"""AdaMuon — orthogonalized momentum with factored, quantized variance adaptation.

AdaMuon is Muon's Newton-Schulz orthogonalization (Jordan et al.; Si et al.,
*AdaMuon: Adaptive Muon Optimizer*, arXiv:2507.11005) grafted onto Adakaon's
memory backend. It targets **AdamW-beating precision** for diffusion fine-tuning
at near-Adafactor memory.

The pipeline (2-D / conv weights) is, in order:

1. **First moment of the RAW gradient** — an EMA ``m = β1·m + (1-β1)·g`` kept in a
   quantized codec (bf16/int8/4bit) exactly like :class:`~kaon.adakaon.Adakaon`.
2. **Orthogonalize** ``m`` with a Newton-Schulz iteration (``ns_steps``, default 2)
   → ``O ≈ U·Vᵀ``.
3. **Factored second moment OF ``O``** (Adafactor row+col EMA) → ``u = O·inv_sqrt(v)``
   (uncorrected unless ``bias_correction=True``; ``clip_threshold`` absorbs the cold-``v``
   overshoot otherwise — see its docstring).
4. **RMS scaling** to a shape-independent target (see below) → apply at ``lr``.

This is the key difference from Adakaon, which factors the second moment of the
*gradient* and takes momentum of the *normalized update*. AdaMuon reverses the
order: momentum is on the raw gradient (it feeds Newton-Schulz), and the factored
second moment is computed on the *orthogonalized* signal ``O`` — that variance
adaptation is what AdaMuon adds over plain Muon, and what the literature credits
for closing/overtaking AdamW.

**Update-norm note (why ``0.2``, not ``0.2·√max(R,C)``).** Plain Muon scales the
orthogonal factor ``O`` (which has RMS ``≈ 1/√max(R,C)``) by ``0.2·√max(R,C)`` to
get a shape-independent applied RMS of ``0.2``. In AdaMuon the factored
``inv_sqrt(v)`` already rescales ``u`` to RMS ``≈ 1`` (the ``c_factor`` term is
``≈ √max(R,C)``), so reapplying ``√max(R,C)`` would double-count the shape and
make the update grow with layer size. (``RMS ≈ 1`` is the *asymptotic* statement:
with no bias correction the actual pre-clip RMS is ``≈ 1/√(1-β₂ᵗ)``, and
``clip_threshold`` is what pins it to 1 for the first ``~1/(1-β₂)`` steps.) We therefore scale by the **constant**
``_UPDATE_RMS`` (0.2) only. Every parameter — 2-D and 1-D alike — is normalized to
an applied RMS of ``≈ 0.2·lr``, so a single ``lr`` governs the whole model (no
separate Adam LR for biases/norms, unlike plain Muon).

**Momentum semantics differ from plain Muon** (Jordan et al.). Muon uses a
heavy-ball buffer (``m = momentum·m + g``) with optional Nesterov; AdaMuon uses an
Adam-style EMA lerp (``m = β1·m + (1-β1)·g``), which is the canonical AdaMuon form
and what the shared momentum codec implements — giving int8/4bit momentum and
bit-exact checkpoint resume for free. A learning rate tuned for ``Muon`` will not
transfer directly.

State cost: factored second moment (row+col, ~0) + one quantized first moment
(~2 B/param bf16, ~1 B int8, ~0.5 B 4bit) — Adafactor-class memory, well under
AdamW. 1-D params (biases, norm scales) are not orthogonalized; they use
Adakaon's non-factored Adam-style path (full per-coordinate second moment, same
quantized momentum), RMS-normalized to the same ``0.2·lr`` target.

It is a standard ``torch.optim.Optimizer``. ``foreach=True`` (default) batches the
step across parameters — the decisive win for LoRA/LoKr adapters (hundreds of tiny
2-D weights), where a per-parameter Python loop is the dominant throughput cost.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, Literal

import torch
from torch import Tensor
from torch.optim import Optimizer

from kaon._autolr import DEFAULT_FUSE_REL, AutoLRMixin
from kaon._backend import (
    FOREACH_BATCH_CUTOFF,
    cautious_batched_,
    cautious_one_,
    centralize_grads_,
    foreach_budget,
    is_low_precision,
    rms,
    subtract_batched_,
    subtract_one_,
)
from kaon._factored import factored_inv_sqrt_factors, update_factored_state
from kaon._foreach_plan import ForeachChunk, ForeachPlanMixin, ForeachSpec
from kaon._momentum_codec import (
    _FOURBIT_BLOCK,
    _make_codec,
    _MomentumCodec,
    load_state_dict_preserving_dtypes,
    warn_if_4bit_high_beta1,
)

__all__ = ["AdaMuon"]

MomentumDtype = Literal["bfloat16", "float32", "int8", "4bit"]

# Shape-independent applied-update RMS target (before lr). The factored
# inv_sqrt(v) already brings ``u`` to RMS≈1, so this is the only magnitude scale
# applied — equal to Muon's per-element RMS (``0.2``). See the module docstring on
# why ``√max(R,C)`` is NOT reapplied here.
_UPDATE_RMS = 0.2

# Foreach batching knobs — mirror Adakaon's (kept local so AdaMuon is a
# standalone module, not coupled to adakaon.py internals). See
# docs/foreach-batching.md for the rationale behind each constant.
_STACK_BYTES_PER_ELEM = 48



def zeropower_via_newtonschulz5(grad: Tensor, steps: int) -> Tensor:
    """Newton-Schulz quintic iteration: approximate the orthogonal factor of ``grad``.

    Returns ``U`` (≈ ``U @ V.T`` of ``grad = U S V.T``) in bf16. Runs in bf16 for
    speed/memory — the iteration is robust to it. ``grad`` must be 2-D. (Muon's
    orthogonalization, Jordan et al.; the per-parameter path uses it directly, the
    foreach path the batched ``_stacked`` variant below.)
    """
    assert grad.ndim == 2, "Newton-Schulz expects a 2-D matrix"
    a, b, c = 3.4445, -4.7750, 2.0315
    x = grad.bfloat16()
    transposed = x.size(0) > x.size(1)
    if transposed:  # iterate on the smaller inner dimension
        x = x.mT
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        aa = x @ x.mT
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x
    if transposed:
        x = x.mT
    return x


def zeropower_via_newtonschulz5_stacked(grad: Tensor, steps: int) -> Tensor:
    """Batched Newton-Schulz quintic iteration over a stack of 2-D matrices.

    ``grad`` is ``[N, R, C]`` (all slices share ``R, C``). Returns ``[N, R, C]``
    bf16 orthogonal factors, one per slice — element-for-element the per-slice
    :func:`zeropower_via_newtonschulz5` but with a single ``bmm`` per
    iteration instead of ``N`` matmuls (the LoRA throughput win). Each slice is
    normalized by its own Frobenius norm and transposed to its smaller inner
    dimension (uniform across the bucket since all slices share the shape).

    bf16 matmul reduction order differs between ``bmm`` and per-slice ``@``, so this
    matches the per-slice helper closely but not bit-for-bit; both are unbiased.
    """
    assert grad.ndim == 3, "stacked Newton-Schulz expects [N, R, C]"
    a, b, c = 3.4445, -4.7750, 2.0315
    x = grad.bfloat16()
    transposed = x.size(1) > x.size(2)
    if transposed:  # iterate on the smaller inner dimension (uniform per bucket)
        x = x.mT
    n = x.shape[0]
    fro = x.reshape(n, -1).norm(dim=1).clamp_min(1e-7).view(n, 1, 1)
    x = x / fro
    for _ in range(steps):
        aa = torch.bmm(x, x.mT)
        bb = b * aa + c * torch.bmm(aa, aa)
        x = a * x + torch.bmm(bb, x)
    if transposed:
        x = x.mT
    return x


# --------------------------------------------------------------------------- #
# Pure-tensor step math (the ``torch.compile`` unit).
#
# ``compile=True`` compiles THESE functions, not the step body. The step body
# reads ``p.grad`` / ``self.state[p]`` / ``group[...]`` per parameter, which makes
# Dynamo install a guard per parameter on *whether that parameter has a gradient*
# and on the *value* of ``group["lr"]``. Both change during normal training — a
# MoE / CFG-dropout / partial-accumulation step changes the grad set, and any LR
# schedule changes ``lr`` every step — so the whole step recompiled until
# ``recompile_limit`` (8) was hit and compilation silently fell back to eager
# (measured: 8 graphs in both scenarios; ``add_param_group`` cost 2 more, ~8 s).
#
# These functions take *stacked tensors* and *constant* configuration only, so the
# guards are on shapes/dtypes (which automatic-dynamic generalizes after the second
# distinct shape) and on config values that do not change during a run. Everything
# that touches Python containers — the grad filter, the bucketing, the momentum
# codec, the state write-back — stays outside the graph. Step-varying scalars
# (``lr``, the bias-correction factor) are passed as 0-D tensors when compiled so
# Dynamo cannot specialize on their value; a 0-D tensor multiply is bit-identical
# to the Python-float multiply used in eager.
# --------------------------------------------------------------------------- #

Scalar = Tensor | float


def _add_scaled_(delta: Tensor, other: Tensor, scale: Scalar) -> Tensor:
    """``delta += scale * other``. ``scale`` is a Python float in eager (one fused
    ``add_(alpha=)`` kernel, bit-exact with the pre-refactor code) and a 0-D tensor
    under ``torch.compile`` (``alpha=`` would specialize the graph on the value);
    Inductor fuses both forms into the surrounding elementwise chain anyway."""
    if isinstance(scale, Tensor):
        return delta.add_(other * scale)
    return delta.add_(other, alpha=scale)


def _factored_math(
    m: Tensor,
    grad: Tensor,
    row: Tensor,
    col: Tensor,
    p_fp32: Tensor | None,
    ns_steps: int,
    beta2: float,
    eps1: float,
    clip: float,
    bc_scale: Scalar | None,
    lr_scale: Scalar,
    wd_scale: Scalar | None,
    cautious: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Stacked 2-D core: Newton-Schulz -> factored second moment -> clip -> scale.

    ``m`` / ``grad`` / ``p_fp32`` are ``[N, R, C]`` fp32; ``row`` / ``col`` are the
    ``[N, R]`` / ``[N, C]`` *stacked copies* of the per-param state (mutated here and
    written back by the caller). Returns ``(row, col, delta)``.
    """
    N, R, C = m.shape  # noqa: N806 — matrix dims
    ortho = zeropower_via_newtonschulz5_stacked(m, ns_steps).float()   # [N, R, C]

    # Factored second moment OF the orthogonalized signal (HF eps placement).
    omb = 1.0 - beta2
    ortho_sq = ortho * ortho
    if eps1 > 0:
        ortho_sq = ortho_sq.add_(eps1)
    row.lerp_(ortho_sq.mean(dim=-1), omb)
    col.lerp_(ortho_sq.mean(dim=-2), omb)

    r_factor = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)  # [N, R, 1]
    c_factor = col.rsqrt().unsqueeze(-2)                                       # [N, 1, C]
    update = ortho.mul(r_factor).mul_(c_factor)                               # [N, R, C], RMS≈1
    if bc_scale is not None:
        update.mul_(bc_scale)

    # Clip ceiling (RMS≈1 domain) then the constant shape-independent scale.
    rms_ = update.reshape(N, -1).norm(2, dim=1) / math.sqrt(R * C)
    update.div_(rms_.div_(clip).clamp_(min=1.0).view(N, 1, 1))
    update.mul_(lr_scale)
    delta = update

    if wd_scale is not None:
        delta = _add_scaled_(delta, p_fp32, wd_scale)
    if cautious:
        delta = cautious_batched_(delta, grad)
    return row, col, delta


def _nonfactored_pre_math(
    grad: Tensor,
    v: Tensor,
    beta2: float,
    eps1: float,
    clip: float,
    bc_scale: Scalar | None,
    lr_scale: Scalar,
) -> tuple[Tensor, Tensor]:
    """Stacked ``ndim<=1`` core, part 1: second moment -> normalize -> clip -> scale.

    Split in two because the momentum codec sits between the halves and is
    Python-container work that must stay out of the graph. ``grad`` / ``v`` are
    ``[N, L]`` fp32 (``v`` is the stacked copy of the state). Returns ``(v, update)``.
    """
    N, length = grad.shape  # noqa: N806
    omb = 1.0 - beta2
    grad_sq = grad * grad
    if eps1 > 0:
        grad_sq = grad_sq.add_(eps1)
    v.lerp_(grad_sq, omb)

    update = grad.mul(v.rsqrt())                                      # [N, L], RMS≈1
    if bc_scale is not None:
        update.mul_(bc_scale)
    rms_ = update.norm(2, dim=1) / math.sqrt(length)
    update.div_(rms_.div_(clip).clamp_(min=1.0).view(N, 1))
    update.mul_(lr_scale)
    return v, update


def _post_math(
    delta: Tensor,
    grad: Tensor,
    p_fp32: Tensor | None,
    wd_scale: Scalar | None,
    cautious: bool,
) -> Tensor:
    """Stacked tail of the ``ndim<=1`` bucket: weight decay + cautious mask."""
    if wd_scale is not None:
        delta = _add_scaled_(delta, p_fp32, wd_scale)
    if cautious:
        delta = cautious_batched_(delta, grad)
    return delta


def _factored_one_math(
    m: Tensor,
    grad: Tensor,
    row: Tensor,
    col: Tensor,
    p_fp32: Tensor | None,
    ns_steps: int,
    beta2: float,
    eps1: float,
    clip: float,
    bc_scale: Scalar | None,
    lr_scale: Scalar,
    wd_scale: Scalar | None,
    cautious: bool,
) -> Tensor:
    """Per-parameter 2-D core. Tensors are ``[R, C]``; ``row`` / ``col`` are the state
    tensors themselves (updated in place). Returns the ``[R, C]`` delta."""
    ortho = zeropower_via_newtonschulz5(m, ns_steps).float()          # [R, C]
    update_factored_state(ortho, row, col, beta2, eps1)
    r_factor, c_factor = factored_inv_sqrt_factors(row, col)
    update = ortho.mul(r_factor).mul_(c_factor)                       # [R, C], RMS≈1
    if bc_scale is not None:
        update.mul_(bc_scale)
    if clip > 0:
        update.div_((rms(update) / clip).clamp_(min=1.0))
    update.mul_(lr_scale)
    delta = update
    if wd_scale is not None:
        delta = _add_scaled_(delta, p_fp32, wd_scale)
    if cautious:
        delta = cautious_one_(delta, grad)
    return delta


def _nonfactored_one_pre_math(
    grad: Tensor,
    v: Tensor,
    beta2: float,
    eps1: float,
    clip: float,
    bc_scale: Scalar | None,
    lr_scale: Scalar,
) -> Tensor:
    """Per-parameter ``ndim<=1`` core, part 1 (see :func:`_nonfactored_pre_math`).
    ``v`` is the state tensor, updated in place; returns the scaled update."""
    grad_sq = grad * grad
    if eps1 > 0:
        grad_sq.add_(eps1)
    v.lerp_(grad_sq, 1.0 - beta2)
    update = grad.mul(v.rsqrt())
    if bc_scale is not None:
        update.mul_(bc_scale)
    if clip > 0:
        update.div_((rms(update) / clip).clamp_(min=1.0))
    update.mul_(lr_scale)
    return update


def _post_one_math(
    delta: Tensor,
    grad: Tensor,
    p_fp32: Tensor | None,
    wd_scale: Scalar | None,
    cautious: bool,
) -> Tensor:
    """Per-parameter tail: decoupled weight decay + cautious mask."""
    if wd_scale is not None:
        delta = _add_scaled_(delta, p_fp32, wd_scale)
    if cautious:
        delta = cautious_one_(delta, grad)
    return delta


class AdaMuon(AutoLRMixin, ForeachPlanMixin, Optimizer):
    """Orthogonalized-momentum optimizer with factored quantized variance.

    Args:
        params: parameters or param-group dicts.
        lr: learning rate. AdaMuon (like Muon) takes a larger LR than Adam because
            updates are RMS-normalized; ``~2e-2`` is a typical starting point. A
            single ``lr`` governs both 2-D and 1-D params (all normalized to
            applied RMS ``≈ 0.2·lr``).
        betas: ``(beta1, beta2)``. ``beta1`` is the first-moment EMA (lerp) decay —
            ``beta1=0`` orthogonalizes the raw gradient with no momentum buffer
            (minimum memory). ``beta2`` is the factored second-moment decay.
        eps: ``(eps1, eps2)``. ``eps1`` is added to ``O**2`` before the factored
            reductions (HF Adafactor convention). ``eps2`` is reserved/unused.
        weight_decay: decoupled weight decay (folded into the per-step delta).
        ns_steps: Newton-Schulz iteration steps. **Default ``2``** (LLM Muon uses
            5). On a paired pixel-DDPM sweep ``2`` was both faster (~0.9 ms/step per
            saved iteration) AND lower val than ``5``, and ``1`` lost the edge over
            Adakaon. The mechanism is **aspect-ratio dependent**, not
            "over-orthogonalization": the quintic settles into a singular-value band
            ≈[0.67, 1.20] and never leaves it, and on skinny matrices (LoRA shapes,
            and conv weights matrixized to ``(out, in·kh·kw)``) ``ns=2`` is already
            inside it (mean sv 0.89-1.06 at 4:1-16:1), so steps 3-5 buy nothing. On
            *square* matrices ``ns=2`` is heavily **under**-orthogonalized instead
            (mean sv 0.56 at 256x256, 0.31 at 1024x1024, with near-zero directions),
            which the diffusion proxy never exercises — re-sweep ``ns_steps`` on a
            model with large square weights. See docs/adamuon.md for the table.
        clip_threshold: RMS ceiling on the normalized update (``rms(u) <= thr``),
            applied in the RMS≈1 domain (so ``1.0`` matches Adakaon's semantics).
            **A first-order hyperparameter, not a safety net.** The factored second
            moment has no bias correction by default, so on a real proxy-U-Net run
            the mean ``rms(u)`` before clipping measures ``31.9`` at step 1, ``3.56``
            at 100, ``1.20`` at 1000 and ``0.98`` at 2999 (β₂=0.999) — almost exactly
            ``1/√(1-β₂ᵗ)``. The clip is therefore *active on every weight bucket* for
            the first ``~1/(1-β₂)`` steps and on 70-80% after, which makes it both the
            thing that sets the early effective step size AND, in practice, the second
            moment's bias correction (see ``bias_correction``). docs/adamuon.md has the
            measured lr/clip coupling.
        bias_correction: divide the factored second moment by ``1 - β₂ᵗ`` (Adam's
            bias correction, per parameter — ``state["step"]``, so a weight that only
            sometimes gets a gradient is corrected by its own update count). Because
            the row factor is a *ratio* of row stats the correction cancels there and
            survives only in the column factor, so it reduces exactly to scaling the
            normalized update by ``√(1 - β₂ᵗ)`` before the clip — one multiply, no
            state beyond the counter. **Default ``False`` because ``clip_threshold``
            already does this job, harder.** The uncorrected ``rms(u)`` measures
            almost exactly ``1/√(1-β₂ᵗ)``, which is what the correction divides out,
            so while the clip binds both settings emit the *same* update (measured
            applied RMS in units of ``0.2·lr``: 1.0000 vs 1.0000 at step 1, 1.0000 vs
            0.9953 at 1000, 1.0000 vs 0.9905 at 3000 — a ≤1% rescale, never more), and
            the paired pixel-DDPM A/B finds no gain. The two are alternative
            implementations of the same normalization: with the clip *disabled* the
            correction recovers 92% of its value (proxy 2-seed mean val 0.0701 with
            ``clip=1.0`` alone, 0.0870 clip-off uncorrected, 0.0714 clip-off
            corrected). Turn it on if you raise or disable ``clip_threshold``, or when
            composing with a layer that assumes an unbiased ``1/√v``. See
            docs/adamuon.md for the tables.
        momentum_dtype: storage for the first moment when ``beta1>0`` —
            ``"bfloat16"`` (default, ~2 B/param), ``"float32"`` (4 B), ``"int8"``
            (~1 B, per-row absmax), or ``"4bit"`` (~0.5 B, per-block absmax). Newton-
            Schulz runs in bf16 internally regardless.
        momentum_4bit_block: block size for ``momentum_dtype="4bit"`` (default 128;
            ``0``/negative = whole-tensor).
        cautious: cautious masking (Liang et al. 2024) — zero update coordinates
            whose sign disagrees with the *raw gradient*. **On by default**:
            initially left off (its interaction with orthogonalized updates was
            unvalidated for the Muon family), but a paired sweep showed it helps
            substantially — it flips AdaMuon from a loss to a win vs Adakaon (~2%
            on all seeds). Set ``False`` to recover the un-masked Muon-family update.
        bf16_method: low-precision weight-update strategy —
            ``"stochastic_rounding"`` (default), ``"kahan"`` (+2 B/param), or
            ``"none"``. No-op on fp32 params.
        foreach: batch the step across parameters with stacked ops (default
            ``True``). Bucketed by shape: ``ndim>=2`` factored ``[N,R,C]`` (with a
            batched ``bmm`` Newton-Schulz), ``ndim==1`` non-factored ``[N,L]``. The
            batched 2-D path matches the per-parameter path within bf16 Newton-
            Schulz tolerance (both unbiased); 1-D and all fp32 ops are bit-exact.
        foreach_batch_cutoff: per-tensor element count above which a weight loops
            instead of stacking (performance knob; default ``2_000_000``).
        foreach_stack_budget: max elements per stacked chunk. ``None`` (default)
            adapts to free VRAM; an int pins a fixed cap.
        compile: ``torch.compile`` the step's tensor math, fusing its elementwise
            chain. Only the pure-tensor bucket kernels are compiled — the parameter
            bookkeeping (grad filter, bucketing, momentum codec, state write-back)
            stays in eager Python, so the compiled graphs are guarded on shapes and
            dtypes only. Compiling the whole step body instead made Dynamo guard on
            *which parameters have a gradient* and on the *value* of ``lr``, so a MoE
            / CFG-dropout / partial-accumulation grad set — or any LR schedule —
            burned through ``recompile_limit`` (8) and silently fell back to eager
            (measured: 8 graphs in both scenarios, 1 after the fix).
            **Workload-dependent — benchmark it.** The win scales with how much
            (fusable) elementwise math the step does, so for AdaMuon (heavy
            Newton-Schulz + factored + cautious + scale) it helps broadly. Measured
            eager->compiled ratios (RTX 3000 Ada, one scenario per process): 0.58x on
            two 512² weights, 0.71x on a 128-weight LoRA bag and on a single 2048²
            weight, 0.76x on 12 *distinct*-shaped small weights, 0.84x on a U-Net-like
            mix, ~1.00x on a 4x1024² full-fine-tune-like set. Trading the old
            whole-step graph for per-bucket kernels costs peak throughput on
            multi-shape models when lr is *constant* (many-distinct 7.1 -> 19.2 ms,
            U-Net-like 6.5 -> 10.4) and gains on the ``foreach``-batched and
            single-weight cases — but that peak needs a constant lr. *With* an LR
            schedule the two multi-shape sets measured 1.03x / 1.00x before the fix
            (compile did nothing at all) versus 0.29x / 0.79x after, so in the regime
            real runs are in the flag went from a no-op to a 1.3-3.4x step speedup.
            One-time warmup (~4 s); a no-op when the model fwd/bwd dominates (SDXL is
            UNet-bound). Numerically equivalent to eager: the 0-D-tensor scalars are
            bit-identical, and the one place the compiled kernel is *written*
            differently (weight decay as ``p*(lr·wd)`` rather than
            ``add_(alpha=lr·wd)``, so Dynamo cannot specialize on the value) is fused
            away by Inductor — measured bit-identical on the non-orthogonalized
            buckets with ``wd`` on. The residual compiled-vs-eager difference is
            Inductor reassociating the **bf16 Newton-Schulz**, so it appears only on
            ``ndim>=2`` weights at ~1e-5 relative — the same order as the existing
            foreach-vs-per-param bf16 gap. SR stays unbiased (no host syncs). Not recommended on CPU (inconsistent). NB: compiling
            *only* the Newton-Schulz does NOT help on LoRA-rank matrices — the win is
            fusing the whole bucket. Default ``False``.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 2e-2,
        betas: tuple[float, float] = (0.95, 0.999),
        eps: tuple[float, float] = (1e-30, 1e-3),
        weight_decay: float = 0.0,
        *,
        ns_steps: int = 2,
        clip_threshold: float = 1.0,
        bias_correction: bool = False,
        momentum_dtype: MomentumDtype = "bfloat16",
        momentum_4bit_block: int = _FOURBIT_BLOCK,
        cautious: bool = True,
        gradient_centralization: bool = False,
        bf16_method: str = "stochastic_rounding",
        foreach: bool = True,
        foreach_batch_cutoff: int = FOREACH_BATCH_CUTOFF,
        foreach_stack_budget: int | None = None,
        compile: bool = False,  # noqa: A002 — public kwarg name
        auto_lr: bool = False,
        auto_lr_scale: float = 1.0,
        auto_lr_fuse_rel: float = DEFAULT_FUSE_REL,
        auto_lr_d0: float | None = None,
    ) -> None:
        beta1, beta2 = float(betas[0]), float(betas[1])
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"betas[0] must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"betas[1] must be in [0, 1), got {beta2}")
        if lr < 0.0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be >= 1, got {ns_steps}")
        if clip_threshold <= 0.0:
            raise ValueError(f"clip_threshold must be > 0, got {clip_threshold}")
        if momentum_dtype not in ("bfloat16", "float32", "int8", "4bit"):
            raise ValueError(
                f"momentum_dtype must be bfloat16/float32/int8/4bit, got {momentum_dtype!r}"
            )
        if bf16_method not in ("stochastic_rounding", "kahan", "none"):
            raise ValueError(f"bf16_method must be stochastic_rounding/kahan/none, got {bf16_method!r}")
        if foreach_batch_cutoff < 1:
            raise ValueError(f"foreach_batch_cutoff must be >= 1, got {foreach_batch_cutoff}")
        warn_if_4bit_high_beta1(beta1, momentum_dtype)
        defaults = {
            "lr": lr,
            "betas": (beta1, beta2),
            "eps": (float(eps[0]), float(eps[1])),
            "weight_decay": weight_decay,
            "ns_steps": ns_steps,
            "clip_threshold": clip_threshold,
            "bias_correction": bias_correction,
            "momentum_dtype": momentum_dtype,
            "momentum_4bit_block": momentum_4bit_block,
            "cautious": cautious,
            "gradient_centralization": gradient_centralization,
            "bf16_method": bf16_method,
        }
        super().__init__(params, defaults)
        self._foreach = foreach
        self._foreach_batch_cutoff = foreach_batch_cutoff
        self._foreach_stack_budget = foreach_stack_budget
        # Optional torch.compile of the step's tensor math. The compiled unit is the
        # set of pure-tensor kernels above, NOT the step body: see the comment there
        # for why compiling the body recompiled on every grad-set / lr change. No
        # host syncs in the step, so stochastic rounding stays unbiased.
        self._compile = bool(compile)
        self._factored_math = torch.compile(_factored_math) if compile else _factored_math
        self._nonfactored_pre_math = (
            torch.compile(_nonfactored_pre_math) if compile else _nonfactored_pre_math
        )
        self._post_math = torch.compile(_post_math) if compile else _post_math
        self._factored_one_math = (
            torch.compile(_factored_one_math) if compile else _factored_one_math
        )
        self._nonfactored_one_pre_math = (
            torch.compile(_nonfactored_one_pre_math) if compile else _nonfactored_one_pre_math
        )
        self._post_one_math = torch.compile(_post_one_math) if compile else _post_one_math
        # 0-D scalar buffers for the step-varying scalars under compile (see _scalar).
        self._scalars: dict[tuple[str, Any], Tensor] = {}
        # One momentum codec per dtype string (stateless beyond the dtype).
        self._codecs: dict[str, _MomentumCodec] = {}

        # Composable parameter-free LR (continuous Mechanic) via AutoLRMixin. off -> zero overhead.
        self._init_autolr(auto_lr, auto_lr_scale, auto_lr_fuse_rel, auto_lr_d0)

    def _scalar(self, name: str, value: float, device: Any) -> Scalar:
        """A step-varying scalar in the form the math kernels want.

        Eager: the Python float itself. Compiled: a cached 0-D tensor refilled in
        place, because Dynamo specializes the graph on a float argument's *value* —
        an ``lr`` schedule (or the per-step bias-correction factor) would otherwise
        recompile every iteration until ``recompile_limit`` drops us back to eager.
        The multiply is bit-identical either way; the ``fill_`` is stream-ordered
        after any kernel still reading the buffer, so reusing it across groups and
        buckets within a step is safe.
        """
        if not self._compile:
            return value
        key = (name, device)
        buf = self._scalars.get(key)
        if buf is None:
            buf = self._scalars[key] = torch.empty((), dtype=torch.float32, device=device)
        return buf.fill_(value)

    def _bc_scale(self, t: int, beta2: float, device: Any) -> Scalar:
        """``√(1 - β₂ᵗ)`` — the whole of the factored second moment's bias correction.

        Correcting ``v`` means dividing *both* row and col stats by ``1 - β₂ᵗ``. The
        row factor is ``rsqrt(row / mean(row))``, a ratio, so the correction cancels
        there and only the column factor ``rsqrt(col)`` keeps it: the net effect on
        the normalized update is exactly one multiply by ``√(1 - β₂ᵗ)``, applied
        *before* the clip (that is the point — an uncorrected update is up to ~30x
        too large on the first steps and the clip is what absorbs it).

        Always a *scalar*: ``t`` is part of the bucket key when ``bias_correction`` is
        on (see :meth:`_step_foreach`), so every member of a bucket shares it. That
        keeps this off the H2D path — a per-bucket ``torch.tensor(..., device=cuda)``
        of per-slice factors would be a pageable host-to-device copy on every bucket
        of every step, for a value that is one float.
        """
        return self._scalar("bc", math.sqrt(1.0 - beta2 ** t), device)

    def _codec(self, group: dict[str, Any]) -> _MomentumCodec:
        md = group["momentum_dtype"]
        codec = self._codecs.get(md)
        if codec is None:
            codec = self._codecs[md] = _make_codec(md)
        return codec

    @torch.no_grad()
    def _init_state(self, p: Tensor, state: dict[str, Any], group: dict[str, Any]) -> None:
        grad = p.grad
        factored = p.ndim >= 2
        if factored:
            # Factored second moment is over O (same matrixized shape as the grad):
            # ndim==2 is its own matrix; ndim>2 (conv) reshapes to [out, in·kh·kw].
            gv = grad if p.ndim == 2 else grad.reshape(grad.shape[0], -1)
            row_shape = gv.shape[:-1]
            col_shape = gv.shape[:-2] + gv.shape[-1:]
            state["row"] = torch.zeros(row_shape, dtype=torch.float32, device=p.device)
            state["col"] = torch.zeros(col_shape, dtype=torch.float32, device=p.device)
        else:
            state["v"] = torch.zeros_like(grad, dtype=torch.float32)
        # Per-parameter update counter for the (optional) second-moment bias
        # correction. Always maintained — it costs one int and it lets
        # ``bias_correction`` be turned on mid-run / across a resume without the
        # correction restarting from a cold t. See _bc_scale on why it is per
        # parameter rather than per group.
        state["step"] = 0
        # First moment stores the EMA of the RAW gradient, in the param's original
        # shape (the codec matrixizes it per-step for Newton-Schulz).
        if group["betas"][0] > 0:
            self._codec(group).init_state(state, grad, group)
        if is_low_precision(p) and group["bf16_method"] == "kahan":
            state["shift"] = torch.zeros_like(p)

    @torch.no_grad()
    def _step_impl(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._run_step()
        return loss

    @torch.no_grad()
    def _run_step(self) -> None:
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                # A group can be entirely gradient-less on a given step (MoE routing,
                # CFG dropout, partial gradient accumulation, or a plain step() with
                # nothing backwarded). Skipping it early also keeps the foreach path's
                # ``params[0].device`` probe below from indexing an empty list.
                continue
            for p in params:
                if p.grad.is_sparse:
                    raise RuntimeError("AdaMuon does not support sparse gradients")
            if group["gradient_centralization"]:
                centralize_grads_(params)
            if self._foreach and self._group_foreach_eligible(group):
                chunk_budget = foreach_budget(self._foreach_stack_budget, self._foreach_batch_cutoff, _STACK_BYTES_PER_ELEM, params[0].device)
                cutoff = min(self._foreach_batch_cutoff, chunk_budget // 2)
                fast: list[Tensor] = []
                slow: list[Tensor] = []
                for p in params:
                    (fast if self._param_foreach_eligible(p, group, cutoff) else slow).append(p)
                if len(fast) >= 2:
                    self._step_foreach(fast, group, chunk_budget)
                    for p in slow:
                        self._step_one_param(p, group)
                else:
                    self._drop_foreach_plan(group)
                    for p in params:
                        self._step_one_param(p, group)
            else:
                # Per-parameter fallback for the whole group: drop any cached plan for it,
                # so a cached plan only ever describes a group the foreach path stepped.
                self._drop_foreach_plan(group)
                for p in params:
                    self._step_one_param(p, group)

    def state_dict(self) -> dict[str, Any]:
        """Base state + the auto_lr tuner blob (via AutoLRMixin) when auto_lr is on."""
        return self._autolr_state_dict(super().state_dict())

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state, preserving the quantized first moment's stored dtype.

        torch's default ``load_state_dict`` upcasts every state tensor to the
        param's dtype (fp32), which would inflate a bf16/int8/4bit momentum back to
        fp32 on resume — losing the memory the codec saves and breaking bit-exact
        resume. Delegate to the shared dtype-preserving helper.

        It also **replaces** each ``param_groups`` dict with the checkpoint's (only
        ``params`` is carried over), so a checkpoint written by an older kaon has no
        entry for a hyperparameter added since — reading it would raise ``KeyError``
        on the first step. Backfill any key the checkpoint predates from
        ``self.defaults``; keys the checkpoint *does* carry win, so a resumed run
        keeps its own tuning.
        """
        self._autolr_load(state_dict, lambda sd: load_state_dict_preserving_dtypes(self, sd))
        for group in self.param_groups:
            for key, value in self.defaults.items():
                group.setdefault(key, value)
        # The loader REPLACES state["v"] / state["m"] (and every param_groups dict), so a
        # cached foreach plan would keep stepping detached buffers under a dead group id.
        self._clear_foreach_plans()

    def _autolr_reset_base_state(self) -> None:
        """Reset the base optimizer after an AutoLR rollback: the cleared state is
        reallocated by the next step, so the cached view plan must go with it."""
        super()._autolr_reset_base_state()
        self._clear_foreach_plans()

    # ----------------------------------------------------------------- foreach
    # Bucketing, chunking and the cached view plan live in kaon._foreach_plan. ``row`` /
    # ``col`` (factored) and ``v`` (non-factored) are the state buffers the bucket bodies
    # stack and write back through. With ``bias_correction`` on, ``t`` joins the bucket key
    # so every slice in a bucket shares the correction factor and it stays a Python float
    # (see ``_bc_scale``); params that step together keep the same ``t`` and so stay in one
    # bucket, and only intermittent gradients fragment it. ADOPT groups by its per-param
    # step for the same reason. ``single_alias`` is ``_stack_fp32``'s zero-copy
    # ``unsqueeze`` for a bucket of one, kept on the plan's stacking helpers.
    _FOREACH_SPEC = ForeachSpec(
        factored_state=("row", "col"),
        flat_state=("v",),
        extra_key=lambda state, group: (
            state.get("step", 0) if group.get("bias_correction", False) else 0
        ),
        momentum_cache=lambda group: (
            group["betas"][0] > 0 and group["momentum_dtype"] != "4bit"
        ),
        single_alias=True,
    )

    @staticmethod
    def _group_foreach_eligible(group: dict[str, Any]) -> bool:
        return (
            group["clip_threshold"] > 0
            and group["bf16_method"] != "kahan"  # kahan needs a per-param shift buffer
        )

    @staticmethod
    def _param_foreach_eligible(p: Tensor, group: dict[str, Any], cutoff: int) -> bool:
        # 0-D scalars are NOT excluded: they ride the non-factored bucket as
        # length-1 rows (see kaon._backend.flat_view). Only the per-tensor size cap
        # and the awkward dtype/contiguity cases fall back to the per-param loop.
        if p.numel() > cutoff:
            return False
        if (
            group["bf16_method"] == "stochastic_rounding"
            and is_low_precision(p)
            and p.dtype != torch.bfloat16
        ):
            return False
        if p.ndim > 2:
            return p.data.is_contiguous() and p.grad.is_contiguous()
        return True

    @torch.no_grad()
    def _step_foreach(self, params: list[Tensor], group: dict[str, Any], budget: int) -> None:
        """Batched step. Buckets: ``ndim>=2`` factored ``[N,R,C]`` (orthogonalized),
        ``ndim==1`` non-factored ``[N,L]`` (Adam-style)."""
        beta1, beta2 = group["betas"]
        eps1, _eps2 = group["eps"]
        lr, clip = group["lr"], group["clip_threshold"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]
        ns_steps = group["ns_steps"]
        bias_correction = group.get("bias_correction", False)
        codec = self._codec(group)

        for chunk in self._foreach_chunks(params, group, budget):
            if chunk.eff is not None:
                self._factored_bucket(
                    chunk, ns_steps,
                    beta1, beta2, eps1, lr, clip, wd, cautious, bias_correction,
                    bf16_method, codec,
                )
            else:
                self._nonfactored_bucket(
                    chunk,
                    beta1, beta2, eps1, lr, clip, wd, cautious, bias_correction,
                    bf16_method, codec,
                )

    @staticmethod
    def _bump_steps(states: list[dict[str, Any]]) -> int:
        """Advance each parameter's update counter and return the (shared) new value.

        ``.get`` (not ``[]``) so a checkpoint written before the counter existed
        resumes at ``t=1``. Callers pass one bucket, and buckets are keyed on ``t``
        when the correction is on, so the values agree; when it is off the return is
        unused.
        """
        t = 0
        for s in states:
            t = s["step"] = s.get("step", 0) + 1
        return t

    @torch.no_grad()
    def _factored_bucket(
        self,
        chunk: ForeachChunk,
        ns_steps: int,
        beta1: float,
        beta2: float,
        eps1: float,
        lr: float,
        clip: float,
        wd: float,
        cautious: bool,
        bias_correction: bool,
        bf16_method: str,
        codec: _MomentumCodec,
    ) -> None:
        R, C = chunk.eff  # noqa: N806 — matrix dims (stacked tensor is [N, R, C])

        states = chunk.states
        t = self._bump_steps(states)
        rows, cols = chunk.state_views
        pviews = chunk.pviews

        grad = chunk.grad_stack()                                         # [N, R, C]

        # First moment of the RAW gradient (codec owns dequant→EMA→requant). Stays
        # in eager: it walks per-param state dicts, exactly the Python-container
        # work the compiled kernels must not see.
        m = codec.ema_stacked(states, grad, chunk.mat, (R, C), beta1) if beta1 > 0 else grad

        dev = grad.device
        p_fp32 = chunk.param_stack() if wd != 0 else None
        row, col, delta = self._factored_math(
            m, grad, torch.stack(rows), torch.stack(cols), p_fp32,
            ns_steps, beta2, eps1, clip,
            self._bc_scale(t, beta2, dev) if bias_correction else None,
            self._scalar("lr", _UPDATE_RMS * lr, dev),
            self._scalar("wd", lr * wd, dev) if wd != 0 else None,
            cautious,
        )
        torch._foreach_copy_(rows, list(row.unbind(0)))
        torch._foreach_copy_(cols, list(col.unbind(0)))

        subtract_batched_(pviews, delta, bf16_method)

    @torch.no_grad()
    def _nonfactored_bucket(
        self,
        chunk: ForeachChunk,
        beta1: float,
        beta2: float,
        eps1: float,
        lr: float,
        clip: float,
        wd: float,
        cautious: bool,
        bias_correction: bool,
        bf16_method: str,
        codec: _MomentumCodec,
    ) -> None:
        """Non-factored Adam-style update for ``ndim <= 1`` params (biases, norm
        scales, plus 0-D scalars as length-1 rows).

        Not orthogonalized (Newton-Schulz needs a matrix). RMS-normalized to the
        same ``0.2·lr`` target as the 2-D path so a single ``lr`` is consistent
        across the model.

        0-D scalars join the ``L == 1`` bucket as length-1 **views**
        (:func:`~kaon._backend.flat_view`) of the same storage, so ``v``, the codec
        write-back and the weight subtract all reach the original 0-D tensors. Those
        views are the cached plan's (``chunk.state_views`` / ``chunk.pviews``), not
        rebuilt per param per step. At ``L == 1`` the per-slice ``norm(dim=1)/sqrt(1)``
        is exactly the per-param ``rms()`` of a scalar, so the clip is the same op the
        per-param path applies.
        """
        states = chunk.states
        t = self._bump_steps(states)
        (vs,) = chunk.state_views                                         # each [L], fp32
        pviews = chunk.pviews

        grad = chunk.grad_stack()                                         # [N, L]
        dev = grad.device
        v, update = self._nonfactored_pre_math(
            grad, torch.stack(vs), beta2, eps1, clip,
            self._bc_scale(t, beta2, dev) if bias_correction else None,
            self._scalar("lr", _UPDATE_RMS * lr, dev),
        )
        torch._foreach_copy_(vs, list(v.unbind(0)))

        if beta1 > 0:
            delta = codec.ema_stacked(states, update, chunk.mat, (chunk.length,), beta1)  # [N, L]
        else:
            delta = update

        if wd != 0 or cautious:
            p_fp32 = chunk.param_stack() if wd != 0 else None
            delta = self._post_math(
                delta, grad, p_fp32,
                self._scalar("wd", lr * wd, dev) if wd != 0 else None,
                cautious,
            )

        subtract_batched_(pviews, delta, bf16_method)

    @torch.no_grad()
    def _step_one_param(self, p: Tensor, group: dict[str, Any]) -> None:
        beta1, beta2 = group["betas"]
        eps1, _eps2 = group["eps"]
        lr, clip = group["lr"], group["clip_threshold"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]
        ns_steps = group["ns_steps"]

        state = self.state[p]
        if not state:
            self._init_state(p, state, group)
        t = self._bump_steps([state])

        grad_fp32 = p.grad if p.grad.dtype == torch.float32 else p.grad.float()
        ndim = grad_fp32.ndim
        dev = grad_fp32.device
        lr_scale = self._scalar("lr", _UPDATE_RMS * lr, dev)
        wd_scale = self._scalar("wd", lr * wd, dev) if wd != 0 else None
        bc_scale = (
            self._scalar("bc", math.sqrt(1.0 - beta2 ** t), dev)
            if group.get("bias_correction", False) else None
        )
        p_fp32 = (p.data if p.dtype == torch.float32 else p.data.float()) if wd != 0 else None

        if ndim >= 2:
            matrixize = ndim > 2

            def mat(x: Tensor | None) -> Tensor | None:
                # Weight decay and the cautious mask are elementwise, so running them
                # on the matrixized view is equivalent to the original shape.
                return x if x is None or not matrixize else x.reshape(x.shape[0], -1)

            # 1. First moment of the raw gradient (original shape).
            m = self._codec(group).ema_one(state, grad_fp32, beta1) if beta1 > 0 else grad_fp32
            # 2-4. Orthogonalize -> factored second moment OF it -> clip -> scale.
            delta = self._factored_one_math(
                mat(m), mat(grad_fp32), state["row"], state["col"], mat(p_fp32),
                ns_steps, beta2, eps1, clip, bc_scale, lr_scale, wd_scale, cautious,
            )
            if matrixize:
                delta = delta.view_as(grad_fp32)
        else:
            # 1-D: Adam-style (no orthogonalization), RMS-normalized to 0.2·lr.
            update = self._nonfactored_one_pre_math(
                grad_fp32, state["v"], beta2, eps1, clip, bc_scale, lr_scale,
            )
            delta = self._codec(group).ema_one(state, update, beta1) if beta1 > 0 else update
            if wd != 0 or cautious:
                delta = self._post_one_math(delta, grad_fp32, p_fp32, wd_scale, cautious)

        subtract_one_(p, delta, state, bf16_method)
