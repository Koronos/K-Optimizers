"""AdaPNM — Adam + Positive-Negative Momentum on kaon's memory backend.

AdaPNM — the adaptive variant of **Positive-Negative Momentum** (Xie et al. 2021,
*Positive-Negative Momentum: Manipulating Stochastic Gradient Noise to Improve
Generalization*, ICML 2021, arXiv:2103.17182) — implemented on top of the
precision and memory machinery already proven in
:class:`~kaon.adakaon.Adakaon`. It is the **generalization-bucket**
optimizer: an *implicit regularizer* that improves flat-minima / train-val-gap
behaviour **without** the extra forward-backward of SAM. Keeping it a separate
class lets it be A/B'd against Adakaon cleanly (Adakaon is left byte-for-byte
unchanged).

(Developed under the provisional code name *Janus*.)

**The idea (PNM / AdaPNM).** Vanilla momentum averages away the stochastic
gradient noise. PNM instead maintains **two** momentum buffers and, on each step,
feeds the gradient to only **one** of them (alternating which), then forms the
update direction as a *positive-negative mix*

.. code-block:: text

    pn = ((1 + beta0) * m_pos  -  beta0 * m_neg) / noise_norm
    noise_norm = sqrt((1 + beta0)^2 + beta0^2)

The ``(1 + beta0)`` / ``-beta0`` coefficients **amplify the momentum signal and
enlarge the variance of the injected gradient noise** in a controlled way; the
larger, more isotropic noise is what biases SGD toward flatter minima (better
generalization). ``noise_norm`` renormalizes so the *effective* step magnitude is
preserved (the two coefficients have squared-norm ``(1+beta0)^2 + beta0^2``).
AdaPNM divides that pos-neg numerator by the Adam second-moment denominator
``sqrt(v_hat) + eps`` — so it fits the factored-``v`` framework directly.

**The exact update (matches kozistr ``pytorch_optimizer`` ``AdaPNM``, defaults
``ams_bound=False`` here — see below):**

.. code-block:: text

    # per group, step t (1-indexed); alternate which buffer is "positive":
    t odd : (m_pos, m_neg) = (exp_avg,      neg_exp_avg)
    t even: (m_pos, m_neg) = (neg_exp_avg,  exp_avg)

    beta1_sq = beta1 ** 2                      # NOTE: momentum decay is beta1^2
    m_pos   = beta1_sq * m_pos + (1 - beta1_sq) * grad     # only m_pos sees grad
    v       = beta2 * v + (1 - beta2) * grad^2             # Adam second moment

    bc1     = 1 - beta1 ** t                   # bias corrections use beta1 (not ^2),
    bc2_sq  = sqrt(1 - beta2 ** t)             #   matching kozistr exactly
    denom   = sqrt(v_hat) + eps                # v_hat = v / bc2 (AdaPNM folds bc2 in)
    pn      = ((1 + beta0) * m_pos - beta0 * m_neg) / noise_norm
    p      -= (lr / bc1) * pn / denom

Two subtleties carried over verbatim from kozistr: (1) the **first-moment decay
is ``beta1**2``** (their ``beta1_p2``), while the **bias correction uses
``beta1``** (their ``debias(beta1, step)``); (2) only the *positive* buffer is
EMA-updated each step — the negative buffer is the *stale* (one-step-old, because
of the alternation) momentum that gets subtracted. ``beta0`` is kozistr's
``beta3`` (the pos-neg coefficient); their default ``beta3 = 1.0`` gives
``pn = (2*m_pos - m_neg)/sqrt(5)``.

**The factored second moment.** ``v`` reuses Adakaon's backend exactly:
``ndim >= 2`` weights factor ``v`` into row+column EMAs (conv kernels matrixized
to ``[out, in*kh*kw]`` first), ``ndim == 1`` keeps a full per-coordinate ``v``.
The denominator is Adakaon's ``r_factor * c_factor`` reconstruction of
``1/sqrt(v_hat)`` with the same RMS-clip; ``eps`` is added Adafactor-style via the
factored ``eps1`` (the kozistr scalar ``eps`` on the denominator has no factored
analogue, so it is exposed as ``eps`` and applied on the non-factored path /
folded into ``eps1`` — documented under ``eps``). Bias correction ``bc2`` is
applied by scaling the reconstructed inverse-denominator.

**AMSGrad / ``ams_bound``.** kozistr's AdaPNM defaults ``ams_bound=True`` (a
running element-wise max of ``v``). A *factored* ``v`` has no materialized matrix
to take a max over (``max`` of two rank-1 reconstructions is not rank-1), so
AMSBound cannot be applied to 2-D weights without giving up the factoring that is
the whole memory story. AdaPNM therefore **defaults ``ams_bound=False``** and, when
enabled, applies the running max only on the **non-factored (1-D) path** (where
``v`` is full); on factored weights it is silently a no-op. This is the one
deliberate deviation from the kozistr default, made for the factored backend; the
1-D path then matches kozistr's AdaPNM exactly.

**Cautious masking and the pos-neg direction.** Cautious (Liang et al. 2024) zeroes
the update coordinates whose sign disagrees with the *current* gradient
(``delta * g <= 0``), rescaling survivors to preserve the mean magnitude — the same
semantics Adakaon/Lion use, applied here to the **final** ``delta`` (the
pos-neg-mixed, denominator-divided, WD-folded step) against the raw gradient. Note
the tension: PNM's amplified ``-beta0 * m_neg`` term is *designed* to let the step
oppose the instantaneous gradient on noisy coordinates (that is the
noise-manipulation mechanism). Cautious masking removes exactly those coordinates,
so it partially damps the implicit regularizer. We keep ``cautious=True`` by default
for consistency with the rest of kaon and because the rescale preserves step
size, but **this is the knob to ablate first** when measuring AdaPNM's
generalization benefit — try ``cautious=False`` to let the pos-neg mechanism run
unmasked. Like Adakaon, with momentum effectively always on here the mask is not
a no-op.

**What is reused vs new.** Reused from Adakaon's backend: the factored
second-moment helpers (:mod:`kaon._factored`), the momentum **storage layout**
and quant/dequant primitives in :mod:`kaon._momentum_codec` (int8 per-row
absmax; 4-bit per-block absmax, nibble-packed), the stochastic-rounding bf16
weight update (:func:`kaon._stochastic_rounding.add_stochastic_`),
``load_state_dict_preserving_dtypes`` for dtype-safe checkpoint resume, and the
bucketed foreach batching pattern. New here: **two** momentum buffers with the
PNM alternation + positive-negative mixing and ``noise_norm`` renormalization, the
``beta1**2`` first-moment decay with ``beta1`` bias correction, and the read-it-
yourself EMA (the shared codec's ``ema_*`` helpers do an Adam *momentum-of-update*
and update a *single* buffer, so they cannot be reused verbatim; AdaPNM uses the
codec's storage + *read-only* ``_dequant``/requant primitives and runs the
raw-gradient EMA on the positive buffer itself).

It is a standard ``torch.optim.Optimizer`` with a single per-parameter step, so it
drops into per-parameter / gradient-release training loops unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, Literal

import torch
from torch import Tensor
from torch.optim import Optimizer

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
from kaon._autolr import AutoLRMixin
from kaon._factored import factored_inv_sqrt_factors, update_factored_state
from kaon._momentum_codec import (
    fourbit_block_size,
    _FOURBIT_BLOCK,
    _dequant_4bit,
    _dequant_4bit_stacked,
    _quant_4bit,
    _quant_4bit_stacked,
    _quant_int8,
    _quant_int8_stacked,
    load_state_dict_preserving_dtypes,
)

__all__ = ["AdaPNM"]

MomentumDtype = Literal["bfloat16", "float32", "int8", "4bit"]

# Performance / memory knobs mirror Adakaon (see that module for the rationale).
_STACK_BYTES_PER_ELEM = 64  # two momenta + factored v: a touch above Adakaon's 48



def _rms_clip_one_(u: Tensor, clip: float) -> Tensor:
    """Adafactor RMS clip on a single normalized update: ``rms(u) <= clip``. In place."""
    if clip > 0.0:
        u.div_((rms(u) / clip).clamp_(min=1.0))
    return u


def _rms_clip_batched_(u: Tensor, clip: float) -> Tensor:
    """Per-slice RMS clip on a stacked ``[N, *shape]`` normalized update. In place."""
    if clip > 0.0:
        n = u.shape[0]
        per = max(u[0].numel(), 1) if n else 1
        rms = u.reshape(n, -1).norm(2, dim=1) / math.sqrt(per)
        u.div_(rms.div_(clip).clamp_(min=1.0).view(n, *([1] * (u.ndim - 1))))
    return u


# --------------------------------------------------------------------- diagnostics
# Env-gated, zero-overhead-when-off stability probe used to diagnose the real-training
# NaN divergence. Set KAON_PROBE_LOG=/path/to/log to enable. It records, per group step:
#   * the worst factored denominator multiplier  max(r_factor * c_factor * bc2_sq)  and which
#     param produced it (the suspected blowup channel: a near-zero col-EMA -> huge rsqrt), and
#   * the FIRST non-finite parameter, with its shape, fused routing (one-block/big/native),
#     and the offending channel's row/col/momentum stats — i.e. the exact culprit, not a guess.
import os  # noqa: E402

_PROBE_LOG = os.environ.get("KAON_PROBE_LOG")
_PROBE_EVERY = int(os.environ.get("KAON_PROBE_EVERY", "25"))


def _probe_write(line: str) -> None:
    with open(_PROBE_LOG, "a") as fh:  # noqa: SIM115 — short append, diagnostics only
        fh.write(line + "\n")


@torch.no_grad()
def _probe_group(opt: AdaPNM, group: dict[str, Any]) -> None:
    """Inspect every factored param after a step: worst denom multiplier + first non-finite."""
    step = group["step"]
    c = opt._coeffs(group)
    bc2_sq = c["bc2_sq"]
    routing = _probe_routing(opt, group)
    worst_mult, worst_shape = 0.0, None
    for p in group["params"]:
        st = opt.state.get(p)
        if not st or "col" not in st:
            continue
        col = st["col"]
        row = st["row"]
        cfac_max = col.clamp_min(1e-30).rsqrt().max().item()
        rfac_max = row.div(row.mean().clamp_min(1e-30)).clamp_min(1e-30).rsqrt().max().item()
        mult = rfac_max * cfac_max * bc2_sq
        if mult > worst_mult:
            worst_mult, worst_shape = mult, tuple(p.shape)
        if not torch.isfinite(p).all():
            _probe_write(
                f"[NONFINITE] step={step} shape={tuple(p.shape)} route={routing.get(id(p),'?')} "
                f"col_min={col.min().item():.3e} col_max={col.max().item():.3e} "
                f"cfac_max={cfac_max:.3e} rfac_max={rfac_max:.3e} denom_mult={mult:.3e} "
                f"p_absmax={p.detach().abs().float().amax().item():.3e} "
                f"grad_absmax={(p.grad.detach().abs().float().amax().item() if p.grad is not None else float('nan')):.3e}"
            )
    if step == 1 or step % _PROBE_EVERY == 0:
        _probe_write(f"[denom] step={step} worst_mult={worst_mult:.3e} shape={worst_shape}")


def _native_reason(p: Tensor, md: str, bf16m: str, cap: int, ft: Any) -> str:
    """Why does this param miss the fused path? (the 'falling to native' census)."""
    if p.ndim != 2:
        return f"ndim={p.ndim}(1d/conv)"
    if not p.is_cuda:
        return "not_cuda"
    if not p.is_contiguous():
        return "non_contiguous"
    if p.dtype not in (torch.float32, torch.bfloat16):
        return f"dtype={p.dtype}"
    if p.dtype == torch.bfloat16 and bf16m != "stochastic_rounding":
        return f"bf16_method={bf16m}"
    if md == "4bit" and p.shape[1] % 2 != 0:
        return "4bit_odd_C"
    br, bc = ft.next_pow2_tile(*p.shape)
    if br * bc > cap:
        return f"tile>{cap}({br}x{bc})"
    return "unknown"


def _probe_census(one_block: list, big: list, native: list, md: str, bf16m: str, cap: int, ft: Any) -> None:
    from collections import Counter
    reasons = Counter(_native_reason(p, md, bf16m, cap, ft) for p in native)
    shapes_nat = Counter(tuple(p.shape) for p in native)
    _probe_write(
        f"[census] one_block={len(one_block)} big={len(big)} native={len(native)} "
        f"native_reasons={dict(reasons)} native_shapes={dict(shapes_nat)}"
    )
    _probe_write(f"[census] one_block_shapes={dict(Counter(tuple(p.shape) for p in one_block))}")
    _probe_write(f"[census] big_shapes={dict(Counter(tuple(p.shape) for p in big))}")


def _probe_routing(opt: AdaPNM, group: dict[str, Any]) -> dict[int, str]:
    """Map id(param) -> 'one_block' | 'big' | 'native' from the cached fused partition."""
    out: dict[int, str] = {}
    if not getattr(opt, "_fused", False):
        return out
    cached = opt._fused_part.get(id(group))
    if cached is None:
        return out
    _ids, one_block, big, one_dim, native = cached
    for p in one_block:
        out[id(p)] = "one_block"
    for p in big:
        out[id(p)] = "big"
    for p in one_dim:
        out[id(p)] = "one_dim"
    for p in native:
        out[id(p)] = "native"
    return out


class AdaPNM(AutoLRMixin, Optimizer):
    """AdaPNM (Adam + Positive-Negative Momentum) on Adakaon's memory backend.

    Args:
        params: parameters or param-group dicts.
        lr: learning rate.
        betas: ``(beta1, beta2)``. ``beta1`` is the first-moment decay — note the
            actual EMA decay is ``beta1**2`` (matching kozistr's AdaPNM), while the
            bias correction uses ``beta1``. ``beta2`` is the (factored) second-moment
            decay. Default ``(0.8, 0.999)``. **``beta1`` is the loss↔gap dial**: on
            the synthetic proxy the loss is U-shaped in ``beta1`` and bottoms at
            ``0.8`` while the train–val gap stays low; ``0.9`` (the usual Adam value)
            is measurably worse here, and below ``~0.7`` the gap climbs with no loss
            gain. ``beta2=0.999`` is the sweet spot. Raise ``beta1`` toward ``0.95``
            for more regularization (lower gap, higher loss).
        beta0: the **positive-negative momentum coefficient** (kozistr's ``beta3``).
            The update direction is ``((1+beta0)*m_pos - beta0*m_neg)/noise_norm``
            with ``noise_norm = sqrt((1+beta0)^2 + beta0^2)``. ``beta0`` must be in
            ``[0, 1]``. ``beta0=0`` collapses to plain (debiased) Adam-momentum (the
            PNM noise-injection is then off — measurably worse on the proxy, so PNM is
            load-bearing); ``beta0=1`` is the canonical PNM ``(2*m_pos-m_neg)/sqrt(5)``.
            Default ``0.5`` (the measured sweet spot — best loss/gap on the proxy).
        eps: term added to the second-moment denominator for stability. On the
            non-factored (1-D) path it is added to ``sqrt(v_hat)`` exactly as
            kozistr does. On the factored path it is folded into the Adafactor
            ``eps1`` (added to ``grad**2`` before the row/col reductions).
        clip_threshold: Adafactor-style RMS clip on the (v_hat-normalized) update —
            ``rms(pn / sqrt(v_hat)) <= clip_threshold`` before the lr scale, exactly as
            Adakaon. **On by default (``1.0``).** This is the stability guard for the
            factored denominator: a near-zero ``col`` EMA makes ``c_factor = rsqrt(col)``
            explode (~1e4), so a fresh gradient on that channel produces an unbounded
            step → NaN. The clip bounds that runaway (measured: real Cosmos LoKr
            training diverged to NaN without it). ``0`` disables the clip (the original
            unclamped PNM update — diverges on real diffusion training, kept only for
            ablation). Set looser (e.g. ``> 1``) to recover more of the raw PNM step if
            a generalization measurement shows the clip costs gap.
        weight_decay: decoupled (AdamW-style) weight decay. Applied multiplicatively
            ``p *= (1 - lr*weight_decay)`` *before* the moment updates, matching
            kozistr's ``weight_decouple=True`` default (not folded into the cautious
            delta — so cautious does not gate weight decay, unlike Adakaon).
        cautious: cautious masking (Liang et al. 2024) on the final pos-neg step vs
            the gradient. **On by default.** See the class docstring: it interacts
            with — and partially damps — PNM's noise-manipulation mechanism; ablate
            it first when measuring generalization.
        ams_bound: AMSGrad-style running max of ``v``. **Off by default** (kozistr's
            AdaPNM defaults it on, but a factored ``v`` cannot be max'd). When on, it
            applies only to the non-factored 1-D path; a no-op on factored weights.
        momentum_dtype: storage for **both** momentum buffers — ``"bfloat16"``
            (default, ~2 B/param each), ``"float32"`` (4 B/param each), ``"int8"``
            (~1 B/param each, per-row absmax), or ``"4bit"`` (~0.5 B/param each,
            per-block absmax, nibble-packed). Same layout as Adakaon's first
            moment, so checkpoints resume bit-exactly via ``load_state_dict``. Note
            AdaPNM carries *two* momenta, so its momentum floor is ~2x a
            single-momentum optimizer at the same dtype (the price of PNM).
        momentum_4bit_block: block size for ``momentum_dtype="4bit"``. Default
            ``128``. ``0``/negative means whole-tensor.
        bf16_method: weight-update strategy for low-precision params —
            ``"stochastic_rounding"`` (default), ``"kahan"`` (+2 B/param), or
            ``"none"``. No-op on fp32 params.
        foreach: batch the step across parameters with stacked multi-tensor ops.
            Default ``True``. Numerically matches the per-parameter path
            (stochastic-rounding draws differ, unbiased either way). 0-D scalars,
            kahan, and fp16+SR fall back to the per-parameter path.
        foreach_batch_cutoff: per-tensor element count above which a weight loops
            instead of stacking (a performance knob; default ``2_000_000``).
        foreach_stack_budget: max elements per stacked chunk. ``None`` (default)
            adapts to free VRAM; an int pins a fixed cap.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.8, 0.999),
        beta0: float = 0.5,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        clip_threshold: float = 1.0,
        cautious: bool = True,
        gradient_centralization: bool = True,
        ams_bound: bool = False,
        momentum_dtype: MomentumDtype = "bfloat16",
        momentum_4bit_block: int = _FOURBIT_BLOCK,
        bf16_method: str = "stochastic_rounding",
        foreach: bool = True,
        foreach_batch_cutoff: int = FOREACH_BATCH_CUTOFF,
        foreach_stack_budget: int | None = None,
        fused: bool = False,
        fused_tile_cap: int | None = None,
        auto_lr: bool = False,
        auto_lr_freeze: int | str | None = "auto",
        auto_lr_scale: float = 1.0,
        auto_lr_fuse_rel: float = 100.0,
    ) -> None:
        beta1, beta2 = float(betas[0]), float(betas[1])
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"betas[0] must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"betas[1] must be in [0, 1), got {beta2}")
        if not 0.0 <= beta0 <= 1.0:
            raise ValueError(f"beta0 must be in [0, 1], got {beta0}")
        if lr < 0.0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if eps < 0.0:
            raise ValueError(f"eps must be >= 0, got {eps}")
        if clip_threshold < 0.0:
            raise ValueError(f"clip_threshold must be >= 0, got {clip_threshold}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        if momentum_dtype not in ("bfloat16", "float32", "int8", "4bit"):
            raise ValueError(
                f"momentum_dtype must be bfloat16/float32/int8/4bit, got {momentum_dtype!r}"
            )
        if bf16_method not in ("stochastic_rounding", "kahan", "none"):
            raise ValueError(
                f"bf16_method must be stochastic_rounding/kahan/none, got {bf16_method!r}"
            )
        if foreach_batch_cutoff < 1:
            raise ValueError(f"foreach_batch_cutoff must be >= 1, got {foreach_batch_cutoff}")
        defaults = {
            "lr": lr,
            "betas": (beta1, beta2),
            "beta0": float(beta0),
            "eps": float(eps),
            "clip_threshold": float(clip_threshold),
            "weight_decay": weight_decay,
            "cautious": cautious,
            "gradient_centralization": gradient_centralization,
            "ams_bound": ams_bound,
            "momentum_dtype": momentum_dtype,
            "momentum_4bit_block": momentum_4bit_block,
            "bf16_method": bf16_method,
            "step": 0,
        }
        super().__init__(params, defaults)
        self._foreach = foreach
        self._foreach_batch_cutoff = foreach_batch_cutoff
        self._foreach_stack_budget = foreach_stack_budget
        # Optional Triton-fused step (same math + state). Eligible 2-D weights run through the fused
        # PNM kernels (one-block / chunked); everything else falls back to the native path, in-place
        # on the SAME state, so fused and non-fused interoperate and resume from each other.
        self._fused = bool(fused)
        # When True, the many-same-shape big regime (>tile_cap) runs the batched chunked kernel; set
        # False to revert to the batched-native-foreach path (the A/B baseline). See _fused_big.
        self._fused_big_batched = True
        # EXPERIMENTAL (candidate #4): fuse the batched-big reductions into Triton (grad via pointer
        # array, no [N,R,C] stack, GC in-kernel). Default False until the A/B confirms a win.
        self._fused_reductions = True
        self._fused_part: dict[int, tuple] = {}
        self._fused_ob_caches: dict[int, Any] = {}
        self._fused_od_caches: dict[int, Any] = {}       # group id -> OneDimPnmCache (1-D)
        if self._fused:
            from kaon._fused_triton import HAS_TRITON, TILE_CAP
            if not HAS_TRITON:
                raise RuntimeError("AdaPNM(fused=True) requires Triton (a GPU-only optional dependency)")
            self._fused_tile_cap = TILE_CAP if fused_tile_cap is None else fused_tile_cap

        # Composable parameter-free LR (update-space DoWG) via AutoLRMixin. off -> zero overhead.
        self._init_autolr(auto_lr, auto_lr_freeze, auto_lr_scale, auto_lr_fuse_rel)

    # ------------------------------------------------------------------- state
    @torch.no_grad()
    def _alloc_momentum(
        self, prefix: str, grad: Tensor, state: dict[str, Any], group: dict[str, Any]
    ) -> None:
        """Allocate one momentum buffer (keys ``f"{prefix}"``, ``f"{prefix}_scale"`` …).

        Storage layout matches :mod:`kaon._momentum_codec` exactly (per-row int8
        scale; per-block 4-bit scale, zero == nibble 8) so the two momenta resume
        bit-exactly via ``load_state_dict_preserving_dtypes``.
        """
        md = group["momentum_dtype"]
        if md in ("bfloat16", "float32"):
            dtype = torch.bfloat16 if md == "bfloat16" else torch.float32
            state[prefix] = torch.zeros_like(grad, dtype=dtype)
        elif md == "int8":
            state[prefix] = torch.zeros_like(grad, dtype=torch.int8)
            state[f"{prefix}_scale"] = torch.ones(
                (grad.shape[0],) + (1,) * (grad.ndim - 1) if grad.ndim >= 2 else (),
                dtype=torch.float32, device=grad.device,
            )
        else:  # 4bit
            numel = grad.numel()
            bs = fourbit_block_size(grad, group)
            nblocks = (numel + bs - 1) // bs
            state[prefix] = torch.full(
                ((numel + 1) // 2,), 0x88, dtype=torch.uint8, device=grad.device
            )
            state[f"{prefix}_scale"] = torch.ones(nblocks, dtype=torch.float32, device=grad.device)
            state[f"{prefix}_numel"] = numel
            state[f"{prefix}_block"] = bs

    @torch.no_grad()
    def _init_state(self, p: Tensor, state: dict[str, Any], group: dict[str, Any]) -> None:
        grad = p.grad
        factored = p.ndim >= 2
        if factored:
            gv = grad if p.ndim == 2 else grad.reshape(grad.shape[0], -1)
            row_shape = gv.shape[:-1]
            col_shape = gv.shape[:-2] + gv.shape[-1:]
            state["row"] = torch.zeros(row_shape, dtype=torch.float32, device=p.device)
            state["col"] = torch.zeros(col_shape, dtype=torch.float32, device=p.device)
        else:
            state["v"] = torch.zeros_like(grad, dtype=torch.float32)
            if group["ams_bound"]:
                state["max_v"] = torch.zeros_like(grad, dtype=torch.float32)
        # Two momenta (pos / neg), each through the shared codec layout.
        self._alloc_momentum("m_pos", grad, state, group)
        self._alloc_momentum("m_neg", grad, state, group)
        if is_low_precision(p) and group["bf16_method"] == "kahan":
            state["shift"] = torch.zeros_like(p)

    # -------------------------------------------------- momentum read / write
    @staticmethod
    def _dequant_one(state: dict[str, Any], prefix: str, md: str, like: Tensor) -> Tensor:
        """Read one stored momentum back as a fresh fp32 tensor shaped like ``like``.

        Buffers are stored in the param's original shape; conv kernels are
        matrixized at use-site so ``like`` may be the ``[R, C]`` view — reshape to
        match (the per-row int8 scale's row grouping is preserved across the
        reshape because dim-0 is unchanged).
        """
        if md in ("bfloat16", "float32"):
            return state[prefix].float().reshape_as(like)
        if md == "int8":
            return state[prefix].float().mul_(state[f"{prefix}_scale"]).reshape_as(like)
        m = _dequant_4bit(
            state[prefix], state[f"{prefix}_scale"], state[f"{prefix}_numel"], state[f"{prefix}_block"]
        )
        return m.view_as(like)

    @staticmethod
    def _store_one(state: dict[str, Any], prefix: str, md: str, m_fp32: Tensor) -> None:
        """Write an updated fp32 momentum back into the configured storage layout.

        ``m_fp32`` may be the matrixized ``[R, C]`` view; the int8 per-row scale and
        the float buffer both reduce/store over the param's *original* shape, so
        reshape back first (dim-0 — the int8 row axis — is preserved).
        """
        if md in ("bfloat16", "float32"):
            tgt = state[prefix]
            tgt.copy_(m_fp32.reshape(tgt.shape))
        elif md == "int8":
            m_orig = m_fp32.reshape(state[prefix].shape)
            state[prefix], state[f"{prefix}_scale"] = _quant_int8(m_orig)
        else:  # 4bit
            packed, scale, _ = _quant_4bit(m_fp32, state[f"{prefix}_block"])
            state[prefix], state[f"{prefix}_scale"] = packed, scale

    @staticmethod
    def _dequant_stacked(
        states: list[dict[str, Any]], prefix: str, md: str, shape: tuple[int, ...]
    ) -> Tensor:
        """Stacked fp32 momentum ``[N, *shape]`` from per-param storage (see Lion)."""
        n = len(states)
        per = math.prod(shape)
        if md in ("bfloat16", "float32"):
            # Buffers are stored in the param's original shape; reshape to the
            # effective (matrixized [R, C] / flat [L]) view so stacking aligns.
            return torch.stack([s[prefix].reshape(shape) for s in states]).float()
        if md == "int8":
            row = shape[0] if len(shape) >= 2 else 1
            rest = max(per // row, 1)
            m = torch.stack([s[prefix].reshape(row, rest) for s in states]).float()  # [N, R, rest]
            scale = torch.stack([s[f"{prefix}_scale"].reshape(row, 1) for s in states])  # [N, R, 1]
            return m.mul_(scale).reshape((n, *shape))
        packed = torch.stack([s[prefix] for s in states])
        sc = torch.stack([s[f"{prefix}_scale"] for s in states])
        bs = states[0][f"{prefix}_block"]
        return _dequant_4bit_stacked(packed, sc, per, bs).reshape((n, *shape))

    @staticmethod
    def _store_stacked(
        states: list[dict[str, Any]], prefix: str, md: str, m_fp32: Tensor
    ) -> None:
        """Write stacked fp32 momentum ``[N, *shape]`` back into per-param storage."""
        n = m_fp32.shape[0]
        shape = tuple(m_fp32.shape[1:])
        per = math.prod(shape)
        if md in ("bfloat16", "float32"):
            ms = [s[prefix].reshape(shape) for s in states]
            torch._foreach_copy_(ms, list(m_fp32.unbind(0)))
        elif md == "int8":
            row = shape[0] if len(shape) >= 2 else 1
            rest = max(per // row, 1)
            q, new_scale = _quant_int8_stacked(m_fp32.reshape(n, row, rest))  # [N,R,rest]->[N,R,1]
            torch._foreach_copy_(
                [s[prefix].reshape(row, rest) for s in states], list(q.unbind(0))
            )
            for s, sc in zip(states, new_scale.unbind(0), strict=True):  # sc: [R, 1]
                s[f"{prefix}_scale"] = sc.reshape(row, 1) if len(shape) >= 2 else sc.reshape(1)
        else:  # 4bit
            bs = states[0][f"{prefix}_block"]
            new_packed, new_scale = _quant_4bit_stacked(m_fp32.reshape(n, per), bs)
            torch._foreach_copy_([s[prefix] for s in states], list(new_packed.unbind(0)))
            for s, sc in zip(states, new_scale.unbind(0), strict=True):
                s[f"{prefix}_scale"].copy_(sc)

    @staticmethod
    def _pos_neg_prefixes(step: int) -> tuple[str, str]:
        """Which stored buffer plays positive / negative this step (the alternation).

        On odd steps ``m_pos`` is positive; on even steps the roles swap, so the
        buffer that received the gradient last step becomes the (stale) negative
        momentum that is subtracted. ``step`` is 1-indexed (incremented before use).
        """
        return ("m_pos", "m_neg") if step % 2 == 1 else ("m_neg", "m_pos")

    # -------------------------------------------------------------------- step
    @torch.no_grad()
    def _step_impl(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self._fused:
            return self._fused_step(loss)
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            for p in params:
                if p.grad.is_sparse:
                    raise RuntimeError("AdaPNM does not support sparse gradients")
            group["step"] += 1
            if not params:
                continue
            if group["gradient_centralization"]:
                centralize_grads_(params)
            self._native_dispatch(params, group)
            if _PROBE_LOG:
                _probe_group(self, group)
        return loss

    @torch.no_grad()
    def _native_dispatch(self, params: list[Tensor], group: dict[str, Any]) -> None:
        """The native (non-fused) step over ``params`` — foreach where eligible, else per-param.
        Gradient Centralization is the caller's responsibility (done per-subset)."""
        if not params:
            return
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
                for p in params:
                    self._step_one_param(p, group)
        else:
            for p in params:
                self._step_one_param(p, group)

    # ----------------------------------------------------------- fused (Triton) step
    @torch.no_grad()
    def _fused_step(self, loss: Any) -> Any:
        import kaon._fused_triton as ft

        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            for p in params:
                if p.grad.is_sparse:
                    raise RuntimeError("AdaPNM does not support sparse gradients")
            group["step"] += 1
            if not params:
                continue
            c = self._coeffs(group)
            pos_pref, neg_pref = self._pos_neg_prefixes(group["step"])
            one_block, big, one_dim, native = self._fused_partition(group, params, ft)
            if native:
                if group["gradient_centralization"]:
                    centralize_grads_(native)
                self._native_dispatch(native, group)
            if one_block:
                self._fused_one_block(one_block, group, ft, c)
            if big:
                self._fused_big(big, group, ft, c, pos_pref, neg_pref)
            if one_dim:
                self._fused_one_dim(one_dim, group, ft, c, pos_pref, neg_pref)
            if _PROBE_LOG:
                _probe_group(self, group)
        return loss

    def _fused_partition(self, group: dict[str, Any], params: list[Tensor], ft: Any) -> tuple:
        gid = id(group)
        ids = tuple(id(p) for p in params)
        cached = self._fused_part.get(gid)
        if cached is not None and cached[0] == ids:
            return cached[1], cached[2], cached[3], cached[4]
        md, bf16m, cap = group["momentum_dtype"], group["bf16_method"], self._fused_tile_cap
        float_mom = md in ("bfloat16", "float32")  # the 1-D kernel handles only fp32/bf16 momentum
        no_ams = not group["ams_bound"]            # ams_bound 1-D (full max_v) -> native
        one_block: list[Tensor] = []
        big: list[Tensor] = []
        one_dim: list[Tensor] = []
        native: list[Tensor] = []
        for p in params:
            bf_ok = (p.dtype != torch.bfloat16) or (bf16m == "stochastic_rounding")
            # ndim>2 (conv) is matrixized to (out, in*kh*kw); needs a contiguous grad + fp32/bf16
            # momentum (quant's per-row requant would reshape the conv state) -> else native.
            conv_ok = p.ndim <= 2 or (p.grad is not None and p.grad.is_contiguous() and float_mom)
            ok = bf_ok and conv_ok and ft.fused_eligible(p, cap)
            if ok and md == "4bit" and ft.eff_2d(p)[1] % 2 != 0:
                ok = False
            big_ok = (bf_ok and conv_ok and p.ndim >= 2 and p.is_cuda and p.is_contiguous()
                      and p.dtype in (torch.float32, torch.bfloat16)
                      and ft.next_pow2_tile(*ft.eff_2d(p))[0] * ft.next_pow2_tile(*ft.eff_2d(p))[1] > cap)
            if ok:
                one_block.append(p)
            elif big_ok:
                big.append(p)
            elif bf_ok and float_mom and no_ams and ft.fused_1d_eligible(p, cap):
                one_dim.append(p)
            else:
                native.append(p)
        self._fused_part[gid] = (ids, one_block, big, one_dim, native)
        if _PROBE_LOG:
            _probe_census(one_block, big, native, md, bf16m, cap, ft)
        return one_block, big, one_dim, native

    def _fused_one_block(self, plist: list[Tensor], group: dict[str, Any], ft: Any, c: dict) -> None:
        for p in plist:
            st = self.state[p]
            if not st:
                self._init_state(p, st, group)
        ids = tuple(id(p) for p in plist)
        gid = id(group)
        cache = self._fused_ob_caches.get(gid)
        if cache is None or cache.ids != ids:
            cache = ft.AdaPnmCache(plist, lambda p: self.state[p])
            self._fused_ob_caches[gid] = cache
        cache.refresh_grads()
        odd = group["step"] % 2 == 1
        lr, wd, eps1 = group["lr"], group["weight_decay"], group["eps"]
        cautious, gc = group["cautious"], group["gradient_centralization"]
        clip = group["clip_threshold"]
        sc, inv_noise = c["bc2_sq"] * c["step_size"], 1.0 / c["noise_norm"]
        clip_eff = clip * c["step_size"]        # rms(upd) <= clip*step_size == rms(pn/sqrt(v_hat)) <= clip
        for bk in cache.buckets:
            # which physical buffer plays positive this step (alternation): the m_pos slot if odd.
            if odd:
                kpos, kneg, kposc, knegc = bk["pos_addr"], bk["neg_addr"], bk["posc_addr"], bk["negc_addr"]
            else:
                kpos, kneg, kposc, knegc = bk["neg_addr"], bk["pos_addr"], bk["negc_addr"], bk["posc_addr"]
            lanes = bk["BR"] * bk["BC"]
            ft._adapnm_tile_kernel[(len(bk["plist"]),)](
                bk["g_addr"], bk["p_addr"], kpos, kneg, kposc, knegc, bk["row_addr"], bk["col_addr"],
                bk["Rs"], bk["Cs"], c["beta1_sq"], c["beta0"], inv_noise, c["beta2"], sc, lr * wd, eps1,
                clip_eff, group["step"], LOWP=bk["lowp"], MOM=bk["mom"], CAUTIOUS=cautious, WD=wd != 0,
                GC=gc, SR=bk["lowp"], CLIP=clip > 0.0, BR=bk["BR"], BC=bk["BC"], num_warps=ft.warps_for(lanes),
            )

    def _fused_one_dim(self, plist: list[Tensor], group: dict[str, Any], ft: Any, c: dict,
                       pos_pref: str, neg_pref: str) -> None:
        """One-block non-factored kernel over eligible 1-D weights (biases / norm scales). GC is a no-op
        on 1-D. fp32/bf16 momenta only; ams_bound and quant route to native (excluded in the partition)."""
        for p in plist:
            st = self.state[p]
            if not st:
                self._init_state(p, st, group)
        ids = tuple(id(p) for p in plist)
        gid = id(group)
        cache = self._fused_od_caches.get(gid)
        if cache is None or cache.ids != ids:
            cache = ft.OneDimPnmCache(plist, lambda p: self.state[p])
            self._fused_od_caches[gid] = cache
        cache.refresh_grads()
        odd = group["step"] % 2 == 1
        lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
        cautious, clip = group["cautious"], group["clip_threshold"]
        inv_noise = 1.0 / c["noise_norm"]
        for bk in cache.buckets:
            # which physical buffer plays positive this step (alternation): the m_pos slot if odd.
            kpos, kneg = (bk["pos_addr"], bk["neg_addr"]) if odd else (bk["neg_addr"], bk["pos_addr"])
            ft._adapnm_1d_kernel[(len(bk["plist"]),)](
                bk["g_addr"], bk["p_addr"], kpos, kneg, bk["v_addr"], bk["Ls"],
                c["beta1_sq"], c["beta0"], inv_noise, c["beta2"], c["step_size"], c["bc2_sq"], eps,
                lr * wd, clip, group["step"], LOWP=bk["lowp"], MOM=bk["mom"], CAUTIOUS=cautious,
                WD=wd != 0, CLIP=clip > 0.0, SR=bk["lowp"], BL=bk["BL"], num_warps=ft.warps_for(bk["BL"]),
            )

    @torch.no_grad()
    def _fused_big(self, big: list[Tensor], group: dict[str, Any], ft: Any, c: dict,
                   pos_pref: str, neg_pref: str) -> None:
        """Dispatch the >tile-cap ("big") 2-D factors.

        The per-tensor fused-chunked kernel is launch-bound (~8 kernels/tensor): for the many
        same-shape big factors a real LoKr run has (e.g. 236x 512x512), the **batched native
        foreach** path is ~5x faster (measured 14ms vs 69ms) because it stacks same-shape tensors
        and amortizes launches. A lone big tensor has no batch to amortize, so it keeps the
        fused-chunked kernel. Same math + state either way (both RMS-clip), so they interoperate.
        """
        if len(big) >= 2 and not self._fused_big_batched:
            if group["gradient_centralization"]:
                centralize_grads_(big)
            self._native_dispatch(big, group)
            return
        by_shape: dict[tuple[int, int], list[Tensor]] = {}
        for p in big:
            by_shape.setdefault(tuple(p.shape), []).append(p)
        for plist in by_shape.values():
            if len(plist) >= 2:
                self._chunked_step_batched(plist, group, ft, c, pos_pref, neg_pref)
            else:
                self._chunked_step(plist[0], group, ft, c, pos_pref, neg_pref)

    def _chunked_reductions(self, p: Tensor, group: dict[str, Any], st: dict[str, Any]) -> tuple:
        b2, eps1 = group["betas"][1], group["eps"]
        g = p.grad.float().reshape(p.shape[0], p.numel() // p.shape[0])  # conv -> matrixized (out, in*kh*kw)
        if group["gradient_centralization"]:
            g = g - g.mean(dim=1, keepdim=True)
        g = g.contiguous()
        gsq = g * g
        st["row"].lerp_(gsq.mean(1).add_(eps1), 1.0 - b2)
        st["col"].lerp_(gsq.mean(0).add_(eps1), 1.0 - b2)
        return g, st["row"].div(st["row"].mean()).rsqrt_(), st["col"].rsqrt()

    def _chunked_step(self, p: Tensor, group: dict[str, Any], ft: Any, c: dict,
                      pos_pref: str, neg_pref: str) -> None:
        st = self.state[p]
        if not st:
            self._init_state(p, st, group)
        R, C = p.shape[0], p.numel() // p.shape[0]  # noqa: N806 — matrix dims (conv -> matrixized)
        n = R * C
        md = group["momentum_dtype"]
        lr, wd, cautious = group["lr"], group["weight_decay"], group["cautious"]
        sr = (p.dtype == torch.bfloat16) and (group["bf16_method"] == "stochastic_rounding")
        g, r, cfac = self._chunked_reductions(p, group, st)
        m_pos = self._dequant_one(st, pos_pref, md, g).reshape(R, C)   # fp32 temp (EMA'd in-kernel)
        m_neg = self._dequant_one(st, neg_pref, md, g).reshape(R, C)   # fp32 temp (read-only)
        sc, inv_noise = c["bc2_sq"] * c["step_size"], 1.0 / c["noise_norm"]
        keep = torch.zeros(1, dtype=torch.int32, device=p.device)
        gf, posf, negf, pf = g.reshape(-1), m_pos.reshape(-1), m_neg.reshape(-1), p.reshape(-1)
        grid = ((n + 1023) // 1024,)
        ft._adapnm_chunked_mom[grid](gf, posf, negf, pf, r, cfac, keep, C, n, c["beta1_sq"], c["beta0"],
                                     inv_noise, sc, lr * wd, CAUTIOUS=cautious, WD=wd != 0, BLOCK=1024)
        self._store_one(st, pos_pref, md, m_pos)                       # requant/store updated positive
        # Adafactor RMS-clip on the v_hat-normalized update U = pn * r * c * bc2_sq (== delta/step_size):
        # bound rms(U) <= clip by folding 1/max(rms(U)/clip, 1) into the lr-scale `sc` (one [R,C] temp).
        sc_apply = sc
        clip = group["clip_threshold"]
        if clip > 0.0:
            u = m_pos.mul(1.0 + c["beta0"]).sub_(m_neg, alpha=c["beta0"])   # pn numerator (fresh temp)
            u.mul_(r.reshape(R, 1)).mul_(cfac.reshape(1, C))
            rms_u = float(u.norm()) * inv_noise * c["bc2_sq"] / math.sqrt(n)
            sc_apply = sc / max(rms_u / clip, 1.0)
        inv_mean = 1.0 / max(keep.item() / n, 1e-8) if cautious else 1.0
        ft._adapnm_chunked_apply[grid](gf, posf, negf, pf, r, cfac, C, n, c["beta0"], inv_noise, sc_apply,
                                       lr * wd, inv_mean, group["step"], CAUTIOUS=cautious, WD=wd != 0,
                                       SR=sr, BLOCK=1024)

    # ----------------------------------------------- batched chunked (many same-shape big tensors)
    @torch.no_grad()
    def _chunked_reductions_batched(self, plist: list[Tensor], group: dict[str, Any]) -> tuple:
        """Stacked AdaPNM reductions for a same-shape big bucket: GC (on the fp32 copy) + row/col EMA.
        Returns the stacked fp32 grad ``[N, R, C]`` and the stacked r/c factors ``[N, R]``/``[N, C]``
        (contiguous). Mirrors :meth:`_chunked_reductions` per tensor (eps1 on the means; no rms — the
        AdaPNM clip is computed separately, folded into ``sc`` in :meth:`_chunked_step_batched`)."""
        b2, eps1 = group["betas"][1], group["eps"]
        states = [self.state[p] for p in plist]
        R, C = plist[0].shape[0], plist[0].numel() // plist[0].shape[0]    # noqa: N806 — conv -> matrixized
        g = torch.stack([p.grad.float().reshape(R, C) for p in plist])     # [N, R, C]
        if group["gradient_centralization"]:
            g.sub_(g.mean(dim=-1, keepdim=True))
        gsq = g * g
        row = torch.stack([s["row"] for s in states])                    # [N, R]
        col = torch.stack([s["col"] for s in states])                    # [N, C]
        row.lerp_(gsq.mean(dim=-1).add_(eps1), 1.0 - b2)
        col.lerp_(gsq.mean(dim=-2).add_(eps1), 1.0 - b2)
        torch._foreach_copy_([s["row"] for s in states], list(row.unbind(0)))
        torch._foreach_copy_([s["col"] for s in states], list(col.unbind(0)))
        r = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_()             # [N, R]
        c = col.rsqrt()                                                  # [N, C]
        return g, r.contiguous(), c.contiguous()

    @torch.no_grad()
    def _chunked_step_batched(self, plist: list[Tensor], group: dict[str, Any], ft: Any, c: dict,
                              pos_pref: str, neg_pref: str) -> None:
        """A bucket of >=2 same-shape big 2-D tensors via the batched AdaPNM chunked kernels (~2
        launches). fp32/bf16 momenta are read/written in place via the pos/neg pointer arrays (no
        temps); int8/4bit dequant to fp32 temps, step on them, requant the positive between passes.
        The Adafactor RMS-clip's per-tensor sum-of-squares is accumulated in-kernel (no torch momentum
        temp in the float case), turned into ``sc_apply[N]`` between passes."""
        for p in plist:
            st = self.state[p]
            if not st:
                self._init_state(p, st, group)
        N = len(plist)  # noqa: N806
        R, C = plist[0].shape[0], plist[0].numel() // plist[0].shape[0]  # noqa: N806 — conv -> matrixized
        n = R * C
        dev = plist[0].device
        md = group["momentum_dtype"]
        lr, wd, cautious = group["lr"], group["weight_decay"], group["cautious"]
        clip = group["clip_threshold"]
        gc = group["gradient_centralization"]
        lowp = plist[0].dtype == torch.bfloat16
        sr = lowp and (group["bf16_method"] == "stochastic_rounding")
        states = [self.state[p] for p in plist]
        fused_red = self._fused_reductions
        if fused_red:  # grad via pointer array, no [N,R,C] stack (candidate #4); rowmean carries GC
            g_addr, rowmean, r, cfac = self._chunked_reductions_fused(plist, group, ft, R, C, lowp)
        else:
            g, r, cfac = self._chunked_reductions_batched(plist, group)   # g [N,R,C], r [N,R], c [N,C]
            gf = g.reshape(-1)
        sc, inv_noise = c["bc2_sq"] * c["step_size"], 1.0 / c["noise_norm"]

        quant = md in ("int8", "4bit")
        if quant:  # dequant both momenta to stacked fp32 temps; arrays point at the temp slices
            pos_temp = torch.empty(N, R, C, dtype=torch.float32, device=dev)
            neg_temp = torch.empty(N, R, C, dtype=torch.float32, device=dev)
            for i, st in enumerate(states):
                pos_temp[i] = self._dequant_one(st, pos_pref, md, pos_temp[i])
                neg_temp[i] = self._dequant_one(st, neg_pref, md, neg_temp[i])
            pos_addr = ft.ptr_array(list(pos_temp), dev)
            neg_addr = ft.ptr_array(list(neg_temp), dev)
            mom = ft.MOM_FP32
        else:      # fp32/bf16: pointer arrays straight to the stored buffers (EMA positive in place)
            pos_addr = ft.ptr_array([s[pos_pref] for s in states], dev)
            neg_addr = ft.ptr_array([s[neg_pref] for s in states], dev)
            mom = ft.MOM_BF16 if md == "bfloat16" else ft.MOM_FP32
        p_addr = ft.ptr_array(plist, dev)
        keep = torch.zeros(N, dtype=torch.int32, device=dev)
        rms_acc = torch.zeros(N, dtype=torch.float32, device=dev)
        K = (n + 1023) // 1024  # noqa: N806
        grid = (N * K,)
        if fused_red:
            ft._adapnm_chunked_mom_batched_g[grid](
                g_addr, rowmean, pos_addr, neg_addr, r, cfac, keep, rms_acc, R, C, n, K,
                c["beta1_sq"], c["beta0"], inv_noise,
                LOWP=lowp, MOM=mom, GC=gc, CAUTIOUS=cautious, CLIP=clip > 0.0, BLOCK=1024,
            )
        else:
            ft._adapnm_chunked_mom_batched[grid](
                gf, pos_addr, neg_addr, r, cfac, keep, rms_acc, R, C, n, K, c["beta1_sq"], c["beta0"],
                inv_noise, MOM=mom, CAUTIOUS=cautious, CLIP=clip > 0.0, BLOCK=1024,
            )
        if quant:  # requant the updated positive temp back into storage (apply reads the temp)
            for i, st in enumerate(states):
                self._store_one(st, pos_pref, md, pos_temp[i])
        # Per-tensor Adafactor RMS-clip: rms_u = bc2_sq * sqrt(rms_acc / n); fold into sc_apply[N].
        if clip > 0.0:
            rms_u = rms_acc.div_(n).sqrt_().mul_(c["bc2_sq"])             # [N]
            sc_apply = (sc / rms_u.div_(clip).clamp_(min=1.0)).contiguous()
        else:
            sc_apply = torch.full((N,), sc, dtype=torch.float32, device=dev)
        inv_mean = (1.0 / (keep.float() / n).clamp_(min=1e-8)) if cautious else torch.ones(N, device=dev)
        if fused_red:
            ft._adapnm_chunked_apply_batched_g[grid](
                g_addr, rowmean, pos_addr, neg_addr, p_addr, r, cfac, sc_apply, inv_mean, R, C, n, K,
                c["beta0"], inv_noise, lr * wd, group["step"],
                LOWP=lowp, MOM=mom, GC=gc, CAUTIOUS=cautious, WD=wd != 0, SR=sr, BLOCK=1024,
            )
        else:
            ft._adapnm_chunked_apply_batched[grid](
                gf, pos_addr, neg_addr, p_addr, r, cfac, sc_apply, inv_mean, R, C, n, K, c["beta0"],
                inv_noise, lr * wd, group["step"], LOWP=lowp, MOM=mom, CAUTIOUS=cautious, WD=wd != 0,
                SR=sr, BLOCK=1024,
            )

    @torch.no_grad()
    def _chunked_reductions_fused(self, plist, group, ft, R, C, lowp):  # noqa: N803
        """Candidate #4 for AdaPNM: row/col EMA factors via the Triton reduction kernel reading grad
        from a pointer array (no [N,R,C] stack; GC in-kernel). No rms here — AdaPNM's clip is computed
        in the mom kernel. Returns (g_addr, rowmean, r_factor[N,R], c_factor[N,C])."""
        b2, eps1 = group["betas"][1], group["eps"]
        N = len(plist)  # noqa: N806
        dev = plist[0].device
        states = [self.state[p] for p in plist]
        g_addr = ft.ptr_array([p.grad for p in plist], dev)
        BR, BC, RB = ft.reduction_tile(R, C)  # noqa: N806
        rowmean = torch.empty(N * R, dtype=torch.float32, device=dev)
        rowsum = torch.empty(N * R, dtype=torch.float32, device=dev)
        colsum = torch.zeros(N * C, dtype=torch.float32, device=dev)
        ft._reduce_rowcol[(N * RB,)](
            g_addr, rowmean, rowsum, colsum, R, C, RB,
            LOWP=lowp, GC=group["gradient_centralization"], BR=BR, BC=BC, num_warps=ft.warps_for(BR * BC),
        )
        rowsum = rowsum.view(N, R)
        colsum = colsum.view(N, C)
        row = torch.stack([s["row"] for s in states])
        col = torch.stack([s["col"] for s in states])
        row.lerp_(rowsum.div(C).add_(eps1), 1.0 - b2)
        col.lerp_(colsum.div(R).add_(eps1), 1.0 - b2)
        torch._foreach_copy_([s["row"] for s in states], list(row.unbind(0)))
        torch._foreach_copy_([s["col"] for s in states], list(col.unbind(0)))
        r = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_().contiguous()
        cfac = col.rsqrt().contiguous()
        return g_addr, rowmean, r, cfac

    def state_dict(self) -> dict[str, Any]:
        """Base state + the auto_lr tuner blob (via AutoLRMixin) when auto_lr is on."""
        return self._autolr_state_dict(super().state_dict())

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state, preserving both quantized momenta's stored dtype.

        torch's default ``load_state_dict`` upcasts every state tensor to the
        param's dtype (fp32), which would silently inflate bf16/int8/4bit momenta
        back to fp32 on resume. Delegate to the shared helper that restores each
        tensor to how it was checkpointed.
        """
        self._autolr_load(state_dict, lambda sd: load_state_dict_preserving_dtypes(self, sd))

    # ----------------------------------------------------------- coefficients
    @staticmethod
    def _coeffs(group: dict[str, Any]) -> dict[str, float]:
        """All per-step scalar coefficients (shared by the per-param and foreach paths)."""
        beta1, beta2 = group["betas"]
        beta0 = group["beta0"]
        step = group["step"]
        beta1_sq = beta1 * beta1
        noise_norm = math.sqrt((1.0 + beta0) ** 2 + beta0 ** 2)
        bc1 = 1.0 - beta1 ** step          # bias correction uses beta1, NOT beta1^2
        bc2_sq = math.sqrt(1.0 - beta2 ** step)
        return {
            "beta1_sq": beta1_sq,
            "beta2": beta2,
            "beta0": beta0,
            "noise_norm": noise_norm,
            "bc1": bc1,
            "bc2_sq": bc2_sq,
            "step_size": group["lr"] / bc1,
        }

    # ----------------------------------------------------------------- foreach

    @staticmethod
    def _group_foreach_eligible(group: dict[str, Any]) -> bool:
        return group["bf16_method"] != "kahan"  # kahan needs a per-param shift buffer

    @staticmethod
    def _param_foreach_eligible(p: Tensor, group: dict[str, Any], cutoff: int) -> bool:
        if p.ndim == 0 or p.numel() > cutoff:
            return False
        if (
            group["bf16_method"] == "stochastic_rounding"
            and is_low_precision(p)
            and p.dtype != torch.bfloat16  # fp16+SR unsupported -> per-param (raises)
        ):
            return False
        if p.ndim > 2:
            # Matrixized conv writes back through a reshaped view -> needs contiguity.
            return p.data.is_contiguous() and p.grad.is_contiguous()
        return True

    @torch.no_grad()
    def _step_foreach(self, params: list[Tensor], group: dict[str, Any], budget: int) -> None:
        """Batched step. Factored (ndim>=2) and non-factored (ndim==1) buckets, by shape."""
        c = self._coeffs(group)
        md = group["momentum_dtype"]
        pos, neg = self._pos_neg_prefixes(group["step"])

        factored_buckets: dict[tuple[Any, ...], list[Tensor]] = {}
        flat_buckets: dict[tuple[Any, ...], list[Tensor]] = {}
        for p in params:
            state = self.state[p]
            if not state:
                self._init_state(p, state, group)
            g = p.grad
            if g.ndim >= 2:
                matrixize = g.ndim > 2
                eff = (g.shape[0], g.numel() // g.shape[0]) if matrixize else tuple(g.shape)
                factored_buckets.setdefault((eff, p.dtype, matrixize), []).append(p)
            else:
                flat_buckets.setdefault((g.shape[0], p.dtype), []).append(p)

        for (eff, _dtype, matrixize), plist in factored_buckets.items():
            stepn = max(1, budget // max(eff[0] * eff[1], 1))
            for i in range(0, len(plist), stepn):
                self._factored_bucket(plist[i:i + stepn], eff, matrixize, md, pos, neg, c, group)
        for (length, _dtype), plist in flat_buckets.items():
            stepn = max(1, budget // max(length, 1))
            for i in range(0, len(plist), stepn):
                self._nonfactored_bucket(plist[i:i + stepn], length, md, pos, neg, c, group)

    @torch.no_grad()
    def _factored_bucket(
        self,
        plist: list[Tensor],
        eff: tuple[int, int],
        matrixize: bool,
        md: str,
        pos: str,
        neg: str,
        c: dict[str, float],
        group: dict[str, Any],
    ) -> None:
        R, C = eff  # noqa: N806
        eps1 = group["eps"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]

        def mat(t: Tensor) -> Tensor:
            return t.view(R, C) if matrixize else t

        states = [self.state[p] for p in plist]
        rows = [s["row"] for s in states]
        cols = [s["col"] for s in states]

        grad = torch.stack([mat(p.grad).float() for p in plist])          # [N, R, C]
        row = torch.stack(rows)                                           # [N, R]
        col = torch.stack(cols)                                           # [N, C]

        # Decoupled weight decay BEFORE moment updates (kozistr order): p *= (1 - lr*wd).
        if wd != 0:
            self._apply_decoupled_wd_batched(plist, mat, group["lr"] * wd)

        # Factored second-moment EMA (HF eps1 placement).
        omb2 = 1.0 - c["beta2"]
        grad_sq = grad * grad
        if eps1 > 0:
            grad_sq = grad_sq.add_(eps1)
        row.lerp_(grad_sq.mean(dim=-1), omb2)
        col.lerp_(grad_sq.mean(dim=-2), omb2)
        torch._foreach_copy_(rows, list(row.unbind(0)))
        torch._foreach_copy_(cols, list(col.unbind(0)))

        r_factor = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)  # [N, R, 1]
        c_factor = col.rsqrt().unsqueeze(-2)                                       # [N, 1, C]
        inv_denom = (r_factor * c_factor).mul_(c["bc2_sq"])                        # 1/sqrt(v_hat)

        # Positive-negative momentum mixing (read both, EMA only the positive).
        pn = self._pn_stacked(states, pos, neg, md, (R, C), grad, c)               # [N, R, C]

        update = _rms_clip_batched_(pn.mul_(inv_denom), group["clip_threshold"])   # rms(u)<=clip
        delta = update.mul_(c["step_size"])                                        # full step

        if cautious:
            delta = cautious_batched_(delta, grad)

        subtract_batched_([mat(p.data) for p in plist], delta, bf16_method)

    @torch.no_grad()
    def _nonfactored_bucket(
        self,
        plist: list[Tensor],
        length: int,
        md: str,
        pos: str,
        neg: str,
        c: dict[str, float],
        group: dict[str, Any],
    ) -> None:
        eps1 = group["eps"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]
        ams_bound = group["ams_bound"]

        states = [self.state[p] for p in plist]
        vs = [s["v"] for s in states]

        grad = torch.stack([p.grad.float() for p in plist])               # [N, L]
        v = torch.stack(vs)                                               # [N, L]

        if wd != 0:
            self._apply_decoupled_wd_batched(plist, lambda t: t, group["lr"] * wd)

        # Full per-coordinate second moment (1-D). eps here goes on the denominator
        # (kozistr), NOT folded into grad^2; eps1==eps for the 1-D path.
        v.mul_(c["beta2"]).addcmul_(grad, grad, value=1.0 - c["beta2"])
        torch._foreach_copy_(vs, list(v.unbind(0)))

        if ams_bound:
            max_vs = [s["max_v"] for s in states]
            max_v = torch.stack(max_vs)
            torch.maximum(max_v, v, out=max_v)
            torch._foreach_copy_(max_vs, list(max_v.unbind(0)))
            de_nom = max_v.add(1e-15).sqrt_().add_(eps1)
        else:
            de_nom = v.add(1e-15).sqrt_().add_(eps1)
        de_nom.div_(c["bc2_sq"])                                          # v_hat denom

        pn = self._pn_stacked(states, pos, neg, md, (length,), grad, c)   # [N, L]
        update = _rms_clip_batched_(pn.div_(de_nom), group["clip_threshold"])
        delta = update.mul_(c["step_size"])

        if cautious:
            delta = cautious_batched_(delta, grad)

        subtract_batched_([p.data for p in plist], delta, bf16_method)

    def _pn_stacked(
        self,
        states: list[dict[str, Any]],
        pos: str,
        neg: str,
        md: str,
        shape: tuple[int, ...],
        grad: Tensor,
        c: dict[str, float],
    ) -> Tensor:
        """Stacked positive-negative momentum numerator (EMA the positive buffer).

        ``grad`` is the stacked, reshaped, post-WD gradient (``[N, *shape]``). Reads
        both momenta as fp32, EMA-updates only the positive buffer with the raw
        gradient (decay ``beta1**2``), stores it back, and returns the renormalized
        ``((1+beta0)*m_pos - beta0*m_neg)/noise_norm``.
        """
        n = grad.shape[0]
        m_pos = self._dequant_stacked(states, pos, md, shape).reshape((n, *shape))
        m_neg = self._dequant_stacked(states, neg, md, shape).reshape((n, *shape))
        m_pos.mul_(c["beta1_sq"]).add_(grad, alpha=1.0 - c["beta1_sq"])
        self._store_stacked(states, pos, md, m_pos.reshape((n, *shape)))
        # m_pos is a fresh stacked tensor (torch.stack copies) and is already stored, so
        # the pos-neg mix can run in-place on it — no extra [N, *shape] allocation.
        pn = m_pos.mul_(1.0 + c["beta0"]).add_(m_neg, alpha=-c["beta0"]).mul_(1.0 / c["noise_norm"])
        return pn

    @torch.no_grad()
    def _apply_decoupled_wd_batched(self, plist: list[Tensor], mat: Any, factor: float) -> None:
        """In-place decoupled WD ``p *= (1 - factor)`` on the (matrixized) weights."""
        scale = 1.0 - factor
        torch._foreach_mul_([mat(p.data) for p in plist], scale)

    # ---------------------------------------------------------- per-parameter
    @torch.no_grad()
    def _step_one_param(self, p: Tensor, group: dict[str, Any]) -> None:
        c = self._coeffs(group)
        md = group["momentum_dtype"]
        eps1 = group["eps"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]
        ams_bound = group["ams_bound"]
        pos, neg = self._pos_neg_prefixes(group["step"])

        state = self.state[p]
        if not state:
            self._init_state(p, state, group)

        grad = p.grad if p.grad.dtype == torch.float32 else p.grad.float()
        ndim = grad.ndim
        factored = ndim >= 2

        # Decoupled weight decay BEFORE the moment updates (kozistr order).
        if wd != 0:
            p.data.mul_(1.0 - group["lr"] * wd)

        if factored:
            matrixize = ndim > 2
            gv = grad.reshape(grad.shape[0], -1) if matrixize else grad
            update_factored_state(gv, state["row"], state["col"], c["beta2"], eps1)
            r_factor, c_factor = factored_inv_sqrt_factors(state["row"], state["col"])
            inv_denom = (r_factor * c_factor).mul_(c["bc2_sq"])           # 1/sqrt(v_hat)
            pn = self._pn_one(state, pos, neg, md, gv, c)
            update = _rms_clip_one_(pn.mul_(inv_denom), group["clip_threshold"])
            delta = update.mul_(c["step_size"])
            if matrixize:
                delta = delta.reshape_as(grad)
        else:
            v = state["v"]
            v.mul_(c["beta2"]).addcmul_(grad, grad, value=1.0 - c["beta2"])
            if ams_bound:
                max_v = state["max_v"]
                torch.maximum(max_v, v, out=max_v)
                de_nom = max_v.add(1e-15).sqrt_().add_(eps1)
            else:
                de_nom = v.add(1e-15).sqrt_().add_(eps1)
            de_nom.div_(c["bc2_sq"])
            pn = self._pn_one(state, pos, neg, md, grad, c)
            update = _rms_clip_one_(pn.div_(de_nom), group["clip_threshold"])
            delta = update.mul_(c["step_size"])

        if cautious:
            delta = cautious_one_(delta, grad)

        subtract_one_(p, delta, state, bf16_method)

    def _pn_one(
        self,
        state: dict[str, Any],
        pos: str,
        neg: str,
        md: str,
        grad: Tensor,
        c: dict[str, float],
    ) -> Tensor:
        """Per-param pos-neg numerator (EMA the positive buffer with the raw grad)."""
        m_pos = self._dequant_one(state, pos, md, grad)
        m_neg = self._dequant_one(state, neg, md, grad)
        m_pos.mul_(c["beta1_sq"]).add_(grad, alpha=1.0 - c["beta1_sq"])
        self._store_one(state, pos, md, m_pos)
        return m_pos.mul(1.0 + c["beta0"]).add_(m_neg, alpha=-c["beta0"]).mul_(1.0 / c["noise_norm"])

