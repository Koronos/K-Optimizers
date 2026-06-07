"""AdaMuon — orthogonalized momentum with factored, quantized variance adaptation.

AdaMuon is Muon's Newton-Schulz orthogonalization (Jordan et al.; Si et al.,
*AdaMuon: Adaptive Muon Optimizer*, arXiv:2507.11005) grafted onto Adakaon's
memory backend. It targets **AdamW-beating precision** for diffusion fine-tuning
at near-Adafactor memory.

The pipeline (2-D / conv weights) is, in order:

1. **First moment of the RAW gradient** — an EMA ``m = β1·m + (1-β1)·g`` kept in a
   quantized codec (bf16/int8/4bit) exactly like :class:`~kaon.adakaon.Adakaon`.
2. **Orthogonalize** ``m`` with a 5-step Newton-Schulz iteration → ``O ≈ U·Vᵀ``.
3. **Factored second moment OF ``O``** (Adafactor row+col EMA) → ``u = O·inv_sqrt(v̂)``.
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
``inv_sqrt(v̂)`` already rescales ``u`` to RMS ``≈ 1`` (the ``c_factor`` term is
``≈ √max(R,C)``), so reapplying ``√max(R,C)`` would double-count the shape and
make the update grow with layer size. We therefore scale by the **constant**
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

from kaon._backend import (
    cautious_batched_,
    is_low_precision,
    subtract_batched_,
    subtract_one_,
)
from kaon._factored import factored_inv_sqrt_factors, update_factored_state
from kaon._momentum_codec import (
    _FOURBIT_BLOCK,
    _make_codec,
    _MomentumCodec,
    load_state_dict_preserving_dtypes,
)

__all__ = ["AdaMuon"]

MomentumDtype = Literal["bfloat16", "float32", "int8", "4bit"]

# Shape-independent applied-update RMS target (before lr). The factored
# inv_sqrt(v̂) already brings ``u`` to RMS≈1, so this is the only magnitude scale
# applied — equal to Muon's per-element RMS (``0.2``). See the module docstring on
# why ``√max(R,C)`` is NOT reapplied here.
_UPDATE_RMS = 0.2

# Foreach batching knobs — mirror Adakaon's (kept local so AdaMuon is a
# standalone module, not coupled to adakaon.py internals). See
# docs/foreach-batching.md for the rationale behind each constant.
_STACK_SAFETY_FRACTION = 0.10
_STACK_BYTES_PER_ELEM = 48
_MIN_STACK_ELEMS = 262_144
_DEFAULT_STACK_ELEMS = 64_000_000
_FOREACH_BATCH_CUTOFF = 2_000_000


def _rms(t: Tensor) -> Tensor:
    return t.norm(2) / math.sqrt(max(t.numel(), 1))


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


class AdaMuon(Optimizer):
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
            5). On a paired pixel-DDPM sweep, ``5`` *over-orthogonalizes*: flattening
            the momentum's singular spectrum too hard discards useful curvature, so
            ``2`` was both faster (~0.9 ms/step per saved iteration) AND lower val
            than ``5`` — a strict win. ``1`` under-orthogonalizes (loses the edge over
            Adakaon); ``2`` was the sweet spot. Re-tune per task/model.
        clip_threshold: RMS ceiling on the normalized update (``rms(u) <= thr``).
            Applied in the RMS≈1 domain, so ``1.0`` matches Adakaon's semantics
            and is load-bearing for the first few steps (before the factored second
            moment warms up). Steady-state it is a near no-op.
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
        compile: ``torch.compile`` the whole step body (``fullgraph=False``), fusing
            the step's elementwise chain. **Workload-dependent — benchmark it.** The
            win scales with how much (fusable) elementwise math the step does, so for
            AdaMuon (heavy Newton-Schulz + factored + cautious + scale) it helps
            broadly: measured ~0.34x (-66%) on many small *distinct*-shaped tensors
            (which defeat ``foreach`` batching), and ~0.6-0.75x on few/tiny/single
            params. It is ~neutral for compute-bound full fine-tunes and for already-
            ``foreach``-batched pure-LoRA sets, and a no-op when the model fwd/bwd
            dominates (SDXL is UNet-bound). One-time warmup; numerically equivalent to
            eager (bit-exact per step; SR unbiased; no crashes across dtypes/shapes).
            Not recommended on CPU (inconsistent). NB: compiling *only* the
            Newton-Schulz does NOT help on LoRA-rank matrices — the win is the
            whole-step fusion. Default ``False``.
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
        momentum_dtype: MomentumDtype = "bfloat16",
        momentum_4bit_block: int = _FOURBIT_BLOCK,
        cautious: bool = True,
        bf16_method: str = "stochastic_rounding",
        foreach: bool = True,
        foreach_batch_cutoff: int = _FOREACH_BATCH_CUTOFF,
        foreach_stack_budget: int | None = None,
        compile: bool = False,  # noqa: A002 — public kwarg name
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
        defaults = {
            "lr": lr,
            "betas": (beta1, beta2),
            "eps": (float(eps[0]), float(eps[1])),
            "weight_decay": weight_decay,
            "ns_steps": ns_steps,
            "clip_threshold": clip_threshold,
            "momentum_dtype": momentum_dtype,
            "momentum_4bit_block": momentum_4bit_block,
            "cautious": cautious,
            "bf16_method": bf16_method,
        }
        super().__init__(params, defaults)
        self._foreach = foreach
        self._foreach_batch_cutoff = foreach_batch_cutoff
        self._foreach_stack_budget = foreach_stack_budget
        # Optional torch.compile of the whole step body. fullgraph=False tolerates
        # the param-group Python loop; fuses the elementwise chain across foreach
        # buckets. Measured ~16% faster on many-small-tensor (LoRA-shaped) loads
        # where the optimizer is a real fraction of the step — a no-op win when the
        # model fwd/bwd dominates (e.g. SDXL is UNet-bound). NB: compiling ONLY the
        # Newton-Schulz does NOT help on LoRA-rank matrices (too small); the win is
        # the whole-step fusion. No host syncs in the step, so SR stays unbiased.
        self._compiled_step = torch.compile(self._run_step, fullgraph=False) if compile else None
        # One momentum codec per dtype string (stateless beyond the dtype).
        self._codecs: dict[str, _MomentumCodec] = {}

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
        # First moment stores the EMA of the RAW gradient, in the param's original
        # shape (the codec matrixizes it per-step for Newton-Schulz).
        if group["betas"][0] > 0:
            self._codec(group).init_state(state, grad, group)
        if is_low_precision(p) and group["bf16_method"] == "kahan":
            state["shift"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        (self._compiled_step or self._run_step)()
        return loss

    @torch.no_grad()
    def _run_step(self) -> None:
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            for p in params:
                if p.grad.is_sparse:
                    raise RuntimeError("AdaMuon does not support sparse gradients")
            if self._foreach and self._group_foreach_eligible(group):
                chunk_budget = self._foreach_budget(params[0].device)
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

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state, preserving the quantized first moment's stored dtype.

        torch's default ``load_state_dict`` upcasts every state tensor to the
        param's dtype (fp32), which would inflate a bf16/int8/4bit momentum back to
        fp32 on resume — losing the memory the codec saves and breaking bit-exact
        resume. Delegate to the shared dtype-preserving helper.
        """
        load_state_dict_preserving_dtypes(self, state_dict)

    # ----------------------------------------------------------------- foreach
    def _foreach_budget(self, device: torch.device) -> int:
        """Max elements per stacked chunk (mirrors Adakaon's adaptive budget)."""
        if self._foreach_stack_budget is not None:
            return self._foreach_stack_budget
        cap = 4 * self._foreach_batch_cutoff
        if device.type == "cuda":
            free_bytes = torch.cuda.mem_get_info(device)[0]
            adaptive = int(free_bytes * _STACK_SAFETY_FRACTION / _STACK_BYTES_PER_ELEM)
            return max(_MIN_STACK_ELEMS, min(adaptive, cap))
        return min(_DEFAULT_STACK_ELEMS, cap)

    @staticmethod
    def _group_foreach_eligible(group: dict[str, Any]) -> bool:
        return (
            group["clip_threshold"] > 0
            and group["bf16_method"] != "kahan"  # kahan needs a per-param shift buffer
        )

    @staticmethod
    def _param_foreach_eligible(p: Tensor, group: dict[str, Any], cutoff: int) -> bool:
        if p.ndim == 0:
            return False
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
        codec = self._codec(group)

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
            step = max(1, budget // max(eff[0] * eff[1], 1))
            for i in range(0, len(plist), step):
                self._factored_bucket(
                    plist[i:i + step], eff, matrixize, ns_steps,
                    beta1, beta2, eps1, lr, clip, wd, cautious, bf16_method, codec,
                )
        for (length, _dtype), plist in flat_buckets.items():
            step = max(1, budget // max(length, 1))
            for i in range(0, len(plist), step):
                self._nonfactored_bucket(
                    plist[i:i + step], length,
                    beta1, beta2, eps1, lr, clip, wd, cautious, bf16_method, codec,
                )

    @torch.no_grad()
    def _factored_bucket(
        self,
        plist: list[Tensor],
        eff: tuple[int, int],
        matrixize: bool,
        ns_steps: int,
        beta1: float,
        beta2: float,
        eps1: float,
        lr: float,
        clip: float,
        wd: float,
        cautious: bool,
        bf16_method: str,
        codec: _MomentumCodec,
    ) -> None:
        R, C = eff  # noqa: N806 — matrix dims (stacked tensor is [N, R, C])
        N = len(plist)  # noqa: N806

        def mat(t: Tensor) -> Tensor:
            return t.view(R, C) if matrixize else t

        rows = [self.state[p]["row"] for p in plist]
        cols = [self.state[p]["col"] for p in plist]

        grad = torch.stack([mat(p.grad).float() for p in plist])          # [N, R, C]

        # 1. First moment of the RAW gradient (codec owns dequant→EMA→requant).
        if beta1 > 0:
            states = [self.state[p] for p in plist]
            m = codec.ema_stacked(states, grad, mat, (R, C), beta1)        # [N, R, C]
        else:
            m = grad

        # 2. Orthogonalize the momentum, per slice (batched bmm Newton-Schulz).
        ortho = zeropower_via_newtonschulz5_stacked(m, ns_steps).float()   # [N, R, C]

        # 3. Factored second moment OF the orthogonalized signal (HF eps placement).
        row = torch.stack(rows)                                           # [N, R]
        col = torch.stack(cols)                                           # [N, C]
        omb = 1.0 - beta2
        ortho_sq = ortho * ortho
        if eps1 > 0:
            ortho_sq = ortho_sq.add_(eps1)
        row.lerp_(ortho_sq.mean(dim=-1), omb)
        col.lerp_(ortho_sq.mean(dim=-2), omb)
        torch._foreach_copy_(rows, list(row.unbind(0)))
        torch._foreach_copy_(cols, list(col.unbind(0)))

        r_factor = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)  # [N, R, 1]
        c_factor = col.rsqrt().unsqueeze(-2)                                       # [N, 1, C]
        update = ortho.mul(r_factor).mul_(c_factor)                               # [N, R, C], RMS≈1

        # 4. Clip ceiling (RMS≈1 domain) then the constant shape-independent scale.
        rms = update.reshape(N, -1).norm(2, dim=1) / math.sqrt(R * C)
        update.div_(rms.div_(clip).clamp_(min=1.0).view(N, 1, 1))
        update.mul_(_UPDATE_RMS * lr)
        delta = update

        if wd != 0:
            p_fp32 = torch.stack([mat(p.data).float() for p in plist])
            delta = delta.add_(p_fp32, alpha=lr * wd)

        if cautious:
            delta = cautious_batched_(delta, grad)

        subtract_batched_([mat(p.data) for p in plist], delta, bf16_method)

    @torch.no_grad()
    def _nonfactored_bucket(
        self,
        plist: list[Tensor],
        length: int,
        beta1: float,
        beta2: float,
        eps1: float,
        lr: float,
        clip: float,
        wd: float,
        cautious: bool,
        bf16_method: str,
        codec: _MomentumCodec,
    ) -> None:
        """Non-factored Adam-style update for 1-D params (biases, norm scales).

        Not orthogonalized (Newton-Schulz needs a matrix). RMS-normalized to the
        same ``0.2·lr`` target as the 2-D path so a single ``lr`` is consistent
        across the model.
        """
        N = len(plist)  # noqa: N806
        vs = [self.state[p]["v"] for p in plist]                          # each [L], fp32

        grad = torch.stack([p.grad.float() for p in plist])               # [N, L]
        v = torch.stack(vs)                                               # [N, L]

        omb = 1.0 - beta2
        grad_sq = grad * grad
        if eps1 > 0:
            grad_sq = grad_sq.add_(eps1)
        v.lerp_(grad_sq, omb)
        torch._foreach_copy_(vs, list(v.unbind(0)))

        update = grad.mul(v.rsqrt())                                      # [N, L], RMS≈1
        rms = update.norm(2, dim=1) / math.sqrt(length)
        update.div_(rms.div_(clip).clamp_(min=1.0).view(N, 1))
        update.mul_(_UPDATE_RMS * lr)

        if beta1 > 0:
            states = [self.state[p] for p in plist]
            delta = codec.ema_stacked(states, update, lambda t: t, (length,), beta1)  # [N, L]
        else:
            delta = update

        if wd != 0:
            p_fp32 = torch.stack([p.data.float() for p in plist])
            delta = delta.add_(p_fp32, alpha=lr * wd)

        if cautious:
            delta = cautious_batched_(delta, grad)

        subtract_batched_([p.data for p in plist], delta, bf16_method)

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

        grad_fp32 = p.grad if p.grad.dtype == torch.float32 else p.grad.float()
        ndim = grad_fp32.ndim
        factored = ndim >= 2

        if factored:
            matrixize = ndim > 2
            # 1. First moment of the raw gradient (original shape).
            m = self._codec(group).ema_one(state, grad_fp32, beta1) if beta1 > 0 else grad_fp32
            mv = m.reshape(m.shape[0], -1) if matrixize else m
            # 2. Orthogonalize.
            ortho = zeropower_via_newtonschulz5(mv, ns_steps).float()         # [R, C]
            # 3. Factored second moment OF the orthogonalized signal.
            update_factored_state(ortho, state["row"], state["col"], beta2, eps1)
            r_factor, c_factor = factored_inv_sqrt_factors(state["row"], state["col"])
            update = ortho.mul(r_factor).mul_(c_factor)                       # [R, C], RMS≈1
            # 4. Clip ceiling then constant scale; reshape back if conv.
            if clip > 0:
                update.div_((_rms(update) / clip).clamp_(min=1.0))
            update.mul_(_UPDATE_RMS * lr)
            if matrixize:
                update = update.view_as(grad_fp32)
            delta = update
            cautious_ref = grad_fp32
        else:
            # 1-D: Adam-style (no orthogonalization), RMS-normalized to 0.2·lr.
            v = state["v"]
            grad_sq = grad_fp32 * grad_fp32
            if eps1 > 0:
                grad_sq.add_(eps1)
            v.lerp_(grad_sq, 1.0 - beta2)
            update = grad_fp32.mul(v.rsqrt())
            if clip > 0:
                update.div_((_rms(update) / clip).clamp_(min=1.0))
            update.mul_(_UPDATE_RMS * lr)
            delta = self._codec(group).ema_one(state, update, beta1) if beta1 > 0 else update
            cautious_ref = grad_fp32

        if wd != 0:
            p_fp32 = p.data if p.dtype == torch.float32 else p.data.float()
            delta = delta.add_(p_fp32, alpha=lr * wd)

        if cautious:
            mask = (delta * cautious_ref > 0).to(delta.dtype)
            delta = delta.mul_(mask).div_(mask.mean().clamp_(min=1e-8))

        subtract_one_(p, delta, state, bf16_method)


