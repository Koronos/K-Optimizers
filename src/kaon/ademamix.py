"""AdEMAMix — Adam with a mixture of two gradient EMAs, on kaon's memory backend.

AdEMAMix (Pagliardini et al. 2024, *The AdEMAMix Optimizer: Better, Faster,
Older*, arXiv:2409.03137) keeps Adam's second moment ``v`` but replaces the single
first-moment EMA with a **mixture of two** EMAs of the gradient: a *fast* EMA
(``beta1 ≈ 0.9``, exactly Adam's) and a *slow* EMA (``beta3 ≈ 0.9999``) that
retains very old gradients. The slow buffer is what lets a single (very large)
momentum horizon stay useful — old gradients keep contributing — which is where
AdEMAMix's "better, faster, older" generalization edge lives.

Implemented on the precision/memory machinery proven in
:class:`~kaon.adapnm.AdaPNM` (its closest sibling: two momentum buffers + a
factored quantized ``v`` + foreach + cautious + gradient-centralization +
dtype-safe checkpoint).

**The exact update AS IMPLEMENTED (matches the official Apple ``ml-ademamix``
PyTorch reference, ``apple/ml-ademamix``):**

.. code-block:: text

    m1 <- beta1 * m1 + (1 - beta1) * g            # fast EMA (Adam's first moment)
    m2 <- beta3_t * m2 + (1 - beta3_t) * g        # slow EMA (long-horizon)
    v  <- beta2 * v + (1 - beta2) * g^2           # Adam second moment (factored)

    bc1   = 1 - beta1^t                           # fast-EMA bias correction
    bc2   = 1 - beta2^t
    denom = sqrt(v) / sqrt(bc2) + eps
    num   = m1 / bc1 + alpha_t * m2               # ONLY m1 is bias-corrected
    p    -= lr * (num / denom + weight_decay * p) # decoupled WD, coupled to lr

Two correctness points carried verbatim from the official reference:

1. **Only the fast EMA ``m1`` is bias-corrected** (``m1 / bc1``). The slow EMA
   ``m2`` is **NOT** bias-corrected — this is deliberate in the paper/official
   code (an early-training ``m2`` with ``beta3=0.9999`` is intentionally tiny;
   ``alpha`` + the ``beta3`` warmup, not a ``1/(1-beta3^t)`` blow-up, control its
   ramp-in). The kozistr ``pytorch_optimizer`` port instead folds the bias
   correction into ``step_size = lr / bc1`` and applies it to the *whole*
   numerator ``(m1 + alpha*m2)``, i.e. it scales ``alpha*m2`` by ``1/bc1`` too —
   a discrepancy in the first few steps (``bc1 -> 1`` fast for ``beta1=0.9``).
   **We follow the official:** divide only ``m1`` by ``bc1``.

2. **Weight decay is decoupled but coupled to ``lr``** (AdamW form as written in
   the official code): the update is ``num/denom + wd*p`` and the *whole thing* is
   scaled by ``-lr``, so the effective decay is ``p -= lr*wd*p``. We fold ``wd*p``
   into ``delta`` (so it rides the same ``lr`` scale and the bf16-correct
   write-back) rather than doing a separate ``p *= (1 - lr*wd)`` — matching the
   official arithmetic. Note this means cautious masking (below) sees the WD term
   as part of ``delta``; that mirrors how Adakaon folds WD into its cautious step.

**Alpha and beta3 warmup (official ``linear_warmup_scheduler`` /
``linear_hl_warmup_scheduler``).** A large ``beta3`` from step 0 is unstable, so
both ``alpha`` and ``beta3`` are warmed up over ``t_alpha_beta3`` steps:

.. code-block:: text

    # alpha: plain linear 0 -> alpha_final over t_alpha_beta3 steps
    a = step / t_alpha_beta3
    alpha_t = (1-a)*0 + a*alpha_final          (clamped to alpha_final)

    # beta3: linear interpolation in HALF-LIFE space from beta1 -> beta3_final
    f(b)     = log(0.5) / log(b + 1e-8) - 1    # b's half-life (in steps), minus 1
    f_inv(t) = 0.5 ** (1 / (t + 1))
    beta3_t  = f_inv((1-a)*f(beta1) + a*f(beta3_final))

Half-life-space interpolation (not naive linear-in-beta) keeps the *memory
horizon* growing smoothly. The kozistr port uses a different (log-reciprocal)
beta3 interpolation; we follow the official half-life form. ``t_alpha_beta3=None``
disables both warmups (constant ``alpha_final`` / ``beta3_final``) — useful for a
clean reference and for short proxy runs where the warmup horizon would dominate.

**The factored second moment** reuses Adakaon/AdaPNM's backend exactly: ``ndim >=
2`` weights factor ``v`` into row+column EMAs (conv kernels matrixized to ``[out,
in*kh*kw]`` first), ``ndim == 1`` keeps a full per-coordinate ``v``. ``eps`` goes
on the denominator on the 1-D path (matching the reference) and is folded into the
Adafactor ``eps1`` on the factored path.

**Momentum precision.** BOTH momenta are stored through the shared codec
(:mod:`kaon._momentum_codec`); ``momentum_dtype`` selects bf16/fp32/int8/4bit.
The slow EMA ``m2`` holds long-horizon signal (``beta3=0.9999`` => half-life ~6930
steps), so it is, a priori, more precision-sensitive than ``m1`` — quantizing it
hard could erode the very "old gradient" information AdEMAMix exists to keep.
Default is ``bfloat16`` for both. (A single ``momentum_dtype`` is used for both
buffers for checkpoint/layout simplicity, matching AdaPNM; if int8/4bit on ``m2``
proves lossy in practice, a per-buffer dtype split would be the follow-up.)

Standard ``torch.optim.Optimizer`` with a single per-parameter step, so it drops
into per-parameter / gradient-release loops unchanged. ``foreach=True`` batches the
step and is bit-exact with the per-parameter path (verified by the parity test).
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
    subtract_batched_,
    subtract_one_,
)
from kaon._factored import factored_inv_sqrt_factors, update_factored_state
from kaon._momentum_codec import (
    _FOURBIT_BLOCK,
    _dequant_4bit,
    _dequant_4bit_stacked,
    _quant_4bit,
    _quant_4bit_stacked,
    _quant_int8,
    _quant_int8_stacked,
    load_state_dict_preserving_dtypes,
)

__all__ = ["AdEMAMix"]

MomentumDtype = Literal["bfloat16", "float32", "int8", "4bit"]

# Two momenta (fast + slow) + factored v: same stacked working-set estimate as AdaPNM.
_STACK_BYTES_PER_ELEM = 64


def schedule_alpha(step: int, alpha_final: float, t_alpha_beta3: int | None) -> float:
    """Official ``linear_warmup_scheduler``: linear 0 -> ``alpha_final`` over ``t_alpha_beta3``.

    ``step`` is 1-indexed. ``t_alpha_beta3 is None`` (or <= 0) disables warmup. At
    ``step >= t_alpha_beta3`` returns ``alpha_final`` exactly (the official clamps
    via the ``step < warmup`` branch; we clamp equivalently).
    """
    if not t_alpha_beta3 or t_alpha_beta3 <= 0:
        return alpha_final
    if step >= t_alpha_beta3:
        return alpha_final
    a = step / float(t_alpha_beta3)
    return a * alpha_final  # alpha_start = 0


def schedule_beta3(
    step: int, beta1: float, beta3_final: float, t_alpha_beta3: int | None
) -> float:
    """Official ``linear_hl_warmup_scheduler``: interpolate beta1 -> beta3 in half-life space.

    ``f(b) = log(0.5)/log(b+eps) - 1`` maps a decay to its half-life (steps) minus
    one; we linearly interpolate the *half-lives* and map back with
    ``f_inv(t) = 0.5**(1/(t+1))``. ``step`` is 1-indexed; ``t_alpha_beta3 is None``
    (or <= 0) disables warmup.
    """
    if not t_alpha_beta3 or t_alpha_beta3 <= 0:
        return beta3_final
    if step >= t_alpha_beta3:
        return beta3_final

    def f(beta: float, eps: float = 1e-8) -> float:
        return math.log(0.5) / math.log(beta + eps) - 1.0

    def f_inv(t: float) -> float:
        return math.pow(0.5, 1.0 / (t + 1.0))

    a = step / float(t_alpha_beta3)
    return f_inv((1.0 - a) * f(beta1) + a * f(beta3_final))


class AdEMAMix(Optimizer):
    """AdEMAMix (Adam + a mixture of two gradient EMAs) on kaon's memory backend.

    Args:
        params: parameters or param-group dicts.
        lr: learning rate.
        betas: ``(beta1, beta2, beta3)``. ``beta1`` is the **fast** first-moment
            decay (Adam's; default ``0.9``, bias-corrected). ``beta2`` is the
            (factored) second-moment decay (default ``0.999``). ``beta3`` is the
            **slow** first-moment decay (default ``0.9999``) — the long-horizon EMA
            that retains old gradients; it is NOT bias-corrected and is warmed up
            from ``beta1`` (see ``t_alpha_beta3``).
        alpha: mixing coefficient for the slow EMA in the update numerator
            ``m1_hat + alpha*m2``. Default ``5.0`` (the paper's recommended value;
            the official code's class default is ``2.0`` but the paper's headline
            language-model runs use ``alpha=5``). Warmed up 0 -> ``alpha`` over
            ``t_alpha_beta3``. **This and ``beta3`` are AdEMAMix's signature knobs.**
        t_alpha_beta3: number of warmup steps over which ``alpha`` (linearly) and
            ``beta3`` (in half-life space, from ``beta1``) ramp to their final
            values. ``None`` (default) disables both warmups (constant
            ``alpha``/``beta3`` from step 1). The paper uses a long horizon
            (e.g. matched to the LR warmup / a large fraction of training); set a
            small value for short fine-tunes.
        eps: term added to the second-moment denominator for stability. On the
            non-factored (1-D) path it is added to ``sqrt(v)/sqrt(bc2)`` exactly as
            the reference does; on the factored path it is folded into the
            Adafactor ``eps1`` (added to ``grad**2`` before the row/col reductions).
        weight_decay: decoupled (AdamW-style) weight decay, coupled to ``lr`` and
            folded into the update numerator as the official code does:
            ``p -= lr * (num/denom + weight_decay * p)``.
        cautious: cautious masking (Liang et al. 2024) on the final update vs the
            gradient. **On by default** (kaon convention).
        gradient_centralization: centralize ``ndim>=2`` grads before the step.
            **On by default** (kaon convention); pin ``False`` for reference parity.
        momentum_dtype: storage for **both** momentum buffers — ``"bfloat16"``
            (default), ``"float32"``, ``"int8"`` (per-row absmax) or ``"4bit"``
            (per-block absmax, nibble-packed). Note: the slow EMA ``m2`` carries
            long-horizon signal and may be more sensitive to aggressive quantization
            than ``m1``; default bf16. AdEMAMix carries *two* momenta, so its
            momentum floor is ~2x a single-momentum optimizer at the same dtype.
        momentum_4bit_block: block size for ``momentum_dtype="4bit"`` (default 128).
        bf16_method: weight-update strategy for low-precision params —
            ``"stochastic_rounding"`` (default), ``"kahan"`` (+2 B/param), ``"none"``.
        foreach: batch the step across parameters. Default ``True``; bit-exact with
            the per-parameter path (SR draws differ, unbiased either way).
        foreach_batch_cutoff: per-tensor element count above which a weight loops
            instead of stacking (default ``2_000_000``).
        foreach_stack_budget: max elements per stacked chunk; ``None`` adapts to VRAM.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 5.0,
        t_alpha_beta3: int | None = None,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        cautious: bool = True,
        gradient_centralization: bool = True,
        momentum_dtype: MomentumDtype = "bfloat16",
        momentum_4bit_block: int = _FOURBIT_BLOCK,
        bf16_method: str = "stochastic_rounding",
        foreach: bool = True,
        foreach_batch_cutoff: int = FOREACH_BATCH_CUTOFF,
        foreach_stack_budget: int | None = None,
    ) -> None:
        beta1, beta2, beta3 = float(betas[0]), float(betas[1]), float(betas[2])
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"betas[0] must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"betas[1] must be in [0, 1), got {beta2}")
        if not 0.0 <= beta3 < 1.0:
            raise ValueError(f"betas[2] must be in [0, 1), got {beta3}")
        if alpha < 0.0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if t_alpha_beta3 is not None and t_alpha_beta3 < 0:
            raise ValueError(f"t_alpha_beta3 must be >= 0 or None, got {t_alpha_beta3}")
        if lr < 0.0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if eps < 0.0:
            raise ValueError(f"eps must be >= 0, got {eps}")
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
            "betas": (beta1, beta2, beta3),
            "alpha": float(alpha),
            "t_alpha_beta3": t_alpha_beta3,
            "eps": float(eps),
            "weight_decay": weight_decay,
            "cautious": cautious,
            "gradient_centralization": gradient_centralization,
            "momentum_dtype": momentum_dtype,
            "momentum_4bit_block": momentum_4bit_block,
            "bf16_method": bf16_method,
            "step": 0,
        }
        super().__init__(params, defaults)
        self._foreach = foreach
        self._foreach_batch_cutoff = foreach_batch_cutoff
        self._foreach_stack_budget = foreach_stack_budget

    # ------------------------------------------------------------------- state
    @staticmethod
    def _block_size(grad: Tensor, group: dict[str, Any]) -> int:
        bs = group["momentum_4bit_block"]
        numel = grad.numel()
        return numel if bs <= 0 else (min(bs, numel) if numel > 0 else 1)

    @torch.no_grad()
    def _alloc_momentum(
        self, prefix: str, grad: Tensor, state: dict[str, Any], group: dict[str, Any]
    ) -> None:
        """Allocate one momentum buffer in the configured codec layout.

        Layout matches :mod:`kaon._momentum_codec` exactly (per-row int8 scale;
        per-block 4-bit scale, zero == nibble 8) so each buffer resumes bit-exactly
        via ``load_state_dict_preserving_dtypes``.
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
            bs = self._block_size(grad, group)
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
        # Two momenta: fast (m1) + slow (m2), each through the shared codec layout.
        self._alloc_momentum("m1", grad, state, group)
        self._alloc_momentum("m2", grad, state, group)
        if is_low_precision(p) and group["bf16_method"] == "kahan":
            state["shift"] = torch.zeros_like(p)

    # -------------------------------------------------- momentum read / write
    @staticmethod
    def _dequant_one(state: dict[str, Any], prefix: str, md: str, like: Tensor) -> Tensor:
        """Read one stored momentum back as a **fresh** fp32 tensor shaped like ``like``.

        Must not alias the stored buffer: the caller EMA-updates it in place and then
        builds the (mutating) numerator on top, so for the fp32 codec — where
        ``.float()`` is a no-op that returns the buffer itself — we ``clone()``.
        """
        if md == "float32":
            return state[prefix].clone().reshape_as(like)
        if md == "bfloat16":
            return state[prefix].float().reshape_as(like)  # .float() copies bf16->fp32
        if md == "int8":
            return state[prefix].float().mul_(state[f"{prefix}_scale"]).reshape_as(like)
        m = _dequant_4bit(
            state[prefix], state[f"{prefix}_scale"], state[f"{prefix}_numel"], state[f"{prefix}_block"]
        )
        return m.view_as(like)

    @staticmethod
    def _store_one(state: dict[str, Any], prefix: str, md: str, m_fp32: Tensor) -> None:
        """Write an updated fp32 momentum back into the configured storage layout."""
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
        """Stacked fp32 momentum ``[N, *shape]`` from per-param storage."""
        n = len(states)
        per = 1
        for d in shape:
            per *= d
        if md in ("bfloat16", "float32"):
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
        per = 1
        for d in shape:
            per *= d
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

    # -------------------------------------------------------------------- step
    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            for p in params:
                if p.grad.is_sparse:
                    raise RuntimeError("AdEMAMix does not support sparse gradients")
            group["step"] += 1
            if not params:
                continue
            if group["gradient_centralization"]:
                centralize_grads_(params)
            if self._foreach and self._group_foreach_eligible(group):
                chunk_budget = foreach_budget(
                    self._foreach_stack_budget, self._foreach_batch_cutoff,
                    _STACK_BYTES_PER_ELEM, params[0].device,
                )
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
        return loss

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state, preserving both quantized momenta's stored dtype."""
        load_state_dict_preserving_dtypes(self, state_dict)

    # ----------------------------------------------------------- coefficients
    @staticmethod
    def _coeffs(group: dict[str, Any]) -> dict[str, float]:
        """All per-step scalar coefficients (shared by the per-param and foreach paths)."""
        beta1, beta2, beta3_final = group["betas"]
        step = group["step"]
        t = group["t_alpha_beta3"]
        alpha_t = schedule_alpha(step, group["alpha"], t)
        beta3_t = schedule_beta3(step, beta1, beta3_final, t)
        bc1 = 1.0 - beta1 ** step
        bc2_sq = math.sqrt(1.0 - beta2 ** step)
        return {
            "beta1": beta1,
            "beta2": beta2,
            "beta3_t": beta3_t,
            "alpha_t": alpha_t,
            "bc1": bc1,
            "bc2_sq": bc2_sq,
            "lr": group["lr"],
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
                self._factored_bucket(plist[i:i + stepn], eff, matrixize, md, c, group)
        for (length, _dtype), plist in flat_buckets.items():
            stepn = max(1, budget // max(length, 1))
            for i in range(0, len(plist), stepn):
                self._nonfactored_bucket(plist[i:i + stepn], length, md, c, group)

    @torch.no_grad()
    def _factored_bucket(
        self,
        plist: list[Tensor],
        eff: tuple[int, int],
        matrixize: bool,
        md: str,
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

        # Two-EMA mixture numerator: m1/bc1 + alpha*m2 (only m1 bias-corrected).
        num = self._mix_stacked(states, md, (R, C), grad, c)                       # [N, R, C]

        delta = num.mul_(inv_denom)                                               # num/denom
        # Decoupled WD folded into the update (official: p -= lr*(num/denom + wd*p)).
        if wd != 0:
            weights = torch.stack([mat(p.data).float() for p in plist])
            delta = delta.add_(weights, alpha=wd)
        delta = delta.mul_(c["lr"])

        if cautious:
            delta = cautious_batched_(delta, grad)

        subtract_batched_([mat(p.data) for p in plist], delta, bf16_method)

    @torch.no_grad()
    def _nonfactored_bucket(
        self,
        plist: list[Tensor],
        length: int,
        md: str,
        c: dict[str, float],
        group: dict[str, Any],
    ) -> None:
        eps1 = group["eps"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]

        states = [self.state[p] for p in plist]
        vs = [s["v"] for s in states]

        grad = torch.stack([p.grad.float() for p in plist])               # [N, L]
        v = torch.stack(vs)                                               # [N, L]

        # Full per-coordinate second moment (1-D). eps on the denominator (reference).
        v.mul_(c["beta2"]).addcmul_(grad, grad, value=1.0 - c["beta2"])
        torch._foreach_copy_(vs, list(v.unbind(0)))

        de_nom = v.sqrt().div_(c["bc2_sq"]).add_(eps1)                    # sqrt(v)/sqrt(bc2)+eps

        num = self._mix_stacked(states, md, (length,), grad, c)           # [N, L]
        delta = num.div_(de_nom)
        if wd != 0:
            weights = torch.stack([p.data.float() for p in plist])
            delta = delta.add_(weights, alpha=wd)
        delta = delta.mul_(c["lr"])

        if cautious:
            delta = cautious_batched_(delta, grad)

        subtract_batched_([p.data for p in plist], delta, bf16_method)

    def _mix_stacked(
        self,
        states: list[dict[str, Any]],
        md: str,
        shape: tuple[int, ...],
        grad: Tensor,
        c: dict[str, float],
    ) -> Tensor:
        """Stacked two-EMA mixture numerator ``m1/bc1 + alpha*m2``.

        ``grad`` is the stacked, reshaped gradient (``[N, *shape]``). EMA-updates
        both momenta (fast decay ``beta1``, slow decay ``beta3_t``), stores them
        back, then returns the (bias-corrected-fast + alpha-mixed-slow) numerator.
        Only ``m1`` is bias-corrected (``/bc1``); ``m2`` is not (matches official).
        """
        n = grad.shape[0]
        m1 = self._dequant_stacked(states, "m1", md, shape).reshape((n, *shape))
        m2 = self._dequant_stacked(states, "m2", md, shape).reshape((n, *shape))
        m1.mul_(c["beta1"]).add_(grad, alpha=1.0 - c["beta1"])
        m2.mul_(c["beta3_t"]).add_(grad, alpha=1.0 - c["beta3_t"])
        self._store_stacked(states, "m1", md, m1.reshape((n, *shape)))
        self._store_stacked(states, "m2", md, m2.reshape((n, *shape)))
        # m1 is a fresh stacked tensor (already stored) -> build the numerator in place.
        num = m1.div_(c["bc1"]).add_(m2, alpha=c["alpha_t"])
        return num

    # ---------------------------------------------------------- per-parameter
    @torch.no_grad()
    def _step_one_param(self, p: Tensor, group: dict[str, Any]) -> None:
        c = self._coeffs(group)
        md = group["momentum_dtype"]
        eps1 = group["eps"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]

        state = self.state[p]
        if not state:
            self._init_state(p, state, group)

        grad = p.grad if p.grad.dtype == torch.float32 else p.grad.float()
        ndim = grad.ndim
        factored = ndim >= 2

        if factored:
            matrixize = ndim > 2
            gv = grad.reshape(grad.shape[0], -1) if matrixize else grad
            update_factored_state(gv, state["row"], state["col"], c["beta2"], eps1)
            r_factor, c_factor = factored_inv_sqrt_factors(state["row"], state["col"])
            inv_denom = (r_factor * c_factor).mul_(c["bc2_sq"])           # 1/sqrt(v_hat)
            num = self._mix_one(state, md, gv, c)
            delta = num.mul_(inv_denom)
            if matrixize:
                delta = delta.reshape_as(grad)
        else:
            v = state["v"]
            v.mul_(c["beta2"]).addcmul_(grad, grad, value=1.0 - c["beta2"])
            de_nom = v.sqrt().div_(c["bc2_sq"]).add_(eps1)
            num = self._mix_one(state, md, grad, c)
            delta = num.div_(de_nom)

        # Decoupled WD folded into the update (official: p -= lr*(num/denom + wd*p)).
        if wd != 0:
            pdata = p.data.float() if p.dtype != torch.float32 else p.data
            delta = delta.add_(pdata, alpha=wd)
        delta = delta.mul_(c["lr"])

        if cautious:
            delta = cautious_one_(delta, grad)

        subtract_one_(p, delta, state, bf16_method)

    def _mix_one(
        self,
        state: dict[str, Any],
        md: str,
        grad: Tensor,
        c: dict[str, float],
    ) -> Tensor:
        """Per-param two-EMA mixture numerator ``m1/bc1 + alpha*m2``."""
        m1 = self._dequant_one(state, "m1", md, grad)
        m2 = self._dequant_one(state, "m2", md, grad)
        m1.mul_(c["beta1"]).add_(grad, alpha=1.0 - c["beta1"])
        m2.mul_(c["beta3_t"]).add_(grad, alpha=1.0 - c["beta3_t"])
        self._store_one(state, "m1", md, m1)
        self._store_one(state, "m2", md, m2)
        return m1.div_(c["bc1"]).add_(m2, alpha=c["alpha_t"])
