"""Orphan — the ADOPT update rule fused with koptim's memory-efficient backend.

**Orphan** (``class Orphan``) is a *code-named, experimental* optimizer: the
**ADOPT** algorithm (Taniguchi et al., *Modified Adam Can Converge with Any
β2 with the Optimal Rate*, NeurIPS 2024, arXiv:2411.02853) running on the same
precision/memory machinery as :class:`~koptim.adafusion.Adafusion` and
:class:`~koptim.liofusion.Liofusion`. It is a separate class (Adafusion and
Liofusion are left byte-for-byte unchanged) and is named provisionally — it
earns a real name only once it proves itself in training. The code name nods to
the fact that the update is "adopted" from the ADOPT paper onto an existing
backend it did not grow up with.

**What ADOPT changes vs Adam (the convergence fix).** Vanilla Adam divides the
gradient by ``sqrt(v_t)`` where ``v_t`` already *includes the current*
``g_t**2`` — that coupling is exactly what breaks Adam's convergence for large
``β2``. ADOPT instead

1. normalizes by the **previous-step** second moment ``v_{t-1}`` (which does
   *not* see ``g_t``),
2. takes the first-moment EMA **of that normalized gradient** (not of the raw
   gradient),
3. steps, and
4. **only then** updates ``v_t = β2·v_{t-1} + (1-β2)·g_t**2``.

Plus an early-training elementwise clip on the normalized gradient
(``|normed_g| <= clip_lambda(step)``, default ``step**0.25``) for stability. The
``v_{t-1}``-before-update ordering is the whole point of the algorithm and is
preserved exactly here. With this change ADOPT converges at the optimal rate for
**any** ``β2`` — hence its high default ``β2 = 0.9999``.

**Why this fuses naturally with the koptim backend.** "Normalize the gradient,
then take momentum of the normalized update" is *structurally* what Adafusion
already does (Adafusion normalizes by a factored second moment, then momenta the
normalized update). Orphan reuses that structure and just reorders the second-
moment update to come *after* the normalization (ADOPT's ordering) instead of
*before* (Adam/Adafusion's ordering), and swaps Adafusion's RMS-clip for ADOPT's
elementwise clip.

**The exact per-parameter update Orphan implements** (matching the kozistr
``pytorch_optimizer`` ADOPT reference, decoupled WD / cautious adapted to
koptim's conventions):

.. code-block:: text

    # step 1 (per param): initialize the second moment, take NO step
    v <- g**2                       # factored: row/col means of g**2 (no EMA)

    # step >= 2:
    denom       = sqrt(v) (clamped) # v is v_{t-1}: does NOT include this g
    normed_g    = g / denom
    normed_g    = clamp(normed_g, -c, +c),  c = clip_lambda(step)   # default step**0.25
    m           <- lerp(m, normed_g, 1 - beta1)   # EMA of the *normalized* grad
    update      = m
    # cautious (default): zero coords where update disagrees with g, renormalize
    p          -= lr * (update + weight_decay * p)   # decoupled WD folded in
    v           <- beta2 * v + (1 - beta2) * g**2     # updated AFTER the step

**Second moment: factored *and* quantized (Adafactor-class memory).** Orphan uses
Adafusion's conv-aware factored second moment — a 2-D weight's ``v`` is row+col
EMAs (≈0 state), and a 4-D conv kernel ``[out,in,kh,kw]`` is reshaped to
``[out, in·kh·kw]`` before factoring. **The factored state IS the previous-step
``v``**: Orphan reconstructs ``1/sqrt(v_{t-1})`` from the *current* row/col,
normalizes, steps, and only then advances row/col with ``g**2`` — so ADOPT's
ordering holds with the factored representation, no separate full-``v`` buffer
needed. 1-D params (biases/norms) keep a full per-coordinate ``v`` (fp32), same
as Adafusion's non-factored branch.

  *Divergence flagged:* ADOPT clamps the **reconstructed** denominator,
  ``max(sqrt(v), eps)``. The factored path never materializes the full denom, so
  the ``eps`` guard is applied as an additive ``eps1`` on ``g**2`` *before* the
  row/col reductions (the Adafactor/HF convention Adafusion uses) rather than as
  a post-hoc clamp on ``sqrt(v)``. The 1-D non-factored path *does* clamp
  ``sqrt(v)`` by ``eps`` exactly as ADOPT does. On well-conditioned gradients the
  two eps placements are numerically indistinguishable; the additive form is what
  makes the factored memory win possible.

**First moment: the shared momentum codec.** The EMA-of-normalized-grad buffer is
stored through the same codec as Adafusion/Liofusion — ``bfloat16`` (~2 B/param),
``float32`` (4 B/param), ``int8`` (~1 B/param, per-row absmax) or ``4bit``
(~0.5 B/param, per-block absmax, nibble-packed). The codec's ``ema_*`` helpers do
precisely ADOPT's ``m.lerp_(normed_g, 1-beta1)`` and return the fp32 first moment
as the step delta, so they are reused verbatim.

**Reused vs new.** Reused from the shared backend: the factored second moment
(:mod:`koptim._factored`), the momentum codec and its dtype-safe checkpoint
restore (:mod:`koptim._momentum_codec`), the stochastic-rounding bf16 weight
update (:func:`koptim._stochastic_rounding.add_stochastic_`), cautious masking,
and the bucketed foreach batching. New here: ADOPT's ``v_{t-1}``-before-update
ordering, the step-1 second-moment initialization (no EMA, no step), and the
elementwise ``clip_lambda`` schedule (a scalar function of the step count, *not*
per-parameter state).

**Hyper-parameters.** ADOPT's default ``betas=(0.9, 0.9999)`` (the high ``β2`` is
safe by construction). ``lr`` is Adam-scale. ``clip_lambda`` is the early-step
clip schedule, a callable ``step -> float`` (default ``step**0.25``); pass
``None`` to disable the clip entirely.

It is a standard ``torch.optim.Optimizer`` with a single per-parameter step, so
it drops into per-parameter / gradient-release training loops unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any, Literal

import torch
from torch import Tensor
from torch.optim import Optimizer

from koptim._factored import update_factored_state
from koptim._momentum_codec import (
    _FOURBIT_BLOCK,
    _make_codec,
    _MomentumCodec,
    load_state_dict_preserving_dtypes,
)
from koptim._stochastic_rounding import add_stochastic_

__all__ = ["Orphan"]

_LOW_PRECISION = (torch.bfloat16, torch.float16)
MomentumDtype = Literal["bfloat16", "float32", "int8", "4bit"]
ClipLambda = Callable[[int], float] | Literal["default"] | None

# Mirrors Adafusion: the same adaptive-foreach constants (the batching machinery
# is shared verbatim; only the per-coordinate math differs).
_STACK_SAFETY_FRACTION = 0.10
_STACK_BYTES_PER_ELEM = 48
_MIN_STACK_ELEMS = 262_144
_DEFAULT_STACK_ELEMS = 64_000_000
_FOREACH_BATCH_CUTOFF = 2_000_000


def _default_clip_lambda(step: int) -> float:
    """ADOPT's default early-step clip schedule: ``step ** 0.25``."""
    return math.pow(step, 0.25)


def _is_low_precision(t: Tensor) -> bool:
    return t.dtype in _LOW_PRECISION


class Orphan(Optimizer):
    """ADOPT (arXiv:2411.02853) on koptim's factored-quantized backend (experimental).

    Args:
        params: parameters or param-group dicts.
        lr: learning rate (Adam-scale).
        betas: ``(beta1, beta2)``. ``beta1`` is the first-moment EMA decay (the EMA
            runs over the *normalized* gradient); ``beta2`` is the second-moment
            decay. ADOPT's default ``(0.9, 0.9999)`` — the high ``beta2`` is safe
            by construction (that is the paper's result).
        eps: ``(eps1, eps2)``. ``eps1`` is added to ``grad**2`` before the factored
            row/col reductions (Adafactor/HF convention) **and** is the
            ``max(sqrt(v), eps1)`` denominator clamp on the 1-D non-factored path
            (ADOPT's ``eps``). ``eps2`` is reserved/unused.
        weight_decay: decoupled (AdamW-style) weight decay, folded into the
            per-step delta.
        clip_lambda: ADOPT's early-step elementwise clip on the normalized
            gradient — a callable ``step -> float`` giving the symmetric clip
            bound ``|normed_g| <= clip_lambda(step)``. ``"default"`` (the default)
            uses ``step**0.25``; pass ``None`` to disable clipping; pass your own
            callable to customize. This is a scalar function of the global step
            count, **not** per-parameter state.
        cautious: cautious masking (Liang et al. 2024) — zero the update
            coordinates whose sign disagrees with the gradient, then renormalize to
            preserve the mean step magnitude. **On by default.**
        momentum_dtype: storage for the first-moment buffer — ``"bfloat16"``
            (default, ~2 B/param), ``"float32"`` (4 B/param), ``"int8"`` (~1
            B/param, per-row absmax) or ``"4bit"`` (~0.5 B/param, per-block absmax,
            nibble-packed). Same layout as Adafusion/Liofusion, so checkpoints
            resume bit-exactly.
        momentum_4bit_block: block size for ``momentum_dtype="4bit"``. Default
            ``128``. ``0``/negative means whole-tensor (single scale).
        bf16_method: weight-update strategy for low-precision params —
            ``"stochastic_rounding"`` (default), ``"kahan"`` (+2 B/param), or
            ``"none"``. No-op on fp32 params.
        foreach: batch the step across parameters with stacked multi-tensor ops
            instead of a per-parameter Python loop. Default ``True``. Matches the
            per-parameter path numerically (stochastic-rounding draws differ,
            unbiased either way). 0-D scalars, kahan and fp16+SR fall back to the
            per-parameter path.
        foreach_batch_cutoff: per-tensor element count above which a weight loops
            instead of stacking (a performance knob; default ``2_000_000``).
        foreach_stack_budget: max elements per stacked chunk. ``None`` (default)
            adapts to free VRAM; an int pins a fixed cap.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.9999),
        eps: tuple[float, float] = (1e-6, 1e-3),
        weight_decay: float = 0.0,
        *,
        clip_lambda: ClipLambda = "default",
        cautious: bool = True,
        momentum_dtype: MomentumDtype = "bfloat16",
        momentum_4bit_block: int = _FOURBIT_BLOCK,
        bf16_method: str = "stochastic_rounding",
        foreach: bool = True,
        foreach_batch_cutoff: int = _FOREACH_BATCH_CUTOFF,
        foreach_stack_budget: int | None = None,
    ) -> None:
        beta1, beta2 = float(betas[0]), float(betas[1])
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"betas[0] must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"betas[1] must be in [0, 1), got {beta2}")
        if lr < 0.0:
            raise ValueError(f"lr must be >= 0, got {lr}")
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

        # Resolve the clip schedule once. It is a scalar fn of the step count,
        # shared by all params/groups (NOT per-parameter state), so it lives on the
        # optimizer instance rather than in ``defaults``/``state``.
        if clip_lambda == "default":
            self._clip_lambda: Callable[[int], float] | None = _default_clip_lambda
        elif clip_lambda is None:
            self._clip_lambda = None
        elif callable(clip_lambda):
            self._clip_lambda = clip_lambda
        else:
            raise ValueError(
                f"clip_lambda must be 'default', None, or callable, got {clip_lambda!r}"
            )

        defaults = {
            "lr": lr,
            "betas": (beta1, beta2),
            "eps": (float(eps[0]), float(eps[1])),
            "weight_decay": weight_decay,
            "momentum_dtype": momentum_dtype,
            "momentum_4bit_block": momentum_4bit_block,
            "cautious": cautious,
            "bf16_method": bf16_method,
        }
        super().__init__(params, defaults)
        self._foreach = foreach
        self._foreach_batch_cutoff = foreach_batch_cutoff
        self._foreach_stack_budget = foreach_stack_budget
        self._codecs: dict[str, _MomentumCodec] = {}

    def _codec(self, group: dict[str, Any]) -> _MomentumCodec:
        md = group["momentum_dtype"]
        codec = self._codecs.get(md)
        if codec is None:
            codec = self._codecs[md] = _make_codec(md)
        return codec

    def _clip_bound(self, step: int) -> float | None:
        return None if self._clip_lambda is None else self._clip_lambda(step)

    # ------------------------------------------------------------------- state
    @torch.no_grad()
    def _init_state(self, p: Tensor, state: dict[str, Any], group: dict[str, Any]) -> None:
        """Allocate the factored/full second moment and the first-moment buffer.

        ``state["step"]`` counts this param's own steps so the ADOPT step-1
        initialization (``v <- g**2``, no weight step) and the ``clip_lambda``
        schedule are well-defined even for params added mid-run. The factored
        ``row``/``col`` (ndim>=2) or full ``v`` (ndim==1) is **left zero here** and
        seeded on the first step (ADOPT initializes ``v`` from the first gradient,
        not from an EMA against zero).
        """
        grad = p.grad
        state["step"] = 0
        factored = p.ndim >= 2
        if factored:
            gv = grad if p.ndim == 2 else grad.reshape(grad.shape[0], -1)
            row_shape = gv.shape[:-1]
            col_shape = gv.shape[:-2] + gv.shape[-1:]
            state["row"] = torch.zeros(row_shape, dtype=torch.float32, device=p.device)
            state["col"] = torch.zeros(col_shape, dtype=torch.float32, device=p.device)
        else:
            state["v"] = torch.zeros_like(grad, dtype=torch.float32)
        self._codec(group).init_state(state, grad, group)
        if _is_low_precision(p) and group["bf16_method"] == "kahan":
            state["shift"] = torch.zeros_like(p)

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
                    raise RuntimeError("Orphan does not support sparse gradients")
            if self._foreach and self._group_foreach_eligible(group) and params:
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
        return loss

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state, preserving the quantized first moment's stored dtype."""
        load_state_dict_preserving_dtypes(self, state_dict)

    # ----------------------------------------------------------------- foreach
    def _foreach_budget(self, device: torch.device) -> int:
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
        return group["bf16_method"] != "kahan"  # kahan needs a per-param shift buffer

    @staticmethod
    def _param_foreach_eligible(p: Tensor, group: dict[str, Any], cutoff: int) -> bool:
        if p.ndim == 0 or p.numel() > cutoff:
            return False
        if (
            group["bf16_method"] == "stochastic_rounding"
            and _is_low_precision(p)
            and p.dtype != torch.bfloat16  # fp16+SR is unsupported -> per-param (raises)
        ):
            return False
        if p.ndim > 2:
            return p.data.is_contiguous() and p.grad.is_contiguous()
        return True

    @torch.no_grad()
    def _step_foreach(self, params: list[Tensor], group: dict[str, Any], budget: int) -> None:
        """Batched ADOPT step.

        Params are bucketed exactly as Adafusion's: ``ndim >= 2`` factored
        ``[N, R, C]``, ``ndim == 1`` non-factored ``[N, L]``. Each bucket is further
        split by whether its members are on their step-1 (seed ``v``, no weight
        step) or step ``>= 2`` (the normalize-step-then-update path) — the two
        cannot share a stacked kernel because step 1 takes no weight step. In
        practice all params in a group share a step count, so this split is a
        no-op after the first iteration.
        """
        beta1, beta2 = group["betas"]
        eps1, _eps2 = group["eps"]
        lr = group["lr"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]
        codec = self._codec(group)

        factored_buckets: dict[tuple[Any, ...], list[Tensor]] = {}
        flat_buckets: dict[tuple[Any, ...], list[Tensor]] = {}
        for p in params:
            state = self.state[p]
            if not state:
                self._init_state(p, state, group)
            state["step"] += 1
            seeding = state["step"] == 1
            g = p.grad
            if g.ndim >= 2:
                matrixize = g.ndim > 2
                eff = (g.shape[0], g.numel() // g.shape[0]) if matrixize else tuple(g.shape)
                factored_buckets.setdefault((eff, p.dtype, matrixize, seeding), []).append(p)
            else:
                flat_buckets.setdefault((g.shape[0], p.dtype, seeding), []).append(p)

        for (eff, _dtype, matrixize, seeding), plist in factored_buckets.items():
            step = max(1, budget // max(eff[0] * eff[1], 1))
            for i in range(0, len(plist), step):
                self._factored_bucket(
                    plist[i:i + step], eff, matrixize, seeding,
                    beta1, beta2, eps1, lr, wd, cautious, bf16_method, codec,
                )
        for (length, _dtype, seeding), plist in flat_buckets.items():
            step = max(1, budget // max(length, 1))
            for i in range(0, len(plist), step):
                self._nonfactored_bucket(
                    plist[i:i + step], length, seeding,
                    beta1, beta2, eps1, lr, wd, cautious, bf16_method, codec,
                )

    @torch.no_grad()
    def _factored_bucket(
        self,
        plist: list[Tensor],
        eff: tuple[int, int],
        matrixize: bool,
        seeding: bool,
        beta1: float,
        beta2: float,
        eps1: float,
        lr: float,
        wd: float,
        cautious: bool,
        bf16_method: str,
        codec: _MomentumCodec,
    ) -> None:
        R, C = eff  # noqa: N806
        N = len(plist)  # noqa: N806

        def mat(t: Tensor) -> Tensor:
            return t.view(R, C) if matrixize else t

        rows = [self.state[p]["row"] for p in plist]
        cols = [self.state[p]["col"] for p in plist]
        grad = torch.stack([mat(p.grad).float() for p in plist])  # [N, R, C]

        grad_sq = grad * grad
        if eps1 > 0:
            grad_sq = grad_sq.add_(eps1)

        if seeding:
            # ADOPT step 1: v <- g**2 (no EMA against zero), take NO weight step.
            # Factored: seed row/col to the row/col means of g**2 directly.
            row = grad_sq.mean(dim=-1)  # [N, R]
            col = grad_sq.mean(dim=-2)  # [N, C]
            torch._foreach_copy_(rows, list(row.unbind(0)))
            torch._foreach_copy_(cols, list(col.unbind(0)))
            return

        # Normalize by v_{t-1} (the CURRENT row/col, not yet updated with this g).
        row = torch.stack(rows)  # [N, R]
        col = torch.stack(cols)  # [N, C]
        r_factor = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)  # [N, R, 1]
        c_factor = col.rsqrt().unsqueeze(-2)  # [N, 1, C]
        normed = grad.mul(r_factor).mul_(c_factor)  # [N, R, C]
        self._clip_(normed, plist)

        # First-moment EMA of the normalized grad, then the step. The codec returns
        # the fp32 first moment as the delta.
        states = [self.state[p] for p in plist]
        delta = codec.ema_stacked(states, normed, mat, (R, C), beta1)  # [N, R, C]

        if wd != 0:
            p_fp32 = torch.stack([mat(p.data).float() for p in plist])
            delta = delta.add_(p_fp32, alpha=wd)

        if cautious:
            mask = (delta * grad > 0).to(delta.dtype)
            denom = mask.reshape(N, -1).mean(dim=1).clamp_(min=1e-8).view(N, 1, 1)
            delta = delta.mul_(mask).div_(denom)

        delta.mul_(lr)
        pviews = [mat(p.data) for p in plist]
        weights = torch.stack(pviews)
        self._apply_subtract_batched(weights, delta, bf16_method)
        torch._foreach_copy_(pviews, list(weights.unbind(0)))

        # ADOPT: update v_t = beta2*v_{t-1} + (1-beta2)*g**2, AFTER the step.
        row.lerp_(grad_sq.mean(dim=-1), 1.0 - beta2)
        col.lerp_(grad_sq.mean(dim=-2), 1.0 - beta2)
        torch._foreach_copy_(rows, list(row.unbind(0)))
        torch._foreach_copy_(cols, list(col.unbind(0)))

    @torch.no_grad()
    def _nonfactored_bucket(
        self,
        plist: list[Tensor],
        length: int,
        seeding: bool,
        beta1: float,
        beta2: float,
        eps1: float,
        lr: float,
        wd: float,
        cautious: bool,
        bf16_method: str,
        codec: _MomentumCodec,
    ) -> None:
        N = len(plist)  # noqa: N806
        vs = [self.state[p]["v"] for p in plist]
        grad = torch.stack([p.grad.float() for p in plist])  # [N, L]
        grad_sq = grad * grad

        if seeding:
            # ADOPT step 1: v <- g**2, no weight step.
            torch._foreach_copy_(vs, list(grad_sq.unbind(0)))
            return

        v = torch.stack(vs)  # [N, L] = v_{t-1}
        denom = v.sqrt().clamp_(min=eps1)  # ADOPT's max(sqrt(v), eps), exact on 1-D
        normed = grad.div(denom)
        self._clip_(normed, plist)

        states = [self.state[p] for p in plist]
        delta = codec.ema_stacked(states, normed, lambda t: t, (length,), beta1)  # [N, L]

        if wd != 0:
            p_fp32 = torch.stack([p.data.float() for p in plist])
            delta = delta.add_(p_fp32, alpha=wd)

        if cautious:
            mask = (delta * grad > 0).to(delta.dtype)
            denom_c = mask.mean(dim=1).clamp_(min=1e-8).view(N, 1)
            delta = delta.mul_(mask).div_(denom_c)

        delta.mul_(lr)
        pviews = [p.data for p in plist]
        weights = torch.stack(pviews)
        self._apply_subtract_batched(weights, delta, bf16_method)
        torch._foreach_copy_(pviews, list(weights.unbind(0)))

        # Update v AFTER the step.
        v.mul_(beta2).add_(grad_sq, alpha=1.0 - beta2)
        torch._foreach_copy_(vs, list(v.unbind(0)))

    def _clip_(self, normed: Tensor, plist: list[Tensor]) -> None:
        """In-place ADOPT clip on the normalized grad using the shared step count.

        All params in a foreach bucket share the same step (they were bucketed only
        within one ``step()`` call and split by seeding), so the clip bound is read
        from the first param's step.
        """
        bound = self._clip_bound(self.state[plist[0]]["step"])
        if bound is not None:
            normed.clamp_(-bound, bound)

    @staticmethod
    def _apply_subtract_batched(weights: Tensor, delta_fp32: Tensor, bf16_method: str) -> None:
        if (
            _is_low_precision(weights)
            and bf16_method == "stochastic_rounding"
            and weights.dtype == torch.bfloat16
        ):
            add_stochastic_(weights, delta_fp32, alpha=-1.0)
        else:
            weights.sub_(delta_fp32.to(weights.dtype))

    # ---------------------------------------------------------- per-parameter
    @torch.no_grad()
    def _step_one_param(self, p: Tensor, group: dict[str, Any]) -> None:
        beta1, beta2 = group["betas"]
        eps1, _eps2 = group["eps"]
        lr = group["lr"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]

        state = self.state[p]
        if not state:
            self._init_state(p, state, group)
        state["step"] += 1
        step = state["step"]

        grad = p.grad if p.grad.dtype == torch.float32 else p.grad.float()
        ndim = grad.ndim
        factored = ndim >= 2

        grad_sq = grad * grad
        if eps1 > 0 and factored:
            grad_sq = grad_sq.add_(eps1)

        if step == 1:
            # ADOPT step 1: seed v from g**2, take NO weight step.
            if factored:
                matrixize = ndim > 2
                gsq = grad_sq.reshape(grad_sq.shape[0], -1) if matrixize else grad_sq
                state["row"].copy_(gsq.mean(dim=-1))
                state["col"].copy_(gsq.mean(dim=-2))
            else:
                state["v"].copy_(grad_sq)
            return

        # Normalize by v_{t-1} (state BEFORE this step's update).
        if factored:
            matrixize = ndim > 2
            gv = grad.reshape(grad.shape[0], -1) if matrixize else grad
            row, col = state["row"], state["col"]
            r_factor = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
            c_factor = col.rsqrt().unsqueeze(-2)
            normed = gv.mul(r_factor).mul_(c_factor)
            if matrixize:
                normed = normed.view_as(grad)
        else:
            v = state["v"]
            denom = v.sqrt().clamp_(min=eps1)  # ADOPT's max(sqrt(v), eps)
            normed = grad.div(denom)

        bound = self._clip_bound(step)
        if bound is not None:
            normed.clamp_(-bound, bound)

        # First-moment EMA of the normalized grad; codec returns fp32 delta.
        delta = self._codec(group).ema_one(state, normed, beta1)

        if wd != 0:
            p_fp32 = p.data if p.dtype == torch.float32 else p.data.float()
            delta = delta.add_(p_fp32, alpha=wd)

        if cautious:
            mask = (delta * grad > 0).to(delta.dtype)
            delta = delta.mul_(mask).div_(mask.mean().clamp_(min=1e-8))

        delta.mul_(lr)
        self._apply_subtract(p, delta, state, bf16_method)

        # ADOPT: update v AFTER the step.
        if factored:
            matrixize = ndim > 2
            gv = grad.reshape(grad.shape[0], -1) if matrixize else grad
            update_factored_state(gv, state["row"], state["col"], beta2, eps1)
        else:
            state["v"].mul_(beta2).add_(grad_sq, alpha=1.0 - beta2)

    @staticmethod
    def _apply_subtract(
        p: Tensor, delta_fp32: Tensor, state: dict[str, Any], bf16_method: str
    ) -> None:
        low = _is_low_precision(p)
        if low and bf16_method == "kahan":
            shift = state["shift"]
            shift.sub_(delta_fp32.to(p.dtype))
            p_before = p.detach().clone()
            p.add_(shift)
            shift.add_(p_before.sub_(p))
        elif low and bf16_method == "stochastic_rounding" and p.dtype == torch.bfloat16:
            add_stochastic_(p.data, delta_fp32, alpha=-1.0)
        else:
            p.data.sub_(delta_fp32.to(p.dtype))
