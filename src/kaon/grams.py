"""Grams — Gradient descent with Adaptive Momentum Scaling on kaon's backend.

Grams (Cao, Li, Liang, Song et al. 2024, *Grams: Gradient Descent with Adaptive
Momentum Scaling*, arXiv:2412.17107, ICLR 2025 Workshop) is **Adam's adaptive
magnitude with the update direction taken from the current gradient's sign**
rather than from the momentum's sign. It is implemented here on the precision and
memory machinery already proven in :class:`~kaon.adakaon.Adakaon` /
:class:`~kaon.adapnm.AdaPNM` (factored quantized second moment, quantized
first-moment storage, stochastic-rounding bf16 write-back, foreach batching).

**The idea (the Grams twist).** A standard Adam step is

.. code-block:: text

    update = m_hat / (sqrt(v_hat) + eps)        # direction = sign(m_hat)

whose *direction* is the sign of the (debiased) first moment ``m_hat``. Grams
keeps Adam's *magnitude* ``|m_hat| / (sqrt(v_hat) + eps)`` but replaces the
direction with the sign of the **current** gradient ``g``:

.. code-block:: text

    update = sign(g) * |m_hat| / (sqrt(v_hat) + eps)     # direction = sign(g)

i.e. ``update = grad.sign() * |exp_avg_hat|`` divided by the Adam denominator.
The momentum is used *solely* to scale the per-coordinate step magnitude; the
direction always agrees with the instantaneous gradient. The paper shows this
decoupling gives faster loss descent and better generalization than Adam, Lion,
and their cautious variants.

**The exact update (matches the official ``Gunale0926/Grams``):**

.. code-block:: text

    m  = beta1 * m + (1 - beta1) * g                 # Adam first moment
    v  = beta2 * v + (1 - beta2) * g^2               # Adam second moment (factored on >=2D)
    bc1 = 1 - beta1^t ; bc2 = 1 - beta2^t            # bias corrections
    step_size = lr * sqrt(bc2) / bc1
    denom     = sqrt(v) + eps
    update    = sign(g) * |m|                        # |m| (the /bc1 lives in step_size)
    p        -= lr * weight_decay * p                # decoupled (AdamW) WD
    p        -= step_size * update / denom

The ``1/bc1`` factor is folded into ``step_size`` and multiplies ``|m|`` (so the
effective numerator is ``|m_hat| = |m|/bc1``); ``sqrt(bc2)`` is likewise folded
into ``step_size`` so the effective denominator is ``sqrt(v_hat) = sqrt(v/bc2)``.
This reproduces the official line ``grad.sign_().mul_(exp_avg.abs())`` /
``denom = exp_avg_sq.sqrt().add_(eps)`` / ``step_size = lr*sqrt(bc2)/bc1``
exactly on the non-factored (1-D) fp32 path. (kozistr's ``pytorch_optimizer``
``Grams`` is identical except it adds ``eps`` to ``sqrt(v/bc2)`` instead of to the
raw ``sqrt(v)`` — a placement difference of order ``eps`` that is numerically
negligible; this port follows the official.)

**The factored second moment.** ``v`` reuses Adakaon's backend exactly:
``ndim >= 2`` weights factor ``v`` into row+column EMAs (conv kernels matrixized
to ``[out, in*kh*kw]`` first); ``ndim == 1`` keeps a full per-coordinate ``v``.
The denominator on the factored path is Adakaon's ``r_factor * c_factor``
reconstruction of ``1/sqrt(v_hat)`` (with ``bc2`` folded in), and ``eps`` is added
Adafactor-style via the factored ``eps1`` (added to ``grad**2`` before the
row/col reductions) since a scalar ``eps`` has no rank-1 analogue. On the 1-D
path ``eps`` is added to ``sqrt(v)`` exactly as the official does.

**Cautious masking and the sign(g) direction.** Cautious (Liang et al. 2024)
zeroes the update coordinates whose sign disagrees with the *current* gradient
(``delta * g <= 0``) and rescales survivors to preserve the mean magnitude.
Grams' update direction is **already** ``sign(g)`` per coordinate, so on the
factored path the final ``delta`` agrees with ``g`` wherever the reconstructed
denominator and ``|m_hat|`` are non-negative — which they always are — making the
mask all-ones (a literal no-op). It is wired anyway for consistency with the rest
of kaon and defaults to ``True``; on the **factored** path the reconstructed
``r_factor * c_factor`` denominator can leave a coordinate whose magnitude is zero
(``m_hat == 0``), which the mask ``delta*g > 0`` drops, so cautious is at most a
zero-magnitude no-op. See the tests: with the sign(g) direction cautious does not
change the trajectory.

**What is reused vs new.** Reused from Adakaon/AdaPNM's backend: the factored
second-moment helpers (:mod:`kaon._factored`), the momentum storage layout and
quant/dequant primitives in :mod:`kaon._momentum_codec` (int8 per-row absmax;
4-bit per-block absmax, nibble-packed), the stochastic-rounding bf16 weight update
(:func:`kaon._stochastic_rounding.add_stochastic_`),
``load_state_dict_preserving_dtypes`` for dtype-safe resume, and the bucketed
foreach batching pattern. New here: the ``sign(g) * |m_hat|`` numerator (the
Grams twist) with a single standard-Adam momentum buffer. The shared codec's
``ema_*`` helpers compute a *momentum of the (lr-scaled) update* and return that
as the delta, which is the *wrong* quantity for Grams (Grams needs the raw
first-moment EMA so it can take ``|m_hat|`` and pair it with ``sign(g)``), so —
exactly as AdaPNM does — Grams uses the codec's storage + *read-only*
dequant/requant primitives and runs the raw-gradient EMA itself.

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

__all__ = ["Grams"]

MomentumDtype = Literal["bfloat16", "float32", "int8", "4bit"]

# Performance / memory knobs mirror Adakaon (see that module for the rationale).
# One momentum + factored v: Adakaon's 48 covers this single-momentum layout.
_STACK_BYTES_PER_ELEM = 48


def _rms(t: Tensor) -> Tensor:
    return t.norm(2) / math.sqrt(max(t.numel(), 1))


class Grams(Optimizer):
    """Grams (gradient descent with Adaptive Momentum Scaling) on kaon's backend.

    The update direction is the **sign of the current gradient**; Adam's
    bias-corrected first/second moments supply only the per-coordinate magnitude
    ``|m_hat| / (sqrt(v_hat) + eps)``. See the module docstring for the exact rule.

    Args:
        params: parameters or param-group dicts.
        lr: learning rate. Default ``1e-3``.
        betas: ``(beta1, beta2)`` — first/second-moment EMA decays. Default
            ``(0.9, 0.999)`` (the paper / official defaults).
        eps: term added to the second-moment denominator for stability. Default
            ``1e-6`` (the official Grams default). On the non-factored (1-D) path
            it is added to ``sqrt(v)`` exactly as the official does; on the
            factored path it is folded into the Adafactor ``eps1`` (added to
            ``grad**2`` before the row/col reductions), the only consistent
            analogue for a factored ``v``.
        weight_decay: decoupled (AdamW-style) weight decay, applied as
            ``p -= lr * weight_decay * p`` independently of the cautious-gated
            update (matching the official, where WD is a separate term).
        cautious: cautious masking (Liang et al. 2024) on the final step vs the
            gradient. **On by default** for consistency with the rest of kaon, but
            Grams' direction is *already* ``sign(g)``, so the mask is at most a
            zero-magnitude no-op — verified by the direction/cautious tests.
        gradient_centralization: Gradient Centralization (Yong et al. 2020) on
            ``ndim >= 2`` grads before the step. Default ``True`` (as Adakaon).
        momentum_dtype: storage for the first-moment buffer — ``"bfloat16"``
            (default, ~2 B/param), ``"float32"`` (4 B/param), ``"int8"``
            (~1 B/param, per-row absmax), or ``"4bit"`` (~0.5 B/param, per-block
            absmax, nibble-packed). Same layout as Adakaon's first moment, so
            checkpoints resume bit-exactly via ``load_state_dict``.
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
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
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
        beta1, beta2 = float(betas[0]), float(betas[1])
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"betas[0] must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"betas[1] must be in [0, 1), got {beta2}")
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
            "betas": (beta1, beta2),
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
        # Single first-moment buffer through the shared codec storage layout.
        md = group["momentum_dtype"]
        if md in ("bfloat16", "float32"):
            dtype = torch.bfloat16 if md == "bfloat16" else torch.float32
            state["m"] = torch.zeros_like(grad, dtype=dtype)
        elif md == "int8":
            state["m"] = torch.zeros_like(grad, dtype=torch.int8)
            state["m_scale"] = torch.ones(
                (grad.shape[0],) + (1,) * (grad.ndim - 1) if grad.ndim >= 2 else (),
                dtype=torch.float32, device=grad.device,
            )
        else:  # 4bit
            numel = grad.numel()
            bs = self._block_size(grad, group)
            nblocks = (numel + bs - 1) // bs
            state["m"] = torch.full(
                ((numel + 1) // 2,), 0x88, dtype=torch.uint8, device=grad.device
            )
            state["m_scale"] = torch.ones(nblocks, dtype=torch.float32, device=grad.device)
            state["m_numel"] = numel
            state["m_block"] = bs
        if is_low_precision(p) and group["bf16_method"] == "kahan":
            state["shift"] = torch.zeros_like(p)

    # -------------------------------------------------- momentum read / write
    @staticmethod
    def _dequant_one(state: dict[str, Any], md: str, like: Tensor) -> Tensor:
        """Read the stored momentum back as a fresh fp32 tensor shaped like ``like``.

        Buffers are stored in the param's original shape; conv kernels are
        matrixized at use-site so ``like`` may be the ``[R, C]`` view — reshape to
        match (the per-row int8 scale's row grouping is preserved across the
        reshape because dim-0 is unchanged).
        """
        if md in ("bfloat16", "float32"):
            # .clone() the fp32 path: ``.float()`` returns the SAME tensor for an
            # fp32 buffer, and Grams mutates the returned momentum in place
            # (``mul_``/``add_``/``abs_``), so an alias would corrupt stored state
            # (the known kaon fp32-aliasing footgun).
            m = state["m"]
            return (m.clone() if m.dtype == torch.float32 else m.float()).reshape_as(like)
        if md == "int8":
            return state["m"].float().mul_(state["m_scale"]).reshape_as(like)
        m = _dequant_4bit(state["m"], state["m_scale"], state["m_numel"], state["m_block"])
        return m.view_as(like)

    @staticmethod
    def _store_one(state: dict[str, Any], md: str, m_fp32: Tensor) -> None:
        """Write an updated fp32 momentum back into the configured storage layout."""
        if md in ("bfloat16", "float32"):
            tgt = state["m"]
            tgt.copy_(m_fp32.reshape(tgt.shape))
        elif md == "int8":
            m_orig = m_fp32.reshape(state["m"].shape)
            state["m"], state["m_scale"] = _quant_int8(m_orig)
        else:  # 4bit
            packed, scale, _ = _quant_4bit(m_fp32, state["m_block"])
            state["m"], state["m_scale"] = packed, scale

    @staticmethod
    def _dequant_stacked(
        states: list[dict[str, Any]], md: str, shape: tuple[int, ...]
    ) -> Tensor:
        """Stacked fp32 momentum ``[N, *shape]`` from per-param storage."""
        n = len(states)
        per = 1
        for d in shape:
            per *= d
        if md in ("bfloat16", "float32"):
            return torch.stack([s["m"].reshape(shape) for s in states]).float()
        if md == "int8":
            row = shape[0] if len(shape) >= 2 else 1
            rest = max(per // row, 1)
            m = torch.stack([s["m"].reshape(row, rest) for s in states]).float()  # [N, R, rest]
            scale = torch.stack([s["m_scale"].reshape(row, 1) for s in states])   # [N, R, 1]
            return m.mul_(scale).reshape((n, *shape))
        packed = torch.stack([s["m"] for s in states])
        sc = torch.stack([s["m_scale"] for s in states])
        bs = states[0]["m_block"]
        return _dequant_4bit_stacked(packed, sc, per, bs).reshape((n, *shape))

    @staticmethod
    def _store_stacked(states: list[dict[str, Any]], md: str, m_fp32: Tensor) -> None:
        """Write stacked fp32 momentum ``[N, *shape]`` back into per-param storage."""
        n = m_fp32.shape[0]
        shape = tuple(m_fp32.shape[1:])
        per = 1
        for d in shape:
            per *= d
        if md in ("bfloat16", "float32"):
            ms = [s["m"].reshape(shape) for s in states]
            torch._foreach_copy_(ms, list(m_fp32.unbind(0)))
        elif md == "int8":
            row = shape[0] if len(shape) >= 2 else 1
            rest = max(per // row, 1)
            q, new_scale = _quant_int8_stacked(m_fp32.reshape(n, row, rest))  # [N,R,rest]->[N,R,1]
            torch._foreach_copy_(
                [s["m"].reshape(row, rest) for s in states], list(q.unbind(0))
            )
            for s, sc in zip(states, new_scale.unbind(0), strict=True):  # sc: [R, 1]
                s["m_scale"] = sc.reshape(row, 1) if len(shape) >= 2 else sc.reshape(1)
        else:  # 4bit
            bs = states[0]["m_block"]
            new_packed, new_scale = _quant_4bit_stacked(m_fp32.reshape(n, per), bs)
            torch._foreach_copy_([s["m"] for s in states], list(new_packed.unbind(0)))
            for s, sc in zip(states, new_scale.unbind(0), strict=True):
                s["m_scale"].copy_(sc)

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
                    raise RuntimeError("Grams does not support sparse gradients")
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
        """Restore state, preserving the quantized first moment's stored dtype.

        torch's default ``load_state_dict`` upcasts every state tensor to the
        param's dtype (fp32), which would silently inflate bf16/int8/4bit momentum
        back to fp32 on resume. Delegate to the shared helper that restores each
        tensor to how it was checkpointed.
        """
        load_state_dict_preserving_dtypes(self, state_dict)

    # ----------------------------------------------------------- coefficients
    @staticmethod
    def _coeffs(group: dict[str, Any]) -> dict[str, float]:
        """Per-step scalar coefficients (shared by the per-param and foreach paths)."""
        beta1, beta2 = group["betas"]
        step = group["step"]
        bc1 = 1.0 - beta1 ** step
        bc2_sq = math.sqrt(1.0 - beta2 ** step)
        return {
            "beta1": beta1,
            "beta2": beta2,
            "bc2_sq": bc2_sq,
            # step_size = lr * sqrt(bc2) / bc1  (official: the 1/bc1 debiases |m|,
            # the sqrt(bc2) debiases the denom).
            "step_size": group["lr"] * bc2_sq / bc1,
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

        # Factored second-moment EMA (HF eps1 placement: eps1 into grad^2).
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

        # First-moment EMA (standard Adam decay), then the Grams numerator
        # sign(g) * |m|. (The 1/bc1 debias of |m| lives in step_size.)
        m = self._dequant_stacked(states, md, (R, C)).reshape((len(plist), R, C))
        m.mul_(c["beta1"]).add_(grad, alpha=1.0 - c["beta1"])
        self._store_stacked(states, md, m)
        # The Grams twist: magnitude from |m|, direction from sign(g).
        numer = grad.sign().mul_(m.abs_())                                # [N, R, C]

        delta = numer.mul_(inv_denom).mul_(c["step_size"])               # full update step

        if cautious:
            delta = cautious_batched_(delta, grad)

        # Decoupled weight decay (separate term, like the official): p -= lr*wd*p.
        if wd != 0:
            self._apply_decoupled_wd_batched(plist, mat, group["lr"] * wd)

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
        eps = group["eps"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]

        states = [self.state[p] for p in plist]
        vs = [s["v"] for s in states]

        grad = torch.stack([p.grad.float() for p in plist])               # [N, L]
        v = torch.stack(vs)                                               # [N, L]

        # Full per-coordinate second moment (1-D). eps goes on the denominator
        # (official), NOT folded into grad^2.
        v.mul_(c["beta2"]).addcmul_(grad, grad, value=1.0 - c["beta2"])
        torch._foreach_copy_(vs, list(v.unbind(0)))

        denom = v.sqrt().add_(eps)                                        # sqrt(v) + eps

        m = self._dequant_stacked(states, md, (length,)).reshape((len(plist), length))
        m.mul_(c["beta1"]).add_(grad, alpha=1.0 - c["beta1"])
        self._store_stacked(states, md, m)
        numer = grad.sign().mul_(m.abs_())                                # sign(g) * |m|

        delta = numer.div_(denom).mul_(c["step_size"])                   # [N, L]

        if cautious:
            delta = cautious_batched_(delta, grad)

        if wd != 0:
            self._apply_decoupled_wd_batched(plist, lambda t: t, group["lr"] * wd)

        subtract_batched_([p.data for p in plist], delta, bf16_method)

    @torch.no_grad()
    def _apply_decoupled_wd_batched(self, plist: list[Tensor], mat: Any, factor: float) -> None:
        """In-place decoupled WD ``p *= (1 - factor)`` on the (matrixized) weights."""
        torch._foreach_mul_([mat(p.data) for p in plist], 1.0 - factor)

    # ---------------------------------------------------------- per-parameter
    @torch.no_grad()
    def _step_one_param(self, p: Tensor, group: dict[str, Any]) -> None:
        c = self._coeffs(group)
        md = group["momentum_dtype"]
        eps = group["eps"]
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
            update_factored_state(gv, state["row"], state["col"], c["beta2"], eps)
            r_factor, c_factor = factored_inv_sqrt_factors(state["row"], state["col"])
            inv_denom = (r_factor * c_factor).mul_(c["bc2_sq"])           # 1/sqrt(v_hat)
            m = self._dequant_one(state, md, gv)
            m.mul_(c["beta1"]).add_(gv, alpha=1.0 - c["beta1"])
            self._store_one(state, md, m)
            numer = gv.sign().mul_(m.abs_())                             # sign(g) * |m|
            delta = numer.mul_(inv_denom).mul_(c["step_size"])
            if matrixize:
                delta = delta.reshape_as(grad)
        else:
            v = state["v"]
            v.mul_(c["beta2"]).addcmul_(grad, grad, value=1.0 - c["beta2"])
            denom = v.sqrt().add_(eps)                                    # sqrt(v) + eps
            m = self._dequant_one(state, md, grad)
            m.mul_(c["beta1"]).add_(grad, alpha=1.0 - c["beta1"])
            self._store_one(state, md, m)
            numer = grad.sign().mul_(m.abs_())                           # sign(g) * |m|
            delta = numer.div_(denom).mul_(c["step_size"])

        if cautious:
            delta = cautious_one_(delta, grad)

        # Decoupled weight decay (separate term, like the official): p -= lr*wd*p.
        if wd != 0:
            p.data.mul_(1.0 - group["lr"] * wd)

        subtract_one_(p, delta, state, bf16_method)
