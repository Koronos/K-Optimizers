"""Adan — Adaptive Nesterov Momentum Algorithm on kaon's memory backend.

Adan (Xie et al. 2022, *Adan: Adaptive Nesterov Momentum Algorithm for Faster
Optimizing Deep Models*, arXiv:2208.06677) reformulated Nesterov's extrapolation
as an explicit *gradient-difference* momentum, giving a faster, more robust
optimizer than Adam/AdamW across ViT / ResNet / **Stable Diffusion**. This port
keeps Adan's exact update while reusing the precision and memory machinery proven
in :class:`~kaon.adakaon.Adakaon` / :class:`~kaon.adapnm.AdaPNM`: a factored
quantized second moment (~0 state for ``ndim >= 2`` weights) and the bf16-correct
weight write-back, so Adan fits the same ≤16GB diffusion fine-tuning budget.

**The three buffers (Adan's signature).** Adan keeps, per parameter:

* ``m``    — gradient momentum (Adam's first moment), decay ``beta1``.
* ``diff`` — momentum of the *gradient difference* ``g_t - g_{t-1}``, decay ``beta2``.
* ``n``    — second moment of a **look-ahead** quantity, decay ``beta3``.
* ``g_prev`` — the previous step's gradient (to form the difference).

**The exact update (matches the official ``sail-sg/Adan`` and kozistr's port).**
Betas are stored as the official does — the **retention** factors (large), so
``m = beta1*m + (1-beta1)*g`` etc. The official default is
``betas=(0.98, 0.92, 0.99)`` (equivalently ``1-beta = (0.02, 0.08, 0.01)`` as the
paper writes them):

.. code-block:: text

    diff = g_t - g_prev                                  # 0 at t=1 (g_prev := g_1)
    m    = beta1*m    + (1-beta1)*g_t
    diff_ema = beta2*diff_ema + (1-beta2)*diff
    u    = g_t + beta2*diff                              # the look-ahead quantity
    n    = beta3*n    + (1-beta3)*u^2

    bc1  = 1 - beta1^t ;  bc2 = 1 - beta2^t ;  bc3 = 1 - beta3^t
    denom = sqrt(n)/sqrt(bc3) + eps
    step_size      = lr / bc1
    step_size_diff = lr * beta2 / bc2
    # prox (no_prox=False, default):
    p -= step_size*(m/denom) + step_size_diff*(diff_ema/denom)
    p  = p / (1 + lr*weight_decay)
    # no_prox=True:
    p  = p * (1 - lr*weight_decay)
    p -= step_size*(m/denom) + step_size_diff*(diff_ema/denom)
    g_prev = g_t

The two ``step_size`` terms share the *same* ``1/denom`` and the *same* sign-able
direction, so they combine into one fp32 ``update`` numerator
``m/bc1 + beta2*diff_ema/bc2`` that is then divided by ``denom`` and scaled by
``lr`` — exactly the official two-``addcdiv`` result, but as a single delta so the
backend's cautious mask / bf16 write-back apply once.

**The look-ahead term ``u = g_t + beta2*diff`` is Adan's signature and the easiest
to get subtly wrong.** Note ``beta2`` here is the *retention* factor (0.92 by
default), NOT ``1-beta2``; this is the official's ``neg_grad_or_diff.mul_(beta2)
.add_(grad)`` line. ``u`` is what gets squared into ``n`` (the second moment is of
the gradient *plus* the look-ahead difference, not of the gradient alone).

**The factored second moment.** ``n`` reuses Adakaon's backend exactly: ``ndim >=
2`` weights factor ``n`` into row+column EMAs (conv kernels matrixized to
``[out, in*kh*kw]`` first), ``ndim == 1`` keeps a full per-coordinate ``n``. Since
:func:`kaon._factored.update_factored_state` *squares its input internally* and
mixes with weight ``1-beta``, we pass the **linear** look-ahead quantity ``u``
(not ``u^2``) and ``beta3`` as its ``beta2`` argument — the convention lines up
(decay ``beta3`` ⇒ mix weight ``1-beta3``). ``bc3`` is folded into the
reconstructed inverse denominator (``* sqrt(bc3)``), matching the official
``sqrt(n)/sqrt(bc3)``.

**Memory cost (be honest — Adan is the heaviest kaon candidate).** Adan carries
THREE full-size-ish codec buffers (``m``, ``diff``, ``g_prev``) plus the ~0-state
factored ``n``. At ``momentum_dtype="bfloat16"`` that is ~6 B/param vs Adam's ~4
B/param (one momentum + factored v ≈ 2 B/param in kaon); at ``"int8"`` ~3 B/param;
at ``"4bit"`` ~1.5 B/param. So Adan's floor is ~3x a single-momentum kaon
optimizer at the same dtype — the price of the gradient-difference machinery. For
the tightest budgets prefer ``int8``/``4bit`` momentum.

**Reused vs new.** Reused from Adakaon's backend: the factored second-moment
helpers (:mod:`kaon._factored`), the momentum storage layout + quant/dequant
primitives (:mod:`kaon._momentum_codec`), the stochastic-rounding bf16 weight
update, ``load_state_dict_preserving_dtypes``, gradient centralization, cautious
masking, and the bucketed foreach batching. New here: the **three**-buffer
gradient-difference bookkeeping (read ``g_prev``, form ``diff``, EMA ``m`` and
``diff_ema``, build the look-ahead ``u``, store ``g_prev``), the prox vs no_prox
weight-decay forms, and the two-step-size combined numerator. Like AdaPNM, the
shared codec's ``ema_*`` helpers do a single-buffer Adam EMA and cannot be reused
verbatim, so Adan uses the codec's storage + *read-only* dequant/requant
primitives and runs the three raw EMAs itself.

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

__all__ = ["Adan"]

MomentumDtype = Literal["bfloat16", "float32", "int8", "4bit"]

# Three codec buffers (m, diff, g_prev) + factored n: heavier working set than
# AdaPNM's two momenta, so a touch above its 64.
_STACK_BYTES_PER_ELEM = 80


class Adan(Optimizer):
    """Adan (Adaptive Nesterov Momentum) on Adakaon's memory backend.

    Args:
        params: parameters or param-group dicts.
        lr: learning rate. **Adan's LR scale differs from AdamW** — the official
            suggests it can be ~5-10x larger than the AdamW LR for the same task.
            Default ``1e-3`` (the official default).
        betas: ``(beta1, beta2, beta3)`` as the official stores them — the EMA
            **retention** factors (large): ``beta1`` for the gradient momentum
            ``m``, ``beta2`` for the gradient-difference momentum ``diff_ema`` AND
            the look-ahead coefficient ``u = g + beta2*diff``, ``beta3`` for the
            second moment ``n``. Default ``(0.98, 0.92, 0.99)`` (paper's
            ``1-beta = (0.02, 0.08, 0.01)``).
        eps: term added to the second-moment denominator ``sqrt(n)/sqrt(bc3)`` for
            stability (official placement: *added to the denominator*, not folded
            into ``grad^2``). On the factored path it is applied as the denominator
            ``eps`` on the reconstructed ``1/sqrt(n_hat)`` so the placement matches
            the non-factored path; the factored ``eps1`` is kept ``0``.
        weight_decay: decoupled weight decay. With ``no_prox=False`` (default,
            Adan's proximal form) applied as ``p /= (1 + lr*weight_decay)`` AFTER
            the moment step; with ``no_prox=True`` as ``p *= (1 - lr*weight_decay)``
            BEFORE the step. Not gated by cautious.
        no_prox: select the weight-decay form. ``False`` (default) is Adan's
            proximal update (the paper's default); ``True`` is the decoupled-AdamW
            ordering.
        cautious: cautious masking (Liang et al. 2024) on the final combined step
            vs the gradient. On by default (kaon convention); pin ``False`` for
            base-correctness parity with the reference.
        gradient_centralization: subtract the per-output-row gradient mean for
            ``ndim >= 2`` weights before the step (Yong et al. 2020). On by default
            (kaon convention); pin ``False`` in base-correctness tests.
        max_grad_norm: optional global gradient-norm clip (the official's
            ``max_grad_norm``). ``0`` (default) disables it — kaon users typically
            clip outside the optimizer; exposed for parity. When ``> 0``, all grads
            in the optimizer are scaled by ``min(1, max_grad_norm / ||g||)`` before
            the step.
        momentum_dtype: storage for the THREE codec buffers (``m``, ``diff``,
            ``g_prev``) — ``"bfloat16"`` (default), ``"float32"``, ``"int8"``
            (per-row absmax) or ``"4bit"`` (per-block absmax, nibble-packed). Adan
            carries three such buffers, so its momentum floor is ~3x a
            single-momentum optimizer at the same dtype (the price of Adan).
        momentum_4bit_block: block size for ``momentum_dtype="4bit"``. Default
            ``128``. ``0``/negative means whole-tensor.
        bf16_method: weight-update strategy for low-precision params —
            ``"stochastic_rounding"`` (default), ``"kahan"`` (+2 B/param), or
            ``"none"``.
        foreach: batch the step across parameters. Default ``True``. Numerically
            matches the per-parameter path (verified bit-exact in tests; SR draws
            differ, unbiased either way). 0-D scalars, kahan, and fp16+SR fall back
            to the per-parameter path.
        foreach_batch_cutoff: per-tensor element count above which a weight loops
            instead of stacking (a performance knob; default ``2_000_000``).
        foreach_stack_budget: max elements per stacked chunk. ``None`` (default)
            adapts to free VRAM; an int pins a fixed cap.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.98, 0.92, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        no_prox: bool = False,
        cautious: bool = True,
        gradient_centralization: bool = True,
        max_grad_norm: float = 0.0,
        momentum_dtype: MomentumDtype = "bfloat16",
        momentum_4bit_block: int = _FOURBIT_BLOCK,
        bf16_method: str = "stochastic_rounding",
        foreach: bool = True,
        foreach_batch_cutoff: int = FOREACH_BATCH_CUTOFF,
        foreach_stack_budget: int | None = None,
    ) -> None:
        beta1, beta2, beta3 = (float(betas[0]), float(betas[1]), float(betas[2]))
        for name, b in (("betas[0]", beta1), ("betas[1]", beta2), ("betas[2]", beta3)):
            if not 0.0 <= b < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {b}")
        if lr < 0.0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if eps < 0.0:
            raise ValueError(f"eps must be >= 0, got {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        if max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be >= 0, got {max_grad_norm}")
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
            "eps": float(eps),
            "weight_decay": weight_decay,
            "no_prox": no_prox,
            "cautious": cautious,
            "gradient_centralization": gradient_centralization,
            "max_grad_norm": float(max_grad_norm),
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
    def _alloc_buffer(
        self,
        prefix: str,
        grad: Tensor,
        state: dict[str, Any],
        group: dict[str, Any],
        init: Tensor | None = None,
    ) -> None:
        """Allocate one codec buffer (keys ``prefix``, ``prefix_scale`` …).

        Storage layout matches :mod:`kaon._momentum_codec` exactly so all three
        buffers resume bit-exactly via ``load_state_dict_preserving_dtypes``. If
        ``init`` is given (fp32, param's original shape), the buffer is initialised
        to that value instead of zero — used for ``g_prev := g_1`` at t=1, which
        makes the first-step gradient difference exactly zero (matching the
        official's ``neg_pre_grad = -grad`` initialisation).
        """
        md = group["momentum_dtype"]
        if md in ("bfloat16", "float32"):
            dtype = torch.bfloat16 if md == "bfloat16" else torch.float32
            state[prefix] = (
                torch.zeros_like(grad, dtype=dtype)
                if init is None
                else init.to(dtype)
            )
        elif md == "int8":
            if init is None:
                state[prefix] = torch.zeros_like(grad, dtype=torch.int8)
                state[f"{prefix}_scale"] = torch.ones(
                    (grad.shape[0],) + (1,) * (grad.ndim - 1) if grad.ndim >= 2 else (),
                    dtype=torch.float32, device=grad.device,
                )
            else:
                q, scale = _quant_int8(init)
                state[prefix] = q
                state[f"{prefix}_scale"] = scale
        else:  # 4bit
            numel = grad.numel()
            bs = self._block_size(grad, group)
            nblocks = (numel + bs - 1) // bs
            if init is None:
                state[prefix] = torch.full(
                    ((numel + 1) // 2,), 0x88, dtype=torch.uint8, device=grad.device
                )
                state[f"{prefix}_scale"] = torch.ones(
                    nblocks, dtype=torch.float32, device=grad.device
                )
            else:
                packed, scale, _ = _quant_4bit(init, bs)
                state[prefix] = packed
                state[f"{prefix}_scale"] = scale
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
            state["n"] = torch.zeros_like(grad, dtype=torch.float32)
        # m and diff start at zero; g_prev starts at the (possibly clipped) first
        # gradient so the t=1 difference is exactly zero (official semantics).
        self._alloc_buffer("m", grad, state, group)
        self._alloc_buffer("diff", grad, state, group)
        self._alloc_buffer("g_prev", grad, state, group, init=grad.float())
        if is_low_precision(p) and group["bf16_method"] == "kahan":
            state["shift"] = torch.zeros_like(p)

    # ----------------------------------------------------- codec read / write
    @staticmethod
    def _dequant_one(state: dict[str, Any], prefix: str, md: str, like: Tensor) -> Tensor:
        """Read one stored buffer back as a fresh fp32 tensor shaped like ``like``.

        Always returns a tensor the caller may mutate in place: for the fp32 codec
        ``.float()`` is a no-op that would alias the stored buffer, so clone it (the
        bf16/int8/4bit paths already materialise a fresh fp32 tensor).
        """
        if md == "float32":
            return state[prefix].clone().reshape_as(like)
        if md == "bfloat16":
            return state[prefix].float().reshape_as(like)
        if md == "int8":
            return state[prefix].float().mul_(state[f"{prefix}_scale"]).reshape_as(like)
        m = _dequant_4bit(
            state[prefix], state[f"{prefix}_scale"],
            state[f"{prefix}_numel"], state[f"{prefix}_block"],
        )
        return m.view_as(like)

    @staticmethod
    def _store_one(state: dict[str, Any], prefix: str, md: str, m_fp32: Tensor) -> None:
        """Write an updated fp32 buffer back into the configured storage layout."""
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
        """Stacked fp32 buffer ``[N, *shape]`` from per-param storage."""
        n = len(states)
        per = 1
        for d in shape:
            per *= d
        if md in ("bfloat16", "float32"):
            return torch.stack([s[prefix].reshape(shape) for s in states]).float()
        if md == "int8":
            row = shape[0] if len(shape) >= 2 else 1
            rest = max(per // row, 1)
            m = torch.stack([s[prefix].reshape(row, rest) for s in states]).float()
            scale = torch.stack([s[f"{prefix}_scale"].reshape(row, 1) for s in states])
            return m.mul_(scale).reshape((n, *shape))
        packed = torch.stack([s[prefix] for s in states])
        sc = torch.stack([s[f"{prefix}_scale"] for s in states])
        bs = states[0][f"{prefix}_block"]
        return _dequant_4bit_stacked(packed, sc, per, bs).reshape((n, *shape))

    @staticmethod
    def _store_stacked(
        states: list[dict[str, Any]], prefix: str, md: str, m_fp32: Tensor
    ) -> None:
        """Write stacked fp32 buffer ``[N, *shape]`` back into per-param storage."""
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
            q, new_scale = _quant_int8_stacked(m_fp32.reshape(n, row, rest))
            torch._foreach_copy_(
                [s[prefix].reshape(row, rest) for s in states], list(q.unbind(0))
            )
            for s, sc in zip(states, new_scale.unbind(0), strict=True):
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
        # Optional global grad-norm clip (official max_grad_norm). Computed across
        # ALL params with grads, then folded into each grad in place before the step.
        clip = self._global_clip()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            for p in params:
                if p.grad.is_sparse:
                    raise RuntimeError("Adan does not support sparse gradients")
            group["step"] += 1
            if not params:
                continue
            if clip is not None:
                for p in params:
                    p.grad.mul_(clip)
            if group["gradient_centralization"]:
                centralize_grads_(params)
            if self._foreach and self._group_foreach_eligible(group):
                chunk_budget = foreach_budget(
                    self._foreach_stack_budget, self._foreach_batch_cutoff,
                    _STACK_BYTES_PER_ELEM, params[0].device,
                )
                cutoff = min(self._foreach_batch_cutoff, chunk_budget // 3)
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

    @torch.no_grad()
    def _global_clip(self) -> float | None:
        """Return ``min(1, max_grad_norm / ||g||)`` over all grads, or None if off.

        The clip is a single scalar shared by every param group (the official
        computes one global norm). Returns None when no group enables it.
        """
        mgn = max(
            (g["max_grad_norm"] for g in self.param_groups), default=0.0
        )
        if mgn <= 0.0:
            return None
        eps = min((g["eps"] for g in self.param_groups), default=1e-8)
        total = None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    sq = (p.grad.float() ** 2).sum()
                    total = sq if total is None else total + sq
        if total is None:
            return None
        norm = total.sqrt().item()
        return min(1.0, mgn / (norm + eps))

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state, preserving each quantized buffer's stored dtype."""
        load_state_dict_preserving_dtypes(self, state_dict)

    # ----------------------------------------------------------- coefficients
    @staticmethod
    def _coeffs(group: dict[str, Any]) -> dict[str, float]:
        """All per-step scalar coefficients (shared by per-param and foreach paths)."""
        beta1, beta2, beta3 = group["betas"]
        step = group["step"]
        lr = group["lr"]
        bc1 = 1.0 - beta1 ** step
        bc2 = 1.0 - beta2 ** step
        bc3_sqrt = math.sqrt(1.0 - beta3 ** step)
        return {
            "beta1": beta1,
            "beta2": beta2,
            "beta3": beta3,
            "bc3_sqrt": bc3_sqrt,
            "step_size": lr / bc1,                 # weights m / denom
            "step_size_diff": lr * beta2 / bc2,    # weights diff_ema / denom
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
        eps = group["eps"]
        wd = group["weight_decay"]
        no_prox = group["no_prox"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]

        def mat(t: Tensor) -> Tensor:
            return t.view(R, C) if matrixize else t

        states = [self.state[p] for p in plist]
        rows = [s["row"] for s in states]
        cols = [s["col"] for s in states]

        grad = torch.stack([mat(p.grad).float() for p in plist])          # [N, R, C]
        row = torch.stack(rows)                                           # [N, R]
        col = torch.stack(cols)                                           # [N, C]

        # diff = g - g_prev, then update g_prev := g (store back).
        g_prev = self._dequant_stacked(states, "g_prev", md, (R, C))      # [N, R, C]
        diff = grad - g_prev
        self._store_stacked(states, "g_prev", md, grad)

        # m and diff_ema EMAs (read both, EMA, store back).
        m = self._dequant_stacked(states, "m", md, (R, C))
        diff_ema = self._dequant_stacked(states, "diff", md, (R, C))
        m.mul_(c["beta1"]).add_(grad, alpha=1.0 - c["beta1"])
        diff_ema.mul_(c["beta2"]).add_(diff, alpha=1.0 - c["beta2"])
        self._store_stacked(states, "m", md, m)
        self._store_stacked(states, "diff", md, diff_ema)

        # Look-ahead u = g + beta2*diff -> squared into the factored second moment.
        u = diff.mul_(c["beta2"]).add_(grad)                              # reuse diff buffer
        omb3 = 1.0 - c["beta3"]
        u_sq = u * u
        row.lerp_(u_sq.mean(dim=-1), omb3)
        col.lerp_(u_sq.mean(dim=-2), omb3)
        torch._foreach_copy_(rows, list(row.unbind(0)))
        torch._foreach_copy_(cols, list(col.unbind(0)))

        r_factor = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)  # [N,R,1]
        c_factor = col.rsqrt().unsqueeze(-2)                                       # [N,1,C]
        # inv_denom approximates 1/(sqrt(n)/sqrt(bc3) + eps): fold bc3 into the
        # reconstruction and add eps on the denominator (1/(d+eps)) to match the
        # official's denominator-eps placement on the factored path.
        denom = (r_factor * c_factor).reciprocal_().mul_(1.0 / c["bc3_sqrt"]).add_(eps)

        # Combined numerator: m*step_size + diff_ema*step_size_diff, then / denom.
        update = m.mul_(c["step_size"]).add_(diff_ema, alpha=c["step_size_diff"])
        delta = update.div_(denom)

        if no_prox:
            if wd != 0:
                self._scale_weights_batched(plist, mat, 1.0 - group["lr"] * wd)
            if cautious:
                delta = cautious_batched_(delta, grad)
            subtract_batched_([mat(p.data) for p in plist], delta, bf16_method)
        else:
            if cautious:
                delta = cautious_batched_(delta, grad)
            subtract_batched_([mat(p.data) for p in plist], delta, bf16_method)
            if wd != 0:
                self._scale_weights_batched(plist, mat, 1.0 / (1.0 + group["lr"] * wd))

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
        no_prox = group["no_prox"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]

        states = [self.state[p] for p in plist]
        ns = [s["n"] for s in states]

        grad = torch.stack([p.grad.float() for p in plist])               # [N, L]
        n = torch.stack(ns)                                               # [N, L]

        g_prev = self._dequant_stacked(states, "g_prev", md, (length,))
        diff = grad - g_prev
        self._store_stacked(states, "g_prev", md, grad)

        m = self._dequant_stacked(states, "m", md, (length,))
        diff_ema = self._dequant_stacked(states, "diff", md, (length,))
        m.mul_(c["beta1"]).add_(grad, alpha=1.0 - c["beta1"])
        diff_ema.mul_(c["beta2"]).add_(diff, alpha=1.0 - c["beta2"])
        self._store_stacked(states, "m", md, m)
        self._store_stacked(states, "diff", md, diff_ema)

        u = diff.mul_(c["beta2"]).add_(grad)
        n.mul_(c["beta3"]).addcmul_(u, u, value=1.0 - c["beta3"])
        torch._foreach_copy_(ns, list(n.unbind(0)))

        denom = n.sqrt().div_(c["bc3_sqrt"]).add_(eps)
        update = m.mul_(c["step_size"]).add_(diff_ema, alpha=c["step_size_diff"])
        delta = update.div_(denom)

        if no_prox:
            if wd != 0:
                torch._foreach_mul_([p.data for p in plist], 1.0 - group["lr"] * wd)
            if cautious:
                delta = cautious_batched_(delta, grad)
            subtract_batched_([p.data for p in plist], delta, bf16_method)
        else:
            if cautious:
                delta = cautious_batched_(delta, grad)
            subtract_batched_([p.data for p in plist], delta, bf16_method)
            if wd != 0:
                torch._foreach_mul_([p.data for p in plist], 1.0 / (1.0 + group["lr"] * wd))

    @torch.no_grad()
    def _scale_weights_batched(self, plist: list[Tensor], mat: Any, factor: float) -> None:
        torch._foreach_mul_([mat(p.data) for p in plist], factor)

    # ---------------------------------------------------------- per-parameter
    @torch.no_grad()
    def _step_one_param(self, p: Tensor, group: dict[str, Any]) -> None:
        c = self._coeffs(group)
        md = group["momentum_dtype"]
        eps = group["eps"]
        wd = group["weight_decay"]
        no_prox = group["no_prox"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]

        state = self.state[p]
        if not state:
            self._init_state(p, state, group)

        grad = p.grad if p.grad.dtype == torch.float32 else p.grad.float()
        ndim = grad.ndim
        factored = ndim >= 2
        matrixize = ndim > 2
        gv = grad.reshape(grad.shape[0], -1) if matrixize else grad

        # diff = g - g_prev, then g_prev := g.
        g_prev = self._dequant_one(state, "g_prev", md, gv)
        diff = gv - g_prev
        self._store_one(state, "g_prev", md, gv)

        # m and diff_ema EMAs.
        m = self._dequant_one(state, "m", md, gv)
        diff_ema = self._dequant_one(state, "diff", md, gv)
        m.mul_(c["beta1"]).add_(gv, alpha=1.0 - c["beta1"])
        diff_ema.mul_(c["beta2"]).add_(diff, alpha=1.0 - c["beta2"])
        self._store_one(state, "m", md, m)
        self._store_one(state, "diff", md, diff_ema)

        # Look-ahead u = g + beta2*diff.
        u = diff.mul_(c["beta2"]).add_(gv)

        if factored:
            update_factored_state(u, state["row"], state["col"], c["beta3"], 0.0)
            r_factor, c_factor = factored_inv_sqrt_factors(state["row"], state["col"])
            denom = (r_factor * c_factor).reciprocal_().mul_(1.0 / c["bc3_sqrt"]).add_(eps)
        else:
            n = state["n"]
            n.mul_(c["beta3"]).addcmul_(u, u, value=1.0 - c["beta3"])
            denom = n.sqrt().div_(c["bc3_sqrt"]).add_(eps)

        update = m.mul_(c["step_size"]).add_(diff_ema, alpha=c["step_size_diff"])
        delta = update.div_(denom)
        if matrixize:
            delta = delta.reshape_as(grad)

        if no_prox:
            if wd != 0:
                p.data.mul_(1.0 - group["lr"] * wd)
            if cautious:
                delta = cautious_one_(delta, grad)
            subtract_one_(p, delta, state, bf16_method)
        else:
            if cautious:
                delta = cautious_one_(delta, grad)
            subtract_one_(p, delta, state, bf16_method)
            if wd != 0:
                p.data.mul_(1.0 / (1.0 + group["lr"] * wd))
