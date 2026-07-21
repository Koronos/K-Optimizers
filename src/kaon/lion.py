"""Lion — sign-momentum update on Adakaon's precision/memory backend.

Lion (Chen et al. 2023, *Symbolic Discovery of Optimization Algorithms*,
arXiv:2302.06675) — a sign-of-momentum optimizer that keeps a **single** momentum
buffer and **no second moment** — implemented on top of the precision and memory
machinery already proven in :class:`~kaon.adakaon.Adakaon`. It is a
deliberate generalization / ablation vehicle: same quantized-momentum store, same
bf16-correct stochastic rounding, same cautious masking and foreach batching as
Adakaon, but with Lion's update rule instead of the factored-second-moment
Adam-style step. Keeping it a separate class lets it be A/B'd against Adakaon
cleanly (Adakaon is left byte-for-byte unchanged).

(Developed under the provisional code name *Liofusion*.)

**The update (per parameter, decoupled weight decay):**

.. code-block:: text

    c       = sign(beta1 * m + (1 - beta1) * g)   # interpolated-momentum direction
    update  = c                                   # +1 / 0 / -1 per coordinate
    p      -= lr * (update + weight_decay * p)    # decoupled WD, folded into delta
    m       = beta2 * m + (1 - beta2) * g         # momentum EMA, updated AFTER c

The direction uses the *old* momentum interpolated with the current gradient at
``beta1``; the stored momentum is then advanced with the (usually larger)
``beta2``. This is exactly Lion. Note the EMA is on the **raw gradient**, unlike
Adakaon (which takes momentum of the already-normalized update).

**Why this is cheap (the headline):**

* **One** state buffer (the momentum), **no** second moment — half the live
  optimizer state of Adam/Adakaon-with-momentum before quantization.
* That single buffer is stored through the **shared momentum codec layout**
  (``bfloat16`` ~2 B/param, ``int8`` ~1 B/param, ``4bit`` ~0.5 B/param), so
  Lion's optimizer-state floor is Lion-class or better.
* The step itself is a ``sign`` plus two cheap EMAs — no ``rsqrt``, no factored
  reconstruction, no RMS clip.

**lr is Lion-scale.** Lion's sign update has unit magnitude per coordinate, so a
good ``lr`` is typically **~3-10x smaller** than the AdamW/Adakaon lr for the
same model. Weight decay is decoupled (AdamW-style) and Lion usually wants it a
bit larger than Adam to compensate for the unit-magnitude steps.

**Cautious masking** (Liang et al. 2024) is supported and on by default. The
update is already ``sign(c)``; cautious zeroes the coordinates where that sign
disagrees with the current gradient sign (``update * g <= 0``) and rescales the
survivors to preserve the mean step magnitude — the same semantics Adakaon
uses. With pure sign updates this is a per-coordinate agreement filter between
the momentum-interpolated direction and the instantaneous gradient.

**What is reused vs new.** Reused from Adakaon's backend: the momentum storage
layout and the quant/dequant primitives in :mod:`kaon._momentum_codec`
(int8 per-row absmax; 4-bit per-block absmax, nibble-packed), the
stochastic-rounding bf16 weight update (:func:`kaon._stochastic_rounding.add_stochastic_`),
``load_state_dict_preserving_dtypes`` for dtype-safe checkpoint resume, and the
bucketed foreach batching pattern for many small tensors. New here: the Lion
sign-momentum update rule and its dual-beta momentum handling (the shared codec's
``ema_*`` helpers do an Adam-style *momentum-of-update* and so cannot be reused
verbatim; Lion dequants the momentum, computes the Lion direction and the
``beta2`` EMA itself, then requants through the same storage layout).

It is a standard ``torch.optim.Optimizer`` with a single per-parameter step, so
it drops into per-parameter / gradient-release training loops unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, Literal

import torch
from torch import Tensor
from torch.optim import Optimizer

from kaon._autolr import AutoLRMixin
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

__all__ = ["Lion"]

MomentumDtype = Literal["bfloat16", "float32", "int8", "4bit"]

# Performance cutoff (mirrors Adakaon): weights larger than this loop instead
# of being stacked — batching only pays off while per-tensor kernel-launch
# overhead dominates (small tensors); a large weight's sign step is
# bandwidth-bound, so stacking just adds copy traffic.
_STACK_BYTES_PER_ELEM = 48


class Lion(AutoLRMixin, Optimizer):
    """Lion sign-momentum optimizer on Adakaon's quantized-momentum backend.

    Args:
        params: parameters or param-group dicts.
        lr: learning rate. **Lion-scale** — typically ~3-10x smaller than the
            AdamW/Adakaon lr for the same model (the sign update has unit
            magnitude per coordinate).
        betas: ``(beta1, beta2)``. ``beta1`` interpolates the *direction*
            (``sign(beta1*m + (1-beta1)*g)``); ``beta2`` is the momentum EMA decay
            (updated after the direction). Lion's defaults ``(0.9, 0.99)``.
        weight_decay: decoupled (AdamW-style) weight decay, folded into the
            per-step delta. Lion usually wants this a touch larger than Adam.
        momentum_dtype: storage for the single momentum buffer — ``"bfloat16"``
            (default, ~2 B/param), ``"float32"`` (4 B/param), ``"int8"`` (~1
            B/param, per-row absmax), or ``"4bit"`` (~0.5 B/param, per-block
            absmax, nibble-packed). Same storage layout as Adakaon's first
            moment, so checkpoints resume bit-exactly via ``load_state_dict``.
        momentum_4bit_block: block size for ``momentum_dtype="4bit"`` (consecutive
            flattened elements sharing one absmax scale). Default ``128``.
            ``0``/negative means whole-tensor (a single scale).
        cautious: cautious masking (Liang et al. 2024) — zero the update
            coordinates whose sign disagrees with the gradient, then rescale the
            survivors to preserve the mean step magnitude. **On by default.** For
            Lion's pure-sign update this filters coordinates where the
            momentum-interpolated direction disagrees with the instantaneous
            gradient.
        bf16_method: weight-update strategy for low-precision params —
            ``"stochastic_rounding"`` (default), ``"kahan"`` (+2 B/param), or
            ``"none"``. No-op on fp32 params.
        foreach: batch the step across parameters with stacked multi-tensor ops
            instead of a per-parameter Python loop. Default ``True`` (the win for
            LoRA/LoKr adapters and the many 1-D biases/norms of a full fine-tune).
            Numerically matches the per-parameter path (stochastic-rounding draws
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
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        *,
        momentum_dtype: MomentumDtype = "bfloat16",
        momentum_4bit_block: int = _FOURBIT_BLOCK,
        cautious: bool = True,
        gradient_centralization: bool = True,
        bf16_method: str = "stochastic_rounding",
        foreach: bool = True,
        foreach_batch_cutoff: int = FOREACH_BATCH_CUTOFF,
        foreach_stack_budget: int | None = None,
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
        defaults = {
            "lr": lr,
            "betas": (beta1, beta2),
            "weight_decay": weight_decay,
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
        # Composable parameter-free LR (update-space DoWG) via AutoLRMixin. When on, drives
        # the step via _step_impl at the discovered lr=S; off (default) -> step == _step_impl.
        self._init_autolr(auto_lr, auto_lr_freeze, auto_lr_scale, auto_lr_fuse_rel)

    # ------------------------------------------------------------------- state
    @torch.no_grad()
    def _init_state(self, p: Tensor, state: dict[str, Any], group: dict[str, Any]) -> None:
        """Allocate the single momentum buffer in the configured storage layout.

        Layout matches :mod:`kaon._momentum_codec` exactly (per-row int8 scale,
        per-block 4-bit scale, zero == nibble 8) so checkpoint resume and
        ``load_state_dict_preserving_dtypes`` behave identically to Adakaon.
        """
        grad = p.grad
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
            bs = fourbit_block_size(grad, group)
            nblocks = (numel + bs - 1) // bs
            # zero momentum -> nibble 8 (zero level after the +8 shift); 0x88 byte.
            state["m"] = torch.full(
                ((numel + 1) // 2,), 0x88, dtype=torch.uint8, device=grad.device
            )
            state["m_scale"] = torch.ones(nblocks, dtype=torch.float32, device=grad.device)
            state["m_numel"] = numel
            state["m_block"] = bs
        if is_low_precision(p) and group["bf16_method"] == "kahan":
            state["shift"] = torch.zeros_like(p)

    # -------------------------------------------------------- momentum (codec)
    @staticmethod
    def _dequant_one(state: dict[str, Any], md: str, like: Tensor) -> Tensor:
        """Read the stored momentum back as a fresh fp32 tensor shaped like ``like``."""
        if md in ("bfloat16", "float32"):
            return state["m"].float()
        if md == "int8":
            return state["m"].float().mul_(state["m_scale"])
        m = _dequant_4bit(state["m"], state["m_scale"], state["m_numel"], state["m_block"])
        return m.view_as(like)

    @staticmethod
    def _store_one(state: dict[str, Any], md: str, m_fp32: Tensor) -> None:
        """Write an updated fp32 momentum back into the configured storage layout."""
        if md in ("bfloat16", "float32"):
            state["m"].copy_(m_fp32)
        elif md == "int8":
            state["m"], state["m_scale"] = _quant_int8(m_fp32)
        else:  # 4bit
            packed, scale, _ = _quant_4bit(m_fp32, state["m_block"])
            state["m"], state["m_scale"] = packed, scale

    @staticmethod
    def _dequant_stacked(states: list[dict[str, Any]], md: str, shape: tuple[int, ...]) -> Tensor:
        """Stacked fp32 momentum ``[N, *shape]`` from the per-param storage.

        Reductions are arranged to be element-for-element identical to the
        per-param path: int8 collapses to a per-row (dim-0 of the param) scale
        — a stacked ``[N, R, rest]`` view reduces its trailing axis, which is the
        same value set the per-param ``_quant_int8`` reduces over dims ``>= 1``.
        4-bit uses the flat per-param block layout (block boundaries match).
        """
        n = len(states)
        per = math.prod(shape)
        if md in ("bfloat16", "float32"):
            return torch.stack([s["m"] for s in states]).float()
        if md == "int8":
            # Per-param int8 uses a per-row (dim-0) scale for ndim >= 2 and a
            # single scalar scale for ndim == 1 (see _quant_int8). Mirror both:
            # ndim >= 2 -> [N, R, rest] (R rows), ndim == 1 -> [N, 1, L] (1 row).
            row = shape[0] if len(shape) >= 2 else 1
            rest = max(per // row, 1)
            m = torch.stack([s["m"].reshape(row, rest) for s in states]).float()  # [N, R, rest]
            scale = torch.stack([s["m_scale"].reshape(row, 1) for s in states])    # [N, R, 1]
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
        per = math.prod(shape)
        if md in ("bfloat16", "float32"):
            ms = [s["m"] for s in states]
            torch._foreach_copy_(ms, list(m_fp32.unbind(0)))
        elif md == "int8":
            row = shape[0] if len(shape) >= 2 else 1
            rest = max(per // row, 1)
            q, new_scale = _quant_int8_stacked(m_fp32.reshape(n, row, rest))  # [N,R,rest]->[N,R,1]
            torch._foreach_copy_(
                [s["m"].reshape(row, rest) for s in states], list(q.unbind(0))
            )
            # Per-param scale is shaped (R, 1) (ndim>=2) or (1,) (ndim==1, scalar).
            for s, sc in zip(states, new_scale.unbind(0), strict=True):  # sc: [R, 1]
                s["m_scale"] = sc.reshape(row, 1) if len(shape) >= 2 else sc.reshape(1)
        else:  # 4bit
            bs = states[0]["m_block"]
            new_packed, new_scale = _quant_4bit_stacked(m_fp32.reshape(n, per), bs)
            torch._foreach_copy_([s["m"] for s in states], list(new_packed.unbind(0)))
            for s, sc in zip(states, new_scale.unbind(0), strict=True):
                s["m_scale"].copy_(sc)

    # -------------------------------------------------------------------- step
    # step() is the AutoLRMixin router (drives the DoWG tuner when auto_lr is on, else
    # _step_impl; re-imposes the frozen LR each step vs a harness clobber).
    @torch.no_grad()
    def _step_impl(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            for p in params:
                if p.grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")
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
                    for p in params:
                        self._step_one_param(p, group)
            else:
                for p in params:
                    self._step_one_param(p, group)
        return loss

    def state_dict(self) -> dict[str, Any]:
        """Base state + the auto_lr tuner blob (via AutoLRMixin) when auto_lr is on."""
        return self._autolr_state_dict(super().state_dict())

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state, preserving the quantized momentum's stored dtype (torch's default
        would upcast bf16/int8/4bit momentum to fp32 on resume). The auto_lr tuner blob is
        peeled off first by AutoLRMixin."""
        self._autolr_load(state_dict, lambda sd: load_state_dict_preserving_dtypes(self, sd))

    # ----------------------------------------------------------------- foreach

    @staticmethod
    def _group_foreach_eligible(group: dict[str, Any]) -> bool:
        return group["bf16_method"] != "kahan"  # kahan needs a per-param shift buffer

    @staticmethod
    def _param_foreach_eligible(p: Tensor, group: dict[str, Any], cutoff: int) -> bool:
        if p.ndim == 0 or p.numel() > cutoff:
            return False
        # fp16+SR is unsupported (raises) -> route to the per-param path.
        return not (
            group["bf16_method"] == "stochastic_rounding"
            and is_low_precision(p)
            and p.dtype != torch.bfloat16
        )

    @torch.no_grad()
    def _step_foreach(self, params: list[Tensor], group: dict[str, Any], budget: int) -> None:
        """Batched Lion step.

        Lion's update is fully per-coordinate (no factoring), so params are
        bucketed by exact shape (and dtype) and each bucket stacks to ``[N, L]``
        (flattened) for the sign/EMA/cautious/WD math — element-for-element the
        same as :meth:`_step_one_param`. The momentum helpers recover the param's
        row structure internally so the int8 per-row scale matches the per-param
        path exactly.
        """
        beta1, beta2 = group["betas"]
        lr = group["lr"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]
        md = group["momentum_dtype"]

        buckets: dict[tuple[Any, ...], list[Tensor]] = {}
        for p in params:
            state = self.state[p]
            if not state:
                self._init_state(p, state, group)
            buckets.setdefault((tuple(p.shape), p.dtype), []).append(p)

        for (shape, _dtype), plist in buckets.items():
            length = 1
            for d in shape:
                length *= d
            chunk = max(1, budget // max(length, 1))
            for i in range(0, len(plist), chunk):
                self._bucket(
                    plist[i:i + chunk], shape, length, md,
                    beta1, beta2, lr, wd, cautious, bf16_method,
                )

    @torch.no_grad()
    def _bucket(
        self,
        plist: list[Tensor],
        shape: tuple[int, ...],
        length: int,
        md: str,
        beta1: float,
        beta2: float,
        lr: float,
        wd: float,
        cautious: bool,
        bf16_method: str,
    ) -> None:
        n = len(plist)
        states = [self.state[p] for p in plist]
        grad = torch.stack([p.grad.reshape(length).float() for p in plist])  # [N, L]
        m = self._dequant_stacked(states, md, shape).reshape(n, length)      # [N, L] fp32

        # Lion direction: sign of the beta1-interpolated momentum.
        c = m.mul(beta1).add_(grad, alpha=1.0 - beta1)
        delta = torch.sign(c)                                                # +1/0/-1

        # Momentum EMA with beta2 (on the raw gradient), then requant + store.
        m.mul_(beta2).add_(grad, alpha=1.0 - beta2)
        self._store_stacked(states, md, m.reshape((n, *shape)))

        if wd != 0:
            p_fp32 = torch.stack([p.data.reshape(length).float() for p in plist])
            delta = delta.add_(p_fp32, alpha=wd)

        if cautious:
            delta = cautious_batched_(delta, grad)

        delta.mul_(lr)
        subtract_batched_([p.data.reshape(length) for p in plist], delta, bf16_method)

    # ---------------------------------------------------------- per-parameter
    @torch.no_grad()
    def _step_one_param(self, p: Tensor, group: dict[str, Any]) -> None:
        beta1, beta2 = group["betas"]
        lr = group["lr"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]
        md = group["momentum_dtype"]

        state = self.state[p]
        if not state:
            self._init_state(p, state, group)

        grad = p.grad if p.grad.dtype == torch.float32 else p.grad.float()
        m = self._dequant_one(state, md, grad)                  # fp32, grad-shaped

        # Lion direction: sign(beta1 * m + (1 - beta1) * g).
        c = m.mul(beta1).add_(grad, alpha=1.0 - beta1)
        delta = torch.sign(c)

        # Momentum EMA with beta2, then requant + store.
        m.mul_(beta2).add_(grad, alpha=1.0 - beta2)
        self._store_one(state, md, m)

        if wd != 0:
            p_fp32 = p.data if p.dtype == torch.float32 else p.data.float()
            delta = delta.add_(p_fp32, alpha=wd)

        if cautious:
            delta = cautious_one_(delta, grad)

        delta.mul_(lr)
        subtract_one_(p, delta, state, bf16_method)

