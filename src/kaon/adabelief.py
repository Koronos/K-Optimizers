"""AdaBelief — Adapting Stepsizes by the Belief in Observed Gradients — on kaon's backend.

AdaBelief (Zhuang et al. 2020, *AdaBelief Optimizer: Adapting Stepsizes by the
Belief in Observed Gradients*, NeurIPS 2020, arXiv:2010.07468) implemented on top
of the precision and memory machinery already proven in
:class:`~kaon.adakaon.Adakaon` and :class:`~kaon.adapnm.AdaPNM`. It is the
**"belief" optimizer**: structurally Adam, but the second moment tracks the
variance of the gradient *around its own EMA prediction* — the running squared
*residual* ``(g - m)**2`` — instead of the raw ``g**2``. When the gradient agrees
with the momentum (a "trusted" / confident direction) the residual is small, the
denominator is small, and the step is large; when the gradient surprises the
prediction the denominator grows and the step shrinks. That single swap is what
gives AdaBelief Adam-like fast convergence with SGD-like generalization on the
authors' benchmarks.

**The exact update (matches kozistr ``pytorch_optimizer`` ``AdaBelief``,
``rectify=False`` / ``ams_bound=False`` v1 here):**

.. code-block:: text

    # per param, step t (1-indexed):
    m  = beta1 * m + (1 - beta1) * g                       # plain Adam first moment
    s  = beta2 * s + (1 - beta2) * (g - m)**2 + eps        # the "belief" / residual variance

    bc1    = 1 - beta1 ** t
    bc2_sq = sqrt(1 - beta2 ** t)
    de_nom = (sqrt(s) + eps) / bc2_sq                      # s already has +eps folded in
    p     -= (lr / bc1) * m / de_nom

    # decoupled (AdamW-style) weight decay, applied BEFORE the moment updates:
    p     *= (1 - lr * weight_decay)

Two eps placements are carried over verbatim from kozistr: ``eps`` is added **both**
inside the second-moment EMA (``+ eps`` after the ``(g-m)**2`` term) **and** to
``sqrt(s)`` in the denominator. The first-moment decay and the bias correction both
use ``beta1`` (no ``beta1**2`` quirk, unlike AdaPNM). Default ``eps = 1e-16`` and
``betas = (0.9, 0.999)`` per the paper.

**The kaon twist — factored "belief".** The residual second moment ``s`` reuses
Adakaon's backend: ``ndim >= 2`` weights factor ``s`` into row+column EMAs (conv
kernels matrixized to ``[out, in*kh*kw]`` first), feeding the **residual**
``(g - m_dequant)`` to :func:`~kaon._factored.update_factored_state` (which squares
its input internally — so we pass the residual, *not* its square). ``ndim == 1``
keeps a full per-coordinate ``s``. The denominator is Adakaon's
``r_factor * c_factor`` reconstruction of ``1/sqrt(s_hat)``; the bias correction
``bc2`` is folded in by scaling that reconstructed inverse-denominator.

**eps placement on the factored path.** The factored backend (matching HF
Adafactor) adds ``eps1`` to ``(g-m)**2`` *before* the row/col mean reductions — the
direct analogue of AdaBelief's "``+ eps`` inside the EMA". The *second* AdaBelief
eps (added to ``sqrt(s)`` in the denominator) has **no factored analogue** (there is
no materialized ``sqrt(s)`` to add to — the reconstruction is a product of two
rsqrt factors), so on the factored 2-D path it is silently dropped, exactly as
AdaPNM drops kozistr's scalar denominator ``eps`` there. The non-factored 1-D path
applies **both** eps placements and matches kozistr's AdaBelief 1-D update exactly.
This is the single deliberate factored-backend deviation; it is documented under
``eps`` and is the reason the numpy parity test targets the 1-D fp32 path.

**Cautious masking.** Cautious (Liang et al. 2024) zeroes the update coordinates
whose sign disagrees with the *current* gradient (``delta * g <= 0``), rescaling
survivors to preserve the mean magnitude — the same semantics as the rest of kaon,
applied to the **final** ``delta`` (the ``m / de_nom`` step) against the raw
gradient. On by default.

**What is reused vs new.** Reused from Adakaon's backend: the factored second-moment
helpers (:mod:`kaon._factored`), the momentum storage layout and quant/dequant
primitives in :mod:`kaon._momentum_codec` (int8 per-row absmax; 4-bit per-block
absmax, nibble-packed), the stochastic-rounding bf16 weight update
(:func:`kaon._stochastic_rounding.add_stochastic_` via
:func:`~kaon._backend.subtract_one_`/:func:`~kaon._backend.subtract_batched_`),
cautious masking and Gradient Centralization (:mod:`kaon._backend`),
``load_state_dict_preserving_dtypes`` for dtype-safe resume, and the bucketed
foreach batching pattern. New here: feeding the **residual** ``(g - m)`` (not ``g``)
to the factored second moment, which forces the momentum EMA to run *before* the
second-moment update and to be read back from the codec via the *read-only*
``_dequant`` primitives (the codec's ``ema_*`` helpers fold the EMA into the update
and cannot supply the pre-update ``m`` the residual needs; AdaBelief runs the raw
``m`` EMA itself, mirroring AdaPNM's read-it-yourself pattern).

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

__all__ = ["AdaBelief"]

MomentumDtype = Literal["bfloat16", "float32", "int8", "4bit"]

# Performance / memory knobs mirror Adakaon (see that module for the rationale).
# One momentum + factored s: same working set as Adakaon.
_STACK_BYTES_PER_ELEM = 48


class AdaBelief(AutoLRMixin, Optimizer):
    """AdaBelief (belief-in-observed-gradients) on Adakaon's memory backend.

    Args:
        params: parameters or param-group dicts.
        lr: learning rate.
        betas: ``(beta1, beta2)``. ``beta1`` is the first-moment EMA decay,
            ``beta2`` the (factored) residual second-moment decay. Both the decay
            and the bias correction use ``beta1`` directly (no ``beta1**2``).
            Default ``(0.9, 0.999)`` per the paper.
        eps: stability constant. Added in **two** places (matching kozistr's
            AdaBelief): inside the second-moment EMA (``+ eps`` after the
            ``(g-m)**2`` term) and to ``sqrt(s)`` in the denominator. On the
            factored 2-D path only the first placement applies (folded into the
            Adafactor ``eps1``); the denominator ``eps`` has no factored analogue
            and is dropped there. Default ``1e-16`` per the paper.
        weight_decay: decoupled (AdamW-style) weight decay, applied
            multiplicatively ``p *= (1 - lr*weight_decay)`` *before* the moment
            updates (kozistr ``weight_decouple=True``; not folded into the cautious
            delta — cautious does not gate weight decay).
        cautious: cautious masking (Liang et al. 2024) on the final step vs the
            gradient. **On by default.**
        gradient_centralization: subtract each ``ndim>=2`` gradient's fan-in mean
            before the step (Yong et al. 2020). **On by default.** 1-D params
            untouched.
        momentum_dtype: storage for the first moment ``m`` — ``"bfloat16"``
            (default, ~2 B/param), ``"float32"`` (4 B/param), ``"int8"`` (~1 B/param,
            per-row absmax), or ``"4bit"`` (~0.5 B/param, per-block absmax,
            nibble-packed). Same layout as Adakaon, so checkpoints resume bit-exactly.
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
        eps: float = 1e-16,
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

        # Composable parameter-free LR (update-space DoWG) via AutoLRMixin. off -> zero overhead.
        self._init_autolr(auto_lr, auto_lr_freeze, auto_lr_scale, auto_lr_fuse_rel)

    # ------------------------------------------------------------------- state
    @torch.no_grad()
    def _alloc_momentum(
        self, prefix: str, grad: Tensor, state: dict[str, Any], group: dict[str, Any]
    ) -> None:
        """Allocate the momentum buffer (keys ``f"{prefix}"``, ``f"{prefix}_scale"`` …).

        Storage layout matches :mod:`kaon._momentum_codec` exactly (per-row int8
        scale; per-block 4-bit scale, zero == nibble 8) so the momentum resumes
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
            state["s"] = torch.zeros_like(grad, dtype=torch.float32)
        self._alloc_momentum("m", grad, state, group)
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
            # .float() is a no-op view for an fp32 buffer, so clone to avoid mutating
            # the stored momentum when the returned tensor is updated in place.
            m = state["m"]
            return (m.float() if m.dtype != torch.float32 else m.clone()).reshape_as(like)
        if md == "int8":
            return state["m"].float().mul_(state["m_scale"]).reshape_as(like)
        m = _dequant_4bit(state["m"], state["m_scale"], state["m_numel"], state["m_block"])
        return m.view_as(like)

    @staticmethod
    def _store_one(state: dict[str, Any], md: str, m_fp32: Tensor) -> None:
        """Write an updated fp32 momentum back into the configured storage layout.

        ``m_fp32`` may be the matrixized ``[R, C]`` view; the int8 per-row scale and
        the float buffer both reduce/store over the param's *original* shape, so
        reshape back first (dim-0 — the int8 row axis — is preserved).
        """
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
        """Stacked fp32 momentum ``[N, *shape]`` from per-param storage (see AdaPNM)."""
        n = len(states)
        per = math.prod(shape)
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
        per = math.prod(shape)
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
    def _step_impl(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            for p in params:
                if p.grad.is_sparse:
                    raise RuntimeError("AdaBelief does not support sparse gradients")
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

    def state_dict(self) -> dict[str, Any]:
        """Base state + the auto_lr tuner blob (via AutoLRMixin) when auto_lr is on."""
        return self._autolr_state_dict(super().state_dict())

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state, preserving the quantized momentum's stored dtype.

        torch's default ``load_state_dict`` upcasts every state tensor to the
        param's dtype (fp32), which would silently inflate bf16/int8/4bit momentum
        back to fp32 on resume. Delegate to the shared helper that restores each
        tensor to how it was checkpointed.
        """
        self._autolr_load(state_dict, lambda sd: load_state_dict_preserving_dtypes(self, sd))

    # ----------------------------------------------------------- coefficients
    @staticmethod
    def _coeffs(group: dict[str, Any]) -> dict[str, float]:
        """All per-step scalar coefficients (shared by the per-param and foreach paths)."""
        beta1, beta2 = group["betas"]
        step = group["step"]
        bc1 = 1.0 - beta1 ** step
        bc2_sq = math.sqrt(1.0 - beta2 ** step)
        return {
            "beta1": beta1,
            "beta2": beta2,
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

        # Decoupled weight decay BEFORE moment updates (kozistr order): p *= (1 - lr*wd).
        if wd != 0:
            self._apply_decoupled_wd_batched(plist, mat, group["lr"] * wd)

        # First-moment EMA (read both, mutate, store). m must update BEFORE the
        # residual second moment so the residual (g - m) uses the *new* m.
        m = self._dequant_stacked(states, md, (R, C))                     # [N, R, C]
        m.mul_(c["beta1"]).add_(grad, alpha=1.0 - c["beta1"])
        self._store_stacked(states, md, m.reshape((len(plist), R, C)))

        # Factored "belief" second moment of the residual (g - m). HF eps1 placement:
        # update_factored_state squares its input and adds eps1 to the square.
        residual = grad - m                                               # [N, R, C]
        omb2 = 1.0 - c["beta2"]
        res_sq = residual * residual
        if eps1 > 0:
            res_sq = res_sq.add_(eps1)
        row.lerp_(res_sq.mean(dim=-1), omb2)
        col.lerp_(res_sq.mean(dim=-2), omb2)
        torch._foreach_copy_(rows, list(row.unbind(0)))
        torch._foreach_copy_(cols, list(col.unbind(0)))

        r_factor = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)  # [N, R, 1]
        c_factor = col.rsqrt().unsqueeze(-2)                                       # [N, 1, C]
        inv_denom = (r_factor * c_factor).mul_(c["bc2_sq"])                        # 1/sqrt(s_hat)

        delta = m.mul_(inv_denom).mul_(c["step_size"])                    # m / de_nom * step_size

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
        eps = group["eps"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]

        states = [self.state[p] for p in plist]
        ss = [s["s"] for s in states]

        grad = torch.stack([p.grad.float() for p in plist])               # [N, L]
        s = torch.stack(ss)                                               # [N, L]

        if wd != 0:
            self._apply_decoupled_wd_batched(plist, lambda t: t, group["lr"] * wd)

        # First-moment EMA before the residual second moment.
        m = self._dequant_stacked(states, md, (length,))                  # [N, L]
        m.mul_(c["beta1"]).add_(grad, alpha=1.0 - c["beta1"])
        self._store_stacked(states, md, m.reshape((len(plist), length)))

        # Full per-coordinate "belief" second moment of (g - m), kozistr 1-D:
        # s = beta2*s + (1-beta2)*(g-m)^2 + eps; de_nom = (sqrt(s) + eps) / bc2_sq.
        residual = grad - m
        s.mul_(c["beta2"]).addcmul_(residual, residual, value=1.0 - c["beta2"]).add_(eps)
        torch._foreach_copy_(ss, list(s.unbind(0)))

        de_nom = s.sqrt().add_(eps).div_(c["bc2_sq"])
        delta = m.div_(de_nom).mul_(c["step_size"])

        if cautious:
            delta = cautious_batched_(delta, grad)

        subtract_batched_([p.data for p in plist], delta, bf16_method)

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
        eps = group["eps"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]

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

            # First-moment EMA before the residual second moment.
            m = self._dequant_one(state, md, gv)
            m.mul_(c["beta1"]).add_(gv, alpha=1.0 - c["beta1"])
            self._store_one(state, md, m)

            # Factored "belief" second moment of the residual (g - m). pass the
            # residual (NOT its square): update_factored_state squares internally.
            residual = gv - m
            update_factored_state(residual, state["row"], state["col"], c["beta2"], eps)
            r_factor, c_factor = factored_inv_sqrt_factors(state["row"], state["col"])
            inv_denom = (r_factor * c_factor).mul_(c["bc2_sq"])           # 1/sqrt(s_hat)
            delta = m.mul_(inv_denom).mul_(c["step_size"])
            if matrixize:
                delta = delta.reshape_as(grad)
        else:
            # First-moment EMA before the residual second moment.
            m = self._dequant_one(state, md, grad)
            m.mul_(c["beta1"]).add_(grad, alpha=1.0 - c["beta1"])
            self._store_one(state, md, m)

            s = state["s"]
            residual = grad - m
            s.mul_(c["beta2"]).addcmul_(residual, residual, value=1.0 - c["beta2"]).add_(eps)
            de_nom = s.sqrt().add_(eps).div_(c["bc2_sq"])
            delta = m.div_(de_nom).mul_(c["step_size"])

        if cautious:
            delta = cautious_one_(delta, grad)

        subtract_one_(p, delta, state, bf16_method)
