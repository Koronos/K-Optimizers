"""AdamP — AdamW with a per-channel radial (scale-invariant) projection, on kaon's
memory backend.

AdamP (Heo et al. 2021, *Slowing Down the Weight Norm Increase in Momentum-based
Optimizers*, ICLR 2021, arXiv:2006.08217) is plain AdamW with **one** extra step:
before applying the Adam update direction it removes the component of that update
that is **parallel to the weight** (the *radial* direction) — but only for weights
that are effectively **scale-invariant** (i.e. immediately followed by a
normalization, so their magnitude does not affect the function). Removing the
radial component slows the spurious weight-norm growth that momentum induces;
that growth otherwise shrinks the *effective* learning rate over training and
hurts generalization. AdamP gets the AdamW step quality back at the original
effective LR, with a measured generalization win.

**Scale-invariance is detected cheaply** by the cosine similarity between the
*gradient* and the *weight*: for a scale-invariant weight the gradient is (almost)
orthogonal to the weight, so a *low* cosine is the proxy for "project this one".
The test is applied per output channel (``channel_view``: the weight viewed as
``[out, -1]``) and, failing that, per whole layer (``layer_view``: ``[1, -1]``),
exactly as the official ``clovaai/AdamP``.

**The exact update (matches the official ``clovaai/AdamP`` ``_projection`` +
``step``):**

.. code-block:: text

    m = beta1*m + (1-beta1)*g
    v = beta2*v + (1-beta2)*g^2
    bc1 = 1 - beta1^t ; bc2 = 1 - beta2^t
    denom   = sqrt(v) / sqrt(bc2) + eps          # official eps placement
    perturb = m / denom                          # Adam direction (m NOT debiased here;
                                                 #   bias-correction1 folds into step_size)
    # --- the AdamP projection (only for ndim >= 2 weights) ---
    wd_ratio = 1
    for view in (channel_view=[out,-1], layer_view=[1,-1]):
        cos = | cosine_similarity(view(g), view(p)) |        # per row of the view
        if cos.max() < delta / sqrt(view_dim):              # "scale-invariant?" proxy
            p_n = p / (||view(p)||_row + eps)               # unit weight direction, per row
            perturb -= p_n * <p_n, perturb>_row             # remove the radial component
            wd_ratio = wd_ratio_hp                          # and damp WD on this weight
            break                                           # channel view wins if it triggers
    # decoupled weight decay (scaled by wd_ratio when the projection fired):
    p *= (1 - lr*weight_decay*wd_ratio)
    p -= (lr/bc1) * perturb

**1-D params (biases, norm scales)** are never projected (``len(p.shape) > 1``
gate in the official) — they *are* the scale parameters. They take the plain Adam
step with full (decoupled) weight decay.

**The factored second moment.** ``v`` reuses Adakaon's backend: ``ndim >= 2``
weights factor ``v`` into row+column EMAs (conv kernels matrixized to
``[out, in*kh*kw]`` first); ``ndim == 1`` keeps a full per-coordinate ``v``. The
factored path adds ``eps`` Adafactor-style via ``eps1`` (folded into ``grad**2``
before the row/col reductions) — the official's scalar ``eps`` on the denominator
has no factored analogue. The **1-D path matches the official eps placement
exactly**: ``denom = sqrt(v)/sqrt(bc2) + eps``.

**The momentum** is the *raw* Adam first moment (not a pre-mixed delta), because
AdamP post-processes the update *direction* before applying it. It is stored
through the shared :mod:`kaon._momentum_codec` storage + read-only
dequant/requant primitives (the same pattern AdaPNM uses), so ``momentum_dtype``
can be ``bfloat16``/``float32``/``int8``/``4bit``.

**foreach parity with the projection.** The projection is a *per-tensor* branch
(channel-view fires, else layer-view fires, else nothing) with a *per-channel*
reduction. The foreach buckets are keyed by shape, so every slice in a bucket
shares the same view dimensions; the per-tensor branch is then a boolean mask over
the stack and the radial removal is a masked, broadcast subtract — element-for-
element identical to the per-param path (verified by the parity test, including
weights that DO trigger the projection).

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

__all__ = ["AdamP"]

MomentumDtype = Literal["bfloat16", "float32", "int8", "4bit"]

# One momentum + factored v; on par with Adakaon's single-momentum working set.
_STACK_BYTES_PER_ELEM = 48


class AdamP(Optimizer):
    """AdamP (AdamW + per-channel radial projection) on kaon's memory backend.

    Args:
        params: parameters or param-group dicts.
        lr: learning rate. Default ``1e-3``.
        betas: ``(beta1, beta2)`` — first- and (factored) second-moment EMA decays.
            Default ``(0.9, 0.999)`` (the official AdamP defaults).
        eps: term added to the second-moment denominator. On the non-factored (1-D)
            path it follows the **official** placement ``denom = sqrt(v)/sqrt(bc2) +
            eps``. On the factored path it is folded into the Adafactor ``eps1``
            (added to ``grad**2`` before the row/col reductions). Default ``1e-8``.
        weight_decay: decoupled (AdamW-style) weight decay, applied multiplicatively
            ``p *= (1 - lr*weight_decay*wd_ratio)`` *before* the parameter update.
            When the projection fires on a weight, ``wd_ratio`` is used (see below).
            Default ``0``.
        delta: cosine-similarity threshold for the scale-invariance proxy. The
            projection fires when ``max_row |cos(g, p)| < delta / sqrt(view_dim)``.
            Default ``0.1`` (official).
        wd_ratio: the factor weight decay is multiplied by **on weights where the
            projection fires** (the radial WD on a scale-invariant weight is mostly
            redundant, so it is damped). Default ``0.1`` (official).
        nesterov: use the Nesterov-style lookahead numerator
            ``(beta1*m + (1-beta1)*g) / denom`` instead of ``m / denom`` (official
            ``nesterov`` flag). Default ``False``.
        cautious: cautious masking (Liang et al. 2024) on the final (projected) step
            vs the gradient. **On by default** (consistency with the rest of kaon).
            Pin ``False`` to recover the literal official AdamP step.
        gradient_centralization: subtract the per-output-row gradient mean for
            ``ndim >= 2`` weights before the step (Yong et al. 2020). **On by
            default**; pin ``False`` for the literal official step.
        momentum_dtype: storage for the first moment — ``"bfloat16"`` (default,
            ~2 B/param), ``"float32"`` (4 B/param), ``"int8"`` (~1 B/param, per-row
            absmax), or ``"4bit"`` (~0.5 B/param, per-block absmax, nibble-packed).
        momentum_4bit_block: block size for ``momentum_dtype="4bit"``. Default
            ``128``. ``0``/negative means whole-tensor.
        bf16_method: weight-update strategy for low-precision params —
            ``"stochastic_rounding"`` (default), ``"kahan"`` (+2 B/param), or
            ``"none"``. No-op on fp32 params.
        foreach: batch the step across parameters with stacked multi-tensor ops.
            Default ``True``. Numerically matches the per-parameter path (including
            the per-channel projection). 0-D scalars, kahan, and fp16+SR fall back to
            the per-parameter path.
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
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        delta: float = 0.1,
        wd_ratio: float = 0.1,
        nesterov: bool = False,
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
        if delta < 0.0:
            raise ValueError(f"delta must be >= 0, got {delta}")
        if not 0.0 <= wd_ratio <= 1.0:
            raise ValueError(f"wd_ratio must be in [0, 1], got {wd_ratio}")
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
            "delta": float(delta),
            "wd_ratio": float(wd_ratio),
            "nesterov": nesterov,
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
    def _alloc_momentum(self, grad: Tensor, state: dict[str, Any], group: dict[str, Any]) -> None:
        """Allocate the first-moment buffer in the configured codec layout.

        Storage layout matches :mod:`kaon._momentum_codec` exactly (per-row int8
        scale; per-block 4-bit scale, zero == nibble 8) so checkpoints resume
        bit-exactly via ``load_state_dict_preserving_dtypes``.
        """
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
        self._alloc_momentum(grad, state, group)
        if is_low_precision(p) and group["bf16_method"] == "kahan":
            state["shift"] = torch.zeros_like(p)

    # -------------------------------------------------- momentum read / write
    @staticmethod
    def _dequant_one(state: dict[str, Any], md: str, like: Tensor) -> Tensor:
        """Read the stored first moment back as a fresh fp32 tensor shaped like ``like``.

        Buffers are stored in the param's original shape; conv kernels are
        matrixized at use-site, so reshape to ``like`` (the per-row int8 scale's row
        grouping is preserved because dim-0 is unchanged).

        NOTE (the kaon footgun): the float path uses ``.float()`` which on an fp32
        buffer returns the SAME tensor — so it is ``.clone()``'d to avoid corrupting
        stored state when the caller mutates the returned tensor in place.
        """
        if md in ("bfloat16", "float32"):
            m = state["m"].float()
            if state["m"].dtype == torch.float32:
                m = m.clone()
            return m.reshape_as(like)
        if md == "int8":
            return state["m"].float().mul_(state["m_scale"]).reshape_as(like)
        m = _dequant_4bit(state["m"], state["m_scale"], state["m_numel"], state["m_block"])
        return m.view_as(like)

    @staticmethod
    def _store_one(state: dict[str, Any], md: str, m_fp32: Tensor) -> None:
        """Write the updated fp32 first moment back into the configured storage layout."""
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
        """Stacked fp32 first moment ``[N, *shape]`` from per-param storage."""
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
        """Write stacked fp32 first moment ``[N, *shape]`` back into per-param storage."""
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

    # ------------------------------------------------------------- projection
    @staticmethod
    def _projection_channel(
        p: Tensor, grad: Tensor, perturb: Tensor, delta: float, eps: float
    ) -> tuple[Tensor, bool]:
        """Per-param **channel-view** projection (``p`` viewed as ``[out, -1]``).

        Returns ``(perturb, fired)``. ``perturb`` is modified in place when fired.
        Mirrors the official ``_projection`` first iteration exactly.
        """
        view = p.reshape(p.shape[0], -1)
        gview = grad.reshape(grad.shape[0], -1)
        cos = torch.nn.functional.cosine_similarity(gview, view, dim=1, eps=eps).abs_()
        if cos.max() < delta / math.sqrt(view.shape[1]):
            expand = [-1] + [1] * (perturb.ndim - 1)
            p_n = p / view.norm(dim=1).view(expand).add_(eps)
            pv = p_n.reshape(p_n.shape[0], -1)
            dot = (pv * perturb.reshape(perturb.shape[0], -1)).sum(dim=1).view(expand)
            perturb.sub_(p_n * dot)
            return perturb, True
        return perturb, False

    @staticmethod
    def _projection_layer(
        p: Tensor, grad: Tensor, perturb: Tensor, delta: float, eps: float
    ) -> tuple[Tensor, bool]:
        """Per-param **layer-view** projection (``p`` viewed as ``[1, -1]``).

        Returns ``(perturb, fired)``. Mirrors the official ``_projection`` second
        iteration exactly.
        """
        view = p.reshape(1, -1)
        gview = grad.reshape(1, -1)
        cos = torch.nn.functional.cosine_similarity(gview, view, dim=1, eps=eps).abs_()
        if cos.max() < delta / math.sqrt(view.shape[1]):
            p_n = p / view.norm(dim=1).add_(eps)  # scalar norm over the whole tensor
            dot = (p_n.reshape(1, -1) * perturb.reshape(1, -1)).sum()
            perturb.sub_(p_n * dot)
            return perturb, True
        return perturb, False

    def _project_one(
        self, p: Tensor, grad: Tensor, perturb: Tensor, group: dict[str, Any]
    ) -> tuple[Tensor, float]:
        """Official per-param projection: channel view first, else layer view, else none.

        ``grad`` and ``p`` are in the param's ORIGINAL shape (not matrixized) so the
        ``[out, -1]`` / ``[1, -1]`` views match the official. Returns
        ``(perturb, wd_ratio)``.
        """
        delta, eps = group["delta"], group["eps"]
        perturb, fired = self._projection_channel(p, grad, perturb, delta, eps)
        if fired:
            return perturb, group["wd_ratio"]
        perturb, fired = self._projection_layer(p, grad, perturb, delta, eps)
        if fired:
            return perturb, group["wd_ratio"]
        return perturb, 1.0

    @staticmethod
    def _project_stacked(
        p_stack: Tensor,
        g_stack: Tensor,
        perturb: Tensor,
        delta: float,
        eps: float,
        wd_ratio_hp: float,
    ) -> tuple[Tensor, Tensor]:
        """Batched projection over a same-shape bucket. ``[N, R, C]`` stacks.

        Every slice in a bucket shares the view dims (bucket keyed by shape), so the
        per-tensor branch (channel fires / else layer fires / else none) is a boolean
        mask over the stack. The radial removal is then a masked broadcast subtract,
        element-for-element identical to the per-param path.

        ``p_stack``/``g_stack``/``perturb`` are the matrixized ``[N, R, C]`` stacks
        (``R = out``, ``C = fan-in``). Returns ``(perturb, wd_ratio[N,1,1])``.
        """
        n, r, c = p_stack.shape

        # --- channel view: [N, R, C], cosine per (slice, row) over C ---
        cos_ch = torch.nn.functional.cosine_similarity(g_stack, p_stack, dim=-1, eps=eps).abs_()  # [N, R]
        ch_fire = cos_ch.amax(dim=1) < (delta / math.sqrt(c))  # [N] bool

        # --- layer view: [N, 1, R*C], cosine per slice over R*C ---
        gflat = g_stack.reshape(n, 1, r * c)
        pflat = p_stack.reshape(n, 1, r * c)
        cos_ly = torch.nn.functional.cosine_similarity(gflat, pflat, dim=-1, eps=eps).abs_()  # [N, 1]
        ly_fire = (cos_ly.amax(dim=1) < (delta / math.sqrt(r * c))) & (~ch_fire)  # [N] bool

        # Channel radial removal: p_n = p / (rownorm + eps); subtract p_n*<p_n,perturb>_row.
        ch_norm = p_stack.norm(dim=-1, keepdim=True).add_(eps)              # [N, R, 1]
        pn_ch = p_stack / ch_norm                                          # [N, R, C]
        radial_ch = pn_ch * (pn_ch * perturb).sum(dim=-1, keepdim=True)    # [N, R, C]

        # Layer radial removal: p_n over the whole [R*C] block.
        ly_norm = p_stack.reshape(n, 1, r * c).norm(dim=-1, keepdim=True).add_(eps)  # [N, 1, 1]
        pn_ly = (p_stack.reshape(n, 1, r * c) / ly_norm).reshape(n, r, c)  # [N, R, C]
        ly_dot = (pn_ly.reshape(n, r * c) * perturb.reshape(n, r * c)).sum(dim=-1)  # [N]
        radial_ly = pn_ly * ly_dot.view(n, 1, 1)                           # [N, R, C]

        ch_mask = ch_fire.view(n, 1, 1)
        ly_mask = ly_fire.view(n, 1, 1)
        zero = perturb.new_zeros(())
        perturb = perturb - torch.where(ch_mask, radial_ch, zero)
        perturb = perturb - torch.where(ly_mask, radial_ly, zero)

        fired = (ch_fire | ly_fire).view(n, 1, 1)
        wd_ratio = torch.where(
            fired, perturb.new_full((), wd_ratio_hp), perturb.new_ones(())
        )  # [N, 1, 1]
        return perturb, wd_ratio

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
                    raise RuntimeError("AdamP does not support sparse gradients")
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
        """Restore state, preserving the quantized momentum's stored dtype.

        torch's default ``load_state_dict`` upcasts every state tensor to the
        param's dtype (fp32), which would silently inflate bf16/int8/4bit momentum
        back to fp32 on resume. Delegate to the shared helper.
        """
        load_state_dict_preserving_dtypes(self, state_dict)

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
        """Batched step. Factored (ndim>=2, projected) and non-factored (ndim==1) buckets."""
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
        nesterov = group["nesterov"]

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

        # First-moment EMA (raw Adam momentum), read, EMA, store back.
        m = self._dequant_stacked(states, md, (R, C))                             # [N, R, C]
        m.mul_(c["beta1"]).add_(grad, alpha=1.0 - c["beta1"])
        self._store_stacked(states, md, m.reshape((len(plist), R, C)))

        if nesterov:
            perturb = (m.mul(c["beta1"]).add_(grad, alpha=1.0 - c["beta1"])).mul_(inv_denom)
        else:
            perturb = m.mul(inv_denom)                                            # [N, R, C]

        # AdamP projection (per-channel radial removal on the matrixized [R, C] view).
        p_stack = torch.stack([mat(p.data).float() for p in plist])               # [N, R, C]
        perturb, wd_ratio = self._project_stacked(
            p_stack, grad, perturb, group["delta"], group["eps"], group["wd_ratio"]
        )

        # Decoupled weight decay (scaled per-slice by wd_ratio), then the step.
        if wd != 0:
            scale = (1.0 - group["lr"] * wd * wd_ratio)                           # [N, 1, 1]
            pviews = [mat(p.data) for p in plist]
            torch._foreach_mul_(pviews, list(scale.reshape(-1)))

        delta = perturb.mul_(c["step_size"])
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
        nesterov = group["nesterov"]

        states = [self.state[p] for p in plist]
        vs = [s["v"] for s in states]

        grad = torch.stack([p.grad.float() for p in plist])               # [N, L]
        v = torch.stack(vs)                                               # [N, L]

        v.mul_(c["beta2"]).addcmul_(grad, grad, value=1.0 - c["beta2"])
        torch._foreach_copy_(vs, list(v.unbind(0)))

        # Official 1-D denom: sqrt(v)/sqrt(bc2) + eps.
        de_nom = v.sqrt().div_(c["bc2_sq"]).add_(eps)

        m = self._dequant_stacked(states, md, (length,))                  # [N, L]
        m.mul_(c["beta1"]).add_(grad, alpha=1.0 - c["beta1"])
        self._store_stacked(states, md, m.reshape((len(plist), length)))

        if nesterov:
            perturb = (m.mul(c["beta1"]).add_(grad, alpha=1.0 - c["beta1"])).div_(de_nom)
        else:
            perturb = m.div(de_nom)

        # 1-D params are NEVER projected (official: len(p.shape) > 1 gate). Full WD.
        if wd != 0:
            torch._foreach_mul_([p.data for p in plist], 1.0 - group["lr"] * wd)

        delta = perturb.mul_(c["step_size"])
        if cautious:
            delta = cautious_batched_(delta, grad)
        subtract_batched_([p.data for p in plist], delta, bf16_method)

    # ---------------------------------------------------------- per-parameter
    @torch.no_grad()
    def _step_one_param(self, p: Tensor, group: dict[str, Any]) -> None:
        c = self._coeffs(group)
        md = group["momentum_dtype"]
        eps = group["eps"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]
        nesterov = group["nesterov"]

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
            inv_denom = (r_factor * c_factor).mul_(c["bc2_sq"])           # 1/sqrt(v_hat) [R, C]
            m = self._dequant_one(state, md, gv)
            m.mul_(c["beta1"]).add_(gv, alpha=1.0 - c["beta1"])
            self._store_one(state, md, m)
            if nesterov:
                perturb = (m.mul(c["beta1"]).add_(gv, alpha=1.0 - c["beta1"])).mul_(inv_denom)
            else:
                perturb = m.mul(inv_denom)                               # [R, C] matrixized view
            # Projection operates on the ORIGINAL-shape p / grad (official views).
            perturb_orig = perturb.reshape_as(grad)
            perturb_orig, wd_ratio = self._project_one(p.data, grad, perturb_orig, group)
            if wd != 0:
                p.data.mul_(1.0 - group["lr"] * wd * wd_ratio)
            delta = perturb_orig.mul_(c["step_size"])
        else:
            v = state["v"]
            v.mul_(c["beta2"]).addcmul_(grad, grad, value=1.0 - c["beta2"])
            de_nom = v.sqrt().div_(c["bc2_sq"]).add_(eps)                # official 1-D placement
            m = self._dequant_one(state, md, grad)
            m.mul_(c["beta1"]).add_(grad, alpha=1.0 - c["beta1"])
            self._store_one(state, md, m)
            if nesterov:
                perturb = (m.mul(c["beta1"]).add_(grad, alpha=1.0 - c["beta1"])).div_(de_nom)
            else:
                perturb = m.div(de_nom)
            # 1-D params are never projected; full decoupled WD.
            if wd != 0:
                p.data.mul_(1.0 - group["lr"] * wd)
            delta = perturb.mul_(c["step_size"])

        if cautious:
            delta = cautious_one_(delta, grad)
        subtract_one_(p, delta, state, bf16_method)
