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
    # decoupled weight decay (scaled by wd_ratio when the projection fired), on the
    # pre-step weight and folded into the same fp32 write as the step:
    p -= (lr/bc1) * perturb + lr*weight_decay*wd_ratio * p

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
the stack and the radial removal is a masked broadcast coefficient applied with
``addcmul_``. Reassociating the fp32 radial expression changes results by about
5e-8 relative versus the historical normalized-vector implementation.

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

from kaon._autolr import DEFAULT_FUSE_REL, AutoLRMixin
from kaon._backend import (
    FOREACH_BATCH_CUTOFF,
    SRSeedState,
    cautious_batched_,
    cautious_one_,
    centralize_grads_,
    foreach_budget,
    init_bf16_state,
    is_low_precision,
    per_param_only_bf16_method,
    subtract_batched_,
    subtract_one_,
    validate_bf16_method,
    weight_value,
)
from kaon._decoupled_wd import decay_batched_, decay_one_
from kaon._factored import factored_inv_sqrt_factors, update_factored_state
from kaon._foreach_plan import ForeachChunk, ForeachPlanMixin, ForeachSpec
from kaon._momentum_codec import (
    _FOURBIT_BLOCK,
    _dequant_4bit,
    _make_codec,
    fourbit_block_size,
    load_state_dict_preserving_dtypes,
    warn_if_4bit_high_beta1,
)

__all__ = ["AdamP"]

def _zero_safe_inv_sqrt_factors(row: Tensor, col: Tensor) -> tuple[Tensor, Tensor]:
    """``factored_inv_sqrt_factors`` for ``eps1 == 0``, where exact-zero statistics are reachable.

    With no ``eps1`` inside the square, a row (or column) whose gradient has always been
    exactly zero keeps a statistic of exactly 0 — a dead unit, a frozen input, or a whole
    all-zero gradient. ``rsqrt(0) = inf`` then meets a first moment that is 0 there too, and
    ``0 * inf`` NaN'd the weights; a whole all-zero ``row`` was already ``0 / 0`` in the
    mean. A zero second moment with a zero first moment is a zero update, so those factors
    are 0 instead. Only an EXACTLY zero mean is replaced (not floored: a subnormal mean is a
    finite update, see ``kaon._factored``), and NaNs from the gradient still propagate (they
    are not inf). Called only when ``eps1 == 0``: every other configuration keeps the plain
    reconstruction, bit for bit. Shapes as ``factored_inv_sqrt_factors`` (any leading dims).
    """
    row_mean = row.mean(dim=-1, keepdim=True)
    row_mean.masked_fill_(row_mean == 0, 1.0)
    r_factor = row.div(row_mean).rsqrt_()
    r_factor.masked_fill_(r_factor.isinf(), 0.0)
    c_factor = col.rsqrt()
    c_factor.masked_fill_(c_factor.isinf(), 0.0)
    return r_factor.unsqueeze(-1), c_factor.unsqueeze(-2)


MomentumDtype = Literal["bfloat16", "float32", "int8", "4bit"]

# One momentum + factored v; on par with Adakaon's single-momentum working set.
_STACK_BYTES_PER_ELEM = 48


class AdamP(AutoLRMixin, ForeachPlanMixin, SRSeedState, Optimizer):
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
        weight_decay: decoupled (AdamW-style) weight decay,
            ``p *= (1 - lr*weight_decay*wd_ratio)`` on the pre-step weight. When the
            projection fires on a weight, ``wd_ratio`` is used (see below). Added to the
            fp32 step after the cautious mask, so bf16 weights decay through stochastic
            rounding / Kahan instead of rounding the factor away (see
            ``kaon._decoupled_wd``). Default ``0``.
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
            default**; pin ``False`` for the literal official step. Skipped where the
            fan-in is 1 (``(out, 1)``, ``(out, 1, 1, 1)``), because subtracting a
            one-element row's own mean zeroes the gradient — see
            ``kaon._backend.gc_applies``.
        momentum_dtype: storage for the first moment — ``"bfloat16"`` (default,
            ~2 B/param), ``"float32"`` (4 B/param), ``"int8"`` (~1 B/param, per-row
            absmax), or ``"4bit"`` (~0.5 B/param, per-block absmax, nibble-packed).
        momentum_4bit_block: block size for ``momentum_dtype="4bit"``. Default
            ``128``. ``0``/negative means whole-tensor.
        bf16_method: weight-update strategy for low-precision params —
            ``"stochastic_rounding"`` (default), ``"kahan8"`` (+1 B/param,
            compact fixed-point Kahan, see ``docs/research/compact-kahan.md``),
            ``"kahan16"`` (+2 B/param, bit-exact fp32 master weight split in two),
            ``"kahan"`` (+2 B/param, legacy per-param only), or
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
        validate_bf16_method(bf16_method)
        if foreach_batch_cutoff < 1:
            raise ValueError(f"foreach_batch_cutoff must be >= 1, got {foreach_batch_cutoff}")
        warn_if_4bit_high_beta1(beta1, momentum_dtype)
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
        self._codecs: dict[str, Any] = {}

        # Composable parameter-free LR (continuous Mechanic) via AutoLRMixin. off -> zero overhead.
        self._init_autolr(auto_lr, auto_lr_scale, auto_lr_fuse_rel, auto_lr_d0)

    # ------------------------------------------------------------------- state
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
            bs = fourbit_block_size(grad, group)
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
        self._alloc_momentum(grad, state, group)
        init_bf16_state(p, state, group["bf16_method"])

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

    def _codec(self, md: str):
        codec = self._codecs.get(md)
        if codec is None:
            codec = self._codecs[md] = _make_codec(md)
        return codec

    def _store_one(self, state: dict[str, Any], md: str, m_fp32: Tensor) -> None:
        """Write the updated fp32 first moment back into the configured storage layout.

        Delegates to the shared codec so codes/scales are written **in place**
        (MSAM caches ``data_ptr`` of ``m`` / ``m_scale``).
        """
        self._codec(md).store_one(state, m_fp32)

    # ----------------------------------------------------------- coefficients
    @staticmethod
    def _coeffs(group: dict[str, Any], step: int) -> dict[str, float]:
        """All per-step scalar coefficients (shared by the per-param and foreach paths)."""
        beta1, beta2 = group["betas"]
        bc1 = 1.0 - beta1 ** step
        bc2_sq = math.sqrt(1.0 - beta2 ** step)
        return {
            "beta1": beta1,
            "beta2": beta2,
            "bc1": bc1,
            "bc2_sq": bc2_sq,
            "step_size": group["lr"] / bc1,
        }

    def _prepare_param_step(self, p: Tensor, group: dict[str, Any]) -> None:
        """Initialize and advance the bias-correction clock for one active param."""
        state = self.state[p]
        if not state:
            self._init_state(p, state, group)
            state["step"] = 1
        elif "step" not in state:
            # Legacy checkpoints only stored the group clock. The global clock
            # has already advanced for this call, so it reconstructs the normal
            # all-params-active trajectory exactly.
            state["step"] = group["step"]
        else:
            state["step"] += 1

    # ------------------------------------------------------------- projection
    def _project_one(
        self, p: Tensor, grad: Tensor, perturb: Tensor, group: dict[str, Any]
    ) -> tuple[Tensor, Tensor]:
        """Device-only per-param projection: channel first, else layer, else none.

        ``grad`` and ``p`` are in the param's ORIGINAL shape (not matrixized) so the
        official views are preserved. The one-element batch avoids converting either
        projection predicate to a Python bool (and synchronizing CUDA). ``p`` is a
        low-precision weight on bf16/fp16 runs, so it is widened here — the same
        thing :meth:`_factored_bucket` does when it stacks — because the whole
        projection (``bmm``, the norms, the in-place ``addcmul_`` into the fp32
        ``perturb``) has to run in one working dtype.
        """
        rows = p.shape[0]
        cols = p.numel() // rows
        projected, wd_ratio = self._project_stacked(
            p.detach().float().reshape(1, rows, cols),
            grad.float().reshape(1, rows, cols),
            perturb.reshape(1, rows, cols),
            group["delta"],
            group["eps"],
            group["wd_ratio"],
        )
        return projected.reshape_as(perturb), wd_ratio.reshape(())

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
        mask over the stack. The radial removal is a masked broadcast coefficient;
        reassociation versus the normalized-vector form is bounded by the parity tests.

        ``p_stack``/``g_stack``/``perturb`` are the matrixized ``[N, R, C]`` stacks
        (``R = out``, ``C = fan-in``). Returns ``(perturb, wd_ratio[N,1,1])``.

        ``perturb`` (the Adam direction, always fp32) sets the working dtype: the
        weights and gradients are widened to it, and the result is written back to
        the parameter's own dtype by the caller's subtract.
        """
        n, r, c = p_stack.shape
        work = perturb.dtype
        if p_stack.dtype != work:
            p_stack = p_stack.to(work)
        if g_stack.dtype != work:
            g_stack = g_stack.to(work)

        # Every quantity the projection needs is a per-row statistic of three [N, R, C]
        # stacks — ||p||, ||g||, p.g and p.perturb over C — plus the flat ||p|| of the layer
        # view. Each is ONE reduction pass (bmm rows for the dots), and the layer cosine is
        # assembled from the row ones. ``F.cosine_similarity`` instead normalizes both
        # inputs into fresh [N, R, C] copies, twice (channel and layer view), which was
        # ~40% of AdamP's step on DiT shapes. Its formula is kept: each norm is clamped at
        # ``eps`` separately, ``g.p / (max(||g||, eps) * max(||p||, eps))``. The cosines
        # only feed the two fire comparisons, so this moves no weight unless a cosine sits
        # within rounding (~3e-8 relative) of its threshold.
        rows = n * r
        g_norm = torch.linalg.vector_norm(g_stack, dim=-1)                       # [N, R]
        p_norm = torch.linalg.vector_norm(p_stack, dim=-1)                       # [N, R]
        gp_dot = torch.bmm(g_stack.reshape(rows, 1, c), p_stack.reshape(rows, c, 1)).view(n, r)

        # --- channel view: cosine per (slice, row) over C ---
        cos_ch = gp_dot.div(g_norm.clamp_min(eps).mul_(p_norm.clamp_min(eps))).abs_()  # [N, R]
        ch_fire = cos_ch.amax(dim=1) < (delta / math.sqrt(c))  # [N] bool

        # --- layer view: cosine per slice over R*C, from the row statistics ---
        ly_norm = p_stack.reshape(n, r * c).norm(dim=-1)                          # [N]
        g_ly = g_norm.square().sum(dim=1).sqrt_()                                 # [N]
        cos_ly = gp_dot.sum(dim=1).div_(g_ly.clamp_min_(eps).mul_(ly_norm.clamp_min(eps))).abs_()
        ly_fire = (cos_ly < (delta / math.sqrt(r * c))) & (~ch_fire)  # [N] bool

        ch_mask = ch_fire.view(n, 1, 1)
        ly_mask = ly_fire.view(n, 1, 1)

        # radial = p * (sum(p * perturb) / (||p|| + eps)^2). bmm produces only
        # [N,R,1] coefficients; one addcmul_ applies them without any [N,R,C] radial
        # or normalized-weight temporaries. Layer fire excludes channel fire, so
        # the pre-channel row dots are also valid for every layer-fired slice, and the
        # two coefficients are never both non-zero: their sum is exactly the one that
        # fired, so a single pass replaces the channel pass + layer pass bit for bit.
        row_dot = torch.bmm(
            p_stack.reshape(rows, 1, c),
            perturb.reshape(rows, c, 1),
        ).reshape(n, r, 1)
        ch_norm = p_norm.unsqueeze(-1).add_(eps)
        zero = perturb.new_zeros(())
        ch_coef = torch.where(ch_mask, row_dot.div(ch_norm.square()), zero)
        ly_norm = ly_norm.view(n, 1, 1).add_(eps)
        ly_coef = torch.where(
            ly_mask,
            row_dot.sum(dim=1, keepdim=True).div_(ly_norm.square()),
            zero,
        )
        perturb.addcmul_(p_stack, ch_coef.add_(ly_coef), value=-1.0)

        fired = (ch_fire | ly_fire).view(n, 1, 1)
        wd_ratio = torch.where(
            fired, perturb.new_full((), wd_ratio_hp), perturb.new_ones(())
        )  # [N, 1, 1]
        return perturb, wd_ratio

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
                    raise RuntimeError("AdamP does not support sparse gradients")
            group["step"] += 1
            if not params:
                continue
            if group["gradient_centralization"]:
                centralize_grads_(params)
            for p in params:
                self._prepare_param_step(p, group)
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
                    self._drop_foreach_plan(group)
                    for p in params:
                        self._step_one_param(p, group)
            else:
                # Per-parameter fallback for the whole group: drop any cached plan for it,
                # so a cached plan only ever describes a group the foreach path stepped.
                self._drop_foreach_plan(group)
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
        back to fp32 on resume. Delegate to the shared helper.

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
    # stack and write back through; the per-parameter step joins the bucket key so every
    # slice of a bucket shares one bias correction (and one ``_coeffs`` dict) — and, with
    # it, the projection's per-channel view dims. The momentum goes through the shared
    # codec's stacked read/write, whose own per-param view lists come from
    # ``chunk.momentum_views``; ``chunk.view`` rides along as the codec's ``mat``
    # fallback for a layout those views declined.
    _FOREACH_SPEC = ForeachSpec(
        factored_state=("row", "col"),
        flat_state=("v",),
        extra_key=lambda state, group: max(state["step"], 1),
    )

    @staticmethod
    def _group_foreach_eligible(group: dict[str, Any]) -> bool:
        return not per_param_only_bf16_method(group["bf16_method"])  # kahan needs a per-param shift buffer

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
            and p.dtype != torch.bfloat16  # fp16+SR unsupported -> per-param (raises)
        ):
            return False
        if p.ndim > 2:
            # Matrixized conv writes back through a reshaped view -> needs contiguity.
            return p.data.is_contiguous() and p.grad.is_contiguous()
        return True

    @torch.no_grad()
    def _step_foreach(self, params: list[Tensor], group: dict[str, Any], budget: int) -> None:
        """Batched step. Factored (ndim>=2, projected) and non-factored (ndim<=1) buckets.

        0-D scalars ride the non-factored bucket keyed by ``numel() == 1``, sharing it
        with real shape-(1,) params. The ``max(step, 1)`` clamp in the bucket key only
        covers a state allocated by the plan builder itself (step 0), whose bias
        correction would otherwise be ``1 - beta**0 == 0``; ``_step_impl`` advances every
        param's step before this runs."""
        md = group["momentum_dtype"]
        coeffs: dict[int, dict[str, float]] = {}
        for chunk in self._foreach_chunks(params, group, budget):
            c = coeffs.get(chunk.key)
            if c is None:
                c = coeffs[chunk.key] = self._coeffs(group, chunk.key)
            bucket = self._factored_bucket if chunk.eff is not None else self._nonfactored_bucket
            bucket(chunk, md, c, group)

    @torch.no_grad()
    def _factored_bucket(
        self,
        chunk: ForeachChunk,
        md: str,
        c: dict[str, float],
        group: dict[str, Any],
    ) -> None:
        R, C = chunk.eff  # noqa: N806
        eps1 = group["eps"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]
        nesterov = group["nesterov"]

        states = chunk.states
        rows, cols = chunk.state_views
        pviews = chunk.pviews

        grad = chunk.grad_stack()                                         # [N, R, C]
        row = torch.stack(rows)                                           # [N, R]
        col = torch.stack(cols)                                           # [N, C]

        # Factored second-moment EMA (HF eps1 placement).
        omb2 = 1.0 - c["beta2"]
        grad_sq = grad * grad
        if eps1 > 0:
            grad_sq = grad_sq.add_(eps1)
        row.lerp_(grad_sq.mean(dim=-1), omb2)
        col.lerp_(grad_sq.mean(dim=-2), omb2)
        del grad_sq                                   # dead [N, R, C] before inv_denom / m
        torch._foreach_copy_(rows, list(row.unbind(0)))
        torch._foreach_copy_(cols, list(col.unbind(0)))

        if eps1 > 0:
            r_factor = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)  # [N, R, 1]
            c_factor = col.rsqrt().unsqueeze(-2)                                       # [N, 1, C]
        else:  # exact-zero statistics are reachable: 0 * inf would NaN
            r_factor, c_factor = _zero_safe_inv_sqrt_factors(row, col)
        inv_denom = (r_factor * c_factor).mul_(c["bc2_sq"])                        # 1/sqrt(v_hat)

        # First-moment EMA (raw Adam momentum), read, EMA, store back.
        codec = self._codec(md)
        views = chunk.momentum_views(codec)
        m = codec.dequant_stacked(states, chunk.view, (R, C), views=views)         # [N, R, C]
        m.mul_(c["beta1"]).add_(grad, alpha=1.0 - c["beta1"])
        codec.store_stacked(states, m.reshape((chunk.n, R, C)), views=views)

        # perturb is written into inv_denom's buffer (the product commutes: same bits).
        if nesterov:
            perturb = inv_denom.mul_(m.mul(c["beta1"]).add_(grad, alpha=1.0 - c["beta1"]))
        else:
            perturb = inv_denom.mul_(m)                                           # [N, R, C]
        del m, inv_denom

        # AdamP projection (per-channel radial removal on the matrixized [R, C] view). It
        # reads the weight's full value: decoded under kahan8/kahan16 (the decoded stacks are
        # reused by the write), the plain fp32 stack otherwise (kaon._backend.weight_value).
        p_stack, stacks = chunk.value_and_stacks(bf16_method)                     # [N, R, C]
        perturb, wd_ratio = self._project_stacked(
            p_stack, grad, perturb, group["delta"], group["eps"], group["wd_ratio"]
        )

        delta = perturb.mul_(c["step_size"])
        if cautious:
            delta = cautious_batched_(delta, grad)
        # Decoupled weight decay (scaled per-slice by wd_ratio) on the pre-step value the
        # projection read, in the fp32 delta, outside the mask (kaon._decoupled_wd).
        if wd != 0:
            decay_batched_(delta, chunk, bf16_method, group["lr"] * wd, ratio=wd_ratio,
                           value=p_stack)
        subtract_batched_(pviews, delta, bf16_method, sr=self.sr_stream, comp=chunk.cviews,
                          stacked=stacks)

    @torch.no_grad()
    def _nonfactored_bucket(
        self,
        chunk: ForeachChunk,
        md: str,
        c: dict[str, float],
        group: dict[str, Any],
    ) -> None:
        """Non-factored (full per-coordinate ``v``) update for ``ndim <= 1`` params.

        0-D scalars share the ``L == 1`` bucket with shape-``(1,)`` params as length-1
        **views** (:func:`~kaon._backend.flat_view`) of the same storage, so the state
        write-backs and the weight subtract reach the original 0-D tensors. Those views
        are the cached plan's (``chunk.state_views`` / ``chunk.pviews``), not rebuilt per
        param per step. Like every other param in this bucket they are never projected —
        which is exactly the official AdamP ``len(p.shape) > 1`` gate the per-param path
        applies to them.
        """
        eps = group["eps"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]
        nesterov = group["nesterov"]

        length = chunk.length
        states = chunk.states
        (vs,) = chunk.state_views
        pviews = chunk.pviews

        grad = chunk.grad_stack()                                         # [N, L]
        v = torch.stack(vs)                                               # [N, L]

        v.mul_(c["beta2"]).addcmul_(grad, grad, value=1.0 - c["beta2"])
        torch._foreach_copy_(vs, list(v.unbind(0)))

        # Official 1-D denom: sqrt(v)/sqrt(bc2) + eps.
        de_nom = v.sqrt().div_(c["bc2_sq"]).add_(eps)

        codec = self._codec(md)
        views = chunk.momentum_views(codec)
        m = codec.dequant_stacked(states, chunk.view, (length,), views=views)  # [N, L]
        m.mul_(c["beta1"]).add_(grad, alpha=1.0 - c["beta1"])
        codec.store_stacked(states, m.reshape((chunk.n, length)), views=views)

        if nesterov:
            perturb = (m.mul(c["beta1"]).add_(grad, alpha=1.0 - c["beta1"])).div_(de_nom)
        else:
            perturb = m.div(de_nom)

        # ndim<=1 params are NEVER projected (official: len(p.shape) > 1 gate). Full WD.
        delta = perturb.mul_(c["step_size"])
        if cautious:
            delta = cautious_batched_(delta, grad)
        stacks = decay_batched_(delta, chunk, bf16_method, group["lr"] * wd) if wd != 0 else None
        subtract_batched_(pviews, delta, bf16_method, sr=self.sr_stream, comp=chunk.cviews,
                          stacked=stacks)

    # ---------------------------------------------------------- per-parameter
    @torch.no_grad()
    def _step_one_param(self, p: Tensor, group: dict[str, Any]) -> None:
        state = self.state[p]
        if not state or "step" not in state:
            self._prepare_param_step(p, group)
        c = self._coeffs(group, state["step"])
        md = group["momentum_dtype"]
        eps = group["eps"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]
        nesterov = group["nesterov"]

        grad = p.grad if p.grad.dtype == torch.float32 else p.grad.float()
        ndim = grad.ndim
        factored = ndim >= 2

        if factored:
            matrixize = ndim > 2
            gv = grad.reshape(grad.shape[0], -1) if matrixize else grad
            update_factored_state(gv, state["row"], state["col"], c["beta2"], eps)
            if eps > 0:
                r_factor, c_factor = factored_inv_sqrt_factors(state["row"], state["col"])
            else:  # exact-zero statistics are reachable: 0 * inf would NaN
                r_factor, c_factor = _zero_safe_inv_sqrt_factors(state["row"], state["col"])
            inv_denom = (r_factor * c_factor).mul_(c["bc2_sq"])           # 1/sqrt(v_hat) [R, C]
            m = self._dequant_one(state, md, gv)
            m.mul_(c["beta1"]).add_(gv, alpha=1.0 - c["beta1"])
            self._store_one(state, md, m)
            # perturb is written into inv_denom's buffer (the product commutes: same bits).
            if nesterov:
                perturb = inv_denom.mul_(m.mul(c["beta1"]).add_(gv, alpha=1.0 - c["beta1"]))
            else:
                perturb = inv_denom.mul_(m)                              # [R, C] matrixized view
            del m, inv_denom
            # Projection operates on the ORIGINAL-shape p / grad (official views).
            # (the decoded full value under kahan8/kahan16 — see kaon._backend.weight_value)
            p_value = weight_value(p, state, bf16_method)
            perturb_orig = perturb.reshape_as(grad)
            perturb_orig, wd_ratio = self._project_one(p_value, grad, perturb_orig, group)
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
            p_value, wd_ratio = None, None
            delta = perturb.mul_(c["step_size"])

        if cautious:
            delta = cautious_one_(delta, grad)
        # Decoupled weight decay p*(1 - lr*wd*wd_ratio) - delta, folded into the fp32 delta
        # so a bf16 weight decays through SR / Kahan too (see kaon._decoupled_wd).
        if wd != 0:
            delta = decay_one_(delta, p, state, bf16_method, group["lr"] * wd,
                               ratio=wd_ratio, value=p_value)
        subtract_one_(p, delta, state, bf16_method, sr=self.sr_stream)
