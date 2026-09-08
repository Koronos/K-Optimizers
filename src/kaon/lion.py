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

import warnings
from collections.abc import Iterable
from typing import Any, Literal

import torch
from torch import Tensor
from torch.optim import Optimizer

from kaon._autolr import DEFAULT_FUSE_REL, AutoLRMixin
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
from kaon._foreach_plan import ForeachChunk, ForeachPlanMixin, ForeachSpec
from kaon._momentum_codec import (
    _FOURBIT_BLOCK,
    _make_codec,
    _MomentumCodec,
    load_state_dict_preserving_dtypes,
    warn_if_4bit_high_beta1,
)

__all__ = ["Lion"]

MomentumDtype = Literal["bfloat16", "float32", "int8", "4bit"]

# Performance cutoff (mirrors Adakaon): weights larger than this loop instead
# of being stacked — batching only pays off while per-tensor kernel-launch
# overhead dominates (small tensors); a large weight's sign step is
# bandwidth-bound, so stacking just adds copy traffic.
_STACK_BYTES_PER_ELEM = 48


class Lion(AutoLRMixin, ForeachPlanMixin, Optimizer):
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
            B/param, per-row absmax; **recommended cheap option**), or
            ``"4bit"`` (~0.5 B/param, per-block absmax, nibble-packed). Same
            storage layout as Adakaon's first moment, so checkpoints resume
            bit-exactly via ``load_state_dict``. **Warning:** Lion's update is
            ``sign(β1·m + (1-β1)·g)``, so 4-bit quantization noise flips signs
            (~12–13% of coordinates measured); final loss was ~32× worse vs bf16
            at Lion-scale lr, and ~3916× worse at lr=1e-3; ``cautious=True``
            (default) made loss worse still. Prefer ``"int8"``. ``"4bit"`` remains
            accepted for checkpoint compatibility.
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
        # 4bit + Lion is harmful (sign flips); still accepted for checkpoint compat.
        if momentum_dtype == "4bit":
            warnings.warn(
                "Lion(momentum_dtype='4bit'): measured ~12–13% sign flips from "
                "quantization noise; final loss ~32× worse vs bf16 (Lion-scale lr) "
                "and ~3916× worse at lr=1e-3; cautious=True (default) raised loss "
                "further. Prefer momentum_dtype='int8' (~1 B/param, near-lossless). "
                "4bit remains accepted so old checkpoints still load.",
                UserWarning,
                stacklevel=2,
            )
        warn_if_4bit_high_beta1(beta1, momentum_dtype)
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
        self._codecs: dict[str, Any] = {}
        # Composable continuous Mechanic LR via AutoLRMixin. When on, drives
        # the step via _step_impl at the discovered lr=S; off (default) -> step == _step_impl.
        self._init_autolr(auto_lr, auto_lr_scale, auto_lr_fuse_rel, auto_lr_d0)

    # ------------------------------------------------------------------- state
    @torch.no_grad()
    def _init_state(self, p: Tensor, state: dict[str, Any], group: dict[str, Any]) -> None:
        """Allocate the single momentum buffer in the configured storage layout.

        The layout IS :mod:`kaon._momentum_codec`'s (per-row int8 scale, per-block 4-bit
        scale, zero == nibble 8), so the codec allocates it — there is one owner of that
        layout and Lion is not it. Checkpoint resume and
        ``load_state_dict_preserving_dtypes`` therefore behave identically to Adakaon.
        """
        self._codec(group["momentum_dtype"]).init_state(state, p.grad, group)
        if is_low_precision(p) and group["bf16_method"] == "kahan":
            state["shift"] = torch.zeros_like(p)

    # -------------------------------------------------------- momentum (codec)
    def _codec(self, md: str) -> _MomentumCodec:
        codec = self._codecs.get(md)
        if codec is None:
            codec = self._codecs[md] = _make_codec(md)
        return codec

    # -------------------------------------------------------------------- step
    # step() is the AutoLRMixin router (drives Mechanic when auto_lr is on, else
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
                    # The whole group steps per-parameter: drop any cached plan for it, so
                    # a cached plan only ever describes a group the foreach path stepped.
                    self._drop_foreach_plan(group)
                    for p in params:
                        self._step_one_param(p, group)
            else:
                self._drop_foreach_plan(group)
                for p in params:
                    self._step_one_param(p, group)
        return loss

    def state_dict(self) -> dict[str, Any]:
        """Base state + the auto_lr tuner blob (via AutoLRMixin) when auto_lr is on."""
        return self._autolr_state_dict(super().state_dict())

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state, preserving the quantized momentum's stored dtype (torch's default
        would upcast bf16/int8/4bit momentum to fp32 on resume). The auto_lr tuner blob is
        peeled off first by AutoLRMixin.

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
        # The loader REPLACES every state tensor the cached views alias (and every group
        # dict, which the plan is keyed on), so the plan cannot survive it.
        self._clear_foreach_plans()

    def _autolr_reset_base_state(self) -> None:
        """Reset the base optimizer after an AutoLR rollback: the cleared state is
        reallocated by the next step, so the cached view plan must go with it."""
        super()._autolr_reset_base_state()
        self._clear_foreach_plans()

    # ----------------------------------------------------------------- foreach

    # Lion has NO state of its own to cache: its single buffer is the momentum, and the
    # codec's own cached view lists (``ForeachChunk.momentum_views``) cover it — hence
    # no ``factored_state`` / ``flat_state`` and no ``momentum_cache``.
    #
    # The four bucketing flags reproduce the bucketing Lion carried before it moved onto
    # the shared plan — ONE dict keyed by ``(exact shape, dtype)``, stepped in
    # first-appearance order — because that partition and that order are what the
    # stochastic-rounding draws are consumed in, and therefore reach bf16 weights
    # (``tests/test_foreach_plan.py::test_pinned_bf16_sr_vector_matches_the_pre_plan_tree``
    # is the anchor). Concretely: ``raw_shape_key`` keeps two convs that matrixize to the
    # same ``[R, C]`` apart, ``scalar_bucket`` keeps 0-D params out of the ``L == 1``
    # bucket shared with shape-``(1,)`` params, and ``insertion_order`` emits the buckets
    # in first-appearance order instead of all-factored-then-all-flat.
    #
    # Lion's update is fully per-coordinate, so it does not *need* any of those splits —
    # merging would be a (small) win it forgoes to keep bf16+SR runs reproducible. What it
    # does keep is ``matrixize``: a conv bucket works in ``[N, R, C]``, which is exactly
    # the layout the int8 per-row scale is defined on (``R`` = dim 0 of the weight), so
    # the quantized momentum matches the per-parameter path row for row.
    _FOREACH_SPEC = ForeachSpec(
        raw_shape_key=True,
        scalar_bucket=True,
        insertion_order=True,
    )

    @staticmethod
    def _group_foreach_eligible(group: dict[str, Any]) -> bool:
        return group["bf16_method"] != "kahan"  # kahan needs a per-param shift buffer

    @staticmethod
    def _param_foreach_eligible(p: Tensor, group: dict[str, Any], cutoff: int) -> bool:
        # 0-D scalars are NOT excluded: they ride the non-factored bucket as
        # length-1 rows (flattened to L=1 alongside shape-(1,) params). Only the
        # per-tensor size cap and the awkward dtype/contiguity cases fall back
        # to the per-param loop.
        if p.numel() > cutoff:
            return False
        # fp16+SR is unsupported (raises) -> route to the per-param path.
        if (
            group["bf16_method"] == "stochastic_rounding"
            and is_low_precision(p)
            and p.dtype != torch.bfloat16
        ):
            return False
        if p.ndim > 2:
            # A matrixized conv bucket writes back through a ``view(R, C)``; a
            # channels_last (or otherwise non-contiguous) conv has no such view.
            # Fall back to per-param.
            return p.data.is_contiguous() and p.grad.is_contiguous()
        return True

    @torch.no_grad()
    def _step_foreach(self, params: list[Tensor], group: dict[str, Any], budget: int) -> None:
        """Batched Lion step over the shared plan's cached chunks.

        Lion's update is fully per-coordinate (no factoring), so a chunk's effective
        layout is irrelevant to the sign/EMA/cautious/WD math — it is element-for-element
        :meth:`_step_one_param`'s either way. What the layout DOES decide is the int8
        per-row scale, which is why a conv bucket is worked in ``[N, R, C]`` (``R`` = dim
        0 of the weight, the axis ``_quant_int8`` reduces around).
        """
        beta1, beta2 = group["betas"]
        lr = group["lr"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]
        codec = self._codec(group["momentum_dtype"])
        for chunk in self._foreach_chunks(params, group, budget):
            self._bucket(chunk, codec, beta1, beta2, lr, wd, cautious, bf16_method)

    @torch.no_grad()
    def _bucket(
        self,
        chunk: ForeachChunk,
        codec: _MomentumCodec,
        beta1: float,
        beta2: float,
        lr: float,
        wd: float,
        cautious: bool,
        bf16_method: str,
    ) -> None:
        # ``[N, R, C]`` for a matrix/conv bucket, ``[N, L]`` for the flat one (0-D params
        # ride it as length-1 rows).
        eff = chunk.eff if chunk.eff is not None else (chunk.length,)
        states = chunk.states
        views = chunk.momentum_views(codec)

        grad = chunk.grad_stack()                                            # [N, *eff]
        m = codec.dequant_stacked(states, chunk.mat, eff, views=views)       # [N, *eff] fp32

        # Lion direction: sign of the beta1-interpolated momentum.
        c = m.mul(beta1).add_(grad, alpha=1.0 - beta1)
        delta = torch.sign(c)                                                # +1/0/-1

        # Momentum EMA with beta2 (on the raw gradient), then requant + store.
        m.mul_(beta2).add_(grad, alpha=1.0 - beta2)
        codec.store_stacked(states, m, views=views)

        if wd != 0:
            delta = delta.add_(chunk.param_stack(), alpha=wd)

        if cautious:
            delta = cautious_batched_(delta, grad)

        delta.mul_(lr)
        subtract_batched_(chunk.pviews, delta, bf16_method)

    # ---------------------------------------------------------- per-parameter
    @torch.no_grad()
    def _step_one_param(self, p: Tensor, group: dict[str, Any]) -> None:
        beta1, beta2 = group["betas"]
        lr = group["lr"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]
        codec = self._codec(group["momentum_dtype"])

        state = self.state[p]
        if not state:
            self._init_state(p, state, group)

        grad = p.grad if p.grad.dtype == torch.float32 else p.grad.float()
        m = codec.dequant_one(state, grad)                      # fp32, grad-shaped

        # Lion direction: sign(beta1 * m + (1 - beta1) * g).
        c = m.mul(beta1).add_(grad, alpha=1.0 - beta1)
        delta = torch.sign(c)

        # Momentum EMA with beta2, then requant + store. The codec writes codes and
        # scales IN PLACE (MSAM caches ``data_ptr`` of ``m`` / ``m_scale``).
        m.mul_(beta2).add_(grad, alpha=1.0 - beta2)
        codec.store_one(state, m)

        if wd != 0:
            p_fp32 = p.data if p.dtype == torch.float32 else p.data.float()
            delta = delta.add_(p_fp32, alpha=wd)

        if cautious:
            delta = cautious_one_(delta, grad)

        delta.mul_(lr)
        subtract_one_(p, delta, state, bf16_method)

