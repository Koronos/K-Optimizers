"""Shared per-step primitives reused by every kaon optimizer's update.

One implementation each — so a fix or a perf change lands everywhere at once — for the
cross-cutting pieces that used to be copy-pasted into each optimizer:

* the low-precision dtype check,
* the 0-D -> length-1 view used by every non-factored foreach bucket,
* the bf16-correct weight write (``p -= delta``), per-param and batched (foreach),
* cautious masking (Liang et al. 2024), per-param and batched.

All are bit-exact with the per-optimizer copies they replaced (same arithmetic); the
``foreach == per-param`` parity tests across every optimizer/dtype are the proof.

ONE EXCEPTION, since 0.7.12: the bf16 stochastic-rounding write prefers a Triton kernel on
CUDA (:data:`SR_TRITON`), whose noise comes from a different RNG than the torch path's. Both
are unbiased, but they do not reproduce each other's draws, and only the torch path keeps
``foreach == per-param`` for the *noise* as well as the arithmetic (its generator is one
running sequence, so a stacked draw and the equivalent per-param draws consume the same
numbers; the kernel seeds each launch separately). ``kaon.reseed_stochastic_rounding()``
resets both streams; ``SR_TRITON = False`` pins the torch reference implementation.

Both paths draw from the **caller's** noise stream (:class:`SRSeedState`, threaded as ``sr``
through the writers below), so where the noise sits is per-optimizer, checkpointed state
instead of a process-global counter — which is what makes a bf16 resume bit-identical to
the run it continues.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from kaon._stochastic_rounding import SRStream, add_stochastic_

__all__ = [
    "DEFAULT_STACK_ELEMS",
    "FOREACH_BATCH_CUTOFF",
    "LOW_PRECISION",
    "MIN_STACK_ELEMS",
    "STACK_SAFETY_FRACTION",
    "SRSeedState",
    "cautious_batched_",
    "cautious_one_",
    "centralize_grads_",
    "flat_view",
    "foreach_budget",
    "is_low_precision",
    "rms",
    "subtract_batched_",
    "subtract_one_",
]

LOW_PRECISION = (torch.bfloat16, torch.float16)


def is_low_precision(t: Tensor) -> bool:
    return t.dtype in LOW_PRECISION


def rms(t: Tensor) -> Tensor:
    """Root-mean-square of ``t`` (Adafactor-style update normalizer)."""
    return t.norm(2) / math.sqrt(max(t.numel(), 1))


def flat_view(t: Tensor) -> Tensor:
    """0-D scalar -> length-1 **view**; anything else returned untouched.

    The single place every optimizer's non-factored foreach bucket goes through to
    admit 0-D params (LyCORIS ``use_scalar`` gates and friends). A bag of scalars on
    the per-parameter path costs ~20 CUDA launches *per scalar per step* — pure CPU
    dispatch, measured ~6x the wall time of the same params shaped ``(1,)`` — which is
    exactly what foreach batching exists to remove. Bucketing them by ``numel()``
    drops them into the ``L == 1`` non-factored bucket alongside real shape-``(1,)``
    params, and the whole update (grad stack, state stack, codec write-back, weight
    subtract) then flows through length-1 rows.

    It must be a *view*, not a reshape-copy: the bucket writes state and weights back
    through it, so those writes have to reach the original 0-D storage and leave the
    persisted state at its per-param shape (checkpoints stay compatible across paths).
    For ``L == 1`` the batched per-slice reductions degenerate to the per-param scalar
    ones — ``norm(dim=1)/sqrt(1) == rms(x)``, and a one-element mask's mean is the mask
    — so the batched and per-param paths stay element-for-element identical.
    """
    return t.view(1) if t.ndim == 0 else t


# ----------------------------- foreach budget -----------------------------
# Shared foreach-batching knobs (the per-optimizer ``bytes_per_elem`` differs and is
# passed in; everything else is identical across optimizers). See docs/foreach-batching.md.
FOREACH_BATCH_CUTOFF = 2_000_000   # per-tensor element cap above which a weight loops (perf)
STACK_SAFETY_FRACTION = 0.10       # use at most ~10% of currently-free VRAM per stacked chunk
MIN_STACK_ELEMS = 262_144          # still batch small tensors even under memory pressure
DEFAULT_STACK_ELEMS = 64_000_000   # CPU / unknown device: no VRAM limit to respect


def foreach_budget(stack_budget: int | None, batch_cutoff: int, bytes_per_elem: int,
                   device: torch.device) -> int:
    """Max elements per stacked chunk for the foreach path.

    An explicit ``stack_budget`` is returned verbatim. Otherwise the chunk is
    ``min(adaptive_to_free_VRAM, 4 * batch_cutoff)``: the VRAM term shrinks the chunk when a
    big model already fills the card (OOM safety) and grows it on a roomy one; the
    ``4 * batch_cutoff`` cap stops over-stacking (beyond a few cutoff-sized tensors, stacking
    medium weights just adds copy bandwidth). ``bytes_per_elem`` is the optimizer's stacked
    working-set estimate per element (more momenta -> larger).
    """
    if stack_budget is not None:
        return stack_budget
    cap = 4 * batch_cutoff
    if device.type == "cuda":
        free_bytes = torch.cuda.mem_get_info(device)[0]
        adaptive = int(free_bytes * STACK_SAFETY_FRACTION / bytes_per_elem)
        return max(MIN_STACK_ELEMS, min(adaptive, cap))
    return min(DEFAULT_STACK_ELEMS, cap)


# ----------------------------- bf16 stochastic-rounding write -----------------------------
# The bf16 + stochastic-rounding write is the one weight write that is NOT a single torch op:
# ``add_stochastic_`` upcasts, draws a parameter-sized int32 noise tensor, masks it, adds,
# masks the mantissa and casts back — ~7-8 kernels and two parameter-sized temporaries per
# call. A Triton kernel does the whole thing in one launch with no temporary
# (``kaon._fused_triton.sr_add_``), so this module prefers it whenever the tensors qualify
# (CUDA, contiguous, bf16 target / fp32 source) and Triton is installed. CPU, fp16, strided
# views and Triton-less builds keep the torch path, which stays the reference implementation.
#
# The two draw from DIFFERENT noise streams. Both are unbiased, which is the only property
# stochastic rounding is relied on for, but a run is only bit-reproducible against itself —
# set ``kaon._backend.SR_TRITON = False`` to pin the torch path (also the A/B switch the
# speedup below is measured with).
SR_TRITON = True


def _sr_write_(
    target: Tensor,
    source: Tensor,
    alpha: float,
    triton: bool | None = None,
    sr: SRStream | None = None,
) -> None:
    """``target += alpha * source`` with bf16 SR, through Triton when it applies.

    ``sr`` is the calling optimizer's noise stream (see :class:`SRSeedState`). BOTH paths
    consume one draw from it, so which path a write took is invisible to the counter and a
    run may cross between them without desynchronising its noise. Leaving it ``None`` falls
    back to the process-wide streams, whose position no checkpoint saves.
    """
    use = SR_TRITON if triton is None else triton
    if use:
        from kaon import _fused_triton as ft
        if ft.sr_add_supported(target, source):
            ft.sr_add_(target, source, alpha, sr)
            return
    add_stochastic_(target, source, alpha=alpha, sr=sr)


# ----------------------------- per-optimizer SR noise stream -----------------------------
class SRSeedState:
    """Mixin: the optimizer's own stochastic-rounding noise stream, in its ``state_dict``.

    Where a bf16 SR weight write sits in its noise — a launch counter for the Triton kernel,
    a generator state for the torch reference path — is **optimizer state**: an optimizer
    that restarts it at 0 on resume applies different noise than the run it continues, so a
    resumed run's weights diverge from the continuous run's even with every state tensor
    restored bit-exactly. Until 0.7.13 the counter was one process-global, per-device one
    that no checkpoint saved, and no optimizer with bf16 params on the native path resumed
    bit-identically (measured ~4.7e-2 max abs four steps after a resume).

    Mixed in **before** ``Optimizer`` in the bases, so a subclass's
    ``super().state_dict()`` picks the meta up with no per-optimizer save code; the restore
    side is :func:`kaon._momentum_codec.load_state_dict_preserving_dtypes`, which every
    kaon optimizer's ``load_state_dict`` already funnels through (wrappers do it for their
    own stream in :class:`kaon._wrappers.WrapsInnerOptimizer`). What each optimizer *does*
    have to do is hand :attr:`sr_stream` to the writers (``sr=self.sr_stream``) — an
    optimizer that forgets silently falls back to the process-wide counter and stops
    resuming exactly, which ``tests/test_sr_seed_checkpoint.py`` sweeps for.

    The stream is created on first use rather than in ``__init__``: no optimizer has to
    grow a constructor line, and ``state_dict`` on an optimizer that never stepped stays
    free of a stream id (so it does not consume one and perturb the ids of the streams that
    follow).
    """

    SR_META_KEY = "_sr_meta"

    @property
    def sr_stream(self) -> SRStream:
        """This optimizer's noise stream (allocated on first access)."""
        stream = self.__dict__.get("_sr_stream")
        if stream is None:
            stream = self.__dict__["_sr_stream"] = SRStream()
        return stream

    def _sr_save(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Add the stream's position to ``state_dict`` (no-op if it was never used)."""
        stream = self.__dict__.get("_sr_stream")
        if stream is not None:
            state_dict[self.SR_META_KEY] = stream.snapshot()
        return state_dict

    def _sr_load(self, state_dict: dict[str, Any]) -> None:
        """Adopt the checkpoint's stream position; a checkpoint without one resets.

        An optimizer with neither a saved position nor a live stream is left WITHOUT one:
        a load must not allocate a stream id for an owner that never draws (the wrappers
        whose weight writes are all the inner optimizer's), because that would shift the
        ids of the streams allocated after it away from what the live run assigned.
        """
        meta = state_dict.get(self.SR_META_KEY)
        if meta is None and self.__dict__.get("_sr_stream") is None:
            return
        self.sr_stream.restore(meta)

    def state_dict(self) -> dict[str, Any]:
        return self._sr_save(super().state_dict())  # type: ignore[misc]


# ----------------------------- weight write: p -= delta -----------------------------
@torch.no_grad()
def subtract_one_(p: Tensor, delta_fp32: Tensor, state: dict, bf16_method: str,
                  alpha: float = 1.0, triton: bool | None = None,
                  sr: SRStream | None = None) -> None:
    """Per-parameter ``p -= alpha * delta`` with the configured bf16 handling.

    ``kahan`` keeps a per-param compensation buffer (``state['shift']``); ``stochastic_
    rounding`` does the unbiased bf16 round; otherwise a plain cast-and-subtract.

    ``alpha`` is the learning rate the caller would otherwise have applied with a separate
    ``delta.mul_(lr)`` pass — see :func:`subtract_batched_`. The two must stay in step:
    ``Tensor.sub_(d, alpha=lr)`` and ``torch._foreach_sub_([p], [d], alpha=lr)`` are
    BIT-IDENTICAL (verified on CPU and CUDA), which is what keeps the per-parameter path
    and the foreach path bit-exact with each other — the invariant
    ``tests/test_adakaon.py::test_foreach_matches_per_param`` guards. Folding in only one
    of them breaks it, because ``mul_`` then subtract and ``sub_(alpha=)`` differ by a
    sub-ulp contraction.

    ``kahan`` is the exception and keeps the explicit fp32 product: its compensation buffer
    is in the param's dtype, so scaling during the narrowing subtract would round the
    product to bf16 and defeat the compensation. Kahan never reaches the foreach path
    (``Adakaon._group_foreach_eligible`` rejects it), so no invariant depends on it.
    """
    low = is_low_precision(p)
    if low and bf16_method == "kahan":
        shift = state["shift"]
        shift.sub_((delta_fp32 * alpha if alpha != 1.0 else delta_fp32).to(p.dtype))
        p_before = p.detach().clone()
        p.add_(shift)
        shift.add_(p_before.sub_(p))
    elif low and bf16_method == "stochastic_rounding" and p.dtype == torch.bfloat16:
        _sr_write_(p.data, delta_fp32, -alpha, triton, sr)
    elif p.dtype == delta_fp32.dtype:
        p.data.sub_(delta_fp32, alpha=alpha)
    elif alpha == 1.0:
        p.data.sub_(delta_fp32.to(p.dtype))
    else:  # scale in fp32 before the narrowing cast (see subtract_batched_)
        p.data.sub_((delta_fp32 * alpha).to(p.dtype))


@torch.no_grad()
def subtract_batched_(pviews: list[Tensor], delta: Tensor, bf16_method: str,
                      alpha: float = 1.0, triton: bool | None = None,
                      sr: SRStream | None = None) -> None:
    """In-place ``p -= alpha * delta`` over a foreach bucket of (matrixized) param views.

    ``pviews`` is the list of N same-shape param views (each ``[*shape]``); ``delta`` is
    the stacked fp32 step ``[N, *shape]`` (row i applies to ``pviews[i]``).

    Only the **bf16 + stochastic-rounding** case needs a materialized stacked-weights
    tensor (``add_stochastic_`` operates on the stack). Every other case — notably the
    fp32 regime, including LoRA's many-tiny-tensor buckets — subtracts the delta slices
    straight into the param views with ``_foreach_sub_``, skipping *both* the stack-weights
    allocation and the copy-back, which are pure overhead in the launch-bound regime.

    ``alpha`` exists so a caller can hand over the learning rate instead of scaling the
    delta itself: ``delta.mul_(lr)`` is a full read-modify-write pass over the stacked
    bucket, while every writer below already takes an ``alpha`` and folds the multiply
    into the pass it was going to make anyway. Both bf16 branches keep the product in
    **fp32** (``add_stochastic_`` upcasts; the plain-cast branch scales before the cast),
    so folding never costs precision. Not bit-identical to a separate ``mul_``: an
    ``a - alpha*b`` kernel may contract to an FMA where a separate multiply rounds. The
    difference is sub-ulp but NOT zero — measured 2.8e-8 to 5.6e-8 relative on fp32 params
    over 6 steps, and 0 on bf16 params (the narrowing write absorbs it). What IS bit-identical, and
    is what the ``foreach == per-param`` invariant rests on, is ``Tensor.sub_(d, alpha=lr)``
    against ``torch._foreach_sub_([p], [d], alpha=lr)``."""
    p0 = pviews[0]
    if p0.dtype == torch.bfloat16 and bf16_method == "stochastic_rounding":
        weights = torch.stack(pviews)
        _sr_write_(weights, delta, -alpha, triton, sr)
        torch._foreach_copy_(pviews, list(weights.unbind(0)))
    elif p0.dtype == delta.dtype:
        torch._foreach_sub_(pviews, list(delta.unbind(0)), alpha=alpha)
    elif alpha == 1.0:
        torch._foreach_sub_(pviews, list(delta.to(p0.dtype).unbind(0)))
    else:
        # Scale in fp32 BEFORE the narrowing cast: a bf16 ``alpha * delta`` would round
        # the product to bf16 and lose exactly the small updates this path exists to keep.
        torch._foreach_sub_(pviews, list((delta * alpha).to(p0.dtype).unbind(0)))


# ----------------------------- cautious masking -----------------------------
@torch.no_grad()
def cautious_batched_(delta: Tensor, grad: Tensor) -> Tensor:
    """Cautious masking on a foreach bucket ``[N, *shape]``: zero the update coordinates
    whose sign disagrees with the gradient (``delta*grad <= 0``) and rescale the survivors
    by their per-slice surviving fraction so the mean step magnitude is preserved. Modifies
    and returns ``delta``."""
    mask = (delta * grad > 0).to(delta.dtype)
    n = delta.shape[0]
    denom = mask.reshape(n, -1).mean(dim=1).clamp_(min=1e-8).view(n, *([1] * (delta.ndim - 1)))
    return delta.mul_(mask).div_(denom)


@torch.no_grad()
def cautious_one_(delta: Tensor, grad: Tensor) -> Tensor:
    """Per-parameter cautious masking (scalar rescale). Modifies and returns ``delta``."""
    mask = (delta * grad > 0).to(delta.dtype)
    return delta.mul_(mask).div_(mask.mean().clamp_(min=1e-8))


# ----------------------------- gradient preprocessing -----------------------------
@torch.no_grad()
def centralize_grads_(params: list[Tensor]) -> None:
    """Gradient Centralization (Yong et al. 2020, arXiv:2004.01461), in place.

    For every ``ndim >= 2`` weight, subtract the gradient's mean over the fan-in dims (all
    dims except the output channel, dim 0) per output row. A **zero-state** gradient
    preprocessor applied at the top of the step, before the optimizer reads ``p.grad``;
    1-D params (biases / norm scales) are left untouched.

    Measured (proxy, gap lens, 3 seeds-pairs): a free held-out-loss win (~-0.003..-0.006) for
    the factored-Adam and sign optimizers (Adakaon, Lion, AdaPNM, KProdigy); neutral-to-negative
    for the orthogonalized AdaMuon, so it is per-optimizer opt-out (``gradient_centralization``).

    **Batched by shape** so the LoRA many-tiny-tensor regime stays fast: a naive per-param Python
    loop here added ~1024 kernel launches/step on a 512-adapter bag (3x slower). Same-shape grads
    are stacked and centralized in a handful of ops; lone shapes go in place.
    """
    by_key: dict[tuple[tuple[int, ...], torch.device, torch.dtype], list[Tensor]] = {}
    for p in params:
        g = p.grad
        if g is not None and g.ndim >= 2:
            key = (tuple(g.shape), g.device, g.dtype)
            by_key.setdefault(key, []).append(g)
    for grads in by_key.values():
        if len(grads) == 1:
            g = grads[0]
            g.sub_(g.mean(dim=tuple(range(1, g.ndim)), keepdim=True))
        else:
            # Stack so the per-param means are one reduction, centralize the stack, scatter
            # back. (A broadcasting ``_foreach_sub_`` falls to a slow path and is ~2x worse.)
            gs = torch.stack(grads)  # [N, *shape]; fan-in dims are 2..end (dim 1 = output row)
            gs.sub_(gs.mean(dim=tuple(range(2, gs.ndim)), keepdim=True))
            torch._foreach_copy_(grads, list(gs.unbind(0)))
