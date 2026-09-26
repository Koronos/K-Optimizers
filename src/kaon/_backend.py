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
import warnings
from typing import Any

import torch
from torch import Tensor

from kaon._compact_kahan import (
    RESIDUAL_KEY,
    compensated_add_,
    convert_residual,
    decode,
    init_residual,
    is_compact_kahan,
    residual_bits,
    residual_dtype,
)
from kaon._stochastic_rounding import SRStream, add_stochastic_

__all__ = [
    "BF16_METHODS",
    "DEFAULT_STACK_ELEMS",
    "FOREACH_BATCH_CUTOFF",
    "LOW_PRECISION",
    "MIN_STACK_ELEMS",
    "STACK_SAFETY_FRACTION",
    "SRSeedState",
    "cautious_batched_",
    "cautious_one_",
    "centralize_grads_",
    "ensure_residuals",
    "flat_view",
    "foreach_budget",
    "gc_applies",
    "init_bf16_state",
    "is_low_precision",
    "per_param_only_bf16_method",
    "residual_ok",
    "rms",
    "subtract_batched_",
    "subtract_one_",
    "validate_bf16_method",
    "weight_value",
]

LOW_PRECISION = (torch.bfloat16, torch.float16)

#: Every ``bf16_method`` a kaon optimizer accepts. ``"kahan"`` is the legacy bf16
#: compensation buffer (+2 B/param, per-param path only); ``"kahan8"`` the compact
#: fixed-point residual (+1 B/param, every path) and ``"kahan16"`` its 16-bit twin
#: (+2 B/param, every path: ``(bf16, residual)`` is bit for bit an fp32 master weight) —
#: see :mod:`kaon._compact_kahan`.
BF16_METHODS = ("stochastic_rounding", "kahan", "kahan8", "kahan16", "none")


def is_low_precision(t: Tensor) -> bool:
    return t.dtype in LOW_PRECISION


def validate_bf16_method(bf16_method: str) -> None:
    if bf16_method not in BF16_METHODS:
        raise ValueError(
            f"bf16_method must be one of {'/'.join(BF16_METHODS)}, got {bf16_method!r}"
        )


def per_param_only_bf16_method(bf16_method: str) -> bool:
    """Does this method exist only on the per-parameter writer?

    ``"kahan"`` keeps a bf16 ``state['shift']`` that only :func:`subtract_one_` knows how to
    carry, so every foreach/fused route has to reject it. ``"kahan8"`` / ``"kahan16"`` are
    NOT in this set: their residual rides the batched writer (``comp=``) and Adakaon's
    fused kernels.
    """
    return bf16_method == "kahan"


_LAZY_RESIDUAL_WARNED = False
_CONVERTED_RESIDUAL_WARNED = False


def ensure_residuals(params: list[Tensor], states: list[dict], bits: int = 8) -> bool:
    """Give every bf16 param in ``params`` a ``kahan_lo`` of the ``bits``-wide codec.

    Returns True if any residual was allocated or replaced (the caller then rebuilds
    whatever baked the old buffers). Two mid-run cases, both reached only when a group's
    ``bf16_method`` was changed AFTER the state existed (a scheduler/user reaching into the
    group dict):

    * **no residual** (switched on from SR / none / kahan): a zero residual is allocated —
      the weights start compensating from here, taken as exact, the same thing a fresh run
      does. Safe.
    * **a residual of the other width** (``kahan8 <-> kahan16``): it is CONVERTED, not
      dropped — :func:`kaon._compact_kahan.convert_residual` decodes the value with the
      codec that wrote it and re-encodes it at ``bits``. Widening is exact; narrowing is
      one round-half-away at the 8-bit grid (``<= ulp/512``). The new tensor replaces the
      old one in the state (a watched rebinding: every pointer table and plan rebuilds).

    Each case warns once, because silently accepting a mid-run switch is how a typo would
    hide. Every writer (per-param, foreach, fused) goes through this rather than
    substituting another buffer for the missing one or decoding a residual with the wrong
    width.
    """
    global _LAZY_RESIDUAL_WARNED, _CONVERTED_RESIDUAL_WARNED
    want = residual_dtype(bits)
    made = converted = False
    for p, st in zip(params, states, strict=True):
        if p.dtype != torch.bfloat16:
            continue
        lo = st.get(RESIDUAL_KEY)
        if lo is None:
            st[RESIDUAL_KEY] = init_residual(p, bits)
            made = True
        elif lo.dtype != want:
            st[RESIDUAL_KEY] = convert_residual(p.data, lo, bits)
            converted = True
    if made and not _LAZY_RESIDUAL_WARNED:
        _LAZY_RESIDUAL_WARNED = True
        warnings.warn(
            f"bf16_method='kahan{bits}' was enabled on a group whose parameters already had "
            "optimizer state: their compensation residual ('kahan_lo') starts at zero from "
            "this step (the weights are taken as exact). Set bf16_method at construction to "
            "avoid this.",
            stacklevel=3,
        )
    if converted and not _CONVERTED_RESIDUAL_WARNED:
        _CONVERTED_RESIDUAL_WARNED = True
        how = "exact" if bits == 16 else "one rounding at ulp/512"
        warnings.warn(
            f"bf16_method was switched to 'kahan{bits}' on a group whose compensation "
            f"residuals ('kahan_lo') were written by the other compact-Kahan width: they are "
            f"re-encoded at {bits} bits from this step ({how}).",
            stacklevel=3,
        )
    return made or converted


def residual_ok(state: dict, bits: int) -> bool:
    """Does ``state`` already hold a residual of the ``bits``-wide codec? (The per-param
    writers' fast check before :func:`ensure_residuals`.)"""
    lo = state.get(RESIDUAL_KEY)
    return lo is not None and lo.dtype == residual_dtype(bits)


@torch.no_grad()
def weight_value(p: Tensor, state: dict, bf16_method: str) -> Tensor:
    """The fp32 value of ``p`` an update term that READS the weight must use.

    Under ``kahan8`` / ``kahan16`` a bf16 weight is only the rounded half of the value the
    optimizer is tracking: the full value is ``(p, state['kahan_lo'])`` decoded
    (:func:`kaon._compact_kahan.decode`). Decoupled weight decay (``delta += wd * p``) reading
    the bare bf16 instead rounds ``p`` to the bf16 grid inside the update — at wd=0.1 that
    was the whole gap between a ``kahan16`` run and its fp32 twin (2.3 / 7.4 ulp at lr
    1e-4 / 3e-4, ``benchmarks/lowlr_bf16``). With the decoded value ``kahan16`` is the fp32
    run bit for bit and ``kahan8`` reads its ~1/256-ulp value.

    Every other case is the historical read: ``p.data`` itself for an fp32 param (an ALIAS —
    callers only read it) and ``p.data.float()`` otherwise, so SR / none / legacy ``kahan``
    and fp32 params are bit-identical to before. A missing / other-width residual (method
    switched mid-run) goes through :func:`ensure_residuals` first, the same lazy contract as
    the writer that follows.
    """
    if p.dtype == torch.float32:
        return p.data
    if p.dtype == torch.bfloat16 and is_compact_kahan(bf16_method):
        bits = residual_bits(bf16_method)
        if not residual_ok(state, bits):
            ensure_residuals([p], [state], bits)
        return decode(p.data, state[RESIDUAL_KEY], bits)
    return p.data.float()


def init_bf16_state(p: Tensor, state: dict, bf16_method: str) -> None:
    """Allocate the per-param compensation buffer ``bf16_method`` needs on a low-precision
    ``p`` (nothing for SR / none / fp32 params). One call in every optimizer's ``_init_state``."""
    if not is_low_precision(p):
        return
    if bf16_method == "kahan":
        state["shift"] = torch.zeros_like(p)
    elif is_compact_kahan(bf16_method):
        if p.dtype != torch.bfloat16:
            raise NotImplementedError(
                f"bf16_method={bf16_method!r} holds bf16 weights only (got {p.dtype}); "
                "use bf16_method='kahan' for fp16 parameters, or keep them in fp32"
            )
        state[RESIDUAL_KEY] = init_residual(p, residual_bits(bf16_method))


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

    ``sr`` is the calling optimizer's noise stream (see :class:`SRSeedState`). The stream
    holds a position per path and the write advances the one it actually took — the kernel's
    launch counter here, the torch path's generator inside ``add_stochastic_``; a write that
    takes Triton does not move the generator and vice versa. Both positions are
    checkpointed, so a run that crosses between the paths still resumes exactly. Leaving
    ``sr`` at ``None`` falls back to the process-wide stream and generator, whose positions
    no checkpoint saves.
    """
    use = SR_TRITON if triton is None else triton
    if use:
        from kaon import _fused_triton as ft
        if ft.sr_add_supported(target, source):
            ft.sr_add_(target, source, alpha, sr)
            return
    add_stochastic_(target, source, alpha=alpha, sr=sr)


def _ck_write_(
    target: Tensor,
    lo: Tensor,
    source: Tensor,
    alpha: float,
    bits: int,
    triton: bool | None = None,
    sr: SRStream | None = None,
    rows: bool = False,
) -> None:
    """``(target, lo) += alpha * source`` with the compact-Kahan codec, through Triton when
    it applies (CUDA, contiguous, bf16/residual/fp32) and the torch reference otherwise.

    ``rows`` marks a stacked foreach bucket: the torch path's ``kahan16`` add then runs per
    row (see :func:`kaon._compact_kahan.compensated_add_`); the Triton kernel is elementwise
    and layout-invariant already.

    ``lo`` must be stored in the ``bits``-wide codec's dtype (``uint8`` for 8, ``int16``
    for 16): a residual of the other width is REFUSED, never decoded with the wrong grid —
    the callers convert it first (:func:`ensure_residuals`).

    Same stream contract as :func:`_sr_write_`: the kernel takes a launch seed from ``sr``,
    the torch path draws its residual noise from ``sr``'s generator. Same caveat too — the
    two paths' noise does not reproduce each other, both are unbiased. ``SR_TRITON`` pins
    the torch path for both writers (it is the same A/B switch).
    """
    if lo.dtype != residual_dtype(bits):
        raise ValueError(
            f"compact-Kahan write with bits={bits} got a {lo.dtype} residual (written by the "
            "other width); convert it first with kaon._backend.ensure_residuals"
        )
    use = SR_TRITON if triton is None else triton
    if use:
        from kaon import _fused_triton as ft
        if ft.ck_add_supported(target, lo, source, bits):
            ft.ck_add_(target, lo, source, alpha, bits, sr)
            return
    compensated_add_(target, lo, source, alpha, bits, sr, rows=rows)


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

    The stream is created on first *access* and claims its noise identity only on its
    first real *draw*, so no optimizer has to grow a constructor line and one that never
    rounds (fp32 params, ``bf16_method="kahan"``, Adakaon's fused path) neither consumes a
    stream id nor grows an ``_sr_meta`` key.

    :attr:`sr_stream` is installed as a plain instance attribute by :meth:`__getattr__`
    rather than served by a ``property``, because the writers read it once per weight on
    the per-parameter path: a descriptor call measured ~260 ns against ~71 ns for an
    instance lookup — ~+0.11 ms/step on a 428-adapter bag, for nothing.
    """

    SR_META_KEY = "_sr_meta"

    def __getattr__(self, name: str) -> Any:
        # Only ever reached when normal attribute lookup FAILS, so the lazy install below
        # happens once and every later read is a plain ``__dict__`` hit.
        if name == "sr_stream":
            stream = SRStream()
            self.__dict__[name] = stream
            return stream
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def _sr_save(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Add the stream's position to ``state_dict`` (no-op if it never drew)."""
        stream = self.__dict__.get("sr_stream")
        if stream is not None:
            meta = stream.snapshot()
            if meta is not None:
                state_dict[self.SR_META_KEY] = meta
        return state_dict

    def _sr_load(self, state_dict: dict[str, Any]) -> None:
        """Adopt the checkpoint's stream position; a checkpoint without one resets.

        An optimizer with neither a saved position nor a live stream is left WITHOUT one:
        a load must not allocate a stream for an owner that never draws (the wrappers
        whose weight writes are all the inner optimizer's).
        """
        meta = state_dict.get(self.SR_META_KEY)
        if meta is None and self.__dict__.get("sr_stream") is None:
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

    ``kahan8`` decodes the compensated fp32 value ``z`` from ``(p, state['kahan_lo'])``,
    does ``z.add_(delta, alpha=-alpha)`` — the same ``alpha`` fold as the batched writer —
    and re-encodes with stochastic rounding at the residual grid
    (:func:`kaon._compact_kahan.compensated_add_`). Per-param and batched consume the
    same generator sequence, so the torch path keeps ``foreach == per-param`` for it.
    ``kahan16`` is the same codec at 16 bits: ``z`` is the exact fp32 master, the add is
    fp32's own round-to-nearest and the encode drops nothing (no noise is drawn), so the
    decoded weight is bit for bit what an fp32 parameter would hold.
    """
    low = is_low_precision(p)
    if low and is_compact_kahan(bf16_method):
        bits = residual_bits(bf16_method)
        if not residual_ok(state, bits):
            ensure_residuals([p], [state], bits)  # method switched mid-run: see ensure_residuals
        _ck_write_(p.data, state[RESIDUAL_KEY], delta_fp32, -alpha, bits, triton, sr)
    elif low and bf16_method == "kahan":
        shift = state.get("shift")
        if shift is None:
            # switched to legacy kahan after the state existed: start compensating from a
            # zero buffer (a bare KeyError otherwise), like ensure_residuals does for kahan8/16
            shift = state["shift"] = torch.zeros_like(p)
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
                      sr: SRStream | None = None,
                      comp: list[Tensor] | None = None) -> None:
    """In-place ``p -= alpha * delta`` over a foreach bucket of (matrixized) param views.

    ``pviews`` is the list of N same-shape param views (each ``[*shape]``); ``delta`` is
    the stacked fp32 step ``[N, *shape]`` (row i applies to ``pviews[i]``). ``comp`` is the
    matching list of ``state['kahan_lo']`` views for ``bf16_method="kahan8"`` / ``"kahan16"``
    (:attr:`kaon._foreach_plan.ForeachChunk.cviews`); a bf16 bucket under that method
    without it is refused rather than silently written uncompensated, and so is one whose
    residuals are of the other width (see :func:`_ck_write_`).

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
    if p0.dtype == torch.bfloat16 and is_compact_kahan(bf16_method):
        if comp is None:
            raise ValueError(
                f"subtract_batched_ with bf16_method={bf16_method!r} needs the bucket's "
                "residual views (comp=...); the caller must route this method per-param "
                "or hand over ForeachChunk.cviews"
            )
        bits = residual_bits(bf16_method)
        want = residual_dtype(bits)
        # EVERY view, not the stack: torch.stack promotes a uint8 + int16 mix to int16, which
        # _ck_write_ would then accept and decode the uint8 rows on the 16-bit grid.
        if any(c.dtype != want for c in comp):
            raise ValueError(
                f"subtract_batched_ with bf16_method={bf16_method!r} got residuals of the "
                f"other width ({sorted({str(c.dtype) for c in comp})}, want {want}); convert "
                "them first with kaon._backend.ensure_residuals"
            )
        weights = torch.stack(pviews)
        lows = torch.stack(comp)
        _ck_write_(weights, lows, delta, -alpha, bits, triton, sr, rows=True)
        torch._foreach_copy_(pviews, list(weights.unbind(0)))
        torch._foreach_copy_(comp, list(lows.unbind(0)))
    elif p0.dtype == torch.bfloat16 and bf16_method == "stochastic_rounding":
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
def gc_applies(shape: torch.Size | tuple[int, ...]) -> bool:
    """Whether Gradient Centralization is DEFINED for a weight of this shape.

    **The one definition of the predicate.** GC subtracts, per output row, the gradient's
    mean over the fan-in dims (every dim but dim 0). That needs two things:

    * ``ndim >= 2`` — a 1-D bias or norm scale has no fan-in dims at all (unchanged
      behaviour: GC always skipped these);
    * ``fan_in > 1`` — over a row of ONE element the mean *is* the element, so
      ``g - mean(g)`` is identically zero. That is not centralization, it is erasure: it
      destroys the update signal and freezes the parameter. Shapes that hit it are
      ordinary — a rank-1 LoRA up-projection ``(out, 1)``, a one-input 1x1 conv
      ``(out, 1, 1, 1)``, some projections — so GC is skipped there (0.7.13; before that
      those params silently never moved). ``fan_in == 0`` (an empty weight) falls on the
      skip side of the very same comparison, which is all that is claimed for it: the
      pre-fix code was already harmless there, because the NaN mean it computed was
      subtracted into a zero-element destination and wrote nothing.

    GC is implemented at nine host sites and sixteen Triton kernels, and a skip applied at
    fewer than all of them makes the fused and native routes disagree. So every one of them
    resolves the flag through THIS function — the torch sites call it directly, the Triton
    ones get it per shape/tile bucket as their ``GC`` ``tl.constexpr`` (see
    :func:`kaon._fused_triton.bucket_gc_ok` and ``docs/FUSED_REDUCTIONS_DESIGN.md``).
    It is called once per BUCKET, never per parameter per step: the callers evaluate it
    where the plan/pointer cache is built, so a steady-state step pays nothing for it.
    """
    return len(shape) >= 2 and math.prod(shape[1:]) > 1


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

    Weights GC is not defined for are skipped — see :func:`gc_applies`. The skip is applied to
    the BUCKET, not to each parameter, for two reasons: the bucket key already carries the shape
    so the predicate is a pure function of it (one call per distinct shape instead of one per
    param per step — ~4 calls instead of 428 on the LoRA bag), and dropping a whole shape cannot
    perturb any other bucket, which is what keeps fan-in >= 2 bit-identical.
    """
    by_key: dict[tuple[tuple[int, ...], torch.device, torch.dtype], list[Tensor]] = {}
    for p in params:
        g = p.grad
        if g is not None and g.ndim >= 2:
            key = (tuple(g.shape), g.device, g.dtype)
            by_key.setdefault(key, []).append(g)
    for (shape, _dev, _dtype), grads in by_key.items():
        if not gc_applies(shape):
            continue
        if len(grads) == 1:
            g = grads[0]
            g.sub_(g.mean(dim=tuple(range(1, g.ndim)), keepdim=True))
        else:
            # Stack so the per-param means are one reduction, centralize the stack, scatter
            # back. (A broadcasting ``_foreach_sub_`` falls to a slow path and is ~2x worse.)
            gs = torch.stack(grads)  # [N, *shape]; fan-in dims are 2..end (dim 1 = output row)
            gs.sub_(gs.mean(dim=tuple(range(2, gs.ndim)), keepdim=True))
            torch._foreach_copy_(grads, list(gs.unbind(0)))
