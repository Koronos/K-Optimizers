"""Adakaon — a conv-aware factored optimizer aimed at AdamW quality at
Adafactor memory, for bf16 diffusion fine-tuning.

Adakaon is the flagship of the library and the optimizer that most fully exercises
the **kaon** shared backend (factored second moment, quantized momentum codec,
stochastic rounding, foreach, cautious) — hence the name. Every other optimizer in
the package reuses pieces of Adakaon's machinery. (Formerly named *Adafusion*.)

Design (validated by benchmarks/bench_convergence-style experiments):

* **Conv-aware factored second moment.** Like Adafactor/Compactor, the second
  moment of a 2-D weight is factored into row+column EMAs (≈0 state). The fix
  over Compactor/HF-Adafactor: a 4-D conv kernel ``[out,in,kh,kw]`` is first
  **reshaped to ``[out, in·kh·kw]``** and factored over *that* matrix — instead
  of factoring the tiny spatial dims, which barely compresses a 3×3 kernel and
  was the entire optimizer-state floor on a diffusion UNet (≈26× more conv state
  for no quality gain).
* **Optional momentum in bf16.** A first-moment buffer recovers AdamW-level
  convergence; kept in bf16 it costs ~2 B/param (half of fp32 momentum) with no
  measured quality loss → AdamW-quality at ~1/4 of AdamW's optimizer memory.
* **bf16-correct weight updates** via stochastic rounding (no extra state) or
  Kahan summation.
* **Cautious masking** (Liang et al. 2024): zero the update coordinates whose
  sign disagrees with the gradient, renormalized to keep the step size. **On by
  default** — measured ~1.4% lower held-out val loss with momentum (paired
  t=-4.07); a literal no-op without momentum. Set ``cautious=False`` for
  no-momentum configs.

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

from kaon._autolr import DEFAULT_FUSE_REL, AutoLRMixin
from kaon._backend import (
    FOREACH_BATCH_CUTOFF,
    cautious_batched_,
    cautious_one_,
    centralize_grads_,
    flat_view,
    foreach_budget,
    is_low_precision,
    rms,
    subtract_batched_,
    subtract_one_,
)
from kaon._factored import factored_inv_sqrt_factors, update_factored_state
from kaon._momentum_codec import (
    _FOURBIT_BLOCK,
    _dequant_4bit,
    _dequant_4bit_stacked,
    _FloatCodec,
    _FourBitCodec,
    _Int8Codec,
    _make_codec,
    _MomentumCodec,
    _pack_nibbles,
    _quant_4bit,
    _quant_4bit_stacked,
    _quant_int8,
    _quant_int8_stacked,
    _unpack_nibbles,
    load_state_dict_preserving_dtypes,
    warn_if_4bit_high_beta1,
)

__all__ = ["Adakaon"]

# Re-exported codec internals (kept importable from ``kaon.adakaon`` for
# backwards compatibility with existing tests/benchmarks). The implementations
# now live in :mod:`kaon._momentum_codec`, shared with KProdigy.
_ = (
    _dequant_4bit, _dequant_4bit_stacked, _FloatCodec, _FourBitCodec, _Int8Codec,
    _MomentumCodec, _pack_nibbles, _quant_4bit, _quant_4bit_stacked, _quant_int8,
    _quant_int8_stacked, _unpack_nibbles,
)

MomentumDtype = Literal["bfloat16", "float32", "int8", "4bit"]
CautiousWD = Literal["masked", "full"]

# Diagnostic kernel-routing override (env-gated, zero cost when unset): a comma list of
# fused subsets to force onto the native path — e.g. KAON_FUSED_DISABLE="one_block" or
# "big,one_dim". Used to bisect which Triton kernel a real-training divergence lives in
# (the 2026-06-10 Nekaon NaN hunt); harmless to leave in.
import os as _os  # noqa: E402

_FUSED_DISABLE = frozenset(
    s.strip() for s in _os.environ.get("KAON_FUSED_DISABLE", "").split(",") if s.strip()
)
_PROBE_LOG_PATH = _os.environ.get("KAON_PROBE_LOG")

# Stacking a foreach bucket allocates several transient copies of the stacked
# tensor (grad fp32, the reconstruction, the SR intermediate, ...), so an unbounded
# bucket of large weights can OOM a full fine-tune — which would undercut
# Adakaon's whole memory story. We therefore cap the per-chunk element count and
# split bigger buckets. The cap is **adaptive to free VRAM** rather than a fixed
# constant: a card with lots of headroom batches whole buckets (and even stacks
# large weights), while a constrained card shrinks the chunk and stays safe. The
# budget is `free_bytes * SAFETY_FRACTION / BYTES_PER_ELEM`; the divisor accounts
# for the ~handful of simultaneous transient copies a chunk touches at peak.
# Peak transient bytes per stacked element. This is a property of the optimizer's
# intermediate tensors, NOT of the model: measured byte-for-byte identical on SDXL
# and Cosmos shapes. It depends only on the path and config — 2-D factored 24 B
# (common) / 38 B (momentum+wd+cautious), 1-D non-factored 28 B / 42 B (it also
# stacks the full second-moment state). 48 = worst measured (42.1) + margin.
_STACK_BYTES_PER_ELEM = 48

# Per-tensor element count above which a weight is stepped by the per-parameter
# loop instead of being stacked. This is a PERFORMANCE threshold, deliberately
# decoupled from the VRAM-safety budget: batching pays off only while per-tensor
# kernel-launch overhead dominates (small tensors); a large weight's update is
# compute/bandwidth-bound, so stacking it just adds copy traffic and is slower.
# A budget sweep on SDXL and Cosmos full fine-tunes showed a broad flat optimum
# for cutoffs of ~0.1-4 M elements and a sharp slowdown beyond ~4 M, on both
# models — i.e. the crossover is an absolute element count, NOT a fraction of
# VRAM (so it must not scale with the card). 2 M sits in the middle of that
# plateau. See docs/foreach-batching.md.


def _identity(t: Tensor) -> Tensor:
    """``mat`` for buckets whose effective layout *is* the tensor (no view needed)."""
    return t


# Unbound, so the per-step staleness guard is a C-level ``map()`` instead of a genexpr.
_DATA_PTR = Tensor.data_ptr
_IS_CONTIG = Tensor.is_contiguous


class _ForeachChunk:
    """Cached derived views for ONE stacked chunk of the native foreach path.

    Rebuilding these lists every step is what made a bag of 0-D scalars slow: with 448
    params the non-factored bucket calls :func:`~kaon._backend.flat_view` ~5x per param per
    step (the ``v`` stack, the codec's ``mat`` callback, the weight-decay stack, the final
    subtract) and every one of those calls on a 0-D tensor materializes a fresh
    ``aten::view`` TensorImpl. They are all pure functions of tensors the optimizer
    **already owns** — the params and their ``self.state`` buffers — so caching them pins
    no memory at all.

    Gradient views are deliberately **not** cached: a retained view of ``p.grad`` keeps the
    previous step's gradient storage alive (``set_to_none=True`` allocates a fresh grad
    every backward), which would add a whole gradient set to peak memory — an unacceptable
    trade for an optimizer whose entire pitch is memory. :meth:`grad_stack` avoids them a
    different way instead: ``torch.stack`` always writes a contiguous output, so stacking
    the **raw** grads and reshaping the *stack* once is element-for-element the same buffer
    as stacking N per-param reshapes — one ``view`` per step instead of N. That covers every
    bucket whose params share an ``ndim`` (all convs, all 1-D, all 0-D); only a bucket that
    genuinely mixes 0-D with shape-``(1,)`` params still builds per-param views.

    Staleness is the caller's job (:meth:`Adakaon._foreach_plan`): the plan is rebuilt when
    the group's param identities or ``p.data`` pointers change, and dropped outright by
    :meth:`Adakaon._invalidate_fused_caches` (state reset / checkpoint load) and by any step
    that falls back to the per-parameter path.
    """

    __slots__ = ("cols", "eff", "grad_reshape", "grad_uniform", "length", "mat", "plist",
                 "pviews", "rows", "states", "view", "vs")

    def __init__(
        self,
        plist: list[Tensor],
        states: list[dict[str, Any]],
        group: dict[str, Any],
        eff: tuple[int, int] | None,
        matrixize: bool,
        length: int,
        cached: bool = True,
    ) -> None:
        self.plist, self.states, self.eff, self.length = plist, states, eff, length
        n = len(plist)
        # ``grad_uniform``: the bucket's grads all share an ndim, so they stack raw and the
        # STACK gets reshaped (``grad_reshape``, ``None`` when it is already [N, *eff]).
        self.grad_uniform = cached
        self.grad_reshape: tuple[int, ...] | None = None
        if eff is not None:                                   # factored bucket
            R, C = eff  # noqa: N806 — matrix dims (stacked tensor is [N, R, C])
            self.view = (lambda t: t.view(R, C)) if matrixize else _identity
            self.rows = [s["row"] for s in states]
            self.cols = [s["col"] for s in states]
            self.vs = None
            if matrixize:                                     # conv [N,O,I,kh,kw] -> [N,R,C]
                self.grad_reshape = (n, R, C)
        else:                                                 # non-factored bucket
            self.rows = self.cols = None
            ndims = {p.ndim for p in plist}
            self.view = _identity if ndims == {1} else flat_view
            self.grad_uniform = cached and len(ndims) == 1
            if ndims == {0}:                                  # 0-D bag: [N] -> [N, 1]
                self.grad_reshape = (n, 1)
            self.vs = [self.view(s["v"]) for s in states]
        self.pviews = [self.view(p.data) for p in plist]
        # The codec calls ``mat(state["m"])`` once (float) or twice (int8) per param per
        # step; when that call would build a view, an identity-keyed lookup of prebuilt
        # views is cheaper. Only then — ``Tensor.__hash__`` is a Python-level call in
        # torch, so a dict hit is *more* expensive than ``_identity`` when there is no view
        # to save. 4bit is excluded too: its ``m`` is a packed byte string, not a momentum
        # in the effective layout (``view`` would raise), and its codec never calls ``mat``.
        self.mat = self.view
        if (cached and self.view is not _identity and group["betas"][0] > 0
                and group["momentum_dtype"] != "4bit"):
            self.mat = {s["m"]: self.view(s["m"]) for s in states}.__getitem__

    # ponytail: known ceiling. Profiling a 448x 0-D bag after this cache shows the residue is
    # three ``stack`` + three ``unbind`` per step (~1344 ``aten::select``). Only ONE of each
    # pair lives here (the ``v`` stack and its write-back); the other two are inside
    # ``_momentum_codec.ema_stacked`` and ``_backend.subtract_batched_``, which this change is
    # scoped out of. Persistent ``cat(out=)`` scratch buffers with cached unbind slices would
    # therefore buy only the local pair (~10% of the remaining step) while pinning a stacked
    # fp32 buffer per bucket for the process's lifetime — a bad trade for an optimizer whose
    # pitch is memory, and it would feed back into the free-VRAM-adaptive chunk budget.
    # Removing the other two needs the codec to hand back a reusable buffer: follow-up work.
    def grad_stack(self) -> Tensor:
        """This step's stacked fp32 gradient ``[N, *eff]``. Never cached — see the class
        docstring; the reshape-the-stack trick keeps it to one ``view`` per bucket."""
        if self.grad_uniform:
            g = torch.stack([p.grad for p in self.plist])
            if self.grad_reshape is not None:
                g = g.view(self.grad_reshape)
            return g.float()
        view = self.view
        return torch.stack([view(p.grad) for p in self.plist]).float()


class _ForeachPlan:
    """One param group's foreach bucketing plus the per-chunk view caches.

    ``buckets`` preserves the order the uncached code stepped in (factored buckets first,
    then non-factored, each in first-seen param order). ``chunks`` is that bucketing split
    by the current memory budget; the split is re-derived only when a bucket's chunk length
    actually changes, because the adaptive VRAM budget wobbles every step while
    ``budget // length`` almost never does.

    ``cached=False`` (the ``_foreach_cache_enabled`` A/B switch) makes every chunk rebuild
    its lists per param per step exactly as the pre-cache code did, so the two arms differ
    only in this optimization and nothing else.
    """

    __slots__ = ("buckets", "chunks", "steps", "witness")

    def __init__(self, witness: tuple, buckets: list[tuple]) -> None:
        self.witness, self.buckets = witness, buckets
        self.chunks: list[_ForeachChunk] | None = None
        self.steps: tuple[int, ...] = ()

    def rechunk(self, budget: int, group: dict[str, Any],
                cached: bool = True) -> list[_ForeachChunk]:
        steps = tuple(max(1, budget // size) for size, *_ in self.buckets)
        if self.chunks is not None and steps == self.steps:
            return self.chunks
        self.steps = steps
        self.chunks = [
            _ForeachChunk(plist[i:i + n], states[i:i + n], group, eff, matrixize, length, cached)
            for n, (_size, plist, states, eff, matrixize, length) in zip(steps, self.buckets, strict=True)
            for i in range(0, len(plist), n)
        ]
        return self.chunks


def _param_witness(plist: list[Tensor]) -> tuple:
    """Per-step staleness witness for the NATIVE foreach plan over ``plist``: ``(ids, data_ptrs,
    contiguity)``, one flat tuple each.

    Deliberately duplicated from :func:`kaon._fused_triton.param_witness` rather than imported:
    this one guards the native plan, which has to work in a build without Triton, and
    ``kaon.adakaon`` never imports the Triton module at module scope.

    Each field is load-bearing and none implies another. ``id`` catches a changed param set.
    ``data_ptr`` catches ``p.data = <fresh storage>`` — an external EMA, a ``.to(dtype/device)``,
    Rengu-Flow's block-swap offloader — and, through it, every dtype and device change.
    Contiguity catches ``p.data = p.data.t()``, which on a SQUARE weight keeps id, pointer AND
    shape and moves only the strides, while every kernel indexes row-major off ``data_ptr``.

    THIS ONE IS FIXED AT THREE FIELDS; the fused one takes an optional fourth. A shape-changing
    rebind (``p.data = p.data.view(64, 16)``) is invisible to all three fields above, and the two
    paths are in very different positions about that. The native plan re-stacks by effective shape
    every step, so the rebind raises a size mismatch out of ``torch.stack`` on the very next step
    and no witness field would add anything — only cost. The FUSED path freezes its whole geometry
    at plan-build time (the cached partition, ``Rs``/``Cs``, the row/col pointer arrays), so it
    needs to be told; that is ``kaon._fused_triton.SHAPE_WITNESS``, opt-in, priced on that flag.
    :meth:`_fused_partition` therefore calls ``ft.param_witness`` and :meth:`_foreach_plan` calls
    this one — which also keeps the fused key EXACTLY as strong as the pointer caches' own
    witness, and that is what makes their O(1) ``_WitnessedCache.built_from`` revalidation sound.

    LIMIT — a shape-changing rebind is REFUSED, never supported. ``state["row"]``/``state["col"]``
    were allocated for the old effective 2-D shape and an EMA has no meaningful migration onto a
    different factorization. What 0.7.13 guarantees, at no per-step cost and with no flag, is that
    the fused path cannot step a plan that disagrees with the state: every pointer cache validates
    the state geometry at build (:func:`kaon._fused_triton.check_state_geometry`), so any rebind
    that moves the witness at all — a fresh buffer, a dtype or device change, a transpose, or a
    reshape bundled with any of those — raises with an actionable message instead of pointing the
    new ``R``/``C`` at the old buffers and reading past them (which is what main did: measured,
    silent, on ``p.data = p.data.view(64, 16).clone()``). The recovery is to reshape before
    constructing the optimizer, or ``del opt.state[p]`` to restart that parameter's second moment.

    Left over, and the reason ``SHAPE_WITNESS`` exists: a rebind that changes the shape and
    NOTHING else moves no field, so the plan is never rebuilt and never revalidated. That weight
    keeps being stepped as the shape it used to have. For a plain ``view`` it stays in bounds (the
    storage is the same size), so it degrades quality rather than corrupting memory, which is why
    detecting it is opt-in and closing the corruption paths is not. A NARROWING rebind
    (``p.data[:8]``) is the sharper version and the flag does not see it either — strides do not
    move when only ``R`` shrinks; see ``SHAPE_WITNESS`` for what that costs a sibling view of the
    same storage.

    COST on the 428-param LoRA-shaped bag (200x(256,256) + 100x(512,) + 128x 0-D),
    ``benchmarks/fused/bench_shape_witness.py``, paired against the 3-field base: 67 µs for the
    three fields here (median of 15x250 reps, build + compare). The witness runs once per group in
    :meth:`_fused_partition` (the route caches then revalidate by list identity) plus once per big
    shape bucket — 2 calls per step on that bag, 5 on a UNet-shaped one — and the per-step grad
    contiguity sweep in :meth:`_fused_demote` adds ~75 µs on the same bag. ``SHAPE_WITNESS``
    itself is priced on the flag, per bag.
    """
    return (tuple(map(id, plist)), tuple(map(_DATA_PTR, plist)), tuple(map(_IS_CONTIG, plist)))


def _same_shape_device_buckets(plist: list[Tensor]) -> dict[tuple, list[Tensor]]:
    """Group big 2-D tensors for the batched chunked kernel: EXACT shape, dtype AND device.

    A bucket is launched as one grid against pointer arrays that :class:`BigPointerCache` builds
    on ``plist[0].device``; two CUDA devices sharing a shape would run the second one's tensors
    against index tensors from the first. The device belongs in the grouping, not only in the
    cache key that is derived from it.
    """
    by_shape: dict[tuple, list[Tensor]] = {}
    for p in plist:
        by_shape.setdefault((tuple(p.shape), p.dtype, p.device), []).append(p)
    return by_shape


def _demote_non_contiguous_grads(
    one_block: list[Tensor], big: list[Tensor], one_dim: list[Tensor], native: list[Tensor],
) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
    """Move any param whose GRAD is not contiguous onto the native subset FOR THIS STEP.

    Every fused kernel addresses the gradient as ``base + ri*C + ci`` (or ``base + offs``) read
    straight off ``grad.data_ptr()``. A transposed (``grad = x.t()``) or strided
    (``grad = buf[::2]``) gradient has the right shape and the wrong layout, so the kernel
    silently steps the wrong numbers — measured as a ~1e-3 relative divergence from native with
    no error raised anywhere. ``fused_eligible`` / ``fused_1d_eligible`` check
    ``p.is_contiguous()``, never the grad's.

    Contiguity belongs to THIS step's gradient (a fresh tensor every backward), so it cannot be
    frozen into the cached routing partition: this runs per step, returns new lists and leaves
    the cache untouched. Only the offending tensors move — one strided grad on one adapter must
    not cost the rest of its bucket the fused path.
    """
    fused = (one_block, big, one_dim)
    if all(p.grad.is_contiguous() for sub in fused for p in sub):
        return one_block, big, one_dim, native
    kept: tuple[list[Tensor], ...] = ([], [], [])
    demoted: list[Tensor] = []
    for keep, sub in zip(kept, fused, strict=True):
        for p in sub:
            (keep if p.grad.is_contiguous() else demoted).append(p)
    return kept[0], kept[1], kept[2], native + demoted


class Adakaon(AutoLRMixin, Optimizer):
    """Conv-aware factored optimizer with optional bf16 momentum.

    Args:
        params: parameters or param-group dicts.
        lr: learning rate.
        betas: ``(beta1, beta2)``. ``beta1=0`` disables momentum (minimum memory,
            Adafactor-like). ``beta1>0`` enables momentum (AdamW-like quality).
        eps: ``(eps1, eps2)``. ``eps1`` is added to ``grad**2`` before the
            factored reductions (HF Adafactor convention). ``eps2`` is currently
            unused (reserved).
        weight_decay: decoupled weight decay (folded into the per-step delta; see
            ``cautious_wd`` for its placement relative to the cautious mask).
        clip_threshold: Adafactor RMS update clipping (``rms(update) <= thr``).
        momentum_dtype: storage for the first-moment buffer when ``beta1>0`` —
            ``"bfloat16"`` (default; ~2 B/param), ``"float32"`` (4 B/param),
            ``"int8"`` (~1 B/param, per-row absmax quantized; Lion8bit-class
            memory but with the factored adaptive second moment), or ``"4bit"``
            (~0.5 B/param: signed linear 4-bit, two nibbles per byte, with a
            per-block absmax scale — block size ``momentum_4bit_block``). On real
            SDXL gradients block-128 4-bit matched int8's delta cosine vs fp32.
        momentum_4bit_block: block size (consecutive flattened elements sharing one
            absmax scale) for ``momentum_dtype="4bit"``. Default ``128``. Smaller
            blocks raise fidelity at the cost of more scale bytes
            (``4/block`` B/param); ``128`` adds ~0.03 B/param for a ~0.53 B/param
            total. ``0``/negative means whole-tensor (single scale).
        cautious: cautious masking (Liang et al. 2024) — zero the update
            coordinates whose sign disagrees with the gradient. **On by default**:
            it improves convergence when momentum is on (``beta1>0``) and is a
            literal no-op without momentum (the mask is all-ones — verified). Turn
            it off for no-momentum configs to skip the then-useless masking op.
        cautious_wd: where decoupled ``weight_decay`` sits relative to the cautious
            mask. ``"masked"`` (default, historical behaviour) folds it into the
            delta BEFORE the mask, so a coordinate the mask rejects gets **no**
            decay at all and a survivor gets it multiplied by the survivor rescale
            ``1/keep``. Measured (fraction of the REQUESTED ``lr*wd*p`` that each
            coordinate actually receives): at ``keep=0.64``, 1.49x on survivors
            and 0.005x on rejected coordinates; at ``keep=0.50``, 1.985x and
            0.0013x. The aggregate shrinkage is preserved; its per-coordinate
            distribution is not.
            ``"full"`` applies ``lr*wd*p`` to **every** coordinate, outside the
            mask, and masks only the momentum/update term — which is what the
            Cautious Optimizers paper (Liang et al. 2024) does. With
            ``cautious=False`` the mask is a no-op and the two orders are the same
            add, so the modes coincide — bit-identically wherever the path itself is
            deterministic (the fused big bucket's fp32-atomic reductions are
            run-to-run nondeterministic unless ``deterministic_reductions=True``).
            No effect when ``weight_decay == 0``.
        bf16_method: weight-update strategy for low-precision params —
            ``"stochastic_rounding"`` (default), ``"kahan"`` (+2 B/param), or
            ``"none"``. No-op on fp32 params.
        foreach: batch the step across parameters with multi-tensor (stacked) ops
            instead of a per-parameter Python loop. Default ``True``. Huge win when
            many tensors are stepped at once — LoRA/LoKr adapters (hundreds of tiny
            2-D tensors) *and* full fine-tunes (thousands of weights incl. all the
            1-D biases/norms). Params are bucketed by shape and each bucket steps
            as a few stacked kernels: ``ndim >= 2`` factored ``[N, R, C]``, ``ndim
            <= 1`` non-factored ``[N, L]`` (0-D scalars — e.g. LyCORIS
            ``use_scalar`` gates — ride that bucket as length-1 rows). Matches the
            per-parameter path numerically (stochastic-rounding draws differ,
            unbiased either way); int8 momentum is also batched (per-row absmax
            dequant/EMA/requant on the stacked layout), as is 4bit (per-block
            absmax, packed nibbles). The rest (kahan, fp16+SR,
            non-contiguous matrixized convs, single-param groups) transparently
            falls back to it. Set ``False`` to force the per-parameter path.
        foreach_batch_cutoff: per-tensor element count above which a weight is
            stepped by the per-parameter loop instead of being stacked. A
            **performance** knob, decoupled from VRAM: batching only pays off while
            launch overhead dominates (small tensors), so large weights loop. The
            default ``2_000_000`` is the middle of a flat optimum measured on SDXL
            and Cosmos full fine-tunes; raise it only if profiling your GPU shows a
            higher crossover. See ``docs/foreach-batching.md``.
        deterministic_reductions: make the fused big-tensor path bit-reproducible
            across runs. The batched reductions accumulate the column sums and the
            RMS with **fp32 atomics**, whose completion order the scheduler picks, so
            the same inputs give slightly different results on every run: measured
            spread over 4 runs of 6 steps on 3x(512,512), max|Δp| / weight scale,
            5.1e-8 (fp32 momentum), 3.9e-6 (bf16), 8.1e-6 (int8) and **7.0e-4**
            (4bit — one ulp on the momentum can flip an adjacent 4-bit code). With
            this on, each row-block writes a partial only it owns and a second pass
            sums the partials in a fixed order — same arithmetic, one order. Costs an
            ``N * ceil(R/BR) * C`` fp32 buffer per big bucket (7.7 MB for a
            236x(512,512) LoKr bucket) plus one extra launch per bucket per
            reduction. Default ``False``: the nondeterminism is at or below the
            dtype's own noise for every momentum kind except 4bit, and the flag is
            not free. Turn it on for run-to-run reproducibility work, to debug a
            divergence, or with ``momentum_dtype="4bit"`` where the amplification is
            real. Only affects ``fused=True`` big buckets; nothing else in kaon uses
            a float atomic.
        foreach_stack_budget: the **memory-safety** ceiling — max elements in a
            single stacked ``foreach`` chunk. ``None`` (default) adapts to
            currently-free VRAM each step (roomy card → bigger chunks, full card →
            smaller, OOM-safe). Pass an int to pin a fixed cap (reproducibility, or
            a hard ceiling on a shared GPU). Decoupled from ``foreach_batch_cutoff``
            so raising it never pulls large weights into stacking.

    Note: Adakaon deliberately exposes **no** ``compile`` flag. A whole-step
    ``torch.compile`` was measured ~neutral on most shapes here and a slight loss on
    trivial steps (Adakaon's step has little fusable elementwise math), so it is
    not worth the API surface — Adakaon stays lean. The flag lives on
    :class:`~kaon.adamuon.AdaMuon`, whose heavy Newton-Schulz math it actually
    speeds up.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: tuple[float, float] = (1e-30, 1e-3),
        weight_decay: float = 0.0,
        *,
        clip_threshold: float = 1.0,
        momentum_dtype: MomentumDtype = "bfloat16",
        momentum_4bit_block: int = _FOURBIT_BLOCK,
        cautious: bool = True,
        cautious_wd: CautiousWD = "masked",
        gradient_centralization: bool = True,
        bf16_method: str = "stochastic_rounding",
        foreach: bool = True,
        foreach_batch_cutoff: int = FOREACH_BATCH_CUTOFF,
        foreach_stack_budget: int | None = None,
        fused: bool = False,
        fused_tile_cap: int | None = None,
        deterministic_reductions: bool = False,
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
        if clip_threshold <= 0.0:
            raise ValueError(f"clip_threshold must be > 0, got {clip_threshold}")
        if momentum_dtype not in ("bfloat16", "float32", "int8", "4bit"):
            raise ValueError(
                f"momentum_dtype must be bfloat16/float32/int8/4bit, got {momentum_dtype!r}"
            )
        # One shared helper, not a per-optimizer copy: 4-bit momentum under a high beta1 has the
        # dequant->EMA->requant loop amplify its own quantization error ~1/sqrt(1-beta1^2). Same
        # one-liner as lion.py / adabelief.py / adamuon.py / ... — see docs/momentum.md.
        warn_if_4bit_high_beta1(beta1, momentum_dtype)
        if cautious_wd not in ("masked", "full"):
            raise ValueError(f"cautious_wd must be 'masked' or 'full', got {cautious_wd!r}")
        if bf16_method not in ("stochastic_rounding", "kahan", "none"):
            raise ValueError(f"bf16_method must be stochastic_rounding/kahan/none, got {bf16_method!r}")
        if foreach_batch_cutoff < 1:
            raise ValueError(f"foreach_batch_cutoff must be >= 1, got {foreach_batch_cutoff}")
        defaults = {
            "lr": lr,
            "betas": (beta1, beta2),
            "eps": (float(eps[0]), float(eps[1])),
            "weight_decay": weight_decay,
            "clip_threshold": clip_threshold,
            "momentum_dtype": momentum_dtype,
            "momentum_4bit_block": momentum_4bit_block,
            "cautious": cautious,
            "cautious_wd": cautious_wd,
            "gradient_centralization": gradient_centralization,
            "bf16_method": bf16_method,
        }
        super().__init__(params, defaults)
        # Multi-tensor (foreach) batching of the factored fast path. Collapses the
        # per-parameter Python loop + per-tensor kernel launches into a handful of
        # stacked-tensor ops per (shape, dtype) bucket — the decisive win when many
        # small weights are trained at once (LoRA/LoKr adapters). Numerically
        # matches the per-parameter path; stochastic-rounding draws differ
        # (unbiased either way). Anything it doesn't cover falls back per-param.
        self._foreach = foreach
        # Performance cutoff: weights larger than this loop instead of stacking
        # (batching only helps while launch overhead dominates). Decoupled from
        # the VRAM-safety chunk budget below.
        self._foreach_batch_cutoff = foreach_batch_cutoff
        # Memory-safety ceiling: max elements per stacked chunk. None -> adaptive
        # to free VRAM (see _foreach_budget); an int forces a fixed cap.
        self._foreach_stack_budget = foreach_stack_budget
        # One momentum codec per dtype string (the codec is stateless beyond the
        # dtype). Encapsulates every dequant→EMA→requant detail so the three step
        # functions stay dtype-agnostic.
        self._codecs: dict[str, _MomentumCodec] = {}
        # Optional Triton-fused step (same math + state, faster on GPU). Eligible 2-D weights run
        # through the fused kernels (one-block tile / chunked big-tensor); everything else falls back
        # to the native path below, in-place on the SAME state, so fused/non-fused interoperate and
        # resume from each other's checkpoints.
        self._fused = bool(fused)
        # When True, the many-same-shape big regime (>tile_cap) runs the batched chunked kernel; set
        # False to revert to the batched-native-foreach path (the A/B baseline). See _fused_big.
        self._fused_big_batched = True
        # Batched-big reductions stay in Triton (grad via pointer arrays, no [N,R,C] stack,
        # GC in-kernel). The toggle remains internal for parity/performance A/B tests.
        self._fused_reductions = True
        # Two-pass (partials -> reduce) colsum/rms instead of fp32 atomics: same arithmetic in
        # a FIXED order, so a run reproduces itself bit for bit. Costs an N*RB*C fp32 partial
        # buffer and one extra launch per bucket per reduction. Default OFF - see the kwarg.
        self._deterministic_reductions = bool(deterministic_reductions)
        # A LONE big tensor also takes the batched (N=1) chunked path. The per-tensor
        # ``_chunked_step`` it replaces costs two CPU<->GPU syncs per tensor per step
        # (``float(rms)`` and ``keep.item()``) and materializes fp32 g and g² — see
        # _fused_big. False reverts to the per-tensor kernel (the A/B baseline).
        self._fused_big_lone_batched = True
        # The momentum buffer stores an LR-INDEPENDENT direction (lr scales the
        # final delta, it is never folded into the EMA). Wrappers that convert
        # momentum back to step units (MSAM/Nekaon's raw-momentum lookahead) key
        # off this marker; optimizers without it keep lr-scaled momentum.
        self._momentum_is_unscaled = True
        self._t = 0
        self._fused_part: dict[int, tuple] = {}          # group id -> (witness, one_block, big, one_dim, native)
        self._fused_demoted: dict[int, tuple] = {}       # group id -> memo of the non-contiguous-grad demotion
        self._fused_ob_caches: dict[int, Any] = {}       # group id -> PointerArrayCache (one-block)
        self._fused_od_caches: dict[int, Any] = {}       # group id -> OneDimPointerCache (1-D)
        self._fused_big_caches: dict[tuple[int, tuple[int, ...], Any], Any] = {}
        # Native foreach path: group id -> _ForeachPlan (bucketing + per-chunk views). Set
        # ``_foreach_cache_enabled = False`` to rebuild every step — the A/B switch the
        # cache's speedup is measured with; it is numerically a no-op either way.
        self._foreach_plans: dict[int, _ForeachPlan] = {}
        self._foreach_cache_enabled = True
        # --- native factored-bucket work reducers (0.7.12). Each is a separate A/B toggle
        # so its own contribution stays measurable; see _factored_bucket for the identity
        # each one relies on and for the exactness each one costs (nothing, at defaults).
        self._eps1_on_means = True        # eps1 on the [N,R]/[N,C] means, not the [N,R,C] square
        self._write_fold_lr = True        # lr as the write's alpha, not a delta.mul_(lr) pass
        # REJECTED BY MEASUREMENT, kept behind the toggle so another GPU can re-test.
        # RMS-clip via matvec with the clip divisor folded into r_factor removes one
        # [N,R,C] pass but adds ~12 tiny launches and a batched GEMV. Paired A/B on an
        # RTX 3000 Ada (geometric mean of 150-300 per-rep ratios, 95% CI, fp32 params /
        # bf16 momentum): 200x(256,256) 0.964x [0.935, 0.993] — a SIGNIFICANT LOSS —
        # 50x(512,512) 1.002x [0.989, 1.016], 8x(1024,1024) 0.993x [0.970, 1.016],
        # 400x(64,64) 0.979x [0.931, 1.030]; not a significant win anywhere. It also
        # routes the clip divisor through a matmul, so a process with
        # ``torch.backends.cuda.matmul.allow_tf32`` on would lose mantissa bits there.
        self._native_rms_matvec = False
        # Batched-big int8 momentum dequant/EMA/requant entirely in Triton (the mirror of the
        # 4-bit direct path) instead of the codec's stacked fp32 temp. False = codec fallback.
        self._direct_int8 = True
        if self._fused:
            from kaon._fused_triton import HAS_TRITON, TILE_CAP
            if not HAS_TRITON:
                raise RuntimeError("Adakaon(fused=True) requires Triton (a GPU-only optional dependency)")
            # 2-D one_block/chunked crossover ONLY. The non-factored (ndim<=1) ceiling is
            # ft.TILE_CAP_1D and is deliberately not tunable from here: a 1-D tensor over its
            # cap falls to native rather than to a better kernel, so the two are independent.
            self._fused_tile_cap = TILE_CAP if fused_tile_cap is None else fused_tile_cap

        # Optional continuous Mechanic step-size controller. It owns group["lr"] while
        # adapting and drives the base update through _step_impl.
        # Off (default) -> zero overhead, step == _step_impl.
        self._init_autolr(auto_lr, auto_lr_scale, auto_lr_fuse_rel, auto_lr_d0)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Add a group, rejecting the fp16 + stochastic-rounding combination up front.

        ``Optimizer.__init__`` routes every constructor group through here too, so one check
        covers construction and later additions. There is no fp16 stochastic-rounding write in
        kaon: ``kaon.add_stochastic_`` raises ``NotImplementedError`` for it and the fused
        kernel's ``sr_round`` is a bf16-only bit trick. The step used to reach neither — the
        param fell off the foreach fast path and the fused routing, and the per-param fallback
        wrote round-to-nearest — so an fp16 run silently lost every update smaller than half an
        fp16 ulp, which is exactly the loss stochastic rounding exists to prevent. Say so at
        construction instead, and point at the supported route.

        The check runs BEFORE ``super()``: a caller that catches the error must not be left
        holding a group the optimizer cannot step.
        """
        if not isinstance(param_group, dict):
            super().add_param_group(param_group)   # let torch raise its own TypeError
            return
        params = param_group["params"]
        if isinstance(params, torch.Tensor):
            params = [params]
        else:
            # ``params`` may be a generator; materialize it here (exactly as torch is about to)
            # so validating it does not consume what ``super()`` still has to read.
            params = list(params)
            param_group["params"] = params
        method = param_group.get("bf16_method", self.defaults.get("bf16_method"))
        if method == "stochastic_rounding":
            for p in params:
                if p.dtype == torch.float16:
                    raise NotImplementedError(
                        "bf16_method='stochastic_rounding' does not support torch.float16 "
                        "parameters (stochastic rounding is implemented for bfloat16 only); "
                        "use bf16_method='kahan' for fp16 parameters, or keep them in fp32"
                    )
        super().add_param_group(param_group)

    def _invalidate_fused_caches(self) -> None:
        """Drop every host-side cache that may retain pointers or views into optimizer state.

        State tensors are allocated lazily by the next step.  The fused pointer caches and
        the native foreach view caches therefore must never survive a state reset or
        checkpoint load, even when the parameter identities themselves have not changed:
        ``load_state_dict`` *replaces* ``state["v"]`` / ``state["m"]`` (the dtype-preserving
        loader casts them back), so anything still holding the old buffers would keep
        stepping a detached copy of the state.
        """
        self._fused_part.clear()
        self._fused_demoted.clear()
        self._fused_ob_caches.clear()
        self._fused_od_caches.clear()
        self._fused_big_caches.clear()
        self._foreach_plans.clear()

    def _autolr_reset_base_state(self) -> None:
        """Reset Adakaon's base optimizer after an AutoLR rollback/contact."""
        self.state.clear()
        self._t = 0
        self._invalidate_fused_caches()

    def _codec(self, group: dict[str, Any]) -> _MomentumCodec:
        md = group["momentum_dtype"]
        codec = self._codecs.get(md)
        if codec is None:
            codec = self._codecs[md] = _make_codec(md)
        return codec

    @torch.no_grad()
    def _init_state(self, p: Tensor, state: dict[str, Any], group: dict[str, Any]) -> None:
        grad = p.grad
        factored = p.ndim >= 2
        if factored:
            # ndim==2 is already its own matrix; ndim>2 (conv) reshapes to
            # [out, in·kh·kw] before factoring (the conv-aware fix, always on).
            gv = grad if p.ndim == 2 else grad.reshape(grad.shape[0], -1)
            row_shape = gv.shape[:-1]
            col_shape = gv.shape[:-2] + gv.shape[-1:]
            state["row"] = torch.zeros(row_shape, dtype=torch.float32, device=p.device)
            state["col"] = torch.zeros(col_shape, dtype=torch.float32, device=p.device)
        else:
            state["v"] = torch.zeros_like(grad, dtype=torch.float32)
        if group["betas"][0] > 0:
            self._codec(group).init_state(state, grad, group)
        if is_low_precision(p) and group["bf16_method"] == "kahan":
            state["shift"] = torch.zeros_like(p)

    # step() is the AutoLRMixin router (drives Mechanic when auto_lr is on, else
    # calls _step_impl); it also re-imposes the frozen LR each step vs a harness clobber.
    @torch.no_grad()
    def _step_impl(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self._fused:
            return self._fused_step(loss)
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            for p in params:
                if p.grad.is_sparse:
                    raise RuntimeError("Adakaon does not support sparse gradients")
            if group["gradient_centralization"]:
                centralize_grads_(params)
            self._native_dispatch(params, group)
        return loss

    @torch.no_grad()
    def _native_dispatch(self, params: list[Tensor], group: dict[str, Any]) -> None:
        """The native (non-fused) step over ``params`` — foreach batching where eligible, else
        per-param. Gradient Centralization is the caller's responsibility (done per-subset)."""
        if not params:
            return
        if self._foreach and self._group_foreach_eligible(group):
            chunk_budget = foreach_budget(self._foreach_stack_budget, self._foreach_batch_cutoff, _STACK_BYTES_PER_ELEM, params[0].device)
            # Effective cutoff = the performance threshold, lowered only if the
            # memory budget can't fit two of a tensor in a chunk (so batching
            # would be a wasteful stack-of-1). Roomy card -> cutoff wins;
            # constrained card -> safety wins.
            cutoff = min(self._foreach_batch_cutoff, chunk_budget // 2)
            fast: list[Tensor] = []
            slow: list[Tensor] = []
            for p in params:
                (fast if self._param_foreach_eligible(p, group, cutoff) else slow).append(p)
            if len(fast) >= 2:
                self._step_foreach(fast, group, chunk_budget)
                for p in slow:
                    self._step_one_param(p, group)
                return
        # Per-parameter fallback for the whole group: drop any cached plan for it. The
        # shared codecs requant IN PLACE since 4d980ee (``state["m"].copy_(q)``), so this
        # is no longer load-bearing for int8/4bit the way it was — but a group that steps
        # per-param has no use for a plan anyway, and dropping it keeps the invariant
        # "a cached plan only ever describes a group the foreach path actually stepped".
        # The ids/ptrs guard cannot see a fallback: the param set is unchanged.
        self._foreach_plans.pop(id(group), None)
        for p in params:
            self._step_one_param(p, group)

    # ----------------------------------------------------------------- fused (Triton) step
    @torch.no_grad()
    def _fused_step(self, loss: Any) -> Any:
        """Triton-fused step: eligible 2-D weights run through the one-block / chunked kernels
        (same math + state as native); everything else falls back to :meth:`_native_dispatch`."""
        import kaon._fused_triton as ft

        self._t += 1
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            for p in params:
                if p.grad.is_sparse:
                    raise RuntimeError("Adakaon does not support sparse gradients")
            parts = self._fused_partition(group, params, ft)
            # Routing is cached; grad CONTIGUITY is not cacheable (fresh tensor every backward).
            one_block, big, one_dim, native = self._fused_demote(id(group), parts)
            if native:  # GC for the native subset (fused subsets centralize in-kernel / in-reductions)
                if group["gradient_centralization"]:
                    centralize_grads_(native)
                self._native_dispatch(native, group)
            if one_block:
                self._fused_one_block(one_block, group, ft)
            if big:
                self._fused_big(big, group, ft)
            if one_dim:
                self._fused_one_dim(one_dim, group, ft)
        return loss

    @torch.no_grad()
    def _fused_big(self, big: list[Tensor], group: dict[str, Any], ft: Any) -> None:
        """Dispatch the >tile-cap ("big") 2-D weights. The per-tensor fused-chunked kernel is
        launch-bound (~7 kernels/tensor); for the many same-shape big factors a real LoKr run has
        (e.g. 236x 512x512) we run the **batched chunked kernel** (``_chunked_step_batched``) —
        the whole same-shape bucket in ~2 launches, p/m written in place via a pointer array. Same
        math + state either way. ``self._fused_big_batched=False`` reverts the multi-tensor case to
        the batched-native-foreach path (the A/B baseline).

        A LONE big tensor (``N == 1``) takes the same batched kernel since 0.7.12. It has no batch
        to amortize launches over, but that was never the per-tensor path's problem: the batched
        route is fully device-resident, while ``_chunked_step`` blocks the CPU TWICE per tensor per
        step (``float(rms)`` in ``_chunked_reductions`` and ``keep.item()`` for the cautious mean)
        and materializes an fp32 ``g`` plus a ``g*g`` of the tensor's size. On a bag of DISTINCT big
        shapes — the UNet/DiT case, where every shape bucket holds exactly one tensor — those syncs
        serialize the whole step. ``self._fused_big_lone_batched=False`` reverts to the per-tensor
        kernel (the A/B baseline)."""
        if len(big) >= 2 and not self._fused_big_batched:
            if group["gradient_centralization"]:
                centralize_grads_(big)
            self._native_dispatch(big, group)
            return
        # Group by EXACT shape (and dtype and DEVICE); same-shape buckets of >=2 take the
        # batched chunked kernel, lone tensors take the per-tensor chunked kernel.
        for plist in _same_shape_device_buckets(big).values():
            # One device scope per bucket, covering every launch inside the chunked steps (see
            # _fused_one_block): the bucket's pointer arrays and scratch live on plist[0].device,
            # and a Triton launch targets the CURRENT device. PLAUSIBLE, not verified — one GPU here.
            with torch.cuda.device(plist[0].device):
                if (len(plist) >= 2 or group["betas"][0] == 0.0
                        or self._fused_big_lone_batched):
                    self._chunked_step_batched(plist, group, ft)
                else:
                    self._chunked_step(plist[0], group, ft)

    def _fused_partition(self, group: dict[str, Any], params: list[Tensor], ft: Any) -> tuple:
        """Split a group's params into (one-block, chunked-big, one-dim, native), cached per param-set.

        Keyed on :func:`kaon._fused_triton.param_witness` — ids, ``data_ptr``s and contiguity,
        plus strides under ``ft.SHAPE_WITNESS`` — because each is a routing input the partition
        (and the pointer arrays derived from it) bakes in, and ``p.data = ...`` can change any of
        them while the Parameter object stays the same. An id-only key kept dispatching a stale
        plan at memory the optimizer no longer owns. The FUSED witness is used here rather than
        :func:`_param_witness` deliberately — see that function on why the native plan carries no
        shape field, and on why this key must be exactly as strong as the pointer caches' own.

        A rebuild is where the state geometry gets revalidated (the caches call
        ``ft.check_state_geometry``), so this key moving is what turns a shape-changing rebind
        into a clear error instead of a plan pointed at the wrong buffers.

        Grad properties deliberately stay OUT of this key: a gradient is a new tensor every
        backward, so its contiguity is re-checked per step in :func:`_demote_non_contiguous_grads`
        rather than frozen into the routing.
        """
        gid = id(group)
        witness = ft.param_witness(params)
        cached = self._fused_part.get(gid)
        if cached is not None and cached[0] == witness:
            return cached[1], cached[2], cached[3], cached[4]
        md, bf16m, cap = group["momentum_dtype"], group["bf16_method"], self._fused_tile_cap
        one_block: list[Tensor] = []
        big: list[Tensor] = []
        one_dim: list[Tensor] = []
        native: list[Tensor] = []
        momentum = group["betas"][0] > 0
        for p in params:
            # bf16 params need stochastic rounding (the kernel's only bf16 write); kahan/none -> native
            bf_ok = (p.dtype != torch.bfloat16) or (bf16m == "stochastic_rounding")
            # ndim>2 (conv) is matrixized to (out, in*kh*kw); every momentum layout is row-major
            # compatible with that view (int8 scales dim-0, 4-bit blocks the same flat storage).
            # The matrixized write-back needs a contiguous GRAD — enforced per step by
            # _demote_non_contiguous_grads, not here, because the grad changes every backward.
            two_d = bf_ok and p.ndim >= 2 and p.is_cuda and p.is_contiguous() \
                and p.dtype in (torch.float32, torch.bfloat16)
            ok = bf_ok and ft.fused_eligible(p, cap)
            if ok and momentum and md == "4bit" and ft.eff_2d(p)[1] % 2 != 0:
                ok = False                                  # one-block 4bit needs even C
                # NOTE the block-size guard that used to live here is gone (0.7.12): the tile
                # kernel takes ``state["m_block"]`` as a RUNTIME scalar and
                # ``PointerArrayCache`` buckets by it, so every ``momentum_4bit_block`` keeps
                # this route instead of degrading to native. Only the even-C packing
                # assumption is still a real constraint.
            if ok:
                one_block.append(p)
            elif two_d and ft.next_pow2_tile(*ft.eff_2d(p))[0] * ft.next_pow2_tile(*ft.eff_2d(p))[1] > cap:
                big.append(p)
            elif bf_ok and (
                # No cap argument on purpose: 1-D is bounded by ft.TILE_CAP_1D, not by the 2-D
                # ``cap``. Above the 2-D cap a tensor goes to the chunked kernel; above the 1-D
                # cap it goes to NATIVE (there is no chunked 1-D route), so the two ceilings
                # answer different questions and moved apart in 0.7.10.
                ft.fused_1d_eligible(p)
                # 0-D scalars (LyCORIS use_scalar gates) ride the 1-D kernel as
                # length-1 tensors: it works on base pointers + element counts,
                # never on shapes, so a valid pointer with numel()==1 is enough.
                # Per-param dispatch for them is ~22 launches/scalar/step (CPU
                # launch-bound); the kernel is one launch for the whole bag.
                or (p.ndim == 0 and p.is_cuda and p.dtype in (torch.float32, torch.bfloat16))
            ):
                one_dim.append(p)
            else:
                native.append(p)
        if _FUSED_DISABLE:  # diagnostic routing override (see module note)
            if "one_block" in _FUSED_DISABLE:
                native += one_block
                one_block = []
            if "big" in _FUSED_DISABLE:
                native += big
                big = []
            if "one_dim" in _FUSED_DISABLE:
                native += one_dim
                one_dim = []
        if _PROBE_LOG_PATH:  # routing census + grad-contiguity audit (probe runs only)
            noncontig = [tuple(p.shape) for p in params if p.grad is not None and not p.grad.is_contiguous()]
            with open(_PROBE_LOG_PATH, "a") as fh:  # noqa: SIM115 — diagnostics only
                fh.write(
                    f"[census] one_block={len(one_block)} big={len(big)} one_dim={len(one_dim)} "
                    f"native={len(native)} noncontig_grads={noncontig[:8]} disable={sorted(_FUSED_DISABLE)}\n"
                )
        self._fused_part[gid] = (witness, one_block, big, one_dim, native)
        return one_block, big, one_dim, native

    def _fused_demote(self, gid: int, parts: tuple) -> tuple:
        """This step's routing, with any non-contiguous-grad tensor moved to the native subset.

        Thin memo over :func:`_demote_non_contiguous_grads`. The demotion has to build fresh
        route lists, and a fresh list means every downstream pointer cache sees a new object and
        rebuilds itself (``_WitnessedCache.built_from``) — so a tensor whose grad is PERSISTENTLY
        strided would throw away the whole bucket's index tensors on every step. Reuse the lists
        while both the partition (identity of its four lists) and the demoted SET are unchanged;
        the contiguity sweep itself still runs every step, since that is what detects the change.
        """
        demoted = tuple(id(p) for sub in parts[:3] for p in sub if not p.grad.is_contiguous())
        if not demoted:
            self._fused_demoted.pop(gid, None)
            return parts
        cached = self._fused_demoted.get(gid)
        if (cached is not None and cached[0] == demoted
                and all(a is b for a, b in zip(cached[1], parts, strict=True))):
            return cached[2]
        out = _demote_non_contiguous_grads(*parts)
        self._fused_demoted[gid] = (demoted, parts, out)
        return out

    def _fused_one_block(self, plist: list[Tensor], group: dict[str, Any], ft: Any) -> None:
        """Launch the one-block pointer-array kernel over the eligible small 2-D weights."""
        for p in plist:
            st = self.state[p]
            if not st:
                self._init_state(p, st, group)
        gid = id(group)
        cache = self._fused_ob_caches.get(gid)
        # ``built_from`` (identity), not ``stale`` (tuple rebuild): _fused_partition already
        # revalidated ids+data_ptrs for the whole group this step and only returns this exact
        # list object while nothing moved. See _WitnessedCache.built_from.
        if cache is None or not cache.built_from(plist):
            cache = ft.PointerArrayCache(plist, lambda p: self.state[p], None)
            self._fused_ob_caches[gid] = cache
        cache.refresh_grads()
        b1, b2 = group["betas"]
        lr, eps1 = group["lr"], group["eps"][0]
        clip, wd = group["clip_threshold"], group["weight_decay"]
        cautious, gc = group["cautious"], group["gradient_centralization"]
        wd_full = wd != 0 and group["cautious_wd"] == "full"   # see _step_one_param
        # A bucket's index arrays live on ITS device (PointerArrayCache buckets by device), and a
        # Triton launch goes to the CURRENT device, not to the one the arguments came from. A group
        # spanning cuda:0 and cuda:1 would otherwise launch every bucket on whichever device
        # happened to be current. PLAUSIBLE, not verified: this machine has one GPU, so the
        # multi-device path is reasoned about rather than tested.
        for bk in cache.buckets:
            lanes = bk["BR"] * bk["BC"]
            with torch.cuda.device(bk["dev"]):
                ft._adakaon_tile_kernel[(len(bk["plist"]),)](
                    bk["g_addr"], bk["p_addr"], bk["m_addr"], bk["mscale_addr"], bk["row_addr"],
                    bk["col_addr"], bk["Rs"], bk["Cs"], bk["mscale_n"],
                    lr, b1, b2, eps1, clip, wd, self._t, bk["blk"],
                    LOWP=bk["lowp"], MOM=bk["mom"], CAUTIOUS=cautious, WD=wd != 0, GC=gc,
                    SR=bk["lowp"], MOMENTUM=bk["momentum"], WDFULL=wd_full,
                    BR=bk["BR"], BC=bk["BC"], EXACT=bk["exact4"], FBLK=bk["fblk"],
                    num_warps=ft.warps_for(lanes),
                )

    def _fused_one_dim(self, plist: list[Tensor], group: dict[str, Any], ft: Any) -> None:
        """One-block non-factored kernel over the eligible ``ndim <= 1`` weights (biases / norm
        scales, plus 0-D scalars as length-1 tensors). GC is a no-op on ndim<2 (``centralize_grads_``
        skips them), so it never enters this path."""
        for p in plist:
            st = self.state[p]
            if not st:
                self._init_state(p, st, group)
        gid = id(group)
        cache = self._fused_od_caches.get(gid)
        if cache is None or not cache.built_from(plist):   # see _fused_one_block
            cache = ft.OneDimPointerCache(plist, lambda p: self.state[p])
            self._fused_od_caches[gid] = cache
        cache.refresh_grads()
        b1, b2 = group["betas"]
        lr, eps1 = group["lr"], group["eps"][0]
        clip, wd = group["clip_threshold"], group["weight_decay"]
        cautious = group["cautious"]
        wd_full = wd != 0 and group["cautious_wd"] == "full"   # see _step_one_param
        for bk in cache.buckets:                       # see _fused_one_block on the device scope
            with torch.cuda.device(bk["dev"]):
                ft._adam_1d_kernel[(len(bk["plist"]),)](
                    bk["g_addr"], bk["p_addr"], bk["m_addr"], bk["mscale_addr"], bk["v_addr"],
                    bk["Ls"], lr, b1, b2, eps1, clip, wd, self._t,
                    LOWP=bk["lowp"], MOM=bk["mom"], MOMENTUM=bk["momentum"], CAUTIOUS=cautious,
                    WD=wd != 0, SR=bk["lowp"], BL=bk["BL"], FBLOCK=bk["block"], WDFULL=wd_full,
                    num_warps=ft.warps_for(bk["BL"]),
                )

    def _chunked_reductions(self, p: Tensor, group: dict[str, Any], st: dict[str, Any]) -> tuple:
        """Shared torch part of a big-tensor step: GC + row/col EMA + rms (matvec, no [R,C] temp).
        ``ndim>2`` convs are matrixized to ``(out, in*kh*kw)`` (the row/col state's shape)."""
        n = p.numel()
        R = p.shape[0]  # noqa: N806
        b2, eps1 = group["betas"][1], group["eps"][0]
        clip = group["clip_threshold"]
        g = p.grad.float().reshape(R, n // R)
        if group["gradient_centralization"]:
            g = g - g.mean(dim=1, keepdim=True)
        g = g.contiguous()
        gsq = g * g
        st["row"].lerp_(gsq.mean(1).add_(eps1), 1.0 - b2)
        st["col"].lerp_(gsq.mean(0).add_(eps1), 1.0 - b2)
        r = st["row"].div(st["row"].mean()).rsqrt_()
        c = st["col"].rsqrt()
        rms = (((r * r) * gsq.matmul(c * c)).sum() / n).sqrt_()
        return g, r, c, 1.0 / float(rms.div_(clip).clamp_(min=1.0))

    def _chunked_step(self, p: Tensor, group: dict[str, Any], ft: Any) -> None:
        """One big 2-D tensor via the chunked kernels; int8/4bit momentum through the codec (dequant
        -> fp32 temp -> kernels -> requant) so the weight update uses the exact pre-requant momentum."""
        st = self.state[p]
        if not st:
            self._init_state(p, st, group)
        R, C = p.shape[0], p.numel() // p.shape[0]  # noqa: N806 — matrix dims (conv -> matrixized)
        n = R * C
        md, b1 = group["momentum_dtype"], group["betas"][0]
        lr, wd, cautious = group["lr"], group["weight_decay"], group["cautious"]
        wd_full = wd != 0 and group["cautious_wd"] == "full"   # see _step_one_param
        sr = (p.dtype == torch.bfloat16) and (group["bf16_method"] == "stochastic_rounding")
        g, r, c, inv_rms = self._chunked_reductions(p, group, st)
        quant = md in ("int8", "4bit")
        if quant:
            m_fp32 = self._codec(group).dequant_one(st, torch.empty(R, C, device=p.device)).reshape(R, C)
            mf = m_fp32.reshape(-1)
        else:
            mf = st["m"].reshape(-1)
        keep = torch.zeros(1, dtype=torch.int32, device=p.device)
        gf, pf = g.reshape(-1), p.reshape(-1)
        grid = ((n + 1023) // 1024,)
        ft._chunked_mom[grid](gf, mf, pf, r, c, keep, C, n, inv_rms, wd, b1,
                              CAUTIOUS=cautious, WD=wd != 0, BLOCK=1024, WDFULL=wd_full)
        if quant:
            # Requant IN PLACE: cached pointer tables (MSAM's fused axpy plan, batched-step
            # plans) hold raw data_ptrs into these buffers — replacing the tensors leaves the
            # tables dangling (illegal memory access once the old block is empty_cache()d).
            if md == "int8":
                q, sc = _quant_int8(m_fp32)
                st["m"].copy_(q.reshape(st["m"].shape))
                st["m_scale"].copy_(sc.reshape(st["m_scale"].shape))
            else:
                packed, sc, _ = _quant_4bit(m_fp32.reshape(-1), st["m_block"])
                st["m"].copy_(packed)
                st["m_scale"].copy_(sc)
        inv_mean = 1.0 / max(keep.item() / n, 1e-8) if cautious else 1.0
        ft._chunked_apply[grid](gf, mf, pf, n, inv_mean, lr, wd, self._t,
                                CAUTIOUS=cautious, WD=wd != 0, SR=sr, BLOCK=1024, WDFULL=wd_full)

    # ----------------------------------------------- batched chunked (many same-shape big tensors)
    @torch.no_grad()
    def _chunked_reductions_batched(self, plist: list[Tensor], group: dict[str, Any]) -> tuple:
        """Stacked torch reductions for a same-shape big bucket: GC (on the fp32 copy) + row/col EMA
        + per-tensor rms (matvec, no [N,R,C] beyond grad/gsq). Returns the stacked fp32 grad ``[N,n]``,
        the stacked r/c factors ``[N,R]``/``[N,C]`` (contiguous), and ``inv_rms`` ``[N]`` — the same
        quantities ``_chunked_reductions`` returns per tensor. Mirrors that math exactly (eps1 added to
        the row/col means; rms uses raw gsq)."""
        b2, eps1 = group["betas"][1], group["eps"][0]
        clip = group["clip_threshold"]
        N = len(plist)  # noqa: N806
        R, C = plist[0].shape[0], plist[0].numel() // plist[0].shape[0]  # noqa: N806 — conv -> matrixized
        n = R * C
        states = [self.state[p] for p in plist]
        g = torch.stack([p.grad.reshape(R, C) for p in plist]).float()     # [N, R, C]
        if group["gradient_centralization"]:
            g.sub_(g.mean(dim=-1, keepdim=True))                          # GC on the fp32 copy
        gsq = g * g                                                       # raw (eps1 goes on the means)
        row = torch.stack([s["row"] for s in states])                    # [N, R]
        col = torch.stack([s["col"] for s in states])                    # [N, C]
        row.lerp_(gsq.mean(dim=-1).add_(eps1), 1.0 - b2)
        col.lerp_(gsq.mean(dim=-2).add_(eps1), 1.0 - b2)
        torch._foreach_copy_([s["row"] for s in states], list(row.unbind(0)))
        torch._foreach_copy_([s["col"] for s in states], list(col.unbind(0)))
        r = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_()             # [N, R]
        c = col.rsqrt()                                                  # [N, C]
        # rms per tensor via matvec: sqrt( sum_i r_i^2 * (gsq @ c^2)_i / n )  (no [N,R,C] temp)
        rms = (r * r).mul_(torch.bmm(gsq, (c * c).unsqueeze(-1)).squeeze(-1)).sum(-1).div_(n).sqrt_()
        inv_rms = rms.div_(clip).clamp_(min=1.0).reciprocal_()           # [N]
        return g.reshape(N, n), r.contiguous(), c.contiguous(), inv_rms.contiguous()

    @torch.no_grad()
    def _chunked_step_batched(self, plist: list[Tensor], group: dict[str, Any], ft: Any) -> None:
        """A same-shape big 2-D bucket via the batched chunked kernels (~2 launches). ``N == 1``
        included — see :meth:`_fused_big`.

        fp32/bf16 momentum is read/written in place via the m pointer array. **Every** quantized
        momentum now has an in-Triton route too, with no momentum-sized fp32 temporary: 4-bit
        when its absmax blocks tile a chunk (``m_block <= 1024`` and divides it), int8 when a
        chunk owns whole ROWS (``C <= 1024`` and divides it — int8 scales per row). Only the
        shapes that fail those alignment conditions keep the codec fallback (dequant to a
        stacked fp32 temp, generic kernels, requant), which is correct but costs ~15 torch
        kernels, a host->device pointer copy (a CPU<->GPU sync) and 4 B/param of transient."""
        for p in plist:
            st = self.state[p]
            if not st:
                self._init_state(p, st, group)
        N = len(plist)  # noqa: N806
        R, C = plist[0].shape[0], plist[0].numel() // plist[0].shape[0]  # noqa: N806 — conv -> matrixized
        n = R * C
        dev = plist[0].device
        md, b1 = group["momentum_dtype"], group["betas"][0]
        lr, wd, cautious = group["lr"], group["weight_decay"], group["cautious"]
        wd_full = wd != 0 and group["cautious_wd"] == "full"   # see _step_one_param
        gc = group["gradient_centralization"]
        lowp = plist[0].dtype == torch.bfloat16
        sr = lowp and (group["bf16_method"] == "stochastic_rounding")
        states = [self.state[p] for p in plist]
        cache_key = (id(group), tuple(plist[0].shape), plist[0].dtype, plist[0].device)
        cache = self._fused_big_caches.get(cache_key)
        # ``cache.gc`` is part of the validity, not only the param witness: with GC off the
        # cache aliases ``rowmean`` onto ``rowsum`` to save N*R floats, and a group dict
        # flipped mid-run (schedulers do this) would then have the reduction kernel write
        # the row means over the row sums. Nothing moved, so ``stale()`` cannot see it.
        if cache is None or cache.stale(plist) or cache.gc != gc:
            cache = ft.BigPointerCache(plist, lambda p: self.state[p], R, C, gc=gc)
            self._fused_big_caches[cache_key] = cache
        cache.refresh_grads()

        if b1 == 0.0:
            self._chunked_step_batched_nomom(
                plist, group, ft, R, C, n, lowp, sr, states, cache
            )
            return

        # Reductions: fused (grad via pointer array, no [N,R,C] stack — candidate #4) or torch.
        fused_red = self._fused_reductions
        clip = group["clip_threshold"]
        if fused_red:
            # ``rms`` is the RAW sum-of-squares accumulator; every ``_g`` consumer turns it
            # into the clip factor itself, and ``zero_accumulators`` has already cleared
            # ``keep`` along with it (one launch for the three).
            g_addr, rowmean, r, c, rms = self._chunked_reductions_fused(
                plist, group, ft, R, C, n, lowp, cache
            )
            keep = cache.keep
        else:
            g, r, c, inv_rms = self._chunked_reductions_batched(plist, group)
            keep = cache.keep.zero_()

        p_addr = cache.p_addr
        K = (n + 1023) // 1024  # noqa: N806
        grid = (N * K,)
        direct_4bit = md == "4bit" and states[0]["m_block"] <= 1024 \
            and 1024 % states[0]["m_block"] == 0
        # int8's scale is per ROW, so a chunk may requantize a row only if it owns the whole
        # row: C <= BLOCK and BLOCK % C == 0 (see _chunked_int8_apply_batched_g). Otherwise the
        # per-row absmax would need a cross-program reduction and the codec fallback stands.
        direct_int8 = (md == "int8" and self._direct_int8
                       and C <= 1024 and 1024 % C == 0)
        if direct_int8 and fused_red:
            if cautious:
                ft._chunked_int8_keep_batched_g[grid](
                    g_addr, rowmean, cache.m_addr, cache.mscale_addr, p_addr, r, c,
                    keep, rms, clip, wd, b1, R, C, n, K,
                    LOWP=lowp, GC=gc, WD=wd != 0, BLOCK=1024, WDFULL=wd_full,
                )
            ft._chunked_int8_apply_batched_g[grid](
                g_addr, rowmean, cache.m_addr, cache.mscale_addr, p_addr, r, c,
                keep, rms, clip, lr, wd, b1, self._t, R, C, n, K,
                LOWP=lowp, GC=gc, CAUTIOUS=cautious, WD=wd != 0, SR=sr,
                CSEG=C, RPC=1024 // C, BLOCK=1024, WDFULL=wd_full,
            )
            return
        if direct_4bit and fused_red:
            block = states[0]["m_block"]
            if cautious:
                ft._chunked_4bit_keep_batched_g[grid](
                    g_addr, rowmean, cache.m_addr, cache.mscale_addr, p_addr, r, c,
                    keep, rms, clip, wd, b1, R, C, n, K,
                    LOWP=lowp, GC=gc, WD=wd != 0, FBLOCK=block, BLOCK=1024, WDFULL=wd_full,
                )
            ft._chunked_4bit_apply_batched_g[grid](
                g_addr, rowmean, cache.m_addr, cache.mscale_addr, p_addr, r, c,
                keep, rms, clip, lr, wd, b1, self._t, R, C, n, K,
                LOWP=lowp, GC=gc, CAUTIOUS=cautious, WD=wd != 0, SR=sr,
                FBLOCK=block, BLOCK=1024, WDFULL=wd_full,
            )
            return

        quant = md in ("int8", "4bit")
        if quant:  # dequant whole bucket to a stacked fp32 temp; kernel m pointers index its slices
            # Batched: dequant_stacked runs the whole bucket's codec in a handful of kernels.
            # The per-tensor dequant_one loop was ~8 tiny torch ops x N tensors, CPU-dispatch
            # bound: measured ~87 ms/step on a 528-tensor LoRA-r32 fleet (4080), vs <2 ms batched.
            temp = self._codec(group).dequant_stacked(
                states, lambda t: t, (R, C)
            ).reshape(N, R, C).contiguous()
            m_addr = ft.ptr_array(list(temp), dev)
            mom = ft.MOM_FP32
        else:
            m_addr = cache.m_addr
            mom = ft.MOM_BF16 if md == "bfloat16" else ft.MOM_FP32
        if fused_red:
            ft._chunked_mom_batched_g[grid](
                g_addr, rowmean, m_addr, p_addr, r, c, keep, rms, clip, wd, b1, R, C, n, K,
                LOWP=lowp, MOM=mom, GC=gc, CAUTIOUS=cautious, WD=wd != 0, BLOCK=1024, WDFULL=wd_full,
            )
        else:
            ft._chunked_mom_batched[grid](
                g, m_addr, p_addr, r, c, keep, inv_rms, wd, b1, R, C, n, K,
                LOWP=lowp, MOM=mom, CAUTIOUS=cautious, WD=wd != 0, BLOCK=1024, WDFULL=wd_full,
            )
        if quant:  # requant the updated fp32 temp back into per-tensor storage (apply reads the temp)
            # Batched requant (same write pattern as ema_stacked: in-place copies keep the
            # state tensors' identities stable for pointer-array caches).
            if md == "int8":
                q8, new_scale = _quant_int8_stacked(temp)            # per-row scale, [N, R, 1]
                torch._foreach_copy_(
                    [st["m"] for st in states],
                    [q.view_as(st["m"]) for st, q in zip(states, q8.unbind(0), strict=True)],
                )
                for st, sc in zip(states, new_scale.unbind(0), strict=True):
                    st["m_scale"].copy_(sc.view_as(st["m_scale"]))
            else:
                new_packed, new_scale = _quant_4bit_stacked(temp.reshape(N, -1), states[0]["m_block"])
                torch._foreach_copy_([st["m"] for st in states], list(new_packed.unbind(0)))
                for st, sc in zip(states, new_scale.unbind(0), strict=True):
                    st["m_scale"].copy_(sc)
        if fused_red:
            ft._chunked_apply_batched_g[grid](
                g_addr, rowmean, m_addr, p_addr, keep, lr, wd, self._t, R, C, n, K,
                LOWP=lowp, MOM=mom, GC=gc, CAUTIOUS=cautious, WD=wd != 0, SR=sr, BLOCK=1024, WDFULL=wd_full,
            )
        else:
            inv_mean = (
                (1.0 / (keep.float() / n).clamp_(min=1e-8))
                if cautious else torch.ones(N, device=dev)
            )
            ft._chunked_apply_batched[grid](
                g, m_addr, p_addr, inv_mean, lr, wd, self._t, n, K,
                LOWP=lowp, MOM=mom, CAUTIOUS=cautious, WD=wd != 0, SR=sr, BLOCK=1024, WDFULL=wd_full,
            )

    def _chunked_step_batched_nomom(
        self, plist, group, ft, R, C, n, lowp, sr, states, cache  # noqa: N803
    ) -> None:
        """Chunked big-tensor path for beta1=0 with no momentum allocation."""
        if not self._fused_reductions:
            if group["gradient_centralization"]:
                centralize_grads_(plist)
            self._native_dispatch(plist, group)
            return
        N = len(plist)  # noqa: N806
        lr, wd = group["lr"], group["weight_decay"]
        cautious = group["cautious"]
        wd_full = wd != 0 and group["cautious_wd"] == "full"   # see _step_one_param
        gc = group["gradient_centralization"]
        clip = group["clip_threshold"]
        g_addr, rowmean, r, c, rms = self._chunked_reductions_fused(
            plist, group, ft, R, C, n, lowp, cache
        )
        p_addr = cache.p_addr
        K = (n + 1023) // 1024  # noqa: N806
        grid = (N * K,)
        keep = cache.keep          # already zeroed with colsum/rms (one launch)
        if cautious:
            ft._chunked_nomom_keep_batched_g[grid](
                g_addr, rowmean, p_addr, r, c, keep, rms, clip,
                wd, R, C, n, K, LOWP=lowp, GC=gc, WD=wd != 0, BLOCK=1024, WDFULL=wd_full,
            )
        ft._chunked_nomom_apply_batched_g[grid](
            g_addr, rowmean, p_addr, r, c, rms, clip, keep,
            lr, wd, self._t, R, C, n, K, LOWP=lowp, GC=gc,
            CAUTIOUS=cautious, WD=wd != 0, SR=sr, BLOCK=1024, WDFULL=wd_full,
        )

    @torch.no_grad()
    def _chunked_reductions_fused(self, plist, group, ft, R, C, n, lowp, cache):  # noqa: N803
        """Candidate #4: row/col EMA factors + the rms accumulator via Triton reduction kernels
        reading grad from a pointer array (NO [N,R,C] stack; GC in-kernel). Returns
        ``(g_addr, rowmean, r, c, rms)`` — the mom/apply ``_g`` kernels re-read grad via
        ``g_addr``, GC via ``rowmean``, and turn ``rms`` into the clip factor themselves
        (:func:`kaon._fused_triton.inv_rms_clip`).

        Three launches per bucket per step live here that used to be four more (0.7.12):

        * ONE ``zero_()`` for all three atomic accumulators (``colsum``, ``rms``, ``keep``),
          which :class:`~kaon._fused_triton.BigPointerCache` keeps adjacent for exactly this.
        * NO ``_finish_rms``: that was a ``grid=1`` kernel — one program for the whole
          bucket — computing N scalars the consumers then loaded anyway. Each consumer now
          derives its own from the raw accumulator.

        Those four launches were a FIXED per-bucket cost (measured 62-94 us for the set),
        which is what dominates a step over many small-ish big buckets: a 40-bucket step
        paid it 40 times, 1.2-1.9 ms/step of pure launch overhead.
        """
        b2, eps1 = group["betas"][1], group["eps"][0]
        gc = group["gradient_centralization"]
        N = len(plist)  # noqa: N806
        g_addr = cache.g_addr
        BR, BC, RB = ft.reduction_tile(R, C)  # noqa: N806
        rowmean = cache.rowmean
        rowsum = cache.rowsum
        colsum = cache.colsum
        det = self._deterministic_reductions
        cache.zero_accumulators()          # colsum + rms + keep, one launch
        if det:
            colpart, rmspart = cache.partials(RB)
            CB = (C + 255) // 256  # noqa: N806
            ft._reduce_rowcol_det[(N * RB,)](
                g_addr, rowmean, rowsum, colpart, R, C, RB,
                LOWP=lowp, GC=gc, BR=BR, BC=BC, num_warps=ft.warps_for(BR * BC),
            )
            ft._reduce_colpart[(N * CB,)](colpart, colsum, C, RB, CB, BCT=256)
        else:
            ft._reduce_rowcol[(N * RB,)](
                g_addr, rowmean, rowsum, colsum, R, C, RB,
                LOWP=lowp, GC=gc, BR=BR, BC=BC, num_warps=ft.warps_for(BR * BC),
            )
        # Update persistent row/col state directly via pointer arrays and emit
        # factors in one launch (no stack/scatter or eager elementwise chain). ``r``/``c``
        # ALIAS ``rowsum``/``colsum``: program t reads and writes the same indices, so the
        # factor overwrites the sum it came from (see BigPointerCache).
        r = cache.rfac
        c = cache.cfac
        row_addr = cache.row_addr
        col_addr = cache.col_addr
        FR = ft.triton.next_power_of_2(R)  # noqa: N806
        FC = ft.triton.next_power_of_2(C)  # noqa: N806
        ft._factor_rowcol_batched[(N,)](
            row_addr, col_addr, rowsum, colsum, r, c, R, C, b2, eps1,
            BR=FR, BC=FC, num_warps=ft.warps_for(max(FR, FC)),
        )
        rms = cache.rms
        if det:
            ft._reduce_rms_det[(N * RB,)](
                g_addr, rowmean, r, c, rmspart, R, C, RB,
                LOWP=lowp, GC=gc, BR=BR, BC=BC, num_warps=ft.warps_for(BR * BC),
            )
            ft._reduce_rmspart[(N,)](rmspart, rms, RB)
        else:
            ft._reduce_rms[(N * RB,)](
                g_addr, rowmean, r, c, rms, R, C, RB,
                LOWP=lowp, GC=gc, BR=BR, BC=BC, num_warps=ft.warps_for(BR * BC),
            )
        return g_addr, rowmean, r, c, rms

    def state_dict(self) -> dict[str, Any]:
        """Base state + the auto_lr tuner blob (via AutoLRMixin) when auto_lr is on."""
        state_dict = self._autolr_state_dict(super().state_dict())
        # momentum_units=2: the first moment is an LR-independent direction (lr is
        # applied to the final delta each step). Absent/1 marks the pre-0.7.11
        # layout where lr was folded into the EMA; load_state_dict migrates it.
        state_dict["_adakaon_meta"] = {"fused_step": self._t, "momentum_units": 2}
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state, preserving the quantized first moment's stored dtype.

        torch's default ``load_state_dict`` upcasts every state tensor to the param's
        dtype (fp32), silently inflating a bf16/int8/4bit ``momentum_dtype`` back to fp32
        on resume — losing the memory the codec was chosen to save. Delegate to the
        preserving helper; the auto_lr tuner blob is peeled off first by AutoLRMixin.
        """
        copied = dict(state_dict)
        meta = copied.pop("_adakaon_meta", {})
        fused_step = int(meta.get("fused_step", 0))
        if fused_step < 0:
            raise ValueError("Adakaon checkpoint has an invalid fused step counter")
        self._autolr_load(copied, lambda sd: load_state_dict_preserving_dtypes(self, sd))
        # torch restores param_groups from the CHECKPOINT's dicts, so a checkpoint written
        # before a group key existed comes back without it and the next step raises KeyError.
        # Back-fill the constructor default (the historical behaviour for ``cautious_wd``).
        for g in self.param_groups:
            g.setdefault("cautious_wd", self.defaults.get("cautious_wd", "masked"))
        self._t = fused_step
        if int(meta.get("momentum_units", 1)) < 2:
            self._migrate_momentum_to_direction_units()
        self._invalidate_fused_caches()

    @torch.no_grad()
    def _migrate_momentum_to_direction_units(self) -> None:
        """Rescale a pre-0.7.11 checkpoint's momentum (lr-scaled) to direction units.

        The old layout folded lr into the EMA, so ``m_old == lr * m_direction`` at
        the checkpoint's (restored) group lr. ``codec.scale_`` multiplies the
        quantized codecs' ``m_scale`` (no requant error) and the float codecs' ``m``
        directly. ``lr == 0`` is left untouched: under the old layout that momentum
        content is zero (every EMA contribution was multiplied by that lr), so
        there is nothing to rescale and ``1/lr`` would poison the buffer with inf.
        """
        for group in self.param_groups:
            lr = group["lr"]
            if lr == 0:
                continue
            codec = self._codec(group)
            for p in group["params"]:
                st = self.state.get(p)
                if st and "m" in st:
                    codec.scale_(st, 1.0 / lr)

    # ----------------------------------------------------------------- foreach

    @staticmethod
    def _group_foreach_eligible(group: dict[str, Any]) -> bool:
        """Group-level options the batched fast path supports."""
        return (
            group["clip_threshold"] > 0          # clip always applied in the batched path
            and group["bf16_method"] != "kahan"  # kahan needs a per-param shift buffer
        )

    @staticmethod
    def _param_foreach_eligible(p: Tensor, group: dict[str, Any], cutoff: int) -> bool:
        """Per-parameter shapes/dtypes the batched fast path can stack.

        Both branches are covered: ``ndim >= 2`` uses the factored bucket, ``ndim
        <= 1`` (biases/norms — the bulk of a full fine-tune — plus 0-D scalars,
        which ride the non-factored bucket as length-1 rows) uses the
        non-factored bucket. Only the awkward dtype/contiguity cases fall back:
        a bag of 0-D scalars on the per-param path is ~22 CUDA launches *per
        scalar per step* (pure CPU dispatch, measured ~592x slower than the same
        params as shape ``(1,)``), which is exactly what batching exists to fix.

        ``cutoff`` is the effective per-tensor size limit (performance threshold,
        possibly lowered by the memory budget) — bigger weights loop.
        """
        if p.numel() > cutoff:
            # Compute/bandwidth-bound: the per-tensor launch overhead is noise for
            # it, so looping is as fast and skips the stack/copy traffic.
            return False
        if (
            group["bf16_method"] == "stochastic_rounding"
            and is_low_precision(p)
            and p.dtype != torch.bfloat16        # fp16+SR is unsupported -> per-param (raises)
        ):
            return False
        if p.ndim > 2:
            # Matrixized conv writes back through a reshaped view -> needs contiguity.
            return p.data.is_contiguous() and p.grad.is_contiguous()
        return True

    def _foreach_buckets(self, params: list[Tensor], group: dict[str, Any]) -> list[tuple]:
        """Bucket ``params`` so each bucket stacks into one tensor, and init their state.

        * ``ndim >= 2`` -> factored bucket, keyed by effective 2-D shape ``[N, R, C]``.
        * ``ndim <= 1`` (biases/norms, 0-D scalars) -> non-factored bucket, keyed
          by element count ``[N, L]`` (a 0-D scalar is a length-1 row, sharing the
          ``L == 1`` bucket with shape-``(1,)`` params).

        Returns ``(chunk_size, plist, states, eff, matrixize, length)`` per bucket —
        factored buckets first, then non-factored, each in first-seen param order (the
        order the pre-cache code stepped them in).
        """
        factored: dict[tuple[Any, ...], tuple[list, list]] = {}
        flat: dict[tuple[Any, ...], tuple[list, list]] = {}
        for p in params:
            state = self.state[p]
            if not state:
                self._init_state(p, state, group)
            g = p.grad
            # DEVICE is part of every bucket key: a bucket is stacked with ``torch.stack``,
            # which refuses to mix devices ("Expected all tensors to be on the same device"),
            # so a group holding a CPU and a CUDA weight of the same shape used to crash the
            # whole step rather than step each on its own device.
            if g.ndim >= 2:
                matrixize = g.ndim > 2  # conv kernels always reshape to 2-D before factoring
                eff = (g.shape[0], g.numel() // g.shape[0]) if matrixize else tuple(g.shape)
                bucket = factored.setdefault((eff, p.dtype, p.device, matrixize), ([], []))
            else:  # ndim <= 1 — 0-D scalars ride as length 1 (g.numel() == g.shape[0] for 1-D)
                bucket = flat.setdefault((g.numel(), p.dtype, p.device), ([], []))
            bucket[0].append(p)
            bucket[1].append(state)
        buckets = [(max(eff[0] * eff[1], 1), pl, st, eff, matrixize, 0)
                   for (eff, _dtype, _device, matrixize), (pl, st) in factored.items()]
        buckets += [(max(length, 1), pl, st, None, False, length)
                    for (length, _dtype, _device), (pl, st) in flat.items()]
        return buckets

    def _foreach_plan(self, params: list[Tensor], group: dict[str, Any],
                      budget: int) -> list[_ForeachChunk]:
        """This step's chunks for ``group``, from the cached plan when it is still valid.

        The plan is rebuilt whenever :func:`_param_witness` moves: the param set (``id``), the
        storage a param points at (``data_ptr``, which also covers dtype and device), or its
        contiguity. Contiguity matters because the plan caches ``p.data`` VIEWS: ``p.data =
        p.data.t()`` on a square weight keeps both id and pointer while moving the strides, and
        the cached view would keep stepping the pre-transpose layout. A SHAPE-changing rebind is
        an unsupported operation (see :func:`_param_witness`); the stale bucketing makes the next
        step raise a size mismatch, which is the intended outcome. Everything else that can
        invalidate the plan is event-driven: see :meth:`_invalidate_fused_caches` and the
        per-parameter fallback in :meth:`_native_dispatch`.
        """
        gid = id(group)
        cached = self._foreach_cache_enabled
        witness = _param_witness(params)
        plan = self._foreach_plans.get(gid) if cached else None
        if plan is None or plan.witness != witness:
            plan = _ForeachPlan(witness, self._foreach_buckets(params, group))
            if cached:
                self._foreach_plans[gid] = plan
        return plan.rechunk(budget, group, cached)

    @torch.no_grad()
    def _step_foreach(self, params: list[Tensor], group: dict[str, Any], budget: int) -> None:
        """Batched step for many params at once.

        Each bucket is stacked into a single tensor and stepped with a handful of kernels —
        element-for-element the same math as :meth:`_step_one_param`. The bucketing and
        every view it derives come from the cached :class:`_ForeachPlan`.
        """
        beta1, beta2 = group["betas"]
        eps1, _eps2 = group["eps"]
        lr, clip = group["lr"], group["clip_threshold"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]
        wd_full = wd != 0 and group["cautious_wd"] == "full"   # see _step_one_param
        codec = self._codec(group)
        for chunk in self._foreach_plan(params, group, budget):
            bucket = self._factored_bucket if chunk.eff is not None else self._nonfactored_bucket
            bucket(chunk, beta1, beta2, eps1, lr, clip, wd, cautious, wd_full, bf16_method, codec)

    @torch.no_grad()
    def _factored_bucket(
        self,
        chunk: _ForeachChunk,
        beta1: float,
        beta2: float,
        eps1: float,
        lr: float,
        clip: float,
        wd: float,
        cautious: bool,
        wd_full: bool,
        bf16_method: str,
        codec: _MomentumCodec,
    ) -> None:
        R, C = chunk.eff  # noqa: N806 — matrix dims (stacked tensor is [N, R, C])
        N = len(chunk.plist)  # noqa: N806
        rows, cols = chunk.rows, chunk.cols

        grad = chunk.grad_stack()                                         # [N, R, C]
        row = torch.stack(rows)                                           # [N, R]
        col = torch.stack(cols)                                           # [N, C]

        # Second-moment EMA weight (fixed beta2). row/col are [N, R]/[N, C], so the
        # scalar broadcasts cleanly.
        omb = 1.0 - beta2

        # Factored second-moment EMA (HF eps placement: eps1 before the means).
        grad_sq = grad * grad
        matvec = self._native_rms_matvec
        if self._eps1_on_means:
            # eps1 goes on the [N,R]/[N,C] MEANS, not on the [N,R,C] square. ``mean(x +
            # eps) == mean(x) + eps``, and the elementwise ``add_`` was a whole extra
            # read-modify-write pass over the bucket (13.1 M elements = ~157 MB of traffic
            # on 200x(256,256)) to compute R+C numbers. The fused reductions
            # (``_chunked_reductions``) always did it this way; this is the native path
            # catching up. Bit-identical at the default ``eps1=1e-30`` with ordinary
            # gradients — 1e-30 is far below the ulp of a grad^2 near 1, so the
            # elementwise add was a literal no-op — and differs by fp32 ulps (rel ~1e-7)
            # only when eps1 is comparable to grad^2 (a large eps1, or gradients ~1e-10),
            # where summing before and after the add rounds differently.
            row_mean, col_mean = grad_sq.mean(dim=-1), grad_sq.mean(dim=-2)
            if eps1 > 0:
                row_mean = row_mean.add_(eps1)
                col_mean = col_mean.add_(eps1)
        else:  # A/B baseline: the pre-0.7.12 eps-on-the-square order.
            # ``add`` (copy) rather than ``add_`` only when ``grad_sq`` is still needed
            # raw by the matvec below; the baseline-baseline arm keeps the in-place cost.
            gsq_eps = (grad_sq + eps1) if (eps1 > 0 and matvec) else (
                grad_sq.add_(eps1) if eps1 > 0 else grad_sq)
            row_mean, col_mean = gsq_eps.mean(dim=-1), gsq_eps.mean(dim=-2)
        row.lerp_(row_mean, omb)
        col.lerp_(col_mean, omb)
        torch._foreach_copy_(rows, list(row.unbind(0)))
        torch._foreach_copy_(cols, list(col.unbind(0)))

        # Reconstruct 1/sqrt(v_hat) = r_factor * c_factor, then clip. Momentum is
        # kept in LR-independent direction units; lr scales the complete delta at
        # the end, so a scheduler moves the CURRENT step instead of being baked
        # into the momentum's history.
        r = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_()                       # [N, R]
        c = col.rsqrt()                                                            # [N, C]
        if matvec:
            # RMS-clip WITHOUT materializing the unclipped update: rms^2 per slice ==
            # sum_i r_i^2 * (gsq @ c^2)_i / n. That is one [N,R,C] READ (the bmm, whose
            # ``gsq`` we already own) in place of a [N,R,C] mul, a norm pass and a div_
            # pass; the clip divisor then folds into ``r`` (an [N,R] multiply) so the
            # reconstruction costs two [N,R,C] passes instead of four. Same identity
            # ``_chunked_reductions``/``_chunked_reductions_batched`` use on the fused
            # path. Differs from the norm-of-the-update form by fp32 ulps (rel 1-2e-7):
            # the two sum the same terms in different orders. NOTE it is a matmul, so a
            # process that has switched ``torch.backends.cuda.matmul.allow_tf32`` on
            # trades ~3 mantissa bits of the CLIP DIVISOR (never of the update itself);
            # the fused path has had that property since 0.7.7.
            rms = (r * r).mul_(
                torch.bmm(grad_sq, (c * c).unsqueeze(-1)).squeeze(-1)
            ).sum(-1).div_(R * C).sqrt_()                                          # [N]
            r = r.mul_(rms.div_(clip).clamp_(min=1.0).reciprocal_().unsqueeze(-1))
            update = grad.mul(r.unsqueeze(-1)).mul_(c.unsqueeze(-2))               # [N, R, C]
        else:  # A/B baseline: build the update, then norm it and divide it down.
            update = grad.mul(r.unsqueeze(-1)).mul_(c.unsqueeze(-2))               # [N, R, C]
            rms = update.reshape(N, -1).norm(2, dim=1) / math.sqrt(R * C)          # per-slice RMS
            update.div_(rms.div_(clip).clamp_(min=1.0).view(N, 1, 1))

        if beta1 > 0:
            # The codec owns every dtype's dequant → fp32 EMA → requant detail; this
            # block is identical for fp32/bf16/int8/4bit (and to _step_one_param).
            delta = codec.ema_stacked(chunk.states, update, chunk.mat, (R, C), beta1)  # [N, R, C]
        else:
            delta = update

        if wd != 0 and not wd_full:                  # "masked": decay inside the mask
            p_fp32 = torch.stack(chunk.pviews).float()
            delta = delta.add_(p_fp32, alpha=wd)

        if cautious:
            delta = cautious_batched_(delta, grad)

        if wd_full:                                  # "full": decay outside the mask
            p_fp32 = torch.stack(chunk.pviews).float()
            delta = delta.add_(p_fp32, alpha=wd)

        # lr rides the weight write (``alpha``) instead of a separate ``delta.mul_(lr)``
        # pass over the stacked bucket - see :func:`kaon._backend.subtract_batched_`.
        if self._write_fold_lr:
            subtract_batched_(chunk.pviews, delta, bf16_method, alpha=lr)
        else:                                        # A/B baseline (pre-0.7.12 order)
            delta.mul_(lr)
            subtract_batched_(chunk.pviews, delta, bf16_method)

    @torch.no_grad()
    def _nonfactored_bucket(
        self,
        chunk: _ForeachChunk,
        beta1: float,
        beta2: float,
        eps1: float,
        lr: float,
        clip: float,
        wd: float,
        cautious: bool,
        wd_full: bool,
        bf16_method: str,
        codec: _MomentumCodec,
    ) -> None:
        """Non-factored update (full per-coordinate second moment) for ``ndim <= 1`` params.

        The bulk of a full fine-tune is biases and norm weights. Their update is
        the plain Adam-style ``grad / sqrt(v)`` — no row/col factoring — so a
        bucket of equal-length 1-D tensors stacks to ``[N, L]`` and steps as a few
        kernels. Mirrors the ``not factored`` branch of :meth:`_step_one_param`.

        0-D scalars (LyCORIS ``use_scalar`` gates and friends) join the ``L == 1``
        bucket as length-1 **views** of the same storage (:func:`flat_view`): stacking,
        the codec's write-back and the final subtract all go through the views, so
        the persisted state keeps its original 0-D shape (checkpoint-compatible
        with the per-param path). For ``L == 1`` the RMS clip
        ``|update| / sqrt(1)`` equals the per-param ``rms()`` of a scalar, and the
        cautious mask's per-slice mean is the scalar mask itself — element-for-
        element the same math as :meth:`_step_one_param`.
        """
        N = len(chunk.plist)  # noqa: N806 — matrix dim (stacked tensor is [N, L])
        length = chunk.length
        vs = chunk.vs                                                     # each [L], fp32

        grad = chunk.grad_stack()                                         # [N, L]
        v = torch.stack(vs)                                               # [N, L]

        # Second-moment EMA weight (fixed beta2).
        omb = 1.0 - beta2

        grad_sq = grad * grad
        if eps1 > 0:
            grad_sq = grad_sq.add_(eps1)
        v.lerp_(grad_sq, omb)
        torch._foreach_copy_(vs, list(v.unbind(0)))

        update = grad.mul(v.rsqrt())                                      # [N, L]
        rms = update.norm(2, dim=1) / math.sqrt(length)                   # per-slice RMS
        update.div_(rms.div_(clip).clamp_(min=1.0).view(N, 1))

        if beta1 > 0:
            # Same codec entry point as the factored bucket; mat flattens a 0-D
            # ``m`` to its length-1 view (identity on 1-D) and the effective
            # per-param shape is the 1-D length (so int8 reduces the whole L axis
            # to one scalar scale, 4bit blocks over L).
            delta = codec.ema_stacked(chunk.states, update, chunk.mat, (length,), beta1)  # [N, L]
        else:
            delta = update

        if wd != 0 and not wd_full:                  # "masked": decay inside the mask
            p_fp32 = torch.stack(chunk.pviews).float()
            delta = delta.add_(p_fp32, alpha=wd)

        if cautious:
            delta = cautious_batched_(delta, grad)

        if wd_full:                                  # "full": decay outside the mask
            p_fp32 = torch.stack(chunk.pviews).float()
            delta = delta.add_(p_fp32, alpha=wd)

        if self._write_fold_lr:                      # see _factored_bucket
            subtract_batched_(chunk.pviews, delta, bf16_method, alpha=lr)
        else:
            delta.mul_(lr)
            subtract_batched_(chunk.pviews, delta, bf16_method)

    @torch.no_grad()
    def _step_one_param(self, p: Tensor, group: dict[str, Any]) -> None:
        beta1, beta2 = group["betas"]
        eps1, _eps2 = group["eps"]
        lr, clip = group["lr"], group["clip_threshold"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]

        state = self.state[p]
        if not state:
            self._init_state(p, state, group)

        grad_fp32 = p.grad if p.grad.dtype == torch.float32 else p.grad.float()
        ndim = grad_fp32.ndim
        factored = ndim >= 2

        if factored:
            matrixize = ndim > 2  # conv kernels always reshape to 2-D before factoring
            gv = grad_fp32.reshape(grad_fp32.shape[0], -1) if matrixize else grad_fp32
            update_factored_state(gv, state["row"], state["col"], beta2, eps1)
            r_factor, c_factor = factored_inv_sqrt_factors(state["row"], state["col"])
            update = gv.mul(r_factor).mul_(c_factor)
            if matrixize:
                update = update.view_as(grad_fp32)
        else:
            v = state["v"]
            grad_sq = grad_fp32 * grad_fp32
            if eps1 > 0:
                grad_sq.add_(eps1)
            v.lerp_(grad_sq, 1.0 - beta2)
            update = grad_fp32.mul(v.rsqrt())

        if clip > 0:
            update.div_((rms(update) / clip).clamp_(min=1.0))

        # Single codec call owns dequant → fp32 EMA → requant for every dtype.
        # The stored momentum is an LR-independent direction; lr scales the
        # complete (momentum + weight-decay) delta below.
        delta = self._codec(group).ema_one(state, update, beta1) if beta1 > 0 else update

        # ``cautious_wd`` decides whether the decay rides INSIDE the cautious mask (historical
        # "masked": rejected coordinates get no decay, survivors get it rescaled by 1/keep) or
        # outside it ("full": every coordinate decays by the same lr*wd, only the update is
        # masked — the Cautious Optimizers paper's own placement). With ``cautious=False`` the
        # two branches are the same ``add_`` on an untouched delta, hence bit-identical.
        wd_full = wd != 0 and group["cautious_wd"] == "full"
        if wd != 0 and not wd_full:
            p_fp32 = p.data if p.dtype == torch.float32 else p.data.float()
            delta = delta.add_(p_fp32, alpha=wd)

        if cautious:
            delta = cautious_one_(delta, grad_fp32)

        if wd_full:
            p_fp32 = p.data if p.dtype == torch.float32 else p.data.float()
            delta = delta.add_(p_fp32, alpha=wd)

        # lr rides the write, exactly as the foreach buckets do — the two must fold it the
        # same way or they stop being bit-exact with each other (see subtract_one_).
        if self._write_fold_lr:
            subtract_one_(p, delta, state, bf16_method, alpha=lr)
        else:                                        # A/B baseline (pre-0.7.12 order)
            delta.mul_(lr)
            subtract_one_(p, delta, state, bf16_method)


