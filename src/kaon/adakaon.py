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
    SRSeedState,
    cautious_batched_,
    cautious_one_,
    centralize_grads_,
    ensure_residuals,
    foreach_budget,
    gc_applies,
    init_bf16_state,
    is_low_precision,
    per_param_only_bf16_method,
    rms,
    subtract_batched_,
    subtract_one_,
    validate_bf16_method,
    weight_value,
)
from kaon._compact_kahan import is_compact_kahan, residual_bits
from kaon._factored import factored_inv_sqrt_factors, update_factored_state
from kaon._foreach_plan import (
    ForeachChunk,
    ForeachPlanMixin,
    ForeachSpec,
    WatchedStateMixin,
    state_generation,
)
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


def _grad_unfusable(p: Tensor) -> bool:
    """True when ``p``'s gradient cannot be read by a fused kernel THIS step: it is not
    contiguous, or its dtype is not the param's. See :func:`_demote_unfusable_grads`."""
    g = p.grad
    return g.dtype != p.dtype or not g.is_contiguous()


def _demote_unfusable_grads(
    one_block: list[Tensor], big: list[Tensor], one_dim: list[Tensor], native: list[Tensor],
) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
    """Move any param whose GRAD a fused kernel cannot read onto the native subset FOR THIS STEP.

    Every fused kernel addresses the gradient as ``base + ri*C + ci`` (or ``base + offs``) read
    straight off ``grad.data_ptr()``, TYPED BY THE PARAM's dtype (``LOWP`` casts the grad pointer
    to bf16 exactly when the weight is bf16). Two properties of the gradient break that and
    neither is visible to ``fused_eligible`` / ``fused_1d_eligible``, which only look at the param:

    * layout — a transposed (``grad = x.t()``) or strided (``grad = buf[::2]``) gradient has the
      right shape and the wrong layout, so the kernel silently steps the wrong numbers
      (measured as a ~1e-3 relative divergence from native, no error raised anywhere);
    * dtype — an fp32 gradient on a bf16 weight (``p.grad_dtype = None`` in torch 2.12, or a
      trainer that keeps fp32 grads) is read as bf16 pairs of its fp32 words: NaN/garbage
      weights from the first step, on every route and every ``bf16_method``. The native path
      reads any grad dtype correctly (it upcasts to fp32), so the param simply goes there.

    Both belong to THIS step's gradient (a fresh tensor every backward), so they cannot be
    frozen into the cached routing partition: this runs per step, returns new lists and leaves
    the cache untouched (the callers' ``_fused_demote`` memo keys on the demoted SET, so a grad
    whose dtype changes between steps moves the key). Only the offending tensors move — one
    strided or fp32 grad on one adapter must not cost the rest of its bucket the fused path.
    """
    fused = (one_block, big, one_dim)
    if not any(_grad_unfusable(p) for sub in fused for p in sub):
        return one_block, big, one_dim, native
    kept: tuple[list[Tensor], ...] = ([], [], [])
    demoted: list[Tensor] = []
    for keep, sub in zip(kept, fused, strict=True):
        for p in sub:
            (demoted if _grad_unfusable(p) else keep).append(p)
    return kept[0], kept[1], kept[2], native + demoted


def _widened_grad(p: Tensor) -> bool:
    """An fp32 gradient on a bf16 weight — the one grad/param dtype mismatch Adakaon's fused
    kernels read directly (their ``GF32`` constexpr types the grad pointer fp32 while the weight
    stays bf16; the grad is upcast to fp32 on load either way, as native's ``grad.float()``)."""
    return p.dtype == torch.bfloat16 and p.grad.dtype == torch.float32


def _adakaon_demoted(parts: tuple) -> tuple:
    """Ids of the fused-routed params Adakaon must step natively THIS step.

    :func:`_grad_unfusable` minus the widened case: a non-contiguous grad, or a grad whose dtype
    is neither the param's nor the widened fp32-on-bf16 (fp16 grads, a bf16 grad on an fp32
    weight, ...). A WIDENED grad stays fused — ``GF32`` is one constexpr per launch, resolved from
    the bucket's first grad, so every bf16 weight of a route must agree on its grad dtype: when a
    route mixes bf16-grad and fp32-grad bf16 weights, the fp32-grad ones are demoted as before
    (the rare case; the common ones — every grad fp32, or every grad bf16 — keep the whole route).
    AdaPNM keeps :func:`_grad_unfusable` (its kernels type the grad by the weight)."""
    out: list[int] = []
    for sub in parts[:3]:
        wide: list[int] = []
        narrow = False
        for p in sub:
            g = p.grad
            if not g.is_contiguous():
                out.append(id(p))
            elif g.dtype == p.dtype:
                narrow = narrow or p.dtype == torch.bfloat16
            elif _widened_grad(p):
                wide.append(id(p))
            else:
                out.append(id(p))
        if narrow:
            out += wide
    return tuple(out)


def _demote_by_id(parts: tuple, demoted: tuple) -> tuple:
    """``parts`` with the params whose ids are in ``demoted`` moved onto the native subset."""
    ids = set(demoted)
    kept: tuple[list[Tensor], ...] = ([], [], [])
    moved: list[Tensor] = []
    for keep, sub in zip(kept, parts[:3], strict=True):
        for p in sub:
            (moved if id(p) in ids else keep).append(p)
    return kept[0], kept[1], kept[2], parts[3] + moved


def _gf32(plist: list[Tensor]) -> bool:
    """The ``GF32`` constexpr of a launch over ``plist`` (a bucket of one param dtype): True for a
    bf16 bucket whose grads are fp32. Uniform across the bucket by :func:`_adakaon_demoted`."""
    return _widened_grad(plist[0])


class Adakaon(AutoLRMixin, WatchedStateMixin, ForeachPlanMixin, SRSeedState, Optimizer):
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
            ``"stochastic_rounding"`` (default), ``"kahan8"`` (+1 B/param,
            compact fixed-point Kahan, see ``docs/research/compact-kahan.md``),
            ``"kahan16"`` (+2 B/param, bit-exact fp32 master weight split in two),
            ``"kahan"`` (+2 B/param, legacy per-param only), or
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
        validate_bf16_method(bf16_method)
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
        # ``self.state`` reports every change of state IDENTITY, so no cross-step cache can
        # keep stepping a retired ``m``/``row``/``col``/``v`` (see
        # ``kaon._foreach_plan.WatchedState``). Zero cost per step; reinstalled by
        # ``WatchedStateMixin.__setstate__`` on every load / unpickle / deepcopy.
        self._install_state_watch()
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
        # group id -> (param witness, state-identity generation, bf16_method, one_block, big,
        # one_dim, native). The three leading fields are the cache KEY; read the routes off the END
        # (``entry[-4:]``) so a future field cannot silently break a positional consumer.
        self._fused_part: dict[int, tuple] = {}
        self._fused_demoted: dict[int, tuple] = {}       # group id -> memo of the non-contiguous-grad demotion
        self._fused_ob_caches: dict[int, Any] = {}       # group id -> PointerArrayCache (one-block)
        self._fused_od_caches: dict[int, Any] = {}       # group id -> OneDimPointerCache (1-D)
        self._fused_big_caches: dict[tuple[int, tuple[int, ...], Any], Any] = {}
        # group id -> (the ``big`` route list the buckets were split from, the buckets)
        self._fused_big_buckets: dict[int, tuple[list[Tensor], list[list[Tensor]]]] = {}
        # The native foreach path's own cache — the bucketing + per-chunk view plan — is NOT
        # allocated here: it lives in ForeachPlanMixin as the lazy ``_foreach_plans``
        # property, keyed by group id (see _FOREACH_SPEC below).
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
        if method == "stochastic_rounding" or is_compact_kahan(method):
            for p in params:
                if p.dtype == torch.float16:
                    raise NotImplementedError(
                        f"bf16_method={method!r} does not support torch.float16 parameters "
                        "(implemented for bfloat16 only); use bf16_method='kahan' for fp16 "
                        "parameters, or keep them in fp32"
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
        self._fused_big_buckets.clear()
        self._clear_foreach_plans()

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
        init_bf16_state(p, state, group["bf16_method"])

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
            self._centralize_native(params, group)
            self._native_dispatch(params, group)
        return loss

    @staticmethod
    def _gc_in_step(group: dict[str, Any], p: Tensor) -> bool:
        """Does the native step centralize this gradient ITSELF, in fp32 (instead of
        :func:`kaon._backend.centralize_grads_` in place on ``p.grad``)?

        Under ``kahan8`` / ``kahan16`` the gradient of a low-precision weight is centralized on
        the fp32 copy the update reads, not in its own dtype: ``centralize_grads_`` on a bf16
        ``p.grad`` rounds both the row mean and ``g - mean`` to bf16, an input error of up to
        half a bf16 ulp of the gradient per step that the compensated weight then faithfully
        integrates — the other half (with weight decay) of the gap between a ``kahan16``
        run and its fp32 twin. The fused kernels always did GC in fp32 registers; this makes
        the native routes agree. Free: the update already materializes that fp32 copy (the
        per-param ``p.grad.float()``, the foreach ``grad_stack()``), and it saves the
        separate stack + copy-back ``centralize_grads_`` does. The group's other methods
        (SR, none, legacy kahan) keep the historical in-place bf16 GC, bit for bit — and so
        does ``p.grad`` of a compact-Kahan group: it is left UNcentralized (the step reads
        its centralized copy).

        In-step exactly when the grad OR the weight is not fp32: a bf16 grad on an fp32 weight
        is centralized on its fp32 copy as well (in place it would round the mean to bf16),
        and an fp32 grad on a bf16 weight joins its bucket, which a bf16 weight decides whole
        (see :meth:`_gc32_rows`). An fp32 grad on an fp32 weight keeps the in-place GC.
        Per param, on the per-param path and in :meth:`_centralize_native`; the foreach bucket
        resolves it per row in :meth:`_gc32_rows`, so both paths pick the same set."""
        return (group["gradient_centralization"] and is_compact_kahan(group["bf16_method"])
                and (p.dtype != torch.float32 or p.grad.dtype != torch.float32))

    def _gc32_rows(self, group: dict[str, Any], plist: list[Tensor]) -> Any:
        """The in-step fp32 GC decision of one foreach bucket, row for row the same as
        :meth:`_gc_in_step` (so every row is centralized exactly once, in place by
        :meth:`_centralize_native` or on the stacked copy): ``False`` (no row), ``True`` (every
        row) or a per-row bool list for a MIXED bucket.

        A bucket shares its param dtype (the plan keys on it), so a bf16/fp16 bucket is
        ``True`` whole in one compare. Only an fp32-weight bucket of a compact-Kahan + GC group
        looks at its grads — one dtype compare per row, in a group configuration (fp32 weights
        under kahan8/kahan16) that the default setups do not produce."""
        if not (group["gradient_centralization"] and is_compact_kahan(group["bf16_method"])):
            return False
        if plist[0].dtype != torch.float32:
            return True
        rows = [p.grad.dtype != torch.float32 for p in plist]
        if not any(rows):
            return False
        return True if all(rows) else rows

    def _centralize_native(self, params: list[Tensor], group: dict[str, Any]) -> None:
        """Gradient Centralization for a native subset: in place on ``p.grad``, except the
        gradients the step centralizes itself in fp32 (:meth:`_gc_in_step`)."""
        if not group["gradient_centralization"]:
            return
        if is_compact_kahan(group["bf16_method"]):
            params = [p for p in params if not self._gc_in_step(group, p)]
        centralize_grads_(params)

    @torch.no_grad()
    def _native_dispatch(self, params: list[Tensor], group: dict[str, Any]) -> None:
        """The native (non-fused) step over ``params`` — foreach batching where eligible, else
        per-param. Gradient Centralization is the caller's responsibility (done per-subset)."""
        if not params:
            return
        if self._foreach and self._group_foreach_eligible(group):
            chunk_budget = foreach_budget(
                self._foreach_stack_budget, self._foreach_batch_cutoff,
                _STACK_BYTES_PER_ELEM + self._ck_bits(group) // 8,  # + residual stack (0/1/2 B)
                params[0].device,
            )
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
        self._drop_foreach_plan(group)
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
            # ONE pass over the grads: the None filter, the sparse check and whether any grad is
            # one a fused kernel cannot read (dtype != param's, or non-contiguous — see
            # _demote_unfusable_grads). ``unfusable`` False (the steady state) lets _fused_demote
            # skip its own sweep; True for a param routed native anyway is harmless (the sweep
            # then moves nothing). Every grad is checked, never one per bucket: a bucket can mix
            # grad dtypes.
            #
            # An fp32 grad on a bf16 weight (``wide``) is READ by the fused kernels (``GF32``), so
            # it only needs the sweep when the group also has a bf16 weight with a bf16 grad —
            # checked lazily, so the default all-bf16 steady state pays nothing new here.
            params: list[Tensor] = []
            unfusable = False
            wide = False
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                if g.is_sparse:
                    raise RuntimeError("Adakaon does not support sparse gradients")
                params.append(p)
                if g.dtype != p.dtype:
                    if g.dtype == torch.float32 and p.dtype == torch.bfloat16:
                        wide = True
                    else:
                        unfusable = True
                if not g.is_contiguous():
                    unfusable = True
            if wide and not unfusable:
                unfusable = any(p.dtype == torch.bfloat16 and p.grad.dtype == torch.bfloat16
                                for p in params)
            parts = self._fused_partition(group, params, ft)
            # Routing is cached; grad CONTIGUITY is not cacheable (fresh tensor every backward).
            one_block, big, one_dim, native = self._fused_demote(id(group), parts,
                                                                 unfusable)
            if native:  # GC for the native subset (fused subsets centralize in-kernel / in-reductions)
                self._centralize_native(native, group)
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
            self._centralize_native(big, group)
            self._native_dispatch(big, group)
            return
        # Group by EXACT shape (and dtype and DEVICE); same-shape buckets of >=2 take the
        # batched chunked kernel, lone tensors take the per-tensor chunked kernel.
        for plist in self._big_shape_buckets(id(group), big):
            # One device scope per bucket, covering every launch inside the chunked steps (see
            # _fused_one_block): the bucket's pointer arrays and scratch live on plist[0].device,
            # and a Triton launch targets the CURRENT device. PLAUSIBLE, not verified — one GPU here.
            with torch.cuda.device(plist[0].device):
                if (len(plist) >= 2 or group["betas"][0] == 0.0
                        or self._fused_big_lone_batched):
                    self._chunked_step_batched(plist, group, ft)
                else:
                    self._chunked_step(plist[0], group, ft)

    def _big_shape_buckets(self, gid: int, big: list[Tensor]) -> list[list[Tensor]]:
        """``big`` split into same-shape/dtype/device buckets, memoized per group.

        Same trick, and the same contract, as :meth:`_fused_demote`'s memo. Rebuilding
        these lists every step was not the cost — it is what the cost was *made of*: a
        fresh list means :class:`~kaon._fused_triton.BigPointerCache` cannot recognize
        it (``built_from`` is identity), so it had to revalidate with ``stale()``, i.e.
        a full :func:`~kaon._foreach_plan.param_witness` tuple over the bucket, once per
        bucket per step, on top of the one :meth:`_fused_partition` already ran for the
        whole group. On an 80-param UNet-shaped bag with four big shapes that is five
        witness sweeps per step where one suffices; on the 428-param LoRA bag the sweep
        alone is ~25 µs.

        VALIDITY is ``big``'s list IDENTITY, which is exactly as strong as recomparing
        the witness because of what produces that object: :meth:`_fused_partition`
        returns the very same list only while ids, ``data_ptr``s and contiguity all
        hold for the group's whole param set (so a rebind to fresh storage, a dtype or
        device change, a ``p.data.t()``, a param entering or leaving the set — including
        via ``p.grad = None`` — all hand back a FRESH list), and :meth:`_fused_demote`
        rebuilds its lists whenever the non-contiguous-grad set moves. Shape is the one
        field neither watches, and a shape-changing rebind is unsupported for the same
        reason everywhere else (see :meth:`_fused_partition`). Event-driven drops
        (checkpoint load, state reset) go through :meth:`_invalidate_fused_caches`,
        which clears this memo with the rest — and the cached list is held here, so its
        ``id`` cannot be reused by a new list while the entry lives.

        WHAT THE ARGUMENT DOES NOT COVER: it is made entirely over PARAMETER identity
        and says nothing about the identity of the STATE. Deleting a parameter's state
        behind the optimizer's back (``del opt.state[p]``, then stepping) leaves every
        route's pointer tables addressing the retired ``row``/``col``/``m`` buffers and
        the step writes them, because no witness anywhere observes ``self.state``. That
        predates this memo and is byte-for-byte identical with and without it (the
        native plan has the same blind spot, via ``ForeachChunk``'s state views); it is
        tracked as its own item, not fixed here. The supported way to discard state is
        :meth:`_invalidate_fused_caches` — which is what ``load_state_dict`` and the
        AutoLR reset call.
        """
        cached = self._fused_big_buckets.get(gid)
        if cached is not None and cached[0] is big:
            return cached[1]
        buckets = list(_same_shape_device_buckets(big).values())
        self._fused_big_buckets[gid] = (big, buckets)
        return buckets

    def _fused_partition(self, group: dict[str, Any], params: list[Tensor], ft: Any) -> tuple:
        """Split a group's params into (one-block, chunked-big, one-dim, native), cached per param-set.

        Keyed on :func:`kaon._fused_triton.param_witness` — ids, ``data_ptr``s and contiguity,
        plus strides under ``ft.SHAPE_WITNESS`` — because each is a routing input the partition
        (and the pointer arrays derived from it) bakes in, and ``p.data = ...`` can change any of
        them while the Parameter object stays the same. An id-only key kept dispatching a stale
        plan at memory the optimizer no longer owns. The FUSED witness is used here rather than
        the native :func:`kaon._foreach_plan.param_witness` deliberately, and the ``ft`` module is
        already a parameter of this method so nothing imports Triton at module scope: see that
        function on why the native plan carries no shape field, and on why this key must be
        exactly as strong as the pointer caches' own — that is what makes their O(1)
        ``_WitnessedCache.built_from`` revalidation sound. With the flag off the two witnesses are
        the same three tuples, so the default path is unchanged, bit for bit and cost for cost.

        A rebuild is where the state geometry gets revalidated (the caches call
        ``ft.check_state_geometry``), so this key moving is what turns a shape-changing rebind
        into a clear error instead of a plan pointed at the wrong buffers.

        SECOND KEY FIELD — the STATE-identity generation
        (:class:`~kaon._foreach_plan.WatchedState`), and it is checked FIRST because it is one
        integer compare against a witness tuple's ``len(params)`` element compares. The param
        witness cannot see a state buffer being retired: ``del opt.state[p]``,
        ``opt.state[p].clear()`` and ``opt.state[p]["m"] = ...`` move no parameter at all, and
        every route's pointer tables went on addressing the retired ``m``/``row``/``col``/``v``
        and writing them (measured on all four routes, and on the native plan through
        ``ForeachChunk.state_views``). ONE check here covers every fused route, because a
        rebuild hands back FOUR FRESH LISTS: ``_fused_demote``'s memo, ``_big_shape_buckets``'
        memo and every ``_WitnessedCache.built_from`` downstream all revalidate by list
        IDENTITY, so they rebuild together — the same argument that lets them skip a second
        witness sweep, now carrying the state as well. The generation is read once per group
        per step and never moves in steady state (``_init_state`` populating a fresh state
        does not move it — see ``WatchedParamState.__setitem__``), so nothing here rebuilds
        for it and the cost is a ``getattr`` plus a list index.

        SHAPE-REBIND LIMIT — ``p.data = p.data.view(...)`` mid-training is REFUSED, never
        supported. ``state["row"]``/``state["col"]`` were allocated for the old effective 2-D
        shape and an EMA has no meaningful migration onto a different factorization. What 0.7.13
        guarantees, at no per-step cost and with no flag, is that the fused path cannot step a plan
        that disagrees with the state: every pointer cache validates the state geometry at build
        (:func:`kaon._fused_triton.check_state_geometry`), so any rebind that moves the witness at
        all — a fresh buffer, a dtype or device change, a transpose, or a reshape bundled with any
        of those — raises with an actionable message instead of pointing the new ``R``/``C`` at the
        old buffers and reading past them (which is what main did: measured, silent, on
        ``p.data = p.data.view(64, 16).clone()``). The native path refuses on its own, out of the
        stack. The recovery is to reshape before constructing the optimizer, or ``del opt.state[p]``
        to restart that parameter's second moment.

        Left over, and the reason ``SHAPE_WITNESS`` exists: a rebind that changes the shape and
        NOTHING else moves no field, so the plan is never rebuilt and never revalidated. That
        weight keeps being stepped as the shape it used to have. For a plain ``view`` it stays in
        bounds (the storage is the same size), so it degrades quality rather than corrupting
        memory, which is why detecting it is opt-in and closing the corruption paths is not. A
        NARROWING rebind (``p.data[:8]``) is the sharper version and the flag does not see it
        either — strides do not move when only ``R`` shrinks; see ``SHAPE_WITNESS`` for what that
        costs a sibling view of the same storage.

        COST on the 428-param LoRA-shaped bag (200x(256,256) + 100x(512,) + 128x 0-D),
        ``benchmarks/fused/bench_shape_witness.py``, paired against the 3-field base: 67 µs for
        the three base fields (median of 15x250 reps, build + compare). The witness runs once per
        group here (the route caches then revalidate by list identity — see
        ``_WitnessedCache.built_from``) plus once per big shape bucket — 2 calls per step on that
        bag, 5 on a UNet-shaped one — and the per-step grad contiguity sweep in
        :meth:`_fused_demote` adds ~75 µs on the same bag. ``SHAPE_WITNESS`` itself is priced on
        the flag, per bag.

        Grad properties deliberately stay OUT of this key: a gradient is a new tensor every
        backward, so its contiguity and dtype are re-checked per step in
        :func:`_demote_unfusable_grads`
        rather than frozen into the routing.
        """
        gid = id(group)
        witness = ft.param_witness(params)
        gen = state_generation(self.state)
        cached = self._fused_part.get(gid)
        bf16m = group["bf16_method"]
        # ``bf16_method`` is a ROUTING input too (a bf16 param is fused-eligible only under SR
        # or compact Kahan), and a group dict can be switched mid-run. Without it in the key a
        # switch kahan16 -> "none" (or legacy "kahan") kept the bf16 params on the cached
        # fused routes, whose kernels then wrote them with stochastic rounding (``SR = lowp
        # and not CK``) — neither the method asked for nor what native does; found by the
        # 0.7.16 stale-residual twin test. One string compare per group per step.
        if (cached is not None and cached[1] == gen and cached[2] == bf16m
                and cached[0] == witness):
            return cached[-4:]
        md, cap = group["momentum_dtype"], self._fused_tile_cap
        one_block: list[Tensor] = []
        big: list[Tensor] = []
        one_dim: list[Tensor] = []
        native: list[Tensor] = []
        momentum = group["betas"][0] > 0
        for p in params:
            # bf16 params need a kernel bf16 write: stochastic rounding or the compact-Kahan
            # (bf16 + residual byte) codec; legacy kahan / none -> native
            bf_ok = (p.dtype != torch.bfloat16) or (bf16m == "stochastic_rounding") \
                or is_compact_kahan(bf16m)
            # ndim>2 (conv) is matrixized to (out, in*kh*kw); every momentum layout is row-major
            # compatible with that view (int8 scales dim-0, 4-bit blocks the same flat storage).
            # The matrixized write-back needs a contiguous GRAD — enforced per step by
            # _demote_unfusable_grads, not here, because the grad changes every backward.
            # The chunked kernels index one tensor in int32 (``k * BLOCK + arange``), so a
            # single weight of >= 2**31 elements stays native instead of wrapping its offsets.
            two_d = bf_ok and p.ndim >= 2 and p.is_cuda and p.is_contiguous() \
                and p.dtype in (torch.float32, torch.bfloat16) \
                and p.numel() < ft.I64_THRESHOLD
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
        self._fused_part[gid] = (witness, gen, bf16m, one_block, big, one_dim, native)
        return one_block, big, one_dim, native

    def _fused_demote(self, gid: int, parts: tuple, unfusable: bool = True) -> tuple:
        """This step's routing, with any tensor whose grad is non-contiguous or not of the
        param's dtype (fp32 grad on a bf16 weight) moved to the native subset.

        Thin memo over :func:`_demote_unfusable_grads`. The demotion has to build fresh
        route lists, and a fresh list means every downstream pointer cache sees a new object and
        rebuilds itself (``_WitnessedCache.built_from``) — so a tensor whose grad is PERSISTENTLY
        strided would throw away the whole bucket's index tensors on every step. Reuse the lists
        while both the partition (identity of its four lists) and the demoted SET are unchanged;
        the check itself runs every step, since that is what detects the change (a grad
        switching dtype between steps moves the demoted set, hence the memo key) — folded into
        the caller's grad pass (``unfusable``), so the steady state pays no second sweep here.
        """
        # ``unfusable`` is the caller's verdict from the grad pass it already makes (False: no
        # grad in the group is unfusable, so there is nothing to sweep for).
        demoted = () if not unfusable else _adakaon_demoted(parts)
        if not demoted:
            self._fused_demoted.pop(gid, None)
            return parts
        cached = self._fused_demoted.get(gid)
        if (cached is not None and cached[0] == demoted
                and all(a is b for a, b in zip(cached[1], parts, strict=True))):
            return cached[2]
        out = _demote_by_id(parts, demoted)
        self._fused_demoted[gid] = (demoted, parts, out)
        return out

    @staticmethod
    def _ck_bits(group: dict[str, Any]) -> int:
        """The compact-Kahan residual width the fused kernels' ``CK`` constexpr takes for this
        group's bf16 buckets (0 = not compact Kahan; fp32 buckets always pass 0)."""
        m = group["bf16_method"]
        return residual_bits(m) if is_compact_kahan(m) else 0

    @staticmethod
    def _fused_sr(group: dict[str, Any], lowp: bool) -> bool:
        """The ``SR`` constexpr of a fused launch: bf16 stochastic rounding exactly when the
        GROUP's method is ``"stochastic_rounding"`` (was ``lowp and not CK``, which read any
        other method as SR). A bf16 bucket whose method no fused kernel writes (``"none"``,
        legacy ``"kahan"``) must never get here — the partition routes it native, keyed on
        the method — so reaching a launch with one is refused instead of written with the
        wrong rounding. For SR and compact-Kahan groups the flags are the ones compiled
        before (``SR=True``/``CK=0`` resp. ``SR=False``/``CK=bits``): same kernel variants."""
        if not lowp:
            return False
        method = group["bf16_method"]
        if method == "stochastic_rounding":
            return True
        if not is_compact_kahan(method):
            raise RuntimeError(
                f"kaon fused step: a bf16 bucket reached a fused kernel under "
                f"bf16_method={method!r}, which no fused kernel implements (it belongs on the "
                "native path — stale routing cache?)"
            )
        return False

    def _ck_prepare(self, plist: list[Tensor], group: dict[str, Any]) -> tuple[int, bool]:
        """``(ck, allocated)`` for a fused launch over ``plist``: the ``CK`` constexpr, and
        whether residuals had to be created or converted (a mid-run switch to kahan8/kahan16,
        or between them — see :func:`kaon._backend.ensure_residuals`), in which case the
        caller rebuilds its pointer cache so the new buffers get a ``c_addr``."""
        ck = self._ck_bits(group)
        if not ck:
            return 0, False
        return ck, ensure_residuals(plist, [self.state[p] for p in plist], ck)

    @staticmethod
    def _c_addr_arg(c_addr: Any, p_addr: Tensor, ck: int) -> Tensor:
        """The residual pointer array a launch passes. With ``CK`` off the kernel never
        dereferences it, so any valid array (``p_addr``) will do; with ``CK`` on a missing
        array is REFUSED — substituting ``p_addr`` would have the kernel write residue bytes
        over the weights (silently: weights up to 1e36 were observed)."""
        if c_addr is not None:
            return c_addr
        if ck:
            raise RuntimeError(
                f"kaon fused step: bf16_method='kahan{ck}' but the bucket's pointer cache carries "
                "no residual ('kahan_lo') array — the cache predates the residuals; rebuild it"
            )
        return p_addr

    def _fused_one_block(self, plist: list[Tensor], group: dict[str, Any], ft: Any) -> None:
        """Launch the one-block pointer-array kernel over the eligible small 2-D weights."""
        for p in plist:
            st = self.state[p]
            if not st:
                self._init_state(p, st, group)
        gid = id(group)
        ck, made = self._ck_prepare(plist, group)
        cache = self._fused_ob_caches.get(gid)
        if cache is not None and (made or (ck and any(bk["c_addr"] is None for bk in cache.buckets
                                                     if bk["lowp"]))):
            cache = None                              # rebuild: residuals are new (B3)
        # ``built_from`` (identity), not ``stale`` (tuple rebuild): _fused_partition already
        # revalidated ids+data_ptrs for the whole group this step and only returns this exact
        # list object while nothing moved. See _WitnessedCache.built_from.
        #
        # ``gen`` is the state-identity generation the tables were baked at, checked here as
        # well as in _fused_partition: this cache is what holds the ``m``/``row``/``col``
        # pointers, so it is where the check belongs (and AdaPNM's ``revalidate`` routes prove
        # the partition's own check does not reach every cache on its own).
        gen = state_generation(self.state)
        if cache is None or not cache.built_from(plist, gen):
            cache = ft.PointerArrayCache(plist, lambda p: self.state[p], None, gen=gen)
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
                    bk["g_addr"], bk["p_addr"],
                    self._c_addr_arg(bk["c_addr"], bk["p_addr"], ck if bk["lowp"] else 0),
                    bk["m_addr"], bk["mscale_addr"], bk["row_addr"],
                    bk["col_addr"], bk["Rs"], bk["Cs"], bk["mscale_n"],
                    lr, b1, b2, eps1, clip, wd, self._t, bk["blk"],
                    # GC is PER BUCKET: a tile of fan-in-1 tensors (BC == 1) has no fan-in
                    # mean to subtract, and centralizing it would zero the gradient. The
                    # predicate is resolved at cache build (``bucket_gc_ok``), so this is a
                    # bool AND, not a shape walk. See kaon._backend.gc_applies.
                    LOWP=bk["lowp"], MOM=bk["mom"], CAUTIOUS=cautious, WD=wd != 0,
                    GC=gc and bk["gc_ok"],
                    SR=self._fused_sr(group, bk["lowp"]), CK=ck if bk["lowp"] else 0,
                    MOMENTUM=bk["momentum"], WDFULL=wd_full, GF32=_gf32(bk["plist"]),
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
        ck, made = self._ck_prepare(plist, group)
        cache = self._fused_od_caches.get(gid)
        if cache is not None and (made or (ck and any(bk["c_addr"] is None for bk in cache.buckets
                                                     if bk["lowp"]))):
            cache = None                              # rebuild: residuals are new (B3)
        gen = state_generation(self.state)
        if cache is None or not cache.built_from(plist, gen):   # see _fused_one_block
            cache = ft.OneDimPointerCache(plist, lambda p: self.state[p], gen=gen)
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
                    bk["g_addr"], bk["p_addr"],
                    self._c_addr_arg(bk["c_addr"], bk["p_addr"], ck if bk["lowp"] else 0),
                    bk["m_addr"], bk["mscale_addr"],
                    bk["v_addr"], bk["Ls"], lr, b1, b2, eps1, clip, wd, self._t,
                    LOWP=bk["lowp"], MOM=bk["mom"], MOMENTUM=bk["momentum"], CAUTIOUS=cautious,
                    WD=wd != 0, SR=self._fused_sr(group, bk["lowp"]), CK=ck if bk["lowp"] else 0,
                    BL=bk["BL"], FBLOCK=bk["block"], WDFULL=wd_full, GF32=_gf32(bk["plist"]),
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
        # ``gc_applies`` (not the raw flag): a fan-in-1 row has no mean to subtract — see
        # kaon._backend.gc_applies. Once per tensor per step on the LONE-big fallback route,
        # where each tensor is megabytes.
        if group["gradient_centralization"] and gc_applies(p.shape):
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
        sr = self._fused_sr(group, p.dtype == torch.bfloat16)
        ck = self._ck_prepare([p], group)[0] if p.dtype == torch.bfloat16 else 0
        g, r, c, inv_rms = self._chunked_reductions(p, group, st)
        quant = md in ("int8", "4bit")
        if quant:
            m_fp32 = self._codec(group).dequant_one(st, torch.empty(R, C, device=p.device)).reshape(R, C)
            mf = m_fp32.reshape(-1)
        else:
            mf = st["m"].reshape(-1)
        keep = torch.zeros(1, dtype=torch.int32, device=p.device)
        gf, pf = g.reshape(-1), p.reshape(-1)
        cf = st["kahan_lo"].reshape(-1) if ck else pf
        grid = ((n + 1023) // 1024,)
        ft._chunked_mom[grid](gf, mf, pf, cf, r, c, keep, C, n, inv_rms, wd, b1,
                              CAUTIOUS=cautious, WD=wd != 0, BLOCK=1024, WDFULL=wd_full, CK=ck)
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
        ft._chunked_apply[grid](gf, mf, pf, cf, n, inv_mean, lr, wd, self._t,
                                CAUTIOUS=cautious, WD=wd != 0, SR=sr, BLOCK=1024, WDFULL=wd_full,
                                CK=ck)

    # ----------------------------------------------- batched chunked (many same-shape big tensors)
    @torch.no_grad()
    def _chunked_reductions_batched(
        self, plist: list[Tensor], group: dict[str, Any], gc: bool
    ) -> tuple:
        """Stacked torch reductions for a same-shape big bucket: GC (on the fp32 copy) + row/col EMA
        + per-tensor rms (matvec, no [N,R,C] beyond grad/gsq). Returns the stacked fp32 grad ``[N,n]``,
        the stacked r/c factors ``[N,R]``/``[N,C]`` (contiguous), and ``inv_rms`` ``[N]`` — the same
        quantities ``_chunked_reductions`` returns per tensor. Mirrors that math exactly (eps1 added to
        the row/col means; rms uses raw gsq).

        ``gc`` is the EFFECTIVE per-bucket flag the caller resolved
        (``group["gradient_centralization"] and cache.gc_ok``), not the raw group flag: GC is
        undefined on a fan-in-1 shape and a bucket is one exact shape — see
        :func:`kaon._backend.gc_applies`."""
        b2, eps1 = group["betas"][1], group["eps"][0]
        clip = group["clip_threshold"]
        N = len(plist)  # noqa: N806
        R, C = plist[0].shape[0], plist[0].numel() // plist[0].shape[0]  # noqa: N806 — conv -> matrixized
        n = R * C
        states = [self.state[p] for p in plist]
        g = torch.stack([p.grad.reshape(R, C) for p in plist]).float()     # [N, R, C]
        if gc:
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
        gc_flag = group["gradient_centralization"]
        lowp = plist[0].dtype == torch.bfloat16
        sr = self._fused_sr(group, lowp)
        ck, made = self._ck_prepare(plist, group) if lowp else (0, False)
        states = [self.state[p] for p in plist]
        cache_key = (id(group), tuple(plist[0].shape), plist[0].dtype, plist[0].device)
        cache = self._fused_big_caches.get(cache_key)
        if cache is not None and (made or (ck and cache.c_addr is None)):
            cache = None                              # rebuild: residuals are new (B3)
        # ``built_from`` (identity), not ``stale`` (a second witness tuple over the bucket):
        # ``plist`` comes from :meth:`_big_shape_buckets`, which hands back the same list
        # object only while :meth:`_fused_partition`'s witness — ids + ``data_ptr``s +
        # contiguity over the whole group, already computed this step — holds. Same contract
        # as _fused_one_block; see _big_shape_buckets and _WitnessedCache.built_from.
        #
        # ``cache.gc`` is part of the validity, not only the param witness: with GC off the
        # cache aliases ``rowmean`` onto ``rowsum`` to save N*R floats, and a group dict
        # flipped mid-run (schedulers do this) would then have the reduction kernel write
        # the row means over the row sums. Nothing moved, so no witness could see it.
        #
        # The comparison is against the EFFECTIVE flag: the cache resolves GC's shape
        # predicate once (``cache.gc_ok``, a fan-in-1 bucket has no GC to do — see
        # kaon._backend.gc_applies), so ``gc_flag and cache.gc_ok`` costs a bool AND per
        # step instead of a shape walk, and a fresh cache is built with the raw flag and
        # applies the predicate itself. ``gen`` is the state generation the tables were
        # baked against (see kaon._foreach_plan.state_generation).
        gen = state_generation(self.state)
        if (cache is None or not cache.built_from(plist, gen)
                or cache.gc != (gc_flag and cache.gc_ok)):
            cache = ft.BigPointerCache(plist, lambda p: self.state[p], R, C, gc=gc_flag, gen=gen)
            self._fused_big_caches[cache_key] = cache
        cache.refresh_grads()
        gc = cache.gc          # the per-bucket constexpr every launch below is given
        gf32 = _gf32(plist)    # fp32 grads on bf16 weights, read in place (GF32)

        if b1 == 0.0:
            self._chunked_step_batched_nomom(
                plist, group, ft, R, C, n, lowp, sr, states, cache, gc
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
                plist, group, ft, R, C, n, lowp, cache, gc
            )
            keep = cache.keep
        else:
            g, r, c, inv_rms = self._chunked_reductions_batched(plist, group, gc)
            keep = cache.keep.zero_()

        p_addr = cache.p_addr
        c_addr = self._c_addr_arg(cache.c_addr, p_addr, ck)
        K = (n + 1023) // 1024  # noqa: N806
        grid = (N * K,)
        direct_4bit = md == "4bit" and states[0]["m_block"] <= 1024 \
            and 1024 % states[0]["m_block"] == 0
        # int8's scale is per ROW, so a chunk may requantize a row in ONE pass only if it owns
        # the whole row: C <= BLOCK and BLOCK % C == 0 (``ft.int8_route`` == "aligned", see
        # _chunked_int8_apply_batched_g). Rows that span chunks take the two-pass cross-program
        # row absmax ("rows", C >= 128); only narrower misaligned rows keep the codec fallback.
        int8_route = ft.int8_route(C) if md == "int8" and self._direct_int8 else "codec"
        direct_int8 = int8_route == "aligned"
        if direct_int8 and fused_red:
            if cautious:
                ft._chunked_int8_keep_batched_g[grid](
                    g_addr, rowmean, cache.m_addr, cache.mscale_addr, p_addr, c_addr, r, c,
                    keep, rms, clip, wd, b1, R, C, n, K,
                    LOWP=lowp, GC=gc, WD=wd != 0, BLOCK=1024, WDFULL=wd_full, CK=ck, GF32=gf32,
                )
            ft._chunked_int8_apply_batched_g[grid](
                g_addr, rowmean, cache.m_addr, cache.mscale_addr, p_addr, c_addr, r, c,
                keep, rms, clip, lr, wd, b1, self._t, R, C, n, K,
                LOWP=lowp, GC=gc, CAUTIOUS=cautious, WD=wd != 0, SR=sr, CK=ck,
                CSEG=C, RPC=1024 // C, BLOCK=1024, WDFULL=wd_full, GF32=gf32,
            )
            return
        if int8_route == "rows" and fused_red and cache.rowmax is not None:
            # Rows span chunks: the two-pass cross-program row absmax (see
            # ``ft._chunked_int8_rowmax_batched_g``). Pass 1 also counts the cautious keep,
            # so this route costs the aligned one's two launches, with or without cautious.
            ft._chunked_int8_rowmax_batched_g[grid](
                g_addr, rowmean, cache.m_addr, cache.mscale_addr, p_addr, c_addr, r, c,
                keep, rms, cache.rowmax, cache.oldscale, clip, wd, b1, R, C, n, K,
                LOWP=lowp, GC=gc, CAUTIOUS=cautious, WD=wd != 0, BLOCK=1024,
                WDFULL=wd_full, CK=ck, GF32=gf32,
            )
            ft._chunked_int8_apply_rows_batched_g[grid](
                g_addr, rowmean, cache.m_addr, cache.mscale_addr, p_addr, c_addr, r, c,
                keep, rms, cache.rowmax, cache.oldscale, clip, lr, wd, b1, self._t,
                R, C, n, K,
                LOWP=lowp, GC=gc, CAUTIOUS=cautious, WD=wd != 0, SR=sr, CK=ck,
                BLOCK=1024, WDFULL=wd_full, GF32=gf32,
            )
            return
        if direct_4bit and fused_red:
            block = states[0]["m_block"]
            if cautious:
                ft._chunked_4bit_keep_batched_g[grid](
                    g_addr, rowmean, cache.m_addr, cache.mscale_addr, p_addr, c_addr, r, c,
                    keep, rms, clip, wd, b1, R, C, n, K,
                    LOWP=lowp, GC=gc, WD=wd != 0, FBLOCK=block, BLOCK=1024, WDFULL=wd_full,
                    CK=ck, GF32=gf32,
                )
            ft._chunked_4bit_apply_batched_g[grid](
                g_addr, rowmean, cache.m_addr, cache.mscale_addr, p_addr, c_addr, r, c,
                keep, rms, clip, lr, wd, b1, self._t, R, C, n, K,
                LOWP=lowp, GC=gc, CAUTIOUS=cautious, WD=wd != 0, SR=sr, CK=ck,
                FBLOCK=block, BLOCK=1024, WDFULL=wd_full, GF32=gf32,
            )
            return

        quant = md in ("int8", "4bit")
        if quant:  # dequant whole bucket to a stacked fp32 temp; kernel m pointers index its slices
            # Batched: dequant_stacked runs the whole bucket's codec in a handful of kernels.
            # The per-tensor dequant_one loop was ~8 tiny torch ops x N tensors, CPU-dispatch
            # bound: measured ~87 ms/step on a 528-tensor LoRA-r32 fleet (4080), vs <2 ms batched.
            #
            # NO ``views=`` HERE, deliberately. The native path hands the codec cached view
            # lists (``ForeachChunk.momentum_views``) because it walks them once per param
            # per step on bags of hundreds of TINY tensors. This is the opposite regime: it
            # is the fallback of the *big* route, so N is small (one shape bucket of weights
            # each above the tile cap) and every one of them is megabytes — the N views the
            # codec rebuilds are noise next to the dequant/requant traffic. It is also cold
            # on top of that: it runs only when the in-Triton momentum routes decline the
            # alignment (int8 with C > 1024 or C not dividing it, 4-bit with m_block > 1024
            # or not dividing a chunk), or with ``_fused_reductions=False``. And ``mat``
            # here is identity, not the ``(R, C)`` layout ``eff`` names, so a cached
            # ``_StackedViews`` would have to be built with a different callback than the
            # one passed — a second, subtly different contract for no measurable gain.
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
                g_addr, rowmean, m_addr, p_addr, c_addr, r, c, keep, rms, clip, wd, b1, R, C, n, K,
                LOWP=lowp, MOM=mom, GC=gc, CAUTIOUS=cautious, WD=wd != 0, BLOCK=1024, WDFULL=wd_full,
                CK=ck, GF32=gf32,
            )
        else:
            ft._chunked_mom_batched[grid](
                g, m_addr, p_addr, c_addr, r, c, keep, inv_rms, wd, b1, R, C, n, K,
                LOWP=lowp, MOM=mom, CAUTIOUS=cautious, WD=wd != 0, BLOCK=1024, WDFULL=wd_full,
                CK=ck, I64=ft.needs_i64(N * K * 1024),
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
                g_addr, rowmean, m_addr, p_addr, c_addr, keep, lr, wd, self._t, R, C, n, K,
                LOWP=lowp, MOM=mom, GC=gc, CAUTIOUS=cautious, WD=wd != 0, SR=sr, CK=ck,
                BLOCK=1024, WDFULL=wd_full, GF32=gf32,
            )
        else:
            inv_mean = (
                (1.0 / (keep.float() / n).clamp_(min=1e-8))
                if cautious else torch.ones(N, device=dev)
            )
            ft._chunked_apply_batched[grid](
                g, m_addr, p_addr, c_addr, inv_mean, lr, wd, self._t, n, K,
                LOWP=lowp, MOM=mom, CAUTIOUS=cautious, WD=wd != 0, SR=sr, CK=ck,
                BLOCK=1024, WDFULL=wd_full, I64=ft.needs_i64(N * K * 1024),
            )

    def _chunked_step_batched_nomom(
        self, plist, group, ft, R, C, n, lowp, sr, states, cache, gc  # noqa: N803
    ) -> None:
        """Chunked big-tensor path for beta1=0 with no momentum allocation.

        ``gc`` is the caller's effective per-bucket flag (``cache.gc``); the native fallback
        below goes through ``centralize_grads_``, which applies the same predicate itself."""
        if not self._fused_reductions:
            self._centralize_native(plist, group)
            self._native_dispatch(plist, group)
            return
        N = len(plist)  # noqa: N806
        lr, wd = group["lr"], group["weight_decay"]
        cautious = group["cautious"]
        wd_full = wd != 0 and group["cautious_wd"] == "full"   # see _step_one_param
        clip = group["clip_threshold"]
        g_addr, rowmean, r, c, rms = self._chunked_reductions_fused(
            plist, group, ft, R, C, n, lowp, cache, gc
        )
        p_addr = cache.p_addr
        gf32 = _gf32(plist)
        ck = self._ck_bits(group) if lowp else 0
        c_addr = self._c_addr_arg(cache.c_addr, p_addr, ck)
        K = (n + 1023) // 1024  # noqa: N806
        grid = (N * K,)
        keep = cache.keep          # already zeroed with colsum/rms (one launch)
        if cautious:
            ft._chunked_nomom_keep_batched_g[grid](
                g_addr, rowmean, p_addr, c_addr, r, c, keep, rms, clip,
                wd, R, C, n, K, LOWP=lowp, GC=gc, WD=wd != 0, BLOCK=1024, WDFULL=wd_full,
                CK=ck, GF32=gf32,
            )
        ft._chunked_nomom_apply_batched_g[grid](
            g_addr, rowmean, p_addr, c_addr, r, c, rms, clip, keep,
            lr, wd, self._t, R, C, n, K, LOWP=lowp, GC=gc,
            CAUTIOUS=cautious, WD=wd != 0, SR=sr, CK=ck, BLOCK=1024, WDFULL=wd_full, GF32=gf32,
        )

    @torch.no_grad()
    def _chunked_reductions_fused(self, plist, group, ft, R, C, n, lowp, cache, gc):  # noqa: N803
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

        ``gc`` is the caller's effective per-bucket flag (``cache.gc``), and it MUST be the
        same value the mom/apply ``_g`` kernels get: those re-apply GC from the ``rowmean``
        this function's reduction kernel writes, so a disagreement would centralize the update
        with a mean the reductions never subtracted. See :func:`kaon._backend.gc_applies`.
        """
        b2, eps1 = group["betas"][1], group["eps"][0]
        N = len(plist)  # noqa: N806
        g_addr = cache.g_addr
        ct = ft.triton.next_power_of_2(C) > ft.REDUCTION_C_CAP   # rows too wide: tiled kernels
        BR, BC, RB = ft.reduction_tile(R, C, cap=ft.REDUCTION_C_TILE if ct else None)  # noqa: N806
        rowmean = cache.rowmean
        rowsum = cache.rowsum
        colsum = cache.colsum
        det = self._deterministic_reductions
        cache.zero_accumulators()          # colsum + rms + keep, one launch
        gf32 = _gf32(plist)
        nw = ft.warps_for(BR * BC)
        if det:
            colpart, rmspart = cache.partials(RB)
            CB = (C + 255) // 256  # noqa: N806
        if ct:
            ft._reduce_rowcol_ct[(N * RB,)](
                g_addr, rowmean, rowsum, colsum, colpart if det else colsum, R, C, RB,
                LOWP=lowp, GC=gc, BR=BR, BC=BC, DET=det, num_warps=nw, GF32=gf32,
            )
            if det:
                ft._reduce_colpart[(N * CB,)](colpart, colsum, C, RB, CB, BCT=256)
        elif det:
            ft._reduce_rowcol_det[(N * RB,)](
                g_addr, rowmean, rowsum, colpart, R, C, RB,
                LOWP=lowp, GC=gc, BR=BR, BC=BC, num_warps=nw, GF32=gf32,
            )
            ft._reduce_colpart[(N * CB,)](colpart, colsum, C, RB, CB, BCT=256)
        else:
            ft._reduce_rowcol[(N * RB,)](
                g_addr, rowmean, rowsum, colsum, R, C, RB,
                LOWP=lowp, GC=gc, BR=BR, BC=BC, num_warps=nw, GF32=gf32,
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
        if ct:
            ft._factor_rowcol_batched_ct[(N,)](
                row_addr, col_addr, rowsum, colsum, r, c, R, C, b2, eps1,
                BR=FR, BC=ft.REDUCTION_C_TILE,
                num_warps=ft.warps_for(max(FR, ft.REDUCTION_C_TILE)),
            )
        else:
            ft._factor_rowcol_batched[(N,)](
                row_addr, col_addr, rowsum, colsum, r, c, R, C, b2, eps1,
                BR=FR, BC=FC, num_warps=ft.warps_for(max(FR, FC)),
            )
        rms = cache.rms
        if ct:
            ft._reduce_rms_ct[(N * RB,)](
                g_addr, rowmean, r, c, rmspart if det else rms, R, C, RB,
                LOWP=lowp, GC=gc, BR=BR, BC=BC, DET=det, num_warps=nw, GF32=gf32,
            )
            if det:
                ft._reduce_rmspart[(N,)](rmspart, rms, RB)
        elif det:
            ft._reduce_rms_det[(N * RB,)](
                g_addr, rowmean, r, c, rmspart, R, C, RB,
                LOWP=lowp, GC=gc, BR=BR, BC=BC, num_warps=nw, GF32=gf32,
            )
            ft._reduce_rmspart[(N,)](rmspart, rms, RB)
        else:
            ft._reduce_rms[(N * RB,)](
                g_addr, rowmean, r, c, rms, R, C, RB,
                LOWP=lowp, GC=gc, BR=BR, BC=BC, num_warps=nw, GF32=gf32,
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
        # torch restores param_groups from the CHECKPOINT's dicts (only ``params`` is carried
        # over from the live optimizer), so a checkpoint written before a group key existed
        # comes back without it and the next step raises KeyError. Back-fill EVERY missing
        # key from the constructor defaults, as the rest of the optimizers do since 0.7.12:
        # this used to name ``cautious_wd`` alone — the one key that had bitten — which left
        # every other key added after a checkpoint was written (``momentum_4bit_block``,
        # ``bf16_method``, ``clip_threshold``, ...) a first-step KeyError on resume.
        # ``setdefault``, so a value the checkpoint DOES carry always wins.
        for g in self.param_groups:
            for key, value in self.defaults.items():
                g.setdefault(key, value)
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
            and not per_param_only_bf16_method(group["bf16_method"])  # kahan needs a per-param shift buffer
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

    # Bucketing, chunking and the cached view plan live in kaon._foreach_plan, shared with
    # AdaBelief / AdamP / AdaMuon / ADOPT / ScheduleFree. ``row``/``col`` (factored) and
    # ``v`` (non-factored) are the state buffers the bucket bodies stack and write back
    # through; ``v`` goes through ``flat_view``, which is what admits 0-D params into the
    # ``L == 1`` bucket as length-1 views. There is no ``extra_key``: every coefficient here
    # is per group per step (fixed beta2, no bias correction), so nothing per-parameter
    # enters the bucket key and the plan survives every step that keeps the param set.
    # The momentum goes through the codec's stacked EMA, whose own per-param view lists
    # (``mat(state["m"])``, the per-row ``m_scale`` views, the requant's write-back
    # targets) come from ``chunk.momentum_views``; ``chunk.view`` is handed over as the
    # codec's ``mat`` argument only as the fallback for a layout those views declined (a
    # non-contiguous buffer). Invalidation is the mixin's witness plus
    # :meth:`_invalidate_fused_caches` (state reset / checkpoint load) and the
    # per-parameter fallback in :meth:`_native_dispatch`. Set ``_foreach_cache_enabled =
    # False`` on an instance to rebuild the plan every step — the A/B switch the cache's
    # speedup is measured with; it is numerically a no-op either way.
    _FOREACH_SPEC = ForeachSpec(
        factored_state=("row", "col"),
        flat_state=("v",),
    )

    @torch.no_grad()
    def _step_foreach(self, params: list[Tensor], group: dict[str, Any], budget: int) -> None:
        """Batched step for many params at once.

        Each bucket is stacked into a single tensor and stepped with a handful of kernels —
        element-for-element the same math as :meth:`_step_one_param`. The bucketing and
        every view it derives come from the cached
        :class:`~kaon._foreach_plan.ForeachPlan`.
        """
        beta1, beta2 = group["betas"]
        eps1, _eps2 = group["eps"]
        lr, clip = group["lr"], group["clip_threshold"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]
        wd_full = wd != 0 and group["cautious_wd"] == "full"   # see _step_one_param
        codec = self._codec(group)
        for chunk in self._foreach_chunks(params, group, budget):
            if chunk.eff is not None:
                # fp32 GC on the stacked copy, row for row as _gc_in_step decides it
                gc32 = self._gc32_rows(group, chunk.plist)
                self._factored_bucket(chunk, beta1, beta2, eps1, lr, clip, wd, cautious,
                                      wd_full, bf16_method, codec, gc32=gc32)
            else:
                self._nonfactored_bucket(chunk, beta1, beta2, eps1, lr, clip, wd, cautious,
                                         wd_full, bf16_method, codec)

    @torch.no_grad()
    def _factored_bucket(
        self,
        chunk: ForeachChunk,
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
        gc32: Any = False,
    ) -> None:
        R, C = chunk.eff  # noqa: N806 — matrix dims (stacked tensor is [N, R, C])
        N = chunk.n  # noqa: N806
        rows, cols = chunk.state_views

        grad = chunk.grad_stack()                                         # [N, R, C]
        if gc32 is not False and C > 1:
            # fp32 GC on the stacked copy (compact Kahan, see _gc_in_step); the fan-in of the
            # matrixized [out, in*kh*kw] layout is the last dim, and C > 1 is gc_applies.
            # A lone bucket's stack of an fp32 grad is a VIEW of ``p.grad``; copy it so the
            # step leaves ``p.grad`` uncentralized (the compact-Kahan contract).
            if N == 1 and grad.data_ptr() == chunk.plist[0].grad.data_ptr():
                grad = grad.clone()
            if gc32 is True:
                grad.sub_(grad.mean(dim=-1, keepdim=True))
            else:  # mixed fp32-weight bucket: only the rows _centralize_native left alone
                idx = torch.tensor([i for i, r in enumerate(gc32) if r], device=grad.device)
                sub = grad.index_select(0, idx)
                grad.index_copy_(0, idx, sub.sub_(sub.mean(dim=-1, keepdim=True)))
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
            del gsq_eps
        if not matvec:
            # Dead from here on: dropping it lets the allocator hand its [N,R,C] block to
            # ``update`` instead of holding both through the EMA/decay/cautious tail (one
            # bucket-sized fp32 buffer off the foreach step's peak).
            del grad_sq
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
            del grad_sq                                  # dead: see the non-matvec branch
            r = r.mul_(rms.div_(clip).clamp_(min=1.0).reciprocal_().unsqueeze(-1))
            update = grad.mul(r.unsqueeze(-1)).mul_(c.unsqueeze(-2))               # [N, R, C]
        else:  # A/B baseline: build the update, then norm it and divide it down.
            update = grad.mul(r.unsqueeze(-1)).mul_(c.unsqueeze(-2))               # [N, R, C]
            rms = update.reshape(N, -1).norm(2, dim=1) / math.sqrt(R * C)          # per-slice RMS
            update.div_(rms.div_(clip).clamp_(min=1.0).view(N, 1, 1))

        if beta1 > 0:
            # The codec owns every dtype's dequant → fp32 EMA → requant detail; this
            # block is identical for fp32/bf16/int8/4bit (and to _step_one_param).
            # ``views`` hands it this chunk's cached per-param view lists (built once);
            # ``chunk.view`` is the ``mat`` fallback for a layout it declined.
            delta = codec.ema_stacked(chunk.states, update, chunk.view, (R, C), beta1,
                                      views=chunk.momentum_views(codec))       # [N, R, C]
        else:
            delta = update

        # The decay reads the weight's full VALUE: under kahan8/kahan16 the decoded
        # (bf16 + residual) value, bit-identical to param_stack() for every other method
        # (see kaon._backend.weight_value).
        ck_stacks = None                             # reused by the write (no second stack)
        if wd != 0 and not wd_full:                  # "masked": decay inside the mask
            p_fp32, ck_stacks = chunk.value_and_stacks(bf16_method)
            delta = delta.add_(p_fp32, alpha=wd)

        if cautious:
            delta = cautious_batched_(delta, grad)

        if wd_full:                                  # "full": decay outside the mask
            p_fp32, ck_stacks = chunk.value_and_stacks(bf16_method)
            delta = delta.add_(p_fp32, alpha=wd)

        # lr rides the weight write (``alpha``) instead of a separate ``delta.mul_(lr)``
        # pass over the stacked bucket - see :func:`kaon._backend.subtract_batched_`.
        if self._write_fold_lr:
            subtract_batched_(chunk.pviews, delta, bf16_method, alpha=lr, sr=self.sr_stream,
                              comp=chunk.cviews, stacked=ck_stacks)
        else:                                        # A/B baseline (pre-0.7.12 order)
            delta.mul_(lr)
            subtract_batched_(chunk.pviews, delta, bf16_method, sr=self.sr_stream,
                          comp=chunk.cviews, stacked=ck_stacks)

    @torch.no_grad()
    def _nonfactored_bucket(
        self,
        chunk: ForeachChunk,
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
        bucket as length-1 **views** of the same storage (:func:`~kaon._backend.flat_view`): stacking,
        the codec's write-back and the final subtract all go through the views, so
        the persisted state keeps its original 0-D shape (checkpoint-compatible
        with the per-param path). For ``L == 1`` the RMS clip
        ``|update| / sqrt(1)`` equals the per-param ``rms()`` of a scalar, and the
        cautious mask's per-slice mean is the scalar mask itself — element-for-
        element the same math as :meth:`_step_one_param`.
        """
        N = chunk.n  # noqa: N806 — matrix dim (stacked tensor is [N, L])
        length = chunk.length
        (vs,) = chunk.state_views                                         # each [L], fp32

        grad = chunk.grad_stack()                                         # [N, L]
        v = torch.stack(vs)                                               # [N, L]

        # Second-moment EMA weight (fixed beta2).
        omb = 1.0 - beta2

        grad_sq = grad * grad
        if eps1 > 0:
            grad_sq = grad_sq.add_(eps1)
        v.lerp_(grad_sq, omb)
        del grad_sq                    # dead: frees its [N, L] block for ``update``
        torch._foreach_copy_(vs, list(v.unbind(0)))

        # rsqrt(v) * grad, in the rsqrt's own buffer: one [N, L] temp instead of two
        # (the product is commutative, so this is bit-identical to ``grad.mul(v.rsqrt())``).
        update = v.rsqrt().mul_(grad)                                     # [N, L]
        rms = update.norm(2, dim=1) / math.sqrt(length)                   # per-slice RMS
        update.div_(rms.div_(clip).clamp_(min=1.0).view(N, 1))

        if beta1 > 0:
            # Same codec entry point as the factored bucket; ``chunk.view`` flattens a
            # 0-D ``m`` to its length-1 view (identity on 1-D) and the effective
            # per-param shape is the 1-D length (so int8 reduces the whole L axis
            # to one scalar scale, 4bit blocks over L).
            delta = codec.ema_stacked(chunk.states, update, chunk.view, (length,), beta1,
                                      views=chunk.momentum_views(codec))          # [N, L]
        else:
            delta = update

        # The decay reads the weight's full VALUE: under kahan8/kahan16 the decoded
        # (bf16 + residual) value, bit-identical to param_stack() for every other method
        # (see kaon._backend.weight_value).
        ck_stacks = None                             # reused by the write (no second stack)
        if wd != 0 and not wd_full:                  # "masked": decay inside the mask
            p_fp32, ck_stacks = chunk.value_and_stacks(bf16_method)
            delta = delta.add_(p_fp32, alpha=wd)

        if cautious:
            delta = cautious_batched_(delta, grad)

        if wd_full:                                  # "full": decay outside the mask
            p_fp32, ck_stacks = chunk.value_and_stacks(bf16_method)
            delta = delta.add_(p_fp32, alpha=wd)

        if self._write_fold_lr:                      # see _factored_bucket
            subtract_batched_(chunk.pviews, delta, bf16_method, alpha=lr, sr=self.sr_stream,
                              comp=chunk.cviews, stacked=ck_stacks)
        else:
            delta.mul_(lr)
            subtract_batched_(chunk.pviews, delta, bf16_method, sr=self.sr_stream,
                          comp=chunk.cviews, stacked=ck_stacks)

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
        if factored and self._gc_in_step(group, p) and gc_applies(p.shape):
            # fp32 GC on the fresh fp32 copy (compact Kahan, see _gc_in_step). An fp32 grad on
            # a bf16 weight IS ``p.grad`` here: copy it, so ``p.grad`` is left uncentralized
            # as on the foreach path (the compact-Kahan contract).
            if grad_fp32 is p.grad:
                grad_fp32 = grad_fp32.clone()
            grad_fp32.sub_(grad_fp32.mean(dim=tuple(range(1, ndim)), keepdim=True))

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
        # The decay reads the weight's full VALUE (decoded under kahan8/kahan16; the
        # historical ``p.data`` / ``p.data.float()`` otherwise) — see kaon._backend.weight_value.
        wd_full = wd != 0 and group["cautious_wd"] == "full"
        if wd != 0 and not wd_full:
            p_fp32 = weight_value(p, state, bf16_method)
            delta = delta.add_(p_fp32, alpha=wd)

        if cautious:
            delta = cautious_one_(delta, grad_fp32)

        if wd_full:
            p_fp32 = weight_value(p, state, bf16_method)
            delta = delta.add_(p_fp32, alpha=wd)

        # lr rides the write, exactly as the foreach buckets do — the two must fold it the
        # same way or they stop being bit-exact with each other (see subtract_one_).
        if self._write_fold_lr:
            subtract_one_(p, delta, state, bf16_method, alpha=lr, sr=self.sr_stream)
        else:                                        # A/B baseline (pre-0.7.12 order)
            delta.mul_(lr)
            subtract_one_(p, delta, state, bf16_method, sr=self.sr_stream)


