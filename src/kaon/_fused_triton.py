"""Triton fused-step building blocks for kaon (experimental — not in the public API).

These are the kernels + host plumbing behind ``Adakaon(fused=True)`` (see ``adakaon.py``, which owns
the step orchestration, partition, and state). One Triton program owns one tensor and runs the whole
factored step in-block, reading each tensor's base address from a MultiTensorApply-style POINTER ARRAY
and writing ``p``/``m`` IN PLACE (no stacking, no scatter) — 18-39x over native foreach on the
launch-bound (many-small / low-rank LoRA) regime; a chunked multi-block path (``_chunked_mom`` /
``_chunked_apply``) handles large tensors (~2.5x). Same math, state, and fidelity as the native path.

────────────────────────────────────────────────────────────────────────────────────────────────
REUSE MAP — for the planned shared "kaon Triton núcleo" that other optimizers will build on.
What is already optimizer-AGNOSTIC vs Adakaon-SPECIFIC here:

  Host-side, reuse as-is for ANY fused optimizer:
    * ``next_pow2_tile`` / ``warps_for``         — tile sizing + launch config
    * ``fused_eligible``                         — the "does one block own this tensor?" predicate
    * ``PointerArrayCache``                      — per-tensor pointer arrays, bucketed by tile,
                                                   cached across steps (grad ptrs refreshed on
                                                   realloc). The hard, reusable plumbing.

  Device-side ``@triton.jit`` helpers (the device-side mirror of ``kaon._momentum_codec``):
    * ``ck_decode`` / ``ck_store`` (compact-Kahan bf16 + residual weight write, ``bf16_method="kahan8"`` /
      ``"kahan16"``; ``ck_ptr`` types a residual pointer-array entry for the width)
    * ``sr_round``  (bf16 stochastic-rounding)   — REUSABLE by every bf16 optimizer (Lion, AdaPNM,
                                                   AdaMuon, …); pure, no Adakaon assumptions.
    * ``dequant_int8`` / ``requant_int8``        — per-row int8 momentum codec, in-kernel. REUSABLE by
                                                   any factored-family fused optimizer (the EMA formula
                                                   between them is the only optimizer-specific part).
    * ``dequant_4bit`` / ``requant_4bit``        — per-128-block 4-bit packed codec (segmented absmax +
                                                   nibble pack via reshape/``tl.split``), in-kernel.
                                                   REUSABLE the same way. 0.5 B/param at fused speed.
    * ``gradient_centralize``                    — GC (subtract per-row fan-in mean). REUSABLE by the
                                                   factored family and any conv optimizer.
    * ``factored_rc``                            — row/col 2nd-moment EMA -> rsqrt r/c factors.
                                                   REUSABLE by Adakaon / AdaPNM / KProdigy.

  4bit packs 2 codes/byte over row-major-flat elements, so the fused path needs an EVEN column count
  (keeps each byte's pair within one row); odd-C tensors route to the native Adakaon.

  Adakaon-SPECIFIC (Adakaon reimplements; another factored optimizer would swap only this):
    * ``_adakaon_tile_kernel`` (one-block) + ``_chunked_mom``/``_chunked_apply`` (big) — the factored
      step (r/c-factor + RMS-clip + EMA + cautious). AdaPNM would add a 2nd (negative) momentum;
      AdaMuon would swap in orthogonalization. ``Adakaon._fused_step`` orchestrates the partition +
      launches over its own state/codec; this module holds no optimizer class.

  Native fallback remains for fp16 parameters, non-contiguous storage, bf16 write modes other than
  stochastic rounding, and small odd-column 4-bit matrices. Convs, 1-D tensors, beta1==0 and large
  tensors above ``TILE_CAP`` all have dedicated fused routes.
────────────────────────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import torch

from kaon._backend import gc_applies
from kaon._stochastic_rounding import SRStream

try:  # Triton is an optional, GPU-only dependency — keep ``import kaon`` working without it.
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised only on triton-less installs
    triton = None
    tl = None
    _HAS_TRITON = False

__all__ = ["fused_eligible", "fused_1d_eligible", "eff_2d", "warps_for", "next_pow2_tile", "TILE_CAP",
           "TILE_CAP_1D", "HAS_TRITON"]

HAS_TRITON = _HAS_TRITON
# Largest padded tile a single program owns — i.e. the ONE_BLOCK vs CHUNKED crossover.
#
# One program == one tensor, so a bag of N tensors is N CTAs no matter how big each one is: past a
# few thousand lanes the tile needs many warps and the whole tensor is serialized inside a single
# CTA, leaving most of the GPU idle (200 tensors = 200 fat CTAs over 36 SMs). The batched chunked
# path splits the same work into N * ceil(n/BLOCK) CTAs and fills the machine instead. The cap was
# 131072 from an era when the alternative was the NATIVE foreach path; 0.7.7 rewrote the chunked
# kernel and moved the crossover down by 16x.
#
# Re-measured on 0.7.7+ kernels (RTX 3000 Ada Laptop, 36 SMs; min of 60 A/B-interleaved reps,
# cautious+GC+wd, bf16 momentum). Ratios are the worse-arm/better-arm over N=50 and N=200 bags of
# same-shape tensors, fp32 and bf16 params (all four configs agree on the crossover):
#     lanes    winner              margin
#      1024    one_block           2.1-2.5x
#      4096    one_block           1.5-2.1x
#      8192    one_block           1.1-1.3x
#     16384    chunked             1.0-1.9x
#     32768    chunked             1.5-1.8x
# End-to-end, cap 131072 -> 8192: 200x (256,256) steps 10.18 -> 2.26 ms fp32 (4.5x) and
# 10.87 -> 1.55 ms bf16 (7.0x); a mixed 2-D bag (50x each of (64,64)...(256,512)) 8.46 -> 1.93 ms
# (4.4x); bags already below the cap (200x (64,64)) are unchanged within noise.
#
# The crossover is an occupancy effect (tensors-per-CTA vs SM count), so the optimum is GPU
# dependent — a part with many more SMs may tolerate a higher cap. It is not hard-coded into the
# routing: ``Adakaon(fused_tile_cap=...)`` / ``AdaPNM(fused_tile_cap=...)`` override this default
# per optimizer. That kwarg means *this* number — the 2-D one_block/chunked crossover — and does
# NOT move the 1-D ceiling below (see :data:`TILE_CAP_1D`).
TILE_CAP = 1 << 13  # 8192 padded lanes

# Largest padded block the NON-FACTORED (ndim <= 1) one-block kernel accepts.
#
# Deliberately a separate constant from :data:`TILE_CAP`, because the two answer different
# questions. For 2-D, exceeding the cap means "hand this to the chunked kernel", which fills the
# machine better — so the cap is an occupancy crossover and wants to be low. For 1-D there is NO
# chunked route: exceeding the cap drops the tensor to the NATIVE foreach path. So this cap is the
# one-program capability bound (how big a block a single program can still own profitably), and
# lowering it past the point where native catches up is a pure loss.
#
# Coupling them is what made 0.7.10's TILE_CAP 131072 -> 8192 a 2.0-2.2x REGRESSION for 1-D tensors
# of length 16384 — a reachable shape (a 14336-wide FFN's norm weight pads to 16384). Measured
# fused-1D vs native, same harness as above:
#     length     fp32                    bf16
#      16384     fused  2.03x faster     fused  2.16x faster
#      32768     native 1.06x faster     fused  1.17x faster
#      65536     native 1.50x faster     native 1.22x faster
#     131072     native 1.23x faster     fused  1.06x faster
# 131072 keeps the pre-0.7.10 1-D behaviour exactly (zero regression, which is the point of the
# split). The data does suggest the true 1-D crossover is nearer 32768 — 65536 favours native by
# 1.2-1.5x — so there is a further win available here, but tightening it is a separate measurement
# with its own regression risk and is intentionally not bundled with the 2-D change.
TILE_CAP_1D = 1 << 17  # 131072 padded lanes
DEV = "cuda"
# Bound methods/getters looked up once: the per-step staleness witness calls each of them per
# param (see :class:`_WitnessedCache`), and the attribute lookup is a measurable share of the cost.
_DATA_PTR = torch.Tensor.data_ptr
_IS_CONTIG = torch.Tensor.is_contiguous
_STRIDE = torch.Tensor.stride


# Opt-in fourth witness field: per-param STRIDES, i.e. shape-change detection (0.7.13).
#
# OFF by default because it cannot be made cheap enough. The witness is a host sweep over every
# param in the group, once per step per group plus once per big shape bucket, and a shape field is
# one more sweep. Measured with ``benchmarks/fused/bench_shape_witness.py`` (paired and
# interleaved) against the step's MIN wall time on the same bag. Ranges span repeat runs on TWO
# machines, both laptop GPUs shared with other jobs — the ORDERING of the candidate fields was
# stable in every run, the absolutes were not, so the spread is given:
#
#     bag                          witness calls/step   added host    % of the min step
#     428 LoRA-ish (200x(256,256)
#       + 100x(512,) + 128x 0-D)   2 (428p + 200p)      +61..65 µs       3.0..3.6%
#     448x 0-D (launch-bound)      1 (448p)             +35..55 µs         8..18%
#     UNet/DiT-ish, 80 params      5 (80p + 4 buckets)  +14..17 µs       1.1..1.7%
#
# Nothing about the FIELD is what makes that expensive — it is the sweep. ``Tensor.stride`` is
# the cheapest form measured on the 428-param bag (+42..65 µs, against +65..104 for
# ``Tensor.size`` and +44 for a ``chain``-flattened stride tuple; ``map(Tensor.stride, pl,
# repeat(0))`` is WORSE at +52 despite allocating no tuple, and a lazy ``all(map(eq, ...))``
# compare is worse still). The floor for ANY per-param field is ``Tensor.dim``, which returns
# cached small ints and cannot even see the bug being guarded: +13..25 µs, already ~1% of the
# 428-param step and >4% of the 0-D one. So no implementation of a per-step shape witness fits a
# 1%-of-step budget, and the 0-D/1-D launch-bound bag — the regime this module spent
# 0.7.10-0.7.12 optimizing — is where it hurts most, because there the host sweep IS the step
# (main's three fields alone are already ~20-25% of that step).
#
# What is always on instead is :func:`check_state_geometry`, which costs nothing per step and
# closes the cases that actually corrupt memory (see its docstring). This flag adds detection of
# a rebind that changes the shape and NOTHING else, so no other field moves — MOST of it, not all:
#
# BLIND SPOT, with the flag on. Strides do NOT see a truncation along dim 0 of a contiguous
# tensor. ``(16,64) -> p.data[:8]`` keeps the strides ``(64, 1)``, the pointer and contiguity, and
# a 1-D ``[:256]`` keeps ``(1,)`` — only ``numel`` moves, and numel is not a field here. Such a
# param keeps being stepped at its PRE-narrowing extent, which means the optimizer writes past the
# param's current ``numel`` *inside the original storage*: measured 324 of the 512 elements beyond
# a narrowed ``(16,64)`` modified, max delta 1.5e-3. So a SIBLING VIEW of the same storage — a
# split QKV, anything from ``chunk()``/``split()`` — is silently rewritten. Neither this flag nor
# ``check_state_geometry`` closes that (the guard never runs, because nothing triggers a rebuild);
# what closes it is the narrowing also moving the storage, which is the common case and is caught.
# ``numel`` is the complementary field and is CHEAPER than strides (+22..32 µs vs +32..65 on the
# 428-param bag) but blind to the reported ``view`` case, which preserves numel. Only
# ``strides + numel`` is complete, at the sum of both costs.
#
# Set the flag when something in the training loop rebinds ``p.data`` (an external EMA, an
# offloader, a resolution-switching harness) and you want a loud failure instead of a weight
# quietly optimized as the shape it used to have::
#
#     import kaon._fused_triton as ft
#     ft.SHAPE_WITNESS = True
#
# Flipping it mid-run is safe: the cached witnesses then differ in ARITY from the fresh ones, so
# every plan rebuilds once (which also runs ``check_state_geometry`` over everything) and the new
# setting takes effect from the next step.
SHAPE_WITNESS = False


def param_witness(plist) -> tuple:
    """The per-step staleness witness for any cached pointer plan over ``plist``.

    Three flat tuples — ``(ids, data_ptrs, contiguity)`` — one entry per param, because each is
    a routing input the caches bake in and none of the others implies it:

    * ``id``        — the param set itself changed.
    * ``data_ptr``  — ``p.data = ...`` rebound the SAME Parameter to different storage (an
      external EMA, a ``.to(dtype/device)``, a block-swap offloader). Covers dtype and device
      too: either necessarily moves the storage.
    * ``contiguity``— ``p.data = p.data.t()`` on a SQUARE weight keeps id, pointer AND shape;
      only the strides move, and the kernels index row-major from ``data_ptr``.

    Plus a FOURTH, ``strides``, when :data:`SHAPE_WITNESS` is on — see that flag for what it
    detects, what it costs, and the one geometry change it does NOT see. It is a shape PROXY, not
    the shape: for a contiguous tensor ``stride(0)`` is the ``C`` of :func:`eff_2d` and the rest
    of the tuple pins every inner dim, so a change to ``C`` or to the dim count moves it — but a
    change to ``R`` alone does not. ``(16,64) -> p.data[:8]`` keeps ``(64, 1)``, and a 1-D
    ``[:256]`` keeps ``(1,)``; catching those needs ``numel`` as well.

    Detection is only half of it either way. Moving the witness REBUILDS the plan, and a rebuilt
    plan would otherwise point the new ``R``/``C`` at the old ``row``/``col`` buffers — so every
    cache calls :func:`check_state_geometry` at build, flag or no flag.

    The native counterpart, :func:`kaon._foreach_plan.param_witness`, is fixed at the three base
    fields and never grows the fourth: the native plan re-stacks by effective shape every step, so
    the shape change raises there on its own. The fourth field is defined here, once, and
    ``Adakaon._fused_partition`` routes off THIS function precisely so the fused key is never
    weaker than the pointer caches it feeds.
    """
    base = (tuple(map(id, plist)), tuple(map(_DATA_PTR, plist)), tuple(map(_IS_CONTIG, plist)))
    if SHAPE_WITNESS:
        return (*base, tuple(map(_STRIDE, plist)))
    return base


def check_state_geometry(plist, state_of, factored: bool) -> None:
    """Refuse a param whose optimizer state no longer describes its current shape.

    Runs at cache BUILD only — never per step — so it is free in steady state and fires exactly
    once, on the step that follows a shape-changing ``p.data`` rebind (:func:`param_witness`
    moves, the partition and every pointer cache rebuild, this runs).

    The state IS the shape. A factored second moment holds ``row[R]`` and ``col[C]`` for the
    effective 2-D shape (:func:`eff_2d`); the non-factored 1-D/0-D route holds ``v[numel]``. A
    ``view``/``reshape`` rebind changes ``(R, C)`` (or the route) while every one of those buffers
    keeps its old length, and an EMA has no meaningful migration onto a different factorization —
    so the contract is REFUSE, not adapt. Without this the rebuilt pointer arrays would carry the
    new ``R``/``C`` against the old buffers and read/write past them.

    ``factored`` picks which invariant applies, i.e. which route the caller built the cache for.
    An EMPTY state is skipped: on a param's first step the caller's ``_init_state`` allocates
    against the shape the param has now, so there is nothing stale to catch.

    Not only rebinds. ``load_state_dict`` reaches here too: a checkpoint saved from ``(512, 16)``
    weights loads onto ``(16, 512)`` ones without complaint (same param count, same numel), and
    the factored state then arrives with ``row=512``/``col=16`` against an effective shape of
    ``(16, 512)`` — 2048 bytes written into a 64-byte ``col``, which left ``p`` non-finite with no
    exception before this check existed.
    """
    bad = []
    for p in plist:
        st = state_of(p)
        if not st:
            continue
        if factored:
            row, col = st.get("row"), st.get("col")
            if p.ndim < 2:                       # a 2-D route with a 0-D/1-D param: stale routing
                bad.append((tuple(p.shape), "ndim<2 on the factored route"))
            elif row is None or col is None:
                bad.append((tuple(p.shape), "no row/col state (built as a 1-D param)"))
            else:
                r, c = eff_2d(p)
                if (row.numel(), col.numel()) != (r, c):
                    bad.append((tuple(p.shape),
                                f"row/col are {row.numel()}/{col.numel()}, this shape needs {r}/{c}"))
        else:
            v = st.get("v")
            if v is None:
                bad.append((tuple(p.shape), "no v state (built as a factored 2-D param)"))
            elif v.numel() != p.numel():
                bad.append((tuple(p.shape), f"v is {v.numel()} elements, this shape needs {p.numel()}"))
    if bad:
        detail = "; ".join(f"{sh}: {why}" for sh, why in bad[:4])
        raise RuntimeError(
            f"kaon fused step: {len(bad)} parameter(s) have an optimizer state that does not "
            f"describe their current SHAPE -- {detail}. Either a shape-changing ``p.data`` rebind "
            "(view/reshape/narrow) mid-training, or a ``load_state_dict`` from a checkpoint whose "
            "shapes differ; neither is supported, because the second moment is allocated for the "
            "old geometry and an EMA cannot be migrated onto a different factorization. "
            "RECOVERY: reshape before constructing the optimizer, or drop that parameter's state "
            "(``del opt.state[p]``) to restart its second moment from the new shape -- that drop "
            "is now OBSERVED (kaon._foreach_plan.WatchedState), so every pointer table and cached "
            "view over it is rebuilt on the next step instead of being left addressing the buffers "
            "you just dropped. NOTE this "
            "step may be PARTIALLY APPLIED: the fused subsets are dispatched in order "
            "(native, one-block, big, 1-D) and the ones before this one already launched, so do "
            "not retry the step -- fix the state and carry on from the next one."
        )

# Momentum storage kinds (passed to the kernel as a constexpr so the unused branches compile away).
MOM_FP32, MOM_BF16, MOM_INT8, MOM_4BIT = 0, 1, 2, 3
# int8 (/127) and 4bit (/7) scale divisors are inlined as literals in the @jit device helpers
# (Triton kernels can't read module globals).


# ============================================================ host-side reusable helpers
def next_pow2_tile(R: int, C: int) -> tuple[int, int]:
    """Padded block tile (BR, BC) for a tensor of shape (R, C). Optimizer-agnostic."""
    return triton.next_power_of_2(R), triton.next_power_of_2(C)


def ptr_array(tensors: list, device) -> torch.Tensor:
    """int64 device array of the tensors' base addresses — the MultiTensorApply-style pointer array a
    batched kernel indexes by program id. Shared by every per-step host launch in the batched paths."""
    return torch.tensor([t.data_ptr() for t in tensors], dtype=torch.int64, device=device)


def bucket_gc_ok(plist: list) -> bool:
    """Whether Gradient Centralization applies to a WHOLE bucket, as one ``tl.constexpr``.

    ``GC`` reaches every kernel as a ``tl.constexpr``: a Triton program cannot call
    :func:`kaon._backend.gc_applies`, and one launch serves one bucket, so the predicate has
    to be uniform across the bucket and resolved on the host. It *is* uniform for every
    bucketing this module uses, but for different reasons, so this checks rather than assumes:

    * the big routes bucket by EXACT shape, so the predicate is trivially constant;
    * the one-block routes bucket by the PADDED tile ``(BR, BC) = next_pow2(R), next_pow2(C)``,
      and ``BC == 1`` exactly when ``C == 1`` — i.e. the fan-in-1 tensors land in tiles of
      their own and can never share one with a tensor GC applies to.

    A mixed bucket has no correct answer (either the fan-in-1 members freeze or the healthy
    ones lose GC), so it raises instead of silently picking one. Called at cache BUILD only —
    a steady-state step reads the cached bool.
    """
    flags = {gc_applies(p.shape) for p in plist}
    if len(flags) != 1:
        raise RuntimeError(
            "fused bucket mixes shapes Gradient Centralization does and does not apply to "
            f"({[tuple(p.shape) for p in plist][:6]}); GC is one tl.constexpr per launch, so "
            "the bucket key must separate them (see kaon._backend.gc_applies)"
        )
    return flags.pop()


def reduction_tile(R: int, C: int, work: int = 16384) -> tuple[int, int, int]:
    """Row-block sizing for the fused reduction kernels: ``(BR, BC, RB)`` — one program owns ``BR``
    rows × ``BC = next_pow2(C)`` cols (≈ ``work`` lanes) and ``RB = ceil(R/BR)`` blocks tile the rows.

    ``BR`` is a POWER OF TWO. It reaches the kernels as a ``tl.constexpr`` driving
    ``tl.arange(0, BR)``, and Triton rejects anything else outright:
    ``CompilationError: arange's range must be a power of 2``. The unrounded row count broke
    the batched-big path for perfectly ordinary weights — ``(96, 96)``, ``(9, 640)``,
    ``(12, 1024)``, ``(65, 65)`` — which simply failed to compile.

    Rounding UP (not down) keeps a program's block at or above the work target. The padding
    is free of numerical consequence: every consumer of this ``BR``
    (:func:`_reduce_rowcol`, :func:`_reduce_rms`, in both Adakaon's and AdaPNM's
    ``_chunked_reductions_fused``) masks its rows with ``ri < R`` on every load, store and
    atomic, and ``RB`` below is derived from the PADDED ``BR`` so the row blocks still tile
    ``R`` exactly once.
    """
    BC = triton.next_power_of_2(C)  # noqa: N806
    BR = triton.next_power_of_2(max(1, min(R, max(1, work // BC))))  # noqa: N806
    RB = (R + BR - 1) // BR  # noqa: N806
    return BR, BC, RB


def eff_2d(p: torch.Tensor) -> tuple[int, int]:
    """The effective 2-D matrix shape ``(R, C)`` the factored step works on.

    For a plain 2-D weight this is ``p.shape``. For an ``ndim>2`` conv kernel
    ``[out, in, kh, kw]`` it is the conv-aware matrixization ``(out, in*kh*kw)`` — and because a
    CONTIGUOUS tensor's row-major storage is identical under that reshape, the fused kernels (which
    index the flat buffer as ``ri*C + ci``) operate on a conv exactly as on the 2-D view, with no
    copy. The row/col second-moment state is already allocated in this matrixized shape (see each
    optimizer's ``_init_state``)."""
    return p.shape[0], p.numel() // p.shape[0]


def warps_for(lanes: int) -> int:
    """num_warps for a per-tensor program owning ``lanes`` padded elements. Optimizer-agnostic."""
    if lanes <= 512:
        return 1
    if lanes <= 2048:
        return 2
    if lanes <= 8192:
        return 4
    if lanes <= 32768:
        return 8
    return 16


def fused_eligible(p: torch.Tensor, tile_cap: int = TILE_CAP) -> bool:
    """Does ONE Triton block own this tensor? (ndim>=2, contiguous, fp32/bf16, fits a tile.)

    Optimizer-agnostic: any single-block fused kernel shares this predicate. ``ndim>2`` conv kernels
    are matrixized to ``(out, in*kh*kw)`` via :func:`eff_2d` (valid because they're contiguous); the
    caller must also confirm the GRAD is contiguous (the matrixized write-back needs it). Everything
    that returns False is routed to the native fallback.
    """
    if p.ndim < 2 or not p.is_cuda or not p.is_contiguous():
        return False
    if p.dtype not in (torch.float32, torch.bfloat16):  # fp16 SR unsupported -> native
        return False
    BR, BC = next_pow2_tile(*eff_2d(p))
    return tile_cap >= BR * BC


def fused_1d_eligible(p: torch.Tensor, tile_cap: int = TILE_CAP_1D) -> bool:
    """Does ONE Triton block own this 1-D tensor? (1-D, contiguous, fp32/bf16, fits a block.)

    The non-factored (full per-coordinate ``v``) Adam step for biases / norm scales. Same
    one-block-per-tensor pointer-array idea as the 2-D path, so a bag of many tiny 1-D tensors
    (the launch-bound regime) steps in one launch instead of a torch-foreach stack. Quant momentum
    (int8/4bit) uses the scalar/per-block form of the same codecs inside the 1-D kernel.

    Note the default is :data:`TILE_CAP_1D`, **not** the 2-D :data:`TILE_CAP`: falling off this
    cap means dropping to the native foreach path, not to the chunked kernel, so the two bounds
    are unrelated. Callers pass no cap — ``fused_tile_cap=`` tunes the 2-D crossover only."""
    if p.ndim != 1 or not p.is_cuda or not p.is_contiguous():
        return False
    if p.dtype not in (torch.float32, torch.bfloat16):
        return False
    return tile_cap >= triton.next_power_of_2(p.shape[0])


# ============================================================ device-side reusable helpers
if _HAS_TRITON:
    from triton.language.extra import libdevice  # libdevice.rint == torch.round (half-to-even)

    @triton.jit
    def sr_round(res, seed, offs):
        """Stochastic-round an fp32 value to bf16 via the int32 bit-trick (== kaon.add_stochastic_).

        REUSABLE device primitive: any bf16-parameter optimizer writes weights through this. Unbiased
        (E[sr_round(x)] == x) for both signs. ``offs`` drives the per-lane noise; vary ``seed`` per step.

        NON-FINITE INPUTS COME BACK BIT-FOR-BIT. Both halves of the trick corrupt them:
        ``ibits + noise`` OVERFLOWS int32 for a NaN with a payload near ``0x7FFFFFFF`` (it lands
        on ``-0.0``), and the ``& 0xFFFF0000`` mantissa truncation alone turns a low-payload NaN
        such as ``0x7F800001`` into ``+inf``. Either way a finite (or wrongly-signalling) weight
        is manufactured out of a NaN, burying a diverged run instead of surfacing it. The
        finiteness policy is PROPAGATE, identically to the torch path, so the ORIGINAL bits are
        returned whenever ``res`` is not finite (``tl.abs`` of a NaN or an inf is never
        ``<= FLT_MAX``). Finite values are untouched — including an fp32 above bf16's largest
        finite, which still carries into inf exactly as ``.to(torch.bfloat16)`` would.
        """
        ibits = res.to(tl.int32, bitcast=True)
        noise = tl.minimum((tl.rand(seed, offs) * 65536.0).to(tl.int32), 65535)
        rounded = (ibits + noise) & -65536  # 0xFFFF0000 as a two's-complement int32
        # 3.4028...e38 is FLT_MAX: the comparison is False for NaN and for +-inf. Inlined
        # because a Triton kernel cannot read a module global.
        finite = tl.abs(res) <= 3.4028234663852886e+38
        return tl.where(finite, rounded, ibits).to(tl.float32, bitcast=True)

    @triton.jit
    def ck_ptr(c_addr, t, BITS: tl.constexpr):
        """The residual pointer of tensor ``t`` in a compact-Kahan pointer array: ``uint8``
        for ``kahan8``, ``int16`` (the uint16 pattern in signed storage) for ``kahan16``."""
        base = tl.load(c_addr + t)
        if BITS == 16:
            return base.to(tl.pointer_type(tl.int16))
        else:
            return base.to(tl.pointer_type(tl.uint8))

    @triton.jit
    def ck_decode(pp, cp, idx, mask, BITS: tl.constexpr):
        """Compact Kahan: the exact compensated fp32 value of ``(bf16 weight, residual)``.

        Integer-only mirror of :func:`kaon._compact_kahan.decode`: the pair IS an fp32 whose
        low ``16 - BITS`` mantissa bits are zero — ``(trunc16 << 16) | (q << (16 - BITS))``
        with ``trunc16 = w16 - (q >> (BITS - 1))`` (the stored bf16 is round-half-away of the
        value, so a set top residual bit means the pattern carried one unit up). No float
        arithmetic touches the residual, so a subnormal residual cannot be flushed.
        ``BITS == 16`` (``kahan16``): ``q`` is the fp32's low half, loaded from int16 storage
        (sign-extended, hence masked) — the pair is then bit for bit the fp32 master.
        """
        w16 = tl.load(pp + idx, mask=mask, other=0.0).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        q = tl.load(cp + idx, mask=mask, other=0).to(tl.int32)
        if BITS == 16:
            q = q & 0xFFFF
        # No carry off a +-0 pattern (an externally zeroed weight with a set top residual
        # bit): it would wrap the magnitude to a NaN pattern — see kaon._compact_kahan.decode.
        carry = tl.where((w16 & 0x7FFF) != 0, q >> (BITS - 1), 0)
        b = ((w16 - carry) << 16) | (q << (16 - BITS))
        return b.to(tl.float32, bitcast=True)

    @triton.jit
    def wd_keep_flat(delta, g, wd, pp, cp, idx, mask, CK: tl.constexpr):
        """``delta + wd * value(p)`` for a cautious KEEP pass (only its sign against ``g``
        matters): the survivor count must see the same ``delta`` the apply pass writes with,
        i.e. the DECODED compact-Kahan value under ``CK``.

        Decoding every lane made the keep pass read the residual too (+1-2 B/elem on a
        memory-bound kernel: ``_chunked_nomom_keep_batched_g`` 200 -> 292 us, retime-016).
        The decision only depends on the decoded value where the bare-bf16 ``x_p = delta +
        wd*p`` is within reach of it: ``|z - p| <= |p|*2^-7`` (+ a subnormal-sized term at
        ``p == 0``), so the residual is loaded (masked load: untouched sectors are not
        fetched) only for lanes with ``|x_p| <= 4*wd*(|p|*2^-7) + 1e-36`` or ``|x_p*g| <
        1e-30`` (product near underflow). Elsewhere ``x_p`` and ``x_z`` have the same sign and
        neither product underflows, so ``keep`` is bit-identical to the full decode.
        Non-finite ``p``/``delta`` give NaN/inf on both sides alike. ``cp``: typed residual
        pointer (ignored without ``CK``)."""
        p = tl.load(pp + idx, mask=mask, other=0.0).to(tl.float32)
        x = delta + wd * p
        if CK:
            thr = 4.0 * wd * (tl.abs(p) * 0.0078125) + 1e-36
            amb = mask & ((tl.abs(x) <= thr) | (tl.abs(x * g) < 1e-30))
            # Block-uniform skip: ambiguous lanes are rare (a sign within ~1% of the decay term
            # of a zero crossing), so most blocks never run the decode at all.
            if tl.max(amb.to(tl.int32)) > 0:
                z = ck_decode(pp, cp, idx, amb, CK)
                x = tl.where(amb, delta + wd * z, x)
        return x

    @triton.jit
    def wd_keep(delta, g, wd, pp, c_addr, t, idx, mask, CK: tl.constexpr):
        """:func:`wd_keep_flat` for tensor ``t`` of a pointer-array launch."""
        if CK:
            return wd_keep_flat(delta, g, wd, pp, ck_ptr(c_addr, t, CK), idx, mask, CK)
        else:
            return delta + wd * tl.load(pp + idx, mask=mask, other=0.0).to(tl.float32)

    @triton.jit
    def ck_store_noise(pp, cp, idx, mask, res, noise, BITS: tl.constexpr):
        """Compact Kahan: store fp32 ``res`` as ``(bf16 weight, residual byte)``, rounding the
        residual with the GIVEN int32 ``noise`` in ``[0, 2**(16-BITS))`` (mirror of
        :func:`kaon._compact_kahan.encode_`). ``ck_store`` draws the noise; this entry point
        exists so a test can enumerate every noise value and require an exactly zero bias.

        The stored bf16 is round-half-away-from-zero of the kept value (the carry ``q >> (BITS-1)``
        on the 16-bit pattern), so the forward pass sees the nearest bf16. Non-finite values
        are cast as-is with a zero residual — PROPAGATE, exactly like ``sr_round``.
        """
        UNIT: tl.constexpr = 1 << (16 - BITS)
        ibits = res.to(tl.int32, bitcast=True)
        finite = tl.abs(res) <= 3.4028234663852886e+38
        br = (ibits + noise) & -UNIT
        q = tl.where(finite, (br >> (16 - BITS)) & ((1 << BITS) - 1), 0)
        w16 = ((br >> 16) & 0xFFFF) + (q >> (BITS - 1))
        w16 = (w16 << 16) >> 16                                   # sign-extend to int16 range
        w = tl.where(finite, w16.to(tl.int16).to(tl.bfloat16, bitcast=True), res.to(tl.bfloat16))
        tl.store(pp + idx, w, mask=mask)
        if BITS == 16:
            tl.store(cp + idx, q.to(tl.int16), mask=mask)         # the uint16 pattern, wrapped
        else:
            tl.store(cp + idx, q.to(tl.uint8), mask=mask)

    @triton.jit
    def ck_store(pp, cp, idx, mask, res, seed, BITS: tl.constexpr):
        """:func:`ck_store_noise` with stochastic rounding: the same ``tl.rand(seed, idx)``
        draw ``sr_round`` uses, scaled to the ``2**(16-BITS)`` dropped bits instead of the 16 a
        bf16 cast drops — unbiased at the finer grid. The clamp to ``UNIT-1`` is a guard, not
        a no-op by contract: ``tl.rand`` is documented as ``[0, 1)`` but its value comes from
        an int->float scaling whose top end sits one fp32 rounding away from 1.0 (the
        current Triton scale keeps it at ``1 - 2**-24``, and a power-of-two ``UNIT`` keeps
        the product exactly below ``UNIT``). Should a draw ever reach 1.0, an unclamped
        noise of ``UNIT`` would round every such value up by one extra grid unit even when
        it sits exactly on the grid; the clamp keeps the noise in ``[0, UNIT)``, the range
        the unbiasedness argument (and the torch path's ``randint``) assumes.

        ``BITS == 16`` (``kahan16``) drops no bit: the value is split exactly, no draw is made
        (the rounding already happened in the fp32 arithmetic that produced ``res``)."""
        if BITS == 16:
            ck_store_noise(pp, cp, idx, mask, res, 0, BITS)
        else:
            UNIT: tl.constexpr = 1 << (16 - BITS)
            noise = tl.minimum((tl.rand(seed, idx) * UNIT).to(tl.int32), UNIT - 1)
            ck_store_noise(pp, cp, idx, mask, res, noise, BITS)

    @triton.jit
    def _sr_axpy_kernel(p_ptr, d_ptr, alpha, n, seed, BLOCK: tl.constexpr):
        """``p += alpha * d`` for a bf16 ``p`` and an fp32 ``d``, stochastically rounded.

        ONE kernel and ZERO temporaries, against the torch path's chain inside
        ``kaon._stochastic_rounding._add_stochastic_bf16_``: ``target.float()`` (a full-size
        fp32 temp), ``add_``, ``randint`` (a full-size int32 temp — 4 B/elem of transient
        noise), ``where``, ``add_``, ``bitwise_and_``, ``copy_``. Same rounding rule and the
        same finiteness policy: ``sr_round`` returns a non-finite input bit-for-bit, which is
        exactly what the torch path's NaN-masked noise achieves.

        The NOISE STREAM differs from the torch path's (Philox via ``tl.rand`` vs a
        ``torch.Generator``), so switching implementations mid-run does not reproduce the
        other's draws. Both are unbiased — ``E[sr(x)] == x``, the only property stochastic
        rounding is relied on for — and the fused kernels have written weights through
        ``tl.rand``-seeded ``sr_round`` since 0.7.5, so this introduces no new kind of
        nondeterminism, only a second place that uses it.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        d = tl.load(d_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(p_ptr + offs, sr_round(p + alpha * d, seed, offs).to(tl.bfloat16), mask=mask)

    @triton.jit
    def _ck_axpy_kernel(p_ptr, c_ptr, d_ptr, alpha, n, seed, BITS: tl.constexpr, BLOCK: tl.constexpr):
        """``(p, lo) += alpha * d`` for a compact-Kahan bf16 ``p`` + residual ``lo`` and an
        fp32 ``d``: ONE launch, ZERO temporaries — the ``kahan8`` twin of ``_sr_axpy_kernel``.
        The torch reference (``kaon._compact_kahan.compensated_add_``) is a dozen integer
        kernels with parameter-sized int32 scratch (measured ~23 B/elem transient on a
        stacked bucket); this reads 3 B/elem and writes 3 B/elem (4 and 4 for ``kahan16``)."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        z = ck_decode(p_ptr, c_ptr, offs, mask, BITS)
        d = tl.load(d_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        ck_store(p_ptr, c_ptr, offs, mask, z + alpha * d, seed, BITS)

    @triton.jit
    def _ck_decode_kernel(p_ptr, c_ptr, out_ptr, n, BITS: tl.constexpr, BLOCK: tl.constexpr):
        """``out = decode(p, lo)`` as fp32, one launch (the torch reference,
        ``kaon._compact_kahan.decode``, is ~10 integer kernels with three int32 temporaries —
        measured +50-80% self-CUDA on a foreach Adakaon step when the weight-decay read went
        through it). Integer-identical: ``ck_decode`` is the same bit manipulation."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, ck_decode(p_ptr, c_ptr, offs, mask, BITS), mask=mask)

    @triton.jit
    def dequant_int8(code_ptr, idx, mask, scale_ptr, rr, R):
        """Per-row int8 momentum codes -> fp32. REUSABLE by any factored-family fused optimizer.

        ``code_ptr`` is the int8 [R,C] codes; ``scale_ptr`` the fp32 [R] per-row absmax scales.
        Mirrors ``kaon._momentum_codec._Int8Codec`` dequant (code * row_scale)."""
        scr = tl.load(scale_ptr + rr, mask=rr < R, other=0.0)          # [BR] per-row scale
        code = tl.load(code_ptr + idx, mask=mask, other=0).to(tl.float32)
        return code * scr[:, None]

    @triton.jit
    def requant_int8(m_new, m2, code_ptr, idx, scale_ptr, rr, R):
        """fp32 momentum -> per-row int8 codes + scale, stored in place. REUSABLE.

        Per-row (dim-0) absmax / 127, round half-to-even (libdevice.rint == torch.round), clamp
        [-127, 127]. Element-for-element identical to ``_Int8Codec`` requant."""
        amax = tl.max(tl.where(m2, tl.abs(m_new), 0.0), axis=1)        # [BR] per-row absmax
        amax = tl.where(amax < 1e-12, 1e-12, amax)
        new_scale = amax / 127.0                                       # symmetric int8 -> [-127, 127]
        q = libdevice.rint(m_new / new_scale[:, None])
        q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
        tl.store(code_ptr + idx, q, mask=m2)
        tl.store(scale_ptr + rr, new_scale, mask=rr < R)

    @triton.jit
    def dequant_4bit(packed_ptr, scale_ptr, ri, ci, idx, Chalf, mask, BLK, NS):
        """Per-block 4-bit packed momentum -> fp32. REUSABLE by any factored-family fused optimizer.

        Nibble-packed (2 codes/byte) over the row-major-flattened tensor with a per-block absmax
        scale; assumes an EVEN column count so a byte's pair stays within one row. Mirrors
        ``kaon._momentum_codec._FourBitCodec`` dequant (unpack nibble - 8, * block scale).

        ``NS`` is the CAPACITY of ``scale_ptr`` (the caller's ``m_scale.numel()``): the block index
        ``idx // BLK`` is derived from ``BLK``, not from the stored layout, so a mismatch reads
        past the buffer. Bounding the load keeps that an out-of-range 0 instead of whatever the
        allocator left there. See :func:`requant_4bit` for the write side."""
        byte = tl.load(packed_ptr + (ri * Chalf + ci // 2), mask=mask, other=0)
        nib = tl.where((ci % 2) == 0, byte & 0xF, (byte >> 4) & 0xF)
        q = nib.to(tl.float32) - 8.0
        blk = idx // BLK
        sc = tl.load(scale_ptr + blk, mask=mask & (blk < NS), other=0.0)   # per-block scale
        return q * sc

    @triton.jit
    def requant_4bit(m_new, m2, idx, R, C, Chalf, packed_ptr, scale_ptr, NB, NS, BLK,
                     BR: tl.constexpr, BC: tl.constexpr,
                     EXACT: tl.constexpr = False, FBLK: tl.constexpr = 0):
        """fp32 momentum -> per-block 4-bit codes + scale, stored in place. REUSABLE.

        Pass 1: segmented per-block absmax / 7 (a runtime loop over the tensor's ``NB`` blocks).
        Pass 2: round half-to-even (libdevice.rint) + clamp [-7, 7] + 8 shift -> nibbles, packed
        two-per-byte via reshape + ``tl.split`` (no cross-lane write hazard). Element-identical to
        ``_FourBitCodec``.

        ``NS`` is the CAPACITY of ``scale_ptr`` in fp32 elements — the caller's
        ``state["m_scale"].numel()`` for this tensor — and it bounds BOTH scale accesses. What it
        protects against is precisely an ``m_scale`` SHORTER than the ``NB`` blocks this kernel
        addresses: ``NB`` follows from ``BLK``, so such a caller would write ``NB - NS`` floats past
        the end of the buffer and read the missing blocks back out of it. It does NOT make a
        different block LAYOUT correct — a state quantized with a smaller ``momentum_4bit_block``
        has its scales at other offsets entirely. The caller's job is to pass THIS tensor's real
        block: ``Adakaon`` does it by bucketing :class:`PointerArrayCache` on ``state["m_block"]``
        and handing the kernel that block as a runtime scalar, so ``NS == NB`` here by
        construction; the masks are the second line of defence that turns a future bucketing
        mistake into dropped stores and a neutral 1.0 scale instead of memory corruption."""
        blk = idx // BLK
        am = tl.where(m2, tl.abs(m_new), 0.0)
        if EXACT:
            # SINGLE-REDUCTION FAST PATH (0.7.12). The general loop below runs ``NB`` times
            # over the WHOLE tile — O(numel * NB) — which measured 1.8x (64,128), 1.9x
            # (128,64) and 3.2x (16,512) slower than the bf16 momentum path at NB=64, against
            # only 1.07x at NB=16: the signature of exactly that product.
            #
            # A block is a segment of the FLAT ``ri*C + ci`` index, so it coincides with a
            # reshape of the ``[BR, BC]`` tile only when the tile is UNPADDED (``BR == R`` and
            # ``BC == C``) — then ``ri*C + ci == ri*BC + ci`` is the tile's own row-major
            # order. That is a per-tensor property and the reshape needs constexpr extents, so
            # the host decides it per BUCKET (``PointerArrayCache``: every tensor in the
            # bucket has the tile as its exact shape) and passes ``EXACT``/``FBLK``.
            #
            # ``FBLK`` MUST DIVIDE ``BR*BC`` — the reshape below is otherwise invalid and Triton
            # raises a ``CompilationError`` at launch (verified:
            # ``test_exact_requant_rejects_a_block_that_does_not_divide_the_tile``). Since 0.7.12
            # ``FBLK`` is the bucket's real ``momentum_4bit_block``, not a hardcoded 128, so that
            # divisibility is no longer automatic: :class:`PointerArrayCache` is what enforces it
            # (``(BR*BC) % blk == 0`` in its ``exact4`` predicate) and falls back to the general
            # loop below otherwise. ``BR*BC`` is a power of two, so any divisor of it is one too
            # and ``nb`` stays a legal Triton extent.
            nb: tl.constexpr = (BR * BC) // FBLK
            seg = tl.max(tl.reshape(am, (nb, FBLK)), axis=1)           # [nb] per-block absmax
            seg = tl.where(seg < 1e-12, 1e-12, seg) / 7.0
            bb = tl.arange(0, nb)
            tl.store(scale_ptr + bb, seg, mask=bb < NS)
            # Broadcast the per-block scale back over the tile in the same order.
            sc = tl.reshape(tl.broadcast_to(seg[:, None], (nb, FBLK)), (BR, BC))
        else:
            for b in range(NB):                                        # segmented per-block absmax
                bmax = tl.max(tl.where(blk == b, am, 0.0))
                bmax = tl.where(bmax < 1e-12, 1e-12, bmax)
                tl.store(scale_ptr + b, bmax / 7.0, mask=b < NS)       # symmetric 4-bit -> [-7, 7]
            # ``blk < NS`` for the same reason as the store above; the neutral 1.0 keeps the
            # quantization defined (and finite) for a block whose scale was never written.
            sc = tl.load(scale_ptr + blk, mask=m2 & (blk < NS), other=1.0)  # per-lane block scale
        q = libdevice.rint(m_new / sc)
        q = tl.minimum(tl.maximum(q, -7.0), 7.0)
        # Canonical odd-length padding matches ``_pack_nibbles``: the unused
        # high nibble is zero, never an arbitrary quantized value.
        nib = tl.where(m2, (q + 8.0).to(tl.uint8), 0)                   # [BR, BC]
        lo, hi = tl.split(tl.reshape(nib, (BR, BC // 2, 2)))           # pair adjacent columns
        byte = lo | (hi << 4)                                         # [BR, BC//2]
        rr = tl.arange(0, BR)[:, None]
        jj = tl.arange(0, BC // 2)[None, :]
        tl.store(packed_ptr + rr * Chalf + jj, byte, mask=(rr < R) & (jj < Chalf))

    @triton.jit
    def gradient_centralize(g, m2, Cf):
        """Gradient Centralization: subtract each row's mean over the fan-in (C) axis.

        REUSABLE by the whole factored-Adam family (and any conv optimizer). ``m2`` masks padded
        lanes to 0 so they don't bias the mean and stay 0 after."""
        gmean = tl.sum(g, axis=1) / Cf
        return tl.where(m2, g - gmean[:, None], 0.0)

    @triton.jit
    def factored_rc(g, rowp, colp, rr, cc, R, C, Rf, Cf, beta2, eps1):
        """Factored second-moment EMA (row/col) -> the rsqrt reconstruction factors (r, c).

        REUSABLE by Adakaon / AdaPNM / KProdigy (every factored-Adam optimizer). Updates the row/col
        EMA state in place (HF eps placement) and returns ``(r_factor [BR], c_factor [BC])`` such that
        1/sqrt(v_hat)[i,j] == r_factor[i] * c_factor[j]. Mirrors ``kaon._factored``."""
        gsq = g * g
        row_mean = tl.sum(gsq, axis=1) / Cf + eps1
        col_mean = tl.sum(gsq, axis=0) / Rf + eps1
        omb = 1.0 - beta2
        row_new = tl.load(rowp + rr, mask=rr < R, other=0.0)
        row_new = row_new + omb * (row_mean - row_new)
        col_new = tl.load(colp + cc, mask=cc < C, other=0.0)
        col_new = col_new + omb * (col_mean - col_new)
        tl.store(rowp + rr, row_new, mask=rr < R)
        tl.store(colp + cc, col_new, mask=cc < C)
        row_mean_all = tl.sum(tl.where(rr < R, row_new, 0.0)) / Rf
        return tl.rsqrt(row_new / row_mean_all), tl.rsqrt(col_new)

    @triton.jit
    def _adakaon_tile_kernel(
        g_addr, p_addr, c_addr, m_addr, mscale_addr, row_addr, col_addr, Rs_ptr, Cs_ptr, Ns_ptr,
        lr, beta1, beta2, eps1, clip, wd, seed, m_blk,
        LOWP: tl.constexpr, MOM: tl.constexpr, MOMENTUM: tl.constexpr,
        CAUTIOUS: tl.constexpr, WD: tl.constexpr,
        GC: tl.constexpr, SR: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
        EXACT: tl.constexpr = False, FBLK: tl.constexpr = 0,
        WDFULL: tl.constexpr = False, CK: tl.constexpr = 0,
    ):
        """One program == one tensor. Whole factored Adakaon step, in place via pointer-array.

        Padded lanes are masked to 0 so the reductions and the 0*inf factor corners stay finite.

        ``m_blk`` is the 4-bit absmax block size (``state["m_block"]``) as a RUNTIME scalar, one
        per launch: :class:`PointerArrayCache` buckets by it, so every tensor in the launch shares
        it. It used to be a hardcoded ``min(R*C, 128)``, which forced every other
        ``momentum_4bit_block`` off this route and onto the native path. Ignored (and never read)
        for every non-4-bit momentum — the branch is constexpr-elided — where the host passes 0.

        Runtime, not ``constexpr``, so the KERNEL BODY does not specialize per block: Triton
        keys an int argument only on ``== 1`` and ``% 16 == 0``. Measured on a padded-tile bucket
        (``EXACT`` off), blocks 128/64/256/0/32 compile to **one** shared variant. What does
        specialize is the ``EXACT`` fast path's ``FBLK``, which has to be constant for
        ``requant_4bit``'s ``(nb, FBLK)`` reshape: on an unpadded power-of-two tile each distinct
        block size costs **+1** variant of this kernel (measured 1 per block over 128/64/256/0/32).
        A single training run configures one block size, so in practice that is +0.
        """
        t = tl.program_id(0)
        R = tl.load(Rs_ptr + t)
        C = tl.load(Cs_ptr + t)
        Rf = R.to(tl.float32)
        Cf = C.to(tl.float32)

        gi = tl.load(g_addr + t)
        pi = tl.load(p_addr + t)
        mi = tl.load(m_addr + t)
        rowp = tl.load(row_addr + t).to(tl.pointer_type(tl.float32))
        colp = tl.load(col_addr + t).to(tl.pointer_type(tl.float32))
        if LOWP:
            gp = gi.to(tl.pointer_type(tl.bfloat16))
            pp = pi.to(tl.pointer_type(tl.bfloat16))
        else:
            gp = gi.to(tl.pointer_type(tl.float32))
            pp = pi.to(tl.pointer_type(tl.float32))

        ri = tl.arange(0, BR)[:, None]
        ci = tl.arange(0, BC)[None, :]
        rr = tl.arange(0, BR)
        cc = tl.arange(0, BC)
        m2 = (ri < R) & (ci < C)
        idx = ri * C + ci
        g = tl.load(gp + idx, mask=m2, other=0.0).to(tl.float32)
        if CK == 8:
            # Where the decode sits is a MEASURED choice (same formula anywhere, but the
            # compiler's FMA contraction of the later terms can follow the placement: kahan16
            # stays late, where it is bit-exact to the fp32 variant). kahan8: UP FRONT,
            # so its loads overlap the second-moment and momentum math — decoding where the
            # decay first needs it (before the cautious tl.sum) put the residual load + integer
            # decode on the critical path of this latency-bound kernel: +27% (4-bit) / +36% (no
            # momentum) on a LoRA bag with wd>0 + cautious (retime-016); up front it is at or
            # below 0.7.15. kahan16: up front measured +19-23% (the 16-bit residual tile held
            # across the whole kernel), late +1% / +8%, so it decodes late (below).
            cp = ck_ptr(c_addr, t, CK)
            zc = ck_decode(pp, cp, idx, m2, CK)

        # --- REUSABLE (factored family): GC + row/col second moment -> r/c factors ---
        if GC:
            g = gradient_centralize(g, m2, Cf)
        r_factor, c_factor = factored_rc(g, rowp, colp, rr, cc, R, C, Rf, Cf, beta2, eps1)

        # --- Adakaon-specific: reconstructed update, RMS-clip (lr scales the final delta) ---
        upd = tl.where(m2, g * r_factor[:, None] * c_factor[None, :], 0.0)  # 0*inf corners -> 0
        rms = tl.sqrt(tl.sum(upd * upd) / (Rf * Cf))
        denom = rms / clip
        denom = tl.where(denom < 1.0, 1.0, denom)
        upd = upd / denom

        # --- momentum EMA (storage fp32 / bf16 / int8 / 4bit; EMA always runs in fp32) ---
        # dequant the stored momentum to fp32 (quant primitives are codec-level -> reusable)
        if MOMENTUM:
            if MOM == 2:  # int8 codes + per-row scale
                code_ptr = mi.to(tl.pointer_type(tl.int8))
                scale_ptr = tl.load(mscale_addr + t).to(tl.pointer_type(tl.float32))
                m_old = dequant_int8(code_ptr, idx, m2, scale_ptr, rr, R)
            elif MOM == 3:  # 4bit packed codes + per-block scale (even C only; odd C -> native)
                packed_ptr = mi.to(tl.pointer_type(tl.uint8))
                scale_ptr = tl.load(mscale_addr + t).to(tl.pointer_type(tl.float32))
                Chalf = C // 2
                BLK = m_blk                                            # flat elems per 4-bit block
                # Ns_ptr[t] is this tensor's ALLOCATED m_scale length; it bounds the dequant's
                # scale load and the requant's scale store/reload below. ``m_blk`` comes from the
                # bucket key, so NS == NB here by construction; the bound is the backstop that
                # turns a future bucketing mistake into dropped scales, never foreign memory.
                NS = tl.load(Ns_ptr + t)
                m_old = dequant_4bit(packed_ptr, scale_ptr, ri, ci, idx, Chalf, m2, BLK, NS)
            elif MOM == 1:  # bf16
                m_old = tl.load(mi.to(tl.pointer_type(tl.bfloat16)) + idx, mask=m2, other=0.0).to(tl.float32)
            else:  # fp32
                m_old = tl.load(mi.to(tl.pointer_type(tl.float32)) + idx, mask=m2, other=0.0).to(tl.float32)
            m_new = beta1 * m_old + (1.0 - beta1) * upd
            # requant the updated momentum back to storage (m_new stays fp32 for delta below)
            if MOM == 2:
                requant_int8(m_new, m2, code_ptr, idx, scale_ptr, rr, R)
            elif MOM == 3:
                NB = (R * C + BLK - 1) // BLK
                requant_4bit(m_new, m2, idx, R, C, Chalf, packed_ptr, scale_ptr,
                             NB, NS, BLK, BR, BC, EXACT, FBLK)
            elif MOM == 1:
                tl.store(mi.to(tl.pointer_type(tl.bfloat16)) + idx, m_new.to(tl.bfloat16), mask=m2)
            else:
                tl.store(mi.to(tl.pointer_type(tl.float32)) + idx, m_new, mask=m2)
        else:
            m_new = upd

        # --- decoupled weight decay (AdamW-style), placement per ``cautious_wd`` (like native) ---
        # WDFULL=False ("masked", the default): folded into delta BEFORE the cautious mask, so a
        # rejected coordinate decays by ~0 and a survivor by wd/keep. WDFULL=True ("full"): applied
        # AFTER the mask to every coordinate at the same lr*wd (the Cautious Optimizers placement).
        p_old = tl.load(pp + idx, mask=m2, other=0.0).to(tl.float32)
        if CK == 16:  # late for kahan16 — see the measured note where kahan8 decodes
            cp = ck_ptr(c_addr, t, CK)
            zc = ck_decode(pp, cp, idx, m2, CK)
        if CK:
            p_old = zc  # decay (WD) reads the full decoded value, never the bare bf16
        delta = m_new
        if WD and not WDFULL:
            delta = delta + wd * p_old                 # momentum requant above used m_new (sans wd)

        # --- REUSABLE-ish: cautious masking + survivor rescale (operates on delta incl. wd) ---
        if CAUTIOUS:
            keep = (delta * g) > 0.0
            keepf = tl.where(keep, 1.0, 0.0)
            mm = tl.sum(keepf) / (Rf * Cf)
            mm = tl.where(mm < 1e-8, 1e-8, mm)
            # MULTIPLY by the 0/1 survivor mask, never ``tl.where``: 0 * NaN == NaN, so a
            # non-finite delta PROPAGATES exactly as native's ``delta.mul_(mask).div_(denom)``
            # does (_backend.cautious_batched_). ``tl.where`` substituted a hard 0 and FROZE the
            # tensor for the rest of the run — a silently dead weight instead of a visible NaN.
            # Finite values are unchanged: x * 1.0 == x. A REJECTED coordinate subtracts -0.0
            # rather than the old +0.0, which is the same weight except that a stored -0.0 flips
            # to +0.0 (they compare equal; the only difference is the sign bit).
            delta = (delta * keepf) / mm

        if WD and WDFULL:
            delta = delta + wd * p_old

        # --- weight write (plain fp32 or bf16 stochastic rounding); lr on the FULL delta ---
        if CK:  # compact Kahan: exact compensated value in, (bf16, residual) out
            ck_store(pp, cp, idx, m2, zc - lr * delta, seed + t, CK)
        else:
            res = p_old - lr * delta
            if SR:
                res = sr_round(res, seed + t, idx)
            tl.store(pp + idx, res.to(pp.dtype.element_ty), mask=m2)

    # ---- chunked (multi-block) path for tensors too large for one block ----
    # The per-tensor reductions (row/col EMA, rms via matvec, cautious mean) are cheap and stay in
    # torch; these two elementwise kernels do the heavy [R,C] passes (momentum + write) chunked over
    # a flat view, so a big weight matrix costs ~few memory passes instead of native's ~30.

    @triton.jit
    def _chunked_mom(g_ptr, m_ptr, p_ptr, c_ptr, rfac_ptr, cfac_ptr, keep_ptr, C, n, inv_rms, wd,
                     beta1, CAUTIOUS: tl.constexpr, WD: tl.constexpr, BLOCK: tl.constexpr,
                     WDFULL: tl.constexpr = False, CK: tl.constexpr = 0):
        """Momentum EMA of the normalized (LR-independent) update over a flat chunk; accumulates
        the cautious keep count (on delta incl. wd under ``cautious_wd="masked"``, on the bare
        momentum under ``"full"`` — matching native either way; the mask is invariant to the
        positive lr scale). m is fp32 or bf16 (EMA runs in fp32)."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        i = offs // C
        j = offs % C
        g = tl.load(g_ptr + offs, mask=mask, other=0.0)
        rf = tl.load(rfac_ptr + i, mask=mask, other=0.0)
        cf = tl.load(cfac_ptr + j, mask=mask, other=0.0)
        upd = g * rf * cf * inv_rms
        m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        m = beta1 * m + (1.0 - beta1) * upd
        tl.store(m_ptr + offs, m.to(m_ptr.dtype.element_ty), mask=mask)
        if CAUTIOUS:
            delta = m
            if WD and not WDFULL:
                # the decoded value where it can matter, as _chunked_apply's decay reads it
                delta = wd_keep_flat(delta, g, wd, p_ptr, c_ptr, offs, mask, CK)
            keep = ((delta * g) > 0.0) & mask
            tl.atomic_add(keep_ptr, tl.sum(keep.to(tl.int32)))

    @triton.jit
    def _chunked_apply(g_ptr, m_ptr, p_ptr, c_ptr, n, inv_mean, lr, wd, seed,
                       CAUTIOUS: tl.constexpr, WD: tl.constexpr, SR: tl.constexpr, BLOCK: tl.constexpr,
                       WDFULL: tl.constexpr = False, CK: tl.constexpr = 0):
        """delta = cautious(m + wd*p, g) ["masked"] or cautious(m, g) + wd*p ["full"];
        p -= lr*delta, with bf16 stochastic rounding if SR."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        if CK:
            cp = c_ptr
            zc = ck_decode(p_ptr, cp, offs, mask, CK)
            p = zc  # decay (WD) reads the full decoded value, never the bare bf16
        delta = m
        if WD and not WDFULL:
            delta = delta + wd * p
        if CAUTIOUS:
            g = tl.load(g_ptr + offs, mask=mask, other=0.0)
            keep = (delta * g) > 0.0
            # Mask by MULTIPLICATION so a non-finite delta propagates (see _adakaon_tile_kernel).
            delta = delta * tl.where(keep, inv_mean, 0.0)
        if WD and WDFULL:
            delta = delta + wd * p
        if CK:  # compact Kahan: exact compensated value in, (bf16, residual) out
            ck_store(p_ptr, cp, offs, mask, zc - lr * delta, seed, CK)
        else:
            res = p - lr * delta
            if SR:
                res = sr_round(res, seed, offs)
            tl.store(p_ptr + offs, res.to(p_ptr.dtype.element_ty), mask=mask)

    # ---- batched chunked (multi-block, pointer-array) path for the many-same-shape big regime ----
    # The dominant real workload (Cosmos LoKr: 236x 512x512 factors, all > TILE_CAP) used to route
    # to native torch foreach (~15-25 launches + ~10 full [N,R,C] passes per bucket). These two
    # kernels run the heavy [N,R,C] elementwise work (momentum EMA, cautious keep-count, WD, subtract,
    # SR) over the WHOLE same-shape bucket in ONE launch each, reading p/m per-tensor from a pointer
    # array (write IN PLACE, no stacking of p/m). The cheap reductions (row/col EMA, rms via matvec)
    # stay in torch on the stacked grad — see ``Adakaon._chunked_reductions_batched``. Same math +
    # state as the per-tensor ``_chunked_mom``/``_chunked_apply`` (which a lone big tensor still uses).
    #
    # Grid is ``(N * K,)`` with ``K = ceil(n/BLOCK)`` chunks per tensor (same n=R*C across the bucket,
    # so K is a constant): ``t = pid // K`` selects the tensor, ``k = pid % K`` the chunk. Grad is the
    # stacked fp32 [N, n] (GC already folded into the copy); r/c factors are stacked [N, R]/[N, C];
    # per-tensor scalars (``inv_rms``/``inv_mean``) are float32[N] arrays indexed by ``t``. Momentum
    # is read/written via the m pointer array with the same MOM constexpr as the one-block kernel for
    # fp32/bf16; int8/4bit momentum is dequant'd to an fp32 temp host-side (the m array then points at
    # the temp's per-tensor slices, MOM==FP32) and requant'd in torch between the two passes — exactly
    # the per-tensor ``_chunked_step`` precedent, so odd-C 4bit works (it packs flat, not per-tile).

    @triton.jit
    def _chunked_mom_batched(
        g_ptr, m_addr, p_addr, c_addr, rfac_ptr, cfac_ptr, keep_ptr, inv_rms_ptr,
        wd, beta1, R, C, n, K,
        LOWP: tl.constexpr, MOM: tl.constexpr, CAUTIOUS: tl.constexpr, WD: tl.constexpr,
        BLOCK: tl.constexpr, WDFULL: tl.constexpr = False, CK: tl.constexpr = 0,
    ):
        """Batched pass 1: momentum EMA of the normalized update over a flat chunk of tensor ``t``;
        accumulates the cautious keep-count (on delta incl. WD unless ``WDFULL``, matching native)
        into ``keep_ptr[t]``."""
        pid = tl.program_id(0)
        t = pid // K
        k = pid % K
        offs = k * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        i = offs // C
        j = offs % C
        g = tl.load(g_ptr + t * n + offs, mask=mask, other=0.0)            # stacked fp32 grad (GC'd)
        rf = tl.load(rfac_ptr + t * R + i, mask=mask, other=0.0)
        cf = tl.load(cfac_ptr + t * C + j, mask=mask, other=0.0)
        inv_rms = tl.load(inv_rms_ptr + t)
        upd = g * rf * cf * inv_rms
        mi = tl.load(m_addr + t)
        if MOM == 1:  # bf16 storage
            mp = mi.to(tl.pointer_type(tl.bfloat16))
            m = tl.load(mp + offs, mask=mask, other=0.0).to(tl.float32)
        else:         # fp32 storage (also the dequant'd int8/4bit temp)
            mp = mi.to(tl.pointer_type(tl.float32))
            m = tl.load(mp + offs, mask=mask, other=0.0)
        m = beta1 * m + (1.0 - beta1) * upd
        if MOM == 1:
            tl.store(mp + offs, m.to(tl.bfloat16), mask=mask)
        else:
            tl.store(mp + offs, m, mask=mask)
        if CAUTIOUS:
            delta = m
            if WD and not WDFULL:
                pi = tl.load(p_addr + t)
                if LOWP:
                    pp = pi.to(tl.pointer_type(tl.bfloat16))
                else:
                    pp = pi.to(tl.pointer_type(tl.float32))
                delta = wd_keep(delta, g, wd, pp, c_addr, t, offs, mask, CK)
            keep = ((delta * g) > 0.0) & mask
            tl.atomic_add(keep_ptr + t, tl.sum(keep.to(tl.int32)))

    @triton.jit
    def _chunked_apply_batched(
        g_ptr, m_addr, p_addr, c_addr, inv_mean_ptr, lr, wd, seed, n, K,
        LOWP: tl.constexpr, MOM: tl.constexpr, CAUTIOUS: tl.constexpr, WD: tl.constexpr,
        SR: tl.constexpr, BLOCK: tl.constexpr, WDFULL: tl.constexpr = False, CK: tl.constexpr = 0,
    ):
        """Batched pass 2: delta = cautious(m + wd*p, g) ["masked"] or cautious(m, g) + wd*p
        ["full"]; p -= lr*delta (bf16 SR if LOWP+SR)."""
        pid = tl.program_id(0)
        t = pid // K
        k = pid % K
        offs = k * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        mi = tl.load(m_addr + t)
        if MOM == 1:
            m = tl.load(mi.to(tl.pointer_type(tl.bfloat16)) + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            m = tl.load(mi.to(tl.pointer_type(tl.float32)) + offs, mask=mask, other=0.0)
        pi = tl.load(p_addr + t)
        pp = pi.to(tl.pointer_type(tl.bfloat16)) if LOWP else pi.to(tl.pointer_type(tl.float32))
        p = tl.load(pp + offs, mask=mask, other=0.0).to(tl.float32)
        if CK:
            cp = ck_ptr(c_addr, t, CK)
            zc = ck_decode(pp, cp, offs, mask, CK)
            p = zc  # decay (WD) reads the full decoded value, never the bare bf16
        delta = m
        if WD and not WDFULL:
            delta = delta + wd * p
        if CAUTIOUS:
            g = tl.load(g_ptr + t * n + offs, mask=mask, other=0.0)
            inv_mean = tl.load(inv_mean_ptr + t)
            keep = (delta * g) > 0.0
            # Mask by MULTIPLICATION so a non-finite delta propagates (see _adakaon_tile_kernel).
            delta = delta * tl.where(keep, inv_mean, 0.0)
        if WD and WDFULL:
            delta = delta + wd * p
        if CK:  # compact Kahan: exact compensated value in, (bf16, residual) out
            ck_store(pp, cp, offs, mask, zc - lr * delta, seed + t, CK)
        else:
            res = p - lr * delta
            if SR:
                res = sr_round(res, seed + t, offs)
            tl.store(pp + offs, res.to(pp.dtype.element_ty), mask=mask)

    # ---- candidate #4: FUSED REDUCTIONS for the batched big regime (no 248MB torch stack) ----
    # The torch reductions (stack -> GC -> gsq -> row/col EMA -> rms) were measured at 73-80% of the
    # batched big step, dominated by the [N,R,C] fp32 stack + the GC pass. These kernels read each
    # tensor's grad straight from a POINTER ARRAY (no stack), do GC in-kernel, and produce the per-
    # tensor row/col sums + rms; the mom/apply "_g" variants below then re-read grad via the same
    # pointer array (GC via the precomputed rowmean) so NOTHING materializes the [N,R,C] grad.
    # One program owns BR rows x C cols of tensor t (grid = N * ceil(R/BR)). rowsum/rowmean are stored
    # directly (the program owns those rows); colsum accumulates across row-blocks via atomic_add.

    @triton.jit
    def _reduce_rowcol(
        g_addr, rowmean_ptr, rowsum_ptr, colsum_ptr, R, C, RB,
        LOWP: tl.constexpr, GC: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
    ):
        """Per (tensor, row-block): GC (per-row mean over C) -> rowmean[N,R], rowsum_gsq[N,R] (direct),
        colsum_gsq[N,C] (atomic). Padded cols load 0 so the per-row mean over the real C is exact."""
        pid = tl.program_id(0)
        t = pid // RB
        rb = pid % RB
        ri = rb * BR + tl.arange(0, BR)
        ci = tl.arange(0, BC)
        rmask = ri < R
        cmask = ci < C
        m2 = rmask[:, None] & cmask[None, :]
        gbase = tl.load(g_addr + t)
        gp = gbase.to(tl.pointer_type(tl.bfloat16)) if LOWP else gbase.to(tl.pointer_type(tl.float32))
        g = tl.load(gp + ri[:, None] * C + ci[None, :], mask=m2, other=0.0).to(tl.float32)
        if GC:
            # ``C * 1.0``, never ``C.to(tl.float32)``: Triton SPECIALIZES an int argument whose
            # value is 1 into a Python int, which has no ``.to`` — a (20000, 1) weight on the big
            # route failed to compile at all. Multiplying promotes either form to fp32.
            rmean = tl.sum(g, axis=1) / (C * 1.0)               # [BR] per-row mean over real C
            tl.store(rowmean_ptr + t * R + ri, rmean, mask=rmask)
            g = tl.where(m2, g - rmean[:, None], 0.0)
        gsq = g * g
        tl.store(rowsum_ptr + t * R + ri, tl.sum(gsq, axis=1), mask=rmask)
        tl.atomic_add(colsum_ptr + t * C + ci, tl.sum(gsq, axis=0), mask=cmask)

    @triton.jit
    def _reduce_rms(
        g_addr, rowmean_ptr, rfac_ptr, cfac_ptr, rms_ptr, R, C, RB,
        LOWP: tl.constexpr, GC: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
    ):
        """Per (tensor, row-block): accumulate sum( (g' * r_factor * c_factor)^2 ) into rms_ptr[t]
        (atomic), re-reading grad via the pointer array and GC via the precomputed rowmean."""
        pid = tl.program_id(0)
        t = pid // RB
        rb = pid % RB
        ri = rb * BR + tl.arange(0, BR)
        ci = tl.arange(0, BC)
        rmask = ri < R
        cmask = ci < C
        m2 = rmask[:, None] & cmask[None, :]
        gbase = tl.load(g_addr + t)
        gp = gbase.to(tl.pointer_type(tl.bfloat16)) if LOWP else gbase.to(tl.pointer_type(tl.float32))
        g = tl.load(gp + ri[:, None] * C + ci[None, :], mask=m2, other=0.0).to(tl.float32)
        if GC:
            rmean = tl.load(rowmean_ptr + t * R + ri, mask=rmask, other=0.0)
            g = tl.where(m2, g - rmean[:, None], 0.0)
        rf = tl.load(rfac_ptr + t * R + ri, mask=rmask, other=0.0)
        cf = tl.load(cfac_ptr + t * C + ci, mask=cmask, other=0.0)
        u = g * rf[:, None] * cf[None, :]
        tl.atomic_add(rms_ptr + t, tl.sum(u * u))

    # ---- deterministic (two-pass) variants of the two fp32-atomic reductions ----
    # ``_reduce_rowcol`` and ``_reduce_rms`` accumulate ``colsum`` and ``rms`` with
    # ``tl.atomic_add`` on fp32. Float addition is not associative, and the order row-blocks
    # reach the atomic is decided by the scheduler, so the SAME inputs give slightly different
    # sums on every run. Measured spread over 4 runs of 6 steps on 3x(512,512), max|dp| /
    # scale: 5.1e-8 (fp32 momentum), 3.9e-6 (bf16), 8.1e-6 (int8), 7.0e-4 (4bit — a ulp on the
    # momentum flips an adjacent 4-bit code, so quantized momenta amplify it by four orders).
    #
    # The two-pass form the design doc kept in reserve for atomic CONTENTION (see
    # docs/FUSED_REDUCTIONS_DESIGN.md, "Risks to validate") removes it: pass 1 stores one
    # PARTIAL per row-block — a plain store to an address only that program owns — and pass 2
    # sums the partials in a fixed sequential order. Same arithmetic, one fixed order, so a run
    # reproduces itself bit for bit.
    #
    # ``keep`` needs no such treatment: it is an INT32 atomic, and integer addition is
    # associative and exact whatever the order.
    #
    # Cost is the partial buffer, ``N * RB * C`` fp32 (7.7 MB for the 236x(512,512) LoKr
    # bucket at RB=16), allocated only when the flag is on. See ``Adakaon(deterministic_
    # reductions=True)``.

    @triton.jit
    def _reduce_rowcol_det(
        g_addr, rowmean_ptr, rowsum_ptr, colpart_ptr, R, C, RB,
        LOWP: tl.constexpr, GC: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
    ):
        """``_reduce_rowcol`` with the colsum atomic replaced by a per-row-block PARTIAL store."""
        pid = tl.program_id(0)
        t = pid // RB
        rb = pid % RB
        ri = rb * BR + tl.arange(0, BR)
        ci = tl.arange(0, BC)
        rmask = ri < R
        cmask = ci < C
        m2 = rmask[:, None] & cmask[None, :]
        gbase = tl.load(g_addr + t)
        gp = gbase.to(tl.pointer_type(tl.bfloat16)) if LOWP else gbase.to(tl.pointer_type(tl.float32))
        g = tl.load(gp + ri[:, None] * C + ci[None, :], mask=m2, other=0.0).to(tl.float32)
        if GC:
            rmean = tl.sum(g, axis=1) / (C * 1.0)          # see _reduce_rowcol on the `* 1.0`
            tl.store(rowmean_ptr + t * R + ri, rmean, mask=rmask)
            g = tl.where(m2, g - rmean[:, None], 0.0)
        gsq = g * g
        tl.store(rowsum_ptr + t * R + ri, tl.sum(gsq, axis=1), mask=rmask)
        # This program is the ONLY writer of partial row-block ``rb`` of tensor ``t``.
        tl.store(colpart_ptr + (t * RB + rb) * C + ci, tl.sum(gsq, axis=0), mask=cmask)

    @triton.jit
    def _reduce_rms_det(
        g_addr, rowmean_ptr, rfac_ptr, cfac_ptr, rmspart_ptr, R, C, RB,
        LOWP: tl.constexpr, GC: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
    ):
        """``_reduce_rms`` with the rms atomic replaced by a per-row-block PARTIAL store."""
        pid = tl.program_id(0)
        t = pid // RB
        rb = pid % RB
        ri = rb * BR + tl.arange(0, BR)
        ci = tl.arange(0, BC)
        rmask = ri < R
        cmask = ci < C
        m2 = rmask[:, None] & cmask[None, :]
        gbase = tl.load(g_addr + t)
        gp = gbase.to(tl.pointer_type(tl.bfloat16)) if LOWP else gbase.to(tl.pointer_type(tl.float32))
        g = tl.load(gp + ri[:, None] * C + ci[None, :], mask=m2, other=0.0).to(tl.float32)
        if GC:
            rmean = tl.load(rowmean_ptr + t * R + ri, mask=rmask, other=0.0)
            g = tl.where(m2, g - rmean[:, None], 0.0)
        rf = tl.load(rfac_ptr + t * R + ri, mask=rmask, other=0.0)
        cf = tl.load(cfac_ptr + t * C + ci, mask=cmask, other=0.0)
        u = g * rf[:, None] * cf[None, :]
        tl.store(rmspart_ptr + t * RB + rb, tl.sum(u * u))

    @triton.jit
    def _reduce_colpart(colpart_ptr, colsum_ptr, C, RB, CB, BCT: tl.constexpr):
        """Sum the row-block partials into ``colsum``, in a FIXED sequential order."""
        pid = tl.program_id(0)
        t = pid // CB
        cb = pid % CB
        ci = cb * BCT + tl.arange(0, BCT)
        cmask = ci < C
        acc = tl.zeros((BCT,), dtype=tl.float32)
        for rb in range(RB):
            acc += tl.load(colpart_ptr + (t * RB + rb) * C + ci, mask=cmask, other=0.0)
        tl.store(colsum_ptr + t * C + ci, acc, mask=cmask)

    @triton.jit
    def _reduce_rmspart(rmspart_ptr, rms_ptr, RB):
        """Sum one tensor's rms partials, in a FIXED sequential order."""
        t = tl.program_id(0)
        acc = tl.zeros((1,), dtype=tl.float32)
        for rb in range(RB):
            acc += tl.load(rmspart_ptr + t * RB + rb)
        tl.store(rms_ptr + t, tl.sum(acc))

    @triton.jit
    def _factor_rowcol_batched(
        row_addr, col_addr, rowsum_ptr, colsum_ptr, rfac_ptr, cfac_ptr,
        R, C, beta2, eps1,
        BR: tl.constexpr, BC: tl.constexpr,
    ):
        """Update factored EMA state in place and emit inverse-sqrt factors."""
        t = tl.program_id(0)
        rr = tl.arange(0, BR)
        cc = tl.arange(0, BC)
        rmask = rr < R
        cmask = cc < C
        rowp = tl.load(row_addr + t).to(tl.pointer_type(tl.float32))
        colp = tl.load(col_addr + t).to(tl.pointer_type(tl.float32))
        row_old = tl.load(rowp + rr, mask=rmask, other=0.0)
        col_old = tl.load(colp + cc, mask=cmask, other=0.0)
        omb = 1.0 - beta2
        row_new = row_old + omb * (
            tl.load(rowsum_ptr + t * R + rr, mask=rmask, other=0.0) / C + eps1 - row_old
        )
        col_new = col_old + omb * (
            tl.load(colsum_ptr + t * C + cc, mask=cmask, other=0.0) / R + eps1 - col_old
        )
        tl.store(rowp + rr, row_new, mask=rmask)
        tl.store(colp + cc, col_new, mask=cmask)
        row_mean = tl.sum(tl.where(rmask, row_new, 0.0)) / (R * 1.0)  # see _reduce_rowcol on `* 1.0`
        tl.store(rfac_ptr + t * R + rr, tl.rsqrt(row_new / row_mean), mask=rmask)
        tl.store(cfac_ptr + t * C + cc, tl.rsqrt(col_new), mask=cmask)

    @triton.jit
    def _finish_rms(rms_ptr, inv_rms_ptr, n, clip, N, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < N
        rms = tl.sqrt(tl.load(rms_ptr + offs, mask=mask, other=0.0) / n)
        denom = tl.maximum(rms / clip, 1.0)
        tl.store(inv_rms_ptr + offs, 1.0 / denom, mask=mask)


    @triton.jit
    def inv_rms_clip(rms_ptr, t, n, clip):
        """``1 / max(rms/clip, 1)`` for tensor ``t`` from the RAW sum-of-squares accumulator.

        Folds what the ``grid=1`` :func:`_finish_rms` launch used to precompute into every
        consumer, which is where the value was going anyway: one scalar load plus four scalar
        ops per program, against a whole extra kernel launch per bucket per step (a fixed
        62-94 us cost that a 40-bucket step pays 40 times). ``_finish_rms`` itself stays for
        AdaPNM's reduction path, which still precomputes.
        """
        return 1.0 / tl.maximum(tl.sqrt(tl.load(rms_ptr + t) / n) / clip, 1.0)

    # mom/apply that read grad via the pointer array (+ GC via rowmean) instead of a stacked g_ptr.
    @triton.jit
    def _chunked_mom_batched_g(
        g_addr, rowmean_ptr, m_addr, p_addr, c_addr, rfac_ptr, cfac_ptr, keep_ptr, rms_ptr,
        clip, wd, beta1, R, C, n, K,
        LOWP: tl.constexpr, MOM: tl.constexpr, GC: tl.constexpr, CAUTIOUS: tl.constexpr,
        WD: tl.constexpr, BLOCK: tl.constexpr, WDFULL: tl.constexpr = False, CK: tl.constexpr = 0,
    ):
        """As ``_chunked_mom_batched`` but grad comes from the pointer array (GC via rowmean[t, row])."""
        pid = tl.program_id(0)
        t = pid // K
        k = pid % K
        offs = k * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        i = offs // C
        j = offs % C
        gbase = tl.load(g_addr + t)
        gp = gbase.to(tl.pointer_type(tl.bfloat16)) if LOWP else gbase.to(tl.pointer_type(tl.float32))
        g = tl.load(gp + offs, mask=mask, other=0.0).to(tl.float32)
        if GC:
            g = g - tl.load(rowmean_ptr + t * R + i, mask=mask, other=0.0)
        rf = tl.load(rfac_ptr + t * R + i, mask=mask, other=0.0)
        cf = tl.load(cfac_ptr + t * C + j, mask=mask, other=0.0)
        upd = g * rf * cf * inv_rms_clip(rms_ptr, t, n, clip)
        mi = tl.load(m_addr + t)
        if MOM == 1:
            mp = mi.to(tl.pointer_type(tl.bfloat16))
            m = tl.load(mp + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            mp = mi.to(tl.pointer_type(tl.float32))
            m = tl.load(mp + offs, mask=mask, other=0.0)
        m = beta1 * m + (1.0 - beta1) * upd
        if MOM == 1:
            tl.store(mp + offs, m.to(tl.bfloat16), mask=mask)
        else:
            tl.store(mp + offs, m, mask=mask)
        if CAUTIOUS:
            delta = m
            if WD and not WDFULL:
                pi = tl.load(p_addr + t)
                pp = pi.to(tl.pointer_type(tl.bfloat16)) if LOWP else pi.to(tl.pointer_type(tl.float32))
                delta = wd_keep(delta, g, wd, pp, c_addr, t, offs, mask, CK)
            keep = ((delta * g) > 0.0) & mask
            tl.atomic_add(keep_ptr + t, tl.sum(keep.to(tl.int32)))

    @triton.jit
    def _chunked_apply_batched_g(
        g_addr, rowmean_ptr, m_addr, p_addr, c_addr, inv_mean_ptr, lr, wd, seed, R, C, n, K,
        LOWP: tl.constexpr, MOM: tl.constexpr, GC: tl.constexpr, CAUTIOUS: tl.constexpr,
        WD: tl.constexpr, SR: tl.constexpr, BLOCK: tl.constexpr, WDFULL: tl.constexpr = False, CK: tl.constexpr = 0,
    ):
        """As ``_chunked_apply_batched`` but grad (for cautious) comes from the pointer array + GC."""
        pid = tl.program_id(0)
        t = pid // K
        k = pid % K
        offs = k * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        mi = tl.load(m_addr + t)
        if MOM == 1:
            m = tl.load(mi.to(tl.pointer_type(tl.bfloat16)) + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            m = tl.load(mi.to(tl.pointer_type(tl.float32)) + offs, mask=mask, other=0.0)
        pi = tl.load(p_addr + t)
        pp = pi.to(tl.pointer_type(tl.bfloat16)) if LOWP else pi.to(tl.pointer_type(tl.float32))
        p = tl.load(pp + offs, mask=mask, other=0.0).to(tl.float32)
        if CK:
            cp = ck_ptr(c_addr, t, CK)
            zc = ck_decode(pp, cp, offs, mask, CK)
            p = zc  # decay (WD) reads the full decoded value, never the bare bf16
        delta = m
        if WD and not WDFULL:
            delta = delta + wd * p
        if CAUTIOUS:
            i = offs // C
            gbase = tl.load(g_addr + t)
            gp = gbase.to(tl.pointer_type(tl.bfloat16)) if LOWP else gbase.to(tl.pointer_type(tl.float32))
            g = tl.load(gp + offs, mask=mask, other=0.0).to(tl.float32)
            if GC:
                g = g - tl.load(rowmean_ptr + t * R + i, mask=mask, other=0.0)
            count = tl.load(inv_mean_ptr + t).to(tl.float32)
            inv_mean = n.to(tl.float32) / tl.maximum(count, 1.0)
            keep = (delta * g) > 0.0
            # Mask by MULTIPLICATION so a non-finite delta propagates (see _adakaon_tile_kernel).
            delta = delta * tl.where(keep, inv_mean, 0.0)
        if WD and WDFULL:
            delta = delta + wd * p
        if CK:  # compact Kahan: exact compensated value in, (bf16, residual) out
            ck_store(pp, cp, offs, mask, zc - lr * delta, seed + t, CK)
        else:
            res = p - lr * delta
            if SR:
                res = sr_round(res, seed + t, offs)
            tl.store(pp + offs, res.to(pp.dtype.element_ty), mask=mask)

    # ---- one-block non-factored 1-D path (biases / norm scales) ----
    # A bag of many tiny 1-D tensors is the same launch-bound regime the 2-D one-block kernel wins
    # big on; a native torch-foreach stacks them (stack overhead dominates). This kernel owns one 1-D
    # tensor per program and runs the whole non-factored Adam step (full per-coordinate v, in registers)
    # — no row/col factoring. Every momentum codec is updated in place. Same math as the
    # native ``_nonfactored_bucket`` (eps1 added to grad^2; RMS-clip; momentum EMA on the clipped
    # update; WD folded into delta before cautious; SR weight write).

    @triton.jit
    def _adam_1d_kernel(
        g_addr, p_addr, c_addr, m_addr, mscale_addr, v_addr, Ls_ptr,
        lr, beta1, beta2, eps1, clip, wd, seed,
        LOWP: tl.constexpr, MOM: tl.constexpr, MOMENTUM: tl.constexpr, CAUTIOUS: tl.constexpr,
        WD: tl.constexpr, SR: tl.constexpr, BL: tl.constexpr, FBLOCK: tl.constexpr,
        WDFULL: tl.constexpr = False, CK: tl.constexpr = 0,
    ):
        """One program == one 1-D tensor. Whole non-factored Adam step, in place via pointer-array."""
        t = tl.program_id(0)
        L = tl.load(Ls_ptr + t)
        Lf = L.to(tl.float32)
        gi = tl.load(g_addr + t)
        pi = tl.load(p_addr + t)
        vp = tl.load(v_addr + t).to(tl.pointer_type(tl.float32))
        if LOWP:
            gp = gi.to(tl.pointer_type(tl.bfloat16))
            pp = pi.to(tl.pointer_type(tl.bfloat16))
        else:
            gp = gi.to(tl.pointer_type(tl.float32))
            pp = pi.to(tl.pointer_type(tl.float32))
        offs = tl.arange(0, BL)
        mask = offs < L
        g = tl.load(gp + offs, mask=mask, other=0.0).to(tl.float32)

        # full per-coordinate second moment (HF eps placement: eps1 into grad^2)
        v = tl.load(vp + offs, mask=mask, other=0.0)
        v = beta2 * v + (1.0 - beta2) * (g * g + eps1)
        tl.store(vp + offs, v, mask=mask)
        update = tl.where(mask, g * tl.rsqrt(v), 0.0)

        # Adafactor RMS-clip on the update (lr scales the final delta)
        rms = tl.sqrt(tl.sum(update * update) / Lf)
        denom = rms / clip
        denom = tl.where(denom < 1.0, 1.0, denom)
        update = update / denom

        # Momentum EMA in fp32, then write back through the selected persistent codec.
        if MOMENTUM:
            if MOM == 1:
                mp = tl.load(m_addr + t).to(tl.pointer_type(tl.bfloat16))
                m_old = tl.load(mp + offs, mask=mask, other=0.0).to(tl.float32)
            elif MOM == 2:
                mp = tl.load(m_addr + t).to(tl.pointer_type(tl.int8))
                sp = tl.load(mscale_addr + t).to(tl.pointer_type(tl.float32))
                scale = tl.load(sp)
                m_old = tl.load(mp + offs, mask=mask, other=0).to(tl.float32) * scale
            elif MOM == 3:
                mp = tl.load(m_addr + t).to(tl.pointer_type(tl.uint8))
                sp = tl.load(mscale_addr + t).to(tl.pointer_type(tl.float32))
                byte = tl.load(mp + offs // 2, mask=mask, other=0)
                nib = tl.where((offs % 2) == 0, byte & 0xF, (byte >> 4) & 0xF)
                scale = tl.load(sp + offs // FBLOCK, mask=mask, other=0.0)
                m_old = (nib.to(tl.float32) - 8.0) * scale
            else:
                mp = tl.load(m_addr + t).to(tl.pointer_type(tl.float32))
                m_old = tl.load(mp + offs, mask=mask, other=0.0)
            m_new = beta1 * m_old + (1.0 - beta1) * update
            if MOM == 1:
                tl.store(mp + offs, m_new.to(tl.bfloat16), mask=mask)
            elif MOM == 2:
                amax = tl.maximum(tl.max(tl.where(mask, tl.abs(m_new), 0.0)), 1e-12)
                new_scale = amax / 127.0
                q = libdevice.rint(m_new / new_scale)
                q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
                tl.store(mp + offs, q, mask=mask)
                tl.store(sp, new_scale)
            elif MOM == 3:
                # Reuse the matrix codec with a synthetic [1,L] row. BL is at least 2,
                # so adjacent nibbles always have one unambiguous writer.
                idx = offs
                # FBLOCK is this bucket's REAL m_block, so the block count is exactly the
                # allocated m_scale length -> NB and NS coincide here by construction.
                NB = (L + FBLOCK - 1) // FBLOCK
                requant_4bit(
                    m_new[None, :], mask[None, :], idx[None, :], 1, L, (L + 1) // 2,
                    mp, sp, NB, NB, FBLOCK, BR=1, BC=BL,
                )
            else:
                tl.store(mp + offs, m_new, mask=mask)
            delta = m_new
        else:
            delta = update

        p_old = tl.load(pp + offs, mask=mask, other=0.0).to(tl.float32)
        if CK:
            cp = ck_ptr(c_addr, t, CK)
            zc = ck_decode(pp, cp, offs, mask, CK)
            p_old = zc  # decay (WD) reads the full decoded value, never the bare bf16
        if WD and not WDFULL:                          # ``cautious_wd`` — see _adakaon_tile_kernel
            delta = delta + wd * p_old
        if CAUTIOUS:
            keep = (delta * g) > 0.0
            keepf = tl.where(keep, 1.0, 0.0)
            mm = tl.sum(keepf) / Lf
            mm = tl.where(mm < 1e-8, 1e-8, mm)
            # Mask by MULTIPLICATION so a non-finite delta propagates (see _adakaon_tile_kernel).
            delta = (delta * keepf) / mm
        if WD and WDFULL:
            delta = delta + wd * p_old
        if CK:  # compact Kahan: exact compensated value in, (bf16, residual) out
            ck_store(pp, cp, offs, mask, zc - lr * delta, seed + t, CK)
        else:
            res = p_old - lr * delta
            if SR:
                res = sr_round(res, seed + t, offs)
            tl.store(pp + offs, res.to(pp.dtype.element_ty), mask=mask)

    # ============================================================ AdaPNM (positive-negative momentum)
    # Reuses the núcleo (gradient_centralize, factored_rc, dequant/requant_*, sr_round). New vs Adakaon:
    # TWO momenta (pos/neg, roles alternate by step parity — the host passes them swapped), the
    # raw-grad EMA on only the positive buffer (decay beta1^2), the pos-neg mix / noise_norm, and
    # decoupled WD applied BEFORE the step (p *= 1-lr*wd). Like Adakaon it RMS-clips the update
    # (``CLIP``, threshold ``clip_eff == clip * step_size``) — load-bearing: without it the factored
    # 1/sqrt(v_hat) blows up on a cold col and diverges. ``sc`` folds bc2_sq * step_size; ``inv_noise`` = 1/noise_norm.

    @triton.jit
    def _adapnm_tile_kernel(
        g_addr, p_addr, pos_addr, neg_addr, posc_addr, negc_addr, row_addr, col_addr,
        Rs_ptr, Cs_ptr, Ns_ptr,
        beta1_sq, beta0, inv_noise, beta2, sc, lrwd, eps1, clip_eff, seed,
        LOWP: tl.constexpr, MOM: tl.constexpr, CAUTIOUS: tl.constexpr, WD: tl.constexpr,
        GC: tl.constexpr, SR: tl.constexpr, CLIP: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
    ):
        t = tl.program_id(0)
        R = tl.load(Rs_ptr + t)
        C = tl.load(Cs_ptr + t)
        Rf = R.to(tl.float32)
        Cf = C.to(tl.float32)
        gi = tl.load(g_addr + t)
        pi = tl.load(p_addr + t)
        posi = tl.load(pos_addr + t)
        negi = tl.load(neg_addr + t)
        rowp = tl.load(row_addr + t).to(tl.pointer_type(tl.float32))
        colp = tl.load(col_addr + t).to(tl.pointer_type(tl.float32))
        if LOWP:
            gp = gi.to(tl.pointer_type(tl.bfloat16))
            pp = pi.to(tl.pointer_type(tl.bfloat16))
        else:
            gp = gi.to(tl.pointer_type(tl.float32))
            pp = pi.to(tl.pointer_type(tl.float32))
        ri = tl.arange(0, BR)[:, None]
        ci = tl.arange(0, BC)[None, :]
        rr = tl.arange(0, BR)
        cc = tl.arange(0, BC)
        m2 = (ri < R) & (ci < C)
        idx = ri * C + ci
        g = tl.load(gp + idx, mask=m2, other=0.0).to(tl.float32)
        if GC:
            g = gradient_centralize(g, m2, Cf)
        r_factor, c_factor = factored_rc(g, rowp, colp, rr, cc, R, C, Rf, Cf, beta2, eps1)

        # dequant both momenta (codec-level, reusable); EMA only the positive
        Chalf = C // 2
        BLK = tl.minimum(R * C, 128)
        if MOM == 2:  # int8 per-row
            posp = posi.to(tl.pointer_type(tl.int8))
            negp = negi.to(tl.pointer_type(tl.int8))
            pscale = tl.load(posc_addr + t).to(tl.pointer_type(tl.float32))
            nscale = tl.load(negc_addr + t).to(tl.pointer_type(tl.float32))
            m_pos = dequant_int8(posp, idx, m2, pscale, rr, R)
            m_neg = dequant_int8(negp, idx, m2, nscale, rr, R)
        elif MOM == 3:  # 4bit per-block
            posp = posi.to(tl.pointer_type(tl.uint8))
            negp = negi.to(tl.pointer_type(tl.uint8))
            pscale = tl.load(posc_addr + t).to(tl.pointer_type(tl.float32))
            nscale = tl.load(negc_addr + t).to(tl.pointer_type(tl.float32))
            # ``Ns_ptr[t]`` is this tensor's ALLOCATED scale length (the smaller of the two
            # momenta's, which the launcher validates are equal) — NOT the block count derived
            # from ``BLK``. It bounds the dequant loads and the requant's store/reload below, so
            # a layout the routing guard failed to divert can only lose scales, never touch
            # foreign memory. The old ``NS = NB`` made the two agree by construction and wrote
            # ``NB - NS`` floats past a shorter ``m_pos_scale``/``m_neg_scale``.
            NS = tl.load(Ns_ptr + t)
            m_pos = dequant_4bit(posp, pscale, ri, ci, idx, Chalf, m2, BLK, NS)
            m_neg = dequant_4bit(negp, nscale, ri, ci, idx, Chalf, m2, BLK, NS)
        elif MOM == 1:  # bf16
            m_pos = tl.load(posi.to(tl.pointer_type(tl.bfloat16)) + idx, mask=m2, other=0.0).to(tl.float32)
            m_neg = tl.load(negi.to(tl.pointer_type(tl.bfloat16)) + idx, mask=m2, other=0.0).to(tl.float32)
        else:  # fp32
            m_pos = tl.load(posi.to(tl.pointer_type(tl.float32)) + idx, mask=m2, other=0.0)
            m_neg = tl.load(negi.to(tl.pointer_type(tl.float32)) + idx, mask=m2, other=0.0)
        m_pos = beta1_sq * m_pos + (1.0 - beta1_sq) * g
        if MOM == 2:
            requant_int8(m_pos, m2, posp, idx, pscale, rr, R)
        elif MOM == 3:
            NB = (R * C + BLK - 1) // BLK
            requant_4bit(m_pos, m2, idx, R, C, Chalf, posp, pscale, NB, NS, BLK, BR, BC)
        elif MOM == 1:
            tl.store(posi.to(tl.pointer_type(tl.bfloat16)) + idx, m_pos.to(tl.bfloat16), mask=m2)
        else:
            tl.store(posi.to(tl.pointer_type(tl.float32)) + idx, m_pos, mask=m2)

        pn = ((1.0 + beta0) * m_pos - beta0 * m_neg) * inv_noise
        upd = tl.where(m2, pn * r_factor[:, None] * c_factor[None, :] * sc, 0.0)
        # Adafactor RMS-clip on the (lr-scaled) update: rms(upd) <= clip_eff == clip * step_size,
        # i.e. rms(pn / sqrt(v_hat)) <= clip. Bounds the cold-col rsqrt blowup -> no NaN runaway.
        if CLIP:
            rms = tl.sqrt(tl.sum(upd * upd) / (Rf * Cf))
            d = rms / clip_eff
            d = tl.where(d < 1.0, 1.0, d)
            upd = upd / d
        delta = upd
        if CAUTIOUS:
            keep = (upd * g) > 0.0
            keepf = tl.where(keep, 1.0, 0.0)
            mm = tl.sum(keepf) / (Rf * Cf)
            mm = tl.where(mm < 1e-8, 1e-8, mm)
            # MULTIPLY by the 0/1 survivor mask, never ``tl.where``: 0 * NaN == NaN, so a
            # non-finite update PROPAGATES exactly as native's ``delta.mul_(mask).div_(denom)``
            # does (_backend.cautious_batched_). ``tl.where`` substituted a hard 0 and FROZE the
            # tensor for the rest of the run — a silently dead weight instead of a visible NaN.
            # Finite values are unchanged: x * 1.0 == x. A REJECTED coordinate subtracts -0.0
            # rather than the old +0.0, which is the same weight except that a stored -0.0 flips
            # to +0.0 (they compare equal; the only difference is the sign bit).
            delta = (upd * keepf) / mm
        p_old = tl.load(pp + idx, mask=m2, other=0.0).to(tl.float32)
        if WD:
            p_old = p_old * (1.0 - lrwd)               # decoupled WD BEFORE (kozistr order)
        res = p_old - delta
        if SR:
            res = sr_round(res, seed + t, idx)
        tl.store(pp + idx, res.to(pp.dtype.element_ty), mask=m2)

    @triton.jit
    def _adapnm_chunked_mom(g_ptr, pos_ptr, neg_ptr, p_ptr, rfac_ptr, cfac_ptr, keep_ptr, C, n,
                            beta1_sq, beta0, inv_noise, sc, lrwd, CAUTIOUS: tl.constexpr,
                            WD: tl.constexpr, BLOCK: tl.constexpr):
        """Chunked AdaPNM pass 1: EMA the positive momentum (pos_ptr is an fp32 temp), accumulate the
        cautious keep-count on the pos-neg delta (incl. WD-on-p). pos/neg are fp32 temps; p is the weight."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        i = offs // C
        j = offs % C
        g = tl.load(g_ptr + offs, mask=mask, other=0.0)
        m_pos = tl.load(pos_ptr + offs, mask=mask, other=0.0)
        m_neg = tl.load(neg_ptr + offs, mask=mask, other=0.0)
        m_pos = beta1_sq * m_pos + (1.0 - beta1_sq) * g
        tl.store(pos_ptr + offs, m_pos, mask=mask)
        if CAUTIOUS:
            rf = tl.load(rfac_ptr + i, mask=mask, other=0.0)
            cf = tl.load(cfac_ptr + j, mask=mask, other=0.0)
            pn = ((1.0 + beta0) * m_pos - beta0 * m_neg) * inv_noise
            delta = pn * rf * cf * sc
            keep = ((delta * g) > 0.0) & mask
            tl.atomic_add(keep_ptr, tl.sum(keep.to(tl.int32)))

    @triton.jit
    def _adapnm_chunked_apply(g_ptr, pos_ptr, neg_ptr, p_ptr, rfac_ptr, cfac_ptr, C, n,
                              beta0, inv_noise, sc, lrwd, inv_mean, seed,
                              CAUTIOUS: tl.constexpr, WD: tl.constexpr, SR: tl.constexpr,
                              BLOCK: tl.constexpr):
        """Chunked AdaPNM pass 2: delta = cautious(pn * r*c * sc, g); p = p*(1-lr*wd) - delta."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        i = offs // C
        j = offs % C
        m_pos = tl.load(pos_ptr + offs, mask=mask, other=0.0)
        m_neg = tl.load(neg_ptr + offs, mask=mask, other=0.0)
        rf = tl.load(rfac_ptr + i, mask=mask, other=0.0)
        cf = tl.load(cfac_ptr + j, mask=mask, other=0.0)
        pn = ((1.0 + beta0) * m_pos - beta0 * m_neg) * inv_noise
        delta = pn * rf * cf * sc
        if CAUTIOUS:
            g = tl.load(g_ptr + offs, mask=mask, other=0.0)
            keep = (delta * g) > 0.0
            # Mask by MULTIPLICATION so a non-finite delta propagates (see _adakaon_tile_kernel).
            delta = delta * tl.where(keep, inv_mean, 0.0)
        p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        if WD:
            p = p * (1.0 - lrwd)
        res = p - delta
        if SR:
            res = sr_round(res, seed, offs)
        tl.store(p_ptr + offs, res.to(p_ptr.dtype.element_ty), mask=mask)

    # ---- batched chunked (multi-block, pointer-array) AdaPNM path for the many-same-shape big regime ----
    # Mirrors the Adakaon batched pair, plus AdaPNM's two-momentum structure. As in the per-tensor
    # AdaPNM chunked path, BOTH momenta are fp32 stacked temps (dequant'd host-side via the codec,
    # EMA only the positive) — both momenta are read/written IN PLACE via the pos/neg pointer arrays
    # (fp32 or bf16, MOM constexpr; int8/4bit dequant'd to fp32 temps host-side, the arrays then point
    # at the temps and the positive is requant'd between passes). Only ``p`` and the two momenta are
    # per-tensor pointer arrays; grad and r/c are stacked fp32 (GC folded into grad). The Adafactor
    # RMS-clip's per-tensor ``sum((pn*r*c)^2)`` is accumulated IN-KERNEL (``rms_ptr[t]``, atomic) so the
    # float case needs NO torch momentum temp at all — host turns it into ``sc_ptr[N]`` between passes.
    # WD is decoupled-on-p (``p *= 1-lr*wd``), applied in pass 2 and NOT gated by cautious.

    @triton.jit
    def _adapnm_chunked_mom_batched(
        g_ptr, pos_addr, neg_addr, rfac_ptr, cfac_ptr, keep_ptr, rms_ptr,
        R, C, n, K, beta1_sq, beta0, inv_noise,
        MOM: tl.constexpr, CAUTIOUS: tl.constexpr, CLIP: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """Batched AdaPNM pass 1: EMA the positive momentum in place (via pos pointer array). Per
        tensor, accumulate the cautious keep-count (``keep_ptr[t]``) and the clip's running
        ``sum((pn*r*c)^2)`` (``rms_ptr[t]``) from the pos-neg numerator."""
        pid = tl.program_id(0)
        t = pid // K
        k = pid % K
        offs = k * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        g = tl.load(g_ptr + t * n + offs, mask=mask, other=0.0)
        posi = tl.load(pos_addr + t)
        if MOM == 1:  # bf16 storage
            posp = posi.to(tl.pointer_type(tl.bfloat16))
            m_pos = tl.load(posp + offs, mask=mask, other=0.0).to(tl.float32)
        else:         # fp32 storage (also the dequant'd int8/4bit temp)
            posp = posi.to(tl.pointer_type(tl.float32))
            m_pos = tl.load(posp + offs, mask=mask, other=0.0)
        m_pos = beta1_sq * m_pos + (1.0 - beta1_sq) * g
        if MOM == 1:
            tl.store(posp + offs, m_pos.to(tl.bfloat16), mask=mask)
        else:
            tl.store(posp + offs, m_pos, mask=mask)
        if CAUTIOUS or CLIP:
            negi = tl.load(neg_addr + t)
            if MOM == 1:
                m_neg = tl.load(negi.to(tl.pointer_type(tl.bfloat16)) + offs, mask=mask, other=0.0).to(tl.float32)
            else:
                m_neg = tl.load(negi.to(tl.pointer_type(tl.float32)) + offs, mask=mask, other=0.0)
            i = offs // C
            j = offs % C
            rf = tl.load(rfac_ptr + t * R + i, mask=mask, other=0.0)
            cf = tl.load(cfac_ptr + t * C + j, mask=mask, other=0.0)
            urc = ((1.0 + beta0) * m_pos - beta0 * m_neg) * inv_noise * rf * cf  # == U / bc2_sq
            if CLIP:
                tl.atomic_add(rms_ptr + t, tl.sum(tl.where(mask, urc * urc, 0.0)))
            if CAUTIOUS:
                keep = ((urc * g) > 0.0) & mask                    # sign-invariant to the positive sc
                tl.atomic_add(keep_ptr + t, tl.sum(keep.to(tl.int32)))

    @triton.jit
    def _adapnm_chunked_apply_batched(
        g_ptr, pos_addr, neg_addr, p_addr, rfac_ptr, cfac_ptr, sc_ptr, inv_mean_ptr,
        R, C, n, K, beta0, inv_noise, lrwd, seed,
        LOWP: tl.constexpr, MOM: tl.constexpr, CAUTIOUS: tl.constexpr, WD: tl.constexpr,
        SR: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """Batched AdaPNM pass 2: delta = cautious(pn * r*c * sc[t], g); p = p*(1-lr*wd) - delta."""
        pid = tl.program_id(0)
        t = pid // K
        k = pid % K
        offs = k * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        i = offs // C
        j = offs % C
        posi = tl.load(pos_addr + t)
        negi = tl.load(neg_addr + t)
        if MOM == 1:
            m_pos = tl.load(posi.to(tl.pointer_type(tl.bfloat16)) + offs, mask=mask, other=0.0).to(tl.float32)
            m_neg = tl.load(negi.to(tl.pointer_type(tl.bfloat16)) + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            m_pos = tl.load(posi.to(tl.pointer_type(tl.float32)) + offs, mask=mask, other=0.0)
            m_neg = tl.load(negi.to(tl.pointer_type(tl.float32)) + offs, mask=mask, other=0.0)
        rf = tl.load(rfac_ptr + t * R + i, mask=mask, other=0.0)
        cf = tl.load(cfac_ptr + t * C + j, mask=mask, other=0.0)
        sc = tl.load(sc_ptr + t)
        pn = ((1.0 + beta0) * m_pos - beta0 * m_neg) * inv_noise
        delta = pn * rf * cf * sc
        if CAUTIOUS:
            g = tl.load(g_ptr + t * n + offs, mask=mask, other=0.0)
            inv_mean = tl.load(inv_mean_ptr + t)
            keep = (delta * g) > 0.0
            # Mask by MULTIPLICATION so a non-finite delta propagates (see _adakaon_tile_kernel).
            delta = delta * tl.where(keep, inv_mean, 0.0)
        pi = tl.load(p_addr + t)
        pp = pi.to(tl.pointer_type(tl.bfloat16)) if LOWP else pi.to(tl.pointer_type(tl.float32))
        p = tl.load(pp + offs, mask=mask, other=0.0).to(tl.float32)
        if WD:
            p = p * (1.0 - lrwd)
        res = p - delta
        if SR:
            res = sr_round(res, seed + t, offs)
        tl.store(pp + offs, res.to(pp.dtype.element_ty), mask=mask)

    # AdaPNM mom/apply that read grad via the pointer array (+ GC via rowmean) — candidate #4 path.
    @triton.jit
    def _adapnm_chunked_mom_batched_g(
        g_addr, rowmean_ptr, pos_addr, neg_addr, rfac_ptr, cfac_ptr, keep_ptr, rms_ptr,
        R, C, n, K, beta1_sq, beta0, inv_noise,
        LOWP: tl.constexpr, MOM: tl.constexpr, GC: tl.constexpr, CAUTIOUS: tl.constexpr,
        CLIP: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """As ``_adapnm_chunked_mom_batched`` but grad comes from the pointer array (GC via rowmean)."""
        pid = tl.program_id(0)
        t = pid // K
        k = pid % K
        offs = k * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        i = offs // C
        gbase = tl.load(g_addr + t)
        gp = gbase.to(tl.pointer_type(tl.bfloat16)) if LOWP else gbase.to(tl.pointer_type(tl.float32))
        g = tl.load(gp + offs, mask=mask, other=0.0).to(tl.float32)
        if GC:
            g = g - tl.load(rowmean_ptr + t * R + i, mask=mask, other=0.0)
        posi = tl.load(pos_addr + t)
        if MOM == 1:
            posp = posi.to(tl.pointer_type(tl.bfloat16))
            m_pos = tl.load(posp + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            posp = posi.to(tl.pointer_type(tl.float32))
            m_pos = tl.load(posp + offs, mask=mask, other=0.0)
        m_pos = beta1_sq * m_pos + (1.0 - beta1_sq) * g
        if MOM == 1:
            tl.store(posp + offs, m_pos.to(tl.bfloat16), mask=mask)
        else:
            tl.store(posp + offs, m_pos, mask=mask)
        if CAUTIOUS or CLIP:
            negi = tl.load(neg_addr + t)
            if MOM == 1:
                m_neg = tl.load(negi.to(tl.pointer_type(tl.bfloat16)) + offs, mask=mask, other=0.0).to(tl.float32)
            else:
                m_neg = tl.load(negi.to(tl.pointer_type(tl.float32)) + offs, mask=mask, other=0.0)
            j = offs % C
            rf = tl.load(rfac_ptr + t * R + i, mask=mask, other=0.0)
            cf = tl.load(cfac_ptr + t * C + j, mask=mask, other=0.0)
            urc = ((1.0 + beta0) * m_pos - beta0 * m_neg) * inv_noise * rf * cf
            if CLIP:
                tl.atomic_add(rms_ptr + t, tl.sum(tl.where(mask, urc * urc, 0.0)))
            if CAUTIOUS:
                keep = ((urc * g) > 0.0) & mask
                tl.atomic_add(keep_ptr + t, tl.sum(keep.to(tl.int32)))

    @triton.jit
    def _adapnm_chunked_apply_batched_g(
        g_addr, rowmean_ptr, pos_addr, neg_addr, p_addr, rfac_ptr, cfac_ptr, sc_ptr, inv_mean_ptr,
        R, C, n, K, beta0, inv_noise, lrwd, seed,
        LOWP: tl.constexpr, MOM: tl.constexpr, GC: tl.constexpr, CAUTIOUS: tl.constexpr,
        WD: tl.constexpr, SR: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """As ``_adapnm_chunked_apply_batched`` but grad (for cautious) comes from the pointer array + GC."""
        pid = tl.program_id(0)
        t = pid // K
        k = pid % K
        offs = k * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        i = offs // C
        j = offs % C
        posi = tl.load(pos_addr + t)
        negi = tl.load(neg_addr + t)
        if MOM == 1:
            m_pos = tl.load(posi.to(tl.pointer_type(tl.bfloat16)) + offs, mask=mask, other=0.0).to(tl.float32)
            m_neg = tl.load(negi.to(tl.pointer_type(tl.bfloat16)) + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            m_pos = tl.load(posi.to(tl.pointer_type(tl.float32)) + offs, mask=mask, other=0.0)
            m_neg = tl.load(negi.to(tl.pointer_type(tl.float32)) + offs, mask=mask, other=0.0)
        rf = tl.load(rfac_ptr + t * R + i, mask=mask, other=0.0)
        cf = tl.load(cfac_ptr + t * C + j, mask=mask, other=0.0)
        sc = tl.load(sc_ptr + t)
        pn = ((1.0 + beta0) * m_pos - beta0 * m_neg) * inv_noise
        delta = pn * rf * cf * sc
        if CAUTIOUS:
            gbase = tl.load(g_addr + t)
            gp = gbase.to(tl.pointer_type(tl.bfloat16)) if LOWP else gbase.to(tl.pointer_type(tl.float32))
            g = tl.load(gp + offs, mask=mask, other=0.0).to(tl.float32)
            if GC:
                g = g - tl.load(rowmean_ptr + t * R + i, mask=mask, other=0.0)
            inv_mean = tl.load(inv_mean_ptr + t)
            keep = (delta * g) > 0.0
            # Mask by MULTIPLICATION so a non-finite delta propagates (see _adakaon_tile_kernel).
            delta = delta * tl.where(keep, inv_mean, 0.0)
        pi = tl.load(p_addr + t)
        pp = pi.to(tl.pointer_type(tl.bfloat16)) if LOWP else pi.to(tl.pointer_type(tl.float32))
        p = tl.load(pp + offs, mask=mask, other=0.0).to(tl.float32)
        if WD:
            p = p * (1.0 - lrwd)
        res = p - delta
        if SR:
            res = sr_round(res, seed + t, offs)
        tl.store(pp + offs, res.to(pp.dtype.element_ty), mask=mask)

    # ---- one-block non-factored 1-D AdaPNM path (biases / norm scales) ----
    # Mirrors ``_adam_1d_kernel`` plus AdaPNM's two-momentum structure: full per-coordinate v, EMA only
    # the positive (host passes pos/neg swapped by parity), pos-neg mix / noise_norm, eps on the denom
    # (not eps1), RMS-clip on the v_hat-normalized update, decoupled WD on p (``p *= 1-lr*wd``). fp32/bf16
    # momenta in place (quant 1-D and ams_bound -> native). Matches the native ``_nonfactored_bucket``.

    @triton.jit
    def _adapnm_1d_kernel(
        g_addr, p_addr, pos_addr, neg_addr, v_addr, Ls_ptr,
        beta1_sq, beta0, inv_noise, beta2, step_size, bc2_sq, eps, lrwd, clip, seed,
        LOWP: tl.constexpr, MOM: tl.constexpr, CAUTIOUS: tl.constexpr, WD: tl.constexpr,
        CLIP: tl.constexpr, SR: tl.constexpr, BL: tl.constexpr,
    ):
        """One program == one 1-D tensor. Whole non-factored AdaPNM step, in place via pointer-array."""
        t = tl.program_id(0)
        L = tl.load(Ls_ptr + t)
        Lf = L.to(tl.float32)
        gi = tl.load(g_addr + t)
        pi = tl.load(p_addr + t)
        vp = tl.load(v_addr + t).to(tl.pointer_type(tl.float32))
        if LOWP:
            gp = gi.to(tl.pointer_type(tl.bfloat16))
            pp = pi.to(tl.pointer_type(tl.bfloat16))
        else:
            gp = gi.to(tl.pointer_type(tl.float32))
            pp = pi.to(tl.pointer_type(tl.float32))
        offs = tl.arange(0, BL)
        mask = offs < L
        g = tl.load(gp + offs, mask=mask, other=0.0).to(tl.float32)

        # full per-coordinate second moment (no eps1 in grad^2 for AdaPNM; eps goes on the denom)
        v = tl.load(vp + offs, mask=mask, other=0.0)
        v = beta2 * v + (1.0 - beta2) * (g * g)
        tl.store(vp + offs, v, mask=mask)

        posi = tl.load(pos_addr + t)
        negi = tl.load(neg_addr + t)
        if MOM == 1:  # bf16
            posp = posi.to(tl.pointer_type(tl.bfloat16))
            m_pos = tl.load(posp + offs, mask=mask, other=0.0).to(tl.float32)
            m_neg = tl.load(negi.to(tl.pointer_type(tl.bfloat16)) + offs, mask=mask, other=0.0).to(tl.float32)
        else:         # fp32
            posp = posi.to(tl.pointer_type(tl.float32))
            m_pos = tl.load(posp + offs, mask=mask, other=0.0)
            m_neg = tl.load(negi.to(tl.pointer_type(tl.float32)) + offs, mask=mask, other=0.0)
        m_pos = beta1_sq * m_pos + (1.0 - beta1_sq) * g          # EMA only the positive
        if MOM == 1:
            tl.store(posp + offs, m_pos.to(tl.bfloat16), mask=mask)
        else:
            tl.store(posp + offs, m_pos, mask=mask)

        denom = (tl.sqrt(v + 1e-15) + eps) / bc2_sq
        pn = ((1.0 + beta0) * m_pos - beta0 * m_neg) * inv_noise
        upd = tl.where(mask, pn / denom, 0.0)
        if CLIP:
            rms = tl.sqrt(tl.sum(upd * upd) / Lf)
            d = rms / clip
            d = tl.where(d < 1.0, 1.0, d)
            upd = upd / d
        delta = upd * step_size
        if CAUTIOUS:
            keep = (delta * g) > 0.0
            keepf = tl.where(keep, 1.0, 0.0)
            mm = tl.sum(keepf) / Lf
            mm = tl.where(mm < 1e-8, 1e-8, mm)
            # Mask by MULTIPLICATION so a non-finite delta propagates (see _adakaon_tile_kernel).
            delta = (delta * keepf) / mm
        p_old = tl.load(pp + offs, mask=mask, other=0.0).to(tl.float32)
        if WD:
            p_old = p_old * (1.0 - lrwd)
        res = p_old - delta
        if SR:
            res = sr_round(res, seed + t, offs)
        tl.store(pp + offs, res.to(pp.dtype.element_ty), mask=mask)

    @triton.jit
    def _axpy_4bit_batched(
        p_addr, pk_addr, sc_addr, alpha, clamp, n, K, seed,
        FBLOCK: tl.constexpr, LOWP: tl.constexpr, SR: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """``p += alpha * dequant_4bit(m)`` over a same-shape bucket, one pass, no fp32 temp.

        The MSAM/Nekaon perturbation reads the 4-bit momentum twice per step (climb +
        removal); the torch path materializes the fp32 momentum (unpack nibbles -> sub
        zero -> scale -> stack -> axpy, several kernels + temps). This does the whole
        thing in one launch per bucket: per flat element ``i`` of tensor ``t``,
        ``m = (nibble(i) - 8) * scale[i // FBLOCK]`` (the exact `_dequant_4bit` math:
        low nibble at even ``i``, high at odd), then a bf16-SR-correct in-place axpy.
        Determinism note: dequant is exact, so a +alpha/-alpha round trip lands where
        the torch path lands (same fp32 adds)."""
        pid = tl.program_id(0)
        t = pid // K
        k = pid % K
        offs = k * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        pkp = tl.load(pk_addr + t).to(tl.pointer_type(tl.uint8))
        byte = tl.load(pkp + (offs >> 1), mask=mask, other=0)
        nib = tl.where((offs & 1) == 0, byte & 0x0F, (byte >> 4) & 0x0F).to(tl.float32)
        scp = tl.load(sc_addr + t).to(tl.pointer_type(tl.float32))
        sc = tl.load(scp + offs // FBLOCK, mask=mask, other=0.0)
        m = (nib - 8.0) * sc
        e = alpha * m
        e = tl.where(e != e, 0.0, e)  # NaN (e.g. 0*inf from a blown block scale) -> zero climb
        e = tl.minimum(tl.maximum(e, -clamp), clamp)  # per-element climb bound (MSAM._climb_bound)
        pbase = tl.load(p_addr + t)
        pp = pbase.to(tl.pointer_type(tl.bfloat16)) if LOWP else pbase.to(tl.pointer_type(tl.float32))
        res = tl.load(pp + offs, mask=mask, other=0.0).to(tl.float32) + e
        if LOWP and SR:
            res = sr_round(res, seed + t, offs)
        tl.store(pp + offs, res.to(pp.dtype.element_ty), mask=mask)

    @triton.jit
    def _chunked_nomom_keep_batched_g(
        g_addr, rowmean_ptr, p_addr, c_addr, rfac_ptr, cfac_ptr, keep_ptr,
        rms_ptr, clip, wd, R, C, n, K,
        LOWP: tl.constexpr, GC: tl.constexpr, WD: tl.constexpr,
        BLOCK: tl.constexpr, WDFULL: tl.constexpr = False, CK: tl.constexpr = 0,
    ):
        """Count cautious survivors for a no-momentum chunked update."""
        pid = tl.program_id(0)
        t = pid // K
        k = pid % K
        offs = k * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        i = offs // C
        j = offs % C
        gbase = tl.load(g_addr + t)
        gp = gbase.to(tl.pointer_type(tl.bfloat16)) if LOWP else gbase.to(tl.pointer_type(tl.float32))
        g = tl.load(gp + offs, mask=mask, other=0.0).to(tl.float32)
        if GC:
            g = g - tl.load(rowmean_ptr + t * R + i, mask=mask, other=0.0)
        rf = tl.load(rfac_ptr + t * R + i, mask=mask, other=0.0)
        cf = tl.load(cfac_ptr + t * C + j, mask=mask, other=0.0)
        delta = g * rf * cf * inv_rms_clip(rms_ptr, t, n, clip)
        if WD and not WDFULL:
            pbase = tl.load(p_addr + t)
            pp = pbase.to(tl.pointer_type(tl.bfloat16)) if LOWP else pbase.to(tl.pointer_type(tl.float32))
            delta = wd_keep(delta, g, wd, pp, c_addr, t, offs, mask, CK)
        keep = ((delta * g) > 0.0) & mask
        tl.atomic_add(keep_ptr + t, tl.sum(keep.to(tl.int32)))

    @triton.jit
    def _chunked_nomom_apply_batched_g(
        g_addr, rowmean_ptr, p_addr, c_addr, rfac_ptr, cfac_ptr, rms_ptr, clip,
        inv_mean_ptr, lr, wd, seed, R, C, n, K,
        LOWP: tl.constexpr, GC: tl.constexpr, CAUTIOUS: tl.constexpr,
        WD: tl.constexpr, SR: tl.constexpr, BLOCK: tl.constexpr,
        WDFULL: tl.constexpr = False, CK: tl.constexpr = 0,
    ):
        """Apply a chunked factored update without materializing momentum."""
        pid = tl.program_id(0)
        t = pid // K
        k = pid % K
        offs = k * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        i = offs // C
        j = offs % C
        gbase = tl.load(g_addr + t)
        gp = gbase.to(tl.pointer_type(tl.bfloat16)) if LOWP else gbase.to(tl.pointer_type(tl.float32))
        g = tl.load(gp + offs, mask=mask, other=0.0).to(tl.float32)
        if GC:
            g = g - tl.load(rowmean_ptr + t * R + i, mask=mask, other=0.0)
        rf = tl.load(rfac_ptr + t * R + i, mask=mask, other=0.0)
        cf = tl.load(cfac_ptr + t * C + j, mask=mask, other=0.0)
        delta = g * rf * cf * inv_rms_clip(rms_ptr, t, n, clip)
        pbase = tl.load(p_addr + t)
        pp = pbase.to(tl.pointer_type(tl.bfloat16)) if LOWP else pbase.to(tl.pointer_type(tl.float32))
        p = tl.load(pp + offs, mask=mask, other=0.0).to(tl.float32)
        if CK:
            cp = ck_ptr(c_addr, t, CK)
            zc = ck_decode(pp, cp, offs, mask, CK)
            p = zc  # decay (WD) reads the full decoded value, never the bare bf16
        if WD and not WDFULL:
            delta += wd * p
        if CAUTIOUS:
            keep = (delta * g) > 0.0
            count = tl.load(inv_mean_ptr + t).to(tl.float32)
            inv_mean = n.to(tl.float32) / tl.maximum(count, 1.0)
            # Mask by MULTIPLICATION so a non-finite delta propagates (see _adakaon_tile_kernel).
            delta = delta * tl.where(keep, inv_mean, 0.0)
        if WD and WDFULL:
            delta += wd * p
        if CK:  # compact Kahan: exact compensated value in, (bf16, residual) out
            ck_store(pp, cp, offs, mask, zc - lr * delta, seed + t, CK)
        else:
            res = p - lr * delta
            if SR:
                res = sr_round(res, seed + t, offs)
            tl.store(pp + offs, res.to(pp.dtype.element_ty), mask=mask)

    @triton.jit
    def _chunked_4bit_keep_batched_g(
        g_addr, rowmean_ptr, packed_addr, scale_addr, p_addr, c_addr, rfac_ptr, cfac_ptr,
        keep_ptr, rms_ptr, clip, wd, beta1, R, C, n, K,
        LOWP: tl.constexpr, GC: tl.constexpr, WD: tl.constexpr,
        FBLOCK: tl.constexpr, BLOCK: tl.constexpr, WDFULL: tl.constexpr = False,
        CK: tl.constexpr = 0,
    ):
        """Count cautious survivors from the exact pre-requantized 4-bit EMA.

        State is deliberately left untouched: the apply kernel recomputes the
        same EMA, uses it for the weight update, then requantizes. This preserves
        native codec semantics without an fp32 momentum-sized temporary.
        """
        pid = tl.program_id(0)
        t = pid // K
        k = pid % K
        local = tl.arange(0, BLOCK)
        offs = k * BLOCK + local
        mask = offs < n
        i = offs // C
        j = offs % C
        gbase = tl.load(g_addr + t)
        gp = gbase.to(tl.pointer_type(tl.bfloat16)) if LOWP else gbase.to(tl.pointer_type(tl.float32))
        g = tl.load(gp + offs, mask=mask, other=0.0).to(tl.float32)
        if GC:
            g -= tl.load(rowmean_ptr + t * R + i, mask=mask, other=0.0)
        upd = g * tl.load(rfac_ptr + t * R + i, mask=mask, other=0.0)
        upd *= tl.load(cfac_ptr + t * C + j, mask=mask, other=0.0)
        upd *= inv_rms_clip(rms_ptr, t, n, clip)
        packed = tl.load(packed_addr + t).to(tl.pointer_type(tl.uint8))
        scales = tl.load(scale_addr + t).to(tl.pointer_type(tl.float32))
        byte = tl.load(packed + offs // 2, mask=mask, other=0)
        nib = tl.where((offs & 1) == 0, byte & 0x0F, (byte >> 4) & 0x0F)
        old = (nib.to(tl.float32) - 8.0) * tl.load(scales + offs // FBLOCK, mask=mask, other=0.0)
        momentum = beta1 * old + (1.0 - beta1) * upd
        delta = momentum
        if WD and not WDFULL:
            pbase = tl.load(p_addr + t)
            pp = pbase.to(tl.pointer_type(tl.bfloat16)) if LOWP else pbase.to(tl.pointer_type(tl.float32))
            delta = wd_keep(delta, g, wd, pp, c_addr, t, offs, mask, CK)
        keep = ((delta * g) > 0.0) & mask
        tl.atomic_add(keep_ptr + t, tl.sum(keep.to(tl.int32)))

    @triton.jit
    def _chunked_4bit_apply_batched_g(
        g_addr, rowmean_ptr, packed_addr, scale_addr, p_addr, c_addr, rfac_ptr, cfac_ptr,
        keep_ptr, rms_ptr, clip, lr, wd, beta1, seed, R, C, n, K,
        LOWP: tl.constexpr, GC: tl.constexpr, CAUTIOUS: tl.constexpr,
        WD: tl.constexpr, SR: tl.constexpr, FBLOCK: tl.constexpr, BLOCK: tl.constexpr,
        WDFULL: tl.constexpr = False, CK: tl.constexpr = 0,
    ):
        """Exact update plus in-kernel 4-bit requantization for a chunked tensor.

        ``BLOCK`` is an integer multiple of ``FBLOCK``; consequently each codec
        block has exactly one writer and requires no cross-program reduction.
        """
        pid = tl.program_id(0)
        t = pid // K
        k = pid % K
        local = tl.arange(0, BLOCK)
        offs = k * BLOCK + local
        mask = offs < n
        i = offs // C
        j = offs % C
        gbase = tl.load(g_addr + t)
        gp = gbase.to(tl.pointer_type(tl.bfloat16)) if LOWP else gbase.to(tl.pointer_type(tl.float32))
        g = tl.load(gp + offs, mask=mask, other=0.0).to(tl.float32)
        if GC:
            g -= tl.load(rowmean_ptr + t * R + i, mask=mask, other=0.0)
        upd = g * tl.load(rfac_ptr + t * R + i, mask=mask, other=0.0)
        upd *= tl.load(cfac_ptr + t * C + j, mask=mask, other=0.0)
        upd *= inv_rms_clip(rms_ptr, t, n, clip)
        packed = tl.load(packed_addr + t).to(tl.pointer_type(tl.uint8))
        scales = tl.load(scale_addr + t).to(tl.pointer_type(tl.float32))
        byte = tl.load(packed + offs // 2, mask=mask, other=0)
        old_nib = tl.where((offs & 1) == 0, byte & 0x0F, (byte >> 4) & 0x0F)
        old_scale = tl.load(scales + offs // FBLOCK, mask=mask, other=0.0)
        momentum = beta1 * ((old_nib.to(tl.float32) - 8.0) * old_scale) + (1.0 - beta1) * upd

        pbase = tl.load(p_addr + t)
        pp = pbase.to(tl.pointer_type(tl.bfloat16)) if LOWP else pbase.to(tl.pointer_type(tl.float32))
        p = tl.load(pp + offs, mask=mask, other=0.0).to(tl.float32)
        if CK:
            cp = ck_ptr(c_addr, t, CK)
            zc = ck_decode(pp, cp, offs, mask, CK)
            p = zc  # decay (WD) reads the full decoded value, never the bare bf16
        delta = momentum
        if WD and not WDFULL:
            delta += wd * p
        if CAUTIOUS:
            count = tl.load(keep_ptr + t).to(tl.float32)
            keep = (delta * g) > 0.0
            # Mask by MULTIPLICATION so a non-finite delta propagates (see _adakaon_tile_kernel).
            delta = delta * tl.where(keep, n.to(tl.float32) / tl.maximum(count, 1.0), 0.0)
        if WD and WDFULL:
            # EXPLICIT fma: ``delta * scale + wd * p`` has two legal contractions, and the
            # compiler picked ``fma(wd, p, delta*scale)`` for the fp32 variant but
            # ``fma(delta, scale, wd*p)`` for the CK one — kahan16 then left its fp32 twin by
            # an fp32 ulp per step. Pinning the one the fp32 / SR variants already compiled
            # to keeps them bit-identical and makes CK agree.
            delta = tl.fma(wd, p, delta)
        if CK:  # compact Kahan: exact compensated value in, (bf16, residual) out
            ck_store(pp, cp, offs, mask, zc - lr * delta, seed + t, CK)
        else:
            res = p - lr * delta
            if SR:
                res = sr_round(res, seed + t, offs)
            tl.store(pp + offs, res.to(pp.dtype.element_ty), mask=mask)

        # Segmented absmax/requant. Chunk and codec block boundaries are aligned,
        # including the final partial chunk; padded lanes quantize to the zero nibble.
        #
        # The per-block absmax is ONE axis reduction over a ``(BLOCK//FBLOCK, FBLOCK)``
        # reshape and the scale is broadcast back FROM REGISTERS. It used to be a loop of
        # ``BLOCK//FBLOCK`` full-tile maxima followed by ``tl.load(scales + offs // FBLOCK)``
        # — reloading the values this same program had just stored. That reload was a genuine
        # RACE: the storing lane and the reading lanes are different lanes of the same
        # program, with no barrier between the store and the load, so a lane could quantize
        # against the PREVIOUS step's scale. It showed up as run-to-run nondeterminism that
        # survived ``deterministic_reductions`` (5.1e-4 relative spread over 4 identical runs
        # with 4-bit momentum, against 0 for every other momentum dtype once the reductions
        # were made two-pass). Keeping the scale in registers removes both the race and the
        # loop; see ``requant_4bit``'s EXACT path for the same reduction shape.
        base_block = (k * BLOCK) // FBLOCK
        nb: tl.constexpr = BLOCK // FBLOCK
        seg = tl.max(tl.abs(tl.reshape(tl.where(mask, momentum, 0.0), (nb, FBLOCK))), axis=1)
        seg = tl.maximum(seg, 1e-12) / 7.0                       # [nb] per-block scale
        bb = tl.arange(0, nb)
        tl.store(scales + base_block + bb, seg, mask=(base_block + bb) * FBLOCK < n)
        new_scale = tl.reshape(tl.broadcast_to(seg[:, None], (nb, FBLOCK)), (BLOCK,))
        q = libdevice.rint(momentum / new_scale)
        q = tl.minimum(tl.maximum(q, -7.0), 7.0)
        nib = tl.where(mask, (q + 8.0).to(tl.uint8), 0)
        lo, hi = tl.split(tl.reshape(nib, (BLOCK // 2, 2)))
        packed_byte = lo | (hi << 4)
        jj = tl.arange(0, BLOCK // 2)
        byte_offs = (k * BLOCK) // 2 + jj
        tl.store(packed + byte_offs, packed_byte, mask=byte_offs < (n + 1) // 2)

    # ---- direct in-kernel INT8 momentum for the batched chunked path ----
    # The mirror of the 4-bit pair above, and it exists for the same reason: without it an int8
    # bucket routes through the host codec — ``dequant_stacked`` to a momentum-sized fp32 [N,R,C]
    # temp, the two generic ``_g`` kernels against that temp, then ``_quant_int8_stacked`` back —
    # which is ~15 extra torch kernels, one ``ptr_array`` host->device copy (a CPU<->GPU sync) and
    # 4 B/param of transient the rest of the path spent 0.7.7-0.7.11 removing.
    #
    # WHY IT NEEDS A CONDITION. int8's scale is per ROW (``_quant_int8`` reduces dim 0), so the
    # requant is a segmented absmax whose segments are rows. A chunk can own whole rows only when
    # ``BLOCK % C == 0`` (and ``C <= BLOCK``); then every row has exactly ONE writing program and
    # no cross-program reduction is needed. Chunk boundaries are multiples of ``BLOCK`` hence of
    # ``C``, and ``n == R*C``, so even the final partial chunk ends on a row boundary. Buckets that
    # fail the condition keep the codec fallback — see ``Adakaon._chunked_step_batched``.
    #
    # ``CSEG``/``RPC`` are constexpr so the per-row reduction is ONE ``tl.max`` over a
    # ``(RPC, CSEG)`` reshape instead of an ``RPC``-iteration loop over the whole tile (which is
    # what ``requant_4bit`` does, and is O(numel * blocks) in registers). The cost is one JIT
    # variant per distinct big shape; the host already caches per shape bucket.
    @triton.jit
    def _chunked_int8_keep_batched_g(
        g_addr, rowmean_ptr, code_addr, scale_addr, p_addr, c_addr, rfac_ptr, cfac_ptr,
        keep_ptr, rms_ptr, clip, wd, beta1, R, C, n, K,
        LOWP: tl.constexpr, GC: tl.constexpr, WD: tl.constexpr, BLOCK: tl.constexpr,
        WDFULL: tl.constexpr = False, CK: tl.constexpr = 0,
    ):
        """Count cautious survivors from the exact pre-requantized int8 EMA.

        State is deliberately left untouched: the apply kernel recomputes the same EMA, uses it
        for the weight update, then requantizes — the same two-pass shape as the 4-bit pair, so
        the delta the weight sees is the exact fp32 EMA (native codec semantics) with no temp.
        """
        pid = tl.program_id(0)
        t = pid // K
        k = pid % K
        offs = k * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        i = offs // C
        j = offs % C
        gbase = tl.load(g_addr + t)
        gp = gbase.to(tl.pointer_type(tl.bfloat16)) if LOWP else gbase.to(tl.pointer_type(tl.float32))
        g = tl.load(gp + offs, mask=mask, other=0.0).to(tl.float32)
        if GC:
            g -= tl.load(rowmean_ptr + t * R + i, mask=mask, other=0.0)
        upd = g * tl.load(rfac_ptr + t * R + i, mask=mask, other=0.0)
        upd *= tl.load(cfac_ptr + t * C + j, mask=mask, other=0.0)
        upd *= inv_rms_clip(rms_ptr, t, n, clip)
        codes = tl.load(code_addr + t).to(tl.pointer_type(tl.int8))
        scales = tl.load(scale_addr + t).to(tl.pointer_type(tl.float32))
        old = tl.load(codes + offs, mask=mask, other=0).to(tl.float32)
        old *= tl.load(scales + i, mask=mask, other=0.0)          # per-row dequant
        delta = beta1 * old + (1.0 - beta1) * upd
        if WD and not WDFULL:
            pbase = tl.load(p_addr + t)
            pp = pbase.to(tl.pointer_type(tl.bfloat16)) if LOWP else pbase.to(tl.pointer_type(tl.float32))
            delta = wd_keep(delta, g, wd, pp, c_addr, t, offs, mask, CK)
        keep = ((delta * g) > 0.0) & mask
        tl.atomic_add(keep_ptr + t, tl.sum(keep.to(tl.int32)))

    @triton.jit
    def _chunked_int8_apply_batched_g(
        g_addr, rowmean_ptr, code_addr, scale_addr, p_addr, c_addr, rfac_ptr, cfac_ptr,
        keep_ptr, rms_ptr, clip, lr, wd, beta1, seed, R, C, n, K,
        LOWP: tl.constexpr, GC: tl.constexpr, CAUTIOUS: tl.constexpr, WD: tl.constexpr,
        SR: tl.constexpr, CSEG: tl.constexpr, RPC: tl.constexpr, BLOCK: tl.constexpr,
        WDFULL: tl.constexpr = False, CK: tl.constexpr = 0,
    ):
        """Exact update plus in-kernel per-row int8 requantization for a chunked tensor.

        ``BLOCK == RPC * CSEG`` with ``CSEG == C``: this program owns ``RPC`` COMPLETE rows, so
        the per-row absmax has a single writer and needs no cross-program reduction.
        """
        pid = tl.program_id(0)
        t = pid // K
        k = pid % K
        offs = k * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        i = offs // C
        j = offs % C
        gbase = tl.load(g_addr + t)
        gp = gbase.to(tl.pointer_type(tl.bfloat16)) if LOWP else gbase.to(tl.pointer_type(tl.float32))
        g = tl.load(gp + offs, mask=mask, other=0.0).to(tl.float32)
        if GC:
            g -= tl.load(rowmean_ptr + t * R + i, mask=mask, other=0.0)
        upd = g * tl.load(rfac_ptr + t * R + i, mask=mask, other=0.0)
        upd *= tl.load(cfac_ptr + t * C + j, mask=mask, other=0.0)
        upd *= inv_rms_clip(rms_ptr, t, n, clip)
        codes = tl.load(code_addr + t).to(tl.pointer_type(tl.int8))
        scales = tl.load(scale_addr + t).to(tl.pointer_type(tl.float32))
        old = tl.load(codes + offs, mask=mask, other=0).to(tl.float32)
        old *= tl.load(scales + i, mask=mask, other=0.0)
        momentum = beta1 * old + (1.0 - beta1) * upd

        pbase = tl.load(p_addr + t)
        pp = pbase.to(tl.pointer_type(tl.bfloat16)) if LOWP else pbase.to(tl.pointer_type(tl.float32))
        p = tl.load(pp + offs, mask=mask, other=0.0).to(tl.float32)
        if CK:
            cp = ck_ptr(c_addr, t, CK)
            zc = ck_decode(pp, cp, offs, mask, CK)
            p = zc  # decay (WD) reads the full decoded value, never the bare bf16
        delta = momentum
        if WD and not WDFULL:
            delta += wd * p
        if CAUTIOUS:
            count = tl.load(keep_ptr + t).to(tl.float32)
            keep = (delta * g) > 0.0
            # Mask by MULTIPLICATION so a non-finite delta propagates (see _adakaon_tile_kernel).
            delta = delta * tl.where(keep, n.to(tl.float32) / tl.maximum(count, 1.0), 0.0)
        if WD and WDFULL:
            # EXPLICIT fma: ``delta * scale + wd * p`` has two legal contractions, and the
            # compiler picked ``fma(wd, p, delta*scale)`` for the fp32 variant but
            # ``fma(delta, scale, wd*p)`` for the CK one — kahan16 then left its fp32 twin by
            # an fp32 ulp per step. Pinning the one the fp32 / SR variants already compiled
            # to keeps them bit-identical and makes CK agree.
            delta = tl.fma(wd, p, delta)
        if CK:  # compact Kahan: exact compensated value in, (bf16, residual) out
            ck_store(pp, cp, offs, mask, zc - lr * delta, seed + t, CK)
        else:
            res = p - lr * delta
            if SR:
                res = sr_round(res, seed + t, offs)
            tl.store(pp + offs, res.to(pp.dtype.element_ty), mask=mask)

        # Per-row absmax / 127, round half-to-even, clamp — element-for-element
        # ``_quant_int8``. One reduce over the row axis of a (RPC, CSEG) reshape; padded lanes
        # of a final partial chunk carry 0.0 so they neither bias a row's absmax nor get stored.
        mrows = tl.reshape(tl.where(mask, momentum, 0.0), (RPC, CSEG))
        amax = tl.maximum(tl.max(tl.abs(mrows), axis=1), 1e-12)
        new_scale = amax / 127.0                                  # [RPC]
        q = libdevice.rint(mrows / new_scale[:, None])
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        tl.store(codes + offs, tl.reshape(q, (BLOCK,)).to(tl.int8), mask=mask)
        rows = (k * BLOCK) // CSEG + tl.arange(0, RPC)
        tl.store(scales + rows, new_scale, mask=rows < R)

    @triton.jit
    def _axpy_momentum_batched(
        p_addr, c_addr, m_addr, mscale_addr, alpha, clamp, n, K, row_width, seed,
        MOM: tl.constexpr, FBLOCK: tl.constexpr, LOWP: tl.constexpr,
        SR: tl.constexpr, BLOCK: tl.constexpr, CK: tl.constexpr = 0,
    ):
        """Fused ``p += alpha*m`` for every Kaon momentum storage format.

        This is the shared MSAM/Nekaon perturbation pass.  One program owns a
        flat chunk of one tensor and dequantizes momentum directly from its
        persistent storage, avoiding a stacked fp32 temporary and one Python
        stochastic-rounding call per parameter.

        ``CK`` (compact Kahan, ``bf16_method="kahan8"`` = 8, ``"kahan16"`` = 16): the climb is
        applied to the DECODED compensated value and re-encoded through ``ck_store``
        (stochastic rounding of the residual, seeded from ``seed``; none at 16 bits), so a
        climb/removal pair leaves the clean value intact to ~1/256 ulp, unbiased (to fp32's
        own rounding for ``kahan16``). Perturbing the bare bf16 weight instead loses the
        sub-ulp part of the climb coherently on every step — see kaon._compact_kahan.
        """
        pid = tl.program_id(0)
        t = pid // K
        k = pid % K
        offs = k * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        mb = tl.load(m_addr + t)
        if MOM == 3:  # packed 4-bit, flat block scales
            mp = mb.to(tl.pointer_type(tl.uint8))
            byte = tl.load(mp + (offs >> 1), mask=mask, other=0)
            nib = tl.where((offs & 1) == 0, byte & 0x0F, (byte >> 4) & 0x0F).to(tl.float32)
            sp = tl.load(mscale_addr + t).to(tl.pointer_type(tl.float32))
            scale = tl.load(sp + offs // FBLOCK, mask=mask, other=0.0)
            m = (nib - 8.0) * scale
        elif MOM == 2:  # int8, one scale per leading-dimension row
            mp = mb.to(tl.pointer_type(tl.int8))
            code = tl.load(mp + offs, mask=mask, other=0).to(tl.float32)
            sp = tl.load(mscale_addr + t).to(tl.pointer_type(tl.float32))
            scale = tl.load(sp + offs // row_width, mask=mask, other=0.0)
            m = code * scale
        elif MOM == 1:
            mp = mb.to(tl.pointer_type(tl.bfloat16))
            m = tl.load(mp + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            mp = mb.to(tl.pointer_type(tl.float32))
            m = tl.load(mp + offs, mask=mask, other=0.0)

        e = alpha * m
        e = tl.where(e != e, 0.0, e)
        e = tl.minimum(tl.maximum(e, -clamp), clamp)
        pbase = tl.load(p_addr + t)
        pp = pbase.to(tl.pointer_type(tl.bfloat16)) if LOWP else pbase.to(tl.pointer_type(tl.float32))
        if CK:
            cp = ck_ptr(c_addr, t, CK)
            ck_store(pp, cp, offs, mask, ck_decode(pp, cp, offs, mask, CK) + e, seed + t, CK)
        else:
            res = tl.load(pp + offs, mask=mask, other=0.0).to(tl.float32) + e
            if LOWP and SR:
                res = sr_round(res, seed + t, offs)
            tl.store(pp + offs, res.to(pp.dtype.element_ty), mask=mask)


# ============================================================ pointer-array cache (reusable)
class _WitnessedCache:
    """Shared staleness witness for every cached pointer array — see :func:`param_witness`.

    ``id(p)`` ALONE IS NOT A WITNESS. ``p.data = ...`` rebinds the SAME Parameter object to
    fresh storage — an external EMA writing weights back, a ``.to(dtype/device)``, or a
    block-swap offloader that re-materializes a block on every pull and frees it on evict.
    The ids still match, the cached ``data_ptr`` does not, and the kernel writes the whole
    step into memory the optimizer no longer owns: silent corruption while the buffer is
    still mapped, an illegal memory access once the allocator hands it back. Confirmed on
    all four routes (one-block, 1-D, 0-D, big batched). Contiguity closes the one rebind that
    keeps the pointer and only moves the layout (``p.data.t()``); the one that moves the SHAPE
    (``p.data.view(...)``) needs the opt-in :data:`SHAPE_WITNESS` field.

    A rebind that changes the shape is an UNSUPPORTED operation, so detecting it means refusing
    the step, never adapting to it — and the refusal is :func:`check_state_geometry`, which every
    subclass calls at build right after the witness. That is what makes a witness MOVE safe: a
    rebuilt plan would otherwise carry the new ``R``/``C`` against the old ``row``/``col``
    buffers and read past them. The native plan guards itself with the same three base fields
    (``kaon._foreach_plan.param_witness``) and MSAM with ``_plan_addrs_valid``; the grad side is
    already covered per step by ``refresh_grads``.

    ``gen`` IS THE SECOND HALF OF THE WITNESS, and it watches what no parameter field can:
    the identity of the STATE. The tables above bake ``data_ptr``s of ``m``/``m_scale``/
    ``row``/``col``/``v`` at build; the fields above observe the PARAMS. So a state buffer
    retired while every param stood still left the tables addressing a dead tensor and the
    step wrote it — measured on all four routes for ``del opt.state[p]`` (the recovery
    :func:`check_state_geometry` documents), for ``opt.state[p].clear()`` (worse: the same
    dict is refilled with fresh buffers, so both generations are live at once) and for
    ``opt.state[p]["m"] = ...`` (an external EMA, a partial ``load_state_dict`` that bypasses
    the optimizer's own loader, a requant that does not follow the in-place codec contract).
    ``gen`` is ``kaon._foreach_plan.state_generation(opt.state)`` — a counter the state mapping
    itself moves on every such rebinding (:class:`~kaon._foreach_plan.WatchedState`), so the
    check costs ONE integer compare instead of a per-param sweep and never fires in steady
    state. It is checked BEFORE the tuple compare in every method here, and it defaults to
    ``_foreach_plan._NO_GENERATION`` (0) so a caller that does not watch its state — or a
    test building a cache directly — behaves exactly as before.
    """

    #: Overwritten per instance by :meth:`_witness`; a class default keeps a cache built
    #: through an older code path comparable.
    gen: int = 0

    def _witness(self, plist, gen: int = 0) -> None:
        """Record the witness for ``plist`` (call once, at cache build)."""
        self.witness = param_witness(plist)
        self.ids, self.ptrs = self.witness[0], self.witness[1]
        self.src = plist
        self.gen = gen

    def stale(self, plist, gen: int = 0) -> bool:
        """True when the cached pointer arrays no longer describe ``plist`` or its state.

        Called per step by callers that build ``plist`` fresh each time (the big-tensor shape
        buckets): see :func:`param_witness` for what it observes and what each field costs.
        """
        return self.gen != gen or self.witness != param_witness(plist)

    def built_from(self, plist, gen: int = 0) -> bool:
        """O(1) alternative to :meth:`stale` for a caller that ALREADY revalidated the witness.

        ``Adakaon._fused_partition`` compares ids and ``data_ptr``s across the whole group
        every step and hands back the very same route lists while nothing changed; any change
        (or the per-step non-contiguous-grad demotion) rebuilds them into FRESH list objects.
        Under that contract, "this is the list I was built from" is exactly as strong as
        recomparing the tuples — and skips a second witness sweep per step (~131 µs on the
        428-param bag). Callers without that contract must use :meth:`stale` — or
        :meth:`revalidate`, which is that combination done without desynchronising.

        List identity says nothing about the STATE, so ``gen`` is checked here too — see the
        class docstring. It is not redundant with the caller's own generation check even where
        one exists (``Adakaon._fused_partition`` has it): that one makes the partition hand
        back fresh lists, which is a route from the check to this cache, not the check itself.
        """
        return plist is self.src and self.gen == gen

    def revalidate(self, plist, gen: int = 0) -> bool:
        """True when this cache still describes ``plist``, ADOPTING a fresh list object
        that describes the same parameters. False means the tables must be rebuilt.

        For callers whose ``plist`` is *usually* the list the cache was built from but
        can legitimately be re-derived into a new object over unchanged parameters —
        AdaPNM's routes, where a mixed-lag group re-splits every step and a
        non-contiguous-grad demotion rebuilds all four route lists for one step.

        The naive combination ``not built_from(plist) and stale(plist)`` looks
        equivalent and is not: it leaves the cache holding ``src`` from the PREVIOUS
        generation whenever the witness agrees, so ``built_from`` fails from then on and
        the ``param_witness`` sweep comes back **permanently** — measured on AdaPNM as
        1.0 → 3.17 sweeps per step after a single strided-grad step, and it never
        recovers. The missing half is the rebind on this line: when the witness says
        nothing moved, the fresh list is as good as the old one, so take it and let the
        next step hit the O(1) identity path again.

        One witness sweep at most, and only when identity misses.

        ``gen`` IS LOAD-BEARING HERE AND NOWHERE MORE SO. This method deliberately ADOPTS a
        fresh list whenever the param witness agrees — which is precisely what stopped a
        caller's own state check from ever reaching AdaPNM's routes: the partition rebuilt,
        handed the route a brand-new list, and ``revalidate`` said "same parameters, take it"
        while the tables still addressed the retired ``m_pos``/``row`` (measured: 4/4 buffers
        written on the one-block route, 3/3 on 1-D). Checking the generation FIRST is what
        closes it, and it must stay first: adopting the list before refusing would mark the
        stale tables as describing the current generation.
        """
        if self.gen != gen:
            return False
        if plist is self.src:
            return True
        witness = param_witness(plist)
        if witness != self.witness:
            return False
        self.src = plist            # same parameters, new list object: adopt it
        return True


# ============================================================ Triton bf16 SR weight write
# Fallback noise stream for :func:`_sr_axpy_kernel`, for a caller that hands over no stream
# of its own. Every kaon optimizer DOES hand one over (that is how its noise position gets
# into its ``state_dict`` and stays independent of the other optimizers in the process —
# see :class:`kaon._stochastic_rounding.SRStream`), so this serves direct callers of
# :func:`sr_add_` only: benchmarks, tests, downstream code.
#
# It takes ``stream_id=0`` explicitly rather than an allocated id, for two reasons: it must
# not consume the id the first optimizer needs (stream 0 on cuda:0 is the compatibility
# anchor that reproduces 0.7.12's sequence), and the fallback itself should keep producing
# exactly the sequence 0.7.12's global counter produced. Sharing the id with the first
# optimizer is harmless: streams collide only if they are used at the same time, and a run
# that threads streams never reaches here.
#
# ``kaon.reseed_stochastic_rounding()`` restarts it like every other stream (the module
# epoch, checked on use) — when this counter was left out of that call, a bf16 run on a
# Triton build silently stopped reproducing under ``torch.manual_seed(s)`` + that call.
_PROCESS_SR_STREAM = SRStream(stream_id=0)


def sr_add_supported(target, source) -> bool:
    """Can :func:`sr_add_` take this pair? bf16 CUDA target, fp32 source, both contiguous.

    Contiguity is not a nicety: the kernel indexes both buffers as ``base + offs``, so a
    strided view would be read and written in the wrong places (the same trap
    ``_demote_non_contiguous_grads`` exists for on the gradient side).
    """
    return (
        _HAS_TRITON
        and target.is_cuda
        and target.dtype == torch.bfloat16
        and source.dtype == torch.float32
        and target.is_contiguous()
        and source.is_contiguous()
        and target.numel() == source.numel()
    )


@torch.no_grad()
def sr_add_(target, source, alpha: float = 1.0, sr: SRStream | None = None) -> None:
    """``target += alpha * source`` with bf16 stochastic rounding, in ONE Triton launch.

    The caller must have checked :func:`sr_add_supported`.

    ``sr`` is the caller's noise stream; its next draw becomes the launch's seed. Passing
    one costs the same three integer ops the process-wide fallback costs (no extra kernel,
    no sync) and is what makes the draw checkpointable — see
    :class:`kaon._stochastic_rounding.SRStream`.
    """
    n = target.numel()
    stream = _PROCESS_SR_STREAM if sr is None else sr
    # Drawn OUTSIDE the device guard, as the global counter was: for a device without an
    # explicit index the draw reads the ambient current device, and moving it inside would
    # silently change which device's global seed the stream binds to.
    seed = stream.next_seed(target.device)
    with torch.cuda.device(target.device):   # a launch targets the CURRENT device
        _sr_axpy_kernel[((n + 1023) // 1024,)](
            target, source, alpha, n, seed, BLOCK=1024,
        )


def ck_add_supported(target, lo, source, bits: int = 8) -> bool:
    """Can :func:`ck_add_` take this triple? bf16 CUDA target, a residual in the ``bits``-wide
    codec's dtype (``uint8`` / ``int16``), fp32 source, all contiguous and the same numel (the
    kernel indexes all three as ``base + offs``)."""
    return (
        _HAS_TRITON
        and target.is_cuda
        and target.dtype == torch.bfloat16
        and lo.dtype == (torch.int16 if bits == 16 else torch.uint8)
        and source.dtype == torch.float32
        and target.is_contiguous()
        and lo.is_contiguous()
        and source.is_contiguous()
        and target.numel() == source.numel() == lo.numel()
        and lo.device == target.device
    )


def ck_decode_supported(p, lo) -> bool:
    """Can :func:`ck_decode_fast` take this pair? bf16 CUDA weight, a uint8 / int16 residual,
    both contiguous, same numel and device."""
    return (
        _HAS_TRITON
        and p.is_cuda
        and p.dtype == torch.bfloat16
        and lo.dtype in (torch.uint8, torch.int16)
        and p.is_contiguous()
        and lo.is_contiguous()
        and p.numel() == lo.numel()
        and lo.device == p.device
    )


@torch.no_grad()
def ck_decode_fast(p, lo, bits: int):
    """The fp32 value of ``(p, lo)`` in ONE Triton launch (caller checked
    :func:`ck_decode_supported`); bit-identical to :func:`kaon._compact_kahan.decode`."""
    out = torch.empty(p.shape, dtype=torch.float32, device=p.device)
    n = p.numel()
    if n:
        with torch.cuda.device(p.device):
            _ck_decode_kernel[((n + 1023) // 1024,)](p, lo, out, n, BITS=bits, BLOCK=1024)
    return out


@torch.no_grad()
def ck_add_(target, lo, source, alpha: float = 1.0, bits: int = 8, sr: SRStream | None = None) -> None:
    """``(target, lo) += alpha * source`` with the compact-Kahan codec, in ONE Triton launch.

    The caller must have checked :func:`ck_add_supported`. ``sr`` is the caller's noise
    stream, exactly as for :func:`sr_add_` (its next draw seeds the residual's stochastic
    rounding; checkpointed, so a resume reproduces it).
    """
    n = target.numel()
    stream = _PROCESS_SR_STREAM if sr is None else sr
    seed = stream.next_seed(target.device)
    with torch.cuda.device(target.device):
        _ck_axpy_kernel[((n + 1023) // 1024,)](
            target, lo, source, alpha, n, seed, BITS=bits, BLOCK=1024,
        )


def fourbit_kernel_blocks(numel: int, block: int = 0) -> int:
    """Number of 4-bit absmax blocks a one-block tile kernel writes for an ``numel``-element
    tensor under ``block``-element absmax blocks — i.e. the ``m_scale`` capacity that layout
    needs. ``block <= 0`` means the legacy hardcoded ``min(numel, 128)`` (still what
    :class:`AdaPnmCache` gets, since ``_adapnm_tile_kernel`` keeps the constant); Adakaon's
    tile kernel takes the block as a runtime scalar and passes the bucket's real
    ``state["m_block"]`` (see :class:`PointerArrayCache`)."""
    bs = min(numel, 128) if block <= 0 else block
    return (numel + max(bs, 1) - 1) // max(bs, 1)


class PointerArrayCache(_WitnessedCache):
    """Per-tensor pointer arrays, BUCKETED by padded tile, cached across steps.

    Optimizer-agnostic plumbing. A single global (BR,BC)=max would pad every tiny adapter up to the
    largest tensor's tile (and run *slower* than native); bucketing by exact tile keeps each
    tensor's work proportional to its own size — one kernel launch per distinct tile. Stable tensors
    (p / m / row / col) are addressed once; the grad pointer array is rebuilt only when a grad
    tensor is reallocated (identity check), so the steady-state per-step host cost is ~0.
    """

    def __init__(self, plist, state_of, mom_dtype, gen: int = 0):
        self._witness(plist, gen)
        check_state_geometry(plist, state_of, factored=True)
        # The device is part of the bucket key (and every index array is built ON that device):
        # one launch owns one device, and a group holding params on two of them would otherwise
        # hand a kernel a pointer array from the wrong context.
        #
        # ``m_block`` (0 for every non-4-bit momentum, so it never fragments them) joins the key
        # because the tile kernel takes the 4-bit absmax block as ONE runtime scalar for the whole
        # launch: two tensors sharing a tile can still carry different block layouts (a non-default
        # ``momentum_4bit_block``, ``0`` = whole-tensor with different numels, or a checkpoint's
        # layout), and one launch cannot serve both.
        groups: dict[tuple[int, int, int, torch.dtype, torch.device], list] = {}
        for p in plist:
            br, bc = next_pow2_tile(*eff_2d(p))
            groups.setdefault((br, bc, state_of(p).get("m_block", 0), p.dtype, p.device), []).append(p)
        self.buckets = []
        for (BR, BC, blk, _dtype, dev), bl in groups.items():  # noqa: N806
            i64 = lambda xs, _d=dev: torch.tensor(xs, dtype=torch.int64, device=_d)  # noqa: E731
            i32 = lambda xs, _d=dev: torch.tensor(xs, dtype=torch.int32, device=_d)  # noqa: E731
            st = [state_of(p) for p in bl]
            momentum = "m" in st[0]
            mdtype = st[0]["m"].dtype if momentum else torch.float32
            if mdtype == torch.int8:
                mom = MOM_INT8
            elif mdtype == torch.uint8:
                mom = MOM_4BIT
            elif mdtype == torch.bfloat16:
                mom = MOM_BF16
            else:
                mom = MOM_FP32
            m_addr = (
                i64([s["m"].data_ptr() for s in st])
                if momentum else i64([s["row"].data_ptr() for s in st])
            )
            # int8/4bit need a per-tensor pointer array to the fp32 scales; float kinds never
            # dereference mscale (constexpr-elided), so reuse m_addr as a harmless valid pointer.
            quant = mom in (MOM_INT8, MOM_4BIT)
            mscale_addr = i64([s["m_scale"].data_ptr() for s in st]) if quant else m_addr
            Rs = i32([p.shape[0] for p in bl])  # noqa: N806
            # Per-tensor m_scale CAPACITY for the 4-bit requant's bounded scale store. The tile
            # kernel writes ``fourbit_kernel_blocks(numel, blk)`` scales for THIS bucket's runtime
            # block; a shorter buffer would run off the end of ``m_scale``. In practice the two
            # coincide by construction (the bucket key IS ``m_block``), so this is the second line
            # of defence, not the routing decision it used to back. Float momenta never
            # dereference this array (the branch is constexpr-elided), so they reuse ``Rs``.
            if mom == MOM_4BIT:
                have = [s["m_scale"].numel() for s in st]
                short = [(tuple(q.shape), h) for q, h in zip(bl, have, strict=True)
                         if h < fourbit_kernel_blocks(q.numel(), blk)]
                if short:
                    raise RuntimeError(
                        f"fused one-block 4-bit momentum: m_scale is shorter than the {blk}-element "
                        f"absmax block layout needs for {short[:4]} - route them to the native path "
                        "(see Adakaon._fused_partition)"
                    )
                mscale_n = i32(have)
            else:
                mscale_n = Rs
            # 4-bit single-reduction fast path: every tensor in the bucket must fill the
            # padded tile exactly, so the flat block segments line up with a reshape of the
            # tile (see requant_4bit), AND the block must divide the tile. ``BR*BC`` is a power
            # of two, so a divisor of it is one too and ``tl.reshape(am, (nb, FBLK))`` is legal.
            # Powers of two only, which is the common LoRA/adapter case AND the one where NB is
            # largest and the general loop hurts most.
            exact4 = (mom == MOM_4BIT and blk > 0 and (BR * BC) % blk == 0
                      and all(eff_2d(p) == (BR, BC) for p in bl))
            self.buckets.append(dict(
                plist=bl, BR=BR, BC=BC, mom=mom, momentum=momentum, dev=dev, blk=blk,
                exact4=exact4, fblk=blk if exact4 else 0,
                # GC's tl.constexpr for this tile, resolved ONCE here (see bucket_gc_ok):
                # the caller ANDs it with the group flag, so a steady-state step reads a bool.
                gc_ok=bucket_gc_ok(bl),
                p_addr=i64([p.data_ptr() for p in bl]),
                # compact-Kahan residual bytes (``kahan_lo``), or None: a launch with ``CK``
                # set must REFUSE a None (never substitute another array — the kernel would
                # write residue bytes over whatever it pointed at), see Adakaon._c_addr_arg.
                c_addr=(i64([s["kahan_lo"].data_ptr() for s in st])
                        if all("kahan_lo" in s for s in st) else None),
                m_addr=m_addr, mscale_addr=mscale_addr, mscale_n=mscale_n,
                row_addr=i64([s["row"].data_ptr() for s in st]),
                col_addr=i64([s["col"].data_ptr() for s in st]),
                Rs=Rs, Cs=i32([eff_2d(p)[1] for p in bl]),
                lowp=bl[0].dtype == torch.bfloat16,
                g_addr=i64([p.grad.data_ptr() for p in bl]),
                grad_ptrs=tuple(p.grad.data_ptr() for p in bl),
            ))

    def refresh_grads(self):
        """Rebuild a bucket's grad pointer array iff ANY grad tensor was reallocated.

        The staleness sentinel must cover EVERY grad in the bucket. The original
        first-grad-only check silently kept stale pointers whenever the caching
        allocator reused tensor #0's address while moving the others — which is
        exactly what happens when a new latent shape changes the backward's
        allocation pattern. The kernels then read freed/reused memory as gradients
        (garbage/NaN) and the row/col EMAs rot from arithmetically-impossible
        inputs. Root cause of the 2026-06-10 real-training Nekaon NaN (forensics:
        finite tiny grads + 100% NaN row/col, reproducibly adjacent to a
        "[compile] new latent shape" event). Full-tuple compare costs ~µs/step."""
        for b in self.buckets:
            ptrs = tuple(p.grad.data_ptr() for p in b["plist"])
            if b["grad_ptrs"] != ptrs:
                b["g_addr"] = torch.tensor(ptrs, dtype=torch.int64, device=b["dev"])
                b["grad_ptrs"] = ptrs


class BigPointerCache(_WitnessedCache):
    """Stable pointer arrays and reusable reduction scratch for one big shape bucket.

    The scratch layout is three optimizations in one allocation (0.7.12):

    * **One zeroed region, one ``zero_()``.** ``colsum``, ``rms`` and ``keep`` are all
      atomic accumulation targets that must start each step at 0. Zeroing them separately
      was three kernel launches per bucket per step — pure fixed cost, ~62-94 µs on a
      40-bucket step. They are now adjacent slices of ``_zeros``, so one launch clears all
      three. (``keep`` is an int32 VIEW of its fp32 slice; all-zero bits are 0 in both, and
      the two never alias in a live range.)
    * **``rfac``/``cfac`` in place over ``rowsum``/``colsum``.** ``_factor_rowcol_batched``
      is launched with ``grid=(N,)`` and program ``t`` reads ``rowsum[t*R + rr]`` and writes
      ``rfac[t*R + rr]`` — the same element, in the same program, with no other reader in
      flight — so the factor can overwrite the sum it was derived from. Saves ``N*(R+C)``
      fp32 of permanently pinned scratch.
    * **``rowmean`` only under GC.** It is dereferenced exclusively inside ``if GC:``
      branches (a ``tl.constexpr``, so the load compiles away when GC is off). Without GC
      the attribute aliases ``rowsum``: a valid, correctly-sized pointer that nothing reads,
      the same trick ``mscale_addr`` uses for float momenta.

      That aliasing is only safe while GC STAYS off, so ``gc`` is recorded on the cache and
      the caller rebuilds when it changes (``Adakaon._chunked_step_batched``). A param group
      is a plain mutable dict and schedulers do reach in and flip flags mid-run; with the
      alias live under ``GC=True``, ``_reduce_rowcol`` writes the per-row MEANS on top of the
      row SUMS it just stored, and the factored EMA is then built from means — measured
      1.1e-3 relative divergence from the native path, silently. The param witness cannot
      catch it: no parameter moved.

      ``gc`` is the EFFECTIVE flag: the constructor ANDs the group's flag with
      :attr:`gc_ok`, this bucket's shape predicate (:func:`bucket_gc_ok` — GC is undefined
      for a fan-in of 1). So a fan-in-1 bucket takes the aliased, cheaper layout for free,
      and the caller's validity check is ``cache.gc != (group_flag and cache.gc_ok)``.

    Together those drop the per-bucket scratch from ``N*(3R + 2C) + 3N`` to
    ``N*(R + C) + 2N`` fp32 (``+ N*R`` with GC) — measured -3.07 MB across 1000 tensors of
    scratch. ``inv_rms`` is gone entirely: the consumer kernels derive it from the raw
    ``rms`` accumulator, which removes the ``grid=1`` ``_finish_rms`` launch as well.
    """

    def __init__(self, plist, state_of, R, C, gc=True, gen: int = 0):  # noqa: N803
        self._witness(plist, gen)
        check_state_geometry(plist, state_of, factored=True)
        self.plist = plist
        self.N, self.R, self.C = len(plist), R, C  # noqa: N806
        dev = plist[0].device
        states = [state_of(p) for p in plist]
        self.p_addr = ptr_array(plist, dev)
        # compact-Kahan residual bytes, or None (a CK launch refuses None — see Adakaon._c_addr_arg)
        self.c_addr = (ptr_array([s["kahan_lo"] for s in states], dev)
                       if all("kahan_lo" in s for s in states) else None)
        self.row_addr = ptr_array([s["row"] for s in states], dev)
        self.col_addr = ptr_array([s["col"] for s in states], dev)
        self.m_addr = ptr_array([s["m"] for s in states], dev) if "m" in states[0] else None
        self.mscale_addr = (
            ptr_array([s["m_scale"] for s in states], dev)
            if "m_scale" in states[0] else self.m_addr
        )
        self.g_addr = ptr_array([p.grad for p in plist], dev)
        self.grad_ptrs = tuple(p.grad.data_ptr() for p in plist)
        n_c = self.N * C
        # The three per-step-zeroed accumulators, contiguous so one zero_() clears them.
        self._zeros = torch.zeros(n_c + 2 * self.N, dtype=torch.float32, device=dev)
        self.colsum = self._zeros[:n_c]
        self.cfac = self.colsum                     # written in place (see the class docstring)
        self.rms = self._zeros[n_c:n_c + self.N]
        self.keep = self._zeros[n_c + self.N:].view(torch.int32)
        self.rowsum = torch.empty(self.N * R, dtype=torch.float32, device=dev)
        self.rfac = self.rowsum                     # written in place
        # Whether GC is DEFINED for this bucket's shape (fan-in >= 2), resolved once here —
        # the shape cannot change under a cache (a shape-rebind moves the witness, see
        # Adakaon._fused_partition), so the caller gets the effective flag as
        # ``group["gradient_centralization"] and cache.gc_ok`` at no per-step cost.
        self.gc_ok = bucket_gc_ok(plist)
        # The EFFECTIVE flag: recorded so the caller can rebuild when the group flips it —
        # see the class docstring; the alias below is correct ONLY while GC stays off.
        self.gc = bool(gc) and self.gc_ok
        self.rowmean = (
            torch.empty(self.N * R, dtype=torch.float32, device=dev) if self.gc else self.rowsum
        )

    def zero_accumulators(self) -> None:
        """Clear ``colsum`` + ``rms`` + ``keep`` for this step in ONE launch."""
        self._zeros.zero_()

    def partials(self, RB):  # noqa: N803
        """``(colpart[N*RB*C], rmspart[N*RB])`` for the deterministic two-pass reductions.

        Allocated on FIRST USE and only under ``deterministic_reductions``, so the default
        path never pays for them. ``RB`` is a pure function of the bucket's ``(R, C)``
        (:func:`reduction_tile`), hence constant for the cache's lifetime; it is still keyed
        so a future change to the tiling cannot silently reuse a wrongly-sized buffer.

        These need NO zeroing: every element is written by exactly one pass-1 program before
        pass 2 reads it. (``colpart`` is fully covered because ``RB`` row-blocks tile ``R``
        exactly and each stores all ``C`` columns.)
        """
        got = getattr(self, "_partials", None)
        if got is None or got[0] != RB:
            dev = self.rowsum.device
            got = (RB,
                   torch.empty(self.N * RB * self.C, dtype=torch.float32, device=dev),
                   torch.empty(self.N * RB, dtype=torch.float32, device=dev))
            self._partials = got
        return got[1], got[2]

    def refresh_grads(self) -> None:
        ptrs = tuple(p.grad.data_ptr() for p in self.plist)
        if ptrs != self.grad_ptrs:
            self.g_addr = torch.tensor(ptrs, dtype=torch.int64, device=self.plist[0].device)
            self.grad_ptrs = ptrs


class AdaPnmCache(_WitnessedCache):
    """Like :class:`PointerArrayCache` but for AdaPNM's TWO momenta (``m_pos`` / ``m_neg``).

    Stores the two physical momentum buffers' pointer arrays (+ their fp32 scales for int8/4bit);
    the optimizer passes them to the kernel in (positive, negative) order, swapping by step parity.
    Bucketed by padded tile, grad pointers refreshed on realloc — same plumbing as the single-momentum
    cache."""

    def __init__(self, plist, state_of, gen: int = 0):
        self._witness(plist, gen)
        check_state_geometry(plist, state_of, factored=True)
        groups: dict[tuple[int, int, torch.dtype, torch.device], list] = {}
        for p in plist:
            br, bc = next_pow2_tile(*eff_2d(p))
            groups.setdefault((br, bc, p.dtype, p.device), []).append(p)
        self.buckets = []
        for (BR, BC, _dtype, dev), bl in groups.items():  # noqa: N806
            i64 = lambda xs, _d=dev: torch.tensor(xs, dtype=torch.int64, device=_d)  # noqa: E731
            i32 = lambda xs, _d=dev: torch.tensor(xs, dtype=torch.int32, device=_d)  # noqa: E731
            st = [state_of(p) for p in bl]
            mdtype = st[0]["m_pos"].dtype
            mom = (MOM_INT8 if mdtype == torch.int8 else MOM_4BIT if mdtype == torch.uint8
                   else MOM_BF16 if mdtype == torch.bfloat16 else MOM_FP32)
            quant = mom in (MOM_INT8, MOM_4BIT)
            pos_addr = i64([s["m_pos"].data_ptr() for s in st])
            neg_addr = i64([s["m_neg"].data_ptr() for s in st])
            posc = i64([s["m_pos_scale"].data_ptr() for s in st]) if quant else pos_addr
            negc = i64([s["m_neg_scale"].data_ptr() for s in st]) if quant else neg_addr
            Rs = i32([p.shape[0] for p in bl])  # noqa: N806
            # Per-tensor scale CAPACITY for the 4-bit requant's bounded scale store, taken as the
            # SMALLER of the two momenta's buffers (they are allocated identically, so this is a
            # belt-and-braces min, not a real asymmetry). The tile kernel writes
            # ``fourbit_kernel_blocks(numel)`` scales (its block size is a hardcoded 128); a
            # shorter buffer means a non-default ``momentum_4bit_block`` was routed here, which
            # used to run off the end of ``m_pos_scale``/``m_neg_scale`` — 32 floats past a
            # 32-entry buffer for a (64,128) weight at block=256. Float momenta never dereference
            # this array (the branch is constexpr-elided), so they reuse ``Rs``.
            if mom == MOM_4BIT:
                have = [min(s["m_pos_scale"].numel(), s["m_neg_scale"].numel()) for s in st]
                short = [(tuple(q.shape), h) for q, h in zip(bl, have, strict=True)
                         if h < fourbit_kernel_blocks(q.numel())]
                if short:
                    raise RuntimeError(
                        "fused one-block 4-bit momentum needs 128-element absmax blocks; these "
                        f"tensors carry a different layout: {short[:4]} - route them to the "
                        "native path (see AdaPNM._fused_partition)"
                    )
                mscale_n = i32(have)
            else:
                mscale_n = Rs
            self.buckets.append(dict(
                plist=bl, BR=BR, BC=BC, mom=mom, dev=dev,
                gc_ok=bucket_gc_ok(bl),        # see PointerArrayCache / bucket_gc_ok
                p_addr=i64([p.data_ptr() for p in bl]),
                pos_addr=pos_addr, neg_addr=neg_addr, posc_addr=posc, negc_addr=negc,
                mscale_n=mscale_n,
                row_addr=i64([s["row"].data_ptr() for s in st]),
                col_addr=i64([s["col"].data_ptr() for s in st]),
                Rs=Rs, Cs=i32([eff_2d(p)[1] for p in bl]),
                lowp=bl[0].dtype == torch.bfloat16,
                g_addr=i64([p.grad.data_ptr() for p in bl]),
                grad_ptrs=tuple(p.grad.data_ptr() for p in bl),
            ))

    refresh_grads = PointerArrayCache.refresh_grads


class OneDimPointerCache(_WitnessedCache):
    """Per-tensor pointer arrays for the non-factored ``ndim <= 1`` path, bucketed by (padded block
    ``BL`` = ``next_pow2(numel)``, momentum kind, 4-bit block, param dtype) — one launch per distinct
    bucket. Holds ``g/p/m/v`` base-address arrays + the true element counts ``Ls`` (for masking). The
    kernel is shape-free (base pointer + count), so a 0-D scalar rides as ``numel() == 1`` — for 1-D
    tensors ``numel() == shape[0]``, so this is the same bucketing as before. Quantized momentum
    additionally caches its scale pointer and 4-bit block size; ``beta1==0`` (no ``m``) reuses
    ``v_addr`` as a harmless valid pointer. Same plumbing as :class:`PointerArrayCache` (grad
    pointers refreshed on realloc)."""

    def __init__(self, plist, state_of, gen: int = 0):
        self._witness(plist, gen)
        check_state_geometry(plist, state_of, factored=False)
        groups: dict[tuple[int, int, int, torch.dtype, torch.device], list] = {}
        for p in plist:
            st = state_of(p)
            momentum = "m" in st
            mdtype = st["m"].dtype if momentum else torch.float32
            mom = (MOM_INT8 if mdtype == torch.int8 else MOM_4BIT if mdtype == torch.uint8
                   else MOM_BF16 if mdtype == torch.bfloat16 else MOM_FP32)
            block = st.get("m_block", 1)
            # 4-bit packing requires a pair of lanes even for a scalar parameter.
            bl = max(2 if mom == MOM_4BIT else 1, triton.next_power_of_2(max(p.numel(), 1)))
            groups.setdefault((bl, mom, block, p.dtype, p.device), []).append(p)
        self.buckets = []
        for (BL, mom, block, _dtype, dev), bl in groups.items():  # noqa: N806
            i64 = lambda xs, _d=dev: torch.tensor(xs, dtype=torch.int64, device=_d)  # noqa: E731
            i32 = lambda xs, _d=dev: torch.tensor(xs, dtype=torch.int32, device=_d)  # noqa: E731
            st = [state_of(p) for p in bl]
            momentum = "m" in st[0]
            quant = mom in (MOM_INT8, MOM_4BIT)
            v_addr = i64([s["v"].data_ptr() for s in st])
            m_addr = i64([s["m"].data_ptr() for s in st]) if momentum else v_addr
            mscale_addr = i64([s["m_scale"].data_ptr() for s in st]) if quant else m_addr
            self.buckets.append(dict(
                plist=bl, BL=BL, mom=mom, momentum=momentum, block=block, dev=dev,
                p_addr=i64([p.data_ptr() for p in bl]),
                c_addr=(i64([s["kahan_lo"].data_ptr() for s in st])
                        if all("kahan_lo" in s for s in st) else None),   # see PointerArrayCache
                m_addr=m_addr, mscale_addr=mscale_addr, v_addr=v_addr,
                Ls=i32([p.numel() for p in bl]),
                lowp=bl[0].dtype == torch.bfloat16,
                g_addr=i64([p.grad.data_ptr() for p in bl]),
                grad_ptrs=tuple(p.grad.data_ptr() for p in bl),
            ))

    refresh_grads = PointerArrayCache.refresh_grads


class OneDimPnmCache(_WitnessedCache):
    """Like :class:`OneDimPointerCache` but for AdaPNM's two 1-D momenta (``m_pos``/``m_neg``). Stores
    both physical buffers' address arrays + ``v``; the optimizer passes them (positive, negative) by
    step parity. fp32/bf16 only (quant 1-D and ams_bound route to native). Bucketed by ``next_pow2(L)``."""

    def __init__(self, plist, state_of, gen: int = 0):
        self._witness(plist, gen)
        check_state_geometry(plist, state_of, factored=False)
        groups: dict[tuple[int, torch.dtype, torch.device], list] = {}
        for p in plist:
            key = (triton.next_power_of_2(p.shape[0]), p.dtype, p.device)
            groups.setdefault(key, []).append(p)
        self.buckets = []
        for (BL, _dtype, dev), bl in groups.items():  # noqa: N806
            i64 = lambda xs, _d=dev: torch.tensor(xs, dtype=torch.int64, device=_d)  # noqa: E731
            i32 = lambda xs, _d=dev: torch.tensor(xs, dtype=torch.int32, device=_d)  # noqa: E731
            st = [state_of(p) for p in bl]
            mom = MOM_BF16 if st[0]["m_pos"].dtype == torch.bfloat16 else MOM_FP32
            self.buckets.append(dict(
                plist=bl, BL=BL, mom=mom, dev=dev,
                p_addr=i64([p.data_ptr() for p in bl]),
                pos_addr=i64([s["m_pos"].data_ptr() for s in st]),
                neg_addr=i64([s["m_neg"].data_ptr() for s in st]),
                v_addr=i64([s["v"].data_ptr() for s in st]),
                Ls=i32([p.shape[0] for p in bl]),
                lowp=bl[0].dtype == torch.bfloat16,
                g_addr=i64([p.grad.data_ptr() for p in bl]),
                grad_ptrs=tuple(p.grad.data_ptr() for p in bl),
            ))

    refresh_grads = PointerArrayCache.refresh_grads


class BigPnmCache(_WitnessedCache):
    """Stable pointer arrays and reusable reduction scratch for one big AdaPNM shape bucket.

    The AdaPNM counterpart of :class:`BigPointerCache`, with the TWO momenta (``m_pos``/``m_neg``)
    instead of one. The batched chunked path used to rebuild every one of these arrays from a
    fresh ``torch.tensor([...])`` on EVERY step — six host-to-device index allocations per bucket
    per step (grad, p, both momenta, plus the reduction scratch), which the one-block and 1-D
    routes have cached since 0.7.9. The two momentum arrays hold the PHYSICAL buffers; the
    optimizer swaps them into (positive, negative) order by step parity, exactly as
    :class:`AdaPnmCache` does. Quantized momenta step on host-side fp32 temps and therefore keep
    building their own per-step arrays — only ``p``/``grad``/the scratch are reused there.
    """

    def __init__(self, plist, state_of, R, C, gen: int = 0):  # noqa: N803
        self._witness(plist, gen)
        check_state_geometry(plist, state_of, factored=True)
        self.plist = plist
        self.N, self.R, self.C = len(plist), R, C  # noqa: N806
        dev = plist[0].device
        self.dev = dev
        states = [state_of(p) for p in plist]
        self.p_addr = ptr_array(plist, dev)
        self.pos_addr = ptr_array([s["m_pos"] for s in states], dev)
        self.neg_addr = ptr_array([s["m_neg"] for s in states], dev)
        self.g_addr = ptr_array([p.grad for p in plist], dev)
        self.grad_ptrs = tuple(p.grad.data_ptr() for p in plist)
        # Whether GC is DEFINED for this bucket's shape — see BigPointerCache.gc_ok. This cache
        # always allocates ``rowmean`` for real (no aliasing trick), so unlike Adakaon's it does
        # not need the effective flag as part of its validity.
        self.gc_ok = bucket_gc_ok(plist)
        self.rowmean = torch.empty(self.N * R, dtype=torch.float32, device=dev)
        self.rowsum = torch.empty(self.N * R, dtype=torch.float32, device=dev)
        self.colsum = torch.empty(self.N * C, dtype=torch.float32, device=dev)
        self.keep = torch.empty(self.N, dtype=torch.int32, device=dev)
        self.rms_acc = torch.empty(self.N, dtype=torch.float32, device=dev)

    def momenta(self, pos_first: bool):
        """The (positive, negative) pointer arrays for this step's parity."""
        return (self.pos_addr, self.neg_addr) if pos_first else (self.neg_addr, self.pos_addr)

    refresh_grads = BigPointerCache.refresh_grads
