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

    NOT observed: ``p.shape``. A rebind that changes the shape while keeping the storage
    (``p.data = p.data.view(...)``) is UNSUPPORTED — see the limit documented on
    :func:`kaon.adakaon._param_witness` — and collecting a ``torch.Size`` per param roughly
    DOUBLES the witness, for an operation that stays broken either way. Cost on a 428-param
    LoRA-shaped bag: the review's run measured 80 µs for main's ``(ids, data_ptr)``, 131 µs
    (8.0% of a 1633 µs step) for the three fields kept here and 254 µs (15.6%) with shapes; a
    re-run here on a differently-composed bag measured 48 / 66 / 112 µs. Same ratio either way.
    """
    return (tuple(map(id, plist)), tuple(map(_DATA_PTR, plist)), tuple(map(_IS_CONTIG, plist)))

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
        noise = (tl.rand(seed, offs) * 65536.0).to(tl.int32)
        rounded = (ibits + noise) & -65536  # 0xFFFF0000 as a two's-complement int32
        # 3.4028...e38 is FLT_MAX: the comparison is False for NaN and for +-inf. Inlined
        # because a Triton kernel cannot read a module global.
        finite = tl.abs(res) <= 3.4028234663852886e+38
        return tl.where(finite, rounded, ibits).to(tl.float32, bitcast=True)

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
        g_addr, p_addr, m_addr, mscale_addr, row_addr, col_addr, Rs_ptr, Cs_ptr, Ns_ptr,
        lr, beta1, beta2, eps1, clip, wd, seed, m_blk,
        LOWP: tl.constexpr, MOM: tl.constexpr, MOMENTUM: tl.constexpr,
        CAUTIOUS: tl.constexpr, WD: tl.constexpr,
        GC: tl.constexpr, SR: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
        EXACT: tl.constexpr = False, FBLK: tl.constexpr = 0,
        WDFULL: tl.constexpr = False,
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
        res = p_old - lr * delta
        if SR:
            res = sr_round(res, seed + t, idx)
        tl.store(pp + idx, res.to(pp.dtype.element_ty), mask=m2)

    # ---- chunked (multi-block) path for tensors too large for one block ----
    # The per-tensor reductions (row/col EMA, rms via matvec, cautious mean) are cheap and stay in
    # torch; these two elementwise kernels do the heavy [R,C] passes (momentum + write) chunked over
    # a flat view, so a big weight matrix costs ~few memory passes instead of native's ~30.

    @triton.jit
    def _chunked_mom(g_ptr, m_ptr, p_ptr, rfac_ptr, cfac_ptr, keep_ptr, C, n, inv_rms, wd, beta1,
                     CAUTIOUS: tl.constexpr, WD: tl.constexpr, BLOCK: tl.constexpr,
                     WDFULL: tl.constexpr = False):
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
                delta = delta + wd * tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            keep = ((delta * g) > 0.0) & mask
            tl.atomic_add(keep_ptr, tl.sum(keep.to(tl.int32)))

    @triton.jit
    def _chunked_apply(g_ptr, m_ptr, p_ptr, n, inv_mean, lr, wd, seed,
                       CAUTIOUS: tl.constexpr, WD: tl.constexpr, SR: tl.constexpr, BLOCK: tl.constexpr,
                       WDFULL: tl.constexpr = False):
        """delta = cautious(m + wd*p, g) ["masked"] or cautious(m, g) + wd*p ["full"];
        p -= lr*delta, with bf16 stochastic rounding if SR."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
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
        g_ptr, m_addr, p_addr, rfac_ptr, cfac_ptr, keep_ptr, inv_rms_ptr,
        wd, beta1, R, C, n, K,
        LOWP: tl.constexpr, MOM: tl.constexpr, CAUTIOUS: tl.constexpr, WD: tl.constexpr,
        BLOCK: tl.constexpr, WDFULL: tl.constexpr = False,
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
                p_old = tl.load(pp + offs, mask=mask, other=0.0).to(tl.float32)
                delta = delta + wd * p_old
            keep = ((delta * g) > 0.0) & mask
            tl.atomic_add(keep_ptr + t, tl.sum(keep.to(tl.int32)))

    @triton.jit
    def _chunked_apply_batched(
        g_ptr, m_addr, p_addr, inv_mean_ptr, lr, wd, seed, n, K,
        LOWP: tl.constexpr, MOM: tl.constexpr, CAUTIOUS: tl.constexpr, WD: tl.constexpr,
        SR: tl.constexpr, BLOCK: tl.constexpr, WDFULL: tl.constexpr = False,
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
        g_addr, rowmean_ptr, m_addr, p_addr, rfac_ptr, cfac_ptr, keep_ptr, rms_ptr,
        clip, wd, beta1, R, C, n, K,
        LOWP: tl.constexpr, MOM: tl.constexpr, GC: tl.constexpr, CAUTIOUS: tl.constexpr,
        WD: tl.constexpr, BLOCK: tl.constexpr, WDFULL: tl.constexpr = False,
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
                delta = delta + wd * tl.load(pp + offs, mask=mask, other=0.0).to(tl.float32)
            keep = ((delta * g) > 0.0) & mask
            tl.atomic_add(keep_ptr + t, tl.sum(keep.to(tl.int32)))

    @triton.jit
    def _chunked_apply_batched_g(
        g_addr, rowmean_ptr, m_addr, p_addr, inv_mean_ptr, lr, wd, seed, R, C, n, K,
        LOWP: tl.constexpr, MOM: tl.constexpr, GC: tl.constexpr, CAUTIOUS: tl.constexpr,
        WD: tl.constexpr, SR: tl.constexpr, BLOCK: tl.constexpr, WDFULL: tl.constexpr = False,
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
        g_addr, p_addr, m_addr, mscale_addr, v_addr, Ls_ptr,
        lr, beta1, beta2, eps1, clip, wd, seed,
        LOWP: tl.constexpr, MOM: tl.constexpr, MOMENTUM: tl.constexpr, CAUTIOUS: tl.constexpr,
        WD: tl.constexpr, SR: tl.constexpr, BL: tl.constexpr, FBLOCK: tl.constexpr,
        WDFULL: tl.constexpr = False,
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
        g_addr, rowmean_ptr, p_addr, rfac_ptr, cfac_ptr, keep_ptr,
        rms_ptr, clip, wd, R, C, n, K,
        LOWP: tl.constexpr, GC: tl.constexpr, WD: tl.constexpr,
        BLOCK: tl.constexpr, WDFULL: tl.constexpr = False,
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
            delta += wd * tl.load(pp + offs, mask=mask, other=0.0).to(tl.float32)
        keep = ((delta * g) > 0.0) & mask
        tl.atomic_add(keep_ptr + t, tl.sum(keep.to(tl.int32)))

    @triton.jit
    def _chunked_nomom_apply_batched_g(
        g_addr, rowmean_ptr, p_addr, rfac_ptr, cfac_ptr, rms_ptr, clip,
        inv_mean_ptr, lr, wd, seed, R, C, n, K,
        LOWP: tl.constexpr, GC: tl.constexpr, CAUTIOUS: tl.constexpr,
        WD: tl.constexpr, SR: tl.constexpr, BLOCK: tl.constexpr,
        WDFULL: tl.constexpr = False,
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
        res = p - lr * delta
        if SR:
            res = sr_round(res, seed + t, offs)
        tl.store(pp + offs, res.to(pp.dtype.element_ty), mask=mask)

    @triton.jit
    def _chunked_4bit_keep_batched_g(
        g_addr, rowmean_ptr, packed_addr, scale_addr, p_addr, rfac_ptr, cfac_ptr,
        keep_ptr, rms_ptr, clip, wd, beta1, R, C, n, K,
        LOWP: tl.constexpr, GC: tl.constexpr, WD: tl.constexpr,
        FBLOCK: tl.constexpr, BLOCK: tl.constexpr, WDFULL: tl.constexpr = False,
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
            delta += wd * tl.load(pp + offs, mask=mask, other=0.0).to(tl.float32)
        keep = ((delta * g) > 0.0) & mask
        tl.atomic_add(keep_ptr + t, tl.sum(keep.to(tl.int32)))

    @triton.jit
    def _chunked_4bit_apply_batched_g(
        g_addr, rowmean_ptr, packed_addr, scale_addr, p_addr, rfac_ptr, cfac_ptr,
        keep_ptr, rms_ptr, clip, lr, wd, beta1, seed, R, C, n, K,
        LOWP: tl.constexpr, GC: tl.constexpr, CAUTIOUS: tl.constexpr,
        WD: tl.constexpr, SR: tl.constexpr, FBLOCK: tl.constexpr, BLOCK: tl.constexpr,
        WDFULL: tl.constexpr = False,
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
        delta = momentum
        if WD and not WDFULL:
            delta += wd * p
        if CAUTIOUS:
            count = tl.load(keep_ptr + t).to(tl.float32)
            keep = (delta * g) > 0.0
            # Mask by MULTIPLICATION so a non-finite delta propagates (see _adakaon_tile_kernel).
            delta = delta * tl.where(keep, n.to(tl.float32) / tl.maximum(count, 1.0), 0.0)
        if WD and WDFULL:
            delta += wd * p
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
        g_addr, rowmean_ptr, code_addr, scale_addr, p_addr, rfac_ptr, cfac_ptr,
        keep_ptr, rms_ptr, clip, wd, beta1, R, C, n, K,
        LOWP: tl.constexpr, GC: tl.constexpr, WD: tl.constexpr, BLOCK: tl.constexpr,
        WDFULL: tl.constexpr = False,
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
            delta += wd * tl.load(pp + offs, mask=mask, other=0.0).to(tl.float32)
        keep = ((delta * g) > 0.0) & mask
        tl.atomic_add(keep_ptr + t, tl.sum(keep.to(tl.int32)))

    @triton.jit
    def _chunked_int8_apply_batched_g(
        g_addr, rowmean_ptr, code_addr, scale_addr, p_addr, rfac_ptr, cfac_ptr,
        keep_ptr, rms_ptr, clip, lr, wd, beta1, seed, R, C, n, K,
        LOWP: tl.constexpr, GC: tl.constexpr, CAUTIOUS: tl.constexpr, WD: tl.constexpr,
        SR: tl.constexpr, CSEG: tl.constexpr, RPC: tl.constexpr, BLOCK: tl.constexpr,
        WDFULL: tl.constexpr = False,
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
        delta = momentum
        if WD and not WDFULL:
            delta += wd * p
        if CAUTIOUS:
            count = tl.load(keep_ptr + t).to(tl.float32)
            keep = (delta * g) > 0.0
            # Mask by MULTIPLICATION so a non-finite delta propagates (see _adakaon_tile_kernel).
            delta = delta * tl.where(keep, n.to(tl.float32) / tl.maximum(count, 1.0), 0.0)
        if WD and WDFULL:
            delta += wd * p
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
        p_addr, m_addr, mscale_addr, alpha, clamp, n, K, row_width, seed,
        MOM: tl.constexpr, FBLOCK: tl.constexpr, LOWP: tl.constexpr,
        SR: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """Fused ``p += alpha*m`` for every Kaon momentum storage format.

        This is the shared MSAM/Nekaon perturbation pass.  One program owns a
        flat chunk of one tensor and dequantizes momentum directly from its
        persistent storage, avoiding a stacked fp32 temporary and one Python
        stochastic-rounding call per parameter.
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
    all four routes (one-block, 1-D, 0-D, big batched). Contiguity closes the one SUPPORTED
    rebind that keeps the pointer (``p.data.t()``); the other one (``view``, which changes the
    shape) is an unsupported operation, documented on ``Adakaon._param_witness``. Same guard the
    native plan uses (``Adakaon._foreach_plan``) and MSAM's ``p_witness``; the grad side is
    already covered per step by ``refresh_grads``.
    """

    def _witness(self, plist) -> None:
        """Record the witness for ``plist`` (call once, at cache build)."""
        self.witness = param_witness(plist)
        self.ids, self.ptrs = self.witness[0], self.witness[1]
        self.src = plist

    def stale(self, plist) -> bool:
        """True when the cached pointer arrays no longer describe ``plist``.

        Called per step by callers that build ``plist`` fresh each time (the big-tensor shape
        buckets): see :func:`param_witness` for what it observes and what each field costs.
        """
        return self.witness != param_witness(plist)

    def built_from(self, plist) -> bool:
        """O(1) alternative to :meth:`stale` for a caller that ALREADY revalidated the witness.

        ``Adakaon._fused_partition`` compares ids and ``data_ptr``s across the whole group
        every step and hands back the very same route lists while nothing changed; any change
        (or the per-step non-contiguous-grad demotion) rebuilds them into FRESH list objects.
        Under that contract, "this is the list I was built from" is exactly as strong as
        recomparing the tuples — and skips a second witness sweep per step (~131 µs on the
        428-param bag). Callers without that contract must use :meth:`stale`.
        """
        return plist is self.src


# ============================================================ Triton bf16 SR weight write
# Seed stream for :func:`_sr_axpy_kernel`, isolated from the global RNG exactly as
# ``kaon._stochastic_rounding`` isolates its generator, and derived from the global INITIAL
# seed so ``torch.manual_seed(s)`` at the top of a run reproduces the sequence — the counter
# restarts whenever that seed changes.
#
# Same limitation as the torch path: re-seeding to the SAME value mid-process is not
# observable through the global RNG. That is why ``_reseed_sr_kernel`` is REGISTERED below as
# a ``kaon._stochastic_rounding`` reseed hook instead of being a second public entry point.
# ``kaon.reseed_stochastic_rounding()`` is the one call users are told about, and it has to
# reset every SR noise stream kaon owns — when this counter was left out of it, a bf16 run on
# a Triton build stopped reproducing under ``torch.manual_seed(s)`` + that call, silently,
# for all ten optimizers (the reseed-to-the-same-value case is exactly the one the torch
# module documents as needing it).
_sr_seed_state: dict[int, list[int]] = {}


def _sr_next_seed(device) -> int:
    """Next per-call seed for :func:`_sr_axpy_kernel` on ``device``."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    base = torch.cuda.default_generators[idx].initial_seed()
    entry = _sr_seed_state.get(idx)
    if entry is None or entry[0] != base:
        entry = [base, 0]
        _sr_seed_state[idx] = entry
    entry[1] += 1
    # Odd Weyl increment (golden-ratio constant): consecutive calls land far apart in the
    # Philox stream, so two buckets stepped back to back do not share lane noise.
    return (base + entry[1] * 0x9E3779B1) & 0x7FFFFFFF


def _reseed_sr_kernel() -> None:
    """Restart the Triton SR seed stream. INTERNAL — reached through
    ``kaon.reseed_stochastic_rounding()``, which is the single public reseed entry point."""
    _sr_seed_state.clear()


# Register with the torch SR module so ONE user-facing call resets both noise streams.
# ``_backend`` imports this module lazily, so a Triton-less build never gets here — and never
# has a kernel counter to reset either. The membership guard keeps a re-import of this module
# object from stacking duplicates; an ``importlib.reload`` would still register the new
# function (the stale one then clears an orphaned dict, which is harmless).
from kaon._stochastic_rounding import _reseed_hooks as _sr_reseed_hooks  # noqa: E402

if _reseed_sr_kernel not in _sr_reseed_hooks:
    _sr_reseed_hooks.append(_reseed_sr_kernel)


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
def sr_add_(target, source, alpha: float = 1.0) -> None:
    """``target += alpha * source`` with bf16 stochastic rounding, in ONE Triton launch.

    The caller must have checked :func:`sr_add_supported`.
    """
    n = target.numel()
    with torch.cuda.device(target.device):   # a launch targets the CURRENT device
        _sr_axpy_kernel[((n + 1023) // 1024,)](
            target, source, alpha, n, _sr_next_seed(target.device), BLOCK=1024,
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

    def __init__(self, plist, state_of, mom_dtype):
        self._witness(plist)
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
                p_addr=i64([p.data_ptr() for p in bl]),
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

    Together those drop the per-bucket scratch from ``N*(3R + 2C) + 3N`` to
    ``N*(R + C) + 2N`` fp32 (``+ N*R`` with GC) — measured -3.07 MB across 1000 tensors of
    scratch. ``inv_rms`` is gone entirely: the consumer kernels derive it from the raw
    ``rms`` accumulator, which removes the ``grid=1`` ``_finish_rms`` launch as well.
    """

    def __init__(self, plist, state_of, R, C, gc=True):  # noqa: N803
        self._witness(plist)
        self.plist = plist
        self.N, self.R, self.C = len(plist), R, C  # noqa: N806
        dev = plist[0].device
        states = [state_of(p) for p in plist]
        self.p_addr = ptr_array(plist, dev)
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
        # Recorded so the caller can rebuild when the group flips it — see the class
        # docstring; the alias below is correct ONLY while GC stays off.
        self.gc = bool(gc)
        self.rowmean = (
            torch.empty(self.N * R, dtype=torch.float32, device=dev) if gc else self.rowsum
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

    def __init__(self, plist, state_of):
        self._witness(plist)
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

    def __init__(self, plist, state_of):
        self._witness(plist)
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

    def __init__(self, plist, state_of):
        self._witness(plist)
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

    def __init__(self, plist, state_of, R, C):  # noqa: N803
        self._witness(plist)
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
        self.rowmean = torch.empty(self.N * R, dtype=torch.float32, device=dev)
        self.rowsum = torch.empty(self.N * R, dtype=torch.float32, device=dev)
        self.colsum = torch.empty(self.N * C, dtype=torch.float32, device=dev)
        self.keep = torch.empty(self.N, dtype=torch.int32, device=dev)
        self.rms_acc = torch.empty(self.N, dtype=torch.float32, device=dev)

    def momenta(self, pos_first: bool):
        """The (positive, negative) pointer arrays for this step's parity."""
        return (self.pos_addr, self.neg_addr) if pos_first else (self.neg_addr, self.pos_addr)

    refresh_grads = BigPointerCache.refresh_grads
