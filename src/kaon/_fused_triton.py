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
    rows × ``BC = next_pow2(C)`` cols (≈ ``work`` lanes) and ``RB = ceil(R/BR)`` blocks tile the rows."""
    BC = triton.next_power_of_2(C)  # noqa: N806
    BR = max(1, min(R, max(1, work // BC)))  # noqa: N806
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
        """
        ibits = res.to(tl.int32, bitcast=True)
        noise = (tl.rand(seed, offs) * 65536.0).to(tl.int32)
        ibits = (ibits + noise) & -65536  # 0xFFFF0000 as a two's-complement int32
        return ibits.to(tl.float32, bitcast=True)

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
    def dequant_4bit(packed_ptr, scale_ptr, ri, ci, idx, Chalf, mask, BLK):
        """Per-block 4-bit packed momentum -> fp32. REUSABLE by any factored-family fused optimizer.

        Nibble-packed (2 codes/byte) over the row-major-flattened tensor with a per-128-block absmax
        scale; assumes an EVEN column count so a byte's pair stays within one row. Mirrors
        ``kaon._momentum_codec._FourBitCodec`` dequant (unpack nibble - 8, * block scale)."""
        byte = tl.load(packed_ptr + (ri * Chalf + ci // 2), mask=mask, other=0)
        nib = tl.where((ci % 2) == 0, byte & 0xF, (byte >> 4) & 0xF)
        q = nib.to(tl.float32) - 8.0
        sc = tl.load(scale_ptr + (idx // BLK), mask=mask, other=0.0)    # per-block scale
        return q * sc

    @triton.jit
    def requant_4bit(m_new, m2, idx, R, C, Chalf, packed_ptr, scale_ptr, NB, BLK,
                     BR: tl.constexpr, BC: tl.constexpr):
        """fp32 momentum -> per-block 4-bit codes + scale, stored in place. REUSABLE.

        Pass 1: segmented per-128-block absmax / 7 (a runtime loop over the tensor's blocks). Pass 2:
        round half-to-even (libdevice.rint) + clamp [-7, 7] + 8 shift -> nibbles, packed two-per-byte
        via reshape + ``tl.split`` (no cross-lane write hazard). Element-identical to ``_FourBitCodec``."""
        blk = idx // BLK
        for b in range(NB):                                            # segmented per-block absmax
            bmax = tl.max(tl.where((blk == b) & m2, tl.abs(m_new), 0.0))
            bmax = tl.where(bmax < 1e-12, 1e-12, bmax)
            tl.store(scale_ptr + b, bmax / 7.0)                        # symmetric 4-bit -> [-7, 7]
        sc = tl.load(scale_ptr + blk, mask=m2, other=1.0)              # per-lane block scale
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
        g_addr, p_addr, m_addr, mscale_addr, row_addr, col_addr, Rs_ptr, Cs_ptr,
        lr, beta1, beta2, eps1, clip, wd, seed,
        LOWP: tl.constexpr, MOM: tl.constexpr, MOMENTUM: tl.constexpr,
        CAUTIOUS: tl.constexpr, WD: tl.constexpr,
        GC: tl.constexpr, SR: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
    ):
        """One program == one tensor. Whole factored Adakaon step, in place via pointer-array.

        Padded lanes are masked to 0 so the reductions and the 0*inf factor corners stay finite.
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
                BLK = tl.minimum(R * C, 128)                           # flat elems per 4-bit block
                m_old = dequant_4bit(packed_ptr, scale_ptr, ri, ci, idx, Chalf, m2, BLK)
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
                requant_4bit(m_new, m2, idx, R, C, Chalf, packed_ptr, scale_ptr, NB, BLK, BR, BC)
            elif MOM == 1:
                tl.store(mi.to(tl.pointer_type(tl.bfloat16)) + idx, m_new.to(tl.bfloat16), mask=m2)
            else:
                tl.store(mi.to(tl.pointer_type(tl.float32)) + idx, m_new, mask=m2)
        else:
            m_new = upd

        # --- decoupled weight decay (AdamW-style): folded into delta BEFORE cautious, like native ---
        p_old = tl.load(pp + idx, mask=m2, other=0.0).to(tl.float32)
        delta = m_new
        if WD:
            delta = delta + wd * p_old                 # momentum requant above used m_new (sans wd)

        # --- REUSABLE-ish: cautious masking + survivor rescale (operates on delta incl. wd) ---
        if CAUTIOUS:
            keep = (delta * g) > 0.0
            keepf = tl.where(keep, 1.0, 0.0)
            mm = tl.sum(keepf) / (Rf * Cf)
            mm = tl.where(mm < 1e-8, 1e-8, mm)
            delta = tl.where(keep, delta / mm, 0.0)

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
                     CAUTIOUS: tl.constexpr, WD: tl.constexpr, BLOCK: tl.constexpr):
        """Momentum EMA of the normalized (LR-independent) update over a flat chunk; accumulates
        the cautious keep count (on delta incl. wd, matching native — the mask is invariant to
        the positive lr scale). m is fp32 or bf16 (EMA runs in fp32)."""
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
            if WD:
                delta = delta + wd * tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            keep = ((delta * g) > 0.0) & mask
            tl.atomic_add(keep_ptr, tl.sum(keep.to(tl.int32)))

    @triton.jit
    def _chunked_apply(g_ptr, m_ptr, p_ptr, n, inv_mean, lr, wd, seed,
                       CAUTIOUS: tl.constexpr, WD: tl.constexpr, SR: tl.constexpr, BLOCK: tl.constexpr):
        """delta = cautious(m + wd*p, g); p -= lr*delta, with bf16 stochastic rounding if SR."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        delta = m
        if WD:
            delta = delta + wd * p
        if CAUTIOUS:
            g = tl.load(g_ptr + offs, mask=mask, other=0.0)
            keep = (delta * g) > 0.0
            delta = tl.where(keep, delta * inv_mean, 0.0)
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
        BLOCK: tl.constexpr,
    ):
        """Batched pass 1: momentum EMA of the normalized update over a flat chunk of tensor ``t``;
        accumulates the cautious keep-count (on delta incl. WD, matching native) into ``keep_ptr[t]``."""
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
            if WD:
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
        SR: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """Batched pass 2: delta = cautious(m + wd*p, g); p -= lr*delta (bf16 SR if LOWP+SR)."""
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
        if WD:
            delta = delta + wd * p
        if CAUTIOUS:
            g = tl.load(g_ptr + t * n + offs, mask=mask, other=0.0)
            inv_mean = tl.load(inv_mean_ptr + t)
            keep = (delta * g) > 0.0
            delta = tl.where(keep, delta * inv_mean, 0.0)
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
            rmean = tl.sum(g, axis=1) / C.to(tl.float32)        # [BR] per-row mean over real C
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
        row_mean = tl.sum(tl.where(rmask, row_new, 0.0)) / R.to(tl.float32)
        tl.store(rfac_ptr + t * R + rr, tl.rsqrt(row_new / row_mean), mask=rmask)
        tl.store(cfac_ptr + t * C + cc, tl.rsqrt(col_new), mask=cmask)

    @triton.jit
    def _finish_rms(rms_ptr, inv_rms_ptr, n, clip, N, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < N
        rms = tl.sqrt(tl.load(rms_ptr + offs, mask=mask, other=0.0) / n)
        denom = tl.maximum(rms / clip, 1.0)
        tl.store(inv_rms_ptr + offs, 1.0 / denom, mask=mask)

    # mom/apply that read grad via the pointer array (+ GC via rowmean) instead of a stacked g_ptr.
    @triton.jit
    def _chunked_mom_batched_g(
        g_addr, rowmean_ptr, m_addr, p_addr, rfac_ptr, cfac_ptr, keep_ptr, inv_rms_ptr,
        wd, beta1, R, C, n, K,
        LOWP: tl.constexpr, MOM: tl.constexpr, GC: tl.constexpr, CAUTIOUS: tl.constexpr,
        WD: tl.constexpr, BLOCK: tl.constexpr,
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
        inv_rms = tl.load(inv_rms_ptr + t)
        upd = g * rf * cf * inv_rms
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
            if WD:
                pi = tl.load(p_addr + t)
                pp = pi.to(tl.pointer_type(tl.bfloat16)) if LOWP else pi.to(tl.pointer_type(tl.float32))
                delta = delta + wd * tl.load(pp + offs, mask=mask, other=0.0).to(tl.float32)
            keep = ((delta * g) > 0.0) & mask
            tl.atomic_add(keep_ptr + t, tl.sum(keep.to(tl.int32)))

    @triton.jit
    def _chunked_apply_batched_g(
        g_addr, rowmean_ptr, m_addr, p_addr, inv_mean_ptr, lr, wd, seed, R, C, n, K,
        LOWP: tl.constexpr, MOM: tl.constexpr, GC: tl.constexpr, CAUTIOUS: tl.constexpr,
        WD: tl.constexpr, SR: tl.constexpr, BLOCK: tl.constexpr,
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
        if WD:
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
            delta = tl.where(keep, delta * inv_mean, 0.0)
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
                requant_4bit(
                    m_new[None, :], mask[None, :], idx[None, :], 1, L, (L + 1) // 2,
                    mp, sp, (L + FBLOCK - 1) // FBLOCK, FBLOCK, BR=1, BC=BL,
                )
            else:
                tl.store(mp + offs, m_new, mask=mask)
            delta = m_new
        else:
            delta = update

        p_old = tl.load(pp + offs, mask=mask, other=0.0).to(tl.float32)
        if WD:
            delta = delta + wd * p_old
        if CAUTIOUS:
            keep = (delta * g) > 0.0
            keepf = tl.where(keep, 1.0, 0.0)
            mm = tl.sum(keepf) / Lf
            mm = tl.where(mm < 1e-8, 1e-8, mm)
            delta = tl.where(keep, delta / mm, 0.0)
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
        g_addr, p_addr, pos_addr, neg_addr, posc_addr, negc_addr, row_addr, col_addr, Rs_ptr, Cs_ptr,
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
            m_pos = dequant_4bit(posp, pscale, ri, ci, idx, Chalf, m2, BLK)
            m_neg = dequant_4bit(negp, nscale, ri, ci, idx, Chalf, m2, BLK)
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
            requant_4bit(m_pos, m2, idx, R, C, Chalf, posp, pscale, NB, BLK, BR, BC)
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
            delta = tl.where(keep, upd / mm, 0.0)
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
            delta = tl.where(keep, delta * inv_mean, 0.0)
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
            delta = tl.where(keep, delta * inv_mean, 0.0)
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
            delta = tl.where(keep, delta * inv_mean, 0.0)
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
            delta = tl.where(keep, delta / mm, 0.0)
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
        inv_rms_ptr, wd, R, C, n, K,
        LOWP: tl.constexpr, GC: tl.constexpr, WD: tl.constexpr,
        BLOCK: tl.constexpr,
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
        delta = g * rf * cf * tl.load(inv_rms_ptr + t)
        if WD:
            pbase = tl.load(p_addr + t)
            pp = pbase.to(tl.pointer_type(tl.bfloat16)) if LOWP else pbase.to(tl.pointer_type(tl.float32))
            delta += wd * tl.load(pp + offs, mask=mask, other=0.0).to(tl.float32)
        keep = ((delta * g) > 0.0) & mask
        tl.atomic_add(keep_ptr + t, tl.sum(keep.to(tl.int32)))

    @triton.jit
    def _chunked_nomom_apply_batched_g(
        g_addr, rowmean_ptr, p_addr, rfac_ptr, cfac_ptr, inv_rms_ptr,
        inv_mean_ptr, lr, wd, seed, R, C, n, K,
        LOWP: tl.constexpr, GC: tl.constexpr, CAUTIOUS: tl.constexpr,
        WD: tl.constexpr, SR: tl.constexpr, BLOCK: tl.constexpr,
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
        delta = g * rf * cf * tl.load(inv_rms_ptr + t)
        pbase = tl.load(p_addr + t)
        pp = pbase.to(tl.pointer_type(tl.bfloat16)) if LOWP else pbase.to(tl.pointer_type(tl.float32))
        p = tl.load(pp + offs, mask=mask, other=0.0).to(tl.float32)
        if WD:
            delta += wd * p
        if CAUTIOUS:
            keep = (delta * g) > 0.0
            count = tl.load(inv_mean_ptr + t).to(tl.float32)
            inv_mean = n.to(tl.float32) / tl.maximum(count, 1.0)
            delta = tl.where(keep, delta * inv_mean, 0.0)
        res = p - lr * delta
        if SR:
            res = sr_round(res, seed + t, offs)
        tl.store(pp + offs, res.to(pp.dtype.element_ty), mask=mask)

    @triton.jit
    def _chunked_4bit_keep_batched_g(
        g_addr, rowmean_ptr, packed_addr, scale_addr, p_addr, rfac_ptr, cfac_ptr,
        keep_ptr, inv_rms_ptr, wd, beta1, R, C, n, K,
        LOWP: tl.constexpr, GC: tl.constexpr, WD: tl.constexpr,
        FBLOCK: tl.constexpr, BLOCK: tl.constexpr,
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
        upd *= tl.load(inv_rms_ptr + t)
        packed = tl.load(packed_addr + t).to(tl.pointer_type(tl.uint8))
        scales = tl.load(scale_addr + t).to(tl.pointer_type(tl.float32))
        byte = tl.load(packed + offs // 2, mask=mask, other=0)
        nib = tl.where((offs & 1) == 0, byte & 0x0F, (byte >> 4) & 0x0F)
        old = (nib.to(tl.float32) - 8.0) * tl.load(scales + offs // FBLOCK, mask=mask, other=0.0)
        momentum = beta1 * old + (1.0 - beta1) * upd
        delta = momentum
        if WD:
            pbase = tl.load(p_addr + t)
            pp = pbase.to(tl.pointer_type(tl.bfloat16)) if LOWP else pbase.to(tl.pointer_type(tl.float32))
            delta += wd * tl.load(pp + offs, mask=mask, other=0.0).to(tl.float32)
        keep = ((delta * g) > 0.0) & mask
        tl.atomic_add(keep_ptr + t, tl.sum(keep.to(tl.int32)))

    @triton.jit
    def _chunked_4bit_apply_batched_g(
        g_addr, rowmean_ptr, packed_addr, scale_addr, p_addr, rfac_ptr, cfac_ptr,
        keep_ptr, inv_rms_ptr, lr, wd, beta1, seed, R, C, n, K,
        LOWP: tl.constexpr, GC: tl.constexpr, CAUTIOUS: tl.constexpr,
        WD: tl.constexpr, SR: tl.constexpr, FBLOCK: tl.constexpr, BLOCK: tl.constexpr,
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
        upd *= tl.load(inv_rms_ptr + t)
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
        if WD:
            delta += wd * p
        if CAUTIOUS:
            count = tl.load(keep_ptr + t).to(tl.float32)
            keep = (delta * g) > 0.0
            delta = tl.where(keep, delta * (n.to(tl.float32) / tl.maximum(count, 1.0)), 0.0)
        res = p - lr * delta
        if SR:
            res = sr_round(res, seed + t, offs)
        tl.store(pp + offs, res.to(pp.dtype.element_ty), mask=mask)

        # Segmented absmax/requant. Chunk and codec block boundaries are aligned,
        # including the final partial chunk; padded lanes quantize to the zero nibble.
        block_in_chunk = local // FBLOCK
        base_block = (k * BLOCK) // FBLOCK
        for b in range(BLOCK // FBLOCK):
            amax = tl.max(tl.where((block_in_chunk == b) & mask, tl.abs(momentum), 0.0))
            amax = tl.maximum(amax, 1e-12)
            tl.store(scales + base_block + b, amax / 7.0, mask=(base_block + b) * FBLOCK < n)
        new_scale = tl.load(scales + offs // FBLOCK, mask=mask, other=1.0)
        q = libdevice.rint(momentum / new_scale)
        q = tl.minimum(tl.maximum(q, -7.0), 7.0)
        nib = tl.where(mask, (q + 8.0).to(tl.uint8), 0)
        lo, hi = tl.split(tl.reshape(nib, (BLOCK // 2, 2)))
        packed_byte = lo | (hi << 4)
        jj = tl.arange(0, BLOCK // 2)
        byte_offs = (k * BLOCK) // 2 + jj
        tl.store(packed + byte_offs, packed_byte, mask=byte_offs < (n + 1) // 2)

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
class PointerArrayCache:
    """Per-tensor pointer arrays, BUCKETED by padded tile, cached across steps.

    Optimizer-agnostic plumbing. A single global (BR,BC)=max would pad every tiny adapter up to the
    largest tensor's tile (and run *slower* than native); bucketing by exact tile keeps each
    tensor's work proportional to its own size — one kernel launch per distinct tile. Stable tensors
    (p / m / row / col) are addressed once; the grad pointer array is rebuilt only when a grad
    tensor is reallocated (identity check), so the steady-state per-step host cost is ~0.
    """

    def __init__(self, plist, state_of, mom_dtype):
        self.ids = tuple(id(p) for p in plist)
        i64 = lambda xs: torch.tensor(xs, dtype=torch.int64, device=DEV)  # noqa: E731
        i32 = lambda xs: torch.tensor(xs, dtype=torch.int32, device=DEV)  # noqa: E731
        groups: dict[tuple[int, int, torch.dtype], list] = {}
        for p in plist:
            br, bc = next_pow2_tile(*eff_2d(p))
            groups.setdefault((br, bc, p.dtype), []).append(p)
        self.buckets = []
        for (BR, BC, _dtype), bl in groups.items():  # noqa: N806
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
            self.buckets.append(dict(
                plist=bl, BR=BR, BC=BC, mom=mom, momentum=momentum,
                p_addr=i64([p.data_ptr() for p in bl]),
                m_addr=m_addr, mscale_addr=mscale_addr,
                row_addr=i64([s["row"].data_ptr() for s in st]),
                col_addr=i64([s["col"].data_ptr() for s in st]),
                Rs=i32([p.shape[0] for p in bl]), Cs=i32([eff_2d(p)[1] for p in bl]),
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
                b["g_addr"] = torch.tensor(ptrs, dtype=torch.int64, device=DEV)
                b["grad_ptrs"] = ptrs


class BigPointerCache:
    """Stable pointer arrays and reusable reduction scratch for one big shape bucket."""

    def __init__(self, plist, state_of, R, C):  # noqa: N803
        self.ids = tuple(id(p) for p in plist)
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
        self.rowmean = torch.empty(self.N * R, dtype=torch.float32, device=dev)
        self.rowsum = torch.empty(self.N * R, dtype=torch.float32, device=dev)
        self.colsum = torch.empty(self.N * C, dtype=torch.float32, device=dev)
        self.rfac = torch.empty(self.N * R, dtype=torch.float32, device=dev)
        self.cfac = torch.empty(self.N * C, dtype=torch.float32, device=dev)
        self.rms = torch.empty(self.N, dtype=torch.float32, device=dev)
        self.inv_rms = torch.empty(self.N, dtype=torch.float32, device=dev)
        self.keep = torch.empty(self.N, dtype=torch.int32, device=dev)

    def refresh_grads(self) -> None:
        ptrs = tuple(p.grad.data_ptr() for p in self.plist)
        if ptrs != self.grad_ptrs:
            self.g_addr = torch.tensor(ptrs, dtype=torch.int64, device=self.plist[0].device)
            self.grad_ptrs = ptrs


class AdaPnmCache:
    """Like :class:`PointerArrayCache` but for AdaPNM's TWO momenta (``m_pos`` / ``m_neg``).

    Stores the two physical momentum buffers' pointer arrays (+ their fp32 scales for int8/4bit);
    the optimizer passes them to the kernel in (positive, negative) order, swapping by step parity.
    Bucketed by padded tile, grad pointers refreshed on realloc — same plumbing as the single-momentum
    cache."""

    def __init__(self, plist, state_of):
        self.ids = tuple(id(p) for p in plist)
        i64 = lambda xs: torch.tensor(xs, dtype=torch.int64, device=DEV)  # noqa: E731
        i32 = lambda xs: torch.tensor(xs, dtype=torch.int32, device=DEV)  # noqa: E731
        groups: dict[tuple[int, int, torch.dtype], list] = {}
        for p in plist:
            br, bc = next_pow2_tile(*eff_2d(p))
            groups.setdefault((br, bc, p.dtype), []).append(p)
        self.buckets = []
        for (BR, BC, _dtype), bl in groups.items():  # noqa: N806
            st = [state_of(p) for p in bl]
            mdtype = st[0]["m_pos"].dtype
            mom = (MOM_INT8 if mdtype == torch.int8 else MOM_4BIT if mdtype == torch.uint8
                   else MOM_BF16 if mdtype == torch.bfloat16 else MOM_FP32)
            quant = mom in (MOM_INT8, MOM_4BIT)
            pos_addr = i64([s["m_pos"].data_ptr() for s in st])
            neg_addr = i64([s["m_neg"].data_ptr() for s in st])
            posc = i64([s["m_pos_scale"].data_ptr() for s in st]) if quant else pos_addr
            negc = i64([s["m_neg_scale"].data_ptr() for s in st]) if quant else neg_addr
            self.buckets.append(dict(
                plist=bl, BR=BR, BC=BC, mom=mom,
                p_addr=i64([p.data_ptr() for p in bl]),
                pos_addr=pos_addr, neg_addr=neg_addr, posc_addr=posc, negc_addr=negc,
                row_addr=i64([s["row"].data_ptr() for s in st]),
                col_addr=i64([s["col"].data_ptr() for s in st]),
                Rs=i32([p.shape[0] for p in bl]), Cs=i32([eff_2d(p)[1] for p in bl]),
                lowp=bl[0].dtype == torch.bfloat16,
                g_addr=i64([p.grad.data_ptr() for p in bl]),
                grad_ptrs=tuple(p.grad.data_ptr() for p in bl),
            ))

    refresh_grads = PointerArrayCache.refresh_grads


class OneDimPointerCache:
    """Per-tensor pointer arrays for the non-factored ``ndim <= 1`` path, bucketed by (padded block
    ``BL`` = ``next_pow2(numel)``, momentum kind, 4-bit block, param dtype) — one launch per distinct
    bucket. Holds ``g/p/m/v`` base-address arrays + the true element counts ``Ls`` (for masking). The
    kernel is shape-free (base pointer + count), so a 0-D scalar rides as ``numel() == 1`` — for 1-D
    tensors ``numel() == shape[0]``, so this is the same bucketing as before. Quantized momentum
    additionally caches its scale pointer and 4-bit block size; ``beta1==0`` (no ``m``) reuses
    ``v_addr`` as a harmless valid pointer. Same plumbing as :class:`PointerArrayCache` (grad
    pointers refreshed on realloc)."""

    def __init__(self, plist, state_of):
        self.ids = tuple(id(p) for p in plist)
        i64 = lambda xs: torch.tensor(xs, dtype=torch.int64, device=DEV)  # noqa: E731
        i32 = lambda xs: torch.tensor(xs, dtype=torch.int32, device=DEV)  # noqa: E731
        groups: dict[tuple[int, int, int, torch.dtype], list] = {}
        for p in plist:
            st = state_of(p)
            momentum = "m" in st
            mdtype = st["m"].dtype if momentum else torch.float32
            mom = (MOM_INT8 if mdtype == torch.int8 else MOM_4BIT if mdtype == torch.uint8
                   else MOM_BF16 if mdtype == torch.bfloat16 else MOM_FP32)
            block = st.get("m_block", 1)
            # 4-bit packing requires a pair of lanes even for a scalar parameter.
            bl = max(2 if mom == MOM_4BIT else 1, triton.next_power_of_2(max(p.numel(), 1)))
            groups.setdefault((bl, mom, block, p.dtype), []).append(p)
        self.buckets = []
        for (BL, mom, block, _dtype), bl in groups.items():  # noqa: N806
            st = [state_of(p) for p in bl]
            momentum = "m" in st[0]
            quant = mom in (MOM_INT8, MOM_4BIT)
            v_addr = i64([s["v"].data_ptr() for s in st])
            m_addr = i64([s["m"].data_ptr() for s in st]) if momentum else v_addr
            mscale_addr = i64([s["m_scale"].data_ptr() for s in st]) if quant else m_addr
            self.buckets.append(dict(
                plist=bl, BL=BL, mom=mom, momentum=momentum, block=block,
                p_addr=i64([p.data_ptr() for p in bl]),
                m_addr=m_addr, mscale_addr=mscale_addr, v_addr=v_addr,
                Ls=i32([p.numel() for p in bl]),
                lowp=bl[0].dtype == torch.bfloat16,
                g_addr=i64([p.grad.data_ptr() for p in bl]),
                grad_ptrs=tuple(p.grad.data_ptr() for p in bl),
            ))

    refresh_grads = PointerArrayCache.refresh_grads


class OneDimPnmCache:
    """Like :class:`OneDimPointerCache` but for AdaPNM's two 1-D momenta (``m_pos``/``m_neg``). Stores
    both physical buffers' address arrays + ``v``; the optimizer passes them (positive, negative) by
    step parity. fp32/bf16 only (quant 1-D and ams_bound route to native). Bucketed by ``next_pow2(L)``."""

    def __init__(self, plist, state_of):
        self.ids = tuple(id(p) for p in plist)
        i64 = lambda xs: torch.tensor(xs, dtype=torch.int64, device=DEV)  # noqa: E731
        i32 = lambda xs: torch.tensor(xs, dtype=torch.int32, device=DEV)  # noqa: E731
        groups: dict[tuple[int, torch.dtype], list] = {}
        for p in plist:
            groups.setdefault((triton.next_power_of_2(p.shape[0]), p.dtype), []).append(p)
        self.buckets = []
        for (BL, _dtype), bl in groups.items():  # noqa: N806
            st = [state_of(p) for p in bl]
            mom = MOM_BF16 if st[0]["m_pos"].dtype == torch.bfloat16 else MOM_FP32
            self.buckets.append(dict(
                plist=bl, BL=BL, mom=mom,
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
