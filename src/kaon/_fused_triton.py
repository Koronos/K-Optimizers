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

  NOT covered (Adakaon's native path handles them): fp16 params, conv ndim>2, 1-D, small odd-C 4bit,
  beta1==0, and per-param-group configs. (Large tensors above ``TILE_CAP`` use the chunked path.)
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

__all__ = ["fused_eligible", "warps_for", "next_pow2_tile", "TILE_CAP", "HAS_TRITON"]

HAS_TRITON = _HAS_TRITON
# Largest padded tile a single program owns. Measured crossover (RTX 4080, fp32): the one-block
# kernel beats native up to ~131072 lanes (2.7x @ 65K, 1.2-1.4x @ 131K) and loses past ~262144
# (register spill; >=1M lanes won't even compile). 131072 is the measured sweet spot; this cap is
# also the safety guard that keeps truly-large tensors (full-FT matrices) on the native path. A
# chunked multi-block kernel for those is future work (native goes bandwidth-bound there -> headroom).
TILE_CAP = 1 << 17  # 131072 padded lanes
DEV = "cuda"

# Momentum storage kinds (passed to the kernel as a constexpr so the unused branches compile away).
MOM_FP32, MOM_BF16, MOM_INT8, MOM_4BIT = 0, 1, 2, 3
# int8 (/127) and 4bit (/7) scale divisors are inlined as literals in the @jit device helpers
# (Triton kernels can't read module globals).


# ============================================================ host-side reusable helpers
def next_pow2_tile(R: int, C: int) -> tuple[int, int]:
    """Padded block tile (BR, BC) for a tensor of shape (R, C). Optimizer-agnostic."""
    return triton.next_power_of_2(R), triton.next_power_of_2(C)


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
    """Does ONE Triton block own this tensor? (2-D, contiguous, fp32/bf16, fits a tile.)

    Optimizer-agnostic: any single-block fused kernel shares this predicate. Everything that
    returns False is routed to the native fallback.
    """
    if p.ndim != 2 or not p.is_cuda or not p.is_contiguous():
        return False
    if p.dtype not in (torch.float32, torch.bfloat16):  # fp16 SR unsupported -> native
        return False
    BR, BC = next_pow2_tile(p.shape[0], p.shape[1])
    return tile_cap >= BR * BC


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
        nib = (q + 8.0).to(tl.uint8)                                   # [BR, BC]
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
        LOWP: tl.constexpr, MOM: tl.constexpr, CAUTIOUS: tl.constexpr, WD: tl.constexpr,
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

        # --- Adakaon-specific: reconstructed update, RMS-clip, lr scale ---
        upd = tl.where(m2, g * r_factor[:, None] * c_factor[None, :], 0.0)  # 0*inf corners -> 0
        rms = tl.sqrt(tl.sum(upd * upd) / (Rf * Cf))
        denom = rms / clip
        denom = tl.where(denom < 1.0, 1.0, denom)
        upd = upd * (lr / denom)

        # --- momentum EMA (storage fp32 / bf16 / int8 / 4bit; EMA always runs in fp32) ---
        # dequant the stored momentum to fp32 (quant primitives are codec-level -> reusable)
        if MOM == 2:  # int8 codes + per-row scale
            code_ptr = mi.to(tl.pointer_type(tl.int8))
            scale_ptr = tl.load(mscale_addr + t).to(tl.pointer_type(tl.float32))
            m_old = dequant_int8(code_ptr, idx, m2, scale_ptr, rr, R)
        elif MOM == 3:  # 4bit packed codes + per-block scale (even C only; odd C -> native)
            packed_ptr = mi.to(tl.pointer_type(tl.uint8))
            scale_ptr = tl.load(mscale_addr + t).to(tl.pointer_type(tl.float32))
            Chalf = C // 2
            BLK = tl.minimum(R * C, 128)                               # flat elems per 4-bit block
            m_old = dequant_4bit(packed_ptr, scale_ptr, ri, ci, idx, Chalf, m2, BLK)
        elif MOM == 1:  # bf16
            m_old = tl.load(mi.to(tl.pointer_type(tl.bfloat16)) + idx, mask=m2, other=0.0).to(tl.float32)
        else:  # fp32
            m_old = tl.load(mi.to(tl.pointer_type(tl.float32)) + idx, mask=m2, other=0.0).to(tl.float32)
        m_new = beta1 * m_old + (1.0 - beta1) * upd
        # requant the updated momentum back to storage (m_new stays fp32 for delta/cautious below)
        if MOM == 2:
            requant_int8(m_new, m2, code_ptr, idx, scale_ptr, rr, R)
        elif MOM == 3:
            NB = (R * C + BLK - 1) // BLK
            requant_4bit(m_new, m2, idx, R, C, Chalf, packed_ptr, scale_ptr, NB, BLK, BR, BC)
        elif MOM == 1:
            tl.store(mi.to(tl.pointer_type(tl.bfloat16)) + idx, m_new.to(tl.bfloat16), mask=m2)
        else:
            tl.store(mi.to(tl.pointer_type(tl.float32)) + idx, m_new, mask=m2)

        # --- decoupled weight decay (AdamW-style): folded into delta BEFORE cautious, like native ---
        p_old = tl.load(pp + idx, mask=m2, other=0.0).to(tl.float32)
        delta = m_new
        if WD:
            delta = delta + (lr * wd) * p_old          # momentum requant above used m_new (sans wd)

        # --- REUSABLE-ish: cautious masking + survivor rescale (operates on delta incl. wd) ---
        if CAUTIOUS:
            keep = (delta * g) > 0.0
            keepf = tl.where(keep, 1.0, 0.0)
            mm = tl.sum(keepf) / (Rf * Cf)
            mm = tl.where(mm < 1e-8, 1e-8, mm)
            delta = tl.where(keep, delta / mm, 0.0)

        # --- weight write (plain fp32 or bf16 stochastic rounding) ---
        res = p_old - delta
        if SR:
            res = sr_round(res, seed + t, idx)
        tl.store(pp + idx, res.to(pp.dtype.element_ty), mask=m2)

    # ---- chunked (multi-block) path for tensors too large for one block ----
    # The per-tensor reductions (row/col EMA, rms via matvec, cautious mean) are cheap and stay in
    # torch; these two elementwise kernels do the heavy [R,C] passes (momentum + write) chunked over
    # a flat view, so a big weight matrix costs ~few memory passes instead of native's ~30.

    @triton.jit
    def _chunked_mom(g_ptr, m_ptr, p_ptr, rfac_ptr, cfac_ptr, keep_ptr, C, n, inv_rms_lr, lrwd, beta1,
                     CAUTIOUS: tl.constexpr, WD: tl.constexpr, BLOCK: tl.constexpr):
        """Momentum EMA of the normalized update over a flat chunk; accumulates the cautious keep
        count (on delta incl. wd, matching native). m is fp32 or bf16 (EMA runs in fp32)."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        i = offs // C
        j = offs % C
        g = tl.load(g_ptr + offs, mask=mask, other=0.0)
        rf = tl.load(rfac_ptr + i, mask=mask, other=0.0)
        cf = tl.load(cfac_ptr + j, mask=mask, other=0.0)
        upd = g * rf * cf * inv_rms_lr
        m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        m = beta1 * m + (1.0 - beta1) * upd
        tl.store(m_ptr + offs, m.to(m_ptr.dtype.element_ty), mask=mask)
        if CAUTIOUS:
            delta = m
            if WD:
                delta = delta + lrwd * tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            keep = ((delta * g) > 0.0) & mask
            tl.atomic_add(keep_ptr, tl.sum(keep.to(tl.int32)))

    @triton.jit
    def _chunked_apply(g_ptr, m_ptr, p_ptr, n, inv_mean, lrwd, seed,
                       CAUTIOUS: tl.constexpr, WD: tl.constexpr, SR: tl.constexpr, BLOCK: tl.constexpr):
        """delta = cautious(m + lr*wd*p, g); p -= delta, with bf16 stochastic rounding if SR."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        delta = m
        if WD:
            delta = delta + lrwd * p
        if CAUTIOUS:
            g = tl.load(g_ptr + offs, mask=mask, other=0.0)
            keep = (delta * g) > 0.0
            delta = tl.where(keep, delta * inv_mean, 0.0)
        res = p - delta
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
    # per-tensor scalars (``inv_rms_lr``/``inv_mean``) are float32[N] arrays indexed by ``t``. Momentum
    # is read/written via the m pointer array with the same MOM constexpr as the one-block kernel for
    # fp32/bf16; int8/4bit momentum is dequant'd to an fp32 temp host-side (the m array then points at
    # the temp's per-tensor slices, MOM==FP32) and requant'd in torch between the two passes — exactly
    # the per-tensor ``_chunked_step`` precedent, so odd-C 4bit works (it packs flat, not per-tile).

    @triton.jit
    def _chunked_mom_batched(
        g_ptr, m_addr, p_addr, rfac_ptr, cfac_ptr, keep_ptr, inv_rms_lr_ptr,
        lrwd, beta1, R, C, n, K,
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
        inv_rms_lr = tl.load(inv_rms_lr_ptr + t)
        upd = g * rf * cf * inv_rms_lr
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
                delta = delta + lrwd * p_old
            keep = ((delta * g) > 0.0) & mask
            tl.atomic_add(keep_ptr + t, tl.sum(keep.to(tl.int32)))

    @triton.jit
    def _chunked_apply_batched(
        g_ptr, m_addr, p_addr, inv_mean_ptr, lrwd, seed, n, K,
        LOWP: tl.constexpr, MOM: tl.constexpr, CAUTIOUS: tl.constexpr, WD: tl.constexpr,
        SR: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """Batched pass 2: delta = cautious(m + lr*wd*p, g); p -= delta (bf16 SR if LOWP+SR)."""
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
            delta = delta + lrwd * p
        if CAUTIOUS:
            g = tl.load(g_ptr + t * n + offs, mask=mask, other=0.0)
            inv_mean = tl.load(inv_mean_ptr + t)
            keep = (delta * g) > 0.0
            delta = tl.where(keep, delta * inv_mean, 0.0)
        res = p - delta
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
    # EMA only the positive, requant'd back after pass 1) — so pos/neg are passed as stacked fp32
    # ``[N, n]`` (flat ``t*n + offs``), not pointer arrays; only the weight ``p`` is a per-tensor
    # pointer array written in place. Grad is stacked fp32 (GC folded in). The Adafactor RMS-clip is
    # folded per-tensor into ``sc_ptr[N]`` host-side (pass 2 only — pass 1's keep-count is
    # sign-invariant to the positive ``sc`` scalar, so it uses the unclipped scalar, matching native).
    # WD is decoupled-on-p (``p *= 1-lr*wd``), applied in pass 2 and NOT gated by cautious.

    @triton.jit
    def _adapnm_chunked_mom_batched(
        g_ptr, pos_ptr, neg_ptr, rfac_ptr, cfac_ptr, keep_ptr,
        R, C, n, K, beta1_sq, beta0, inv_noise, sc,
        CAUTIOUS: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """Batched AdaPNM pass 1: EMA the positive momentum (fp32 stacked temp); accumulate the
        cautious keep-count on the pos-neg delta into ``keep_ptr[t]`` (sc is the unclipped scalar)."""
        pid = tl.program_id(0)
        t = pid // K
        k = pid % K
        offs = k * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        base = t * n + offs
        g = tl.load(g_ptr + base, mask=mask, other=0.0)
        m_pos = tl.load(pos_ptr + base, mask=mask, other=0.0)
        m_pos = beta1_sq * m_pos + (1.0 - beta1_sq) * g
        tl.store(pos_ptr + base, m_pos, mask=mask)
        if CAUTIOUS:
            m_neg = tl.load(neg_ptr + base, mask=mask, other=0.0)
            i = offs // C
            j = offs % C
            rf = tl.load(rfac_ptr + t * R + i, mask=mask, other=0.0)
            cf = tl.load(cfac_ptr + t * C + j, mask=mask, other=0.0)
            pn = ((1.0 + beta0) * m_pos - beta0 * m_neg) * inv_noise
            delta = pn * rf * cf * sc
            keep = ((delta * g) > 0.0) & mask
            tl.atomic_add(keep_ptr + t, tl.sum(keep.to(tl.int32)))

    @triton.jit
    def _adapnm_chunked_apply_batched(
        g_ptr, pos_ptr, neg_ptr, p_addr, rfac_ptr, cfac_ptr, sc_ptr, inv_mean_ptr,
        R, C, n, K, beta0, inv_noise, lrwd, seed,
        LOWP: tl.constexpr, CAUTIOUS: tl.constexpr, WD: tl.constexpr, SR: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Batched AdaPNM pass 2: delta = cautious(pn * r*c * sc[t], g); p = p*(1-lr*wd) - delta."""
        pid = tl.program_id(0)
        t = pid // K
        k = pid % K
        offs = k * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        base = t * n + offs
        i = offs // C
        j = offs % C
        m_pos = tl.load(pos_ptr + base, mask=mask, other=0.0)
        m_neg = tl.load(neg_ptr + base, mask=mask, other=0.0)
        rf = tl.load(rfac_ptr + t * R + i, mask=mask, other=0.0)
        cf = tl.load(cfac_ptr + t * C + j, mask=mask, other=0.0)
        sc = tl.load(sc_ptr + t)
        pn = ((1.0 + beta0) * m_pos - beta0 * m_neg) * inv_noise
        delta = pn * rf * cf * sc
        if CAUTIOUS:
            g = tl.load(g_ptr + base, mask=mask, other=0.0)
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
        groups: dict[tuple[int, int], list] = {}
        for p in plist:
            groups.setdefault(next_pow2_tile(p.shape[0], p.shape[1]), []).append(p)
        self.buckets = []
        for (BR, BC), bl in groups.items():  # noqa: N806
            st = [state_of(p) for p in bl]
            mdtype = st[0]["m"].dtype
            if mdtype == torch.int8:
                mom = MOM_INT8
            elif mdtype == torch.uint8:
                mom = MOM_4BIT
            elif mdtype == torch.bfloat16:
                mom = MOM_BF16
            else:
                mom = MOM_FP32
            m_addr = i64([s["m"].data_ptr() for s in st])
            # int8/4bit need a per-tensor pointer array to the fp32 scales; float kinds never
            # dereference mscale (constexpr-elided), so reuse m_addr as a harmless valid pointer.
            quant = mom in (MOM_INT8, MOM_4BIT)
            mscale_addr = i64([s["m_scale"].data_ptr() for s in st]) if quant else m_addr
            self.buckets.append(dict(
                plist=bl, BR=BR, BC=BC, mom=mom,
                p_addr=i64([p.data_ptr() for p in bl]),
                m_addr=m_addr, mscale_addr=mscale_addr,
                row_addr=i64([s["row"].data_ptr() for s in st]),
                col_addr=i64([s["col"].data_ptr() for s in st]),
                Rs=i32([p.shape[0] for p in bl]), Cs=i32([p.shape[1] for p in bl]),
                lowp=bl[0].dtype == torch.bfloat16,
                g_addr=i64([p.grad.data_ptr() for p in bl]),
                grad_first=bl[0].grad.data_ptr(),
            ))

    def refresh_grads(self):
        """Rebuild a bucket's grad pointer array iff its first grad tensor was reallocated."""
        for b in self.buckets:
            gf = b["plist"][0].grad.data_ptr()
            if b["grad_first"] != gf:
                b["g_addr"] = torch.tensor([p.grad.data_ptr() for p in b["plist"]],
                                           dtype=torch.int64, device=DEV)
                b["grad_first"] = gf


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
        groups: dict[tuple[int, int], list] = {}
        for p in plist:
            groups.setdefault(next_pow2_tile(p.shape[0], p.shape[1]), []).append(p)
        self.buckets = []
        for (BR, BC), bl in groups.items():  # noqa: N806
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
                Rs=i32([p.shape[0] for p in bl]), Cs=i32([p.shape[1] for p in bl]),
                lowp=bl[0].dtype == torch.bfloat16,
                g_addr=i64([p.grad.data_ptr() for p in bl]),
                grad_first=bl[0].grad.data_ptr(),
            ))

    refresh_grads = PointerArrayCache.refresh_grads
