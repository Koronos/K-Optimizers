"""Triton fused optimizer kernels for kaon (experimental — not in the public API yet).

Graduated from the ``benchmarks/control/fused_adakaon_v2.py`` PoC. One Triton program owns one
tensor and runs the whole factored step in-block, reading each tensor's base address from a
MultiTensorApply-style POINTER ARRAY and writing ``p``/``m`` IN PLACE (no stacking, no scatter).
Tensors the kernel can't own fall back to a real :class:`kaon.Adakaon`, so the result is always
correct; fusion is a strict speed overlay on the launch-bound (many-small-tensor / low-rank LoRA)
regime, where it is 18-39x faster than the native foreach step at identical memory and fidelity.

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
    * GC + factored row/col second moment        — currently INLINE in ``_adakaon_tile_kernel`` but
                                                   marked; reusable by the whole factored-Adam family
                                                   (Adakaon / AdaPNM / KProdigy) once extracted.

  4bit packs 2 codes/byte over row-major-flat elements, so the fused path needs an EVEN column count
  (keeps each byte's pair within one row); odd-C tensors route to the native Adakaon.

  Adakaon-SPECIFIC (the part each optimizer reimplements):
    * ``_adakaon_tile_kernel``                   — the factored step (r/c-factor + RMS-clip + EMA +
                                                   cautious). AdaPNM would add a 2nd (negative)
                                                   momentum; AdaMuon would swap in orthogonalization.

  NOT yet covered (native fallback handles them): int8/4bit momentum, weight_decay != 0, fp16
  params, conv ndim>2, tiles above ``TILE_CAP`` (high-rank LoRA), per-param-group configs.
────────────────────────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import torch
from torch.optim import Optimizer

try:  # Triton is an optional, GPU-only dependency — keep ``import kaon`` working without it.
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised only on triton-less installs
    triton = None
    tl = None
    _HAS_TRITON = False

__all__ = ["FusedAdakaon", "fused_eligible", "warps_for", "next_pow2_tile", "TILE_CAP", "HAS_TRITON"]

HAS_TRITON = _HAS_TRITON
TILE_CAP = 1 << 16  # 65536 padded lanes — the largest tile we let a single program own
DEV = "cuda"

# Momentum storage kinds (passed to the kernel as a constexpr so the unused branches compile away).
MOM_FP32, MOM_BF16, MOM_INT8, MOM_4BIT = 0, 1, 2, 3
# int8 (/127) and 4bit (/7) scale divisors are inlined as literals in the @jit device helpers
# (Triton kernels can't read module globals). _FOURBIT_BLOCK is the host-side block size.
_FOURBIT_BLOCK = 128   # flat elements per 4-bit absmax block (matches _momentum_codec._FOURBIT_BLOCK)


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
    return BR * BC <= tile_cap


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
    def _adakaon_tile_kernel(
        g_addr, p_addr, m_addr, mscale_addr, row_addr, col_addr, Rs_ptr, Cs_ptr,
        lr, beta1, beta2, eps1, clip, seed,
        LOWP: tl.constexpr, MOM: tl.constexpr, CAUTIOUS: tl.constexpr,
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
        m2 = (ri < R) & (ci < C)
        idx = ri * C + ci
        g = tl.load(gp + idx, mask=m2, other=0.0).to(tl.float32)

        # --- REUSABLE (factored family): Gradient Centralization, mean over fan-in (C) ---
        if GC:
            gmean = tl.sum(g, axis=1) / Cf
            g = tl.where(m2, g - gmean[:, None], 0.0)

        # --- REUSABLE (factored family): row/col second-moment EMA + r/c factor ---
        gsq = g * g
        row_mean = tl.sum(gsq, axis=1) / Cf + eps1
        col_mean = tl.sum(gsq, axis=0) / Rf + eps1
        rr = tl.arange(0, BR)
        cc = tl.arange(0, BC)
        row_old = tl.load(rowp + rr, mask=rr < R, other=0.0)
        col_old = tl.load(colp + cc, mask=cc < C, other=0.0)
        omb = 1.0 - beta2
        row_new = row_old + omb * (row_mean - row_old)
        col_new = col_old + omb * (col_mean - col_old)
        tl.store(rowp + rr, row_new, mask=rr < R)
        tl.store(colp + cc, col_new, mask=cc < C)
        row_valid = tl.where(rr < R, row_new, 0.0)
        row_mean_all = tl.sum(row_valid) / Rf
        r_factor = tl.rsqrt(row_new / row_mean_all)
        c_factor = tl.rsqrt(col_new)

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

        # --- REUSABLE-ish: cautious masking + survivor rescale ---
        delta = m_new
        if CAUTIOUS:
            keep = (m_new * g) > 0.0
            keepf = tl.where(keep, 1.0, 0.0)
            mm = tl.sum(keepf) / (Rf * Cf)
            mm = tl.where(mm < 1e-8, 1e-8, mm)
            delta = tl.where(keep, m_new / mm, 0.0)

        # --- weight write (plain fp32 or bf16 stochastic rounding) ---
        p_old = tl.load(pp + idx, mask=m2, other=0.0).to(tl.float32)
        res = p_old - delta
        if SR:
            res = sr_round(res, seed + t, idx)
        tl.store(pp + idx, res.to(pp.dtype.element_ty), mask=m2)


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


# ============================================================ Adakaon-specific optimizer
class FusedAdakaon(Optimizer):
    """Adakaon-bf16 with a pointer-array fused Triton kernel for small 2-D weights; native otherwise.

    Fuses ``momentum_dtype`` ∈ {bf16, fp32, int8, 4bit} in-kernel via the reusable dequant/requant
    device helpers (per-row int8: 1 B/param; per-128-block 4bit: 0.5 B/param) at ``weight_decay == 0``.
    Odd-column 4bit, weight decay, conv, and high-rank tensors are routed to an inner
    :class:`kaon.Adakaon` (always correct). Experimental: imported from ``kaon._fused_triton``, not
    yet in the public API.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps1=1e-30, clip=1.0,
                 cautious=True, gradient_centralization=True, momentum_dtype="bfloat16",
                 tile_cap=TILE_CAP):
        if not _HAS_TRITON:
            raise RuntimeError("FusedAdakaon requires Triton (a GPU-only optional dependency)")
        if momentum_dtype not in ("bfloat16", "float32", "int8", "4bit"):
            raise ValueError("FusedAdakaon momentum supports bf16/fp32/int8 (4bit -> native fallback)")
        beta1, beta2 = betas
        defaults = dict(lr=lr, beta1=float(beta1), beta2=float(beta2), eps1=eps1, clip=clip,
                        cautious=cautious, gc=gradient_centralization, mdtype=momentum_dtype)
        super().__init__(params, defaults)
        self._tile_cap = tile_cap
        self._t = 0
        self._partitioned = False
        self._fused: list[torch.Tensor] = []
        self._fused_group: dict[int, dict] = {}
        self.inner = None        # native Adakaon over the fallback params
        self._cache: PointerArrayCache | None = None

    def _partition(self):
        from kaon import Adakaon
        fb = []
        for group in self.param_groups:
            is_4bit = group["mdtype"] == "4bit"
            for p in group["params"]:
                ok = fused_eligible(p, self._tile_cap)
                # 4bit packs 2 nibbles/byte over row-major-flat elements; an EVEN column count keeps
                # each byte's pair within one row (clean reshape-pack). Odd C -> native.
                if ok and is_4bit and p.shape[1] % 2 != 0:
                    ok = False
                if ok:
                    self._fused.append(p)
                    self._fused_group[id(p)] = group
                else:
                    fb.append(p)
        if fb:
            grp = self.param_groups[0]
            self.inner = Adakaon(
                fb, lr=grp["lr"], betas=(grp["beta1"], grp["beta2"]), eps=(grp["eps1"], 1e-30),
                clip_threshold=grp["clip"], cautious=grp["cautious"],
                gradient_centralization=grp["gc"], momentum_dtype=grp["mdtype"],
            )
        self._partitioned = True

    def _ensure_state(self, p):
        st = self.state[p]
        if "m" not in st:
            R, C = p.shape
            mdtype = self._fused_group[id(p)]["mdtype"]
            if mdtype == "int8":
                st["m"] = torch.zeros(R, C, dtype=torch.int8, device=p.device)
                # per-row absmax scale; init to ones so a fresh (zero) momentum dequants to 0
                st["m_scale"] = torch.ones(R, dtype=torch.float32, device=p.device)
            elif mdtype == "4bit":
                # packed nibbles [R, C//2] row-major (C even); 0x88 = two zero-level nibbles ->
                # a fresh dequant returns 0. Per-128-block absmax scales, init to ones.
                numel = R * C
                blk = min(_FOURBIT_BLOCK, numel)
                nblocks = (numel + blk - 1) // blk
                st["m"] = torch.full((R * (C // 2),), 0x88, dtype=torch.uint8, device=p.device)
                st["m_scale"] = torch.ones(nblocks, dtype=torch.float32, device=p.device)
            else:
                md = torch.bfloat16 if mdtype == "bfloat16" else torch.float32
                st["m"] = torch.zeros(R, C, dtype=md, device=p.device)
            st["row"] = torch.zeros(R, dtype=torch.float32, device=p.device)
            st["col"] = torch.zeros(C, dtype=torch.float32, device=p.device)

    @torch.no_grad()
    def step(self, closure=None):  # noqa: ANN001
        loss = closure() if closure is not None else None
        self._t += 1
        if not self._partitioned:
            self._partition()

        plist = [p for p in self._fused if p.grad is not None]
        if plist:
            ids = tuple(id(p) for p in plist)
            if self._cache is None or self._cache.ids != ids:
                for p in plist:
                    self._ensure_state(p)
                self._cache = PointerArrayCache(plist, lambda p: self.state[p], None)
            self._cache.refresh_grads()
            for b in self._cache.buckets:
                grp = self._fused_group[id(b["plist"][0])]  # bucket shares one config (PoC limit)
                lanes = b["BR"] * b["BC"]
                _adakaon_tile_kernel[(len(b["plist"]),)](
                    b["g_addr"], b["p_addr"], b["m_addr"], b["mscale_addr"], b["row_addr"], b["col_addr"],
                    b["Rs"], b["Cs"],
                    grp["lr"], grp["beta1"], grp["beta2"], grp["eps1"], grp["clip"], self._t,
                    LOWP=b["lowp"], MOM=b["mom"], CAUTIOUS=grp["cautious"], GC=grp["gc"],
                    SR=b["lowp"], BR=b["BR"], BC=b["BC"], num_warps=warps_for(lanes),
                )

        if self.inner is not None:
            base = self.param_groups[0]
            for ig in self.inner.param_groups:
                ig["lr"] = base["lr"]
            self.inner.step()
        return loss
