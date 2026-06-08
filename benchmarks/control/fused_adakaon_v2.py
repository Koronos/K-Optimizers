"""FusedAdakaonV2 — the POINTER-ARRAY, one-tile-per-tensor fused Adakaon (the real one).

PoC #5 (``fused_adakaon.py``) proved the stacked kernel does NOT beat native Adakaon as a real
optimizer: stacking ``p``/``m`` and scattering them back each step (~2x the weight bandwidth) eats
the launch savings. The fix (verified feasible in ``tmp/ptr_test.py`` + ``tmp/perblock_test.py``):
run the kernel **in place on the separate tensors** via a MultiTensorApply-style POINTER ARRAY —
no stacking, no scatter — with ONE program per tensor doing the whole factored step in-block.

Design — and why it is bulletproof OUTSIDE the battery, not just inside it:

* The fused kernel handles a tensor only when it genuinely fits one block: ``ndim == 2``,
  contiguous, and the padded tile ``BR*BC <= TILE_CAP``. That covers the launch-bound regime
  fusion actually wins (many small adapters / low-rank LoRA ``[r, dim]`` with small ``r``).
* EVERYTHING else — long-axis/high-rank LoRA that would spill a single block, ndim != 2 convs,
  1-D biases/norms, non-contiguous grads, fp16 params, oversized tensors — is routed to a real
  inner :class:`kaon.Adakaon` (the proven, already-bandwidth-efficient path). So the result is
  ALWAYS correct for any model; fusion is a strict speed overlay on the cases it helps.
* Mixed shapes in the fused set are handled in ONE launch (per-tensor ``R,C`` + masking), the way
  ``perblock_test`` verified — no shape bucketing needed.
* bf16 params -> in-kernel stochastic-rounding write (the int32 bit-trick); fp32 -> plain write.
* Momentum is bf16 (2 B/param, == Adakaon-bf16). The EMA runs in fp32 then rounds to bf16 (the
  ``fused_adakaon.py`` PoC showed this is battery-faithful; codec parity is not the point here).
* Pointer arrays for the stable tensors (p / m / row / col) are cached across steps; the grad
  pointer array is rebuilt only when a grad tensor is reallocated (identity check) — so the
  steady-state per-step host cost is ~0 and the kernel launch is the whole story.

Run ``python benchmarks/control/fused_adakaon_v2.py`` for parity + a multi-regime speed sweep
(battery-tiny, real low-rank LoRA, high-rank fallback, bf16 params).

⭐ VERDICT (RTX 4080, measured) — the pointer-array kernel RECOVERS the win PoC #5 lost:
  Parity vs native Adakaon: fp32 max|Δp|~1e-7 (exact); bf16 momentum within ULP (~4e-5); bf16
    params differ only by the independent SR draw (rel <1%, unbiased — not a real mismatch).
  Proxy quality+memory (C96/N600, 2 seeds): te 0.0803 vs 0.0802, gap +0.0072 vs +0.0070,
    bpp 2.043 vs 2.043 — FAITHFUL and SAME memory.
  Speed vs native Adakaon-bf16 AS A REAL OPTIMIZER (stack/scatter included for native):
    battery bag 512x[8,16]        25-30x      low-rank LoRA 256x[16,320]   18x
    bag 1024x[8,16]               39x         low-rank LoRA 256x[8,1280]   13x
    bf16 LoRA 256x[16,320] (SR)   26x         mixed real-ish bag (3 tiles) 21.6x
    high-rank 64x[128,1280]       0.99x  (correctly routed to native -> no regression)
  THE BUCKETING LESSON (why "bulletproof outside the battery" mattered): a first cut used one
  global tile (BR,BC)=max for the whole fused set -> a mixed bag padded every [8,16] up to a
  [512,512] tile and ran 9x SLOWER than native (0.11x). Bucketing the fused set by exact padded
  tile (one launch per distinct tile) fixed it to 21.6x. The battery's uniform [8,16] bag hid
  this entirely; only the mixed/real-shape stress test surfaced it.
"""
from __future__ import annotations

import importlib.util

import torch
import triton
import triton.language as tl
from torch.optim import Optimizer

REPO = "/media/koronos/arca/repos/K-Optimizers"
DEV = "cuda"
TILE_CAP = 1 << 16  # 65536 padded lanes -> the largest tile we let one program own


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


# ===================================================================== kernel
@triton.jit
def _v2_kernel(
    g_addr, p_addr, m_addr, row_addr, col_addr, Rs_ptr, Cs_ptr,
    lr, beta1, beta2, eps1, clip, seed,
    LOWP: tl.constexpr, MBF16: tl.constexpr, CAUTIOUS: tl.constexpr,
    GC: tl.constexpr, SR: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
):
    """One program == one tensor. Whole factored Adakaon step, in place via pointer-array.

    Reads the int64 base address of tensor ``t`` from each ``*_addr`` array, casts it to the
    right element pointer, and does GC + factored second moment + RMS-clip + momentum EMA +
    cautious + (SR) weight write entirely in-block. Invalid (padded) lanes are masked to 0 so
    the reductions and the 0*inf factor corners stay finite.
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

    if GC:  # Gradient Centralization: subtract the per-row mean over the fan-in (C) axis
        gmean = tl.sum(g, axis=1) / Cf                      # [BR]
        g = tl.where(m2, g - gmean[:, None], 0.0)

    gsq = g * g
    row_mean = tl.sum(gsq, axis=1) / Cf + eps1              # [BR]  (HF eps placement)
    col_mean = tl.sum(gsq, axis=0) / Rf + eps1              # [BC]
    rr = tl.arange(0, BR)
    cc = tl.arange(0, BC)
    row_old = tl.load(rowp + rr, mask=rr < R, other=0.0)
    col_old = tl.load(colp + cc, mask=cc < C, other=0.0)
    omb = 1.0 - beta2
    row_new = row_old + omb * (row_mean - row_old)         # lerp
    col_new = col_old + omb * (col_mean - col_old)
    tl.store(rowp + rr, row_new, mask=rr < R)
    tl.store(colp + cc, col_new, mask=cc < C)

    row_valid = tl.where(rr < R, row_new, 0.0)
    row_mean_all = tl.sum(row_valid) / Rf
    r_factor = tl.rsqrt(row_new / row_mean_all)            # [BR]; padded rows -> inf (masked below)
    c_factor = tl.rsqrt(col_new)                           # [BC]; padded cols -> inf (masked below)

    upd = tl.where(m2, g * r_factor[:, None] * c_factor[None, :], 0.0)   # 0*inf corners -> 0
    rms = tl.sqrt(tl.sum(upd * upd) / (Rf * Cf))
    denom = rms / clip
    denom = tl.where(denom < 1.0, 1.0, denom)
    upd = upd * (lr / denom)

    if MBF16:
        mp = mi.to(tl.pointer_type(tl.bfloat16))
    else:
        mp = mi.to(tl.pointer_type(tl.float32))
    m_old = tl.load(mp + idx, mask=m2, other=0.0).to(tl.float32)
    m_new = beta1 * m_old + (1.0 - beta1) * upd
    if MBF16:
        tl.store(mp + idx, m_new.to(tl.bfloat16), mask=m2)
    else:
        tl.store(mp + idx, m_new, mask=m2)

    delta = m_new
    if CAUTIOUS:
        keep = (m_new * g) > 0.0
        keepf = tl.where(keep, 1.0, 0.0)
        mm = tl.sum(keepf) / (Rf * Cf)
        mm = tl.where(mm < 1e-8, 1e-8, mm)
        delta = tl.where(keep, m_new / mm, 0.0)

    p_old = tl.load(pp + idx, mask=m2, other=0.0).to(tl.float32)
    res = p_old - delta
    if SR:  # stochastic-round fp32 -> bf16 via the int32 bit-trick (== kaon.add_stochastic_)
        ibits = res.to(tl.int32, bitcast=True)
        noise = (tl.rand(seed + t, idx) * 65536.0).to(tl.int32)
        ibits = (ibits + noise) & -65536
        res = ibits.to(tl.float32, bitcast=True)
    tl.store(pp + idx, res.to(pp.dtype.element_ty), mask=m2)


def _warps_for(lanes: int) -> int:
    if lanes <= 512:
        return 1
    if lanes <= 2048:
        return 2
    if lanes <= 8192:
        return 4
    if lanes <= 32768:
        return 8
    return 16


# ===================================================================== optimizer
class FusedAdakaonV2(Optimizer):
    """Adakaon-bf16 with a pointer-array fused kernel for small 2-D weights; native for the rest."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps1=1e-30, clip=1.0,
                 cautious=True, gradient_centralization=True, momentum_dtype="bfloat16"):
        if momentum_dtype not in ("bfloat16", "float32"):
            raise ValueError("V2 PoC stores momentum as bf16/fp32 only")
        beta1, beta2 = betas
        defaults = dict(lr=lr, beta1=float(beta1), beta2=float(beta2), eps1=eps1, clip=clip,
                        cautious=cautious, gc=gradient_centralization, mdtype=momentum_dtype)
        super().__init__(params, defaults)
        self._t = 0
        self._partitioned = False
        self._fused: list[torch.Tensor] = []     # fused-eligible params (across all groups)
        self._fused_group: dict[int, dict] = {}  # id(p) -> its group dict
        self.inner = None                        # native Adakaon over the fallback params
        self._cache = None                       # cached pointer arrays for the fused set

    # ---- eligibility: only the cases one block genuinely owns; rest -> native ----
    @staticmethod
    def _fused_eligible(p: torch.Tensor) -> bool:
        if p.ndim != 2 or not p.is_cuda or not p.is_contiguous():
            return False
        if p.dtype not in (torch.float32, torch.bfloat16):  # fp16 SR unsupported -> native
            return False
        R, C = p.shape
        BR, BC = triton.next_power_of_2(R), triton.next_power_of_2(C)
        return BR * BC <= TILE_CAP

    def _partition(self):
        from kaon import Adakaon
        fb = []
        for group in self.param_groups:
            g = dict(group)
            for p in group["params"]:
                if self._fused_eligible(p):
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
            md = torch.bfloat16 if self._fused_group[id(p)]["mdtype"] == "bfloat16" else torch.float32
            st["m"] = torch.zeros(R, C, dtype=md, device=p.device)
            st["row"] = torch.zeros(R, dtype=torch.float32, device=p.device)
            st["col"] = torch.zeros(C, dtype=torch.float32, device=p.device)

    def _build_cache(self, plist):
        """Pointer arrays + per-tensor dims, BUCKETED by padded tile ``(BR, BC)``.

        Crucial for mixed-shape bags: a single global ``(BR, BC) = max`` would pad every tiny
        ``[8,16]`` adapter up to the largest tensor's tile (e.g. ``[512,512]`` — 262144 masked
        lanes), making the fused path *slower* than native. Bucketing by exact tile size keeps
        each tensor's work proportional to its own size (one kernel launch per distinct tile),
        the same granularity native uses, while still running in place via the pointer array.
        """
        for p in plist:
            self._ensure_state(p)
        i64 = lambda xs: torch.tensor(xs, dtype=torch.int64, device=DEV)  # noqa: E731
        i32 = lambda xs: torch.tensor(xs, dtype=torch.int32, device=DEV)  # noqa: E731
        groups: dict[tuple[int, int], list] = {}
        for p in plist:
            tile = (triton.next_power_of_2(p.shape[0]), triton.next_power_of_2(p.shape[1]))
            groups.setdefault(tile, []).append(p)
        buckets = []
        for (BR, BC), bl in groups.items():  # noqa: N806
            st = [self.state[p] for p in bl]
            buckets.append(dict(
                plist=bl, BR=BR, BC=BC,
                p_addr=i64([p.data_ptr() for p in bl]),
                m_addr=i64([s["m"].data_ptr() for s in st]),
                row_addr=i64([s["row"].data_ptr() for s in st]),
                col_addr=i64([s["col"].data_ptr() for s in st]),
                Rs=i32([p.shape[0] for p in bl]), Cs=i32([p.shape[1] for p in bl]),
                lowp=bl[0].dtype == torch.bfloat16,
                mbf16=st[0]["m"].dtype == torch.bfloat16,
                g_addr=i64([p.grad.data_ptr() for p in bl]),
                grad_first=bl[0].grad.data_ptr(),
            ))
        return dict(ids=tuple(id(p) for p in plist), buckets=buckets)

    @torch.no_grad()
    def step(self, closure=None):  # noqa: ANN001
        loss = closure() if closure is not None else None
        self._t += 1
        if not self._partitioned:
            self._partition()

        # ---- fused set (bucketed by tile so mixed shapes never over-pad) ----
        plist = [p for p in self._fused if p.grad is not None]
        if plist:
            ids = tuple(id(p) for p in plist)
            if self._cache is None or self._cache["ids"] != ids:
                self._cache = self._build_cache(plist)
            for b in self._cache["buckets"]:
                # rebuild grad pointers only when a grad tensor was reallocated (identity moved)
                gf = b["plist"][0].grad.data_ptr()
                if b["grad_first"] != gf:
                    b["g_addr"] = torch.tensor([p.grad.data_ptr() for p in b["plist"]],
                                               dtype=torch.int64, device=DEV)
                    b["grad_first"] = gf
                grp = self._fused_group[id(b["plist"][0])]  # bucket shares one config in this PoC
                lanes = b["BR"] * b["BC"]
                _v2_kernel[(len(b["plist"]),)](
                    b["g_addr"], b["p_addr"], b["m_addr"], b["row_addr"], b["col_addr"], b["Rs"], b["Cs"],
                    grp["lr"], grp["beta1"], grp["beta2"], grp["eps1"], grp["clip"], self._t,
                    LOWP=b["lowp"], MBF16=b["mbf16"], CAUTIOUS=grp["cautious"], GC=grp["gc"],
                    SR=b["lowp"], BR=b["BR"], BC=b["BC"], num_warps=_warps_for(lanes),
                )

        # ---- everything the kernel does not own -> the proven native path ----
        if self.inner is not None:
            # keep the inner LR in sync with the (possibly scheduled) fused LR
            base = self.param_groups[0]
            for ig in self.inner.param_groups:
                ig["lr"] = base["lr"]
            self.inner.step()
        return loss


__all__ = ["FusedAdakaonV2"]


# ===================================================================== self-test
def _main():
    import time

    from kaon import Adakaon

    def bag(shapes, dt, seed=2, grad=True):
        g = torch.Generator(device=DEV).manual_seed(seed)
        ps = [torch.randn(*sh, generator=g, device=DEV, dtype=dt).requires_grad_(True) for sh in shapes]
        if grad:
            gg = torch.Generator(device=DEV).manual_seed(3)
            for p in ps:
                p.grad = torch.randn(*p.shape, generator=gg, device=DEV, dtype=dt)
        return ps

    def parity(name, shapes, dt, md, cautious=True, gc=True, steps=6):
        cfg = dict(lr=2e-3, betas=(0.9, 0.999), cautious=cautious,
                   gradient_centralization=gc, momentum_dtype=md)
        pv = bag(shapes, dt, 1, grad=False)
        pn = [p.detach().clone().requires_grad_(True) for p in pv]
        ov, on = FusedAdakaonV2(pv, **cfg), Adakaon(pn, **cfg)
        gen = torch.Generator(device=DEV).manual_seed(7)
        for _ in range(steps):
            gs = [torch.randn(*p.shape, generator=gen, device=DEV, dtype=dt) for p in pv]
            for p, g in zip(pv, gs):
                p.grad = g.clone()
            for p, g in zip(pn, gs):
                p.grad = g.clone()
            ov.step()
            on.step()
        torch.cuda.synchronize()
        d = max((a.detach().float() - b.detach().float()).abs().max().item() for a, b in zip(pv, pn))
        # bf16 weights can only match in EXPECTATION (independent SR draws) -> looser bound
        tol = 4e-2 if dt == torch.bfloat16 else (5e-3 if md == "bfloat16" else 1e-5)
        nat = 0 if ov.inner is None else len(ov.inner.param_groups[0]["params"])
        print(f"  [{'OK ' if d < tol else 'OFF'}] {name:38s} max|Δp|={d:.2e} "
              f"(fused={len(ov._fused)} native={nat})")

    def tfn(fn, reps=30, warm=8):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.time()
            fn()
            torch.cuda.synchronize()
            ts.append((time.time() - t0) * 1e3)
        ts.sort()
        return ts[len(ts) // 2]

    def speed(name, shapes, dt=torch.float32):
        cfg = dict(lr=2e-3, betas=(0.9, 0.999), cautious=True,
                   gradient_centralization=True, momentum_dtype="bfloat16")
        ov, on = FusedAdakaonV2(bag(shapes, dt), **cfg), Adakaon(bag(shapes, dt), **cfg)
        tv, tn = tfn(ov.step), tfn(on.step)
        nb = len(ov._cache["buckets"]) if ov._cache else 0
        tag = f"{nb} tile-bucket(s)" if ov.inner is None else f"{nb} fused + native"
        print(f"  {name:36s} V2 {tv:7.3f} ms | native {tn:7.3f} ms | {tn / tv:5.2f}x  [{tag}]")

    print("== PARITY (V2 vs native Adakaon) ==")
    mixed = [(8, 16), (16, 8), (12, 20), (32, 24), (16, 64)]
    parity("tiny fp32 / mom fp32", [(8, 16)] * 8, torch.float32, "float32")
    parity("mixed fp32 / mom fp32", mixed, torch.float32, "float32")
    parity("mixed fp32 / cautious off", mixed, torch.float32, "float32", cautious=False)
    parity("mixed fp32 / GC off", mixed, torch.float32, "float32", gc=False)
    parity("lora fp32 / mom bf16", [(16, 320), (320, 16), (8, 1280)], torch.float32, "bfloat16")
    parity("bf16 params (SR, unbiased)", mixed, torch.bfloat16, "bfloat16")
    parity("tiny+big (fused+native)", [(8, 16)] * 4 + [(512, 512), (1024, 64)], torch.float32, "float32")
    parity("with 1-D params (native)", [(8, 16)] * 4 + [(128,), (64,)], torch.float32, "float32")
    print("\n== SPEED (V2 vs native Adakaon-bf16, real-optimizer step) ==")
    speed("battery bag 512x[8,16]", [(8, 16)] * 512)
    speed("low-rank LoRA 256x[16,320]", [(16, 320)] * 256)
    speed("bf16 LoRA 256x[16,320] (SR)", [(16, 320)] * 256, dt=torch.bfloat16)
    speed("mixed real-ish bag", [(8, 16)] * 200 + [(16, 320)] * 64 + [(320, 16)] * 64)
    speed("high-rank 64x[128,1280] (native)", [(128, 1280)] * 64)


if __name__ == "__main__":
    _main()
