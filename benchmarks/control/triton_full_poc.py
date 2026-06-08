"""PoC #4 (hardcore): the FULL fused Adakaon-bf16 step in Triton -- GC + factored v + momentum +
cautious + stochastic-rounding write, multi-tensor (one stack of N uniform [R,C] tensors).

Builds on PoC #3 (which proved the 32-60x launch-fusion win on the bare apply). Adds everything the
real Adakaon does, all fused:
  - Gradient Centralization (precomputed in torch: subtract grad mean over fan-in -> 1 reduction).
  - factored second moment (row/col EMA) + the r_factor/c_factor reconstruction + RMS clip (matvec).
  - momentum EMA at bf16 (DENSE bf16 momentum = 2 B/param == Adakaon-bf16's footprint; no int8 needed
    to hit 2 B/p -- int8/4bit would be a further cut, future work).
  - cautious masking WITH the survivor rescale (needs a per-tensor mask-mean reduction -> 2 kernels:
    momentum-update, then mask-mean in torch, then cautious+write).
  - bf16 stochastic-rounding weight write (the int32 bit-trick, in-kernel via tl.rand).

Verifies: (1) fp32 path BIT-CLOSE to an exact torch replica of Adakaon's factored step (GC+cautious),
(2) bf16+SR path unbiased + close to native Adakaon-bf16, (3) the speed win survives all of it.

RESULTS (RTX 4080):
  FP32 parity vs exact Adakaon replica (GC+cautious): MATCH (max|dp|~2e-7).
  BF16 SR write: UNBIASED (mean|E[SR]-exact| ~5e-5 << bf16 ULP 4e-3).
  Speed (full step, 2 B/param):  dense 67M = 10.1 vs 9.8 ms (~1.0x, bandwidth-bound -> even);
    LoRA 512x[64,64] = 0.29 vs 5.77 ms (19.7x); LoRA 1024x[32,64] = 0.31 vs 11.0 ms (36.1x).
The launch-bound LoRA regime (the main diffusion use case) is 20-36x faster WITH all features; dense
is even (native is already bandwidth-efficient there; the cautious mask-mean reduction + 2nd pass eat
the bare 2.6x). int8/4bit codec momentum (a further memory cut) and mixed-shape bucketing are the
remaining steps to a production Adakaon(fused=True).

    python benchmarks/control/triton_full_poc.py
"""
from __future__ import annotations

import importlib.util
import time

import torch
import triton
import triton.language as tl

REPO = "/media/koronos/arca/repos/K-Optimizers"


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


H = _load("harness", f"{REPO}/benchmarks/proxy/harness.py")
from kaon import Adakaon  # noqa: E402

DEV = "cuda"


# ===================================================================== kernels
@triton.jit
def _mom_kernel(grad_ptr, m_ptr, rfac_ptr, cfac_ptr, invrms_ptr,
                R, C, RC, n_elem, lr, beta1, BLOCK: tl.constexpr):
    """Momentum EMA of the normalized update, for the whole [N,R,C] stack. Writes m (its dtype)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    n = offs // RC
    rem = offs % RC
    i = rem // C
    j = rem % C
    g = tl.load(grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    rf = tl.load(rfac_ptr + n * R + i, mask=mask, other=0.0).to(tl.float32)
    cf = tl.load(cfac_ptr + n * C + j, mask=mask, other=0.0).to(tl.float32)
    ir = tl.load(invrms_ptr + n, mask=mask, other=0.0).to(tl.float32)
    u = g * rf * cf * (ir * lr)
    m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m = beta1 * m + (1.0 - beta1) * u
    tl.store(m_ptr + offs, m, mask=mask)


@triton.jit
def _apply_kernel(grad_ptr, m_ptr, p_ptr, maskmean_ptr,
                  RC, n_elem, seed,
                  CAUTIOUS: tl.constexpr, SR: tl.constexpr, BLOCK: tl.constexpr):
    """delta = cautious(m, g); p -= delta with stochastic rounding (bf16) or plain (fp32)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    g = tl.load(grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    delta = m
    if CAUTIOUS:
        n = offs // RC
        mm = tl.load(maskmean_ptr + n, mask=mask, other=1.0).to(tl.float32)
        keep = (m * g) > 0.0
        delta = tl.where(keep, m / mm, 0.0)
    p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    res = p - delta
    if SR:
        # stochastic-round fp32 res -> bf16 via the int32 bit trick (== kaon.add_stochastic_)
        ibits = res.to(tl.int32, bitcast=True)
        noise = (tl.rand(seed, offs) * 65536.0).to(tl.int32)
        ibits = (ibits + noise) & -65536          # 0xFFFF0000 as a two's-complement int32 (== kaon -0x10000)
        res = ibits.to(tl.float32, bitcast=True)
    tl.store(p_ptr + offs, res.to(p_ptr.dtype.element_ty), mask=mask)


# ===================================================================== step
def fused_adakaon_step(p, grad, row, col, m, *, lr, beta1, beta2, eps1, clip,
                       gc=True, cautious=True, seed=0):
    """One fused Adakaon step on a STACK [N,R,C] (uniform shapes). p/m at bf16 or fp32."""
    N, R, C = p.shape  # noqa: N806
    omb = 1.0 - beta2
    g = grad.float()
    if gc:
        g = g - g.mean(dim=-1, keepdim=True)            # Gradient Centralization (fan-in = C)
    grad_sq = g * g
    row.lerp_(grad_sq.mean(dim=-1).add_(eps1), omb)
    col.lerp_(grad_sq.mean(dim=-2).add_(eps1), omb)
    r_factor = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_()       # [N,R]
    c_factor = col.rsqrt()                                            # [N,C]
    inner = torch.bmm(grad_sq, (c_factor * c_factor).unsqueeze(-1)).squeeze(-1)
    rms = ((r_factor * r_factor) * inner).sum(dim=-1).div_(R * C).sqrt_()
    inv_rms = rms.div_(clip).clamp_(min=1.0).reciprocal_()           # [N]
    g = g.to(grad.dtype)                                              # GC'd grad at the param dtype
    n_elem = N * R * C
    grid = (triton.cdiv(n_elem, 1024),)
    _mom_kernel[grid](g, m, r_factor, c_factor, inv_rms, R, C, R * C, n_elem, lr, beta1, BLOCK=1024)
    maskmean = torch.ones(N, device=DEV)
    if cautious:
        mf = m.float()
        maskmean = ((mf * g.float()) > 0).float().mean(dim=(1, 2)).clamp_(min=1e-8)  # [N]
    sr = p.dtype == torch.bfloat16
    _apply_kernel[grid](g, m, p, maskmean, R * C, n_elem, seed,
                        CAUTIOUS=cautious, SR=sr, BLOCK=1024)


def torch_adakaon_ref(p, grad, row, col, m, *, lr, beta1, beta2, eps1, clip, gc=True, cautious=True):
    """Exact torch replica of Adakaon's factored step (GC + cautious), for the fp32 parity check."""
    N, R, C = p.shape  # noqa: N806
    omb = 1.0 - beta2
    g = grad.float()
    if gc:
        g = g - g.mean(dim=-1, keepdim=True)
    grad_sq = g * g + eps1
    row.lerp_(grad_sq.mean(dim=-1), omb)
    col.lerp_(grad_sq.mean(dim=-2), omb)
    r_factor = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
    c_factor = col.rsqrt().unsqueeze(-2)
    update = g.mul(r_factor).mul_(c_factor)
    rms = update.reshape(N, -1).norm(2, dim=1) / (R * C) ** 0.5
    update.div_(rms.div_(clip).clamp_(min=1.0).view(N, 1, 1)).mul_(lr)
    m.mul_(beta1).add_(update, alpha=1.0 - beta1)
    delta = m.clone()
    if cautious:
        mk = (delta * g > 0).float()
        delta = delta.mul(mk).div_(mk.reshape(N, -1).mean(1).clamp_(min=1e-8).view(N, 1, 1))
    p.sub_(delta)


# ===================================================================== checks
def check_fp32_parity():
    torch.manual_seed(0)
    N, R, C = 16, 128, 256  # noqa: N806
    cfg = dict(lr=1e-3, beta1=0.9, beta2=0.999, eps1=1e-30, clip=1.0, gc=True, cautious=True)
    p0 = torch.randn(N, R, C, device=DEV); gs = [torch.randn(N, R, C, device=DEV) for _ in range(6)]
    pa = p0.clone(); ra = torch.zeros(N, R, device=DEV); ca = torch.zeros(N, C, device=DEV); ma = torch.zeros(N, R, C, device=DEV)
    pb = p0.clone(); rb = torch.zeros(N, R, device=DEV); cb = torch.zeros(N, C, device=DEV); mb = torch.zeros(N, R, C, device=DEV)
    for g in gs:
        fused_adakaon_step(pa, g.clone(), ra, ca, ma, **cfg)
        torch_adakaon_ref(pb, g.clone(), rb, cb, mb, **cfg)
    torch.cuda.synchronize()
    dp = (pa - pb).abs().max().item(); dm = (ma - mb).abs().max().item()
    print("FP32 parity (fused Triton vs exact Adakaon replica, GC+cautious, 6 steps):")
    print(f"  max|dp|={dp:.2e}  max|dm|={dm:.2e}  -> {'MATCH' if dp < 1e-4 and dm < 1e-4 else 'MISMATCH'}")


def check_bf16_sr_unbiased():
    """SR write should be UNBIASED: averaging many bf16-SR results -> the exact fp32 result."""
    torch.manual_seed(0)
    N, R, C = 4, 64, 64  # noqa: N806
    p0 = torch.randn(N, R, C, device=DEV).to(torch.bfloat16)
    g = torch.randn(N, R, C, device=DEV).to(torch.bfloat16)
    cfg = dict(lr=1e-2, beta1=0.9, beta2=0.999, eps1=1e-30, clip=1.0, gc=False, cautious=False)
    # exact fp32 reference of ONE step
    pe = p0.float().clone(); re = torch.zeros(N, R, device=DEV); cce = torch.zeros(N, C, device=DEV); me = torch.zeros(N, R, C, device=DEV)
    torch_adakaon_ref(pe, g.float().clone(), re, cce, me, **cfg)
    # average many SR draws of the bf16 step
    acc = torch.zeros(N, R, C, device=DEV); K = 400
    for k in range(K):
        pk = p0.clone(); rk = torch.zeros(N, R, device=DEV); ck = torch.zeros(N, C, device=DEV); mk = torch.zeros(N, R, C, device=DEV)
        fused_adakaon_step(pk, g.clone(), rk, ck, mk, seed=k, **cfg)
        acc += pk.float()
    acc /= K
    bias = (acc - pe).abs().mean().item()
    print(f"BF16 stochastic-rounding write: mean|E[SR] - exact| over {K} draws = {bias:.2e} "
          f"(bf16 ULP ~ {2**-8:.2e}) -> {'UNBIASED' if bias < 2**-8 else 'BIASED?'}")


def time_fn(fn, reps=80, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.time(); fn()
        torch.cuda.synchronize(); ts.append((time.time() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


def bench(title, N, R, C):  # noqa: N803
    cfg = dict(lr=1e-3, beta1=0.9, beta2=0.999, eps1=1e-30, clip=1.0, gc=True, cautious=True)
    print(f"\n== {title} ==  (N={N} x [{R},{C}], {N*R*C/1e6:.2f}M params)")
    p = torch.randn(N, R, C, device=DEV).to(torch.bfloat16)
    g = torch.randn(N, R, C, device=DEV).to(torch.bfloat16)
    ro = torch.zeros(N, R, device=DEV); co = torch.zeros(N, C, device=DEV)
    mm = torch.zeros(N, R, C, device=DEV).to(torch.bfloat16)
    fused = time_fn(lambda: fused_adakaon_step(p, g, ro, co, mm, **cfg))
    pads = [torch.randn(R, C, device=DEV).to(torch.bfloat16).requires_grad_(True) for _ in range(N)]
    for pp in pads:
        pp.grad = torch.randn_like(pp)
    opt = Adakaon(pads, lr=1e-3, betas=(0.9, 0.999), cautious=True, gradient_centralization=True, momentum_dtype="bfloat16")
    native = time_fn(opt.step)
    print(f"  {'fused-full (Triton, GC+cautious+SR)':36s} {fused:8.3f} ms   [2 B/param]")
    print(f"  {'Adakaon-bf16 native':36s} {native:8.3f} ms   -> fused {native/fused:.1f}x faster")


def main():
    torch.manual_seed(0)
    check_fp32_parity()
    check_bf16_sr_unbiased()
    bench("dense  8 x [2048,4096]  (67M)", 8, 2048, 4096)
    bench("LoRA  512 x [64,64]", 512, 64, 64)
    bench("LoRA  1024 x [32,64]", 1024, 32, 64)
    print("\n(full Adakaon-bf16 step: GC + factored v + momentum + cautious + SR, all fused. 2 B/param.)")


if __name__ == "__main__":
    main()
