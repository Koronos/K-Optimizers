"""PoC #3: a fused TRITON kernel for kaon's factored-Adam step (Adakaon), keeping ~2 B/param.

The AdamW fused kernel can't ride our quantized/factored state (PoC #2). The only route to
fused-speed-AT-kaon-memory is our OWN kernel. This fuses the per-element APPLY pipeline of
Adakaon's factored step into a single Triton launch:

    u     = grad * r_factor[i] * c_factor[j]      # reconstruct the normalized update
    u     = u / rms_clip * lr                      # Adafactor RMS clip + lr
    m     = beta1*m + (1-beta1)*u                   # momentum EMA (dense here; int8 = +1 requant pass)
    p    -= m                                       # weight write

The cheap REDUCTIONS (row/col EMA, the r_factor row-mean, the rms norm) stay in torch -- they have
tiny outputs and are 1-2 launches. The HEAVY O(R*C) elementwise chain (~8 torch launches today)
becomes ONE Triton kernel. This is where the launch overhead lives.

Verifies correctness vs an exact torch replica of Adakaon's _factored_bucket math, then benchmarks
the fused step vs the all-torch step and vs native Adakaon-bf16.

RESULTS (RTX 4080, correctness MATCH in all cases, fp32 reference):
  dense 67M  : fused 3.7 ms vs Adakaon native 9.8 ms  -> 2.6x
  LoRA 512x[64,64]  : fused-batched 0.18 ms vs native 5.85 ms  -> 32x
  LoRA 1024x[32,64] : fused-batched 0.19 ms vs native 11.2 ms  -> 59x
The multi-tensor (one-launch-for-the-whole-stack) kernel is the decisive win in the launch-bound
LoRA regime -- exactly where kaon hurts most.

HONEST CAVEATS (this is a PoC, not production):
  - momentum here is DENSE fp32 (not the int8/4bit codec) -> not yet at kaon's 2 B/param; adding the
    codec requant (an extra pass or in-kernel) costs some of the headroom, but there's a LOT (32-60x).
  - no cautious / GC / bf16 stochastic-rounding write yet -- all fuseable INTO the kernel (sign mask,
    a grad-mean reduction, the SR bit-trick), so the native comparison does slightly more work today.
  - uniform-shape stack only; mixed adapter shapes need shape-bucketing (few buckets) or a
    pointer-array multi-tensor kernel. The path to production is clear and the prize is quantified.

    python benchmarks/control/triton_poc.py
"""
from __future__ import annotations

import importlib.util
import math
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


# ============================================================= the fused kernel
@triton.jit
def _factored_apply_kernel(
    grad_ptr, m_ptr, p_ptr, rfac_ptr, cfac_ptr,
    C, n_elem, lr, beta1, inv_rms,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    i = offs // C          # row index
    j = offs % C           # col index
    g = tl.load(grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    rf = tl.load(rfac_ptr + i, mask=mask, other=0.0).to(tl.float32)
    cf = tl.load(cfac_ptr + j, mask=mask, other=0.0).to(tl.float32)
    u = g * rf * cf * (inv_rms * lr)                  # normalized update, RMS-clipped, lr-scaled
    m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m = beta1 * m + (1.0 - beta1) * u                 # momentum EMA
    p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    p = p - m                                          # weight write (delta == m)
    tl.store(m_ptr + offs, m, mask=mask)
    tl.store(p_ptr + offs, p, mask=mask)


def fused_factored_step(p, grad, row, col, m, *, lr, beta1, beta2, eps1, clip):
    """One factored-Adam step on a 2-D weight: torch reductions + the fused Triton apply."""
    R, C = p.shape  # noqa: N806
    omb = 1.0 - beta2
    # --- reductions (torch; tiny outputs). Keep grad_sq RAW (eps1 added to the means, which is
    #     identical: mean(g^2+eps1) == mean(g^2)+eps1), so it stays usable for the rms matvec. ---
    grad_sq = grad * grad                                      # raw g^2, [R,C]
    row.lerp_(grad_sq.mean(dim=-1).add_(eps1), omb)            # [R]
    col.lerp_(grad_sq.mean(dim=-2).add_(eps1), omb)            # [C]
    r_factor = row.div(row.mean()).rsqrt_()                    # [R]
    c_factor = col.rsqrt()                                     # [C]
    # rms over the would-be update WITHOUT materializing [R,C]: sum_i rfac_i^2 (sum_j g_ij^2 cfac_j^2);
    # the inner sum is a matvec (BLAS reduction -> [R], no R*C temp).
    inner = grad_sq.matmul(c_factor * c_factor)                # [R]
    rms = (((r_factor * r_factor) * inner).sum() / (R * C)).sqrt_()
    inv_rms = 1.0 / max(float(rms) / clip, 1.0)
    # --- fused apply (Triton) ---
    n = R * C
    grid = (triton.cdiv(n, 1024),)
    _factored_apply_kernel[grid](
        grad.contiguous(), m, p, r_factor.contiguous(), c_factor.contiguous(),
        C, n, lr, beta1, inv_rms, BLOCK=1024,
    )


@triton.jit
def _factored_apply_batched_kernel(
    grad_ptr, m_ptr, p_ptr, rfac_ptr, cfac_ptr, invrms_ptr,
    R, C, RC, n_elem, lr, beta1,
    BLOCK: tl.constexpr,
):
    """Multi-tensor version: one launch processes ALL N stacked [R,C] tensors ([N,R,C] flat)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    n = offs // RC          # which tensor
    rem = offs % RC
    i = rem // C            # row in that tensor
    j = rem % C             # col
    g = tl.load(grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    rf = tl.load(rfac_ptr + n * R + i, mask=mask, other=0.0).to(tl.float32)
    cf = tl.load(cfac_ptr + n * C + j, mask=mask, other=0.0).to(tl.float32)
    ir = tl.load(invrms_ptr + n, mask=mask, other=0.0).to(tl.float32)
    u = g * rf * cf * (ir * lr)
    m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m = beta1 * m + (1.0 - beta1) * u
    p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    p = p - m
    tl.store(m_ptr + offs, m, mask=mask)
    tl.store(p_ptr + offs, p, mask=mask)


def fused_factored_step_batched(p, grad, row, col, m, *, lr, beta1, beta2, eps1, clip):
    """One factored-Adam step on a STACK of N uniform [R,C] tensors ([N,R,C]) — ONE Triton launch
    for the whole stack (multi-tensor). Reductions are batched torch ops over the stack."""
    N, R, C = p.shape  # noqa: N806
    omb = 1.0 - beta2
    grad_sq = grad * grad                                              # [N,R,C]
    row.lerp_(grad_sq.mean(dim=-1).add_(eps1), omb)                    # [N,R]
    col.lerp_(grad_sq.mean(dim=-2).add_(eps1), omb)                    # [N,C]
    r_factor = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_()        # [N,R]
    c_factor = col.rsqrt()                                             # [N,C]
    inner = torch.bmm(grad_sq, (c_factor * c_factor).unsqueeze(-1)).squeeze(-1)  # [N,R]
    rms = ((r_factor * r_factor) * inner).sum(dim=-1).div_(R * C).sqrt_()        # [N]
    inv_rms = rms.div_(clip).clamp_(min=1.0).reciprocal_()                       # [N]
    n_elem = N * R * C
    grid = (triton.cdiv(n_elem, 1024),)
    _factored_apply_batched_kernel[grid](
        grad, m, p, r_factor, c_factor, inv_rms, R, C, R * C, n_elem, lr, beta1, BLOCK=1024,
    )


def torch_factored_step(p, grad, row, col, m, *, lr, beta1, beta2, eps1, clip):
    """Exact same math, but the apply chain in plain torch (the reference + the no-Triton baseline)."""
    R, C = p.shape  # noqa: N806
    omb = 1.0 - beta2
    grad_sq = grad * grad
    if eps1 > 0:
        grad_sq = grad_sq + eps1
    row.lerp_(grad_sq.mean(dim=-1), omb)
    col.lerp_(grad_sq.mean(dim=-2), omb)
    r_factor = row.div(row.mean()).rsqrt_().unsqueeze(-1)
    c_factor = col.rsqrt().unsqueeze(-2)
    update = grad.mul(r_factor).mul_(c_factor)
    rms = update.norm(2) / math.sqrt(R * C)
    update.div_(max(float(rms) / clip, 1.0)).mul_(lr)
    m.mul_(beta1).add_(update, alpha=1.0 - beta1)
    p.sub_(m)


# ============================================================= correctness
def check_correctness():
    torch.manual_seed(0)
    R, C = 512, 1024  # noqa: N806
    cfg = dict(lr=1e-3, beta1=0.9, beta2=0.999, eps1=1e-30, clip=1.0)
    p0 = torch.randn(R, C, device=DEV); g_seq = [torch.randn(R, C, device=DEV) for _ in range(6)]

    pa = p0.clone(); ra = torch.zeros(R, device=DEV); ca = torch.zeros(C, device=DEV); ma = torch.zeros(R, C, device=DEV)
    pb = p0.clone(); rb = torch.zeros(R, device=DEV); cb = torch.zeros(C, device=DEV); mb = torch.zeros(R, C, device=DEV)
    for g in g_seq:
        fused_factored_step(pa, g.clone(), ra, ca, ma, **cfg)
        torch_factored_step(pb, g.clone(), rb, cb, mb, **cfg)
    torch.cuda.synchronize()
    dp = (pa - pb).abs().max().item(); dm = (ma - mb).abs().max().item()
    print("correctness (fused Triton vs torch reference, fp32, 6 steps):")
    print(f"  max|dp|={dp:.2e}  max|dm|={dm:.2e}  -> {'MATCH' if dp < 1e-4 and dm < 1e-4 else 'MISMATCH'}")


# ============================================================= speed
def time_fn(fn, reps=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.time(); fn()
        torch.cuda.synchronize(); ts.append((time.time() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


def bench(title, shapes):
    cfg = dict(lr=1e-3, beta1=0.9, beta2=0.999, eps1=1e-30, clip=1.0)
    ntot = sum(r * c for r, c in shapes)
    print(f"\n== {title} ==  ({len(shapes)} tensors, {ntot/1e6:.2f}M params)")

    def setup():
        ps = [torch.randn(r, c, device=DEV) for r, c in shapes]
        gs = [torch.randn(r, c, device=DEV) for r, c in shapes]
        rows = [torch.zeros(r, device=DEV) for r, c in shapes]
        cols = [torch.zeros(c, device=DEV) for r, c in shapes]
        ms = [torch.zeros(r, c, device=DEV) for r, c in shapes]
        return ps, gs, rows, cols, ms

    ps, gs, rows, cols, ms = setup()
    fused = time_fn(lambda: [fused_factored_step(ps[i], gs[i], rows[i], cols[i], ms[i], **cfg) for i in range(len(ps))])
    ps, gs, rows, cols, ms = setup()
    torchstep = time_fn(lambda: [torch_factored_step(ps[i], gs[i], rows[i], cols[i], ms[i], **cfg) for i in range(len(ps))])

    # native Adakaon-bf16 reference (full optimizer, bf16 momentum)
    pads = [torch.randn(r, c, device=DEV).to(torch.bfloat16).requires_grad_(True) for r, c in shapes]
    for p in pads:
        p.grad = torch.randn_like(p)
    opt = Adakaon(pads, lr=1e-3, betas=(0.9, 0.999), cautious=True, momentum_dtype="bfloat16")
    native = time_fn(opt.step)

    print(f"  {'method':28s} {'ms/step':>9}")
    print(f"  {'fused (torch redux + Triton)':28s} {fused:9.3f}   ({torchstep/fused:.1f}x vs all-torch, {native/fused:.1f}x vs Adakaon)")
    print(f"  {'all-torch (same math)':28s} {torchstep:9.3f}")
    print(f"  {'Adakaon-bf16 native':28s} {native:9.3f}")


def bench_batched(title, N, R, C):  # noqa: N803
    """The MULTI-TENSOR win: N uniform [R,C] tensors as one [N,R,C] stack, one Triton launch,
    vs native Adakaon on the N separate tensors (its real per-bucket-stacked path)."""
    cfg = dict(lr=1e-3, beta1=0.9, beta2=0.999, eps1=1e-30, clip=1.0)
    print(f"\n== {title} ==  (N={N} x [{R},{C}], {N*R*C/1e6:.2f}M params, multi-tensor)")
    # correctness: batched stack vs per-tensor torch reference
    torch.manual_seed(1)
    pst = torch.randn(N, R, C, device=DEV); g = torch.randn(N, R, C, device=DEV)
    ro = torch.zeros(N, R, device=DEV); co = torch.zeros(N, C, device=DEV); mst = torch.zeros(N, R, C, device=DEV)
    pref = pst.clone(); mref = torch.zeros(N, R, C, device=DEV)
    rref = [torch.zeros(R, device=DEV) for _ in range(N)]; cref = [torch.zeros(C, device=DEV) for _ in range(N)]
    fused_factored_step_batched(pst, g.clone(), ro, co, mst, **cfg)
    for k in range(N):
        torch_factored_step(pref[k], g[k].clone(), rref[k], cref[k], mref[k], **cfg)
    dp = (pst - pref).abs().max().item()
    print(f"  correctness vs per-tensor ref: max|dp|={dp:.2e} -> {'MATCH' if dp < 1e-4 else 'MISMATCH'}")

    def setup_stack():
        return (torch.randn(N, R, C, device=DEV), torch.randn(N, R, C, device=DEV),
                torch.zeros(N, R, device=DEV), torch.zeros(N, C, device=DEV), torch.zeros(N, R, C, device=DEV))

    p, g, ro, co, mst = setup_stack()
    fused = time_fn(lambda: fused_factored_step_batched(p, g, ro, co, mst, **cfg))
    pads = [torch.randn(R, C, device=DEV).to(torch.bfloat16).requires_grad_(True) for _ in range(N)]
    for pp in pads:
        pp.grad = torch.randn_like(pp)
    opt = Adakaon(pads, lr=1e-3, betas=(0.9, 0.999), cautious=True, momentum_dtype="bfloat16")
    native = time_fn(opt.step)
    print(f"  {'fused-batched (1 Triton launch)':32s} {fused:8.3f} ms")
    print(f"  {'Adakaon-bf16 native (N tensors)':32s} {native:8.3f} ms   -> fused is {native/fused:.1f}x faster")


def main():
    torch.manual_seed(0)
    check_correctness()
    bench("dense  ~67M params (8 x [2048,4096])", [(2048, 4096)] * 8)
    bench("dense  ~4M params (8 x [512,1024])", [(512, 1024)] * 8)
    print("\n--- the multi-tensor (batched) kernel: the LoRA / launch-bound regime ---")
    bench_batched("LoRA  512 x [64,64]", 512, 64, 64)
    bench_batched("LoRA  1024 x [32,64]", 1024, 32, 64)
    print("\n(ms = step() ONLY, median. fused = elementwise apply in 1 Triton launch.)")


if __name__ == "__main__":
    main()
