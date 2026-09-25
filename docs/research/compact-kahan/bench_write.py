"""Kernel microbenchmark: the bf16 weight write by ``bf16_method`` (ORIENTATIVE, see the doc).

Measures (CUDA events, median of reps) on a 2^22-element bf16 weight:
  * the standalone native writer (``subtract_one_``): SR (Triton), kahan (torch), kahan8 (Triton),
    kahan8 torch reference, none;
  * a full fused Adakaon step over a LoRA-shaped bag and over two big (1024x1200) weights, SR vs kahan8.
Prints the GPU power/clock state so the numbers can be read in context.
"""
import subprocess
import sys

import torch

import kaon
from kaon import Adakaon
from kaon import _backend as bk

dev = "cuda"
print("kaon:", kaon.__file__)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,power.limit,power.draw,clocks.max.sm,clocks.sm,temperature.gpu,utilization.gpu",
                      "--format=csv"], capture_output=True, text=True).stdout.strip())


def timeit(fn, reps=50, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2], ts[len(ts) // 4], ts[3 * len(ts) // 4]


rows = []
n = 1 << 22
torch.manual_seed(0)
p = (torch.randn(n, device=dev) * 0.05).to(torch.bfloat16)
d = torch.randn(n, device=dev)
for label, method, triton in [("none", "none", None), ("stochastic_rounding (Triton)", "stochastic_rounding", None),
                              ("stochastic_rounding (torch)", "stochastic_rounding", False),
                              ("kahan (bf16 shift, torch)", "kahan", None),
                              ("kahan8 (Triton axpy)", "kahan8", None), ("kahan8 (torch reference)", "kahan8", False)]:
    pp = p.clone()
    st = {}
    bk.init_bf16_state(pp, st, method)
    med, q1, q3 = timeit(
        lambda pp=pp, st=st, method=method, triton=triton:
            bk.subtract_one_(pp, d, st, method, alpha=1e-5, triton=triton)
    )
    rows.append((f"writer {label}", med, q1, q3))
    print(f"writer {label:32s} 2^22 elems: {med:.3f} ms  [{q1:.3f}, {q3:.3f}]", flush=True)


def bag(shapes):
    torch.manual_seed(0)
    ps = [torch.nn.Parameter((torch.randn(s, device=dev) * 0.05).to(torch.bfloat16)) for s in shapes]
    for q in ps:
        q.grad = (torch.randn_like(q.float()) * 0.02).to(torch.bfloat16)
    return ps


for name, shapes in [("LoRA bag 200x(256,256)+100x(512,)", [(256, 256)] * 200 + [(512,)] * 100),
                     ("big 2x(1024,1200)", [(1024, 1200)] * 2),
                     ("UNet-ish 8x(1024,1024)+16x(4096,)", [(1024, 1024)] * 8 + [(4096,)] * 16)]:
    for method in ("stochastic_rounding", "kahan8"):
        for fused in (True, False):
            ps = bag(shapes)
            opt = Adakaon(ps, lr=1e-4, bf16_method=method, fused=fused)
            opt.step()
            med, q1, q3 = timeit(opt.step, reps=30, warm=3)
            rows.append((f"step {name} {method} fused={fused}", med, q1, q3))
            print(f"step {name:38s} {method:20s} fused={fused!s:5s}: {med:.3f} ms  [{q1:.3f}, {q3:.3f}]", flush=True)
            del opt, ps
if len(sys.argv) > 1:
    with open(sys.argv[1], "w") as fh:
        fh.write("| what | median ms | q1 | q3 |\n|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r[0]} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.3f} |\n")
