"""Bytes per parameter of Adakaon's optimizer state by ``bf16_method`` (bf16 weights).

Two measurements per method: the exact sum of the state tensors' storage after one step, and
``torch.cuda.max_memory_allocated`` across the first step (state allocation + step transients).
"""
import sys

import torch

import kaon
from kaon import Adakaon

dev = "cuda"
print("kaon:", kaon.__file__, flush=True)
shapes = [(1024, 1024)] * 8 + [(4096,)] * 16 + [(64, 3, 3, 3)] * 8
n_params = sum(torch.Size(s).numel() for s in shapes)
rows = []
for method in ("stochastic_rounding", "kahan", "kahan8", "none"):
    for fused in (False, True):
        if method == "kahan" and fused:
            continue          # kahan never reaches the fused kernels (native fallback)
        torch.manual_seed(0)
        params = [torch.nn.Parameter((torch.randn(s, device=dev) * 0.05).to(torch.bfloat16)) for s in shapes]
        for p in params:
            p.grad = (torch.randn_like(p.float()) * 0.02).to(torch.bfloat16)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        opt = Adakaon(params, lr=1e-4, momentum_dtype="bfloat16", bf16_method=method, fused=fused)
        opt.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        state_bytes = sum(t.numel() * t.element_size() for st in opt.state.values() for t in st.values() if torch.is_tensor(t))
        comp_bytes = sum(t.numel() * t.element_size() for st in opt.state.values() for k, t in st.items()
                         if k in ("shift", "kahan_lo"))
        rows.append((method, fused, state_bytes / n_params, comp_bytes / n_params, peak / n_params))
        print(f"{method:20s} fused={fused!s:5s} state={state_bytes / n_params:.3f} B/param  "
              f"(compensation {comp_bytes / n_params:.3f} B/param)  first-step peak={peak / n_params:.3f} B/param", flush=True)
        del opt, params
print(f"params: {n_params} ({n_params * 2 / 2**20:.1f} MiB of bf16 weights)")
if len(sys.argv) > 1:
    with open(sys.argv[1], "w") as fh:
        fh.write("| bf16_method | fused | state B/param | compensation B/param | first-step peak B/param |\n|---|---|---|---|---|\n")
        for m, f, s, c, p in rows:
            fh.write(f"| {m} | {f} | {s:.3f} | {c:.3f} | {p:.3f} |\n")
