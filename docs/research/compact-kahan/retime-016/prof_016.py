"""Stable-step profile for the retime-016 ABBA campaign: Adakaon fused SR, Adakaon fused
kahan8, Nekaon fused SR, on commit <tag>. Confirms no new per-step syncs or H2D/D2H copies
introduced by feature/nekaon-kahan's fused-kernel changes vs main 6201c1f.

Usage: <python> prof_016.py <tag> [--table]   (--table dumps the full profiler table too)
"""
import sys
import time
import warnings

import torch
from torch.profiler import profile, ProfilerActivity

import kaon
from kaon import Adakaon, Nekaon

tag = sys.argv[1]
dump_table = "--table" in sys.argv[2:]

UNET = [(1024, 1024)] * 8 + [(4096,)] * 16
dev = "cuda"


def make_params(shapes, seed=0):
    torch.manual_seed(seed)
    ps = [torch.nn.Parameter((torch.randn(s, device=dev) * 0.05).to(torch.bfloat16)) for s in shapes]
    for q in ps:
        q.grad = (torch.randn_like(q.float()) * 0.02).to(torch.bfloat16)
    return ps


def make_adakaon(method):
    ps = make_params(UNET)
    return Adakaon(ps, lr=1e-4, weight_decay=0.1, cautious=True, bf16_method=method,
                    fused=True, foreach=True, betas=(0.0, 0.999), momentum_dtype="bfloat16")


def make_nekaon(method):
    ps = make_params(UNET)
    return Nekaon(ps, lr=1e-4, fused=True, foreach=True, bf16_method=method)


CASES = {
    "adakaon_fused_sr": lambda: make_adakaon("stochastic_rounding"),
    "adakaon_fused_kahan8": lambda: make_adakaon("kahan8"),
    "nekaon_fused_sr": lambda: make_nekaon("stochastic_rounding"),
}

print("kaon:", kaon.__file__, flush=True)
print("torch:", torch.__version__, "cuda:", torch.version.cuda, flush=True)

for label, factory in CASES.items():
    o = factory()
    for _ in range(15):
        o.step()
    torch.cuda.synchronize()

    torch.cuda.set_sync_debug_mode("warn")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o.step()
        torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode(0)
    nsync = sum("synchroniz" in str(x.message) for x in w)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        o.step()
        torch.cuda.synchronize()
    ev = prof.key_averages()
    n_cuda_events = sum(e.count for e in ev if e.device_type == torch.autograd.DeviceType.CUDA)
    h2d = sum(e.count for e in ev if "Memcpy HtoD" in e.key)
    d2h = sum(e.count for e in ev if "Memcpy DtoH" in e.key)

    cpu0 = time.perf_counter()
    for _ in range(50):
        o.step()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - cpu0) / 50 * 1e3

    print(f"PROF {tag} {label}: syncs={nsync} cuda_events={n_cuda_events} "
          f"HtoD={h2d} DtoH={d2h} wall_ms={wall:.4f}", flush=True)
    if dump_table:
        print(ev.table(sort_by="cuda_time_total", row_limit=25))
    del o
