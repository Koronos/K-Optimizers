import sys, warnings, time
import torch
from torch.profiler import profile, ProfilerActivity
import kaon
from kaon import Adakaon
tag = sys.argv[1]
UNET = [(1024, 1024)] * 8 + [(4096,)] * 16
def make_opt(method, fused, foreach=True):
    torch.manual_seed(0)
    ps = [torch.nn.Parameter((torch.randn(s, device="cuda") * 0.05).to(torch.bfloat16)) for s in UNET]
    for q in ps: q.grad = (torch.randn_like(q.float()) * 0.02).to(torch.bfloat16)
    return Adakaon(ps, lr=1e-4, bf16_method=method, fused=fused, foreach=foreach, betas=(0.0, 0.999), momentum_dtype="bfloat16")
for label, m, fu in (("fused_sr", "stochastic_rounding", True), ("fused_k8", "kahan8", True), ("foreach_sr", "stochastic_rounding", False)):
    try: o = make_opt(m, fu)
    except Exception as e: print(label, "n/a"); continue
    for _ in range(10): o.step()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("warn")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o.step(); torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode(0)
    nsync = sum("synchroniz" in str(x.message) for x in w)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        o.step(); torch.cuda.synchronize()
    ev = prof.key_averages()
    kern = sum(e.count for e in ev if e.device_type == torch.autograd.DeviceType.CUDA)
    h2d = sum(e.count for e in ev if "Memcpy HtoD" in e.key)
    d2h = sum(e.count for e in ev if "Memcpy DtoH" in e.key)
    cpu = time.perf_counter(); 
    for _ in range(50): o.step()
    torch.cuda.synchronize(); wall = (time.perf_counter()-cpu)/50*1e3
    print(f"{tag} {label}: syncs={nsync} cuda_events={kern} HtoD={h2d} DtoH={d2h} wall={wall:.3f}ms")
    if label == "fused_sr" and len(sys.argv) > 2:
        print(ev.table(sort_by="cpu_time_total", row_limit=25))
