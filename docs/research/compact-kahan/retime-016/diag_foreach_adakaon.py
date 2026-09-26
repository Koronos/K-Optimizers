import sys, warnings, time
import torch
from torch.profiler import profile, ProfilerActivity
import kaon
from kaon import Adakaon
tag = sys.argv[1]
UNET = [(1024, 1024)] * 8 + [(4096,)] * 16
def mk(method):
    torch.manual_seed(0)
    ps = [torch.nn.Parameter((torch.randn(s, device="cuda") * 0.05).to(torch.bfloat16)) for s in UNET]
    for q in ps: q.grad = (torch.randn_like(q.float()) * 0.02).to(torch.bfloat16)
    return Adakaon(ps, lr=1e-4, weight_decay=0.1, cautious=True, bf16_method=method,
                   fused=False, foreach=True, betas=(0.0, 0.999), momentum_dtype="bfloat16")
print("kaon:", kaon.__file__)
for label, method in (("foreach_sr","stochastic_rounding"), ("foreach_kahan8","kahan8")):
    o = mk(method)
    for _ in range(15): o.step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        o.step(); torch.cuda.synchronize()
    ev = prof.key_averages()
    n_cuda = sum(e.count for e in ev if e.device_type == torch.autograd.DeviceType.CUDA)
    print(ev.table(sort_by="self_cuda_time_total", row_limit=15))
