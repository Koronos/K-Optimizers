import sys, warnings, torch
from torch.profiler import profile, ProfilerActivity
from kaon import Nekaon, Adakaon
warnings.simplefilter("ignore")
method, shape, md, mode = sys.argv[1:5]
SH = {"unet": [(1024, 1024)] * 8 + [(4096,)] * 16, "lora": [(64, 16)] * 200 + [(16, 64)] * 200 + [(64,)] * 28}
torch.manual_seed(0)
ps = [torch.nn.Parameter((torch.randn(s, device="cuda") * 0.05).to(torch.bfloat16)) for s in SH[shape]]
for q in ps: q.grad = (torch.randn_like(q.float()) * 0.02).to(torch.bfloat16)
kw = dict(betas=(0.0, 0.999), momentum_dtype="bfloat16") if md == "nomom" else dict(momentum_dtype="4bit")
o = Adakaon(ps, lr=1e-4, weight_decay=float(sys.argv[5]) if len(sys.argv)>5 else 0.1, cautious=(sys.argv[6]=="1") if len(sys.argv)>6 else True, bf16_method=method, fused=mode == "fused", foreach=True, **kw)
for _ in range(10): o.step()
torch.cuda.synchronize()
best = {}
for _ in range(15):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        o.step(); torch.cuda.synchronize()
    for e in prof.key_averages():
        if e.self_device_time_total > 0:
            k = e.key[:60]; best[k] = min(best.get(k, 1e18), e.self_device_time_total)
for k, t in sorted(best.items(), key=lambda kv: -kv[1])[:10]:
    print(f"{t:9.1f} us  {k}")
