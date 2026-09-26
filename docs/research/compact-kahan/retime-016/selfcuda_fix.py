"""Self-CUDA time (profiler, sum of kernel self time) per optimizer step, median of 7 steps."""
import sys, statistics, warnings, torch
from torch.profiler import profile, ProfilerActivity
import kaon
from kaon import Adakaon, Nekaon
warnings.simplefilter("ignore")
tag = sys.argv[1]
SH = {"unet": [(1024, 1024)] * 8 + [(4096,)] * 16,
      "lora": [(64, 16)] * 200 + [(16, 64)] * 200 + [(64,)] * 28}
def ps_of(shapes):
    torch.manual_seed(0)
    ps = [torch.nn.Parameter((torch.randn(s, device="cuda") * 0.05).to(torch.bfloat16)) for s in shapes]
    for q in ps: q.grad = (torch.randn_like(q.float()) * 0.02).to(torch.bfloat16)
    return ps
def ada(shape, mode, method, md):
    kw = dict(betas=(0.0, 0.999), momentum_dtype="bfloat16") if md == "nomom" else dict(momentum_dtype="4bit")
    return Adakaon(ps_of(SH[shape]), lr=1e-4, weight_decay=0.1, cautious=True, bf16_method=method,
                   fused=mode == "fused", foreach=True, **kw)
def nek(shape, mode, method, md=None):
    return Nekaon(ps_of(SH[shape]), lr=1e-4, fused=mode == "fused", foreach=True, bf16_method=method)
cases = []
SHAPES = sys.argv[2].split(",") if len(sys.argv) > 2 else ("unet", "lora")
MODES = sys.argv[3].split(",") if len(sys.argv) > 3 else ("foreach", "fused")
for shape in SHAPES:
    for mode in MODES:
        for method in ("stochastic_rounding", "kahan8", "kahan16"):
            cases.append((f"{shape}/ada-nomom/{mode}/{method}", lambda s=shape, m=mode, me=method: ada(s, m, me, "nomom")))
            cases.append((f"{shape}/ada-4bit/{mode}/{method}", lambda s=shape, m=mode, me=method: ada(s, m, me, "4bit")))
            cases.append((f"{shape}/nekaon/{mode}/{method}", lambda s=shape, m=mode, me=method: nek(s, m, me)))
print("kaon:", kaon.__file__, flush=True)
for label, f in cases:
    o = f()
    for _ in range(10): o.step()
    torch.cuda.synchronize()
    ts = []
    for _ in range(15):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            o.step(); torch.cuda.synchronize()
        ts.append(sum(e.self_device_time_total for e in prof.key_averages()) / 1e3)
    print(f"SELF {tag} {label} {min(ts):.3f}", flush=True)
    del o
