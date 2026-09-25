"""Micro-bench: Adakaon fused regression bisect. Prints one JSON line (prefixed RESULT)."""
import json, sys
import torch
import kaon
from kaon import Adakaon

tag = sys.argv[1]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 40
dev = "cuda"
UNET = [(1024, 1024)] * 8 + [(4096,)] * 16
BIG = [(1024, 1200)] * 2
MOM = {"betas": (0.0, 0.999), "momentum_dtype": "bfloat16"}

def make_opt(shapes, method, fused, foreach=True):
    torch.manual_seed(0)
    ps = [torch.nn.Parameter((torch.randn(s, device=dev) * 0.05).to(torch.bfloat16)) for s in shapes]
    for q in ps:
        q.grad = (torch.randn_like(q.float()) * 0.02).to(torch.bfloat16)
    return Adakaon(ps, lr=1e-4, bf16_method=method, fused=fused, foreach=foreach, **MOM)

def med_iqr(ts):
    ts = sorted(ts); n = len(ts)
    med = ts[n // 2] if n % 2 else 0.5 * (ts[n // 2 - 1] + ts[n // 2])
    return med, ts[n // 4], ts[(3 * n) // 4]

def bench(arms, reps, warmup=15):
    for f in arms.values():
        for _ in range(warmup):
            f()
    torch.cuda.synchronize()
    t = {k: [] for k in arms}
    for _ in range(reps):
        for k, f in arms.items():
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); f(); e.record(); torch.cuda.synchronize()
            t[k].append(s.elapsed_time(e))
    return {k: med_iqr(v) for k, v in t.items()}

has_k8 = True
try:
    make_opt([(4,)], "kahan8", fused=True).step()
except Exception as ex:
    has_k8 = False

out = {"tag": tag, "file": kaon.__file__}
# isolated arms (one optimizer benched alone)
iso = {"unet_fused_sr": (UNET, "stochastic_rounding", True, True),
       "unet_foreach_sr": (UNET, "stochastic_rounding", False, True),
       "big_fused_sr": (BIG, "stochastic_rounding", True, True)}
if has_k8:
    iso["unet_fused_k8"] = (UNET, "kahan8", True, True)
for name, (sh, m, fu, fe) in iso.items():
    o = make_opt(sh, m, fu, fe)
    out["iso_" + name] = bench({name: o.step}, reps)[name]
    del o
# interleaved fused SR / kahan8 (like the definitive script)
if has_k8:
    a = make_opt(UNET, "stochastic_rounding", True); b = make_opt(UNET, "kahan8", True)
    r = bench({"sr": a.step, "k8": b.step}, reps)
    out["il_unet_fused_sr"] = r["sr"]; out["il_unet_fused_k8"] = r["k8"]
print("RESULT " + json.dumps(out), flush=True)
