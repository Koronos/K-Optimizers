"""End-to-end: does the MSAM / Nekaon climb keep the compact-Kahan advantage?

One 64x64 bf16 weight, lr 1e-5 (steps ~0.04 ulp), 300 steps, the same bf16 gradients fed to
an fp32-parameter reference of the same optimizer and to bf16 runs under ``kahan8`` and
``stochastic_rounding``. Error = max |z - z_ref| over the tensor, in bf16 ulps at the RMS
weight, where ``z`` is the compensated value (``decode(w, lo)``, or ``w`` for SR), read with
the climb removed (``optimizer.eval()`` for the wrappers). Usage: ``climb_e2e.py [out.md]``.
"""
import sys

import torch

import kaon
from kaon import MSAM, Adakaon, Nekaon
from kaon._compact_kahan import RESIDUAL_KEY, decode

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("kaon:", kaon.__file__, "| device:", dev, flush=True)


def ulp_ref(z):
    return float(torch.exp2(torch.floor(torch.log2(z.float().pow(2).mean().sqrt())) - 7))


def build(kind, params, method, fused):
    kw = dict(lr=1e-5, betas=(0.9, 0.999), cautious=False, foreach=True, fused=fused)
    if method is not None:
        kw["bf16_method"] = method
    if kind == "adakaon":
        return Adakaon(params, **kw)
    if kind == "msam":
        return MSAM(params, rho=0.05, **kw)
    return Nekaon(params, k=1.5, momentum_dtype="bfloat16", weight_decay=0.0, **kw)


def run(kind, fused, steps=300, seed=0):
    torch.manual_seed(seed)
    w = (torch.randn(64, 64, device=dev) * 0.05).to(torch.bfloat16)
    ref = torch.nn.Parameter(w.float())
    runs = {"fp32": ([ref], build(kind, [ref], None, fused))}
    for method in ("kahan8", "stochastic_rounding"):
        p = torch.nn.Parameter(w.clone())
        runs[method] = ([p], build(kind, [p], method, fused))
    for opt in runs.values():
        if hasattr(opt[1], "train"):
            opt[1].train()
    gg = torch.Generator(device=dev).manual_seed(seed + 1)
    g0 = torch.randn(64, 64, generator=gg, device=dev)
    for _ in range(steps):
        g = (g0 + 0.3 * torch.randn(64, 64, generator=gg, device=dev)).to(torch.bfloat16)
        for params, opt in runs.values():
            params[0].grad = g.to(params[0].dtype)
            opt.step()
    for _params, opt in runs.values():
        if hasattr(opt, "eval"):
            opt.eval()          # remove the climb: measure the CLEAN weights
    z_ref = runs["fp32"][0][0].data
    u = ulp_ref(z_ref)
    out = {}
    for method in ("kahan8", "stochastic_rounding"):
        params, opt = runs[method]
        p = params[0]
        inner = opt
        while hasattr(inner, "inner"):
            inner = inner.inner
        st = inner.state[p]
        z = decode(p.data, st[RESIDUAL_KEY]) if RESIDUAL_KEY in st else p.data.float()
        out[method] = float(((z - z_ref).abs() / u).max())
    moved = float(((z_ref - w.float()).abs() / u).max())
    return out, moved


rows = []
for kind in ("adakaon", "msam", "nekaon"):
    for fused in ([False, True] if dev == "cuda" else [False]):
        out, moved = run(kind, fused)
        rows.append((kind, fused, out["kahan8"], out["stochastic_rounding"], moved))
        print(f"{kind:8s} fused={fused!s:5s}  max err vs fp32 [ulp]: kahan8 {out['kahan8']:.3f}   "
              f"SR {out['stochastic_rounding']:.3f}   (reference moved up to {moved:.2f} ulp)", flush=True)
if len(sys.argv) > 1:
    with open(sys.argv[1], "w") as fh:
        fh.write("| optimizer | fused | kahan8 max err (ulp) | SR max err (ulp) | ref moved (ulp) |\n|---|---|---|---|---|\n")
        for k, f, a, b, m in rows:
            fh.write(f"| {k} | {f} | {a:.3f} | {b:.3f} | {m:.2f} |\n")
