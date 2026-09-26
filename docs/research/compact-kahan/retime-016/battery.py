"""ABBA re-timing battery for kaon 0.7.16 release gate (docs/research/compact-kahan/retime-016/).

Compares A = main 6201c1f (kaon 0.7.15) vs B = feature/nekaon-kahan 026621d.
B touched: Adakaon fused kernels' WD term (reads decoded value on kahan paths), cautious-mask
count with CK, an explicit tl.fma; a per-launch host `_fused_sr` guard; `bf16_method` added to
the fused partition cache key for Adakaon/AdaPNM; fp32 GC on the kahan native path.

Brazos (interleaved within each group, CUDA-event timed, synchronize every rep):
  * Adakaon fused+foreach x {stochastic_rounding, kahan8, kahan16}, weight_decay=0.1,
    cautious=True (default) -- x2 momentum regimes (no_momentum beta1=0 bf16, momentum_4bit).
    -> 12 arms per shape (2 modes x 3 methods x 2 momentum regimes).
  * Nekaon fused+foreach x {stochastic_rounding, kahan8, kahan16}, all other kwargs left at
    Nekaon defaults (betas=(0.5,0.999), weight_decay=0.1, momentum_dtype='4bit').
    -> 6 arms per shape.
  * AdaPNM fused, stochastic_rounding only (all other kwargs at default).
    -> 1 arm per shape.
  19 arms x 3 shapes = 57 arms per round.

Usage: <python> battery.py <tag> <outdir> [reps] [warmup]
Writes <outdir>/battery_<tag>.json and appends one line to <outdir>/battery_log.txt.
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import kaon
from kaon import Adakaon, Nekaon, AdaPNM

tag = sys.argv[1]
outdir = Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 40
warmup = int(sys.argv[4]) if len(sys.argv) > 4 else 15

dev = "cuda"


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


print(f"=== battery {tag} start @ {now()} ===", flush=True)
print("kaon:", kaon.__file__, flush=True)
print("torch:", torch.__version__, "cuda:", torch.version.cuda, flush=True)
smi = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,power.limit,power.draw,clocks.max.sm,clocks.sm,"
     "temperature.gpu,utilization.gpu", "--format=csv"],
    capture_output=True, text=True,
).stdout.strip()
print(smi, flush=True)


def median_iqr(ts):
    ts = sorted(ts)
    n = len(ts)
    med = ts[n // 2] if n % 2 else 0.5 * (ts[n // 2 - 1] + ts[n // 2])
    q1 = ts[n // 4]
    q3 = ts[(3 * n) // 4]
    return med, q1, q3


def interleaved_bench(arms: dict, reps: int, warmup: int):
    labels = list(arms)
    for label in labels:
        for _ in range(warmup):
            arms[label]()
    torch.cuda.synchronize()
    times = {label: [] for label in labels}
    for _r in range(reps):
        for label in labels:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            arms[label]()
            e.record()
            torch.cuda.synchronize()
            times[label].append(s.elapsed_time(e))
    out = {}
    for label in labels:
        med, q1, q3 = median_iqr(times[label])
        out[label] = {"median_ms": med, "q1_ms": q1, "q3_ms": q3, "n": reps}
    return out


def lora_bag(n_pairs=256, r=16, lo=320, hi=1280, seed=0):
    torch.manual_seed(seed)
    dims = torch.linspace(lo, hi, n_pairs).round().long().tolist()
    shapes = []
    for dim in dims:
        shapes.append((r, dim))
        shapes.append((dim, r))
    return shapes


SHAPES = {
    "unet": [(1024, 1024)] * 8 + [(4096,)] * 16,
    "big": [(1024, 1200)] * 2,
    "lora": lora_bag(),
}

MOMENTUM = {
    "no_momentum": {"betas": (0.0, 0.999), "momentum_dtype": "bfloat16"},
    "momentum_4bit": {"betas": (0.9, 0.999), "momentum_dtype": "4bit"},
}

METHODS = ("stochastic_rounding", "kahan8", "kahan16")


def make_params(shapes, seed=0):
    torch.manual_seed(seed)
    ps = [torch.nn.Parameter((torch.randn(s, device=dev) * 0.05).to(torch.bfloat16)) for s in shapes]
    for q in ps:
        q.grad = (torch.randn_like(q.float()) * 0.02).to(torch.bfloat16)
    return ps


def step_fn(opt):
    def fn():
        opt.step()
    return fn


def adakaon_group(shapes, mom_kw):
    """6 arms: {fused,foreach} x {sr,kahan8,kahan16}."""
    arms = {}
    keep = []
    for fused in (True, False):
        mode = "fused" if fused else "foreach"
        for method in METHODS:
            ps = make_params(shapes)
            opt = Adakaon(ps, lr=1e-4, weight_decay=0.1, cautious=True,
                          bf16_method=method, fused=fused, foreach=True, **mom_kw)
            keep.append((opt, ps))
            arms[f"{mode}_{method}"] = step_fn(opt)
    r = interleaved_bench(arms, reps=reps, warmup=warmup)
    del keep
    return r


def nekaon_group(shapes):
    """6 arms: {fused,foreach} x {sr,kahan8,kahan16}, Nekaon defaults otherwise."""
    arms = {}
    keep = []
    for fused in (True, False):
        mode = "fused" if fused else "foreach"
        for method in METHODS:
            ps = make_params(shapes)
            opt = Nekaon(ps, lr=1e-4, fused=fused, foreach=True, bf16_method=method)
            keep.append((opt, ps))
            arms[f"{mode}_{method}"] = step_fn(opt)
    r = interleaved_bench(arms, reps=reps, warmup=warmup)
    del keep
    return r


def adapnm_group(shapes):
    """1 arm: fused stochastic_rounding."""
    ps = make_params(shapes)
    opt = AdaPNM(ps, lr=1e-4, fused=True, bf16_method="stochastic_rounding")
    arms = {"fused_stochastic_rounding": step_fn(opt)}
    r = interleaved_bench(arms, reps=reps, warmup=warmup)
    del opt, ps
    return r


def main():
    t0 = time.time()
    results = {}
    for shape_name, shapes in SHAPES.items():
        print(f"--- shape {shape_name} @ {now()} ---", flush=True)
        results[shape_name] = {}

        results[shape_name]["adakaon"] = {}
        for mom_name, mom_kw in MOMENTUM.items():
            print(f"  adakaon/{mom_name} start @ {now()}", flush=True)
            g = adakaon_group(shapes, mom_kw)
            results[shape_name]["adakaon"][mom_name] = g
            for k, v in g.items():
                print(f"    adakaon {mom_name:14s} {k:22s} median={v['median_ms']:.4f}ms "
                      f"[{v['q1_ms']:.4f},{v['q3_ms']:.4f}]", flush=True)

        print(f"  nekaon start @ {now()}", flush=True)
        gn = nekaon_group(shapes)
        results[shape_name]["nekaon"] = gn
        for k, v in gn.items():
            print(f"    nekaon {k:22s} median={v['median_ms']:.4f}ms "
                  f"[{v['q1_ms']:.4f},{v['q3_ms']:.4f}]", flush=True)

        print(f"  adapnm start @ {now()}", flush=True)
        ga = adapnm_group(shapes)
        results[shape_name]["adapnm"] = ga
        for k, v in ga.items():
            print(f"    adapnm {k:22s} median={v['median_ms']:.4f}ms "
                  f"[{v['q1_ms']:.4f},{v['q3_ms']:.4f}]", flush=True)

        # incremental checkpoint
        (outdir / f"battery_{tag}.partial.json").write_text(json.dumps(results, indent=2))

    wall = time.time() - t0
    out = {"tag": tag, "file": kaon.__file__, "reps": reps, "warmup": warmup,
           "wall_s": wall, "results": results}
    (outdir / f"battery_{tag}.json").write_text(json.dumps(out, indent=2))
    with open(outdir / "battery_log.txt", "a") as fh:
        fh.write(f"{tag} wall={wall:.1f}s @ {now()}\n")
    print(f"=== battery {tag} done, wall={wall:.1f}s @ {now()} ===", flush=True)


if __name__ == "__main__":
    main()
