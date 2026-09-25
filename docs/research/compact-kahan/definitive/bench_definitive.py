"""DEFINITIVE speed measurement: bf16_method="kahan8" (compact Kahan, 1 B/param) vs
"stochastic_rounding" (default, 0 compensation bytes) vs "kahan" (legacy, 2 B/param).

Protocol (see docs/research/compact-kahan/definitive/MANIFEST.md):
  * AC power only, GPU confirmed idle before the run.
  * Arms interleaved (ABC ABC ...) within each comparable group to spread thermal drift.
  * >=30 timed reps per arm after warmup, CUDA-event timed, synchronized every rep.
  * "kahan" (legacy) never reaches foreach/fused (see kaon._backend.per_param_only_bf16_method)
    -> it is only measured on the native per-param path; SR and kahan8 are ALSO measured on
    that same native path so the three methods get one true 3-way apples-to-apples group,
    in addition to their own foreach-vs-foreach and fused-vs-fused groups.

Writes:
  * definitive/writer_results.json + writer_table.md  (isolated writer, 2^22 elems)
  * definitive/step_results.json  + step_table.md      (full Adakaon step, 3 shapes x 2
    momentum regimes x {native, foreach, fused})
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

import kaon
from kaon import Adakaon
from kaon import _backend as bk

OUTDIR = Path(__file__).parent
dev = "cuda"

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


def interleaved_bench(arms: dict, reps: int = 32, warmup: int = 6):
    """arms: {label: callable}. Runs ABC ABC... across all labels, CUDA-event timed."""
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
        out[label] = {"median_ms": med, "q1_ms": q1, "q3_ms": q3, "n": reps, "raw_ms": times[label]}
    return out


# --------------------------------------------------------------------------- writer (isolated)
def bench_writer():
    n = 1 << 22
    torch.manual_seed(0)
    p_master = (torch.randn(n, device=dev) * 0.05).to(torch.bfloat16)
    d = torch.randn(n, device=dev)

    arms = {}
    holders = {}
    for method in ("stochastic_rounding", "kahan", "kahan8"):
        pp = p_master.clone()
        st = {}
        bk.init_bf16_state(pp, st, method)
        holders[method] = (pp, st)

        def fn(pp=pp, st=st, method=method):
            bk.subtract_one_(pp, d, st, method, alpha=1e-5)

        arms[method] = fn

    return interleaved_bench(arms, reps=40, warmup=8)


# --------------------------------------------------------------------------- step (full optimizer)
def lora_bag(n_pairs=256, r=16, lo=320, hi=1280, seed=0):
    """512 tensors total: n_pairs x (A: r x dim, B: dim x r), dims spread across [lo, hi]."""
    torch.manual_seed(seed)
    dims = torch.linspace(lo, hi, n_pairs).round().long().tolist()
    shapes = []
    for dim in dims:
        shapes.append((r, dim))
        shapes.append((dim, r))
    return shapes


SHAPES = {
    "big 2x(1024,1200)": [(1024, 1200)] * 2,
    "LoRA bag 512x r=16 dims 320-1280": lora_bag(),
    "UNet-ish 8x(1024,1024)+16x(4096,)": [(1024, 1024)] * 8 + [(4096,)] * 16,
}

MOMENTUM = {
    "no_momentum (beta1=0)": {"betas": (0.0, 0.999), "momentum_dtype": "bfloat16"},
    "momentum_4bit": {"betas": (0.9, 0.999), "momentum_dtype": "4bit"},
}


def make_params(shapes, seed=0):
    torch.manual_seed(seed)
    ps = [torch.nn.Parameter((torch.randn(s, device=dev) * 0.05).to(torch.bfloat16)) for s in shapes]
    for q in ps:
        q.grad = (torch.randn_like(q.float()) * 0.02).to(torch.bfloat16)
    return ps


def make_opt(shapes, method, mom_kw, fused, foreach=True, seed=0):
    ps = make_params(shapes, seed=seed)
    opt = Adakaon(ps, lr=1e-4, bf16_method=method, fused=fused, foreach=foreach, **mom_kw)
    return opt, ps


def step_fn(opt):
    def fn():
        opt.step()
    return fn


def bench_steps():
    results = {}
    for shape_name, shapes in SHAPES.items():
        results[shape_name] = {}
        n_reps = 40 if shape_name.startswith("big") else 32
        warm = 6
        for mom_name, mom_kw in MOMENTUM.items():
            print(f"start: {shape_name} / {mom_name}  @ {now()}", flush=True)
            group = {}

            # --- native (per-param): SR, kahan, kahan8 -- true 3-way ABC ---
            native_arms = {}
            native_opts = {}
            for method in ("stochastic_rounding", "kahan", "kahan8"):
                opt, ps = make_opt(shapes, method, mom_kw, fused=False, foreach=False)
                native_opts[method] = (opt, ps)
                native_arms[method] = step_fn(opt)
            group["native"] = interleaved_bench(native_arms, reps=n_reps, warmup=warm)
            del native_opts

            # --- foreach (fused=False, foreach=True default): SR, kahan8 only ---
            foreach_arms = {}
            foreach_opts = {}
            for method in ("stochastic_rounding", "kahan8"):
                opt, ps = make_opt(shapes, method, mom_kw, fused=False, foreach=True)
                foreach_opts[method] = (opt, ps)
                foreach_arms[method] = step_fn(opt)
            group["foreach"] = interleaved_bench(foreach_arms, reps=n_reps, warmup=warm)
            del foreach_opts

            # --- fused=True: SR, kahan8 only ---
            fused_arms = {}
            fused_opts = {}
            for method in ("stochastic_rounding", "kahan8"):
                opt, ps = make_opt(shapes, method, mom_kw, fused=True)
                fused_opts[method] = (opt, ps)
                fused_arms[method] = step_fn(opt)
            group["fused"] = interleaved_bench(fused_arms, reps=n_reps, warmup=warm)
            del fused_opts

            results[shape_name][mom_name] = group
            print(f"done: {shape_name} / {mom_name}  @ {now()}", flush=True)
            for mode, arms in group.items():
                for method, r in arms.items():
                    print(f"  {mode:8s} {method:22s} median={r['median_ms']:.4f} ms "
                          f"[{r['q1_ms']:.4f}, {r['q3_ms']:.4f}]", flush=True)
            # incremental checkpoint: a kill/interruption mid-run should not lose completed groups
            (OUTDIR / "step_results.partial.json").write_text(json.dumps(strip_raw(results), indent=2))
    return results


def strip_raw(obj):
    """Drop raw_ms lists for the compact JSON (kept separately if needed)."""
    if isinstance(obj, dict):
        return {k: strip_raw(v) for k, v in obj.items() if k != "raw_ms"}
    return obj


def main():
    t0 = time.time()
    print(f"=== writer isolated (2^22 elements) === @ {now()}", flush=True)
    writer = bench_writer()
    print(f"writer done @ {now()}", flush=True)
    for method, r in writer.items():
        print(f"  {method:22s} median={r['median_ms']:.4f} ms [{r['q1_ms']:.4f}, {r['q3_ms']:.4f}]", flush=True)

    (OUTDIR / "writer_results.json").write_text(json.dumps(strip_raw(writer), indent=2))
    with open(OUTDIR / "writer_table.md", "w") as fh:
        fh.write("| method | median ms | q1 ms | q3 ms |\n|---|---|---|---|\n")
        for method, r in writer.items():
            fh.write(f"| {method} | {r['median_ms']:.4f} | {r['q1_ms']:.4f} | {r['q3_ms']:.4f} |\n")

    print("\n=== full step (SR / kahan / kahan8, native / foreach / fused) ===", flush=True)
    steps = bench_steps()
    (OUTDIR / "step_results.json").write_text(json.dumps(strip_raw(steps), indent=2))

    with open(OUTDIR / "step_table.md", "w") as fh:
        for shape_name, moms in steps.items():
            fh.write(f"\n## {shape_name}\n\n")
            for mom_name, modes in moms.items():
                fh.write(f"\n### {mom_name}\n\n")
                fh.write("| mode | method | median ms | q1 ms | q3 ms |\n|---|---|---|---|---|\n")
                for mode, arms in modes.items():
                    for method, r in arms.items():
                        fh.write(f"| {mode} | {method} | {r['median_ms']:.4f} | {r['q1_ms']:.4f} | {r['q3_ms']:.4f} |\n")

    print(f"\ntotal wall time: {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
