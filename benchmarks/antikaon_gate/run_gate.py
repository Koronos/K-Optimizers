"""Antikaon quality gate — staged CONSTANT-LR control-battery run.

Only quality (held-out loss, train-val gap) matters here: no ms/step is measured or
reported (the GPU is shared with another job during this run). Reuses the control
battery's own harness/dataset/registry and its exact ``train()`` loop (imported by file
path, same as ``battery.py`` does) so results are directly comparable to the rest of the
battery's constant-LR ("continuity") numbers.

Protocol (docs/research/antikaon-design.md §6, adapted per dispatch):
  stage1: C=40, N=600,  seed=0        -> reject divergence/underfitting cheaply
  stage2: C=40, N=2000, seeds=0,1     -> exposes late overfitting

Arms: A0 Adakaon-nomom, A1 Nekaon (defaults), B1-B3 Antikaon k_sigma in {1.5,5,15},
C1 (shape=none), C2 (antithetic), C3 (sigma_ref=weight), plus B2 at lr x0.5 and x2 — all
pulled straight from ``benchmarks/control/registry.py``'s ``Antikaon``/``Nekaon``/
``Adakaon-nomom`` entries so the gate can never drift from the registry's own config.

Usage:
    python run_gate.py --stage 1
    python run_gate.py --stage 2 --arms "A0,A1,B1,B2,B3"
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


CTRL = _load("control_battery", f"{REPO}/benchmarks/control/battery.py")
H = CTRL.H
D = CTRL.D
REG = CTRL.REG
OPTIMIZERS = REG.OPTIMIZERS
DEV = CTRL.DEV

ANTI = OPTIMIZERS["Antikaon"]
NEK = OPTIMIZERS["Nekaon"]
ADA0 = OPTIMIZERS["Adakaon-nomom"]
AV = ANTI["variants"]

# every arm's `make` and constant-LR come straight from the registry (no re-declaration
# of the optimizer configs here) -- only the lr override for the B2 x0.5/x2 anchor probe
# is arm-specific.
ARMS = {
    "A0 Adakaon-nomom": dict(make=ADA0["make"], lr=ADA0["lr_const"]),
    "A1 Nekaon (defaults)": dict(make=NEK["make"], lr=NEK["lr_const"]),
    # B1-B3 monotonically trade loss for gap as k_sigma grows (stage-2 finding); this arm
    # extends the sweep DOWN (loss keeps worsening as k_sigma increases -> check if it keeps
    # improving below 1.5) per the dispatch's monotonicity rule.
    "B0 k_sigma=0.5": dict(make=lambda p, lr: REG.Antikaon(p, lr=lr, k_sigma=0.5), lr=ANTI["lr_const"]),
    "B1 k_sigma=1.5": dict(make=AV["B1 k_sigma=1.5"], lr=ANTI["lr_const"]),
    "B2 k_sigma=5": dict(make=AV["B2 k_sigma=5"], lr=ANTI["lr_const"]),
    "B3 k_sigma=15": dict(make=AV["B3 k_sigma=15"], lr=ANTI["lr_const"]),
    "C1 shape=none": dict(make=AV["C1 k_sigma=5 shape=none"], lr=ANTI["lr_const"]),
    "C2 antithetic": dict(make=AV["C2 k_sigma=5 antithetic"], lr=ANTI["lr_const"]),
    "C3 sigma_ref=weight": dict(make=AV["C3 k_sigma=5 sigma_ref=weight"], lr=ANTI["lr_const"]),
    "B2 lr x0.5": dict(make=AV["B2 k_sigma=5"], lr=ANTI["lr_const"] * 0.5),
    "B2 lr x2": dict(make=AV["B2 k_sigma=5"], lr=ANTI["lr_const"] * 2.0),
}

STAGES = {
    1: dict(C=40, N=600, seeds=[0]),
    2: dict(C=40, N=2000, seeds=[0, 1]),
}


def mean(xs):
    return sum(xs) / len(xs)


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def run_stage(stage_no, arm_names, out_path):
    scfg = STAGES[stage_no]
    C, N, seeds = scfg["C"], scfg["N"], scfg["seeds"]
    ds = D.build_proxy_dataset()
    fp = D.fingerprint(ds)
    data = {k: v.to(DEV).to(H.DT) for k, v in ds["DATA"].items()}
    tr, te = ds["TR"], ds["TE"]
    ac = H.make_alphas()
    seq = CTRL.seq_prog(N)

    results = {}
    t_wall0 = time.time()
    for name in arm_names:
        arm = ARMS[name]
        per_seed = []
        for seed in seeds:
            r = CTRL.train(arm["make"], arm["lr"], schedule="const", seq=seq, seed=seed,
                            data=data, tr=tr, te=te, ac=ac, channels=C, bs=8, n=N)
            per_seed.append(dict(seed=seed, train=r["tr"], test=r["te"], gap=r["gap"]))
            print(f"  [{name}] seed={seed} train={r['tr']:.5f} test={r['te']:.5f} "
                  f"gap={r['gap']:+.5f} bpp={r['bpp']:.3f}", flush=True)
        te_vals = [p["test"] for p in per_seed]
        gap_vals = [p["gap"] for p in per_seed]
        results[name] = dict(
            per_seed=per_seed,
            test_mean=mean(te_vals), test_std=stdev(te_vals),
            gap_mean=mean(gap_vals), gap_std=stdev(gap_vals),
            bpp=r["bpp"],
        )
        print(f"{name:24s} test={results[name]['test_mean']:.5f}"
              f"(+-{results[name]['test_std']:.5f}) "
              f"gap={results[name]['gap_mean']:+.5f}(+-{results[name]['gap_std']:.5f})",
              flush=True)

    manifest = dict(
        commit=os.popen("git rev-parse HEAD").read().strip(),
        branch=os.popen("git rev-parse --abbrev-ref HEAD").read().strip(),
        stage=stage_no, C=C, N=N, seeds=seeds, bs=8, schedule="const",
        proxy_weight_dtype=str(H.DT), dataset_fingerprint=fp,
        wall_s=time.time() - t_wall0,
        arms={n: dict(lr=ARMS[n]["lr"]) for n in arm_names},
    )
    out = dict(manifest=manifest, results=results)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {out_path}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2])
    ap.add_argument("--arms", type=str, default=None, help="comma-sep arm names (default: all)")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    arm_names = [s.strip() for s in a.arms.split(",")] if a.arms else list(ARMS)
    out_path = a.out or f"{HERE}/stage{a.stage}.json"
    run_stage(a.stage, arm_names, out_path)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
