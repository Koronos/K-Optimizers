"""Sensitivity sweep for the unexported AdamG prototype."""

from __future__ import annotations

import argparse
import json
import math

import torch
from momo_adakaon_battery import _problems, _run_fixed

from kaon._adamg import AdamG


def _run(problem, *, cap: float, steps: int) -> dict[str, float | bool]:
    param = torch.nn.Parameter(problem.initial.clone())
    opt = AdamG([param], cap=cap)
    initial = float(problem.loss(param, 0).detach())
    maximum = initial
    for step in range(steps):
        opt.zero_grad()
        loss = problem.loss(param, step)
        loss.backward()
        if problem.name.endswith("clipped"):
            torch.nn.utils.clip_grad_norm_([param], 1.0)
        try:
            opt.step()
        except FloatingPointError:
            return {"finite": False, "initial": initial, "final": math.inf, "damage": math.inf}
        maximum = max(maximum, float(loss.detach()))
        if not bool(torch.isfinite(param).all()):
            return {"finite": False, "initial": initial, "final": math.inf, "damage": math.inf}
    final = float(problem.loss(param, steps).detach())
    return {
        "finite": math.isfinite(final),
        "initial": initial,
        "final": final,
        "damage": maximum / max(initial, 1e-30),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    args = parser.parse_args()
    caps = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, math.inf]
    fixed_lrs = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
    records = []
    for seed in args.seeds:
        for problem in _problems(seed):
            oracle = min(
                _run_fixed(problem, beta1=0.9, lr=lr, steps=args.steps)
                for lr in fixed_lrs
            )
            for cap in caps:
                run = _run(problem, cap=cap, steps=args.steps)
                initial = float(run["initial"])
                final = float(run["final"])
                epsilon = 1e-12 * max(initial, 1.0)
                denominator = math.log(max(initial, epsilon) / max(oracle, epsilon))
                if not bool(run["finite"]):
                    progress = -math.inf
                elif denominator <= math.log(2.0):
                    progress = 1.0
                else:
                    progress = math.log(max(initial, epsilon) / max(final, epsilon)) / denominator
                passed = (
                    bool(run["finite"])
                    and float(run["damage"]) <= 4.0
                    and progress >= 0.75
                    and final <= max(2.0 * oracle, 1e-8 * initial)
                )
                records.append(
                    {
                        "problem": problem.name,
                        "cap": cap,
                        "pass": passed,
                        "progress": progress,
                        "oracle": oracle,
                        **run,
                    }
                )
    summary = []
    for cap in caps:
        selected = [record for record in records if record["cap"] == cap]
        summary.append(
            {
                "cap": cap,
                "pass_count": sum(bool(record["pass"]) for record in selected),
                "total": len(selected),
                "worst_progress": min(float(record["progress"]) for record in selected),
                "worst_damage": max(float(record["damage"]) for record in selected),
            }
        )
    print(json.dumps(summary, indent=2))
    print("cap=1 detail")
    print(
        json.dumps(
            [record for record in records if record["cap"] == 1.0],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
