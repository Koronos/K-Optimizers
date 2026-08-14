"""Paired multi-seed Lion + Mechanic research battery.

This small nonlinear regression proxy isolates global scale selection without the
runtime cost of the image proxy.  All arms in a seed share model initialization,
training data, minibatch order, and held-out data.  Fixed-LR Lion provides the
oracle grid; faithful Mechanic and the guarded prototype wrap a unit-LR Lion.

Example::

    python benchmarks/lion_mechanic_battery.py --output lion_mechanic.json
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from kaon import Lion, __version__
from kaon._mechanic_addon import MechanicAddon

SEEDS = (17, 29, 43)
FIXED_LRS = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
ARMS = ("mechanic", "mechanic_guard")


def build_problem(seed: int, device: torch.device):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(320, 32, generator=generator)
    teacher = nn.Sequential(nn.Linear(32, 48), nn.Tanh(), nn.Linear(48, 8))
    with torch.no_grad():
        for parameter in teacher.parameters():
            parameter.copy_(0.4 * torch.randn(parameter.shape, generator=generator))
        targets = teacher(features) + 0.03 * torch.randn(320, 8, generator=generator)
    torch.manual_seed(seed + 1000)
    model = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 8))
    return model.to(device), features.to(device), targets.to(device)


def make_optimizer(arm: str, params, *, lr: float | None):
    common = dict(
        betas=(0.9, 0.99),
        weight_decay=0.0,
        cautious=False,
        gradient_centralization=False,
        momentum_dtype="float32",
        foreach=True,
        auto_lr=False,
    )
    if arm == "fixed":
        if lr is None:
            raise ValueError("fixed Lion requires an LR")
        return Lion(params, lr=lr, **common)
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    inner = Lion(params, lr=1.0, **common)
    return MechanicAddon(inner, guard=arm == "mechanic_guard")


def transient_damage(values: list[float]) -> float:
    running_best = values[0]
    result = 0.0
    for value in values[1:]:
        running_best = min(running_best, value)
        result = max(result, value / max(running_best, 1e-12) - 1.0)
    return result


@dataclass
class Result:
    seed: int
    arm: str
    configured_lr: float | None
    finite: bool
    initial_loss: float
    final_loss: float | None
    best_loss: float
    damage: float
    scale_trace: list[float]
    trajectory: list[dict[str, float | int | None]]
    time_to_quality: int | None = None
    quality_target: float | None = None


def run_arm(
    *,
    seed: int,
    arm: str,
    lr: float | None,
    steps: int,
    batch_size: int,
    checkpoints: int,
    device: torch.device,
) -> Result:
    model, features, targets = build_problem(seed, device)
    optimizer = make_optimizer(arm, model.parameters(), lr=lr)
    train_x, test_x = features[:64], features[64:]
    train_y, test_y = targets[:64], targets[64:]
    with torch.no_grad():
        initial = float(torch.nn.functional.mse_loss(model(test_x), test_y))
    observed = [initial]
    trajectory: list[dict[str, float | int | None]] = [
        {"step": 0, "train_loss": None, "heldout_loss": initial}
    ]
    scale_trace: list[float] = []
    every = max(1, steps // checkpoints)
    finite = True
    for step in range(steps):
        start = (step * batch_size) % len(train_x)
        indices = torch.arange(start, start + batch_size, device=device) % len(train_x)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(train_x[indices]), train_y[indices])
        train_loss = float(loss.detach())
        if not math.isfinite(train_loss):
            finite = False
            break
        loss.backward()
        try:
            optimizer.step()
        except FloatingPointError:
            finite = False
            break
        scale = float(optimizer.get_scale()) if isinstance(optimizer, MechanicAddon) else float(lr)
        scale_trace.append(scale)
        if not math.isfinite(scale) or not all(
            bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
        ):
            finite = False
            break
        point: dict[str, float | int | None] = {
            "step": step + 1,
            "train_loss": train_loss,
            "heldout_loss": None,
        }
        if (step + 1) % every == 0 or step + 1 == steps:
            with torch.no_grad():
                heldout = float(torch.nn.functional.mse_loss(model(test_x), test_y))
            if not math.isfinite(heldout):
                finite = False
                break
            point["heldout_loss"] = heldout
            observed.append(heldout)
        trajectory.append(point)
    return Result(
        seed=seed,
        arm=arm,
        configured_lr=lr,
        finite=finite and len(scale_trace) == steps,
        initial_loss=initial,
        final_loss=observed[-1] if finite and len(scale_trace) == steps else None,
        best_loss=min(observed),
        damage=transient_damage(observed),
        scale_trace=scale_trace,
        trajectory=trajectory,
    )


def annotate_quality(rows: list[dict[str, Any]], tolerance: float = 0.05) -> None:
    for seed in sorted({row["seed"] for row in rows}):
        fixed = [
            row
            for row in rows
            if row["seed"] == seed and row["arm"] == "fixed" and row["final_loss"] is not None
        ]
        target = min(row["final_loss"] for row in fixed) * (1.0 + tolerance)
        for row in (candidate for candidate in rows if candidate["seed"] == seed):
            row["quality_target"] = target
            row["time_to_quality"] = next(
                (
                    int(point["step"])
                    for point in row["trajectory"]
                    if point["heldout_loss"] is not None and point["heldout_loss"] <= target
                ),
                None,
            )


def run_battery(
    *, seeds: tuple[int, ...], steps: int, batch_size: int, checkpoints: int, device: torch.device
) -> dict[str, Any]:
    results = []
    for seed in seeds:
        for lr in FIXED_LRS:
            results.append(
                run_arm(
                    seed=seed,
                    arm="fixed",
                    lr=lr,
                    steps=steps,
                    batch_size=batch_size,
                    checkpoints=checkpoints,
                    device=device,
                )
            )
        for arm in ARMS:
            results.append(
                run_arm(
                    seed=seed,
                    arm=arm,
                    lr=None,
                    steps=steps,
                    batch_size=batch_size,
                    checkpoints=checkpoints,
                    device=device,
                )
            )
    rows = [asdict(result) for result in results]
    annotate_quality(rows)
    summary = {}
    for arm in ("fixed", *ARMS):
        members = [row for row in rows if row["arm"] == arm]
        final = [row["final_loss"] for row in members if row["final_loss"] is not None]
        reached = [row["time_to_quality"] for row in members if row["time_to_quality"] is not None]
        summary[arm] = {
            "runs": len(members),
            "all_finite": all(row["finite"] for row in members),
            "mean_final_loss": sum(final) / len(final) if final else None,
            "max_damage": max(row["damage"] for row in members),
            "quality_reached": len(reached),
            "mean_time_to_quality": sum(reached) / len(reached) if reached else None,
        }
    return {
        "kaon_version": __version__,
        "device": str(device),
        "steps": steps,
        "seeds": list(seeds),
        "fixed_lrs": list(FIXED_LRS),
        "summary": summary,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoints", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_battery(
        seeds=tuple(args.seeds),
        steps=args.steps,
        batch_size=args.batch_size,
        checkpoints=args.checkpoints,
        device=torch.device(args.device),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
