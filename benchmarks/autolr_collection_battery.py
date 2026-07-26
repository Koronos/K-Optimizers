"""Paired multi-seed AutoLR battery for Adakaon, Nekaon, and Lion."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn

from kaon import Adakaon, Lion, Nekaon, __version__

SEEDS = (17, 29, 43)
FIXED_LRS = {
    "adakaon": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
    "nekaon": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
    "lion": (3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
}


def _problem(seed: int, device: torch.device):
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


def _optimizer(name: str, params, *, lr: float, auto_lr: bool):
    shared = dict(
        lr=lr,
        weight_decay=0.0,
        cautious=False,
        gradient_centralization=False,
        momentum_dtype="float32",
        foreach=True,
        auto_lr=auto_lr,
    )
    if name == "adakaon":
        return Adakaon(params, betas=(0.9, 0.999), **shared)
    if name == "nekaon":
        return Nekaon(params, k=1.5, betas=(0.5, 0.999), **shared)
    if name == "lion":
        return Lion(params, betas=(0.9, 0.99), **shared)
    raise ValueError(name)


def _evaluate(optimizer, fn):
    eval_fn = getattr(optimizer, "eval", None)
    train_fn = getattr(optimizer, "train", None)
    if callable(eval_fn):
        eval_fn()
    try:
        return fn()
    finally:
        if callable(train_fn):
            train_fn()


def _damage(values: list[float]) -> float:
    best = values[0]
    damage = 0.0
    for value in values[1:]:
        best = min(best, value)
        damage = max(damage, value / max(best, 1e-12) - 1.0)
    return damage


def _run(name: str, seed: int, steps: int, *, lr: float, auto_lr: bool, device):
    model, features, targets = _problem(seed, device)
    optimizer = _optimizer(name, model.parameters(), lr=lr, auto_lr=auto_lr)
    train_x, test_x = features[:64], features[64:]
    train_y, test_y = targets[:64], targets[64:]
    with torch.no_grad():
        initial = _evaluate(
            optimizer, lambda: float(torch.nn.functional.mse_loss(model(test_x), test_y))
        )
    heldout = [initial]
    trace = []
    finite = True
    for step in range(steps):
        indices = torch.arange(step * 16, step * 16 + 16, device=device) % len(train_x)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(train_x[indices]), train_y[indices])
        if not math.isfinite(float(loss.detach())):
            finite = False
            break
        loss.backward()
        try:
            optimizer.step()
        except FloatingPointError:
            finite = False
            break
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            finite = False
            break
        if (step + 1) % 8 == 0:
            with torch.no_grad():
                value = _evaluate(
                    optimizer,
                    lambda: float(torch.nn.functional.mse_loss(model(test_x), test_y)),
                )
            heldout.append(value)
            trace.append({"step": step + 1, "loss": value, "lr": optimizer.get_d()})
            if not math.isfinite(value):
                finite = False
                break
    return {
        "optimizer": name,
        "seed": seed,
        "arm": "autolr" if auto_lr else "fixed",
        "configured_lr": None if auto_lr else lr,
        "finite": finite and len(trace) == steps // 8,
        "initial_loss": initial,
        "final_loss": heldout[-1] if finite else None,
        "tail_loss": (
            sum(point["loss"] for point in trace[-4:]) / min(4, len(trace))
            if finite and trace else None
        ),
        "best_loss": min(heldout),
        "damage": _damage(heldout),
        "final_lr": optimizer.get_d(),
        "trace": trace,
    }


def run_battery(names, seeds, steps, device, *, auto_only: bool = False):
    rows = []
    for name in names:
        for seed in seeds:
            if not auto_only:
                for lr in FIXED_LRS[name]:
                    rows.append(_run(name, seed, steps, lr=lr, auto_lr=False, device=device))
            rows.append(_run(name, seed, steps, lr=1.0, auto_lr=True, device=device))

    if auto_only:
        summary = {}
        for name in names:
            selected = [row for row in rows if row["optimizer"] == name]
            summary[name] = {
                "all_finite": all(row["finite"] for row in selected),
                "max_damage": max(row["damage"] for row in selected),
                "seeds": [
                    {
                        "seed": row["seed"],
                        "final_loss": row["final_loss"],
                        "best_loss": row["best_loss"],
                        "final_lr": row["final_lr"],
                        "tail_lr_min": min(point["lr"] for point in row["trace"][-64:]),
                        "tail_lr_max": max(point["lr"] for point in row["trace"][-64:]),
                        "finite": row["finite"],
                    }
                    for row in selected
                ],
            }
        return {"kaon_version": __version__, "steps": steps, "summary": summary, "rows": rows}

    summary = {}
    for name in names:
        per_seed = []
        for seed in seeds:
            fixed = [
                row for row in rows
                if row["optimizer"] == name and row["seed"] == seed
                and row["arm"] == "fixed" and row["finite"]
            ]
            auto = next(
                row for row in rows
                if row["optimizer"] == name and row["seed"] == seed and row["arm"] == "autolr"
            )
            # A single final minibatch checkpoint is noisy. The mean of the last
            # four held-out evaluations is the terminal-quality estimate used by
            # the release gate; raw final_loss remains in every row for auditing.
            oracle = min(fixed, key=lambda row: row["tail_loss"])
            per_seed.append({
                "seed": seed,
                "oracle_lr": oracle["configured_lr"],
                "oracle_loss": oracle["tail_loss"],
                "autolr_loss": auto["tail_loss"],
                "relative": None if not auto["finite"] else auto["tail_loss"] / oracle["tail_loss"],
                "autolr_damage": auto["damage"],
                "autolr_final_lr": auto["final_lr"],
                "finite": auto["finite"],
            })
        summary[name] = {
            "all_finite": all(row["finite"] for row in per_seed),
            "worst_relative": max(row["relative"] for row in per_seed if row["relative"] is not None),
            "mean_relative": sum(row["relative"] for row in per_seed if row["relative"] is not None) / len(per_seed),
            "max_damage": max(row["autolr_damage"] for row in per_seed),
            "seeds": per_seed,
        }
    return {"kaon_version": __version__, "steps": steps, "summary": summary, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizers", nargs="+", choices=tuple(FIXED_LRS), default=list(FIXED_LRS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--auto-only",
        action="store_true",
        help="run only the AutoLR arms (useful for inexpensive long-horizon stability checks)",
    )
    args = parser.parse_args()
    result = run_battery(
        args.optimizers,
        args.seeds,
        args.steps,
        torch.device(args.device),
        auto_only=args.auto_only,
    )
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2) if args.summary_only else rendered)


if __name__ == "__main__":
    main()
