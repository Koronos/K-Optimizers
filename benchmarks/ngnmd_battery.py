"""Three-seed no-scheduler battery for the NGN-MDv1 reference."""

from __future__ import annotations

import argparse
import json
import math
from statistics import mean

import torch
from torch import nn

from kaon import NGNMD, Adakaon, Lion, Nekaon

GRIDS = {
    "adakaon": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
    "nekaon": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
    "lion": (3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
    "ngnmd": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0),
}


def problem(seed: int, device: torch.device):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(320, 32, generator=generator)
    teacher = nn.Sequential(nn.Linear(32, 48), nn.Tanh(), nn.Linear(48, 8))
    with torch.no_grad():
        for parameter in teacher.parameters():
            parameter.copy_(0.4 * torch.randn(parameter.shape, generator=generator))
        targets = teacher(features) + 0.03 * torch.randn(320, 8, generator=generator)
    torch.manual_seed(seed + 1000)
    model = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 8)).to(device)
    return model, features.to(device), targets.to(device)


def make_optimizer(name: str, params, lr: float):
    shared = dict(
        lr=lr, weight_decay=0.0, cautious=False, gradient_centralization=False,
        momentum_dtype="float32", foreach=True,
    )
    if name == "adakaon":
        return Adakaon(params, betas=(0.9, 0.999), **shared)
    if name == "nekaon":
        return Nekaon(params, k=1.5, betas=(0.5, 0.999), **shared)
    if name == "lion":
        return Lion(params, betas=(0.9, 0.99), **shared)
    if name == "ngnmd":
        return NGNMD(params, lr=lr, betas=(0.9, 0.999))
    raise ValueError(name)


def evaluate(model, optimizer, x, y) -> float:
    eval_fn, train_fn = getattr(optimizer, "eval", None), getattr(optimizer, "train", None)
    if callable(eval_fn):
        eval_fn()
    try:
        with torch.no_grad():
            return float(torch.nn.functional.mse_loss(model(x), y))
    finally:
        if callable(train_fn):
            train_fn()


def run(name: str, lr: float, seed: int, steps: int, device: torch.device):
    model, x, y = problem(seed, device)
    optimizer = make_optimizer(name, model.parameters(), lr)
    train_x, test_x, train_y, test_y = x[:64], x[64:], y[:64], y[64:]
    trace = []
    for step in range(steps):
        indices = torch.arange(step * 16, step * 16 + 16, device=device) % len(train_x)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(train_x[indices]), train_y[indices])
        loss.backward()
        if name == "ngnmd":
            optimizer.step(loss=loss.detach())
        else:
            optimizer.step()
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            return {"name": name, "lr": lr, "seed": seed, "finite": False}
        if (step + 1) % 8 == 0:
            trace.append(evaluate(model, optimizer, test_x, test_y))
    effective = optimizer.get_effective_lr() if name == "ngnmd" else optimizer.param_groups[0]["lr"]
    return {
        "name": name, "lr": lr, "seed": seed, "finite": all(map(math.isfinite, trace)),
        "tail": mean(trace[-4:]), "best": min(trace), "effective_lr": effective,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizers", nargs="+", default=list(GRIDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    rows = [
        run(name, lr, seed, args.steps, device)
        for name in args.optimizers for seed in args.seeds for lr in GRIDS[name]
    ]
    summary = {}
    for name in args.optimizers:
        selected = [row for row in rows if row["name"] == name and row["finite"]]
        per_lr = {
            str(lr): mean(row["tail"] for row in selected if row["lr"] == lr)
            for lr in GRIDS[name]
            if any(row["lr"] == lr for row in selected)
        }
        best_lr, best_loss = min(per_lr.items(), key=lambda item: item[1])
        summary[name] = {
            "best_lr": best_lr, "best_loss": best_loss, "all_finite": len(selected) == len(GRIDS[name]) * len(args.seeds),
            "per_lr": per_lr,
        }
    payload = {"summary": summary} if args.summary_only else {"summary": summary, "rows": rows}
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
