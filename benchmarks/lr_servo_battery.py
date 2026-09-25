"""Local-LR servo battery: fixed LR versus the same LR with correction enabled."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch
from torch import nn

from kaon import Adakaon, Lion, Nekaon

FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)
REFERENCE_LR = {"adakaon": 3e-3, "nekaon": 3e-3, "lion": 1e-3}


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


def make_optimizer(name: str, params, lr: float, servo: bool):
    shared = dict(
        lr=lr, lr_servo=servo, weight_decay=0.0, cautious=False,
        gradient_centralization=False, momentum_dtype="float32", foreach=True,
    )
    if name == "adakaon":
        return Adakaon(params, betas=(0.9, 0.999), **shared)
    if name == "nekaon":
        return Nekaon(params, k=1.5, betas=(0.5, 0.999), **shared)
    if name == "lion":
        return Lion(params, betas=(0.9, 0.99), **shared)
    raise ValueError(name)


def eval_loss(model, optimizer, x, y) -> float:
    eval_fn, train_fn = getattr(optimizer, "eval", None), getattr(optimizer, "train", None)
    if callable(eval_fn):
        eval_fn()
    try:
        with torch.no_grad():
            return float(torch.nn.functional.mse_loss(model(x), y))
    finally:
        if callable(train_fn):
            train_fn()


def run(name: str, seed: int, steps: int, factor: float, servo: bool, device: torch.device):
    model, features, targets = problem(seed, device)
    lr = REFERENCE_LR[name] * factor
    optimizer = make_optimizer(name, model.parameters(), lr, servo)
    train_x, test_x = features[:64], features[64:]
    train_y, test_y = targets[:64], targets[64:]
    tail = []
    start = time.perf_counter()
    for step in range(steps):
        indices = torch.arange(step * 16, step * 16 + 16, device=device) % len(train_x)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(train_x[indices]), train_y[indices])
        loss.backward()
        optimizer.step()
        if (step + 1) % 8 == 0:
            tail.append(eval_loss(model, optimizer, test_x, test_y))
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "optimizer": name,
        "seed": seed,
        "factor": factor,
        "servo": servo,
        "tail_loss": sum(tail[-4:]) / min(4, len(tail)),
        "final_lr": optimizer.param_groups[0]["lr"],
        "scale": optimizer.get_lr_servo_scale(),
        "ms_step": 1000.0 * elapsed / steps,
        "finite": math.isfinite(tail[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizers", nargs="+", default=["adakaon", "nekaon", "lion"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    rows = [
        run(name, seed, args.steps, factor, servo, device)
        for name in args.optimizers
        for seed in args.seeds
        for factor in FACTORS
        for servo in (False, True)
    ]
    pairs = []
    for fixed, servo in zip(rows[::2], rows[1::2], strict=True):
        pairs.append({
            "optimizer": fixed["optimizer"], "seed": fixed["seed"], "factor": fixed["factor"],
            "fixed_loss": fixed["tail_loss"], "servo_loss": servo["tail_loss"],
            "relative": servo["tail_loss"] / fixed["tail_loss"],
            "final_scale": servo["scale"], "overhead": servo["ms_step"] / fixed["ms_step"],
        })
    summary = {}
    for name in args.optimizers:
        summary[name] = {}
        for factor in FACTORS:
            selected = [p for p in pairs if p["optimizer"] == name and p["factor"] == factor]
            summary[name][str(factor)] = {
                "mean_relative": statistics.mean(p["relative"] for p in selected),
                "worst_relative": max(p["relative"] for p in selected),
                "mean_final_scale": statistics.mean(p["final_scale"] for p in selected),
                "median_overhead": statistics.median(p["overhead"] for p in selected),
            }
    payload = {"summary": summary} if args.summary_only else {
        "summary": summary, "pairs": pairs, "rows": rows,
    }
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
