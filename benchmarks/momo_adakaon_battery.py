"""Diagnostic cap sweep for the experimental MoMo-Adakaon implementation.

This is intentionally not a release benchmark yet.  It asks the first blocking
question: does one broad internal MoMo cap work across differently scaled and
conditioned objectives, or is the cap merely a hidden learning rate?
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from kaon import Adakaon
from kaon._momo_adakaon import MomoAdakaon


@dataclass
class Problem:
    name: str
    initial: torch.Tensor
    loss: Callable[[torch.Tensor, int], torch.Tensor]


def _problems(seed: int) -> list[Problem]:
    generator = torch.Generator().manual_seed(seed)
    shape = (4, 8)
    target = torch.randn(shape, generator=generator)
    spectrum = torch.logspace(0.0, 4.0, target.numel()).reshape(shape)
    q, _ = torch.linalg.qr(torch.randn(target.numel(), target.numel(), generator=generator))
    rotated_spectrum = torch.logspace(0.0, 3.0, target.numel())

    def isotropic(param: torch.Tensor, step: int) -> torch.Tensor:
        del step
        return 0.5 * (param - target).square().mean()

    def conditioned(param: torch.Tensor, step: int) -> torch.Tensor:
        del step
        return 0.5 * (spectrum * (param - target).square()).mean()

    def rotated(param: torch.Tensor, step: int) -> torch.Tensor:
        del step
        delta = (param - target).flatten()
        projected = q.T @ delta
        return 0.5 * (rotated_spectrum * projected.square()).mean()

    def quartic(param: torch.Tensor, step: int) -> torch.Tensor:
        del step
        delta = param - target
        return (0.25 * delta.pow(4) + 0.05 * delta.square()).mean()

    # A deterministic minibatch least-squares stream.  Every optimizer arm sees
    # the same cyclic samples, so comparisons are paired rather than statistical.
    features = torch.randn(64, target.numel(), generator=generator)
    features *= torch.logspace(0.0, 2.0, target.numel())
    labels = features @ target.flatten() + 0.01 * torch.randn(64, generator=generator)

    def least_squares(param: torch.Tensor, step: int) -> torch.Tensor:
        start = (step * 8) % 64
        indices = torch.arange(start, start + 8) % 64
        residual = features[indices] @ param.flatten() - labels[indices]
        return 0.5 * residual.square().mean()

    initial = torch.zeros(shape)
    return [
        Problem("isotropic", initial, isotropic),
        Problem("conditioned_1e4", initial, conditioned),
        Problem("rotated_1e3", initial, rotated),
        Problem("quartic", initial, quartic),
        Problem("least_squares_clipped", initial, least_squares),
    ]


def _run_momo(
    problem: Problem,
    *,
    beta1: float,
    cap: float,
    steps: int,
) -> dict[str, float | bool]:
    param = torch.nn.Parameter(problem.initial.clone())
    opt = MomoAdakaon([param], cap=cap, betas=(beta1, 0.999))
    initial_loss = float(problem.loss(param, 0).detach())
    maximum_loss = initial_loss
    cap_active = 0
    for step in range(steps):
        opt.zero_grad()
        loss = problem.loss(param, step)
        loss.backward()
        if problem.name.endswith("clipped"):
            torch.nn.utils.clip_grad_norm_([param], 1.0)
        try:
            opt.step(loss=loss)
        except (FloatingPointError, ValueError):
            return {
                "finite": False,
                "initial": initial_loss,
                "final": math.inf,
                "damage": math.inf,
                "cap_active_fraction": cap_active / max(step + 1, 1),
            }
        maximum_loss = max(maximum_loss, float(loss.detach()))
        cap_active += int(opt._momo_last.cap_active)
        if not bool(torch.isfinite(param).all()):
            return {"finite": False, "initial": initial_loss, "final": math.inf}
    final_loss = float(problem.loss(param, steps).detach())
    return {
        "finite": math.isfinite(final_loss),
        "initial": initial_loss,
        "final": final_loss,
        "damage": maximum_loss / max(initial_loss, 1e-30),
        "cap_active_fraction": cap_active / steps,
    }


def _run_fixed(
    problem: Problem,
    *,
    beta1: float,
    lr: float,
    steps: int,
) -> float:
    param = torch.nn.Parameter(problem.initial.clone())
    opt = Adakaon(
        [param],
        lr=lr,
        betas=(beta1, 0.999),
        momentum_dtype="float32",
        cautious=False,
        gradient_centralization=False,
        bf16_method="none",
        foreach=False,
    )
    for step in range(steps):
        opt.zero_grad()
        loss = problem.loss(param, step)
        loss.backward()
        if problem.name.endswith("clipped"):
            torch.nn.utils.clip_grad_norm_([param], 1.0)
        opt.step()
        if not bool(torch.isfinite(param).all()):
            return math.inf
    return float(problem.loss(param, steps).detach())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--caps", type=float, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    default_caps = [
        1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3,
        1e-2, 3e-2, 1e-1, 3e-1, 1.0, math.inf,
    ]
    caps = default_caps if args.caps is None else args.caps
    fixed_lrs = copy.copy(default_caps[:-1])
    records: list[dict[str, object]] = []

    for seed in args.seeds:
        for problem in _problems(seed):
            for beta1 in (0.0, 0.9):
                oracle = min(
                    _run_fixed(problem, beta1=beta1, lr=lr, steps=args.steps)
                    for lr in fixed_lrs
                )
                for cap in caps:
                    run = _run_momo(
                        problem,
                        beta1=beta1,
                        cap=cap,
                        steps=args.steps,
                    )
                    initial = float(run["initial"])
                    final = float(run["final"])
                    epsilon = 1e-12 * max(initial, 1.0)
                    oracle_error = max(oracle, epsilon)
                    auto_error = max(final, epsilon)
                    denominator = math.log(max(initial, epsilon) / oracle_error)
                    if not bool(run["finite"]):
                        progress = -math.inf
                    else:
                        progress = (
                            math.log(max(initial, epsilon) / auto_error) / denominator
                            if denominator > math.log(2.0)
                            else 1.0
                        )
                    passed = (
                        bool(run["finite"])
                        and float(run["damage"]) <= 4.0
                        and progress >= 0.75
                        and final <= max(2.0 * oracle, 1e-8 * initial)
                    )
                    records.append(
                        {
                            "seed": seed,
                            "problem": problem.name,
                            "beta1": beta1,
                            "cap": cap,
                            "oracle": oracle,
                            "progress": progress,
                            "pass": passed,
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
                "mean_cap_active_fraction": sum(
                    float(record["cap_active_fraction"]) for record in selected
                )
                / len(selected),
            }
        )
    output = {"steps": args.steps, "seeds": args.seeds, "summary": summary, "records": records}
    rendered = json.dumps(output, indent=2)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
