"""Reproducible AutoLR release battery.

Runs three deterministic seeds for Adakaon with and without momentum, compares
an AutoLR ``d0`` sweep against a fixed-LR grid and KProdigy, and writes JSON.
Run this file once from the candidate checkout and once with ``PYTHONPATH``
pointing at a 0.7.3 checkout to obtain a directly comparable legacy result.

Example::

    python benchmarks/auto_lr_battery.py --device cuda --output current.json
    PYTHONPATH=/tmp/kaon-073/src python benchmarks/auto_lr_battery.py \
        --device cuda --output legacy.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch
from torch import nn

from kaon import Adakaon, KProdigy, __version__

SEEDS = (17, 29, 43)
FIXED_LRS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
D0S = (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1.0)


def _problem(seed: int, device: torch.device) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    """Small ill-conditioned regression problem with frozen, deterministic data."""
    cpu = torch.Generator().manual_seed(seed)
    features = torch.randn(192, 24, generator=cpu)
    # Keep the optimum in Kaon's intended fine-tuning regime (~1e-3 to 1e-2).
    # A previous version used scales down to 0.1, which made target coefficients
    # as large as 10 and turned this into a large-displacement pretraining task.
    scales = torch.logspace(1.0, 2.0, features.shape[1])
    features.mul_(scales)
    target_w = torch.randn(24, 6, generator=cpu) / scales[:, None]
    targets = features @ target_w + 0.01 * torch.randn(192, 6, generator=cpu)
    torch.manual_seed(seed + 1000)
    model = nn.Linear(24, 6, bias=True).to(device)
    return model, features.to(device), targets.to(device)


def _snapshot_bytes(opt: torch.optim.Optimizer) -> int:
    tuner = getattr(opt, "_autolr", None)
    if tuner is None:
        return 0
    return sum(ref.numel() * ref.element_size() for ref in tuner._x0.values())


def _run(
    seed: int,
    device: torch.device,
    make_opt: Any,
    *,
    steps: int,
) -> dict[str, Any]:
    model, features, targets = _problem(seed, device)
    optimizer = make_opt(model.parameters())
    peak_snapshot = 0
    for step in range(steps):
        # Fixed cyclic minibatches make every optimizer see identical examples.
        start = (step * 24) % len(features)
        indices = torch.arange(start, start + 24, device=device) % len(features)
        loss = torch.nn.functional.mse_loss(model(features[indices]), targets[indices])
        if not torch.isfinite(loss):
            return {"loss": float("inf"), "finite": False, "step": step}
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        peak_snapshot = max(peak_snapshot, _snapshot_bytes(optimizer))

    with torch.no_grad():
        final = float(torch.nn.functional.mse_loss(model(features), targets))
    result: dict[str, Any] = {
        "loss": final,
        "finite": math.isfinite(final),
        "peak_snapshot_bytes": peak_snapshot,
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
    }
    if hasattr(optimizer, "get_d"):
        result["lr"] = float(optimizer.get_d())
    if getattr(optimizer, "_autolr", None) is not None:
        tuner = optimizer._autolr
        result.update(
            frozen=bool(optimizer.is_frozen()),
            freeze_step=int(tuner._t),
            freeze_reason=getattr(tuner, "freeze_reason", None),
        )
    return result


def run_battery(device: torch.device, steps: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for momentum in (False, True):
        betas = (0.9, 0.999) if momentum else (0.0, 0.999)
        for seed in SEEDS:
            fixed: dict[str, Any] = {}
            for lr in FIXED_LRS:
                fixed[f"{lr:g}"] = _run(
                    seed,
                    device,
                    lambda params, lr=lr, betas=betas, momentum=momentum: Adakaon(
                        params,
                        lr=lr,
                        betas=betas,
                        cautious=momentum,
                        momentum_dtype="float32",
                    ),
                    steps=steps,
                )
            best_lr, best = min(fixed.items(), key=lambda item: item[1]["loss"])
            kprodigy = _run(
                seed,
                device,
                lambda params, betas=betas, momentum=momentum: KProdigy(
                    params,
                    lr=1.0,
                    betas=betas,
                    cautious=momentum,
                    momentum_dtype="float32",
                ),
                steps=steps,
            )
            auto: dict[str, Any] = {}
            for d0 in D0S:
                auto[f"{d0:g}"] = _run(
                    seed,
                    device,
                    lambda params, d0=d0, betas=betas, momentum=momentum: Adakaon(
                        params,
                        lr=1.0,
                        betas=betas,
                        cautious=momentum,
                        momentum_dtype="float32",
                        auto_lr=True,
                        auto_lr_d0=d0,
                    ),
                    steps=steps,
                )
            rows.append(
                {
                    "momentum": momentum,
                    "seed": seed,
                    "fixed": fixed,
                    "fixed_best_lr": float(best_lr),
                    "fixed_best_loss": best["loss"],
                    "kprodigy": kprodigy,
                    "auto": auto,
                }
            )

    auto_losses = [arm["loss"] for row in rows for arm in row["auto"].values()]
    return {
        "kaon_version": __version__,
        "device": str(device),
        "steps": steps,
        "seeds": list(SEEDS),
        "fixed_lrs": list(FIXED_LRS),
        "d0s": list(D0S),
        "summary": {
            "auto_median_loss": statistics.median(auto_losses),
            "all_finite": all(math.isfinite(loss) for loss in auto_losses),
            "all_frozen_by_budget": all(
                arm.get("frozen", False) and arm.get("freeze_step", steps + 1) <= 192
                for row in rows
                for arm in row["auto"].values()
            ),
            "snapshot_within_one_copy": all(
                arm["peak_snapshot_bytes"] <= arm["parameter_bytes"]
                for row in rows
                for arm in row["auto"].values()
            ),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--compare-legacy",
        type=Path,
        help="0.7.3 JSON from the same battery; enforce no AutoLR arm regresses by >5%%",
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="record results without candidate safety gates (use for the 0.7.3 baseline)",
    )
    args = parser.parse_args()
    result = run_battery(torch.device(args.device), args.steps)
    legacy_ok = True
    if args.compare_legacy:
        legacy = json.loads(args.compare_legacy.read_text(encoding="utf-8"))
        legacy_rows = {
            (row["momentum"], row["seed"]): row for row in legacy["rows"]
        }
        ratios: dict[str, float] = {}
        for row in result["rows"]:
            old = legacy_rows[(row["momentum"], row["seed"])]
            for d0, arm in row["auto"].items():
                old_loss = old["auto"][d0]["loss"]
                ratio = arm["loss"] / old_loss if old_loss > 0.0 else float("inf")
                ratios[f"momentum={row['momentum']},seed={row['seed']},d0={d0}"] = ratio
        worst_name, worst_ratio = max(ratios.items(), key=lambda item: item[1])
        legacy_ok = worst_ratio <= 1.05
        result["legacy_comparison"] = {
            "max_loss_ratio": worst_ratio,
            "worst_arm": worst_name,
            "passes_5_percent": legacy_ok,
            "ratios": ratios,
        }
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not args.no_gate and (not all(result["summary"].values()) or not legacy_ok):
        raise SystemExit("AutoLR release battery failed a safety invariant")


if __name__ == "__main__":
    main()
