"""Passive lagged-alignment battery for Adakaon.

This is an observability experiment, not an AutoLR experiment.  Every arm uses a
fixed LR and the telemetry hook is read-only: it must not alter gradients, optimizer
state, parameters, or the LR.  Arms share initialization, minibatches, DDPM noise,
and held-out evaluation draws.

The deliberately small 32x32/C=8 proxy makes it cheap to answer a narrow question:
does the normalized alignment between the current gradient and the previous learned
direction predict held-out loss or transient damage, including when a low global
gradient clip hides gradient-norm growth?

Examples::

    python benchmarks/alignment_telemetry_battery.py --quick --device cpu
    python benchmarks/alignment_telemetry_battery.py --device cuda --output alignment.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from kaon import Adakaon, __version__

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_LRS = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
DEFAULT_BETA1S = (0.0, 0.9)
DEFAULT_CLIP_NORM = 0.1


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load benchmark helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = _load("alignment_telemetry_harness", REPO / "benchmarks/proxy/harness.py")
D = _load("alignment_telemetry_dataset", REPO / "benchmarks/proxy/dataset.py")


def safe_cosine(dot: float, lhs_norm_sq: float, rhs_norm_sq: float) -> float | None:
    """Return a finite normalized dot product, or None for an unobservable pair."""
    values = (dot, lhs_norm_sq, rhs_norm_sq)
    if not all(math.isfinite(value) for value in values):
        return None
    denom_sq = lhs_norm_sq * rhs_norm_sq
    if denom_sq <= 0.0:
        return None
    # Reduction roundoff can exceed the mathematical range by a few ulps.
    return max(-1.0, min(1.0, dot / math.sqrt(denom_sq)))


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_alignments(values: list[float | None]) -> dict[str, float | int | None]:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return {
        "observations": len(finite),
        "mean": mean(finite),
        "minimum": min(finite) if finite else None,
        "negative_fraction": (
            sum(value < 0.0 for value in finite) / len(finite) if finite else None
        ),
        "severe_negative_fraction": (
            sum(value < -0.25 for value in finite) / len(finite) if finite else None
        ),
    }


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Small dependency-free Pearson correlation used by the JSON aggregation."""
    if len(xs) != len(ys):
        raise ValueError("correlation inputs must have equal length")
    if len(xs) < 2:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    dx = [value - mx for value in xs]
    dy = [value - my for value in ys]
    denom = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    if denom == 0.0:
        return None
    return sum(x * y for x, y in zip(dx, dy, strict=True)) / denom


def _ranks(values: list[float]) -> list[float]:
    """Average ranks for ties, matching the convention used by Spearman."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for position in order[start:end]:
            ranks[position] = rank
        start = end
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys):
        raise ValueError("correlation inputs must have equal length")
    return pearson(_ranks(xs), _ranks(ys))


def correlation_summary(
    rows: list[dict[str, Any]], alignment_key: str = "alignment"
) -> dict[str, Any]:
    """Correlate passive alignment summaries with quality and damage."""
    features = ("mean", "minimum", "negative_fraction", "severe_negative_fraction")
    targets = ("heldout_loss", "damage")
    result: dict[str, Any] = {"arms": len(rows), "correlations": {}}
    for feature in features:
        for target in targets:
            pairs = [
                (row[alignment_key][feature], row[target])
                for row in rows
                if row[alignment_key][feature] is not None and row[target] is not None
            ]
            xs = [float(pair[0]) for pair in pairs]
            ys = [float(pair[1]) for pair in pairs]
            result["correlations"][f"{feature}_vs_{target}"] = {
                "n": len(pairs),
                "pearson": pearson(xs, ys),
                "spearman": spearman(xs, ys),
            }
    return result


@dataclass
class ArmResult:
    seed: int
    lr: float
    beta1: float
    clipping: str
    clip_norm: float | None
    finite: bool
    initial_heldout: float
    heldout_loss: float | None
    best_heldout: float | None
    damage: float | None
    clipped_fraction: float
    alignment: dict[str, float | int | None]
    anchor_alignment: dict[str, float | int | None]
    telemetry: list[dict[str, float | int | bool | None]]
    trajectory: list[dict[str, float | int | None]]


def _telemetry_row(sample: Any) -> dict[str, float | int | bool | None]:
    lagged = safe_cosine(
        float(sample.grad_prev_direction_dot),
        float(sample.grad_norm_sq),
        float(sample.prev_direction_norm_sq),
    )
    current = safe_cosine(
        float(sample.grad_direction_dot),
        float(sample.grad_norm_sq),
        float(sample.direction_norm_sq),
    )
    return {
        "step": int(sample.step),
        "active_numel": int(sample.active_numel),
        "finite": bool(sample.finite),
        "grad_norm": math.sqrt(max(0.0, float(sample.grad_norm_sq))),
        "direction_norm": math.sqrt(max(0.0, float(sample.direction_norm_sq))),
        "lagged_alignment": lagged,
        "current_alignment": current,
    }


def _damage(observed: list[float]) -> float:
    running_best = observed[0]
    damage = 0.0
    for value in observed[1:]:
        running_best = min(running_best, value)
        damage = max(damage, value / max(running_best, 1e-12) - 1.0)
    return damage


def _anchor_signal(params: list[torch.Tensor], anchors: list[torch.Tensor]) -> dict[str, float | None]:
    """Mechanic's passive signal using the live displacement from the initial point.

    For Mechanic's accumulated positive descent direction, fixed positive LR gives
    ``x_t - x_0 = -lr * Delta_t``. Thus this cosine has the opposite sign of
    Mechanic's ``<g_t, Delta_t>`` while avoiding reconstruction assumptions.
    """
    grad_sq = 0.0
    displacement_sq = 0.0
    dot = 0.0
    for param, anchor in zip(params, anchors, strict=True):
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        displacement = param.detach().float() - anchor
        grad_sq += float(grad.square().sum())
        displacement_sq += float(displacement.square().sum())
        dot += float((grad * displacement).sum())
    return {
        "anchor_displacement_dot": dot,
        "anchor_displacement_cosine": safe_cosine(dot, grad_sq, displacement_sq),
    }


def run_arm(
    *,
    seed: int,
    lr: float,
    beta1: float,
    clipping: str,
    clip_norm: float,
    steps: int,
    channels: int,
    batch_size: int,
    checkpoints: int,
    eval_reps: int,
    device: torch.device,
) -> ArmResult:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    data_bundle = D.build_proxy_dataset(seed=7)
    data = {resolution: value.to(device) for resolution, value in data_bundle["DATA"].items()}
    train_idx, test_idx = data_bundle["TR"], data_bundle["TE"]
    old_dev = H.DEV
    H.DEV = str(device)
    try:
        alphas = H.make_alphas()
        net = H.UNet(C=channels).to(device).to(H.DT)
        opt = Adakaon(
            net.parameters(),
            lr=lr,
            betas=(beta1, 0.999),
            cautious=False,
            momentum_dtype="float32",
            gradient_centralization=False,
        )
        hook = getattr(opt, "_set_step_telemetry_hook", None)
        if hook is None:
            raise RuntimeError("this checkout does not expose Adakaon's passive telemetry hook")
        telemetry: list[dict[str, float | int | bool | None]] = []
        hook(lambda sample: telemetry.append(_telemetry_row(sample)))
        params = [param for param in net.parameters() if param.requires_grad]
        anchors = [param.detach().float().clone() for param in params]
        anchor_alignments: list[float | None] = []

        initial = H.eval_loss(net, data[32], test_idx, alphas, reps=eval_reps)
        observed = [initial]
        trajectory: list[dict[str, float | int | None]] = [
            {"step": 0, "heldout_loss": initial}
        ]
        generator = torch.Generator(device=device).manual_seed(seed + 12345)
        every = max(1, steps // checkpoints)
        position = 0
        clipped_steps = 0
        finite = True
        for step in range(steps):
            indices = [train_idx[(position + j) % len(train_idx)] for j in range(batch_size)]
            position += batch_size
            opt.zero_grad(set_to_none=True)
            loss = H.batch_loss(
                net,
                data[32],
                torch.tensor(indices, device=device),
                alphas,
                generator,
            )
            if not bool(torch.isfinite(loss)):
                finite = False
                break
            loss.backward()
            if clipping == "global":
                total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), clip_norm)
                clipped_steps += int(float(total_norm) > clip_norm)
            elif clipping != "none":
                raise ValueError(f"unknown clipping mode: {clipping}")
            anchor = _anchor_signal(params, anchors)
            anchor_alignments.append(anchor["anchor_displacement_cosine"])
            opt.step()
            telemetry[-1].update(anchor)
            if not all(bool(torch.isfinite(param).all()) for param in net.parameters()):
                finite = False
                break
            if (step + 1) % every == 0 or step + 1 == steps:
                heldout = H.eval_loss(net, data[32], test_idx, alphas, reps=eval_reps)
                if not math.isfinite(heldout):
                    finite = False
                    break
                observed.append(heldout)
                trajectory.append({"step": step + 1, "heldout_loss": heldout})
        final = observed[-1] if finite else None
        alignments = [row["lagged_alignment"] for row in telemetry]
        return ArmResult(
            seed=seed,
            lr=lr,
            beta1=beta1,
            clipping=clipping,
            clip_norm=clip_norm if clipping == "global" else None,
            finite=finite and all(bool(row["finite"]) for row in telemetry),
            initial_heldout=initial,
            heldout_loss=final,
            best_heldout=min(observed) if observed else None,
            damage=_damage(observed) if finite else None,
            clipped_fraction=clipped_steps / max(1, len(telemetry)),
            alignment=summarize_alignments(alignments),
            anchor_alignment=summarize_alignments(anchor_alignments),
            telemetry=telemetry,
            trajectory=trajectory,
        )
    finally:
        H.DEV = old_dev


def run_battery(
    *,
    seeds: tuple[int, ...],
    lrs: tuple[float, ...],
    beta1s: tuple[float, ...],
    steps: int,
    channels: int,
    batch_size: int,
    checkpoints: int,
    eval_reps: int,
    clip_norm: float,
    device: torch.device,
) -> dict[str, Any]:
    results = [
        run_arm(
            seed=seed,
            lr=lr,
            beta1=beta1,
            clipping=clipping,
            clip_norm=clip_norm,
            steps=steps,
            channels=channels,
            batch_size=batch_size,
            checkpoints=checkpoints,
            eval_reps=eval_reps,
            device=device,
        )
        for seed in seeds
        for beta1 in beta1s
        for clipping in ("none", "global")
        for lr in lrs
    ]
    rows = [asdict(result) for result in results]
    strata = {}
    for beta1 in beta1s:
        for clipping in ("none", "global"):
            members = [
                row
                for row in rows
                if row["beta1"] == beta1 and row["clipping"] == clipping
            ]
            strata[f"beta1={beta1:g},clipping={clipping}"] = {
                "lagged": correlation_summary(members),
                "anchor": correlation_summary(members, "anchor_alignment"),
            }
    return {
        "kaon_version": __version__,
        "device": str(device),
        "passive": True,
        "resolution": 32,
        "channels": channels,
        "steps": steps,
        "seeds": list(seeds),
        "lrs": list(lrs),
        "beta1s": list(beta1s),
        "clip_norm": clip_norm,
        "summary": {
            "all_finite": all(row["finite"] for row in rows),
            "global": {
                "lagged": correlation_summary(rows),
                "anchor": correlation_summary(rows, "anchor_alignment"),
            },
            "strata": strata,
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17])
    parser.add_argument("--lrs", type=float, nargs="+", default=list(DEFAULT_LRS))
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoints", type=int, default=8)
    parser.add_argument("--eval-reps", type=int, default=2)
    parser.add_argument("--clip-norm", type=float, default=DEFAULT_CLIP_NORM)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="print only the aggregate summary (the output file still contains raw telemetry)",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.steps = min(args.steps, 12)
        args.lrs = [1e-4, 1e-3, 1e-2]
        args.checkpoints = min(args.checkpoints, 3)
        args.eval_reps = 1
    if args.steps <= 0 or args.channels <= 0 or args.batch_size <= 0:
        parser.error("steps, channels, and batch-size must be positive")
    if args.clip_norm <= 0.0:
        parser.error("clip-norm must be positive")
    if not args.lrs or not all(math.isfinite(lr) and lr > 0.0 for lr in args.lrs):
        parser.error("all LRs must be finite and positive")
    result = run_battery(
        seeds=tuple(args.seeds),
        lrs=tuple(args.lrs),
        beta1s=DEFAULT_BETA1S,
        steps=args.steps,
        channels=args.channels,
        batch_size=args.batch_size,
        checkpoints=args.checkpoints,
        eval_reps=args.eval_reps,
        clip_norm=args.clip_norm,
        device=torch.device(args.device),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(
        json.dumps(result["summary"], indent=2, sort_keys=True, allow_nan=False)
        if args.summary_only
        else rendered
    )


if __name__ == "__main__":
    main()
