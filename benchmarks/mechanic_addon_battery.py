"""Active Mechanic-addon battery on Kaon's low-resolution DDPM proxy.

The five paired arms are fixed-LR Adakaon (low/oracle/high), faithful Mechanic,
and Mechanic with the directional guard.  Every arm in a stratum starts from the
same model and observes identical minibatches, diffusion timesteps, and noise.

The addon is intentionally imported only when a Mechanic arm is constructed, so
the pure aggregation tests can run before ``kaon._mechanic_addon`` lands.  This
benchmark does not implement or approximate Mechanic itself.

Examples::

    python benchmarks/mechanic_addon_battery.py --quick --device cpu
    python benchmarks/mechanic_addon_battery.py --device cuda --output mechanic.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import torch

from kaon import Adakaon, __version__

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_SEEDS = (17, 29, 43)
DEFAULT_BETA1S = (0.0, 0.9)
ARMS = ("fixed_low", "fixed_oracle", "fixed_high", "mechanic", "mechanic_guard")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load benchmark helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = _load("mechanic_addon_harness", REPO / "benchmarks/proxy/harness.py")
D = _load("mechanic_addon_dataset", REPO / "benchmarks/proxy/dataset.py")


def fixed_lr(arm: str, oracle_lr: float, lr_ratio: float) -> float | None:
    if arm == "fixed_low":
        return oracle_lr / lr_ratio
    if arm == "fixed_oracle":
        return oracle_lr
    if arm == "fixed_high":
        return oracle_lr * lr_ratio
    if arm in {"mechanic", "mechanic_guard"}:
        return None
    raise ValueError(f"unknown arm: {arm}")


def _finite_float(value: Any) -> float | int | bool | str | None:
    if isinstance(value, bool | str) or value is None:
        return value
    if isinstance(value, int):
        return value
    try:
        result = float(value)
    except (TypeError, ValueError):
        return str(value)
    return result if math.isfinite(result) else None


def mechanic_stats(stats: Any) -> dict[str, float | int | bool | str | None]:
    """Normalize the addon's deliberately small diagnostic record for JSON."""
    if stats is None:
        return {}
    if isinstance(stats, dict):
        raw = stats
    elif is_dataclass(stats):
        raw = asdict(stats)
    else:
        raw = {
            name: getattr(stats, name)
            for name in ("scale", "h", "guarded")
            if hasattr(stats, name)
        }
    return {str(key): _finite_float(value) for key, value in raw.items()}


def transient_damage(observed: list[float]) -> float:
    """Largest held-out rebound relative to the best value already reached."""
    if not observed:
        raise ValueError("damage needs at least one observation")
    running_best = observed[0]
    damage = 0.0
    for value in observed[1:]:
        running_best = min(running_best, value)
        damage = max(damage, value / max(running_best, 1e-12) - 1.0)
    return damage


def first_step_at_quality(
    trajectory: list[dict[str, float | int | None]], target: float
) -> int | None:
    for point in trajectory:
        loss = point.get("heldout_loss")
        if loss is not None and float(loss) <= target:
            return int(point["step"])
    return None


def annotate_time_to_quality(
    rows: list[dict[str, Any]], *, tolerance: float = 0.02
) -> list[dict[str, Any]]:
    """Use each stratum's fixed-oracle final held-out loss as its causal target."""
    if tolerance < 0.0:
        raise ValueError("quality tolerance must be non-negative")
    oracle_by_stratum = {
        (row["seed"], row["beta1"], row["clipping"]): row
        for row in rows
        if row["arm"] == "fixed_oracle"
    }
    annotated = []
    for row in rows:
        result = dict(row)
        oracle = oracle_by_stratum.get((row["seed"], row["beta1"], row["clipping"]))
        oracle_loss = None if oracle is None else oracle["heldout_loss"]
        target = None if oracle_loss is None else float(oracle_loss) * (1.0 + tolerance)
        result["quality_target"] = target
        result["time_to_quality"] = (
            None if target is None else first_step_at_quality(row["trajectory"], target)
        )
        annotated.append(result)
    return annotated


@dataclass
class ArmResult:
    arm: str
    seed: int
    beta1: float
    clipping: str
    configured_lr: float | None
    finite: bool
    initial_heldout: float
    heldout_loss: float | None
    best_heldout: float | None
    damage: float | None
    clipped_fraction: float
    trajectory: list[dict[str, Any]]


def make_optimizer(arm: str, params, *, beta1: float, lr: float | None):
    inner_lr = 1.0 if arm in {"mechanic", "mechanic_guard"} else lr
    if inner_lr is None:
        raise ValueError("fixed Adakaon arm requires an LR")
    inner = Adakaon(
        params,
        lr=inner_lr,
        betas=(beta1, 0.999),
        weight_decay=0.0,
        cautious=False,
        fused=False,
        momentum_dtype="float32",
        gradient_centralization=False,
    )
    if arm.startswith("fixed_"):
        return inner
    from kaon._mechanic_addon import MechanicAddon

    return MechanicAddon(inner, guard=arm == "mechanic_guard")


def run_arm(
    *,
    arm: str,
    seed: int,
    beta1: float,
    clipping: str,
    configured_lr: float | None,
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
    bundle = D.build_proxy_dataset(seed=7)
    data = {resolution: value.to(device) for resolution, value in bundle["DATA"].items()}
    train_idx, test_idx = bundle["TR"], bundle["TE"]
    old_dev = H.DEV
    H.DEV = str(device)
    try:
        alphas = H.make_alphas()
        net = H.UNet(C=channels).to(device).to(H.DT)
        optimizer = make_optimizer(
            arm,
            net.parameters(),
            beta1=beta1,
            lr=configured_lr,
        )
        initial = H.eval_loss(net, data[32], test_idx, alphas, reps=eval_reps)
        heldout_observed = [initial]
        trajectory: list[dict[str, Any]] = [
            {
                "step": 0,
                "train_loss": None,
                "heldout_loss": initial,
                "scale": configured_lr if arm.startswith("fixed_") else float(optimizer.get_scale()),
                "stats": {},
            }
        ]
        generator = torch.Generator(device=device).manual_seed(seed + 12345)
        every = max(1, steps // checkpoints)
        position = 0
        clipped_steps = 0
        completed = 0
        finite = True
        for step in range(steps):
            indices = [train_idx[(position + offset) % len(train_idx)] for offset in range(batch_size)]
            position += batch_size
            optimizer.zero_grad(set_to_none=True)
            loss = H.batch_loss(
                net,
                data[32],
                torch.tensor(indices, device=device),
                alphas,
                generator,
            )
            train_loss = float(loss.detach())
            if not math.isfinite(train_loss):
                finite = False
                break
            loss.backward()
            if clipping == "global":
                total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), clip_norm)
                clipped_steps += int(float(total_norm) > clip_norm)
            elif clipping != "none":
                raise ValueError(f"unknown clipping mode: {clipping}")
            optimizer.step()
            completed += 1
            scale = (
                configured_lr
                if arm.startswith("fixed_")
                else float(optimizer.get_scale())
            )
            finite = (
                math.isfinite(scale)
                and all(bool(torch.isfinite(param).all()) for param in net.parameters())
            )
            point: dict[str, Any] = {
                "step": step + 1,
                "train_loss": train_loss,
                "heldout_loss": None,
                "scale": scale if math.isfinite(scale) else None,
                "stats": (
                    {} if arm.startswith("fixed_") else mechanic_stats(optimizer.last_stats)
                ),
            }
            if finite and ((step + 1) % every == 0 or step + 1 == steps):
                heldout = H.eval_loss(net, data[32], test_idx, alphas, reps=eval_reps)
                if math.isfinite(heldout):
                    point["heldout_loss"] = heldout
                    heldout_observed.append(heldout)
                else:
                    finite = False
            trajectory.append(point)
            if not finite:
                break
        final = heldout_observed[-1] if finite and completed == steps else None
        return ArmResult(
            arm=arm,
            seed=seed,
            beta1=beta1,
            clipping=clipping,
            configured_lr=configured_lr,
            finite=finite and completed == steps,
            initial_heldout=initial,
            heldout_loss=final,
            best_heldout=min(heldout_observed),
            damage=transient_damage(heldout_observed),
            clipped_fraction=clipped_steps / max(1, completed),
            trajectory=trajectory,
        )
    finally:
        H.DEV = old_dev


def run_battery(
    *,
    seeds: tuple[int, ...],
    beta1s: tuple[float, ...],
    clipping_modes: tuple[str, ...],
    arms: tuple[str, ...],
    oracle_lr: float,
    lr_ratio: float,
    clip_norm: float,
    steps: int,
    channels: int,
    batch_size: int,
    checkpoints: int,
    eval_reps: int,
    quality_tolerance: float,
    device: torch.device,
) -> dict[str, Any]:
    raw_rows = [
        asdict(
            run_arm(
                arm=arm,
                seed=seed,
                beta1=beta1,
                clipping=clipping,
                configured_lr=fixed_lr(arm, oracle_lr, lr_ratio),
                clip_norm=clip_norm,
                steps=steps,
                channels=channels,
                batch_size=batch_size,
                checkpoints=checkpoints,
                eval_reps=eval_reps,
                device=device,
            )
        )
        for seed in seeds
        for beta1 in beta1s
        for clipping in clipping_modes
        for arm in arms
    ]
    rows = annotate_time_to_quality(raw_rows, tolerance=quality_tolerance)
    by_arm = {}
    for arm in arms:
        members = [row for row in rows if row["arm"] == arm]
        finite_members = [row for row in members if row["heldout_loss"] is not None]
        reached = [row["time_to_quality"] for row in members if row["time_to_quality"] is not None]
        by_arm[arm] = {
            "runs": len(members),
            "finite_fraction": sum(row["finite"] for row in members) / max(1, len(members)),
            "mean_heldout_loss": (
                sum(row["heldout_loss"] for row in finite_members) / len(finite_members)
                if finite_members
                else None
            ),
            "max_damage": max((row["damage"] for row in members), default=None),
            "quality_reached_fraction": len(reached) / max(1, len(members)),
            "mean_time_to_quality": sum(reached) / len(reached) if reached else None,
        }
    return {
        "kaon_version": __version__,
        "device": str(device),
        "resolution": 32,
        "channels": channels,
        "steps": steps,
        "seeds": list(seeds),
        "beta1s": list(beta1s),
        "arms": list(arms),
        "clipping": {"modes": list(clipping_modes), "global_max_norm": clip_norm},
        "fixed_lrs": {
            "low": oracle_lr / lr_ratio,
            "oracle": oracle_lr,
            "high": oracle_lr * lr_ratio,
        },
        "quality_tolerance": quality_tolerance,
        "summary": {"by_arm": by_arm},
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--beta1s", type=float, nargs="+", default=list(DEFAULT_BETA1S))
    parser.add_argument("--clipping", nargs="+", choices=("none", "global"), default=("none", "global"))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    parser.add_argument("--oracle-lr", type=float, default=1e-3)
    parser.add_argument("--lr-ratio", type=float, default=10.0)
    parser.add_argument("--clip-norm", type=float, default=0.1)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoints", type=int, default=8)
    parser.add_argument("--eval-reps", type=int, default=2)
    parser.add_argument("--quality-tolerance", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="print only aggregate metrics while retaining full trajectories in --output",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.steps = min(args.steps, 12)
        args.seeds = args.seeds[:1]
        args.checkpoints = min(args.checkpoints, 3)
        args.eval_reps = 1
    positive = (
        args.steps,
        args.oracle_lr,
        args.lr_ratio,
        args.clip_norm,
        args.channels,
        args.batch_size,
        args.checkpoints,
        args.eval_reps,
    )
    if not all(math.isfinite(float(value)) and value > 0 for value in positive):
        parser.error("steps and all scale/count arguments must be finite and positive")
    if args.lr_ratio <= 1.0:
        parser.error("lr-ratio must be greater than one")
    result = run_battery(
        seeds=tuple(args.seeds),
        beta1s=tuple(args.beta1s),
        clipping_modes=tuple(args.clipping),
        arms=tuple(args.arms),
        oracle_lr=args.oracle_lr,
        lr_ratio=args.lr_ratio,
        clip_norm=args.clip_norm,
        steps=args.steps,
        channels=args.channels,
        batch_size=args.batch_size,
        checkpoints=args.checkpoints,
        eval_reps=args.eval_reps,
        quality_tolerance=args.quality_tolerance,
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
