"""Paired Adakaon-vs-Nekaon battery.

This benchmark answers one deliberately narrow question: does Nekaon's negative
momentum lookahead still buy anything once the underlying Adakaon core receives the
same learning-rate trajectory?  Unlike the broad control battery, every pair shares
initial weights, minibatches, diffusion noise, optimizer hyperparameters, and LR.
The only algorithmic difference is ``k=0`` (Adakaon) versus ``k>0`` (Nekaon).

Examples::

    python benchmarks/nekaon_relative_battery.py --quick
    python benchmarks/nekaon_relative_battery.py --steps 2000 --seeds 3
    python benchmarks/nekaon_relative_battery.py --lr-trace trace.json

An LR trace is a JSON list of positive multipliers.  It is replayed in both arms,
which is how a future autonomous scale controller must first be evaluated: controller
quality and lookahead quality are separate experiments.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from kaon import Nekaon

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = _load("nekaon_relative_harness", REPO / "benchmarks/proxy/harness.py")
D = _load("nekaon_relative_dataset", REPO / "benchmarks/proxy/dataset.py")
DEV = H.DEV


@dataclass(frozen=True)
class Result:
    arm: str
    regime: str
    seed: int
    lr: float
    initial_train: float
    train_loss: float
    heldout_loss: float
    gap: float
    best_heldout: float
    damage: float
    ms_step: float
    state_bpp: float
    trajectory: list[tuple[int, float, float]]


def rex(step: int, steps: int, d: float = 0.9) -> float:
    z = 1.0 - step / steps
    return z / ((1.0 - d) + d * z)


def lr_multipliers(regime: str, steps: int, replay: list[float] | None = None) -> list[float]:
    if regime == "replay":
        if replay is None or len(replay) < steps:
            raise ValueError("replay regime needs at least --steps positive LR multipliers")
        values = replay[:steps]
    elif regime == "rex":
        values = [rex(i, steps) for i in range(steps)]
    elif regime == "constant":
        values = [1.0] * steps
    else:
        raise ValueError(f"unknown regime: {regime}")
    if not all(math.isfinite(v) and v > 0.0 for v in values):
        raise ValueError("LR multipliers must be finite and positive")
    return values


def make_optimizer(arm: str, params, lr: float, *, beta1: float, weight_decay: float, k: float):
    common = dict(
        lr=lr,
        betas=(beta1, 0.999),
        weight_decay=weight_decay,
        cautious=True,
        momentum_dtype="bfloat16",
    )
    if arm == "adakaon":
        # Use the k=0 wrapper for the causal quality comparison: every constructor path is
        # identical and only k changes. Bare Adakaon is checked separately by unit tests.
        return Nekaon(params, k=0.0, **common)
    if arm == "nekaon":
        return Nekaon(params, k=k, **common)
    raise ValueError(f"unknown arm: {arm}")


def _eval(opt, fn):
    swaps = hasattr(opt, "eval") and hasattr(opt, "train")
    if swaps:
        opt.eval()
    value = fn()
    if swaps:
        opt.train()
    return value


def run_arm(
    arm: str,
    *,
    seed: int,
    base_lr: float,
    multipliers: list[float],
    regime: str,
    data,
    train_idx,
    test_idx,
    alphas,
    channels: int,
    batch_size: int,
    beta1: float,
    weight_decay: float,
    k: float,
    checkpoints: int,
    eval_reps: int,
) -> Result:
    """Run one arm. Pairing is obtained by calling both arms with identical arguments."""
    torch.manual_seed(seed)
    if DEV == "cuda":
        torch.cuda.manual_seed_all(seed)
    net = H.UNet(C=channels).to(DEV).to(H.DT)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = make_optimizer(
        arm, params, base_lr, beta1=beta1, weight_decay=weight_decay, k=k
    )
    generator = torch.Generator(device=DEV)
    generator.manual_seed(seed + 12345)
    sequence = ([32, 64, 48, 64] * ((len(multipliers) + 3) // 4))[: len(multipliers)]
    every = max(1, len(multipliers) // checkpoints)
    trajectory = []
    position = 0
    initial_train, initial = _eval(
        opt,
        lambda: (
            H.eval_loss(net, data[64], train_idx, alphas, reps=eval_reps),
            H.eval_loss(net, data[64], test_idx, alphas, reps=eval_reps),
        ),
    )
    warmup = min(10, max(0, len(multipliers) - 1))
    started = None
    timed = 0
    for step, (resolution, multiplier) in enumerate(zip(sequence, multipliers, strict=True)):
        if step == warmup:
            if DEV == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
        for group in opt.param_groups:
            group["lr"] = base_lr * multiplier
        indices = [train_idx[(position + j) % len(train_idx)] for j in range(batch_size)]
        position += batch_size
        opt.zero_grad()
        loss = H.batch_loss(
            net, data[resolution], torch.tensor(indices, device=DEV), alphas, generator
        )
        loss.backward()
        opt.step()
        if step >= warmup:
            timed += 1
        if (step + 1) % every == 0 or step + 1 == len(multipliers):
            train_at_step, heldout = _eval(
                opt,
                lambda: (
                    H.eval_loss(net, data[64], train_idx, alphas, reps=eval_reps),
                    H.eval_loss(net, data[64], test_idx, alphas, reps=eval_reps),
                ),
            )
            trajectory.append((step + 1, train_at_step, heldout))
    if DEV == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started if started is not None else 0.0
    train_loss, heldout_loss = _eval(
        opt,
        lambda: (
            H.eval_loss(net, data[64], train_idx, alphas, reps=eval_reps),
            H.eval_loss(net, data[64], test_idx, alphas, reps=eval_reps),
        ),
    )
    observed = [initial, *(heldout for _, _, heldout in trajectory)]
    best = min(observed)
    # Maximum loss above the best value already reached. This catches transient model damage;
    # final loss alone can hide a destructive excursion followed by partial recovery.
    running_best = observed[0]
    damage = 0.0
    for value in observed[1:]:
        running_best = min(running_best, value)
        damage = max(damage, value / max(running_best, 1e-12) - 1.0)
    return Result(
        arm=arm,
        regime=regime,
        seed=seed,
        lr=base_lr,
        initial_train=initial_train,
        train_loss=train_loss,
        heldout_loss=heldout_loss,
        gap=heldout_loss - train_loss,
        best_heldout=best,
        damage=damage,
        ms_step=elapsed / max(timed, 1) * 1000.0,
        state_bpp=H.opt_state_bytes_per_param(opt, params),
        trajectory=trajectory,
    )


def mean(values):
    return sum(values) / len(values)


def aggregate(results: list[Result]) -> list[dict]:
    groups = {}
    for result in results:
        groups.setdefault((result.arm, result.regime, result.lr), []).append(result)
    rows = []
    scalar_fields = (
        "train_loss",
        "initial_train",
        "heldout_loss",
        "gap",
        "best_heldout",
        "damage",
        "ms_step",
        "state_bpp",
    )
    for (arm, regime, lr), members in sorted(groups.items()):
        row = {"arm": arm, "regime": regime, "lr": lr, "seeds": len(members)}
        row.update({field: mean([getattr(m, field) for m in members]) for field in scalar_fields})
        rows.append(row)
    return rows


def dominates(a: dict, b: dict, *, loss_tolerance: float = 0.01, gap_tolerance: float = 0.002):
    """Return whether A makes B unnecessary on the loss/generalization plane.

    Tolerances are relative for held-out loss and absolute for the train-validation gap.
    This deliberately prevents a high-loss, underfit run from winning merely by having a
    small gap.
    """
    loss_limit = b["heldout_loss"] * (1.0 + loss_tolerance)
    a_progress = a.get("initial_train", a["train_loss"]) - a["train_loss"]
    b_progress = b.get("initial_train", b["train_loss"]) - b["train_loss"]
    fit_ratio = a_progress / max(b_progress, 1e-12)
    no_worse = (
        fit_ratio >= 0.9
        and a["heldout_loss"] <= loss_limit
        and a["gap"] <= b["gap"] + gap_tolerance
    )
    meaningfully_better = (
        a["heldout_loss"] < b["heldout_loss"] * (1.0 - loss_tolerance)
        or a["gap"] < b["gap"] - gap_tolerance
    )
    return no_worse and meaningfully_better


def decision(rows: list[dict]) -> dict:
    by_regime = {}
    for regime in sorted({row["regime"] for row in rows}):
        adakaon = [r for r in rows if r["arm"] == "adakaon" and r["regime"] == regime]
        nekaon = [r for r in rows if r["arm"] == "nekaon" and r["regime"] == regime]
        nekaon_frontier = [n for n in nekaon if not any(dominates(a, n) for a in adakaon)]
        adakaon_frontier = [a for a in adakaon if not any(dominates(n, a) for n in nekaon)]
        meaningful_nekaon_win = any(dominates(n, a) for n in nekaon for a in adakaon)
        all_nekaon_dominated = bool(nekaon) and all(
            any(dominates(a, n) for a in adakaon) for n in nekaon
        )
        if all_nekaon_dominated:
            verdict = "adakaon_supersedes_nekaon"
        elif meaningful_nekaon_win:
            verdict = "nekaon_retains_a_pareto_advantage"
        else:
            verdict = "inconclusive_frontiers_overlap"
        by_regime[regime] = {
            "verdict": verdict,
            "nekaon_frontier": nekaon_frontier,
            "adakaon_frontier": adakaon_frontier,
        }
    verdicts = {item["verdict"] for item in by_regime.values()}
    if verdicts == {"adakaon_supersedes_nekaon"}:
        verdict = "adakaon_supersedes_nekaon"
    elif "nekaon_retains_a_pareto_advantage" in verdicts:
        verdict = "nekaon_retains_a_pareto_advantage"
    else:
        verdict = "inconclusive_frontiers_overlap"
    return {
        "verdict": verdict,
        "by_regime": by_regime,
        "note": "A low gap is not a win unless held-out loss remains within 1%.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--base-lr", type=float, default=1.2e-3)
    parser.add_argument("--lr-multipliers", type=float, nargs="+", default=(0.5, 1.0, 2.0))
    parser.add_argument("--regimes", nargs="+", choices=("constant", "rex", "replay"), default=("constant", "rex"))
    parser.add_argument("--lr-trace", type=Path)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--k", type=float, default=1.5)
    parser.add_argument("--checkpoints", type=int, default=None)
    parser.add_argument("--eval-reps", type=int, default=None)
    parser.add_argument("--output", type=Path, default=HERE / "nekaon_relative_results.json")
    args = parser.parse_args()
    steps = args.steps or (320 if args.quick else 2000)
    seeds = args.seeds or (1 if args.quick else 8)
    channels = 16 if args.quick else 96
    checkpoints = args.checkpoints or (4 if args.quick else 16)
    eval_reps = args.eval_reps or (2 if args.quick else 32)
    replay = None
    if args.lr_trace:
        replay = json.loads(args.lr_trace.read_text())
    if "replay" in args.regimes and replay is None:
        parser.error("--regimes replay requires --lr-trace")
    dataset = D.build_proxy_dataset()
    data = {key: value.to(DEV).to(H.DT) for key, value in dataset["DATA"].items()}
    alphas = H.make_alphas()
    results = []
    for regime in args.regimes:
        scales = lr_multipliers(regime, steps, replay)
        for lr_factor in args.lr_multipliers:
            lr = args.base_lr * lr_factor
            for seed in range(seeds):
                for arm in ("adakaon", "nekaon"):
                    result = run_arm(
                        arm,
                        seed=seed,
                        base_lr=lr,
                        multipliers=scales,
                        regime=regime,
                        data=data,
                        train_idx=dataset["TR"],
                        test_idx=dataset["TE"],
                        alphas=alphas,
                        channels=channels,
                        batch_size=8,
                        beta1=args.beta1,
                        weight_decay=args.weight_decay,
                        k=args.k,
                        checkpoints=checkpoints,
                        eval_reps=eval_reps,
                    )
                    results.append(result)
                    print(
                        f"{regime:8s} lr={lr:.3g} seed={seed} {arm:8s} "
                        f"te={result.heldout_loss:.4f} gap={result.gap:+.4f} "
                        f"damage={result.damage:.1%} {result.ms_step:.1f}ms",
                        flush=True,
                    )
    rows = aggregate(results)
    payload = {
        "device": DEV,
        "steps": steps,
        "seeds": seeds,
        "pairing": "same initialization, batches, noise, hyperparameters, and LR trajectory",
        "rows": rows,
        "decision": decision(rows),
        "raw": [asdict(result) for result in results],
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["decision"], indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
