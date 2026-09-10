"""Paired constant-LR screening. Outputs raw runs, not a quality claim.

Run from the repository root with PYTHONPATH=src.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
from control import battery as B

from kaon import Adakaon, Nekaon, Rakaon


def target_hits(trajectory):
    """Observed checkpoint crossings; no interpolation or extrapolation."""
    result = {}
    for target in (.10, .09, .08, .077, .07):
        for bounded_gap in (False, True):
            key = f"val<{target:g}" + ("_abs_gap<0.007" if bounded_gap else "")
            hits = [i for i, point in enumerate(trajectory)
                    if point["te"] < target and (not bounded_gap or abs(point["gap"]) < .007)]
            first = trajectory[hits[0]] if hits else None
            sustained = next((trajectory[i] for i in hits if i + 1 in hits), None)
            result[key] = dict(first=first, first_of_two_consecutive=sustained)
    return result


def timed_train(make, lr, args, seed, data, ds):
    """Measure actual forward/backward/step time to joint quality thresholds.

    Training time excludes validation, but includes warmup. Wall time includes
    validation. Both begin after construction; no ms/step multiplication is used.
    """
    torch.manual_seed(seed)
    net = B.H.UNet(C=args.channels).to(B.DEV)
    params = list(net.parameters())
    opt = make(params, lr)
    ac = B.H.make_alphas()
    gen = torch.Generator(device=B.DEV).manual_seed(seed + 12345)
    trajectory = []
    tr = ds["TR"]
    def sync():
        if B.DEV == "cuda":
            torch.cuda.synchronize()
    sync()
    wall_start = train_start = time.perf_counter()
    train_seconds = 0.
    for step, resolution in enumerate(B.seq_prog(args.steps), start=1):
        idx = torch.tensor([tr[((step - 1) * 8 + j) % len(tr)] for j in range(8)], device=B.DEV)
        opt.zero_grad(set_to_none=True)
        B.H.batch_loss(net, data[resolution], idx, ac, gen).backward()
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            sync()
            train_seconds += time.perf_counter() - train_start
            train_loss, val_loss = B.evald(opt, lambda: (
                B.H.eval_loss(net, data[64], ds["TR"], ac),
                B.H.eval_loss(net, data[64], ds["TE"], ac)))
            sync()
            trajectory.append(dict(step=step, training_seconds=train_seconds,
                                   wall_seconds=time.perf_counter() - wall_start,
                                   tr=train_loss, te=val_loss, gap=val_loss - train_loss))
            print(json.dumps(dict(progress_step=step, **{k: trajectory[-1][k] for k in ("te", "gap", "training_seconds")})), flush=True)
            train_start = time.perf_counter()
    last = trajectory[-1]
    return dict(tr=last["tr"], te=last["te"], gap=last["gap"],
                ms=1000 * train_seconds / args.steps,
                bpp=B.H.opt_state_bytes_per_param(opt, params), traj=trajectory,
                targets=target_hits(trajectory))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--lrs", type=float, nargs="+", default=[.0006, .0012, .0024])
    parser.add_argument("--arm-lr", action="append", default=[], help="Per-arm LR, e.g. Rakaon-1=.0024")
    parser.add_argument("--eval-every", type=int, default=0, help="0: final-only legacy protocol; >0: time-to-quality trajectory")
    parser.add_argument("--arms", nargs="+", default=["Adakaon", "Nekaon", "Rakaon-0", "Rakaon-0.1", "Rakaon-0.5", "Rakaon-1"])
    parser.add_argument("--output", type=Path, default=Path("benchmarks/rakaon_screen.json"))
    args = parser.parse_args()
    if args.steps <= 0 or args.eval_every < 0:
        parser.error("steps must be positive and eval-every nonnegative")
    arm_lrs = {name: float(lr) for name, lr in (value.split("=", 1) for value in args.arm_lr)}
    ds = B.D.build_proxy_dataset()
    data = {r: x.to(B.DEV) for r, x in ds["DATA"].items()}
    factories = {
        "AdamW": lambda p, lr: torch.optim.AdamW(p, lr=lr, fused=True),
        "Adakaon": lambda p, lr: Adakaon(p, lr=lr, betas=(0., .999), cautious=False),
        "Nekaon": lambda p, lr: Nekaon(p, lr=lr, k=1.5, betas=(.5, .999), weight_decay=.3, momentum_dtype="4bit"),
    }
    for s in (0., .1, .5, 1.):
        factories[f"Rakaon-{s:g}"] = lambda p, lr, s=s: Rakaon(p, lr=lr, shrinkage=s)
    for beta1 in (.5, .9):
        factories[f"Rakaon-m{beta1:g}"] = lambda p, lr, b=beta1: Rakaon(p, lr=lr, shrinkage=1, beta1=b)
    for block_size in (64, 256, 1024):
        factories[f"Rakaon-block{block_size}"] = lambda p, lr, b=block_size: Rakaon(p, lr=lr, shrinkage=1, block_size=b)
    output = dict(settings=vars(args) | {"output": str(args.output)},
                  torch=torch.__version__, python=platform.python_version(),
                  device=torch.cuda.get_device_name() if B.DEV == "cuda" else "cpu",
                  fingerprint=B.D.fingerprint(ds), runs=[])
    source = Path(__file__).resolve().parents[1] / "src/kaon/rakaon.py"
    output["rakaon_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    for arm in args.arms:
        for lr in ([arm_lrs[arm]] if arm in arm_lrs else args.lrs):
            for seed in args.seeds:
                if args.eval_every:
                    result = timed_train(factories[arm], lr, args, seed, data, ds)
                else:
                    result = B.train(factories[arm], lr, schedule="constant", seq=B.seq_prog(args.steps),
                                     seed=seed, data=data, tr=ds["TR"], te=ds["TE"],
                                     ac=B.H.make_alphas(), channels=args.channels, bs=8, n=args.steps)
                row = dict(arm=arm, lr=lr, seed=seed, **result)
                output["runs"].append(row)
                args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
                print(json.dumps({k: v for k, v in row.items() if k not in ("traj", "targets")}), flush=True)


if __name__ == "__main__":
    main()
