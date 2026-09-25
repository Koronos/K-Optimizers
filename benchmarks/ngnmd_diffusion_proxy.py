"""Low-resolution diffusion proxy for the loss-aware NGN-MDv1 reference."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from statistics import mean

import torch

from kaon import NGNMD, Adakaon, Lion, Nekaon

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


H = load("proxy_harness", ROOT / "benchmarks/proxy/harness.py")
D = load("proxy_dataset", ROOT / "benchmarks/proxy/dataset.py")

GRIDS = {
    "adakaon": (6e-4, 1.2e-3, 2.4e-3, 4.8e-3),
    "nekaon": (6e-4, 1.2e-3, 2.4e-3, 4.8e-3),
    "lion": (1.5e-4, 3e-4, 6e-4, 1.2e-3),
    "ngnmd": (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0),
}

CONFIRM_GRIDS = {
    "adakaon": (2.4e-3, 4.8e-3),
    "nekaon": (1.2e-3, 2.4e-3),
    "lion": (3e-4, 6e-4),
    "ngnmd": (3e-3, 1e-2, 3e-2, 1e-1),
}


def make(name: str, params, lr: float):
    if name == "adakaon":
        return Adakaon(params, lr=lr, betas=(0.9, 0.999), cautious=True, momentum_dtype="bfloat16")
    if name == "nekaon":
        return Nekaon(params, lr=lr, k=1.5, betas=(0.5, 0.999), weight_decay=0.1, momentum_dtype="4bit")
    if name == "lion":
        return Lion(params, lr=lr, betas=(0.95, 0.98), cautious=True, momentum_dtype="bfloat16")
    if name == "ngnmd":
        return NGNMD(params, lr=lr, betas=(0.9, 0.999))
    raise ValueError(name)


def evaluate(opt, fn):
    swap = hasattr(opt, "eval") and hasattr(opt, "train")
    if swap:
        opt.eval()
    result = fn()
    if swap:
        opt.train()
    return result


def run(name: str, lr: float, seed: int, steps: int, channels: int, resolution: int, dataset):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = H.DEV
    data = {key: value.to(device) for key, value in dataset["DATA"].items()}
    net = H.UNet(C=channels).to(device).to(H.DT)
    params = list(net.parameters())
    opt = make(name, params, lr)
    ac = H.make_alphas()
    generator = torch.Generator(device=device).manual_seed(seed + 12345)
    train, test = dataset["TR"], dataset["TE"]
    start = time.perf_counter()
    for step in range(steps):
        indices = [train[(step * 8 + offset) % len(train)] for offset in range(8)]
        opt.zero_grad(set_to_none=True)
        loss = H.batch_loss(net, data[resolution], torch.tensor(indices, device=device), ac, generator)
        loss.backward()
        if name == "ngnmd":
            opt.step(loss=loss.detach())
        else:
            opt.step()
        if not all(bool(torch.isfinite(param).all()) for param in params):
            return {"name": name, "lr": lr, "seed": seed, "finite": False}
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    train_loss, test_loss = evaluate(opt, lambda: (
        H.eval_loss(net, data[resolution], train, ac, reps=4),
        H.eval_loss(net, data[resolution], test, ac, reps=4),
    ))
    return {
        "name": name, "lr": lr, "seed": seed, "finite": True,
        "train": train_loss, "test": test_loss, "gap": test_loss - train_loss,
        "ms_step": 1000.0 * elapsed / steps,
        "state_bpp": H.opt_state_bytes_per_param(opt, params),
        "effective_lr": opt.get_effective_lr() if name == "ngnmd" else lr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizers", nargs="+", default=list(GRIDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--confirm-grid", action="store_true")
    args = parser.parse_args()
    grids = CONFIRM_GRIDS if args.confirm_grid else GRIDS
    dataset = D.build_proxy_dataset()
    rows = [
        run(name, lr, seed, args.steps, args.channels, args.resolution, dataset)
        for name in args.optimizers for seed in args.seeds for lr in grids[name]
    ]
    summary = {}
    for name in args.optimizers:
        selected = [row for row in rows if row["name"] == name and row["finite"]]
        per_lr = {
            str(lr): mean(row["test"] for row in selected if row["lr"] == lr)
            for lr in grids[name]
            if any(row["lr"] == lr for row in selected)
        }
        best_lr, best_test = min(per_lr.items(), key=lambda item: item[1])
        best_rows = [row for row in selected if str(row["lr"]) == best_lr]
        summary[name] = {
            "best_lr": best_lr, "best_test": best_test,
            "best_gap": mean(row["gap"] for row in best_rows),
            "state_bpp": mean(row["state_bpp"] for row in best_rows),
            "ms_step": mean(row["ms_step"] for row in best_rows),
            "all_finite": len(selected) == len(grids[name]) * len(args.seeds),
            "per_lr": per_lr,
        }
    payload = {"summary": summary} if args.summary_only else {"summary": summary, "rows": rows}
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
