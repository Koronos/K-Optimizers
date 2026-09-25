"""CUDA many-tensor latency and peak-memory probe for ``lr_servo``."""

from __future__ import annotations

import argparse
import json

import torch

from kaon import Adakaon, Lion


def run(kind: str, servo: bool, tensors: int, width: int, steps: int) -> dict[str, float]:
    torch.manual_seed(123)
    params = [
        torch.randn(width, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
        for _ in range(tensors)
    ]
    cls = Adakaon if kind == "adakaon" else Lion
    opt = cls(
        params, lr=1e-3, weight_decay=0.0, momentum_dtype="bfloat16",
        cautious=False, gradient_centralization=False, foreach=True, lr_servo=servo,
    )
    grads = [torch.randn_like(p) for p in params]
    for _ in range(8):
        for p, grad in zip(params, grads, strict=True):
            p.grad = grad
        opt.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(steps):
        for p, grad in zip(params, grads, strict=True):
            p.grad = grad
        opt.step()
    end.record()
    torch.cuda.synchronize()
    return {
        "ms_step": start.elapsed_time(end) / steps,
        "peak_mb": torch.cuda.max_memory_allocated() / 2**20,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=("adakaon", "lion"), required=True)
    parser.add_argument("--tensors", type=int, default=512)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    fixed = run(args.optimizer, False, args.tensors, args.width, args.steps)
    servo = run(args.optimizer, True, args.tensors, args.width, args.steps)
    print(json.dumps({
        "optimizer": args.optimizer, "fixed": fixed, "servo": servo,
        "time_ratio": servo["ms_step"] / fixed["ms_step"],
        "peak_delta_mb": servo["peak_mb"] - fixed["peak_mb"],
    }, indent=2))


if __name__ == "__main__":
    main()
