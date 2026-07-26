"""CUDA many-tensor overhead probe for the shared AutoLR mixin."""

from __future__ import annotations

import argparse
import json

import torch

from kaon import Adakaon, Lion


def run(kind: str, auto_lr: bool, tensors: int, width: int, steps: int) -> dict[str, float]:
    torch.manual_seed(123)
    params = [
        torch.randn(width, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
        for _ in range(tensors)
    ]
    cls = Adakaon if kind == "adakaon" else Lion
    kwargs = dict(
        lr=1.0 if auto_lr else 1e-3,
        weight_decay=0.0,
        momentum_dtype="bfloat16",
        cautious=False,
        gradient_centralization=False,
        foreach=True,
        auto_lr=auto_lr,
    )
    opt = cls(params, **kwargs)
    grads = [torch.randn_like(p) for p in params]
    for _ in range(5):
        for p, grad in zip(params, grads, strict=True):
            p.grad = grad
        opt.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        for p, grad in zip(params, grads, strict=True):
            p.grad = grad
        opt.step()
    end.record()
    torch.cuda.synchronize()
    return {
        "ms_per_step": start.elapsed_time(end) / steps,
        "peak_mb": torch.cuda.max_memory_allocated() / 2**20,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=("adakaon", "lion"), required=True)
    parser.add_argument("--tensors", type=int, default=512)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.profile:
        torch.manual_seed(123)
        params = [
            torch.randn(args.width, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
            for _ in range(args.tensors)
        ]
        cls = Adakaon if args.optimizer == "adakaon" else Lion
        opt = cls(
            params, lr=1.0, weight_decay=0.0, momentum_dtype="bfloat16",
            cautious=False, gradient_centralization=False, foreach=True, auto_lr=True,
        )
        grads = [torch.randn_like(p) for p in params]
        for _ in range(5):
            for p, grad in zip(params, grads, strict=True):
                p.grad = grad
            opt.step()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            for p, grad in zip(params, grads, strict=True):
                p.grad = grad
            opt.step()
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
        return
    fixed = run(args.optimizer, False, args.tensors, args.width, args.steps)
    autolr = run(args.optimizer, True, args.tensors, args.width, args.steps)
    print(json.dumps({
        "optimizer": args.optimizer,
        "fixed": fixed,
        "autolr": autolr,
        "time_ratio": autolr["ms_per_step"] / fixed["ms_per_step"],
    }, indent=2))


if __name__ == "__main__":
    main()
