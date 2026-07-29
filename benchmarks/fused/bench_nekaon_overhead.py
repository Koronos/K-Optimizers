"""Measure Nekaon's wrapper overhead against an identically configured Adakaon."""

from __future__ import annotations

import argparse
import time

import torch

from kaon import Adakaon, Nekaon


def bag(regime: str, dtype: torch.dtype) -> list[torch.Tensor]:
    shapes = {
        "lora": [(8, 16)] * 512,
        "mixed": [(8, 16)] * 384 + [(1024,)] * 96 + [(64, 64)] * 32,
        "conv": [(320, 320, 3, 3)] * 24,
        "big": [(512, 512)] * 128,
    }[regime]
    gen = torch.Generator(device="cuda").manual_seed(123)
    params = []
    for shape in shapes:
        p = torch.randn(shape, generator=gen, dtype=dtype, device="cuda").requires_grad_(True)
        p.grad = torch.randn(shape, generator=gen, dtype=dtype, device="cuda")
        params.append(p)
    return params


def measure(opt, reps: int, warmup: int) -> float:
    for _ in range(warmup):
        opt.step()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        opt.step()
        torch.cuda.synchronize()
        samples.append(1000.0 * (time.perf_counter() - start))
    return sorted(samples)[len(samples) // 2]


def make(kind: str, params, momentum_dtype: str, fused: bool):
    common = dict(
        lr=1.2e-3,
        betas=(0.5, 0.999),
        weight_decay=0.1,
        cautious=True,
        gradient_centralization=True,
        momentum_dtype=momentum_dtype,
        fused=fused,
    )
    if kind == "adakaon":
        return Adakaon(params, **common)
    return Nekaon(params, k=1.5, **common)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--regimes", nargs="+", default=["lora", "mixed", "conv", "big"])
    parser.add_argument("--momentum", nargs="+", default=["4bit", "bfloat16"])
    args = parser.parse_args()
    print("kind      fused momentum regime       ms  nekaon/adakaon")
    for momentum in args.momentum:
        for fused in (False, True):
            for regime in args.regimes:
                timings = {}
                for kind in ("adakaon", "nekaon"):
                    params = bag(regime, torch.bfloat16)
                    timings[kind] = measure(
                        make(kind, params, momentum, fused), args.reps, args.warmup
                    )
                    del params
                    torch.cuda.empty_cache()
                ratio = timings["nekaon"] / timings["adakaon"]
                for kind in ("adakaon", "nekaon"):
                    print(
                        f"{kind:<10}{str(fused):<6}{momentum:<9}{regime:<8}"
                        f"{timings[kind]:8.3f}  {ratio:8.3f}"
                    )


if __name__ == "__main__":
    main()
