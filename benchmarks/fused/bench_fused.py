"""Micro-benchmark for the Triton FUSED step vs the native path — the measurement gate
for the "send more to Triton" campaign (docs/EXPERIMENTS_GRAVEYARD.md tracks the verdicts).

Until now the fused kernels had ZERO speed coverage; the 80->15.5ms numbers lived only in
commit messages. This script times ``opt.step()`` (wall-clock, warmup + ``cuda.synchronize``
barriers + median over reps — the house style from ``benchmarks/control/battery.py``
``lora_step_ms``) across the regimes each campaign candidate targets, and prints ms/step +
the fused/native speedup ratio so a candidate is accepted only on a measured win (>1.0x).

Regimes (``--regime``):
  * ``big``  — 236x 512x512 factors (the Cosmos LoKr workload; candidate #1, the >tile_cap
               batched path). The dominant real shape.
  * ``oned`` — many 1-D biases/norms (candidate #2).
  * ``conv`` — ndim>2 conv kernels (candidate #3).
  * ``lora`` — many tiny 2-D adapters (sanity for the already-fused one-block path).

A/B columns:
  * ``native``      — ``fused=False`` (the torch foreach path).
  * ``fused``       — ``fused=True`` (the Triton path; for ``big`` this is the candidate-#1
                      batched chunked path once it lands, else today's native-foreach fallback).
  * ``fused_nobat`` — ``fused=True`` with ``opt._fused_big_batched=False`` if that toggle
                      exists, so the batched big path can be A/B'd against the in-fused
                      native-foreach fallback within one build (skipped if the attr is absent).

GPU pre-flight: refuses to run if the GPU is already busy (>1 GiB used) unless ``--force`` —
the dev box often has a training run live. Pass ``--force`` to override.

    python benchmarks/fused/bench_fused.py --regime big --opt Adakaon AdaPNM
    python benchmarks/fused/bench_fused.py --regime all --reps 50 --warmup 10
"""
from __future__ import annotations

import argparse
import time

import torch

from kaon import Adakaon, AdaPNM

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------- bag builders
def _bag(shapes: list[tuple[int, ...]], dtype: torch.dtype, seed: int = 0) -> list[torch.Tensor]:
    """Leaf params (random) with a random grad attached, on DEV. Grads stay fixed across
    reps (as ``lora_step_ms`` does) — the kernel does the same work each step, so the timing
    is representative; the momentum EMA evolving is irrelevant to wall-clock."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for sh in shapes:
        p = torch.randn(*sh, generator=g).to(DEV).to(dtype).requires_grad_(True)
        p.grad = torch.randn(*sh, generator=g).to(DEV).to(dtype)
        out.append(p)
    return out


def make_bag(regime: str, dtype: torch.dtype) -> list[torch.Tensor]:
    if regime == "big":      # Cosmos LoKr: 236x 512x512 (>tile_cap -> candidate #1)
        return _bag([(512, 512)] * 236, dtype)
    if regime == "oned":     # biases / norm scales (candidate #2)
        return _bag([(1024,)] * 400 + [(2048,)] * 200, dtype)
    if regime == "conv":     # conv kernels ndim>2 (candidate #3)
        return _bag([(320, 320, 3, 3)] * 24, dtype)
    if regime == "lora":     # many tiny 2-D adapters (one-block sanity)
        return _bag([(8, 16)] * 512, dtype)
    raise ValueError(f"unknown regime {regime!r}")


REGIMES = ["big", "oned", "conv", "lora"]


# ----------------------------------------------------------------- timing
def step_ms(opt: torch.optim.Optimizer, reps: int, warmup: int) -> float:
    """Median ms for one ``opt.step()`` (sync barriers around each rep)."""
    for _ in range(warmup):
        opt.step()
    if DEV == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        if DEV == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        opt.step()
        if DEV == "cuda":
            torch.cuda.synchronize()
        ts.append((time.time() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


def build(cls, params, *, fused: bool, big_batched: bool | None):
    """Construct an optimizer in the campaign's real config (cautious+gc+wd+bf16 momentum)."""
    opt = cls(params, lr=1e-3, weight_decay=0.01, cautious=True,
              gradient_centralization=True, momentum_dtype="bfloat16", fused=fused)
    if fused and big_batched is not None and hasattr(opt, "_fused_big_batched"):
        opt._fused_big_batched = big_batched
    return opt


# ----------------------------------------------------------------- main
def gpu_busy_gib() -> float:
    if DEV != "cuda":
        return 0.0
    free, total = torch.cuda.mem_get_info()
    return (total - free) / (1024 ** 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="big", help="big|oned|conv|lora|all")
    ap.add_argument("--opt", nargs="+", default=["Adakaon", "AdaPNM"])
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--force", action="store_true", help="run even if the GPU looks busy")
    A = ap.parse_args()

    if DEV != "cuda":
        raise SystemExit("fused path needs CUDA — no GPU available")
    busy = gpu_busy_gib()
    if busy > 1.0 and not A.force:
        raise SystemExit(
            f"GPU busy ({busy:.1f} GiB used) — likely a live training run. "
            "Re-run with --force only when you're sure it's free (check nvidia-smi)."
        )

    dtype = torch.bfloat16 if A.dtype == "bf16" else torch.float32
    classes = {"Adakaon": Adakaon, "AdaPNM": AdaPNM}
    regimes = REGIMES if A.regime == "all" else [A.regime]

    print(f"# fused micro-bench  dtype={A.dtype} reps={A.reps} warmup={A.warmup} dev={DEV}")
    print(f"{'opt':<8} {'regime':<6} {'native':>9} {'fused':>9} {'fused/nat':>10} "
          f"{'fused_nobat':>12} {'bat/nobat':>10}")
    for name in A.opt:
        cls = classes[name]
        for regime in regimes:
            # fresh bag per config so state alloc cost isn't shared/warmed across configs
            nat = step_ms(build(cls, make_bag(regime, dtype), fused=False, big_batched=None),
                          A.reps, A.warmup)
            fus = step_ms(build(cls, make_bag(regime, dtype), fused=True, big_batched=True),
                          A.reps, A.warmup)
            o_nb = build(cls, make_bag(regime, dtype), fused=True, big_batched=False)
            nobat = step_ms(o_nb, A.reps, A.warmup) if hasattr(o_nb, "_fused_big_batched") else float("nan")
            r1 = nat / fus if fus else float("nan")
            r2 = (nobat / fus) if (fus and nobat == nobat) else float("nan")
            print(f"{name:<8} {regime:<6} {nat:9.3f} {fus:9.3f} {r1:10.2f} "
                  f"{nobat:12.3f} {r2:10.2f}")


if __name__ == "__main__":
    main()
