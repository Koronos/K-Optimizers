"""Micro-benchmark for 0-D scalar batching in the ``foreach``/``fused`` paths — the
measurement gate for the "batch LyCORIS ``use_scalar`` gates instead of looping them"
change (``docs/foreach-batching.md``). Same house style as ``bench_fused.py``: times
``opt.step()`` (wall-clock, warmup + ``cuda.synchronize`` barriers + median over reps),
prints ms/step + the fused/native ratio.

Regimes (``--regime``):
  * ``scalar0d`` — 448 bare 0-D tensors (LyCORIS ``use_scalar`` gates). The case that
                   used to fall back to the per-param loop (~22 launches/scalar/step,
                   CPU-dispatch-bound).
  * ``scalar1d`` — the same count as shape ``(1,)`` instead of ``()``. 0-D scalars ride
                   the same ``L == 1`` bucket as length-1 rows, so this should track
                   ``scalar0d`` closely — a discrepancy would mean the two aren't
                   actually sharing a kernel path.
  * ``mixed``    — a realistic full-fine-tune-with-LyCORIS bag: 2-D weights + 1-D
                   biases + 0-D scalars together, to check bucketing doesn't regress
                   when scalars share a param list with everything else.
  * ``oversize`` — tensors above ``foreach_batch_cutoff`` (default 2M elements/tensor).
                   These always loop regardless of ``foreach``/``fused``; the two
                   should be within noise of each other (stacking never kicks in).

    python benchmarks/fused/bench_scalar0d.py --regime scalar0d --opt Adakaon
    python benchmarks/fused/bench_scalar0d.py --regime all --reps 50 --warmup 10
"""
from __future__ import annotations

import argparse
import time

import torch

from kaon import Adakaon

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------- bag builders
def _bag(shapes: list[tuple[int, ...]], dtype: torch.dtype, seed: int = 0) -> list[torch.Tensor]:
    """Leaf params (random) with a random grad attached, on DEV. Grads stay fixed across
    reps (as ``bench_fused.py`` does) — the kernel does the same work each step, so the
    timing is representative; the momentum EMA evolving is irrelevant to wall-clock."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for sh in shapes:
        # torch.randn(size) needs the shape as one sequence arg — unpacking ``sh`` breaks
        # for 0-D (``()`` unpacks to zero args, and ``randn()`` requires a size).
        p = torch.randn(sh, generator=g).to(DEV).to(dtype).requires_grad_(True)
        p.grad = torch.randn(sh, generator=g).to(DEV).to(dtype)
        out.append(p)
    return out


def make_bag(regime: str, dtype: torch.dtype) -> list[torch.Tensor]:
    if regime == "scalar0d":     # 448 bare 0-D LyCORIS-gate-style scalars
        return _bag([()] * 448, dtype)
    if regime == "scalar1d":     # same count, shape (1,) instead of () -> parity check
        return _bag([(1,)] * 448, dtype)
    if regime == "mixed":        # full-FT-with-LyCORIS: 2-D + 1-D + 0-D together
        return _bag([(256, 256)] * 200, dtype) + _bag([(512,)] * 100, dtype) + _bag([()] * 128, dtype)
    if regime == "oversize":     # > foreach_batch_cutoff (2M) elements/tensor -> always loops
        return _bag([(2048, 1100)] * 4, dtype)  # 2.25M elements each
    raise ValueError(f"unknown regime {regime!r}")


REGIMES = ["scalar0d", "scalar1d", "mixed", "oversize"]


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
        t0 = time.perf_counter()
        opt.step()
        if DEV == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


def build(cls, params, *, fused: bool):
    """Construct an optimizer in a realistic config (cautious+gc+wd+bf16 momentum) —
    gradient_centralization is a no-op below ndim 2, so it doesn't touch the scalars."""
    return cls(params, lr=1e-3, weight_decay=0.01, cautious=True,
               gradient_centralization=True, momentum_dtype="bfloat16", fused=fused)


# ----------------------------------------------------------------- main
def gpu_busy_gib() -> float:
    if DEV != "cuda":
        return 0.0
    free, total = torch.cuda.mem_get_info()
    return (total - free) / (1024 ** 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="scalar0d", help="scalar0d|scalar1d|mixed|oversize|all")
    ap.add_argument("--opt", nargs="+", default=["Adakaon"])
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--force", action="store_true", help="run even if the GPU looks busy")
    A = ap.parse_args()

    if DEV != "cuda":
        raise SystemExit("foreach/fused paths need CUDA — no GPU available")
    busy = gpu_busy_gib()
    if busy > 1.0 and not A.force:
        raise SystemExit(
            f"GPU busy ({busy:.1f} GiB used) — likely a live training run. "
            "Re-run with --force only when you're sure it's free (check nvidia-smi)."
        )

    dtype = torch.bfloat16 if A.dtype == "bf16" else torch.float32
    classes = {"Adakaon": Adakaon}
    regimes = REGIMES if A.regime == "all" else [A.regime]

    print(f"# scalar-0d micro-bench  dtype={A.dtype} reps={A.reps} warmup={A.warmup} dev={DEV}")
    print(f"{'opt':<8} {'regime':<9} {'native':>9} {'fused':>9} {'fused/nat':>10}")
    for name in A.opt:
        cls = classes[name]
        for regime in regimes:
            # fresh bag per config so state alloc cost isn't shared/warmed across configs
            nat = step_ms(build(cls, make_bag(regime, dtype), fused=False), A.reps, A.warmup)
            fus = step_ms(build(cls, make_bag(regime, dtype), fused=True), A.reps, A.warmup)
            r1 = nat / fus if fus else float("nan")
            print(f"{name:<8} {regime:<9} {nat:9.3f} {fus:9.3f} {r1:10.2f}")


if __name__ == "__main__":
    main()
