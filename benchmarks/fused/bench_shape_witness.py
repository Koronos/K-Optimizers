"""What the 0.7.13 shape guard costs, and why ``strides`` is the field that got picked.

The guard has two halves and only one of them can possibly cost anything:

* ``check_state_geometry`` runs at cache BUILD, so in steady state it runs zero times. Not
  measured here beyond confirming the step is unchanged; there is nothing to measure.
* ``param_witness`` runs once per group per step, on the host, over every param in the group.
  That is the whole budget, and this file prices every candidate field for it.

Two measurements, because they answer different questions:

``--case field`` prices the candidate fields ALONE — a host sweep over the bag, no GPU work in
the loop, median of 9 blocks of 200 reps. This is where the field choice is made: it is a pure
CPU cost and the numbers are stable to a few percent even on a shared machine.

``--case step`` puts the chosen field into a real ``Adakaon.step()`` and measures end to end,
PAIRED and interleaved (the design ``bench_perf_a2`` argues for: adjacent arms, alternating
order, log-ratio per pair, geometric mean + 95% CI, so a contended GPU cancels). Both arms live
in the SAME process and differ only in which ``param_witness`` the module holds, so the A/B is
the field and nothing else.

Candidates and what each one can see (all four keep ``(ids, data_ptrs, contiguity)``):

* ``dim``     — the cheapest possible field (``Tensor.dim`` returns cached small ints). Sees
  2-D <-> 1-D rebinds and nothing else: ``(16,64) -> (64,16)``, the reported bug, is invisible.
* ``strides`` — CHOSEN. For a contiguous tensor ``stride(0)`` is the ``C`` of ``eff_2d`` and the
  rest of the tuple pins every inner dim, so every change to the matrix the factored state
  describes moves it. Blind only to a rebind that narrows the ROW count at fixed ``stride(0)``
  (``p.data[:8]``), which needs ``numel``.
* ``numel``   — sees the row-narrowing case and NOT the reported one (a ``view`` preserves numel).
  Complete only together with ``strides``, i.e. at the sum of both costs.
* ``shape``   — complete on its own, and the field the pre-0.7.13 docstrings priced. It loses to
  ``strides`` because ``Tensor.size`` allocates a ``torch.Size`` per param where ``Tensor.stride``
  returns a tuple of (mostly cached) small ints.

Usage::

    python benchmarks/fused/bench_shape_witness.py --case field
    python benchmarks/fused/bench_shape_witness.py --case step --reps 200
"""
from __future__ import annotations

import argparse
import math
import statistics as st
import time
from collections.abc import Callable

import torch
from torch import Tensor

import kaon._fused_triton as ft
from kaon import Adakaon

DEV = "cuda"
_ID, _PTR, _CONTIG = id, Tensor.data_ptr, Tensor.is_contiguous
_STRIDE, _NUMEL, _DIM, _SIZE = Tensor.stride, Tensor.numel, Tensor.dim, Tensor.size


# ----------------------------------------------------------------- witness variants
def w_base(pl) -> tuple:
    """main: (ids, data_ptrs, contiguity)."""
    return (tuple(map(_ID, pl)), tuple(map(_PTR, pl)), tuple(map(_CONTIG, pl)))


def w_strides(pl) -> tuple:
    """0.7.13, shipped."""
    return (tuple(map(_ID, pl)), tuple(map(_PTR, pl)), tuple(map(_CONTIG, pl)),
            tuple(map(_STRIDE, pl)))


def w_dim(pl) -> tuple:
    return (tuple(map(_ID, pl)), tuple(map(_PTR, pl)), tuple(map(_CONTIG, pl)),
            tuple(map(_DIM, pl)))


def w_numel(pl) -> tuple:
    return (tuple(map(_ID, pl)), tuple(map(_PTR, pl)), tuple(map(_CONTIG, pl)),
            tuple(map(_NUMEL, pl)))


def w_shape(pl) -> tuple:
    return (tuple(map(_ID, pl)), tuple(map(_PTR, pl)), tuple(map(_CONTIG, pl)),
            tuple(map(_SIZE, pl)))


def w_strides_numel(pl) -> tuple:
    """The complete pair: strides for the shape, numel for a row-narrowing view."""
    return (tuple(map(_ID, pl)), tuple(map(_PTR, pl)), tuple(map(_CONTIG, pl)),
            tuple(map(_STRIDE, pl)), tuple(map(_NUMEL, pl)))


VARIANTS: dict[str, Callable[[list], tuple]] = {
    "base (main)": w_base,
    "+dim": w_dim,
    "+strides": w_strides,
    "+numel": w_numel,
    "+shape": w_shape,
    "+strides+numel": w_strides_numel,
}


# ----------------------------------------------------------------- bags
def bag(shapes: list[tuple[int, ...]], dtype: torch.dtype = torch.bfloat16,
        seed: int = 0) -> list[Tensor]:
    """Leaf params with a fixed random grad attached (identical kernel work every rep)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for sh in shapes:
        p = torch.randn(sh, generator=g).to(DEV).to(dtype).requires_grad_(True)
        p.grad = torch.randn(sh, generator=g).to(DEV).to(dtype)
        out.append(p)
    return out


BAGS: dict[str, list[tuple[int, ...]]] = {
    # bench_perf_a2's reference mixed bag: the LoRA/adapter regime the witness was priced on.
    "lora428": [(256, 256)] * 200 + [(512,)] * 100 + [()] * 128,
    # Pure 0-D: the launch-bound extreme, and the one where the 1-D route carries everything.
    "zero448": [()] * 448,
    # UNet/DiT-ish: few params, most of them big, one tensor per shape bucket.
    "unet": ([(320, 320)] * 8 + [(320, 1280)] * 8 + [(1280, 320)] * 8 + [(640, 640)] * 4
             + [(320,)] * 40 + [(1280,)] * 12),
}


# ----------------------------------------------------------------- statistics
def host_us(fn: Callable[[], object], reps: int = 200, blocks: int = 9) -> float:
    """Median over ``blocks`` of the mean host µs per call. No GPU work in the loop."""
    fn()
    out = []
    for _ in range(blocks):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        out.append((time.perf_counter() - t0) / reps * 1e6)
    return st.median(out)


def host_delta_us(base: Callable[[], object], var: Callable[[], object],
                  reps: int = 200, blocks: int = 25) -> tuple[float, float, float]:
    """PAIRED host cost of ``var`` over ``base``: µs per call, with a 95% CI.

    Measuring the two arms in separate passes does not work on a shared machine — the base
    drifts by more than the field being priced, which produced *negative* field costs. Here each
    pair times a block of ``base`` and a block of ``var`` ADJACENT in time with the order
    alternating, and the statistic is the per-pair difference. Whatever the other tenants of the
    CPU are doing hits both members of a pair alike.
    """
    base()
    var()
    deltas = []
    for r in range(blocks):
        first, second = (base, var) if r % 2 == 0 else (var, base)
        t0 = time.perf_counter()
        for _ in range(reps):
            first()
        t1 = time.perf_counter()
        for _ in range(reps):
            second()
        t2 = time.perf_counter()
        xb, xv = ((t1 - t0), (t2 - t1)) if r % 2 == 0 else ((t2 - t1), (t1 - t0))
        deltas.append((xv - xb) / reps * 1e6)
    m = st.fmean(deltas)
    half = 1.96 * st.stdev(deltas) / math.sqrt(len(deltas))
    return m, m - half, m + half


def paired(a: Callable[[], None], b: Callable[[], None], reps: int,
           warm: int = 30) -> tuple[float, float, float]:
    """Geometric mean of ``t_a / t_b`` over ``reps`` PAIRS, with a 95% CI (see bench_perf_a2)."""
    for _ in range(warm):
        a()
        b()
    torch.cuda.synchronize()
    logs = []
    for r in range(reps):
        first, second = (a, b) if r % 2 == 0 else (b, a)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        first()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        second()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        xa, xb = ((t1 - t0), (t2 - t1)) if r % 2 == 0 else ((t2 - t1), (t1 - t0))
        logs.append(math.log(xa / xb))
    m = st.fmean(logs)
    half = 1.96 * st.stdev(logs) / math.sqrt(len(logs))
    return math.exp(m), math.exp(m - half), math.exp(m + half)


def step_us(run: Callable[[], None], reps: int = 60) -> tuple[float, float]:
    """``(min, median)`` wall-clock µs of one ``run()``, synchronize-bracketed.

    Both are reported because they answer different questions on a card shared with other jobs:
    the MIN is the closest thing to the uncontended step and is the honest denominator for "what
    fraction of a step is this field"; the median is what the machine actually delivered while
    the measurement ran, and the gap between them is the contention.
    """
    for _ in range(10):
        run()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e6)
    return min(out), st.median(out)


# ----------------------------------------------------------------- cases
def case_field(args: argparse.Namespace) -> None:
    """Price every candidate field on its own, as a host sweep."""
    print("\n## candidate witness fields — host cost of ONE witness call")
    print("   'extra' is PAIRED against the 3-field base, mean of 25 pairs with a 95% CI")
    for name, shapes in BAGS.items():
        pl = bag(shapes)
        base_us = host_us(lambda pl=pl: w_base(pl), args.reps)
        print(f"\n  bag {name}: {len(pl)} params — 3-field base {base_us:.1f} µs")
        for label, fn in VARIANTS.items():
            if fn is w_base:
                continue
            d, lo, hi = host_delta_us(lambda pl=pl: w_base(pl), lambda fn=fn, pl=pl: fn(pl),
                                      args.reps)
            print(f"    {label:<16} extra {d:+7.1f} µs [{lo:+6.1f},{hi:+6.1f}]   "
                  f"= {d / base_us * 100:+5.1f}% of the base witness")


def case_step(args: argparse.Namespace) -> None:
    """The shipped field inside a real step, paired A/B, fused and native.

    ONE OPTIMIZER PER ARM, deliberately. The variant is a module global, so flipping it around a
    single optimizer would make every step see a witness of a different arity from its cached one
    — a forced partition + pointer-cache rebuild per step, which is not what either arm costs.
    Each arm therefore keeps its own optimizer over the SAME params (``bench_perf_a2._two``'s
    trick) and the global is set to that arm's variant immediately before its step, so each
    optimizer only ever compares like with like.
    """
    print("\n## +strides inside Adakaon.step() — paired, interleaved, same process")
    print("   ratio > 1 means the 4-field witness is SLOWER; the CI says whether it resolves")
    for name, shapes in BAGS.items():
        for fused in (True, False):
            pl = bag(shapes)
            cfg = dict(lr=1e-3, weight_decay=0.01, fused=fused)
            opts = {v: Adakaon(pl, **cfg) for v in (w_base, w_strides)}

            def arm(variant, opts=opts):
                def run():
                    ft.param_witness = variant
                    opts[variant].step()
                return run

            # A THIRD optimizer on the base witness, so the same paired test can be run
            # base-vs-base. That NULL arm is the measurement's own noise floor: any effect
            # smaller than its CI is not resolvable by this design, whatever the point estimate.
            opts[None] = Adakaon(pl, **cfg)
            a_base, a_str = arm(w_base), arm(w_strides)

            def a_null(opts=opts):
                ft.param_witness = w_base
                opts[None].step()

            bmin, bmed = step_us(a_base, args.step_reps)
            smin, smed = step_us(a_str, args.step_reps)
            n_g, n_lo, n_hi = paired(a_null, a_base, args.reps)
            g, lo, hi = paired(a_str, a_base, args.reps)
            floor = max(abs(n_lo - 1), abs(n_hi - 1)) * 100
            resolved = "resolved" if (lo - 1) * 100 > floor else "BELOW the noise floor"
            tag = "fused" if fused else "native"
            print(f"\n  bag {name} ({len(pl)} params, {tag})")
            print(f"    step, 3-field witness   min {bmin:8.1f}  median {bmed:8.1f} µs")
            print(f"    step, +strides          min {smin:8.1f}  median {smed:8.1f} µs   "
                  f"min-delta {smin - bmin:+7.1f} µs = {(smin / bmin - 1) * 100:+5.2f}%")
            print(f"    paired null (base/base) {n_g:6.4f}x [{n_lo:.4f},{n_hi:.4f}]"
                  f"   noise floor +-{floor:.2f}%")
            print(f"    paired +strides         {g:6.4f}x [{lo:.4f},{hi:.4f}]"
                  f"   {(g - 1) * 100:+5.2f}% of the step, {resolved}")
    ft.param_witness = w_strides


def case_budget(args: argparse.Namespace) -> None:
    """What the shipped field costs a real step, by accounting rather than by A/B.

    The end-to-end paired A/B (``--case step``) cannot resolve an effect this small on a shared
    card: its own null arm (base vs base, where the true effect is zero) came out at +2.0% with
    a +-4.8% CI. So the primary number is arithmetic, from two things that ARE measurable to a
    few µs: how many times the witness runs per step and over how many params (counted by
    wrapping ``ft.param_witness``), and the paired host cost of the field at each of those sizes.

    The count is not 1. ``_fused_partition`` calls it once over the whole group, and every big
    shape bucket calls ``BigPointerCache.stale()`` once more over ITS params — because
    ``_fused_big`` rebuilds its bucket lists every step, so the O(1) ``built_from`` shortcut
    always misses. On the reference LoRA bag that is 2 calls (428 params + the 200-param
    (256,256) bucket), which is most of why the field costs what it does there.
    """
    print("\n## budget accounting for the shipped +strides field")
    real = ft.param_witness
    for name, shapes in BAGS.items():
        pl = bag(shapes)
        opt = Adakaon(pl, lr=1e-3, weight_decay=0.01, fused=True)
        for _ in range(12):
            opt.step()
        torch.cuda.synchronize()
        sizes: list[int] = []

        def counting(plist, _r=real, _s=sizes):
            _s.append(len(plist))
            return _r(plist)

        ft.param_witness = counting
        opt.step()
        torch.cuda.synchronize()
        ft.param_witness = real
        # Host cost of the field at each call's size, priced on a bag of that size cut from
        # this one (the composition matters: 0-D strides are () and cheaper than a 2-D pair).
        total = 0.0
        legs = []
        for n in sizes:
            sub = pl[:n]
            d, _lo, _hi = host_delta_us(lambda sub=sub: w_base(sub),
                                        lambda sub=sub: w_strides(sub), args.reps, blocks=15)
            total += d
            legs.append(f"{n}p:{d:+.1f}")
        reps = [0.0] * 0
        for _ in range(200):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            opt.step()
            torch.cuda.synchronize()
            reps.append((time.perf_counter() - t0) * 1e6)
        lo_step, med_step = min(reps), st.median(reps)
        print(f"\n  bag {name}: {len(pl)} params, {len(sizes)} witness call(s)/step "
              f"[{', '.join(legs)}]")
        print(f"    predicted added host cost   {total:+7.1f} µs/step")
        print(f"    step (min / median)         {lo_step:8.1f} / {med_step:8.1f} µs")
        print(f"    => {total / lo_step * 100:5.2f}% of the min step, "
              f"{total / med_step * 100:5.2f}% of the median step")


CASES = {"field": case_field, "budget": case_budget, "step": case_step}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=[*CASES, "all"], default="all")
    ap.add_argument("--reps", type=int, default=200, help="host reps / paired reps")
    ap.add_argument("--step-reps", type=int, default=60)
    args = ap.parse_args()
    if not (ft.HAS_TRITON and torch.cuda.is_available()):
        raise SystemExit("needs CUDA + Triton")
    for name in ([args.case] if args.case != "all" else list(CASES)):
        CASES[name](args)


if __name__ == "__main__":
    main()
