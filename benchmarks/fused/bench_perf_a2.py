"""A/B harness for the 0.7.12 fused/native performance batch.

Every optimization in that batch is either guarded by a toggle on the optimizer
(``_fused_big_lone_batched``, ``_eps1_on_means``, ``_native_rms_matvec``,
``_write_fold_lr``, ``_direct_int8``, ``deterministic_reductions``), by a module switch
(``kaon._backend.SR_TRITON``), or by a per-bucket flag the harness can flip on the cached
pointer arrays (``exact4``, for the 4-bit reduction rewrite). So both arms of every
comparison live in the SAME process and the same build, and the only difference between
them is the one thing under test. That matters on a laptop GPU whose clocks drift and
which may be sharing the card with a training run.

Statistics — the lesson of this batch. A ``min``-of-N per arm is fine for a 3x effect and
useless for a 3% one: with another process on the card the per-rep spread here reached
+-40%, which swamped every one of the native-path optimizations. What works is a PAIRED
design: run both arms back to back (order alternating), take the LOG RATIO per pair, and
report the geometric mean with a 95% CI. Contention hits both members of a pair alike, so
it cancels in the ratio, and the CI says outright whether the effect is resolvable at all.
``--reps`` therefore has to be large (150-400) for the small effects; ``paired`` prints
``WIN``/``LOSS``/``n.s.`` rather than a bare number.

Not everything needs the clock. Three statistics here are contention-IMMUNE and are the
primary evidence for the launch-bound changes:

* ``launches()``  — CUDA kernel count for one step (``torch.profiler``).
* ``peak_mb()``   — peak allocated bytes for one step, steady state already warm.
* ``sync_free()`` — whether the step performs any CPU<->GPU synchronization
  (``torch.cuda.set_sync_debug_mode("error")``). A sync per tensor per step is invisible
  in a ``synchronize``-bracketed timing loop and serializes a real training step.

Usage::

    python benchmarks/fused/bench_perf_a2.py --case all
    python benchmarks/fused/bench_perf_a2.py --case native --reps 300
"""
from __future__ import annotations

import argparse
import math
import statistics as st
import time
from collections.abc import Callable
from typing import Any

import torch

import kaon._backend as bk
import kaon._fused_triton as ft
from kaon import Adakaon

DEV = "cuda"


# ----------------------------------------------------------------- bags
def bag(shapes: list[tuple[int, ...]], dtype: torch.dtype, seed: int = 0) -> list[torch.Tensor]:
    """Leaf params with a fixed random grad attached (the kernel does identical work every
    rep, so the timing is representative; the EMA evolving is irrelevant to wall clock)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for sh in shapes:
        # Tuple form, not ``*sh``: ``randn()`` with no varargs is a TypeError, and the
        # realistic bag deliberately contains 0-D params (shape ``()``).
        p = torch.randn(sh, generator=g).to(DEV).to(dtype).requires_grad_(True)
        p.grad = torch.randn(sh, generator=g).to(DEV).to(dtype)
        out.append(p)
    return out


def realistic_bag(dtype: torch.dtype = torch.bfloat16) -> list[torch.Tensor]:
    """The batch's reference mixed bag: 200x (256,256) + 100x (512,) + 128 scalars."""
    return bag([(256, 256)] * 200 + [(512,)] * 100 + [()] * 128, dtype)


# ----------------------------------------------------------------- statistics
def paired(a: Callable[[], None], b: Callable[[], None], reps: int,
           warm: int = 30) -> tuple[float, float, float]:
    """Geometric mean of ``t_a / t_b`` over ``reps`` PAIRS, with a 95% CI.

    Both arms run adjacent in time with the order alternating, so a contended or drifting
    GPU affects both members of a pair alike and cancels in the ratio.
    """
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


def show(label: str, a: Callable[[], None], b: Callable[[], None], reps: int) -> None:
    g, lo, hi = paired(a, b, reps)
    verdict = "WIN " if lo > 1.0 else ("LOSS" if hi < 1.0 else "n.s.")
    print(f"  {label:<44} {g:6.3f}x [{lo:.3f},{hi:.3f}] {verdict}")


def launches(fn: Callable[[], None], warm: int = 4) -> int:
    """CUDA kernel launches for ONE call of ``fn``."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events()
               if e.device_type == torch.autograd.DeviceType.CUDA)


def peak_mb(fn: Callable[[], None], warm: int = 3) -> float:
    """Peak ALLOCATED MiB during one call, steady state already warm — i.e. the step's own
    transient working set, not the one-off state allocation."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def sync_free(fn: Callable[[], None], warm: int = 3) -> str:
    """"clean" when one call performs no CPU<->GPU synchronization, else "SYNC"."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        fn()
        torch.cuda.synchronize()
        return "clean"
    except RuntimeError:
        return "SYNC"
    finally:
        torch.cuda.set_sync_debug_mode("default")


def profile_row(name: str, fn: Callable[[], None]) -> None:
    print(f"    {name:>10}: {launches(fn):4d} launches  {peak_mb(fn):8.2f} MiB peak  "
          f"{sync_free(fn)}")


# ----------------------------------------------------------------- cases
CASES: dict[str, Callable[[argparse.Namespace], None]] = {}


def case(name: str) -> Callable[[Callable], Callable]:
    def deco(fn: Callable) -> Callable:
        CASES[name] = fn
        return fn
    return deco


def _two(params: list[torch.Tensor], flag: str, **kw: Any) -> tuple[Adakaon, Adakaon]:
    """Two optimizers over the same params differing only in ``flag`` (separate state, so
    neither arm's momentum history feeds the other's timing)."""
    base: dict[str, Any] = dict(lr=1e-3, weight_decay=0.01, fused=True)
    base.update(kw)
    off, on = Adakaon(params, **base), Adakaon(params, **base)
    setattr(off, flag, False)
    setattr(on, flag, True)
    return off, on


@case("lone_big")
def _lone_big(args: argparse.Namespace) -> None:
    """Opt 1: the per-tensor chunked kernel vs the batched one at N == 1."""
    print("\n## lone big tensor -> batched (N=1) kernel   [per-tensor / batched]")
    shapes = [(320 * k, 320) for k in range(1, 9)] + [(640, 640 * k) for k in range(1, 5)]
    for dt in [torch.float32, torch.bfloat16]:
        ps = bag(shapes, dt)
        off, on = _two(ps, "_fused_big_lone_batched")
        print(f"  bag of {len(shapes)} DISTINCT big shapes, {dt} "
              "(one tensor per shape bucket - the UNet/DiT case)")
        profile_row("per-tensor", off.step)
        profile_row("batched", on.step)
        show(f"unique-shape big bag {dt}", off.step, on.step, args.reps)
        del ps, off, on
        torch.cuda.empty_cache()
    for shape in [(1024, 1024), (4096, 4096)]:
        for md in ["bfloat16", "4bit", "int8"]:
            ps = bag([shape], torch.float32)
            off, on = _two(ps, "_fused_big_lone_batched", momentum_dtype=md)
            profile_row("per-tensor", off.step)
            profile_row("batched", on.step)
            show(f"lone {shape} mom={md}", off.step, on.step, args.reps)
            del ps, off, on
            torch.cuda.empty_cache()


@case("native")
def _native(args: argparse.Namespace) -> None:
    """Opts 2/3/4: the native factored foreach bucket, one toggle at a time, cumulative.

    ``foreach_stack_budget`` is PINNED so the chunking is identical across arms — the
    default adapts to free VRAM, which wobbles every step and would differ between them.
    """
    flags = ["_eps1_on_means", "_native_rms_matvec", "_write_fold_lr"]
    stages = [("base", (False, False, False)), ("eps1", (True, False, False)),
              ("matvec", (True, True, False)), ("foldlr", (True, True, True))]
    for dt in [torch.float32, torch.bfloat16]:
        print(f"\n## native factored foreach, params={dt}, mom=bf16  [previous stage / this stage]")
        for n, shape in [(400, (64, 64)), (200, (256, 256)), (50, (512, 512)),
                         (8, (1024, 1024))]:
            ps = bag([shape] * n, dt)
            opts = {}
            for name, fl in stages:
                o = Adakaon(ps, lr=1e-3, weight_decay=0.01, fused=False,
                            momentum_dtype="bfloat16", foreach_stack_budget=8_000_000)
                for f, v in zip(flags, fl, strict=True):
                    setattr(o, f, v)
                opts[name] = o
            print(f"  {n}x{shape}")
            for i in range(1, len(stages)):
                prev, cur = stages[i - 1][0], stages[i][0]
                show(f"+{cur}", opts[prev].step, opts[cur].step, args.reps)
            show("CUMULATIVE", opts["base"].step, opts["foldlr"].step, args.reps)
            del ps, opts
            torch.cuda.empty_cache()


@case("int8")
def _int8(args: argparse.Namespace) -> None:
    """Opt 5: in-kernel int8 for the batched big path vs the codec fp32-temp fallback."""
    print("\n## batched big int8 momentum   [codec fallback / direct in-kernel]")
    for n, shape in [(100, (256, 256)), (40, (512, 512)), (8, (1024, 1024))]:
        for dt in [torch.float32, torch.bfloat16]:
            ps = bag([shape] * n, dt)
            off, on = _two(ps, "_direct_int8", momentum_dtype="int8")
            print(f"  {n}x{shape} {dt}")
            profile_row("codec", off.step)
            profile_row("direct", on.step)
            show(f"{n}x{shape} {dt}", off.step, on.step, args.reps)
            del ps, off, on
            torch.cuda.empty_cache()


@case("scratch")
def _scratch(args: argparse.Namespace) -> None:
    """Opt 6: the four per-bucket launches removed (three ``zero_()`` + ``_finish_rms``).

    Timed in ISOLATION rather than through a toggle: the optimization is unconditional (the
    packed scratch has no second layout to switch to), so the honest measurement is the cost
    of the launches themselves, replayed at the real per-bucket counts. Their FIXED cost is
    what a many-bucket step pays over and over.
    """
    print("\n## per-bucket launch overhead removed   [3 zero_ + _finish_rms / 1 zero_]")
    N, R, C = 2, 256, 256
    for nb in [1, 10, 40]:
        zs = [torch.zeros(N * C + 2 * N, device=DEV) for _ in range(nb)]
        col = [z[:N * C] for z in zs]
        rms = [z[N * C:N * C + N] for z in zs]
        keep = [z[N * C + N:].view(torch.int32) for z in zs]
        inv = [torch.empty(N, device=DEV) for _ in range(nb)]
        fn = ft.triton.next_power_of_2(N)

        def old(nb=nb, col=col, rms=rms, keep=keep, inv=inv, fn=fn):
            for i in range(nb):
                col[i].zero_()
                rms[i].zero_()
                keep[i].zero_()
                ft._finish_rms[(1,)](rms[i], inv[i], R * C, 1.0, N, BLOCK=fn)

        def new(nb=nb, zs=zs):
            for i in range(nb):
                zs[i].zero_()

        show(f"{nb} bucket(s)", old, new, args.reps)
        del zs, col, rms, keep, inv
        torch.cuda.empty_cache()
    print("  (also: reduction scratch 0.98->0.59 MiB for 100x(512,512), "
          "9.78->5.87 MiB for 1000x(512,512))")


@case("sr")
def _sr(args: argparse.Namespace) -> None:
    """Opt 7: the torch stochastic-rounding chain vs one Triton kernel."""
    print("\n## bf16 stochastic-rounding write   [torch chain / Triton kernel]")
    for numel in [1 << 20, 1 << 22, 13_000_000]:
        p = torch.randn(numel, device=DEV, dtype=torch.bfloat16)
        d = torch.randn(numel, device=DEV)

        def a(p=p, d=d):
            bk._sr_write_(p, d, -1e-3, triton=False)

        def b(p=p, d=d):
            bk._sr_write_(p, d, -1e-3, triton=True)

        print(f"  add_stochastic_ numel={numel}")
        profile_row("torch", a)
        profile_row("triton", b)
        show(f"numel={numel}", a, b, args.reps)
        del p, d
        torch.cuda.empty_cache()
    pv = [torch.randn(256, 256, device=DEV, dtype=torch.bfloat16) for _ in range(200)]
    delta = torch.randn(200, 256, 256, device=DEV)

    def a():
        bk.subtract_batched_(pv, delta, "stochastic_rounding", alpha=1e-3, triton=False)

    def b():
        bk.subtract_batched_(pv, delta, "stochastic_rounding", alpha=1e-3, triton=True)

    print("  subtract_batched_ 200x(256,256) bf16")
    profile_row("torch", a)
    profile_row("triton", b)
    show("subtract_batched_ 200x(256,256)", a, b, args.reps)


@case("fourbit")
def _fourbit(args: argparse.Namespace) -> None:
    """Item 9a: the 4-bit per-block absmax, NB-loop vs one axis reduction.

    The ``exact4`` flag is flipped on the CACHED pointer arrays after the first step, so
    both arms are the same build and the same bucketing.
    """
    print("\n## one-block 4-bit requant   [NB loop / single-axis reduction], "
          "then 4bit vs bf16")
    for shape in [(64, 128), (128, 64), (16, 512), (64, 64), (32, 64)]:
        ps = bag([shape] * 200, torch.float32)
        loop = Adakaon(ps, lr=1e-3, weight_decay=0.01, fused=True, momentum_dtype="4bit")
        fast = Adakaon(ps, lr=1e-3, weight_decay=0.01, fused=True, momentum_dtype="4bit")
        bf16 = Adakaon(ps, lr=1e-3, weight_decay=0.01, fused=True, momentum_dtype="bfloat16")
        for o in (loop, fast, bf16):
            o.step()                                     # build the pointer caches
        for cache in loop._fused_ob_caches.values():
            for bucket in cache.buckets:
                bucket["exact4"] = False
                bucket["fblk"] = 0
        nb = (shape[0] * shape[1] + 127) // 128
        show(f"{shape} NB={nb} loop/reshape", loop.step, fast.step, args.reps)
        show(f"{shape} 4bit/bf16 after", fast.step, bf16.step, args.reps)
        del ps, loop, fast, bf16
        torch.cuda.empty_cache()


@case("determinism")
def _determinism(args: argparse.Namespace) -> None:
    """Item 9b: what ``deterministic_reductions=True`` costs, and the memory it pins."""
    print("\n## deterministic_reductions   [two-pass / fp32 atomics]  >1 = det is slower")
    for n, shape in [(236, (512, 512)), (100, (256, 256)), (8, (1024, 1024)),
                     (2, (4096, 4096))]:
        ps = bag([shape] * n, torch.float32)
        atomic = Adakaon(ps, lr=1e-3, weight_decay=0.01, fused=True)
        det = Adakaon(ps, lr=1e-3, weight_decay=0.01, fused=True,
                      deterministic_reductions=True)
        for o in (atomic, det):
            o.step()

        def scratch(o):
            return sum(c._zeros.numel() * 4 + c.rowsum.numel() * 4
                       + (c.rowmean.numel() * 4 if c.rowmean is not c.rowsum else 0)
                       + sum(t.numel() * 4 for t in getattr(c, "_partials", (0,))[1:])
                       for c in o._fused_big_caches.values()) / 2**20

        print(f"  {n}x{shape}: scratch {scratch(atomic):.2f} -> {scratch(det):.2f} MiB, "
              f"launches {launches(atomic.step)} -> {launches(det.step)}")
        show(f"{n}x{shape}", det.step, atomic.step, args.reps)
        del ps, atomic, det
        torch.cuda.empty_cache()


@case("realistic")
def _realistic(args: argparse.Namespace) -> None:
    """The reference mixed bag with every accepted toggle off vs on."""
    print("\n## realistic bag 200x(256,256) + 100x(512,) + 128 scalars   [all off / all on]")
    flags = ["_fused_big_lone_batched", "_eps1_on_means", "_write_fold_lr", "_direct_int8"]
    for fused in [False, True]:
        for dt in [torch.bfloat16, torch.float32]:
            ps = realistic_bag(dt)
            off = Adakaon(ps, lr=1e-3, weight_decay=0.01, fused=fused)
            on = Adakaon(ps, lr=1e-3, weight_decay=0.01, fused=fused)
            for f in flags:
                setattr(off, f, False)
                setattr(on, f, True)
            saved = bk.SR_TRITON
            try:
                def a(off=off):
                    bk.SR_TRITON = False
                    off.step()

                def b(on=on):
                    bk.SR_TRITON = True
                    on.step()

                show(f"fused={fused} {dt}", a, b, args.reps)
            finally:
                bk.SR_TRITON = saved
            del ps, off, on
            torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all", choices=[*CASES, "all"])
    ap.add_argument("--reps", type=int, default=150,
                    help="PAIRS per comparison. 150-400 for the few-percent native effects; "
                         "40 is plenty for the 3x fused ones.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    # Contention warning, NOT a gate. ``mem_get_info`` cannot attribute memory to a process
    # and this script's own CUDA context is already ~1 GiB by the time it runs, so any
    # threshold here would either fire on an idle card or miss a real neighbour. Print the
    # number and let the reader judge: the 3x fused comparisons survive a busy card, the
    # few-percent native ones do not (measured: per-rep spread went from ~5% idle to +-40%
    # against three other processes, which is why ``paired`` reports a CI at all).
    torch.zeros(1, device=DEV)                       # force context creation before measuring
    free, total = torch.cuda.mem_get_info()
    others = (total - free) / 2**30
    if others > 1.6 and not args.force:
        print(f"NOTE: {others:.2f} GiB in use on this GPU (this process included). If another "
              "job is on the card, treat every result narrower than ~1.10x as unresolved.")
    for name in (list(CASES) if args.case == "all" else [args.case]):
        CASES[name](args)


if __name__ == "__main__":
    main()
