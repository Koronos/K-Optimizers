"""What the state-identity guard costs, and why the counter beat the two witness fields.

The guard closes a hole no PARAMETER witness can see: a state buffer retired while every
param stands still (``del opt.state[p]``, ``opt.state[p].clear()``,
``opt.state[p]["m"] = ...``) left every fused pointer table and every cached foreach view
addressing it, and the step wrote it. Three candidates were priced; this file is the
measurement they were chosen on.

* **(a) a state-tensor field on the witness** — ``tuple(map(Tensor.data_ptr, (st["m"] for
  st in states)))``, i.e. one more per-param host sweep next to ``param_witness``' three,
  plus the ``states`` lookup the witness does not otherwise need. Complete for a rebind;
  BLIND to ``state[p].clear()`` + refill (the dict, the params and the pointers can all
  come back identical).
* **(b) a state-DICT-id field** — ``tuple(map(id, map(state.__getitem__, params)))``.
  Cheaper, and sees ``del opt.state[p]`` (a new dict object). Blind to BOTH
  ``state[p]["m"] = ...`` and ``state[p].clear()``, which keep the dict.
* **(c) a generation counter on ``self.state``** — SHIPPED. The state mapping itself
  counts every rebinding of a baked key (``kaon._foreach_plan.WatchedState``), so a cache
  compares ONE integer and there is no sweep at all. Complete for all three cases. Its
  cost is not a sweep but two much smaller things, both measured here: the per-step reads
  (``--case field``) and the write hook every ``state[...] = ...`` now goes through
  (``--case writes``, which is where AdaPNM's per-param ``state["step"] += 1`` lands).

``--case profile`` is the other half of the claim: the guard must add no CUDA kernel, no
launch and no synchronisation, only host work.

READING THE NUMBERS. The absolutes here are one machine and do not travel: the same script
on a second machine returns roughly half of them (e.g. (a) +110 vs +144 µs, the write hook
+56 vs +107 µs). The ORDERING of the candidates and every conclusion drawn from it were
identical on both. The counter's own row is below this harness's noise floor either way —
its CI straddles zero — so it is reported as ACCOUNTING (``--case calls`` x a direct
per-call timing), and the load-independent evidence (``--case profile``, plus the bytecode
count in the CHANGELOG) is what the cost claim actually rests on.

Usage::

    python benchmarks/fused/bench_state_witness.py --case field
    python benchmarks/fused/bench_state_witness.py --case writes
    python benchmarks/fused/bench_state_witness.py --case calls
    python benchmarks/fused/bench_state_witness.py --case profile
"""
from __future__ import annotations

import argparse
import math
import statistics as st
import time
from collections.abc import Callable

import torch
from torch import Tensor

import kaon._foreach_plan as fp
import kaon.adakaon as ak
import kaon.adapnm as ap
from kaon import Adakaon, AdaPNM
from kaon._foreach_plan import WatchedState, state_generation

DEV = "cuda"
_ID, _PTR, _CONTIG = id, Tensor.data_ptr, Tensor.is_contiguous

BAGS: dict[str, list[tuple[int, ...]]] = {
    # bench_perf_a2's / bench_shape_witness' reference LoRA-adapter bag.
    "lora428": [(256, 256)] * 200 + [(512,)] * 100 + [()] * 128,
    # Pure 0-D: the launch-bound extreme, where host work IS the step.
    "zero448": [()] * 448,
    "unet": ([(320, 320)] * 8 + [(320, 1280)] * 8 + [(1280, 320)] * 8 + [(640, 640)] * 4
             + [(320,)] * 40 + [(1280,)] * 12),
}

CFG = dict(lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01, momentum_dtype="bfloat16")


def bag(shapes: list[tuple[int, ...]], dtype: torch.dtype = torch.bfloat16,
        seed: int = 0) -> list[Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = []
    for sh in shapes:
        p = torch.randn(sh, generator=g).to(DEV).to(dtype).requires_grad_(True)
        p.grad = torch.randn(sh, generator=g).to(DEV).to(dtype)
        out.append(p)
    return out


# ----------------------------------------------------------------- candidates
_GET_M = lambda s: s["m"]           # noqa: E731 — priced as a lambda on purpose (see below)


def w_base(pl, state) -> tuple:
    """main / 0.7.13: (ids, data_ptrs, contiguity) over the params only."""
    return (tuple(map(_ID, pl)), tuple(map(_PTR, pl)), tuple(map(_CONTIG, pl)))


def w_state_ptrs(pl, state) -> tuple:
    """(a) + the momentum buffers' addresses."""
    get = state.__getitem__
    return (tuple(map(_ID, pl)), tuple(map(_PTR, pl)), tuple(map(_CONTIG, pl)),
            tuple(map(_PTR, map(_GET_M, map(get, pl)))))


def w_state_ids(pl, state) -> tuple:
    """(b) + the per-param state DICTS' ids."""
    get = state.__getitem__
    return (tuple(map(_ID, pl)), tuple(map(_PTR, pl)), tuple(map(_CONTIG, pl)),
            tuple(map(_ID, map(get, pl))))


def w_generation(pl, state) -> tuple:
    """(c) SHIPPED — the state mapping's own counter, read once."""
    return (tuple(map(_ID, pl)), tuple(map(_PTR, pl)), tuple(map(_CONTIG, pl)),
            state_generation(state))


VARIANTS: dict[str, Callable[[list, object], tuple]] = {
    "(a) +state m data_ptr": w_state_ptrs,
    "(b) +state dict id": w_state_ids,
    "(c) +generation  [SHIPPED]": w_generation,
}


# ----------------------------------------------------------------- statistics
def host_us(fn: Callable[[], object], reps: int = 200, blocks: int = 9) -> float:
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
    """PAIRED host cost of ``var`` over ``base`` in µs per call, with a 95% CI.

    Same design as ``bench_shape_witness.host_delta_us``: the two arms are timed ADJACENT
    with the order alternating, so whatever else the CPU is doing hits both halves of a
    pair alike (separate passes produced negative field costs on this machine).
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


def step_us(run: Callable[[], None], reps: int = 60) -> tuple[float, float]:
    """``(min, median)`` synchronize-bracketed wall µs of one ``run()``."""
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
    """Price the three candidates as the per-step host work each one adds."""
    print("\n## candidate state-staleness fields — host cost of ONE per-step check")
    print("   PAIRED against the 3-field param witness, mean of 25 pairs with a 95% CI")
    for name, shapes in BAGS.items():
        pl = bag(shapes)
        opt = Adakaon(pl, fused=False, **CFG)
        opt.step()                                   # allocate state
        state = opt.state
        base_us = host_us(lambda pl=pl, s=state: w_base(pl, s), args.reps)
        nmin, _ = step_us(opt.step, args.step_reps)
        pf = bag(shapes, seed=1)
        fmin, _ = step_us(Adakaon(pf, fused=True, **CFG).step, args.step_reps)
        print(f"\n  bag {name}: {len(pl)} params — 3-field param witness {base_us:.1f} µs; "
              f"step min: fused {fmin:.1f} µs, native {nmin:.1f} µs")
        for label, fn in VARIANTS.items():
            d, lo, hi = host_delta_us(lambda pl=pl, s=state: w_base(pl, s),
                                      lambda fn=fn, pl=pl, s=state: fn(pl, s), args.reps)
            print(f"    {label:<28} extra {d:+8.2f} µs [{lo:+7.2f},{hi:+7.2f}]   "
                  f"= {d / fmin * 100:+6.3f}% of the FUSED step, "
                  f"{d / nmin * 100:+6.3f}% of the native one")


def case_writes(args: argparse.Namespace) -> None:
    """The write hook: what ``WatchedParamState.__setitem__`` costs the ONE per-step
    state write kaon actually makes (AdaPNM's per-param bias-correction counter).

    Adakaon makes none, so its steady-state write cost is exactly zero; this is priced on
    AdaPNM's 428 ``state["step"] += 1`` per step, against plain dicts.
    """
    print("\n## the write hook — per-step cost of N `state[\"step\"] += 1`")
    print("   PAIRED against the same loop over PLAIN dicts, 25 pairs with a 95% CI")
    set_raw = dict.__setitem__
    for name, shapes in BAGS.items():
        n = len(shapes)
        watched = WatchedState()
        plain: list[dict] = []
        keys = [torch.zeros(1) for _ in range(n)]
        for k in keys:
            watched[k]["step"] = 0
            plain.append({"step": 0})
        w_states = [watched[k] for k in keys]

        def bump_hooked(states=w_states):
            for s in states:
                s["step"] += 1                      # REJECTED: a Python-level __setitem__

        def bump_bypass(states=w_states, set_raw=set_raw):
            for s in states:
                set_raw(s, "step", s["step"] + 1)   # SHIPPED: dict.__setitem__

        def bump_plain(states=plain):
            for s in states:
                s["step"] += 1

        def bump_plain_call(states=plain, set_raw=set_raw):
            # The shipped CALL FORM on a plain dict: isolates "the unbound-method call"
            # from "the dict is a subclass". The subclass costs nothing on a read or on a
            # C-level write (same slots); the call form is the whole difference.
            for s in states:
                set_raw(s, "step", s["step"] + 1)

        pl = bag(shapes)
        opt = AdaPNM(pl, fused=False, **CFG)
        opt.step()
        nmin, _ = step_us(opt.step, args.step_reps)
        pf = bag(shapes, seed=1)
        fmin, _ = step_us(AdaPNM(pf, fused=True, **CFG).step, args.step_reps)
        print(f"\n  bag {name}: {n} state dicts — AdaPNM step min: "
              f"fused {fmin:.1f} µs, native {nmin:.1f} µs")
        for label, fn in (("through the hook [REJECTED]", bump_hooked),
                          ("dict.__setitem__ [SHIPPED]", bump_bypass),
                          ("  ...of which the call form", bump_plain_call)):
            d, lo, hi = host_delta_us(bump_plain, fn, args.reps)
            print(f"    {label:<28} extra {d:+8.2f} µs [{lo:+7.2f},{hi:+7.2f}]   "
                  f"= {d / fmin * 100:+6.3f}% of the FUSED step, "
                  f"{d / nmin * 100:+6.3f}% of the native one")


def case_calls(args: argparse.Namespace) -> None:
    """How many times the guard is read per step, per route — the accounting multiplier."""
    print("\n## state_generation() calls per step (the accounting multiplier)")
    real = fp.state_generation
    for name, shapes in BAGS.items():
        for kind in ("adakaon", "adapnm"):
            for fused in (False, True):
                pl = bag(shapes)
                cls = Adakaon if kind == "adakaon" else AdaPNM
                opt = cls(pl, fused=fused, **CFG)
                opt.step()
                opt.step()
                counter = {"n": 0}

                def counted(state, counter=counter, real=real):
                    counter["n"] += 1
                    return real(state)

                fp.state_generation = counted
                ak.state_generation = counted
                ap.state_generation = counted
                try:
                    opt.step()
                finally:
                    fp.state_generation = real
                    ak.state_generation = real
                    ap.state_generation = real
                tag = "fused" if fused else "native"
                print(f"  {name:9} {kind:8} {tag:7} {counter['n']:3d} calls/step")


def case_profile(args: argparse.Namespace) -> None:
    """No new kernel, no new launch, no new synchronisation — only host work.

    Reports the profiler's CUDA-kernel count, total launches and the
    ``cudaDeviceSynchronize`` / ``cudaMemcpy`` counts for one step of each route. Compare
    the numbers against the same run on the pre-guard tree; the point of the case is that
    every CUDA-side column is identical and only the CPU total moves.
    """
    from torch.profiler import ProfilerActivity, profile
    print("\n## torch.profiler — one step per config")
    for name, shapes in BAGS.items():
        for kind in ("adakaon", "adapnm"):
            for fused in (False, True):
                pl = bag(shapes)
                cls = Adakaon if kind == "adakaon" else AdaPNM
                opt = cls(pl, fused=fused, **CFG)
                for _ in range(5):
                    opt.step()
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
                    opt.step()
                    torch.cuda.synchronize()
                evs = pr.key_averages()
                kernels = sum(e.count for e in evs if e.device_type.name == "CUDA"
                              and not e.key.startswith("cuda"))
                launches = sum(e.count for e in evs if e.key == "cudaLaunchKernel")
                syncs = sum(e.count for e in evs
                            if e.key in ("cudaDeviceSynchronize", "cudaStreamSynchronize",
                                         "cudaMemcpyAsync", "cudaMemcpy"))
                cpu_us = sum(e.self_cpu_time_total for e in evs) / 1e3
                tag = "fused" if fused else "native"
                print(f"  {name:9} {kind:8} {tag:7} kernels {kernels:5d}  launches {launches:5d}"
                      f"  syncs/copies {syncs:4d}  self CPU {cpu_us:9.1f} µs")


CASES = {"field": case_field, "writes": case_writes, "calls": case_calls,
         "profile": case_profile}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", choices=sorted(CASES), default="field")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--step-reps", type=int, default=60)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("this benchmark needs CUDA")
    CASES[args.case](args)


if __name__ == "__main__":
    main()
