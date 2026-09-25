"""What the optimizer itself costs, per step, at diffusion scale.

The question this answers is narrow on purpose: *for a fine-tune whose tensor shapes look
like SDXL/Anima, how much wall clock, how much VRAM and how many bytes of state does the
optimizer add?* It does not measure quality — nothing here trains anything to a loss.
Quality lives in the Anima campaign ([anima/campaign.py](anima/campaign.py)); this file is
the cost half of the same evidence, and the two are meant to be read together.

Design notes, i.e. why the numbers can be trusted:

* **One bag, one seed, every arm.** ``sdxl_bag(R)`` is a parametric stand-in for the
  trainable tensors of an SDXL-shaped UNet: 12R attention/ff squares (1280x1280), 8R fat
  MLP rectangles (640x5120 and its transpose), 14R conv kernels and 120R 1-D norm/bias
  vectors. ``R=3`` is 468 tensors / 397.9 M params; ``R=5`` (663 M) is the intended
  measurement scale because two bf16 arms still fit side by side in 8 GB. Gradients are
  random and *fixed* — the kernel does the same work every rep, which is what makes the
  clock representative.

* **Paired, alternating, with a CI.** Absolute ms/step on a laptop GPU drifts with
  temperature and with whatever else holds the card. Every arm is therefore also timed
  back-to-back against ``adamw_bf16`` with the order alternating ABAB, and reported as the
  geometric mean of the per-pair ratio with a 95% CI (the design of
  ``benchmarks/fused/bench_perf_a2.py::paired``, extended here to keep the absolute
  samples as well so one pass yields both). A ratio whose CI straddles 1.0 is printed
  ``n.s.`` and means exactly that: not resolvable, not "equal".

* **Two AdamW arms, because "AdamW costs 8 B/p" is only true of fp32 training.**
  ``torch.optim.AdamW`` given bf16 params allocates bf16 exp_avg/exp_avg_sq, i.e. **4
  B/p** — that is ``adamw_bf16``, and it is what a user gets who simply casts the model
  and does not reach for kaon (no stochastic rounding, so its own bf16 updates are lossy;
  that cost shows up in quality, not here). ``adamw_fp32`` is the other honest baseline:
  pure fp32 training — 4 B/p weights + 4 B/p grad + 8 B/p state = 16 B/p resident, no
  bf16 copy anywhere — and it is the configuration the 8 B/p folklore refers to. Both are
  reported; the paired ratios use ``adamw_bf16``.

* **Peak VRAM is measured with one arm resident at a time**, after the state has been
  allocated (state is lazy: it appears on the first ``step()``), so the number is
  "resident state + this step's transient", not a half-built optimizer.

Usage (from the worktree root, ``PYTHONPATH=src``)::

    python benchmarks/nekaon_evidence/step_cost.py --R 5 --reps 150
    python benchmarks/nekaon_evidence/step_cost.py --R 5 --reps 60 --fraction --C 256
    python benchmarks/nekaon_evidence/step_cost.py --capacity --capacity-max 12
    python benchmarks/nekaon_evidence/step_cost.py --render-only results/step_cost_R5.json

Timing runs are refused on battery: this laptop drops from a 60 W to a 35 W power limit
when unplugged, which moves the clock by more than any effect measured here.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics as st
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]
DEV = "cuda"
BASELINE = "adamw_bf16"
#: The paired phase holds two arms at once. ``adamw_bf16`` is ~8 B/p resident (bf16
#: weights + bf16 grads + 4 B/p state + transient), so on an 8 GB card R=3 is the largest
#: scale at which a second bf16 arm reliably fits beside it; R=5 is fine for the solo
#: phase, where one arm is resident. Override with ``--paired-R`` on a bigger card.
DEFAULT_PAIRED_R = 3

# ----------------------------------------------------------------- the bag
#: Shapes of one SDXL-like "unit"; ``sdxl_bag(R)`` repeats each count R times. The 1-D
#: entries stand for GroupNorm/LayerNorm weights and biases at the two dominant widths.
UNIT_SHAPES: tuple[tuple[int, tuple[int, ...]], ...] = (
    (12, (1280, 1280)),        # attention projections / ff squares
    (4, (640, 5120)),          # ff up
    (4, (5120, 640)),          # ff down
    (4, (1280, 1280, 3, 3)),   # deep conv blocks
    (6, (640, 640, 3, 3)),
    (6, (320, 320, 3, 3)),
    (60, (1280,)),             # norms / biases
    (60, (640,)),
)


def bag_shapes(R: int) -> list[tuple[int, ...]]:
    """The full shape list for scale ``R`` (no allocation — cheap enough to test on CPU)."""
    if R < 1:
        raise ValueError(f"R must be >= 1, got {R}")
    out: list[tuple[int, ...]] = []
    for count, shape in UNIT_SHAPES:
        out.extend([shape] * (count * R))
    return out


def bag_numel(R: int) -> int:
    """Trainable parameter count at scale ``R``."""
    return sum(math.prod(s) for s in bag_shapes(R))


def sdxl_bag(R: int, dtype: torch.dtype, seed: int = 0) -> list[torch.Tensor]:
    """Leaf params with fixed random grads attached, same values for every arm.

    The generator is seeded once and driven in a fixed shape order, so two arms at the
    same ``R`` see identical inputs regardless of which one ran first.
    """
    gen = torch.Generator(device=DEV).manual_seed(seed)
    params = []
    for shape in bag_shapes(R):
        p = torch.randn(shape, generator=gen, dtype=torch.float32, device=DEV).to(dtype)
        p.requires_grad_(True)
        p.grad = torch.randn(shape, generator=gen, dtype=torch.float32, device=DEV).to(dtype)
        params.append(p)
    return params


# ----------------------------------------------------------------- arms
def _kaon_common(fused: bool) -> dict[str, Any]:
    """The Anima fine-tune configuration (benchmarks/anima/generate_comparison.py)."""
    return dict(
        lr=1e-4,
        betas=(0.5, 0.999),
        weight_decay=0.1,
        cautious=True,
        gradient_centralization=True,
        bf16_method="stochastic_rounding",
        auto_lr=False,
        fused=fused,
    )


def _adakaon(params, momentum_dtype: str, fused: bool):
    from kaon import Adakaon

    return Adakaon(params, clip_threshold=1.0, momentum_dtype=momentum_dtype,
                   **_kaon_common(fused))


def _nekaon(params, momentum_dtype: str, fused: bool, k: float = 1.5):
    from kaon import Nekaon

    return Nekaon(params, k=k, momentum_dtype=momentum_dtype, **_kaon_common(fused))


def _adamw(params, fused: bool = True):
    return torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01,
                             eps=1e-8, fused=fused)


#: name -> (param dtype, builder(params, native) -> optimizer, note for the report)
ARMS: dict[str, tuple[torch.dtype, Callable[[Any, bool], Any], str]] = {
    "adamw_bf16": (
        torch.bfloat16,
        lambda p, native: _adamw(p, fused=True),
        "torch.optim.AdamW(fused) on bf16 params: state is bf16 too, 4 B/p. No stochastic "
        "rounding — the cheap thing people actually do. Baseline of the paired ratios.",
    ),
    "adamw_fp32": (
        torch.float32,
        lambda p, native: _adamw(p, fused=True),
        "Plain fp32 training: fp32 weights (4 B/p) + fp32 grad (4 B/p) + fp32 AdamW state "
        "(8 B/p) = 16 B/p resident. There is no bf16 copy and hence no master weight — the "
        "8 B/p is the STATE alone, which is what the 'AdamW costs 8 bytes per parameter' "
        "folklore means. Only fits at small R.",
    ),
    "adakaon_bf16": (
        torch.bfloat16,
        lambda p, native: _adakaon(p, "bfloat16", not native),
        "Adakaon: bf16 momentum + factored second moment, stochastic-rounded bf16 writes.",
    ),
    "adakaon_4bit": (
        torch.bfloat16,
        lambda p, native: _adakaon(p, "4bit", not native),
        "Adakaon with 4-bit quantized momentum.",
    ),
    "nekaon_bf16": (
        torch.bfloat16,
        lambda p, native: _nekaon(p, "bfloat16", not native),
        "Nekaon k=1.5 over the bf16-momentum Adakaon: the negative-momentum lookahead on "
        "top of the same inner step.",
    ),
    "nekaon_4bit": (
        torch.bfloat16,
        lambda p, native: _nekaon(p, "4bit", not native),
        "Nekaon k=1.5 with 4-bit momentum — the Anima configuration.",
    ),
    "nekaon_k0_4bit": (
        torch.bfloat16,
        lambda p, native: _nekaon(p, "4bit", not native, k=0.0),
        "Nekaon with k=0: the lookahead is inert, so the gap to nekaon_4bit isolates what "
        "the lookahead itself costs and the gap to adakaon_4bit is the wrapper's own tax.",
    ),
}

DEFAULT_ARMS: tuple[str, ...] = tuple(ARMS)


def disarm_inert_telemetry(opt):
    """Silence MSAM's inert-lookahead heuristic, so no arm is charged for it.

    ``MSAM._warn_if_inert`` (``src/kaon/msam.py``) samples weight scales every
    ``inert_check_interval`` climbs (default 10) for at most ``_INERT_MAX_CHECKS`` (200)
    climbs, and each sample converts device reductions to Python floats — a host sync. It
    is skipped entirely when ``rho == 0``, so only ``nekaon_bf16``/``nekaon_4bit`` would
    pay it: ~``reps / 10`` syncs over a 60-rep phase that ``adamw_*``, ``adakaon_*`` and
    ``nekaon_k0_4bit`` (k=0 -> rho=0) never pay, charged by the geometric mean to the
    update itself. Setting the "already warned" latch is the heuristic's own documented
    short-circuit: the check returns before it even increments its counter.

    What this excludes is a BOUNDED cost of the first ~2000 steps of a run (200 checks at
    interval 10), not a per-step tax — the point here is the update in regime. In the
    Anima campaign, which is 200 steps long, the telemetry IS paid and is part of the wall
    clock reported there.

    A no-op on arms that have no such heuristic, which is what makes it uniform: it is
    called for every arm, and every arm ends up measured without it.
    """
    for o in (opt, getattr(opt, "inner", None)):
        if o is not None and hasattr(o, "_inert_warned"):
            o._inert_warned = True
    return opt


def build_arm(name: str, R: int, native: bool, seed: int = 0):
    """``(params, optimizer)`` for one arm, allocated fresh, telemetry disarmed."""
    dtype, factory, _ = ARMS[name]
    params = sdxl_bag(R, dtype, seed=seed)
    return params, disarm_inert_telemetry(factory(params, native))


# ----------------------------------------------------------------- statistics
#: Two-sided 95% Student-t critical values by degrees of freedom (n-1); the same table
#: ``benchmarks/control/battery.py::ci95`` uses, so the two reports quote intervals built
#: the same way. Past df=30 the normal approximation (1.96) is within 1.5%.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
        23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
        30: 2.042}


def gmean_ci(logs: Sequence[float]) -> tuple[float, float, float]:
    """Geometric mean of ratios from their logs, with a 95% CI (as in ``bench_perf_a2``).

    ``bench_perf_a2`` hardcodes z=1.96, which is the large-sample limit and is fine at the
    default 60 reps. A short ``--reps`` run would understate the interval, so the critical
    value comes from ``_T95`` (Student-t, n-1 d.o.f.) and falls back to 1.96 past df=30 —
    the same rule as ``battery.py::ci95``, so intervals are comparable between the two
    reports.
    """
    m = st.fmean(logs)
    if len(logs) < 2:
        return math.exp(m), float("nan"), float("nan")
    half = _T95.get(len(logs) - 1, 1.96) * st.stdev(logs) / math.sqrt(len(logs))
    return math.exp(m), math.exp(m - half), math.exp(m + half)


def verdict_of(lo: float, hi: float) -> str:
    """``slower``/``faster``/``n.s.`` for a ratio arm/baseline whose CI is ``[lo, hi]``."""
    if lo != lo or hi != hi:  # NaN
        return "n.s."
    if lo > 1.0:
        return "slower"
    if hi < 1.0:
        return "faster"
    return "n.s."


#: Warmup pairs before the measured loop. The paired phase runs at ``--paired-R``, i.e. a
#: different bag from the solo phase, so every Triton kernel re-autotunes on its first
#: calls at the new shapes; 10 pairs left part of that autotune inside the samples.
DEFAULT_PAIRED_WARM = 30
#: ...and the first few measured pairs still ride the clock/power ramp back up, so they
#: are dropped from BOTH arms (dropping whole pairs keeps the pairing intact).
DEFAULT_PAIRED_DISCARD = 5


def paired_samples(a: Callable[[], None], b: Callable[[], None], reps: int,
                   warm: int = DEFAULT_PAIRED_WARM,
                   discard: int = DEFAULT_PAIRED_DISCARD) -> tuple[list[float], list[float]]:
    """Per-pair wall times (ms) of ``a`` and ``b``, run adjacent with alternating order.

    ``benchmarks/fused/bench_perf_a2.py::paired`` returns only the ratio; the absolute
    samples are wanted here too, and running the loop twice would double an already long
    measurement, so its loop is repeated here with the samples kept.

    The first ``discard`` measured pairs are dropped from both arms; a run too short to
    afford that (``reps <= 2 * discard``, i.e. a smoke test) keeps all of them rather than
    returning almost nothing.
    """
    for _ in range(warm):
        a()
        b()
    torch.cuda.synchronize()
    ta: list[float] = []
    tb: list[float] = []
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
        ta.append(1000.0 * xa)
        tb.append(1000.0 * xb)
    drop = discard if reps > 2 * discard else 0
    return ta[drop:], tb[drop:]


def median_ms(fn: Callable[[], None], reps: int, warm: int = 5) -> float:
    """Median ms of ``fn``, each rep bracketed by a synchronize."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(1000.0 * (time.perf_counter() - t0))
    return st.median(samples)


def launches(fn: Callable[[], None], warm: int = 3) -> int:
    """CUDA kernel launches for ONE call (contention-immune, unlike the clock)."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)


def state_bytes_per_param(opt, params) -> float:
    """Optimizer-state bytes/param, walking the ``.inner`` chain of wrappers.

    Same accounting as ``benchmarks/proxy/harness.py::opt_state_bytes_per_param``, inlined
    so this script imports nothing that pulls in the proxy dataset.
    """
    total = 0
    o = opt
    while o is not None:
        for entry in o.state.values():
            for v in entry.values():
                if torch.is_tensor(v):
                    total += v.numel() * v.element_size()
        o = getattr(o, "inner", None)
    return total / max(1, sum(p.numel() for p in params))


# ----------------------------------------------------------------- environment
def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env dependent
        return f"<unavailable: {exc}>"
    return (out.stdout or out.stderr).strip()


def battery_status() -> str:
    """Raw ``Win32_Battery.BatteryStatus`` text ("" when the machine has no battery)."""
    return _run(["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Battery).BatteryStatus"])


def on_ac(status: str) -> tuple[bool, str]:
    """``(ok, explanation)``. 2 = on AC; no battery at all = a desktop, also fine."""
    text = status.strip()
    if not text:
        return True, "no battery reported (desktop, or CIM unavailable)"
    first = text.splitlines()[0].strip()
    if first == "2":
        return True, "on AC (BatteryStatus=2)"
    return False, f"on battery (BatteryStatus={first!r})"


def collect_meta() -> dict[str, Any]:
    import kaon

    try:
        import triton

        triton_version = triton.__version__
    except Exception:  # pragma: no cover - triton is present in this environment
        triton_version = None
    meta: dict[str, Any] = {
        "kaon_version": kaon.__version__,
        "kaon_file": str(Path(kaon.__file__).resolve()),
        "commit": _run(["git", "-C", str(WORKTREE), "rev-parse", "HEAD"]),
        "branch": _run(["git", "-C", str(WORKTREE), "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(_run(["git", "-C", str(WORKTREE), "status", "--porcelain"])),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton_version,
        "python": sys.version.split()[0],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["gpu_total_bytes"] = props.total_memory
        meta["gpu_capability"] = f"{props.major}.{props.minor}"
    meta["nvidia_smi"] = _run([
        "nvidia-smi",
        "--query-gpu=power.limit,power.max_limit,power.default_limit,clocks.max.sm,"
        "temperature.gpu",
        "--format=csv,noheader",
    ])
    status = battery_status()
    ok, why = on_ac(status)
    meta["battery_status"] = status
    meta["on_ac"] = ok
    meta["power_note"] = why
    return meta


def reclaim() -> None:
    """Collect, return the cached blocks to the driver and re-arm the peak counters.

    It takes no arguments on purpose. The obvious ``free_all(*objs)`` helper does not
    work: ``del o`` inside the helper unbinds only the *helper's* local name, so the
    caller's reference keeps the arm alive and ``empty_cache()`` runs with the previous
    optimizer still resident. Its blocks then stay reserved and land in the NEXT arm's
    ``peak_reserved_bytes`` and, in the capacity sweep, in the next arm's OOM point —
    which, with ``nekaon_*`` last in ``ARMS``, biases exactly the arms under test.
    Callers must therefore drop their own references (including bound methods and
    closures, which hold the optimizer) BEFORE calling this.
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ----------------------------------------------------------------- phases
def measure_solo(names: Sequence[str], R: int, native: bool, reps: int,
                 want_launches: bool) -> dict[str, dict[str, Any]]:
    """Per-arm memory, state bytes and a solo (unpaired) median ms, one arm resident.

    An arm that does not fit at this ``R`` at all (``adamw_fp32`` needs 16 B/p, i.e.
    10.6 GB at R=5) is recorded as ``"oom": true`` and the sweep continues.
    """
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        params = opt = step = None
        try:
            params, opt = build_arm(name, R, native)
            step = opt.step
            for _ in range(3):  # state is lazy: it appears on the first step
                step()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base_alloc = torch.cuda.memory_allocated()
            step()
            torch.cuda.synchronize()
            rec = {
                "param_dtype": str(ARMS[name][0]).replace("torch.", ""),
                "params": sum(p.numel() for p in params),
                "tensors": len(params),
                "state_bytes_per_param": state_bytes_per_param(opt, params),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "resident_before_step_bytes": base_alloc,
                "step_transient_bytes": torch.cuda.max_memory_allocated() - base_alloc,
                "peak_bytes_per_param": (torch.cuda.max_memory_allocated()
                                         / max(1, sum(p.numel() for p in params))),
                "ms_step_solo": median_ms(step, reps),
                "note": ARMS[name][2],
            }
            if want_launches:
                rec["cuda_launches"] = launches(step)
            out[name] = rec
            print(f"  {name:<18} {rec['ms_step_solo']:7.2f} ms  "
                  f"{rec['state_bytes_per_param']:5.2f} B/p  "
                  f"peak {rec['peak_allocated_bytes'] / 2**30:5.2f} GiB"
                  + (f"  {rec['cuda_launches']:5d} launches" if want_launches else ""))
        except torch.OutOfMemoryError:
            out[name] = {"param_dtype": str(ARMS[name][0]).replace("torch.", ""),
                         "R": R, "oom": True, "note": ARMS[name][2]}
            print(f"  {name:<18} OOM at R={R} — does not fit on this GPU")
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            out[name] = {"param_dtype": str(ARMS[name][0]).replace("torch.", ""),
                         "R": R, "oom": True, "note": ARMS[name][2]}
            print(f"  {name:<18} OOM at R={R} — does not fit on this GPU")
        finally:
            del params, opt, step  # `step` is a bound METHOD: it pins the optimizer too
            reclaim()
    return out


def measure_paired(names: Sequence[str], R: int, native: bool, reps: int,
                   baseline: str = BASELINE) -> dict[str, dict[str, Any]]:
    """Every arm timed ABAB against ``baseline``, both resident in the same process.

    ``baseline`` itself is excluded from the contenders — it IS the reference the others
    are read against, so a self-pair would just report a ratio of 1.0 with no content.

    Holding two arms at once is what costs memory here: at R=5 the baseline alone is
    ~8 B/p of resident memory, so a heavy partner (``adamw_fp32``, 16 B/p) does not
    fit on an 8 GB card. Such a pair is recorded as ``"oom": true`` and the sweep
    continues — the run is not lost, and the gap is visible in the report.
    """
    base_params, base_opt = build_arm(baseline, R, native)
    result: dict[str, dict[str, Any]] = {}
    try:
        for name in names:
            if name == baseline:
                continue
            params = opt = None
            try:
                params, opt = build_arm(name, R, native)
                ta, tb = paired_samples(opt.step, base_opt.step, reps)
                logs = [math.log(x / y) for x, y in zip(ta, tb, strict=True)]
                ratio, lo, hi = gmean_ci(logs)
                rec = {
                    "baseline": baseline,
                    "reps": reps,
                    "pairs_used": len(logs),  # reps minus the discarded ramp-up pairs
                    "ms_arm_median": st.median(ta),
                    "ms_baseline_median": st.median(tb),
                    "ratio": ratio,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "verdict": verdict_of(lo, hi),
                }
                result[name] = rec
                print(f"  {name:<18} {rec['ms_arm_median']:7.2f} ms vs "
                      f"{rec['ms_baseline_median']:7.2f} ms  "
                      f"{ratio:5.3f}x [{lo:.3f},{hi:.3f}] {rec['verdict']}")
            except torch.OutOfMemoryError:
                result[name] = {"baseline": baseline, "reps": reps, "R": R, "oom": True}
                print(f"  {name:<18} OOM alongside {baseline} at R={R} — pair skipped")
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                result[name] = {"baseline": baseline, "reps": reps, "R": R, "oom": True}
                print(f"  {name:<18} OOM alongside {baseline} at R={R} — pair skipped")
            finally:
                del params, opt
                reclaim()
    finally:
        del base_params, base_opt
        reclaim()
    return result


def measure_capacity(names: Sequence[str], native: bool, start: int, stop: int,
                     step_r: int) -> dict[str, dict[str, Any]]:
    """Largest ``R`` (hence param count) each arm fits on this GPU before OOM."""
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        best_R = 0
        oom_at = None
        for R in range(start, stop + 1, step_r):
            params = opt = None
            try:
                params, opt = build_arm(name, R, native)
                opt.step()
                opt.step()
                torch.cuda.synchronize()
                best_R = R
            except torch.OutOfMemoryError:
                oom_at = R
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                oom_at = R
            finally:
                del params, opt
                reclaim()
            if oom_at is not None:
                break
        out[name] = {
            "max_R": best_R,
            "params": bag_numel(best_R) if best_R else 0,
            "oom_at_R": oom_at,
            "oom_at_params": bag_numel(oom_at) if oom_at else None,
            "swept": [start, stop, step_r],
        }
        print(f"  {name:<18} max R={best_R} ({out[name]['params'] / 1e6:6.1f} M params)"
              f"  OOM at R={oom_at}")
    return out


def measure_fraction(names: Sequence[str], C: int, batch: int, px: int, steps: int,
                     native: bool) -> dict[str, Any]:
    """Optimizer ms as a fraction of a real fwd+bwd+step on the proxy UNet at width ``C``."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "proxy_harness", WORKTREE / "benchmarks" / "proxy" / "harness.py")
    assert spec and spec.loader
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)

    # ``UNet.forward`` calls the module-global ``temb``, which builds its sinusoidal
    # embedding in fp32; a bf16 net then hits a dtype mismatch at the first Linear. Cast
    # it on this private copy of the module rather than editing the shared harness.
    base_temb = harness.temb

    arms: dict[str, Any] = {}
    for name in names:
        dtype, factory, _ = ARMS[name]
        harness.temb = lambda t, dim=64, _dt=dtype: base_temb(t, dim).to(_dt)
        torch.manual_seed(0)
        net = harness.UNet(C=C).to(DEV).to(dtype)
        params = list(net.parameters())
        opt = disarm_inert_telemetry(factory(params, native))
        gen = torch.Generator(device=DEV).manual_seed(7)
        x = torch.randn(batch, 1, px, px, generator=gen, device=DEV, dtype=dtype)
        t = torch.randint(0, 1000, (batch,), device=DEV, generator=gen)
        target = torch.randn(batch, 1, px, px, generator=gen, device=DEV, dtype=dtype)
        loss = None  # bound in the loop below; named here so `finally` can always clear it
        try:
            totals: list[float] = []
            opts: list[float] = []
            # Triton autotunes on the first few steps of a kaon arm; half the run is
            # thrown away so that cost does not land in the median.
            warm = max(5, steps // 2)
            for i in range(steps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                loss = torch.nn.functional.mse_loss(net(x, t), target)
                net.zero_grad(set_to_none=True)
                loss.backward()
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                opt.step()
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                if i >= warm:  # drop warmup / Triton autotune
                    totals.append(1000.0 * (t2 - t0))
                    opts.append(1000.0 * (t2 - t1))
            if len(totals) < 3:
                raise ValueError(
                    f"--fraction-steps {steps} leaves only {len(totals)} measured steps "
                    "after warmup; use at least 16")
            rec = {
                "ms_step_total": st.median(totals),
                "ms_optimizer": st.median(opts),
                "params": sum(p.numel() for p in params),
                "measured_steps": len(totals),
            }
            rec["optimizer_fraction"] = rec["ms_optimizer"] / rec["ms_step_total"]
            arms[name] = rec
            print(f"  {name:<18} total {rec['ms_step_total']:7.2f} ms  opt "
                  f"{rec['ms_optimizer']:6.2f} ms  {100 * rec['optimizer_fraction']:5.1f}%")
        finally:
            # `loss` still owns an autograd graph over the net's activations.
            del net, params, opt, x, t, target, loss
            harness.temb = base_temb  # the patched closure captures the arm's dtype
            reclaim()
    return {"C": C, "batch": batch, "px": px, "steps": steps, "arms": arms}


# ----------------------------------------------------------------- report
def _gib(b: float | None) -> str:
    return "-" if b is None else f"{b / 2**30:.2f}"


def render_markdown(payload: dict[str, Any]) -> str:
    """The whole report, rebuildable from the JSON alone (``--render-only``)."""
    meta = payload.get("meta", {})
    cfg = payload.get("config", {})
    bag = payload.get("bag", {})
    L: list[str] = []
    L.append("# Optimizer step cost at diffusion scale")
    L.append("")
    L.append("Cost only — wall clock, VRAM and state bytes of the optimizer step. Nothing "
             "on this page says anything about the quality of the resulting model.")
    L.append("")
    L.append(f"* kaon `{meta.get('kaon_version')}` @ `{str(meta.get('commit'))[:12]}`"
             + (" (dirty worktree)" if meta.get("dirty") else ""))
    L.append(f"* torch {meta.get('torch')} / CUDA {meta.get('torch_cuda')} / triton "
             f"{meta.get('triton')} / python {meta.get('python')}")
    L.append(f"* {meta.get('gpu')} — `{meta.get('nvidia_smi')}` "
             "(power.limit, power.max_limit, power.default_limit, clocks.max.sm, temp)")
    L.append(f"* power: {meta.get('power_note')} — {meta.get('timestamp')}")
    if bag:
        L.append(f"* bag: R={bag.get('R')}, {bag.get('tensors')} tensors, "
                 f"{bag.get('params', 0) / 1e6:.1f} M params; paired reps="
                 f"{cfg.get('reps')} at R={cfg.get('paired_R', cfg.get('R'))}; kaon path="
                 f"{'native/foreach' if cfg.get('native') else 'Triton-fused'}")
    L.append("")

    arms = payload.get("arms") or {}
    if arms:
        L.append("## Per step, per arm (one arm resident)")
        L.append("")
        has_launch = any("cuda_launches" in a for a in arms.values())
        head = ("| arm | params | ms/step | state B/p | resident B/p | peak alloc GiB "
                "| peak reserved GiB | step transient GiB |")
        sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        if has_launch:
            head += " launches |"
            sep += " ---: |"
        L += [head, sep]
        for name, a in arms.items():
            if a.get("oom"):
                L.append(f"| `{name}` | - | - | - | - | - | - | OOM at R={a.get('R')} |"
                         + (" - |" if has_launch else ""))
                continue
            row = (f"| `{name}` | {a['params'] / 1e6:.1f} M | {a['ms_step_solo']:.2f} | "
                   f"{a['state_bytes_per_param']:.2f} | "
                   f"{a.get('peak_bytes_per_param', float('nan')):.2f} | "
                   f"{_gib(a['peak_allocated_bytes'])} | "
                   f"{_gib(a['peak_reserved_bytes'])} | {_gib(a['step_transient_bytes'])} |")
            if has_launch:
                row += f" {a.get('cuda_launches', '-')} |"
            L.append(row)
        L.append("")
        L.append("`state B/p` counts only the optimizer's own tensors; `resident B/p` is "
                 "peak allocated over the parameter count — weights + grads + state + the "
                 "step's transient, which is what actually decides whether a run fits.")
        L.append("")
        L.append("The solo ms/step column is unpaired and therefore the weakest number "
                 "here — read the paired ratios below instead.")
        L.append("")

    paired = payload.get("paired") or {}
    if paired:
        paired_baseline = cfg.get("baseline", BASELINE)
        L.append(f"## Paired against `{paired_baseline}` (ABAB, geometric mean, 95% CI)")
        L.append("")
        L.append("| arm | ms/step | baseline ms/step | ratio | 95% CI | verdict |")
        L.append("| --- | ---: | ---: | ---: | :---: | --- |")
        for name, p in paired.items():
            if p.get("oom"):
                L.append(f"| `{name}` | - | - | - | - | did not fit beside the baseline "
                         f"at R={p.get('R')} |")
                continue
            L.append(f"| `{name}` | {p['ms_arm_median']:.2f} | {p['ms_baseline_median']:.2f} "
                     f"| {p['ratio']:.3f}x | [{p['ci_lo']:.3f}, {p['ci_hi']:.3f}] | "
                     f"{p['verdict']} |")
        L.append("")
        L.append("`n.s.` = the CI straddles 1.0, i.e. the difference is not resolvable at "
                 "this rep count — not a claim of equality. The interval is Student-t "
                 f"(n-1 d.o.f.), as in `benchmarks/control/battery.py`. Each pair is "
                 f"preceded by {DEFAULT_PAIRED_WARM} warm-up pairs and the first "
                 f"{DEFAULT_PAIRED_DISCARD} measured pairs are dropped from both arms, so "
                 "Triton's re-autotune at the paired scale is outside the samples. MSAM's "
                 "inert-lookahead telemetry is disarmed on every arm: it syncs to the host "
                 "only when `rho != 0`, so leaving it on would charge the two lookahead "
                 "arms for a bounded first-~2000-steps cost. The Anima campaign (200 "
                 "steps) does pay it, and reports it.")
        L.append("")

    frac = payload.get("fraction")
    if frac and frac.get("arms"):
        L.append(f"## Share of a real training step (proxy UNet C={frac['C']}, batch "
                 f"{frac['batch']}, {frac['px']}px)")
        L.append("")
        L.append("The full-step column should agree across arms to within the optimizer "
                 "difference — where it does not, the run was contended and the share is "
                 "only indicative.")
        L.append("")
        L.append("| arm | trainable params | full step ms | optimizer ms | optimizer share |")
        L.append("| --- | ---: | ---: | ---: | ---: |")
        for name, f in frac["arms"].items():
            L.append(f"| `{name}` | {f['params'] / 1e6:.2f} M | {f['ms_step_total']:.2f} | "
                     f"{f['ms_optimizer']:.2f} | {100 * f['optimizer_fraction']:.1f}% |")
        L.append("")

    cap = payload.get("capacity")
    if cap:
        L.append("## Capacity on this GPU (largest bag that fits before OOM)")
        L.append("")
        L.append("| arm | max R | trainable params | OOM at R | OOM at params |")
        L.append("| --- | ---: | ---: | ---: | ---: |")
        for name, c in cap.items():
            oom_p = c.get("oom_at_params")
            L.append(f"| `{name}` | {c['max_R']} | {c['params'] / 1e6:.1f} M | "
                     f"{c['oom_at_R'] if c['oom_at_R'] else '-'} | "
                     f"{f'{oom_p / 1e6:.1f} M' if oom_p else '-'} |")
        L.append("")
        L.append("Capacity is a ceiling for THIS bag on THIS card with nothing else "
                 "resident; a real trainer also holds activations.")
        L.append("")

    if arms:
        L.append("## What each arm is")
        L.append("")
        for name, a in arms.items():
            L.append(f"* **`{name}`** ({a['param_dtype']} params) — {a['note']}")
        L.append("")
    return "\n".join(L)


# ----------------------------------------------------------------- driver
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--R", type=int, default=5, help="bag scale (5 ~ 663 M params)")
    ap.add_argument("--reps", type=int, default=60, help="paired reps per arm")
    ap.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS),
                    choices=list(DEFAULT_ARMS))
    ap.add_argument("--baseline", default=BASELINE, choices=list(DEFAULT_ARMS),
                    help="arm the paired phase holds fixed as the reference (default: "
                         f"{BASELINE}); excluded from the paired contenders and added to "
                         "--arms automatically if missing")
    ap.add_argument("--native", action="store_true",
                    help="route the kaon arms through the torch/foreach path instead of "
                         "the Triton-fused one (the AdamW arms stay fused either way)")
    ap.add_argument("--launches", action="store_true",
                    help="also count CUDA kernel launches per step")
    ap.add_argument("--paired-R", type=int, default=None,
                    help="scale for the paired phase, which holds TWO arms at once and so "
                         "needs roughly twice the memory (default: min(--R, 3), which is "
                         "what fits beside the 8 B/p baseline on an 8 GB card)")
    ap.add_argument("--skip-paired", action="store_true")
    ap.add_argument("--capacity", action="store_true", help="sweep R upward until OOM")
    ap.add_argument("--capacity-start", type=int, default=1)
    ap.add_argument("--capacity-max", type=int, default=16)
    ap.add_argument("--capacity-step", type=int, default=1)
    ap.add_argument("--fraction", action="store_true",
                    help="optimizer ms vs a full fwd+bwd+step on the proxy UNet")
    ap.add_argument("--C", type=int, default=256, help="proxy UNet width for --fraction")
    ap.add_argument("--fraction-batch", type=int, default=8)
    ap.add_argument("--fraction-px", type=int, default=64)
    ap.add_argument("--fraction-steps", type=int, default=24,
                    help="half of these are warmup (Triton autotune)")
    ap.add_argument("--out", type=Path, default=None,
                    help="JSON path (default results/step_cost_R{R}.json next to this file)")
    ap.add_argument("--markdown", type=Path, default=None,
                    help="markdown path (default RESULTS_step_cost.md next to this file)")
    ap.add_argument("--render-only", type=Path, default=None,
                    help="re-render the markdown from an existing JSON and exit")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    md_path = args.markdown or (HERE / "RESULTS_step_cost.md")

    if args.render_only is not None:
        payload = json.loads(Path(args.render_only).read_text(encoding="utf-8"))
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown(payload), encoding="utf-8")
        print(f"rendered {args.render_only} -> {md_path}")
        return 0

    if not torch.cuda.is_available():
        print("no CUDA device — this benchmark measures GPU step cost", file=sys.stderr)
        return 2

    meta = collect_meta()
    if not meta["on_ac"]:
        print(f"REFUSING to measure: {meta['power_note']}. This laptop drops from a 60 W "
              "to a 35 W power limit on battery, which moves the clock more than anything "
              "measured here. Plug it in and re-run.", file=sys.stderr)
        return 3

    baseline = args.baseline
    arms = list(args.arms)
    if baseline not in arms:
        arms.append(baseline)
        print(f"note: --baseline {baseline} was not in --arms; added it (the paired phase "
              "needs it resident as the reference)")

    out_path = args.out or (HERE / "results" / f"step_cost_R{args.R}.json")
    payload: dict[str, Any] = {
        "schema": "nekaon_evidence.step_cost/1",
        "meta": meta,
        "config": {
            "R": args.R,
            "reps": args.reps,
            "arms": arms,
            "native": bool(args.native),
            "baseline": baseline,
        },
        "bag": {"R": args.R, "tensors": len(bag_shapes(args.R)),
                "params": bag_numel(args.R)},
    }
    print(f"bag R={args.R}: {payload['bag']['tensors']} tensors, "
          f"{payload['bag']['params'] / 1e6:.1f} M params  |  kaon {meta['kaon_version']} "
          f"@ {str(meta['commit'])[:12]}  |  {meta.get('gpu')}")

    print("solo (memory, state bytes, unpaired ms):")
    payload["arms"] = measure_solo(arms, args.R, args.native,
                                   max(5, args.reps // 4), args.launches)

    if not args.skip_paired and len(arms) > 1:
        paired_R = args.paired_R or min(args.R, DEFAULT_PAIRED_R)
        payload["config"]["paired_R"] = paired_R
        print(f"paired vs {baseline} at R={paired_R} ({args.reps} reps, ABAB):")
        payload["paired"] = measure_paired(arms, paired_R, args.native, args.reps,
                                           baseline=baseline)

    if args.fraction:
        print(f"fraction of a full step (UNet C={args.C}):")
        payload["fraction"] = measure_fraction(arms, args.C, args.fraction_batch,
                                               args.fraction_px, args.fraction_steps,
                                               args.native)

    if args.capacity:
        print(f"capacity sweep R={args.capacity_start}..{args.capacity_max}:")
        payload["capacity"] = measure_capacity(arms, args.native, args.capacity_start,
                                               args.capacity_max, args.capacity_step)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote {out_path}\nwrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
