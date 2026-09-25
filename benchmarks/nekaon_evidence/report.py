"""Consolidate the three Nekaon evidence measurements into one EVIDENCE.md + evidence.json.

The question this answers is "does Nekaon beat Adakaon and AdamW on speed, quality and
memory for diffusion fine-tuning?", and the only honest answer is per axis, per level and
per comparison. So this script never writes an adjective it did not compute: every verdict
below comes from a stated rule applied to an interval, and a difference whose 95% interval
covers zero is printed as ``n.s.``, with the same prominence as a win.

It reads, all optional:

* ``--anima``      ``results.json`` from ``anima/aggregate.py`` -- the real model
  (Anima/Pets LoRA), paired by seed, bootstrap intervals;
* ``--step-cost``  the JSON from ``step_cost.py`` -- the optimizer step alone at SDXL
  parameter scale: paired ratios, state bytes per param, peak VRAM, capacity, and the
  fraction of a real training step the optimizer actually is;
* ``--step-cost-vs-adakaon`` a second ``step_cost.py`` JSON whose ``--baseline`` is
  ``adakaon_4bit`` -- the only file that measures Nekaon *directly* against Adakaon at
  SDXL scale, inside one process, with a real paired interval;
* ``--step-cost-native`` a third ``step_cost.py`` JSON run with ``--native --skip-paired``
  -- the non-Triton path, solo timings only, printed as an unpaired informative table;
* ``--battery``    ``results_evidence.json`` from ``benchmarks/control/battery.py`` -- the
  5-seed proxy-diffusion control battery, whose ``per_seed`` lists make the comparison
  paired (same seed = same init, same batch order, same noise).

A source that is not passed, or that is missing on disk, becomes a "not measured" section
and pushes its axis to ``no evidence``. Nothing is dropped silently.

Two facts about *this* machine bound what the step-cost files can be read to mean, and both
are applied here as rules rather than by hand. Windows drives this GPU through WDDM, whose
allocator oversubscribes into host RAM instead of raising OOM: a capacity sweep that never
OOMs has not found a fit limit, and a timing whose arm held more bytes than the card has is
a measurement of the PCIe bus. See ``INVALID`` and ``WDDM_NO_OOM_NOTE`` below.

    python benchmarks/nekaon_evidence/report.py \\
        --anima benchmarks/nekaon_evidence/anima/results.json \\
        --step-cost benchmarks/nekaon_evidence/results/step_cost_R5.json \\
        --step-cost-vs-adakaon \\
            benchmarks/nekaon_evidence/results/step_cost_R3_vs_adakaon.json \\
        --step-cost-native benchmarks/nekaon_evidence/results/step_cost_R5_native.json \\
        --battery benchmarks/control/results_evidence.json \\
        --out benchmarks/nekaon_evidence/EVIDENCE.md \\
        --json-out benchmarks/nekaon_evidence/evidence.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path
from typing import Any

SCHEMA = "nekaon-evidence-report-v1"

# ------------------------------------------------------------------ verdict vocabulary
#: The challenger is better and the 95% interval excludes the null.
WINS = "wins"
#: The challenger is worse and the 95% interval excludes the null.
LOSES = "loses"
#: The 95% interval covers the null. Not a win, not a loss -- the measurement cannot tell.
NS = "n.s."
#: Two arms that are identical by construction on this metric (same code path, same state).
TIE = "tie by construction"
#: The measurement needed for this cell was not supplied, or the source could not answer it.
NO_EVIDENCE = "no evidence"
#: Findings on one axis disagree; the axis has no single answer and the rows are the answer.
MIXED = "mixed"
#: The row was measured while the allocator held more bytes than the GPU has. Not a result.
INVALID = "invalid (oversubscribed)"

#: What a memory row that exceeded VRAM is: still a real byte count, but not a VRAM figure.
SPILLED_LABEL = "exceeds VRAM: spilled to host"

#: Why a capacity sweep that never OOMed has not measured a fit limit on this machine.
WDDM_NO_OOM_NOTE = ("no OOM reached on any arm: the CUDA allocator oversubscribed into host "
                    "memory (Windows WDDM), so the fit-limit is not measurable on this "
                    "machine")

#: Why a timing taken while the arm was spilled to host RAM is thrown away rather than shown.
OVERSUBSCRIBED_NOTE = ("measured while the allocator held more bytes than this GPU has: on "
                       "Windows WDDM the surplus is served from host RAM over PCIe, so this "
                       "row times the bus, not the optimizer. It is excluded from every "
                       "verdict")

VERDICT_RULES = """\
Every verdict on this page is one of seven words, each produced by a rule, never by reading a
number and choosing an adjective:

* **wins** -- the paired 95% interval for the difference (or the ratio) lies entirely on the
  side that favours the challenger.
* **loses** -- the interval lies entirely on the other side.
* **n.s.** -- the interval covers zero (or 1.0 for a ratio). *Not a win.* With five seeds the
  interval is wide, so `n.s.` is the expected outcome of a small real effect, and it is
  reported exactly as loudly as a win.
* **tie by construction** -- the two arms run the same code path with the same state, so the
  metric is identical by definition and no measurement is needed (see the memory axis).
* **no evidence** -- the source that would answer this was not supplied, or could not answer.
* **invalid (oversubscribed)** -- the arm or its baseline held more bytes than this GPU has
  while the row was being measured. Under Windows WDDM the CUDA allocator serves the surplus
  from host RAM instead of raising OOM, so the number is a PCIe measurement wearing an
  optimizer's name. The row is printed, and it enters no verdict.
* **mixed** -- the rows rolled up into this axis disagree; the rows, not the roll-up, are the
  result.

Intervals: the Anima campaign uses the percentile bootstrap of the paired mean that
`anima/aggregate.py` computed; the control battery and the SDXL step cost use Student-t with
n-1 d.o.f. (the same `_T95` table `battery.py::ci95` and `step_cost.py::gmean_ci` use, so the
three sets of intervals are built the same way). N is printed next to every verdict, because
a verdict from five seeds is not the same object as a verdict from sixty reps.
"""

MEMORY_BY_CONSTRUCTION = (
    "Nekaon vs Adakaon at the same `momentum_dtype` hold the **same memory, by "
    "construction**: Nekaon is a negative-momentum lookahead layer over Adakaon that "
    "climbs and descends the existing momentum buffer in place and allocates no state of "
    "its own. Any B/p difference measured between them is instrumentation noise, not a "
    "property of the optimizers -- read that row as a tie unless the two are running "
    "different `momentum_dtype`s. The memory advantage over AdamW is a different claim "
    "entirely: it comes from Adakaon's factored second moment and its quantized momentum "
    "codec, and the lookahead contributes exactly nothing to it."
)

#: Why there is no "lookahead at SDXL scale" row in the attribution table.
SDXL_LOOKAHEAD_NOTE = (
    "There is deliberately **no `nekaon_4bit / nekaon_k0_4bit` row at SDXL scale here.** "
    "Both arms are paired against `adakaon_4bit`, not against each other, and the stored "
    "file keeps only each pair's geometric mean and interval — not the per-rep times. "
    "Dividing the two ratios would give a point estimate whose interval cannot be "
    "reconstructed (the two pairs share a baseline, so their errors are correlated by an "
    "unknown amount, and treating them as independent would understate it). A correct "
    "interval needs a `step_cost.py` run whose `--baseline` is `nekaon_k0_4bit`; until then "
    "the lookahead's isolated cost is the battery's single-knob pair below, and the SDXL "
    "rows above state only what was measured: each arm against `adakaon_4bit`."
)

# ----------------------------------------------------------------------- arm names
ANIMA_CHALLENGER = "nekaon_fused"
ANIMA_BASELINE = "adamw_fused"
ANIMA_SIBLING = "adakaon_fused"

STEP_COST_BASELINE = "adamw_bf16"
STEP_COST_CHALLENGER = "nekaon_4bit"
STEP_COST_SIBLING = "adakaon_4bit"

#: Arms whose ratio against `adakaon_4bit` *is* a Nekaon-vs-Adakaon result. An arm outside
#: this set (e.g. `adakaon_bf16`) is printed for context and rolled up into nothing: it is
#: Adakaon against Adakaon, which is a codec knob and not this report's question.
STEP_COST_NEKAON_ARMS = ("nekaon_4bit", "nekaon_k0_4bit", "nekaon_bf16")

BATTERY_CHALLENGER = "Nekaon-fused"
BATTERY_BASELINE = "torch.AdamW (fused)"
BATTERY_SIBLING = "Adakaon-4bit-fused"

#: The two comparisons the whole report is organised around.
VS_ADAMW = "nekaon_vs_adamw"
VS_ADAKAON = "nekaon_vs_adakaon"
COMPARISONS = (VS_ADAMW, VS_ADAKAON)
COMPARISON_LABELS = {VS_ADAMW: "Nekaon vs AdamW", VS_ADAKAON: "Nekaon vs Adakaon"}

AXES = ("quality", "speed", "memory")
AXIS_LABELS = {"quality": "Quality", "speed": "Speed", "memory": "Memory"}

#: Single-knob pairs: each isolates one mechanism, because only one setting differs.
ATTRIBUTION_PAIRS = (
    ("lookahead (k=1.5 vs k=0)", BATTERY_CHALLENGER, "Nekaon-k0-4bit-fused"),
    ("4-bit codec on Nekaon", BATTERY_CHALLENGER, "Nekaon-bf16-fused"),
    ("4-bit codec on Adakaon", "Adakaon-4bit-fused", "Adakaon-bf16-fused"),
)

#: Battery metrics and the axis each one belongs to. All are lower-is-better.
BATTERY_METRICS = {
    "te": ("Test loss (REX)", "quality"),
    "gap": ("Train-test gap (REX)", "quality"),
    "cgap": ("Train-test gap (const LR)", "quality"),
    "ms": ("ms / iteration", "speed"),
    "opt_ms": ("ms / optimizer step", "speed"),
    "bpp": ("Bytes per param (state)", "memory"),
}

#: Anima metrics, as ``aggregate.py::METRICS`` names them. All are lower-is-better.
ANIMA_METRICS = {
    "final_val": ("Final val eps-MSE", "quality"),
    "raw_gap": ("Raw gap (val - train_eval)", "quality"),
    "active_seconds": ("Active train seconds", "speed"),
    "ms_per_step": ("ms / step", "speed"),
    "peak_gib": ("Peak allocator GiB", "memory"),
}


# ------------------------------------------------------------------------- statistics
# Two-sided 95% Student-t critical values by degrees of freedom (n-1); past df=30 the normal
# approximation (1.96) is within 1.5%. Copied rather than imported: `battery.py` executes
# `benchmarks/proxy/harness.py` and `dataset.py` at import time and binds `DEV` from them, so
# importing it to borrow `ci95` would drag a torch/CUDA harness into a pure-text report.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
        23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
        30: 2.042}


def ci95(xs: list[float] | None) -> float | None:
    """Half-width of the 95% CI of the mean (Student-t, n-1 d.o.f.); ``None`` for n < 2.

    The same rule as ``benchmarks/control/battery.py::ci95``, so an interval printed here is
    the interval that report would have printed.
    """
    clean = [float(x) for x in (xs or []) if x is not None and math.isfinite(float(x))]
    n = len(clean)
    if n < 2:
        return None
    return _T95.get(n - 1, 1.96) * st.stdev(clean) / (n ** 0.5)


def paired_diff(left: list[float] | None, right: list[float] | None) -> dict[str, Any] | None:
    """Per-seed paired difference ``left - right`` with its 95% t interval.

    Pairing is positional: the battery writes every arm's ``per_seed`` lists for seeds
    0..N-1 in order, and the same seed means the same init, the same batch order and the
    same diffusion noise -- so this is a within-seed difference, not two samples compared.
    Lists of unequal length are refused rather than truncated: a silent zip would pair one
    arm's seed 3 against another's seed 4 and the interval would be fiction.
    """
    if left is None or right is None:
        return None
    if len(left) != len(right):
        raise ValueError(f"cannot pair {len(left)} values against {len(right)}")
    diffs = [float(a) - float(b) for a, b in zip(left, right, strict=True)
             if a is not None and b is not None]
    if not diffs:
        return None
    mean = st.fmean(diffs)
    half = ci95(diffs)
    if half is None:
        return {"mean": mean, "ci_low": None, "ci_high": None, "half_width": None,
                "n": len(diffs), "crosses_zero": None, "per_seed": diffs}
    return {
        "mean": mean,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "half_width": half,
        "n": len(diffs),
        "crosses_zero": (mean - half) <= 0.0 <= (mean + half),
        "per_seed": diffs,
    }


def verdict_from_interval(low: float | None, high: float | None, *,
                          lower_is_better: bool = True, null: float = 0.0) -> str:
    """``wins``/``loses``/``n.s.`` for an interval on a challenger-minus-baseline difference.

    A missing interval (n < 2, or a source that stored a bare mean) is ``n.s.``: a single
    sample makes no claim, and calling it a win would be inventing the evidence this page
    exists to supply.
    """
    if low is None or high is None or not math.isfinite(low) or not math.isfinite(high):
        return NS
    if high < null:
        return WINS if lower_is_better else LOSES
    if low > null:
        return LOSES if lower_is_better else WINS
    return NS


def verdict_from_ratio(low: float | None, high: float | None) -> str:
    """The same rule for an arm/baseline time ratio: the null is 1.0 and lower is faster."""
    return verdict_from_interval(low, high, lower_is_better=True, null=1.0)


def roll_up(verdicts: list[str]) -> str:
    """One word for an axis, from its rows. Disagreement stays disagreement.

    ``no evidence`` and ``invalid (oversubscribed)`` are both *absences* of a result, for
    different reasons, and neither may tilt an axis: a row that measured the PCIe bus is no
    more a speed result than a file that was never supplied.
    """
    meaningful = [v for v in verdicts if v not in (NO_EVIDENCE, INVALID)]
    if not meaningful:
        return NO_EVIDENCE
    wins = sum(1 for v in meaningful if v == WINS)
    losses = sum(1 for v in meaningful if v == LOSES)
    if wins and losses:
        return MIXED
    if wins:
        return WINS
    if losses:
        return LOSES
    if all(v == TIE for v in meaningful):
        return TIE
    return NS


# ---------------------------------------------------------------------------- findings


def finding(*, axis: str, comparison: str, level: str, metric: str, verdict: str,
            source: str, stat: dict[str, Any] | None = None, units: str = "",
            note: str | None = None, value: float | None = None,
            rollup: bool = True) -> dict[str, Any]:
    """One measured cell: an axis, a level within it, a comparison and a rule-made verdict.

    ``rollup=False`` prints the row with its real verdict but keeps it out of the axis
    roll-up, for a row that is measured and true yet answers a different question than the
    column it is printed under.
    """
    row: dict[str, Any] = {
        "axis": axis, "comparison": comparison, "level": level, "metric": metric,
        "verdict": verdict, "source": source, "units": units, "note": note,
        "mean": None, "ci_low": None, "ci_high": None, "n": None, "value": value,
        "rollup": rollup,
    }
    if stat:
        row.update({k: stat.get(k) for k in ("mean", "ci_low", "ci_high", "n")})
    return row


def load_source(path: str | Path | None) -> tuple[dict[str, Any] | None, str | None]:
    """The JSON at ``path``, or ``(None, reason)``. A bad file is a printed reason, not a crash."""
    if path is None:
        return None, "not supplied on the command line"
    p = Path(path)
    if not p.exists():
        return None, f"`{p}` does not exist"
    try:
        return json.loads(p.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"`{p}` could not be read ({exc})"


# ------------------------------------------------------------------------ anima source


def anima_findings(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Quality / speed / memory rows from the real-model campaign, paired by seed."""
    if not result:
        return []
    rows: list[dict[str, Any]] = []
    power = result.get("power") or {}
    timings_ok = bool(power.get("timings_comparable", False)) if power else False
    pairs = {
        VS_ADAMW: (result.get("paired") or {}).get(f"{ANIMA_CHALLENGER}_minus_{ANIMA_BASELINE}"),
        VS_ADAKAON: (result.get("paired") or {}).get(f"{ANIMA_CHALLENGER}_minus_{ANIMA_SIBLING}"),
    }
    levels = {"quality": "real model (Anima/Pets LoRA), paired by seed",
              "speed": "(b) full training step on the real model",
              "memory": "(b) peak VRAM during real training"}
    for comparison, block in pairs.items():
        if not block:
            continue
        for metric, (label, axis) in ANIMA_METRICS.items():
            entry = ((block.get("metrics") or {}).get(metric) or {}).get("bootstrap")
            if not entry:
                continue
            verdict = verdict_from_interval(entry.get("ci_low"), entry.get("ci_high"),
                                            lower_is_better=True)
            note = None
            if axis == "speed" and not timings_ok:
                verdict = NO_EVIDENCE
                note = ("the runs behind this number were measured under different power "
                        "signatures (60 W on AC vs 35 W on battery), so active seconds and "
                        "ms/step are not comparable at all; the quality rows are unaffected")
            rows.append(finding(
                axis=axis, comparison=comparison, level=levels[axis], metric=label,
                verdict=verdict, source="anima", stat=entry,
                units="paired difference (negative favours Nekaon)", note=note,
            ))
    return rows


# -------------------------------------------------------------------- step-cost source


#: Allocator peaks are not deterministic to the byte -- block rounding and a handful of
#: workspace tensors move them by kilobytes between two arms running the same buffers. A
#: peak difference under this relative tolerance is that, and calling it a memory result
#: would manufacture a winner out of allocator bookkeeping.
PEAK_REL_TOL = 1e-3


def _memory_verdict(a: float, b: float, comparison: str, *,
                    higher_is_better: bool = False, rel_tol: float = 1e-9) -> str:
    """Verdict for a deterministic memory quantity, which has no interval to test.

    Equality against Adakaon is ``tie by construction`` rather than ``n.s.``: the two
    optimizers hold the same buffers at the same ``momentum_dtype``, so the equality is a
    fact about the code and not a measurement that failed to separate them. Equality
    against AdamW is ``n.s.``, because there it would be a coincidence of two different
    state layouts.
    """
    if math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-9):
        return TIE if comparison == VS_ADAKAON else NS
    better = a > b if higher_is_better else a < b
    return WINS if better else LOSES


def _sc_arm(payload: dict[str, Any], name: str) -> dict[str, Any] | None:
    arm = (payload.get("arms") or {}).get(name)
    if not arm or arm.get("oom"):
        return None
    return arm


def gpu_total_bytes(payload: dict[str, Any] | None) -> int | None:
    """The VRAM this GPU actually has, as ``step_cost.py`` recorded it (``None`` if absent).

    Absent means unknown, and unknown may never be read as "it fit": a payload without
    ``meta.gpu_total_bytes`` flags nothing, exactly as a payload whose arms all fit.
    """
    total = ((payload or {}).get("meta") or {}).get("gpu_total_bytes")
    try:
        total = int(total)
    except (TypeError, ValueError):
        return None
    return total if total > 0 else None


def oversubscribed_arms(payload: dict[str, Any] | None) -> dict[str, int]:
    """``{arm: peak_allocated_bytes}`` for every arm that held more bytes than the GPU has.

    Under Windows WDDM a CUDA allocation larger than VRAM does not raise: the driver pages
    the surplus into host RAM and the step keeps running, hundreds of milliseconds slower.
    So a peak above ``gpu_total_bytes`` is not a near miss, it is the signature of a run
    whose timings measure PCIe traffic -- and it contaminates the *other* arms too, because
    the baseline is re-timed inside the same process.
    """
    total = gpu_total_bytes(payload)
    if total is None:
        return {}
    out: dict[str, int] = {}
    for name, arm in ((payload or {}).get("arms") or {}).items():
        if not isinstance(arm, dict):
            continue
        peak = arm.get("peak_allocated_bytes")
        if isinstance(peak, (int, float)) and math.isfinite(peak) and peak > total:
            out[name] = int(peak)
    return out


def _paired_speed_rows(payload: dict[str, Any], *, comparison: str, source: str,
                       arms: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    """One speed-(a) row per paired arm, with the oversubscription rule already applied."""
    spilled = oversubscribed_arms(payload)
    rows: list[dict[str, Any]] = []
    paired = payload.get("paired") or {}
    for name in (arms if arms is not None else tuple(paired)):
        entry = paired.get(name)
        if not entry or entry.get("oom") or entry.get("ratio") is None:
            continue
        base = entry.get("baseline", STEP_COST_BASELINE)
        verdict = verdict_from_ratio(entry.get("ci_lo"), entry.get("ci_hi"))
        note, rollup = None, True
        if comparison == VS_ADAKAON and name not in STEP_COST_NEKAON_ARMS:
            note = ("context row: this pair is Adakaon against Adakaon -- the 4-bit codec "
                    "knob at SDXL scale, not Nekaon against Adakaon -- so it is printed "
                    "with its interval and rolled up into nothing")
            rollup = False
        bad = [a for a in (name, base) if a in spilled]
        if bad:
            verdict, rollup = INVALID, False
            note = (f"{OVERSUBSCRIBED_NOTE} ({', '.join(bad)} peaked at "
                    f"{', '.join(f'{spilled[a] / 2 ** 30:.2f} GiB' for a in bad)} on a "
                    f"{gpu_total_bytes(payload) / 2 ** 30:.2f} GiB card)")
        rows.append(finding(
            axis="speed", comparison=comparison,
            level="(a) optimizer step alone, SDXL parameter scale",
            metric=f"{name} / {base} time ratio",
            verdict=verdict, source=source,
            stat={"mean": entry.get("ratio"), "ci_low": entry.get("ci_lo"),
                  "ci_high": entry.get("ci_hi"), "n": entry.get("pairs_used")},
            units="ratio (<1 favours the left arm)", note=note, rollup=rollup,
        ))
    return rows


def step_cost_findings(payload: dict[str, Any] | None,
                       vs_adakaon: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Speed levels (a) and (c), and memory levels (a) and (c), at SDXL parameter scale.

    ``vs_adakaon`` is a second ``step_cost.py`` payload whose baseline is ``adakaon_4bit``.
    When it is supplied, the Nekaon-vs-Adakaon speed-(a) cell is a *real paired ratio with
    an interval*, measured in one process; when it is not, the cell stays the interval-less
    ratio-of-ratios it has always been, and earns no verdict.
    """
    if not payload:
        return []
    rows: list[dict[str, Any]] = []

    # --- speed (a): the optimizer step alone, paired against adamw_bf16 in one process.
    paired = payload.get("paired") or {}
    challenger = paired.get(STEP_COST_CHALLENGER)
    rows += _paired_speed_rows(payload, comparison=VS_ADAMW, source="step_cost",
                               arms=(STEP_COST_CHALLENGER,))

    # --- speed (a), Nekaon against Adakaon: directly, if a run with that baseline exists.
    if vs_adakaon:
        rows += _paired_speed_rows(vs_adakaon, comparison=VS_ADAKAON,
                                   source="step_cost_vs_adakaon")
    sibling = paired.get(STEP_COST_SIBLING)
    if (not vs_adakaon and challenger and sibling and not challenger.get("oom")
            and not sibling.get("oom") and sibling.get("ratio")):
        # Both arms are timed against adamw_bf16, never against each other. Dividing the two
        # geometric means gives a point estimate, but its interval is NOT the two intervals
        # combined -- those are different pairs, measured in different processes. So the
        # number is printed and no verdict is claimed from it.
        rows.append(finding(
            axis="speed", comparison=VS_ADAKAON,
            level="(a) optimizer step alone, SDXL parameter scale",
            metric=f"{STEP_COST_CHALLENGER} / {STEP_COST_SIBLING} (indirect)",
            verdict=NO_EVIDENCE, source="step_cost",
            value=challenger["ratio"] / sibling["ratio"],
            units="ratio of two ratios, no interval",
            note=("`step_cost.py` pairs every arm against `adamw_bf16`, never Nekaon against "
                  "Adakaon, so this point estimate carries no paired interval and earns no "
                  "verdict. The lookahead's own cost is measured by the single-knob battery "
                  "pair in the attribution table, or would need a step_cost run whose "
                  "baseline is `adakaon_4bit`."),
        ))

    # --- speed (c): how much of a real training step the optimizer even is.
    for name, entry in ((payload.get("fraction") or {}).get("arms") or {}).items():
        share = entry.get("optimizer_fraction")
        if share is None:
            continue
        comparison = VS_ADAKAON if name == STEP_COST_SIBLING else VS_ADAMW
        rows.append(finding(
            axis="speed", comparison=comparison,
            level="(c) optimizer share of a full fwd+bwd+step",
            metric=f"{name} optimizer fraction",
            verdict=NS, source="step_cost", value=share, units="fraction of the step",
            note=("a ceiling, not a contest: making this optimizer infinitely fast would cut "
                  f"the whole step by at most {100 * share:.1f}%"),
        ))

    # --- memory (a): state bytes per param. Deterministic arithmetic; no interval exists.
    arms = {name: _sc_arm(payload, name)
            for name in (STEP_COST_CHALLENGER, STEP_COST_BASELINE, STEP_COST_SIBLING)}
    ch = arms[STEP_COST_CHALLENGER]
    for comparison, other_name in ((VS_ADAMW, STEP_COST_BASELINE),
                                  (VS_ADAKAON, STEP_COST_SIBLING)):
        other = arms[other_name]
        if not ch or not other:
            continue
        a, b = ch.get("state_bytes_per_param"), other.get("state_bytes_per_param")
        if a is None or b is None:
            continue
        verdict = _memory_verdict(a, b, comparison)
        rows.append(finding(
            axis="memory", comparison=comparison,
            level="(a) optimizer state bytes per parameter",
            metric=f"{STEP_COST_CHALLENGER} {a:.3f} B/p vs {other_name} {b:.3f} B/p",
            verdict=verdict, source="step_cost", value=a - b,
            units="B/p difference (deterministic, no interval)",
            note=("identical by construction: Nekaon is a lookahead layer over Adakaon and "
                  "allocates no state of its own, so at the same `momentum_dtype` the two "
                  "hold the same buffers" if verdict == TIE else None),
        ))

    # --- memory (b): peak allocator bytes at SDXL scale. A byte count either way, but above
    # `gpu_total_bytes` it is no longer a VRAM figure: that arm is partly resident in host
    # RAM, so the row is labelled rather than silently read as "peak VRAM".
    total = gpu_total_bytes(payload)
    for comparison, other_name in ((VS_ADAMW, STEP_COST_BASELINE),
                                  (VS_ADAKAON, STEP_COST_SIBLING)):
        other = arms[other_name]
        if not ch or not other:
            continue
        a, b = ch.get("peak_allocated_bytes"), other.get("peak_allocated_bytes")
        if a is None or b is None:
            continue
        bad = [name for name, peak in ((STEP_COST_CHALLENGER, a), (other_name, b))
               if total is not None and peak > total]
        rows.append(finding(
            axis="memory", comparison=comparison,
            level="(b) peak allocator bytes at SDXL parameter scale",
            metric=(f"{STEP_COST_CHALLENGER} {a / 2 ** 30:.2f} GiB vs "
                    f"{other_name} {b / 2 ** 30:.2f} GiB"),
            verdict=_memory_verdict(a, b, comparison, rel_tol=PEAK_REL_TOL),
            source="step_cost",
            value=(a - b) / 2 ** 30,
            units=(f"GiB difference (allocator peak; anything under "
                   f"{PEAK_REL_TOL:.1%} is bookkeeping, not a result)"),
            note=(f"{SPILLED_LABEL} — {', '.join(bad)} peaked above this card's "
                  f"{total / 2 ** 30:.2f} GiB, so that arm's bytes are real but its "
                  "residency is not VRAM" if bad else None),
        ))

    # --- memory (c): how many parameters each arm holds before it OOMs on this GPU.
    capacity = payload.get("capacity") or {}
    cap_ch = capacity.get(STEP_COST_CHALLENGER)
    # A sweep in which no arm ever OOMed did not find a fit limit -- it found the end of the
    # sweep. On WDDM the allocator spills to host RAM instead of raising, so `max_R` there is
    # the largest R tried, not the largest R that fits, and comparing two of them is comparing
    # two identical loop bounds.
    no_oom = bool(capacity) and not any(
        (entry or {}).get("oom_at_R") is not None or (entry or {}).get("oom_at_params") is not None
        for entry in capacity.values())
    for comparison, other_name in ((VS_ADAMW, STEP_COST_BASELINE),
                                  (VS_ADAKAON, STEP_COST_SIBLING)):
        cap_other = capacity.get(other_name)
        if not cap_ch or not cap_other:
            continue
        a, b = cap_ch.get("params") or 0, cap_other.get("params") or 0
        verdict = (NO_EVIDENCE if no_oom
                   else _memory_verdict(a, b, comparison, higher_is_better=True))
        rows.append(finding(
            axis="memory", comparison=comparison,
            level="(c) largest parameter count that fits on this GPU",
            metric=f"{STEP_COST_CHALLENGER} {a / 1e6:.1f} M vs {other_name} {b / 1e6:.1f} M",
            verdict=verdict, source="step_cost", value=(a - b) / 1e6,
            units="M parameters (higher is better; a coarse sweep in R, not an interval)",
            note=(WDDM_NO_OOM_NOTE + f"; both arms simply reached the top of the sweep "
                  f"(R={cap_ch.get('max_R')} and R={cap_other.get('max_R')})"
                  if no_oom else None),
        ))
    return rows


def integrity_table(sources: dict[str, dict[str, Any] | None]) -> dict[str, Any] | None:
    """Per arm, per step-cost file: did it fit, and is its timing therefore usable?

    This is the audit trail behind every ``invalid (oversubscribed)`` verdict above. It
    lists the solo phase and the paired phase separately, because they run at different R
    and an arm can spill in one and not the other -- and it prints the arms that fit too, so
    the reader can see that the rule was applied to all of them and bit only some.
    """
    entries: list[dict[str, Any]] = []
    totals: set[int] = set()
    for source, payload in sources.items():
        if not payload:
            continue
        total = gpu_total_bytes(payload)
        if total is None:
            continue
        totals.add(total)
        spilled = oversubscribed_arms(payload)
        paired = payload.get("paired") or {}
        for name, arm in (payload.get("arms") or {}).items():
            if not isinstance(arm, dict):
                continue
            peak = arm.get("peak_allocated_bytes")
            pair = paired.get(name) or {}
            base = pair.get("baseline")
            entries.append({
                "source": source,
                "arm": name,
                "R": (payload.get("bag") or {}).get("R"),
                "peak_allocated_bytes": peak,
                "exceeds_vram": name in spilled,
                "solo_ms": arm.get("ms_step_solo"),
                "solo_status": INVALID if name in spilled else "usable",
                "paired_R": (payload.get("config") or {}).get("paired_R"),
                "paired_ratio": pair.get("ratio") if pair else None,
                "paired_baseline": base,
                "paired_baseline_ms": pair.get("ms_baseline_median") if pair else None,
                "paired_status": (None if not pair else
                                  INVALID if (name in spilled or base in spilled)
                                  else "usable"),
            })
    if not entries:
        return None
    capacity = ((sources.get("step_cost") or {}).get("capacity")) or {}
    no_oom = bool(capacity) and not any(
        (e or {}).get("oom_at_R") is not None or (e or {}).get("oom_at_params") is not None
        for e in capacity.values())
    return {
        "gpu_total_bytes": max(totals) if totals else None,
        "capacity_no_oom": no_oom,
        "capacity_note": WDDM_NO_OOM_NOTE if no_oom else None,
        "arms": entries,
    }


def native_table(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """The ``--native`` run: solo milliseconds per arm, unpaired, with no interval at all.

    ``step_cost.py --native --skip-paired`` times the reference (non-Triton) kernels one arm
    after another. There is no interleaving and no ratio, so there is no interval and no
    verdict: the table is printed to show which path the Triton numbers are being compared
    against, and it is labelled as ordering evidence, not as a result.
    """
    if not payload:
        return None
    total = gpu_total_bytes(payload)
    rows = []
    for name, arm in (payload.get("arms") or {}).items():
        if not isinstance(arm, dict) or arm.get("oom"):
            continue
        peak = arm.get("peak_allocated_bytes")
        rows.append({
            "arm": name,
            "ms_step_solo": arm.get("ms_step_solo"),
            "state_bytes_per_param": arm.get("state_bytes_per_param"),
            "peak_allocated_bytes": peak,
            "exceeds_vram": bool(total is not None and isinstance(peak, (int, float))
                                 and peak > total),
        })
    if not rows:
        return None
    return {
        "R": (payload.get("bag") or {}).get("R"),
        "params": (payload.get("bag") or {}).get("params"),
        "reps": (payload.get("config") or {}).get("reps"),
        "gpu_total_bytes": total,
        "rows": sorted(rows, key=lambda r: r["ms_step_solo"] or math.inf),
    }


# ---------------------------------------------------------------------- battery source


def _per_seed(store: dict[str, Any], arm: str, key: str) -> list[float] | None:
    entry = store.get(arm)
    if not isinstance(entry, dict):
        return None
    values = (entry.get("per_seed") or {}).get(key)
    return list(values) if values else None


def battery_findings(store: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Paired-by-seed rows from the 5-seed control battery on the proxy diffusion task."""
    if not store:
        return []
    rows: list[dict[str, Any]] = []
    levels = {"quality": "control battery (proxy diffusion), paired by seed",
              "speed": "(b) full training step on the proxy model",
              "memory": "(a) optimizer state bytes per parameter"}
    for comparison, other in ((VS_ADAMW, BATTERY_BASELINE), (VS_ADAKAON, BATTERY_SIBLING)):
        for key, (label, axis) in BATTERY_METRICS.items():
            left = _per_seed(store, BATTERY_CHALLENGER, key)
            right = _per_seed(store, other, key)
            if left is None or right is None or len(left) != len(right):
                continue
            stat = paired_diff(left, right)
            if not stat:
                continue
            note = None
            if axis == "memory":
                # Bytes per param is arithmetic, not a sample: every seed reports the same
                # number, so a t interval on it is degenerate and would print `n.s.` for two
                # arms that are identical by construction. Compare the values themselves.
                verdict = _memory_verdict(st.fmean(left), st.fmean(right), comparison)
                note = ("state bytes per param is deterministic: it is compared directly, "
                        "not through an interval, and any spread across seeds would be "
                        "measurement noise rather than seed variance")
            else:
                verdict = verdict_from_interval(stat["ci_low"], stat["ci_high"],
                                                lower_is_better=True)
            rows.append(finding(
                axis=axis, comparison=comparison, level=levels[axis],
                metric=f"{label} ({BATTERY_CHALLENGER} - {other})",
                verdict=verdict, source="battery", stat=stat,
                units="paired difference (negative favours Nekaon)", note=note,
            ))
    return rows


def attribution_table(store: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Single-knob pairs: what the lookahead does, and what the 4-bit codec does.

    Each pair differs in exactly one setting, so its difference is attributable to that
    setting alone. A pair whose arms are not both in the store is reported as missing, never
    quietly dropped. With no battery at all there are no pairs to form, so the table is
    empty and the page says the source is missing instead of listing six unmeasured rows.
    """
    if not store:
        return []
    rows: list[dict[str, Any]] = []
    for label, left_arm, right_arm in ATTRIBUTION_PAIRS:
        for key, (metric_label, axis) in BATTERY_METRICS.items():
            left = _per_seed(store, left_arm, key)
            right = _per_seed(store, right_arm, key)
            if left is None or right is None or len(left) != len(right):
                rows.append({"knob": label, "left": left_arm, "right": right_arm,
                             "metric": metric_label, "axis": axis, "verdict": NO_EVIDENCE,
                             "mean": None, "ci_low": None, "ci_high": None, "n": None,
                             "note": "one of the two arms is absent from the battery store"})
                continue
            stat = paired_diff(left, right)
            if not stat:
                continue
            if axis == "memory":
                # Deterministic again: a knob that leaves the state layout alone shows a
                # degenerate [0, 0] interval, which is a tie by construction and not a
                # measurement that failed.
                verdict = (TIE if math.isclose(st.fmean(left), st.fmean(right),
                                               rel_tol=1e-9, abs_tol=1e-9)
                           else WINS if st.fmean(left) < st.fmean(right) else LOSES)
            else:
                verdict = verdict_from_interval(stat["ci_low"], stat["ci_high"],
                                                lower_is_better=True)
            rows.append({
                "knob": label, "left": left_arm, "right": right_arm,
                "metric": metric_label, "axis": axis, "verdict": verdict,
                "mean": stat["mean"], "ci_low": stat["ci_low"], "ci_high": stat["ci_high"],
                "n": stat["n"], "note": None,
            })
    return rows


# --------------------------------------------------------------------------- caveats


def caveats(anima: dict[str, Any] | None, step_cost: dict[str, Any] | None,
            battery: dict[str, Any] | None, *,
            step_cost_vs_adakaon: dict[str, Any] | None = None) -> list[str]:
    """Everything that bounds what the numbers above may be used to claim.

    The static entries are true of the design; the rest are derived from the sources, so a
    caveat appears exactly when a measurement earned it.
    """
    out: list[str] = []

    if anima:
        out.append("**Hyperparameter confound, AdamW vs the kaon arms (Anima).** "
                   + (anima.get("confound_note") or ""))
        boundary = [arm for arm, e in (anima.get("lr_boundary") or {}).items()
                    if e.get("boundary")]
        if boundary:
            out.append(f"**LR selected at the edge of the screened grid for "
                       f"{', '.join(sorted(boundary))}.** The optimum was never bracketed "
                       "there, so that arm's result is a lower bound on what it can do, not "
                       "an estimate of its best.")
        phase_a = anima.get("phase_a") or {}
        if not phase_a.get("complete", True):
            out.append(f"**Phase A is incomplete** ({phase_a.get('runs')}/"
                       f"{phase_a.get('expected')} screening runs). The learning rates came "
                       "from a partial screen, so the paired comparison is provisional.")
        power = anima.get("power") or {}
        if power and not power.get("timings_comparable", True):
            out.append("**The Anima runs do not share an electrical signature.** This GPU is "
                       "power limited to 60 W on AC and 35 W on battery, so active seconds "
                       "and ms/step across those runs are not comparable at all; quality is "
                       "unaffected.")
        if anima.get("telemetry_note"):
            out.append("**Inert-lookahead telemetry counts against Nekaon on time.** "
                       + anima["telemetry_note"])
        validation = anima.get("validation") or {}
        if validation and not validation.get("ok", True):
            out.append("**The paired set failed its own pairing check**: within a seed the "
                       "arms did not share the initial adapter SHA-256 or the step-zero "
                       "losses, so the differences are not purely the optimizer.")
        seeds = ((anima.get("seeds") or {}).get("paired")) or []
        if len(seeds) < 10:
            out.append(f"**n is small** -- {len(seeds)} paired seed(s) on the real model. A "
                       "bootstrap interval over that many points is wide; `n.s.` there means "
                       "the campaign is too small to resolve the effect, not that the effect "
                       "is zero.")

    if step_cost:
        meta = step_cost.get("meta") or {}
        if meta.get("on_ac") is False:
            out.append("**The SDXL step-cost run was measured on battery** "
                       f"({meta.get('power_note')}), where this GPU drops to a 35 W cap. "
                       "Those timings are not comparable with anything measured on AC.")
        oom = [name for name, arm in (step_cost.get("arms") or {}).items()
               if isinstance(arm, dict) and arm.get("oom")]
        if oom:
            out.append(f"**Arms that did not fit on this GPU, and are therefore unmeasured on "
                       f"time: {', '.join(sorted(oom))}.** Their absence is a memory result, "
                       "not a missing speed result.")
        spilled = {**oversubscribed_arms(step_cost),
                   **oversubscribed_arms(step_cost_vs_adakaon)}
        capacity = step_cost.get("capacity") or {}
        no_oom = bool(capacity) and not any(
            (e or {}).get("oom_at_R") is not None or (e or {}).get("oom_at_params") is not None
            for e in capacity.values())
        if spilled or no_oom:
            total = gpu_total_bytes(step_cost) or gpu_total_bytes(step_cost_vs_adakaon)
            detail = []
            if no_oom:
                detail.append("the capacity sweep reached the top of its range on every arm "
                              "without a single OOM, so no fit limit was found")
            if spilled:
                detail.append("and " * bool(no_oom) + ", ".join(
                    f"`{name}` peaked at {peak / 2 ** 30:.2f} GiB"
                    for name, peak in sorted(spilled.items()))
                    + (f" on a card holding {total / 2 ** 30:.2f} GiB" if total else ""))
            joined = "; ".join(detail)
            out.append("**This GPU is driven through Windows WDDM, whose CUDA allocator "
                       "oversubscribes into host RAM instead of raising OOM.** "
                       + joined[:1].upper() + joined[1:]
                       + ". Memory level (c) is therefore `no evidence` "
                       "on this machine rather than a result, and every timing taken while "
                       "an arm was spilled is marked `invalid (oversubscribed)` and enters "
                       "no verdict. Both need a Linux/TCC driver, or a card the workload "
                       "fits on, to become measurable.")
        out.append("**The SDXL scale is a parameter bag, not SDXL.** `step_cost.py` allocates "
                   "a tensor-shape census of an SDXL-like UNet and steps the optimizer on it. "
                   "That reproduces the optimizer's cost at that parameter count and dtype; "
                   "it runs no diffusion and says nothing about the resulting model.")

    if battery:
        out.append("**The control battery is a synthetic proxy, not perception.** It ranks "
                   "objective loss, overfitting gap and convergence on a small diffusion "
                   "proxy at LRs roughly 100x a real fine-tune. There is no FID, no KID and "
                   "no human judgement anywhere in this report: nothing here licenses a claim "
                   "about image quality. Confirm on a real LoRA with FID/KID before believing "
                   "the quality axis.")
        seeds = (((battery.get(BATTERY_CHALLENGER) or {}).get("per_seed") or {})
                 .get("seeds")) or []
        if seeds and len(seeds) < 10:
            out.append(f"**The battery runs {len(seeds)} seeds.** Student-t on {len(seeds)} "
                       "points gives a wide interval; most small real differences will land "
                       "on `n.s.`.")
        metas = [e["meta"] for e in battery.values()
                 if isinstance(e, dict) and isinstance(e.get("meta"), dict)]
        if any(m.get("on_ac") is False for m in metas):
            out.append("**Part of the battery was measured on battery power**, so its `ms` "
                       "and `opt_ms` columns mix two power caps and are not comparable "
                       "between rows.")

    out.append("**All timings come from a power-limited laptop GPU** (60 W on AC, 35 W on "
               "battery). Ratios measured adjacently inside one process survive that; "
               "absolute milliseconds do not transfer to a desktop or a datacentre card.")
    out.append("**`_warn_if_inert` is armed on Nekaon and on no other arm.** Its "
               "device-to-host synchronisations are real cost for a user running this "
               "configuration today, but they are telemetry, not optimizer arithmetic -- a "
               "Nekaon speed loss of that magnitude is a property of the build, not of the "
               "algorithm.")
    return out


# --------------------------------------------------------------------------- assembly


def _anima_gpu(anima: dict[str, Any] | None) -> str | None:
    """The GPU name, wherever ``aggregate.py`` happened to record it, or ``None``.

    Older ``results.json`` files carry no device name at all -- only the power signature
    keys, which name the electrical state and not the card. In that case the column stays
    `—` rather than borrowing the GPU of another source, which would be a claim about where
    this campaign ran that the file does not support.
    """
    prov = (anima or {}).get("provenance") or {}
    power = (anima or {}).get("power") or {}
    for holder in (prov, power):
        for key in ("gpu", "gpu_name", "device_name", "device"):
            value = holder.get(key)
            if isinstance(value, str) and value.strip() and value.strip() != "cuda":
                return value.strip()
    return None


def _power_cell(power: dict[str, Any]) -> str | None:
    """The Anima electrical state, naming the signature(s) the runs were actually under."""
    if not power:
        return None
    names = sorted(power.get("signatures") or {})
    detail = f" ({', '.join(names)})" if names else ""
    if power.get("timings_comparable"):
        return f"all runs share one signature{detail}"
    return f"MIXED power signatures -- timings not comparable{detail}"


def source_header(anima: dict[str, Any] | None, anima_why: str | None,
                  step_cost: dict[str, Any] | None, step_cost_why: str | None,
                  battery: dict[str, Any] | None, battery_why: str | None,
                  vs_adakaon: dict[str, Any] | None = None,
                  vs_adakaon_why: str | None = None,
                  native: dict[str, Any] | None = None,
                  native_why: str | None = None) -> list[dict[str, Any]]:
    """Per source: present or why not, kaon version, commit, GPU and electrical state."""
    rows: list[dict[str, Any]] = []

    prov = (anima or {}).get("provenance") or {}
    power = (anima or {}).get("power") or {}
    rows.append({
        "source": "anima", "present": anima is not None, "reason": anima_why,
        "kaon_version": prov.get("kaon_version"), "commit": prov.get("commit"),
        "gpu": _anima_gpu(anima),
        "power": _power_cell(power),
        "detail": (f"{len(((anima or {}).get('seeds') or {}).get('paired') or [])} paired seeds"
                   if anima else None),
    })

    def _step_cost_row(name: str, payload: dict[str, Any] | None, why: str | None,
                       extra: str) -> dict[str, Any]:
        meta = (payload or {}).get("meta") or {}
        bag = (payload or {}).get("bag") or {}
        return {
            "source": name, "present": payload is not None, "reason": why,
            "kaon_version": meta.get("kaon_version"), "commit": meta.get("commit"),
            "gpu": meta.get("gpu"), "power": meta.get("power_note"),
            "detail": (f"R={bag.get('R')}, {(bag.get('params') or 0) / 1e6:.1f} M params"
                       f"{extra}" if payload else None),
        }

    rows.append(_step_cost_row("step_cost", step_cost, step_cost_why, ""))
    # The two extra step-cost files are optional by design: a row appears when one was
    # supplied, or when one was supplied and could not be read -- never as an accusation
    # against a run that never asked for them.
    def _optional(payload: dict[str, Any] | None, why: str | None) -> bool:
        return payload is not None or bool(why and "not supplied" not in why)

    if _optional(vs_adakaon, vs_adakaon_why):
        base = ((vs_adakaon or {}).get("config") or {}).get("baseline")
        rows.append(_step_cost_row("step_cost_vs_adakaon", vs_adakaon, vs_adakaon_why,
                                   f", baseline `{base}`"))
    if _optional(native, native_why):
        rows.append(_step_cost_row("step_cost_native", native, native_why,
                                   ", native (non-Triton), unpaired"))

    metas = [e["meta"] for e in (battery or {}).values()
             if isinstance(e, dict) and isinstance(e.get("meta"), dict)]
    last = max(metas, key=lambda m: m.get("timestamp") or "") if metas else {}
    rows.append({
        "source": "battery", "present": battery is not None, "reason": battery_why,
        "kaon_version": last.get("kaon_version"), "commit": last.get("commit"),
        "gpu": last.get("gpu"),
        "power": (("AC" if last.get("on_ac")
                   else f"BATTERY (BatteryStatus={last.get('battery_status')})")
                  if last else None),
        "detail": (f"{len(battery)} arms" if battery else None),
    })
    return rows


def build_report(anima: dict[str, Any] | None = None,
                 step_cost: dict[str, Any] | None = None,
                 battery: dict[str, Any] | None = None, *,
                 step_cost_vs_adakaon: dict[str, Any] | None = None,
                 step_cost_native: dict[str, Any] | None = None,
                 anima_reason: str | None = None, step_cost_reason: str | None = None,
                 battery_reason: str | None = None,
                 step_cost_vs_adakaon_reason: str | None = None,
                 step_cost_native_reason: str | None = None) -> dict[str, Any]:
    """The whole consolidated result, as data. ``render_markdown`` reads only this."""
    rows = (anima_findings(anima)
            + step_cost_findings(step_cost, step_cost_vs_adakaon)
            + battery_findings(battery))

    matrix: dict[str, dict[str, Any]] = {}
    for axis in AXES:
        matrix[axis] = {}
        for comparison in COMPARISONS:
            printed = [r for r in rows
                       if r["axis"] == axis and r["comparison"] == comparison]
            # A row may be printed and still be no part of the roll-up: an oversubscribed
            # timing, or a pair that answers a neighbouring question.
            cell = [r["verdict"] for r in printed if r.get("rollup", True)]
            matrix[axis][comparison] = {
                "verdict": roll_up(cell) if cell else NO_EVIDENCE,
                "rows": len(printed),
                "wins": sum(1 for v in cell if v == WINS),
                "losses": sum(1 for v in cell if v == LOSES),
                "not_significant": sum(1 for v in cell if v == NS),
                "ties": sum(1 for v in cell if v == TIE),
                "no_evidence": sum(1 for v in cell if v == NO_EVIDENCE),
                "invalid": sum(1 for r in printed if r["verdict"] == INVALID),
                "excluded": sum(1 for r in printed if not r.get("rollup", True)),
            }

    missing = [name for name, payload in
               (("anima", anima), ("step_cost", step_cost), ("battery", battery))
               if payload is None]
    return {
        "schema": SCHEMA,
        "question": ("Does Nekaon beat Adakaon and AdamW on speed, quality and memory for "
                     "diffusion fine-tuning?"),
        "sources": source_header(anima, anima_reason, step_cost, step_cost_reason,
                                 battery, battery_reason,
                                 step_cost_vs_adakaon, step_cost_vs_adakaon_reason,
                                 step_cost_native, step_cost_native_reason),
        "missing_sources": missing,
        "verdict_rules": VERDICT_RULES,
        "verdict_matrix": matrix,
        "findings": rows,
        "attribution": attribution_table(battery),
        "attribution_note": SDXL_LOOKAHEAD_NOTE,
        "integrity": integrity_table({"step_cost": step_cost,
                                      "step_cost_vs_adakaon": step_cost_vs_adakaon,
                                      "step_cost_native": step_cost_native}),
        "native": native_table(step_cost_native),
        "memory_by_construction": MEMORY_BY_CONSTRUCTION,
        "caveats": caveats(anima, step_cost, battery,
                           step_cost_vs_adakaon=step_cost_vs_adakaon),
    }


# --------------------------------------------------------------------------- markdown


def _num(value: float | None, spec: str = ".6g") -> str:
    if value is None:
        return "—"
    value = float(value)
    return "—" if not math.isfinite(value) else format(value, spec)


def _interval(row: dict[str, Any], spec: str = ".6g") -> str:
    if row.get("ci_low") is None or row.get("ci_high") is None:
        if row.get("mean") is not None:
            return f"{_num(row['mean'], spec)} (no interval)"
        if row.get("value") is not None:
            return f"{_num(row['value'], spec)} (deterministic)"
        return "—"
    return (f"{_num(row['mean'], spec)} "
            f"[{_num(row['ci_low'], spec)}, {_num(row['ci_high'], spec)}]")


def _finding_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| Level | Metric | Value / 95% CI | N | Verdict | Source |",
             "| --- | --- | ---: | ---: | --- | --- |"]
    for row in rows:
        note = f"<br>*{row['note']}*" if row.get("note") else ""
        n = row.get("n")
        lines.append(f"| {row['level']} | {row['metric']}{note} | {_interval(row)} | "
                     f"{n if n is not None else '—'} | **{row['verdict']}** | "
                     f"`{row['source']}` |")
    lines.append("")
    return lines


AXIS_PREAMBLE = {
    "quality": ("Final validation loss and the train/val gap. Nothing here is perceptual: "
                "there is no FID and no KID in this report, so a quality win is a win on an "
                "objective loss, on these tasks, at these learning rates."),
    "speed": ("Three levels that are never mixed into one sentence, because they answer "
              "three different questions: **(a)** the optimizer step alone at SDXL parameter "
              "scale, **(b)** the full training step on a real model, and **(c)** how much of "
              "that step the optimizer even is -- the ceiling on what any optimizer speedup "
              "can buy end to end."),
    "memory": ("Also three levels: **(a)** optimizer state bytes per parameter, which is "
               "arithmetic and carries no interval; **(b)** peak VRAM during real training; "
               "**(c)** the largest parameter count that still fits on this GPU."),
}


def _render_native(native: dict[str, Any] | None) -> list[str]:
    """The native-path table: ordering evidence, explicitly not a paired result."""
    if not native:
        return []
    L = ["## Native (non-Triton) path, unpaired", "",
         f"`step_cost.py --native --skip-paired` at R={native.get('R')} "
         f"({(native.get('params') or 0) / 1e6:.1f} M params, {native.get('reps')} reps per "
         "arm). **Every number below is a solo timing with no interval and no pairing**: the "
         "arms ran one after another, not interleaved, so nothing here is comparable with "
         "the paired ratios above and nothing here carries a verdict. It is printed to show "
         "what the Triton kernels are standing in for, and its only safe reading is the "
         "ordering of the arms.", "",
         "| Arm | ms / step (solo, no interval) | State B/p | Peak allocated |",
         "| --- | ---: | ---: | --- |"]
    for row in native["rows"]:
        peak = row.get("peak_allocated_bytes")
        peak_cell = "—" if peak is None else f"{peak / 2 ** 30:.2f} GiB"
        if row.get("exceeds_vram"):
            peak_cell += f" — **{SPILLED_LABEL}**"
        L.append(f"| `{row['arm']}` | {_num(row.get('ms_step_solo'), '.4g')} | "
                 f"{_num(row.get('state_bytes_per_param'), '.3f')} | {peak_cell} |")
    L.append("")
    return L


def _render_integrity(integrity: dict[str, Any] | None) -> list[str]:
    """Which rows survived the oversubscription rule, and which did not — for every arm."""
    if not integrity:
        return []
    total = integrity.get("gpu_total_bytes")
    L = ["## Measurement integrity: VRAM oversubscription", "",
         f"This GPU has {total / 2 ** 30:.2f} GiB of VRAM." if total else
         "This GPU's VRAM size was not recorded.",
         "", "Windows drives it through WDDM, where an allocation larger than VRAM does not "
         "raise `CUDA out of memory`: the driver pages the surplus into host RAM and the "
         "step keeps running over PCIe. So an arm whose peak exceeds the line above was not "
         "measured on this card alone, and neither its own timing nor the baseline it was "
         "interleaved with is a measurement of an optimizer. The rule is applied below to "
         "every arm of every step-cost file, and the arms it did not touch are listed too. "
         "The peak is the one the solo phase recorded; an arm that spilled there is treated "
         "as spilled in its paired phase too, even though the pair runs at a smaller R — "
         "which the baseline medians in the last column corroborate, since an interleaved "
         "baseline slows down by an order of magnitude next to a spilled arm.",
         ""]
    if integrity.get("capacity_no_oom"):
        L += [f"> **Capacity: {integrity['capacity_note']}.** Memory level (c) is `no "
              "evidence` on this machine — the sweep measured where the loop stopped, not "
              "where the GPU did.", ""]
    L += ["| File | Arm | Peak allocated | Solo timing | Paired timing |",
          "| --- | --- | --- | --- | --- |"]
    for row in integrity["arms"]:
        peak = row.get("peak_allocated_bytes")
        peak_cell = "—" if peak is None else f"{peak / 2 ** 30:.2f} GiB"
        if row.get("exceeds_vram"):
            peak_cell += f" — **{SPILLED_LABEL}**"
        solo = (f"{_num(row.get('solo_ms'), '.4g')} ms" if row.get("solo_ms") is not None
                else "—")
        if row.get("solo_status") == INVALID:
            solo += f" — **{INVALID}**"
        if row.get("paired_status") is None:
            pair = "not paired in this file"
        else:
            pair = (f"ratio {_num(row.get('paired_ratio'), '.4g')} vs "
                    f"`{row.get('paired_baseline')}` at R={row.get('paired_R')} "
                    f"(baseline median {_num(row.get('paired_baseline_ms'), '.4g')} ms)")
            if row["paired_status"] == INVALID:
                pair += f" — **{INVALID}**"
        L.append(f"| `{row['source']}` (R={row.get('R')}) | `{row['arm']}` | {peak_cell} | "
                 f"{solo} | {pair} |")
    L.append("")
    return L


def render_markdown(report: dict[str, Any]) -> str:
    """The whole page, rebuildable from ``evidence.json`` alone."""
    L: list[str] = ["# Nekaon: what is measured, and what it does and does not show", "",
                    report["question"], "",
                    "This page exists so the claim can be checked rather than believed. Where "
                    "Nekaon wins, where it loses and where the measurement cannot tell are "
                    "printed in the same tables, in the same font, next to the intervals that "
                    "produced them.", ""]

    L += ["## Sources", "",
          "| Source | Present | kaon | Commit | GPU | Electrical state | Detail |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in report["sources"]:
        present = "yes" if row["present"] else f"**NO** — {row['reason']}"
        L.append(f"| `{row['source']}` | {present} | {row.get('kaon_version') or '—'} | "
                 f"`{str(row.get('commit') or '—')[:12]}` | {row.get('gpu') or '—'} | "
                 f"{row.get('power') or '—'} | {row.get('detail') or '—'} |")
    L.append("")
    if report["missing_sources"]:
        L += [f"> **Not measured: {', '.join(report['missing_sources'])}.** Every axis that "
              "depended on those files reads `no evidence` below. An absent measurement is "
              "never rendered as a neutral result, and never as a favourable one.", ""]

    L += ["## Verdict", "",
          "| Axis | " + " | ".join(COMPARISON_LABELS[c] for c in COMPARISONS) + " |",
          "| --- | " + " | ".join("---" for _ in COMPARISONS) + " |"]
    for axis in AXES:
        cells = []
        for comparison in COMPARISONS:
            cell = report["verdict_matrix"][axis][comparison]
            counts = (f"{cell['wins']}W / {cell['losses']}L / {cell['not_significant']} n.s. / "
                      f"{cell['ties']} tie" if cell["rows"] else "no rows")
            if cell.get("excluded"):
                counts += (f" / {cell['excluded']} excluded"
                           + (f" ({cell['invalid']} oversubscribed)"
                              if cell.get("invalid") else ""))
            cells.append(f"**{cell['verdict']}** ({counts})")
        L.append(f"| {AXIS_LABELS[axis]} | " + " | ".join(cells) + " |")
    L += ["",
          "A roll-up is never the finding. `mixed` means the rows below disagree, and the rows "
          "are the answer; `n.s.` means every row's interval covered zero. Read the level "
          "tables.", "",
          "### How each verdict is made", "", report["verdict_rules"], ""]

    for axis in AXES:
        L += [f"## {AXIS_LABELS[axis]}", "", AXIS_PREAMBLE[axis], ""]
        if axis == "memory":
            L += ["> " + report["memory_by_construction"], ""]
        for comparison in COMPARISONS:
            rows = [r for r in report["findings"]
                    if r["axis"] == axis and r["comparison"] == comparison]
            L += [f"### {COMPARISON_LABELS[comparison]} — {AXIS_LABELS[axis].lower()}", ""]
            if not rows:
                L += ["**no evidence** — no supplied source measured this axis for this "
                      "comparison.", ""]
                continue
            L += _finding_table(sorted(rows, key=lambda r: (r["level"], r["metric"])))

    L += ["## Attribution: which knob did it", "",
          "Each pair below differs in exactly one setting, so its difference is attributable "
          "to that setting and to nothing else. This is the only place in the report where a "
          "mechanism — rather than a whole optimizer — is what is being measured.", ""]
    attribution = report["attribution"]
    if not attribution:
        L += ["**no evidence** — the control battery was not supplied, so no single-knob pair "
              "could be formed.", ""]
    else:
        L += ["| Knob | Metric | Paired difference [95% CI] | N | Verdict |",
              "| --- | --- | ---: | ---: | --- |"]
        for row in attribution:
            n = row.get("n")
            L.append(f"| {row['knob']} | {row['metric']} | {_interval(row)} | "
                     f"{n if n is not None else '—'} | **{row['verdict']}** |")
        L += ["",
              "A negative difference favours the left arm of the pair. The lookahead row is "
              "the only measurement in this report of what Nekaon's own mechanism does, "
              "isolated from the codec and from Adakaon.", ""]
    if report.get("attribution_note"):
        L += ["> " + report["attribution_note"], ""]

    L += _render_native(report.get("native"))
    L += _render_integrity(report.get("integrity"))

    L += ["## Caveats", "",
          "None of these are hedges; each one bounds what the tables above may be used to "
          "claim.", ""]
    L += [f"* {item}" for item in report["caveats"]]
    L.append("")
    return "\n".join(L)


# ------------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Consolidate the Nekaon evidence sources into EVIDENCE.md + evidence.json.")
    parser.add_argument("--anima", help="results.json from anima/aggregate.py")
    parser.add_argument("--step-cost", dest="step_cost", help="JSON from step_cost.py")
    parser.add_argument("--step-cost-vs-adakaon", dest="step_cost_vs_adakaon",
                        help="JSON from step_cost.py --baseline adakaon_4bit")
    parser.add_argument("--step-cost-native", dest="step_cost_native",
                        help="JSON from step_cost.py --native --skip-paired")
    parser.add_argument("--battery",
                        help="results_evidence.json from benchmarks/control/battery.py")
    parser.add_argument("--out", default="EVIDENCE.md", help="markdown output path")
    parser.add_argument("--json-out", dest="json_out", help="machine-readable output path")
    args = parser.parse_args(argv)

    anima, anima_why = load_source(args.anima)
    step_cost, step_cost_why = load_source(args.step_cost)
    battery, battery_why = load_source(args.battery)
    vs_adakaon, vs_adakaon_why = load_source(args.step_cost_vs_adakaon)
    native, native_why = load_source(args.step_cost_native)

    report = build_report(anima, step_cost, battery, anima_reason=anima_why,
                          step_cost_reason=step_cost_why, battery_reason=battery_why,
                          step_cost_vs_adakaon=vs_adakaon,
                          step_cost_vs_adakaon_reason=vs_adakaon_why,
                          step_cost_native=native, step_cost_native_reason=native_why)

    out = Path(args.out)
    if out.parent != Path(""):
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {out}")
    if args.json_out:
        json_out = Path(args.json_out)
        if json_out.parent != Path(""):
            json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {json_out}")

    for axis in AXES:
        cells = " | ".join(
            f"{COMPARISON_LABELS[c]}: {report['verdict_matrix'][axis][c]['verdict']}"
            for c in COMPARISONS)
        print(f"  {AXIS_LABELS[axis]:<8} {cells}")
    if report["missing_sources"]:
        print(f"  NOT MEASURED: {', '.join(report['missing_sources'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
