"""CPU-side tests for the Nekaon evidence consolidator.

Nothing here touches a GPU or a real measurement: the three sources are synthesised so
that every verdict rule can be exercised on a known answer -- a clear win, a clear loss,
an interval that covers zero, a source that is simply absent -- and so the memory
identity between Nekaon and Adakaon at the same ``momentum_dtype`` is asserted rather
than assumed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


report = _load("nekaon_evidence_report", ROOT / "benchmarks" / "nekaon_evidence" / "report.py")


# --------------------------------------------------------------------- fixture builders


def _bootstrap(values: list[float], low: float, high: float) -> dict[str, Any]:
    """An ``aggregate.py``-shaped bootstrap block with the interval stated outright."""
    mean = sum(values) / len(values)
    return {"mean": mean, "ci_low": low, "ci_high": high, "n": len(values),
            "resamples": 4000, "seed": 20240607, "crosses_zero": low <= 0.0 <= high}


def anima_fixture(*, quality: str = "win", timings_comparable: bool = True,
                  boundary: bool = True, phase_a_complete: bool = False) -> dict[str, Any]:
    """A synthetic ``anima/aggregate.py`` result with a dialable quality outcome."""
    intervals = {
        "win": (-0.0040, -0.0010),      # entirely below zero: Nekaon lower val loss
        "loss": (0.0010, 0.0040),       # entirely above zero
        "ns": (-0.0030, 0.0025),        # covers zero
    }[quality]

    def paired(left: str, right: str) -> dict[str, Any]:
        return {
            "left": left, "right": right, "seeds": [43, 44, 45, 46, 47],
            "metrics": {
                "final_val": {"bootstrap": _bootstrap([-0.002] * 5, *intervals),
                              "left_wins": 4, "n": 5, "direction": "lower is better"},
                "raw_gap": {"bootstrap": _bootstrap([-0.001] * 5, -0.002, 0.0015),
                            "left_wins": 3, "n": 5, "direction": "lower is better"},
                "active_seconds": {"bootstrap": _bootstrap([9.0] * 5, 4.0, 14.0),
                                   "left_wins": 0, "n": 5, "direction": "lower is better"},
                "ms_per_step": {"bootstrap": _bootstrap([45.0] * 5, 20.0, 70.0),
                                "left_wins": 0, "n": 5, "direction": "lower is better"},
                # Against AdamW the codec is worth real VRAM; against Adakaon the two hold
                # the same buffers, so their peak differs only by allocator noise.
                "peak_gib": ({"bootstrap": _bootstrap([-0.30] * 5, -0.45, -0.15),
                              "left_wins": 5, "n": 5, "direction": "lower is better"}
                             if right == "adamw_fused" else
                             {"bootstrap": _bootstrap([0.001] * 5, -0.004, 0.006),
                              "left_wins": 2, "n": 5, "direction": "lower is better"}),
            },
        }

    return {
        "schema": "nekaon-evidence-anima-aggregate-v1",
        "provenance": {"kaon_version": "0.7.13", "commit": "abcdef0123456789",
                       "worktree": "/home/k/K-Optimizers"},
        "confound_note": "Every arm runs its own house configuration and only the LR is tuned.",
        "telemetry_note": "The nekaon_fused arm pays MSAM's inert-lookahead telemetry.",
        "phase_a": {"complete": phase_a_complete, "runs": 5, "expected": 9},
        "lr_boundary": {
            "nekaon_fused": {"lr": 5e-5, "grid": [5e-5, 1e-4, 2e-4], "boundary": boundary},
            "adakaon_fused": {"lr": 1e-4, "grid": [5e-5, 1e-4, 2e-4], "boundary": False},
            "adamw_fused": {"lr": 1e-4, "grid": [5e-5, 1e-4, 2e-4], "boundary": False},
        },
        "arm_optimizer_configs": {
            "nekaon_fused": {"run_id": "b_seed43", "optimizer": {"type": "kaon.Nekaon"}},
        },
        "seeds": {"paired": [43, 44, 45, 46, 47], "held_out": [44, 45, 46, 47],
                  "phase_a_seed": 43},
        "paired": {
            "nekaon_fused_minus_adamw_fused": paired("nekaon_fused", "adamw_fused"),
            "nekaon_fused_minus_adakaon_fused": paired("nekaon_fused", "adakaon_fused"),
        },
        "validation": {"ok": True, "seeds": {}},
        "power": {
            "signatures": {"AC/60W": ["b_seed43"]},
            "timings_comparable": timings_comparable,
            "note": "The GPU is power limited to 60 W on AC and 35 W on battery.",
        },
    }


def step_cost_fixture(*, nekaon_ratio: tuple[float, float, float] = (1.10, 1.04, 1.16),
                      nekaon_bpp: float = 0.56, adakaon_bpp: float = 0.56,
                      adamw_bpp: float = 4.0, on_ac: bool = True,
                      gpu_total_bytes: int | None = None, oom_reached: bool = True,
                      spilled_peak: int | None = None) -> dict[str, Any]:
    """A synthetic ``step_cost.py`` payload: ratios, B/p, capacity and the step fraction.

    ``gpu_total_bytes`` + ``spilled_peak`` reproduce a WDDM run in which an arm allocated
    more than the card holds and kept going; ``oom_reached=False`` reproduces a capacity
    sweep that ran off the top of its range without a single OOM.
    """
    def arm(bpp: float, peak: int = 3 * 2 ** 30) -> dict[str, Any]:
        return {"param_dtype": "bfloat16", "params": 397_862_400, "tensors": 468,
                "state_bytes_per_param": bpp, "peak_allocated_bytes": peak,
                "ms_step_solo": 40.0, "note": "synthetic"}

    ratio, lo, hi = nekaon_ratio
    meta = {"kaon_version": "0.7.13", "commit": "abcdef0123456789",
            "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU", "on_ac": on_ac,
            "power_note": "on AC (BatteryStatus=2)" if on_ac else "on battery",
            "timestamp": "2026-09-18T10:00:00-0600"}
    if gpu_total_bytes is not None:
        meta["gpu_total_bytes"] = gpu_total_bytes
    payload = {
        "schema": "nekaon_evidence.step_cost/1",
        "meta": meta,
        "config": {"R": 5, "reps": 60, "native": False, "paired_R": 3},
        "bag": {"R": 5, "tensors": 780, "params": 663_104_000},
        "arms": {
            "adamw_bf16": arm(adamw_bpp),
            "adamw_fp32": {"param_dtype": "float32", "R": 5, "oom": True, "note": "16 B/p"},
            "adakaon_4bit": arm(adakaon_bpp),
            "nekaon_4bit": arm(nekaon_bpp),
        },
        "paired": {
            "nekaon_4bit": {"baseline": "adamw_bf16", "reps": 60, "pairs_used": 55,
                            "ms_arm_median": 44.0, "ms_baseline_median": 40.0,
                            "ratio": ratio, "ci_lo": lo, "ci_hi": hi,
                            "verdict": "slower" if lo > 1 else "n.s."},
            "adakaon_4bit": {"baseline": "adamw_bf16", "reps": 60, "pairs_used": 55,
                             "ms_arm_median": 42.0, "ms_baseline_median": 40.0,
                             "ratio": 1.05, "ci_lo": 1.01, "ci_hi": 1.09,
                             "verdict": "slower"},
        },
        "fraction": {"C": 128, "batch": 8, "px": 64, "steps": 32, "arms": {
            "adamw_bf16": {"ms_step_total": 100.0, "ms_optimizer": 5.0,
                           "optimizer_fraction": 0.05, "params": 4_000_000,
                           "measured_steps": 16},
            "adakaon_4bit": {"ms_step_total": 100.0, "ms_optimizer": 8.0,
                             "optimizer_fraction": 0.08, "params": 4_000_000,
                             "measured_steps": 16},
            "nekaon_4bit": {"ms_step_total": 100.0, "ms_optimizer": 9.0,
                            "optimizer_fraction": 0.09, "params": 4_000_000,
                            "measured_steps": 16},
        }},
        "capacity": {
            "adamw_bf16": {"max_R": 7, "params": 928_345_600, "oom_at_R": 8,
                           "swept": [1, 12, 1]},
            "adakaon_4bit": {"max_R": 9, "params": 1_193_587_200, "oom_at_R": 10,
                             "swept": [1, 12, 1]},
            "nekaon_4bit": {"max_R": 9, "params": 1_193_587_200, "oom_at_R": 10,
                            "swept": [1, 12, 1]},
        },
    }
    if not oom_reached:
        # Every arm ran to the top of the sweep and none of them ever raised: the sweep
        # found the end of its own range, not the end of the GPU.
        for entry in payload["capacity"].values():
            entry.update({"max_R": 12, "params": 1_591_449_600, "oom_at_R": None,
                          "oom_at_params": None})
    if spilled_peak is not None:
        payload["arms"]["adamw_fp32"] = arm(8.0, peak=spilled_peak)
        payload["arms"]["adamw_fp32"]["param_dtype"] = "float32"
        payload["arms"]["adamw_fp32"]["ms_step_solo"] = 842.0
        payload["paired"]["adamw_fp32"] = {
            "baseline": "adamw_bf16", "reps": 60, "pairs_used": 55,
            "ms_arm_median": 165.0, "ms_baseline_median": 411.0,
            "ratio": 0.40, "ci_lo": 0.399, "ci_hi": 0.404, "verdict": "faster"}
    return payload


def vs_adakaon_fixture(*, nekaon_ratio: tuple[float, float, float] = (1.45, 1.44, 1.46),
                       k0_ratio: tuple[float, float, float] = (1.0006, 0.9986, 1.0026),
                       ) -> dict[str, Any]:
    """A ``step_cost.py`` payload whose ``--baseline`` is ``adakaon_4bit``.

    This is the only shape that measures Nekaon against Adakaon *directly*, in one process,
    with a real paired interval -- which is what makes the ratio-of-ratios row unnecessary.
    """
    def pair(values: tuple[float, float, float]) -> dict[str, Any]:
        ratio, lo, hi = values
        return {"baseline": "adakaon_4bit", "reps": 150, "pairs_used": 145,
                "ms_arm_median": 58.0, "ms_baseline_median": 40.0,
                "ratio": ratio, "ci_lo": lo, "ci_hi": hi,
                "verdict": "slower" if lo > 1 else "n.s."}

    def arm(bpp: float) -> dict[str, Any]:
        return {"param_dtype": "bfloat16", "params": 397_862_400, "tensors": 468,
                "state_bytes_per_param": bpp, "peak_allocated_bytes": 2 * 2 ** 30,
                "ms_step_solo": 55.0, "note": "synthetic"}

    return {
        "schema": "nekaon_evidence.step_cost/1",
        "meta": {"kaon_version": "0.7.13", "commit": "abcdef0123456789",
                 "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU", "gpu_total_bytes": 8 * 10 ** 9,
                 "on_ac": True, "power_note": "on AC (BatteryStatus=2)"},
        "config": {"R": 3, "reps": 150, "native": False, "baseline": "adakaon_4bit",
                   "paired_R": 3},
        "bag": {"R": 3, "tensors": 468, "params": 397_862_400},
        "arms": {"adakaon_4bit": arm(0.56), "nekaon_4bit": arm(0.56),
                 "nekaon_k0_4bit": arm(0.56), "adakaon_bf16": arm(2.03)},
        "paired": {"nekaon_4bit": pair(nekaon_ratio), "nekaon_k0_4bit": pair(k0_ratio),
                   "adakaon_bf16": pair((1.18, 1.17, 1.19))},
    }


def native_fixture() -> dict[str, Any]:
    """A ``--native --skip-paired`` payload: solo timings, no ``paired`` block at all."""
    def arm(bpp: float, ms: float, peak: int) -> dict[str, Any]:
        return {"param_dtype": "bfloat16", "params": 663_104_000, "tensors": 780,
                "state_bytes_per_param": bpp, "peak_allocated_bytes": peak,
                "ms_step_solo": ms, "note": "synthetic native"}

    return {
        "schema": "nekaon_evidence.step_cost/1",
        "meta": {"kaon_version": "0.7.13", "commit": "abcdef0123456789",
                 "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU", "gpu_total_bytes": 8 * 10 ** 9,
                 "on_ac": True, "power_note": "on AC (BatteryStatus=2)"},
        "config": {"R": 5, "reps": 100, "native": True},
        "bag": {"R": 5, "tensors": 780, "params": 663_104_000},
        "arms": {"adamw_bf16": arm(4.0, 48.8, 5 * 2 ** 30),
                 "nekaon_4bit": arm(0.56, 585.3, 3 * 2 ** 30),
                 "adamw_fp32": arm(8.0, 828.7, 10_610_063_360)},
    }


def _entry(te: list[float], gap: list[float], ms: list[float], bpp: float,
           *, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    n = len(te)
    entry = {
        "te": sum(te) / n, "gap": sum(gap) / n, "ms": sum(ms) / n,
        "bpp": bpp, "cgap": sum(gap) / n, "cte": sum(te) / n,
        "opt_ms": sum(ms) / n / 4.0, "lms": 3.0, "tr": 0.06, "traj": [],
        "family": "in-house", "blurb": "synthetic", "sig": "C128_N2000_s5",
        "per_seed": {
            "seeds": list(range(n)),
            "te": list(te), "gap": list(gap), "tr": [0.06] * n,
            "ms": list(ms), "opt_ms": [m / 4.0 for m in ms], "bpp": [bpp] * n,
            "cte": list(te), "cgap": list(gap), "ctr": [0.06] * n,
        },
    }
    if meta:
        entry["meta"] = meta
    return entry


BATTERY_META = {"timestamp": "2026-09-18T09:00:00-06:00", "kaon_version": "0.7.13",
                "commit": "abcdef0", "device": "cuda", "gpu": "RTX 4060 Laptop",
                "power": "60.00 W", "battery_status": "2", "on_ac": True}


def battery_fixture(*, nekaon_te: list[float] | None = None) -> dict[str, Any]:
    """A synthetic battery store: Nekaon clearly better on loss, clearly slower on ms."""
    nekaon_te = nekaon_te or [0.0700, 0.0702, 0.0698, 0.0701, 0.0699]
    return {
        "torch.AdamW (fused)": _entry([0.0800, 0.0802, 0.0798, 0.0801, 0.0799],
                                      [0.0160, 0.0162, 0.0158, 0.0161, 0.0159],
                                      [12.0, 12.1, 11.9, 12.0, 12.1], 8.0,
                                      meta=BATTERY_META),
        "Adakaon-bf16-fused": _entry([0.0750, 0.0752, 0.0748, 0.0751, 0.0749],
                                     [0.0140, 0.0142, 0.0138, 0.0141, 0.0139],
                                     [13.0, 13.1, 12.9, 13.0, 13.1], 2.03,
                                     meta=BATTERY_META),
        "Adakaon-4bit-fused": _entry([0.0752, 0.0753, 0.0749, 0.0752, 0.0750],
                                     [0.0141, 0.0143, 0.0139, 0.0142, 0.0140],
                                     [13.2, 13.3, 13.1, 13.2, 13.3], 0.56,
                                     meta=BATTERY_META),
        "Nekaon-fused": _entry(nekaon_te,
                               [0.0120, 0.0122, 0.0118, 0.0121, 0.0119],
                               [14.0, 14.1, 13.9, 14.0, 14.1], 0.56,
                               meta=BATTERY_META),
        "Nekaon-bf16-fused": _entry([0.0701, 0.0703, 0.0699, 0.0702, 0.0700],
                                    [0.0121, 0.0123, 0.0119, 0.0122, 0.0120],
                                    [13.8, 13.9, 13.7, 13.8, 13.9], 2.03,
                                    meta=BATTERY_META),
        "Nekaon-k0-4bit-fused": _entry([0.0710, 0.0712, 0.0708, 0.0711, 0.0709],
                                       [0.0150, 0.0152, 0.0148, 0.0151, 0.0149],
                                       [13.3, 13.4, 13.2, 13.3, 13.4], 0.56,
                                       meta=BATTERY_META),
    }


def _rows(built: dict[str, Any], axis: str, comparison: str,
          source: str | None = None) -> list[dict[str, Any]]:
    return [r for r in built["findings"]
            if r["axis"] == axis and r["comparison"] == comparison
            and (source is None or r["source"] == source)]


# --------------------------------------------------------------------- the verdict rules


def test_interval_entirely_below_zero_is_a_win_and_above_zero_is_a_loss() -> None:
    assert report.verdict_from_interval(-0.004, -0.001) == report.WINS
    assert report.verdict_from_interval(0.001, 0.004) == report.LOSES
    # Higher-is-better flips both, and nothing else.
    assert report.verdict_from_interval(-0.004, -0.001, lower_is_better=False) == report.LOSES
    assert report.verdict_from_interval(0.001, 0.004, lower_is_better=False) == report.WINS


def test_interval_that_covers_zero_is_never_a_win() -> None:
    assert report.verdict_from_interval(-0.003, 0.002) == report.NS
    assert report.verdict_from_interval(0.0, 0.004) == report.NS       # touching the null
    assert report.verdict_from_interval(-0.004, 0.0) == report.NS


def test_a_missing_interval_is_not_significant_rather_than_a_win() -> None:
    assert report.verdict_from_interval(None, None) == report.NS
    assert report.verdict_from_interval(float("nan"), float("nan")) == report.NS


def test_ratio_verdicts_are_taken_against_one_not_zero() -> None:
    assert report.verdict_from_ratio(0.90, 0.97) == report.WINS     # faster
    assert report.verdict_from_ratio(1.04, 1.16) == report.LOSES    # slower
    assert report.verdict_from_ratio(0.97, 1.05) == report.NS


def test_roll_up_keeps_disagreement_visible() -> None:
    assert report.roll_up([report.WINS, report.WINS]) == report.WINS
    assert report.roll_up([report.WINS, report.LOSES]) == report.MIXED
    assert report.roll_up([report.NS, report.NS]) == report.NS
    assert report.roll_up([report.TIE, report.TIE]) == report.TIE
    assert report.roll_up([]) == report.NO_EVIDENCE
    assert report.roll_up([report.NO_EVIDENCE]) == report.NO_EVIDENCE
    # A single no-evidence row never launders a loss into something softer.
    assert report.roll_up([report.NO_EVIDENCE, report.LOSES]) == report.LOSES


def test_paired_differences_use_the_battery_student_t_table() -> None:
    stat = report.paired_diff([1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.0, 0.0, 0.0, 0.0])
    assert stat["n"] == 5
    assert stat["mean"] == pytest.approx(3.0)
    # 2.776 * stdev(1..5)/sqrt(5) = 2.776 * 1.5811/2.2360 = 1.9629
    assert stat["half_width"] == pytest.approx(1.96294, rel=1e-4)
    assert stat["crosses_zero"] is False


def test_pairing_refuses_to_zip_lists_of_different_lengths() -> None:
    with pytest.raises(ValueError, match="cannot pair 3 values against 5"):
        report.paired_diff([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0, 5.0])


# ------------------------------------------------------------------------- clear outcomes


def test_a_clear_quality_win_is_reported_as_a_win_with_its_interval() -> None:
    built = report.build_report(anima_fixture(quality="win"), None, battery_fixture())
    rows = _rows(built, "quality", report.VS_ADAMW, "anima")
    val = next(r for r in rows if "Final val" in r["metric"])
    assert val["verdict"] == report.WINS
    assert val["ci_high"] < 0
    assert val["n"] == 5
    assert built["verdict_matrix"]["quality"][report.VS_ADAMW]["verdict"] == report.WINS


def test_a_clear_quality_loss_is_reported_as_a_loss() -> None:
    built = report.build_report(anima_fixture(quality="loss"), None, None)
    val = next(r for r in _rows(built, "quality", report.VS_ADAMW, "anima")
               if "Final val" in r["metric"])
    assert val["verdict"] == report.LOSES
    assert built["verdict_matrix"]["quality"][report.VS_ADAMW]["verdict"] == report.LOSES
    assert report.WINS not in report.render_markdown(built).split("## Quality")[1][:600]


def test_an_interval_that_crosses_zero_is_rendered_as_not_significant() -> None:
    built = report.build_report(anima_fixture(quality="ns"), None, None)
    val = next(r for r in _rows(built, "quality", report.VS_ADAMW, "anima")
               if "Final val" in r["metric"])
    assert val["verdict"] == report.NS
    assert val["ci_low"] < 0 < val["ci_high"]
    assert built["verdict_matrix"]["quality"][report.VS_ADAMW]["not_significant"] >= 1


def test_a_slower_optimizer_is_printed_as_a_speed_loss_not_omitted() -> None:
    built = report.build_report(None, step_cost_fixture(), None)
    row = next(r for r in _rows(built, "speed", report.VS_ADAMW, "step_cost")
               if "time ratio" in r["metric"])
    assert row["verdict"] == report.LOSES
    assert row["mean"] == pytest.approx(1.10)
    assert "**loses**" in report.render_markdown(built)


# --------------------------------------------------------------------- absent sources


def test_an_absent_source_marks_its_axes_no_evidence_and_says_so() -> None:
    built = report.build_report(None, None, None, anima_reason="not supplied on the command line",
                                step_cost_reason="`x.json` does not exist",
                                battery_reason="not supplied on the command line")
    assert built["missing_sources"] == ["anima", "step_cost", "battery"]
    for axis in report.AXES:
        for comparison in report.COMPARISONS:
            assert built["verdict_matrix"][axis][comparison]["verdict"] == report.NO_EVIDENCE
    text = report.render_markdown(built)
    assert "Not measured: anima, step_cost, battery" in text
    assert "does not exist" in text
    # Every axis section still exists: an unmeasured axis is stated, never dropped.
    for label in ("## Quality", "## Speed", "## Memory"):
        assert label in text
    assert text.count("**no evidence**") >= len(report.AXES) * len(report.COMPARISONS)


def test_only_the_missing_axis_loses_its_verdict() -> None:
    built = report.build_report(None, None, battery_fixture())
    assert built["missing_sources"] == ["anima", "step_cost"]
    assert built["verdict_matrix"]["quality"][report.VS_ADAMW]["verdict"] == report.WINS
    # The battery measures B/p, so memory is not evidence-free even without step_cost.
    assert built["verdict_matrix"]["memory"][report.VS_ADAMW]["verdict"] == report.WINS


def test_a_missing_file_is_reported_rather_than_raised(tmp_path: Path) -> None:
    payload, why = report.load_source(tmp_path / "nope.json")
    assert payload is None and "does not exist" in why
    payload, why = report.load_source(None)
    assert payload is None and "not supplied" in why
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    payload, why = report.load_source(broken)
    assert payload is None and "could not be read" in why


# ---------------------------------------------------------------------------- memory


def test_nekaon_and_adakaon_at_the_same_dtype_are_a_tie_by_construction() -> None:
    built = report.build_report(None, step_cost_fixture(nekaon_bpp=0.56, adakaon_bpp=0.56), None)
    bpp = next(r for r in _rows(built, "memory", report.VS_ADAKAON, "step_cost")
               if "B/p" in r["metric"])
    assert bpp["verdict"] == report.TIE
    assert "allocates no state of its own" in bpp["note"]
    capacity = next(r for r in _rows(built, "memory", report.VS_ADAKAON, "step_cost")
                    if "M vs" in r["metric"])
    assert capacity["verdict"] == report.TIE
    assert capacity["value"] == pytest.approx(0.0)


def test_the_memory_advantage_over_adamw_is_attributed_to_the_codec_not_the_lookahead() -> None:
    built = report.build_report(None, step_cost_fixture(), None)
    bpp = next(r for r in _rows(built, "memory", report.VS_ADAMW, "step_cost")
               if "B/p" in r["metric"])
    assert bpp["verdict"] == report.WINS
    assert bpp["value"] == pytest.approx(0.56 - 4.0)
    text = report.render_markdown(built)
    assert "same memory, by" in text and "construction" in text
    assert "factored second moment" in text
    assert "the lookahead contributes exactly nothing to it" in text


def test_a_heavier_nekaon_state_would_be_printed_as_a_memory_loss() -> None:
    built = report.build_report(None, step_cost_fixture(nekaon_bpp=2.03, adakaon_bpp=0.56), None)
    bpp = next(r for r in _rows(built, "memory", report.VS_ADAKAON, "step_cost")
               if "B/p" in r["metric"])
    assert bpp["verdict"] == report.LOSES


# ----------------------------------------------------------------------------- speed


def test_the_three_speed_levels_stay_separate_and_each_carries_its_own_verdict() -> None:
    built = report.build_report(anima_fixture(timings_comparable=True), step_cost_fixture(),
                                battery_fixture())
    levels = {r["level"] for r in built["findings"] if r["axis"] == "speed"}
    assert any(level.startswith("(a)") for level in levels)
    assert any(level.startswith("(b)") for level in levels)
    assert any(level.startswith("(c)") for level in levels)
    fraction = next(r for r in _rows(built, "speed", report.VS_ADAMW, "step_cost")
                    if r["metric"] == "nekaon_4bit optimizer fraction")
    assert fraction["value"] == pytest.approx(0.09)
    assert "at most 9.0%" in fraction["note"]


def test_nekaon_versus_adakaon_at_sdxl_scale_is_declared_unmeasured_not_inferred() -> None:
    built = report.build_report(None, step_cost_fixture(), None)
    indirect = next(r for r in _rows(built, "speed", report.VS_ADAKAON, "step_cost")
                    if "indirect" in r["metric"])
    assert indirect["verdict"] == report.NO_EVIDENCE
    assert indirect["ci_low"] is None
    assert indirect["value"] == pytest.approx(1.10 / 1.05)
    assert "never Nekaon against" in indirect["note"]


def test_mixed_power_signatures_void_the_anima_timings_but_not_its_quality() -> None:
    built = report.build_report(anima_fixture(timings_comparable=False), None, None)
    for row in _rows(built, "speed", report.VS_ADAMW, "anima"):
        assert row["verdict"] == report.NO_EVIDENCE
        assert "power signatures" in row["note"]
    assert built["verdict_matrix"]["speed"][report.VS_ADAMW]["verdict"] == report.NO_EVIDENCE
    assert built["verdict_matrix"]["quality"][report.VS_ADAMW]["verdict"] == report.WINS


def test_a_slower_battery_step_is_a_loss_the_report_prints() -> None:
    built = report.build_report(None, None, battery_fixture())
    ms = next(r for r in _rows(built, "speed", report.VS_ADAMW, "battery")
              if r["metric"].startswith("ms / iteration"))
    assert ms["verdict"] == report.LOSES
    assert ms["mean"] == pytest.approx(2.0, abs=1e-9)


# ------------------------------------------------- WDDM oversubscription on this machine


def test_a_capacity_sweep_without_a_single_oom_measured_nothing() -> None:
    """Every arm reaching the top of the sweep is not seven equal fit limits: on WDDM the
    allocator spills into host RAM instead of raising, so the sweep ended where the loop
    ended. The honest answer is `no evidence`, not `tie`."""
    built = report.build_report(None, step_cost_fixture(gpu_total_bytes=8 * 10 ** 9,
                                                        oom_reached=False), None)
    for comparison in report.COMPARISONS:
        row = next(r for r in _rows(built, "memory", comparison, "step_cost")
                   if r["level"].startswith("(c)"))
        assert row["verdict"] == report.NO_EVIDENCE
        assert "no OOM reached on any arm" in row["note"]
        assert "Windows WDDM" in row["note"]
    text = report.render_markdown(built)
    assert report.WDDM_NO_OOM_NOTE in text
    # The row is counted as the absence it is, and the axis is decided by the rows that
    # did measure something -- here the two identical state layouts.
    cell = built["verdict_matrix"]["memory"][report.VS_ADAKAON]
    assert cell["no_evidence"] == 1
    assert cell["verdict"] == report.TIE


def test_a_capacity_sweep_that_did_oom_keeps_its_result() -> None:
    built = report.build_report(None, step_cost_fixture(gpu_total_bytes=8 * 10 ** 9), None)
    row = next(r for r in _rows(built, "memory", report.VS_ADAMW, "step_cost")
               if r["level"].startswith("(c)"))
    assert row["verdict"] == report.WINS
    assert row["note"] is None


def test_a_timing_taken_above_vram_is_invalid_and_enters_no_verdict() -> None:
    """`adamw_fp32` holds 16 B/p, which does not fit; WDDM serves the surplus from host RAM
    rather than raising, and the interleaved baseline slows down with it. That row is a
    PCIe measurement, so it is printed, labelled, and kept out of every roll-up."""
    payload = step_cost_fixture(gpu_total_bytes=8 * 10 ** 9, spilled_peak=10_610_063_360)
    spilled = report.oversubscribed_arms(payload)
    assert spilled == {"adamw_fp32": 10_610_063_360}

    rows = report._paired_speed_rows(payload, comparison=report.VS_ADAMW, source="step_cost")
    bad = next(r for r in rows if r["metric"].startswith("adamw_fp32"))
    assert bad["verdict"] == report.INVALID
    assert bad["rollup"] is False
    assert "9.88 GiB" in bad["note"]
    # The arms that fit are untouched by the rule.
    good = next(r for r in rows if r["metric"].startswith("nekaon_4bit"))
    assert good["verdict"] == report.LOSES and good["rollup"] is True

    # An invalid row cannot tilt an axis, in either direction.
    assert report.roll_up([report.INVALID]) == report.NO_EVIDENCE
    assert report.roll_up([report.INVALID, report.WINS]) == report.WINS


def test_the_integrity_table_prints_every_arm_and_flags_only_the_spilled_one() -> None:
    payload = step_cost_fixture(gpu_total_bytes=8 * 10 ** 9, spilled_peak=10_610_063_360)
    built = report.build_report(None, payload, None, step_cost_native=native_fixture())
    integrity = built["integrity"]
    assert integrity["gpu_total_bytes"] == 8 * 10 ** 9
    flagged = {r["arm"] for r in integrity["arms"] if r["exceeds_vram"]}
    assert flagged == {"adamw_fp32"}
    fp32 = next(r for r in integrity["arms"]
                if r["arm"] == "adamw_fp32" and r["source"] == "step_cost")
    assert fp32["solo_status"] == report.INVALID
    assert fp32["paired_status"] == report.INVALID
    assert fp32["paired_baseline_ms"] == pytest.approx(411.0)
    nekaon = next(r for r in integrity["arms"]
                  if r["arm"] == "nekaon_4bit" and r["source"] == "step_cost")
    assert nekaon["solo_status"] == "usable" and nekaon["paired_status"] == "usable"
    text = report.render_markdown(built)
    assert "## Measurement integrity: VRAM oversubscription" in text
    assert report.SPILLED_LABEL in text
    assert f"**{report.INVALID}**" in text


def test_a_payload_without_a_recorded_vram_size_flags_nothing() -> None:
    """Unknown is not "it fit". A file that never recorded `gpu_total_bytes` yields no
    oversubscription claims at all, rather than claims made against a guessed capacity."""
    assert report.oversubscribed_arms(step_cost_fixture()) == {}
    assert report.gpu_total_bytes(step_cost_fixture()) is None
    built = report.build_report(None, step_cost_fixture(), None)
    assert built["integrity"] is None
    assert not any(r["verdict"] == report.INVALID for r in built["findings"])


# -------------------------------------------- Nekaon against Adakaon at SDXL scale


def test_the_direct_pair_replaces_the_indirect_ratio_when_its_run_is_supplied() -> None:
    built = report.build_report(None, step_cost_fixture(), None,
                                step_cost_vs_adakaon=vs_adakaon_fixture())
    rows = _rows(built, "speed", report.VS_ADAKAON, "step_cost_vs_adakaon")
    direct = next(r for r in rows if r["metric"] == "nekaon_4bit / adakaon_4bit time ratio")
    assert direct["verdict"] == report.LOSES
    assert direct["mean"] == pytest.approx(1.45)
    assert direct["ci_low"] == pytest.approx(1.44)
    assert direct["ci_high"] == pytest.approx(1.46)
    assert direct["n"] == 145
    # The interval-less ratio-of-ratios is gone: there is no reason to print a worse
    # estimate of a quantity that was measured properly.
    assert not any("indirect" in r["metric"]
                   for r in _rows(built, "speed", report.VS_ADAKAON))
    assert "step_cost_vs_adakaon" in report.render_markdown(built)


def test_without_that_run_the_cell_stays_an_indirect_estimate_with_no_verdict() -> None:
    built = report.build_report(None, step_cost_fixture(), None)
    indirect = next(r for r in _rows(built, "speed", report.VS_ADAKAON, "step_cost")
                    if "indirect" in r["metric"])
    assert indirect["verdict"] == report.NO_EVIDENCE
    assert indirect["ci_low"] is None


def test_the_k0_arm_is_rolled_up_and_the_adakaon_pair_is_context_only() -> None:
    """`nekaon_k0_4bit / adakaon_4bit` is Nekaon against Adakaon and counts. `adakaon_bf16 /
    adakaon_4bit` is the codec knob on Adakaon: true, printed, and no part of this column."""
    built = report.build_report(None, step_cost_fixture(), None,
                                step_cost_vs_adakaon=vs_adakaon_fixture())
    rows = {r["metric"]: r for r in _rows(built, "speed", report.VS_ADAKAON,
                                          "step_cost_vs_adakaon")}
    k0 = rows["nekaon_k0_4bit / adakaon_4bit time ratio"]
    assert k0["verdict"] == report.NS and k0["rollup"] is True
    context = rows["adakaon_bf16 / adakaon_4bit time ratio"]
    assert context["rollup"] is False
    assert "Adakaon against Adakaon" in context["note"]
    assert built["verdict_matrix"]["speed"][report.VS_ADAKAON]["excluded"] >= 1


def test_no_sdxl_lookahead_row_is_manufactured_and_the_page_says_why() -> None:
    """A `nekaon_4bit / nekaon_k0_4bit` ratio at SDXL scale would need an interval the
    stored summaries cannot supply. It is omitted, and the omission is stated."""
    built = report.build_report(None, step_cost_fixture(), battery_fixture(),
                                step_cost_vs_adakaon=vs_adakaon_fixture())
    assert not any("nekaon_k0_4bit" in row["metric"] and "nekaon_4bit /" in row["metric"]
                   for row in built["findings"])
    text = report.render_markdown(built)
    assert "no `nekaon_4bit / nekaon_k0_4bit` row at SDXL scale" in text
    assert "--baseline` is `nekaon_k0_4bit" in text


def test_the_native_table_is_printed_as_unpaired_and_carries_no_verdict() -> None:
    built = report.build_report(None, step_cost_fixture(), None,
                                step_cost_native=native_fixture())
    native = built["native"]
    assert [row["arm"] for row in native["rows"]] == ["adamw_bf16", "nekaon_4bit",
                                                      "adamw_fp32"]
    assert next(r for r in native["rows"] if r["arm"] == "adamw_fp32")["exceeds_vram"]
    # The native file contributes no findings at all: it is context, not a result.
    assert not any(r["source"] == "step_cost_native" for r in built["findings"])
    text = report.render_markdown(built)
    assert "## Native (non-Triton) path, unpaired" in text
    assert "no interval and no pairing" in text


def test_the_anima_gpu_column_uses_the_file_or_stays_empty() -> None:
    """It may not borrow another source's GPU: a campaign that did not record where it ran
    has not said where it ran."""
    assert report._anima_gpu(anima_fixture()) is None
    enriched = anima_fixture()
    enriched["provenance"]["gpu"] = "NVIDIA RTX 3000 Ada Generation Laptop GPU"
    assert report._anima_gpu(enriched) == "NVIDIA RTX 3000 Ada Generation Laptop GPU"
    row = next(r for r in report.build_report(enriched, None, None)["sources"]
               if r["source"] == "anima")
    assert row["gpu"] == "NVIDIA RTX 3000 Ada Generation Laptop GPU"
    assert "AC/60W" in row["power"]


# ------------------------------------------------------------------------ attribution


def test_attribution_isolates_the_lookahead_and_both_codec_pairs() -> None:
    rows = report.attribution_table(battery_fixture())
    knobs = {row["knob"] for row in rows}
    assert knobs == {"lookahead (k=1.5 vs k=0)", "4-bit codec on Nekaon",
                     "4-bit codec on Adakaon"}
    gap = next(r for r in rows
               if r["knob"].startswith("lookahead") and r["metric"] == "Train-test gap (REX)")
    # k=1.5 tightens the gap by 0.003 against its own k=0 twin: one knob, one effect.
    assert gap["mean"] == pytest.approx(-0.003, abs=1e-9)
    assert gap["verdict"] == report.WINS
    assert gap["n"] == 5
    assert gap["ci_low"] is not None and gap["ci_high"] is not None


def test_attribution_reports_a_missing_arm_instead_of_dropping_the_pair() -> None:
    store = battery_fixture()
    del store["Nekaon-k0-4bit-fused"]
    rows = [r for r in report.attribution_table(store) if r["knob"].startswith("lookahead")]
    assert rows and all(r["verdict"] == report.NO_EVIDENCE for r in rows)
    assert all("absent from the battery store" in r["note"] for r in rows)


def test_attribution_is_absent_but_announced_when_the_battery_is() -> None:
    built = report.build_report(anima_fixture(), step_cost_fixture(), None)
    assert built["attribution"] == []
    text = report.render_markdown(built)
    assert "## Attribution: which knob did it" in text
    assert "the control battery was not supplied" in text


# --------------------------------------------------------------------------- caveats


def test_every_mandatory_caveat_is_present_when_the_sources_earn_it() -> None:
    built = report.build_report(
        anima_fixture(timings_comparable=False, boundary=True, phase_a_complete=False),
        step_cost_fixture(on_ac=False), battery_fixture())
    blob = "\n".join(built["caveats"])
    for needle in (
        "Hyperparameter confound",          # AdamW vs kaon house configs
        "edge of the screened grid",        # LR boundary
        "Phase A is incomplete",            # partial screen
        "electrical signature",             # Anima timings not comparable
        "telemetry",                        # inert-lookahead instrumentation
        "_warn_if_inert",                   # named explicitly
        "n is small",                       # small-n on the real model
        "5 seeds",                          # small-n on the battery
        "no FID",                           # proxy != perception
        "power-limited laptop GPU",         # absolute times do not transfer
        "did not fit on this GPU",          # adamw_fp32 OOM
        "not SDXL",                         # the bag is not the model
    ):
        assert needle in blob, needle
    assert all(f"* {item}" in report.render_markdown(built) for item in built["caveats"])


def test_caveats_that_the_sources_do_not_earn_are_not_invented() -> None:
    built = report.build_report(anima_fixture(timings_comparable=True, boundary=False,
                                              phase_a_complete=True), None, None)
    blob = "\n".join(built["caveats"])
    assert "edge of the screened grid" not in blob
    assert "Phase A is incomplete" not in blob
    assert "electrical signature" not in blob
    # The unconditional ones survive, because they are true of the design itself.
    assert "power-limited laptop GPU" in blob


# ------------------------------------------------------------------ the rendered page


def test_the_rules_are_printed_on_the_page_that_applies_them() -> None:
    text = report.render_markdown(report.build_report(anima_fixture(), step_cost_fixture(),
                                                      battery_fixture()))
    assert "### How each verdict is made" in text
    for word in (report.WINS, report.LOSES, report.NS, report.TIE, report.NO_EVIDENCE,
                 report.MIXED):
        assert word in text
    assert "covers zero" in text


def test_the_header_carries_provenance_and_the_electrical_state_of_each_source() -> None:
    text = report.render_markdown(report.build_report(anima_fixture(), step_cost_fixture(),
                                                      battery_fixture()))
    assert "0.7.13" in text
    assert "abcdef012345" in text                      # commit, truncated
    assert "NVIDIA GeForce RTX 4060 Laptop GPU" in text
    assert "on AC (BatteryStatus=2)" in text
    assert "5 paired seeds" in text


def test_the_json_round_trips_and_the_markdown_rebuilds_from_it_alone() -> None:
    built = report.build_report(anima_fixture(), step_cost_fixture(), battery_fixture())
    restored = json.loads(json.dumps(built))
    assert report.render_markdown(restored) == report.render_markdown(built)
    assert restored["schema"] == report.SCHEMA


def test_the_cli_writes_both_artifacts_and_survives_a_missing_source(tmp_path: Path) -> None:
    anima_path = tmp_path / "results.json"
    anima_path.write_text(json.dumps(anima_fixture()), encoding="utf-8")
    battery_path = tmp_path / "results_evidence.json"
    battery_path.write_text(json.dumps(battery_fixture()), encoding="utf-8")
    out = tmp_path / "EVIDENCE.md"
    json_out = tmp_path / "evidence.json"

    code = report.main(["--anima", str(anima_path), "--battery", str(battery_path),
                        "--step-cost", str(tmp_path / "absent.json"),
                        "--out", str(out), "--json-out", str(json_out)])

    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Nekaon:")
    assert "Not measured: step_cost" in text
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["missing_sources"] == ["step_cost"]
    assert payload["schema"] == report.SCHEMA


def test_the_battery_bpp_row_is_a_tie_rather_than_a_degenerate_interval() -> None:
    """Both arms report 0.56 B/p on every seed. A t interval on that is [0, 0], which
    `verdict_from_interval` would call `n.s.` -- but these two are identical by
    construction, and saying so is a stronger statement than failing to separate them."""
    built = report.build_report(None, None, battery_fixture())
    bpp = next(r for r in _rows(built, "memory", report.VS_ADAKAON, "battery"))
    assert bpp["verdict"] == report.TIE
    assert bpp["mean"] == pytest.approx(0.0)
    against_adamw = next(r for r in _rows(built, "memory", report.VS_ADAMW, "battery"))
    assert against_adamw["verdict"] == report.WINS


def test_attribution_separates_a_knob_that_costs_memory_from_one_that_does_not() -> None:
    rows = report.attribution_table(battery_fixture())
    bpp = {r["knob"]: r for r in rows if r["metric"] == "Bytes per param (state)"}
    # k=1.5 vs k=0 changes no buffer at all.
    assert bpp["lookahead (k=1.5 vs k=0)"]["verdict"] == report.TIE
    # The codec does: 0.56 B/p against 2.03 B/p, on both optimizers.
    assert bpp["4-bit codec on Nekaon"]["verdict"] == report.WINS
    assert bpp["4-bit codec on Adakaon"]["verdict"] == report.WINS
    assert bpp["4-bit codec on Nekaon"]["mean"] == pytest.approx(0.56 - 2.03)

