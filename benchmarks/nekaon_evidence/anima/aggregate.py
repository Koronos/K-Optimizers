"""Aggregate the Nekaon evidence campaign into RESULTS.md and results.json.

Reads every ``*_results.json`` written by ``campaign.py`` and reports, without
claiming more than the numbers support:

* the phase-A learning-rate screen, and the LR the fixed rule selected per arm;
* per-arm mean with a 95% percentile bootstrap interval over paired seeds, for final
  validation epsilon-MSE, raw gap, active training seconds, milliseconds per step and
  peak allocator memory;
* per-seed paired differences Nekaon-AdamW and Nekaon-Adakaon, with their bootstrap
  intervals and the per-seed sign (how many of N seeds each side wins);
* the ``[optimizer]`` block each arm actually ran, parsed from its own TOML, because
  only the learning rate is matched across arms and everything else is each optimizer's
  house configuration;
* whether an arm's selected learning rate sits at the edge of the screened grid, in
  which case its optimum is not bracketed;
* the pairing validation: within a seed, every arm must share the initial adapter
  SHA-256 and agree on the step-zero losses;
* the electrical state of every run, because the GPU is power limited to 60 W on AC
  and 35 W on battery. Runs whose state differs are reported as not comparable on
  time.

An interval that crosses zero is reported as crossing zero, not as a win.

    python benchmarks/nekaon_evidence/anima/aggregate.py --root tmp/nekaon-evidence-anima
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

try:  # Python 3.11+; ``tomli`` is the identical backport for 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - interpreter dependent
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[2]

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260918
TOLERANCE = 1.0e-6


_CAMPAIGN_MODULE = "nekaon_evidence_campaign"


def _load_campaign():
    existing = sys.modules.get(_CAMPAIGN_MODULE)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(_CAMPAIGN_MODULE, HERE / "campaign.py")
    if spec is None or spec.loader is None:  # pragma: no cover - packaging accident
        raise RuntimeError("cannot load campaign.py")
    module = importlib.util.module_from_spec(spec)
    # Registering first keeps `dataclasses` able to resolve the defining module on 3.10.
    sys.modules[_CAMPAIGN_MODULE] = module
    spec.loader.exec_module(module)
    return module


campaign = _load_campaign()

BASELINE = "adamw_fused"
CHALLENGER = "nekaon_fused"
SIBLING = "adakaon_fused"

CONFOUND_NOTE = (
    "Every arm runs its own house configuration and only the learning rate is tuned: "
    "the kaon arms use betas (0.5, 0.999), weight decay 0.1, cautious updates, "
    "gradient centralization, 4-bit momentum and stochastic rounding, while "
    "`adamw_fused` uses betas (0.9, 0.999), weight decay 0.01 and none of those. "
    "A difference between AdamW and a kaon arm is therefore NOT attributable to the "
    "algorithm alone; the table of `[optimizer]` blocks in this report is the exact "
    "confound."
)

BOUNDARY_NOTE = (
    "LR selected at the boundary of the grid for {arms}: the optimum is not bracketed "
    "there, so the reported result is a lower bound on that arm's performance — a "
    "learning rate outside the screened grid may well be better."
)

TELEMETRY_NOTE = (
    "The `nekaon_fused` arm (k = 1.5) pays MSAM's inert-lookahead telemetry that "
    "`adakaon_fused` and `adamw_fused` do not: `_warn_if_inert` stays armed for its "
    "first 200 climbs and samples every 10th, so a 200-step run here is instrumented "
    "end to end and performs roughly twenty device-to-host synchronizations. That cost "
    "is real for a user running this configuration today, but it is telemetry rather "
    "than optimizer arithmetic, and it counts against Nekaon in active seconds and "
    "ms/step."
)


# ----------------------------------------------------------------------- extraction


def _last(record: dict[str, Any], metric: str) -> float | None:
    points = ((record.get("run") or {}).get(metric) or {}).get("points")
    if not points:
        return None
    return float(max(points, key=lambda point: point["step"])["value"])


def _initial(record: dict[str, Any], metric: str) -> float | None:
    points = ((record.get("run") or {}).get(metric) or {}).get("points")
    if not points:
        return None
    point = next((item for item in points if item["step"] == 0), None)
    return None if point is None else float(point["value"])


def _bench(record: dict[str, Any]) -> tuple[float | None, float | None]:
    csvs = (record.get("run") or {}).get("bench_csv") or []
    points = [point for csv in csvs for point in csv.get("points") or []]
    if not points:
        return None, None
    last = max(points, key=lambda point: point["step"])
    peaks = [float(p["cuda_peak_gb"]) for p in points if p.get("cuda_peak_gb") is not None]
    active = last.get("active_train_seconds")
    return (float(active) if active is not None else None), (max(peaks) if peaks else None)


def _active_seconds(record: dict[str, Any]) -> float | None:
    return _bench(record)[0]


def _peak_gib(record: dict[str, Any]) -> float | None:
    return _bench(record)[1]


def _ms_per_step(record: dict[str, Any]) -> float | None:
    active = _active_seconds(record)
    steps = record.get("steps")
    if active is None or not steps:
        return None
    return active / float(steps) * 1000.0


METRICS: dict[str, dict[str, Any]] = {
    "final_val": {"label": "Final val eps-MSE", "get": lambda r: _last(r, "val"), "lower_is_better": True, "fmt": ".6f"},
    "raw_gap": {"label": "Raw gap (val - train_eval)", "get": lambda r: _last(r, "gap"), "lower_is_better": True, "fmt": "+.6f"},
    "active_seconds": {"label": "Active train seconds", "get": _active_seconds, "lower_is_better": True, "fmt": ".1f"},
    "ms_per_step": {"label": "ms / step", "get": _ms_per_step, "lower_is_better": True, "fmt": ".1f"},
    "peak_gib": {"label": "Peak allocator GiB", "get": _peak_gib, "lower_is_better": True, "fmt": ".3f"},
}


# --------------------------------------------------------------- declared confounds


def _parse_optimizer(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """The ``[optimizer]`` table of a run's own TOML, or a reason it is unavailable."""
    text = record.get("config")
    if not text:
        return None, "the run record carries no config TOML"
    if tomllib is None:  # pragma: no cover - interpreter dependent
        return None, "no TOML parser available (install tomli on Python 3.10)"
    try:
        data = tomllib.loads(text)
    except Exception as exc:  # noqa: BLE001 - any malformed TOML is reported, not raised
        return None, f"could not parse the config TOML ({exc})"
    optimizer = data.get("optimizer")
    if not isinstance(optimizer, dict):
        return None, "the config TOML has no [optimizer] table"
    return optimizer, None


def arm_configs(
    records: list[dict[str, Any]],
    *,
    arms: tuple[str, ...] = campaign.ARMS,
    selected: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Per arm, the exact ``[optimizer]`` block it ran, read from the run's own TOML.

    Only the learning rate is matched across arms; everything else differs. Reporting
    the blocks verbatim is what turns an undeclared confound into a declared one.
    """
    selected = selected or {}
    table: dict[str, Any] = {}
    for arm in arms:
        candidates = [record for record in records if record.get("arm") == arm]
        if not candidates:
            table[arm] = {"run_id": None, "optimizer": None, "reason": "no run for this arm"}
            continue
        lr = selected.get(arm)
        preferred = [
            record for record in candidates
            if lr is not None and math.isclose(float(record["lr"]), lr, rel_tol=1e-12, abs_tol=0.0)
        ]
        record = sorted(preferred or candidates, key=lambda item: str(item.get("run_id")))[0]
        optimizer, reason = _parse_optimizer(record)
        table[arm] = {"run_id": record.get("run_id"), "optimizer": optimizer, "reason": reason}
    return table


def lr_grid(records: list[dict[str, Any]], arm: str) -> list[float]:
    """The learning rates actually screened for ``arm``, read from the records."""
    return sorted({float(record["lr"]) for record in records
                   if record.get("phase") == "a" and record.get("arm") == arm})


def boundary_flags(
    records: list[dict[str, Any]],
    selected: dict[str, float],
    *,
    arms: tuple[str, ...] = campaign.ARMS,
) -> dict[str, Any]:
    """Per arm, whether the selected LR sits at an end of the screened grid.

    A selection at the edge means the optimum was never bracketed: the arm's result is a
    lower bound, not an estimate of its best achievable performance.
    """
    flags: dict[str, Any] = {}
    for arm in arms:
        grid = lr_grid(records, arm)
        lr = selected.get(arm)
        at_edge = None
        if lr is not None and grid:
            at_edge = bool(
                math.isclose(lr, grid[0], rel_tol=1e-12, abs_tol=0.0)
                or math.isclose(lr, grid[-1], rel_tol=1e-12, abs_tol=0.0)
            )
        flags[arm] = {"lr": lr, "grid": grid, "boundary": at_edge}
    return flags


def phase_a_status(
    records: list[dict[str, Any]],
    *,
    arms: tuple[str, ...] = campaign.ARMS,
    phase_a_lrs: tuple[float, ...] = campaign.PHASE_A_LRS,
) -> dict[str, Any]:
    """Whether phase A finished. Selecting an LR before it does is provisional.

    ``campaign.resolve_lrs`` refuses to resolve at all until every arm has run every
    candidate; the aggregator still reports, but says the selection is provisional.
    """
    per_arm = {
        arm: sum(1 for record in records if record.get("phase") == "a" and record.get("arm") == arm)
        for arm in arms
    }
    expected_per_arm = len(phase_a_lrs)
    return {
        "complete": all(count >= expected_per_arm for count in per_arm.values()),
        "runs": sum(per_arm.values()),
        "expected": len(arms) * expected_per_arm,
        "per_arm": per_arm,
        "expected_per_arm": expected_per_arm,
    }


# ------------------------------------------------------------------------ bootstrap


def bootstrap_ci(
    values: list[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict[str, Any] | None:
    """Percentile bootstrap of the mean. Deterministic for a given seed."""
    clean = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not clean:
        return None
    mean = statistics.fmean(clean)
    if len(clean) < 2:
        return {"mean": mean, "ci_low": None, "ci_high": None, "n": len(clean),
                "resamples": 0, "seed": seed, "crosses_zero": None}
    rng = random.Random(seed)
    size = len(clean)
    means = []
    for _ in range(resamples):
        means.append(sum(clean[rng.randrange(size)] for _ in range(size)) / size)
    means.sort()
    low = means[int(math.floor((alpha / 2) * (resamples - 1)))]
    high = means[int(math.ceil((1 - alpha / 2) * (resamples - 1)))]
    return {
        "mean": mean, "ci_low": low, "ci_high": high, "n": size,
        "resamples": resamples, "seed": seed, "crosses_zero": low <= 0.0 <= high,
    }


# ------------------------------------------------------------------------- assembly


def _by_arm_seed(records: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    table: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = (record["arm"], int(record["seed"]))
        if key in table:
            raise ValueError(f"duplicate run for arm {key[0]} seed {key[1]}")
        table[key] = record
    return table


def confirmation_set(records: list[dict[str, Any]], selected: dict[str, float]) -> list[dict[str, Any]]:
    """Records at each arm's selected LR — phase A's selected run included."""
    chosen = []
    for record in records:
        lr = selected.get(record["arm"])
        if lr is not None and math.isclose(float(record["lr"]), lr, rel_tol=1e-12, abs_tol=0.0):
            chosen.append(record)
    return chosen


def paired_seeds(records: list[dict[str, Any]], arms: tuple[str, ...]) -> list[int]:
    seeds = sorted({int(record["seed"]) for record in records})
    table = _by_arm_seed(records)
    return [seed for seed in seeds if all((arm, seed) in table for arm in arms)]


def validate_pairing(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Within each seed, every arm must share the adapter SHA and step-zero losses."""
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_seed.setdefault(int(record["seed"]), []).append(record)
    report: dict[str, Any] = {"tolerance": TOLERANCE, "seeds": {}, "ok": True}
    for seed, group in sorted(by_seed.items()):
        shas = [record.get("adapter_initial_sha256") for record in group]
        sha_ok = bool(shas[0]) and all(sha == shas[0] for sha in shas)
        losses: dict[str, Any] = {}
        losses_ok = True
        for metric in ("train_eval", "val"):
            values = [_initial(record, metric) for record in group]
            if any(value is None or not math.isfinite(value) for value in values):
                losses_ok = False
                losses[metric] = {"values": values, "ok": False, "reason": "missing step-zero point"}
                continue
            spread = max(values) - min(values)  # type: ignore[type-var]
            ok = spread <= TOLERANCE
            losses_ok = losses_ok and ok
            losses[metric] = {"value": values[0], "spread": spread, "ok": ok}
        report["seeds"][str(seed)] = {
            "arms": sorted(record["arm"] for record in group),
            "adapter_initial_sha256": shas[0],
            "adapter_sha_matches": sha_ok,
            "initial_losses": losses,
            "ok": sha_ok and losses_ok,
        }
        report["ok"] = report["ok"] and sha_ok and losses_ok
    return report


def power_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    signatures: dict[str, list[str]] = {}
    for record in records:
        signature = record.get("power_signature") or "unrecorded"
        signatures.setdefault(signature, []).append(record["run_id"])
    comparable = len(signatures) == 1
    return {
        "signatures": {key: sorted(value) for key, value in sorted(signatures.items())},
        "timings_comparable": comparable,
        "note": (
            "The GPU is power limited to 60 W on AC and 35 W on battery. Runs under "
            "different signatures are NOT comparable on active seconds or ms/step; "
            "quality metrics are unaffected."
        ),
    }


def summarize(
    records: list[dict[str, Any]],
    *,
    arms: tuple[str, ...] = campaign.ARMS,
    phase_a_seed: int = campaign.PHASE_A_SEED,
    phase_a_lrs: tuple[float, ...] = campaign.PHASE_A_LRS,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    phase_a = [record for record in records if record.get("phase") == "a"]
    selected = campaign.select_lrs(phase_a, arms=arms) if phase_a else {}
    phase_a_state = phase_a_status(records, arms=arms, phase_a_lrs=phase_a_lrs)
    boundary = boundary_flags(records, selected, arms=arms)

    sweep = []
    for record in sorted(phase_a, key=lambda r: (r["arm"], float(r["lr"]))):
        row = {"arm": record["arm"], "seed": int(record["seed"]), "lr": float(record["lr"])}
        row.update({name: spec["get"](record) for name, spec in METRICS.items()})
        row["selected"] = math.isclose(float(record["lr"]), selected.get(record["arm"], math.nan),
                                       rel_tol=1e-12, abs_tol=0.0)
        sweep.append(row)

    chosen = confirmation_set(records, selected)
    seeds = paired_seeds(chosen, arms)
    table = _by_arm_seed(chosen)
    held_out = [seed for seed in seeds if seed != phase_a_seed]

    def arm_block(seed_list: list[int]) -> dict[str, Any]:
        block: dict[str, Any] = {}
        for arm in arms:
            per_metric = {}
            for name, spec in METRICS.items():
                values = [spec["get"](table[(arm, seed)]) for seed in seed_list]
                per_metric[name] = {
                    "per_seed": dict(zip((str(s) for s in seed_list), values, strict=False)),
                    "bootstrap": bootstrap_ci([v for v in values if v is not None], resamples=resamples),
                }
            block[arm] = {
                "lr": selected.get(arm),
                "lr_grid": boundary[arm]["grid"],
                "boundary": boundary[arm]["boundary"],
                "seeds": seed_list,
                "metrics": per_metric,
            }
        return block

    def paired_block(left: str, right: str, seed_list: list[int]) -> dict[str, Any] | None:
        if left not in arms or right not in arms or not seed_list:
            return None
        out: dict[str, Any] = {"left": left, "right": right, "seeds": seed_list, "metrics": {}}
        for name, spec in METRICS.items():
            diffs = []
            for seed in seed_list:
                a = spec["get"](table[(left, seed)])
                b = spec["get"](table[(right, seed)])
                diffs.append(None if a is None or b is None else a - b)
            usable = [d for d in diffs if d is not None]
            lower_better = spec["lower_is_better"]
            wins = sum(1 for d in usable if (d < 0) == lower_better and d != 0)
            out["metrics"][name] = {
                "per_seed": dict(zip((str(s) for s in seed_list), diffs, strict=False)),
                "bootstrap": bootstrap_ci(usable, resamples=resamples),
                "left_wins": wins,
                "n": len(usable),
                "direction": "lower is better" if lower_better else "higher is better",
            }
        return out

    return {
        "schema": "nekaon-evidence-anima-aggregate-v1",
        "purpose": "Paired-seed descriptive comparison of Nekaon, Adakaon and AdamW on Anima/Pets.",
        "provenance": (records[0].get("provenance") if records else None),
        "lr_selection_rule": campaign.LR_SELECTION_RULE,
        "selected_lrs": selected,
        "phase_a": phase_a_state,
        "lr_boundary": boundary,
        "arm_optimizer_configs": arm_configs(records, arms=arms, selected=selected),
        "confound_note": CONFOUND_NOTE,
        "telemetry_note": TELEMETRY_NOTE,
        "lr_sweep": sweep,
        "seeds": {"paired": seeds, "held_out": held_out, "phase_a_seed": phase_a_seed},
        "arms": arm_block(seeds),
        "arms_held_out": arm_block(held_out) if held_out else None,
        "paired": {
            f"{CHALLENGER}_minus_{BASELINE}": paired_block(CHALLENGER, BASELINE, seeds),
            f"{CHALLENGER}_minus_{SIBLING}": paired_block(CHALLENGER, SIBLING, seeds),
        },
        "paired_held_out": {
            f"{CHALLENGER}_minus_{BASELINE}": paired_block(CHALLENGER, BASELINE, held_out),
            f"{CHALLENGER}_minus_{SIBLING}": paired_block(CHALLENGER, SIBLING, held_out),
        } if held_out else None,
        "validation": validate_pairing(chosen),
        "power": power_report(records),
        "bootstrap": {"resamples": resamples, "seed": BOOTSTRAP_SEED, "method": "percentile, 95%"},
        "run_ids": sorted(record["run_id"] for record in records),
    }


# -------------------------------------------------------------------------- markdown


def _fmt(value: float | None, spec: str) -> str:
    return "—" if value is None else format(value, spec)


def _ci(entry: dict[str, Any] | None, spec: str) -> str:
    if not entry:
        return "—"
    if entry.get("ci_low") is None:
        return f"{format(entry['mean'], spec)} (n={entry['n']}, no interval)"
    return f"{format(entry['mean'], spec)} [{format(entry['ci_low'], spec)}, {format(entry['ci_high'], spec)}]"


def _toml_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, "g")
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return f"`{value}`" if isinstance(value, str) else str(value)


def _boundary_arms(result: dict[str, Any]) -> list[str]:
    return [arm for arm, entry in (result.get("lr_boundary") or {}).items() if entry.get("boundary")]


def _provisional_line(result: dict[str, Any]) -> str | None:
    phase_a = result.get("phase_a") or {}
    if phase_a.get("complete", True):
        return None
    return (
        f"**Provisional selection (phase A incomplete: {phase_a.get('runs')}/"
        f"{phase_a.get('expected')} runs).** The learning rates in this report were picked "
        "from a partial screen, so the paired set is not the definitive comparison — the "
        "campaign driver itself refuses to resolve learning rates until phase A finishes."
    )


def _config_table(result: dict[str, Any]) -> list[str]:
    configs = result.get("arm_optimizer_configs") or {}
    arms = list(configs)
    lines = [
        "## Arm configurations as run",
        "",
        CONFOUND_NOTE,
        "",
    ]
    if not arms:
        return lines + ["No run configurations were available to parse.", ""]
    keys: list[str] = []
    for entry in configs.values():
        for key in entry.get("optimizer") or {}:
            if key not in keys:
                keys.append(key)
    missing = [f"`{arm}` ({entry['reason']})" for arm, entry in configs.items() if entry.get("reason")]
    if not keys:
        return lines + ["The `[optimizer]` blocks could not be read: " + ", ".join(missing) + ".", ""]
    lines += [
        "Read verbatim from each run's own TOML"
        + (f" — unavailable for {', '.join(missing)}." if missing else "."),
        "",
        "| `[optimizer]` key | " + " | ".join(arms) + " |",
        "| --- | " + " | ".join("---" for _ in arms) + " |",
    ]
    for key in keys:
        cells = [_toml_value((configs[arm].get("optimizer") or {}).get(key)) for arm in arms]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _arm_means_table(block: dict[str, Any]) -> list[str]:
    lines = [
        "| Arm | LR | " + " | ".join(spec["label"] for spec in METRICS.values()) + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in METRICS) + " |",
    ]
    for arm, entry in block.items():
        cells = [_ci(entry["metrics"][name]["bootstrap"], spec["fmt"]) for name, spec in METRICS.items()]
        lr = entry["lr"]
        label = "—" if lr is None else format(lr, "g")
        if entry.get("boundary"):
            label += " (boundary)"
        lines.append(f"| {arm} | {label} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _paired_tables(blocks: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for block in blocks.values():
        if not block:
            continue
        lines += [
            f"#### {block['left']} − {block['right']}",
            "",
            "A negative difference favours " + block["left"] + " on every metric here "
            "(all are lower-is-better).",
            "",
            "| Metric | Mean difference [95% CI] | Seeds won by "
            + block["left"] + " | Interval crosses zero |",
            "| --- | ---: | ---: | :---: |",
        ]
        for name, spec in METRICS.items():
            entry = block["metrics"][name]
            boot = entry["bootstrap"]
            crosses = "—" if not boot or boot.get("crosses_zero") is None else ("yes" if boot["crosses_zero"] else "no")
            lines.append(
                f"| {spec['label']} | {_ci(boot, spec['fmt'])} | {entry['left_wins']}/{entry['n']} | {crosses} |"
            )
        lines.append("")
    return lines


def to_markdown(result: dict[str, Any]) -> str:
    provenance = result.get("provenance") or {}
    boundary_arms = _boundary_arms(result)
    provisional = _provisional_line(result)
    lines = [
        "# Nekaon vs Adakaon vs AdamW on Anima / Pets",
        "",
        "Rank-16 LoRA on Cosmos Predict2 (Anima), bfloat16, Oxford-IIIT Pets subset at "
        "256 px, constant LR, 200 steps, evaluation before the first step and every 100 "
        "steps on eight fixed images per split at nine noise quantiles. Previews off. "
        "Runs are paired within a seed: the adapter initialization is seeded before the "
        "trainer starts and the initial SHA-256 is required to match across arms.",
        "",
        f"kaon {provenance.get('kaon_version', '?')} at commit {provenance.get('commit', '?')}.",
        "",
        "## Read this before the numbers",
        "",
        f"- **Confound.** {CONFOUND_NOTE}",
    ]
    if boundary_arms:
        lines.append("- **Boundary.** " + BOUNDARY_NOTE.format(
            arms=", ".join(f"`{arm}`" for arm in boundary_arms)))
    else:
        lines.append(
            "- **Boundary.** No arm selected a learning rate at an end of its screened "
            "grid, so every selected LR is bracketed on both sides."
        )
    if provisional:
        lines.append(f"- {provisional}")
    lines += [
        f"- **Telemetry.** {TELEMETRY_NOTE}",
        "",
        f"LR selection rule (fixed before the data was seen): {result['lr_selection_rule']}.",
        "",
        "Selected learning rates: "
        + ", ".join(f"`{arm}` {lr:g}" for arm, lr in sorted(result["selected_lrs"].items()))
        + ".",
        "",
    ]

    lines += _config_table(result)

    lines += [
        "## Phase A — learning-rate screen (seed "
        f"{result['seeds']['phase_a_seed']})",
        "",
    ]
    if provisional:
        lines += [provisional, ""]
    lines += [
        "| Arm | LR | Selected | Final val | Raw gap | Active s | ms/step | Peak GiB |",
        "| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["lr_sweep"]:
        lines.append(
            f"| {row['arm']} | {row['lr']:g} | {'yes' if row['selected'] else ''} | "
            f"{_fmt(row['final_val'], '.6f')} | {_fmt(row['raw_gap'], '+.6f')} | "
            f"{_fmt(row['active_seconds'], '.1f')} | {_fmt(row['ms_per_step'], '.1f')} | "
            f"{_fmt(row['peak_gib'], '.3f')} |"
        )
    lines.append("")
    if boundary_arms:
        lines += [BOUNDARY_NOTE.format(arms=", ".join(f"`{arm}`" for arm in boundary_arms)), ""]

    seeds = result["seeds"]["paired"]
    held_seeds = result["seeds"]["held_out"]
    held_arms = result.get("arms_held_out")
    lines += [
        "## Per-arm means",
        "",
        "Mean with a 95% percentile bootstrap interval "
        f"({result['bootstrap']['resamples']} resamples, seed {result['bootstrap']['seed']}).",
        "",
        f"Seed {result['seeds']['phase_a_seed']} chose the learning rates, so it is shown "
        "second: the held-out seeds carry no selection bias, while the full paired set is "
        "larger but optimistic by construction.",
        "",
    ]
    if held_arms:
        lines += [
            f"### Held-out seeds, no selection bias ({', '.join(str(s) for s in held_seeds)})",
            "",
        ] + _arm_means_table(held_arms)
    lines += [
        f"### All {len(seeds)} paired seeds, includes the LR-selection seed "
        f"({', '.join(str(s) for s in seeds)})",
        "",
    ] + _arm_means_table(result["arms"])

    held_paired = result.get("paired_held_out")
    lines += [
        "## Paired differences by seed",
        "",
        "Held-out first, then the full paired set, over the same metrics.",
        "",
    ]
    if held_paired:
        lines += [
            f"### Held-out seeds, no selection bias ({', '.join(str(s) for s in held_seeds)})",
            "",
        ] + _paired_tables(held_paired)
    lines += [
        f"### All {len(seeds)} paired seeds, includes the LR-selection seed "
        f"({', '.join(str(s) for s in seeds)})",
        "",
    ] + _paired_tables(result["paired"])

    validation = result["validation"]
    lines += [
        "## Pairing validation",
        "",
        f"All seeds pass: **{'yes' if validation['ok'] else 'no'}**. Within each seed the arms "
        f"must share the initial adapter SHA-256 and agree on the step-zero losses to "
        f"{validation['tolerance']:g}.",
        "",
        "| Seed | Arms | Adapter SHA matches | Initial train_eval | Initial val | OK |",
        "| ---: | --- | :---: | ---: | ---: | :---: |",
    ]
    for seed, entry in validation["seeds"].items():
        train = entry["initial_losses"].get("train_eval", {})
        val = entry["initial_losses"].get("val", {})
        lines.append(
            f"| {seed} | {', '.join(entry['arms'])} | {'yes' if entry['adapter_sha_matches'] else 'NO'} | "
            f"{_fmt(train.get('value'), '.6f')} | {_fmt(val.get('value'), '.6f')} | "
            f"{'yes' if entry['ok'] else 'NO'} |"
        )

    power = result["power"]
    lines += [
        "",
        "## Timings and electrical state",
        "",
        power["note"],
        "",
        TELEMETRY_NOTE,
        "",
        f"Timings comparable across all runs: **{'yes' if power['timings_comparable'] else 'no'}**.",
        "",
        "| Power signature | Runs |",
        "| --- | ---: |",
    ]
    for signature, runs in power["signatures"].items():
        lines.append(f"| `{signature}` | {len(runs)} |")

    lines += [
        "",
        "## What this does not show",
        "",
        f"- **The arms are not matched.** {CONFOUND_NOTE}",
        f"- {len(seeds)} paired seeds is a small sample. The bootstrap intervals are wide by "
        "construction and an interval that crosses zero means the campaign did not separate "
        "the arms on that metric, not that they are equal.",
    ]
    if boundary_arms:
        lines.append("- " + BOUNDARY_NOTE.format(arms=", ".join(f"`{arm}`" for arm in boundary_arms)))
    if provisional:
        lines.append(f"- {provisional}")
    lines += [
        "- Validation loss here is epsilon-MSE on eight held-out images at nine fixed noise "
        "quantiles. It is not FID, not a perceptual score, and not a measure of image quality.",
        "- The raw gap is descriptive. Eight images per split cannot estimate population "
        "generalization, and the split does not establish exclusion from Anima's pretraining data.",
        "- Timings come from a single power-limited laptop GPU, one run per configuration, "
        "with thermal drift only partly mitigated by interleaving the arms. Peak allocator "
        "memory includes allocations from earlier in the same process. The Nekaon arm also "
        "carries the inert-lookahead telemetry described above.",
        "- Optimizer state bytes are not measured directly: the trainer reports the allocator "
        "peak and the optimizer parameter count, not a per-optimizer state accounting.",
        "- 200 steps of a rank-16 LoRA on 96 images is a screening protocol. Nothing here "
        "transfers automatically to full fine-tuning, other models, or other datasets.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=WORKTREE / "tmp" / "nekaon-evidence-anima")
    parser.add_argument("--json-out", type=Path, default=HERE / "results.json")
    parser.add_argument("--md-out", type=Path, default=HERE / "RESULTS.md")
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    args = parser.parse_args(argv)

    records = campaign.load_records(args.root)
    if not records:
        raise SystemExit(f"no *_results.json under {campaign.results_dir(args.root)}")
    result = summarize(records, resamples=args.resamples)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.md_out.write_text(to_markdown(result), encoding="utf-8")
    print(f"Wrote {args.md_out} and {args.json_out} from {len(records)} run(s)")
    if not result["validation"]["ok"]:
        print("WARNING: pairing validation failed; see the report")
    if not result["power"]["timings_comparable"]:
        print("WARNING: runs span more than one electrical state; timings are not comparable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
