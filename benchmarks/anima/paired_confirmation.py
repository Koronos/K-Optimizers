"""Pair same-seed Anima reports and emit a descriptive comparison.

The comparison is deliberately within seed only.  It checks that paired runs
used the same model, dataset, and protocol, then reports final validation
metrics and benchmark measurements without turning them into a universal
optimizer ranking.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

TOLERANCE = 1.0e-6


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or not isinstance(value.get("protocol"), dict):
        raise ValueError(f"{path}: expected a report with a protocol manifest")
    return value


def _seed(report: dict[str, Any], path: Path) -> int:
    try:
        return int(report["protocol"]["protocol"]["train_seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: missing protocol.protocol.train_seed") from exc


def _initial(run: dict[str, Any], metric: str, path: Path) -> float:
    points = run.get(metric, {}).get("points")
    if not points:
        raise ValueError(f"{path}: {run.get('arm')}: no {metric} evaluation points")
    point = next((item for item in points if item.get("step") == 0), None)
    if point is None:
        raise ValueError(f"{path}: {run.get('arm')}: {metric} has no step 0 point")
    try:
        value = float(point["value"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {run.get('arm')}: invalid initial {metric}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{path}: {run.get('arm')}: initial {metric} is not finite")
    return value


def _last(run: dict[str, Any], metric: str) -> tuple[int, float] | None:
    points = run.get(metric, {}).get("points")
    if not points:
        return None
    point = max(points, key=lambda item: item.get("step", -1))
    return int(point["step"]), float(point["value"])


def _bench(run: dict[str, Any]) -> tuple[float | None, float | None]:
    points = [point for csv in run.get("bench_csv") or [] for point in csv.get("points") or []]
    if not points:
        return None, None
    last = max(points, key=lambda item: item.get("step", -1))
    active = last.get("active_train_seconds")
    peaks = [float(point["cuda_peak_gb"]) for point in points if point.get("cuda_peak_gb") is not None]
    return (float(active) if active is not None else None), (max(peaks) if peaks else None)


def _check_close(values: list[float], label: str, seed: int) -> None:
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"seed {seed}: initial {label} contains a non-finite value")
    if values and max(values) - min(values) > TOLERANCE:
        raise ValueError(f"seed {seed}: initial {label} differs by more than {TOLERANCE:g}")


def combine(paths: list[Path]) -> dict[str, Any]:
    if len(paths) != 4:
        raise ValueError("exactly four report files are required")
    reports = []
    for path in paths:
        report = _load(path)
        reports.append((path, report, _seed(report, path)))
    by_seed: dict[int, list[tuple[Path, dict[str, Any]]]] = {}
    for path, report, seed in reports:
        by_seed.setdefault(seed, []).append((path, report))
    if len(by_seed) != 2:
        raise ValueError(f"expected two seeds, got {sorted(by_seed)}")

    rows: list[dict[str, Any]] = []
    validation: dict[str, Any] = {"tolerance": TOLERANCE, "seeds": {}}
    for seed, seed_reports in sorted(by_seed.items()):
        manifests = [
            (report["protocol"].get("model"), report["protocol"].get("dataset"), report["protocol"].get("protocol"))
            for _, report in seed_reports
        ]
        if any(manifest != manifests[0] for manifest in manifests[1:]):
            raise ValueError(f"seed {seed}: model, dataset, or protocol differs between paired reports")
        seen: dict[str, Path] = {}
        runs: list[tuple[Path, dict[str, Any]]] = []
        for path, report in seed_reports:
            for run in report.get("runs", []):
                arm = run.get("arm")
                if not isinstance(arm, str):
                    raise ValueError(f"{path}: run has no arm")
                if arm in seen:
                    raise ValueError(f"seed {seed}: duplicate arm {arm} in {seen[arm]} and {path}")
                seen[arm] = path
                runs.append((path, run))
        if not runs:
            raise ValueError(f"seed {seed}: no runs")
        shas = [run.get("adapter_initial_sha256") for _, run in runs]
        if any(not sha or sha != shas[0] for sha in shas):
            raise ValueError(f"seed {seed}: initial adapter SHA differs between arms")
        train_initial = [_initial(run, "train_eval", path) for path, run in runs]
        val_initial = [_initial(run, "val", path) for path, run in runs]
        _check_close(train_initial, "train_eval loss", seed)
        _check_close(val_initial, "val loss", seed)
        validation["seeds"][str(seed)] = {
            "source_files": [path.name for path, _ in seed_reports],
            "model_dataset_protocol_equal": True,
            "initial_sha256": shas[0],
            "initial_train_eval_loss": train_initial[0],
            "initial_val_loss": val_initial[0],
            "initial_losses_within_tolerance": True,
            "arms_unique": True,
        }
        for path, run in sorted(runs, key=lambda item: item[1]["arm"]):
            final_val = _last(run, "val")
            final_gap = _last(run, "gap")
            active, peak = _bench(run)
            rows.append(
                {
                    "seed": seed,
                    "arm": run["arm"],
                    "source_file": path.name,
                    "final": {
                        "step": final_val[0] if final_val else None,
                        "val": final_val[1] if final_val else None,
                        "rawgap": final_gap[1] if final_gap else None,
                        "active_seconds": active,
                        "peak_gib": peak,
                    },
                }
            )
    return {
        "schema": "anima-paired-confirmation-v1",
        "purpose": "Descriptive same-seed pairing; not a universal optimizer ranking.",
        "inputs": [path.name for path in paths],
        "validation": validation,
        "results": rows,
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Anima paired confirmation",
        "",
        "Descriptive same-seed comparison; this is not a universal optimizer ranking.",
        "",
        "Inputs: " + ", ".join(result["inputs"]),
        "",
        "| Seed | Arm | Source | Final step | Val | Raw gap | Active seconds | Peak GiB |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["results"]:
        final = row["final"]
        values = [final[key] for key in ("val", "rawgap", "active_seconds", "peak_gib")]
        rendered = ["" if value is None else f"{value:.6g}" for value in values]
        lines.append(
            f"| {row['seed']} | {row['arm']} | {row['source_file']} | {final['step'] or ''} | "
            + " | ".join(rendered)
            + " |"
        )
    lines.extend(["", "Validation: model, dataset, protocol, initial adapter SHA, and initial losses were checked within each seed.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs=4, type=Path, metavar="REPORT.json")
    parser.add_argument("--json-out", type=Path, default=Path("paired_confirmation.json"))
    parser.add_argument("--md-out", type=Path, default=Path("paired_confirmation.md"))
    args = parser.parse_args()
    result = combine(args.reports)
    args.json_out.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    args.md_out.write_text(_markdown(result), encoding="utf-8")
    print(f"Wrote {args.json_out} and {args.md_out} for seeds {', '.join(str(seed) for seed in result['validation']['seeds'])}")


if __name__ == "__main__":
    main()
