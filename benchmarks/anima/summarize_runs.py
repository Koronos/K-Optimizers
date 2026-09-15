"""Summarize TensorBoard runs without assuming a logger schema.

The script only gives semantic names to scalar tags that are actually present in an
event file. All raw scalar series are retained under ``scalars``; missing metrics are
reported as ``null`` rather than being inferred from a run name or a fixed tag list.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _points(acc: EventAccumulator, tag: str) -> list[dict[str, float | int]]:
    if any(not math.isfinite(float(event.value)) for event in acc.Scalars(tag)):
        raise ValueError(f"Nonfinite metric in {tag}; run cannot be summarized as successful")
    return [
        {"step": int(event.step), "value": float(event.value), "wall_time": float(event.wall_time)}
        for event in acc.Scalars(tag)
    ]


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    values = sorted(values)
    def q(fraction: float) -> float:
        position = (len(values) - 1) * fraction
        low = math.floor(position)
        high = math.ceil(position)
        if low == high:
            return values[low]
        return values[low] + (values[high] - values[low]) * (position - low)
    return {"p10": q(.10), "p50": q(.50), "p90": q(.90)}


def _pick(tags: list[str], groups: tuple[tuple[str, ...], ...], exclude: tuple[str, ...] = ()) -> str | None:
    candidates = []
    for tag in tags:
        lower = tag.lower().replace("\\", "/")
        if any(word in lower for word in exclude):
            continue
        score = sum(1 for group in groups if any(word in lower for word in group))
        if score == len(groups):
            candidates.append((score, len(tag), tag))
    return min(candidates)[2] if candidates else None


def _aligned_gap(train: list[dict[str, Any]] | None, val: list[dict[str, Any]] | None):
    if not train or not val:
        return None
    by_step = {point["step"]: point for point in train}
    return [
        {"step": point["step"], "value": point["value"] - by_step[point["step"]]["value"],
         "wall_time": point["wall_time"]}
        for point in val if point["step"] in by_step
    ] or None


def summarize_run(run_dir: Path) -> dict[str, Any]:
    event_files = sorted(run_dir.rglob("events.out.tfevents*"))
    if len({path.parent for path in event_files}) > 1:
        raise ValueError("Multiple event directories: pass one actual run, not a parent of runs")
    scalars: dict[str, list[dict[str, float | int]]] = {}
    for event_file in event_files:
        acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            # A run may rotate event files; preserve the observed series and sort later.
            scalars.setdefault(tag, []).extend(_points(acc, tag))
    for points in scalars.values():
        points.sort(key=lambda point: (point["step"], point["wall_time"]))
        if len({point["step"] for point in points}) != len(points):
            raise ValueError("Repeated metric steps: resolve restart/rotation overlap explicitly")

    tags = sorted(scalars)
    # Only these named evaluation datasets define our gap. A fuzzy match can
    # silently select train/loss (minibatch noise) or one timestep quantile.
    train_tag = "train_eval/loss" if "train_eval/loss" in scalars else None
    val_tag = "val/loss" if "val/loss" in scalars else None
    train = scalars.get(train_tag) if train_tag else None
    val = scalars.get(val_tag) if val_tag else None
    gap = _aligned_gap(train, val)
    initial = next((point["value"] for point in gap or [] if point["step"] == 0), None)
    excess_gap = ([dict(point, value=point["value"] - initial) for point in gap]
                  if gap and initial is not None else None)

    def matching(groups):
        return {tag: scalars[tag] for tag in tags
                if all(any(word in tag.lower() for word in group) for group in groups)} or None

    evaltime = matching((("eval",), ("time", "seconds", "duration")))
    timings = matching((("time", "step", "throughput", "iter"),))
    memory = matching((("mem", "ram", "vram", "alloc", "peak"),))
    # Rengu writes training-only durations and allocator metrics to bench_steps.csv,
    # separately from TensorBoard. Do not merge several runs into one curve.
    csv_files = sorted(run_dir.rglob("bench_steps.csv"))
    bench = []
    for csv_path in csv_files:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        active = 0.0
        points = []
        for row in rows:
            if not all(math.isfinite(float(row[key])) for key in ("iter_sec", "cuda_peak_gb")):
                raise ValueError(f"Nonfinite timing/memory in {csv_path}")
            if float(row["iter_sec"]) < 0 or float(row["cuda_peak_gb"]) < 0:
                raise ValueError(f"Negative timing/memory in {csv_path}")
            active += float(row["iter_sec"])
            points.append({"step": int(row["step"]), "active_train_seconds": active,
                           "cuda_peak_gb": float(row["cuda_peak_gb"])})
        bench.append({"file": str(csv_path), "points": points,
                      "note": "Training batch duration excludes evaluation and previews; peak includes prior allocations."})
    return {
        "run_dir": str(run_dir),
        "event_files": [str(path) for path in event_files],
        "tags": tags,
        "train_eval": {"tag": train_tag, "points": train},
        "val": {"tag": val_tag, "points": val},
        "gap": {"definition": "val - train_eval", "points": gap},
        "excess_gap": {
            "definition": "(val_t - train_eval_t) - (val_0 - train_eval_0)",
            "points": excess_gap,
            "note": "Descriptive baseline adjustment, not an unbiased generalization bound.",
        },
        "checkpoint_value_quantiles": {
            "train_eval": _quantiles([point["value"] for point in train] if train else []),
            "val": _quantiles([point["value"] for point in val] if val else []),
            "gap": _quantiles([point["value"] for point in gap] if gap else []),
        },
        "evaltime": evaltime,
        "timings": timings,
        "memory": memory,
        "bench_csv": bench,
        "scalars": scalars,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [summarize_run(path) for path in args.run_dirs]
    args.output.write_text(json.dumps({"runs": runs}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
