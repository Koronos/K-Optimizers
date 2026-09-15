"""Negative checks for the Anima same-seed report pairing tool."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

PAIRING = Path(__file__).parents[1] / "benchmarks" / "anima" / "paired_confirmation.py"
spec = importlib.util.spec_from_file_location("paired_confirmation", PAIRING)
assert spec and spec.loader
pairing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pairing)

REPORTS = Path(__file__).parents[1] / "benchmarks" / "anima"


def _report_paths(tmp_path: Path) -> list[Path]:
    names = ("confirmation43_results.json", "confirmation44_results.json", "adam43_results.json", "adam44_results.json")
    paths = []
    for name in names:
        destination = tmp_path / name
        destination.write_text((REPORTS / name).read_text(encoding="utf-8"), encoding="utf-8")
        paths.append(destination)
    return paths


def _mutate(path: Path, mutation) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    mutation(report)
    path.write_text(json.dumps(report), encoding="utf-8")


def test_missing_step_zero_is_rejected(tmp_path: Path) -> None:
    paths = _report_paths(tmp_path)

    def remove_step_zero(report: dict) -> None:
        report["runs"][0]["train_eval"]["points"] = [
            point for point in report["runs"][0]["train_eval"]["points"] if point["step"] != 0
        ]

    _mutate(paths[0], remove_step_zero)
    with pytest.raises(ValueError, match="no step 0"):
        pairing.combine(paths)


def test_nonfinite_initial_loss_is_rejected(tmp_path: Path) -> None:
    paths = _report_paths(tmp_path)

    def add_nan(report: dict) -> None:
        report["runs"][0]["val"]["points"][0]["value"] = float("nan")

    _mutate(paths[0], add_nan)
    with pytest.raises(ValueError, match="not finite"):
        pairing.combine(paths)


def test_duplicate_arm_within_seed_is_rejected(tmp_path: Path) -> None:
    paths = _report_paths(tmp_path)

    def duplicate_arm(report: dict) -> None:
        report["runs"].append(copy.deepcopy(report["runs"][0]))

    _mutate(paths[0], duplicate_arm)
    with pytest.raises(ValueError, match="duplicate arm"):
        pairing.combine(paths)
