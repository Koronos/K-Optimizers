import importlib.util
import sys
from pathlib import Path

import pytest


def _load_battery():
    path = Path(__file__).parents[1] / "benchmarks/nekaon_relative_battery.py"
    spec = importlib.util.spec_from_file_location("nekaon_relative_battery", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


B = _load_battery()


def _row(arm, loss, gap, lr=1e-3):
    return {
        "arm": arm,
        "regime": "constant",
        "lr": lr,
        "initial_train": 1.5,
        "train_loss": loss - gap,
        "heldout_loss": loss,
        "gap": gap,
    }


def test_small_gap_from_underfitting_does_not_dominate():
    underfit = _row("nekaon", loss=1.20, gap=0.001)
    useful = _row("adakaon", loss=1.00, gap=0.010)
    assert not B.dominates(underfit, useful)


def test_gap_improvement_at_matched_loss_dominates():
    flatter = _row("nekaon", loss=1.005, gap=0.005)
    baseline = _row("adakaon", loss=1.00, gap=0.010)
    assert B.dominates(flatter, baseline)


def test_gap_improvement_by_undertraining_does_not_dominate():
    undertrained = _row("nekaon", loss=1.005, gap=0.005)
    undertrained["train_loss"] = 1.45
    baseline = _row("adakaon", loss=1.00, gap=0.010)
    assert not B.dominates(undertrained, baseline)


def test_decision_marks_adakaon_supersession_when_all_nekaon_points_are_dominated():
    rows = [
        _row("adakaon", 0.90, 0.008, 5e-4),
        _row("adakaon", 0.85, 0.012, 1e-3),
        _row("nekaon", 0.92, 0.014, 5e-4),
        _row("nekaon", 0.88, 0.016, 1e-3),
    ]
    assert B.decision(rows)["verdict"] == "adakaon_supersedes_nekaon"


def test_replay_rejects_invalid_or_short_trace():
    with pytest.raises(ValueError):
        B.lr_multipliers("replay", 3, [1.0, 1.0])
    with pytest.raises(ValueError):
        B.lr_multipliers("replay", 2, [1.0, float("nan")])
