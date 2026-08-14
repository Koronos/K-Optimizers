import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch


def _load_battery():
    path = Path(__file__).parents[1] / "benchmarks/alignment_telemetry_battery.py"
    spec = importlib.util.spec_from_file_location("alignment_telemetry_battery", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


B = _load_battery()


def test_safe_cosine_rejects_unobservable_values_and_clamps_roundoff():
    assert B.safe_cosine(1.0, 0.0, 1.0) is None
    assert B.safe_cosine(float("nan"), 1.0, 1.0) is None
    assert B.safe_cosine(1.0 + 1e-12, 1.0, 1.0) == 1.0
    assert B.safe_cosine(-1.0 - 1e-12, 1.0, 1.0) == -1.0


def test_alignment_summary_ignores_missing_and_nonfinite_observations():
    summary = B.summarize_alignments([None, float("nan"), 0.5, -0.5, -0.1])
    assert summary["observations"] == 3
    assert summary["mean"] == pytest.approx(-1.0 / 30.0)
    assert summary["minimum"] == -0.5
    assert summary["negative_fraction"] == pytest.approx(2.0 / 3.0)
    assert summary["severe_negative_fraction"] == pytest.approx(1.0 / 3.0)


def test_correlations_have_expected_direction_and_average_tie_ranks():
    rows = []
    for alignment, loss, damage in ((0.8, 0.8, 0.0), (0.2, 1.0, 0.1), (-0.6, 1.4, 0.5)):
        rows.append(
            {
                "alignment": {
                    "mean": alignment,
                    "minimum": alignment,
                    "negative_fraction": float(alignment < 0.0),
                    "severe_negative_fraction": float(alignment < -0.25),
                },
                "heldout_loss": loss,
                "damage": damage,
            }
        )
    summary = B.correlation_summary(rows)["correlations"]
    assert summary["mean_vs_heldout_loss"]["pearson"] < -0.99
    assert summary["mean_vs_damage"]["spearman"] == pytest.approx(-1.0)
    assert B._ranks([2.0, 1.0, 2.0]) == [1.5, 0.0, 1.5]


def test_correlations_are_json_safe_when_signal_is_constant_or_missing():
    rows = [
        {
            "alignment": {
                "mean": 0.5,
                "minimum": None,
                "negative_fraction": 0.0,
                "severe_negative_fraction": 0.0,
            },
            "heldout_loss": loss,
            "damage": 0.0,
        }
        for loss in (1.0, 2.0)
    ]
    result = B.correlation_summary(rows)
    assert result["correlations"]["mean_vs_heldout_loss"]["pearson"] is None
    assert result["correlations"]["minimum_vs_damage"]["n"] == 0
    assert math.isfinite(result["arms"])


def test_anchor_signal_has_live_displacement_sign_and_is_unobservable_at_anchor():
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    anchor = param.detach().clone()
    param.grad = torch.tensor([2.0, 1.0])
    assert B._anchor_signal([param], [anchor])["anchor_displacement_cosine"] is None
    param.data.add_(torch.tensor([-0.5, -0.5]))
    signal = B._anchor_signal([param], [anchor])
    assert signal["anchor_displacement_dot"] == pytest.approx(-1.5)
    assert signal["anchor_displacement_cosine"] < 0.0
