import importlib.util
import sys
from pathlib import Path

import pytest


def _load_battery():
    path = Path(__file__).parents[1] / "benchmarks/mechanic_addon_battery.py"
    spec = importlib.util.spec_from_file_location("mechanic_addon_battery", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


B = _load_battery()


def test_fixed_lr_arms_are_centered_on_oracle():
    assert B.fixed_lr("fixed_low", 1e-3, 10.0) == pytest.approx(1e-4)
    assert B.fixed_lr("fixed_oracle", 1e-3, 10.0) == pytest.approx(1e-3)
    assert B.fixed_lr("fixed_high", 1e-3, 10.0) == pytest.approx(1e-2)
    assert B.fixed_lr("mechanic", 1e-3, 10.0) is None
    with pytest.raises(ValueError):
        B.fixed_lr("unknown", 1e-3, 10.0)


def test_transient_damage_catches_rebound_after_progress():
    assert B.transient_damage([1.0, 0.8, 1.2, 0.7]) == pytest.approx(0.5)
    assert B.transient_damage([1.0, 0.9, 0.8]) == 0.0


def test_time_to_quality_uses_paired_oracle_final_loss():
    def row(arm, losses):
        return {
            "arm": arm,
            "seed": 17,
            "beta1": 0.9,
            "clipping": "none",
            "heldout_loss": losses[-1],
            "trajectory": [
                {"step": step, "heldout_loss": loss}
                for step, loss in zip((0, 10, 20), losses, strict=True)
            ],
        }

    rows = B.annotate_time_to_quality(
        [row("fixed_oracle", [1.5, 1.1, 1.0]), row("mechanic_guard", [1.5, 1.0, 0.9])]
    )
    assert rows[0]["quality_target"] == pytest.approx(1.02)
    assert rows[0]["time_to_quality"] == 20
    assert rows[1]["time_to_quality"] == 10


def test_mechanic_stats_sanitizes_nonfinite_scalars():
    assert B.mechanic_stats({"scale": 0.25, "h": float("inf"), "guarded": True}) == {
        "scale": 0.25,
        "h": None,
        "guarded": True,
    }
