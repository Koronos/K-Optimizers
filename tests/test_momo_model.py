"""Algebraic tests for the experimental MoMo scalar recurrence."""

from __future__ import annotations

import copy
import math

import pytest

from kaon._momo_model import MomoModel


def _reference_step(
    state: dict[str, float],
    *,
    beta: float,
    cap: float,
    loss: float,
    gx: float,
    dx: float,
    denominator: float,
) -> float:
    state["step"] += 1
    state["loss"] = beta * state["loss"] + (1.0 - beta) * loss
    state["gamma"] = beta * state["gamma"] + (1.0 - beta) * gx
    rho = 1.0 - beta ** int(state["step"])
    model_cap = state["loss"] + dx - state["gamma"]
    if model_cap < rho * state["lb"]:
        state["lb"] = max(0.0, model_cap / (2.0 * rho))
    numerator = state["loss"] - rho * state["lb"] + dx - state["gamma"]
    raw = max(numerator, 0.0) / denominator if denominator > 0.0 else 0.0
    tau = min(cap / rho, raw)
    h = state["loss"] + dx - state["gamma"]
    state["lb"] = max(0.0, (h - 0.5 * tau * denominator) / rho)
    return tau


def test_matches_official_scalar_recurrence_without_weight_decay() -> None:
    model = MomoModel(beta=0.9, cap=1e-2)
    reference = {"step": 0.0, "loss": 0.0, "gamma": 0.0, "lb": 0.0}
    samples = [
        (3.0, 0.4, 0.2, 4.0),
        (2.2, 0.1, 0.3, 2.0),
        (1.7, -0.2, 0.15, 1.5),
        (1.1, -0.4, -0.1, 0.7),
    ]
    for loss, gx, dx, denominator in samples:
        expected = _reference_step(
            reference,
            beta=0.9,
            cap=1e-2,
            loss=loss,
            gx=gx,
            dx=dx,
            denominator=denominator,
        )
        actual = model.update(
            loss=loss,
            grad_dot_param=gx,
            avg_grad_dot_param=dx,
            denominator=denominator,
        )
        assert actual.tau == pytest.approx(expected)
        assert actual.lower_bound == pytest.approx(reference["lb"])


def test_zero_model_or_zero_direction_cannot_inflate_scale() -> None:
    model = MomoModel(beta=0.9, cap=1.0)
    first = model.update(
        loss=0.0,
        grad_dot_param=0.0,
        avg_grad_dot_param=0.0,
        denominator=0.0,
    )
    assert first.tau == 0.0
    assert not first.cap_active


def test_uncapped_model_uses_model_step_without_a_hidden_scale() -> None:
    model = MomoModel(beta=0.0, cap=math.inf, estimate_lower_bound=False)
    step = model.update(
        loss=2.0,
        grad_dot_param=3.0,
        avg_grad_dot_param=3.0,
        denominator=4.0,
    )
    assert step.tau == pytest.approx(0.5)
    assert not step.cap_active


def test_nonfinite_telemetry_fails_closed_without_advancing_state() -> None:
    model = MomoModel(beta=0.9, cap=1.0)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(FloatingPointError):
        model.update(
            loss=float("nan"),
            grad_dot_param=0.0,
            avg_grad_dot_param=0.0,
            denominator=1.0,
        )
    assert model.state_dict() == before


def test_checkpoint_round_trip_is_exact() -> None:
    model = MomoModel(beta=0.8, cap=0.03)
    for index in range(7):
        model.update(
            loss=2.0 / (index + 1),
            grad_dot_param=0.2 - index * 0.01,
            avg_grad_dot_param=0.1 + index * 0.02,
            denominator=1.0 + index,
        )
    state = copy.deepcopy(model.state_dict())
    resumed = MomoModel(beta=0.8, cap=0.03)
    resumed.load_state_dict(state)
    assert resumed.state_dict() == state

    kwargs = dict(
        loss=0.25,
        grad_dot_param=-0.1,
        avg_grad_dot_param=0.4,
        denominator=3.0,
    )
    assert resumed.update(**kwargs) == model.update(**kwargs)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_invalid_denominator_is_rejected(bad: float) -> None:
    model = MomoModel(beta=0.9, cap=1.0)
    error = ValueError if math.isfinite(bad) else FloatingPointError
    with pytest.raises(error):
        model.update(
            loss=1.0,
            grad_dot_param=0.0,
            avg_grad_dot_param=0.0,
            denominator=bad,
        )
