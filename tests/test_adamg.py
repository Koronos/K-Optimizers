"""Tests for the unexported AdamG research implementation."""

from __future__ import annotations

import copy

import pytest
import torch

from kaon._adamg import AdamG


def _step(param: torch.nn.Parameter, opt: AdamG, target: float) -> float:
    opt.zero_grad()
    loss = 0.5 * (param - target).square().sum()
    loss.backward()
    opt.step()
    return float(loss.detach())


def test_first_step_matches_algorithm() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    opt = AdamG([param], cap=1.0)
    _step(param, opt, 0.0)
    v = 0.001
    r = 0.05 * (0.2 * v**0.24)
    m = 0.05 * r
    expected_update = (m / 0.05) / ((v / 0.001) ** 0.5 + 1e-8)
    assert float(param.detach()) == pytest.approx(1.0 - expected_update)


def test_parameter_translation_is_invariant() -> None:
    near = torch.nn.Parameter(torch.tensor([0.2]))
    far = torch.nn.Parameter(torch.tensor([20.0]))
    opt_near = AdamG([near])
    opt_far = AdamG([far])
    for _ in range(64):
        _step(near, opt_near, 0.3)
        _step(far, opt_far, 20.1)
    assert float(far.detach()) - 20.0 == pytest.approx(
        float(near.detach()) - 0.2, rel=2e-3, abs=2e-6
    )


def test_checkpoint_resume_matches() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    opt = AdamG([param])
    for _ in range(17):
        _step(param, opt, 0.25)
    state = copy.deepcopy(opt.state_dict())
    resumed_param = torch.nn.Parameter(param.detach().clone())
    resumed = AdamG([resumed_param])
    resumed.load_state_dict(state)
    for _ in range(64):
        _step(param, opt, 0.25)
        _step(resumed_param, resumed, 0.25)
    assert torch.equal(resumed_param, param)


def test_nonfinite_gradient_fails_before_parameter_update() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    opt = AdamG([param])
    param.grad = torch.tensor([float("nan")])
    with pytest.raises(FloatingPointError):
        opt.step()
    assert float(param.detach()) == 1.0
