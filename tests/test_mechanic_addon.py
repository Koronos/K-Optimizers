from __future__ import annotations

import copy

import pytest
import torch

from kaon._mechanic_addon import MechanicAddon


def _step_quadratic(param: torch.Tensor, opt: MechanicAddon) -> None:
    param.grad = param.detach().clone()
    opt.step()
    opt.zero_grad()


def test_reference_sign_positive_is_favorable_and_negative_triggers_guard():
    p = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    opt = MechanicAddon(torch.optim.SGD([p], lr=0.123), guard=True)

    p.grad = torch.ones_like(p)
    opt.step()
    scale_before = opt.get_scale()

    p.grad = torch.ones_like(p)
    opt.step()
    assert opt._last_h > 0.0
    assert opt._guard_events == 0

    scale_before = opt.get_scale()
    p.grad = -torch.ones_like(p)
    opt.step()
    assert opt._last_h < 0.0
    assert opt._guard_events == 1
    assert opt.get_scale() <= scale_before / 2 + 1e-18
    assert opt.last_stats.h < 0.0
    assert opt.last_stats.guarded
    assert opt.last_stats.candidate_scale >= opt.last_stats.scale


@pytest.mark.parametrize("guard", [False, True])
def test_quadratic_makes_progress_without_nonfinite_values(guard: bool):
    p = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    opt = MechanicAddon(torch.optim.SGD([p], lr=7.0, momentum=0.9), guard=guard)
    initial = p.detach().abs().item()

    for _ in range(300):
        _step_quadratic(p, opt)

    assert torch.isfinite(p).all()
    assert p.detach().abs().item() < initial
    assert opt.get_scale() > 0.0
    assert all(group["lr"] == 1.0 for group in opt.param_groups)


def test_state_dict_roundtrip_resumes_exactly():
    p1 = torch.tensor([2.0, -1.0], dtype=torch.float64, requires_grad=True)
    opt1 = MechanicAddon(torch.optim.SGD([p1], lr=0.2, momentum=0.8), guard=True)
    for _ in range(12):
        _step_quadratic(p1, opt1)

    checkpoint = copy.deepcopy(opt1.state_dict())
    p2 = p1.detach().clone().requires_grad_(True)
    opt2 = MechanicAddon(torch.optim.SGD([p2], lr=9.0, momentum=0.8), guard=True)
    opt2.load_state_dict(checkpoint)

    for _ in range(20):
        _step_quadratic(p1, opt1)
        _step_quadratic(p2, opt2)

    torch.testing.assert_close(p2, p1, rtol=0.0, atol=0.0)
    assert opt2.get_scale() == opt1.get_scale()
    assert opt2._last_h == opt1._last_h


def test_nonfinite_gradient_skips_params_and_inner_state():
    p = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    inner = torch.optim.SGD([p], lr=1.0, momentum=0.9)
    opt = MechanicAddon(inner, guard=True)
    _step_quadratic(p, opt)

    before_param = p.detach().clone()
    before_inner = copy.deepcopy(inner.state_dict())
    before_scale = opt.get_scale()
    p.grad = torch.tensor([float("nan")], dtype=p.dtype)
    opt.step()

    torch.testing.assert_close(p, before_param, rtol=0.0, atol=0.0)
    assert inner.state_dict() == before_inner
    assert opt.get_scale() == before_scale
    assert opt._nonfinite_steps == 1


def test_feedback_overflow_fails_before_advancing_inner_state():
    p = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    inner = torch.optim.SGD([p], lr=1.0, momentum=0.9)
    opt = MechanicAddon(inner)
    _step_quadratic(p, opt)

    p.data.fill_(1e308)
    p.grad = torch.full_like(p, 1e308)
    before_param = p.detach().clone()
    before_inner = copy.deepcopy(inner.state_dict())

    with pytest.raises(FloatingPointError, match="feedback overflowed"):
        opt.step()

    torch.testing.assert_close(p, before_param, rtol=0.0, atol=0.0)
    assert inner.state_dict() == before_inner
    assert opt._nonfinite_steps == 1


def test_zero_grad_and_param_groups_are_delegated():
    p = torch.tensor([1.0], requires_grad=True)
    inner = torch.optim.SGD([p], lr=0.1)
    opt = MechanicAddon(inner)
    assert opt.param_groups is inner.param_groups
    p.grad = torch.ones_like(p)
    opt.zero_grad(set_to_none=True)
    assert p.grad is None


def test_opt_in_diagnostics_are_emitted_by_the_addon(monkeypatch, capsys):
    monkeypatch.setenv("KAON_MECHANIC_LOG_EVERY", "1")
    p = torch.tensor([1.0], requires_grad=True)
    opt = MechanicAddon(torch.optim.SGD([p], lr=0.1))

    _step_quadratic(p, opt)

    output = capsys.readouterr().out
    assert "[kaon.mechanic] step=1" in output
    assert "scale=" in output
    assert "h=" in output
