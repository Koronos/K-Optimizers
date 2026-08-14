"""Focused tests for the unexported MoMo-Adakaon research path."""

from __future__ import annotations

import copy

import pytest
import torch

from kaon._momo_adakaon import MomoAdakaon


def _quadratic_step(param: torch.nn.Parameter, opt: MomoAdakaon, target: float) -> float:
    opt.zero_grad()
    loss = 0.5 * (param - target).square().sum()
    loss.backward()
    opt.step(loss=loss)
    return float(loss.detach())


def test_loss_is_required_and_missing_loss_cannot_mutate_state() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    opt = MomoAdakaon([param], cap=0.01)
    before = param.detach().clone()
    with pytest.raises(ValueError, match="exactly one"):
        opt.step()
    assert torch.equal(param, before)
    assert len(opt.state) == 0


def test_parameter_translation_does_not_change_physical_trajectory() -> None:
    near = torch.nn.Parameter(torch.tensor([0.2]))
    far = torch.nn.Parameter(torch.tensor([20.0]))
    opt_near = MomoAdakaon([near], cap=0.03, betas=(0.0, 0.9))
    opt_far = MomoAdakaon([far], cap=0.03, betas=(0.0, 0.9))

    near_displacements = []
    far_displacements = []
    near_lrs = []
    far_lrs = []
    for _ in range(32):
        _quadratic_step(near, opt_near, 0.3)
        _quadratic_step(far, opt_far, 20.1)
        near_displacements.append(float(near.detach()) - 0.2)
        far_displacements.append(float(far.detach()) - 20.0)
        near_lrs.append(opt_near.get_d())
        far_lrs.append(opt_far.get_d())

    assert far_displacements == pytest.approx(near_displacements, rel=2e-3, abs=2e-6)
    assert far_lrs == pytest.approx(near_lrs, rel=2e-3, abs=2e-6)


@pytest.mark.parametrize("beta1", [0.0, 0.9])
def test_quadratic_converges_without_geometric_lr_ramp(beta1: float) -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    opt = MomoAdakaon([param], cap=0.03, betas=(beta1, 0.99))
    losses = [_quadratic_step(param, opt, 0.0) for _ in range(256)]
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0] * 1e-2
    assert max(losses) <= losses[0] * 1.01
    assert opt.get_d() <= 0.03 / (1.0 - beta1**256)


def test_checkpoint_resume_matches_uninterrupted() -> None:
    first = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    opt_first = MomoAdakaon([first], cap=0.03, betas=(0.9, 0.99))
    for _ in range(17):
        _quadratic_step(first, opt_first, 0.25)
    checkpoint = copy.deepcopy(opt_first.state_dict())

    resumed_param = torch.nn.Parameter(first.detach().clone())
    resumed = MomoAdakaon([resumed_param], cap=0.03, betas=(0.9, 0.99))
    resumed.load_state_dict(checkpoint)
    for _ in range(64):
        _quadratic_step(first, opt_first, 0.25)
        _quadratic_step(resumed_param, resumed, 0.25)

    assert torch.equal(resumed_param, first)
    assert resumed.get_d() == opt_first.get_d()
    assert resumed._momo_model.state_dict() == opt_first._momo_model.state_dict()
