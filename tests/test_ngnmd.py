from __future__ import annotations

import copy

import pytest
import torch

from kaon import NGNMD


def test_first_step_matches_algorithm_2_scalar() -> None:
    p = torch.nn.Parameter(torch.tensor([1.0]))
    opt = NGNMD([p], lr=0.5, betas=(0.9, 0.0), eps=1e-12)
    p.grad = torch.tensor([2.0])
    opt.step(loss=torch.tensor(1.0))
    gamma = 0.5 / (1.0 + 0.5 * 2.0 / (2.0 * 1.0))
    assert opt.get_effective_lr() == pytest.approx(gamma)
    assert p.item() == pytest.approx(1.0 - 0.1 * gamma)


@pytest.mark.parametrize("loss", [None, -1.0, float("nan"), float("inf")])
def test_requires_finite_nonnegative_loss(loss) -> None:
    p = torch.nn.Parameter(torch.ones(1))
    p.grad = torch.ones_like(p)
    with pytest.raises(ValueError, match="non-negative|requires"):
        NGNMD([p]).step(loss=loss)


def test_large_c_remains_finite_on_quadratic() -> None:
    p = torch.nn.Parameter(torch.tensor([10.0]))
    opt = NGNMD([p], lr=1e6, betas=(0.9, 0.999))
    for _ in range(200):
        loss = 0.5 * p.square().sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(loss=loss)
    assert torch.isfinite(p).all()
    assert abs(p.item()) <= 10.0


def test_moderate_c_converges_on_quadratic() -> None:
    p = torch.nn.Parameter(torch.tensor([10.0]))
    opt = NGNMD([p], lr=1.0, betas=(0.9, 0.999))
    for _ in range(400):
        loss = 0.5 * p.square().sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(loss=loss)
    assert abs(p.item()) < 0.1


def test_checkpoint_resume_is_exact() -> None:
    p = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
    opt = NGNMD([p], lr=0.1)
    for _ in range(5):
        loss = p.square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(loss=loss)
    saved = copy.deepcopy(opt.state_dict())
    q = torch.nn.Parameter(p.detach().clone())
    resumed = NGNMD([q], lr=9.0)
    resumed.load_state_dict(saved)
    for _ in range(3):
        for value, optimizer in ((p, opt), (q, resumed)):
            loss = value.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step(loss=loss)
    torch.testing.assert_close(q, p, rtol=0.0, atol=0.0)


def test_loss_scale_is_nearly_invariant() -> None:
    base = torch.tensor([2.0, -1.0])
    params = [torch.nn.Parameter(base.clone()), torch.nn.Parameter(base.clone())]
    opts = [NGNMD([params[0]], lr=0.1, eps=1e-30), NGNMD([params[1]], lr=0.1, eps=1e-30)]
    for _ in range(8):
        losses = (params[0].square().mean(), 100.0 * params[1].square().mean())
        for _param, opt, loss in zip(params, opts, losses, strict=True):
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(loss=loss)
    torch.testing.assert_close(params[1], params[0], rtol=2e-6, atol=2e-6)
