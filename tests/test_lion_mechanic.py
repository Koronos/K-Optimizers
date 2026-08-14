from __future__ import annotations

import copy

import pytest
import torch

from kaon import Lion
from kaon._mechanic_addon import MechanicAddon


def _step(parameter, optimizer, gradient):
    parameter.grad = gradient.clone()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


@pytest.mark.parametrize("guard", [False, True])
@pytest.mark.parametrize("momentum_dtype", ["float32", "int8", "4bit"])
def test_lion_mechanic_checkpoint_resumes_exactly(guard, momentum_dtype):
    torch.manual_seed(91)
    initial = torch.randn(16, 8)
    gradients = [torch.randn_like(initial) for _ in range(14)]
    first = torch.nn.Parameter(initial.clone())
    opt_first = MechanicAddon(
        Lion(
            [first],
            lr=0.123,
            momentum_dtype=momentum_dtype,
            cautious=False,
            gradient_centralization=False,
            auto_lr=False,
        ),
        guard=guard,
    )
    for gradient in gradients[:7]:
        _step(first, opt_first, gradient)

    checkpoint = copy.deepcopy(opt_first.state_dict())
    resumed = torch.nn.Parameter(first.detach().clone())
    opt_resumed = MechanicAddon(
        Lion(
            [resumed],
            lr=9.0,
            momentum_dtype=momentum_dtype,
            cautious=False,
            gradient_centralization=False,
            auto_lr=False,
        ),
        guard=guard,
    )
    opt_resumed.load_state_dict(checkpoint)
    assert opt_resumed.inner.state[resumed]["m"].dtype == opt_first.inner.state[first]["m"].dtype

    for gradient in gradients[7:]:
        _step(first, opt_first, gradient)
        _step(resumed, opt_resumed, gradient)

    torch.testing.assert_close(resumed, first, rtol=0.0, atol=0.0)
    assert opt_resumed.get_scale() == opt_first.get_scale()
    assert opt_resumed.last_stats == opt_first.last_stats


def test_lion_mechanic_forces_unit_inner_lr_and_disables_nested_autolr():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    inner = Lion(
        [parameter],
        lr=3e-4,
        auto_lr=False,
        cautious=False,
        gradient_centralization=False,
    )
    optimizer = MechanicAddon(inner)
    assert inner._autolr is None
    assert optimizer.param_groups[0]["lr"] == 1.0
    _step(parameter, optimizer, torch.tensor([0.5, -0.25]))
    assert optimizer.get_scale() > 0.0
    assert torch.isfinite(parameter).all()
