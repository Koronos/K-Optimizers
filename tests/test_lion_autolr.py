from __future__ import annotations

import copy

import pytest
import torch

from kaon import Lion


def _optimizer(param, *, momentum_dtype="float32", bf16_method="none"):
    return Lion(
        [param],
        lr=1.0,
        betas=(0.9, 0.99),
        weight_decay=0.0,
        momentum_dtype=momentum_dtype,
        cautious=False,
        gradient_centralization=False,
        bf16_method=bf16_method,
        foreach=False,
        auto_lr=True,
    )


def _step(param, optimizer, grad):
    param.grad = torch.full_like(param, grad)
    optimizer.step()
    optimizer.zero_grad()


@pytest.mark.parametrize("momentum_dtype", ["float32", "int8", "4bit"])
def test_checkpoint_resume_is_exact(momentum_dtype: str) -> None:
    param = torch.linspace(-0.3, 0.3, 16).reshape(4, 4).requires_grad_(True)
    optimizer = _optimizer(param, momentum_dtype=momentum_dtype)
    for grad in (0.2, -0.1, 0.4, 0.3):
        _step(param, optimizer, grad)

    state = copy.deepcopy(optimizer.state_dict())
    resumed_param = param.detach().clone().requires_grad_(True)
    resumed = _optimizer(resumed_param, momentum_dtype=momentum_dtype)
    resumed.load_state_dict(state)
    for grad in (-0.25, 0.15, 0.05):
        _step(param, optimizer, grad)
        _step(resumed_param, resumed, grad)

    torch.testing.assert_close(resumed_param, param, rtol=0.0, atol=0.0)
    assert resumed.get_d() == optimizer.get_d()


@pytest.mark.parametrize("bf16_method", ["stochastic_rounding", "kahan"])
def test_bf16_paths_remain_finite(bf16_method: str) -> None:
    param = torch.linspace(-0.3, 0.3, 64, dtype=torch.bfloat16).reshape(8, 8)
    param.requires_grad_(True)
    optimizer = _optimizer(param, momentum_dtype="bfloat16", bf16_method=bf16_method)
    for index in range(64):
        _step(param, optimizer, 0.2 if index % 3 else -0.1)

    assert torch.isfinite(param).all()
    assert 1e-6 < optimizer.get_d() < 1.0


@pytest.mark.parametrize("bf16_method", ["stochastic_rounding", "kahan"])
def test_bf16_uniform_weights_do_not_quantize_mechanic_feedback_to_zero(
    bf16_method: str,
) -> None:
    param = torch.ones(64, dtype=torch.bfloat16, requires_grad=True)
    optimizer = _optimizer(
        param,
        momentum_dtype="bfloat16",
        bf16_method=bf16_method,
    )
    for _ in range(64):
        _step(param, optimizer, 1.0)

    assert optimizer.get_d() > 1e-6
    assert optimizer._autolr._last_h != 0.0
    assert optimizer._autolr._delta[param].abs().max() > 0
