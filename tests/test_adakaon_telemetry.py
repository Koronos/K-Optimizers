"""Passive native/foreach telemetry for Adakaon."""

from __future__ import annotations

import math

import pytest
import torch

from kaon import Adakaon
from kaon._telemetry import AdakaonStepTelemetry


def _optimizer(params, *, foreach: bool, beta1: float = 0.7, weight_decay: float = 0.0):
    return Adakaon(
        params,
        lr=0.1,
        betas=(beta1, 0.9),
        eps=(1e-12, 1e-3),
        weight_decay=weight_decay,
        clip_threshold=1.0,
        momentum_dtype="float32",
        cautious=False,
        gradient_centralization=False,
        foreach=foreach,
    )


@pytest.mark.parametrize("foreach", [False, True])
@pytest.mark.parametrize("beta1", [0.0, 0.7])
def test_callback_is_bit_exactly_passive(foreach: bool, beta1: float) -> None:
    torch.manual_seed(123)
    initial = [torch.randn(3, 4), torch.randn(3, 4), torch.randn(5)]
    plain_params = [torch.nn.Parameter(value.clone()) for value in initial]
    observed_params = [torch.nn.Parameter(value.clone()) for value in initial]
    plain = _optimizer(plain_params, foreach=foreach, beta1=beta1, weight_decay=0.03)
    observed = _optimizer(observed_params, foreach=foreach, beta1=beta1, weight_decay=0.03)
    events: list[AdakaonStepTelemetry] = []
    observed._set_step_telemetry_hook(events.append)

    for seed in range(4):
        generator = torch.Generator().manual_seed(seed)
        grads = [torch.randn(p.shape, generator=generator) for p in plain_params]
        for p, grad in zip(plain_params, grads, strict=True):
            p.grad = grad.clone()
        for p, grad in zip(observed_params, grads, strict=True):
            p.grad = grad.clone()
        plain.step()
        observed.step()

    assert len(events) == 4
    for left, right in zip(plain_params, observed_params, strict=True):
        assert torch.equal(left, right)
        left_state, right_state = plain.state[left], observed.state[right]
        assert left_state.keys() == right_state.keys()
        for key in left_state:
            if torch.is_tensor(left_state[key]):
                assert torch.equal(left_state[key], right_state[key])
            else:
                assert left_state[key] == right_state[key]


@pytest.mark.parametrize("foreach", [False, True])
def test_quadratic_direction_and_lagged_identities(foreach: bool) -> None:
    # beta2=0 makes the learned direction sign(grad) for this 1-D quadratic.
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    opt = Adakaon(
        [p],
        lr=0.1,
        betas=(0.0, 0.0),
        eps=(0.0, 1e-3),
        weight_decay=0.2,
        clip_threshold=10.0,
        cautious=False,
        gradient_centralization=False,
        foreach=foreach,
    )
    events: list[AdakaonStepTelemetry] = []
    opt._set_step_telemetry_hook(events.append)

    p.grad = p.detach().clone()
    opt.step()
    first = events[-1]
    assert first.active_numel == 2
    assert first.previous_numel == 0
    assert first.grad_norm_sq == pytest.approx(5.0)
    assert first.direction_norm_sq == pytest.approx(2.0)
    assert first.grad_direction_dot == pytest.approx(3.0)
    assert first.prev_direction_norm_sq == pytest.approx(0.0)
    assert first.grad_prev_direction_cosine is None
    assert first.decay_direction_norm_sq == pytest.approx(0.2)
    assert first.finite

    second_grad = p.detach().clone()
    p.grad = second_grad
    opt.step()
    second = events[-1]
    expected_prev_dot = float((second_grad * torch.tensor([1.0, -1.0])).sum())
    assert second.previous_numel == 2
    assert second.prev_direction_norm_sq == pytest.approx(2.0)
    assert second.grad_prev_direction_dot == pytest.approx(expected_prev_dot)
    expected_cosine = expected_prev_dot / math.sqrt(float(second_grad.square().sum()) * 2.0)
    assert second.grad_prev_direction_cosine == pytest.approx(expected_cosine)


def test_momentum_uses_existing_buffer_for_lagged_direction() -> None:
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    opt = _optimizer([p], foreach=False, beta1=0.5)
    events: list[AdakaonStepTelemetry] = []
    opt._set_step_telemetry_hook(events.append)

    p.grad = torch.tensor([1.0, -2.0])
    opt.step()
    previous_momentum = opt.state[p]["m"].clone()
    p.grad = torch.tensor([0.5, -1.5])
    expected_dot = float((p.grad * previous_momentum).sum())
    opt.step()

    assert events[-1].previous_numel == p.numel()
    assert events[-1].grad_prev_direction_dot == pytest.approx(expected_dot)


def test_fused_hook_fails_closed() -> None:
    # The check happens before any step/kernel invocation, so a CPU test can
    # exercise it without constructing a fused optimizer.
    p = torch.nn.Parameter(torch.ones(1))
    opt = _optimizer([p], foreach=False)
    opt._fused = True
    with pytest.raises(NotImplementedError, match="not implemented for fused"):
        opt._set_step_telemetry_hook(lambda _event: None)
