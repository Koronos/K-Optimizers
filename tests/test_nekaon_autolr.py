from __future__ import annotations

import copy

import pytest
import torch

from kaon import Nekaon
from kaon._fused_triton import HAS_TRITON


def _optimizer(param: torch.Tensor, *, momentum_dtype: str = "float32", **kwargs) -> Nekaon:
    return Nekaon(
        [param],
        lr=1.0,
        k=1.5,
        betas=(0.5, 0.999),
        weight_decay=0.0,
        momentum_dtype=momentum_dtype,
        cautious=False,
        gradient_centralization=False,
        foreach=False,
        auto_lr=True,
        **kwargs,
    )


def _step(param: torch.Tensor, optimizer: Nekaon, grad: float) -> None:
    param.grad = torch.full_like(param, grad)
    optimizer.step()
    optimizer.zero_grad()


def test_true_and_live_views_share_the_mechanic_scale() -> None:
    param = torch.tensor([1.0], requires_grad=True)
    initial = param.detach().clone()
    optimizer = _optimizer(param)

    _step(param, optimizer, 1.0)
    live = param.detach().clone()
    optimizer.eval()
    true = param.detach().clone()

    true_step = true - initial
    live_climb = live - true
    assert true_step.abs().max() < 1e-4
    torch.testing.assert_close(live_climb, 1.5 * true_step, rtol=2e-4, atol=2e-7)
    optimizer.train()
    torch.testing.assert_close(param, live, rtol=0.0, atol=2e-7)


@pytest.mark.parametrize("momentum_dtype", ["float32", "int8", "4bit"])
def test_checkpoint_resume_preserves_true_and_live_views(momentum_dtype: str) -> None:
    param = torch.linspace(-0.2, 0.2, 16).reshape(4, 4).requires_grad_(True)
    optimizer = _optimizer(param, momentum_dtype=momentum_dtype)
    for grad in (0.2, -0.1, 0.4, 0.3):
        _step(param, optimizer, grad)

    optimizer.eval()
    saved_param = param.detach().clone()
    saved_state = copy.deepcopy(optimizer.state_dict())
    resumed_param = saved_param.clone().requires_grad_(True)
    resumed = _optimizer(resumed_param, momentum_dtype=momentum_dtype)
    resumed.load_state_dict(saved_state)
    optimizer.train()
    resumed.train()
    torch.testing.assert_close(resumed_param, param, rtol=0.0, atol=0.0)

    # Validate the actual first forward/backward after resume. Injecting the
    # same gradient manually would miss a stale true/live view mismatch.
    optimizer.zero_grad()
    resumed.zero_grad()
    (param.square().sum() + 0.3 * param.sum()).backward()
    (resumed_param.square().sum() + 0.3 * resumed_param.sum()).backward()
    torch.testing.assert_close(resumed_param.grad, param.grad, rtol=0.0, atol=0.0)
    optimizer.step()
    resumed.step()
    optimizer.zero_grad()
    resumed.zero_grad()

    for grad in (-0.25, 0.15, 0.05):
        _step(param, optimizer, grad)
        _step(resumed_param, resumed, grad)
    torch.testing.assert_close(resumed_param, param, rtol=0.0, atol=0.0)
    assert resumed.get_d() == optimizer.get_d()

    optimizer.eval()
    resumed.eval()
    torch.testing.assert_close(resumed_param, param, rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_TRITON,
    reason="Nekaon fused AutoLR requires CUDA + Triton",
)
def test_fused_true_live_cycle_remains_finite() -> None:
    param = torch.linspace(-0.2, 0.2, 4096, device="cuda").reshape(64, 64)
    param.requires_grad_(True)
    optimizer = Nekaon(
        [param],
        lr=1.0,
        k=1.5,
        betas=(0.5, 0.999),
        weight_decay=0.0,
        momentum_dtype="4bit",
        cautious=False,
        gradient_centralization=False,
        fused=True,
        auto_lr=True,
    )
    for index in range(8):
        _step(param, optimizer, 0.2 if index % 2 else -0.1)
    optimizer.eval()
    torch.cuda.synchronize()

    assert torch.isfinite(param).all()
    assert 0.0 < optimizer.get_d() < 1.0


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_TRITON,
    reason="Nekaon fused AutoLR requires CUDA + Triton",
)
def test_bf16_fused_resume_restores_exact_first_forward_and_step() -> None:
    initial = torch.linspace(-0.2, 0.2, 4096, device="cuda", dtype=torch.bfloat16).reshape(64, 64)
    param = initial.clone().requires_grad_(True)
    optimizer = Nekaon(
        [param], lr=1.0, k=1.5, betas=(0.5, 0.999), weight_decay=0.0,
        momentum_dtype="4bit", cautious=False, gradient_centralization=False,
        fused=True, auto_lr=True,
    )
    for grad in (0.2, -0.1, 0.4, 0.3):
        _step(param, optimizer, grad)

    optimizer.eval()
    saved_param = param.detach().clone()
    saved_state = copy.deepcopy(optimizer.state_dict())
    optimizer.train()

    resumed_param = saved_param.clone().requires_grad_(True)
    resumed = Nekaon(
        [resumed_param], lr=1.0, k=1.5, betas=(0.5, 0.999), weight_decay=0.0,
        momentum_dtype="4bit", cautious=False, gradient_centralization=False,
        fused=True, auto_lr=True,
    )
    resumed.load_state_dict(saved_state)
    torch.testing.assert_close(resumed_param, param, rtol=0.0, atol=0.0)

    optimizer.zero_grad()
    resumed.zero_grad()
    (param.float().square().sum() + 0.3 * param.float().sum()).backward()
    (resumed_param.float().square().sum() + 0.3 * resumed_param.float().sum()).backward()
    torch.testing.assert_close(resumed_param.grad, param.grad, rtol=0.0, atol=0.0)
    optimizer.step()
    resumed.step()
    torch.cuda.synchronize()
    torch.testing.assert_close(resumed_param, param, rtol=0.0, atol=0.0)
    assert resumed.get_d() == optimizer.get_d()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_TRITON,
    reason="Nekaon fused AutoLR requires CUDA + Triton",
)
def test_bf16_fused_and_native_autolr_remain_numerically_equivalent() -> None:
    initial = torch.linspace(-0.2, 0.2, 4096, device="cuda", dtype=torch.bfloat16).reshape(64, 64)
    params = [initial.clone().requires_grad_(True), initial.clone().requires_grad_(True)]
    optimizers = [
        Nekaon(
            [param], lr=1.0, k=1.5, betas=(0.5, 0.999), weight_decay=0.0,
            momentum_dtype="4bit", cautious=False, gradient_centralization=False,
            fused=fused, foreach=False, auto_lr=True,
        )
        for param, fused in zip(params, (False, True), strict=True)
    ]
    for grad in (0.2, -0.1, 0.4, 0.3, -0.25, 0.15, 0.05, -0.2):
        for param, optimizer in zip(params, optimizers, strict=True):
            _step(param, optimizer, grad)
    for optimizer in optimizers:
        optimizer.eval()
    torch.cuda.synchronize()
    torch.testing.assert_close(params[1], params[0], rtol=2e-2, atol=2e-3)
    assert optimizers[1].get_d() == pytest.approx(optimizers[0].get_d(), rel=2e-2)
