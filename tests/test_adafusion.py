"""Tests for the Adafusion optimizer."""

from __future__ import annotations

import math

import torch

from koptim import Adafusion

from .conftest import train_steps


def test_conv_factoring_reduces_state():
    """Conv-aware factoring stores ~0 state for a conv kernel vs the legacy mode."""
    def state_floats(reshape: bool) -> int:
        p = torch.nn.Parameter(torch.randn(64, 32, 3, 3))
        opt = Adafusion([p], lr=1e-3, betas=(0.0, 0.999), factor_conv_as_matrix=reshape)
        p.grad = torch.randn_like(p)
        opt.step()
        return sum(v.numel() for v in opt.state[p].values() if torch.is_tensor(v))

    fixed = state_floats(True)
    legacy = state_floats(False)
    assert fixed < legacy / 5, f"conv-aware factoring should be much smaller: {fixed} vs {legacy}"


def test_bf16_momentum_is_half_state():
    """bf16 momentum buffer is half the bytes of fp32 momentum."""
    def mom_bytes(dtype: str) -> int:
        p = torch.nn.Parameter(torch.randn(128, 128))
        opt = Adafusion([p], lr=1e-3, betas=(0.9, 0.999), momentum_dtype=dtype)
        p.grad = torch.randn_like(p)
        opt.step()
        return opt.state[p]["m"].numel() * opt.state[p]["m"].element_size()

    assert mom_bytes("bfloat16") * 2 == mom_bytes("float32")


def test_overfits_regression():
    torch.manual_seed(0xC0DE)
    model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 8))
    opt = Adafusion(model.parameters(), lr=3e-3, betas=(0.9, 0.999))
    x = torch.randn(64, 32)
    y = torch.randn(64, 8)
    initial = (model(x) - y).pow(2).mean().item()
    train_steps(model, opt, [(x, y)] * 80)
    final = (model(x) - y).pow(2).mean().item()
    assert final < 0.5 * initial, f"loss did not drop: {initial:.4f} -> {final:.4f}"


def test_conv_net_trains_no_nan():
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(4, 16, 3, padding=1), torch.nn.GELU(),
        torch.nn.Conv2d(16, 4, 3, padding=1),
    )
    opt = Adafusion(net.parameters(), lr=3e-3, betas=(0.9, 0.999))
    x = torch.randn(8, 4, 16, 16)
    y = torch.randn(8, 4, 16, 16)
    for _ in range(30):
        opt.zero_grad()
        loss = (net(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
    assert math.isfinite(loss.item())


def test_bf16_weights_train_no_nan():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 8)).to(torch.bfloat16)
    opt = Adafusion(model.parameters(), lr=3e-3, betas=(0.9, 0.999), bf16_method="stochastic_rounding")
    x = torch.randn(64, 32, dtype=torch.bfloat16)
    y = torch.randn(64, 8, dtype=torch.bfloat16)
    for _ in range(30):
        opt.zero_grad()
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)


def test_cautious_runs():
    torch.manual_seed(0)
    model = torch.nn.Linear(16, 16)
    opt = Adafusion(model.parameters(), lr=1e-3, betas=(0.9, 0.999), cautious=True)
    x = torch.randn(8, 16)
    (model(x)).pow(2).mean().backward()
    opt.step()  # must not raise
