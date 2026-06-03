"""Tests for the Muon hybrid optimizer."""

from __future__ import annotations

import torch

from koptim import Muon
from koptim.muon import zeropower_via_newtonschulz5

from .conftest import train_steps


def test_newton_schulz_orthogonalizes():
    """NS output should have near-unit singular values (≈ orthogonal factor)."""
    torch.manual_seed(0)
    g = torch.randn(64, 48)
    u = zeropower_via_newtonschulz5(g, steps=5).float()
    sv = torch.linalg.svdvals(u)
    assert sv.max() < 1.3 and sv.min() > 0.5, f"singular values not ~1: [{sv.min():.2f}, {sv.max():.2f}]"


def test_routes_by_rank():
    """2-D/4-D params get the Muon momentum buffer; 1-D params get AdamW state."""
    w2d = torch.nn.Parameter(torch.randn(16, 8))
    w4d = torch.nn.Parameter(torch.randn(8, 4, 3, 3))
    b1d = torch.nn.Parameter(torch.randn(16))
    opt = Muon([w2d, w4d, b1d], lr=1e-2, adamw_lr=1e-3)
    for p in (w2d, w4d, b1d):
        p.grad = torch.randn_like(p)
    opt.step()
    assert "momentum_buffer" in opt.state[w2d]
    assert "momentum_buffer" in opt.state[w4d]
    assert "exp_avg" in opt.state[b1d] and "exp_avg_sq" in opt.state[b1d]


def test_muon_overfits_regression():
    """A small MLP + Muon should drive MSE down on a fixed target."""
    torch.manual_seed(0xC0DE)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, 8),
    )
    opt = Muon(model.parameters(), lr=2e-2, adamw_lr=3e-3)
    x = torch.randn(64, 32)
    y = torch.randn(64, 8)
    initial = (model(x) - y).pow(2).mean().item()
    train_steps(model, opt, [(x, y)] * 80)
    final = (model(x) - y).pow(2).mean().item()
    assert final < 0.6 * initial, f"loss did not drop: {initial:.4f} -> {final:.4f}"


def test_muon_bf16_no_nan():
    """bf16 weights + stochastic rounding should train without NaN."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 8)).to(torch.bfloat16)
    opt = Muon(model.parameters(), lr=2e-2, adamw_lr=3e-3, bf16_method="stochastic_rounding")
    x = torch.randn(64, 32, dtype=torch.bfloat16)
    y = torch.randn(64, 8, dtype=torch.bfloat16)
    for _ in range(30):
        opt.zero_grad()
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)
