"""Tests for the Adafusion optimizer."""

from __future__ import annotations

import math

import pytest
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


def _parity_params():
    """A mix that exercises every fast-path branch (factored, conv, and 1-D).

    Includes repeated 2-D shapes and repeated 1-D lengths (so buckets have N>1),
    distinct lengths, and a conv (matrixize).
    """
    g = torch.Generator().manual_seed(0)
    shapes = [
        (64, 128), (128, 64), (64, 128),      # 2-D, one shape repeated -> bucket N=2
        (32, 8, 3, 3),                        # conv (matrixize)
        (8, 96), (96, 8),                     # LoRA-like 2-D
        (40,), (40,), (128,), (320,),         # 1-D: repeated length + distinct lengths
    ]
    return [torch.nn.Parameter(torch.randn(*s, generator=g) * 0.05) for s in shapes]


@pytest.mark.parametrize(
    "cfg",
    [
        dict(lr=1e-3, betas=(0.0, 0.999)),                                  # no momentum
        dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="float32"),        # fp32 momentum
        dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="bfloat16"),       # bf16 momentum
        dict(lr=1e-3, betas=(0.9, 0.999), weight_decay=0.02),               # weight decay
        dict(lr=1e-3, betas=(0.9, 0.999), cautious=True),                   # cautious mask
    ],
)
def test_foreach_matches_per_param(cfg):
    """foreach=True is element-for-element equal to the per-parameter path.

    fp32 params keep stochastic rounding a no-op, so the only difference between
    the two code paths would be a real bug. Bit-exact on CPU.
    """
    pa = _parity_params()
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    oa = Adafusion(pa, foreach=True, **cfg)
    ob = Adafusion(pb, foreach=False, **cfg)
    gg = torch.Generator().manual_seed(7)
    for _ in range(10):
        for a, b in zip(pa, pb):
            grad = torch.randn(*a.shape, generator=gg) * 0.02
            a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


def test_foreach_single_param_uses_fallback():
    """A lone eligible param (e.g. gradient-release) still steps correctly."""
    p = torch.nn.Parameter(torch.randn(16, 16))
    opt = Adafusion([p], lr=1e-3, betas=(0.9, 0.999), foreach=True)
    p.grad = torch.randn_like(p)
    before = p.detach().clone()
    opt.step()
    assert torch.isfinite(p).all() and not torch.equal(before, p.detach())


def test_foreach_bf16_weights_train_no_nan():
    """Batched stochastic-rounding update stays finite over many steps."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 32), torch.nn.GELU(),
        torch.nn.Linear(32, 8),
    ).to(torch.bfloat16)
    opt = Adafusion(model.parameters(), lr=3e-3, betas=(0.0, 0.999),
                    bf16_method="stochastic_rounding", foreach=True)
    x = torch.randn(64, 32, dtype=torch.bfloat16)
    y = torch.randn(64, 8, dtype=torch.bfloat16)
    for _ in range(40):
        opt.zero_grad()
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)
