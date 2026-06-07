"""Tests for the AdaMuon optimizer (orthogonalized momentum + factored variance)."""

from __future__ import annotations

import io

import pytest
import torch

from kaon import AdaMuon
from kaon.adamuon import (
    zeropower_via_newtonschulz5,
    zeropower_via_newtonschulz5_stacked,
)

from .conftest import train_steps


def _parity_params():
    """A mix exercising every fast-path branch (factored, conv, 1-D), with repeated
    shapes so buckets have N>1 and LoRA-like small 2-D weights."""
    g = torch.Generator().manual_seed(0)
    shapes = [
        (64, 128), (128, 64), (64, 128),      # 2-D, one shape repeated -> bucket N=2
        (32, 8, 3, 3),                        # conv (matrixize)
        (8, 96), (96, 8),                     # LoRA-like 2-D
        (40,), (40,), (128,), (320,),         # 1-D: repeated + distinct lengths
    ]
    return [torch.nn.Parameter(torch.randn(*s, generator=g) * 0.05) for s in shapes]


def test_smoke_routes_by_rank():
    """2-D/4-D params get factored row/col + momentum; 1-D get the non-factored v."""
    w2d = torch.nn.Parameter(torch.randn(16, 8))
    w4d = torch.nn.Parameter(torch.randn(8, 4, 3, 3))
    b1d = torch.nn.Parameter(torch.randn(16))
    opt = AdaMuon([w2d, w4d, b1d], lr=1e-2, momentum_dtype="int8")
    for p in (w2d, w4d, b1d):
        p.grad = torch.randn_like(p)
    opt.step()
    for w in (w2d, w4d):
        assert "row" in opt.state[w] and "col" in opt.state[w]
        assert "m" in opt.state[w]                       # quantized momentum, not exp_avg_sq
    assert "v" in opt.state[b1d] and "m" in opt.state[b1d]
    assert "exp_avg_sq" not in opt.state[b1d]            # NOT Muon's fp32 AdamW fallback


def test_overfits_regression():
    """AdaMuon should drive MSE down on a fixed target (orthogonalize+variance works)."""
    torch.manual_seed(0xC0DE)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, 8),
    )
    opt = AdaMuon(model.parameters(), lr=2e-2)
    x = torch.randn(64, 32)
    y = torch.randn(64, 8)
    initial = (model(x) - y).pow(2).mean().item()
    train_steps(model, opt, [(x, y)] * 80)
    final = (model(x) - y).pow(2).mean().item()
    assert final < 0.5 * initial, f"loss did not drop: {initial:.4f} -> {final:.4f}"


def test_conv_net_trains_no_nan():
    """A small Conv2d net exercises the conv matrixize -> NS -> factored path."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(4, 16, 3, padding=1),
        torch.nn.GELU(),
        torch.nn.Conv2d(16, 4, 3, padding=1),
    )
    opt = AdaMuon(model.parameters(), lr=2e-2)
    x = torch.randn(2, 4, 16, 16)
    y = torch.randn(2, 4, 16, 16)
    for _ in range(30):
        opt.zero_grad()
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)


def test_bf16_weights_train_no_nan():
    """bf16 weights + stochastic rounding (NS runs in bf16 internally) train cleanly."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 8)
    ).to(torch.bfloat16)
    opt = AdaMuon(model.parameters(), lr=2e-2, bf16_method="stochastic_rounding")
    x = torch.randn(64, 32, dtype=torch.bfloat16)
    y = torch.randn(64, 8, dtype=torch.bfloat16)
    for _ in range(30):
        opt.zero_grad()
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)


def test_batched_ns_matches_per_slice():
    """Batched bmm Newton-Schulz matches the per-slice helper (within bf16 tol)."""
    torch.manual_seed(0)
    mats = [torch.randn(48, 64) for _ in range(5)]         # R<C and the transpose path
    stacked = torch.stack(mats)
    out_stacked = zeropower_via_newtonschulz5_stacked(stacked, steps=5).float()
    for i, m in enumerate(mats):
        out_one = zeropower_via_newtonschulz5(m, steps=5).float()
        torch.testing.assert_close(out_stacked[i], out_one, rtol=2e-2, atol=2e-2)


def test_orthogonalized_update_singular_values():
    """The orthogonalized signal O should have near-unit singular values."""
    torch.manual_seed(0)
    g = torch.randn(64, 48)
    u = zeropower_via_newtonschulz5(g, steps=5).float()
    sv = torch.linalg.svdvals(u)
    assert sv.max() < 1.3 and sv.min() > 0.5, f"singular values not ~1: [{sv.min():.2f}, {sv.max():.2f}]"


@pytest.mark.parametrize(
    "cfg",
    [
        dict(lr=2e-2, betas=(0.0, 0.999)),                              # no momentum
        dict(lr=2e-2, betas=(0.95, 0.999), momentum_dtype="float32"),   # fp32 momentum
        dict(lr=2e-2, betas=(0.95, 0.999), momentum_dtype="bfloat16"),  # bf16 momentum
        dict(lr=2e-2, betas=(0.95, 0.999), momentum_dtype="int8"),      # int8 momentum
        dict(lr=2e-2, betas=(0.95, 0.999), momentum_dtype="4bit"),      # 4-bit momentum
        dict(lr=2e-2, betas=(0.95, 0.999), weight_decay=0.02),          # weight decay
        dict(lr=2e-2, betas=(0.95, 0.999), cautious=True),             # cautious mask
    ],
)
def test_foreach_matches_per_param(cfg):
    """foreach=True matches the per-parameter path within bf16 Newton-Schulz
    tolerance.

    Unlike Adakaon (all-fp32 math, bit-exact), AdaMuon's 2-D path runs NS in
    bf16, and the batched bmm reduces in a different order than per-slice matmul —
    so the two paths agree closely but not bit-for-bit. 1-D buckets and all the
    fp32 ops are exact; the residual is the bf16 NS on the 2-D weights.
    """
    pa = _parity_params()
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    oa = AdaMuon(pa, foreach=True, **cfg)
    ob = AdaMuon(pb, foreach=False, **cfg)
    gg = torch.Generator().manual_seed(7)
    for _ in range(6):
        for a, b in zip(pa, pb, strict=False):
            grad = torch.randn(*a.shape, generator=gg) * 0.02
            a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb, strict=False):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=2e-2, atol=2e-3)


def test_foreach_1d_bucket_is_bit_exact():
    """The 1-D non-factored bucket (no Newton-Schulz) is bit-exact vs per-param."""
    shapes = [(40,), (40,), (128,), (320,)]
    pa = [torch.nn.Parameter(torch.randn(*s) * 0.05) for s in shapes]
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    oa = AdaMuon(pa, lr=2e-2, betas=(0.95, 0.999), momentum_dtype="int8", foreach=True)
    ob = AdaMuon(pb, lr=2e-2, betas=(0.95, 0.999), momentum_dtype="int8", foreach=False)
    gg = torch.Generator().manual_seed(7)
    for _ in range(8):
        for a, b in zip(pa, pb, strict=False):
            grad = torch.randn(*a.shape, generator=gg) * 0.02
            a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb, strict=False):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
def test_checkpoint_roundtrip_preserves_momentum_dtype(momentum_dtype):
    """torch.save/load resumes bit-exactly and keeps the configured momentum dtype.

    torch's default ``load_state_dict`` upcasts quantized momentum to fp32; AdaMuon
    overrides it to restore the stored dtype (memory + exact resume).
    """
    torch.manual_seed(0)
    p_ref = torch.randn(16, 8)
    grads = [torch.randn(16, 8) for _ in range(10)]

    a = torch.nn.Parameter(p_ref.clone())
    opt_a = AdaMuon([a], lr=2e-2, betas=(0.95, 0.999), momentum_dtype=momentum_dtype)
    for g in grads[:5]:
        a.grad = g.clone()
        opt_a.step()

    buf = io.BytesIO()
    torch.save(opt_a.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, weights_only=False)

    b = torch.nn.Parameter(a.detach().clone())
    opt_b = AdaMuon([b], lr=2e-2, betas=(0.95, 0.999), momentum_dtype=momentum_dtype)
    opt_b.load_state_dict(sd)

    assert opt_b.state[b]["m"].dtype == opt_a.state[a]["m"].dtype

    for g in grads[5:]:
        a.grad = g.clone()
        b.grad = g.clone()
        opt_a.step()
        opt_b.step()
    torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


def test_momentum_is_quantized_state():
    """int8 momentum is ~1 byte/param; bf16 is half of fp32 (Adafactor-class memory)."""
    def mom_bytes(dtype: str) -> int:
        p = torch.nn.Parameter(torch.randn(128, 128))
        opt = AdaMuon([p], lr=2e-2, betas=(0.95, 0.999), momentum_dtype=dtype)
        p.grad = torch.randn_like(p)
        opt.step()
        return opt.state[p]["m"].numel() * opt.state[p]["m"].element_size()

    assert mom_bytes("bfloat16") * 2 == mom_bytes("float32")
    assert mom_bytes("int8") * 4 == mom_bytes("float32")


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(betas=(1.0, 0.999)), "betas\\[0\\]"),
        (dict(betas=(0.9, 1.0)), "betas\\[1\\]"),
        (dict(lr=-1.0), "lr"),
        (dict(ns_steps=0), "ns_steps"),
        (dict(clip_threshold=0.0), "clip_threshold"),
        (dict(momentum_dtype="int4"), "momentum_dtype"),
        (dict(bf16_method="bogus"), "bf16_method"),
    ],
)
def test_invalid_args_rejected(kwargs, match):
    p = torch.nn.Parameter(torch.randn(4, 4))
    with pytest.raises(ValueError, match=match):
        AdaMuon([p], **kwargs)


def test_compile_step_matches_eager():
    """``compile=True`` produces a numerically equivalent update and stays finite."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    ps0 = [torch.randn(16, 24, device=dev) for _ in range(3)]
    gs = [torch.randn(16, 24, device=dev) * 0.1 for _ in range(3)]

    def run(compile):
        ps = [torch.nn.Parameter(p.clone()) for p in ps0]
        opt = AdaMuon(ps, lr=1e-3, betas=(0.95, 0.999), ns_steps=2, cautious=True,
                      momentum_dtype="float32", bf16_method="none", compile=compile)
        for _ in range(2):
            for p, g in zip(ps, gs, strict=True):
                p.grad = g.clone()
            opt.step()
        return [p.detach().clone() for p in ps]

    eager, compiled = run(False), run(True)
    for e, c in zip(eager, compiled, strict=True):
        assert torch.isfinite(c).all()
        # AdaMuon runs Newton-Schulz in bf16 internally -> looser tol than Adakaon
        assert torch.allclose(e, c, rtol=3e-2, atol=2e-3)
