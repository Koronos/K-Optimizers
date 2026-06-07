"""Tests for the Lion (Lion sign-momentum on Adafusion's backend) optimizer."""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
import torch

from kaon import Lion

from .conftest import train_steps


def _ref_lion_step(
    p: np.ndarray,
    g: np.ndarray,
    m: np.ndarray,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    cautious: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference numpy Lion step mirroring Lion's fp32 per-param path.

    Returns ``(new_p, new_m)``. Cautious masking and decoupled weight decay
    follow the same order Lion uses (WD folded into the delta, then the
    cautious mask on ``delta * g``), so this is bit-comparable up to fp rounding.
    """
    c = beta1 * m + (1.0 - beta1) * g
    update = np.sign(c)                       # +1 / 0 / -1
    new_m = beta2 * m + (1.0 - beta2) * g     # EMA updated AFTER the direction
    delta = update + weight_decay * p
    if cautious:
        mask = (delta * g > 0).astype(delta.dtype)
        denom = max(mask.mean(), 1e-8)
        delta = delta * mask / denom
    new_p = p - lr * delta
    return new_p, new_m


def test_construct_and_step():
    """Construct and take a step on tiny CPU tensors (2-D + 1-D + conv)."""
    params = [
        torch.nn.Parameter(torch.randn(8, 4)),
        torch.nn.Parameter(torch.randn(5)),
        torch.nn.Parameter(torch.randn(3, 2, 3, 3)),
    ]
    opt = Lion(params, lr=1e-4, betas=(0.9, 0.99))
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()  # must not raise
    for p in params:
        assert torch.isfinite(p).all()


@pytest.mark.parametrize("cautious", [False, True])
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_matches_numpy_reference(cautious, weight_decay):
    """The fp32 sign-momentum math matches an independent numpy reference."""
    torch.manual_seed(0)
    lr, b1, b2 = 0.01, 0.9, 0.99
    p = torch.nn.Parameter(torch.randn(16, 7, dtype=torch.float64).float())
    opt = Lion(
        [p], lr=lr, betas=(b1, b2), weight_decay=weight_decay,
        momentum_dtype="float32", cautious=cautious, foreach=False,
    )
    pr = p.detach().numpy().copy().astype(np.float64)
    mr = np.zeros_like(pr)
    gg = torch.Generator().manual_seed(3)
    for _ in range(12):
        g = torch.randn(16, 7, generator=gg)
        p.grad = g.clone()
        opt.step()
        pr, mr = _ref_lion_step(
            pr, g.numpy().astype(np.float64), mr, lr, b1, b2, weight_decay, cautious
        )
    torch.testing.assert_close(p.detach().double(), torch.from_numpy(pr), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        opt.state[p]["m"].double(), torch.from_numpy(mr), rtol=1e-5, atol=1e-6
    )


def test_update_is_sign_of_interpolated_momentum():
    """On the very first step (m == 0) the direction is exactly sign(g)."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(64))
    g = torch.randn(64)
    g[g.abs() < 1e-3] = 0.5  # avoid exact-zero coords so sign is unambiguous
    opt = Lion([p], lr=1.0, betas=(0.9, 0.99), weight_decay=0.0,
                    momentum_dtype="float32", cautious=False, foreach=False)
    p.grad = g.clone()
    opt.step()
    # p was 0; with lr=1 and no WD, the step is -sign(g), so p == -sign(g).
    torch.testing.assert_close(p.detach(), -torch.sign(g))


def test_cautious_masks_disagreeing_coords():
    """Cautious zeroes coords where the update sign disagrees with the gradient.

    Build a momentum that points opposite the gradient on chosen coords so the
    interpolated direction disagrees with g there; those coords must not move,
    and the surviving step magnitude is rescaled up by 1/mean(mask).
    """
    torch.manual_seed(0)
    n = 100
    p = torch.nn.Parameter(torch.zeros(n))
    opt = Lion([p], lr=1.0, betas=(0.5, 0.99), weight_decay=0.0,
                    momentum_dtype="float32", cautious=True, foreach=False)
    # Seed momentum opposite to the gradient on the first half of the coords.
    g = torch.ones(n)
    opt.state[p]["m"] = torch.where(
        torch.arange(n) < n // 2, torch.full((n,), -10.0), torch.zeros(n)
    )
    p.grad = g.clone()
    opt.step()
    # c = 0.5*m + 0.5*g: first half -> -4.5 (sign -1, disagrees with g>0 -> masked),
    # second half -> +0.5 (sign +1, agrees). Masked coords stay at 0.
    moved = p.detach() != 0
    assert not moved[: n // 2].any(), "disagreeing coords must be masked (unchanged)"
    assert moved[n // 2 :].all(), "agreeing coords must move"
    # Rescale preserves mean magnitude: surviving step = lr / mean(mask) = 1 / 0.5 = 2.
    torch.testing.assert_close(
        p.detach()[n // 2 :], torch.full((n // 2,), -2.0)
    )


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
def test_momentum_dtype_variants_construct_and_step(momentum_dtype):
    """Every momentum_dtype constructs, steps without NaN, and stores the buffer."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(32, 16))
    opt = Lion([p], lr=1e-3, betas=(0.9, 0.99), momentum_dtype=momentum_dtype)
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()
    assert torch.isfinite(p).all()
    assert "m" in opt.state[p]
    expected = {
        "bfloat16": torch.bfloat16, "float32": torch.float32,
        "int8": torch.int8, "4bit": torch.uint8,
    }[momentum_dtype]
    assert opt.state[p]["m"].dtype == expected


def test_4bit_is_half_byte_per_param():
    """The 4-bit store is a real ~0.5 B/param packed buffer."""
    p = torch.nn.Parameter(torch.randn(256, 256))
    opt = Lion([p], betas=(0.9, 0.99), momentum_dtype="4bit", momentum_4bit_block=128)
    p.grad = torch.randn_like(p)
    opt.step()
    st = opt.state[p]
    assert st["m"].dtype == torch.uint8
    assert st["m"].numel() == (p.numel() + 1) // 2  # exactly 0.5 B/param packed


def test_int8_is_one_byte_per_param():
    p = torch.nn.Parameter(torch.randn(128, 128))
    opt = Lion([p], betas=(0.9, 0.99), momentum_dtype="int8")
    p.grad = torch.randn_like(p)
    opt.step()
    assert opt.state[p]["m"].numel() == p.numel()  # one int8 byte per param


def test_single_momentum_buffer_no_second_moment():
    """Lion keeps ONE momentum buffer and NO second moment (the memory win)."""
    p = torch.nn.Parameter(torch.randn(64, 64))
    opt = Lion([p], lr=1e-3, betas=(0.9, 0.99), momentum_dtype="float32")
    p.grad = torch.randn_like(p)
    opt.step()
    tensor_keys = {k for k, v in opt.state[p].items() if torch.is_tensor(v)}
    assert tensor_keys == {"m"}, f"expected only the momentum buffer, got {tensor_keys}"


def _parity_params():
    g = torch.Generator().manual_seed(0)
    shapes = [
        (64, 128), (128, 64), (64, 128),   # 2-D, one shape repeated -> bucket N=2
        (32, 8, 3, 3),                     # conv
        (8, 96), (96, 8),                  # same numel, different shape (must not co-bucket)
        (40,), (40,), (128,), (320,),      # 1-D
    ]
    return [torch.nn.Parameter(torch.randn(*s, generator=g) * 0.05) for s in shapes]


@pytest.mark.parametrize(
    "cfg",
    [
        dict(lr=1e-3, betas=(0.9, 0.99), momentum_dtype="float32"),
        dict(lr=1e-3, betas=(0.9, 0.99), momentum_dtype="bfloat16"),
        dict(lr=1e-3, betas=(0.9, 0.99), momentum_dtype="int8"),
        dict(lr=1e-3, betas=(0.9, 0.99), momentum_dtype="int8", weight_decay=0.02),
        dict(lr=1e-3, betas=(0.9, 0.99), momentum_dtype="4bit"),
        dict(lr=1e-3, betas=(0.9, 0.99), momentum_dtype="4bit", weight_decay=0.02),
        dict(lr=1e-3, betas=(0.9, 0.99), momentum_dtype="4bit", momentum_4bit_block=64),
        dict(lr=1e-3, betas=(0.9, 0.99), weight_decay=0.02),
        dict(lr=1e-3, betas=(0.9, 0.99), cautious=True),
        dict(lr=1e-3, betas=(0.9, 0.99), cautious=False),
    ],
)
def test_foreach_matches_per_param(cfg):
    """foreach=True is element-for-element equal to the per-parameter path (fp32)."""
    pa = _parity_params()
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    oa = Lion(pa, foreach=True, **cfg)
    ob = Lion(pb, foreach=False, **cfg)
    gg = torch.Generator().manual_seed(7)
    for _ in range(10):
        for a, b in zip(pa, pb, strict=False):
            grad = torch.randn(*a.shape, generator=gg) * 0.02
            a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb, strict=False):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


def test_foreach_chunking_is_exact():
    """A tiny stack budget splits buckets and routes large weights to the loop —
    the result must still equal the per-parameter path exactly (int8 + WD)."""
    pa = _parity_params()
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    oa = Lion(pa, lr=1e-3, betas=(0.9, 0.99), momentum_dtype="int8",
                   weight_decay=0.02, foreach=True, foreach_stack_budget=200)
    ob = Lion(pb, lr=1e-3, betas=(0.9, 0.99), momentum_dtype="int8",
                   weight_decay=0.02, foreach=False)
    gg = torch.Generator().manual_seed(7)
    for _ in range(8):
        for a, b in zip(pa, pb, strict=False):
            grad = torch.randn(*a.shape, generator=gg) * 0.02
            a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb, strict=False):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


def test_overfits_regression():
    torch.manual_seed(0xC0DE)
    model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 8))
    opt = Lion(model.parameters(), lr=3e-3, betas=(0.9, 0.99))
    x = torch.randn(64, 32)
    y = torch.randn(64, 8)
    initial = (model(x) - y).pow(2).mean().item()
    train_steps(model, opt, [(x, y)] * 120)
    final = (model(x) - y).pow(2).mean().item()
    assert final < 0.5 * initial, f"loss did not drop: {initial:.4f} -> {final:.4f}"


def test_bf16_weights_train_no_nan():
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 8)
    ).to(torch.bfloat16)
    opt = Lion(model.parameters(), lr=3e-3, betas=(0.9, 0.99),
                    bf16_method="stochastic_rounding")
    x = torch.randn(64, 32, dtype=torch.bfloat16)
    y = torch.randn(64, 8, dtype=torch.bfloat16)
    for _ in range(30):
        opt.zero_grad()
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
def test_checkpoint_roundtrip_preserves_momentum_dtype(momentum_dtype):
    """torch.save/load resumes BIT-EXACTLY and keeps the configured momentum dtype.

    torch's default load_state_dict upcasts state to the param dtype (fp32);
    Lion overrides load_state_dict to restore the stored dtype.
    """
    torch.manual_seed(0)
    p_ref = torch.randn(16, 8)
    grads = [torch.randn(16, 8) for _ in range(10)]

    a = torch.nn.Parameter(p_ref.clone())
    opt_a = Lion([a], lr=1e-3, betas=(0.9, 0.99), momentum_dtype=momentum_dtype)
    for g in grads[:5]:
        a.grad = g.clone()
        opt_a.step()

    buf = io.BytesIO()
    torch.save(opt_a.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, weights_only=False)

    b = torch.nn.Parameter(a.detach().clone())
    opt_b = Lion([b], lr=1e-3, betas=(0.9, 0.99), momentum_dtype=momentum_dtype)
    opt_b.load_state_dict(sd)

    assert opt_b.state[b]["m"].dtype == opt_a.state[a]["m"].dtype

    for g in grads[5:]:
        a.grad = g.clone()
        opt_a.step()
        b.grad = g.clone()
        opt_b.step()
    assert torch.equal(a, b), "resumed run must continue bit-exactly"


def test_invalid_args_rejected():
    p = [torch.nn.Parameter(torch.randn(4, 4))]
    with pytest.raises(ValueError):
        Lion(p, momentum_dtype="2bit")
    with pytest.raises(ValueError):
        Lion(p, betas=(1.0, 0.99))
    with pytest.raises(ValueError):
        Lion(p, betas=(0.9, 1.0))
    with pytest.raises(ValueError):
        Lion(p, lr=-1.0)
    with pytest.raises(ValueError):
        Lion(p, bf16_method="bogus")


def test_kahan_runs():
    """bf16 + kahan path (per-param, +shift buffer) steps without NaN."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 8, dtype=torch.bfloat16))
    opt = Lion([p], lr=1e-3, betas=(0.9, 0.99), bf16_method="kahan")
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()
    assert torch.isfinite(p).all()
    assert "shift" in opt.state[p]


def test_sparse_grad_rejected():
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = Lion([p], lr=1e-3, betas=(0.9, 0.99))
    p.grad = torch.sparse_coo_tensor(torch.tensor([[0], [0]]), torch.tensor([1.0]), (4, 4))
    with pytest.raises(RuntimeError):
        opt.step()


def test_conv_net_trains_no_nan():
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(4, 16, 3, padding=1), torch.nn.GELU(),
        torch.nn.Conv2d(16, 4, 3, padding=1),
    )
    opt = Lion(net.parameters(), lr=1e-3, betas=(0.9, 0.99))
    x = torch.randn(8, 4, 16, 16)
    y = torch.randn(8, 4, 16, 16)
    for _ in range(30):
        opt.zero_grad()
        loss = (net(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
    assert math.isfinite(loss.item())
