"""Tests for AdaBelief — AdaBelief (belief in observed gradients) on kaon's backend.

The numpy reference mirrors AdaBelief's **non-factored (1-D) fp32 path**, which is
the one that matches kozistr's ``pytorch_optimizer.AdaBelief`` exactly (full
per-coordinate residual second moment ``s``; the factored 2-D path uses Adafactor's
row/col approximation and is only checked for self-consistency / foreach parity).
"""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
import torch

from kaon import AdaBelief

from .conftest import train_steps


def _ref_adabelief_1d(
    p: np.ndarray,
    grads: list[np.ndarray],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> np.ndarray:
    """Reference numpy AdaBelief over a sequence of grads (1-D, full residual s).

    Reproduces the kozistr AdaBelief math the 1-D path mirrors:

    * first moment ``m = beta1*m + (1-beta1)*g`` (decay and bias correction both beta1).
    * residual second moment ``s = beta2*s + (1-beta2)*(g-m)^2 + eps``.
    * denom ``= (sqrt(s) + eps) / sqrt(1 - beta2^t)``.
    * decoupled weight decay ``p *= 1 - lr*wd`` BEFORE the moment updates.
    * ``p -= (lr / (1 - beta1^t)) * m / denom``.
    """
    p = p.copy()
    m = np.zeros_like(p)
    s = np.zeros_like(p)
    for t, g in enumerate(grads, start=1):
        if weight_decay != 0:
            p = p * (1.0 - lr * weight_decay)
        m[...] = beta1 * m + (1.0 - beta1) * g
        residual = g - m
        s[...] = beta2 * s + (1.0 - beta2) * residual * residual + eps
        bc2_sq = math.sqrt(1.0 - beta2 ** t)
        de_nom = (np.sqrt(s) + eps) / bc2_sq
        bc1 = 1.0 - beta1 ** t
        p = p - (lr / bc1) * m / de_nom
    return p


def test_construct_and_step():
    """Construct and take a step on tiny CPU tensors (2-D + 1-D + conv)."""
    params = [
        torch.nn.Parameter(torch.randn(8, 4)),
        torch.nn.Parameter(torch.randn(5)),
        torch.nn.Parameter(torch.randn(3, 2, 3, 3)),
    ]
    opt = AdaBelief(params, lr=1e-3)
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()
    for p in params:
        assert torch.isfinite(p).all()


@pytest.mark.parametrize("weight_decay", [0.0, 0.05])
@pytest.mark.parametrize("betas", [(0.9, 0.999), (0.8, 0.99)])
def test_matches_numpy_reference_1d(betas, weight_decay):
    """AdaBelief's 1-D fp32 path matches the numpy AdaBelief reference (cautious + GC off)."""
    torch.manual_seed(11)
    beta1, beta2 = betas
    n = 13
    p0 = torch.randn(n)
    p = torch.nn.Parameter(p0.clone())
    opt = AdaBelief(
        [p], lr=1e-2, betas=betas, eps=1e-16,
        weight_decay=weight_decay, cautious=False, gradient_centralization=False,
        momentum_dtype="float32", foreach=False,
    )
    grads = [torch.randn(n) for _ in range(10)]
    for g in grads:
        p.grad = g.clone()
        opt.step()
    ref = _ref_adabelief_1d(
        p0.numpy(), [g.numpy() for g in grads],
        lr=1e-2, beta1=beta1, beta2=beta2, eps=1e-16,
        weight_decay=weight_decay,
    )
    np.testing.assert_allclose(p.detach().numpy(), ref, rtol=1e-5, atol=1e-6)


def test_belief_second_moment_uses_residual():
    """The second moment tracks (g - m)^2, not g^2 (the AdaBelief signature)."""
    torch.manual_seed(3)
    n = 7
    p = torch.nn.Parameter(torch.randn(n))
    beta1, beta2 = 0.9, 0.999
    eps = 1e-16
    opt = AdaBelief(
        [p], lr=1e-2, betas=(beta1, beta2), eps=eps, cautious=False,
        gradient_centralization=False, momentum_dtype="float32", foreach=False,
    )
    g1 = torch.randn(n)
    p.grad = g1.clone()
    opt.step()
    st = opt.state[p]
    # step 1: m = (1-beta1)*g1; residual = g1 - m; s = (1-beta2)*residual^2 + eps.
    m_expected = (1.0 - beta1) * g1
    residual = g1 - m_expected
    s_expected = (1.0 - beta2) * residual * residual + eps
    torch.testing.assert_close(st["m"], m_expected, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(st["s"], s_expected, rtol=1e-5, atol=1e-7)
    # Sanity: the residual second moment differs from a plain g^2 second moment.
    s_if_plain = (1.0 - beta2) * g1 * g1 + eps
    assert not torch.allclose(st["s"], s_if_plain)


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
def test_momentum_dtype_variants_construct_and_step(momentum_dtype):
    params = [
        torch.nn.Parameter(torch.randn(8, 4)),
        torch.nn.Parameter(torch.randn(5)),
        torch.nn.Parameter(torch.randn(3, 2, 3, 3)),
    ]
    opt = AdaBelief(params, lr=1e-3, momentum_dtype=momentum_dtype)
    for _ in range(3):
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
    for p in params:
        assert torch.isfinite(p).all()


def test_one_momentum_present():
    """AdaBelief carries a single first-moment buffer (plain Adam, not PNM)."""
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = AdaBelief([p], lr=1e-3, momentum_dtype="float32")
    p.grad = torch.randn_like(p)
    opt.step()
    st = opt.state[p]
    assert "m" in st
    assert "m_neg" not in st
    assert st["m"].shape == p.shape


def test_int8_is_one_byte_per_param():
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = AdaBelief([p], lr=1e-3, momentum_dtype="int8")
    p.grad = torch.randn_like(p)
    opt.step()
    st = opt.state[p]
    assert st["m"].numel() == p.numel()
    assert st["m"].dtype == torch.int8


def test_4bit_is_half_byte_per_param():
    p = torch.nn.Parameter(torch.randn(8, 16))
    opt = AdaBelief([p], lr=1e-3, momentum_dtype="4bit", momentum_4bit_block=16)
    p.grad = torch.randn_like(p)
    opt.step()
    st = opt.state[p]
    assert st["m"].numel() == (p.numel() + 1) // 2
    assert st["m"].dtype == torch.uint8


def test_cautious_masks_disagreeing_coords():
    """cautious=True zeroes coords where the final delta disagrees with the grad."""
    torch.manual_seed(0)
    n = 32
    p0 = torch.randn(n)

    def run(cautious: bool) -> torch.Tensor:
        p = torch.nn.Parameter(p0.clone())
        opt = AdaBelief(
            [p], lr=1e-2, betas=(0.9, 0.999), cautious=cautious,
            gradient_centralization=False, momentum_dtype="float32", foreach=False,
        )
        torch.manual_seed(5)
        for _ in range(4):
            p.grad = torch.randn(n)
            opt.step()
        return p.detach().clone()

    assert not torch.allclose(run(True), run(False))


@pytest.mark.parametrize(
    "cfg",
    [
        dict(momentum_dtype="float32", betas=(0.9, 0.999), weight_decay=0.0, cautious=True),
        dict(momentum_dtype="int8", betas=(0.9, 0.999), weight_decay=0.02, cautious=True),
        dict(momentum_dtype="4bit", betas=(0.9, 0.99), weight_decay=0.01, cautious=False),
        dict(momentum_dtype="bfloat16", betas=(0.9, 0.999), weight_decay=0.0, cautious=True),
        dict(momentum_dtype="float32", betas=(0.9, 0.999), weight_decay=0.0, cautious=False),
        dict(momentum_dtype="int8", betas=(0.9, 0.999), weight_decay=0.0, cautious=False),
        dict(momentum_dtype="4bit", betas=(0.9, 0.999), weight_decay=0.0, cautious=True),
        dict(momentum_dtype="bfloat16", betas=(0.9, 0.999), weight_decay=0.0, cautious=False),
    ],
)
def test_foreach_matches_per_param(cfg):
    """foreach=True is element-for-element equal to the per-parameter path (fp32 weights)."""
    def mk() -> list[torch.nn.Parameter]:
        return [
            torch.nn.Parameter(torch.randn(8, 4)),
            torch.nn.Parameter(torch.randn(7, 3)),
            torch.nn.Parameter(torch.randn(5)),
            torch.nn.Parameter(torch.randn(6)),
            torch.nn.Parameter(torch.randn(3, 2, 3, 3)),
        ]

    torch.manual_seed(1)
    pa = mk()
    torch.manual_seed(1)
    pb = mk()
    oa = AdaBelief(pa, lr=1e-3, foreach=True, bf16_method="none", **cfg)
    ob = AdaBelief(pb, lr=1e-3, foreach=False, bf16_method="none", **cfg)
    torch.manual_seed(7)
    for _ in range(6):
        gs = [torch.randn_like(p) for p in pa]
        for p, g in zip(pa, gs, strict=True):
            p.grad = g.clone()
        for p, g in zip(pb, gs, strict=True):
            p.grad = g.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb, strict=True):
        assert torch.equal(a, b)


def test_foreach_chunking_is_exact():
    """Splitting a foreach bucket into chunks must not change the result."""
    def mk() -> list[torch.nn.Parameter]:
        return [torch.nn.Parameter(torch.randn(6, 5)) for _ in range(7)]

    torch.manual_seed(2)
    pa = mk()
    torch.manual_seed(2)
    pb = mk()
    oa = AdaBelief(pa, lr=1e-3, momentum_dtype="int8", weight_decay=0.02,
                   foreach=True, foreach_stack_budget=120)
    ob = AdaBelief(pb, lr=1e-3, momentum_dtype="int8", weight_decay=0.02, foreach=False)
    torch.manual_seed(3)
    for _ in range(5):
        gs = [torch.randn_like(p) for p in pa]
        for p, g in zip(pa, gs, strict=True):
            p.grad = g.clone()
        for p, g in zip(pb, gs, strict=True):
            p.grad = g.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb, strict=True):
        assert torch.equal(a, b)


def test_overfits_regression(toy_mlp, random_batch):
    """AdaBelief should drive a tiny MLP's training loss down on a fixed batch."""
    x, y = random_batch
    opt = AdaBelief(toy_mlp.parameters(), lr=3e-3)
    losses = []
    for _ in range(80):
        opt.zero_grad()
        loss = (toy_mlp(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5


def test_bf16_weights_train_no_nan():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, 8, dtype=torch.bfloat16))
    opt = AdaBelief([p], lr=1e-3, bf16_method="stochastic_rounding")
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()
    assert torch.isfinite(p).all()


def test_kahan_runs():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 4, dtype=torch.bfloat16))
    opt = AdaBelief([p], lr=1e-3, bf16_method="kahan")
    for _ in range(3):
        p.grad = torch.randn_like(p)
        opt.step()
    assert "shift" in opt.state[p]
    assert torch.isfinite(p).all()


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
def test_checkpoint_roundtrip_preserves_momentum_dtype(momentum_dtype):
    """state_dict round-trip preserves the momentum's stored dtype and resumes bit-exactly."""
    torch.manual_seed(0)
    params_a = [torch.nn.Parameter(torch.randn(8, 4)), torch.nn.Parameter(torch.randn(5))]
    opt_a = AdaBelief(params_a, lr=1e-3, momentum_dtype=momentum_dtype, weight_decay=0.01)
    for _ in range(3):
        for p in params_a:
            p.grad = torch.randn_like(p)
        opt_a.step()

    buf = io.BytesIO()
    torch.save(opt_a.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, weights_only=False)

    params_b = [torch.nn.Parameter(p.detach().clone()) for p in params_a]
    opt_b = AdaBelief(params_b, lr=1e-3, momentum_dtype=momentum_dtype, weight_decay=0.01)
    opt_b.load_state_dict(sd)

    for p_a, p_b in zip(params_a, params_b, strict=True):
        assert opt_b.state[p_b]["m"].dtype == opt_a.state[p_a]["m"].dtype

    torch.manual_seed(123)
    for _ in range(3):
        gs = [torch.randn_like(p) for p in params_a]
        for p, g in zip(params_a, gs, strict=True):
            p.grad = g.clone()
        for p, g in zip(params_b, gs, strict=True):
            p.grad = g.clone()
        opt_a.step()
        opt_b.step()
    for a, b in zip(params_a, params_b, strict=True):
        assert torch.equal(a, b), "resumed run must continue bit-exactly"


def test_conv_net_trains_no_nan():
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(4, 2, 3, padding=1),
    )
    opt = AdaBelief(net.parameters(), lr=1e-3)
    x = torch.randn(2, 3, 8, 8)
    y = torch.randn(2, 2, 8, 8)
    train_steps(net, opt, [(x, y)] * 5)
    for p in net.parameters():
        assert torch.isfinite(p).all()


def test_invalid_args_rejected():
    p = [torch.nn.Parameter(torch.randn(3))]
    with pytest.raises(ValueError):
        AdaBelief(p, lr=-1.0)
    with pytest.raises(ValueError):
        AdaBelief(p, betas=(1.0, 0.999))
    with pytest.raises(ValueError):
        AdaBelief(p, eps=-1e-8)
    with pytest.raises(ValueError):
        AdaBelief(p, weight_decay=-0.1)
    with pytest.raises(ValueError):
        AdaBelief(p, momentum_dtype="fp8")
    with pytest.raises(ValueError):
        AdaBelief(p, bf16_method="bogus")


def test_sparse_grad_rejected():
    p = torch.nn.Parameter(torch.randn(4))
    opt = AdaBelief([p], lr=1e-3)
    idx = torch.tensor([[0, 2]])
    val = torch.tensor([1.0, 1.0])
    p.grad = torch.sparse_coo_tensor(idx, val, (4,))
    with pytest.raises(RuntimeError):
        opt.step()
