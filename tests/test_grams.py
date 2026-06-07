"""Tests for Grams — Adam's adaptive magnitude with the direction = sign(grad).

The numpy reference mirrors Grams' **non-factored (1-D) fp32 path**, which matches
the official ``Gunale0926/Grams`` exactly (full per-coordinate ``v``; the factored
2-D path uses Adafactor's row/col approximation and is only checked for
self-consistency / foreach parity, not numpy parity).

The defining property of Grams is exercised explicitly: the per-coordinate update
direction equals ``sign(current gradient)`` regardless of the momentum's sign.
"""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
import torch

from kaon import Grams

from .conftest import train_steps


def _ref_grams_1d(
    p: np.ndarray,
    grads: list[np.ndarray],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> np.ndarray:
    """Reference numpy Grams over a sequence of grads (1-D, full v).

    Reproduces the official ``Gunale0926/Grams`` math:

    * ``m  = beta1*m + (1-beta1)*g``  (standard Adam first moment)
    * ``v  = beta2*v + (1-beta2)*g^2``
    * ``bc1 = 1 - beta1^t`` ; ``bc2 = 1 - beta2^t``
    * ``step_size = lr * sqrt(bc2) / bc1`` ; ``denom = sqrt(v) + eps``
    * ``update = sign(g) * |m|`` (the ``/bc1`` debias of ``|m|`` is in step_size)
    * decoupled WD ``p -= lr*wd*p`` ; then ``p -= step_size * update / denom``
    """
    p = p.copy()
    m = np.zeros_like(p)
    v = np.zeros_like(p)
    for t, g in enumerate(grads, start=1):
        m[...] = beta1 * m + (1.0 - beta1) * g
        v[...] = beta2 * v + (1.0 - beta2) * g * g
        bc1 = 1.0 - beta1 ** t
        bc2_sq = math.sqrt(1.0 - beta2 ** t)
        step_size = lr * bc2_sq / bc1
        denom = np.sqrt(v) + eps
        update = np.sign(g) * np.abs(m)
        if weight_decay != 0:
            p = p * (1.0 - lr * weight_decay)
        p = p - step_size * update / denom
    return p


def test_construct_and_step():
    """Construct and take a step on tiny CPU tensors (2-D + 1-D + conv)."""
    params = [
        torch.nn.Parameter(torch.randn(8, 4)),
        torch.nn.Parameter(torch.randn(5)),
        torch.nn.Parameter(torch.randn(3, 2, 3, 3)),
    ]
    opt = Grams(params, lr=1e-3)
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()
    for p in params:
        assert torch.isfinite(p).all()


@pytest.mark.parametrize("weight_decay", [0.0, 0.05])
@pytest.mark.parametrize("betas", [(0.9, 0.999), (0.8, 0.99)])
def test_matches_numpy_reference_1d(betas, weight_decay):
    """Grams' 1-D fp32 path matches the numpy Grams reference (cautious + GC off)."""
    torch.manual_seed(11)
    beta1, beta2 = betas
    n = 13
    p0 = torch.randn(n)
    p = torch.nn.Parameter(p0.clone())
    opt = Grams(
        [p], lr=1e-2, betas=betas, eps=1e-6,
        weight_decay=weight_decay, cautious=False, gradient_centralization=False,
        momentum_dtype="float32", foreach=False,
    )
    grads = [torch.randn(n) for _ in range(9)]
    for g in grads:
        p.grad = g.clone()
        opt.step()
    ref = _ref_grams_1d(
        p0.numpy(), [g.numpy() for g in grads],
        lr=1e-2, beta1=beta1, beta2=beta2, eps=1e-6, weight_decay=weight_decay,
    )
    np.testing.assert_allclose(p.detach().numpy(), ref, rtol=1e-5, atol=1e-6)


def test_direction_is_sign_of_current_gradient():
    """THE defining property: the per-coordinate update direction == sign(grad),
    even where the momentum's sign disagrees with the current gradient.

    We engineer a step where momentum (built from a positive gradient) and the
    current gradient point opposite ways on every coordinate, then assert the
    parameter moved *against* the current gradient (i.e. update direction = sign(g))
    on every coordinate — which a momentum-signed Adam would NOT do.
    """
    torch.manual_seed(0)
    n = 64
    p0 = torch.randn(n)
    p = torch.nn.Parameter(p0.clone())
    # GC/cautious off, no WD: isolate the raw sign(g)*|m_hat|/denom step.
    opt = Grams(
        [p], lr=1e-2, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.0, cautious=False, gradient_centralization=False,
        momentum_dtype="float32", foreach=False,
    )
    # Step 1: a strongly positive gradient -> momentum becomes positive everywhere.
    g_pos = torch.full((n,), 1.0)
    p.grad = g_pos.clone()
    opt.step()
    assert (opt.state[p]["m"] > 0).all()  # momentum is positive on every coord

    # Step 2: flip the gradient negative everywhere. Momentum is still positive,
    # so momentum-signed Adam would step in +; Grams must step in -sign(g) = +?
    # update = sign(g)*|m_hat|/denom = negative; p -= step_size*update -> p moves UP.
    # The defining check: the actual per-coordinate weight *delta* sign equals
    # -sign(update) == -sign(g) ... so p moves OPPOSITE to g (descent on g).
    p_before = p.detach().clone()
    g_neg = torch.full((n,), -1.0)
    p.grad = g_neg.clone()
    opt.step()
    delta = p.detach() - p_before  # = -lr*update = -step_size*sign(g)*|m_hat|/denom
    # sign(delta) must be -sign(g) on every coordinate (descent in the g direction),
    # i.e. with g<0 the weight increases. Momentum sign (positive) is irrelevant.
    assert torch.all(delta > 0), "update direction must be -sign(g) (descent), not momentum's sign"
    # And explicitly: sign of the *update* (delta = -lr*update) equals sign(g).
    update_sign = -torch.sign(delta)
    assert torch.equal(update_sign, torch.sign(g_neg))


def test_direction_matches_sign_grad_on_mixed_step():
    """Direction == sign(grad) coordinate-wise on a random step where momentum and
    grad signs disagree on many coordinates (non-factored fp32 path)."""
    torch.manual_seed(3)
    n = 128
    p = torch.nn.Parameter(torch.zeros(n))
    opt = Grams(
        [p], lr=1e-2, betas=(0.9, 0.999), eps=1e-12,
        weight_decay=0.0, cautious=False, gradient_centralization=False,
        momentum_dtype="float32", foreach=False,
    )
    # Build momentum from a fixed-sign gradient.
    p.grad = torch.ones(n)
    opt.step()
    m = opt.state[p]["m"].clone()
    g = torch.randn(n)
    # Ensure plenty of disagreement between momentum sign and current grad sign.
    disagree = (torch.sign(g) != torch.sign(m))
    assert disagree.sum() > n // 4
    p_before = p.detach().clone()
    p.grad = g.clone()
    opt.step()
    delta = p.detach() - p_before
    # delta = -lr_eff * sign(g) * |m_hat|/denom ; |m_hat|/denom > 0 so sign(delta) == -sign(g).
    nz = g != 0
    assert torch.equal(torch.sign(delta[nz]), -torch.sign(g[nz]))


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
def test_momentum_dtype_variants_construct_and_step(momentum_dtype):
    params = [
        torch.nn.Parameter(torch.randn(8, 4)),
        torch.nn.Parameter(torch.randn(5)),
        torch.nn.Parameter(torch.randn(3, 2, 3, 3)),
    ]
    opt = Grams(params, lr=1e-3, momentum_dtype=momentum_dtype)
    for _ in range(3):
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
    for p in params:
        assert torch.isfinite(p).all()


def test_single_momentum_present():
    """Grams carries ONE first-moment buffer (unlike AdaPNM's two)."""
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = Grams([p], lr=1e-3, momentum_dtype="float32")
    p.grad = torch.randn_like(p)
    opt.step()
    st = opt.state[p]
    assert "m" in st and "m_pos" not in st and "m_neg" not in st
    assert st["m"].shape == p.shape


def test_int8_is_one_byte_per_param():
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = Grams([p], lr=1e-3, momentum_dtype="int8")
    p.grad = torch.randn_like(p)
    opt.step()
    st = opt.state[p]
    assert st["m"].numel() == p.numel()
    assert st["m"].dtype == torch.int8


def test_4bit_is_half_byte_per_param():
    p = torch.nn.Parameter(torch.randn(8, 16))
    opt = Grams([p], lr=1e-3, momentum_dtype="4bit", momentum_4bit_block=16)
    p.grad = torch.randn_like(p)
    opt.step()
    st = opt.state[p]
    assert st["m"].numel() == (p.numel() + 1) // 2  # two nibbles per byte
    assert st["m"].dtype == torch.uint8


def test_cautious_is_noop_given_sign_g_direction():
    """Because Grams' direction is already sign(g), cautious masking is a no-op:
    the two trajectories (cautious on/off) must be identical."""
    torch.manual_seed(0)
    n = 32
    p0 = torch.randn(n)

    def run(cautious: bool) -> torch.Tensor:
        p = torch.nn.Parameter(p0.clone())
        opt = Grams(
            [p], lr=1e-2, betas=(0.9, 0.999), cautious=cautious,
            gradient_centralization=False, momentum_dtype="float32", foreach=False,
        )
        torch.manual_seed(5)
        for _ in range(4):
            p.grad = torch.randn(n)
            opt.step()
        return p.detach().clone()

    # sign(g) direction => delta*g >= 0 on every coord => mask all-ones => no-op.
    assert torch.equal(run(True), run(False))


@pytest.mark.parametrize(
    "cfg",
    [
        dict(momentum_dtype="float32", betas=(0.9, 0.999), weight_decay=0.0, cautious=True),
        dict(momentum_dtype="bfloat16", betas=(0.9, 0.999), weight_decay=0.0, cautious=True),
        dict(momentum_dtype="int8", betas=(0.9, 0.999), weight_decay=0.02, cautious=True),
        dict(momentum_dtype="4bit", betas=(0.9, 0.99), weight_decay=0.01, cautious=False),
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
    oa = Grams(pa, lr=1e-3, foreach=True, bf16_method="none", **cfg)
    ob = Grams(pb, lr=1e-3, foreach=False, bf16_method="none", **cfg)
    torch.manual_seed(7)
    for _ in range(7):
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
    oa = Grams(pa, lr=1e-3, momentum_dtype="int8", weight_decay=0.02, foreach=True, foreach_stack_budget=120)
    ob = Grams(pb, lr=1e-3, momentum_dtype="int8", weight_decay=0.02, foreach=False)
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
    """Grams should drive a tiny MLP's training loss down on a fixed batch."""
    x, y = random_batch
    opt = Grams(toy_mlp.parameters(), lr=3e-3)
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
    opt = Grams([p], lr=1e-3, bf16_method="stochastic_rounding")
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()
    assert torch.isfinite(p).all()


def test_kahan_runs():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 4, dtype=torch.bfloat16))
    opt = Grams([p], lr=1e-3, bf16_method="kahan")
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
    opt_a = Grams(params_a, lr=1e-3, momentum_dtype=momentum_dtype, weight_decay=0.01)
    for _ in range(3):
        for p in params_a:
            p.grad = torch.randn_like(p)
        opt_a.step()

    buf = io.BytesIO()
    torch.save(opt_a.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, weights_only=False)

    params_b = [torch.nn.Parameter(p.detach().clone()) for p in params_a]
    opt_b = Grams(params_b, lr=1e-3, momentum_dtype=momentum_dtype, weight_decay=0.01)
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
    opt = Grams(net.parameters(), lr=1e-3)
    x = torch.randn(2, 3, 8, 8)
    y = torch.randn(2, 2, 8, 8)
    train_steps(net, opt, [(x, y)] * 5)
    for p in net.parameters():
        assert torch.isfinite(p).all()


def test_invalid_args_rejected():
    p = [torch.nn.Parameter(torch.randn(3))]
    with pytest.raises(ValueError):
        Grams(p, lr=-1.0)
    with pytest.raises(ValueError):
        Grams(p, betas=(1.0, 0.999))
    with pytest.raises(ValueError):
        Grams(p, eps=-1e-8)
    with pytest.raises(ValueError):
        Grams(p, weight_decay=-0.1)
    with pytest.raises(ValueError):
        Grams(p, momentum_dtype="fp8")
    with pytest.raises(ValueError):
        Grams(p, bf16_method="bogus")


def test_sparse_grad_rejected():
    p = torch.nn.Parameter(torch.randn(4))
    opt = Grams([p], lr=1e-3)
    idx = torch.tensor([[0, 2]])
    val = torch.tensor([1.0, 1.0])
    p.grad = torch.sparse_coo_tensor(idx, val, (4,))
    with pytest.raises(RuntimeError):
        opt.step()
