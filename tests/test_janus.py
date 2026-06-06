"""Tests for Janus — AdaPNM (positive-negative momentum) on koptim's backend.

The numpy reference mirrors Janus's **non-factored (1-D) fp32 path**, which is the
one that matches kozistr's ``pytorch_optimizer.AdaPNM`` exactly (full per-coordinate
``v``; the factored 2-D path uses Adafactor's row/col approximation and is only
checked for self-consistency / foreach parity, not numpy parity).
"""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
import torch

from koptim import Janus

from .conftest import train_steps


def _ref_adapnm_1d(
    p: np.ndarray,
    grads: list[np.ndarray],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    beta0: float,
    eps: float,
    weight_decay: float,
    ams_bound: bool,
) -> np.ndarray:
    """Reference numpy AdaPNM over a sequence of grads (1-D, full v).

    Reproduces the kozistr AdaPNM math Janus's 1-D path mirrors:

    * first-moment decay is ``beta1**2``; bias correction uses ``beta1``.
    * two momenta, alternating which receives the gradient each step.
    * ``noise_norm = sqrt((1+beta0)^2 + beta0^2)``.
    * decoupled weight decay ``p *= 1 - lr*wd`` BEFORE the moment updates.
    * ``v_hat`` denom = ``(sqrt(max_v or v) + eps) / sqrt(1 - beta2^t)``.
    """
    p = p.copy()
    m_pos = np.zeros_like(p)
    m_neg = np.zeros_like(p)
    v = np.zeros_like(p)
    max_v = np.zeros_like(p)
    beta1_sq = beta1 * beta1
    noise_norm = math.sqrt((1.0 + beta0) ** 2 + beta0 ** 2)
    for t, g in enumerate(grads, start=1):
        if weight_decay != 0:
            p = p * (1.0 - lr * weight_decay)
        # alternation: odd step -> (pos, neg) = (m_pos, m_neg); even -> swapped.
        if t % 2 == 1:
            pos, neg = m_pos, m_neg
        else:
            pos, neg = m_neg, m_pos
        pos[...] = beta1_sq * pos + (1.0 - beta1_sq) * g
        v[...] = beta2 * v + (1.0 - beta2) * g * g
        if ams_bound:
            max_v[...] = np.maximum(max_v, v)
            de_nom = np.sqrt(max_v + 1e-15) + eps
        else:
            de_nom = np.sqrt(v + 1e-15) + eps
        de_nom = de_nom / math.sqrt(1.0 - beta2 ** t)
        bc1 = 1.0 - beta1 ** t
        pn = ((1.0 + beta0) * pos - beta0 * neg) / noise_norm
        p = p - (lr / bc1) * pn / de_nom
    return p


def test_construct_and_step():
    """Construct and take a step on tiny CPU tensors (2-D + 1-D + conv)."""
    params = [
        torch.nn.Parameter(torch.randn(8, 4)),
        torch.nn.Parameter(torch.randn(5)),
        torch.nn.Parameter(torch.randn(3, 2, 3, 3)),
    ]
    opt = Janus(params, lr=1e-3)
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()
    for p in params:
        assert torch.isfinite(p).all()


@pytest.mark.parametrize("ams_bound", [False, True])
@pytest.mark.parametrize("beta0", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("weight_decay", [0.0, 0.05])
def test_matches_numpy_reference_1d(ams_bound, beta0, weight_decay):
    """Janus's 1-D fp32 path matches the numpy AdaPNM reference (cautious off)."""
    torch.manual_seed(11)
    n = 13
    p0 = torch.randn(n)
    p = torch.nn.Parameter(p0.clone())
    opt = Janus(
        [p], lr=1e-2, betas=(0.9, 0.999), beta0=beta0, eps=1e-8,
        weight_decay=weight_decay, cautious=False, ams_bound=ams_bound,
        momentum_dtype="float32", foreach=False,
    )
    grads = [torch.randn(n) for _ in range(9)]
    for g in grads:
        p.grad = g.clone()
        opt.step()
    ref = _ref_adapnm_1d(
        p0.numpy(), [g.numpy() for g in grads],
        lr=1e-2, beta1=0.9, beta2=0.999, beta0=beta0, eps=1e-8,
        weight_decay=weight_decay, ams_bound=ams_bound,
    )
    np.testing.assert_allclose(p.detach().numpy(), ref, rtol=1e-5, atol=1e-6)


def test_positive_negative_alternation_and_mixing():
    """The two buffers alternate which receives the gradient; mixing uses noise_norm.

    After step 1 only ``m_pos`` is updated (the EMA of g1 with decay beta1^2); after
    step 2 the roles swap so ``m_neg`` gets g2 and ``m_pos`` is the stale buffer.
    """
    torch.manual_seed(0)
    n = 6
    p = torch.nn.Parameter(torch.randn(n))
    beta1 = 0.9
    opt = Janus(
        [p], lr=1e-2, betas=(beta1, 0.999), beta0=1.0, cautious=False,
        ams_bound=False, momentum_dtype="float32", foreach=False,
    )
    g1 = torch.randn(n)
    p.grad = g1.clone()
    opt.step()
    st = opt.state[p]
    b1sq = beta1 * beta1
    # step 1 (odd): m_pos got the gradient, m_neg untouched (still zero).
    torch.testing.assert_close(st["m_pos"], (1.0 - b1sq) * g1, rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(st["m_neg"]) == 0
    g2 = torch.randn(n)
    p.grad = g2.clone()
    opt.step()
    # step 2 (even): roles swap -> the *neg* buffer received g2; m_pos unchanged.
    torch.testing.assert_close(st["m_neg"], (1.0 - b1sq) * g2, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(st["m_pos"], (1.0 - b1sq) * g1, rtol=1e-5, atol=1e-6)


def test_noise_norm_constant():
    """The pos-neg mix is renormalized by sqrt((1+beta0)^2 + beta0^2)."""
    for beta0 in (0.0, 0.5, 1.0):
        nn = math.sqrt((1.0 + beta0) ** 2 + beta0 ** 2)
        if beta0 == 1.0:
            assert math.isclose(nn, math.sqrt(5.0))
        if beta0 == 0.0:
            assert math.isclose(nn, 1.0)


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
def test_momentum_dtype_variants_construct_and_step(momentum_dtype):
    params = [
        torch.nn.Parameter(torch.randn(8, 4)),
        torch.nn.Parameter(torch.randn(5)),
        torch.nn.Parameter(torch.randn(3, 2, 3, 3)),
    ]
    opt = Janus(params, lr=1e-3, momentum_dtype=momentum_dtype)
    for _ in range(3):
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
    for p in params:
        assert torch.isfinite(p).all()


def test_two_momenta_present():
    """Janus carries TWO momentum buffers (the PNM signature)."""
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = Janus([p], lr=1e-3, momentum_dtype="float32")
    p.grad = torch.randn_like(p)
    opt.step()
    st = opt.state[p]
    assert "m_pos" in st and "m_neg" in st
    assert st["m_pos"].shape == p.shape
    assert st["m_neg"].shape == p.shape


def test_int8_is_one_byte_per_param_per_momentum():
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = Janus([p], lr=1e-3, momentum_dtype="int8")
    p.grad = torch.randn_like(p)
    opt.step()
    st = opt.state[p]
    assert st["m_pos"].numel() == p.numel()
    assert st["m_neg"].numel() == p.numel()
    assert st["m_pos"].dtype == torch.int8


def test_4bit_is_half_byte_per_param_per_momentum():
    p = torch.nn.Parameter(torch.randn(8, 16))
    opt = Janus([p], lr=1e-3, momentum_dtype="4bit", momentum_4bit_block=16)
    p.grad = torch.randn_like(p)
    opt.step()
    st = opt.state[p]
    assert st["m_pos"].numel() == (p.numel() + 1) // 2  # two nibbles per byte
    assert st["m_pos"].dtype == torch.uint8


def test_cautious_masks_disagreeing_coords():
    """cautious=True zeroes coords where the final delta disagrees with the grad."""
    torch.manual_seed(0)
    n = 32
    p0 = torch.randn(n)

    def run(cautious: bool) -> torch.Tensor:
        p = torch.nn.Parameter(p0.clone())
        opt = Janus(
            [p], lr=1e-2, betas=(0.9, 0.999), beta0=1.0, cautious=cautious,
            ams_bound=False, momentum_dtype="float32", foreach=False,
        )
        torch.manual_seed(5)
        for _ in range(4):
            p.grad = torch.randn(n)
            opt.step()
        return p.detach().clone()

    # The two trajectories must differ (cautious is not a no-op with momentum on).
    assert not torch.allclose(run(True), run(False))


@pytest.mark.parametrize(
    "cfg",
    [
        dict(momentum_dtype="float32", betas=(0.9, 0.999), beta0=1.0, weight_decay=0.0, cautious=True),
        dict(momentum_dtype="int8", betas=(0.9, 0.999), beta0=1.0, weight_decay=0.02, cautious=True),
        dict(momentum_dtype="4bit", betas=(0.9, 0.99), beta0=0.5, weight_decay=0.01, cautious=False),
        dict(momentum_dtype="bfloat16", betas=(0.9, 0.999), beta0=1.0, weight_decay=0.0, cautious=True),
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
    oa = Janus(pa, lr=1e-3, foreach=True, bf16_method="none", **cfg)
    ob = Janus(pb, lr=1e-3, foreach=False, bf16_method="none", **cfg)
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
    oa = Janus(pa, lr=1e-3, momentum_dtype="int8", weight_decay=0.02, foreach=True, foreach_stack_budget=120)
    ob = Janus(pb, lr=1e-3, momentum_dtype="int8", weight_decay=0.02, foreach=False)
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
    """Janus should drive a tiny MLP's training loss down on a fixed batch."""
    x, y = random_batch
    opt = Janus(toy_mlp.parameters(), lr=3e-3)
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
    opt = Janus([p], lr=1e-3, bf16_method="stochastic_rounding")
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()
    assert torch.isfinite(p).all()


def test_kahan_runs():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 4, dtype=torch.bfloat16))
    opt = Janus([p], lr=1e-3, bf16_method="kahan")
    for _ in range(3):
        p.grad = torch.randn_like(p)
        opt.step()
    assert "shift" in opt.state[p]
    assert torch.isfinite(p).all()


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
def test_checkpoint_roundtrip_preserves_momentum_dtype(momentum_dtype):
    """state_dict round-trip preserves BOTH momenta's stored dtype and resumes bit-exactly.

    torch's default load_state_dict upcasts state to the param dtype (fp32); Janus
    overrides load_state_dict to restore the stored dtype.
    """
    torch.manual_seed(0)
    params_a = [torch.nn.Parameter(torch.randn(8, 4)), torch.nn.Parameter(torch.randn(5))]
    opt_a = Janus(params_a, lr=1e-3, momentum_dtype=momentum_dtype, weight_decay=0.01)
    for _ in range(3):
        for p in params_a:
            p.grad = torch.randn_like(p)
        opt_a.step()

    buf = io.BytesIO()
    torch.save(opt_a.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, weights_only=False)

    params_b = [torch.nn.Parameter(p.detach().clone()) for p in params_a]
    opt_b = Janus(params_b, lr=1e-3, momentum_dtype=momentum_dtype, weight_decay=0.01)
    opt_b.load_state_dict(sd)

    # stored dtypes preserved for both buffers.
    for p_a, p_b in zip(params_a, params_b, strict=True):
        for key in ("m_pos", "m_neg"):
            assert opt_b.state[p_b][key].dtype == opt_a.state[p_a][key].dtype

    # resumed run continues bit-exactly.
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
    opt = Janus(net.parameters(), lr=1e-3)
    x = torch.randn(2, 3, 8, 8)
    y = torch.randn(2, 2, 8, 8)
    train_steps(net, opt, [(x, y)] * 5)
    for p in net.parameters():
        assert torch.isfinite(p).all()


def test_invalid_args_rejected():
    p = [torch.nn.Parameter(torch.randn(3))]
    with pytest.raises(ValueError):
        Janus(p, lr=-1.0)
    with pytest.raises(ValueError):
        Janus(p, betas=(1.0, 0.999))
    with pytest.raises(ValueError):
        Janus(p, beta0=1.5)
    with pytest.raises(ValueError):
        Janus(p, eps=-1e-8)
    with pytest.raises(ValueError):
        Janus(p, weight_decay=-0.1)
    with pytest.raises(ValueError):
        Janus(p, momentum_dtype="fp8")
    with pytest.raises(ValueError):
        Janus(p, bf16_method="bogus")


def test_sparse_grad_rejected():
    p = torch.nn.Parameter(torch.randn(4))
    opt = Janus([p], lr=1e-3)
    idx = torch.tensor([[0, 2]])
    val = torch.tensor([1.0, 1.0])
    p.grad = torch.sparse_coo_tensor(idx, val, (4,))
    with pytest.raises(RuntimeError):
        opt.step()
