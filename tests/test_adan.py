"""Tests for Adan — Adaptive Nesterov Momentum on kaon's backend.

The numpy reference mirrors Adan's **non-factored (1-D) fp32 path**, which is the
one that matches the official ``sail-sg/Adan`` exactly (full per-coordinate ``n``;
the factored ``ndim >= 2`` path uses Adafactor's row/col approximation and is only
checked for self-consistency / foreach parity, not numpy parity).

Betas are the *retention* factors as the official stores them, e.g.
``betas=(0.98, 0.92, 0.99)``: ``m = beta1*m + (1-beta1)*g``, etc.
"""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
import torch

from kaon import Adan

from .conftest import train_steps


def _ref_adan_1d(
    p: np.ndarray,
    grads: list[np.ndarray],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    beta3: float,
    eps: float,
    weight_decay: float,
    no_prox: bool,
) -> np.ndarray:
    """Reference numpy Adan over a sequence of grads (1-D, full n).

    Reproduces the official ``sail-sg/Adan`` ``_single_tensor_adan`` update:

    * ``g_prev`` initialised to ``g_1`` so the t=1 difference is exactly zero.
    * ``m = beta1*m + (1-beta1)*g`` ; ``diff_ema = beta2*diff_ema + (1-beta2)*(g-g_prev)``.
    * look-ahead ``u = g + beta2*(g - g_prev)`` ; ``n = beta3*n + (1-beta3)*u^2``.
    * ``denom = sqrt(n)/sqrt(1-beta3^t) + eps``.
    * ``step_size = lr/(1-beta1^t)`` ; ``step_size_diff = lr*beta2/(1-beta2^t)``.
    * prox (``no_prox=False``): step then ``p /= 1 + lr*wd``.
    * no_prox (``True``): ``p *= 1 - lr*wd`` then step.
    """
    p = p.copy()
    m = np.zeros_like(p)
    diff_ema = np.zeros_like(p)
    n = np.zeros_like(p)
    g_prev = None
    for t, g in enumerate(grads, start=1):
        if g_prev is None:
            g_prev = g.copy()  # t=1: previous grad := current grad -> diff = 0
        diff = g - g_prev
        m[...] = beta1 * m + (1.0 - beta1) * g
        diff_ema[...] = beta2 * diff_ema + (1.0 - beta2) * diff
        u = g + beta2 * diff
        n[...] = beta3 * n + (1.0 - beta3) * u * u
        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2 ** t
        bc3_sqrt = math.sqrt(1.0 - beta3 ** t)
        denom = np.sqrt(n) / bc3_sqrt + eps
        step_size = lr / bc1
        step_size_diff = lr * beta2 / bc2
        delta = step_size * m / denom + step_size_diff * diff_ema / denom
        if no_prox:
            if weight_decay != 0:
                p = p * (1.0 - lr * weight_decay)
            p = p - delta
        else:
            p = p - delta
            if weight_decay != 0:
                p = p / (1.0 + lr * weight_decay)
        g_prev = g.copy()
    return p


def test_construct_and_step():
    """Construct and take a step on tiny CPU tensors (2-D + 1-D + conv)."""
    params = [
        torch.nn.Parameter(torch.randn(8, 4)),
        torch.nn.Parameter(torch.randn(5)),
        torch.nn.Parameter(torch.randn(3, 2, 3, 3)),
    ]
    opt = Adan(params, lr=1e-3)
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()
    for p in params:
        assert torch.isfinite(p).all()


@pytest.mark.parametrize("no_prox", [False, True])
@pytest.mark.parametrize("weight_decay", [0.0, 0.05])
def test_matches_numpy_reference_1d(no_prox, weight_decay):
    """Adan's 1-D fp32 path matches the numpy Adan reference (cautious + GC off)."""
    torch.manual_seed(11)
    n = 13
    betas = (0.98, 0.92, 0.99)
    p0 = torch.randn(n)
    p = torch.nn.Parameter(p0.clone())
    opt = Adan(
        [p], lr=1e-2, betas=betas, eps=1e-8, weight_decay=weight_decay,
        no_prox=no_prox, cautious=False, gradient_centralization=False,
        momentum_dtype="float32", foreach=False,
    )
    grads = [torch.randn(n) for _ in range(9)]
    for g in grads:
        p.grad = g.clone()
        opt.step()
    ref = _ref_adan_1d(
        p0.numpy(), [g.numpy() for g in grads],
        lr=1e-2, beta1=betas[0], beta2=betas[1], beta3=betas[2], eps=1e-8,
        weight_decay=weight_decay, no_prox=no_prox,
    )
    np.testing.assert_allclose(p.detach().numpy(), ref, rtol=1e-5, atol=1e-6)


def test_t1_difference_is_zero():
    """t=1: g_prev is seeded to g_1, so the gradient difference is exactly zero.

    Consequences at step 1 (matching the official ``neg_pre_grad = -grad`` init):
    * ``diff_ema`` stays at zero (it EMAs in a zero difference).
    * ``n`` is the second moment of ``u = g + beta2*0 = g`` only (no look-ahead yet).
    """
    torch.manual_seed(0)
    n = 7
    beta3 = 0.99
    p = torch.nn.Parameter(torch.randn(n))
    opt = Adan(
        [p], lr=1e-2, betas=(0.98, 0.92, beta3), cautious=False,
        gradient_centralization=False, momentum_dtype="float32", foreach=False,
    )
    g1 = torch.randn(n)
    p.grad = g1.clone()
    opt.step()
    st = opt.state[p]
    # diff_ema EMAs in a zero difference at t=1 -> remains all zeros.
    assert torch.count_nonzero(st["diff"]) == 0
    # n is (1-beta3) * g1^2 (u == g1 at t=1).
    torch.testing.assert_close(st["n"], (1.0 - beta3) * g1 * g1, rtol=1e-5, atol=1e-6)
    # g_prev was stored as g1.
    torch.testing.assert_close(st["g_prev"], g1, rtol=1e-5, atol=1e-6)


def test_three_buffers_present():
    """Adan carries THREE codec buffers: m, diff, g_prev (its signature)."""
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = Adan([p], lr=1e-3, momentum_dtype="float32")
    p.grad = torch.randn_like(p)
    opt.step()
    st = opt.state[p]
    for key in ("m", "diff", "g_prev"):
        assert key in st, key
        assert st[key].shape == p.shape


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
def test_momentum_dtype_variants_construct_and_step(momentum_dtype):
    params = [
        torch.nn.Parameter(torch.randn(8, 4)),
        torch.nn.Parameter(torch.randn(5)),
        torch.nn.Parameter(torch.randn(3, 2, 3, 3)),
    ]
    opt = Adan(params, lr=1e-3, momentum_dtype=momentum_dtype)
    for _ in range(5):
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
    for p in params:
        assert torch.isfinite(p).all()


def test_int8_is_one_byte_per_param_per_buffer():
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = Adan([p], lr=1e-3, momentum_dtype="int8")
    p.grad = torch.randn_like(p)
    opt.step()
    st = opt.state[p]
    for key in ("m", "diff", "g_prev"):
        assert st[key].numel() == p.numel()
        assert st[key].dtype == torch.int8


def test_4bit_is_half_byte_per_param_per_buffer():
    p = torch.nn.Parameter(torch.randn(8, 16))
    opt = Adan([p], lr=1e-3, momentum_dtype="4bit", momentum_4bit_block=16)
    p.grad = torch.randn_like(p)
    opt.step()
    st = opt.state[p]
    for key in ("m", "diff", "g_prev"):
        assert st[key].numel() == (p.numel() + 1) // 2  # two nibbles per byte
        assert st[key].dtype == torch.uint8


@pytest.mark.parametrize(
    "cfg",
    [
        dict(momentum_dtype="float32", betas=(0.98, 0.92, 0.99), weight_decay=0.0, no_prox=False, cautious=True),
        dict(momentum_dtype="bfloat16", betas=(0.98, 0.92, 0.99), weight_decay=0.0, no_prox=False, cautious=True),
        dict(momentum_dtype="int8", betas=(0.98, 0.92, 0.99), weight_decay=0.02, no_prox=False, cautious=True),
        dict(momentum_dtype="int8", betas=(0.9, 0.9, 0.999), weight_decay=0.02, no_prox=True, cautious=False),
        dict(momentum_dtype="bfloat16", betas=(0.95, 0.9, 0.99), weight_decay=0.01, no_prox=True, cautious=True),
    ],
)
def test_foreach_matches_per_param(cfg):
    """foreach=True is element-for-element equal to the per-parameter path (fp32 weights).

    Runs >= 5 steps so the g_prev / gradient-difference machinery is exercised
    well past the t=1 edge case.
    """
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
    oa = Adan(pa, lr=1e-3, foreach=True, bf16_method="none", **cfg)
    ob = Adan(pb, lr=1e-3, foreach=False, bf16_method="none", **cfg)
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
    oa = Adan(pa, lr=1e-3, momentum_dtype="int8", weight_decay=0.02, foreach=True, foreach_stack_budget=120)
    ob = Adan(pb, lr=1e-3, momentum_dtype="int8", weight_decay=0.02, foreach=False)
    torch.manual_seed(3)
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


def test_max_grad_norm_clips():
    """max_grad_norm > 0 scales the global gradient down before the step."""
    torch.manual_seed(0)
    p_clip = torch.nn.Parameter(torch.zeros(4))
    p_free = torch.nn.Parameter(torch.zeros(4))
    g = torch.tensor([10.0, 0.0, 0.0, 0.0])
    oc = Adan([p_clip], lr=1e-1, max_grad_norm=1.0, cautious=False,
              gradient_centralization=False, momentum_dtype="float32", foreach=False)
    of = Adan([p_free], lr=1e-1, max_grad_norm=0.0, cautious=False,
              gradient_centralization=False, momentum_dtype="float32", foreach=False)
    p_clip.grad = g.clone()
    p_free.grad = g.clone()
    oc.step()
    of.step()
    # The clipped run scaled the grad, but the t=1 update is sign-only (m/sqrt(n)
    # ~ +/-1 regardless of scale), so check the stored g_prev reflects the clip.
    assert torch.abs(oc.state[p_clip]["g_prev"]).max() < torch.abs(of.state[p_free]["g_prev"]).max()


def test_cautious_masks_disagreeing_coords():
    """cautious=True changes the trajectory (not a no-op with momentum on)."""
    torch.manual_seed(0)
    n = 32
    p0 = torch.randn(n)

    def run(cautious: bool) -> torch.Tensor:
        p = torch.nn.Parameter(p0.clone())
        opt = Adan(
            [p], lr=1e-2, cautious=cautious, gradient_centralization=False,
            momentum_dtype="float32", foreach=False,
        )
        torch.manual_seed(5)
        for _ in range(5):
            p.grad = torch.randn(n)
            opt.step()
        return p.detach().clone()

    assert not torch.allclose(run(True), run(False))


def test_overfits_regression(toy_mlp, random_batch):
    """Adan should drive a tiny MLP's training loss down on a fixed batch."""
    x, y = random_batch
    opt = Adan(toy_mlp.parameters(), lr=1e-2)
    losses = []
    for _ in range(120):
        opt.zero_grad()
        loss = (toy_mlp(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5


def test_bf16_weights_train_no_nan():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, 8, dtype=torch.bfloat16))
    opt = Adan([p], lr=1e-3, bf16_method="stochastic_rounding")
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()
    assert torch.isfinite(p).all()


def test_kahan_runs():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 4, dtype=torch.bfloat16))
    opt = Adan([p], lr=1e-3, bf16_method="kahan")
    for _ in range(3):
        p.grad = torch.randn_like(p)
        opt.step()
    assert "shift" in opt.state[p]
    assert torch.isfinite(p).all()


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
def test_checkpoint_roundtrip_preserves_momentum_dtype(momentum_dtype):
    """state_dict round-trip preserves all three buffers' stored dtype and resumes bit-exactly."""
    torch.manual_seed(0)
    params_a = [torch.nn.Parameter(torch.randn(8, 4)), torch.nn.Parameter(torch.randn(5))]
    opt_a = Adan(params_a, lr=1e-3, momentum_dtype=momentum_dtype, weight_decay=0.01)
    for _ in range(4):
        for p in params_a:
            p.grad = torch.randn_like(p)
        opt_a.step()

    buf = io.BytesIO()
    torch.save(opt_a.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, weights_only=False)

    params_b = [torch.nn.Parameter(p.detach().clone()) for p in params_a]
    opt_b = Adan(params_b, lr=1e-3, momentum_dtype=momentum_dtype, weight_decay=0.01)
    opt_b.load_state_dict(sd)

    for p_a, p_b in zip(params_a, params_b, strict=True):
        for key in ("m", "diff", "g_prev"):
            assert opt_b.state[p_b][key].dtype == opt_a.state[p_a][key].dtype

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
    opt = Adan(net.parameters(), lr=1e-3)
    x = torch.randn(2, 3, 8, 8)
    y = torch.randn(2, 2, 8, 8)
    train_steps(net, opt, [(x, y)] * 5)
    for p in net.parameters():
        assert torch.isfinite(p).all()


def test_invalid_args_rejected():
    p = [torch.nn.Parameter(torch.randn(3))]
    with pytest.raises(ValueError):
        Adan(p, lr=-1.0)
    with pytest.raises(ValueError):
        Adan(p, betas=(1.0, 0.92, 0.99))
    with pytest.raises(ValueError):
        Adan(p, betas=(0.98, 0.92, 1.0))
    with pytest.raises(ValueError):
        Adan(p, eps=-1e-8)
    with pytest.raises(ValueError):
        Adan(p, weight_decay=-0.1)
    with pytest.raises(ValueError):
        Adan(p, max_grad_norm=-1.0)
    with pytest.raises(ValueError):
        Adan(p, momentum_dtype="fp8")
    with pytest.raises(ValueError):
        Adan(p, bf16_method="bogus")


def test_sparse_grad_rejected():
    p = torch.nn.Parameter(torch.randn(4))
    opt = Adan([p], lr=1e-3)
    p.grad = torch.sparse_coo_tensor(
        torch.tensor([[0, 2]]), torch.tensor([1.0, 2.0]), (4,)
    )
    with pytest.raises(RuntimeError):
        opt.step()
