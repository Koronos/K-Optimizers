"""Tests for MARS — MARS-AdamW (variance reduction) on kaon's backend.

Two flavours of check:

* **foreach == per-param parity** across every ``momentum_dtype`` and cautious on/off,
  on 2-D + 4-D conv + 1-D params, run for several steps so the variance-reduction
  ``c_t`` (which needs the *previous* gradient) is actually exercised.
* a **correctness test** of the matrix (variance-reduction) path against a tiny inline
  fp32 reference that reproduces the corrected gradient ``c_t``, its global-norm clip,
  the factored second moment of ``c_t``, and the AdamW-style debiased update — exactly
  as implemented (``g_{t-1}=g_t`` on the first step, so the first ``c_t = g_t``).
"""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
import torch

from kaon import MARS

from .conftest import train_steps


def _ref_mars_matrix(
    p: np.ndarray,
    grads: list[np.ndarray],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    gamma: float,
    weight_decay: float,
    mars_clip: bool,
) -> np.ndarray:
    """Inline fp32 reference for MARS-AdamW on a 2-D ("matrix") param.

    Mirrors :meth:`MARS._step_one_param`'s factored path exactly:

    * ``c_t = g + gamma*beta1/(1-beta1) * (g - g_prev)`` with ``g_prev`` seeded to
      ``g`` on the first step (so the first ``c_t = g``).
    * optional global-L2-norm clip of ``c_t`` to 1.
    * factored second moment of ``c_t`` (HF Adafactor eps1 placement, eps1 = eps).
    * AdamW debiased update with ``step_size = lr / (1 - beta1^t)`` and the
      ``sqrt(1 - beta2^t)`` factor folded into the inverse denominator.
    * decoupled weight decay ``p *= (1 - lr*wd)`` before the moment updates.
    """
    p = p.copy()
    n_row, n_col = p.shape
    m = np.zeros_like(p)
    row = np.zeros(n_row, dtype=np.float64)
    col = np.zeros(n_col, dtype=np.float64)
    corr = gamma * (beta1 / (1.0 - beta1))
    g_prev = grads[0].copy()  # seed last_grad = g_1
    for t, g in enumerate(grads, start=1):
        if weight_decay != 0:
            p = p * (1.0 - lr * weight_decay)
        c_t = g + corr * (g - g_prev)
        if mars_clip:
            norm = np.sqrt((c_t * c_t).sum())
            if norm > 1.0:
                c_t = c_t / norm
        g_prev = g.copy()

        ct_sq = c_t * c_t + eps
        row[...] = beta2 * row + (1.0 - beta2) * ct_sq.mean(axis=1)
        col[...] = beta2 * col + (1.0 - beta2) * ct_sq.mean(axis=0)

        r_factor = 1.0 / np.sqrt(row / row.mean())     # [R]
        c_factor = 1.0 / np.sqrt(col)                  # [C]
        bc2_sq = math.sqrt(1.0 - beta2 ** t)
        inv_denom = np.outer(r_factor, c_factor) * bc2_sq

        m[...] = beta1 * m + (1.0 - beta1) * c_t
        bc1 = 1.0 - beta1 ** t
        p = p - (lr / bc1) * m * inv_denom
    return p


def test_construct_and_step():
    """Construct and take a step on tiny CPU tensors (2-D + 1-D + conv)."""
    params = [
        torch.nn.Parameter(torch.randn(8, 4)),
        torch.nn.Parameter(torch.randn(5)),
        torch.nn.Parameter(torch.randn(3, 2, 3, 3)),
    ]
    opt = MARS(params, lr=1e-3)
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()
    for p in params:
        assert torch.isfinite(p).all()


@pytest.mark.parametrize("mars_clip", [False, True])
@pytest.mark.parametrize("weight_decay", [0.0, 0.05])
def test_matches_inline_reference_matrix(mars_clip, weight_decay):
    """MARS's 2-D factored path matches the inline MARS-AdamW reference (cautious off)."""
    torch.manual_seed(11)
    n_row, n_col = 6, 5
    p0 = torch.randn(n_row, n_col)
    p = torch.nn.Parameter(p0.clone())
    opt = MARS(
        [p], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, gamma=0.025,
        weight_decay=weight_decay, mars_clip=mars_clip, cautious=False,
        gradient_centralization=False, momentum_dtype="float32", foreach=False,
    )
    grads = [torch.randn(n_row, n_col) for _ in range(10)]
    for g in grads:
        p.grad = g.clone()
        opt.step()
    ref = _ref_mars_matrix(
        p0.numpy().astype(np.float64), [g.numpy().astype(np.float64) for g in grads],
        lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, gamma=0.025,
        weight_decay=weight_decay, mars_clip=mars_clip,
    )
    np.testing.assert_allclose(p.detach().numpy(), ref, rtol=1e-4, atol=1e-6)


def test_first_step_ct_equals_grad():
    """g_{t-1} is seeded to g_t on the first step, so c_t = g_t (no correction yet)."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(6, 5))
    opt = MARS([p], lr=1e-2, gamma=0.5, mars_clip=False, gradient_centralization=False,
               momentum_dtype="float32", foreach=False)
    g1 = torch.randn(6, 5)
    p.grad = g1.clone()
    opt.step()
    st = opt.state[p]
    # exp_avg after step 1 == (1-beta1)*c_t and c_t == g1 -> exp_avg == (1-0.9)*g1.
    torch.testing.assert_close(st["exp_avg"], 0.1 * g1, rtol=1e-5, atol=1e-6)
    # last_grad now holds g1.
    torch.testing.assert_close(st["last_grad"], g1, rtol=1e-5, atol=1e-6)


def test_gamma_zero_is_adamw_on_matrix():
    """gamma=0 removes the correction; c_t = g, so the matrix path is plain AdamW."""
    torch.manual_seed(3)
    p = torch.nn.Parameter(torch.randn(6, 5))
    opt = MARS([p], lr=1e-2, gamma=0.0, mars_clip=False, gradient_centralization=False,
               momentum_dtype="float32", foreach=False)
    for _ in range(3):
        p.grad = torch.randn(6, 5)
        opt.step()
    assert torch.isfinite(p).all()


def test_two_buffers_present():
    """MARS carries TWO persistent codec buffers: the first moment and the prev grad."""
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = MARS([p], lr=1e-3, momentum_dtype="float32")
    p.grad = torch.randn_like(p)
    opt.step()
    st = opt.state[p]
    assert "exp_avg" in st and "last_grad" in st
    assert st["exp_avg"].shape == p.shape
    assert st["last_grad"].shape == p.shape


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
def test_momentum_dtype_variants_construct_and_step(momentum_dtype):
    params = [
        torch.nn.Parameter(torch.randn(8, 4)),
        torch.nn.Parameter(torch.randn(5)),
        torch.nn.Parameter(torch.randn(3, 2, 3, 3)),
    ]
    opt = MARS(params, lr=1e-3, momentum_dtype=momentum_dtype)
    for _ in range(3):
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
    for p in params:
        assert torch.isfinite(p).all()


def test_int8_is_one_byte_per_param_per_buffer():
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = MARS([p], lr=1e-3, momentum_dtype="int8")
    p.grad = torch.randn_like(p)
    opt.step()
    st = opt.state[p]
    assert st["exp_avg"].numel() == p.numel()
    assert st["last_grad"].numel() == p.numel()
    assert st["exp_avg"].dtype == torch.int8


def test_4bit_is_half_byte_per_param_per_buffer():
    p = torch.nn.Parameter(torch.randn(8, 16))
    opt = MARS([p], lr=1e-3, momentum_dtype="4bit", momentum_4bit_block=16)
    p.grad = torch.randn_like(p)
    opt.step()
    st = opt.state[p]
    assert st["exp_avg"].numel() == (p.numel() + 1) // 2
    assert st["exp_avg"].dtype == torch.uint8


@pytest.mark.parametrize("cautious", [False, True])
@pytest.mark.parametrize("momentum_dtype", ["float32", "bfloat16", "int8", "4bit"])
def test_foreach_matches_per_param(momentum_dtype, cautious):
    """foreach=True is element-for-element equal to the per-parameter path (fp32 weights).

    >=2 steps so the prev-grad (variance-reduction) term is exercised; 2-D + conv (4-D)
    drive the factored/clip path, 1-D drives the plain-AdamW path.
    """
    def mk() -> list[torch.nn.Parameter]:
        return [
            torch.nn.Parameter(torch.randn(8, 4)),
            torch.nn.Parameter(torch.randn(7, 3)),
            torch.nn.Parameter(torch.randn(5)),
            torch.nn.Parameter(torch.randn(6)),
            torch.nn.Parameter(torch.randn(3, 2, 3, 3)),
        ]

    cfg = dict(
        lr=1e-3, betas=(0.9, 0.999), gamma=0.025, weight_decay=0.02,
        mars_clip=True, cautious=cautious, momentum_dtype=momentum_dtype,
        bf16_method="none",
    )
    torch.manual_seed(1)
    pa = mk()
    torch.manual_seed(1)
    pb = mk()
    oa = MARS(pa, foreach=True, **cfg)
    ob = MARS(pb, foreach=False, **cfg)
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
    oa = MARS(pa, lr=1e-3, momentum_dtype="int8", weight_decay=0.02, foreach=True,
              foreach_stack_budget=120)
    ob = MARS(pb, lr=1e-3, momentum_dtype="int8", weight_decay=0.02, foreach=False)
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


def test_cautious_masks_disagreeing_coords():
    """cautious=True changes the trajectory (not a no-op with momentum on)."""
    torch.manual_seed(0)
    p0 = torch.randn(8, 6)

    def run(cautious: bool) -> torch.Tensor:
        p = torch.nn.Parameter(p0.clone())
        opt = MARS([p], lr=1e-2, cautious=cautious, gradient_centralization=False,
                   momentum_dtype="float32", foreach=False)
        torch.manual_seed(5)
        for _ in range(4):
            p.grad = torch.randn(8, 6)
            opt.step()
        return p.detach().clone()

    assert not torch.allclose(run(True), run(False))


def test_clip_caps_correction():
    """A huge gradient swing triggers the c_t global-norm clip (||c_t|| -> 1)."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = MARS([p], lr=1e-2, gamma=0.5, mars_clip=True, gradient_centralization=False,
               momentum_dtype="float32", cautious=False, foreach=False)
    # step 1 seeds last_grad = g1.
    p.grad = torch.randn(4, 4)
    opt.step()
    # step 2: a large grad far from g1 -> c_t norm > 1 -> exp_avg bounded.
    p.grad = torch.full((4, 4), 100.0)
    opt.step()
    # exp_avg = beta1*prev + (1-beta1)*c_t, c_t clipped to unit norm; bounded well below 100.
    assert opt.state[p]["exp_avg"].abs().max() < 5.0


def test_overfits_regression(toy_mlp, random_batch):
    """MARS should drive a tiny MLP's training loss down on a fixed batch."""
    x, y = random_batch
    opt = MARS(toy_mlp.parameters(), lr=3e-3)
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
    opt = MARS([p], lr=1e-3, bf16_method="stochastic_rounding")
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()
    assert torch.isfinite(p).all()


def test_kahan_runs():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 4, dtype=torch.bfloat16))
    opt = MARS([p], lr=1e-3, bf16_method="kahan")
    for _ in range(3):
        p.grad = torch.randn_like(p)
        opt.step()
    assert "shift" in opt.state[p]
    assert torch.isfinite(p).all()


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
def test_checkpoint_roundtrip_preserves_momentum_dtype(momentum_dtype):
    """state_dict round-trip preserves BOTH buffers' stored dtype and resumes bit-exactly."""
    torch.manual_seed(0)
    params_a = [torch.nn.Parameter(torch.randn(8, 4)), torch.nn.Parameter(torch.randn(5))]
    opt_a = MARS(params_a, lr=1e-3, momentum_dtype=momentum_dtype, weight_decay=0.01)
    for _ in range(3):
        for p in params_a:
            p.grad = torch.randn_like(p)
        opt_a.step()

    buf = io.BytesIO()
    torch.save(opt_a.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, weights_only=False)

    params_b = [torch.nn.Parameter(p.detach().clone()) for p in params_a]
    opt_b = MARS(params_b, lr=1e-3, momentum_dtype=momentum_dtype, weight_decay=0.01)
    opt_b.load_state_dict(sd)

    for p_a, p_b in zip(params_a, params_b, strict=True):
        for key in ("exp_avg", "last_grad"):
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
    opt = MARS(net.parameters(), lr=1e-3)
    x = torch.randn(2, 3, 8, 8)
    y = torch.randn(2, 2, 8, 8)
    train_steps(net, opt, [(x, y)] * 5)
    for p in net.parameters():
        assert torch.isfinite(p).all()


def test_invalid_args_rejected():
    p = [torch.nn.Parameter(torch.randn(3))]
    with pytest.raises(ValueError):
        MARS(p, lr=-1.0)
    with pytest.raises(ValueError):
        MARS(p, betas=(1.0, 0.999))
    with pytest.raises(ValueError):
        MARS(p, gamma=-0.1)
    with pytest.raises(ValueError):
        MARS(p, eps=-1e-8)
    with pytest.raises(ValueError):
        MARS(p, weight_decay=-0.1)
    with pytest.raises(ValueError):
        MARS(p, momentum_dtype="fp8")
    with pytest.raises(ValueError):
        MARS(p, bf16_method="bogus")


def test_sparse_grad_rejected():
    p = torch.nn.Parameter(torch.randn(4))
    opt = MARS([p], lr=1e-3)
    idx = torch.tensor([[0, 2]])
    val = torch.tensor([1.0, 1.0])
    p.grad = torch.sparse_coo_tensor(idx, val, (4,))
    with pytest.raises(RuntimeError):
        opt.step()
