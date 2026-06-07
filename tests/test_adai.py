"""Tests for Adai — Adaptive Inertia (Xie et al. 2022) on kaon's backend.

The numpy reference mirrors Adai's **1-D fp32 path**, which matches the official
zeke-xie/Adai ``adai.py`` exactly (full per-coordinate ``v``; the factored >=2-D
path uses Adafactor's rank-1 reconstruction for the global ``v_mean`` and is only
checked for self-consistency / foreach parity, not numpy parity). The defining
property — the per-coordinate adaptive momentum (inertia) from the GLOBAL-normalized
second moment, with the step taken ALONG the momentum (no ``1/sqrt(v)``) — is what
the reference verifies.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kaon import Adai


def _ref_adai_1d(
    params: list[np.ndarray],
    grads_seq: list[list[np.ndarray]],
    *,
    lr: float,
    beta0: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    decoupled: bool,
) -> list[np.ndarray]:
    """Reference numpy Adai over a sequence of grads for a GROUP of 1-D params.

    Reproduces the official adai.py two-pass step:

    * pass 1: weight decay (L2 or decoupled) -> ``v`` EMA -> accumulate
      ``sum(v_hat)`` and ``param_size`` across ALL params (the global reduction).
    * pass 2: per-coordinate ``beta1 = clamp(1 - (beta0/v_mean)*v_hat, 0, 1-eps)``,
      ``beta1_prod *= beta1``, ``m = beta1*m + (1-beta1)*g``,
      ``m_hat = m/(1-beta1_prod)``, ``p -= lr*m_hat``.
    """
    ps = [p.copy() for p in params]
    ms = [np.zeros_like(p) for p in params]
    vs = [np.zeros_like(p) for p in params]
    bprod = [np.ones_like(p) for p in params]
    n_steps = len(grads_seq[0])
    for t in range(1, n_steps + 1):
        bc2 = 1.0 - beta2 ** t
        # ---- pass 1 ----
        v_hat_sum = 0.0
        param_size = 0
        gs_t = []
        for j in range(len(ps)):
            g = grads_seq[j][t - 1].copy()
            if weight_decay != 0:
                if decoupled:
                    ps[j] = ps[j] * (1.0 - lr * weight_decay)
                else:
                    g = g + weight_decay * ps[j]
            vs[j][...] = beta2 * vs[j] + (1.0 - beta2) * g * g
            v_hat_sum += (vs[j] / bc2).sum()
            param_size += ps[j].size
            gs_t.append(g)
        v_mean = v_hat_sum / param_size
        # ---- pass 2 ----
        for j in range(len(ps)):
            g = gs_t[j]
            v_hat = vs[j] / bc2
            beta1 = np.clip(1.0 - (beta0 / v_mean) * v_hat, 0.0, 1.0 - eps)
            bprod[j][...] = bprod[j] * beta1
            ms[j][...] = beta1 * ms[j] + (1.0 - beta1) * g
            m_hat = ms[j] / (1.0 - bprod[j])
            ps[j] = ps[j] - lr * m_hat
    return ps


def test_construct_and_step():
    """Construct and take a step on tiny CPU tensors (2-D + 1-D + conv)."""
    params = [
        torch.nn.Parameter(torch.randn(8, 4)),
        torch.nn.Parameter(torch.randn(5)),
        torch.nn.Parameter(torch.randn(3, 2, 3, 3)),
    ]
    opt = Adai(params, lr=1.0)
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()
    for p in params:
        assert torch.isfinite(p).all()


@pytest.mark.parametrize("decoupled", [False, True])
@pytest.mark.parametrize("weight_decay", [0.0, 0.05])
def test_matches_numpy_reference_1d(decoupled, weight_decay):
    """Adai's 1-D fp32 path matches the official-Adai numpy reference (GC + cautious off).

    Uses a GROUP of several 1-D params so the GLOBAL v_mean reduction is genuinely
    exercised (v_mean spans all params in the group).
    """
    torch.manual_seed(11)
    lengths = [13, 7, 5]
    p0s = [torch.randn(n) for n in lengths]
    params = [torch.nn.Parameter(p.clone()) for p in p0s]
    opt = Adai(
        params, lr=0.5, betas=(0.1, 0.99), eps=1e-3,
        weight_decay=weight_decay, decoupled=decoupled,
        cautious=False, gradient_centralization=False,
        momentum_dtype="float32", foreach=False,
    )
    grads_seq = [[torch.randn(n) for _ in range(9)] for n in lengths]
    for t in range(9):
        for j, p in enumerate(params):
            p.grad = grads_seq[j][t].clone()
        opt.step()
    ref = _ref_adai_1d(
        [p.numpy() for p in p0s],
        [[g.numpy() for g in gs] for gs in grads_seq],
        lr=0.5, beta0=0.1, beta2=0.99, eps=1e-3,
        weight_decay=weight_decay, decoupled=decoupled,
    )
    for p, r in zip(params, ref, strict=True):
        np.testing.assert_allclose(p.detach().numpy(), r, rtol=1e-5, atol=1e-6)


def test_inertia_flat_coord_gets_more_momentum():
    """The DEFINING property: a low-variance (flat) coordinate gets a higher
    effective momentum (beta1 closer to 1) than a high-variance (sharp) one.

    Drive one coordinate with large gradients and another with tiny gradients;
    after a few steps the small-``v`` (flat) coordinate must have a larger
    beta1_prod-decay-rate, i.e. its per-step beta1 is closer to 1. We measure the
    implied beta1 directly from the optimizer's stored v and the global v_mean.
    """
    torch.manual_seed(0)
    n = 4
    p = torch.nn.Parameter(torch.zeros(n))
    opt = Adai(
        [p], lr=0.5, betas=(0.1, 0.99), eps=1e-3,
        cautious=False, gradient_centralization=False,
        momentum_dtype="float32", foreach=False,
    )
    # coord 0 = flat (tiny grad), coord 1 = sharp (large grad); 2,3 mid.
    scales = torch.tensor([1e-2, 1.0, 0.2, 0.2])
    for _ in range(8):
        p.grad = scales * torch.randn(n).abs()  # same sign -> consistent curvature
        opt.step()
    st = opt.state[p]
    v = st["v"]
    # Reconstruct the beta1 the last step would compute (relative ordering is the point).
    bc2 = 1.0 - 0.99 ** opt.param_groups[0]["step"]
    v_hat = v / bc2
    v_mean = v_hat.mean()
    beta1 = (1.0 - (0.1 / v_mean) * v_hat).clamp(0.0, 1.0 - 1e-3)
    # Flat coord (0) has the smallest v_hat -> the largest beta1 (most inertia).
    assert beta1[0] > beta1[1], (beta1.tolist())
    assert beta1[0] == beta1.max()
    assert beta1[1] == beta1.min()


def test_step_is_along_momentum_not_normalized():
    """Adai steps ALONG m_hat, NOT m_hat/sqrt(v) — verify the update is not
    divided by sqrt(v) again.

    With a single coordinate and constant gradient, after one step
    ``beta1 = clamp(1 - beta0, 0, 1-eps)`` (since v_hat == v_mean for one coord),
    ``m = (1-beta1)*g``, ``m_hat = m/(1-beta1) = g``, so the first update is
    exactly ``-lr*g`` regardless of the gradient magnitude (no 1/sqrt(v) shrink).
    """
    g_val = 7.0  # large; an Adam-style 1/sqrt(v) would shrink this to ~ -lr*sign(g)
    p = torch.nn.Parameter(torch.zeros(1))
    lr = 0.3
    opt = Adai([p], lr=lr, betas=(0.1, 0.99), eps=1e-3,
               cautious=False, gradient_centralization=False,
               momentum_dtype="float32", foreach=False)
    p.grad = torch.tensor([g_val])
    opt.step()
    # single coord: v_hat == v_mean, beta1 = 1 - beta0 = 0.9, m_hat = g, step = -lr*g.
    assert torch.allclose(p.detach(), torch.tensor([-lr * g_val]), atol=1e-5)


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
def test_momentum_dtype_variants_construct_and_step(momentum_dtype):
    params = [
        torch.nn.Parameter(torch.randn(8, 4)),
        torch.nn.Parameter(torch.randn(5)),
        torch.nn.Parameter(torch.randn(3, 2, 3, 3)),
    ]
    opt = Adai(params, lr=1.0, momentum_dtype=momentum_dtype)
    for _ in range(3):
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
    for p in params:
        assert torch.isfinite(p).all()


def test_beta1_prod_is_fp32():
    """beta1_prod is a per-coordinate fp32 running product (never quantized)."""
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = Adai([p], lr=1.0, momentum_dtype="int8")
    p.grad = torch.randn_like(p)
    opt.step()
    st = opt.state[p]
    assert st["beta1_prod"].dtype == torch.float32
    assert st["beta1_prod"].shape == p.shape
    # After the first step beta1_prod < 1 everywhere (multiplied by beta1 in (0,1)).
    assert (st["beta1_prod"] < 1.0).all()


@pytest.mark.parametrize(
    "cfg",
    [
        dict(momentum_dtype="float32", betas=(0.1, 0.99), weight_decay=0.0, decoupled=True, cautious=True),
        dict(momentum_dtype="int8", betas=(0.1, 0.99), weight_decay=0.02, decoupled=True, cautious=True),
        dict(momentum_dtype="bfloat16", betas=(0.2, 0.999), weight_decay=0.01, decoupled=False, cautious=False),
        dict(momentum_dtype="4bit", betas=(0.1, 0.99), weight_decay=0.0, decoupled=True, cautious=False),
    ],
)
def test_foreach_matches_per_param(cfg):
    """foreach=True is element-for-element equal to the per-parameter path,
    INCLUDING the global v_mean reduction (fp32 weights)."""
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
    oa = Adai(pa, lr=0.5, foreach=True, bf16_method="none", **cfg)
    ob = Adai(pb, lr=0.5, foreach=False, bf16_method="none", **cfg)
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
        assert torch.equal(a, b), (a - b).abs().max()


def test_foreach_chunking_is_exact():
    """Splitting a foreach bucket into chunks must not change the result
    (the v_mean reduction is already global / pass-1, independent of chunking)."""
    def mk() -> list[torch.nn.Parameter]:
        return [torch.nn.Parameter(torch.randn(6, 5)) for _ in range(7)]

    torch.manual_seed(2)
    pa = mk()
    torch.manual_seed(2)
    pb = mk()
    oa = Adai(pa, lr=0.5, momentum_dtype="int8", weight_decay=0.02,
              foreach=True, foreach_stack_budget=120)
    ob = Adai(pb, lr=0.5, momentum_dtype="int8", weight_decay=0.02, foreach=False)
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


def test_state_dict_round_trip_preserves_dtype():
    """load_state_dict preserves the quantized momentum dtype (no fp32 upcast)."""
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = Adai([p], lr=1.0, momentum_dtype="int8")
    p.grad = torch.randn_like(p)
    opt.step()
    sd = opt.state_dict()

    p2 = torch.nn.Parameter(p.detach().clone())
    opt2 = Adai([p2], lr=1.0, momentum_dtype="int8")
    p2.grad = torch.randn_like(p2)
    opt2.step()  # allocate state
    opt2.load_state_dict(sd)
    assert opt2.state[p2]["m"].dtype == torch.int8
    assert opt2.state[p2]["beta1_prod"].dtype == torch.float32


def test_decoupled_vs_l2_differ():
    """decoupled=True (AdaiW) and decoupled=False (L2) take different steps."""
    torch.manual_seed(0)
    p0 = torch.randn(10)

    def run(decoupled: bool) -> torch.Tensor:
        p = torch.nn.Parameter(p0.clone())
        opt = Adai([p], lr=0.5, weight_decay=0.1, decoupled=decoupled,
                   cautious=False, gradient_centralization=False,
                   momentum_dtype="float32", foreach=False)
        torch.manual_seed(5)
        for _ in range(3):
            p.grad = torch.randn(10)
            opt.step()
        return p.detach().clone()

    assert not torch.allclose(run(True), run(False))
