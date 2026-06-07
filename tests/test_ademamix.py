"""Tests for AdEMAMix — Adam + a mixture of two gradient EMAs on kaon's backend.

The numpy reference mirrors AdEMAMix's **non-factored (1-D) fp32 path**, which is
the one that matches the official Apple ``ml-ademamix`` reference exactly (full
per-coordinate ``v``; the factored 2-D path uses Adafactor's row/col approximation
and is only checked for self-consistency / foreach parity, not numpy parity).

The reference and the scheduler tests pin ``gradient_centralization=False`` and use
``momentum_dtype="float32"`` so the comparison is to exact AdEMAMix arithmetic.
"""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
import torch

from kaon import AdEMAMix
from kaon.ademamix import schedule_alpha, schedule_beta3

from .conftest import train_steps


# --------------------------------------------------------------------------- #
#  tiny numpy reference (1-D, full v) — matches official AdEMAMix arithmetic     #
# --------------------------------------------------------------------------- #
def _ref_ademamix_1d(
    p: np.ndarray,
    grads: list[np.ndarray],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    beta3: float,
    alpha: float,
    eps: float,
    weight_decay: float,
) -> np.ndarray:
    """Reference numpy AdEMAMix over a sequence of grads (1-D, full v, NO warmup).

    Reproduces the official update:

    * m1 <- beta1*m1 + (1-beta1)*g ; m2 <- beta3*m2 + (1-beta3)*g
    * v  <- beta2*v + (1-beta2)*g^2
    * denom = sqrt(v)/sqrt(1-beta2^t) + eps
    * num   = m1/(1-beta1^t) + alpha*m2     (only m1 bias-corrected)
    * p    -= lr*(num/denom + weight_decay*p)
    """
    p = p.copy()
    m1 = np.zeros_like(p)
    m2 = np.zeros_like(p)
    v = np.zeros_like(p)
    for t, g in enumerate(grads, start=1):
        m1 = beta1 * m1 + (1.0 - beta1) * g
        m2 = beta3 * m2 + (1.0 - beta3) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        bc1 = 1.0 - beta1 ** t
        bc2_sq = math.sqrt(1.0 - beta2 ** t)
        denom = np.sqrt(v) / bc2_sq + eps
        num = m1 / bc1 + alpha * m2
        update = num / denom + weight_decay * p
        p = p - lr * update
    return p


def test_construct_and_step():
    p = torch.nn.Parameter(torch.randn(5, 7))
    opt = AdEMAMix([p], lr=1e-3)
    p.grad = torch.randn_like(p)
    opt.step()
    assert torch.isfinite(p).all()


@pytest.mark.parametrize("weight_decay", [0.0, 0.05])
@pytest.mark.parametrize("alpha", [2.0, 5.0])
def test_matches_numpy_reference_1d(alpha, weight_decay):
    """1-D fp32 path == exact numpy AdEMAMix (warmup off, GC off, cautious off)."""
    torch.manual_seed(0)
    n = 24
    lr, beta1, beta2, beta3, eps = 2e-3, 0.9, 0.999, 0.9999, 1e-8
    p0 = torch.randn(n, dtype=torch.float64)
    grads = [torch.randn(n, dtype=torch.float64) for _ in range(12)]

    p = torch.nn.Parameter(p0.to(torch.float32).clone())
    opt = AdEMAMix(
        [p], lr=lr, betas=(beta1, beta2, beta3), alpha=alpha, t_alpha_beta3=None,
        eps=eps, weight_decay=weight_decay, cautious=False,
        gradient_centralization=False, momentum_dtype="float32", foreach=False,
    )
    for g in grads:
        p.grad = g.to(torch.float32).clone()
        opt.step()

    ref = _ref_ademamix_1d(
        p0.numpy(), [g.numpy() for g in grads],
        lr=lr, beta1=beta1, beta2=beta2, beta3=beta3, alpha=alpha,
        eps=eps, weight_decay=weight_decay,
    )
    np.testing.assert_allclose(p.detach().numpy(), ref, rtol=1e-5, atol=1e-6)


def test_slow_ema_not_bias_corrected():
    """Sanity: m2 is NOT bias-corrected. With alpha large and step 1, the slow EMA
    contributes ~(1-beta3)*g (tiny), not a bias-corrected ~g; verify the numerator
    matches the un-corrected form."""
    n = 8
    beta1, beta2, beta3, alpha = 0.9, 0.999, 0.9999, 5.0
    g = torch.randn(n)
    p = torch.nn.Parameter(torch.zeros(n))
    opt = AdEMAMix(
        [p], lr=1.0, betas=(beta1, beta2, beta3), alpha=alpha, t_alpha_beta3=None,
        eps=0.0, weight_decay=0.0, cautious=False, gradient_centralization=False,
        momentum_dtype="float32", foreach=False,
    )
    p.grad = g.clone()
    opt.step()
    # After 1 step: m1=(1-b1)*g, m2=(1-b3)*g, v=(1-b2)*g^2.
    # bc1 = 1-b1, bc2_sq = sqrt(1-b2). denom = sqrt(v)/bc2_sq = |g|. (eps=0)
    # num = m1/bc1 + alpha*m2 = g + alpha*(1-b3)*g. update = num/|g| = sign(g)*(1+alpha*(1-b3))
    expected = -1.0 * torch.sign(g) * (1.0 + alpha * (1.0 - beta3))
    assert torch.allclose(p.detach(), expected, atol=1e-5)


# --------------------------------------------------------------------------- #
#  scheduler trajectory tests                                                   #
# --------------------------------------------------------------------------- #
def test_schedule_alpha_trajectory():
    alpha_final, t_warm = 5.0, 100
    assert schedule_alpha(0, alpha_final, t_warm) == pytest.approx(0.0)
    assert schedule_alpha(50, alpha_final, t_warm) == pytest.approx(2.5)
    assert schedule_alpha(100, alpha_final, t_warm) == pytest.approx(5.0)  # clamps at t_warm
    assert schedule_alpha(250, alpha_final, t_warm) == pytest.approx(5.0)
    # disabled -> constant final
    assert schedule_alpha(3, alpha_final, None) == pytest.approx(5.0)


def test_schedule_beta3_trajectory():
    beta1, beta3_final, t_warm = 0.9, 0.9999, 100

    def f(b):
        return math.log(0.5) / math.log(b + 1e-8) - 1.0

    def f_inv(t):
        return math.pow(0.5, 1.0 / (t + 1.0))

    # at step 0 -> beta1; at step t_warm -> beta3_final; midpoint -> half-life-space mean.
    assert schedule_beta3(0, beta1, beta3_final, t_warm) == pytest.approx(f_inv(f(beta1)))
    assert schedule_beta3(100, beta1, beta3_final, t_warm) == pytest.approx(beta3_final)
    assert schedule_beta3(250, beta1, beta3_final, t_warm) == pytest.approx(beta3_final)
    mid = f_inv(0.5 * f(beta1) + 0.5 * f(beta3_final))
    assert schedule_beta3(50, beta1, beta3_final, t_warm) == pytest.approx(mid)
    # monotone increasing toward beta3_final
    traj = [schedule_beta3(s, beta1, beta3_final, t_warm) for s in range(0, 101, 10)]
    assert all(b <= a + 1e-12 for b, a in zip(traj, traj[1:], strict=False))
    assert traj[0] == pytest.approx(beta1, abs=1e-6)
    # disabled -> constant final
    assert schedule_beta3(3, beta1, beta3_final, None) == pytest.approx(beta3_final)


def test_reference_with_warmup_matches_manual():
    """End-to-end 1-D path with warmup ACTIVE matches a manual reference that calls
    the same schedulers (covers the scheduler integration in the step)."""
    torch.manual_seed(1)
    n = 16
    lr, beta1, beta2, beta3, alpha, t_warm = 2e-3, 0.9, 0.999, 0.99, 4.0, 8
    p0 = torch.randn(n, dtype=torch.float64)
    grads = [torch.randn(n, dtype=torch.float64) for _ in range(12)]

    p = torch.nn.Parameter(p0.to(torch.float32).clone())
    opt = AdEMAMix(
        [p], lr=lr, betas=(beta1, beta2, beta3), alpha=alpha, t_alpha_beta3=t_warm,
        eps=1e-8, weight_decay=0.0, cautious=False, gradient_centralization=False,
        momentum_dtype="float32", foreach=False,
    )
    for g in grads:
        p.grad = g.to(torch.float32).clone()
        opt.step()

    # manual reference with per-step alpha_t / beta3_t from the schedulers
    pr = p0.numpy().copy()
    m1 = np.zeros(n)
    m2 = np.zeros(n)
    v = np.zeros(n)
    for t, g in enumerate((gg.numpy() for gg in grads), start=1):
        a_t = schedule_alpha(t, alpha, t_warm)
        b3_t = schedule_beta3(t, beta1, beta3, t_warm)
        m1 = beta1 * m1 + (1 - beta1) * g
        m2 = b3_t * m2 + (1 - b3_t) * g
        v = beta2 * v + (1 - beta2) * g * g
        bc1 = 1 - beta1 ** t
        bc2_sq = math.sqrt(1 - beta2 ** t)
        denom = np.sqrt(v) / bc2_sq + 1e-8
        num = m1 / bc1 + a_t * m2
        pr = pr - lr * (num / denom)
    np.testing.assert_allclose(p.detach().numpy(), pr, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------- #
#  foreach == per-param parity                                                   #
# --------------------------------------------------------------------------- #
def _make_params(seed: int) -> list[torch.nn.Parameter]:
    torch.manual_seed(seed)
    shapes = [(8, 6), (5, 4), (3, 7, 2), (10,), (4, 4), (6,)]
    return [torch.nn.Parameter(torch.randn(*s)) for s in shapes]


@pytest.mark.parametrize(
    "cfg",
    [
        {"momentum_dtype": "bfloat16", "weight_decay": 0.0, "t_alpha_beta3": 6},
        {"momentum_dtype": "bfloat16", "weight_decay": 0.03, "t_alpha_beta3": 6},
        {"momentum_dtype": "int8", "weight_decay": 0.0, "t_alpha_beta3": 6},
        {"momentum_dtype": "int8", "weight_decay": 0.02, "t_alpha_beta3": 6},
        {"momentum_dtype": "4bit", "weight_decay": 0.0, "t_alpha_beta3": 6},
    ],
)
def test_foreach_matches_per_param(cfg):
    """foreach=True is element-for-element equal to the per-parameter path (fp32 weights).

    Runs enough steps (10 > t_alpha_beta3=6) that the alpha/beta3 warmup is partway
    through then completes, exercising the schedulers on both paths.
    """
    pa = _make_params(7)
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    grad_seq = [
        [torch.randn_like(p) for p in pa]
        for _ in range(10)
    ]
    oa = AdEMAMix(pa, lr=1e-3, foreach=True, bf16_method="none", **cfg)
    ob = AdEMAMix(pb, lr=1e-3, foreach=False, bf16_method="none", **cfg)
    for grads in grad_seq:
        for p, g in zip(pa, grads, strict=False):
            p.grad = g.clone()
        for p, g in zip(pb, grads, strict=False):
            p.grad = g.clone()
        oa.step()
        ob.step()
    max_diff = 0.0
    for a, b in zip(pa, pb, strict=False):
        max_diff = max(max_diff, (a.detach() - b.detach()).abs().max().item())
    # quantized momenta round-trip identically on both paths -> bit-exact (fp32 weights).
    assert max_diff == 0.0, f"foreach vs per-param max abs diff = {max_diff}"


def test_foreach_chunking_is_exact():
    """A tight foreach_stack_budget (forcing multi-chunk buckets) stays exact."""
    pa = _make_params(11)
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    oa = AdEMAMix(pa, lr=1e-3, momentum_dtype="int8", weight_decay=0.02,
                  t_alpha_beta3=4, foreach=True, foreach_stack_budget=40)
    ob = AdEMAMix(pb, lr=1e-3, momentum_dtype="int8", weight_decay=0.02,
                  t_alpha_beta3=4, foreach=False)
    for _ in range(6):
        for p in pa:
            p.grad = torch.randn_like(p)
        for pb_, pa_ in zip(pb, pa, strict=False):
            pb_.grad = pa_.grad.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb, strict=False):
        assert torch.equal(a.detach(), b.detach())


# --------------------------------------------------------------------------- #
#  misc behavioural / robustness                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
def test_momentum_dtype_variants_construct_and_step(momentum_dtype):
    p = torch.nn.Parameter(torch.randn(6, 5))
    opt = AdEMAMix([p], lr=1e-3, momentum_dtype=momentum_dtype)
    p.grad = torch.randn_like(p)
    opt.step()
    assert torch.isfinite(p).all()


def test_two_momenta_present():
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = AdEMAMix([p], lr=1e-3, momentum_dtype="float32")
    p.grad = torch.randn_like(p)
    opt.step()
    st = opt.state[p]
    assert "m1" in st and "m2" in st


def test_cautious_changes_update():
    def run(cautious):
        torch.manual_seed(3)
        p = torch.nn.Parameter(torch.randn(6))
        opt = AdEMAMix([p], lr=1e-2, cautious=cautious, gradient_centralization=False,
                       momentum_dtype="float32", foreach=False, t_alpha_beta3=None)
        for _ in range(4):
            p.grad = torch.randn(6)
            opt.step()
        return p.detach().clone()
    assert not torch.allclose(run(True), run(False))


def test_overfits_regression(toy_mlp, random_batch):
    x, y = random_batch
    opt = AdEMAMix(toy_mlp.parameters(), lr=5e-3, t_alpha_beta3=20)
    first = (toy_mlp(x) - y).pow(2).mean().item()
    train_steps(toy_mlp, opt, [(x, y)] * 200)
    last = (toy_mlp(x) - y).pow(2).mean().item()
    assert last < first


def test_bf16_weights_train_no_nan():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 8, dtype=torch.bfloat16))
    opt = AdEMAMix([p], lr=1e-3, bf16_method="stochastic_rounding")
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()
    assert torch.isfinite(p.float()).all()


def test_kahan_runs():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 8, dtype=torch.bfloat16))
    opt = AdEMAMix([p], lr=1e-3, bf16_method="kahan")
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()
    assert "shift" in opt.state[p]
    assert torch.isfinite(p.float()).all()


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "int8", "4bit"])
def test_checkpoint_roundtrip_preserves_momentum_dtype(momentum_dtype):
    p = torch.nn.Parameter(torch.randn(6, 5))
    opt = AdEMAMix([p], lr=1e-3, momentum_dtype=momentum_dtype, t_alpha_beta3=5)
    for _ in range(3):
        p.grad = torch.randn_like(p)
        opt.step()
    want = {k: opt.state[p][k].dtype for k in ("m1", "m2") if k in opt.state[p]}

    buf = io.BytesIO()
    torch.save(opt.state_dict(), buf)
    buf.seek(0)

    p2 = torch.nn.Parameter(p.detach().clone())
    opt2 = AdEMAMix([p2], lr=1e-3, momentum_dtype=momentum_dtype, t_alpha_beta3=5)
    opt2.load_state_dict(torch.load(buf, weights_only=False))
    for k, dt in want.items():
        assert opt2.state[p2][k].dtype == dt, f"{k}: {opt2.state[p2][k].dtype} != {dt}"


def test_conv_net_trains_no_nan():
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(4, 2, 3, padding=1),
    )
    opt = AdEMAMix(net.parameters(), lr=1e-3, t_alpha_beta3=10)
    x = torch.randn(2, 3, 8, 8)
    y = torch.randn(2, 2, 8, 8)
    for _ in range(8):
        opt.zero_grad()
        (net(x) - y).pow(2).mean().backward()
        opt.step()
    for p in net.parameters():
        assert torch.isfinite(p).all()


def test_invalid_args_rejected():
    p = torch.nn.Parameter(torch.randn(3))
    with pytest.raises(ValueError):
        AdEMAMix([p], betas=(1.0, 0.999, 0.9999))
    with pytest.raises(ValueError):
        AdEMAMix([p], betas=(0.9, 0.999, 1.0))
    with pytest.raises(ValueError):
        AdEMAMix([p], alpha=-1.0)
    with pytest.raises(ValueError):
        AdEMAMix([p], t_alpha_beta3=-5)
    with pytest.raises(ValueError):
        AdEMAMix([p], lr=-1.0)
    with pytest.raises(ValueError):
        AdEMAMix([p], momentum_dtype="bogus")


def test_sparse_grad_rejected():
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = AdEMAMix([p], lr=1e-3)
    idx = torch.tensor([[0], [1]])
    val = torch.tensor([1.0])
    p.grad = torch.sparse_coo_tensor(idx, val, (4, 4))
    with pytest.raises(RuntimeError):
        opt.step()
