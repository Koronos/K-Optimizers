"""Tests for ADOPT — modified Adam that converges with any beta2 (arXiv:2411.02853).

The numpy reference mirrors ADOPT's **non-factored (1-D) fp32 path**, which matches
the official ``iShohei220/adopt`` / kozistr ``pytorch_optimizer.ADOPT`` math exactly
(full per-coordinate ``v``; the factored 2-D path uses Adafactor's row/col
approximation and is only checked for self-consistency / foreach parity).

The distinctive ADOPT pieces under test:

* **v-lag**: ``v`` reflects grads up to ``t-1`` when normalizing ``g_t``; ``g_t`` is
  folded into ``v`` only AFTER it has been used.
* **normalize-then-momentum**: the first-moment EMA is of the *normalized, clipped*
  gradient (not the raw gradient).
* **step-0 init**: the very first ``.step()`` only sets ``v = g_0^2`` and does NOT
  update the parameter (no WD either).
* **Algorithm-2 clip**: ``normed_grad`` clamped to ``[-c_t, c_t]`` with
  ``c_t = step ** 0.25`` (0-indexed step).
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch

from kaon import ADOPT

from .conftest import train_steps


def _ref_adopt_1d(
    p: np.ndarray,
    grads: list[np.ndarray],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    clip: bool,
) -> np.ndarray:
    """Reference numpy ADOPT over a sequence of grads (1-D, full v).

    Reproduces the official ordering: step-0 init (v=g0^2, no update), then for
    each subsequent step: decoupled WD, normalize-by-old-v with an eps floor,
    Algorithm-2 clip, momentum EMA of the normalized grad, ``p -= lr*m``, then
    fold ``g_t`` into ``v``.
    """
    p = p.copy()
    m = np.zeros_like(p)
    v = np.zeros_like(p)
    for ostep, g in enumerate(grads):
        if ostep == 0:
            v[...] = g * g
            continue
        if weight_decay != 0:
            p = p * (1.0 - lr * weight_decay)
        denom = np.maximum(np.sqrt(v), eps)
        normed = g / denom
        if clip:
            c = ostep ** 0.25
            normed = np.clip(normed, -c, c)
        m[...] = beta1 * m + (1.0 - beta1) * normed
        p = p - lr * m
        v[...] = beta2 * v + (1.0 - beta2) * g * g
    return p


def test_construct_and_step():
    """Construct and take steps on tiny CPU tensors (2-D + 1-D + conv)."""
    params = [
        torch.nn.Parameter(torch.randn(8, 4)),
        torch.nn.Parameter(torch.randn(5)),
        torch.nn.Parameter(torch.randn(3, 2, 3, 3)),
    ]
    opt = ADOPT(params, lr=1e-3)
    for _ in range(3):
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
    for p in params:
        assert torch.isfinite(p).all()


@pytest.mark.parametrize("clip", [False, True])
@pytest.mark.parametrize("weight_decay", [0.0, 0.05])
def test_matches_numpy_reference_1d(clip, weight_decay):
    """ADOPT's 1-D fp32 path matches the numpy ADOPT reference (cautious + GC off)."""
    torch.manual_seed(11)
    n = 13
    p0 = torch.randn(n)
    p = torch.nn.Parameter(p0.clone())
    opt = ADOPT(
        [p], lr=1e-2, betas=(0.9, 0.9999), eps=1e-6, weight_decay=weight_decay,
        clip=clip, cautious=False, gradient_centralization=False,
        momentum_dtype="float32", foreach=False,
    )
    grads = [torch.randn(n) for _ in range(9)]
    for g in grads:
        p.grad = g.clone()
        opt.step()
    ref = _ref_adopt_1d(
        p0.numpy(), [g.numpy() for g in grads],
        lr=1e-2, beta1=0.9, beta2=0.9999, eps=1e-6,
        weight_decay=weight_decay, clip=clip,
    )
    np.testing.assert_allclose(p.detach().numpy(), ref, rtol=1e-5, atol=1e-6)


def test_step0_initializes_v_and_skips_update():
    """The first .step() sets v = g_0^2 and does NOT move the parameter (no WD)."""
    torch.manual_seed(0)
    n = 7
    p0 = torch.randn(n)
    p = torch.nn.Parameter(p0.clone())
    # weight_decay nonzero to prove WD is also skipped on step 0.
    opt = ADOPT(
        [p], lr=1e-2, weight_decay=0.1, cautious=False,
        gradient_centralization=False, momentum_dtype="float32", foreach=False,
    )
    g0 = torch.randn(n)
    p.grad = g0.clone()
    opt.step()
    # parameter is byte-identical (no update, no WD on the init step).
    assert torch.equal(p.detach(), p0)
    # v was initialized to g0^2 exactly (no beta2 EMA, no bias correction).
    torch.testing.assert_close(opt.state[p]["v"], g0 * g0, rtol=0, atol=0)
    # momentum is still zero (the normalize-then-EMA only runs from step 1).
    assert torch.count_nonzero(opt.state[p]["m"]) == 0


def test_step0_init_factored():
    """For a 2-D weight, step 0 sets the factored row/col stats from g_0^2 and skips."""
    torch.manual_seed(0)
    p0 = torch.randn(6, 5)
    p = torch.nn.Parameter(p0.clone())
    opt = ADOPT([p], lr=1e-2, cautious=False, gradient_centralization=False,
                momentum_dtype="float32", foreach=False)
    g0 = torch.randn(6, 5)
    p.grad = g0.clone()
    opt.step()
    assert torch.equal(p.detach(), p0)  # no update on the init step
    gsq = g0 * g0
    torch.testing.assert_close(opt.state[p]["row"], gsq.mean(dim=-1), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(opt.state[p]["col"], gsq.mean(dim=-2), rtol=1e-6, atol=1e-7)


def test_v_lag_normalizer_independent_of_current_grad():
    """The update at step t must NOT depend on g_t through the normalizer (the v-lag).

    Run two trajectories identical except for the *magnitude* of the final
    gradient's normalizer contribution: because v lags, scaling g_t changes the
    momentum numerator but the denominator (built from v up to t-1) is unchanged —
    so the normed grad is exactly linear in g_t (modulo the clip). We verify the
    second-moment state after the step equals the manual lagged update.
    """
    torch.manual_seed(0)
    n = 9
    p = torch.nn.Parameter(torch.randn(n))
    opt = ADOPT([p], lr=1e-2, betas=(0.9, 0.95), eps=1e-6, clip=False,
                cautious=False, gradient_centralization=False,
                momentum_dtype="float32", foreach=False)
    g0 = torch.randn(n)
    p.grad = g0.clone()
    opt.step()                                # step 0: v = g0^2
    v_after0 = opt.state[p]["v"].clone()
    g1 = torch.randn(n)
    # the denom used at step 1 must be sqrt(v_after0) (NOT including g1).
    denom_expected = v_after0.sqrt().clamp_(min=1e-6)
    m_expected = (1.0 - 0.9) * (g1 / denom_expected)
    p.grad = g1.clone()
    opt.step()                                # step 1
    torch.testing.assert_close(opt.state[p]["m"], m_expected, rtol=1e-5, atol=1e-6)
    # v now folds in g1: v = beta2*v0 + (1-beta2)*g1^2.
    v_expected = 0.95 * v_after0 + 0.05 * g1 * g1
    torch.testing.assert_close(opt.state[p]["v"], v_expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
def test_momentum_dtype_variants_construct_and_step(momentum_dtype):
    params = [
        torch.nn.Parameter(torch.randn(8, 4)),
        torch.nn.Parameter(torch.randn(5)),
        torch.nn.Parameter(torch.randn(3, 2, 3, 3)),
    ]
    opt = ADOPT(params, lr=1e-3, momentum_dtype=momentum_dtype)
    for _ in range(4):
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
    for p in params:
        assert torch.isfinite(p).all()


@pytest.mark.parametrize(
    "cfg",
    [
        dict(momentum_dtype="float32", betas=(0.9, 0.9999), weight_decay=0.0, clip=True, cautious=True),
        dict(momentum_dtype="bfloat16", betas=(0.9, 0.9999), weight_decay=0.0, clip=True, cautious=True),
        dict(momentum_dtype="int8", betas=(0.9, 0.999), weight_decay=0.02, clip=True, cautious=True),
        dict(momentum_dtype="int8", betas=(0.9, 0.9999), weight_decay=0.0, clip=False, cautious=False),
        dict(momentum_dtype="4bit", betas=(0.9, 0.99), weight_decay=0.01, clip=True, cautious=False),
    ],
)
def test_foreach_matches_per_param(cfg):
    """foreach=True is element-for-element equal to the per-parameter path (fp32 weights).

    >= 6 steps so the v-lag and the step**0.25 clip schedule are both exercised.
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
    oa = ADOPT(pa, lr=1e-3, foreach=True, bf16_method="none", **cfg)
    ob = ADOPT(pb, lr=1e-3, foreach=False, bf16_method="none", **cfg)
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
    oa = ADOPT(pa, lr=1e-3, momentum_dtype="int8", weight_decay=0.02, foreach=True, foreach_stack_budget=120)
    ob = ADOPT(pb, lr=1e-3, momentum_dtype="int8", weight_decay=0.02, foreach=False)
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


def test_cautious_masks_disagreeing_coords():
    """cautious=True changes the trajectory (not a no-op with momentum on)."""
    torch.manual_seed(0)
    n = 32
    p0 = torch.randn(n)

    def run(cautious: bool) -> torch.Tensor:
        p = torch.nn.Parameter(p0.clone())
        opt = ADOPT([p], lr=1e-2, cautious=cautious, gradient_centralization=False,
                    momentum_dtype="float32", foreach=False)
        torch.manual_seed(5)
        for _ in range(5):
            p.grad = torch.randn(n)
            opt.step()
        return p.detach().clone()

    assert not torch.allclose(run(True), run(False))


def test_clip_changes_trajectory():
    """clip=True (Algorithm 2) differs from the unclipped revision-1 behaviour."""
    torch.manual_seed(0)
    n = 16
    p0 = torch.randn(n)

    def run(clip: bool) -> torch.Tensor:
        p = torch.nn.Parameter(p0.clone())
        # large grads relative to v so the clip bites on early steps.
        opt = ADOPT([p], lr=1e-2, clip=clip, cautious=False,
                    gradient_centralization=False, momentum_dtype="float32", foreach=False)
        torch.manual_seed(9)
        for _ in range(4):
            p.grad = torch.randn(n) * 5.0
            opt.step()
        return p.detach().clone()

    assert not torch.allclose(run(True), run(False))


def test_overfits_regression(toy_mlp, random_batch):
    """ADOPT should drive a tiny MLP's training loss down on a fixed batch."""
    x, y = random_batch
    opt = ADOPT(toy_mlp.parameters(), lr=3e-3)
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
    opt = ADOPT([p], lr=1e-3, bf16_method="stochastic_rounding")
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()
    assert torch.isfinite(p).all()


def test_kahan_runs():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 4, dtype=torch.bfloat16))
    opt = ADOPT([p], lr=1e-3, bf16_method="kahan")
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
    opt_a = ADOPT(params_a, lr=1e-3, momentum_dtype=momentum_dtype, weight_decay=0.01)
    for _ in range(4):
        for p in params_a:
            p.grad = torch.randn_like(p)
        opt_a.step()

    buf = io.BytesIO()
    torch.save(opt_a.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, weights_only=False)

    params_b = [torch.nn.Parameter(p.detach().clone()) for p in params_a]
    opt_b = ADOPT(params_b, lr=1e-3, momentum_dtype=momentum_dtype, weight_decay=0.01)
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
    opt = ADOPT(net.parameters(), lr=1e-3)
    x = torch.randn(2, 3, 8, 8)
    y = torch.randn(2, 2, 8, 8)
    train_steps(net, opt, [(x, y)] * 6)
    for p in net.parameters():
        assert torch.isfinite(p).all()


def test_invalid_args_rejected():
    p = [torch.nn.Parameter(torch.randn(3))]
    with pytest.raises(ValueError):
        ADOPT(p, lr=-1.0)
    with pytest.raises(ValueError):
        ADOPT(p, betas=(1.0, 0.9999))
    with pytest.raises(ValueError):
        ADOPT(p, betas=(0.9, 1.0))
    with pytest.raises(ValueError):
        ADOPT(p, eps=0.0)
    with pytest.raises(ValueError):
        ADOPT(p, weight_decay=-0.1)
    with pytest.raises(ValueError):
        ADOPT(p, momentum_dtype="fp8")
    with pytest.raises(ValueError):
        ADOPT(p, bf16_method="bogus")


def test_sparse_grad_rejected():
    p = torch.nn.Parameter(torch.randn(4))
    opt = ADOPT([p], lr=1e-3)
    idx = torch.tensor([[0, 2]])
    val = torch.tensor([1.0, 1.0])
    p.grad = torch.sparse_coo_tensor(idx, val, (4,))
    with pytest.raises(RuntimeError):
        opt.step()
