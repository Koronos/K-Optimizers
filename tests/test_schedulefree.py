"""Tests for ScheduleFree — Schedule-Free AdamW (Defazio 2024) on kaon's backend.

The numpy reference mirrors ScheduleFree's **non-factored (1-D) fp32 path**, which
matches the official ``facebookresearch/schedule_free`` ``AdamWScheduleFree`` exactly
(full per-coordinate ``v``; the factored 2-D path uses Adafactor's row/col
approximation and is only checked for self-consistency / foreach parity).

The reference also tracks the three logical sequences (z, x, y) so the train()/eval()
swap can be checked: after ``eval()`` the parameter buffer must equal the reference's
running average ``x``.
"""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
import torch

from kaon import ScheduleFree


def _ref_schedulefree_1d(
    p0: np.ndarray,
    grads_at_y,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    warmup_steps: int = 0,
    r: float = 0.0,
    weight_lr_power: float = 2.0,
    inner_momentum: float = 0.0,
):
    """Reference numpy Schedule-Free AdamW (1-D, full v), tracking z / x / y.

    ``grads_at_y(y) -> g`` is a callable producing the gradient evaluated at the
    current interpolation point ``y`` (so the reference and the optimizer see the
    SAME gradient even though it depends on the iterate). Returns ``(x, y, z)`` after
    all steps, where ``x`` is the kept/averaged sequence (what ``eval()`` exposes).
    """
    x = p0.astype(np.float64).copy()
    z = p0.astype(np.float64).copy()
    v = np.zeros_like(x)
    exp_avg = np.zeros_like(x)
    weight_sum = 0.0
    lr_max = -1.0
    # Official convention: y = beta1*x + (1-beta1)*z (beta1 weights x, not z). At
    # k=0, x0 == z0 == p0 so y0 == p0 too.
    y = beta1 * x + (1.0 - beta1) * z
    for k, gfn in enumerate(grads_at_y):
        t = k + 1
        g = gfn(y).astype(np.float64)
        sched = (t / warmup_steps) if (warmup_steps > 0 and k < warmup_steps) else 1.0
        lr_t = lr * sched
        lr_max = max(lr_t, lr_max)
        weight = (t ** r) * (lr_max ** weight_lr_power)
        weight_sum += weight
        ckp1 = weight / weight_sum if weight_sum != 0 else 0.0

        bc2 = 1.0 - beta2 ** t
        v[...] = beta2 * v + (1.0 - beta2) * g * g
        denom = np.sqrt(v / bc2) + eps
        if inner_momentum != 0:
            exp_avg[...] = inner_momentum * exp_avg + (1.0 - inner_momentum) * g
            bc1 = 1.0 - inner_momentum ** t
            d = (exp_avg / bc1) / denom
        else:
            d = g / denom
        if weight_decay != 0:
            d = d + weight_decay * y

        # y-update (in place, official): y <- (1-ckp1)*y + ckp1*z ; y += d*lr_t*(beta1*(1-ckp1)-1)
        y = (1.0 - ckp1) * y + ckp1 * z
        y = y + d * (lr_t * (beta1 * (1.0 - ckp1) - 1.0))
        # z step
        z = z - lr_t * d
        # x (the average) is implied by y = beta1*x + (1-beta1)*z, i.e. the eval swap
        # x = (y - (1-beta1)*z)/beta1  ==  lerp(y, z, 1 - 1/beta1).
        x = (y - (1.0 - beta1) * z) / beta1
    return x, y, z


def test_construct_and_step():
    """Construct, train(), step on tiny CPU tensors (2-D + 1-D + conv)."""
    params = [
        torch.nn.Parameter(torch.randn(8, 4)),
        torch.nn.Parameter(torch.randn(5)),
        torch.nn.Parameter(torch.randn(3, 2, 3, 3)),
    ]
    opt = ScheduleFree(params, lr=2e-3)
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()
    for p in params:
        assert torch.isfinite(p).all()


def test_step_requires_train_mode():
    """step() outside train mode raises (the Schedule-Free safety check)."""
    p = torch.nn.Parameter(torch.randn(4))
    opt = ScheduleFree([p], lr=1e-3)
    p.grad = torch.randn_like(p)
    opt.step()           # default is train mode -> ok
    opt.eval()
    p.grad = torch.randn_like(p)
    with pytest.raises(RuntimeError):
        opt.step()


@pytest.mark.parametrize("weight_decay", [0.0, 0.05])
@pytest.mark.parametrize("inner_momentum", [0.0, 0.9])
def test_matches_numpy_reference_1d(weight_decay, inner_momentum):
    """1-D fp32 path matches the official Schedule-Free AdamW reference (cautious/GC off).

    Exercises the train()/eval() swap: the EVAL-mode parameter must equal the
    reference's averaged sequence ``x``.
    """
    torch.manual_seed(7)
    n = 11
    lr, beta1, beta2, eps = 2e-2, 0.9, 0.999, 1e-8
    p0 = torch.randn(n, dtype=torch.float64)
    p = torch.nn.Parameter(p0.clone())
    opt = ScheduleFree(
        [p], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay,
        inner_momentum=inner_momentum, cautious=False, gradient_centralization=False,
        momentum_dtype="float32", foreach=False,
    )

    # Fixed gradient sequence (the synthetic quadratic gradient is grad = A@y + b,
    # evaluated at the *current* iterate y so the reference must mirror it).
    torch.manual_seed(3)
    a_mat = torch.randn(n, n, dtype=torch.float64)
    a_mat = (a_mat @ a_mat.t()) / n + torch.eye(n, dtype=torch.float64)  # SPD
    b = torch.randn(n, dtype=torch.float64)

    nsteps = 12
    grad_fns = []
    opt.train()
    for _ in range(nsteps):
        y_now = p.detach().clone()                       # p.data holds y in train mode
        g = a_mat @ y_now + b
        p.grad = g.clone()
        opt.step()
        grad_fns.append((lambda yv, gg=g: gg.numpy()))   # replay the exact grad

    x_ref, y_ref, z_ref = _ref_schedulefree_1d(
        p0.numpy(), grad_fns, lr=lr, beta1=beta1, beta2=beta2, eps=eps,
        weight_decay=weight_decay, inner_momentum=inner_momentum,
    )

    # In train mode, p == y. (kaon keeps the 2nd-moment / z state in fp32 internally,
    # so the match against the fp64 reference is at fp32 precision.)
    np.testing.assert_allclose(p.detach().numpy(), y_ref, rtol=1e-5, atol=1e-6)
    # eval() exposes x (the averaged / kept sequence).
    opt.eval()
    np.testing.assert_allclose(p.detach().numpy(), x_ref, rtol=1e-5, atol=1e-6)
    # z is recoverable: x = y + (1/beta1)(z-y) was used; check the stored z too.
    z_stored = opt.state[p]["z"].double().numpy()
    np.testing.assert_allclose(z_stored, z_ref, rtol=1e-5, atol=1e-6)


def test_train_eval_roundtrip_no_drift():
    """eval() then train() returns the buffer to the y-view unchanged (no drift)."""
    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(8, 4)), torch.nn.Parameter(torch.randn(5))]
    opt = ScheduleFree(params, lr=2e-3, momentum_dtype="float32")
    opt.train()
    for _ in range(4):
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
    y_before = [p.detach().clone() for p in params]
    opt.eval()
    # eval moved p away from y (must differ once z != y).
    assert any(not torch.allclose(p, yb) for p, yb in zip(params, y_before, strict=True))
    opt.train()
    for p, yb in zip(params, y_before, strict=True):
        torch.testing.assert_close(p, yb, rtol=1e-6, atol=1e-6)


def test_train_eval_idempotent():
    """Repeated train()/eval() calls are no-ops (mode-guarded)."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(6, 3))
    opt = ScheduleFree([p], lr=2e-3, momentum_dtype="float32")
    opt.train()
    p.grad = torch.randn_like(p)
    opt.step()
    opt.eval()
    x1 = p.detach().clone()
    opt.eval()  # second eval -> no-op
    torch.testing.assert_close(p, x1)
    opt.train()
    y1 = p.detach().clone()
    opt.train()  # second train -> no-op
    torch.testing.assert_close(p, y1)


@pytest.mark.parametrize(
    "cfg",
    [
        dict(momentum_dtype="float32", betas=(0.9, 0.999), weight_decay=0.0,
             cautious=True, inner_momentum=0.0),
        dict(momentum_dtype="bfloat16", betas=(0.9, 0.999), weight_decay=0.02,
             cautious=True, inner_momentum=0.9),
        dict(momentum_dtype="int8", betas=(0.9, 0.99), weight_decay=0.01,
             cautious=False, inner_momentum=0.0),
        dict(momentum_dtype="int8", betas=(0.95, 0.999), weight_decay=0.0,
             cautious=True, inner_momentum=0.9),
        dict(momentum_dtype="4bit", betas=(0.9, 0.999), weight_decay=0.0,
             cautious=True, inner_momentum=0.0),
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
    oa = ScheduleFree(pa, lr=2e-3, foreach=True, bf16_method="none", **cfg)
    ob = ScheduleFree(pb, lr=2e-3, foreach=False, bf16_method="none", **cfg)
    oa.train()
    ob.train()
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
    # parity must hold after the eval swap too.
    oa.eval()
    ob.eval()
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
    oa = ScheduleFree(pa, lr=2e-3, momentum_dtype="int8", weight_decay=0.02,
                      foreach=True, foreach_stack_budget=120)
    ob = ScheduleFree(pb, lr=2e-3, momentum_dtype="int8", weight_decay=0.02, foreach=False)
    oa.train()
    ob.train()
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


def test_z_buffer_present_and_full_size():
    """ScheduleFree keeps a single full-size z buffer (plus factored v / full v)."""
    p2 = torch.nn.Parameter(torch.randn(8, 4))
    p1 = torch.nn.Parameter(torch.randn(5))
    opt = ScheduleFree([p2, p1], lr=2e-3, momentum_dtype="float32")
    for p in (p2, p1):
        p.grad = torch.randn_like(p)
    opt.step()
    assert opt.state[p2]["z"].shape == p2.shape
    assert "row" in opt.state[p2] and "col" in opt.state[p2]
    assert opt.state[p1]["z"].shape == p1.shape
    assert "v" in opt.state[p1]


def test_int8_z_is_one_byte_per_param():
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = ScheduleFree([p], lr=2e-3, momentum_dtype="int8")
    p.grad = torch.randn_like(p)
    opt.step()
    assert opt.state[p]["z"].numel() == p.numel()
    assert opt.state[p]["z"].dtype == torch.int8


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
def test_momentum_dtype_variants_construct_and_step(momentum_dtype):
    params = [
        torch.nn.Parameter(torch.randn(8, 4)),
        torch.nn.Parameter(torch.randn(5)),
        torch.nn.Parameter(torch.randn(3, 2, 3, 3)),
    ]
    opt = ScheduleFree(params, lr=2e-3, momentum_dtype=momentum_dtype)
    for _ in range(3):
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
    for p in params:
        assert torch.isfinite(p).all()


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
def test_checkpoint_roundtrip_preserves_momentum_dtype(momentum_dtype):
    """state_dict round-trip preserves z's stored dtype and resumes bit-exactly."""
    torch.manual_seed(0)
    params_a = [torch.nn.Parameter(torch.randn(8, 4)), torch.nn.Parameter(torch.randn(5))]
    opt_a = ScheduleFree(params_a, lr=2e-3, momentum_dtype=momentum_dtype, weight_decay=0.01)
    opt_a.train()
    for _ in range(3):
        for p in params_a:
            p.grad = torch.randn_like(p)
        opt_a.step()

    buf = io.BytesIO()
    torch.save(opt_a.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, weights_only=False)

    params_b = [torch.nn.Parameter(p.detach().clone()) for p in params_a]
    opt_b = ScheduleFree(params_b, lr=2e-3, momentum_dtype=momentum_dtype, weight_decay=0.01)
    opt_b.load_state_dict(sd)

    for p_a, p_b in zip(params_a, params_b, strict=True):
        assert opt_b.state[p_b]["z"].dtype == opt_a.state[p_a]["z"].dtype

    opt_b.train()
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


def test_bf16_weights_train_no_nan():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, 8, dtype=torch.bfloat16))
    opt = ScheduleFree([p], lr=2e-3, bf16_method="stochastic_rounding")
    opt.train()
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()
    assert torch.isfinite(p).all()
    opt.eval()
    assert torch.isfinite(p).all()


def test_kahan_runs():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 4, dtype=torch.bfloat16))
    opt = ScheduleFree([p], lr=2e-3, bf16_method="kahan", momentum_dtype="float32")
    opt.train()
    for _ in range(3):
        p.grad = torch.randn_like(p)
        opt.step()
    assert "shift" in opt.state[p]
    assert torch.isfinite(p).all()


def test_overfits_regression():
    """ScheduleFree drives a tiny MLP's training loss down on a fixed batch."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 8)
    )
    x = torch.randn(4, 16)
    y = torch.randn(4, 8)
    opt = ScheduleFree(model.parameters(), lr=4e-3)
    opt.train()
    losses = []
    for _ in range(120):
        opt.zero_grad()
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5


def test_warmup_schedule():
    """warmup_steps linearly ramps the effective LR; weighting uses lr_max."""
    p = torch.nn.Parameter(torch.randn(4))
    opt = ScheduleFree([p], lr=1e-2, warmup_steps=5)
    opt.train()
    g = opt.param_groups[0]
    seen = []
    for _ in range(3):
        p.grad = torch.randn_like(p)
        opt.step()
        seen.append(g["lr_max"])
    # lr_max strictly increases through warmup.
    assert seen[0] < seen[1] < seen[2]
    assert math.isclose(seen[0], 1e-2 * (1 / 5))


def test_invalid_args_rejected():
    p = [torch.nn.Parameter(torch.randn(3))]
    with pytest.raises(ValueError):
        ScheduleFree(p, lr=-1.0)
    with pytest.raises(ValueError):
        ScheduleFree(p, betas=(0.0, 0.999))   # beta1 must be > 0 (1/beta1 swap)
    with pytest.raises(ValueError):
        ScheduleFree(p, betas=(0.9, 1.0))
    with pytest.raises(ValueError):
        ScheduleFree(p, inner_momentum=1.0)
    with pytest.raises(ValueError):
        ScheduleFree(p, eps=-1e-8)
    with pytest.raises(ValueError):
        ScheduleFree(p, weight_decay=-0.1)
    with pytest.raises(ValueError):
        ScheduleFree(p, warmup_steps=-1)
    with pytest.raises(ValueError):
        ScheduleFree(p, momentum_dtype="fp8")
    with pytest.raises(ValueError):
        ScheduleFree(p, bf16_method="bogus")


def test_sparse_grad_rejected():
    p = torch.nn.Parameter(torch.randn(4))
    opt = ScheduleFree([p], lr=1e-3)
    p.grad = torch.randn(4).to_sparse()
    with pytest.raises(RuntimeError):
        opt.step()
