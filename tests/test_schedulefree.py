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

from kaon import ScheduleFree, reseed_stochastic_rounding
from kaon import schedulefree as sf_module


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
    """foreach=True is element-for-element equal to the per-parameter path (fp32 weights).

    Each shape appears twice so `_store_z_stacked` runs with N>1 (the write-back
    that `_foreach_copy_` must actually land). Param order follows real buckets:
    all factored groups contiguous, then flat, matching foreach visit order so
    the replayed RNG feeds the same draw to each coordinate on the per-param path
    — without it a bf16 z would only agree in expectation.
    """
    def mk() -> list[torch.nn.Parameter]:
        return [
            torch.nn.Parameter(torch.randn(8, 4)),
            torch.nn.Parameter(torch.randn(8, 4)),
            torch.nn.Parameter(torch.randn(7, 3)),
            torch.nn.Parameter(torch.randn(7, 3)),
            torch.nn.Parameter(torch.randn(3, 2, 3, 3)),
            torch.nn.Parameter(torch.randn(3, 2, 3, 3)),
            torch.nn.Parameter(torch.randn(5)),
            torch.nn.Parameter(torch.randn(5)),
            torch.nn.Parameter(torch.randn(6)),
            torch.nn.Parameter(torch.randn(6)),
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
        # SR noise comes from kaon's own generator: re-seed it so both paths draw alike.
        reseed_stochastic_rounding()
        oa.step()
        reseed_stochastic_rounding()
        ob.step()
    # The weights are the contract this test has always pinned. How closely the two
    # paths' stored z can agree depends on its storage: a bf16 z is stochastically
    # rounded and the replayed RNG hands both paths the same draws, so it must land
    # bit for bit (a `_store_z_stacked` that never writes back fails here). An fp32 z
    # only differs by the stacked-vs-per-param reassociation ulps, and the quantized
    # codes can straddle a requant boundary, so those are left to the weights.
    md = cfg["momentum_dtype"]
    # Stacks of N=2 same-shape params reassociate fp32 ops (and the bf16 codec runs its
    # EMA in fp32 since this release), so weights agree to fp32 ulps on every path; the
    # bf16 z itself is bit-exact because the re-seeded SR generator feeds both paths.
    def _weights_agree(a: torch.Tensor, b: torch.Tensor) -> None:
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)

    for a, b in zip(pa, pb, strict=True):
        _weights_agree(a, b)
        if md == "bfloat16":
            assert torch.equal(oa.state[a]["z"], ob.state[b]["z"])
        elif md == "float32":
            torch.testing.assert_close(
                oa.state[a]["z"], ob.state[b]["z"], rtol=1e-6, atol=1e-7
            )
    oa.eval()
    ob.eval()
    for a, b in zip(pa, pb, strict=True):
        _weights_agree(a, b)


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
        # A bf16-stored z is written with stochastic rounding from kaon's own generator.
        # Re-seeding it before each step hands both runs the same draws, which is what
        # makes "bit-exact" a statement about the resumed *state* and not about noise.
        reseed_stochastic_rounding()
        opt_a.step()
        reseed_stochastic_rounding()
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


# ============================================================ 0-D scalar params
# LyCORIS ``use_scalar`` gates (and friends) are 0-D weights. They used to be gated
# out of the foreach path and stepped one-by-one, which is ~20 CUDA launches per
# scalar per step; they now ride the non-factored ``L == 1`` bucket as length-1
# views. These tests pin the three things that can silently break: the routing, the
# element-for-element parity with the per-param path, and the state layout (so a
# checkpoint crosses between the two paths).
def _scalar_shapes() -> list[tuple[int, ...]]:
    """0-D scalars plus shape-``(1,)`` bucket-mates.

    Every entry has ``numel() == 1``, so the whole bag lands in the same
    non-factored bucket — 0-D as length-1 views, ``(1,)`` as-is — which is exactly
    the mixed stacking the batched path has to get right.
    """
    return [(), (), (), (1,), (1,)]


def _scalar_bag(shapes, seed: int = 11) -> list[torch.nn.Parameter]:
    g = torch.Generator().manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(s, generator=g) * 0.05) for s in shapes]


def _scalar_parity(cfg, shapes=None, steps: int = 10, seed: int = 11, grad_seed: int = 7):
    """Drive one bag of params and one grad sequence through both paths.

    Returns ``(pa, pb, oa, ob)`` — ``a`` is ``foreach=True``, ``b`` is ``foreach=False``.
    """
    pa = _scalar_bag(_scalar_shapes() if shapes is None else shapes, seed)
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    oa = ScheduleFree(pa, foreach=True, **cfg)
    ob = ScheduleFree(pb, foreach=False, **cfg)
    gg = torch.Generator().manual_seed(grad_seed)
    for _ in range(steps):
        for a, b in zip(pa, pb, strict=True):
            grad = torch.randn(a.shape, generator=gg) * 0.02
            a.grad, b.grad = grad.clone(), grad.clone()
        # SR noise comes from kaon's own generator: re-seed it so both paths draw alike.
        reseed_stochastic_rounding()
        oa.step()
        reseed_stochastic_rounding()
        ob.step()
    return pa, pb, oa, ob


_SCALAR_CFGS = [
    {'lr': 0.0025, 'betas': (0.9, 0.999), 'momentum_dtype': 'float32'},
    {'lr': 0.0025, 'betas': (0.9, 0.999), 'momentum_dtype': 'bfloat16'},
    {'lr': 0.0025, 'betas': (0.9, 0.999), 'momentum_dtype': 'int8'},
    {'lr': 0.0025, 'betas': (0.9, 0.999), 'momentum_dtype': '4bit'},
    {'lr': 0.0025, 'betas': (0.9, 0.999), 'momentum_dtype': 'float32', 'weight_decay': 0.02},
    {'lr': 0.0025, 'betas': (0.9, 0.999), 'momentum_dtype': 'int8', 'weight_decay': 0.02},
    {'lr': 0.0025, 'betas': (0.9, 0.999), 'momentum_dtype': '4bit', 'weight_decay': 0.02},
    {'lr': 0.0025, 'betas': (0.9, 0.999), 'momentum_dtype': 'float32', 'inner_momentum': 0.9},
    {'lr': 0.0025, 'betas': (0.9, 0.999), 'momentum_dtype': 'int8', 'inner_momentum': 0.9, 'weight_decay': 0.02},
    {'lr': 0.0025, 'betas': (0.9, 0.999), 'momentum_dtype': 'float32', 'warmup_steps': 3},
    {'lr': 0.0025, 'betas': (0.9, 0.999), 'momentum_dtype': 'float32', 'cautious': False},
]


@pytest.mark.parametrize("cfg", _SCALAR_CFGS)
def test_foreach_scalar_0d_matches_per_param(cfg):
    """0-D scalars through the batched bucket are element-for-element equal to the
    per-parameter path (fp32 weights, so the *weight* write-back's stochastic rounding
    is not in play).

    For a length-1 slice every per-slice reduction the batched code does must
    degenerate to the per-param scalar one (the RMS clip's ``norm/sqrt(1)``, the
    cautious mask's mean over one element, the int8 absmax over one element) — this
    test is the proof of that, not an assumption.

    A bf16-stored ``z`` is stochastically rounded. Replaying the same RNG state before
    each optimizer step makes every coordinate consume the same draw on both paths.
    """
    torch.manual_seed(0)  # pins the stochastic-rounding draws (bf16 z) run to run
    pa, pb, oa, ob = _scalar_parity(cfg, steps=10)
    for a, b in zip(pa, pb, strict=True):
        assert torch.equal(a, b)
        assert torch.equal(oa.state[a]["z"], ob.state[b]["z"])
    # State keeps the per-param layout: a 0-D param keeps 0-D buffers (checkpoint compat).
    for a in pa:
        if a.ndim == 0:
            assert oa.state[a]["v"].shape == a.shape


def test_foreach_scalar_0d_takes_batched_path(monkeypatch):
    """0-D scalars must actually ride the batched bucket, not silently fall back to
    the per-param loop (the LyCORIS use_scalar pathology this guards against)."""
    looped = []
    orig = ScheduleFree._step_one_param

    def spy(self, p, *args, **kwargs):
        looped.append(p)
        return orig(self, p, *args, **kwargs)

    monkeypatch.setattr(ScheduleFree, "_step_one_param", spy)
    params = _scalar_bag(_scalar_shapes())
    opt = ScheduleFree(params, foreach=True, lr=0.0025, betas=(0.9, 0.999))
    gg = torch.Generator().manual_seed(3)
    for _ in range(3):
        for p in params:
            p.grad = torch.randn(p.shape, generator=gg) * 0.02
        opt.step()
    assert not looped, f"{len(looped)} params fell back to the per-param loop"


def test_foreach_scalar_0d_mixed_with_other_shapes():
    """0-D scalars mixed with 1-D / 2-D / conv params: the scalars stay bit-exact vs
    the per-param path and the other buckets keep their existing contract (the file's
    own parity tests own the exact claim for those; here they get a 1e-7 bound so this
    test is not hostage to value-dependent last-ulp drift)."""
    shapes = [(), (), (1,), (40,), (40,), (8, 16), (8, 16), (4, 4, 3, 3)]
    pa, pb, _oa, _ob = _scalar_parity({'lr': 0.0025, 'betas': (0.9, 0.999), 'momentum_dtype': 'float32'}, shapes=shapes, steps=8, seed=3, grad_seed=5)
    for a, b in zip(pa, pb, strict=True):
        if a.numel() == 1:
            torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)
        else:
            torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=1e-7)


def test_scalar_0d_checkpoint_roundtrip_across_paths():
    """A checkpoint saved mid-training by one path resumes bit-exactly on the other:
    the 0-D state layout is identical on both, in both directions."""
    cfg = {'lr': 0.0025, 'betas': (0.9, 0.999), 'momentum_dtype': 'int8', 'weight_decay': 0.02}
    for save_foreach, load_foreach in [(True, False), (False, True)]:
        pa = _scalar_bag(_scalar_shapes(), seed=29)
        oa = ScheduleFree(pa, foreach=save_foreach, **cfg)
        gg = torch.Generator().manual_seed(31)
        grads = [[torch.randn(p.shape, generator=gg) * 0.02 for p in pa] for _ in range(8)]
        for step in range(4):
            for p, gr in zip(pa, grads[step], strict=True):
                p.grad = gr.clone()
            oa.step()
        buf = io.BytesIO()
        torch.save({"opt": oa.state_dict(), "params": [p.detach().clone() for p in pa]}, buf)
        buf.seek(0)
        ckpt = torch.load(buf, weights_only=False)
        pb = [torch.nn.Parameter(t.clone()) for t in ckpt["params"]]
        ob = ScheduleFree(pb, foreach=load_foreach, **cfg)
        ob.load_state_dict(ckpt["opt"])
        for step in range(4, 8):
            for plist, opt in ((pa, oa), (pb, ob)):
                for p, gr in zip(plist, grads[step], strict=True):
                    p.grad = gr.clone()
                opt.step()
        for a, b in zip(pa, pb, strict=True):
            torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


def test_scalar_0d_train_eval_roundtrip():
    """train()/eval() swap the 0-D weights to the averaged x-iterate and back, and the
    swap is the same on both paths (the y<->x views are per-param but read the batched
    path's ``z``)."""
    pa, pb, oa, ob = _scalar_parity(dict(lr=2.5e-3, betas=(0.9, 0.999), momentum_dtype="float32"))
    for o in (oa, ob):
        o.eval()
    for a, b in zip(pa, pb, strict=True):
        assert a.ndim == b.ndim
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)
    y_a = [p.detach().clone() for p in pa]
    for o in (oa, ob):
        o.train()
    for a, b, y in zip(pa, pb, y_a, strict=True):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)
        assert not torch.equal(a.detach(), y)  # eval view really differed from train view


# ================================================ bf16 z storage: the z-step must survive
# ``z`` is stepped by ``lr_t*d`` and written back to its ``momentum_dtype`` storage on every
# step. At bf16 that step is normally FAR below the ULP of ``z`` itself, so a
# round-to-nearest write-back hands back the OLD value and the z-sequence freezes at its
# initialization — which the iterate average then averages happily. Nothing else in the
# optimizer reveals this (weights keep moving, losses keep dropping, everything is finite),
# so it has to be pinned on the displacement of the *stored* z.
_BF16Z_STEPS = 200
_BF16Z_LR = 1e-4


def _const_grad_z_displacement(momentum_dtype: str, shape, *, foreach: bool) -> torch.Tensor:
    """bf16 weights at magnitude 1.0 under a constant unit gradient; return ``z - z0``.

    Two params, so ``foreach=True`` really batches (it needs >= 2 eligible tensors).
    ``weight_decay=0`` keeps ``d`` independent of the iterate and gradient centralization
    is off (it would subtract the mean of a constant gradient, i.e. zero it), so
    ``d == g / sqrt(v_hat) ~ 1`` on both the factored and non-factored paths and the
    fp32-storage displacement is ``~ -steps*lr`` on every element. Returned flat over all
    elements of all params.
    """
    params = [torch.nn.Parameter(torch.ones(shape, dtype=torch.bfloat16)) for _ in range(2)]
    opt = ScheduleFree(
        params, lr=_BF16Z_LR, momentum_dtype=momentum_dtype, weight_decay=0.0,
        gradient_centralization=False, foreach=foreach,
    )
    opt.train()
    for _ in range(_BF16Z_STEPS):
        for p in params:
            p.grad = torch.ones_like(p)
        opt.step()
    return torch.cat([(opt.state[p]["z"].detach().float() - 1.0).reshape(-1) for p in params])


@pytest.mark.parametrize("shape", [(2048,), (32, 64)], ids=["flat1d", "factored2d"])
@pytest.mark.parametrize("foreach", [False, True], ids=["per_param", "foreach"])
def test_bf16_z_displacement_matches_fp32_storage(shape, foreach):
    """A bf16-stored z must travel, on average, as far as an fp32-stored z.

    200 steps at lr=1e-4 displace z by ~2e-2 in total, while one bf16 ULP just below
    ``|z| = 1`` is ~3.9e-3: *every* individual write is sub-ULP. Round-to-nearest
    therefore gives a mean displacement of exactly 0 (z never leaves 1.0); stochastic
    rounding is unbiased, and the mean over 4096 elements concentrates far inside the 20%
    band asserted here (per-element spread ~9e-3, ~1.4e-4 once averaged).
    """
    torch.manual_seed(0)
    ref = _const_grad_z_displacement("float32", shape, foreach=foreach).mean().item()
    # Re-seed: the reference run also consumes draws (its bf16 *weight* write-back is
    # stochastically rounded), so without this the measured bf16-z displacement would
    # depend on how many of them it happened to consume.
    torch.manual_seed(0)
    got = _const_grad_z_displacement("bfloat16", shape, foreach=foreach).mean().item()
    # Sanity: the reference is the -steps*lr*d (d ~ 1) walk this test's bound assumes.
    assert ref == pytest.approx(-_BF16Z_STEPS * _BF16Z_LR, rel=0.05)
    assert abs(got - ref) <= 0.2 * abs(ref), (
        f"bf16-stored z moved {got:.3e} vs {ref:.3e} at fp32 storage — sub-ULP z-steps "
        "are being lost to the round-to-nearest write-back"
    )


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
@pytest.mark.parametrize("foreach", [False, True], ids=["per_param", "foreach"])
def test_z_store_preserves_storage_identity(momentum_dtype, foreach):
    """Every z write-back lands IN the existing buffers — same object, same ``data_ptr``,
    same dtype and shape — on both paths. The stochastically-rounded bf16 write is the
    new one here; pointer caches over optimizer state must not be left dangling."""
    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.ones(64, dtype=torch.bfloat16)) for _ in range(2)]
    opt = ScheduleFree(params, lr=1e-3, momentum_dtype=momentum_dtype, foreach=foreach)
    opt.train()
    for p in params:
        p.grad = torch.ones_like(p)
    opt.step()
    keys = ["z"] + [k for k in opt.state[params[0]] if k.startswith("z_")]
    before = [
        {k: (id(opt.state[p][k]), opt.state[p][k].data_ptr(), opt.state[p][k].dtype)
         for k in keys if torch.is_tensor(opt.state[p][k])}
        for p in params
    ]
    for _ in range(3):
        for p in params:
            p.grad = torch.ones_like(p)
        opt.step()
    for p, snap in zip(params, before, strict=True):
        for k, (obj_id, ptr, dtype) in snap.items():
            t = opt.state[p][k]
            assert id(t) == obj_id, f"{k} was replaced, not written in place"
            assert t.data_ptr() == ptr, f"{k} moved storage"
            assert t.dtype == dtype


# ================================================ groups that skip a step (no gradients)
def test_group_without_grads_does_not_advance_its_step():
    """A group with no gradients this iteration must be a COMPLETE no-op.

    ``group['step']`` is the ``k`` that feeds ``_coeffs`` (the ``t**r`` average weight and
    the bias corrections), while ``lr_max`` / ``weight_sum`` advance only inside
    ``_coeffs`` — i.e. only on the steps the group actually took. Advancing ``step`` for a
    grad-less group desynchronizes the three: with ``r != 0`` the group's later steps land
    on the wrong ``t`` and its average weighting no longer matches its own ``weight_sum``.
    That is the frozen-then-unfrozen parameter case (and any conditionally-active branch).
    """
    cfg = dict(lr=1e-2, r=1.0, weight_lr_power=2.0, momentum_dtype="float32",
               gradient_centralization=False)
    torch.manual_seed(0)
    p_hot = torch.nn.Parameter(torch.randn(16))                              # always stepped
    late = [torch.nn.Parameter(torch.randn(16)) for _ in range(2)]           # grad-less first
    opt = ScheduleFree([{"params": [p_hot]}, {"params": late}], **cfg)
    # Reference: the same params under the same gradients, in an optimizer that only ever
    # sees the steps they have gradients for.
    ref = [torch.nn.Parameter(p.detach().clone()) for p in late]
    opt_ref = ScheduleFree(ref, **cfg)
    late_group = opt.param_groups[1]

    gen = torch.Generator().manual_seed(5)
    frozen = [p.detach().clone() for p in late]
    for _ in range(3):
        p_hot.grad = torch.randn(16, generator=gen)
        for p in late:
            p.grad = None
        opt.step()
    assert late_group["step"] == 0, "a grad-less group advanced its step counter"
    assert late_group["weight_sum"] == 0.0
    assert late_group["lr_max"] == -1.0
    for p, f in zip(late, frozen, strict=True):
        torch.testing.assert_close(p.detach(), f, rtol=0, atol=0)

    live = 4
    for _ in range(live):
        p_hot.grad = torch.randn(16, generator=gen)
        for p, q in zip(late, ref, strict=True):
            g = torch.randn(16, generator=gen)
            p.grad, q.grad = g.clone(), g.clone()
        opt.step()
        opt_ref.step()

    assert late_group["step"] == live == opt_ref.param_groups[0]["step"]
    # step / weight_sum / lr_max stay mutually consistent: weight_sum is exactly the sum
    # of t**r * lr_max**weight_lr_power over the steps the group actually took.
    assert late_group["weight_sum"] == pytest.approx(
        sum(t * (1e-2 ** 2.0) for t in range(1, live + 1))
    )
    assert late_group["lr_max"] == pytest.approx(1e-2)
    for p, q in zip(late, ref, strict=True):
        torch.testing.assert_close(p.detach(), q.detach(), rtol=0, atol=0)

# --------------------------------------------------------------- foreach view plan
def _plan_run_schedulefree(cache: bool) -> list[torch.Tensor]:
    """Five steps over a 0-D / 1-D / 2-D / conv bag, with a ``p.data`` rebind at step 3."""
    torch.manual_seed(0x5EED)
    shapes = [(), (), (1,), (5,), (5,), (4, 3), (4, 3), (2, 2, 3, 3), (2, 2, 3, 3)]
    g = torch.Generator().manual_seed(7)
    params = [torch.nn.Parameter(torch.randn(s, generator=g)) for s in shapes]
    opt = ScheduleFree(params, lr=1e-2, weight_decay=0.01, momentum_dtype="int8")
    opt._foreach_cache_enabled = cache
    for step in range(1, 6):
        gg = torch.Generator().manual_seed(100 + step)
        for p in params:
            p.grad = torch.randn(p.shape, generator=gg) * 0.05
        if step == 3:                       # fresh storage: only the witness catches it
            params[5].data = params[5].data.clone()
        opt.step()
    # The cached arm must actually have cached something: without this the whole test
    # passes on a tree where ``_foreach_cache_enabled`` is an inert attribute.
    assert not cache or opt._foreach_plans, "the cached arm cached no plan"
    out = []
    for p in params:
        out.append(p.detach().clone())
        out += [v.clone() for _, v in sorted(opt.state[p].items())
                if isinstance(v, torch.Tensor)]
    return out


def test_foreach_plan_cache_is_numerically_invisible():
    """The cached bucketing/view plan (``kaon._foreach_plan``) is a host-side
    optimization only: bit-identical weights and state with it on and off, including
    across a mid-run ``p.data`` rebind that only the plan's witness can see.

    Cross-optimizer coverage of the plan's six invalidation paths lives in
    ``tests/test_foreach_plan.py``; this is ScheduleFree's own tripwire.
    """
    on = _plan_run_schedulefree(cache=True)
    off = _plan_run_schedulefree(cache=False)
    for i, (a, b) in enumerate(zip(on, off, strict=True)):
        assert torch.equal(a, b), f"tensor {i} differs"


# ------------------------------------------------------- SR write-back routing (perf lock)
# Every bf16 stochastic-rounding write in the library goes through
# ``kaon._backend._sr_write_``, which prefers the one-launch, zero-temporary Triton kernel
# on CUDA and falls back to the torch reference elsewhere. ScheduleFree has two such
# writes that used to call ``add_stochastic_`` directly — the ``z`` write-back and the
# per-parameter ``y`` write-back — which cost ~7 extra kernels and two param-sized
# temporaries each. These lock the routing; the numerics are covered by
# ``test_bf16_z_displacement_matches_fp32_storage`` and the storage-identity tests.


def _spy_on_sr_write(monkeypatch):
    """Record every ``_sr_write_`` the ScheduleFree module makes; keep the real behaviour."""
    calls = []
    real = sf_module._sr_write_

    def spy(target, source, alpha, triton=None):
        calls.append((target.dtype, tuple(target.shape), alpha))
        return real(target, source, alpha, triton)

    monkeypatch.setattr(sf_module, "_sr_write_", spy)
    return calls


@pytest.mark.parametrize("foreach", [False, True], ids=["per_param", "foreach"])
@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32"])
def test_bf16_z_store_routes_through_sr_write(monkeypatch, foreach, momentum_dtype):
    """A bf16-stored ``z`` writes back through the shared SR primitive; an fp32 ``z``
    takes the plain codec write and must not touch it at all.

    The parameters are fp32, so the ``y`` write-back never takes an SR branch and every
    recorded call belongs to the ``z`` store.
    """
    torch.manual_seed(0)
    params = [torch.randn(6, 4, requires_grad=True) for _ in range(3)]
    opt = ScheduleFree(params, lr=1e-3, momentum_dtype=momentum_dtype, foreach=foreach)
    calls = _spy_on_sr_write(monkeypatch)
    for _ in range(2):
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
    if momentum_dtype == "float32":
        assert calls == [], "an fp32 z must not be stochastically rounded"
        return
    assert calls, "the bf16 z write-back bypassed kaon._backend._sr_write_"
    assert all(dtype is torch.bfloat16 for dtype, _shape, _alpha in calls)
    # foreach stacks the three same-shape params into one bucket; per-param writes each.
    assert len(calls) == (2 if foreach else 6)


def test_bf16_y_write_above_the_foreach_cutoff_routes_through_sr_write(monkeypatch):
    """A weight too large to batch falls to the per-parameter path even under
    ``foreach=True`` — its bf16 ``y`` write-back must still use the shared primitive.

    ``momentum_dtype="float32"`` keeps the ``z`` store off the SR path, so the recorded
    calls are exactly the ``y`` write-backs.
    """
    torch.manual_seed(0)
    small = torch.randn(32, 32, dtype=torch.bfloat16, requires_grad=True)
    big = torch.randn(64, 64, dtype=torch.bfloat16, requires_grad=True)
    opt = ScheduleFree(
        [small, big], lr=1e-3, momentum_dtype="float32", foreach=True,
        bf16_method="stochastic_rounding", foreach_batch_cutoff=2048,
    )
    calls = _spy_on_sr_write(monkeypatch)
    for p in (small, big):
        p.grad = torch.randn_like(p)
    opt.step()
    shapes = [shape for _dtype, shape, _alpha in calls]
    assert (64, 64) in shapes, (
        "the above-cutoff weight's bf16 write-back bypassed kaon._backend._sr_write_"
    )
    assert all(dtype is torch.bfloat16 for dtype, _shape, _alpha in calls)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton SR kernel needs CUDA")
def test_bf16_z_store_reaches_the_triton_sr_kernel():
    """On CUDA the z write-back must land in the fused kernel (one launch, no temporary),
    not in the multi-kernel torch fallback."""
    import kaon._fused_triton as ft

    if not ft.HAS_TRITON:
        pytest.skip("Triton not installed")
    torch.manual_seed(0)
    params = [torch.randn(6, 4, device="cuda", requires_grad=True) for _ in range(3)]
    opt = ScheduleFree(params, lr=1e-3, momentum_dtype="bfloat16", foreach=True)
    launched = []
    real = ft.sr_add_
    try:
        ft.sr_add_ = lambda target, source, alpha=1.0: (
            launched.append(target.numel()), real(target, source, alpha))[1]
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
    finally:
        ft.sr_add_ = real
    assert launched, "the bf16 z store did not reach kaon._fused_triton.sr_add_"
