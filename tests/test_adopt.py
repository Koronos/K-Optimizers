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


@pytest.mark.parametrize("foreach", [False, True])
def test_late_first_grad_stays_finite(foreach):
    """A param whose first grad arrives after global step 1 must still init v = g^2."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8))
    opt = ADOPT(
        [p], lr=1e-2, cautious=False, gradient_centralization=False,
        momentum_dtype="float32", foreach=foreach,
    )
    for _ in range(3):
        opt.step()  # no grads — global clock advances, param untouched
    g0 = torch.randn(8)
    p.grad = g0.clone()
    opt.step()
    assert torch.isfinite(p).all()
    torch.testing.assert_close(opt.state[p]["v"], g0 * g0, rtol=0, atol=0)
    assert opt.state[p]["step"] == 1


def test_empty_grad_step_then_normal_no_nan():
    """Empty .step() must not perturb a factored weight; next steps match a no-gap run."""
    torch.manual_seed(1)
    shape = (4, 6)
    p0 = torch.randn(shape)
    g1 = torch.randn(shape)
    g2 = torch.randn(shape)

    p_ref = torch.nn.Parameter(p0.clone())
    opt_ref = ADOPT(
        [p_ref], lr=1e-2, clip=True, cautious=False, gradient_centralization=False,
        momentum_dtype="float32", foreach=False,
    )
    p_ref.grad = g1.clone()
    opt_ref.step()
    p_ref.grad = g2.clone()
    opt_ref.step()

    p = torch.nn.Parameter(p0.clone())
    opt = ADOPT(
        [p], lr=1e-2, clip=True, cautious=False, gradient_centralization=False,
        momentum_dtype="float32", foreach=False,
    )
    opt.step()  # no grad — must not corrupt the factored v init path
    p.grad = g1.clone()
    opt.step()
    p.grad = g2.clone()
    opt.step()

    assert torch.isfinite(p).all()
    assert torch.isfinite(opt.state[p]["row"]).all()
    assert torch.isfinite(opt.state[p]["col"]).all()
    torch.testing.assert_close(p.detach(), p_ref.detach(), rtol=0, atol=0)


def test_checkpoint_without_param_step_uses_group_step_for_clip():
    """<=0.7.11 checkpoints without state['step'] resume the clip at group['step'] - 1."""
    torch.manual_seed(0)
    shape = (4, 6)
    grads = [torch.randn(shape) for _ in range(60)]
    g_next = torch.randn(shape)

    p_ref = torch.nn.Parameter(torch.randn(shape))
    p_mig = torch.nn.Parameter(p_ref.detach().clone())
    kw = dict(
        lr=1e-2, clip=True, cautious=False, gradient_centralization=False,
        momentum_dtype="float32", foreach=False,
    )
    opt_ref = ADOPT([p_ref], **kw)
    opt_mig = ADOPT([p_mig], **kw)
    for g in grads:
        p_ref.grad = g.clone()
        p_mig.grad = g.clone()
        opt_ref.step()
        opt_mig.step()
    assert opt_ref.param_groups[0]["step"] == 60
    del opt_mig.state[p_mig]["step"]

    p_ref.grad = g_next.clone()
    p_mig.grad = g_next.clone()
    opt_ref.step()
    opt_mig.step()

    torch.testing.assert_close(p_mig.detach(), p_ref.detach(), rtol=0, atol=0)


def test_per_group_momentum_dtype_codec():
    """Different param groups resolve momentum storage dtype independently."""
    torch.manual_seed(2)
    p_bf16 = torch.nn.Parameter(torch.randn(4))
    p_fp32 = torch.nn.Parameter(torch.randn(4))
    opt = ADOPT(
        [{"params": [p_bf16], "momentum_dtype": "bfloat16"},
         {"params": [p_fp32], "momentum_dtype": "float32"}],
        lr=1e-3,
    )
    for p in (p_bf16, p_fp32):
        p.grad = torch.randn_like(p)
    opt.step()
    assert opt.state[p_bf16]["m"].dtype == torch.bfloat16
    assert opt.state[p_fp32]["m"].dtype == torch.float32


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
    oa = ADOPT(pa, foreach=True, **cfg)
    ob = ADOPT(pb, foreach=False, **cfg)
    gg = torch.Generator().manual_seed(grad_seed)
    for _ in range(steps):
        for a, b in zip(pa, pb, strict=True):
            grad = torch.randn(a.shape, generator=gg) * 0.02
            a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    return pa, pb, oa, ob


_SCALAR_CFGS = [
    {'lr': 0.001, 'betas': (0.9, 0.9999), 'momentum_dtype': 'float32'},
    {'lr': 0.001, 'betas': (0.9, 0.9999), 'momentum_dtype': 'bfloat16'},
    {'lr': 0.001, 'betas': (0.9, 0.9999), 'momentum_dtype': 'int8'},
    {'lr': 0.001, 'betas': (0.9, 0.9999), 'momentum_dtype': '4bit'},
    {'lr': 0.001, 'betas': (0.9, 0.9999), 'momentum_dtype': 'float32', 'weight_decay': 0.02},
    {'lr': 0.001, 'betas': (0.9, 0.9999), 'momentum_dtype': 'int8', 'weight_decay': 0.02},
    {'lr': 0.001, 'betas': (0.9, 0.9999), 'momentum_dtype': '4bit', 'weight_decay': 0.02},
    {'lr': 0.001, 'betas': (0.9, 0.9999), 'momentum_dtype': 'float32', 'clip': False},
    {'lr': 0.001, 'betas': (0.9, 0.9999), 'momentum_dtype': 'float32', 'cautious': False},
]


@pytest.mark.parametrize("cfg", _SCALAR_CFGS)
def test_foreach_scalar_0d_matches_per_param(cfg):
    """0-D scalars through the batched bucket are element-for-element equal to the
    per-parameter path (fp32 weights, so stochastic rounding is not in play).

    For a length-1 slice every per-slice reduction the batched code does must
    degenerate to the per-param scalar one (the RMS clip's ``norm/sqrt(1)``, the
    cautious mask's mean over one element, the int8 absmax over one element) — this
    test is the proof of that, not an assumption.
    """
    pa, pb, oa, _ob = _scalar_parity(cfg)
    for a, b in zip(pa, pb, strict=True):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)
    # State keeps the per-param layout: a 0-D param keeps 0-D buffers (checkpoint compat).
    for a in pa:
        if a.ndim == 0:
            assert oa.state[a]["v"].shape == a.shape


def test_foreach_scalar_0d_takes_batched_path(monkeypatch):
    """0-D scalars must actually ride the batched bucket, not silently fall back to
    the per-param loop (the LyCORIS use_scalar pathology this guards against)."""
    looped = []
    orig = ADOPT._step_one_param

    def spy(self, p, *args, **kwargs):
        looped.append(p)
        return orig(self, p, *args, **kwargs)

    monkeypatch.setattr(ADOPT, "_step_one_param", spy)
    params = _scalar_bag(_scalar_shapes())
    opt = ADOPT(params, foreach=True, lr=0.001, betas=(0.9, 0.9999))
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
    pa, pb, _oa, _ob = _scalar_parity({'lr': 0.001, 'betas': (0.9, 0.9999), 'momentum_dtype': 'float32'}, shapes=shapes, steps=8, seed=3, grad_seed=5)
    for a, b in zip(pa, pb, strict=True):
        if a.numel() == 1:
            torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)
        else:
            torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=1e-7)


def test_scalar_0d_checkpoint_roundtrip_across_paths():
    """A checkpoint saved mid-training by one path resumes bit-exactly on the other:
    the 0-D state layout is identical on both, in both directions."""
    cfg = {'lr': 0.001, 'betas': (0.9, 0.9999), 'momentum_dtype': 'int8', 'weight_decay': 0.02}
    for save_foreach, load_foreach in [(True, False), (False, True)]:
        pa = _scalar_bag(_scalar_shapes(), seed=29)
        oa = ADOPT(pa, foreach=save_foreach, **cfg)
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
        ob = ADOPT(pb, foreach=load_foreach, **cfg)
        ob.load_state_dict(ckpt["opt"])
        for step in range(4, 8):
            for plist, opt in ((pa, oa), (pb, ob)):
                for p, gr in zip(plist, grads[step], strict=True):
                    p.grad = gr.clone()
                opt.step()
        for a, b in zip(pa, pb, strict=True):
            torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)

# --------------------------------------------------------------- foreach view plan
def _plan_run_adopt(cache: bool) -> list[torch.Tensor]:
    """Five steps over a 0-D / 1-D / 2-D / conv bag, with a ``p.data`` rebind at step 3."""
    torch.manual_seed(0x5EED)
    shapes = [(), (), (1,), (5,), (5,), (4, 3), (4, 3), (2, 2, 3, 3), (2, 2, 3, 3)]
    g = torch.Generator().manual_seed(7)
    params = [torch.nn.Parameter(torch.randn(s, generator=g)) for s in shapes]
    opt = ADOPT(params, lr=1e-2, weight_decay=0.01, momentum_dtype="int8")
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
    ``tests/test_foreach_plan.py``; this is ADOPT's own tripwire.
    """
    on = _plan_run_adopt(cache=True)
    off = _plan_run_adopt(cache=False)
    for i, (a, b) in enumerate(zip(on, off, strict=True)):
        assert torch.equal(a, b), f"tensor {i} differs"
