"""Tests for the KProdigy optimizer.

Covers: parameter-free D-adaptation, numerical parity with the reference
``prodigyopt.Prodigy`` (at the exact defaults), the memory variants
(bf16/int8 momentum, factored second moment, sliced D stats), bf16-weight
stochastic-rounding updates, and independent-D for multi-group (SDXL-like)
setups.
"""

from __future__ import annotations

import io
import math

import pytest
import torch

from kaon import KProdigy

from .conftest import train_steps


def _regression_problem(out: int = 32, inp: int = 16, n: int = 256):
    torch.manual_seed(0)
    w_true = torch.randn(out, inp)
    torch.manual_seed(7)
    x = torch.randn(n, inp)
    y = x @ w_true.T
    return x, y


def _run(opt_factory, steps: int = 80, pdtype=torch.float32):
    x, y = _regression_problem()
    lin = torch.nn.Linear(16, 32, bias=False).to(pdtype)
    opt = opt_factory(lin)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(lin(x.to(pdtype)).float(), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses, opt


# -- basics ----------------------------------------------------------------

def test_init_and_step(toy_mlp, random_batch):
    opt = KProdigy(toy_mlp.parameters(), lr=1.0)
    train_steps(toy_mlp, opt, [random_batch])
    assert opt.get_d() >= opt.param_groups[0]["d0"]


def test_sane_defaults():
    """The original repo's footguns must NOT be the defaults."""
    opt = KProdigy([torch.zeros(1, requires_grad=True)])
    g = opt.param_groups[0]
    assert g["d_update_freq"] == 1            # not 5 (which starves D)
    assert g["use_bias_correction"] is False  # not True (which hurt convergence)
    assert g["momentum_dtype"] == "bfloat16"
    assert g["bf16_method"] == "stochastic_rounding"


@pytest.mark.parametrize("bad", [
    {"lr": 0.0}, {"d0": 0.0}, {"eps": 0.0}, {"betas": (1.0, 0.9)},
    {"betas": (0.9, 1.0)}, {"d_update_freq": 0}, {"slice_p": 0},
    {"momentum_dtype": "fp8"}, {"second_moment": "low_rank"}, {"bf16_method": "magic"},
])
def test_invalid_args(bad):
    with pytest.raises(ValueError):
        KProdigy([torch.zeros(1, requires_grad=True)], **bad)


def test_d_rises_and_converges():
    losses, opt = _run(lambda m: KProdigy(m.parameters(), lr=1.0, gradient_centralization=False))
    assert opt.get_d() > 10 * opt.param_groups[0]["d0"]   # D bootstrapped
    assert losses[-1] < 0.05 * losses[0]                  # converged


# -- parity with reference Prodigy -----------------------------------------

def test_parity_with_reference_prodigy():
    """fp32 momentum + full second moment must match konstmish Prodigy."""
    prodigyopt = pytest.importorskip("prodigyopt")

    lp, ref = _run(lambda m: prodigyopt.Prodigy(m.parameters(), lr=1.0, use_bias_correction=False))
    lk, kp = _run(lambda m: KProdigy(m.parameters(), lr=1.0, gradient_centralization=False, momentum_dtype="float32", second_moment="full"))

    d_ref, d_kp = ref.param_groups[0]["d"], kp.get_d()
    assert abs(d_ref - d_kp) / d_ref < 5e-3            # D estimate matches
    assert abs(lp[-1] - lk[-1]) / max(lp[-1], 1e-9) < 0.05  # loss matches


# -- memory variants -------------------------------------------------------

@pytest.mark.parametrize("momentum_dtype", ["float32", "bfloat16", "int8"])
def test_momentum_dtypes_converge(momentum_dtype):
    losses, opt = _run(lambda m: KProdigy(m.parameters(), lr=1.0, gradient_centralization=False, momentum_dtype=momentum_dtype))
    assert losses[-1] < 0.05 * losses[0]


def test_momentum_buffer_dtype():
    for md, dt in [("float32", torch.float32), ("bfloat16", torch.bfloat16), ("int8", torch.int8)]:
        _, opt = _run(lambda m, x=md: KProdigy(m.parameters(), lr=1.0, gradient_centralization=False, momentum_dtype=x), steps=2)
        p = opt.param_groups[0]["params"][0]
        assert opt.state[p]["m"].dtype == dt


def test_factored_second_moment_converges_and_saves_state():
    losses, opt = _run(lambda m: KProdigy(m.parameters(), lr=1.0, gradient_centralization=False, second_moment="factored"))
    assert losses[-1] < 0.1 * losses[0]
    p = opt.param_groups[0]["params"][0]
    state = opt.state[p]
    assert "row" in state and "col" in state and "v" not in state
    # factored stores R + C floats instead of R * C
    assert state["row"].numel() + state["col"].numel() < p.numel()


def test_no_momentum_minimum_state():
    _, opt = _run(lambda m: KProdigy(m.parameters(), lr=1.0, gradient_centralization=False, betas=(0.0, 0.999)), steps=2)
    p = opt.param_groups[0]["params"][0]
    assert "m" not in opt.state[p]


def test_slice_p_reduces_d_state():
    _, opt = _run(lambda m: KProdigy(m.parameters(), lr=1.0, gradient_centralization=False, slice_p=11), steps=2)
    p = opt.param_groups[0]["params"][0]
    assert opt.state[p]["s"].numel() <= p.numel() // 10 + 1


# -- bf16 weights ----------------------------------------------------------

def test_bf16_weights_stochastic_rounding_makes_progress():
    """With bf16 weights and d0=1e-6, naive rounding stalls; SR must not."""
    losses_sr, opt_sr = _run(
        lambda m: KProdigy(m.parameters(), lr=1.0, gradient_centralization=False, bf16_method="stochastic_rounding"),
        pdtype=torch.bfloat16,
    )
    losses_none, _ = _run(
        lambda m: KProdigy(m.parameters(), lr=1.0, gradient_centralization=False, bf16_method="none"),
        pdtype=torch.bfloat16,
    )
    assert losses_sr[-1] < 0.1 * losses_sr[0]            # SR converges
    assert losses_none[-1] > 0.5 * losses_none[0]        # naive rounding stalls
    assert opt_sr.get_d() > 10 * opt_sr.param_groups[0]["d0"]


def test_bf16_weights_kahan_converges():
    losses, _ = _run(lambda m: KProdigy(m.parameters(), lr=1.0, gradient_centralization=False, bf16_method="kahan"), pdtype=torch.bfloat16)
    assert losses[-1] < 0.1 * losses[0]


# -- independent D (multi-group / SDXL) ------------------------------------

def test_independent_d_auto_and_per_group():
    x, y = _regression_problem()
    a = torch.nn.Linear(16, 24, bias=False)
    b = torch.nn.Linear(24, 32, bias=False)
    opt = KProdigy([{"params": a.parameters(), "lr": 1.0},
                    {"params": b.parameters(), "lr": 1.0}])
    assert opt._independent_d is True
    for _ in range(60):
        opt.zero_grad()
        torch.nn.functional.mse_loss(b(a(x)), y).backward()
        opt.step()
    d0, d1 = opt.param_groups[0]["d"], opt.param_groups[1]["d"]
    assert d0 > opt.param_groups[0]["d0"] and d1 > opt.param_groups[1]["d0"]


def test_independent_d_override_off_requires_equal_lr():
    a = torch.nn.Linear(8, 8, bias=False)
    b = torch.nn.Linear(8, 8, bias=False)
    opt = KProdigy([{"params": a.parameters(), "lr": 1.0},
                    {"params": b.parameters(), "lr": 0.5}], independent_d=False)
    x = torch.randn(4, 8)
    opt.zero_grad()
    (b(a(x))).pow(2).mean().backward()
    with pytest.raises(RuntimeError):  # shared-D scope forbids unequal nonzero lr
        opt.step()


def test_state_dict_roundtrip():
    _, opt = _run(lambda m: KProdigy(m.parameters(), lr=1.0, gradient_centralization=False), steps=5)
    sd = opt.state_dict()
    p = opt.param_groups[0]["params"][0]
    opt2 = KProdigy([p], lr=1.0)
    opt2.load_state_dict(sd)
    assert opt2.param_groups[0]["d"] == opt.param_groups[0]["d"]


# -- Adakaon-engine update backend (foreach) -----------------------------

# D stays pinned at d0 unless the estimator works, which would make every
# per-param vs foreach comparison below hold vacuously. The parity tests assert
# the trajectory climbed by at least this factor over their 30 steps (the
# fixture reached ~1.6x when it was measured, so the bar is deliberately loose).
_D_MIN_GROWTH = 1.2


def _assert_d_climbed(ds, d0):
    """The D trajectory has to move, or the parity assertions prove nothing.

    The midpoint is checked too: growth concentrated in the last step (or a
    trajectory that was already at its ceiling) would not exercise the estimator
    over the run. D is non-decreasing by construction here (``growth_rate=inf``
    caps each step at ``d_max``), so the midpoint bound is the informative half.
    """
    assert ds[-1] > _D_MIN_GROWTH * d0
    assert d0 < ds[len(ds) // 2] <= ds[-1]


def _assert_params_close(a, b, rtol=1e-6):
    """Per-param parity, relative, with the absolute floor tied to the scale.

    At ``lr=1.0`` these weights reach ~1e4 after 30 steps of a consistently
    directed gradient, where a fixed ``atol=1e-7`` asserts nothing at all.
    """
    ref = b.detach()
    torch.testing.assert_close(
        a.detach(), ref, rtol=rtol, atol=rtol * float(ref.abs().max())
    )


def _mixed_params(dtype=torch.float32, device="cpu"):
    """2-D + conv (4-D) + 1-D params -> exercises factored, full and flat buckets."""
    g = torch.Generator(device=device).manual_seed(0)
    shapes = [
        (32, 16), (24, 12), (8, 4, 3, 3), (16,), (32,), (10, 5, 1, 1),
        # numel % 16 != 0: low-precision elementwise kernels split into a
        # vectorized body plus a scalar tail, and the two round differently, so
        # a tensor that is batched into a stack must not change value.
        (3, 7), (5, 11), (127,),
    ]
    return [
        torch.nn.Parameter(torch.randn(*s, generator=g, dtype=dtype, device=device) * 0.1)
        for s in shapes
    ]


def _run_kprodigy(ps, *, foreach, steps=30, **kw):
    """D trajectory only; use ``_run_kprodigy_opt`` when the state matters."""
    return _run_kprodigy_opt(ps, foreach=foreach, steps=steps, **kw)[0]


def _run_kprodigy_opt(ps, *, foreach, steps=30, **kw):
    opt = KProdigy(ps, lr=1.0, **{"foreach": foreach, **kw})
    g = torch.Generator(device=ps[0].device).manual_seed(123)
    # A fixed, consistently directed gradient makes <g, p0-p> positive after
    # the first update, so this fixture exercises the D estimator rather than
    # accidentally pinning D at d0.
    directions = [
        torch.randn(p.shape, generator=g, dtype=p.dtype, device=p.device) * 0.5
        for p in ps
    ]
    ds = []
    for _ in range(steps):
        for p, grad in zip(ps, directions, strict=True):
            p.grad = grad.clone()
        opt.step()
        ds.append(opt.get_d())
    return ds, opt


@pytest.mark.parametrize("momentum_dtype", ["float32", "bfloat16", "int8", "4bit"])
@pytest.mark.parametrize("second_moment", ["full", "factored"])
@pytest.mark.parametrize("cautious", [False, True])
def test_foreach_matches_per_param(momentum_dtype, second_moment, cautious):
    """The engine-backed update agrees with the per-param path while D moves."""
    base = _mixed_params()
    pa = [torch.nn.Parameter(p.detach().clone()) for p in base]
    pb = [torch.nn.Parameter(p.detach().clone()) for p in base]
    kw = dict(
        momentum_dtype=momentum_dtype,
        second_moment=second_moment,
        cautious=cautious,
        d0=1e-6,
    )
    d_pp = _run_kprodigy(pa, foreach=False, **kw)
    d_fe = _run_kprodigy(pb, foreach=True, **kw)
    _assert_d_climbed(d_pp, kw["d0"])
    _assert_d_climbed(d_fe, kw["d0"])
    for pp_step, fe_step in zip(d_pp, d_fe, strict=True):
        assert pp_step == pytest.approx(fe_step, rel=1e-6, abs=0)
    for a, b in zip(pa, pb, strict=True):
        _assert_params_close(a, b)


@pytest.mark.parametrize("momentum_dtype", ["float32", "bfloat16", "int8", "4bit"])
@pytest.mark.parametrize("second_moment", ["full", "factored"])
@pytest.mark.parametrize("slice_p", [1, 11])
def test_pass1_foreach_matches_per_param(momentum_dtype, second_moment, slice_p):
    """Batched pass 1 tracks the per-param D trajectory within one ppm."""
    base = _mixed_params()
    pa = [torch.nn.Parameter(p.detach().clone()) for p in base]
    pb = [torch.nn.Parameter(p.detach().clone()) for p in base]
    kw = dict(
        momentum_dtype=momentum_dtype,
        second_moment=second_moment,
        slice_p=slice_p,
        d0=1e-6,
    )
    d_pp = _run_kprodigy(pa, foreach=False, **kw)
    d_fe = _run_kprodigy(pb, foreach=True, **kw)
    _assert_d_climbed(d_pp, kw["d0"])
    _assert_d_climbed(d_fe, kw["d0"])
    assert d_pp == pytest.approx(d_fe, rel=1e-6, abs=0)
    for a, b in zip(pa, pb, strict=True):
        _assert_params_close(a, b)


@pytest.mark.parametrize("second_moment", ["full", "factored"])
def test_bf16_momentum_ema_is_invariant_to_bucket_size(second_moment):
    """bf16 momentum must not change when a tensor is folded into a stack.

    Two conditions have to hold at once for the in-bf16 EMA to be caught, and
    the shared fixture only meets the first: ``numel % 16 != 0`` (CPU
    low-precision elementwise kernels take a vectorized body plus a scalar tail,
    which round differently) AND at least two params per shape, since
    ``_pass1_momentum_foreach`` buckets by ``(momentum_dtype, shape)`` and a
    bucket of one stacks to the same numel as the lone tensor.
    """
    shapes = [(3, 7), (3, 7), (5, 11), (5, 11), (127,), (127,)]
    g = torch.Generator().manual_seed(7)
    base = [torch.nn.Parameter(torch.randn(*s, generator=g) * 0.1) for s in shapes]
    pa = [torch.nn.Parameter(p.detach().clone()) for p in base]
    pb = [torch.nn.Parameter(p.detach().clone()) for p in base]
    kw = dict(momentum_dtype="bfloat16", second_moment=second_moment, d0=1e-6)
    d_pp, opt_pp = _run_kprodigy_opt(pa, foreach=False, **kw)
    d_fe, opt_fe = _run_kprodigy_opt(pb, foreach=True, **kw)

    # The stored momentum is the quantity the EMA rounds, so it is compared bit
    # for bit; D and the weights inherit the pass-1 reduction tolerance.
    for a, b in zip(pa, pb, strict=True):
        ma, mb = opt_pp.state[a]["m"], opt_fe.state[b]["m"]
        assert ma.dtype == torch.bfloat16
        assert float(ma.abs().max()) > 0.0  # the EMA ran at all
        torch.testing.assert_close(mb, ma, rtol=0, atol=0)
    _assert_d_climbed(d_pp, kw["d0"])
    for pp_step, fe_step in zip(d_pp, d_fe, strict=True):
        assert pp_step == pytest.approx(fe_step, rel=1e-6, abs=0)
    for a, b in zip(pa, pb, strict=True):
        _assert_params_close(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pass1_foreach_cuda_d_is_stable_for_30_steps():
    """CUDA's batched reduction tracks the per-param D for 30 steps of growth."""
    base = _mixed_params(device="cuda")
    pa = [torch.nn.Parameter(p.detach().clone()) for p in base]
    pb = [torch.nn.Parameter(p.detach().clone()) for p in base]
    kw = dict(momentum_dtype="bfloat16", second_moment="full", d0=1e-6)
    d_pp = _run_kprodigy(pa, foreach=False, steps=30, **kw)
    d_fe = _run_kprodigy(pb, foreach=True, steps=30, **kw)
    assert all(math.isfinite(d) for d in d_fe)
    _assert_d_climbed(d_fe, kw["d0"])
    # Every step, not just the last: a path that diverged and reconverged would
    # otherwise slip through.
    for pp_step, fe_step in zip(d_pp, d_fe, strict=True):
        assert pp_step == pytest.approx(fe_step, rel=1e-6, abs=0)


@pytest.mark.parametrize("independent_d", [True, False])
def test_pass1_foreach_matches_per_param_multigroup(independent_d):
    """Pass-1 batching keeps the per-group D accumulation order-equivalent: the
    foreach and per-param paths agree within one ppm for a multi-group
    (SDXL-like) setup, with both global and independent D."""
    base = _mixed_params()
    g1, g2 = base[:3], base[3:]

    def build():
        a = [torch.nn.Parameter(p.detach().clone()) for p in g1]
        b = [torch.nn.Parameter(p.detach().clone()) for p in g2]
        return a, b

    def run(a, b, foreach):
        opt = KProdigy(
            [{"params": a, "lr": 1.0}, {"params": b, "lr": 1.0}],
            lr=1.0, foreach=foreach, independent_d=independent_d, slice_p=11, d0=1e-6,
        )
        gen = torch.Generator().manual_seed(99)
        # Consistently directed, as in _run_kprodigy: per-step random gradients
        # leave D sitting at d0 and the comparison below would hold vacuously.
        directions = [
            torch.randn(p.shape, generator=gen, dtype=p.dtype) * 0.5 for p in a + b
        ]
        ds = []
        for _ in range(30):
            for p, grad in zip(a + b, directions, strict=True):
                p.grad = grad.clone()
            opt.step()
            ds.append(tuple(grp["d"] for grp in opt.param_groups))
        return ds

    a1, b1 = build()
    a2, b2 = build()
    d_pp = run(a1, b1, foreach=False)
    d_fe = run(a2, b2, foreach=True)
    for i in range(len(d_fe[-1])):
        _assert_d_climbed([step[i] for step in d_fe], 1e-6)
    for pp_step, fe_step in zip(d_pp, d_fe, strict=True):
        assert pp_step == pytest.approx(fe_step, rel=1e-6, abs=0)
    for x, y in zip(a1 + b1, a2 + b2, strict=True):
        _assert_params_close(x, y)


def test_foreach_4bit_cautious_converges():
    """The new 4bit momentum + cautious path trains and bootstraps D."""
    losses, opt = _run(
        lambda m: KProdigy(m.parameters(), lr=1.0, gradient_centralization=False, momentum_dtype="4bit", cautious=True)
    )
    assert losses[-1] < 0.1 * losses[0]
    assert opt.get_d() > 10 * opt.param_groups[0]["d0"]


def test_4bit_momentum_is_half_byte_per_param():
    p = torch.nn.Parameter(torch.randn(64, 64))
    opt = KProdigy([p], lr=1.0, momentum_dtype="4bit", momentum_4bit_block=128)
    p.grad = torch.randn_like(p)
    opt.step()
    st = opt.state[p]
    assert st["m"].dtype == torch.uint8
    assert st["m"].numel() == (p.numel() + 1) // 2          # 0.5 B/param packed


def test_invalid_momentum_4bit_accepted():
    KProdigy([torch.zeros(1, requires_grad=True)], lr=1.0, momentum_dtype="4bit")


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "int8", "4bit"])
def test_checkpoint_roundtrip_preserves_momentum_dtype(momentum_dtype):
    """A torch.save/load checkpoint resumes BIT-EXACTLY, keeps the configured
    momentum dtype, and restores the D estimate.

    torch's default ``load_state_dict`` upcasts state tensors to the param's dtype
    (fp32), silently inflating quantized momentum back to fp32 on resume.
    ``KProdigy`` overrides ``load_state_dict`` to restore the stored dtype; the D
    bookkeeping (``d``/``d_numerator``/...) rides along in ``param_groups``.
    """
    torch.manual_seed(0)
    p_ref = torch.randn(16, 8)
    grads = [torch.randn(16, 8) for _ in range(10)]

    a = torch.nn.Parameter(p_ref.clone())
    opt_a = KProdigy([a], lr=1.0, momentum_dtype=momentum_dtype)
    for g in grads[:5]:
        a.grad = g.clone()
        opt_a.step()

    buf = io.BytesIO()
    torch.save(opt_a.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, weights_only=False)

    b = torch.nn.Parameter(a.detach().clone())
    opt_b = KProdigy([b], lr=1.0, momentum_dtype=momentum_dtype)
    opt_b.load_state_dict(sd)

    assert opt_b.state[b]["m"].dtype == opt_a.state[a]["m"].dtype
    assert opt_b.get_d() == opt_a.get_d()  # D estimate restored (lives in param_groups)

    for g in grads[5:]:
        a.grad = g.clone()
        opt_a.step()
        b.grad = g.clone()
        opt_b.step()
    assert torch.equal(a, b), "resumed run must continue bit-exactly"


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
    oa = KProdigy(pa, foreach=True, **cfg)
    ob = KProdigy(pb, foreach=False, **cfg)
    gg = torch.Generator().manual_seed(grad_seed)
    for _ in range(steps):
        for a, b in zip(pa, pb, strict=True):
            grad = torch.randn(a.shape, generator=gg) * 0.02
            a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    return pa, pb, oa, ob


_SCALAR_CFGS = [
    {'lr': 1.0, 'betas': (0.9, 0.999), 'momentum_dtype': 'float32'},
    {'lr': 1.0, 'betas': (0.9, 0.999), 'momentum_dtype': 'bfloat16'},
    {'lr': 1.0, 'betas': (0.9, 0.999), 'momentum_dtype': 'int8'},
    {'lr': 1.0, 'betas': (0.9, 0.999), 'momentum_dtype': '4bit'},
    {'lr': 1.0, 'betas': (0.9, 0.999), 'momentum_dtype': 'float32', 'weight_decay': 0.02},
    {'lr': 1.0, 'betas': (0.9, 0.999), 'momentum_dtype': 'int8', 'weight_decay': 0.02},
    {'lr': 1.0, 'betas': (0.9, 0.999), 'momentum_dtype': '4bit', 'weight_decay': 0.02},
    {'lr': 1.0, 'betas': (0.0, 0.999), 'momentum_dtype': 'float32'},
    {'lr': 1.0, 'betas': (0.9, 0.999), 'momentum_dtype': 'float32', 'second_moment': 'factored'},
    {'lr': 1.0, 'betas': (0.9, 0.999), 'momentum_dtype': 'int8', 'second_moment': 'factored'},
    {'lr': 1.0, 'betas': (0.9, 0.999), 'momentum_dtype': 'float32', 'cautious': True},
    {'lr': 1.0, 'betas': (0.9, 0.999), 'momentum_dtype': 'float32', 'weight_decay': 0.02, 'decouple': False},
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
    orig = KProdigy._update_one_param

    def spy(self, p, *args, **kwargs):
        looped.append(p)
        return orig(self, p, *args, **kwargs)

    monkeypatch.setattr(KProdigy, "_update_one_param", spy)
    params = _scalar_bag(_scalar_shapes())
    opt = KProdigy(params, foreach=True, lr=1.0, betas=(0.9, 0.999))
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
    pa, pb, _oa, _ob = _scalar_parity({'lr': 1.0, 'betas': (0.9, 0.999), 'momentum_dtype': 'float32'}, shapes=shapes, steps=8, seed=3, grad_seed=5)
    for a, b in zip(pa, pb, strict=True):
        if a.numel() == 1:
            torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)
        else:
            torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=1e-7)


def test_scalar_0d_checkpoint_roundtrip_across_paths():
    """A checkpoint saved mid-training by one path resumes bit-exactly on the other:
    the 0-D state layout is identical on both, in both directions."""
    cfg = {'lr': 1.0, 'betas': (0.9, 0.999), 'momentum_dtype': 'int8', 'weight_decay': 0.02}
    for save_foreach, load_foreach in [(True, False), (False, True)]:
        pa = _scalar_bag(_scalar_shapes(), seed=29)
        oa = KProdigy(pa, foreach=save_foreach, **cfg)
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
        ob = KProdigy(pb, foreach=load_foreach, **cfg)
        ob.load_state_dict(ckpt["opt"])
        for step in range(4, 8):
            for plist, opt in ((pa, oa), (pb, ob)):
                for p, gr in zip(plist, grads[step], strict=True):
                    p.grad = gr.clone()
                opt.step()
        for a, b in zip(pa, pb, strict=True):
            torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


def test_nd_adafactor_conv_falls_back_to_the_per_param_path():
    """``second_moment="factored"`` + ``factor_conv_as_matrix=False`` on an ``ndim > 2``
    weight must step, and match the per-parameter path.

    That configuration keeps an **N-D** Adafactor pair (``row`` is ``[O, I, kh]`` for a
    4-D kernel, not ``[R]``), which the batched bucket cannot work in — it operates on
    the matrixized ``[N, R, C]`` layout. Regression: the parameter was admitted to the
    batched path anyway and the bucket unpacked ``R, C = eff`` on a rank-4 tuple,
    raising ``ValueError: too many values to unpack``. It now takes the per-parameter
    path, which has always handled it.
    """
    torch.manual_seed(0)
    shape = (2, 2, 3, 3)
    base = [torch.randn(shape) for _ in range(3)]
    fast = [torch.nn.Parameter(t.clone()) for t in base]
    ref = [torch.nn.Parameter(t.clone()) for t in base]
    kw = dict(lr=1.0, d0=1e-2, second_moment="factored", factor_conv_as_matrix=False,
              weight_decay=0.01, bf16_method="none")
    opt = KProdigy(fast, foreach=True, **kw)
    opt_ref = KProdigy(ref, foreach=False, **kw)
    for step in range(1, 4):
        g = torch.Generator().manual_seed(100 + step)
        for p, r in zip(fast, ref, strict=True):
            raw = torch.randn(shape, generator=g).mul_(0.1)
            p.grad, r.grad = raw.clone(), raw.clone()
        opt.step()
        opt_ref.step()
    assert opt._foreach_plans == {}, "the N-D Adafactor conv must not be batched"
    # Pass 1 stays batched (it is a global reduction, not this bucketing), and its
    # ``[B, ...]`` reduction tree is not the per-tensor one — the module's contract is
    # agreement to 1e-6 relative, which is what is checked here.
    assert math.isclose(opt.get_d(), opt_ref.get_d(), rel_tol=1e-6)
    for p, r in zip(fast, ref, strict=True):
        torch.testing.assert_close(p.detach(), r.detach(), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("layout", ["channels_last", "permute"])
def test_non_contiguous_conv_falls_back_under_factored(layout):
    """A non-contiguous 4-D weight must not take the *factored* batched path.

    ``second_moment="factored"`` + ``factor_conv_as_matrix=True`` matrixizes a conv, so
    the bucket reads and writes through ``p.data.view(R, C)`` — which a channels_last or
    permuted kernel does not have (``view`` raises ``view size is not compatible with
    input tensor's size and stride``). The ``ndim > 2`` contiguity guard in
    ``_param_foreach_eligible`` is what keeps them on the per-parameter path.

    **Three params minimum**: ``_apply_updates`` only calls ``_update_foreach`` when the
    eligible list has >= 2 entries, so a smaller bag would pass with the guard deleted.

    ``second_moment="full"`` is deliberately NOT guarded — its bucket keeps the weight in
    its own layout (``ForeachSpec(matrixize=False)``) and batches it fine; that is
    covered by ``test_foreach_plan.py::test_non_contiguous_conv_still_batches_without_matrixize``.
    """
    torch.manual_seed(0)
    if layout == "channels_last":
        def make():
            return torch.randn(4, 8, 3, 3).to(memory_format=torch.channels_last)
    else:
        def make():
            return torch.randn(8, 4, 3, 3).permute(1, 0, 2, 3)

    base = [make() for _ in range(3)]
    for t in base:
        assert not t.is_contiguous()
    fast = [torch.nn.Parameter(t.clone()) for t in base]
    ref = [torch.nn.Parameter(t.clone()) for t in base]
    kw = dict(lr=1.0, d0=1e-2, second_moment="factored", factor_conv_as_matrix=True,
              weight_decay=0.01, bf16_method="none")
    opt = KProdigy(fast, foreach=True, **kw)
    opt_ref = KProdigy(ref, foreach=False, **kw)
    before = [p.detach().clone() for p in fast]
    for step in range(1, 4):
        g = torch.Generator().manual_seed(200 + step)
        for p, r in zip(fast, ref, strict=True):
            gr = torch.randn(p.shape, generator=g).mul_(0.1)
            gr = (gr.to(memory_format=torch.channels_last) if layout == "channels_last"
                  else gr.permute(1, 0, 2, 3).contiguous().permute(1, 0, 2, 3))
            assert not gr.is_contiguous()
            p.grad, r.grad = gr.clone(), gr.clone()
        opt.step()
        opt_ref.step()
    # The guard sent every param to the per-parameter loop, so no plan was ever built.
    assert opt._foreach_plans == {}, "a non-contiguous factored conv must not be batched"
    for p, b in zip(fast, before, strict=True):
        assert float((p.detach() - b).abs().max()) > 0, "the weight was left unchanged"
    # Pass 1 stays batched here (a global reduction, not this bucketing) and its
    # [B, ...] reduction tree is not the per-tensor one; the module's contract is
    # agreement to 1e-6 relative.
    assert math.isclose(opt.get_d(), opt_ref.get_d(), rel_tol=1e-6)
    for p, r in zip(fast, ref, strict=True):
        torch.testing.assert_close(p.detach(), r.detach(), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("foreach", [True, False])
def test_factored_eps_zero_with_an_all_zero_grad_stays_finite(foreach):
    """``eps_factored=0`` and an all-zero gradient history leave ``row == 0``, and
    ``row / mean(row)`` was ``0/0`` — a NaN that the ``1/(d*eps)`` cap cannot remove.
    Flooring the divisor (ADOPT's ``_MIN_NORMAL``) turns it into ``inf``, which the cap
    collapses to ``1/(d*eps)``; with a zero numerator the update is exactly zero."""
    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(8, 6)) for _ in range(3)]
    opt = KProdigy(params, second_moment="factored", eps_factored=0.0, foreach=foreach)
    before = [p.detach().clone() for p in params]
    for p in params:
        p.grad = torch.zeros_like(p)
    opt.step()
    for p, b in zip(params, before, strict=True):
        assert torch.equal(p.detach(), b)
    g = torch.Generator().manual_seed(3)
    for _ in range(3):
        for p in params:
            p.grad = torch.randn(p.shape, generator=g)
        opt.step()
    assert all(torch.isfinite(p).all() for p in params)
    assert math.isfinite(opt.get_d())


@pytest.mark.parametrize("momentum_dtype", ["float32", "bfloat16", "int8", "4bit"])
@pytest.mark.parametrize("second_moment", ["full", "factored"])
def test_pass1_chunking_tracks_the_unchunked_run(momentum_dtype, second_moment):
    """Pass 1 is chunked by the same ``foreach_stack_budget`` as pass 2 (it used to
    stack whole shape buckets and keep every param's fp32 grad alive). Splitting a
    bucket must not move D or the weights beyond the module's 1e-6 pass-1 contract."""
    g = torch.Generator().manual_seed(0)
    shapes = [(32, 16)] * 5 + [(8, 4, 3, 3)] * 3 + [(16,)] * 4 + [(3, 7)] * 3
    base = [torch.nn.Parameter(torch.randn(*s, generator=g) * 0.1) for s in shapes]
    pa = [torch.nn.Parameter(p.detach().clone()) for p in base]
    pb = [torch.nn.Parameter(p.detach().clone()) for p in base]
    kw = dict(momentum_dtype=momentum_dtype, second_moment=second_moment, d0=1e-6)
    d_big, _ = _run_kprodigy_opt(pa, foreach=True, **kw)
    # 600 elements: every 2-D / conv bucket splits into chunks of one or two params.
    d_small, _ = _run_kprodigy_opt(pb, foreach=True, foreach_stack_budget=600, **kw)
    _assert_d_climbed(d_big, kw["d0"])
    for a, b in zip(d_big, d_small, strict=True):
        assert a == pytest.approx(b, rel=1e-6, abs=0)
    for a, b in zip(pa, pb, strict=True):
        _assert_params_close(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("second_moment", ["full", "factored"])
def test_pass1_peak_is_bounded_by_the_stack_budget(second_moment):
    """A bf16 bag of identical blocks (the DiT case): pass 1 used to widen EVERY
    gradient to fp32 and keep it until the end of the pass, and to stack each whole
    shape bucket, so the step's transient peak grew with the model. Chunked, it is a
    few budget-sized stacks, independent of the number of blocks."""
    torch.manual_seed(0)
    n, shape, budget = 80, (256, 1024), 1 << 19
    ps = [torch.nn.Parameter(torch.randn(shape, device="cuda").bfloat16()) for _ in range(n)]
    opt = KProdigy(ps, second_moment=second_moment, foreach_stack_budget=budget)
    for _ in range(2):  # step 1 allocates the state; measure the steady-state step
        for p in ps:
            p.grad = torch.randn(shape, device="cuda").bfloat16()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        opt.step()
        torch.cuda.synchronize()
    transient = torch.cuda.max_memory_allocated() - base
    # The old pass 1 held every widened grad at once, so its transient could not drop
    # below this (measured 600 MiB here vs 40 MiB chunked, on an 80 MiB bound).
    all_fp32_grads = n * math.prod(shape) * 4
    assert transient < all_fp32_grads, (transient / 2**20, all_fp32_grads / 2**20)


@pytest.mark.parametrize("foreach", [True, False])
@pytest.mark.parametrize("slice_p", [1, 3])
def test_p0_is_scalar_for_zero_weights_and_a_slice_copy_otherwise(foreach, slice_p):
    """The first-step state init checks ``norm > 0`` for all params in one batched
    host sync (it used to sync per parameter); the outcome must be what the per-param
    check gave: a 0-D ``p0`` for an all-zero weight (no fp32 copy of a zero-init layer),
    the sliced fp32 weight otherwise."""
    torch.manual_seed(0)
    zero = torch.nn.Parameter(torch.zeros(6, 5))
    live = torch.nn.Parameter(torch.randn(6, 5))
    bias = torch.nn.Parameter(torch.randn(5))
    opt = KProdigy([zero, live, bias], slice_p=slice_p, foreach=foreach)
    for p in (zero, live, bias):
        p.grad = torch.randn_like(p)
    before = {id(p): p.detach().clone() for p in (live, bias)}
    opt.step()
    assert opt.state[zero]["p0"].ndim == 0 and float(opt.state[zero]["p0"]) == 0.0
    for p in (live, bias):
        torch.testing.assert_close(
            opt.state[p]["p0"], before[id(p)].flatten()[::slice_p], rtol=0, atol=0,
        )
