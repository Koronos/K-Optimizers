"""Tests for the KProdigy optimizer.

Covers: parameter-free D-adaptation, numerical parity with the reference
``prodigyopt.Prodigy`` (at the exact defaults), the memory variants
(bf16/int8 momentum, factored second moment, sliced D stats), bf16-weight
stochastic-rounding updates, and independent-D for multi-group (SDXL-like)
setups.
"""

from __future__ import annotations

import io

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

def _mixed_params(dtype=torch.float32):
    """2-D + conv (4-D) + 1-D params -> exercises factored, full and flat buckets."""
    g = torch.Generator().manual_seed(0)
    shapes = [(32, 16), (24, 12), (8, 4, 3, 3), (16,), (32,), (10, 5, 1, 1)]
    return [
        torch.nn.Parameter(torch.randn(*s, generator=g, dtype=dtype) * 0.1)
        for s in shapes
    ]


def _run_kprodigy(ps, *, foreach, steps=12, **kw):
    opt = KProdigy(ps, lr=1.0, **{"foreach": foreach, **kw})
    g = torch.Generator().manual_seed(123)
    ds = []
    for _ in range(steps):
        for p in ps:
            p.grad = torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.05
        opt.step()
        ds.append(opt.get_d())
    return ds


@pytest.mark.parametrize("momentum_dtype", ["float32", "bfloat16", "int8", "4bit"])
@pytest.mark.parametrize("second_moment", ["full", "factored"])
@pytest.mark.parametrize("cautious", [False, True])
def test_foreach_matches_per_param(momentum_dtype, second_moment, cautious):
    """The engine-backed (foreach) update is bit-exact vs the per-param loop on
    fp32 weights, across momentum dtype / second moment / cautious, on 2-D + conv
    + 1-D params. (D-estimation is shared, so D is identical by construction.)"""
    base = _mixed_params()
    pa = [torch.nn.Parameter(p.detach().clone()) for p in base]
    pb = [torch.nn.Parameter(p.detach().clone()) for p in base]
    kw = dict(momentum_dtype=momentum_dtype, second_moment=second_moment, cautious=cautious)
    d_pp = _run_kprodigy(pa, foreach=False, **kw)
    d_fe = _run_kprodigy(pb, foreach=True, **kw)
    assert d_pp == pytest.approx(d_fe, rel=0, abs=0)  # D identical
    for a, b in zip(pa, pb, strict=True):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


@pytest.mark.parametrize("momentum_dtype", ["float32", "bfloat16", "int8", "4bit"])
@pytest.mark.parametrize("second_moment", ["full", "factored"])
@pytest.mark.parametrize("slice_p", [1, 11])
def test_pass1_foreach_matches_per_param(momentum_dtype, second_moment, slice_p):
    """Pass-1 (the D-estimation global reduction + the d-scaled momentum / factored
    second-moment EMAs) must be bit-identical batched (foreach) vs the per-param
    loop: the same D trajectory AND the same final weights, across momentum dtype
    x {full, factored} x slice_p, on 2-D + conv + 1-D params."""
    base = _mixed_params()
    pa = [torch.nn.Parameter(p.detach().clone()) for p in base]
    pb = [torch.nn.Parameter(p.detach().clone()) for p in base]
    kw = dict(momentum_dtype=momentum_dtype, second_moment=second_moment, slice_p=slice_p)
    d_pp = _run_kprodigy(pa, foreach=False, steps=15, **kw)
    d_fe = _run_kprodigy(pb, foreach=True, steps=15, **kw)
    assert d_pp == pytest.approx(d_fe, rel=0, abs=0)  # D trajectory bit-identical
    for a, b in zip(pa, pb, strict=True):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


@pytest.mark.parametrize("independent_d", [True, False])
def test_pass1_foreach_matches_per_param_multigroup(independent_d):
    """Pass-1 batching keeps the per-group D accumulation order-equivalent: the
    foreach and per-param paths give bit-identical per-group D and weights for a
    multi-group (SDXL-like) setup, with both global and independent D."""
    base = _mixed_params()
    g1, g2 = base[:3], base[3:]

    def build():
        a = [torch.nn.Parameter(p.detach().clone()) for p in g1]
        b = [torch.nn.Parameter(p.detach().clone()) for p in g2]
        return a, b

    def run(a, b, foreach):
        opt = KProdigy(
            [{"params": a, "lr": 1.0}, {"params": b, "lr": 1.0}],
            lr=1.0, foreach=foreach, independent_d=independent_d, slice_p=11,
        )
        gen = torch.Generator().manual_seed(99)
        ds = []
        for _ in range(15):
            for p in a + b:
                p.grad = torch.randn(p.shape, generator=gen, dtype=p.dtype) * 0.05
            opt.step()
            ds.append(tuple(grp["d"] for grp in opt.param_groups))
        return ds

    a1, b1 = build()
    a2, b2 = build()
    d_pp = run(a1, b1, foreach=False)
    d_fe = run(a2, b2, foreach=True)
    assert d_pp == d_fe  # per-group D trajectory bit-identical
    for x, y in zip(a1 + b1, a2 + b2, strict=True):
        torch.testing.assert_close(x.detach(), y.detach(), rtol=0, atol=0)


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
