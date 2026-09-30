"""Tests for the AdaMuon optimizer (orthogonalized momentum + factored variance)."""

from __future__ import annotations

import io
import math
import os
import shutil

import pytest
import torch

from kaon import AdaMuon
from kaon.adamuon import (
    zeropower_via_newtonschulz5,
    zeropower_via_newtonschulz5_stacked,
)

from .conftest import train_steps


def _parity_params():
    """A mix exercising every fast-path branch (factored, conv, 1-D), with repeated
    shapes so buckets have N>1 and LoRA-like small 2-D weights."""
    g = torch.Generator().manual_seed(0)
    shapes = [
        (64, 128), (128, 64), (64, 128),      # 2-D, one shape repeated -> bucket N=2
        (32, 8, 3, 3),                        # conv (matrixize)
        (8, 96), (96, 8),                     # LoRA-like 2-D
        (40,), (40,), (128,), (320,),         # 1-D: repeated + distinct lengths
    ]
    return [torch.nn.Parameter(torch.randn(*s, generator=g) * 0.05) for s in shapes]


def test_smoke_routes_by_rank():
    """2-D/4-D params get factored row/col + momentum; 1-D get the non-factored v."""
    w2d = torch.nn.Parameter(torch.randn(16, 8))
    w4d = torch.nn.Parameter(torch.randn(8, 4, 3, 3))
    b1d = torch.nn.Parameter(torch.randn(16))
    opt = AdaMuon([w2d, w4d, b1d], lr=1e-2, momentum_dtype="int8")
    for p in (w2d, w4d, b1d):
        p.grad = torch.randn_like(p)
    opt.step()
    for w in (w2d, w4d):
        assert "row" in opt.state[w] and "col" in opt.state[w]
        assert "m" in opt.state[w]                       # quantized momentum, not exp_avg_sq
    assert "v" in opt.state[b1d] and "m" in opt.state[b1d]
    assert "exp_avg_sq" not in opt.state[b1d]            # NOT Muon's fp32 AdamW fallback


def test_overfits_regression():
    """AdaMuon should drive MSE down on a fixed target (orthogonalize+variance works)."""
    torch.manual_seed(0xC0DE)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, 8),
    )
    opt = AdaMuon(model.parameters(), lr=2e-2)
    x = torch.randn(64, 32)
    y = torch.randn(64, 8)
    initial = (model(x) - y).pow(2).mean().item()
    train_steps(model, opt, [(x, y)] * 80)
    final = (model(x) - y).pow(2).mean().item()
    assert final < 0.5 * initial, f"loss did not drop: {initial:.4f} -> {final:.4f}"


def test_conv_net_trains_no_nan():
    """A small Conv2d net exercises the conv matrixize -> NS -> factored path."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(4, 16, 3, padding=1),
        torch.nn.GELU(),
        torch.nn.Conv2d(16, 4, 3, padding=1),
    )
    opt = AdaMuon(model.parameters(), lr=2e-2)
    x = torch.randn(2, 4, 16, 16)
    y = torch.randn(2, 4, 16, 16)
    for _ in range(30):
        opt.zero_grad()
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)


def test_bf16_weights_train_no_nan():
    """bf16 weights + stochastic rounding (NS runs in bf16 internally) train cleanly."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 8)
    ).to(torch.bfloat16)
    opt = AdaMuon(model.parameters(), lr=2e-2, bf16_method="stochastic_rounding")
    x = torch.randn(64, 32, dtype=torch.bfloat16)
    y = torch.randn(64, 8, dtype=torch.bfloat16)
    for _ in range(30):
        opt.zero_grad()
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)


def test_batched_ns_matches_per_slice():
    """Batched bmm Newton-Schulz matches the per-slice helper (within bf16 tol)."""
    torch.manual_seed(0)
    mats = [torch.randn(48, 64) for _ in range(5)]         # R<C and the transpose path
    stacked = torch.stack(mats)
    out_stacked = zeropower_via_newtonschulz5_stacked(stacked, steps=5).float()
    for i, m in enumerate(mats):
        out_one = zeropower_via_newtonschulz5(m, steps=5).float()
        torch.testing.assert_close(out_stacked[i], out_one, rtol=2e-2, atol=2e-2)


def test_orthogonalized_update_singular_values():
    """The orthogonalized signal O should have near-unit singular values."""
    torch.manual_seed(0)
    g = torch.randn(64, 48)
    u = zeropower_via_newtonschulz5(g, steps=5).float()
    sv = torch.linalg.svdvals(u)
    assert sv.max() < 1.3 and sv.min() > 0.5, f"singular values not ~1: [{sv.min():.2f}, {sv.max():.2f}]"


@pytest.mark.parametrize(
    "cfg",
    [
        dict(lr=2e-2, betas=(0.0, 0.999)),                              # no momentum
        dict(lr=2e-2, betas=(0.95, 0.999), momentum_dtype="float32"),   # fp32 momentum
        dict(lr=2e-2, betas=(0.95, 0.999), momentum_dtype="bfloat16"),  # bf16 momentum
        dict(lr=2e-2, betas=(0.95, 0.999), momentum_dtype="int8"),      # int8 momentum
        dict(lr=2e-2, betas=(0.95, 0.999), momentum_dtype="4bit"),      # 4-bit momentum
        dict(lr=2e-2, betas=(0.95, 0.999), weight_decay=0.02),          # weight decay
        dict(lr=2e-2, betas=(0.95, 0.999), cautious=True),             # cautious mask
        dict(lr=2e-2, betas=(0.95, 0.999), bias_correction=True),      # bias-corrected v
    ],
)
def test_foreach_matches_per_param(cfg):
    """foreach=True matches the per-parameter path within bf16 Newton-Schulz
    tolerance.

    Unlike Adakaon (all-fp32 math, bit-exact), AdaMuon's 2-D path runs NS in
    bf16, and the batched bmm reduces in a different order than per-slice matmul —
    so the two paths agree closely but not bit-for-bit. 1-D buckets and all the
    fp32 ops are exact; the residual is the bf16 NS on the 2-D weights.
    """
    pa = _parity_params()
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    oa = AdaMuon(pa, foreach=True, **cfg)
    ob = AdaMuon(pb, foreach=False, **cfg)
    gg = torch.Generator().manual_seed(7)
    for _ in range(6):
        for a, b in zip(pa, pb, strict=False):
            grad = torch.randn(*a.shape, generator=gg) * 0.02
            a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb, strict=False):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=2e-2, atol=2e-3)


def test_foreach_1d_bucket_is_bit_exact():
    """The 1-D non-factored bucket (no Newton-Schulz) is bit-exact vs per-param."""
    shapes = [(40,), (40,), (128,), (320,)]
    pa = [torch.nn.Parameter(torch.randn(*s) * 0.05) for s in shapes]
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    oa = AdaMuon(pa, lr=2e-2, betas=(0.95, 0.999), momentum_dtype="int8", foreach=True)
    ob = AdaMuon(pb, lr=2e-2, betas=(0.95, 0.999), momentum_dtype="int8", foreach=False)
    gg = torch.Generator().manual_seed(7)
    for _ in range(8):
        for a, b in zip(pa, pb, strict=False):
            grad = torch.randn(*a.shape, generator=gg) * 0.02
            a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb, strict=False):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
def test_checkpoint_roundtrip_preserves_momentum_dtype(momentum_dtype):
    """torch.save/load resumes bit-exactly and keeps the configured momentum dtype.

    torch's default ``load_state_dict`` upcasts quantized momentum to fp32; AdaMuon
    overrides it to restore the stored dtype (memory + exact resume).
    """
    torch.manual_seed(0)
    p_ref = torch.randn(16, 8)
    grads = [torch.randn(16, 8) for _ in range(10)]

    a = torch.nn.Parameter(p_ref.clone())
    opt_a = AdaMuon([a], lr=2e-2, betas=(0.95, 0.999), momentum_dtype=momentum_dtype)
    for g in grads[:5]:
        a.grad = g.clone()
        opt_a.step()

    buf = io.BytesIO()
    torch.save(opt_a.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, weights_only=False)

    b = torch.nn.Parameter(a.detach().clone())
    opt_b = AdaMuon([b], lr=2e-2, betas=(0.95, 0.999), momentum_dtype=momentum_dtype)
    opt_b.load_state_dict(sd)

    assert opt_b.state[b]["m"].dtype == opt_a.state[a]["m"].dtype

    for g in grads[5:]:
        a.grad = g.clone()
        b.grad = g.clone()
        opt_a.step()
        opt_b.step()
    torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


def test_momentum_is_quantized_state():
    """int8 momentum is ~1 byte/param; bf16 is half of fp32 (Adafactor-class memory)."""
    def mom_bytes(dtype: str) -> int:
        p = torch.nn.Parameter(torch.randn(128, 128))
        opt = AdaMuon([p], lr=2e-2, betas=(0.95, 0.999), momentum_dtype=dtype)
        p.grad = torch.randn_like(p)
        opt.step()
        return opt.state[p]["m"].numel() * opt.state[p]["m"].element_size()

    assert mom_bytes("bfloat16") * 2 == mom_bytes("float32")
    assert mom_bytes("int8") * 4 == mom_bytes("float32")


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(betas=(1.0, 0.999)), "betas\\[0\\]"),
        (dict(betas=(0.9, 1.0)), "betas\\[1\\]"),
        (dict(lr=-1.0), "lr"),
        (dict(ns_steps=0), "ns_steps"),
        (dict(clip_threshold=0.0), "clip_threshold"),
        (dict(momentum_dtype="int4"), "momentum_dtype"),
        (dict(bf16_method="bogus"), "bf16_method"),
    ],
)
def test_invalid_args_rejected(kwargs, match):
    p = torch.nn.Parameter(torch.randn(4, 4))
    with pytest.raises(ValueError, match=match):
        AdaMuon([p], **kwargs)


@pytest.mark.parametrize("foreach", [True, False])
@pytest.mark.parametrize("weight_decay", [0.0, 0.02])
def test_compile_step_matches_eager(foreach, weight_decay):
    """``compile=True`` produces a numerically equivalent update and stays finite.

    Both routes are covered: ``foreach=True`` compiles the stacked bucket kernels,
    ``foreach=False`` the per-parameter ones (which mutate ``row``/``col``/``v`` in
    place *inside* the graph), and ``weight_decay`` exercises the one place where the
    compiled kernel is written differently from eager (``p*(lr·wd)`` instead of
    ``add_(alpha=lr·wd)``, so Dynamo cannot specialize on the value).

    That rewrite is numerically free — Inductor fuses both forms into the same kernel,
    and the non-orthogonalized buckets come out bit-identical with ``wd`` on. What is
    left is the bf16 Newton-Schulz on ``ndim>=2`` weights, which Inductor reassociates
    (~1e-5 relative), hence the loose tolerance below.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu" and os.name == "nt" and shutil.which("cl") is None:
        pytest.skip("torch.compile CPU on Windows requires the MSVC cl compiler")
    torch.manual_seed(0)
    shapes = [(16, 24), (16, 24), (12,), (4, 3, 3, 3)]
    ps0 = [torch.randn(s, device=dev) for s in shapes]
    gs = [torch.randn(s, device=dev) * 0.1 for s in shapes]

    def run(compile):
        ps = [torch.nn.Parameter(p.clone()) for p in ps0]
        opt = AdaMuon(ps, lr=1e-3, betas=(0.95, 0.999), ns_steps=2, cautious=True,
                      momentum_dtype="float32", bf16_method="none", compile=compile,
                      foreach=foreach, weight_decay=weight_decay)
        for it in range(3):
            for p, g in zip(ps, gs, strict=True):
                p.grad = g.clone()
            for pg in opt.param_groups:        # an LR schedule, as any real run has
                pg["lr"] = 1e-3 * (1.0 - 0.1 * it)
            opt.step()
        return [p.detach().clone() for p in ps]

    eager, compiled = run(False), run(True)
    for e, c in zip(eager, compiled, strict=True):
        assert torch.isfinite(c).all()
        # AdaMuon runs Newton-Schulz in bf16 internally -> looser tol than Adakaon
        assert torch.allclose(e, c, rtol=3e-2, atol=2e-3)


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
    oa = AdaMuon(pa, foreach=True, **cfg)
    ob = AdaMuon(pb, foreach=False, **cfg)
    gg = torch.Generator().manual_seed(grad_seed)
    for _ in range(steps):
        for a, b in zip(pa, pb, strict=True):
            grad = torch.randn(a.shape, generator=gg) * 0.02
            a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    return pa, pb, oa, ob


_SCALAR_CFGS = [
    {'lr': 0.01, 'betas': (0.95, 0.999), 'momentum_dtype': 'float32'},
    {'lr': 0.01, 'betas': (0.95, 0.999), 'momentum_dtype': 'bfloat16'},
    {'lr': 0.01, 'betas': (0.95, 0.999), 'momentum_dtype': 'int8'},
    {'lr': 0.01, 'betas': (0.95, 0.999), 'momentum_dtype': '4bit'},
    {'lr': 0.01, 'betas': (0.95, 0.999), 'momentum_dtype': 'float32', 'weight_decay': 0.02},
    {'lr': 0.01, 'betas': (0.95, 0.999), 'momentum_dtype': 'int8', 'weight_decay': 0.02},
    {'lr': 0.01, 'betas': (0.95, 0.999), 'momentum_dtype': '4bit', 'weight_decay': 0.02},
    {'lr': 0.01, 'betas': (0.0, 0.999), 'momentum_dtype': 'float32'},
    {'lr': 0.01, 'betas': (0.95, 0.999), 'momentum_dtype': 'float32', 'clip_threshold': 0.5},
    {'lr': 0.01, 'betas': (0.95, 0.999), 'momentum_dtype': 'float32', 'cautious': False},
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
    orig = AdaMuon._step_one_param

    def spy(self, p, *args, **kwargs):
        looped.append(p)
        return orig(self, p, *args, **kwargs)

    monkeypatch.setattr(AdaMuon, "_step_one_param", spy)
    params = _scalar_bag(_scalar_shapes())
    opt = AdaMuon(params, foreach=True, lr=0.01, betas=(0.95, 0.999))
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
    pa, pb, _oa, _ob = _scalar_parity({'lr': 0.01, 'betas': (0.95, 0.999), 'momentum_dtype': 'float32'}, shapes=shapes, steps=8, seed=3, grad_seed=5)
    for a, b in zip(pa, pb, strict=True):
        if a.numel() == 1:
            torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)
        else:
            torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=1e-7)


def test_scalar_0d_checkpoint_roundtrip_across_paths():
    """A checkpoint saved mid-training by one path resumes bit-exactly on the other:
    the 0-D state layout is identical on both, in both directions."""
    cfg = {'lr': 0.01, 'betas': (0.95, 0.999), 'momentum_dtype': 'int8', 'weight_decay': 0.02}
    for save_foreach, load_foreach in [(True, False), (False, True)]:
        pa = _scalar_bag(_scalar_shapes(), seed=29)
        oa = AdaMuon(pa, foreach=save_foreach, **cfg)
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
        ob = AdaMuon(pb, foreach=load_foreach, **cfg)
        ob.load_state_dict(ckpt["opt"])
        for step in range(4, 8):
            for plist, opt in ((pa, oa), (pb, ob)):
                for p, gr in zip(plist, grads[step], strict=True):
                    p.grad = gr.clone()
                opt.step()
        for a, b in zip(pa, pb, strict=True):
            torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


# ====================================================== second-moment bias correction
@pytest.mark.parametrize("foreach", [True, False])
@pytest.mark.parametrize("steps", [1, 3])
def test_bias_correction_is_a_sqrt_scale_on_the_update(foreach, steps):
    """``bias_correction`` == dividing the factored second moment by ``1-β₂ᵗ``.

    Correcting ``v`` cancels out of the row factor (a *ratio* of row stats) and
    survives only in the column factor, so the whole correction is one ``√(1-β₂ᵗ)``
    scale on the normalized update. Checked against the uncorrected update with the
    clip pushed out of the way so nothing else touches the magnitude.
    """
    beta2 = 0.9
    shapes = [(16, 24), (16, 24), (12,), (6, 4, 3, 3)]
    g = torch.Generator().manual_seed(5)
    p0 = [torch.randn(*s, generator=g) * 0.05 for s in shapes]
    grads = [[torch.randn(*s, generator=g) * 0.02 for s in shapes] for _ in range(steps)]

    def last_deltas(bias_correction):
        ps = [torch.nn.Parameter(t.clone()) for t in p0]
        opt = AdaMuon(ps, lr=1e-2, betas=(0.0, beta2), cautious=False, foreach=foreach,
                      clip_threshold=1e9, bias_correction=bias_correction)
        before = None
        for gs in grads:
            before = [p.detach().clone() for p in ps]
            for p, gr in zip(ps, gs, strict=True):
                p.grad = gr.clone()
            opt.step()
        return [p.detach() - b for p, b in zip(ps, before, strict=True)]

    # The residual is the weight subtraction: recovering a ~1e-3 delta by differencing
    # ~5e-2 weights costs an ulp of the *weight*, i.e. ~4e-9 absolute.
    scale = math.sqrt(1.0 - beta2 ** steps)
    for d_off, d_on in zip(last_deltas(False), last_deltas(True), strict=True):
        torch.testing.assert_close(d_on, d_off * scale, rtol=1e-3, atol=5e-8)


def test_bias_correction_tames_the_first_step_update():
    """Uncorrected, the first update is ~``1/√(1-β₂)`` too large and ``clip_threshold``
    is what absorbs it; corrected, it lands at RMS≈1 and the clip is inert."""
    beta2 = 0.999
    p = torch.nn.Parameter(torch.randn(64, 64, generator=torch.Generator().manual_seed(0)))
    grad = torch.randn(64, 64, generator=torch.Generator().manual_seed(1)) * 0.02
    rms_u = {}
    for bc in (False, True):
        q = torch.nn.Parameter(p.detach().clone())
        opt = AdaMuon([q], lr=1.0, betas=(0.0, beta2), cautious=False, foreach=False,
                      clip_threshold=1e9, bias_correction=bc)
        q.grad = grad.clone()
        opt.step()
        # applied RMS is 0.2·lr·rms(u); recover rms(u) with lr = 1.
        rms_u[bc] = float((q.detach() - p.detach()).pow(2).mean().sqrt()) / 0.2
    assert rms_u[False] > 20.0, rms_u          # 1/sqrt(1-beta2) = 31.6
    assert 0.8 < rms_u[True] < 1.25, rms_u     # RMS≈1 -> clip_threshold=1.0 is inert


def test_bias_correction_is_near_inert_under_the_default_clip():
    """With ``clip_threshold=1.0`` the clip already normalizes ``rms(u)`` to 1 every
    step, so ``bias_correction`` is a <=2 % per-tensor rescale — this is *why* the
    default is ``False``, and the claim the docs make.

    The uncorrected ``rms(u)`` is almost exactly ``1/√(1-β₂ᵗ)``, i.e. the very factor
    the correction divides out, so both settings hit the clip and emit the same update
    while it binds.
    """
    beta2 = 0.999
    g = torch.Generator().manual_seed(0)
    base = torch.randn(48, 48, generator=g)
    grads = [torch.randn(48, 48, generator=g) * 0.02 for _ in range(60)]
    probes = (1, 10, 60)

    def applied(bias_correction):
        p = torch.nn.Parameter(base.clone())
        opt = AdaMuon([p], lr=1.0, betas=(0.0, beta2), cautious=False, foreach=False,
                      clip_threshold=1.0, bias_correction=bias_correction)
        out = {}
        for t, grad in enumerate(grads, 1):
            before = p.detach().clone()
            p.grad = grad.clone()
            opt.step()
            if t in probes:
                out[t] = (p.detach() - before) / 0.2      # applied RMS in units of 0.2·lr
        return out

    off, on = applied(False), applied(True)
    for t in probes:
        r_off = float(off[t].pow(2).mean().sqrt())
        r_on = float(on[t].pow(2).mean().sqrt())
        assert abs(r_off - 1.0) < 0.01, f"step {t}: clip should pin rms(u) to 1, got {r_off}"
        ratio = r_on / r_off
        # <=1 (the correction can only shrink) up to fp noise, and never by much.
        assert 0.98 <= ratio <= 1.0 + 1e-5, f"step {t}: not a <=2% shrink ({ratio})"


def test_bias_correction_step_counter_is_per_parameter():
    """A parameter that only sometimes gets a gradient (MoE routing, CFG dropout,
    partial accumulation) is corrected by ITS OWN update count, not the run's."""
    shapes = [(8, 12), (8, 12)]
    g = torch.Generator().manual_seed(2)
    ps = [torch.nn.Parameter(torch.randn(*s, generator=g) * 0.05) for s in shapes]
    opt = AdaMuon(ps, lr=1e-2, betas=(0.95, 0.999), bias_correction=True)
    for it in range(5):
        ps[0].grad = torch.randn(*shapes[0], generator=g) * 0.02
        ps[1].grad = torch.randn(*shapes[1], generator=g) * 0.02 if it == 4 else None
        opt.step()
    assert opt.state[ps[0]]["step"] == 5
    assert opt.state[ps[1]]["step"] == 1
    assert all(torch.isfinite(p).all() for p in ps)


def test_bias_correction_mixed_step_bucket_matches_per_param():
    """Parameters at *different* ``t`` must each get their own correction factor.

    With ``bias_correction`` on, ``t`` is part of the bucket key, so a set of weights
    that share a shape but not an update count (MoE routing, CFG dropout) is split into
    one bucket per ``t`` rather than corrected with a single shared factor. Pin that
    against the per-parameter path, which computes each parameter's ``t`` independently.
    """
    shapes = [(12, 16), (12, 16), (12, 16), (20,), (20,)]
    g = torch.Generator().manual_seed(4)
    p0 = [torch.randn(*sh, generator=g) * 0.05 for sh in shapes]
    pa = [torch.nn.Parameter(t.clone()) for t in p0]
    pb = [torch.nn.Parameter(t.clone()) for t in p0]
    oa = AdaMuon(pa, lr=2e-2, betas=(0.95, 0.999), bias_correction=True, foreach=True)
    ob = AdaMuon(pb, lr=2e-2, betas=(0.95, 0.999), bias_correction=True, foreach=False)

    for it in range(8):
        for i, (a, b) in enumerate(zip(pa, pb, strict=True)):
            # every parameter skips a different subset of steps -> the bucket holds a
            # mix of update counts from step 2 onwards
            if (it + i) % 3 == 0 and it > 0:
                a.grad = b.grad = None
                continue
            grad = torch.randn(*a.shape, generator=g) * 0.02
            a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()

    counts = {oa.state[a]["step"] for a in pa}
    assert len(counts) > 1, "the bucket must actually hold mixed update counts"
    for a, b in zip(pa, pb, strict=True):
        assert oa.state[a]["step"] == ob.state[b]["step"]
        torch.testing.assert_close(a.detach(), b.detach(), rtol=2e-2, atol=2e-3)


def test_bias_correction_checkpoint_roundtrip():
    """``state["step"]`` survives save/load, so the correction resumes at the right t."""
    torch.manual_seed(0)
    grads = [torch.randn(16, 8) for _ in range(8)]
    a = torch.nn.Parameter(torch.randn(16, 8))
    opt_a = AdaMuon([a], lr=2e-2, betas=(0.95, 0.999), bias_correction=True,
                    momentum_dtype="int8")
    for g in grads[:4]:
        a.grad = g.clone()
        opt_a.step()

    buf = io.BytesIO()
    torch.save(opt_a.state_dict(), buf)
    buf.seek(0)
    b = torch.nn.Parameter(a.detach().clone())
    opt_b = AdaMuon([b], lr=2e-2, betas=(0.95, 0.999), bias_correction=True,
                    momentum_dtype="int8")
    opt_b.load_state_dict(torch.load(buf, weights_only=False))
    assert opt_b.state[b]["step"] == 4

    for g in grads[4:]:
        a.grad, b.grad = g.clone(), g.clone()
        opt_a.step()
        opt_b.step()
    torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


def test_resume_from_pre_bias_correction_checkpoint():
    """A checkpoint from a kaon that predates ``bias_correction`` must resume, exactly.

    ``torch.optim.Optimizer.load_state_dict`` **replaces** each ``param_groups`` dict
    with the checkpoint's (it only carries ``params`` over), so every hyperparameter
    added after the checkpoint was written simply vanishes from the live group — the
    next step then died with ``KeyError: 'bias_correction'``. A 0.7.11 checkpoint also
    has no per-param ``state["step"]``.

    Reproduce both by stripping the two keys from the state dict, and require not just
    that the resume runs but that it is **bit-identical** to a run that never
    checkpointed (with the correction off, the restarted ``t`` must not perturb
    anything).
    """
    torch.manual_seed(0)
    shapes = [(16, 8), (12,), (4, 3, 3, 3)]
    base = [torch.randn(*sh) for sh in shapes]
    grads = [[torch.randn(*sh) * 0.1 for sh in shapes] for _ in range(8)]
    cfg = dict(lr=2e-2, betas=(0.95, 0.999), momentum_dtype="int8", weight_decay=0.02)

    control = [torch.nn.Parameter(t.clone()) for t in base]
    opt_c = AdaMuon(control, **cfg)
    resumed = [torch.nn.Parameter(t.clone()) for t in base]
    opt_r = AdaMuon(resumed, **cfg)

    for gs in grads[:4]:                       # both run the first half normally
        for pair in (zip(control, gs, strict=True), zip(resumed, gs, strict=True)):
            for p, g in pair:
                p.grad = g.clone()
        opt_c.step()
        opt_r.step()

    buf = io.BytesIO()
    torch.save(opt_r.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, weights_only=False)
    for pg in sd["param_groups"]:              # as written by 0.7.11
        pg.pop("bias_correction", None)
    sd["state"] = {k: {kk: vv for kk, vv in v.items() if kk != "step"}
                   for k, v in sd["state"].items()}

    fresh = [torch.nn.Parameter(p.detach().clone()) for p in resumed]
    opt_f = AdaMuon(fresh, **cfg)
    opt_f.load_state_dict(sd)
    # the missing key is backfilled from the constructor defaults, so param_groups is
    # complete again and every read site finds it
    assert opt_f.param_groups[0]["bias_correction"] is False
    assert "step" not in opt_f.state[fresh[0]]

    for gs in grads[4:]:
        for pair in (zip(control, gs, strict=True), zip(fresh, gs, strict=True)):
            for p, g in pair:
                p.grad = g.clone()
        opt_c.step()
        opt_f.step()                            # used to raise KeyError here
    for c, f in zip(control, fresh, strict=True):
        torch.testing.assert_close(f.detach(), c.detach(), rtol=0, atol=0)
    assert opt_f.state[fresh[0]]["step"] == 4   # counter restarts, numerics unaffected


def test_load_state_dict_keeps_checkpoint_hyperparameters():
    """The defaults backfill must not clobber values the checkpoint *does* carry."""
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = AdaMuon([p], lr=2e-2, betas=(0.95, 0.999), bias_correction=True,
                  clip_threshold=0.5)
    p.grad = torch.randn(8, 8)
    opt.step()
    sd = opt.state_dict()

    q = torch.nn.Parameter(torch.randn(8, 8))
    opt2 = AdaMuon([q], lr=1e-3, betas=(0.9, 0.99), bias_correction=False,
                   clip_threshold=1.0)
    opt2.load_state_dict(sd)
    g = opt2.param_groups[0]
    assert g["bias_correction"] is True and g["clip_threshold"] == 0.5
    assert g["lr"] == 2e-2 and g["betas"] == (0.95, 0.999)


def test_step_with_a_gradientless_group_is_a_no_op():
    """A param group where nothing has a gradient must be skipped, not crash.

    Same scenario as the recompilation test — a grad set that moves — but the extreme
    of it: MoE routing can leave a whole expert group unrouted for a step, and a bare
    ``opt.step()`` with nothing backwarded is legal too. Both used to raise
    ``IndexError`` from the ``foreach`` path's ``params[0].device`` probe.
    """
    a = torch.nn.Parameter(torch.randn(8, 8))
    b = torch.nn.Parameter(torch.randn(8, 8))
    opt = AdaMuon([{"params": [a]}, {"params": [b]}], lr=1e-3)

    a.grad = torch.randn(8, 8) * 0.1
    b_before, a_before = b.detach().clone(), a.detach().clone()
    opt.step()
    assert torch.equal(b.detach(), b_before), "gradient-less group must not move"
    assert not torch.equal(a.detach(), a_before), "the group with a gradient must move"
    assert b not in opt.state or not opt.state[b]

    a.grad = None                                    # nothing anywhere
    snapshot = [a.detach().clone(), b.detach().clone()]
    opt.step()
    for param, before in zip((a, b), snapshot, strict=True):
        assert torch.equal(param.detach(), before)


# ====================================================== torch.compile recompilation
def test_compile_recompiles_stay_bounded():
    """``compile=True`` must not recompile when the *set of parameters with a gradient*
    changes, when ``lr`` moves, or when a param group is added.

    All three used to be per-step guards (``p.grad is None`` per parameter, the literal
    value of ``group["lr"]``), so a MoE / CFG-dropout / partial-accumulation step or any
    LR schedule burned through ``recompile_limit`` (8) — measured 8 graphs and a silent
    fall back to eager. The compiled unit is now the pure-tensor math, guarded on
    shapes and dtypes only.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu" and os.name == "nt" and shutil.which("cl") is None:
        pytest.skip("torch.compile CPU on Windows requires the MSVC cl compiler")
    dynamo = pytest.importorskip("torch._dynamo")
    from torch._dynamo.utils import counters

    dynamo.reset()
    counters.clear()
    torch.manual_seed(0)
    ps = [torch.nn.Parameter(torch.randn(16, 24, device=dev)) for _ in range(4)]
    extra = [torch.nn.Parameter(torch.randn(16, 24, device=dev))]
    opt = AdaMuon(ps, lr=1e-3, betas=(0.95, 0.999), ns_steps=2, compile=True)
    for it in range(12):
        for i, p in enumerate(ps):
            p.grad = None if i == it % len(ps) else torch.randn(16, 24, device=dev) * 0.1
        for pg in opt.param_groups:                     # an LR schedule
            pg["lr"] = 1e-3 * (1.0 - 0.05 * it)
        if it == 6:
            opt.add_param_group({"params": extra})
        if it >= 6:
            extra[0].grad = torch.randn(16, 24, device=dev) * 0.1
        opt.step()
    graphs = counters["stats"].get("unique_graphs", 0)
    # HEAD measures 2. Keep the bound tight: at 4 a reintroduced guard on the *value*
    # of ``lr`` still passes, which is exactly the regression this test exists for.
    assert graphs <= 2, f"{graphs} compiled graphs — a per-step guard is back"
    assert all(torch.isfinite(p).all() for p in ps + extra)

# --------------------------------------------------------------- foreach view plan
def _plan_run_adamuon(cache: bool) -> list[torch.Tensor]:
    """Five steps over a 0-D / 1-D / 2-D / conv bag, with a ``p.data`` rebind at step 3."""
    torch.manual_seed(0x5EED)
    shapes = [(), (), (1,), (5,), (5,), (4, 3), (4, 3), (2, 2, 3, 3), (2, 2, 3, 3)]
    g = torch.Generator().manual_seed(7)
    params = [torch.nn.Parameter(torch.randn(s, generator=g)) for s in shapes]
    opt = AdaMuon(params, lr=1e-2, weight_decay=0.01, momentum_dtype="int8", bias_correction=True)
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
    ``tests/test_foreach_plan.py``; this is AdaMuon's own tripwire.
    """
    on = _plan_run_adamuon(cache=True)
    off = _plan_run_adamuon(cache=False)
    for i, (a, b) in enumerate(zip(on, off, strict=True)):
        assert torch.equal(a, b), f"tensor {i} differs"


@pytest.mark.parametrize("foreach", [True, False])
@pytest.mark.parametrize("zero_column", [False, True])
def test_eps_zero_with_a_zero_second_moment_stays_finite(foreach, zero_column):
    """``eps1 = 0`` and a weight whose (orthogonalized) momentum is all zero — or has an
    all-zero column, which Newton-Schulz preserves — left ``row == 0`` / ``col == 0``:
    ``row / mean(row)`` was ``0/0``, ``rsqrt(col)`` was ``inf``, ``0 * inf`` NaN'd the
    update and the per-slice RMS clip spread it over the whole weight. A zero second
    moment means a zero orthogonalized signal there, so the update must be zero."""
    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(8, 6)) for _ in range(3)]
    opt = AdaMuon(params, lr=1e-2, eps=(0.0, 1e-3), cautious=False, foreach=foreach)
    before = [p.detach().clone() for p in params]
    g = torch.Generator().manual_seed(11)
    for p in params:
        grad = torch.randn(p.shape, generator=g) if zero_column else torch.zeros_like(p)
        if zero_column:
            grad[:, 2] = 0.0
        p.grad = grad
    opt.step()
    for p, b in zip(params, before, strict=True):
        assert torch.isfinite(p).all()
        if zero_column:
            assert torch.equal(p.detach()[:, 2], b[:, 2])
        else:
            assert torch.equal(p.detach(), b)
    for _ in range(3):
        for p in params:
            p.grad = torch.randn(p.shape, generator=g)
        opt.step()
    assert all(torch.isfinite(p).all() for p in params)


@pytest.mark.parametrize("foreach", [True, False])
@pytest.mark.parametrize("shape", [(7,), ()])
def test_eps_zero_nonfactored_zero_grad_stays_finite(foreach, shape):
    """The non-factored (1-D / 0-D) path had the same ``eps1 = 0`` hole: ``v == 0`` after
    an all-zero gradient made ``grad * rsqrt(v)`` a ``0 * inf`` NaN. The update must be
    exactly zero there."""
    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(shape)) for _ in range(3)]
    opt = AdaMuon(params, lr=1e-2, eps=(0.0, 1e-3), cautious=False, foreach=foreach)
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
