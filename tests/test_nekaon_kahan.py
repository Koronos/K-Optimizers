"""Nekaon (and its Adakaon core) with ``bf16_method="kahan8"`` / ``"kahan16"``, path by path.

The acceptance criterion of the 0.7.16 audit: with the defaults Nekaon ships (wd=0.1,
Gradient Centralization, cautious masking), ``kahan16`` bf16 weights reproduce the fp32-weight
run of the SAME route bit for bit, given the same bf16 gradients — per-param, foreach, native
CUDA and every fused route (one-block, 1-D, batched big with and without fused reductions,
direct 4-bit / int8, no-momentum, lone per-tensor chunked). Before 0.7.16 they did only with
wd=0 and GC off: the decay read the bare bf16 weight and native GC ran in the gradient's bf16.

Around it, the rest of Nekaon's surface under compact Kahan: the plain constructor
(``Nekaon(..., bf16_method="kahan8")``), ``low_vram_above``, mixed fp32/bf16 groups,
parameters without gradients, ``add_param_group``, eval/train, checkpoint round trips (and a
checkpoint from another method), a mid-run method switch, the inert-lookahead warning, and
the public full-precision export (:func:`kaon.decode_weights`).
"""
from __future__ import annotations

import copy
import warnings

import pytest
import torch

import kaon
from kaon import Adakaon, Nekaon
from kaon._compact_kahan import RESIDUAL_KEY, decode, residual_bits_of

CUDA = torch.cuda.is_available()
try:
    from kaon._fused_triton import HAS_TRITON
except Exception:  # pragma: no cover
    HAS_TRITON = False
FUSED = CUDA and HAS_TRITON


# ------------------------------------------------------------------------------ helpers
def _bag(dtype=torch.bfloat16, device="cpu", big=False, seed=0):
    """Factored, conv, fan-in-1 (GC skipped), 1-D, 0-D; ``big`` adds a same-shape big pair
    (batched chunked kernels) and a lone big tensor (N=1 batched / per-tensor chunked)."""
    torch.manual_seed(seed)
    shapes = [(32, 64), (32, 64), (16, 8), (64,), (64,), (), (4, 3, 3, 3), (8, 1)]
    if big:
        shapes += [(1024, 1200), (1024, 1200), (1500, 1100)]
    # bf16-representable starts, so the fp32 twin begins at exactly the same values
    return [torch.nn.Parameter((torch.randn(s, device=device) * 0.05).to(torch.bfloat16).to(dtype))
            for s in shapes]


def _drive(opts, params_lists, steps=6, seed=7, scale=0.02):
    """The SAME bf16 gradients to every run (an fp32 twin gets them upcast)."""
    gg = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        grads = [(torch.randn(p.shape, generator=gg) * scale).to(torch.bfloat16)
                 for p in params_lists[0]]
        for params in params_lists:
            for p, g in zip(params, grads, strict=True):
                if g is not None:
                    p.grad = g.to(device=p.device, dtype=p.dtype).clone()
        for o in opts:
            o.step()


def _owner_state(opt, p):
    while hasattr(opt, "inner"):
        opt = opt.inner
    return opt.state[p]


def _z(opt, p):
    """The full value a param holds: decoded under compact Kahan, else the weight itself."""
    st = _owner_state(opt, p)
    if p.dtype == torch.bfloat16 and RESIDUAL_KEY in st:
        lo = st[RESIDUAL_KEY]
        return decode(p.data, lo, residual_bits_of(lo))
    return p.data.float()


def _bits_equal(a, b):
    return torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


def _mean_ulp(zs, refs):
    """Mean |z - ref| in bf16 ulps of ref (elementwise ulp), averaged over params."""
    out = []
    for z, r in zip(zs, refs, strict=True):
        u = torch.exp2(torch.floor(torch.log2(r.abs().clamp_min(1e-30))) - 7)
        out.append(float(((z - r).abs() / u).mean()))
    return sum(out) / len(out)


_ROUTES = ["per_param", "foreach"] + (["cuda", "fused"] if FUSED else [])


def _route_kw(route):
    kw = dict(fused=route == "fused", foreach=route != "per_param")
    if route == "fused":
        kw["deterministic_reductions"] = True   # the fp32 twin must reproduce itself too
    return kw


def _device(route):
    return "cuda" if route in ("cuda", "fused") else "cpu"


# ------------------------------------------------------------------ 1. bit-exact criterion
@pytest.mark.parametrize("cls", [Adakaon, Nekaon])
@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("md", ["4bit", "int8", "bfloat16"])
@pytest.mark.parametrize("cautious_wd", ["masked", "full"])
def test_kahan16_with_decay_and_gc_is_the_fp32_run(cls, route, md, cautious_wd):
    """Defaults on (wd=0.1, GC, cautious): kahan16 == the fp32-weight run of the same route,
    bit for bit, on every param of a bag that exercises every sub-route."""
    dev = _device(route)
    p32 = _bag(torch.float32, dev, big=route == "fused")
    p16 = _bag(torch.bfloat16, dev, big=route == "fused")
    kw = dict(lr=1e-3, momentum_dtype=md, weight_decay=0.1, cautious_wd=cautious_wd,
              **_route_kw(route))
    o32 = cls(p32, **kw)
    o16 = cls(p16, bf16_method="kahan16", **kw)
    _drive([o32, o16], [p32, p16])
    for a, b in zip(p16, p32, strict=True):
        assert _owner_state(o16, a)[RESIDUAL_KEY].dtype == torch.int16
        assert _bits_equal(_z(o16, a), b.data), (tuple(a.shape), float((_z(o16, a) - b.data).abs().max()))
    if isinstance(o16, Nekaon):                        # and in the eval (true-weight) view
        o16.eval()
        o32.eval()
        for a, b in zip(p16, p32, strict=True):
            assert _bits_equal(_z(o16, a), b.data)


@pytest.mark.skipif(not FUSED, reason="Triton fused kernels need CUDA + Triton")
@pytest.mark.parametrize("variant", ["no_fused_reductions", "lone_per_tensor", "no_momentum",
                                     "codec_fallback_4bit", "codec_fallback_int8"])
def test_kahan16_decay_on_the_remaining_fused_big_routes(variant):
    """The big-tensor sub-routes the default bag does not reach: the torch-reduction batched
    kernels (``_chunked_mom_batched``), the per-tensor ``_chunked_step`` (lone big, batching
    off), beta1=0 (``_chunked_nomom_*``) and the codec fallback (4-bit block > 1024, int8
    with C > 1024)."""
    md = {"codec_fallback_int8": "int8"}.get(variant, "4bit" if "4bit" in variant else "bfloat16")
    extra = {}
    if variant == "no_momentum":
        extra["betas"] = (0.0, 0.999)
    if variant == "codec_fallback_4bit":
        extra["momentum_4bit_block"] = 2048
    runs = []
    for dtype, method in ((torch.float32, None), (torch.bfloat16, "kahan16")):
        ps = _bag(dtype, "cuda", big=True)
        kw = dict(lr=1e-3, momentum_dtype=md, weight_decay=0.1, fused=True,
                  deterministic_reductions=True, **extra)
        if method:
            kw["bf16_method"] = method
        o = Adakaon(ps, **kw)
        if variant == "no_fused_reductions":
            o._fused_reductions = False
        if variant == "lone_per_tensor":
            o._fused_big_lone_batched = False
        runs.append((ps, o))
    _drive([o for _, o in runs], [ps for ps, _ in runs])
    (p32, _), (p16, o16) = runs
    for a, b in zip(p16, p32, strict=True):
        assert _bits_equal(_z(o16, a), b.data), (variant, tuple(a.shape))


@pytest.mark.parametrize("route", _ROUTES)
def test_kahan8_tracks_fp32_far_closer_than_sr(route):
    """kahan8 keeps its advantage with the decoded decay/GC: the clean value stays a small
    fraction of an ulp from the fp32 run, an order of magnitude closer than SR."""
    dev = _device(route)
    ref = _bag(torch.float32, dev)
    runs = {m: _bag(torch.bfloat16, dev) for m in ("stochastic_rounding", "kahan8", "kahan16")}
    kw = dict(lr=1e-4, **_route_kw(route))
    kaon.reseed_stochastic_rounding()
    o_ref = Nekaon(ref, **kw)
    opts = {m: Nekaon(ps, bf16_method=m, **kw) for m, ps in runs.items()}
    _drive([o_ref, *opts.values()], [ref, *runs.values()], steps=30)
    for o in (o_ref, *opts.values()):
        o.eval()
    err = {m: _mean_ulp([_z(opts[m], p) for p in ps], [r.data for r in ref])
           for m, ps in runs.items()}
    assert err["kahan16"] == 0.0, err
    assert err["kahan8"] < 0.1 * err["stochastic_rounding"], err
    assert err["kahan8"] < 0.1, err


def test_sr_and_none_do_not_read_a_residual():
    """The decoded read exists only under compact Kahan: an SR / none group never looks at a
    (stale) residual, and ``weight_value`` is the historical read bit for bit."""
    from kaon._backend import weight_value
    p = torch.nn.Parameter((torch.randn(8, 8) * 0.05).to(torch.bfloat16))
    st = {RESIDUAL_KEY: torch.full((8, 8), 200, dtype=torch.uint8)}
    for method in ("stochastic_rounding", "none", "kahan"):
        assert torch.equal(weight_value(p, st, method), p.data.float())
    q = torch.nn.Parameter(torch.randn(8, 8))
    assert weight_value(q, {}, "kahan16").data_ptr() == q.data_ptr()      # fp32: the alias
    assert not torch.equal(weight_value(p, st, "kahan8"), p.data.float())


# ------------------------------------------------------------------ 2. ergonomics / surface
def test_plain_constructor_works_with_no_other_config():
    """``Nekaon(params, bf16_method="kahan8")`` on a bf16 model: residuals allocated at the
    first step (uint8, +1 B/param), no warning over a run with eval/train cycles."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 4))
    model.to(torch.bfloat16)
    opt = Nekaon(model.parameters(), lr=1e-4, bf16_method="kahan8")
    x = torch.randn(64, 16, dtype=torch.bfloat16)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for i in range(12):
            opt.zero_grad()
            model(x).float().pow(2).mean().backward()
            opt.step()
            if i % 4 == 3:
                opt.eval()
                opt.train()
    for p in model.parameters():
        st = opt.inner.state[p]
        assert st[RESIDUAL_KEY].dtype == torch.uint8 and st[RESIDUAL_KEY].shape == p.shape
        assert torch.isfinite(_z(opt, p)).all()


@pytest.mark.parametrize("route", ["per_param", "foreach"] + (["fused"] if FUSED else []))
def test_low_vram_above_keeps_the_residual_on_the_momentum_free_group(route):
    """The big (momentum-free, no-lookahead) group is still compensated: kahan16 == fp32."""
    dev = _device(route)
    p32, p16 = _bag(torch.float32, dev), _bag(torch.bfloat16, dev)
    kw = dict(lr=1e-3, low_vram_above=100, **_route_kw(route))
    o32, o16 = Nekaon(p32, **kw), Nekaon(p16, bf16_method="kahan16", **kw)
    assert o16.param_groups[1]["betas"][0] == 0.0
    _drive([o32, o16], [p32, p16])
    for a, b in zip(p16, p32, strict=True):
        st = o16.inner.state[a]
        assert RESIDUAL_KEY in st
        assert ("m" in st) == (a.numel() <= 100)
        assert _bits_equal(_z(o16, a), b.data)


@pytest.mark.parametrize("route", _ROUTES)
def test_mixed_fp32_and_bf16_params_in_one_group(route):
    """Only the bf16 params carry a residual; both halves are the fp32 run bit for bit.

    Distinct shapes, so every foreach/fused bucket has the SAME composition in both runs:
    the fp32 optimizer's own stacked reductions are not invariant to how many same-shape
    params share a bucket on CUDA (a fact about the fp32 reference, not about kahan16)."""
    dev = _device(route)
    torch.manual_seed(0)
    shapes = [(32, 64), (16, 8), (64,), (), (4, 3, 3, 3), (48,), (8, 1), (24, 40)]
    base = [(torch.randn(s, device=dev) * 0.05).to(torch.bfloat16).float() for s in shapes]
    p32 = [torch.nn.Parameter(t.detach().clone()) for t in base]
    mixed = [torch.nn.Parameter(t.detach().clone().to(torch.bfloat16 if i % 2 == 0 else torch.float32))
             for i, t in enumerate(base)]
    kw = dict(lr=1e-3, **_route_kw(route))
    o32, om = Nekaon(p32, **kw), Nekaon(mixed, bf16_method="kahan16", **kw)
    _drive([o32, om], [p32, mixed])
    om.eval()
    o32.eval()
    for a, b in zip(mixed, p32, strict=True):
        assert (RESIDUAL_KEY in om.inner.state[a]) == (a.dtype == torch.bfloat16)
        assert _bits_equal(_z(om, a), b.data)


@pytest.mark.parametrize("method", ["kahan8", "kahan16"])
def test_params_without_grad_and_add_param_group(method):
    """A param that has no gradient for a while and a group added mid-run: residuals appear
    lazily with the state, no warning, the added group is compensated like the first."""
    ps = _bag()
    frozen = ps[1]
    o = Nekaon(ps[:4], lr=1e-4, bf16_method=method, foreach=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for step in range(6):
            for p in ps[:4]:
                p.grad = None if (p is frozen and step < 3) else (torch.randn(p.shape) * 0.02).to(torch.bfloat16)
            if step == 2:
                o.add_param_group({"params": ps[4:]})
            for p in ps[4:]:
                p.grad = (torch.randn(p.shape) * 0.02).to(torch.bfloat16) if step >= 2 else None
            o.step()
    o.eval()
    want = torch.uint8 if method == "kahan8" else torch.int16
    for p in ps:
        st = o.inner.state[p]
        assert st[RESIDUAL_KEY].dtype == want
        assert torch.isfinite(_z(o, p)).all()


@pytest.mark.parametrize("route", ["per_param", "foreach"] + (["fused"] if FUSED else []))
@pytest.mark.parametrize("method", ["kahan8", "kahan16"])
def test_resume_model_then_optimizer_is_bit_exact(route, method):
    """eval() -> save (model + optimizer) -> new model, load the MODEL first, then the
    optimizer -> train(): the resumed run is the continuous one bit for bit (weights and
    residuals), kahan8's residual noise stream included."""
    dev = _device(route)
    kw = dict(lr=1e-4, bf16_method=method, **_route_kw(route))
    pa = _bag(device=dev)
    oa = Nekaon(pa, **kw)
    _drive([oa], [pa], steps=4)
    oa.eval()
    sd_opt = copy.deepcopy(oa.state_dict())
    sd_w = [p.detach().clone() for p in pa]
    oa.train()
    pb = [torch.nn.Parameter(torch.zeros_like(p)) for p in pa]
    ob = Nekaon(pb, **kw)
    for p, w in zip(pb, sd_w, strict=True):               # 1) the model weights
        p.data.copy_(w)
    ob.load_state_dict(sd_opt)                             # 2) then the optimizer
    _drive([oa, ob], [pa, pb], steps=4, seed=11)
    for a, b in zip(pa, pb, strict=True):
        assert torch.equal(a.data, b.data)
        assert torch.equal(oa.inner.state[a][RESIDUAL_KEY], ob.inner.state[b][RESIDUAL_KEY])


def test_checkpoint_from_another_method_is_not_a_silent_downgrade():
    """torch restores every group hyperparameter from the checkpoint, ``bf16_method``
    included: a ``kahan8`` Nekaon loading an SR checkpoint used to train with SR from then
    on, silently. The load now warns; setting the method back after loading switches as a
    mid-run switch does (zero residual, one warning). A kahan8 checkpoint into kahan16 is
    the same story with an exact conversion."""
    import kaon._backend as backend
    backend._LAZY_RESIDUAL_WARNED = False
    backend._CONVERTED_RESIDUAL_WARNED = False
    for src, dst, match in (("stochastic_rounding", "kahan8", "kahan_lo"),
                            ("kahan8", "kahan16", "re-encoded")):
        pa = _bag()
        oa = Nekaon(pa, lr=1e-4, bf16_method=src)
        _drive([oa], [pa], steps=3)
        oa.eval()
        sd = copy.deepcopy(oa.state_dict())
        before = [_z(oa, p) for p in pa]
        pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
        ob = Nekaon(pb, lr=1e-4, bf16_method=dst)
        with pytest.warns(UserWarning, match="replaced"):
            ob.load_state_dict(sd)
        assert ob.param_groups[0]["bf16_method"] == src          # the checkpoint's, as torch does
        ob.eval()
        # the value is intact: exactly after an SR checkpoint (round-to-nearest climb round
        # trip); to kahan8's residual grid after the load's climb/removal pair
        assert _mean_ulp([_z(ob, p) for p in pb], before) < (1e-9 if src != "kahan8" else 0.02)
        ob.param_groups[0]["bf16_method"] = dst
        with pytest.warns(UserWarning, match=match):             # at train()'s climb
            ob.train()
            _drive([ob], [pb], steps=1, seed=5)
        want = torch.uint8 if dst == "kahan8" else torch.int16
        assert all(ob.inner.state[p][RESIDUAL_KEY].dtype == want for p in pb)


@pytest.mark.parametrize("route", ["per_param", "foreach"] + (["fused"] if FUSED else []))
def test_switch_back_to_sr_ignores_the_stale_residual(route):
    """kahan16 -> a plain method mid-run: the decay, the climb and decode_weights stop
    reading the residual (the writer no longer maintains it). Discriminating: the stale
    residual is overwritten with random bits (read anywhere, they would move the decayed
    value by up to an ulp, and the carry bit the stored bf16 pattern) and the run is compared
    BIT FOR BIT with an identical twin whose residuals are deleted. The switch goes to
    ``"none"`` (round-to-nearest) so the twins share no noise stream to keep aligned — the
    residual-reading code is the same for every non-compact-Kahan method; an SR switch is
    checked for finiteness on top."""
    dev = _device(route)
    runs = []
    for _ in range(2):                                      # kahan16 is noise-free: twins
        ps = _bag(device=dev)
        # wd*lr = 0.5: a stale residual read by the decay would move the decayed value by up
        # to half an ulp — enough to flip the round-to-nearest write (checked by mutation)
        o = Nekaon(ps, lr=1e-2, weight_decay=50.0, cautious_wd="full",
                   bf16_method="kahan16", **_route_kw(route))
        _drive([o], [ps], steps=3)
        for g in o.param_groups:
            g["bf16_method"] = "none"
        runs.append((ps, o))
    (pa, o), (pb, ob) = runs
    gen = torch.Generator().manual_seed(123)
    for p, q in zip(pa, pb, strict=True):
        assert torch.equal(p.data, q.data)
        lo = o.inner.state[p][RESIDUAL_KEY]                 # poison the stale residual
        lo.copy_(torch.randint(-32768, 32767, lo.shape, generator=gen, dtype=torch.int16))
        del ob.inner.state[q][RESIDUAL_KEY]                 # the twin has none at all
    _drive([o, ob], [pa, pb], steps=3, seed=9)
    o.eval()
    ob.eval()
    vals = kaon.decode_weights(o)
    for p, q in zip(pa, pb, strict=True):
        assert torch.equal(p.data, q.data), tuple(p.shape)
        assert torch.equal(vals[p], p.data.float())
    for g in o.param_groups:                                # and on to SR: still sane
        g["bf16_method"] = "stochastic_rounding"
    o.train()
    _drive([o], [pa], steps=2, seed=10)
    assert all(torch.isfinite(p.data).all() for p in pa)


@pytest.mark.parametrize("method,warns", [("stochastic_rounding", True), ("kahan8", False),
                                          ("kahan16", False)])
def test_no_spurious_inert_warning_at_low_lr(method, warns):
    """At lr 1e-5 (the low-LR regime kahan exists for) the Nekaon climb is below half a bf16
    ulp of typical weights: SR warns (its climb may round away), compact Kahan does not."""
    torch.manual_seed(0)
    p = torch.nn.Parameter((torch.randn(64, 64) * 0.05).to(torch.bfloat16))
    o = Nekaon([p], lr=1e-5, bf16_method=method)
    o.inert_check_interval = 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(60):
            p.grad = (torch.randn(64, 64) * 0.02).to(torch.bfloat16)
            o.step()
    fired = [w for w in caught if "lookahead" in str(w.message)]
    assert bool(fired) == warns, [str(w.message) for w in caught]


def test_gc_leaves_p_grad_uncentralized_only_under_compact_kahan():
    """The compact-Kahan native step centralizes its fp32 copy, not ``p.grad``; SR keeps the
    historical in-place bf16 GC."""
    for method, centralized in (("stochastic_rounding", True), ("kahan8", False)):
        p = torch.nn.Parameter((torch.randn(8, 16) * 0.05).to(torch.bfloat16))
        o = Adakaon([p], lr=1e-4, bf16_method=method, foreach=False)
        g = (torch.randn(8, 16) + 1.0).to(torch.bfloat16)
        p.grad = g.clone()
        o.step()
        moved = not torch.equal(p.grad, g)
        assert moved == centralized, method


# ------------------------------------------------------------------ 3. full-precision export
@pytest.mark.parametrize("route", ["foreach"] + (["fused"] if FUSED else []))
def test_decode_weights_is_the_fp32_master(route):
    dev = _device(route)
    p32, p16 = _bag(torch.float32, dev), _bag(torch.bfloat16, dev)
    kw = dict(lr=1e-3, **_route_kw(route))
    o32, o16 = Nekaon(p32, **kw), Nekaon(p16, bf16_method="kahan16", **kw)
    _drive([o32, o16], [p32, p16])
    with pytest.raises(RuntimeError, match="eval"):
        kaon.decode_weights(o16)                           # train mode: perturbed weights
    o16.eval()
    o32.eval()
    vals = kaon.decode_weights(o16)
    for a, b in zip(p16, p32, strict=True):
        assert vals[a].dtype == torch.float32 and _bits_equal(vals[a], b.data)
        assert vals[a].data_ptr() != a.data_ptr()


def test_full_precision_state_dict_keeps_names_and_buffers():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.BatchNorm1d(8)).to(torch.bfloat16)
    opt = Nekaon(model.parameters(), lr=1e-3, bf16_method="kahan16")
    for _ in range(3):
        opt.zero_grad()
        model(torch.randn(16, 8, dtype=torch.bfloat16)).float().pow(2).mean().backward()
        opt.step()
    opt.eval()
    sd = kaon.full_precision_state_dict(model, opt)
    ref = model.state_dict()
    assert list(sd) == list(ref)
    for name, p in model.named_parameters():
        assert sd[name].dtype == torch.float32
        assert torch.equal(sd[name].to(torch.bfloat16), p.data)          # nearest bf16 of it
    assert sd["1.running_mean"].dtype == torch.bfloat16                  # buffers untouched
    assert torch.equal(sd["1.num_batches_tracked"], ref["1.num_batches_tracked"])


def test_decode_weights_on_other_optimizers_and_lookahead():
    """Generic: any kaon optimizer with compact Kahan (Lion here), SR params upcast as-is,
    and Lookahead refused in eval view (its phi has no residual) but fine in train view."""
    ps = _bag()
    o = kaon.Lion(ps, lr=1e-4, bf16_method="kahan8")
    _drive([o], [ps], steps=2)
    vals = kaon.decode_weights(o)
    assert all(torch.equal(vals[p], _z(o, p)) for p in ps)
    ps2 = _bag()
    o2 = Adakaon(ps2, lr=1e-4)
    _drive([o2], [ps2], steps=1)
    assert all(torch.equal(v, p.data.float()) for p, v in kaon.decode_weights(o2).items())
    ps3 = _bag()
    la = kaon.Lookahead(ps3, lr=1e-4, bf16_method="kahan16")
    _drive([la], [ps3], steps=2)
    assert all(torch.equal(kaon.decode_weights(la)[p], _z(la, p)) for p in ps3)
    la.eval()
    with pytest.raises(RuntimeError, match="eval mode"):
        kaon.decode_weights(la)


@pytest.mark.parametrize("method", ["stochastic_rounding", "kahan16"])
def test_every_param_of_a_mixed_dtype_same_shape_group_is_climbed(method):
    """Regression (any bf16_method): the torch-path climb keyed its leftover buckets by
    (shape, momentum_dtype, group) — an fp32 and a bf16 param of one shape collided and the
    first was never climbed. Each must sit exactly ``k * lr * m`` (clamped) from its eval
    weight after a step."""
    torch.manual_seed(0)
    w = (torch.randn(64) * 0.05).to(torch.bfloat16)
    ps = [torch.nn.Parameter(w.clone()), torch.nn.Parameter(w.float())]
    o = Nekaon(ps, lr=1e-3, bf16_method=method, momentum_dtype="float32", foreach=False)
    for p in ps:
        p.grad = (torch.randn(64) * 0.02).to(p.dtype)
    o.step()
    live = [_z(o, p).clone() for p in ps]
    o.eval()
    for p, z in zip(ps, live, strict=True):
        assert float((z - _z(o, p)).abs().max()) > 1e-4, p.dtype        # a real climb


# ------------------------------------------------------------------ export: memory / misuse
def test_export_streams_to_the_requested_device():
    """``full_precision_state_dict`` defaults to CPU values; ``decode_weights(device=)``
    moves each value; ``None`` keeps the param's device."""
    ps = _bag()
    o = Nekaon(ps, lr=1e-3, bf16_method="kahan16")
    _drive([o], [ps], steps=2)
    o.eval()
    for v in kaon.decode_weights(o, device="cpu").values():
        assert v.device.type == "cpu" and v.dtype == torch.float32
    model = torch.nn.Module()
    for i, p in enumerate(ps):
        model.register_parameter(f"p{i}", p)
    sd = kaon.full_precision_state_dict(model, o)
    assert all(t.device.type == "cpu" and t.dtype == torch.float32 for t in sd.values())


def test_export_decodes_one_tensor_at_a_time(monkeypatch):
    """At most ONE decoded fp32 value is alive on the source device at any time: each is
    moved to the target before the next is decoded (checked by tracking the live
    source-device values through weak references)."""
    import weakref

    import kaon._full_precision as fp
    ps = _bag()
    o = Nekaon(ps, lr=1e-3, bf16_method="kahan16")
    _drive([o], [ps], steps=2)
    o.eval()
    live: list[weakref.ref] = []
    peak = [0]
    real = fp._decoded

    def spy(p, lo, method):
        v = real(p, lo, method)
        live.append(weakref.ref(v))
        peak[0] = max(peak[0], sum(r() is not None for r in live))
        return v

    monkeypatch.setattr(fp, "_decoded", spy)
    real_to = torch.Tensor.to
    # a "device" move that really copies, so the source value can die (CPU -> CPU would alias)
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: real_to(self, *a, **k).clone())
    fp.full_precision_state_dict(torch.nn.ParameterList(ps), o, device="meta")
    assert peak[0] == 1, peak[0]


@pytest.mark.skipif(not CUDA, reason="GPU memory peak")
def test_export_gpu_peak_is_one_tensor_not_the_model():
    """Streaming to CPU: after each tensor the GPU holds nothing extra (``memory_allocated``
    back to the baseline), and the PEAK is one tensor's decode, independent of the model
    size. Measured: :func:`kaon._compact_kahan.decode` on a 1M-element bf16 tensor peaks at
    17 MiB = 4.25x its fp32 size (the int32 pattern and residual, the carry temporaries and
    the fp32 result) — so the bound is 4.5x the largest tensor, against a 64 MiB fp32 model."""
    import kaon._full_precision as fp
    torch.manual_seed(0)
    ps = [torch.nn.Parameter((torch.randn(1024, 1024, device="cuda") * 0.05).to(torch.bfloat16))
          for _ in range(16)]
    o = Nekaon(ps, lr=1e-3, bf16_method="kahan16", foreach=True)
    for p in ps:
        p.grad = (torch.randn_like(p, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    o.step()
    o.eval()
    for p in ps:
        p.grad = None
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for _p, v in fp._iter_decoded(o, "cpu"):                          # one tensor at a time
        assert v.device.type == "cpu"
        assert torch.cuda.memory_allocated() == base                  # nothing left behind
    sd = kaon.full_precision_state_dict(torch.nn.ParameterList(ps), o)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - base
    largest = max(4 * p.numel() for p in ps)                          # 4 MiB
    model_fp32 = sum(4 * p.numel() for p in ps)                       # 64 MiB
    assert extra <= 4.5 * largest, (extra, largest)
    assert extra < model_fp32 // 3, (extra, model_fp32)
    assert all(t.device.type == "cpu" for t in sd.values())
    vals = kaon.decode_weights(o)                                     # device=None: on GPU
    assert all(v.is_cuda for v in vals.values())


def test_full_precision_state_dict_refuses_a_foreign_module():
    ps = _bag()
    o = Nekaon(ps, lr=1e-3, bf16_method="kahan8")
    _drive([o], [ps], steps=1)
    o.eval()
    other = torch.nn.ParameterList([torch.nn.Parameter(p.detach().clone()) for p in ps])
    with pytest.raises(ValueError, match="none of the optimizer's parameters"):
        kaon.full_precision_state_dict(other, o)


def test_export_view_rules_for_schedulefree_and_lookahead():
    """ScheduleFree (no compact Kahan): the train view is y — warns; eval exports x, no
    warning. Lookahead in eval (slow weights phi) is refused with a phi-specific message."""
    ps = _bag()
    sf = kaon.ScheduleFree(ps, lr=1e-3)
    _drive([sf], [ps], steps=2)
    with pytest.warns(UserWarning, match="training view"):
        kaon.decode_weights(sf)
    sf.eval()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vals = kaon.decode_weights(sf)
    assert all(torch.equal(vals[p], p.data.float()) for p in ps)
    ps2 = _bag()
    la = kaon.Lookahead(ps2, lr=1e-4, bf16_method="kahan8")
    _drive([la], [ps2], steps=2)
    la.eval()
    with pytest.raises(RuntimeError, match="slow weights phi"):
        kaon.decode_weights(la)


@pytest.mark.skipif(not FUSED, reason="Triton fused kernels need CUDA + Triton")
@pytest.mark.parametrize("dst", ["none", "kahan"])
def test_fused_partition_follows_a_bf16_method_switch(dst):
    """Regression: the fused partition cache was keyed on the param witness and the state
    generation only, so after a mid-run switch to a method the kernels cannot write
    ("none", legacy "kahan") the bf16 params stayed on the fused routes and were written
    with stochastic rounding. They must move to the native path on the next step."""
    ps = _bag(device="cuda")
    o = Adakaon(ps, lr=1e-3, bf16_method="kahan16", fused=True)
    _drive([o], [ps], steps=2)
    routes = next(iter(o._fused_part.values()))[-4:]
    assert sum(map(len, routes[:3])) > 0                              # fused before
    o.param_groups[0]["bf16_method"] = dst
    _drive([o], [ps], steps=1, seed=3)
    one_block, big, one_dim, native = next(iter(o._fused_part.values()))[-4:]
    assert not (one_block or big or one_dim)
    assert len(native) == len(ps)


def test_adakaon_fused_sr_flag_follows_the_method():
    """SR constexpr from the group's method: SR for SR, off for compact Kahan (CK writes),
    refused for a bf16 bucket under a method no fused kernel implements."""
    f = Adakaon._fused_sr
    assert f({"bf16_method": "stochastic_rounding"}, True) is True
    assert f({"bf16_method": "kahan8"}, True) is False
    assert f({"bf16_method": "kahan16"}, True) is False
    assert f({"bf16_method": "none"}, False) is False
    for m in ("none", "kahan"):
        with pytest.raises(RuntimeError, match="stale routing"):
            f({"bf16_method": m}, True)


@pytest.mark.skipif(not FUSED, reason="Triton fused kernels need CUDA + Triton")
@pytest.mark.parametrize("cls", [Adakaon, Nekaon])
@pytest.mark.parametrize("md", ["4bit", "int8", "bfloat16"])
def test_kahan16_negative_weight_decay_is_the_fp32_run(cls, md):
    """weight_decay is not validated >= 0. The fused keep passes decode only the lanes whose
    cautious sign can depend on the residual, with a threshold proportional to wd: it has to
    use |wd|, or a negative wd leaves no lane ambiguous and the mask diverges from the full
    decode (regression caught in review: 274k mismatching lanes of 25M at wd=-0.1)."""
    p32 = _bag(torch.float32, "cuda", big=True)
    p16 = _bag(torch.bfloat16, "cuda", big=True)
    kw = dict(lr=1e-3, momentum_dtype=md, weight_decay=-0.1, cautious=True, cautious_wd="masked",
              fused=True, deterministic_reductions=True)
    o32 = cls(p32, **kw)
    o16 = cls(p16, bf16_method="kahan16", **kw)
    _drive([o32, o16], [p32, p16])
    for a, b in zip(p16, p32, strict=True):
        assert _bits_equal(_z(o16, a), b.data), (md, tuple(a.shape))


def test_switch_to_legacy_kahan_seeds_shift_from_the_residual():
    """kahan16 -> legacy "kahan" mid-run: the new bf16 ``shift`` starts at the value's sub-ulp
    part (decoded residual, rounded to bf16), not at zero — so the value the legacy pair
    ``p + shift`` carries after the switch is the kahan16 value to bf16-of-residual precision
    (a zero seed would drop up to half an ulp)."""
    ps = _bag()
    o = Adakaon(ps, lr=1e-3, bf16_method="kahan16", foreach=False)
    _drive([o], [ps], steps=3)
    z = {p: _z(o, p) for p in ps}
    o.param_groups[0]["bf16_method"] = "kahan"
    from kaon._backend import subtract_one_
    for p in ps:
        subtract_one_(p, torch.zeros(p.shape), o.state[p], "kahan")   # a zero step
        carried = p.data.float() + o.state[p]["shift"].float()
        ulp = torch.exp2(torch.floor(torch.log2(z[p].abs().clamp_min(1e-30))) - 7)
        assert float(((carried - z[p]).abs() / ulp).max()) < 0.01, tuple(p.shape)
        assert float(((p.data.float() - z[p]).abs() / ulp).max()) > 0.05   # there WAS a residual
