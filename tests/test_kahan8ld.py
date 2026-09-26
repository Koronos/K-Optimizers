"""``bf16_method="kahan8ld"`` (EXPERIMENTAL): kahan8 with a low-discrepancy residual dither.

Covers, per ``docs/research/compact-kahan/lowdisc/``:

* the dither — ``u8 = (h8(i) + r8(i, n // 256) + 159 n) mod 256``: every aligned block of 256
  writes hits each noise value exactly once per element (so the codec is EXACTLY unbiased over
  a full period), the per-element hash depends on the parameter's key and the element's
  index only, and the Triton helpers are its bit twins (noise, the keyed ``ck_ptr`` handle,
  the one-launch axpy);
* parity: per-param == foreach BIT FOR BIT (the noise is deterministic, no pinning needed),
  also when the bucket composition changes (a param without grad drops out of the stack);
  fused == native bit for bit on the 1-D kernel and within the fp32 paths' own difference
  elsewhere (far below the kahan8 SR spread);
* checkpoint: a resume continues bit-identically (the counter is the checkpointed step);
* method switches kahan8 <-> kahan8ld keep the very same uint8 residual (no conversion, no
  warning); the other optimizers and Lookahead's sync refuse it loudly;
* MSAM / Nekaon: three writes per step with distinct counters, climb on the decoded value,
  per-param/foreach/fused agreement, eval/train and resume;
* the point of it: a coherent sub-grid drift is tracked with a bounded error where the SR
  residual walks.
"""
from __future__ import annotations

import copy
import warnings

import pytest
import torch

import kaon
from kaon import MSAM, Adakaon, Lion, Lookahead, Nekaon
from kaon import _backend as bk
from kaon._compact_kahan import (
    CK_LD8,
    RESIDUAL_KEY,
    LDNoise,
    compensated_add_,
    decode,
    encode_,
    ld_key,
    ld_noise,
)

CUDA = torch.cuda.is_available()
try:
    from kaon._fused_triton import HAS_TRITON
except Exception:  # pragma: no cover
    HAS_TRITON = False
FUSED = CUDA and HAS_TRITON
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])


def _ulp_ref(z: torch.Tensor) -> float:
    rms = z.float().pow(2).mean().sqrt()
    return float(torch.exp2(torch.floor(torch.log2(rms)) - 7))


def _z(p: torch.Tensor, st: dict) -> torch.Tensor:
    return decode(p.data, st[RESIDUAL_KEY], 8)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def _bag(seed=0, device="cpu", big=False):
    torch.manual_seed(seed)
    shapes = [(8, 16), (8, 16), (16, 4), (16,), (16,), (), (4, 3, 3, 3)]
    if big:
        shapes += [(1024, 1200)]
    return [torch.nn.Parameter((torch.randn(s, device=device) * 0.05).to(torch.bfloat16)) for s in shapes]


def _drive(opts, params_lists, steps=8, seed=7, skip=None):
    """Same gradients for every run; ``skip`` = index of a param left without grad on odd steps
    (it drops out of its foreach bucket, which changes the bucket's composition)."""
    gg = torch.Generator().manual_seed(seed)
    for s in range(steps):
        grads = [(torch.randn(p.shape, generator=gg) * 0.02) for p in params_lists[0]]
        for params in params_lists:
            for i, (p, g) in enumerate(zip(params, grads, strict=True)):
                if skip is not None and i == skip and s % 2:
                    p.grad = None
                else:
                    p.grad = g.to(device=p.device, dtype=p.dtype).clone()
        for o in opts:
            o.step()


# ------------------------------------------------------------------------------ the dither
@pytest.mark.parametrize("device", DEVICES)
def test_every_block_of_256_writes_is_a_permutation_of_the_noise(device):
    """Per element, the 256 writes of an aligned block take every noise value exactly once
    (159 is odd; the block phase is constant inside the block) — the full-period property the
    codec's unbiasedness rests on. Also: blocks re-draw the phase (they are not all equal)."""
    keys = [ld_key(i) for i in (0, 1, 7, 12345)]
    for blk in (0, 1, 1000):
        u = torch.stack([ld_noise(keys, 300, n, device) for n in range(256 * blk, 256 * (blk + 1))])
        assert u.min() >= 0 and u.max() <= 255
        srt = u.sort(dim=0).values
        want = torch.arange(256, device=device, dtype=torch.int32).view(256, 1, 1)
        assert torch.equal(srt, want.expand_as(srt))
    first = [ld_noise(keys, 300, 256 * b, device) for b in range(4)]
    assert not all(torch.equal(first[0], f) for f in first[1:])


@pytest.mark.parametrize("device", DEVICES)
def test_codec_is_exactly_unbiased_over_a_full_period(device):
    """Encoding the same value with the 256 noises of one block and averaging the decoded
    values gives the value back EXACTLY (float64 mean), for arbitrary values/elements."""
    g = torch.Generator().manual_seed(0)
    z0 = (torch.randn(4, 500, generator=g) * 0.05).to(device)
    z0[0, :3] = torch.tensor([1.0 + 0.3 * 2.0 ** -15, -2.0 + 0.7 * 2.0 ** -14, 3e-3], device=device)
    keys = [ld_key(i) for i in range(4)]
    acc = torch.zeros(4, 500, dtype=torch.float64, device=device)
    for n in range(256 * 5, 256 * 6):
        p = torch.empty(4, 500, dtype=torch.bfloat16, device=device)
        lo = torch.empty(4, 500, dtype=torch.uint8, device=device)
        encode_(z0.clone(), p, lo, 8, ld_noise(keys, 500, n, device))
        acc += decode(p, lo, 8).double()
    # the kept value is on the 16-bit-significand grid: exact mean == the fp32 input
    assert torch.equal(acc / 256, z0.double())


def test_hash_depends_on_the_key_and_the_index_only():
    """A row's noise is the same whatever the other rows of the stack are (bucket-composition
    invariance) and differs between params (keys) and elements."""
    a = ld_noise([ld_key(3)], 64, 77, "cpu")
    b = ld_noise([ld_key(9), ld_key(3), ld_key(1)], 64, 77, "cpu")
    assert torch.equal(a[0], b[1])
    assert not torch.equal(b[0], b[1])
    assert len(set(ld_key(i) for i in range(10000))) == 10000


def test_ld_noise_object_validates_its_counter_and_rows():
    with pytest.raises(ValueError, match="counter"):
        LDNoise([1], -1)
    with pytest.raises(ValueError, match="counter"):
        LDNoise([1], 1 << 31)
    with pytest.raises(ValueError, match="tile"):
        LDNoise([1, 2, 3], 0).noise(torch.Size([4, 5]), torch.device("cpu"))
    p = torch.zeros(4, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="8-bit"):
        compensated_add_(p, torch.zeros(4, dtype=torch.int16), torch.ones(4), 1.0, 16, LDNoise([1], 0))


def test_coherent_sub_grid_drift_is_bounded_where_sr_walks():
    """The reason for the method: a constant step of ~1/8 of the residual grid, 4000 writes.
    The LD dither realises it like a sigma-delta modulator (error stays ~a grid unit); the SR
    residual walks as sqrt(N)."""
    torch.manual_seed(0)
    z0 = (torch.randn(1 << 14) * 0.05).to(torch.bfloat16).float()
    u = _ulp_ref(z0)
    step = (torch.randint(0, 2, z0.shape).float() * 2 - 1) * (u / 256 / 8)
    runs = {}
    for name in ("ld", "sr"):
        p = z0.to(torch.bfloat16)
        lo = torch.zeros(z0.shape, dtype=torch.uint8)
        gen = kaon._stochastic_rounding.SRStream()
        for n in range(4000):
            compensated_add_(p, lo, step, 1.0, 8, LDNoise([ld_key(0)], n) if name == "ld" else gen)
        runs[name] = float(((decode(p, lo, 8) - (z0 + 4000 * step)) / u).std())
    assert runs["ld"] < 0.01, runs
    assert runs["sr"] > 5 * runs["ld"], runs


# ------------------------------------------------------------------------------ Triton twins
@pytest.mark.skipif(not FUSED, reason="Triton fused kernels need CUDA + Triton")
def test_triton_noise_and_keyed_handle_are_the_torch_bits():
    import triton
    import triton.language as tl

    import kaon._fused_triton as ft

    @triton.jit(do_not_specialize=["key", "n"])
    def _noise(out_ptr, key, n, N, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, ft.ld_noise_dev(key.to(tl.uint32), offs, n.to(tl.uint32)),
                 mask=offs < N)

    @triton.jit
    def _write(p_addr, c_addr, d_ptr, n, K, seed, CK: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        t = pid // K
        offs = (pid % K) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        pp = tl.load(p_addr + t).to(tl.pointer_type(tl.bfloat16))
        cp = ft.ck_ptr(c_addr, t, CK)
        e = tl.load(d_ptr + t * n + offs, mask=mask, other=0.0)
        ft.ck_store(pp, cp, offs, mask, ft.ck_decode(pp, cp, offs, mask, CK) + e, seed + t, CK)

    N = 5000
    for key in (ld_key(0), ld_key(7), 0xFFFFFFFF, 0):
        for n in (0, 1, 255, 256, 12345, (1 << 31) - 1):
            out = torch.empty(N, dtype=torch.int32, device="cuda")
            _noise[(triton.cdiv(N, 1024),)](out, key, n, N, BLOCK=1024)
            assert torch.equal(out, ld_noise([key], N, n, "cuda").view(-1)), (key, n)

    torch.manual_seed(0)
    L, T = 3000, 3
    ps = [(torch.randn(L, device="cuda") * 0.05).to(torch.bfloat16) for _ in range(T)]
    los = [torch.randint(0, 256, (L,), device="cuda", dtype=torch.uint8) for _ in range(T)]
    d = torch.randn(T, L, device="cuda") * 1e-5
    keys = [ld_key(i) for i in (5, 2, 9)]
    pr, lr_ = [p.clone() for p in ps], [lo.clone() for lo in los]
    for t in range(T):
        compensated_add_(pr[t], lr_[t], d[t], 1.0, 8, LDNoise([keys[t]], 77))
    K = triton.cdiv(L, 1024)
    _write[(T * K,)](ft.ptr_array(ps, "cuda"), ft.ld_ptr_array(los, keys, "cuda"), d, L, K, 77,
                     CK=CK_LD8, BLOCK=1024)
    for t in range(T):
        assert _same(ps[t], pr[t]) and torch.equal(los[t], lr_[t]), t


@pytest.mark.skipif(not FUSED, reason="Triton fused kernels need CUDA + Triton")
def test_triton_axpy_is_the_torch_reference_bit_for_bit():
    import kaon._fused_triton as ft
    torch.manual_seed(1)
    w = (torch.randn(3, 777, device="cuda") * 0.05).to(torch.bfloat16)
    c = torch.randint(0, 256, (3, 777), device="cuda", dtype=torch.uint8)
    d = torch.randn(3, 777, device="cuda") * 1e-5
    keys = [ld_key(i) for i in (4, 0, 2)]
    w2, c2 = w.clone(), c.clone()
    assert ft.ck_add_supported(w, c, d, 8)
    for n in (1, 300, 70000):
        ft.ck_add_(w, c, d, -0.5, 8, LDNoise(keys, n))
        compensated_add_(w2, c2, d, -0.5, 8, LDNoise(keys, n))
        assert _same(w, w2) and torch.equal(c, c2), n


# ------------------------------------------------------------------------------ optimizer parity
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cfg", [
    dict(momentum_dtype="bfloat16"),
    dict(momentum_dtype="int8", weight_decay=0.01),
    dict(momentum_dtype="4bit", cautious=True),
    dict(betas=(0.0, 0.999)),
])
def test_foreach_matches_per_param_bit_exact(device, cfg):
    """Deterministic noise keyed per param: per-param == foreach bit for bit, weights AND
    residual bytes, with NO pinning — and still when a param without grad drops out of its
    bucket on odd steps (the stack composition changes; the keys do not)."""
    pa, pb = _bag(device=device), _bag(device=device)
    kw = dict(lr=1e-4, bf16_method="kahan8ld", **cfg)
    oa = Adakaon(pa, foreach=False, **kw)
    ob = Adakaon(pb, foreach=True, **kw)
    _drive([oa, ob], [pa, pb], steps=8, skip=1)
    for a, b in zip(pa, pb, strict=True):
        assert _same(a.data, b.data)
        assert torch.equal(oa.state[a][RESIDUAL_KEY], ob.state[b][RESIDUAL_KEY])
        assert ob.state[b][RESIDUAL_KEY].dtype == torch.uint8


@pytest.mark.skipif(not FUSED, reason="Triton fused kernels need CUDA + Triton")
@pytest.mark.parametrize("md", ["bfloat16", "float32", "int8", "4bit"])
def test_fused_routes_match_native(md):
    """Fused kahan8ld: the 1-D/0-D kernel is the native computation bit for bit; the factored
    kernels differ from native only by the fp32 paths' own difference (reduction order, FMA
    contraction), i.e. a few residual-grid units — against the ~0.1 ulp spread two SR noise
    streams give kahan8 over the same run. The big batched bucket is covered too."""
    import kaon._fused_triton as ft
    pf, pn = _bag(device="cuda", big=True), _bag(device="cuda", big=True)
    kw = dict(lr=1e-4, momentum_dtype=md, bf16_method="kahan8ld", gradient_centralization=False)
    of = Adakaon(pf, fused=True, deterministic_reductions=True, **kw)
    on = Adakaon(pn, fused=False, foreach=False, **kw)
    _drive([of, on], [pf, pn], steps=8)
    parts = of._fused_partition(of.param_groups[0], pf, ft)
    assert parts[1], "the big bucket must take the batched fused route"
    one_dim = {id(q) for q in parts[2]}
    assert one_dim
    for i, (a, b) in enumerate(zip(pf, pn, strict=True)):
        za, zb = _z(a, of.state[a]), _z(b, on.state[b])
        if id(a) in one_dim:
            assert _same(a.data, b.data) and torch.equal(of.state[a][RESIDUAL_KEY],
                                                         on.state[b][RESIDUAL_KEY]), i
        tol = 0.8 if (md == "4bit" and a.numel() > 100000) else 0.05
        assert float((za - zb).abs().max()) / _ulp_ref(zb) < tol, (i, md)


@pytest.mark.parametrize("route", ["per_param", "foreach"] + (["fused"] if FUSED else []))
def test_resume_is_bit_exact(route):
    device = "cuda" if route == "fused" else "cpu"
    pb = _bag(device=device)
    kw = dict(lr=1e-4, bf16_method="kahan8ld", fused=route == "fused", foreach=route != "per_param")
    ob = Adakaon(pb, **kw)
    _drive([ob], [pb], steps=5)
    sd = copy.deepcopy(ob.state_dict())
    assert sd["_adakaon_meta"]["fused_step"] == 5
    pc = [torch.nn.Parameter(p.detach().clone()) for p in pb]
    oc = Adakaon(pc, **kw)
    oc.load_state_dict(sd)
    _drive([ob, oc], [pb, pc], steps=5, seed=11)
    for b, c in zip(pb, pc, strict=True):
        assert torch.equal(b.data, c.data)
        assert torch.equal(ob.state[b][RESIDUAL_KEY], oc.state[c][RESIDUAL_KEY])


def test_native_step_counter_advances_and_is_checkpointed():
    p = [torch.nn.Parameter(torch.zeros(4, 4))]
    o = Adakaon(p, lr=1e-3)
    _drive([o], [p], steps=3)
    assert o._t == 3 and o.state_dict()["_adakaon_meta"]["fused_step"] == 3


# ------------------------------------------------------------------------------ switches / validation
@pytest.mark.parametrize("route", ["per_param", "foreach"] + (["fused"] if FUSED else []))
def test_switching_between_kahan8_and_kahan8ld_keeps_the_residual(route):
    """Same codec, same uint8 state: a switch in either direction reuses the residual tensor
    as is (no conversion, no warning) and the decoded value carries straight through."""
    device = "cuda" if route == "fused" else "cpu"
    pa = _bag(device=device)
    o = Adakaon(pa, lr=1e-4, bf16_method="kahan8", fused=route == "fused",
                foreach=route != "per_param")
    _drive([o], [pa], steps=3)
    for method in ("kahan8ld", "kahan8", "kahan8ld"):
        before = {id(p): o.state[p][RESIDUAL_KEY] for p in pa}
        o.param_groups[0]["bf16_method"] = method
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _drive([o], [pa], steps=2)
        for p in pa:
            assert o.state[p][RESIDUAL_KEY] is before[id(p)]
            assert o.state[p][RESIDUAL_KEY].dtype == torch.uint8
    assert all(torch.isfinite(p.float()).all() for p in pa)


def test_state_costs_exactly_one_byte_per_param():
    pa = _bag()
    o = Adakaon(pa, lr=1e-4, bf16_method="kahan8ld", betas=(0.0, 0.999))
    _drive([o], [pa], steps=1)
    for p in pa:
        assert o.state[p][RESIDUAL_KEY].numel() == p.numel()
        assert o.state[p][RESIDUAL_KEY].element_size() == 1


def test_only_adakaon_family_accepts_it_and_other_writers_refuse_it():
    p = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="bf16_method"):
        Lion([p], bf16_method="kahan8ld")
    with pytest.raises(ValueError, match="bf16_method"):
        kaon.ScheduleFree([p], bf16_method="kahan8ld")
    Adakaon([p], bf16_method="kahan8ld")
    # a writer handed no LDNoise (Lion with a group switched to it) is refused, not rounded SR
    q = [torch.nn.Parameter(torch.randn(4, 4).to(torch.bfloat16))]
    o = Lion(q, lr=1e-3, bf16_method="kahan8")
    o.param_groups[0]["bf16_method"] = "kahan8ld"
    q[0].grad = torch.ones_like(q[0])
    with pytest.raises(NotImplementedError, match="kahan8ld"):
        o.step()
    with pytest.raises(NotImplementedError, match="kahan8ld"):
        bk.subtract_batched_([q[0].data], torch.ones(1, 4, 4), "kahan8ld",
                             comp=[torch.zeros(4, 4, dtype=torch.uint8)])


def test_lookahead_sync_refuses_kahan8ld_loudly():
    q = [torch.nn.Parameter(torch.randn(8, 8).to(torch.bfloat16))]
    o = Lookahead(q, lr=1e-3, k=1, bf16_method="kahan8ld")
    q[0].grad = torch.ones_like(q[0])
    with pytest.raises(NotImplementedError, match="kahan8ld"):
        o.step()


# ------------------------------------------------------------------------------ MSAM / Nekaon
def _climb_runs(kind, route, methods, steps=60, seed=0, lr=1e-5):
    device = "cuda" if route == "fused" else "cpu"
    torch.manual_seed(seed)
    kaon.reseed_stochastic_rounding()
    w = (torch.randn(64, 64, device=device) * 0.05).to(torch.bfloat16)
    b = (torch.randn(64, device=device) * 0.05).to(torch.bfloat16)
    kw = dict(lr=lr, betas=(0.9, 0.999), cautious=False, fused=route == "fused",
              foreach=route != "per_param")
    runs = {}
    for method in methods:
        ps = [torch.nn.Parameter(w.float() if method is None else w.clone()),
              torch.nn.Parameter(b.float() if method is None else b.clone())]
        mk = {} if method is None else {"bf16_method": method}
        if kind == "msam":
            o = MSAM(ps, rho=0.05, **kw, **mk)
        else:
            o = Nekaon(ps, k=1.5, momentum_dtype="bfloat16", weight_decay=0.0, **kw, **mk)
        o.train()
        runs[method] = (ps, o)
    gg = torch.Generator(device=device).manual_seed(seed + 1)
    g0 = [torch.randn(64, 64, generator=gg, device=device), torch.randn(64, generator=gg, device=device)]
    for _ in range(steps):
        gs = [(g + 0.3 * torch.randn(g.shape, generator=gg, device=device)).to(torch.bfloat16) for g in g0]
        for ps, o in runs.values():
            for p, g in zip(ps, gs, strict=True):
                p.grad = g.to(p.dtype)
            o.step()
    return runs


@pytest.mark.parametrize("kind", ["msam", "nekaon"])
@pytest.mark.parametrize("route", ["per_param", "foreach"] + (["fused"] if FUSED else []))
def test_msam_nekaon_climb_tracks_fp32_and_widens_the_counter(kind, route):
    """The base step, the removal and the climb each get their own counter (stride 3 on the
    owner); the decoded-value climb keeps the kahan8ld run within a small fraction of an ulp
    of the fp32 run, clean weights after ``eval()``."""
    runs = _climb_runs(kind, route, (None, "kahan8ld"))
    (p32, o32), (pld, old) = runs[None], runs["kahan8ld"]
    assert old.inner._ld_stride == 3 and old.inner._ld_slot == 1
    o32.eval()
    old.eval()
    for a, b in zip(p32, pld, strict=True):
        u = _ulp_ref(a.data)
        err = float(((_z(b, old.inner.state[b]) - a.data).abs() / u).max())
        assert err < 0.3, (route, err)


@pytest.mark.parametrize("kind", ["msam", "nekaon"])
def test_msam_nekaon_per_param_equals_foreach(kind):
    ra = _climb_runs(kind, "per_param", ("kahan8ld",))["kahan8ld"]
    rb = _climb_runs(kind, "foreach", ("kahan8ld",))["kahan8ld"]
    for a, b in zip(ra[0], rb[0], strict=True):
        assert _same(a.data, b.data)
        assert torch.equal(ra[1].inner.state[a][RESIDUAL_KEY], rb[1].inner.state[b][RESIDUAL_KEY])


@pytest.mark.skipif(not FUSED, reason="Triton fused kernels need CUDA + Triton")
def test_nekaon_fused_climb_plan_is_keyed_and_rebuilt_on_a_method_switch():
    pa = _bag(device="cuda")
    o = Nekaon(pa, lr=1e-4, k=1.5, momentum_dtype="bfloat16", bf16_method="kahan8ld",
               weight_decay=0.0, fused=True)
    o.train()
    _drive([o], [pa], steps=2)
    cache = o._axpy_cache
    lowp = [b for b in cache["buckets"] if b["lowp"]]
    assert lowp and all(b["ck"] == CK_LD8 for b in lowp)
    assert all(b["c_addr"].numel() == 2 * b["N"] for b in lowp)       # [ptr, key] per tensor
    assert o._plan_addrs_valid(cache)
    o.param_groups[0]["bf16_method"] = "kahan8"
    assert not o._plan_addrs_valid(cache)
    _drive([o], [pa], steps=1)
    assert all(b["ck"] == 8 for b in o._axpy_cache["buckets"] if b["lowp"])


@pytest.mark.parametrize("kind", ["msam", "nekaon"])
@pytest.mark.parametrize("route", ["foreach"] + (["fused"] if FUSED else []))
def test_msam_nekaon_resume_from_eval_checkpoint_is_bit_exact(kind, route):
    runs = _climb_runs(kind, route, ("kahan8ld",), steps=6)
    ps, o = runs["kahan8ld"]
    o.eval()
    sd = copy.deepcopy(o.state_dict())
    qs = [torch.nn.Parameter(p.detach().clone()) for p in ps]
    kw = dict(lr=1e-5, betas=(0.9, 0.999), cautious=False, fused=route == "fused",
              foreach=True, bf16_method="kahan8ld")
    if kind == "msam":
        o2 = MSAM(qs, rho=0.05, **kw)
    else:
        o2 = Nekaon(qs, k=1.5, momentum_dtype="bfloat16", weight_decay=0.0, **kw)
    o2.load_state_dict(sd)
    o.train()
    o2.train()
    for a, b in zip(ps, qs, strict=True):
        assert _same(a.data, b.data)
    gg = torch.Generator().manual_seed(5)
    for _ in range(4):
        gs = [torch.randn(p.shape, generator=gg).to(device=p.device, dtype=torch.bfloat16) for p in ps]
        for params, opt in ((ps, o), (qs, o2)):
            for p, g in zip(params, gs, strict=True):
                p.grad = g.clone()
            opt.step()
    for a, b in zip(ps, qs, strict=True):
        assert _same(a.data, b.data)
        assert torch.equal(o.inner.state[a][RESIDUAL_KEY], o2.inner.state[b][RESIDUAL_KEY])
