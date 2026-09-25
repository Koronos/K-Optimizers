"""``bf16_method="kahan8"``: the compact (1 B/param) fixed-point Kahan weight write.

Covers, per ``docs/research/compact-kahan.md``:

* the codec — ``(bf16, residual byte)`` <-> fp32 on a 16-bit-significand grid: exactness,
  round-half-away stored weight, binade crossings (the ``+1`` carry), zero/subnormals,
  non-finite propagation, idempotence, unbiased stochastic rounding of the residual;
* parity per-param == foreach (bit-exact on the torch path with the residual noise pinned),
  fused == native (bounded), the Triton one-launch axpy == the torch reference (bounded);
* checkpoint: ``kahan_lo`` is saved as uint8, restored bit-exactly, and a resumed run
  continues bit-identically to the uninterrupted one (noise stream included);
* accuracy: sub-ulp steps tracked to a small fraction of an ulp where plain stochastic
  rounding walks by many ulps, and no stall below the residual grid;
* memory: exactly +1 B/param of state; routing: every fused route admits it, the legacy
  ``kahan`` still takes the per-param path, fp16 params are refused.
"""
from __future__ import annotations

import copy

import pytest
import torch

import kaon
from kaon import Adakaon, Lion
from kaon import _backend as bk
from kaon._compact_kahan import (
    RESIDUAL_KEY,
    compensated_add_,
    decode,
    encode_,
)
from kaon._momentum_codec import load_state_dict_preserving_dtypes

CUDA = torch.cuda.is_available()
try:
    from kaon._fused_triton import HAS_TRITON
except Exception:  # pragma: no cover
    HAS_TRITON = False
FUSED = CUDA and HAS_TRITON
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])


def _ulp_ref(z: torch.Tensor) -> float:
    """bf16 ulp at the RMS magnitude of ``z`` (the sim's error unit)."""
    rms = z.float().pow(2).mean().sqrt()
    return float(torch.exp2(torch.floor(torch.log2(rms)) - 7))


def _grid_unit(z: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Per-coordinate residual grid unit ``ulp(z) / 2**bits`` (bf16 subnormal floor)."""
    e = torch.floor(torch.log2(z.abs().clamp_min(2.0 ** -126)))
    e = torch.where(z.abs() < 2.0 ** -126, torch.full_like(e, -126.0), e)
    return torch.exp2(e - 7 - bits)


# ------------------------------------------------------------------------------ codec
@pytest.mark.parametrize("device", DEVICES)
def test_codec_grid_bound_and_idempotence(device):
    torch.manual_seed(0)
    n = 1 << 15
    z = torch.randn(n, device=device) * torch.exp2(torch.randint(-140, 120, (n,), device=device).float())
    z0 = z.clone()
    p = torch.empty(n, dtype=torch.bfloat16, device=device)
    lo = torch.empty(n, dtype=torch.uint8, device=device)
    encode_(z, p, lo, 8, None)
    zd = decode(p, lo, 8)
    # (1) stored value lies on the 16-significant-bit grid (low 8 fp32 mantissa bits zero)
    assert int((zd.view(torch.int32) & 0xFF).abs().max()) == 0
    # (2) within half a grid unit of the input (round half away)
    assert bool(((zd - z0).abs() <= 0.5 * _grid_unit(z0) + 1e-45).all())
    # (3) the stored bf16 is within half a bf16 ulp of the stored value (nearest bf16)
    assert bool(((zd - p.float()).abs() <= (p.float().abs() * 2.0 ** -8).clamp_min(2.0 ** -134) + 1e-45).all())
    # (4) encode(decode(.)) is the identity
    p2, lo2 = p.clone(), lo.clone()
    encode_(zd.clone(), p2, lo2, 8, None)
    assert torch.equal(p2.view(torch.int16), p.view(torch.int16)) and torch.equal(lo2, lo)


@pytest.mark.parametrize("device", DEVICES)
def test_codec_binade_crossing_zero_and_subnormals(device):
    top = 2.0 - 2.0 ** -7                           # largest bf16 below 2.0
    cases = torch.tensor([
        top + 0.6 * 2.0 ** -7,                      # rounds UP across the binade -> w = 2.0
        -(top + 0.6 * 2.0 ** -7),                   # and its negative
        2.0 - 2.0 ** -15,                           # just below 2.0: w = 2.0, q encodes -1/256 ulp
        1.0 - 2.0 ** -16,                           # just below 1.0 (finer binade below)
        0.0, -0.0,
        2.0 ** -133,                                # smallest bf16 subnormal
        2.0 ** -134,                                # half of it: rounds half away -> 2**-133
        0.3 * 2.0 ** -133,                          # below half: stays 0 with a residual
        2.0 ** -126,                                # smallest bf16 normal
        (2.0 ** -126) * (1 - 2.0 ** -9),            # just below it (largest subnormal region)
    ], dtype=torch.float32, device=device)
    z = cases.clone()
    p = torch.empty_like(cases, dtype=torch.bfloat16)
    lo = torch.empty(cases.shape, dtype=torch.uint8, device=device)
    encode_(z, p, lo, 8, None)
    zd = decode(p, lo, 8)
    assert bool(((zd - cases).abs() <= 0.5 * _grid_unit(cases) + 1e-45).all())
    assert p[0].item() == 2.0 and p[1].item() == -2.0
    assert p[2].item() == 2.0 and lo[2].item() == 0xFF                 # 2.0 - 1 unit of [1,2)
    assert p[4].item() == 0.0 and lo[4].item() == 0
    assert p[5].view(torch.int16).item() == -0x8000 and lo[5].item() == 0   # -0.0 preserved
    assert p[6].item() == 2.0 ** -133 and p[7].item() == 2.0 ** -133
    assert p[8].item() == 0.0 and lo[8].item() > 0                     # sub-half-ulp residual at 0
    assert zd[8].item() == pytest.approx(0.3 * 2.0 ** -133, rel=2.0 ** -7)


@pytest.mark.parametrize("device", DEVICES)
def test_codec_nonfinite_propagates(device):
    z = torch.tensor([float("nan"), float("inf"), -float("inf"), 3.5e38, -3.5e38], device=device)
    p = torch.empty_like(z, dtype=torch.bfloat16)
    lo = torch.full(z.shape, 7, dtype=torch.uint8, device=device)
    encode_(z.clone(), p, lo, 8, torch.randint(0, 256, z.shape, dtype=torch.int32, device=device))
    assert torch.isnan(p[0]) and p[1] == float("inf") and p[2] == -float("inf")
    assert p[3] == float("inf") and p[4] == -float("inf")              # above bf16 max: like RNE
    assert int(lo[:3].max()) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_residual_stochastic_rounding_is_unbiased(device):
    torch.manual_seed(0)
    n = 1 << 18
    for frac in (0.13, 0.5, 0.87):
        z = torch.full((n,), 1.0 + frac * 2.0 ** -15, device=device)   # a fraction of a grid unit above 1
        p = torch.empty(n, dtype=torch.bfloat16, device=device)
        lo = torch.empty(n, dtype=torch.uint8, device=device)
        encode_(z.clone(), p, lo, 8, torch.randint(0, 256, (n,), dtype=torch.int32, device=device))
        got = (decode(p, lo, 8).double() - 1.0).mean().item() / 2.0 ** -15
        se = 0.5 / (n ** 0.5)
        assert abs(got - frac) < 5 * se + 1e-3, (frac, got)


def test_compensated_add_tracks_fp32_far_below_the_ulp():
    """4000 steps of 0.004-ulp pure drift: kahan8 keeps the fp32 iterate to a fraction of an
    ulp, plain bf16 SR walks by ulps, and neither stalls in expectation (SR-of-residual)."""
    torch.manual_seed(0)
    n = 1 << 14
    w0 = (torch.randn(n) * 0.05).to(torch.bfloat16)
    z_ref = w0.float().clone()
    p, lo = w0.clone(), torch.zeros(n, dtype=torch.uint8)
    p_sr = w0.clone()
    step = torch.full((n,), 1e-6) * torch.sign(torch.randn(n))
    for _ in range(4000):
        z_ref.sub_(step)
        compensated_add_(p, lo, step, -1.0, 8, None)
        bk._sr_write_(p_sr, step, -1.0, triton=False)
    # Errors in the bf16 ulp at the RMS weight (the sim's unit): a per-coordinate ulp blows up
    # where a weight crosses zero, while the residual grid it was written on did not.
    u = _ulp_ref(z_ref)
    err_ck = (decode(p, lo, 8) - z_ref) / u
    err_sr = (p_sr.float() - z_ref) / u
    moved = z_ref - w0.float()
    lost_ck = float(-((err_ck * u) * moved).sum() / (moved * moved).sum())
    assert float(err_ck.abs().max()) < 1.5           # sim: max 1.09 after 10k steps
    assert float(err_ck.std()) < 0.15                # sim: 0.114
    assert abs(lost_ck) < 0.02                       # no stall (RTN would lose ~30 %)
    assert float(err_sr.std()) > 5 * float(err_ck.std())   # sim: 1.93 vs 0.114


# ------------------------------------------------------------------------------ parity
def _bag(seed=0, device="cpu"):
    torch.manual_seed(seed)
    shapes = [(8, 16), (8, 16), (16, 4), (16,), (16,), (), (4, 3, 3, 3)]
    return [torch.nn.Parameter((torch.randn(s, device=device) * 0.05).to(torch.bfloat16)) for s in shapes]


def _drive(opts, params_lists, steps=8, seed=7):
    gg = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        grads = [(torch.randn(p.shape, generator=gg) * 0.02) for p in params_lists[0]]
        for params in params_lists:
            for p, g in zip(params, grads, strict=True):
                p.grad = g.to(device=p.device, dtype=p.dtype).clone()
        for o in opts:
            o.step()


@pytest.mark.parametrize("cfg", [
    dict(momentum_dtype="bfloat16"),
    dict(momentum_dtype="int8", weight_decay=0.01),
    dict(momentum_dtype="4bit", cautious=True),
    dict(betas=(0.0, 0.999)),
])
def test_foreach_matches_per_param_bit_exact_with_pinned_noise(cfg, monkeypatch):
    """The torch path is one deterministic codec once the residual noise is pinned: per-param
    and foreach must then agree bit-for-bit (weights AND residual bytes)."""
    monkeypatch.setattr(bk, "SR_TRITON", False)
    real = torch.randint
    monkeypatch.setattr(torch, "randint", lambda *a, **k: torch.zeros_like(real(*a, **k)))
    pa, pb = _bag(), _bag()
    oa = Adakaon(pa, lr=1e-4, foreach=True, bf16_method="kahan8", **cfg)
    ob = Adakaon(pb, lr=1e-4, foreach=False, bf16_method="kahan8", **cfg)
    _drive([oa, ob], [pa, pb])
    for a, b in zip(pa, pb, strict=True):
        assert torch.equal(a.data, b.data)
        assert torch.equal(oa.state[a][RESIDUAL_KEY], ob.state[b][RESIDUAL_KEY])
        assert oa.state[a][RESIDUAL_KEY].dtype == torch.uint8


def test_foreach_and_per_param_agree_with_live_noise():
    """With real residual noise the two paths differ only by the residual grid."""
    pa, pb = _bag(), _bag()
    oa = Adakaon(pa, lr=1e-4, foreach=True, bf16_method="kahan8")
    ob = Adakaon(pb, lr=1e-4, foreach=False, bf16_method="kahan8")
    _drive([oa, ob], [pa, pb], steps=10)
    for a, b in zip(pa, pb, strict=True):
        za, zb = decode(a.data, oa.state[a][RESIDUAL_KEY]), decode(b.data, ob.state[b][RESIDUAL_KEY])
        assert float(((za - zb).abs() / _grid_unit(zb)).max()) <= 10.0 + 1e-6   # <= 1 unit / step


@pytest.mark.skipif(not FUSED, reason="Triton fused kernels need CUDA + Triton")
@pytest.mark.parametrize("md", ["bfloat16", "float32", "int8", "4bit"])
@pytest.mark.parametrize("shapes", [
    [(32, 64), (32, 64), (16, 8), (64,), (64,), (), (4, 3, 3, 3)],   # one-block + 1-D + 0-D + conv
    [(1024, 1200), (1024, 1200)],                                     # big batched (all momentum routes)
])
def test_fused_routes_admit_kahan8_and_match_native(md, shapes):
    import kaon._fused_triton as ft
    torch.manual_seed(0)
    base = [(torch.randn(s, device="cuda") * 0.05).to(torch.bfloat16) for s in shapes]
    pa = [torch.nn.Parameter(b.clone()) for b in base]
    pb = [torch.nn.Parameter(b.clone()) for b in base]
    cfg = dict(lr=1e-4, momentum_dtype=md, bf16_method="kahan8", cautious=False)
    oa = Adakaon(pa, fused=True, **cfg)
    ob = Adakaon(pb, fused=False, foreach=False, **cfg)
    _drive([oa, ob], [pa, pb], steps=6)
    parts = oa._fused_partition(oa.param_groups[0], pa, ft)
    assert len(parts[3]) == (1 if (md == "4bit" and len(shapes) > 2) else 0)   # only 4bit's odd-C conv falls back
    for a, b, b0 in zip(pa, pb, base, strict=True):
        za, zb = decode(a.data, oa.state[a][RESIDUAL_KEY]), decode(b.data, ob.state[b][RESIDUAL_KEY])
        assert torch.isfinite(za).all()
        u = _ulp_ref(zb)
        # different residual noise (Philox vs torch) + reduction order: a few grid units.
        # 4-bit momentum: one requant code flip between the two codec paths is a whole
        # step on that coordinate (the same amplification docs/adakaon.md records).
        assert float((za - zb).abs().max()) / u < (0.5 if md == "4bit" else 0.1), md
        # and the step actually moved the compensated value by a sub-ulp amount
        assert 0.0 < float((zb - b0.float()).abs().max()) / u < 8.0


@pytest.mark.skipif(not FUSED, reason="Triton fused kernels need CUDA + Triton")
def test_triton_axpy_matches_torch_reference():
    import kaon._fused_triton as ft
    torch.manual_seed(0)
    p = (torch.randn(1 << 18, device="cuda") * 0.05).to(torch.bfloat16)
    lo = torch.randint(0, 256, p.shape, dtype=torch.uint8, device="cuda")
    d = torch.randn(p.shape, device="cuda") * 1e-6
    exact = decode(p, lo) - d
    p1, lo1 = p.clone(), lo.clone()
    ft.ck_add_(p1, lo1, d, -1.0, 8, None)
    p2, lo2 = p.clone(), lo.clone()
    compensated_add_(p2, lo2, d, -1.0, 8, None)
    z1, z2 = decode(p1, lo1), decode(p2, lo2)
    unit = _grid_unit(torch.maximum(z1.abs(), z2.abs()))
    assert float(((z1 - z2).abs() / unit).max()) <= 1.0 + 1e-6
    assert float(((z1 - exact).abs() / unit).max()) <= 1.0 + 1e-6
    assert abs(float(((z1 - exact) / unit).mean())) < 5e-3               # unbiased
    # non-finite propagation through the kernel
    p3 = torch.tensor([1.0, 2.0, 3.0], device="cuda").to(torch.bfloat16)
    lo3 = torch.zeros(3, dtype=torch.uint8, device="cuda")
    ft.ck_add_(p3, lo3, torch.tensor([float("nan"), float("inf"), -float("inf")], device="cuda"), 1.0, 8, None)
    assert torch.isnan(p3[0]) and p3[1] == float("inf") and p3[2] == -float("inf") and int(lo3.max()) == 0


@pytest.mark.skipif(not FUSED, reason="Triton fused kernels need CUDA + Triton")
def test_native_cuda_writer_takes_the_kernel_without_int32_scratch():
    pv = [(torch.randn(1 << 16, device="cuda") * 0.05).to(torch.bfloat16) for _ in range(4)]
    cv = [torch.zeros(1 << 16, dtype=torch.uint8, device="cuda") for _ in range(4)]
    delta = torch.randn(4, 1 << 16, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    bk.subtract_batched_(pv, delta, "kahan8", alpha=1e-5, comp=cv)
    torch.cuda.synchronize()
    per_elem = (torch.cuda.max_memory_allocated() - base) / (4 << 16)
    assert per_elem <= 3.5, per_elem          # the stacked weights (2) + stacked residual (1)


# ------------------------------------------------------------------------------ checkpoint
@pytest.mark.parametrize("fused", [False] + ([True] if FUSED else []))
def test_resume_is_bit_exact_and_keeps_uint8(fused):
    """The run that wrote the checkpoint, continued, and a fresh optimizer resumed from it,
    step identically afterwards: residual bytes AND the noise-stream position are restored."""
    device = "cuda" if fused else "cpu"
    pb = _bag(device=device)
    kw = dict(lr=1e-4, bf16_method="kahan8", fused=fused)
    ob = Adakaon(pb, **kw)
    _drive([ob], [pb], steps=4)
    sd = copy.deepcopy(ob.state_dict())
    assert all(s[RESIDUAL_KEY].dtype == torch.uint8 for s in sd["state"].values())
    pc = [torch.nn.Parameter(p.detach().clone()) for p in pb]
    oc = Adakaon(pc, **kw)
    oc.load_state_dict(sd)
    for p, q in zip(pb, pc, strict=True):
        st_b, st_c = ob.state[p], oc.state[q]
        assert st_c[RESIDUAL_KEY].dtype == torch.uint8
        assert torch.equal(st_b[RESIDUAL_KEY], st_c[RESIDUAL_KEY])
    _drive([ob, oc], [pb, pc], steps=4, seed=11)
    for b, c in zip(pb, pc, strict=True):
        assert torch.equal(b.data, c.data)
        assert torch.equal(ob.state[b][RESIDUAL_KEY], oc.state[c][RESIDUAL_KEY])


def test_load_state_dict_preserving_dtypes_keeps_the_residual():
    pa = _bag()
    oa = Adakaon(pa, lr=1e-4, bf16_method="kahan8", foreach=False)
    _drive([oa], [pa], steps=3)
    sd = copy.deepcopy(oa.state_dict())
    pb = _bag()
    ob = Adakaon(pb, lr=1e-4, bf16_method="kahan8", foreach=False)
    load_state_dict_preserving_dtypes(ob, sd)
    for a, b in zip(pa, pb, strict=True):
        assert torch.equal(oa.state[a][RESIDUAL_KEY], ob.state[b][RESIDUAL_KEY])
        assert ob.state[b][RESIDUAL_KEY].dtype == torch.uint8


# ------------------------------------------------------------------------------ accuracy
def test_sub_ulp_training_beats_stochastic_rounding_by_far():
    """Weights ~0.05, lr 1e-6, 400 Adakaon steps (~0.004 ulp/step): kahan8 tracks the fp32
    optimizer to a fraction of an ulp; SR is either stalled or off by whole ulps."""
    torch.manual_seed(0)
    w = (torch.randn(64, 64) * 0.05).to(torch.bfloat16)     # every run starts from the same bf16
    ref = [torch.nn.Parameter(w.float())]
    ck = [torch.nn.Parameter(w.clone())]
    sr = [torch.nn.Parameter(w.clone())]
    kw = dict(lr=1e-6, betas=(0.0, 0.999), cautious=False, foreach=False, fused=False)
    o_ref = Adakaon(ref, **kw)
    o_ck = Adakaon(ck, bf16_method="kahan8", **kw)
    o_sr = Adakaon(sr, bf16_method="stochastic_rounding", **kw)
    gg = torch.Generator().manual_seed(3)
    g0 = torch.randn(64, 64, generator=gg)
    for _ in range(400):
        g = g0 + 0.3 * torch.randn(64, 64, generator=gg)
        ref[0].grad = g.clone()
        ck[0].grad = g.to(torch.bfloat16)
        sr[0].grad = g.to(torch.bfloat16)
        o_ref.step()
        o_ck.step()
        o_sr.step()
    u = _ulp_ref(ref[0].data)                  # bf16 ulp at the RMS weight (see the test above)
    z_ck = decode(ck[0].data, o_ck.state[ck[0]][RESIDUAL_KEY])
    err_ck = ((z_ck - ref[0].data) / u).std().item()
    err_sr = ((sr[0].data.float() - ref[0].data) / u).std().item()
    moved = ((ref[0].data - w.float()).abs() / u).max().item()
    assert moved > 1.0                         # the reference moved by more than an ulp somewhere
    assert err_ck < 0.1, err_ck                # residual grid noise only (same bf16 grads, same deltas)
    assert err_sr > 5 * err_ck, (err_sr, err_ck)


# ------------------------------------------------------------------------------ memory / routing / validation
def test_state_costs_exactly_one_byte_per_param():
    for method, extra in (("kahan8", 1.0), ("kahan", 2.0), ("stochastic_rounding", 0.0)):
        pa = _bag()
        o = Adakaon(pa, lr=1e-4, bf16_method=method, foreach=False)
        _drive([o], [pa], steps=1)
        n = sum(p.numel() for p in pa)
        comp = sum(t.numel() * t.element_size() for st in o.state.values() for k, t in st.items()
                   if k in ("shift", RESIDUAL_KEY))
        assert comp / n == extra, method


def test_legacy_kahan_still_takes_the_per_param_path_and_kahan8_the_foreach_one(monkeypatch):
    calls = []
    real = bk.subtract_batched_
    monkeypatch.setattr(bk, "subtract_batched_", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    import kaon.adakaon as ak
    monkeypatch.setattr(ak, "subtract_batched_", bk.subtract_batched_)
    for method, expect_batched in (("kahan", False), ("kahan8", True)):
        calls.clear()
        pa = _bag()
        o = Adakaon(pa, lr=1e-4, bf16_method=method, foreach=True)
        _drive([o], [pa], steps=1)
        assert (len(calls) > 0) == expect_batched, method
        key = "shift" if method == "kahan" else RESIDUAL_KEY
        assert all(key in o.state[p] for p in pa)


def test_other_optimizers_and_lookahead_accept_kahan8():
    pa = _bag()
    o = Lion(pa, lr=1e-4, bf16_method="kahan8")
    _drive([o], [pa], steps=3)
    assert all(o.state[p][RESIDUAL_KEY].dtype == torch.uint8 for p in pa)
    pb = _bag()
    la = kaon.Lookahead(pb, lr=1e-4, k=2, bf16_method="kahan8")   # sync goes through subtract_one_
    _drive([la], [pb], steps=4)
    assert all(torch.isfinite(p).all() for p in pb)
    assert all(RESIDUAL_KEY in la.inner.state[p] for p in pb)


def test_validation():
    p = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="bf16_method"):
        Adakaon([p], bf16_method="kahan9")
    h = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.float16))
    with pytest.raises(NotImplementedError, match="float16"):
        Adakaon([h], bf16_method="kahan8")
    with pytest.raises(ValueError, match="comp="):
        bk.subtract_batched_([torch.zeros(4, dtype=torch.bfloat16)], torch.zeros(1, 4), "kahan8")
