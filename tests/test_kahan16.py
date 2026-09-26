"""``bf16_method="kahan16"``: the bf16 weight + a 16-bit residual, i.e. an fp32 master split in two.

Covers, per ``docs/research/compact-kahan.md`` §8:

* the codec — ``(bf16, int16-stored uint16)`` IS an fp32, bit for bit, over the whole finite
  domain (exhaustive on CUDA, 2**20 patterns on CPU): ``lo`` is the fp32's low half, the
  stored bf16 its nearest bf16 (half-away), ``+-0``, subnormals, binade crossings and the
  carry into inf included; non-finite values propagate; the Triton helpers store the same
  bits as the torch reference;
* the trajectory: kahan16 bf16 weights reproduce an fp32-weight run of the same optimizer
  BIT FOR BIT (same bf16 gradients, per-param and foreach, CPU and CUDA, fused included) for
  every optimizer whose update does not read the weight; the forward sees the nearest bf16;
* resume bit-exact with the ``int16`` residual preserved; memory exactly +2 B/param;
* the MSAM / Nekaon climb on the decoded value, Lookahead with mixed fp32/bf16 groups;
* mid-run switches: SR -> kahan16 (zero residual), kahan8 <-> kahan16 (converted, exact when
  widening); refused residuals of the wrong width; fp16 and ScheduleFree refused.
"""
from __future__ import annotations

import copy
import inspect
import warnings

import pytest
import torch

import kaon
from kaon import Adakaon
from kaon import _backend as bk
from kaon._compact_kahan import (
    RESIDUAL_KEY,
    compensated_add_,
    convert_residual,
    decode,
    encode_,
    residual_bits_of,
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
    """bf16 ulp at the RMS magnitude of ``z``."""
    rms = z.float().pow(2).mean().sqrt()
    return float(torch.exp2(torch.floor(torch.log2(rms)) - 7))


def _z(p: torch.Tensor, st: dict) -> torch.Tensor:
    """The full value a kahan16 param holds (decode with the stored width)."""
    lo = st[RESIDUAL_KEY]
    return decode(p.data, lo, residual_bits_of(lo))


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


# ------------------------------------------------------------------------------ codec
def _patterns(device: str, start: int, n: int) -> torch.Tensor:
    """fp32 bit patterns ``start .. start+n-1`` (mod 2**32) as int32."""
    i = torch.arange(start, start + n, dtype=torch.int64, device=device)
    return ((i & 0xFFFFFFFF) - ((i & 0x80000000) << 1)).to(torch.int32)


def _cpu_patterns() -> torch.Tensor:
    """Every high half x 16 low halves (the boundaries 0, 1, 0x7FFF, 0x8000 (tie), 0x8001,
    0xFFFF plus fixed random ones): 2**20 patterns, every exponent, sign and bf16 mantissa."""
    lows = torch.tensor([0, 1, 2, 0x7FFE, 0x7FFF, 0x8000, 0x8001, 0xFFFE, 0xFFFF,
                         0x1234, 0x4321, 0x9ABC, 0xCBA9, 0x0F0F, 0xF0F0, 0x5555], dtype=torch.int64)
    hi = torch.arange(65536, dtype=torch.int64)
    i = ((hi[:, None] << 16) | lows[None, :]).reshape(-1)
    return ((i & 0xFFFFFFFF) - ((i & 0x80000000) << 1)).to(torch.int32)


def _check_split(bits: torch.Tensor) -> None:
    """The kahan16 codec over the patterns ``bits`` (int32): exact split on the finite
    domain, nearest (half-away) stored bf16, non-finite propagated, every (w, lo) state with a
    finite weight decoding finite."""
    x = bits.view(torch.float32)
    u = bits.to(torch.int64) & 0xFFFFFFFF
    finite = ((u >> 23) & 0xFF) != 0xFF
    p = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    lo = torch.empty(x.shape, dtype=torch.int16, device=x.device)
    encode_(x.clone(), p, lo, 16, None)
    zd = decode(p, lo, 16)
    # (1) the pair IS the fp32: decode returns the input bits exactly (+-0, subnormals, the
    #     carry across a binade and into the inf pattern all included)
    assert torch.equal(zd.view(torch.int32)[finite], bits[finite])
    # (2) the residual is the fp32's low half, bit for bit (stored in int16)
    lo16 = lo.to(torch.int64) & 0xFFFF
    assert torch.equal(lo16[finite], (u & 0xFFFF)[finite])
    # (3) the stored bf16 is the nearest one, ties away from zero
    w16 = p.view(torch.int16).to(torch.int64) & 0xFFFF
    assert torch.equal(w16[finite], ((u + 0x8000) >> 16)[finite] & 0xFFFF)
    # ... hence the plain RNE cast everywhere except at an exact tie with an even truncation
    rne = x.to(torch.bfloat16).view(torch.int16).to(torch.int64) & 0xFFFF
    differs = finite & (w16 != rne)
    assert bool(((u & 0xFFFF) == 0x8000)[differs].all())
    # (4) non-finite values are stored as the cast stores them, with a zero residual
    nf = ~finite
    assert bool(torch.isnan(p.float()[nf]).eq(torch.isnan(x[nf])).all())
    assert torch.equal(p.float()[nf & ~torch.isnan(x)], x[nf & ~torch.isnan(x)])
    assert int(lo16[nf].abs().max()) == 0 if bool(nf.any()) else True
    # (5) every state (w = high half, lo = low half) with a finite weight decodes finite
    ws = (bits >> 16).to(torch.int16).view(torch.bfloat16)
    ls = bits.to(torch.int16)
    wfin = (((bits.to(torch.int64) >> 23) & 0xFF) != 0xFF)
    assert bool(torch.isfinite(decode(ws, ls, 16))[wfin].all())


def test_codec_is_an_exact_fp32_split_cpu():
    _check_split(_cpu_patterns())


@pytest.mark.skipif(not CUDA, reason="exhaustive sweep needs CUDA")
def test_codec_is_an_exact_fp32_split_exhaustive_cuda():
    """All 2**32 fp32 patterns, in chunks."""
    chunk = 1 << 26
    for start in range(0, 1 << 32, chunk):
        _check_split(_patterns("cuda", start, chunk))


@pytest.mark.parametrize("device", DEVICES)
def test_codec_named_cases(device):
    """The corners by name: binade crossing, +-0, subnormals, the tie, overflow into inf."""
    cases = torch.tensor([
        2.0 - 2.0 ** -23,                 # largest fp32 below 2: w = 2.0 (carry), lo = 0xFFFF
        1.0 + 2.0 ** -8,                  # an exact tie at the bf16 grid: half-away -> up
        1.0 + 3 * 2.0 ** -8,              # tie with an odd truncation: RNE agrees (up)
        0.0, -0.0,
        2.0 ** -149,                      # smallest fp32 subnormal: w = 0, lo = 1
        -(2.0 ** -149),
        2.0 ** -126 * (1 - 2.0 ** -23),   # largest fp32 subnormal
        3.4028234663852886e38,            # FLT_MAX: the stored bf16 carries into inf
    ], dtype=torch.float32, device=device)
    p = torch.empty_like(cases, dtype=torch.bfloat16)
    lo = torch.empty(cases.shape, dtype=torch.int16, device=device)
    encode_(cases.clone(), p, lo, 16, None)
    assert _bits_equal(decode(p, lo, 16), cases)
    lo16 = (lo.to(torch.int32) & 0xFFFF).tolist()
    assert p[0].item() == 2.0 and lo16[0] == 0xFFFF
    assert p[1].item() == 1.0 + 2.0 ** -7                              # half-away up
    assert cases[1:2].to(torch.bfloat16).item() == 1.0                # RNE keeps the even one
    assert p[2].item() == 1.0 + 4 * 2.0 ** -8
    assert p[3].view(torch.int16).item() == 0 and p[4].view(torch.int16).item() == -0x8000
    assert p[5].item() == 0.0 and lo16[5] == 1
    assert p[6].view(torch.int16).item() == -0x8000 and lo16[6] == 1
    assert p[8].item() == float("inf")                                  # the forward sees inf,
    assert decode(p[8:], lo[8:], 16).item() == 3.4028234663852886e38   # the master is exact


@pytest.mark.skipif(not FUSED, reason="Triton fused kernels need CUDA + Triton")
def test_triton_helpers_store_and_decode_the_same_bits_exhaustive():
    """``ck_store_noise(.., 0, 16)`` / ``ck_decode(.., 16)`` against the torch codec over all
    2**32 patterns (as values to store, and as (w, lo) states to decode)."""
    import triton
    import triton.language as tl

    from kaon._fused_triton import ck_decode, ck_store_noise

    @triton.jit
    def _store(z_ptr, p_ptr, c_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        ck_store_noise(p_ptr, c_ptr, offs, mask, tl.load(z_ptr + offs, mask=mask, other=0.0), 0, 16)

    @triton.jit
    def _decode(p_ptr, c_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, ck_decode(p_ptr, c_ptr, offs, mask, 16), mask=mask)

    chunk = 1 << 26
    grid = (chunk // 1024,)
    for start in range(0, 1 << 32, chunk):
        bits = _patterns("cuda", start, chunk)
        x = bits.view(torch.float32)
        p1 = torch.empty(chunk, dtype=torch.bfloat16, device="cuda")
        lo1 = torch.empty(chunk, dtype=torch.int16, device="cuda")
        _store[grid](x, p1, lo1, chunk, BLOCK=1024)
        p2, lo2 = torch.empty_like(p1), torch.empty_like(lo1)
        encode_(x.clone(), p2, lo2, 16, None)
        nan = torch.isnan(x)        # NaN payloads: both sides store *a* NaN, lo = 0
        assert torch.equal(p1.view(torch.int16)[~nan], p2.view(torch.int16)[~nan])
        assert bool(torch.isnan(p1[nan]).all()) and torch.equal(lo1, lo2)
        ws = (bits >> 16).to(torch.int16).view(torch.bfloat16)
        ls = bits.to(torch.int16)
        out = torch.empty(chunk, device="cuda")
        _decode[grid](ws, ls, out, chunk, BLOCK=1024)
        assert torch.equal(out.view(torch.int32), decode(ws, ls, 16).view(torch.int32))


@pytest.mark.skipif(not FUSED, reason="Triton fused kernels need CUDA + Triton")
def test_triton_axpy_is_the_fp32_add():
    """The one-launch native writer at 16 bits: ``(p, lo) += a*d`` is the fp32 ``z + a*d``
    (the same arithmetic torch does on an fp32 tensor), then split."""
    import kaon._fused_triton as ft
    torch.manual_seed(0)
    z = torch.randn(1 << 18, device="cuda") * 0.05
    d = torch.randn(z.shape, device="cuda") * 1e-6
    p = torch.empty(z.shape, dtype=torch.bfloat16, device="cuda")
    lo = torch.empty(z.shape, dtype=torch.int16, device="cuda")
    encode_(z.clone(), p, lo, 16, None)
    assert ft.ck_add_supported(p, lo, d, 16) and not ft.ck_add_supported(p, lo, d, 8)
    p1, lo1 = p.clone(), lo.clone()
    ft.ck_add_(p1, lo1, d, -1e-1, 16, None)
    p2, lo2 = p.clone(), lo.clone()
    compensated_add_(p2, lo2, d, -1e-1, 16, None)
    ref = z.clone().add_(d, alpha=-1e-1)
    assert _bits_equal(decode(p2, lo2, 16), ref)                         # torch reference: exact
    z1 = decode(p1, lo1, 16)
    ulp32 = torch.exp2(torch.floor(torch.log2(ref.abs())) - 23)
    assert float(((z1 - ref).abs() / ulp32).max()) <= 1.0                # <= 1 fp32 ulp (FMA)


@pytest.mark.skipif(not FUSED, reason="Triton fused kernels need CUDA + Triton")
def test_native_cuda_writer_takes_the_kernel_without_int32_scratch():
    pv = [(torch.randn(1 << 16, device="cuda") * 0.05).to(torch.bfloat16) for _ in range(4)]
    cv = [torch.zeros(1 << 16, dtype=torch.int16, device="cuda") for _ in range(4)]
    delta = torch.randn(4, 1 << 16, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    bk.subtract_batched_(pv, delta, "kahan16", alpha=1e-5, comp=cv)
    torch.cuda.synchronize()
    per_elem = (torch.cuda.max_memory_allocated() - base) / (4 << 16)
    assert per_elem <= 4.5, per_elem          # the stacked weights (2) + stacked residual (2)


# ------------------------------------------------------------------------------ trajectory
def _bag(dtype=torch.bfloat16, seed=0, device="cpu"):
    torch.manual_seed(seed)
    shapes = [(8, 16), (8, 16), (16, 4), (16,), (16,), (), (4, 3, 3, 3)]
    # bf16-representable starts, so the fp32 twin begins at exactly the same values
    return [torch.nn.Parameter((torch.randn(s, device=device) * 0.05).to(torch.bfloat16).to(dtype))
            for s in shapes]


def _drive(opts, params_lists, steps=8, seed=7):
    """The SAME bf16 gradients to every run (an fp32 twin gets them upcast)."""
    gg = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        grads = [(torch.randn(p.shape, generator=gg) * 0.02).to(torch.bfloat16) for p in params_lists[0]]
        for params in params_lists:
            for p, g in zip(params, grads, strict=True):
                p.grad = g.to(device=p.device, dtype=p.dtype).clone()
        for o in opts:
            o.step()


def _assert_fp32_master(ps16, opt16, ps32):
    """kahan16 == fp32 twin bit for bit, and the forward weight is the nearest bf16."""
    for b, a in zip(ps16, ps32, strict=True):
        st = opt16.state[b] if b in opt16.state else opt16.inner.state[b]
        assert st[RESIDUAL_KEY].dtype == torch.int16
        z = _z(b, st)
        assert _bits_equal(z, a.data), float((z - a.data).abs().max())
        u = a.data.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        want = (((u + 0x8000) >> 16) & 0xFFFF)
        assert torch.equal(b.data.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF, want)


def _no_gc(cls):
    """Gradient Centralization runs on ``p.grad`` in the GRAD's dtype (bf16 for a bf16 param,
    fp32 for the fp32 twin): an input-preprocessing difference, not a storage one, so the
    bit-exact comparisons switch it off."""
    sig = inspect.signature(cls.__init__).parameters
    return {"gradient_centralization": False} if "gradient_centralization" in sig else {}


@pytest.mark.parametrize("cfg", [
    dict(momentum_dtype="bfloat16"),
    dict(momentum_dtype="int8", cautious=True),
    dict(momentum_dtype="4bit"),
    dict(betas=(0.0, 0.999)),
])
@pytest.mark.parametrize("foreach", [True, False])
def test_adakaon_trajectory_is_bit_exact_to_an_fp32_master(cfg, foreach):
    p32, p16 = _bag(torch.float32), _bag()
    kw = dict(lr=1e-4, foreach=foreach, fused=False, gradient_centralization=False, **cfg)
    o32 = Adakaon(p32, **kw)
    o16 = Adakaon(p16, bf16_method="kahan16", **kw)
    _drive([o32, o16], [p32, p16])
    _assert_fp32_master(p16, o16, p32)


@pytest.mark.parametrize("name", ["Lion", "AdaBelief", "ADOPT", "KProdigy", "AdaMuon", "AdaPNM"])
@pytest.mark.parametrize("foreach", [True, False])
def test_other_optimizers_are_bit_exact_to_an_fp32_master(name, foreach):
    """Every optimizer sharing the backend writer. (AdamP is not in the list: its projection
    reads the stored WEIGHT, i.e. the bf16 ``p``, where the fp32 twin reads its master — see
    ``test_adamp_reads_the_bf16_weight``.)"""
    cls = getattr(kaon, name)
    extra = _no_gc(cls)
    if "fused" in inspect.signature(cls.__init__).parameters:
        extra["fused"] = False
    p32, p16 = _bag(torch.float32), _bag()
    o32 = cls(p32, lr=1e-4, foreach=foreach, **extra)
    o16 = cls(p16, lr=1e-4, foreach=foreach, bf16_method="kahan16", **extra)
    _drive([o32, o16], [p32, p16])
    _assert_fp32_master(p16, o16, p32)


def test_adamp_reads_the_bf16_weight():
    """AdamP's projection uses the stored weight: kahan16 is then fp32-exact in its WRITE
    but the update sees the bf16 ``p``, so the twin differs by far less than an ulp."""
    p32, p16 = _bag(torch.float32), _bag()
    o32 = kaon.AdamP(p32, lr=1e-4, **_no_gc(kaon.AdamP))
    o16 = kaon.AdamP(p16, lr=1e-4, bf16_method="kahan16", **_no_gc(kaon.AdamP))
    _drive([o32, o16], [p32, p16])
    for b, a in zip(p16, p32, strict=True):
        assert float((_z(b, o16.state[b]) - a.data).abs().max()) < 0.05 * _ulp_ref(a.data)


@pytest.mark.skipif(not FUSED, reason="Triton fused kernels need CUDA + Triton")
@pytest.mark.parametrize("md", ["bfloat16", "float32", "int8", "4bit"])
@pytest.mark.parametrize("shapes", [
    [(32, 64), (32, 64), (16, 8), (64,), (64,), (), (4, 3, 3, 3)],   # one-block + 1-D + 0-D + conv
    [(1024, 1200), (1024, 1200)],                                     # big batched
])
def test_fused_routes_are_the_fp32_fused_run_and_match_native(md, shapes):
    """Fused kahan16 == fused fp32 BIT FOR BIT (same kernels, the pair decoded to the
    fp32 master in-register); fused vs native kahan16 differ only as the two fp32 paths do
    (reduction order, FMA contraction): bounded far below an ulp."""
    import kaon._fused_triton as ft
    torch.manual_seed(0)
    base = [(torch.randn(s, device="cuda") * 0.05).to(torch.bfloat16) for s in shapes]
    pf32 = [torch.nn.Parameter(b.float()) for b in base]
    pf = [torch.nn.Parameter(b.clone()) for b in base]
    pn = [torch.nn.Parameter(b.clone()) for b in base]
    cfg = dict(lr=1e-4, momentum_dtype=md, cautious=False, gradient_centralization=False)
    # the big bucket's reductions are fp32 atomics (run-to-run nondeterministic) by default:
    # pin them, or two fp32 fused runs would not be bit-identical to each other either
    of32 = Adakaon(pf32, fused=True, deterministic_reductions=True, **cfg)
    of = Adakaon(pf, fused=True, bf16_method="kahan16", deterministic_reductions=True, **cfg)
    on = Adakaon(pn, fused=False, foreach=False, bf16_method="kahan16", **cfg)
    _drive([of32, of, on], [pf32, pf, pn], steps=6)
    parts = of._fused_partition(of.param_groups[0], pf, ft)
    assert len(parts[3]) == (1 if (md == "4bit" and len(shapes) > 2) else 0)   # 4bit odd-C conv
    _assert_fp32_master(pf, of, pf32)
    for a, b in zip(pf, pn, strict=True):
        za, zb = _z(a, of.state[a]), _z(b, on.state[b])
        assert float((za - zb).abs().max()) / _ulp_ref(zb) < (0.5 if md == "4bit" else 0.01), md


# ------------------------------------------------------------------------------ checkpoint
@pytest.mark.parametrize("route", ["per_param", "foreach"] + (["fused"] if FUSED else []))
def test_resume_is_bit_exact_and_keeps_int16(route):
    device = "cuda" if route == "fused" else "cpu"
    pb = _bag(device=device)
    kw = dict(lr=1e-4, bf16_method="kahan16", fused=route == "fused", foreach=route != "per_param")
    ob = Adakaon(pb, **kw)
    _drive([ob], [pb], steps=4)
    sd = copy.deepcopy(ob.state_dict())
    assert all(s[RESIDUAL_KEY].dtype == torch.int16 for s in sd["state"].values())
    pc = [torch.nn.Parameter(p.detach().clone()) for p in pb]
    oc = Adakaon(pc, **kw)
    oc.load_state_dict(sd)
    for p, q in zip(pb, pc, strict=True):
        assert oc.state[q][RESIDUAL_KEY].dtype == torch.int16
        assert torch.equal(ob.state[p][RESIDUAL_KEY], oc.state[q][RESIDUAL_KEY])
    _drive([ob, oc], [pb, pc], steps=4, seed=11)
    for b, c in zip(pb, pc, strict=True):
        assert torch.equal(b.data, c.data)
        assert torch.equal(ob.state[b][RESIDUAL_KEY], oc.state[c][RESIDUAL_KEY])


def test_load_state_dict_preserving_dtypes_keeps_the_int16_residual():
    pa = _bag()
    oa = Adakaon(pa, lr=1e-4, bf16_method="kahan16", foreach=False)
    _drive([oa], [pa], steps=3)
    sd = copy.deepcopy(oa.state_dict())
    pb = _bag()
    ob = Adakaon(pb, lr=1e-4, bf16_method="kahan16", foreach=False)
    load_state_dict_preserving_dtypes(ob, sd)
    for a, b in zip(pa, pb, strict=True):
        assert ob.state[b][RESIDUAL_KEY].dtype == torch.int16
        assert torch.equal(oa.state[a][RESIDUAL_KEY], ob.state[b][RESIDUAL_KEY])


# ------------------------------------------------------------------------------ memory / validation
def test_state_costs_exactly_two_bytes_per_param():
    for method, extra in (("kahan16", 2.0), ("kahan8", 1.0), ("stochastic_rounding", 0.0)):
        for foreach in (True, False):
            pa = _bag()
            o = Adakaon(pa, lr=1e-4, bf16_method=method, foreach=foreach)
            _drive([o], [pa], steps=1)
            n = sum(p.numel() for p in pa)
            comp = sum(t.numel() * t.element_size() for st in o.state.values()
                       for k, t in st.items() if k in ("shift", RESIDUAL_KEY))
            assert comp / n == extra, (method, foreach)


def test_validation():
    h = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.float16))
    with pytest.raises(NotImplementedError, match="float16"):
        Adakaon([h], bf16_method="kahan16")
    with pytest.raises(NotImplementedError, match="float16"):
        bk.init_bf16_state(h, {}, "kahan16")         # every optimizer's _init_state
    p = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="bf16_method"):
        kaon.ScheduleFree([p], bf16_method="kahan16")
    with pytest.raises(ValueError, match="comp="):
        bk.subtract_batched_([torch.zeros(4, dtype=torch.bfloat16)], torch.zeros(1, 4), "kahan16")
    # a residual of the other width is refused by every writer, never decoded on the wrong grid
    w = torch.zeros(4, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="other width"):
        bk._ck_write_(w, torch.zeros(4, dtype=torch.uint8), torch.zeros(4), 1.0, 16)
    with pytest.raises(ValueError, match="other width"):
        bk.subtract_batched_([w], torch.zeros(1, 4), "kahan16", comp=[torch.zeros(4, dtype=torch.uint8)])


# ------------------------------------------------------------------------------ wrappers
def _climb_runs(kind, fused, steps=60, seed=0):
    from kaon import MSAM, Nekaon
    device = "cuda" if fused else "cpu"
    torch.manual_seed(seed)
    kaon.reseed_stochastic_rounding()
    w = (torch.randn(64, 64, device=device) * 0.05).to(torch.bfloat16)

    def build(params, method):
        kw = dict(lr=1e-5, betas=(0.9, 0.999), cautious=False, foreach=True, fused=fused,
                  gradient_centralization=False)
        if method is not None:
            kw["bf16_method"] = method
        if kind == "msam":
            return MSAM(params, rho=0.05, **kw)
        return Nekaon(params, k=1.5, momentum_dtype="bfloat16", weight_decay=0.0, **kw)

    runs = {}
    for method in (None, "kahan16"):
        p = torch.nn.Parameter(w.float() if method is None else w.clone())
        runs[method] = (p, build([p], method))
        runs[method][1].train()
    gg = torch.Generator(device=device).manual_seed(seed + 1)
    g0 = torch.randn(64, 64, generator=gg, device=device)
    for _ in range(steps):
        g = (g0 + 0.3 * torch.randn(64, 64, generator=gg, device=device)).to(torch.bfloat16)
        for p, opt in runs.values():
            p.grad = g.to(p.dtype)
            opt.step()
    return runs


@pytest.mark.parametrize("kind", ["msam", "nekaon"])
@pytest.mark.parametrize("fused", [False] + ([True] if FUSED else []))
def test_msam_nekaon_climb_is_the_fp32_climb(kind, fused):
    """The climb goes through the decoded value (``_ck_write_`` / ``CK=16`` in the fused
    axpy), which at 16 bits is exactly the fp32 add the fp32 twin does: the kahan16 run is
    the fp32 run, bit for bit, both in the perturbed train state and after ``eval()``."""
    runs = _climb_runs(kind, fused)
    (p32, o32), (p16, o16) = runs[None], runs["kahan16"]
    assert _bits_equal(_z(p16, o16.inner.state[p16]), p32.data)          # perturbed (train)
    o32.eval()
    o16.eval()
    assert _bits_equal(_z(p16, o16.inner.state[p16]), p32.data)          # clean (eval)
    moved = float((p32.data - p32.data.to(torch.bfloat16).float()).abs().max())
    assert moved > 0.0


def _mixed_bag(dtype_bf=torch.bfloat16, device="cpu"):
    torch.manual_seed(1)
    out = []
    for i, s in enumerate([(8, 16), (8, 16), (16,), (16,), (), (4, 3, 3, 3)]):
        t = (torch.randn(s, device=device) * 0.05).to(torch.bfloat16).float()
        out.append(torch.nn.Parameter(t.to(dtype_bf) if i % 2 == 0 else t))
    return out


@pytest.mark.parametrize("foreach", [True, False])
def test_lookahead_kahan16_mixed_precision(foreach):
    """A group mixing fp32 and bf16 params: only the bf16 ones carry an int16 residual (the
    INNER's — it belongs to the weight), the sync writes through it on both sync routes, and
    — since 0.7.16 the sync reads the DECODED ``theta`` (docs §8.5) — the run IS the
    all-fp32 Lookahead, bit for bit."""
    p16, p32 = _mixed_bag(), _mixed_bag(torch.float32)
    kw = dict(lr=1e-4, k=2, foreach=foreach, gradient_centralization=False)
    la16 = kaon.Lookahead(p16, bf16_method="kahan16", **kw)
    la32 = kaon.Lookahead(p32, **kw)
    _drive([la16, la32], [p16, p32], steps=6)
    for a, b in zip(p16, p32, strict=True):
        st = la16.inner.state[a]
        assert (RESIDUAL_KEY in st) == (a.dtype == torch.bfloat16)
        z = _z(a, st) if a.dtype == torch.bfloat16 else a.data
        assert torch.isfinite(z).all()
        if a.dtype == torch.bfloat16:
            assert st[RESIDUAL_KEY].dtype == torch.int16
        assert _bits_equal(z, b.data), float((z - b.data).abs().max())


# ------------------------------------------------------------------------------ mid-run switches
def test_convert_residual_widening_is_exact_and_narrowing_one_rounding():
    torch.manual_seed(0)
    z = torch.randn(1 << 16) * 0.05
    p = torch.empty(z.shape, dtype=torch.bfloat16)
    lo8 = torch.empty(z.shape, dtype=torch.uint8)
    encode_(z.clone(), p, lo8, 8, None)
    z8 = decode(p, lo8, 8)
    w8 = p.clone()
    lo16 = convert_residual(p, lo8, 16)
    assert lo16.dtype == torch.int16 and torch.equal(p.view(torch.int16), w8.view(torch.int16))
    assert _bits_equal(decode(p, lo16, 16), z8)                          # widening: exact
    encode_(z.clone(), p, lo16, 16, None)                                # a full 16-bit value
    lo_back = convert_residual(p, lo16, 8)
    assert lo_back.dtype == torch.uint8
    unit = torch.exp2(torch.floor(torch.log2(z.abs())) - 7 - 8)
    assert bool(((decode(p, lo_back, 8) - z).abs() <= 0.5 * unit + 1e-45).all())


@pytest.mark.parametrize("route", ["per_param", "foreach"] + (["fused"] if FUSED else []))
@pytest.mark.parametrize("switch", [("stochastic_rounding", "kahan16"), ("kahan8", "kahan16"),
                                    ("kahan16", "kahan8")])
def test_switching_method_mid_run(route, switch):
    """SR -> kahan16 allocates a zero int16 residual (one warning); kahan8 <-> kahan16
    CONVERTS the residual (one warning) — the value is kept (exactly when widening), the
    new dtype is used by every route from that step on, and the run continues finite."""
    import kaon._backend as backend
    backend._LAZY_RESIDUAL_WARNED = False
    backend._CONVERTED_RESIDUAL_WARNED = False
    src, dst = switch
    device = "cuda" if route == "fused" else "cpu"
    if route == "fused":
        torch.manual_seed(0)
        shapes = [(32, 64), (32, 64), (64,), (64,), (), (1024, 1200)]
        pa = [torch.nn.Parameter((torch.randn(s, device=device) * 0.05).to(torch.bfloat16))
              for s in shapes]
    else:
        pa = _bag()
    o = Adakaon(pa, lr=1e-4, bf16_method=src, foreach=route == "foreach", fused=route == "fused")
    _drive([o], [pa], steps=2)
    before = [(_z(p, o.state[p]) if RESIDUAL_KEY in o.state[p] else p.data.float()) for p in pa]
    o.param_groups[0]["bf16_method"] = dst
    match = "kahan_lo" if src == "stochastic_rounding" else "re-encoded"
    with pytest.warns(UserWarning, match=match):
        _drive([o], [pa], steps=1, seed=5)
    _drive([o], [pa], steps=2, seed=6)                       # and it keeps going quietly
    want = torch.int16 if dst == "kahan16" else torch.uint8
    for p, b in zip(pa, before, strict=True):
        st = o.state[p]
        assert st[RESIDUAL_KEY].dtype == want and st[RESIDUAL_KEY].shape == p.shape
        z = _z(p, st)
        assert torch.isfinite(z).all()
        # 3 steps of lr 1e-4 on a clipped update: well within a few ulps of where it was
        assert float((z - b).abs().max()) < 4e-4 + 4 * float(b.abs().max()) * 2.0 ** -8


def test_conversion_keeps_the_value_through_the_optimizer():
    """kahan8 -> kahan16 through the per-param writer: the step after the switch starts from
    EXACTLY the kahan8 value (widening is exact), i.e. equals a manual convert + fp32 add."""
    import kaon._backend as backend
    backend._CONVERTED_RESIDUAL_WARNED = False
    pa = _bag()
    o = Adakaon(pa, lr=1e-4, bf16_method="kahan8", foreach=False, gradient_centralization=False)
    _drive([o], [pa], steps=2)
    z8 = [_z(p, o.state[p]) for p in pa]
    o.param_groups[0]["bf16_method"] = "kahan16"
    captured = []
    real = bk._ck_write_

    def spy(target, lo, source, alpha, bits, triton=None, sr=None):
        captured.append((source.clone(), alpha))
        return real(target, lo, source, alpha, bits, triton, sr)

    bk._ck_write_ = spy
    try:
        with pytest.warns(UserWarning, match="re-encoded"):
            _drive([o], [pa], steps=1, seed=5)
    finally:
        bk._ck_write_ = real
    for p, z0, (d, a) in zip(pa, z8, captured, strict=True):
        assert _bits_equal(_z(p, o.state[p]), z0.reshape(d.shape).add(d, alpha=a).reshape(p.shape))


def test_adapnm_and_lion_switch_to_kahan16_lazily():
    import kaon._backend as backend
    for cls in (kaon.AdaPNM, kaon.Lion):
        backend._CONVERTED_RESIDUAL_WARNED = False
        pa = _bag()
        o = cls(pa, lr=1e-4, bf16_method="kahan8", foreach=True)
        _drive([o], [pa], steps=2)
        o.param_groups[0]["bf16_method"] = "kahan16"
        with pytest.warns(UserWarning, match="re-encoded"):
            _drive([o], [pa], steps=1, seed=5)
        _drive([o], [pa], steps=1, seed=6)
        for p in pa:
            assert o.state[p][RESIDUAL_KEY].dtype == torch.int16
            assert torch.isfinite(_z(p, o.state[p])).all()


@pytest.mark.parametrize("fused", [False] + ([True] if FUSED else []))
def test_msam_switch_kahan8_to_kahan16_mid_run_decodes_the_old_width_first(fused):
    """The removal at the top of the step after the switch must decode the uint8 residual the
    climb encoded (the inner step converts it afterwards). Decoding it as int16 would inject
    garbage; the run instead stays within a small fraction of an ulp of the fp32 twin."""
    import kaon._backend as backend
    from kaon import MSAM
    backend._CONVERTED_RESIDUAL_WARNED = False
    device = "cuda" if fused else "cpu"
    torch.manual_seed(0)
    kaon.reseed_stochastic_rounding()
    w = (torch.randn(32, 64, device=device) * 0.05).to(torch.bfloat16)
    kw = dict(rho=0.05, lr=1e-5, betas=(0.9, 0.999), cautious=False, foreach=True, fused=fused,
              gradient_centralization=False)
    p16, p32 = torch.nn.Parameter(w.clone()), torch.nn.Parameter(w.float())
    o16 = MSAM([p16], bf16_method="kahan8", **kw)
    o32 = MSAM([p32], **kw)
    o16.train()
    o32.train()
    gg = torch.Generator(device=device).manual_seed(3)
    for i in range(12):
        if i == 4:
            o16.param_groups[0]["bf16_method"] = "kahan16"
        g = (torch.randn(32, 64, generator=gg, device=device) * 0.02).to(torch.bfloat16)
        p16.grad, p32.grad = g.clone(), g.float()
        if i == 4:
            with pytest.warns(UserWarning, match="re-encoded"):
                o16.step()
        else:
            o16.step()
        o32.step()
    o16.eval()
    o32.eval()
    st = o16.inner.state[p16]
    assert st[RESIDUAL_KEY].dtype == torch.int16
    err = float((_z(p16, st) - p32.data).abs().max()) / _ulp_ref(p32.data)
    # kahan8's residual grid (ulp/256 per write, ~12 writes incl. the climbs) for 4 steps, then
    # exact: measured 0.051. A removal that decoded the uint8 residual as int16 lands >= 1 ulp off.
    assert err < 0.25, err


# ------------------------------------------------------------------------------ review of a6c7fb0
def _tail_bag(dtype=torch.bfloat16):
    """Numels 15 and 7: every tensor ends in a SIMD tail, and a stacked [3, 15] / [3, 7]
    bucket puts the tails in DIFFERENT places than three per-tensor ops do."""
    torch.manual_seed(0)
    shapes = [(5, 3)] * 3 + [(7,)] * 3
    return [torch.nn.Parameter((torch.randn(s) * 0.05).to(torch.bfloat16).to(dtype)) for s in shapes]


@pytest.mark.parametrize("betas", [(0.0, 0.999), (0.9, 0.999)])
def test_misaligned_tail_bucket_is_still_the_fp32_writer(betas):
    """The CPU ``add_(alpha=)`` kernel fuses to an FMA in its vector body but rounds the
    product first in its scalar tail. The first cut did ONE add over the stacked bucket:
    1 fp32 ulp off the fp32 foreach writer on 5 coordinates of this bag after 30 steps. The
    stacked kahan16 add now runs row by row (``compensated_add_(rows=True)``): kahan16
    foreach == fp32 foreach and kahan16 per-param == fp32 per-param, bit for bit. What is
    left between foreach and per-param is the fp32 optimizer's OWN difference (its stacked
    update math has tails too), reproduced exactly."""
    runs = {}
    for foreach in (True, False):
        p32, p16 = _tail_bag(torch.float32), _tail_bag()
        kw = dict(lr=1e-4, betas=betas, foreach=foreach, fused=False, gradient_centralization=False)
        o32 = Adakaon(p32, **kw)
        o16 = Adakaon(p16, bf16_method="kahan16", **kw)
        _drive([o32, o16], [p32, p16], steps=30)
        _assert_fp32_master(p16, o16, p32)
        runs[foreach] = ([a.data.clone() for a in p32], [_z(b, o16.state[b]) for b in p16])
    for f32, pp32, f16, pp16 in zip(runs[True][0], runs[False][0], runs[True][1], runs[False][1],
                                    strict=True):
        assert torch.equal(f16 != pp16, f32 != pp32)                     # same coordinates
        assert _bits_equal(f16 - pp16, f32 - pp32)                       # same differences


def test_stacked_add_rows_is_the_per_tensor_add_and_bounds_the_plain_one():
    """``rows=True`` == the per-tensor adds bit for bit; the plain stacked add (the first
    cut) is within 1 fp32 ulp of it — the bound the SIMD-tail difference can reach."""
    torch.manual_seed(0)
    z = torch.randn(3, 15) * 0.05
    d = torch.randn(3, 15) * 1e-3
    p = torch.empty(3, 15, dtype=torch.bfloat16)
    lo = torch.empty(3, 15, dtype=torch.int16)
    encode_(z.clone(), p, lo, 16, None)
    want = torch.stack([zi.clone().add_(di, alpha=-0.37) for zi, di in zip(z, d, strict=True)])
    pr, lr_ = p.clone(), lo.clone()
    compensated_add_(pr, lr_, d, -0.37, 16, None, rows=True)
    assert _bits_equal(decode(pr, lr_, 16), want)
    pp, lp = p.clone(), lo.clone()
    compensated_add_(pp, lp, d, -0.37, 16, None, rows=False)
    ulp32 = torch.exp2(torch.floor(torch.log2(want.abs())) - 23)
    assert float(((decode(pp, lp, 16) - want).abs() / ulp32).max()) <= 1.0


def test_rows_flag_leaves_kahan8_bit_identical(monkeypatch):
    """kahan8's numerics are not touched: with the same residual noise, rows=True and the
    plain stacked add store the same bits (its ulp/256 grid absorbs the tail difference
    in expectation; here we only require the flag to be a no-op for it)."""
    torch.manual_seed(0)
    z = torch.randn(3, 15) * 0.05
    d = torch.randn(3, 15) * 1e-3
    p = torch.empty(3, 15, dtype=torch.bfloat16)
    lo = torch.empty(3, 15, dtype=torch.uint8)
    encode_(z.clone(), p, lo, 8, None)
    real_randint = torch.randint
    out = []
    for rows in (True, False):
        pa, la = p.clone(), lo.clone()
        g = torch.Generator().manual_seed(5)                    # the same residual noise
        monkeypatch.setattr(
            torch, "randint",
            lambda *a, _r=real_randint, _g=g, **k: _r(*a, **{**k, "generator": _g}))
        compensated_add_(pa, la, d, -0.37, 8, None, rows=rows)
        monkeypatch.setattr(torch, "randint", real_randint)
        out.append((pa.view(torch.int16).clone(), la.clone()))
    assert torch.equal(out[0][0], out[1][0]) and torch.equal(out[0][1], out[1][1])


def test_subtract_batched_refuses_a_mixed_width_bucket():
    """torch.stack promotes a uint8 + int16 mix to int16, which the width check on the stack
    would accept: every view is checked."""
    w = [torch.zeros(4, dtype=torch.bfloat16), torch.zeros(4, dtype=torch.bfloat16)]
    comp = [torch.zeros(4, dtype=torch.int16), torch.zeros(4, dtype=torch.uint8)]
    with pytest.raises(ValueError, match="other width"):
        bk.subtract_batched_(w, torch.zeros(2, 4), "kahan16", alpha=1.0, comp=comp)
    with pytest.raises(ValueError, match="other width"):
        bk.subtract_batched_(w, torch.zeros(2, 4), "kahan8", alpha=1.0, comp=comp)


def test_msam_climb_converts_a_bucket_left_mixed_by_a_param_without_grad():
    """kahan8 -> kahan16 with one param of the bucket no longer getting gradients: the inner
    step converts only the stepped param, so the bucket is mixed (int16 + uint8). The climb
    normalizes it (``_ck_ready``) instead of falling back to the bare-bf16 climb forever."""
    import kaon._backend as backend
    from kaon import MSAM
    from kaon.msam import _ck_climb
    backend._CONVERTED_RESIDUAL_WARNED = False
    torch.manual_seed(0)
    ps = [torch.nn.Parameter((torch.randn(16, 32) * 0.05).to(torch.bfloat16)) for _ in range(2)]
    o = MSAM(ps, rho=0.05, lr=1e-4, betas=(0.9, 0.999), cautious=False, bf16_method="kahan8",
             foreach=True, fused=False)
    o.train()
    _drive([o], [ps], steps=2)
    o.param_groups[0]["bf16_method"] = "kahan16"
    with pytest.warns(UserWarning, match="re-encoded"):
        for _ in range(3):
            ps[0].grad = (torch.randn(16, 32) * 0.02).to(torch.bfloat16)
            ps[1].grad = None                                        # never stepped again
            o.step()
    states = [o.inner.state[p] for p in ps]
    assert all(st[RESIDUAL_KEY].dtype == torch.int16 for st in states)
    assert _ck_climb(o.param_groups[0], ps, states) == 16
    o.eval()
    assert all(torch.isfinite(_z(p, st)).all() for p, st in zip(ps, states, strict=True))


@pytest.mark.parametrize("method,warns", [("stochastic_rounding", True), ("kahan8", False),
                                          ("kahan16", False)])
def test_inert_climb_warning_knows_the_residual_grid(method, warns):
    """|e| ~ 1.9e-3 of the weights: below half a bf16 ulp (3.9e-3 relative), so an SR climb
    may round to nothing and warns; a compact-Kahan climb keeps it in the residual (ulp/256,
    ulp/65536) and must not warn."""
    from kaon import Nekaon
    torch.manual_seed(0)
    p = torch.nn.Parameter((torch.randn(32, 32) * 0.1).to(torch.bfloat16))
    o = Nekaon([p], lr=1e-4, k=1.5, momentum_dtype="bfloat16", weight_decay=0.0,
               bf16_method=method, foreach=True, fused=False)
    o.inert_check_interval = 1
    o.train()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(60):
            p.grad = (torch.randn(32, 32) * 0.02).to(torch.bfloat16)
            o.step()
    fired = [w for w in caught if "lookahead displacement" in str(w.message)]
    assert bool(fired) == warns, [str(w.message) for w in caught]
