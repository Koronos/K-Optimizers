"""Quantized momentum stores must keep ``m`` / ``m_scale`` storage identity.

MSAM (and Nekaon) cache ``data_ptr`` tables into those buffers for the fused climb.
A requant that *reassigns* ``state["m"]`` / ``state["m_scale"]`` leaves the tables
dangling — measured as a climb that reads freed scale memory (error ≈ 76% of the
climb bound with ``MSAM(Lion, momentum_dtype="int8")``).

Regression for Lion / AdaBelief / AdamP / KProdigy migrating onto
``_MomentumCodec.store_one`` / ``store_stacked``.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from kaon import MSAM, AdaBelief, Adakaon, AdamP, AdaPNM, KProdigy, Lion
from kaon._momentum_codec import _make_codec, _quant_4bit, _quant_int8
from kaon.msam import MSAM as _MSAM

BASES = [Lion, AdaBelief, AdamP, KProdigy]
DTYPES = ["int8", "4bit"]
FOREACH = [False, True]


def _base_kwargs(cls, md, foreach):
    kw = dict(lr=1e-3, momentum_dtype=md, foreach=foreach, bf16_method="none")
    if cls is Lion:
        kw["lr"] = 1e-4
    if cls is KProdigy:
        # KProdigy uses lr as a multiplier; pin a tiny d0 so D is quiet for this smoke.
        kw = dict(
            lr=1.0,
            momentum_dtype=md,
            foreach=foreach,
            bf16_method="none",
            d0=1e-6,
            growth_rate=1.0,
        )
    return kw


def _spin(opt, params, steps, device="cpu"):
    for seed in range(steps):
        g = torch.Generator(device=device).manual_seed(seed)
        for p in params:
            p.grad = torch.randn(p.shape, generator=g, dtype=p.dtype, device=device)
        opt.step()
        opt.zero_grad(set_to_none=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA identity repro")
@pytest.mark.parametrize("cls", BASES, ids=[c.__name__ for c in BASES])
@pytest.mark.parametrize("md", DTYPES)
@pytest.mark.parametrize("foreach", FOREACH, ids=["loop", "foreach"])
def test_quantized_store_keeps_m_and_scale_ptrs_cuda(cls, md, foreach):
    """After several steps, ``m`` / ``m_scale`` data_ptr must not change."""
    torch.manual_seed(0)
    device = "cuda"
    params = [
        torch.nn.Parameter(torch.randn(8, 6, device=device) * 0.05),
        torch.nn.Parameter(torch.randn(7, device=device) * 0.05),
    ]
    opt = cls(params, **_base_kwargs(cls, md, foreach))
    _spin(opt, params, 1, device=device)
    ptrs = {
        id(p): (opt.state[p]["m"].data_ptr(), opt.state[p]["m_scale"].data_ptr())
        for p in params
    }
    _spin(opt, params, 5, device=device)
    for p in params:
        m_ptr, sc_ptr = ptrs[id(p)]
        assert opt.state[p]["m"].data_ptr() == m_ptr, f"{cls.__name__}/{md} replaced m"
        assert opt.state[p]["m_scale"].data_ptr() == sc_ptr, (
            f"{cls.__name__}/{md} replaced m_scale"
        )


@pytest.mark.parametrize("cls", BASES, ids=[c.__name__ for c in BASES])
@pytest.mark.parametrize("md", DTYPES)
def test_quantized_store_keeps_ptrs_cpu_foreach_bucket(cls, md):
    """CPU foreach with N>=3 same-shape params exercises ``store_stacked``."""
    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(5, 4) * 0.05) for _ in range(3)]
    opt = cls(params, **_base_kwargs(cls, md, foreach=True))
    _spin(opt, params, 1)
    ptrs = [
        (opt.state[p]["m"].data_ptr(), opt.state[p]["m_scale"].data_ptr()) for p in params
    ]
    _spin(opt, params, 4)
    for p, (m_ptr, sc_ptr) in zip(params, ptrs, strict=True):
        assert opt.state[p]["m"].data_ptr() == m_ptr
        assert opt.state[p]["m_scale"].data_ptr() == sc_ptr


@pytest.mark.parametrize("cls", BASES, ids=[c.__name__ for c in BASES])
@pytest.mark.parametrize("md", DTYPES)
def test_foreach_matches_per_param_store(cls, md):
    """foreach and per-param paths must keep agreeing after the codec migration."""
    torch.manual_seed(0)
    shapes = [(5, 4), (6,), (3, 3, 2, 2)]
    pa = [torch.nn.Parameter(torch.randn(s) * 0.05) for s in shapes]
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    oa = cls(pa, **_base_kwargs(cls, md, foreach=True))
    ob = cls(pb, **_base_kwargs(cls, md, foreach=False))
    for seed in range(3):
        g = torch.Generator().manual_seed(seed)
        for a, b in zip(pa, pb, strict=True):
            gr = torch.randn(a.shape, generator=g) * 0.02
            a.grad, b.grad = gr.clone(), gr.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb, strict=True):
        # Quantized foreach-vs-loop is "one fp32 ULP" in 0.7.11 Adakaon; these
        # bases still match much tighter on a tiny bag — keep a small atol.
        torch.testing.assert_close(a.detach(), b.detach(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("md", DTYPES)
def test_store_stacked_identity_and_codes(md):
    """Direct ``store_stacked``: pointer identity + codes match per-param quant."""
    codec = _make_codec(md)
    group = {"momentum_4bit_block": 16}
    states = []
    fp32s = []
    for _ in range(3):
        st: dict = {}
        g = torch.randn(6, 8)
        codec.init_state(st, g, group)
        states.append(st)
        fp32s.append(torch.randn(6, 8))
    stack = torch.stack(fp32s)
    m_ptrs = [st["m"].data_ptr() for st in states]
    sc_ptrs = [st["m_scale"].data_ptr() for st in states]
    codec.store_stacked(states, stack)
    for st, m_ptr, sc_ptr, m_fp in zip(states, m_ptrs, sc_ptrs, fp32s, strict=True):
        assert st["m"].data_ptr() == m_ptr
        assert st["m_scale"].data_ptr() == sc_ptr
        if md == "int8":
            q, sc = _quant_int8(m_fp)
            assert torch.equal(st["m"], q)
            assert torch.equal(st["m_scale"], sc.reshape_as(st["m_scale"]))
        else:
            packed, sc, _ = _quant_4bit(m_fp, st["m_block"])
            assert torch.equal(st["m"], packed)
            assert torch.equal(st["m_scale"], sc)


def test_msam_rejects_adapnm_base():
    p = torch.nn.Parameter(torch.randn(4, 4))
    with pytest.raises(TypeError, match="m_pos|dual momentum|AdaPNM"):
        MSAM([p], base_optimizer=AdaPNM, lr=1e-3, rho=0.3)


def test_msam_allows_adapnm_when_rho_zero():
    p = torch.nn.Parameter(torch.randn(4, 4))
    MSAM([p], base_optimizer=AdaPNM, lr=1e-3, rho=0.0)  # passthrough — must not raise


def test_msam_warns_when_base_has_no_momentum():
    p = torch.nn.Parameter(torch.randn(8, 8) * 0.02)
    opt = MSAM(
        [p],
        base_optimizer=Adakaon,
        rho=0.3,
        lr=1e-3,
        betas=(0.0, 0.999),
        momentum_dtype="float32",
        foreach=False,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opt.step()  # smoke step, no grads — must NOT warn
        assert not any("no first-moment" in str(w.message) for w in caught)
        for seed in range(2):
            g = torch.Generator().manual_seed(seed)
            p.grad = torch.randn(p.shape, generator=g)
            opt.step()
    msgs = [str(w.message) for w in caught]
    assert any("no first-moment" in m or "no effect" in m for m in msgs), msgs


def _msam_inert_warnings(*, norm, rho, lr, steps=80):
    g = torch.Generator().manual_seed(0)
    p = (torch.randn(32, 32, generator=g) * 0.02).requires_grad_(True)
    opt = MSAM(
        [p],
        base_optimizer=Lion,
        rho=rho,
        norm=norm,
        lr=lr,
        betas=(0.5, 0.99),
        momentum_dtype="float32",
        foreach=False,
        bf16_method="none",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for seed in range(steps):
            gg = torch.Generator().manual_seed(seed)
            p.grad = torch.randn(p.shape, generator=gg, dtype=p.dtype)
            opt.step()
    return [str(w.message) for w in caught]


def test_inert_warning_quiet_under_global_norm_tiny_lr():
    """``norm="global"`` relative is ``|rho|/||w||_2`` — independent of lr."""
    msgs = _msam_inert_warnings(norm="global", rho=0.3, lr=1e-6)
    assert msgs == [], msgs


def test_inert_warning_fires_under_none_norm_tiny_lr():
    msgs = _msam_inert_warnings(norm="none", rho=0.3, lr=1e-8)
    assert len(msgs) == 1 and "inert" in msgs[0]


def test_inert_warning_fires_under_global_tiny_rho():
    """A tiny global radius vs ||w||_2 is inert even at a normal lr."""
    msgs = _msam_inert_warnings(norm="global", rho=1e-12, lr=1e-2)
    assert len(msgs) == 1 and "inert" in msgs[0]


def test_plan_addrs_valid_reads_live_state_dicts():
    """Reassignment via ``st['m'] = ...`` must invalidate — not a swapped witness ref."""
    p = torch.nn.Parameter(torch.randn(4, 4))
    st = {
        "m": torch.zeros(4, 4, dtype=torch.int8),
        "m_scale": torch.ones(4, 1),
    }
    cache = {
        "buckets": [
            dict(
                plist=[p],
                states=[st],
                p_addrs=(p.data_ptr(),),
                m_addrs=(st["m"].data_ptr(),),
                sc_addrs=(st["m_scale"].data_ptr(),),
            )
        ]
    }
    assert _MSAM._plan_addrs_valid(cache)
    st["m"] = st["m"].clone()  # simulate a base that reassigns
    assert not _MSAM._plan_addrs_valid(cache)
    st["m_scale"] = st["m_scale"].clone()
    # m already mismatched; fixing only m_addrs should still fail on scale
    cache["buckets"][0]["m_addrs"] = (st["m"].data_ptr(),)
    assert not _MSAM._plan_addrs_valid(cache)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused plan needs CUDA")
def test_plan_rebuilds_after_momentum_reassignment():
    """After ``st['m'] = clone()``, the next climb rebuilds ``m_addr`` to the live ptrs."""
    p = (torch.randn(64, 64, device="cuda") * 0.02).requires_grad_(True)
    opt = MSAM(
        [p],
        base_optimizer=Lion,
        rho=0.3,
        norm="none",
        lr=1e-3,
        betas=(0.5, 0.99),
        momentum_dtype="int8",
        foreach=False,
        bf16_method="none",
    )
    for seed in range(2):
        g = torch.Generator(device="cuda").manual_seed(seed)
        p.grad = torch.randn(p.shape, generator=g, device="cuda")
        opt.step()
    assert opt._axpy_cache is not None
    assert opt._plan_addrs_valid(opt._axpy_cache)
    owner = opt._momentum_owner()
    st = owner.state[p]
    st["m"] = st["m"].clone()
    st["m_scale"] = st["m_scale"].clone()
    assert not opt._plan_addrs_valid(opt._axpy_cache)
    g = torch.Generator(device="cuda").manual_seed(3)
    p.grad = torch.randn(p.shape, generator=g, device="cuda")
    opt.step()
    assert opt._axpy_cache is not None
    assert opt._plan_addrs_valid(opt._axpy_cache)
    live = tuple(s["m"].data_ptr() for s in opt._axpy_cache["buckets"][0]["states"])
    assert live == opt._axpy_cache["buckets"][0]["m_addrs"]
