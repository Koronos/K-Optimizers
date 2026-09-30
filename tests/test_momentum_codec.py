"""Direct tests for the momentum codecs (``kaon._momentum_codec``).

Until now the codecs were only exercised *indirectly*, through each optimizer's
``foreach == per-param`` parity tests. This module tests them head-on: round-trip
fidelity, the per-param vs stacked bit-exactness contract, scale layout, byte
footprint, the symmetric-level usage (one code intentionally unused), and the
zero-momentum / absmax-floor edge cases.

The bf16 codec runs its EMA in fp32 then ``copy_``s into the bf16 buffer (same
as the fused Triton kernels). A prior leaner in-bf16 lerp rounded the update
before the EMA and was the only dtype where native/fused diverged; that path is
gone. These tests pin behaviour including bf16/fp32-EMA agreement.
"""

from __future__ import annotations

import copy
import math
import warnings

import pytest
import torch

from kaon._momentum_codec import (
    _FOURBIT_BLOCK,
    _FOURBIT_ZERO,
    _INT8_CLAMP,
    _dequant_4bit,
    _dequant_4bit_stacked,
    _FloatCodec,
    _FourBitCodec,
    _Int8Codec,
    _make_codec,
    _pack_nibbles,
    _quant_4bit,
    _quant_4bit_stacked,
    _quant_int8,
    _quant_int8_stacked,
    _unpack_nibbles,
    warn_if_4bit_high_beta1,
)

DTYPES = ["bfloat16", "float32", "int8", "4bit"]


# --------------------------------------------------------------- factory & layout
def test_make_codec_dispatch():
    assert isinstance(_make_codec("bfloat16"), _FloatCodec)
    assert isinstance(_make_codec("float32"), _FloatCodec)
    assert isinstance(_make_codec("int8"), _Int8Codec)
    assert isinstance(_make_codec("4bit"), _FourBitCodec)
    assert _make_codec("bfloat16").dtype == torch.bfloat16
    assert _make_codec("float32").dtype == torch.float32


# --------------------------------------------------------------- int8 round-trip
def test_int8_roundtrip_within_grid():
    """Dequant(quant(m)) is within half a quant step (per-row scale)."""
    m = torch.randn(32, 48)
    q, scale = _quant_int8(m)
    assert q.dtype == torch.int8
    assert scale.shape == (32, 1)                       # per-row (dim-0) scale
    deq = q.float() * scale
    # per-row step = absmax/127; error <= step/2 + rounding slack
    step = m.abs().amax(dim=1, keepdim=True) / 127.0
    assert (deq - m).abs().le(step / 2 + 1e-6).all()


def test_int8_codes_stay_in_symmetric_range():
    """Symmetric int8 never emits -128 (the reserved bottom code)."""
    m = torch.randn(16, 16) * 100.0
    q, _ = _quant_int8(m)
    assert int(q.min()) >= -_INT8_CLAMP
    assert int(q.max()) <= _INT8_CLAMP


def test_int8_stacked_matches_per_param():
    """The batched int8 quantizer is bit-identical to per-param on each 2-D slice."""
    ms = [torch.randn(8, 12) for _ in range(5)]
    stack = torch.stack(ms)
    qS, scaleS = _quant_int8_stacked(stack)
    for i, m in enumerate(ms):
        q, scale = _quant_int8(m)
        assert torch.equal(qS[i], q)
        assert torch.equal(scaleS[i], scale)


def test_int8_zero_momentum_dequants_to_zero():
    """An all-zero row survives the absmax floor and dequants to exactly 0."""
    m = torch.zeros(4, 9)
    q, scale = _quant_int8(m)
    assert torch.equal(q.float() * scale, torch.zeros_like(m))


# --------------------------------------------------------------- 4bit round-trip
def test_nibble_pack_roundtrip_even_and_odd():
    for k in (8, 9):                                    # even and odd lengths
        nib = torch.randint(0, 16, (k,), dtype=torch.uint8)
        packed = _pack_nibbles(nib)
        assert packed.numel() == (k + 1) // 2
        assert torch.equal(_unpack_nibbles(packed, k), nib)


def test_4bit_roundtrip_within_grid():
    m = torch.randn(257)                                # not a block multiple -> exercises padding
    packed, scale, numel = _quant_4bit(m, _FOURBIT_BLOCK)
    assert numel == 257
    assert packed.dtype == torch.uint8
    deq = _dequant_4bit(packed, scale, numel, _FOURBIT_BLOCK)
    assert deq.shape == m.shape
    # per-block step = absmax/7; bound the reconstruction error by step/2 (+slack)
    nb = (257 + _FOURBIT_BLOCK - 1) // _FOURBIT_BLOCK
    pad = nb * _FOURBIT_BLOCK - 257
    blocks = torch.cat([m, m.new_zeros(pad)]).view(nb, _FOURBIT_BLOCK)
    step = (blocks.abs().amax(dim=1) / 7.0).clamp_min(1e-12)
    err = (deq - m).abs().view(-1)
    per_elem_step = step.repeat_interleave(_FOURBIT_BLOCK)[:257]
    assert err.le(per_elem_step / 2 + 1e-6).all()


def test_4bit_nibbles_never_use_reserved_zero_code():
    """Symmetric 4-bit emits signed codes in [-7, 7] -> nibbles in [1, 15]; never 0."""
    m = torch.randn(512) * 50.0
    packed, _, numel = _quant_4bit(m, _FOURBIT_BLOCK)
    nib = _unpack_nibbles(packed, numel)
    assert int(nib.min()) >= 1            # nibble 0 (signed -8) is reserved/unused
    assert int(nib.max()) <= 15
    # and the symmetric mapping is centred on _FOURBIT_ZERO
    assert _FOURBIT_ZERO == 8


def test_4bit_stacked_matches_per_param():
    ms = [torch.randn(130) for _ in range(4)]           # >1 block each
    packedS, scaleS = _quant_4bit_stacked(torch.stack(ms), _FOURBIT_BLOCK)
    deqS = _dequant_4bit_stacked(packedS, scaleS, 130, _FOURBIT_BLOCK)
    for i, m in enumerate(ms):
        packed, scale, numel = _quant_4bit(m, _FOURBIT_BLOCK)
        assert torch.equal(packedS[i], packed)
        assert torch.equal(scaleS[i], scale)
        assert torch.equal(deqS[i], _dequant_4bit(packed, scale, numel, _FOURBIT_BLOCK))


# --------------------------------------------------------------- EMA entry points
def _ema_state(codec, shape):
    state: dict = {}
    grad = torch.zeros(shape)
    codec.init_state(state, grad, {"momentum_4bit_block": _FOURBIT_BLOCK})
    return state


def test_fresh_state_dequants_to_zero_all_codecs():
    """A freshly initialised momentum reads back as exactly zero for every codec."""
    for md in DTYPES:
        codec = _make_codec(md)
        state = _ema_state(codec, (6, 10))
        deq = codec.dequant_one(state, torch.zeros(6, 10))
        assert torch.equal(deq, torch.zeros(6, 10)), md


def test_ema_one_matches_ema_stacked_all_codecs():
    """Per-param ``ema_one`` and batched ``ema_stacked`` agree bit-for-bit (the contract
    every optimizer's foreach path relies on)."""
    torch.manual_seed(1)
    shape = (8, 16)
    updates = [torch.randn(*shape) for _ in range(3)]
    for md in DTYPES:
        cA, cB = _make_codec(md), _make_codec(md)
        sA = [_ema_state(cA, shape) for _ in range(3)]
        sB = [_ema_state(cB, shape) for _ in range(3)]
        beta1 = 0.9
        # per-param
        dA = [cA.ema_one(sA[i], updates[i].clone(), beta1) for i in range(3)]
        # stacked
        upd = torch.stack([u.clone() for u in updates])
        dB = cB.ema_stacked(sB, upd, lambda t: t, shape, beta1)
        for i in range(3):
            assert torch.allclose(dA[i], dB[i], atol=0, rtol=0), f"{md} delta slice {i}"
            assert torch.equal(cA.dequant_one(sA[i], torch.zeros(shape)),
                               cB.dequant_one(sB[i], torch.zeros(shape))), f"{md} stored {i}"


def test_byte_footprint_per_param():
    """int8 == 1 B/param, 4bit == 0.5 B/param (packed), bf16 == 2, fp32 == 4 (state 'm')."""
    shape = (64, 64)
    n = 64 * 64
    expect = {"float32": 4 * n, "bfloat16": 2 * n, "int8": n, "4bit": (n + 1) // 2}
    for md, want in expect.items():
        state = _ema_state(_make_codec(md), shape)
        got = state["m"].numel() * state["m"].element_size()
        assert got == want, f"{md}: {got} != {want}"


def test_scale_folds_value_exactly_for_quantized():
    """``scale_`` multiplies the dequantised momentum exactly for wrapper handoffs."""
    torch.manual_seed(2)
    shape = (5, 7)
    for md in ("int8", "4bit", "bfloat16", "float32"):
        codec = _make_codec(md)
        state = _ema_state(codec, shape)
        codec.ema_one(state, torch.randn(*shape), 0.9)     # populate momentum
        before = codec.dequant_one(state, torch.zeros(shape)).clone()
        codec.scale_(state, 0.25)
        after = codec.dequant_one(state, torch.zeros(shape))
        # quantized codecs scale the per-row/block scale -> exact; float scales in-dtype
        atol = 0 if md in ("int8", "4bit", "float32") else 1e-2
        assert torch.allclose(after, before * 0.25, atol=atol, rtol=1e-5), md


def test_bf16_ema_matches_fp32_ema_then_round():
    """bf16 codec EMA == fp32 lerp then round-to-bf16 (fused-kernel contract)."""
    torch.manual_seed(3)
    shape = (8, 12)
    updates = [torch.randn(*shape) for _ in range(5)]
    beta1 = 0.9

    bf = _make_codec("bfloat16")
    st = _ema_state(bf, shape)
    # Reference: keep an fp32 shadow, lerp in fp32, write bf16.
    shadow = torch.zeros(shape, dtype=torch.float32)
    for u in updates:
        d = bf.ema_one(st, u.clone(), beta1)
        shadow.lerp_(u, 1.0 - beta1)
        assert torch.equal(d, shadow)
        assert torch.equal(st["m"], shadow.bfloat16())
        # Next step must read the rounded store (fused does too).
        shadow = st["m"].float()


def test_bf16_ema_stacked_matches_ema_one():
    """Stacked bf16 EMA stays bit-exact vs per-param after the fp32-EMA change."""
    torch.manual_seed(4)
    shape = (6, 10)
    updates = [torch.randn(*shape) for _ in range(3)]
    cA, cB = _make_codec("bfloat16"), _make_codec("bfloat16")
    sA = [_ema_state(cA, shape) for _ in range(3)]
    sB = [_ema_state(cB, shape) for _ in range(3)]
    dA = [cA.ema_one(sA[i], updates[i].clone(), 0.9) for i in range(3)]
    dB = cB.ema_stacked(sB, torch.stack(updates), lambda t: t, shape, 0.9)
    for i in range(3):
        assert torch.equal(dA[i], dB[i])
        assert torch.equal(sA[i]["m"], sB[i]["m"])


def _legacy_int8_ema_one(state: dict, update: torch.Tensor, beta1: float) -> torch.Tensor:
    """Pre-optimisation ``_Int8Codec.ema_one`` (two fp32 temps + clone + ``.to(int8)``)."""
    m = state["m"].float() * state["m_scale"]
    m.lerp_(update, 1.0 - beta1)
    delta = m.clone()
    q, scale = _quant_int8(m)
    state["m"].copy_(q)
    state["m_scale"].copy_(scale.reshape_as(state["m_scale"]))
    return delta


def _legacy_int8_ema_stacked(states, update, mat, eff, beta1):
    """Pre-optimisation ``_Int8Codec.ema_stacked`` (kept the ``delta = m.clone()``)."""
    rowshape = (eff[0], 1) if len(eff) == 2 else (1,)
    scale = torch.stack([s["m_scale"].view(*rowshape) for s in states])
    m = torch.stack([mat(s["m"]) for s in states]).float().mul_(scale)
    m.lerp_(update, 1.0 - beta1)
    delta = m.clone()
    q, new_scale = _quant_int8_stacked(m)
    torch._foreach_copy_([mat(s["m"]) for s in states], list(q.unbind(0)))
    for s, sc in zip(states, new_scale.unbind(0), strict=True):
        s["m_scale"].copy_(sc.view_as(s["m_scale"]))
    return delta


def test_int8_ema_one_bit_identical_to_legacy():
    """``.float().mul_`` + no-clone + direct code write == legacy bit-for-bit."""
    torch.manual_seed(5)
    shape = (16, 24)
    codec = _Int8Codec()
    st_new = _ema_state(codec, shape)
    st_old = copy.deepcopy(st_new)
    # Seed both with the same codes via one shared path.
    u0 = torch.randn(*shape)
    _legacy_int8_ema_one(st_new, u0.clone(), 0.9)
    st_old = copy.deepcopy(st_new)

    for k in range(5):
        u = torch.randn(*shape)
        if k == 4:
            # An all-zero row exercises the absmax floor of the inlined quantizer.
            u[3] = 0.0
            st_new["m"][3] = 0
            st_old["m"][3] = 0
        d_new = codec.ema_one(st_new, u.clone(), 0.9)
        d_old = _legacy_int8_ema_one(st_old, u.clone(), 0.9)
        assert torch.equal(d_new, d_old)
        assert torch.equal(st_new["m"], st_old["m"])
        assert torch.equal(st_new["m_scale"], st_old["m_scale"])
        assert torch.isfinite(st_new["m_scale"]).all() and (st_new["m_scale"] > 0).all()


def test_int8_ema_stacked_bit_identical_to_legacy():
    torch.manual_seed(6)
    shape = (8, 12)
    codec = _Int8Codec()
    states_new = [_ema_state(codec, shape) for _ in range(3)]
    # Warm up identically.
    warm = torch.stack([torch.randn(*shape) for _ in range(3)])
    codec.ema_stacked(states_new, warm, lambda t: t, shape, 0.9)
    states_old = copy.deepcopy(states_new)

    upd = torch.stack([torch.randn(*shape) for _ in range(3)])
    d_new = codec.ema_stacked(states_new, upd.clone(), lambda t: t, shape, 0.9)
    d_old = _legacy_int8_ema_stacked(states_old, upd.clone(), lambda t: t, shape, 0.9)
    assert torch.equal(d_new, d_old)
    for a, b in zip(states_new, states_old, strict=True):
        assert torch.equal(a["m"], b["m"])
        assert torch.equal(a["m_scale"], b["m_scale"])


def _legacy_4bit_ema_one(state, update, beta1):
    bs = state["m_block"]
    m = _dequant_4bit(state["m"], state["m_scale"], state["m_numel"], bs)
    m = m.view_as(update)
    m.lerp_(update, 1.0 - beta1)
    delta = m.clone()
    packed, scale, _ = _quant_4bit(m, bs)
    state["m"].copy_(packed)
    state["m_scale"].copy_(scale)
    return delta


def test_4bit_ema_one_bit_identical_without_clone():
    torch.manual_seed(7)
    shape = (9, 11)  # numel not a block multiple
    codec = _FourBitCodec()
    st_new = _ema_state(codec, shape)
    u0 = torch.randn(*shape)
    codec.ema_one(st_new, u0.clone(), 0.9)
    st_old = copy.deepcopy(st_new)
    for _ in range(3):
        u = torch.randn(*shape)
        d_new = codec.ema_one(st_new, u.clone(), 0.85)
        d_old = _legacy_4bit_ema_one(st_old, u.clone(), 0.85)
        assert torch.equal(d_new, d_old)
        assert torch.equal(st_new["m"], st_old["m"])
        assert torch.equal(st_new["m_scale"], st_old["m_scale"])


@pytest.mark.parametrize("shape", [(5, 7), (37,), (9, 11)])
def test_4bit_ema_one_matches_ema_stacked_when_per_not_block_multiple(shape):
    """``per % block != 0`` (the common case) keeps per-param and stacked bit-for-bit.

    A ``.contiguous()`` on the strided dequant slice picked a different ``lerp_`` kernel
    and broke this contract (caught in review); pin it with a block that does not
    divide ``numel``.
    """
    torch.manual_seed(8)
    group = {"momentum_4bit_block": 8}
    cA, cB = _FourBitCodec(), _FourBitCodec()
    sA, sB = [], []
    for _ in range(3):
        stA, stB = {}, {}
        cA.init_state(stA, torch.zeros(shape), group)
        cB.init_state(stB, torch.zeros(shape), group)
        sA.append(stA)
        sB.append(stB)
    assert sA[0]["m_numel"] % sA[0]["m_block"] != 0
    for _ in range(4):
        updates = [torch.randn(*shape) for _ in range(3)]
        dA = [cA.ema_one(sA[i], updates[i].clone(), 0.9) for i in range(3)]
        dB = cB.ema_stacked(sB, torch.stack([u.clone() for u in updates]), lambda t: t, shape, 0.9)
        for i in range(3):
            assert torch.equal(dA[i], dB[i]), f"delta slice {i}"
            assert torch.equal(sA[i]["m"], sB[i]["m"]), f"codes {i}"
            assert torch.equal(sA[i]["m_scale"], sB[i]["m_scale"]), f"scale {i}"


def test_warn_if_4bit_high_beta1():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_if_4bit_high_beta1(0.99, "4bit")
        warn_if_4bit_high_beta1(0.9, "4bit")
        warn_if_4bit_high_beta1(0.99, "int8")
    msgs = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
    assert len(msgs) == 1
    assert "1/sqrt" in msgs[0] or "amplif" in msgs[0].lower()


# ------------------------------------------------------------- empty tensors (int8)
# ``(0,)`` / ``(5, 0)`` / ``(3, 0, 2)`` used to raise out of ``amax`` ("reduction over a
# zero-size dimension") on every int8 entry point; 4-bit already survived (its blocks view
# reduces over a size-1 block). A zero-element param is legal (a pruned / zero-width layer)
# and every other codec steps it as a no-op.
EMPTY_SHAPES = [(0,), (5, 0), (3, 0, 2), (0, 4)]


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
def test_int8_quant_of_an_empty_tensor(shape):
    m = torch.zeros(shape)
    q, scale = _quant_int8(m)
    assert q.shape == m.shape and q.dtype == torch.int8
    dims = tuple(range(1, m.ndim)) if m.ndim >= 2 else ()
    expect = torch.zeros(shape).amax(dim=dims, keepdim=True).shape if m.numel() else None
    if expect is not None:
        assert scale.shape == expect
    # stacked, in the effective row layout the foreach buckets use
    row = shape[0] if len(shape) >= 2 else 1
    stack = torch.zeros((3, row, math.prod(shape[1:]) if len(shape) >= 2 else 0))
    qs, ss = _quant_int8_stacked(stack)
    assert qs.shape == stack.shape and ss.shape == (*stack.shape[:-1], 1)


@pytest.mark.parametrize("md", DTYPES)
@pytest.mark.parametrize("shape", EMPTY_SHAPES)
def test_every_codec_steps_an_empty_tensor(md, shape):
    codec = _make_codec(md)
    like = torch.zeros(shape)
    eff = ((shape[0], math.prod(shape[1:])) if len(shape) >= 2 else (math.prod(shape),))

    def mat(t):
        return t.reshape(eff)

    s1, s2 = _ema_state(codec, shape), _ema_state(codec, shape)
    d = codec.ema_one(s1, like.clone(), 0.9)
    assert d.numel() == 0
    codec.store_one(s1, like.clone())
    assert codec.dequant_one(s1, like).shape == like.shape
    upd = torch.zeros((2, *eff))
    states = [s2, _ema_state(codec, shape)]
    codec.ema_stacked(states, upd.clone(), mat, eff, 0.9)
    codec.store_stacked(states, upd.clone())
    assert codec.dequant_stacked(states, mat, eff).shape == upd.shape


# ---------------------------------------- opt-in stochastic requant of the momentum
# ``STOCHASTIC_MOMENTUM_REQUANT`` (default OFF, bit-identical to 0.7.17). Under
# ``m = beta*m_q + (1-beta)*g`` a round-to-nearest requant flattens every coordinate below
# half its block's quant step for good; the stochastic one keeps them in expectation.
def _outlier_block_run(md, stochastic, monkeypatch, steps=200, beta=0.9):
    """Returns ``(sum_t m_q(t), sum_t m_fp32(t), m_q(T))`` on a 128-block with one outlier.

    The SUM over steps is what the weights integrate (every step applies ``m``), so it is
    the quantity a biased requant loses and an unbiased one keeps, even though a single
    stochastic snapshot is noisier than the round-to-nearest one.
    """
    import kaon._momentum_codec as mc

    monkeypatch.setattr(mc, "STOCHASTIC_MOMENTUM_REQUANT", stochastic)
    torch.manual_seed(0)
    codec = _make_codec(md)
    shape = (1, 128)
    state = _ema_state(codec, shape)
    ref = torch.zeros(shape)
    acc_q, acc_ref = torch.zeros(shape), torch.zeros(shape)
    g = torch.Generator().manual_seed(1)
    bias = torch.randn(shape, generator=g) * 0.05          # a persistent per-coordinate signal
    for _ in range(steps):
        grad = bias + torch.randn(shape, generator=g) * 0.1
        grad[0, 0] = 5.0
        ref.lerp_(grad, 1 - beta)
        codec.ema_one(state, grad, beta)
        acc_ref += ref
        acc_q += codec.dequant_one(state, torch.zeros(shape))
    return acc_q, acc_ref, codec.dequant_one(state, torch.zeros(shape))


@pytest.mark.parametrize("md", ["int8", "4bit"])
def test_stochastic_momentum_requant_is_off_by_default(md):
    import kaon._momentum_codec as mc

    assert mc.STOCHASTIC_MOMENTUM_REQUANT is False
    assert mc._momentum_gen(torch.device("cpu")) is None


@pytest.mark.parametrize("md", ["int8", "4bit"])
def test_stochastic_momentum_requant_keeps_the_small_coordinates_alive(md, monkeypatch):
    acc_rne, acc_ref, m_rne = _outlier_block_run(md, False, monkeypatch)
    acc_sr, _, _ = _outlier_block_run(md, True, monkeypatch)
    rest = slice(1, None)
    cos = torch.nn.functional.cosine_similarity
    c_rne = cos(acc_rne[0, rest], acc_ref[0, rest], dim=0).item()
    c_sr = cos(acc_sr[0, rest], acc_ref[0, rest], dim=0).item()
    msg = f"{md}: stochastic requant integrates cos {c_sr:.3f} (round-to-nearest {c_rne:.3f})"
    if md == "4bit":       # the reported defect: the whole block flattened to 0 for good
        assert (m_rne[0, rest] == 0).float().mean().item() > 0.95
        assert c_rne < 0.1, f"4-bit round-to-nearest kept {c_rne:.3f} of the signal"
        assert c_sr > 0.5, msg                      # measured 0.565 vs 0.000
    else:                  # int8's step is 18x finer: RNE already keeps it (0.978 vs 0.977)
        assert c_sr > 0.9, msg


@pytest.mark.parametrize("md", ["int8", "4bit"])
def test_stochastic_momentum_requant_stacked_and_per_param_stay_in_range(md, monkeypatch):
    """With the toggle ON every entry point still produces valid codes and scales."""
    import kaon._momentum_codec as mc

    monkeypatch.setattr(mc, "STOCHASTIC_MOMENTUM_REQUANT", True)
    shape = (8, 16)
    codec = _make_codec(md)
    states = [_ema_state(codec, shape) for _ in range(3)]
    upd = torch.randn(3, *shape)
    d = codec.ema_stacked(states, upd, lambda t: t, shape, 0.9)
    assert torch.isfinite(d).all()
    codec.store_stacked(states, upd)
    codec.store_one(states[0], upd[0])
    codec.ema_one(states[1], upd[1], 0.9)
    for s in states:
        back = codec.dequant_one(s, torch.zeros(shape))
        assert torch.isfinite(back).all()
        assert (back - upd[0]).abs().max() < 10


def test_stochastic_momentum_noise_is_not_the_weight_sr_stream(monkeypatch):
    """The toggle's generator must not replay SR stream 0's words (same global seed)."""
    import kaon._momentum_codec as mc
    from kaon import reseed_stochastic_rounding
    from kaon._stochastic_rounding import SRStream

    monkeypatch.setattr(mc, "STOCHASTIC_MOMENTUM_REQUANT", True)
    torch.manual_seed(123)
    reseed_stochastic_rounding()
    dev = torch.device("cpu")
    a = torch.rand(100_000, generator=mc._momentum_gen(dev))
    b = torch.rand(100_000, generator=SRStream().generator(dev))     # stream 0 after reseed
    assert not torch.equal(a, b)
    corr = torch.corrcoef(torch.stack([a, b]))[0, 1].abs().item()
    assert corr < 0.02, f"momentum requant noise correlates with the weight SR noise ({corr})"
    # ... and it is reproducible under the protocol.
    torch.manual_seed(123)
    reseed_stochastic_rounding()
    assert torch.equal(a, torch.rand(100_000, generator=mc._momentum_gen(dev)))
