"""Direct tests for the momentum codecs (``kaon._momentum_codec``).

Until now the codecs were only exercised *indirectly*, through each optimizer's
``foreach == per-param`` parity tests. This module tests them head-on: round-trip
fidelity, the per-param vs stacked bit-exactness contract, scale layout, byte
footprint, the symmetric-level usage (one code intentionally unused), and the
zero-momentum / absmax-floor edge cases.

Rationale note (measured, 2026-06-07): the bf16 codec runs its EMA in the *stored*
bf16 dtype rather than dequantising to fp32 first (the way int8/4bit do). A proxy
A/B (Adakaon-bf16, C96/N800, 3 seeds) found EMA-in-bf16 vs EMA-in-fp32 identical to
within seed noise (te 0.0780 vs 0.0780, gap +0.0074 vs +0.0073) — the momentum EMA
is a slow average and bf16's 8-bit mantissa is ample, so the leaner in-place bf16
lerp is kept. These tests therefore pin behaviour/fidelity *bounds*, not exact bf16
numerics, so that choice stays free to revisit.
"""

from __future__ import annotations

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
