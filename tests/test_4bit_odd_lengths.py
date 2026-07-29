"""Mandatory odd-length contract for every Kaon 4-bit implementation.

Nibble packing is a framework-level storage format, not an optimizer detail.
Any new codec, foreach implementation or fused backend must support an odd
number of elements, allocate ``ceil(numel / 2)`` bytes, preserve the final real
low nibble, and keep the unused high nibble canonical (zero). These CPU tests
are intentionally broad so the contract runs even when CUDA/Triton is absent;
GPU routing/parity is additionally covered in ``test_fused_triton.py``.
"""

from __future__ import annotations

import pytest
import torch

from kaon._momentum_codec import (
    _dequant_4bit,
    _dequant_4bit_stacked,
    _FourBitCodec,
    _pack_nibbles,
    _quant_4bit,
    _quant_4bit_stacked,
    _unpack_nibbles,
)

ODD_LENGTHS = (1, 3, 7, 9, 63, 65, 127, 129, 255, 257, 1023, 1025)
BLOCK_SIZES = (1, 7, 64, 128)


@pytest.mark.parametrize("length", ODD_LENGTHS)
def test_pack_odd_length_uses_ceil_bytes_and_canonical_padding(length):
    nib = (torch.arange(length, dtype=torch.uint8) % 16).contiguous()
    packed = _pack_nibbles(nib)
    assert packed.numel() == (length + 1) // 2
    assert torch.equal(_unpack_nibbles(packed, length), nib)
    assert int(packed[-1] >> 4) == 0, "unused high nibble must be canonical zero"


@pytest.mark.parametrize("length", ODD_LENGTHS)
@pytest.mark.parametrize("block", BLOCK_SIZES)
def test_quant_dequant_odd_length_respects_layout_and_error_grid(length, block):
    gen = torch.Generator().manual_seed(length * 1000 + block)
    value = torch.randn(length, generator=gen)
    packed, scale, numel = _quant_4bit(value, block)
    restored = _dequant_4bit(packed, scale, numel, block)

    assert numel == length
    assert packed.shape == ((length + 1) // 2,)
    assert scale.shape == ((length + block - 1) // block,)
    assert restored.shape == value.shape
    assert int(packed[-1] >> 4) == 0

    nblocks = (length + block - 1) // block
    pad = nblocks * block - length
    padded = torch.cat((value, value.new_zeros(pad))) if pad else value
    step = padded.view(nblocks, block).abs().amax(dim=1).clamp_min(1e-12) / 7.0
    per_element_step = step.repeat_interleave(block)[:length]
    assert (restored - value).abs().le(per_element_step / 2 + 1e-6).all()


@pytest.mark.parametrize("length", ODD_LENGTHS)
@pytest.mark.parametrize("block", (64, 128))
def test_stacked_odd_length_is_bit_exact_with_per_parameter(length, block):
    gen = torch.Generator().manual_seed(length + block)
    values = torch.randn(3, length, generator=gen)
    packed_stack, scale_stack = _quant_4bit_stacked(values, block)
    restored_stack = _dequant_4bit_stacked(packed_stack, scale_stack, length, block)

    assert packed_stack.shape == (3, (length + 1) // 2)
    assert (packed_stack[:, -1] >> 4).eq(0).all()
    for index, value in enumerate(values):
        packed, scale, numel = _quant_4bit(value, block)
        assert torch.equal(packed_stack[index], packed)
        assert torch.equal(scale_stack[index], scale)
        assert torch.equal(restored_stack[index], _dequant_4bit(packed, scale, numel, block))


@pytest.mark.parametrize("shape", [(1,), (7,), (3, 5), (3, 3, 3), (257,)])
@pytest.mark.parametrize("block", (64, 128))
def test_codec_ema_preserves_odd_storage_contract(shape, block):
    codec = _FourBitCodec()
    state: dict = {}
    update = torch.randn(shape, generator=torch.Generator().manual_seed(len(shape) + block))
    codec.init_state(state, torch.zeros(shape), {"momentum_4bit_block": block})
    delta = codec.ema_one(state, update, beta1=0.5)

    numel = update.numel()
    assert state["m_numel"] == numel
    assert state["m"].numel() == (numel + 1) // 2
    assert state["m_scale"].numel() == (numel + state["m_block"] - 1) // state["m_block"]
    assert int(state["m"][-1] >> 4) == 0
    assert delta.shape == update.shape
    assert codec.dequant_one(state, update).shape == update.shape
