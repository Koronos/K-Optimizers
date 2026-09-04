"""State buffers keep their storage identity across steps (requant must be IN PLACE).

MSAM's fused axpy plan (and the batched chunked steps) cache raw ``data_ptr`` tables into
``st["m"]`` / ``st["m_scale"]``. A requant that REPLACES those tensors leaves the tables
dangling: training silently reads recycled memory, and the first ``empty_cache()`` (previews,
eval) unmaps the old block and the next ``optimizer.train()`` dies with an illegal memory
access. Regression for the lone-big-tensor ``_chunked_step`` requant and ``CodecBuffer.write``.

See also ``tests/test_codec_store_identity.py`` for the Lion / AdaBelief / AdamP / KProdigy
codec-store migration (those bases used to reassign ``m`` / ``m_scale`` on every step).
"""

from __future__ import annotations

import pytest
import torch

from kaon import Nekaon
from kaon._momentum_codec import _make_codec
from kaon._wrappers import CodecBuffer


def _state_tensors(opt) -> dict[int, tuple[torch.Tensor, torch.Tensor | None]]:
    # Strong references to the live buffer OBJECTS, taken from the same _buckets() walk the
    # fused axpy plan builds its pointer tables from. Replacement always breaks `is`-identity,
    # while data_ptr can collide when the allocator recycles the freed block.
    out = {}
    for plist, states, _md, _shape, _group in opt._buckets():
        for p, st in zip(plist, states):
            if "m" in st:
                out[id(p)] = (st["m"], st.get("m_scale"))
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="chunked path requires CUDA + Triton")
def test_chunked_requant_keeps_momentum_storage():
    # Lone 1024x512 (> TILE_CAP lanes, unique shape) routes to _chunked_step, whose 4bit/int8
    # requant must write into the existing buffers, not swap in fresh tensors.
    torch.manual_seed(0)
    params = [
        torch.nn.Parameter(torch.randn(1024, 512, device="cuda")),
        torch.nn.Parameter(torch.randn(640, 512, device="cuda")),
    ]
    opt = Nekaon(params, lr=1e-3, fused=True)  # fused: lone big tensors take _chunked_step

    def step():
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
        opt.zero_grad()

    step()
    before = _state_tensors(opt)
    assert before, "momentum state expected after the first step"
    for _ in range(3):
        step()
    # eval/train around a fake preview (the crash site: train() relaunches the cached plan)
    opt.eval()
    torch.cuda.empty_cache()
    opt.train()
    step()
    after = _state_tensors(opt)
    assert set(after) == set(before)
    for pid, (m_now, sc_now) in after.items():
        m_then, sc_then = before[pid]
        assert m_now is m_then, "momentum buffer was replaced — cached pointer tables dangle"
        assert sc_now is sc_then, "scale buffer was replaced — cached pointer tables dangle"


@pytest.mark.parametrize("dtype", ["int8", "4bit"])
def test_codecbuffer_write_is_in_place(dtype):
    src = torch.randn(64, 16)
    state: dict = {}
    CodecBuffer.alloc(state, "phi", src, dtype, block=64)
    buf_ptr = state["phi"].data_ptr()
    scale_ptr = state["phi_scale"].data_ptr()
    CodecBuffer.write(state, "phi", dtype, torch.randn(64, 16))
    assert state["phi"].data_ptr() == buf_ptr
    assert state["phi_scale"].data_ptr() == scale_ptr


@pytest.mark.parametrize("dtype", ["int8", "4bit"])
def test_codec_store_one_is_in_place(dtype):
    """Direct codec contract: ``store_one`` must not reassign ``m`` / ``m_scale``."""
    codec = _make_codec(dtype)
    state: dict = {}
    g = torch.randn(32, 16)
    codec.init_state(state, g, {"momentum_4bit_block": 64})
    m_ptr, sc_ptr = state["m"].data_ptr(), state["m_scale"].data_ptr()
    codec.store_one(state, torch.randn_like(g))
    assert state["m"].data_ptr() == m_ptr
    assert state["m_scale"].data_ptr() == sc_ptr
