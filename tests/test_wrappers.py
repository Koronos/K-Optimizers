"""Tests for the shared wrapper mixins (:mod:`kaon._wrappers`).

The TrainEvalWeights + WrapsInnerOptimizer mixins are exercised end-to-end through
:class:`kaon.lookahead.Lookahead` (see ``test_lookahead.py``). Here we pin the
:class:`~kaon._wrappers.CodecBuffer` storage primitive directly — its round-trip across
dtypes, the **fresh-fp32 read contract** (the guard against the ``.float()``-aliases-storage
footgun), and per-param == stacked equivalence.
"""
from __future__ import annotations

import pytest
import torch

from kaon._wrappers import CodecBuffer


def _alloc(dtype, src, block=128):
    st: dict = {}
    CodecBuffer.alloc(st, "b", src, dtype, block)
    return st


def test_fp32_roundtrip_exact():
    src = torch.randn(4, 6)
    st = _alloc("float32", src)
    out = CodecBuffer.read(st, "b", "float32", src)
    torch.testing.assert_close(out, src, rtol=0, atol=0)


def test_fresh_fp32_contract():
    """read() must return an INDEPENDENT fp32 tensor — mutating it must NOT touch storage.

    This is the guard against the ``.float()``-aliases-an-fp32-buffer footgun that bit 6 of
    the campaign's candidate optimizers.
    """
    src = torch.randn(4, 6)
    st = _alloc("float32", src)
    out = CodecBuffer.read(st, "b", "float32", src)
    out.add_(123.0)                                   # mutate the returned tensor
    again = CodecBuffer.read(st, "b", "float32", src)
    torch.testing.assert_close(again, src, rtol=0, atol=0)  # storage untouched


def test_quantized_roundtrip_reasonable():
    src = torch.randn(8, 16)
    for dtype, rel in (("bfloat16", 0.05), ("int8", 0.05), ("4bit", 0.25)):
        st = _alloc(dtype, src)
        out = CodecBuffer.read(st, "b", dtype, src)
        assert out.shape == src.shape
        assert torch.isfinite(out).all()
        # correlated reconstruction (loose — quantized is approximate, esp. 4bit)
        err = (out - src).abs().mean() / src.abs().mean()
        assert err < rel, f"{dtype}: rel err {err:.3f} >= {rel}"


def test_write_then_read_updates_storage():
    src = torch.randn(4, 6)
    st = _alloc("int8", src)
    new = torch.randn(4, 6)
    CodecBuffer.write(st, "b", "int8", new)
    back = CodecBuffer.read(st, "b", "int8", src)
    torch.testing.assert_close(back, new, rtol=0, atol=0.15)  # int8 requant of `new`


def test_stacked_matches_perparam():
    for dtype in ("float32", "bfloat16", "int8", "4bit"):
        srcs = [torch.randn(4, 6) for _ in range(3)]
        states = [_alloc(dtype, s) for s in srcs]
        stacked = CodecBuffer.read_stacked(states, "b", dtype, (4, 6))
        assert stacked.shape == (3, 4, 6)
        for i, (st, s) in enumerate(zip(states, srcs, strict=True)):
            torch.testing.assert_close(
                stacked[i], CodecBuffer.read(st, "b", dtype, s), rtol=1e-5, atol=1e-6
            )


def test_write_stacked_matches_write():
    dtype = "int8"
    srcs = [torch.randn(5, 4) for _ in range(3)]
    states_a = [_alloc(dtype, s) for s in srcs]
    states_b = [_alloc(dtype, s) for s in srcs]
    new = torch.randn(3, 5, 4)
    CodecBuffer.write_stacked(states_a, "b", dtype, new)
    for i, st in enumerate(states_b):
        CodecBuffer.write(st, "b", dtype, new[i])
    for sa, sb, s in zip(states_a, states_b, srcs, strict=True):
        torch.testing.assert_close(
            CodecBuffer.read(sa, "b", dtype, s), CodecBuffer.read(sb, "b", dtype, s),
            rtol=1e-5, atol=1e-6,
        )


def test_1d_buffer_roundtrip():
    """1-D buffers (biases / norm scales) go through the codec too."""
    src = torch.randn(10)
    for dtype in ("float32", "bfloat16", "int8", "4bit"):
        st = _alloc(dtype, src)
        out = CodecBuffer.read(st, "b", dtype, src)
        assert out.shape == src.shape and torch.isfinite(out).all()


# --------------------------------------------- 0.7.18 audit: zero-element buffers
_EMPTY_SHAPES = [(5, 0), (0, 5), (0,), (3, 0, 2)]


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "int8", "4bit"])
@pytest.mark.parametrize("shape", _EMPTY_SHAPES)
def test_empty_buffer_per_param_roundtrip(dtype, shape):
    """A zero-element weight (e.g. a (5, 0) projection) must alloc/read/write through
    every codec dtype: the int8 absmax reduction and the ``per // row`` reshape both
    broke on it."""
    src = torch.randn(shape)
    st = _alloc(dtype, src)
    out = CodecBuffer.read(st, "b", dtype, src)
    assert out.shape == src.shape and out.dtype == torch.float32
    CodecBuffer.write(st, "b", dtype, out)
    assert CodecBuffer.read(st, "b", dtype, src).shape == src.shape


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "int8", "4bit"])
@pytest.mark.parametrize("shape", _EMPTY_SHAPES)
def test_empty_buffer_stacked_roundtrip(dtype, shape):
    srcs = [torch.randn(shape) for _ in range(3)]
    states = [_alloc(dtype, s) for s in srcs]
    out = CodecBuffer.read_stacked(states, "b", dtype, shape)
    assert out.shape == (3, *shape) and out.dtype == torch.float32
    CodecBuffer.write_stacked(states, "b", dtype, out)
    assert CodecBuffer.read_stacked(states, "b", dtype, shape).shape == (3, *shape)


@pytest.mark.parametrize("shape", [(0, 5), (5, 0)])
def test_schedulefree_int8_foreach_with_an_empty_param(shape):
    """ScheduleFree foreach + int8 ``z`` with a zero-element param next to a normal one."""
    from kaon import ScheduleFree

    a = torch.nn.Parameter(torch.randn(4, 3))
    e = torch.nn.Parameter(torch.randn(shape))
    b = torch.nn.Parameter(torch.randn(4, 3))
    opt = ScheduleFree([a, e, b], lr=1e-3, momentum_dtype="int8", foreach=True)
    opt.train()
    a0 = a.detach().clone()
    for _ in range(3):
        for p in (a, e, b):
            p.grad = torch.randn(p.shape)
        opt.step()
    opt.eval()
    opt.train()
    assert not torch.equal(a.detach(), a0) and torch.isfinite(a).all()


@pytest.mark.parametrize("slow_dtype", ["int8", "4bit", "bfloat16"])
def test_lookahead_foreach_sync_with_an_empty_param(slow_dtype):
    from kaon import Lookahead

    ps = [torch.nn.Parameter(torch.randn(s)) for s in [(4, 3), (0, 3), (4, 3)]]
    opt = Lookahead(ps, lr=1e-2, k=2, slow_dtype=slow_dtype, foreach=True)
    for _ in range(4):
        for p in ps:
            p.grad = torch.randn(p.shape)
        opt.step()
    opt.eval()
    opt.train()
    assert torch.isfinite(ps[0]).all()
