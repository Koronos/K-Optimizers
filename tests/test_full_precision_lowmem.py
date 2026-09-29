"""``decode_weights`` / ``full_precision_state_dict`` decode with bounded scratch.

The export used the torch reference :func:`kaon._compact_kahan.decode`, which holds ~4.25x
the tensor in int32/bool temporaries at peak. On CUDA it now goes through the one-launch
Triton decode the foreach path already uses (no scratch); on CPU it decodes in chunks into
one preallocated fp32 output. Both must stay BIT-identical to the reference.
"""
from __future__ import annotations

import pytest
import torch

import kaon._full_precision as fp
from kaon._compact_kahan import compensated_add_, decode, init_residual
from kaon._fused_triton import HAS_TRITON


def _pair(shape, bits, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    p = (torch.randn(shape, generator=g) * 0.05).to(device=device, dtype=torch.bfloat16)
    lo = init_residual(p, bits)
    compensated_add_(p, lo, (torch.randn(shape, generator=g) * 1e-4).to(device), bits=bits,
                     stochastic=False)
    return p, lo


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("n", [1, 999, 1000, 1001, 4097])
def test_chunked_cpu_decode_is_bit_identical(bits, n, monkeypatch):
    monkeypatch.setattr(fp, "_DECODE_CHUNK", 1000)
    p, lo = _pair((n,), bits)
    got = fp._decode_lowmem(p, lo, bits)
    assert got.dtype == torch.float32 and got.shape == p.shape
    assert torch.equal(got.view(torch.int32), decode(p, lo, bits).view(torch.int32))


@pytest.mark.parametrize("bits", [8, 16])
def test_chunked_cpu_decode_keeps_the_shape_and_handles_strided(bits, monkeypatch):
    monkeypatch.setattr(fp, "_DECODE_CHUNK", 64)
    p, lo = _pair((33, 17), bits)
    assert torch.equal(fp._decode_lowmem(p, lo, bits), decode(p, lo, bits))
    pt, lt = p.t(), lo.t()                       # non-contiguous: the reference path
    assert torch.equal(fp._decode_lowmem(pt, lt, bits), decode(pt, lt, bits))


@pytest.mark.skipif(not (torch.cuda.is_available() and HAS_TRITON), reason="needs CUDA+Triton")
@pytest.mark.parametrize("bits", [8, 16])
def test_cuda_decode_is_bit_identical_and_scratch_free(bits):
    p, lo = _pair((1536, 1536), bits, device="cuda")
    ref = decode(p, lo, bits)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    got = fp._decode_lowmem(p, lo, bits)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - base
    assert torch.equal(got.view(torch.int32), ref.view(torch.int32))
    out_bytes = p.numel() * 4
    assert extra <= out_bytes + (1 << 20), (
        f"decode peak {extra / 2**20:.1f} MiB > its fp32 output {out_bytes / 2**20:.1f} MiB"
    )
