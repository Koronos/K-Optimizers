"""Tests for kaon._backend shared primitives."""

from __future__ import annotations

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from kaon._backend import centralize_grads_, subtract_batched_
from kaon._stochastic_rounding import add_stochastic_


class _ToCopyCounter(TorchDispatchMode):
    """Counts ``aten::_to_copy`` kernels issued during a region."""

    def __init__(self) -> None:
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001, ANN204
        if func is torch.ops.aten._to_copy.default:
            self.count += 1
        return func(*args, **(kwargs or {}))


def _subtract_batched_casts(n_params: int, shape: tuple[int, ...]) -> int:
    """Dtype-conversion kernels in one subtract_batched_ call (bf16 params, fp32 delta, no SR)."""
    pviews = [torch.zeros(shape, dtype=torch.bfloat16) for _ in range(n_params)]
    delta = torch.randn(n_params, *shape)
    with _ToCopyCounter() as counter:
        subtract_batched_(pviews, delta, bf16_method="none")
    return counter.count


def test_subtract_batched_casts_do_not_scale_with_n() -> None:
    """Stacked delta is cast once per bucket, not once per parameter."""
    small = _subtract_batched_casts(8, (4,))
    large = _subtract_batched_casts(32, (4,))
    assert small > 0, "no cast observed — test no longer covers the dtype-mismatch branch"
    assert large == small, (
        f"cast kernels grew with bucket size ({small} at N=8 -> {large} at N=32)"
    )


def test_centralize_grads_multi_device() -> None:
    """Same-shape grads on different devices must not be stacked together."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    p_cpu = torch.nn.Parameter(torch.randn(8, 16))
    p_gpu = torch.nn.Parameter(torch.randn(8, 16, device="cuda"))
    g_cpu = torch.randn(8, 16)
    g_gpu = torch.randn(8, 16, device="cuda")
    p_cpu.grad = g_cpu.clone()
    p_gpu.grad = g_gpu.clone()
    cpu_before = g_cpu.clone()
    gpu_before = g_gpu.clone()
    centralize_grads_([p_cpu, p_gpu])
    # Each grad should have its fan-in mean subtracted independently.
    cpu_mean = cpu_before.mean(dim=1, keepdim=True)
    gpu_mean = gpu_before.mean(dim=1, keepdim=True)
    assert torch.allclose(p_cpu.grad, cpu_before - cpu_mean)
    assert torch.allclose(p_gpu.grad, gpu_before - gpu_mean)


def test_centralize_grads_batched_same_device() -> None:
    """Same-shape grads on one device are still batched."""
    params = [torch.nn.Parameter(torch.randn(4, 8)) for _ in range(4)]
    grads = [torch.randn(4, 8) for _ in range(4)]
    for p, g in zip(params, grads, strict=True):
        p.grad = g.clone()
    before = [g.clone() for g in grads]
    centralize_grads_(params)
    for p, g0 in zip(params, before, strict=True):
        mean = g0.mean(dim=1, keepdim=True)
        assert torch.allclose(p.grad, g0 - mean)


def test_subtract_batched_sr_matches_per_param(monkeypatch: pytest.MonkeyPatch) -> None:
    """Batched SR is bit-identical to per-param SR given the same noise draws."""
    shape = (3, 4)
    n = 2
    numel = n * shape[0] * shape[1]
    fixed = torch.randint(0, 0x10000, (numel,), dtype=torch.int32)
    offset = [0]
    real_randint = torch.randint

    def patched_randint(*args, **kwargs):  # noqa: ANN002, ANN003
        out = real_randint(*args, **kwargs)
        n_out = out.numel()
        out.copy_(fixed[offset[0] : offset[0] + n_out].view_as(out))
        offset[0] += n_out
        return out

    monkeypatch.setattr(torch, "randint", patched_randint)

    pviews_batch = [torch.ones(shape, dtype=torch.bfloat16) for _ in range(n)]
    pviews_param = [p.clone() for p in pviews_batch]
    delta = torch.full((n, *shape), 0.01, dtype=torch.float32)

    subtract_batched_(pviews_batch, delta.clone(), bf16_method="stochastic_rounding")

    offset[0] = 0
    for i, p in enumerate(pviews_param):
        add_stochastic_(p, delta[i], alpha=-1.0)

    for batched, per_param in zip(pviews_batch, pviews_param, strict=True):
        torch.testing.assert_close(batched, per_param, rtol=0, atol=0)
