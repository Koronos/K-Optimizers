"""Tests for kaon._stochastic_rounding.add_stochastic_."""

from __future__ import annotations

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from kaon._stochastic_rounding import add_stochastic_, reseed_generators


def _device_ids() -> list[str]:
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def _bf16_from_fp32_bits(bits: int, device: str | torch.device = "cpu") -> torch.Tensor:
    """Build a length-1 bf16 tensor from a raw fp32 bit pattern (via fp32 cast)."""
    # int64 first so 0xFFFFFFFF-style patterns wrap into int32 instead of raising.
    raw = torch.tensor([bits], dtype=torch.int64, device=device).to(torch.int32).view(torch.float32)
    return raw.to(torch.bfloat16)


@pytest.mark.parametrize("device", _device_ids())
def test_nonfinite_preserved_nan_canonical_payload(device: str) -> None:
    """CUDA-canonical NaN payload (0x7FFFFFFF) must not become -0.0 after SR."""
    target = _bf16_from_fp32_bits(0x7FFFFFFF, device)
    source = torch.zeros(1, dtype=torch.float32, device=device)
    add_stochastic_(target, source, alpha=1.0)
    assert torch.isnan(target).all()


@pytest.mark.parametrize("device", _device_ids())
def test_nonfinite_preserved_negative_nan(device: str) -> None:
    """Negative-NaN payload (0xFFFFFFFF) must survive SR unchanged."""
    target = _bf16_from_fp32_bits(0xFFFFFFFF, device)
    source = torch.zeros(1, dtype=torch.float32, device=device)
    add_stochastic_(target, source, alpha=1.0)
    assert torch.isnan(target).all()


@pytest.mark.parametrize("device", _device_ids())
def test_finite_above_bf16_max_can_round_to_inf(device: str) -> None:
    """Finites above bf16 max are not clamped: SR sends a fraction of them to inf like RNE."""
    # 1.002 * bf16 max is still a finite fp32 (fp32 max is 3.4e38) and lies between
    # bf16 max and the next bf16 value (inf), so SR must split the outcomes.
    above_max = torch.full(
        (4096,), torch.finfo(torch.bfloat16).max * 1.002, dtype=torch.float32, device=device
    )
    assert above_max.isfinite().all()
    target = torch.zeros(4096, dtype=torch.bfloat16, device=device)
    add_stochastic_(target, above_max, alpha=1.0)
    assert target.isinf().any(), "SR clamped every value below inf"
    assert (target == torch.finfo(torch.bfloat16).max).any()


def test_global_rng_untouched_after_sr() -> None:
    """SR must not consume from the global RNG (dataloader/dropout reproducibility)."""
    target = torch.zeros(16, dtype=torch.bfloat16)
    source = torch.randn(16)
    torch.manual_seed(1234)
    ref = torch.rand(3).clone()
    torch.manual_seed(1234)
    add_stochastic_(target, source, alpha=1.0)
    after = torch.rand(3)
    assert torch.equal(ref, after)


@pytest.mark.parametrize("device", _device_ids())
def test_sr_reproducible_with_manual_seed(device: str) -> None:
    """Re-seeding via manual_seed restores the same SR draws (public API only)."""
    source = torch.randn(32, dtype=torch.float32, device=device)

    def run(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        target = torch.ones(32, dtype=torch.bfloat16, device=device)
        add_stochastic_(target, source, alpha=0.5)
        return target.clone()

    first = run(999)
    other = run(777)  # a different seed in between is picked up automatically
    second = run(999)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    # Same seed twice in a row is not observable through the global RNG: the module
    # generator keeps its stream unless reseed_generators() is called.
    reseed_generators()
    third = run(999)
    assert torch.equal(first, third)


class _ToCopyCounter(TorchDispatchMode):
    """Counts ``aten::_to_copy`` kernels issued during a region."""

    def __init__(self) -> None:
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001, ANN204
        if func is torch.ops.aten._to_copy.default:
            self.count += 1
        return func(*args, **(kwargs or {}))


# bf16 target + fp32 source: one widening cast (target.float()); the write-back is an
# in-place copy_ (aten::copy_, not _to_copy) and the finite mask promotes bool inline.
_EXPECTED_BF16_SR_TO_COPY = 1


def test_add_stochastic_bf16_to_copy_kernel_budget() -> None:
    """Write-back uses copy_(fp32), not a separate .to(bfloat16) allocation."""
    target = torch.randn(64, dtype=torch.bfloat16)
    source = torch.randn(64, dtype=torch.float32)
    with _ToCopyCounter() as counter:
        add_stochastic_(target, source, alpha=1.0)
    assert counter.count == _EXPECTED_BF16_SR_TO_COPY


def test_fp32_passthrough() -> None:
    """fp32 targets use plain add_ with no rounding trick."""
    target = torch.tensor([1.0, 2.0], dtype=torch.float32)
    source = torch.tensor([0.5, -1.0], dtype=torch.float32)
    add_stochastic_(target, source, alpha=2.0)
    assert torch.allclose(target, torch.tensor([2.0, 0.0]))


def test_fp16_raises() -> None:
    with pytest.raises(NotImplementedError, match="fp16"):
        add_stochastic_(torch.zeros(1, dtype=torch.float16), torch.zeros(1))
