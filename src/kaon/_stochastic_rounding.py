"""Stochastic rounding primitive for fp32 -> bf16 weight updates.

When training a model with bf16 parameters, the standard ``p += -lr * update``
truncates the fp32 result to bf16 using round-to-nearest-even. Small updates
that fall below the bf16 ULP of the weight round to zero, and the optimizer
effectively makes no progress on those parameters.

Two well-known mitigations exist:

1. **Kahan summation** — keep a per-parameter compensation buffer that
   accumulates the lost low-order bits across steps. Costs ~2 B/param of
   extra state, equal to the size of the model itself for bf16 weights.
2. **Stochastic rounding** — randomly round up or down with probability
   proportional to the fractional distance. The expected value is the
   exact fp32 result, so updates are preserved *in expectation* without
   any extra state.

This module implements (2): ``add_stochastic_(target, source, alpha)``.

The bf16 implementation uses the integer bit-manipulation trick from
``lodestone-rock/torchastic`` and ``AmericanPresidentJimmyCarter/adamw-bf16``:
add uniform noise in ``[0, 2**16)`` to the int32 view of the fp32 result,
then truncate to the upper 16 bits. The expected value of the rounding
step equals the unbiased rounding of the input.

For fp16 targets (different exponent layout) the bit trick does not apply
directly; ``NotImplementedError`` is raised — fall back to ``bf16_method='kahan'``.

**RNG isolation.** Noise is drawn from a per-device :class:`torch.Generator`
owned by this module (not the global CUDA/CPU RNG). The generator is seeded
from the global initial seed on first use and re-seeded whenever
:func:`torch.manual_seed` / :func:`torch.cuda.manual_seed_all` changes that
seed, so ``torch.manual_seed`` before each run still yields reproducible
training. Subsequent ``torch.rand`` calls are unaffected by stochastic-rounding
steps. Limitation: re-seeding to the *same* value inside one process (with no
different seed in between) is not observable through the global RNG, so the
module generator keeps its stream; call :func:`kaon.reseed_stochastic_rounding` (alias of :func:`reseed_generators`) after
``torch.manual_seed`` in that case (test suites, sweeps in one process).
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["add_stochastic_", "reseed_generators"]

# Per-device RNG — isolated from the global stream so SR does not perturb dataloader/dropout.
# Value is (generator, global_initial_seed at last sync).
_generators: dict[torch.device, tuple[torch.Generator, int]] = {}


def _global_initial_seed(device: torch.device) -> int:
    """Initial seed of the global RNG for ``device`` (multi-GPU safe on CUDA)."""
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return torch.cuda.default_generators[idx].initial_seed()
    return torch.initial_seed()


def reseed_generators() -> None:
    """Drop the module generators so the next SR call re-seeds from the global RNG.

    Needed only when ``torch.manual_seed`` is called again with the *same* seed in one
    process; a seed change is picked up automatically.
    """
    _generators.clear()


def _device_generator(device: torch.device) -> torch.Generator:
    """Return the module-owned generator for ``device``, synced to the global seed."""
    seed = _global_initial_seed(device)
    entry = _generators.get(device)
    if entry is None or entry[1] != seed:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        _generators[device] = (gen, seed)
    return _generators[device][0]


@torch.no_grad()
def add_stochastic_(target: Tensor, source: Tensor, alpha: float = 1.0) -> None:
    """In-place ``target += alpha * source`` with stochastic rounding.

    The rounding step happens on the final cast back to ``target.dtype``.
    For fp32 targets this is just a plain ``target.add_(source, alpha=alpha)``
    — there is no precision loss to compensate. For bf16 targets the
    integer bit trick (see module docstring) is used.

    Args:
        target: Destination tensor, modified in place.
        source: Tensor of the same shape as ``target``. Cast to fp32
            internally if not already.
        alpha: Scalar multiplier applied to ``source`` before adding.

    Raises:
        NotImplementedError: For ``target.dtype`` other than bf16 or fp32.
    """
    if target.dtype == torch.float32:
        target.add_(source, alpha=alpha)
        return
    if target.dtype == torch.bfloat16:
        _add_stochastic_bf16_(target, source, alpha)
        return
    raise NotImplementedError(
        f"add_stochastic_ for target dtype {target.dtype} is not implemented; "
        "currently only torch.bfloat16 and torch.float32 are supported "
        "(use bf16_method='kahan' for fp16 parameters)"
    )


@torch.no_grad()
def _add_stochastic_bf16_(
    target_bf16: Tensor,
    source: Tensor,
    alpha: float,
) -> None:
    source_fp32 = source if source.dtype == torch.float32 else source.float()

    # Compute the exact fp32 result of the addition.
    result_fp32 = target_bf16.float()
    result_fp32.add_(source_fp32, alpha=alpha)

    # Stochastic-round to bf16 via the int32 bit trick.
    # bf16 keeps the top 16 bits of an fp32 representation; the lower 16
    # are dropped on a normal cast. Adding uniform noise in [0, 2^16) to
    # those lower bits and then truncating makes the upper-bit "round up"
    # event happen with probability equal to the fractional distance,
    # which is exactly unbiased stochastic rounding.
    bits = result_fp32.view(torch.int32)
    noise = torch.randint(
        low=0,
        high=0x10000,
        size=bits.shape,
        dtype=torch.int32,
        device=bits.device,
        generator=_device_generator(bits.device),
    )
    # Only NaN needs masking: CUDA canonicalizes fp32 NaN to 0x7FFFFFFF (and
    # -NaN to 0xFFFFFFFF); adding noise overflows to 0x8000xxxx and the AND
    # yields -0.0. ±inf and finites above bf16 max are already safe (inf noise
    # stays in-range; large finites may stochastically round to inf like RNE).
    # One extra kernel: fold the NaN test into the add operand. +-inf need no mask
    # (0x7F800000 + noise <= 0x7F80FFFF and the AND below restores it). Measured
    # cheaper than isfinite() + mul_ (two kernels) on CUDA and CPU.
    noise = torch.where(result_fp32 != result_fp32, 0, noise)
    # In two's complement, ``-0x10000`` is the int32 mask ``0xFFFF0000``.
    bits.add_(noise).bitwise_and_(-0x10000)

    # Lower 16 bits are now zero, so the bf16 cast is exact. copy_ fuses the
    # dtype conversion into the destination without a full-sized bf16 temp.
    # We benchmarked copying the top-16-bit lanes via an int16 view to skip
    # this convert; the strided copy is slower on CUDA than the fused contiguous
    # cast, so copy_ wins. A fused Triton kernel for the whole add+round remains
    # the real optimization — see CHANGELOG.
    target_bf16.copy_(result_fp32)
