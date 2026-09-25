"""Compact Kahan: a bf16 weight plus a few *fixed-point* residual bits, as ONE wider float.

``bf16_method="kahan"`` keeps a bf16 compensation buffer next to every low-precision weight
(+2 B/param, the size of the model again). This module is the cheaper replacement:
``"kahan8"`` stores the residual as ONE byte per parameter, in units of the weight's own
bf16 ulp. The pair ``(w, lo)`` is then simply an fp32 whose low ``16 - BITS`` mantissa bits
are zero — a float with a 16-bit significand (8 from bf16, 8 from the byte) for ``kahan8``:

    fp32 bits(z) == (trunc16 << 16) | (lo << (16 - BITS))        with  trunc16 = w16 - (lo >> (BITS-1))

where ``w16`` is the bf16 bit pattern the model actually holds and ``trunc16`` is ``z``
truncated toward zero to bf16. The stored weight ``w`` is ``z`` rounded half-away to bf16:
whenever the residual's top bit is set (``lo >= 2**(BITS-1)``, i.e. the residual is at least
half an ulp) the pattern carries one unit up in magnitude. The forward pass therefore sees
the *nearest* bf16 to the compensated value, never a truncation biased toward zero.

Why the exponent of ``w`` is the scale (no per-block scale, no memory for it): the residual
of a nearest-rounding is bounded by half an ulp of ``w``, so ``BITS`` bits at ``ulp/2**BITS``
resolution cover it exactly, and bf16's own exponent field already says which ulp that is.
Binade crossings cost nothing: the ``+1`` carry on the 16-bit pattern *is* the crossing
(``0x3FFF + 1 == 0x4000``: ``1.9921875 -> 2.0``), zero and subnormals sit on the same
monotone pattern, and the decode is integer-only, so no float arithmetic on a subnormal
residual is ever performed (no flush-to-zero exposure).

**Rounding of the residual.** Each write computes the exact ``z + alpha * delta`` in fp32
(24-bit significand) and has to drop ``16 - BITS`` low bits. Two policies:

* ``stochastic=True`` (the default, and what the ``kahan8`` method uses): stochastic
  rounding at the residual's grid (uniform noise in ``[0, 2**(16-BITS))`` added to the
  bits before the mask — the same bit trick as :mod:`kaon._stochastic_rounding`, at a
  finer grain). Unbiased for EVERY step size: ``E[z_stored] == z`` conditionally on the
  state, so the error is a martingale with per-step variance ``<= unit**2 / 4`` (``unit =
  ulp / 2**BITS``), i.e. a walk of ``unit * sqrt(N/6)`` after ``N`` steps once the steps
  exceed the grid — 0.16 ulp at 10k steps for ``kahan8``, against 16-40 ulp for plain bf16
  stochastic rounding in the sub-ulp regime, and NO stall for steps below the grid, where
  every round-to-nearest scheme (the bf16 ``kahan`` buffer included) loses 30-40 % of the
  movement (see ``docs/research/compact-kahan.md``).
* ``stochastic=False``: round half away from zero. Deterministic and ``sqrt 2`` tighter
  when the update dithers the residual, but it STALLS on updates below half a grid unit
  (``|alpha*delta| < ulp/512`` for ``kahan8``) exactly as plain RTN stalls below half an
  ulp — kept as a reference/diagnostic policy, not exposed as a ``bf16_method``.

Non-finite values are stored bit-for-bit as the plain cast would store them (a NaN stays a
NaN, an inf an inf) with a zero residual — the same PROPAGATE policy as
``kaon._fused_triton.sr_round``: a diverged run must surface, not be buried.

The Triton kernels in :mod:`kaon._fused_triton` (``ck_decode`` / ``ck_store``, and the
one-launch ``ck_add_`` the native writers prefer on CUDA) implement the identical bit
manipulation; the torch functions here are the reference and the CPU / non-contiguous path.
The torch path is written with in-place integer ops on as few parameter-sized temporaries
as it can (measured ~23 B/elem transient against ~48 B/elem for the naive expression), but
it is still a dozen kernels — which is why the CUDA writers take the Triton kernel.
"""
from __future__ import annotations

import torch
from torch import Tensor

from kaon._stochastic_rounding import SRStream, _device_generator

__all__ = [
    "COMPACT_KAHAN_BITS",
    "COMPACT_KAHAN_METHODS",
    "RESIDUAL_KEY",
    "compensated_add_",
    "decode",
    "encode_",
    "init_residual",
    "is_compact_kahan",
    "residual_bits",
]

#: ``bf16_method`` value -> number of residual bits stored per parameter (in one uint8).
COMPACT_KAHAN_BITS: dict[str, int] = {"kahan8": 8}
COMPACT_KAHAN_METHODS = tuple(COMPACT_KAHAN_BITS)

#: The per-parameter state key holding the residual byte (``torch.uint8``, ``p.shape``).
RESIDUAL_KEY = "kahan_lo"

_FLT_MAX = 3.4028234663852886e38


def is_compact_kahan(bf16_method: str) -> bool:
    return bf16_method in COMPACT_KAHAN_BITS


def residual_bits(bf16_method: str) -> int:
    return COMPACT_KAHAN_BITS[bf16_method]


def init_residual(p: Tensor) -> Tensor:
    """A fresh zero residual for ``p`` (``lo == 0`` means ``z == w`` exactly)."""
    return torch.zeros(p.shape, dtype=torch.uint8, device=p.device)


@torch.no_grad()
def decode(p: Tensor, lo: Tensor, bits: int = 8) -> Tensor:
    """The compensated fp32 value ``z`` of ``(p, lo)``. Exact, integer-only (see module doc).

    Returns a NEW fp32 tensor (the caller may add into it in place). Three int32
    temporaries at peak, all in-place ops otherwise.
    """
    if p.dtype != torch.bfloat16:
        raise TypeError(f"compact kahan holds bf16 weights, got {p.dtype}")
    w = p.view(torch.int16).to(torch.int32)
    w.bitwise_and_(0xFFFF)
    q = lo.to(torch.int32)
    q.bitwise_and_((1 << bits) - 1)              # int16 storage (bits == 16) is sign-extended
    w.sub_(q >> (bits - 1))                      # the carry: back to the truncated pattern
    w.bitwise_left_shift_(16)
    q.bitwise_left_shift_(16 - bits)
    w.bitwise_or_(q)
    return w.view(torch.float32)


@torch.no_grad()
def encode_(z: Tensor, p: Tensor, lo: Tensor, bits: int = 8, noise: Tensor | None = None) -> None:
    """Store ``z`` (fp32) into ``(p, lo)`` in place, rounding it to the ``16 + BITS``-bit grid.

    ``noise`` is an int32 tensor of uniform draws in ``[0, 2**(16-bits))`` for stochastic
    rounding, or ``None`` for round-half-away-from-zero. ``z`` and ``noise`` are CONSUMED
    (used as scratch); do not read them afterwards.
    """
    unit = 1 << (16 - bits)
    b = z.view(torch.int32)
    finite = z.abs() <= _FLT_MAX                 # bool; False for NaN and +-inf
    # The non-finite lanes' bf16 pattern, taken BEFORE the low bits are touched (a NaN
    # whose payload sits only in the low bits would otherwise be masked into an inf).
    pf = z.to(torch.bfloat16).view(torch.int16)
    if noise is None:
        b.add_(unit >> 1)
        q = b >> (16 - bits)
    else:
        noise.mul_(finite)                       # no rounding offset on non-finite lanes
        b.add_(noise)
        q = noise                                # reuse the buffer as the residual scratch
        torch.bitwise_right_shift(b, 16 - bits, out=q)
    q.bitwise_and_((1 << bits) - 1).mul_(finite)
    b.bitwise_and_(-unit)
    w = b >> 16
    w.bitwise_and_(0xFFFF).add_(q >> (bits - 1))
    # ``0x8000..0xFFFF`` are negative patterns: sign-extend before narrowing to int16.
    w.bitwise_left_shift_(16).bitwise_right_shift_(16)
    p.view(torch.int16).copy_(torch.where(finite, w.to(torch.int16), pf))
    lo.copy_(q)                                  # uint8 (bits <= 8) or int16 (bits == 16)


@torch.no_grad()
def compensated_add_(
    p: Tensor,
    lo: Tensor,
    delta: Tensor,
    alpha: float = 1.0,
    bits: int = 8,
    sr: SRStream | None = None,
    stochastic: bool = True,
) -> None:
    """``(p, lo) += alpha * delta`` — decode, exact fp32 add, re-encode; in place. Torch path.

    ``sr`` is the owner's noise stream (:class:`kaon._stochastic_rounding.SRStream`): the
    residual's stochastic rounding draws from the same checkpointed generator the bf16 SR
    write uses, so a resume reproduces it and a stacked (foreach) draw consumes the same
    sequence as the equivalent per-param draws (the torch path's ``foreach == per-param``
    identity). ``None`` falls back to the process-wide generator.
    """
    z = decode(p, lo, bits)
    z.add_(delta if delta.dtype == torch.float32 else delta.float(), alpha=alpha)
    noise = None
    if stochastic and bits < 16:
        gen = _device_generator(z.device) if sr is None else sr.generator(z.device)
        noise = torch.randint(0, 1 << (16 - bits), z.shape, dtype=torch.int32,
                              device=z.device, generator=gen)
    encode_(z, p, lo, bits, noise)
