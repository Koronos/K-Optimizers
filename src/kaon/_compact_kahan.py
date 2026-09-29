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

**``kahan16``** is the same codec at ``BITS = 16`` (+2 B/param, ``int16`` storage of the
uint16 pattern): nothing is dropped, so the pair is bit for bit an fp32 — ``lo`` is the fp32's
low half, ``w`` its high half carried half-away (the nearest bf16, as for ``kahan8``). A write
is plain fp32 arithmetic (round to nearest) followed by an exact split: no noise is drawn, and
given the same gradients the decoded weight is exactly what an fp32 master would hold. The
residual's dtype says which width wrote it (:func:`residual_bits_of`); a switch between the
two widths converts it (:func:`convert_residual`, exact when widening).

Non-finite values are stored bit-for-bit as the plain cast would store them (a NaN stays a
NaN, an inf an inf) with a zero residual — the same PROPAGATE policy as
``kaon._fused_triton.sr_round``: a diverged run must surface, not be buried. Conversely a
finite weight never decodes to a non-finite value, whatever the byte says: every one of
the 2**16 x 2**BITS states with a finite ``w`` decodes finite (``tests/test_compact_kahan.py``
enumerates them), the one corner being a ``+-0`` weight written from outside the optimizer
with a residual whose top bit is set — see :func:`decode`.

**Weights written from outside the codec** (a SAM/MSAM climb, an EMA swap, pruning) leave
``lo`` where it was: the residual then re-attaches to the NEW pattern's ulp. The value is
finite and within one ulp, but a climb-then-remove pair that rounds ``e`` on the bf16 grid
while the base step re-encodes the residual against the perturbed pattern loses the
sub-ulp part of ``e`` COHERENTLY every step — measured 25 ulp over 300 Nekaon steps at lr
1e-5 (worse than plain stochastic rounding). The MSAM/Nekaon climb is therefore
kahan8-aware: it perturbs and restores the DECODED value and re-encodes (with the residual's
stochastic rounding), so the clean value survives to ~1/256 ulp per cycle, unbiased.

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
    "convert_residual",
    "decode",
    "encode_",
    "init_residual",
    "is_compact_kahan",
    "residual_bits",
    "residual_bits_of",
    "residual_dtype",
]

#: ``bf16_method`` value -> number of residual bits stored per parameter. ``kahan8``: one
#: ``uint8``; ``kahan16``: one ``int16`` holding the uint16 pattern (torch's ``uint16`` has no
#: ``_foreach_copy_`` on CUDA, so the bits are stored in the signed type and every reader
#: masks them back to ``[0, 2**16)``). With 16 bits the pair ``(w, lo)`` IS an fp32: the
#: residual is the fp32's low half, bit for bit, and ``w`` its high half rounded half-away.
COMPACT_KAHAN_BITS: dict[str, int] = {"kahan8": 8, "kahan16": 16}
COMPACT_KAHAN_METHODS = tuple(COMPACT_KAHAN_BITS)

#: The per-parameter state key holding the residual (``p.shape``; ``torch.uint8`` under
#: ``kahan8``, ``torch.int16`` under ``kahan16`` — the dtype says which codec wrote it).
RESIDUAL_KEY = "kahan_lo"

_RESIDUAL_DTYPES = {8: torch.uint8, 16: torch.int16}
_BITS_OF_DTYPE = {torch.uint8: 8, torch.int16: 16}

_FLT_MAX = 3.4028234663852886e38


def is_compact_kahan(bf16_method: str) -> bool:
    return bf16_method in COMPACT_KAHAN_BITS


def residual_bits(bf16_method: str) -> int:
    return COMPACT_KAHAN_BITS[bf16_method]


def residual_dtype(bits: int) -> torch.dtype:
    """The storage dtype of a ``bits``-wide residual (``uint8`` for 8, ``int16`` for 16)."""
    return _RESIDUAL_DTYPES[bits]


def residual_bits_of(lo: Tensor) -> int:
    """The width of a STORED residual, read off its dtype — the codec that wrote it.

    Readers that only decode (a climb removal, Antikaon's clean read) use this rather than
    the group's ``bf16_method``: after a mid-run switch between ``kahan8`` and ``kahan16``
    the residual keeps its old encoding until the next write converts it
    (:func:`kaon._backend.ensure_residuals`), and decoding it with the new width would read
    garbage.
    """
    try:
        return _BITS_OF_DTYPE[lo.dtype]
    except KeyError:
        raise TypeError(f"not a compact-Kahan residual dtype: {lo.dtype}") from None


def init_residual(p: Tensor, bits: int = 8) -> Tensor:
    """A fresh zero residual for ``p`` (``lo == 0`` means ``z == w`` exactly)."""
    return torch.zeros(p.shape, dtype=_RESIDUAL_DTYPES[bits], device=p.device)


@torch.no_grad()
def convert_residual(p: Tensor, lo: Tensor, bits: int) -> Tensor:
    """Re-encode ``(p, lo)`` at a ``bits``-wide residual; returns the NEW residual tensor and
    rewrites ``p`` in place.

    ``kahan8 -> kahan16`` is exact (the 16-bit grid contains the 8-bit one; the stored bf16
    and the value are unchanged). ``kahan16 -> kahan8`` drops 8 bits with round-half-away —
    one deterministic rounding of at most half a grid unit (``ulp/512``), the stored bf16
    being re-derived from the kept value (it can move by one ulp at a rounding boundary).
    """
    z = decode(p, lo, residual_bits_of(lo))
    new = init_residual(p, bits)
    encode_(z, p, new, bits, None)
    return new


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
    # The carry, back to the truncated pattern — EXCEPT on a +-0 weight. The encoder never
    # produces (+-0, lo >= 2**(bits-1)) (a set top bit carries the pattern to 1, never to 0),
    # but an external write can: pruning, re-initialising to zero, loading a model while the
    # optimizer state is kept. Subtracting there wraps the magnitude to 0x7FFF (a NaN
    # pattern); the residual is instead read as a sub-ulp value of the same sign, which is
    # what it means everywhere else. 256 states out of 2**24 reach this branch.
    w.sub_((q >> (bits - 1)) * ((w & 0x7FFF) != 0))
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
    # All-finite fast path, CPU ONLY: when every lane is finite the mask is all-True, so the
    # two ``mul_(finite)`` and the final ``where`` are identities and the bf16 fallback
    # pattern ``pf`` is never selected — skipping them is bit-identical and saves four passes
    # plus a param-sized int16 temporary. Deciding it needs ``finite.all()`` on the host,
    # which on CUDA is a device sync per call; this torch path only runs on CUDA when the
    # Triton kernel declines a tensor, so there it keeps the sync-free masked form.
    masked = z.device.type != "cpu" or not bool(finite.all())
    # The non-finite lanes' bf16 pattern, taken BEFORE the low bits are touched (a NaN
    # whose payload sits only in the low bits would otherwise be masked into an inf).
    pf = z.to(torch.bfloat16).view(torch.int16) if masked else None
    if noise is None:
        b.add_(unit >> 1)
        q = b >> (16 - bits)
    else:
        if masked:
            noise.mul_(finite)                   # no rounding offset on non-finite lanes
        b.add_(noise)
        q = noise                                # reuse the buffer as the residual scratch
        torch.bitwise_right_shift(b, 16 - bits, out=q)
    q.bitwise_and_((1 << bits) - 1)
    if masked:
        q.mul_(finite)
    b.bitwise_and_(-unit)
    w = b >> 16
    w.bitwise_and_(0xFFFF).add_(q >> (bits - 1))
    # ``0x8000..0xFFFF`` are negative patterns: sign-extend before narrowing to int16.
    w.bitwise_left_shift_(16).bitwise_right_shift_(16)
    if masked:
        p.view(torch.int16).copy_(torch.where(finite, w.to(torch.int16), pf))
    else:
        p.view(torch.int16).copy_(w)             # int32 -> int16 narrowing, as ``.to`` does
    if bits == 16:
        # int16 storage of the uint16 pattern: sign-extend explicitly rather than rely on
        # the int32 -> int16 narrowing wrapping (it does on CPU and CUDA, but it is
        # implementation-defined in C++).
        q.bitwise_left_shift_(16).bitwise_right_shift_(16)
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
    rows: bool = False,
) -> None:
    """``(p, lo) += alpha * delta`` — decode, exact fp32 add, re-encode; in place. Torch path.

    ``rows=True`` (a stacked foreach bucket ``[N, *shape]``) makes the ``kahan16`` add run
    ROW BY ROW (``_foreach_add_`` over the unbound rows) instead of as one op over the stack.
    The CPU ``add_(alpha=)`` kernel is not layout-invariant: its vector body fuses
    ``z + alpha*d`` into an FMA while its scalar tail rounds the product first, so one op over
    the stack puts the tails in different places than N per-tensor ops do — 1 fp32 ulp on a
    few coordinates, breaking ``kahan16``'s bit-identity with the fp32 foreach writer
    (``_foreach_sub_`` over the per-param views) and with its own per-param path. Row by
    row reproduces the per-tensor tails exactly. ``kahan8`` ignores the flag: its residual
    grid (ulp/256) absorbs the difference and its numerics are left as reviewed.

    ``sr`` is the owner's noise stream (:class:`kaon._stochastic_rounding.SRStream`): the
    residual's stochastic rounding draws from the same checkpointed generator the bf16 SR
    write uses, so a resume reproduces it and a stacked (foreach) draw consumes the same
    sequence as the equivalent per-param draws (the torch path's ``foreach == per-param``
    identity). ``None`` falls back to the process-wide generator.
    """
    z = decode(p, lo, bits)
    d = delta if delta.dtype == torch.float32 else delta.float()
    if rows and bits == 16 and z.ndim > 1:
        torch._foreach_add_(list(z.unbind(0)), list(d.unbind(0)), alpha=alpha)
    else:
        z.add_(d, alpha=alpha)
    noise = None
    if stochastic and bits < 16:
        gen = _device_generator(z.device) if sr is None else sr.generator(z.device)
        noise = torch.randint(0, 1 << (16 - bits), z.shape, dtype=torch.int32,
                              device=z.device, generator=gen)
    encode_(z, p, lo, bits, noise)
