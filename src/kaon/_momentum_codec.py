"""Shared first-moment (momentum) codecs for kaon optimizers.

A *momentum codec* owns, for one ``momentum_dtype``, the entire
dequant -> fp32 EMA -> requant cycle and the underlying storage layout, so the
optimizer step functions never re-implement any quantization detail. Both
:class:`~kaon.adakaon.Adakaon` and :class:`~kaon.kprodigy.KProdigy`
import these classes — Adakaon uses the EMA entry points (it folds the EMA into
the update), while KProdigy does its (``d``-scaled) EMA itself in pass 1 and only
needs the codec's storage + *read-only dequant* in pass 2. The read-only
``dequant_*`` methods were added for KProdigy and are a no-op extension of the
Adakaon-era API (the EMA paths are byte-for-byte unchanged).

The first-moment EMA is always *worked on* as an fp32 tensor in the "effective"
layout — matricized ``[R, C]`` (factored) / flat ``[L]`` (non-factored) per
param, and stacked ``[N, R, C]`` / ``[N, L]`` in the foreach buckets.

Supported ``momentum_dtype`` codecs:

* ``"float32"`` / ``"bfloat16"`` — :class:`_FloatCodec` (store ``m`` directly).
* ``"int8"``                     — :class:`_Int8Codec` (per-row absmax).
* ``"4bit"``                     — :class:`_FourBitCodec` (per-block absmax,
                                   nibble-packed two-per-byte).
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import torch
from torch import Tensor

from kaon._backend import SRSeedState

__all__ = [
    "MomentumDtype",
    "_FOURBIT_BLOCK",
    "_MomentumCodec",
    "_StackedViews",
    "_FloatCodec",
    "_Int8Codec",
    "_FourBitCodec",
    "_make_codec",
    "warn_if_4bit_high_beta1",
    "load_state_dict_preserving_dtypes",
    "_quant_int8",
    "_quant_int8_stacked",
    "_pack_nibbles",
    "_unpack_nibbles",
    "_quant_4bit",
    "_dequant_4bit",
    "_quant_4bit_stacked",
    "_dequant_4bit_stacked",
    "fourbit_block_size",
    "int8_scale_shape",
]

MomentumDtype = ("bfloat16", "float32", "int8", "4bit")

# Block size (number of consecutive flattened elements sharing one absmax scale)
# for 4-bit momentum. Li et al. ("Memory Efficient Optimizers with 4-bit States",
# NeurIPS 2023, arXiv:2309.01507) found small blocks materially help at 4-bit; a
# fidelity replay on real SDXL gradients here confirmed block 128 ≈ int8 fidelity.
_FOURBIT_BLOCK = 128

# Quantization levels and floors, named so the (de)quant math reads identically in every
# helper. Both codecs are SYMMETRIC signed linear quantizers, so they deliberately leave one
# code unused at the bottom of the range (int8: -128; 4-bit: -8) to keep the scale symmetric.
_INT8_ABSMAX = 127.0     # scale = absmax / 127  -> codes span [-127, 127]
_INT8_CLAMP = 127
_FOURBIT_ABSMAX = 7.0    # scale = absmax / 7    -> codes span [-7, 7]
_FOURBIT_CLAMP = 7
_FOURBIT_ZERO = 8        # +8 shift maps signed [-7, 7] -> unsigned nibble [1, 15] (nibble 0 unused)
_ABSMAX_FLOOR = 1e-12    # floor on absmax so an all-zero row/block can't divide by zero


def _absmax(x: Tensor, dims: int | tuple[int, ...]) -> Tensor:
    """``|x|.amax(dims, keepdim=True)`` floored at :data:`_ABSMAX_FLOOR`, empty-safe.

    ``amax`` refuses to reduce over a zero-size dimension, which is every int8 reduction
    of a zero-element param (``(0,)``, ``(5, 0)``, ``(3, 0, 2)``): the floor is exactly the
    scale such a tensor gets, so it is produced directly (same shape ``keepdim`` gives).
    """
    if x.numel() == 0:
        red = range(x.ndim) if dims == () else (dims,) if isinstance(dims, int) else dims
        red = {d % x.ndim for d in red}
        shape = [1 if i in red else n for i, n in enumerate(x.shape)]
        return x.new_full(shape, _ABSMAX_FLOOR)
    return x.abs().amax(dim=dims, keepdim=True).clamp_(min=_ABSMAX_FLOOR)


def _row_rest(shape: tuple[int, ...]) -> tuple[int, int]:
    """``(row, rest)`` of the int8 requant's ``[row, rest]`` layout for a per-param shape.

    ``rest = per // row`` — exact for every non-empty shape (``per`` is a multiple of
    ``row``). An empty shape gets ``rest = 0`` (and a zero-row one no division), so the
    ``[row, rest]`` view of its zero elements is well-formed: the old ``max(per // row, 1)``
    asked a zero-element tensor for ``row`` elements, and ``(0, 4)`` divided by zero.
    """
    per = math.prod(shape) if shape else 1
    row = shape[0] if len(shape) >= 2 else 1
    return row, (per // row if row else 0)


def _quant_int8(m_fp32: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize a momentum tensor to int8 with a per-row (dim-0) absmax scale.

    Per-row scaling keeps a single outlier from collapsing the whole tensor's
    resolution (a coarse stand-in for bitsandbytes' block-wise scheme). 1-D
    tensors use a single scalar scale.
    """
    dims = tuple(range(1, m_fp32.ndim)) if m_fp32.ndim >= 2 else ()
    absmax = _absmax(m_fp32, dims)
    scale = absmax / _INT8_ABSMAX
    q = (m_fp32 / scale).round_().clamp_(-_INT8_CLAMP, _INT8_CLAMP).to(torch.int8)
    return q, scale


def int8_scale_shape(m: Tensor) -> tuple[int, ...]:
    """Shape :func:`_quant_int8` gives the scale of a momentum buffer shaped like ``m``.

    The batched requant reduces a *matrixized* ``[N, R, rest]`` view, so it has to
    reshape each per-param scale back into this layout before storing it. Hardcoding
    ``(R, 1)`` there is wrong for anything that is not exactly 2-D: a conv's scale is
    ``(R, 1, 1, 1)`` and a 0-D param's is a scalar, and a param stepped once by the
    foreach path could then never be stepped per-param again (the stored scale
    mis-broadcasts against the momentum — a hard error, not a silent skew).

    Mirrors :func:`_quant_int8`'s ``keepdim`` reduction exactly: ``ndim >= 2`` keeps
    the dim-0 row axis with ones elsewhere; 1-D reduces over an empty dim tuple to
    ``(1,)``; 0-D stays a scalar.
    """
    if m.ndim >= 2:
        return (m.shape[0],) + (1,) * (m.ndim - 1)
    return (1,) * m.ndim


def _quant_int8_stacked(m_fp32: Tensor) -> tuple[Tensor, Tensor]:
    """Batched :func:`_quant_int8` for a stacked momentum tensor.

    ``m_fp32`` is the stacked momentum in its *effective row layout* — either
    ``[N, R, C]`` (factored bucket) or ``[N, L]`` (non-factored bucket).
    Reducing only the trailing axis here is element-for-element the same set of
    values the per-param path reduces per tensor, so the scales match exactly.
    """
    absmax = _absmax(m_fp32, -1)
    scale = absmax / _INT8_ABSMAX
    q = (m_fp32 / scale).round_().clamp_(-_INT8_CLAMP, _INT8_CLAMP).to(torch.int8)
    return q, scale


def _pack_nibbles(nib: Tensor) -> Tensor:
    """Pack a flat tensor of 4-bit values (``uint8`` in ``[0, 15]``) two-per-byte.

    Operates on the LAST dim so a stacked ``[N, K]`` input packs each row
    independently into ``[N, ceil(K/2)]``. Odd ``K`` is zero-padded (the dangling
    high nibble of the final byte is ignored on unpack).
    """
    k = nib.shape[-1]
    if k % 2:
        nib = torch.cat([nib, nib.new_zeros(*nib.shape[:-1], 1)], dim=-1)
    pair = nib.reshape(*nib.shape[:-1], -1, 2)
    return (pair[..., 0] | (pair[..., 1] << 4)).to(torch.uint8)


def _unpack_nibbles(packed: Tensor, k: int) -> Tensor:
    """Inverse of :func:`_pack_nibbles`: ``[..., ceil(k/2)]`` bytes -> ``[..., k]``."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], -1)
    return out[..., :k]


def _quant_4bit(m_fp32: Tensor, block_size: int) -> tuple[Tensor, Tensor, int]:
    """Quantize ``m_fp32`` to signed linear 4-bit with a per-block absmax scale.

    Returns ``(packed_uint8[ceil(numel/2)], scale_fp32[nblocks], numel)``. The
    flat-block layout is identical whether a single tensor or a stacked bucket is
    quantized, so the batched and per-param paths agree bit-for-bit.
    """
    numel = m_fp32.numel()
    flat = m_fp32.reshape(-1)
    nblocks = (numel + block_size - 1) // block_size
    pad = nblocks * block_size - numel
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(nblocks, block_size)
    absmax = blocks.abs().amax(dim=1, keepdim=True).clamp_(min=_ABSMAX_FLOOR)
    scale = absmax / _FOURBIT_ABSMAX
    q = (blocks / scale).round_().clamp_(-_FOURBIT_CLAMP, _FOURBIT_CLAMP).to(torch.int8)
    nib = (q + _FOURBIT_ZERO).to(torch.uint8).reshape(-1)[:numel]
    packed = _pack_nibbles(nib)
    return packed, scale.reshape(nblocks), numel


def _dequant_4bit(packed: Tensor, scale: Tensor, numel: int, block_size: int) -> Tensor:
    """Inverse of :func:`_quant_4bit`: -> flat fp32 of length ``numel``."""
    nib = _unpack_nibbles(packed, numel)
    q = nib.to(torch.float32) - _FOURBIT_ZERO
    nblocks = scale.shape[0]
    pad = nblocks * block_size - numel
    if pad:
        q = torch.cat([q, q.new_zeros(pad)])
    q = q.view(nblocks, block_size).mul_(scale.view(nblocks, 1))
    return q.reshape(-1)[:numel]


def _quant_4bit_stacked(m_fp32: Tensor, block_size: int) -> tuple[Tensor, Tensor]:
    """Batched :func:`_quant_4bit` for a stacked ``[N, ...]`` momentum tensor.

    Each of the ``N`` slices is flattened and block-quantized independently, so the
    block boundaries match the per-param path exactly. Returns ``(packed[N, B],
    scale[N, nblocks])``.
    """
    n = m_fp32.shape[0]
    per = m_fp32[0].numel()
    flat = m_fp32.reshape(n, per)
    nblocks = (per + block_size - 1) // block_size
    pad = nblocks * block_size - per
    if pad:
        flat = torch.cat([flat, flat.new_zeros(n, pad)], dim=1)
    blocks = flat.view(n, nblocks, block_size)
    absmax = blocks.abs().amax(dim=2, keepdim=True).clamp_(min=_ABSMAX_FLOOR)
    scale = absmax / _FOURBIT_ABSMAX
    q = (blocks / scale).round_().clamp_(-_FOURBIT_CLAMP, _FOURBIT_CLAMP).to(torch.int8)
    nib = (q + _FOURBIT_ZERO).to(torch.uint8).reshape(n, -1)[:, :per]
    packed = _pack_nibbles(nib)                          # [N, ceil(per/2)]
    return packed, scale.reshape(n, nblocks)


def _dequant_4bit_stacked(packed: Tensor, scale: Tensor, per: int, block_size: int) -> Tensor:
    """Inverse of :func:`_quant_4bit_stacked`: -> ``[N, per]`` fp32."""
    n = packed.shape[0]
    nib = _unpack_nibbles(packed, per)                   # [N, per]
    q = nib.to(torch.float32) - _FOURBIT_ZERO
    nblocks = scale.shape[1]
    pad = nblocks * block_size - per
    if pad:
        q = torch.cat([q, q.new_zeros(n, pad)], dim=1)
    q = q.view(n, nblocks, block_size).mul_(scale.view(n, nblocks, 1))
    return q.reshape(n, -1)[:, :per]


def fourbit_block_size(grad: Tensor, group: dict[str, Any]) -> int:
    """4-bit absmax block size for ``grad`` under ``group["momentum_4bit_block"]``.

    A non-positive block setting means "one block over the whole tensor"; otherwise
    the block is clamped to the element count (and to 1 for an empty tensor).
    """
    bs = group["momentum_4bit_block"]
    numel = grad.numel()
    return numel if bs <= 0 else min(bs, numel) if numel > 0 else 1


# --------------------------------------------------------------------- codecs


class _StackedViews:
    """One foreach chunk's precomputed view lists for a codec's stacked path.

    The stacked entry points walk lists that are pure functions of tensors the
    optimizer already owns — ``[mat(s["m"]) for s in states]``, the per-row
    ``[s["m_scale"].view(rowshape) ...]``, the write-back targets — and used to rebuild
    them on **every step**, one ``aten::view`` per param per list. This holds them
    once; the caller (a :class:`~kaon._foreach_plan.ForeachChunk`) keeps it alive for
    as long as it keeps its own view caches and hands it back through ``views=``.

    Fields (content is codec-specific; only the owning codec reads them):

    * :attr:`eff` — the effective per-param shape the lists were built for, checked by
      every consumer so a views object from another bucket is *ignored*, not misread.
      That check is **defense in depth**, not a live code path: all four callers pass
      the same ``chunk.eff`` / ``(chunk.length,)`` they built the views with, and
      :meth:`~kaon._foreach_plan.ForeachPlan.rechunk` hands out *new* chunks (with
      empty caches) whenever the split moves, so nothing today can produce a
      mismatched hand-off. It is here because the failure it prevents is silent — the
      layout-bearing codecs would read back another bucket's shape and the write-backs
      would land in another bucket's buffers — and the check is one tuple compare per
      bucket per step.
    * :attr:`m` — the momentum in the effective layout: ``mat(s["m"])`` for the float
      and int8 codecs, the raw packed ``s["m"]`` for 4-bit (it has no such layout).
      Also the write-back target of ``ema_stacked``.
    * :attr:`scale` — the per-row / per-block scale views, ``None`` for the float codec.
    * :attr:`store` — ``store_stacked``'s write-back targets, which for int8 are the
      ``[row, rest]`` views the batched requant produces (the float and 4-bit codecs
      write through :attr:`m`).

    Like every other cache in the foreach plan this pins **no memory**: each entry is a
    *view* of a live ``state["m"]`` / ``state["m_scale"]``. That is also what keeps the
    storage-identity contract intact — writes through these lists land in the buffers
    MSAM/Nekaon cached ``data_ptr`` of. It follows that they go stale exactly when
    those buffers are *replaced* (``load_state_dict``), which is the plan-level
    invalidation the chunk already performs; a codec whose layout cannot be aliased
    declines the cache by returning ``None`` from :meth:`_MomentumCodec.stacked_views`.
    """

    __slots__ = ("eff", "m", "scale", "store")

    def __init__(
        self,
        eff: tuple[int, ...],
        m: list[Tensor],
        scale: list[Tensor] | None = None,
        store: list[Tensor] | None = None,
    ) -> None:
        self.eff = eff
        self.m = m
        self.scale = scale
        self.store = m if store is None else store


def _stacked_cacheable(states: list[dict[str, Any]], scale: bool) -> bool:
    """Whether ``states``' momentum buffers can be aliased by cached views.

    A non-contiguous buffer cannot be ``view``'d into the effective layout (and
    ``reshape`` would hand back a detached COPY, which the stacked write-backs would
    then write into instead of the state) — those states keep the uncached / per-param
    fallbacks the stacked entry points already carry.
    """
    if scale:
        return all(s["m"].is_contiguous() and s["m_scale"].is_contiguous() for s in states)
    return all(s["m"].is_contiguous() for s in states)


class _MomentumCodec:
    """Base momentum codec. Subclasses own one ``momentum_dtype``'s storage AND the
    full dequant -> fp32 EMA -> requant cycle.

    EMA entry points (Adakaon; perform ``m.lerp_(update, 1-beta1)`` and return
    the fp32 first-moment as the step delta):

    * ``ema_one``     — per-param.
    * ``ema_stacked`` — foreach.

    Write entry points (Lion / AdaBelief / AdamP / KProdigy, which do their own
    EMA arithmetic then hand the fp32 result back for storage):

    * ``store_one``     — per-param requant / copy **in place**.
    * ``store_stacked`` — foreach requant / copy **in place**.

    Read-only entry points (KProdigy, which does its own ``d``-scaled EMA in pass
    1 and only needs to *read* the stored momentum back in pass 2):

    * ``dequant_one``     — per-param: return the fp32 momentum (no mutation).
    * ``dequant_stacked`` — foreach: return the stacked fp32 momentum (no mutation).

    **Storage-identity contract.** ``state["m"]`` and (when present)
    ``state["m_scale"]`` keep the same tensor objects for the lifetime of the
    optimizer state. Writers must ``copy_`` / ``_foreach_copy_`` into those
    buffers — never reassign. MSAM (and Nekaon) cache ``data_ptr`` tables into
    both for the fused climb; a reassignment leaves those tables addressing
    freed memory.

    **Cached views.** Every stacked entry point takes an optional ``views``
    (:class:`_StackedViews`, built by :meth:`stacked_views`) so a caller with a cached
    foreach plan can hand over the per-param view lists instead of having the codec
    rebuild them each step. It is a pure host-side optimization: ``views=None`` runs the
    original code, and passing one is bit-identical.
    """

    def init_state(self, state: dict[str, Any], grad: Tensor, group: dict[str, Any]) -> None:
        raise NotImplementedError

    def stacked_views(
        self, states: list[dict[str, Any]], view: Any, eff: tuple[int, ...]
    ) -> _StackedViews | None:
        """Precompute this bucket's stacked-path view lists, or ``None`` to decline.

        ``view`` is the bucket's effective-layout callback (``t.view(R, C)`` for a
        matrixized conv bucket, :func:`~kaon._backend.flat_view` for one that admits
        0-D params, identity otherwise) and ``eff`` the effective per-param shape —
        ``(R, C)`` factored, ``(L,)`` non-factored. Call it **once per chunk**, not per
        step, and hand the result back as ``views=``.

        ``None`` means "no cache for this layout": a buffer that cannot be aliased into
        the effective layout, or an ``eff`` this codec does not batch. The caller just
        keeps passing ``views=None`` and gets the uncached path.
        """
        return None

    def ema_one(self, state: dict[str, Any], update: Tensor, beta1: float) -> Tensor:
        raise NotImplementedError

    def ema_stacked(
        self, states: list[dict[str, Any]], update: Tensor, mat: Any, eff: tuple[int, ...],
        beta1: float, views: _StackedViews | None = None,
    ) -> Tensor:
        raise NotImplementedError

    def store_one(self, state: dict[str, Any], m_fp32: Tensor) -> None:
        """Write an updated fp32 momentum into ``state`` **in place**.

        Preserves the identity of ``state["m"]`` / ``state["m_scale"]`` (see the
        class docstring). ``m_fp32`` may be a matrixized ``[R, C]`` view; it is
        reshaped to the stored layout before quantizing / copying.
        """
        raise NotImplementedError

    def store_stacked(
        self, states: list[dict[str, Any]], m_fp32: Tensor,
        views: _StackedViews | None = None,
    ) -> None:
        """Write stacked fp32 momentum ``[N, *shape]`` into per-param storage in place.

        Same storage-identity contract as :meth:`store_one`: codes and scales are
        ``copy_``'d into the existing tensors, never replaced.
        """
        raise NotImplementedError

    def dequant_one(self, state: dict[str, Any], like: Tensor) -> Tensor:
        """Return the stored momentum as a fresh fp32 tensor shaped like ``like``."""
        raise NotImplementedError

    def dequant_stacked(
        self, states: list[dict[str, Any]], mat: Any, eff: tuple[int, ...],
        views: _StackedViews | None = None,
    ) -> Tensor:
        """Return the stacked fp32 momentum ``[N, *eff]`` (no mutation)."""
        raise NotImplementedError

    def scale_(self, state: dict[str, Any], factor: float) -> None:
        """Multiply the stored first moment in place by a scalar.

        The quantized codecs scale the per-row/block ``m_scale`` (no requant
        error); the float codec scales ``m`` directly for wrapper handoffs.
        """
        raise NotImplementedError


class _FloatCodec(_MomentumCodec):
    """fp32 / bf16 momentum: store ``m`` directly in ``dtype``.

    The EMA always runs in fp32 (matching the fused Triton kernels). For bf16
    storage the result is ``copy_``'d back into the bf16 buffer; previously bf16
    did ``m.lerp_(update.to(bf16), ...)`` which rounded the update *before* the
    EMA and diverged from fused (~3.5e-4 vs ~1e-7). fp32 storage is unchanged
    (EMA already lived in fp32).
    """

    def __init__(self, dtype: torch.dtype) -> None:
        self.dtype = dtype

    def init_state(self, state: dict[str, Any], grad: Tensor, group: dict[str, Any]) -> None:
        state["m"] = torch.zeros_like(grad, dtype=self.dtype)

    def ema_one(self, state: dict[str, Any], update: Tensor, beta1: float) -> Tensor:
        m = state["m"]
        if m.dtype == torch.float32:
            m.lerp_(update, 1.0 - beta1)
            return m.clone()
        # bf16 (or any non-fp32 store): EMA in fp32, round only on write-back.
        m_fp = m.float()
        m_fp.lerp_(update, 1.0 - beta1)
        m.copy_(m_fp)
        return m_fp

    def stacked_views(
        self, states: list[dict[str, Any]], view: Any, eff: tuple[int, ...]
    ) -> _StackedViews | None:
        """``mat(s["m"])`` per param — the read source *and* the write-back target.

        ``store_stacked``'s target is ``s["m"].view(eff)``, the same storage with the
        same shape as ``view(s["m"])``, so one list serves both.
        """
        if not _stacked_cacheable(states, scale=False):
            return None
        return _StackedViews(tuple(eff), [view(s["m"]) for s in states])

    def ema_stacked(
        self, states: list[dict[str, Any]], update: Tensor, mat: Any, eff: tuple[int, ...],
        beta1: float, views: _StackedViews | None = None,
    ) -> Tensor:
        ms = views.m if views is not None and views.eff == eff else [mat(s["m"]) for s in states]
        mom = torch.stack(ms)                                        # [N, …], momentum dtype
        if mom.dtype == torch.float32:
            mom.lerp_(update, 1.0 - beta1)
            torch._foreach_copy_(ms, list(mom.unbind(0)))
            return mom
        mom_fp = mom.float()
        mom_fp.lerp_(update, 1.0 - beta1)
        torch._foreach_copy_(ms, list(mom_fp.unbind(0)))             # rounds to bf16 on write
        return mom_fp

    def store_one(self, state: dict[str, Any], m_fp32: Tensor) -> None:
        state["m"].copy_(m_fp32.reshape(state["m"].shape))

    def store_stacked(
        self, states: list[dict[str, Any]], m_fp32: Tensor,
        views: _StackedViews | None = None,
    ) -> None:
        shape = tuple(m_fp32.shape[1:])
        vals = list(m_fp32.unbind(0))
        if views is not None and views.eff == shape:
            torch._foreach_copy_(views.store, vals)
            return
        # ``reshape`` of a non-contiguous ``m`` returns a COPY; ``_foreach_copy_``
        # would then write the copy and leave state untouched. ``view`` fails loud
        # when the layout cannot alias; otherwise fall back to per-param ``copy_``.
        if all(s["m"].is_contiguous() for s in states):
            torch._foreach_copy_([s["m"].view(shape) for s in states], vals)
        else:
            for s, v in zip(states, vals, strict=True):
                s["m"].copy_(v.reshape_as(s["m"]))

    def dequant_one(self, state: dict[str, Any], like: Tensor) -> Tensor:
        m = state["m"]
        return m.float() if m.dtype != torch.float32 else m.clone()

    def dequant_stacked(
        self, states: list[dict[str, Any]], mat: Any, eff: tuple[int, ...],
        views: _StackedViews | None = None,
    ) -> Tensor:
        ms = views.m if views is not None and views.eff == eff else [mat(s["m"]) for s in states]
        return torch.stack(ms).float()

    def scale_(self, state: dict[str, Any], factor: float) -> None:
        state["m"].mul_(factor)  # in stored dtype; exact for fp32, rounded for bf16


class _Int8Codec(_MomentumCodec):
    """int8 momentum: per-row (dim-0) absmax scale (see :func:`_quant_int8`)."""

    def init_state(self, state: dict[str, Any], grad: Tensor, group: dict[str, Any]) -> None:
        state["m"] = torch.zeros_like(grad, dtype=torch.int8)
        state["m_scale"] = torch.ones(
            (grad.shape[0],) + (1,) * (grad.ndim - 1) if grad.ndim >= 2 else (),
            dtype=torch.float32, device=grad.device,
        )

    def ema_one(self, state: dict[str, Any], update: Tensor, beta1: float) -> Tensor:
        # One fp32 temp (``.float().mul_``) instead of ``.float() * scale`` (two temps).
        m = state["m"].float().mul_(state["m_scale"])                # dequant
        m.lerp_(update, 1.0 - beta1)
        # ``_quant_int8`` does not mutate ``m`` (uses ``/``, not ``div_``); return it
        # as delta and write codes with a non-mutating ``/`` so the EMA value stays.
        dims = tuple(range(1, m.ndim)) if m.ndim >= 2 else ()
        absmax = _absmax(m, dims)
        scale = absmax / _INT8_ABSMAX
        state["m"].copy_((m / scale).round_().clamp_(-_INT8_CLAMP, _INT8_CLAMP))
        state["m_scale"].copy_(scale.reshape_as(state["m_scale"]))
        return m

    def stacked_views(
        self, states: list[dict[str, Any]], view: Any, eff: tuple[int, ...]
    ) -> _StackedViews | None:
        """``mat(s["m"])`` and the per-row scale views, plus the requant's own targets.

        The batched requant reduces a ``[N, row, rest]`` view, so ``store_stacked``
        writes ``[row, rest]`` slices: that is :attr:`~_StackedViews.m`'s own shape for
        a factored bucket but ``[1, L]`` for a non-factored one, hence the separate
        :attr:`~_StackedViews.store` list. The scale views must ALIAS (a ``reshape``
        copy would freeze the scale at its build-time value), which is why a
        non-contiguous ``m_scale`` declines the cache.
        """
        eff = tuple(eff)
        if len(eff) not in (1, 2) or not _stacked_cacheable(states, scale=True):
            return None
        rowshape = (eff[0], 1) if len(eff) == 2 else (1,)
        row, rest = _row_rest(eff)
        return _StackedViews(
            eff,
            [view(s["m"]) for s in states],
            scale=[s["m_scale"].view(rowshape) for s in states],
            store=[s["m"].view(row, rest) for s in states],
        )

    def ema_stacked(
        self, states: list[dict[str, Any]], update: Tensor, mat: Any, eff: tuple[int, ...],
        beta1: float, views: _StackedViews | None = None,
    ) -> Tensor:
        cached = views is not None and views.eff == eff
        if cached:
            ms = views.m
            scale = torch.stack(views.scale)
        else:
            rowshape = (eff[0], 1) if len(eff) == 2 else (1,)
            scale = torch.stack([s["m_scale"].view(*rowshape) for s in states])
            ms = [mat(s["m"]) for s in states]
        m = torch.stack(ms).float().mul_(scale)                      # dequant
        m.lerp_(update, 1.0 - beta1)
        # No clone: ``_quant_int8_stacked`` does not mutate ``m``.
        q, new_scale = _quant_int8_stacked(m)                        # requant
        # ``_quant_int8_stacked`` reduces the trailing axis of ``[N, *eff]``, so
        # ``new_scale`` is already ``[N, *rowshape]`` — the cached scale views' shape.
        torch._foreach_copy_(ms, list(q.unbind(0)))
        if cached:
            torch._foreach_copy_(views.scale, list(new_scale.unbind(0)))
        else:
            for s, sc in zip(states, new_scale.unbind(0), strict=True):
                s["m_scale"].copy_(sc.view_as(s["m_scale"]))
        return m

    def store_one(self, state: dict[str, Any], m_fp32: Tensor) -> None:
        q, scale = _quant_int8(m_fp32.reshape(state["m"].shape))
        state["m"].copy_(q)
        state["m_scale"].copy_(scale.reshape_as(state["m_scale"]))

    def store_stacked(
        self, states: list[dict[str, Any]], m_fp32: Tensor,
        views: _StackedViews | None = None,
    ) -> None:
        n = m_fp32.shape[0]
        shape = tuple(m_fp32.shape[1:])
        row, rest = _row_rest(shape)
        q, new_scale = _quant_int8_stacked(m_fp32.reshape(n, row, rest))
        qs = list(q.unbind(0))
        if views is not None and views.eff == shape:
            rowshape = (row, 1) if len(shape) >= 2 else (1,)
            torch._foreach_copy_(views.store, qs)
            # ``new_scale`` is [N, row, 1]; the cached views carry the per-param scale
            # layout (see int8_scale_shape), which for a flat bucket is [1], not [1, 1].
            torch._foreach_copy_(views.scale, list(new_scale.view((n, *rowshape)).unbind(0)))
            return
        # ``reshape`` of a non-contiguous ndim>2 buffer is a COPY; writing it would
        # leave ``state["m"]`` unchanged. Prefer ``view`` (aliases storage) and fall
        # back to a per-param ``copy_`` into the original shape when needed.
        if all(s["m"].is_contiguous() for s in states):
            torch._foreach_copy_([s["m"].view(row, rest) for s in states], qs)
        else:
            for s, qi in zip(states, qs, strict=True):
                s["m"].copy_(qi.reshape_as(s["m"]))
        # Same layout `_quant_int8` produces per-param (see int8_scale_shape); copy_
        # into the existing scale buffer so MSAM's cached pointers stay valid.
        for s, sc in zip(states, new_scale.unbind(0), strict=True):
            s["m_scale"].copy_(sc.reshape_as(s["m_scale"]))

    def dequant_one(self, state: dict[str, Any], like: Tensor) -> Tensor:
        return state["m"].float().mul_(state["m_scale"])

    def dequant_stacked(
        self, states: list[dict[str, Any]], mat: Any, eff: tuple[int, ...],
        views: _StackedViews | None = None,
    ) -> Tensor:
        if views is not None and views.eff == eff:
            return torch.stack(views.m).float().mul_(torch.stack(views.scale))
        m = torch.stack([mat(s["m"]) for s in states]).float()       # [N, *mview]
        # Per-row int8 scale: leading axis = dim-0 of the *matrixized* momentum,
        # the rest broadcast (1s). For a 1-D / scalar-scale param this is all 1s.
        per_ndim = m.ndim - 1
        rowshape = (m.shape[1],) + (1,) * (per_ndim - 1) if per_ndim >= 2 else (1,) * per_ndim
        scale = torch.stack([s["m_scale"].reshape(rowshape) for s in states])
        return m.mul_(scale)

    def scale_(self, state: dict[str, Any], factor: float) -> None:
        state["m_scale"].mul_(factor)  # dequant = m * m_scale -> scales value exactly


class _FourBitCodec(_MomentumCodec):
    """4-bit momentum: flat per-block absmax + nibble packing (see :func:`_quant_4bit`).

    Scale layout is flat-over-blocks, NOT per-row; the stacked path operates on each
    param's flattened ``[per]`` view so block boundaries match the per-param path.
    """

    _block_size = staticmethod(fourbit_block_size)

    def init_state(self, state: dict[str, Any], grad: Tensor, group: dict[str, Any]) -> None:
        numel = grad.numel()
        bs = self._block_size(grad, group)
        nblocks = (numel + bs - 1) // bs
        # zero momentum -> nibble _FOURBIT_ZERO (the zero level after the +8 shift); a packed
        # byte of two such nibbles is 0x88 = 136. Scales are 1.0 so a fresh dequant returns 0.
        zero_byte = _FOURBIT_ZERO | (_FOURBIT_ZERO << 4)   # 0x88
        state["m"] = torch.full(((numel + 1) // 2,), zero_byte, dtype=torch.uint8, device=grad.device)
        state["m_scale"] = torch.ones(nblocks, dtype=torch.float32, device=grad.device)
        state["m_numel"] = numel
        state["m_block"] = bs

    def ema_one(self, state: dict[str, Any], update: Tensor, beta1: float) -> Tensor:
        bs = state["m_block"]
        # ``_dequant_4bit`` always returns a fresh tensor (never a view of state).
        m = _dequant_4bit(state["m"], state["m_scale"], state["m_numel"], bs)
        m = m.view_as(update)                                        # dequant -> update shape
        m.lerp_(update, 1.0 - beta1)
        # No clone: ``_quant_4bit`` does not mutate ``m``.
        packed, scale, _ = _quant_4bit(m, bs)                        # requant
        state["m"].copy_(packed)
        state["m_scale"].copy_(scale)
        return m

    def stacked_views(
        self, states: list[dict[str, Any]], view: Any, eff: tuple[int, ...]
    ) -> _StackedViews | None:
        """The raw packed buffers and per-block scales — 4-bit has no effective layout.

        ``m`` is a nibble-packed byte string, so ``view`` is never applied (and the
        codec never calls ``mat``); the cache saves the per-step *list* rebuilds and
        turns the per-param scale write-back into one ``_foreach_copy_``. Buckets mix
        no block sizes: the stacked path already reads ``states[0]["m_block"]``.
        """
        if not _stacked_cacheable(states, scale=True):
            return None
        return _StackedViews(
            tuple(eff),
            [s["m"] for s in states],
            scale=[s["m_scale"] for s in states],
        )

    def ema_stacked(
        self, states: list[dict[str, Any]], update: Tensor, mat: Any, eff: tuple[int, ...],
        beta1: float, views: _StackedViews | None = None,
    ) -> Tensor:
        n = update.shape[0]
        per = math.prod(eff)
        bs = states[0]["m_block"]
        cached = views is not None and views.eff == eff
        ms = views.m if cached else [s["m"] for s in states]
        packed = torch.stack(ms)                                    # [N, ceil(per/2)]
        sc = torch.stack(views.scale if cached else [s["m_scale"] for s in states])  # [N, nblk]
        # ``_dequant_4bit_stacked`` returns a fresh tensor (never a view of the state).
        # When ``per`` is not a block multiple the ``[:, :per]`` slice is non-contiguous;
        # do NOT materialise it: ``lerp_`` picks a different kernel on a contiguous copy
        # and the per-param / stacked paths stop agreeing bit-for-bit. Every consumer of
        # the delta is elementwise, so the strided view is fine.
        m = _dequant_4bit_stacked(packed, sc, per, bs).view_as(update)
        m.lerp_(update, 1.0 - beta1)
        new_packed, new_scale = _quant_4bit_stacked(m.reshape(n, per), bs)  # requant
        torch._foreach_copy_(ms, list(new_packed.unbind(0)))
        if cached:
            torch._foreach_copy_(views.scale, list(new_scale.unbind(0)))
        else:
            for s, sc_i in zip(states, new_scale.unbind(0), strict=True):
                s["m_scale"].copy_(sc_i)
        return m

    def store_one(self, state: dict[str, Any], m_fp32: Tensor) -> None:
        packed, scale, _ = _quant_4bit(m_fp32, state["m_block"])
        state["m"].copy_(packed)
        state["m_scale"].copy_(scale)

    def store_stacked(
        self, states: list[dict[str, Any]], m_fp32: Tensor,
        views: _StackedViews | None = None,
    ) -> None:
        n = m_fp32.shape[0]
        shape = tuple(m_fp32.shape[1:])
        per = math.prod(shape) if m_fp32.ndim > 1 else 1
        bs = states[0]["m_block"]
        new_packed, new_scale = _quant_4bit_stacked(m_fp32.reshape(n, per), bs)
        packs = list(new_packed.unbind(0))
        if views is not None and views.eff == shape:
            torch._foreach_copy_(views.store, packs)
            torch._foreach_copy_(views.scale, list(new_scale.unbind(0)))
            return
        # Packed buffers are 1-D; still guard non-contiguous storage the same way.
        if all(s["m"].is_contiguous() for s in states):
            torch._foreach_copy_([s["m"] for s in states], packs)
        else:
            for s, packed in zip(states, packs, strict=True):
                s["m"].copy_(packed)
        for s, sc in zip(states, new_scale.unbind(0), strict=True):
            s["m_scale"].copy_(sc)

    def dequant_one(self, state: dict[str, Any], like: Tensor) -> Tensor:
        bs = state["m_block"]
        m = _dequant_4bit(state["m"], state["m_scale"], state["m_numel"], bs)
        return m.view_as(like)

    def dequant_stacked(
        self, states: list[dict[str, Any]], mat: Any, eff: tuple[int, ...],
        views: _StackedViews | None = None,
    ) -> Tensor:
        per = math.prod(eff)
        bs = states[0]["m_block"]
        cached = views is not None and views.eff == eff
        packed = torch.stack(views.m if cached else [s["m"] for s in states])
        sc = torch.stack(views.scale if cached else [s["m_scale"] for s in states])
        n = packed.shape[0]
        return _dequant_4bit_stacked(packed, sc, per, bs).reshape((n, *eff))

    def scale_(self, state: dict[str, Any], factor: float) -> None:
        state["m_scale"].mul_(factor)  # per-block scale -> scales the value exactly


def _make_codec(momentum_dtype: str) -> _MomentumCodec:
    if momentum_dtype == "int8":
        return _Int8Codec()
    if momentum_dtype == "4bit":
        return _FourBitCodec()
    return _FloatCodec(torch.bfloat16 if momentum_dtype == "bfloat16" else torch.float32)


def warn_if_4bit_high_beta1(beta1: float, momentum_dtype: str) -> None:
    """Warn when 4-bit momentum is paired with a high EMA decay.

    The dequant→EMA→requant loop amplifies quantization error by roughly
    ``1/sqrt(1-beta1**2)``. Measured (block 128, real SDXL-scale grads):

    =======  =======  =====
    beta1    rel-L2   cos
    =======  =======  =====
    0.9      0.25     0.97
    0.95     0.38     —
    0.99     1.50     0.37
    0.999    4.2      —
    =======  =======  =====

    Call from optimizer constructors after beta validation (one helper, not
    duplicated per optimizer). Threshold ``beta1 >= 0.99``.
    """
    if momentum_dtype == "4bit" and beta1 >= 0.99:
        warnings.warn(
            f"momentum_dtype='4bit' with beta1={beta1} amplifies quantization "
            f"error ~1/sqrt(1-beta1^2) "
            f"(measured block-128: beta1=0.99 → rel-L2≈1.50, cos≈0.37; "
            f"beta1=0.999 → rel-L2≈4.2). Prefer int8, or lower beta1.",
            UserWarning,
            stacklevel=3,
        )


def _warn_bf16_method_replaced(before: list[Any], groups: list[dict[str, Any]]) -> None:
    """Warn when a load REPLACED a group's ``bf16_method``.

    ``torch.optim.Optimizer.load_state_dict`` restores every group hyperparameter from the
    checkpoint, ``bf16_method`` included: ``Nekaon(..., bf16_method="kahan8")`` loading a
    stochastic-rounding checkpoint silently trained with SR from then on (no residual, no
    compensation — the audit of 0.7.16 found it). The checkpoint still wins (that is torch's
    contract, and switching back is one assignment), but it no longer happens silently.
    """
    for i, (old, group) in enumerate(zip(before, groups, strict=False)):
        new = group.get("bf16_method")
        if old is not None and new is not None and old != new:
            warnings.warn(
                f"load_state_dict: the checkpoint's bf16_method={new!r} replaced the "
                f"optimizer's {old!r} on param group {i} (every group hyperparameter is "
                f"restored from the checkpoint). To keep {old!r}, set "
                f"optimizer.param_groups[{i}]['bf16_method'] = {old!r} after loading.",
                stacklevel=3,
            )


def load_state_dict_preserving_dtypes(
    optimizer: torch.optim.Optimizer, state_dict: dict[str, Any]
) -> None:
    """Restore optimizer state byte-identical to the checkpoint.

    ``torch.optim.Optimizer.load_state_dict`` casts every *floating* per-param
    state tensor to the *param's* dtype on load. With bf16/fp16 weights that is
    lossy for fp32 buffers (``m_scale``, ``row``, ``col``, ``v``, fp32 ``m``):
    values round through the param dtype (~0.3% relative drift per resume) and a
    later ``.to(saved_dtype)`` cannot recover the bits. Casting alone was enough
    only when params were fp32 (the historical docstring claim).

    Fix: snapshot the checkpoint tensors (by flattened param index + key), run
    the default load, then put the originals back with ``.to(device=p.device)``
    (dtype and values preserved). When the post-load destination already has the
    correct dtype and shape we ``copy_`` into it (no extra allocation; same
    aliasing behaviour as the base loader); otherwise we reassign a fresh clone
    (unavoidable when torch widened/narrowed the dtype). Note that
    ``Optimizer.load_state_dict`` rebuilds every state tensor, so no pointer that
    existed *before* the load survives either way — MSAM/Nekaon drop their cached
    plan on load and re-validate pointers per step.

    State keys may be ``int`` or ``str`` (JSON round-trip drift); they are
    normalised to ``int`` before the torch load (same convention as
    :mod:`kaon._wrappers`).

    It also restores the optimizer's **stochastic-rounding noise stream** (``_sr_meta``),
    because "byte-identical to the checkpoint" has to include *where in the noise* the run
    was: the bf16 SR weight write is seeded from a counter, and a resume that restarts it
    at 0 rounds differently than the run it continues (~4.7e-2 max abs on bf16 weights four
    steps later) with every state tensor perfectly restored. This is the single restore
    point for it — every kaon optimizer's ``load_state_dict`` funnels through here, so an
    optimizer cannot be forgotten; wrappers restore their OWN stream (Lookahead's phi sync)
    in :meth:`kaon._wrappers.WrapsInnerOptimizer._load_wrapped`.
    """
    if isinstance(optimizer, SRSeedState):
        optimizer._sr_load(state_dict)
    saved = state_dict.get("state", {})
    # Snapshot references + normalise int/str keys for torch (JSON may stringify them).
    # No clone here: the ``copy_`` branch below never aliases, and the reassignment
    # branch clones on its own — cloning everything doubled the resume peak.
    saved_tensors: dict[int, dict[str, Tensor]] = {}
    normalized_state: dict[int, Any] = {}
    for idx, s in saved.items():
        i = int(idx)
        normalized_state[i] = s
        saved_tensors[i] = {k: v for k, v in s.items() if torch.is_tensor(v)}

    sd = dict(state_dict)
    sd["state"] = normalized_state
    methods_before = [g.get("bf16_method") for g in optimizer.param_groups]
    torch.optim.Optimizer.load_state_dict(optimizer, sd)
    _warn_bf16_method_replaced(methods_before, optimizer.param_groups)

    params = [p for group in optimizer.param_groups for p in group["params"]]
    for i, p in enumerate(params):
        src = saved_tensors.get(i)
        if src is None or p not in optimizer.state:
            continue
        st = optimizer.state[p]
        for key, src_t in src.items():
            dst = st.get(key)
            if (
                torch.is_tensor(dst)
                and dst.dtype == src_t.dtype
                and dst.shape == src_t.shape
            ):
                dst.copy_(src_t)  # in place: no allocation, exact dtype + values
            else:
                # torch changed the dtype/shape: put back a private copy of the original.
                st[key] = src_t.detach().clone().to(device=p.device)
