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

from typing import Any

import torch
from torch import Tensor

__all__ = [
    "MomentumDtype",
    "_FOURBIT_BLOCK",
    "_MomentumCodec",
    "_FloatCodec",
    "_Int8Codec",
    "_FourBitCodec",
    "_make_codec",
    "load_state_dict_preserving_dtypes",
    "_quant_int8",
    "_quant_int8_stacked",
    "_pack_nibbles",
    "_unpack_nibbles",
    "_quant_4bit",
    "_dequant_4bit",
    "_quant_4bit_stacked",
    "_dequant_4bit_stacked",
]

MomentumDtype = ("bfloat16", "float32", "int8", "4bit")

# Block size (number of consecutive flattened elements sharing one absmax scale)
# for 4-bit momentum. Li et al. ("Memory Efficient Optimizers with 4-bit States",
# NeurIPS 2023, arXiv:2309.01507) found small blocks materially help at 4-bit; a
# fidelity replay on real SDXL gradients here confirmed block 128 ≈ int8 fidelity.
_FOURBIT_BLOCK = 128


def _quant_int8(m_fp32: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize a momentum tensor to int8 with a per-row (dim-0) absmax scale.

    Per-row scaling keeps a single outlier from collapsing the whole tensor's
    resolution (a coarse stand-in for bitsandbytes' block-wise scheme). 1-D
    tensors use a single scalar scale.
    """
    dims = tuple(range(1, m_fp32.ndim)) if m_fp32.ndim >= 2 else ()
    absmax = m_fp32.abs().amax(dim=dims, keepdim=True).clamp_(min=1e-12)
    scale = absmax / 127.0
    q = (m_fp32 / scale).round_().clamp_(-127, 127).to(torch.int8)
    return q, scale


def _quant_int8_stacked(m_fp32: Tensor) -> tuple[Tensor, Tensor]:
    """Batched :func:`_quant_int8` for a stacked momentum tensor.

    ``m_fp32`` is the stacked momentum in its *effective row layout* — either
    ``[N, R, C]`` (factored bucket) or ``[N, L]`` (non-factored bucket).
    Reducing only the trailing axis here is element-for-element the same set of
    values the per-param path reduces per tensor, so the scales match exactly.
    """
    absmax = m_fp32.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-12)
    scale = absmax / 127.0
    q = (m_fp32 / scale).round_().clamp_(-127, 127).to(torch.int8)
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
    absmax = blocks.abs().amax(dim=1, keepdim=True).clamp_(min=1e-12)
    scale = absmax / 7.0
    q = (blocks / scale).round_().clamp_(-7, 7).to(torch.int8)
    nib = (q + 8).to(torch.uint8).reshape(-1)[:numel]
    packed = _pack_nibbles(nib)
    return packed, scale.reshape(nblocks), numel


def _dequant_4bit(packed: Tensor, scale: Tensor, numel: int, block_size: int) -> Tensor:
    """Inverse of :func:`_quant_4bit`: -> flat fp32 of length ``numel``."""
    nib = _unpack_nibbles(packed, numel)
    q = nib.to(torch.float32) - 8.0
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
    absmax = blocks.abs().amax(dim=2, keepdim=True).clamp_(min=1e-12)
    scale = absmax / 7.0
    q = (blocks / scale).round_().clamp_(-7, 7).to(torch.int8)
    nib = (q + 8).to(torch.uint8).reshape(n, -1)[:, :per]
    packed = _pack_nibbles(nib)                          # [N, ceil(per/2)]
    return packed, scale.reshape(n, nblocks)


def _dequant_4bit_stacked(packed: Tensor, scale: Tensor, per: int, block_size: int) -> Tensor:
    """Inverse of :func:`_quant_4bit_stacked`: -> ``[N, per]`` fp32."""
    n = packed.shape[0]
    nib = _unpack_nibbles(packed, per)                   # [N, per]
    q = nib.to(torch.float32) - 8.0
    nblocks = scale.shape[1]
    pad = nblocks * block_size - per
    if pad:
        q = torch.cat([q, q.new_zeros(n, pad)], dim=1)
    q = q.view(n, nblocks, block_size).mul_(scale.view(n, nblocks, 1))
    return q.reshape(n, -1)[:, :per]


# --------------------------------------------------------------------- codecs


class _MomentumCodec:
    """Base momentum codec. Subclasses own one ``momentum_dtype``'s storage AND the
    full dequant -> fp32 EMA -> requant cycle.

    EMA entry points (Adakaon; perform ``m.lerp_(update, 1-beta1)`` and return
    the fp32 first-moment as the step delta):

    * ``ema_one``     — per-param.
    * ``ema_stacked`` — foreach.

    Read-only entry points (KProdigy, which does its own ``d``-scaled EMA in pass
    1 and only needs to *read* the stored momentum back in pass 2):

    * ``dequant_one``     — per-param: return the fp32 momentum (no mutation).
    * ``dequant_stacked`` — foreach: return the stacked fp32 momentum (no mutation).
    """

    def init_state(self, state: dict[str, Any], grad: Tensor, group: dict[str, Any]) -> None:
        raise NotImplementedError

    def ema_one(self, state: dict[str, Any], update: Tensor, beta1: float) -> Tensor:
        raise NotImplementedError

    def ema_stacked(
        self, states: list[dict[str, Any]], update: Tensor, mat: Any, eff: tuple[int, ...], beta1: float
    ) -> Tensor:
        raise NotImplementedError

    def dequant_one(self, state: dict[str, Any], like: Tensor) -> Tensor:
        """Return the stored momentum as a fresh fp32 tensor shaped like ``like``."""
        raise NotImplementedError

    def dequant_stacked(
        self, states: list[dict[str, Any]], mat: Any, eff: tuple[int, ...]
    ) -> Tensor:
        """Return the stacked fp32 momentum ``[N, *eff]`` (no mutation)."""
        raise NotImplementedError

    def scale_(self, state: dict[str, Any], factor: float) -> None:
        """Multiply the stored first moment in place by a scalar.

        The quantized codecs scale the per-row/block ``m_scale`` (no requant
        error); the float codec scales ``m`` directly. Used by Autokaon's freeze
        to fold the discovered LR into the momentum (the EMA is linear in lr).
        """
        raise NotImplementedError


class _FloatCodec(_MomentumCodec):
    """fp32 / bf16 momentum: store ``m`` directly in ``dtype``.

    The EMA runs in the *stored* dtype (``update.to(m.dtype)``) exactly as the
    original code did, so fp32/bf16 stay bit-for-bit identical after the refactor.
    """

    def __init__(self, dtype: torch.dtype) -> None:
        self.dtype = dtype

    def init_state(self, state: dict[str, Any], grad: Tensor, group: dict[str, Any]) -> None:
        state["m"] = torch.zeros_like(grad, dtype=self.dtype)

    def ema_one(self, state: dict[str, Any], update: Tensor, beta1: float) -> Tensor:
        m = state["m"]
        m.lerp_(update.to(m.dtype), 1.0 - beta1)
        return m.float() if m.dtype != torch.float32 else m.clone()

    def ema_stacked(
        self, states: list[dict[str, Any]], update: Tensor, mat: Any, eff: tuple[int, ...], beta1: float
    ) -> Tensor:
        ms = [mat(s["m"]) for s in states]
        mom = torch.stack(ms)                                        # [N, …], momentum dtype
        mom.lerp_(update.to(mom.dtype), 1.0 - beta1)
        torch._foreach_copy_(ms, list(mom.unbind(0)))
        return mom.float()

    def dequant_one(self, state: dict[str, Any], like: Tensor) -> Tensor:
        m = state["m"]
        return m.float() if m.dtype != torch.float32 else m.clone()

    def dequant_stacked(
        self, states: list[dict[str, Any]], mat: Any, eff: tuple[int, ...]
    ) -> Tensor:
        return torch.stack([mat(s["m"]) for s in states]).float()

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
        m = state["m"].float() * state["m_scale"]                    # dequant
        m.lerp_(update, 1.0 - beta1)
        delta = m.clone()
        state["m"], state["m_scale"] = _quant_int8(m)                # requant
        return delta

    def ema_stacked(
        self, states: list[dict[str, Any]], update: Tensor, mat: Any, eff: tuple[int, ...], beta1: float
    ) -> Tensor:
        rowshape = (eff[0], 1) if len(eff) == 2 else (1,)
        scale = torch.stack([s["m_scale"].view(*rowshape) for s in states])
        m = torch.stack([mat(s["m"]) for s in states]).float().mul_(scale)  # dequant
        m.lerp_(update, 1.0 - beta1)
        delta = m.clone()
        q, new_scale = _quant_int8_stacked(m)                        # requant
        torch._foreach_copy_([mat(s["m"]) for s in states], list(q.unbind(0)))
        for s, sc in zip(states, new_scale.unbind(0), strict=True):
            s["m_scale"].copy_(sc.view_as(s["m_scale"]))
        return delta

    def dequant_one(self, state: dict[str, Any], like: Tensor) -> Tensor:
        return state["m"].float().mul_(state["m_scale"])

    def dequant_stacked(
        self, states: list[dict[str, Any]], mat: Any, eff: tuple[int, ...]
    ) -> Tensor:
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

    @staticmethod
    def _block_size(grad: Tensor, group: dict[str, Any]) -> int:
        bs = group["momentum_4bit_block"]
        numel = grad.numel()
        return numel if bs <= 0 else min(bs, numel) if numel > 0 else 1

    def init_state(self, state: dict[str, Any], grad: Tensor, group: dict[str, Any]) -> None:
        numel = grad.numel()
        bs = self._block_size(grad, group)
        nblocks = (numel + bs - 1) // bs
        # zero momentum -> nibble 8 (the zero level after the +8 shift); a packed
        # byte of two 8-nibbles is 0x88 = 136. Scales are 1.0 so a fresh dequant
        # returns exactly 0.
        state["m"] = torch.full(((numel + 1) // 2,), 0x88, dtype=torch.uint8, device=grad.device)
        state["m_scale"] = torch.ones(nblocks, dtype=torch.float32, device=grad.device)
        state["m_numel"] = numel
        state["m_block"] = bs

    def ema_one(self, state: dict[str, Any], update: Tensor, beta1: float) -> Tensor:
        bs = state["m_block"]
        m = _dequant_4bit(state["m"], state["m_scale"], state["m_numel"], bs)
        m = m.view_as(update)                                        # dequant -> update shape
        m.lerp_(update, 1.0 - beta1)
        delta = m.clone()
        packed, scale, _ = _quant_4bit(m, bs)                        # requant
        state["m"], state["m_scale"] = packed, scale
        return delta

    def ema_stacked(
        self, states: list[dict[str, Any]], update: Tensor, mat: Any, eff: tuple[int, ...], beta1: float
    ) -> Tensor:
        n = update.shape[0]
        per = 1
        for d in eff:
            per *= d
        bs = states[0]["m_block"]
        packed = torch.stack([s["m"] for s in states])              # [N, ceil(per/2)]
        sc = torch.stack([s["m_scale"] for s in states])            # [N, nblocks]
        m = _dequant_4bit_stacked(packed, sc, per, bs).view_as(update)  # dequant
        m.lerp_(update, 1.0 - beta1)
        delta = m.clone()
        new_packed, new_scale = _quant_4bit_stacked(m.reshape(n, per), bs)  # requant
        torch._foreach_copy_([s["m"] for s in states], list(new_packed.unbind(0)))
        for s, sc in zip(states, new_scale.unbind(0), strict=True):
            s["m_scale"].copy_(sc)
        return delta

    def dequant_one(self, state: dict[str, Any], like: Tensor) -> Tensor:
        bs = state["m_block"]
        m = _dequant_4bit(state["m"], state["m_scale"], state["m_numel"], bs)
        return m.view_as(like)

    def dequant_stacked(
        self, states: list[dict[str, Any]], mat: Any, eff: tuple[int, ...]
    ) -> Tensor:
        per = 1
        for d in eff:
            per *= d
        bs = states[0]["m_block"]
        packed = torch.stack([s["m"] for s in states])
        sc = torch.stack([s["m_scale"] for s in states])
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


def load_state_dict_preserving_dtypes(
    optimizer: torch.optim.Optimizer, state_dict: dict[str, Any]
) -> None:
    """Restore optimizer state WITHOUT torch's lossy momentum upcast.

    ``torch.optim.Optimizer.load_state_dict`` casts every per-param state tensor
    to the *param's* dtype (fp32) on load. For kaon's quantized first moment
    that silently inflates bf16/int8/4bit momentum back to fp32 on resume —
    discarding the memory-efficient representation the user configured (e.g. int8
    -> fp32 is 4x the momentum bytes, defeating the point) AND breaking bit-exact
    resume. We snapshot the stored per-tensor dtypes, run the default load, then
    cast each state tensor back to how it was saved: bf16->fp32->bf16 is exact,
    and the int8/uint8 *codes* round-trip through fp32 exactly, so the resumed
    state is byte-identical to the checkpoint.

    torch numbers params ``0..N-1`` in flattened ``param_groups`` order and the
    state keys are those same ids, so the stored dtype for the param at flattened
    position ``i`` is ``saved["state"][i]`` (load_state_dict already required the
    structures to match).
    """
    saved = state_dict.get("state", {})
    saved_dtypes = {
        idx: {k: v.dtype for k, v in s.items() if torch.is_tensor(v)}
        for idx, s in saved.items()
    }
    torch.optim.Optimizer.load_state_dict(optimizer, state_dict)
    params = [p for group in optimizer.param_groups for p in group["params"]]
    for i, p in enumerate(params):
        dtypes = saved_dtypes.get(i)
        if dtypes is None or p not in optimizer.state:
            continue
        st = optimizer.state[p]
        for key, dtype in dtypes.items():
            t = st.get(key)
            if torch.is_tensor(t) and t.dtype != dtype:
                st[key] = t.to(dtype)
