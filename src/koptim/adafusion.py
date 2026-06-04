"""Adafusion — a conv-aware factored optimizer aimed at AdamW quality at
Adafactor memory, for bf16 diffusion fine-tuning.

Design (validated by benchmarks/bench_convergence-style experiments):

* **Conv-aware factored second moment.** Like Adafactor/Compactor, the second
  moment of a 2-D weight is factored into row+column EMAs (≈0 state). The fix
  over Compactor/HF-Adafactor: a 4-D conv kernel ``[out,in,kh,kw]`` is first
  **reshaped to ``[out, in·kh·kw]``** and factored over *that* matrix — instead
  of factoring the tiny spatial dims, which barely compresses a 3×3 kernel and
  was the entire optimizer-state floor on a diffusion UNet (≈26× more conv state
  for no quality gain).
* **Optional momentum in bf16.** A first-moment buffer recovers AdamW-level
  convergence; kept in bf16 it costs ~2 B/param (half of fp32 momentum) with no
  measured quality loss → AdamW-quality at ~1/4 of AdamW's optimizer memory.
* **bf16-correct weight updates** via stochastic rounding (no extra state) or
  Kahan summation.
* **Optional cautious masking** (Liang et al. 2024): zero the update coordinates
  whose sign disagrees with the gradient, renormalized to keep the step size.
  Off by default — it is a regularizer (helps generalization on noisy training),
  not a training-loss-speed booster.

It is a standard ``torch.optim.Optimizer`` with a single per-parameter step, so
it drops into per-parameter / gradient-release training loops unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, Literal

import torch
from torch import Tensor
from torch.optim import Optimizer

from koptim._factored import factored_inv_sqrt_factors, update_factored_state
from koptim._stochastic_rounding import add_stochastic_

__all__ = ["Adafusion"]

_LOW_PRECISION = (torch.bfloat16, torch.float16)
MomentumDtype = Literal["bfloat16", "float32", "int8", "4bit"]

# Block size (number of consecutive flattened elements sharing one absmax scale)
# for 4-bit momentum. Li et al. ("Memory Efficient Optimizers with 4-bit States",
# NeurIPS 2023, arXiv:2309.01507) found small blocks materially help at 4-bit; a
# fidelity replay on real SDXL gradients here confirmed block 128 ≈ int8 fidelity.
_FOURBIT_BLOCK = 128

# Stacking a foreach bucket allocates several transient copies of the stacked
# tensor (grad fp32, the reconstruction, the SR intermediate, ...), so an unbounded
# bucket of large weights can OOM a full fine-tune — which would undercut
# Adafusion's whole memory story. We therefore cap the per-chunk element count and
# split bigger buckets. The cap is **adaptive to free VRAM** rather than a fixed
# constant: a card with lots of headroom batches whole buckets (and even stacks
# large weights), while a constrained card shrinks the chunk and stays safe. The
# budget is `free_bytes * SAFETY_FRACTION / BYTES_PER_ELEM`; the divisor accounts
# for the ~handful of simultaneous transient copies a chunk touches at peak.
_STACK_SAFETY_FRACTION = 0.10   # use at most ~10% of currently-free VRAM per chunk
# Peak transient bytes per stacked element. This is a property of the optimizer's
# intermediate tensors, NOT of the model: measured byte-for-byte identical on SDXL
# and Cosmos shapes. It depends only on the path and config — 2-D factored 24 B
# (common) / 38 B (momentum+wd+cautious), 1-D non-factored 28 B / 42 B (it also
# stacks the full second-moment state). 48 = worst measured (42.1) + margin.
_STACK_BYTES_PER_ELEM = 48
_MIN_STACK_ELEMS = 262_144      # still batch small tensors even under memory pressure
_DEFAULT_STACK_ELEMS = 64_000_000  # CPU / unknown device: no VRAM limit to respect

# Per-tensor element count above which a weight is stepped by the per-parameter
# loop instead of being stacked. This is a PERFORMANCE threshold, deliberately
# decoupled from the VRAM-safety budget: batching pays off only while per-tensor
# kernel-launch overhead dominates (small tensors); a large weight's update is
# compute/bandwidth-bound, so stacking it just adds copy traffic and is slower.
# A budget sweep on SDXL and Cosmos full fine-tunes showed a broad flat optimum
# for cutoffs of ~0.1-4 M elements and a sharp slowdown beyond ~4 M, on both
# models — i.e. the crossover is an absolute element count, NOT a fraction of
# VRAM (so it must not scale with the card). 2 M sits in the middle of that
# plateau. See docs/foreach-batching.md.
_FOREACH_BATCH_CUTOFF = 2_000_000


def _rms(t: Tensor) -> Tensor:
    return t.norm(2) / math.sqrt(max(t.numel(), 1))


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
    ``[N, R, C]`` (factored bucket) or ``[N, L]`` (non-factored bucket). The
    per-param int8 path computes a per-row absmax scale that reduces every
    dimension *except* dim-0 of the original tensor:

    * factored: original ``[R, C]`` (or matrixized conv) -> scale ``[R, 1]``,
      so the stacked scale is ``[N, R, 1]`` (reduce only the last, ``C``, axis);
    * non-factored: original ``[L]`` is 1-D -> a single scalar scale per param,
      so the stacked scale is ``[N, 1]`` (reduce the whole ``L`` axis).

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

    The tensor is flattened, split into consecutive blocks of ``block_size``
    elements (the final block may be short), and each block gets one ``absmax/7``
    scale. Values map to ``[-7, 7]`` (15 used levels), are shifted to the unsigned
    nibble range ``[1, 15]``, and packed two-per-byte for a real 0.5 B/param store.

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
# The first-moment EMA is always *worked on* as an fp32 tensor in the "effective"
# layout — matricized ``[R, C]`` (factored) / flat ``[L]`` (non-factored) per
# param, and stacked ``[N, R, C]`` / ``[N, L]`` in the foreach buckets. A momentum
# codec owns, for one ``momentum_dtype``, the entire dequant → fp32 EMA → requant
# cycle so the three step functions never re-implement it. The whole momentum
# block in each step function collapses to ``delta = codec.ema_*(…, update, beta1)``:
#
#   * ``init_state``   — allocate the stored representation in ``state``.
#   * ``ema_one``      — per-param: dequant ``state`` -> EMA with ``update`` (grad
#                        shape) -> requant + store; returns the fp32 delta.
#   * ``ema_stacked``  — foreach: dequant a list of states -> EMA with the stacked
#                        ``update`` (``[N,…]``) -> requant + store; returns the
#                        stacked fp32 delta.
#
# ``ema_stacked`` takes ``mat`` (the per-param matricizer the factored bucket uses;
# identity for the non-factored bucket) and ``eff`` (the effective per-param shape
# ``(R, C)`` or ``(L,)``) so each codec can present whatever stacked layout it needs
# (per-row for int8, flat-over-blocks for 4bit) while the bucket stays dtype-agnostic.


class _MomentumCodec:
    """Base momentum codec. Subclasses own one ``momentum_dtype``'s storage AND the
    full dequant → fp32 EMA → requant cycle, so the three step functions never
    re-implement any of it. Both ``ema_*`` methods perform the in-place EMA
    ``m.lerp_(update, 1-beta1)`` and return the fp32 first-moment as the step delta.
    """

    def init_state(self, state: dict[str, Any], grad: Tensor, group: dict[str, Any]) -> None:
        raise NotImplementedError

    def ema_one(self, state: dict[str, Any], update: Tensor, beta1: float) -> Tensor:
        """Per-param: EMA the stored momentum with ``update`` (the param's grad
        shape) and return the fp32 delta, storing the new momentum back."""
        raise NotImplementedError

    def ema_stacked(
        self, states: list[dict[str, Any]], update: Tensor, mat: Any, eff: tuple[int, ...], beta1: float
    ) -> Tensor:
        """Stacked: EMA each state's momentum with the matching slice of ``update``
        (``[N, R, C]`` / ``[N, L]``) and return the fp32 delta, storing back."""
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
        # The per-row absmax scale of the stacked [N, R, C]/[N, L] momentum (reduce
        # only the trailing axis) equals each param's per-param scale. Stacked scale
        # layout: factored [R,1] -> [N,R,1]; non-factored scalar -> [N,1].
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


def _make_codec(momentum_dtype: str) -> _MomentumCodec:
    if momentum_dtype == "int8":
        return _Int8Codec()
    if momentum_dtype == "4bit":
        return _FourBitCodec()
    return _FloatCodec(torch.bfloat16 if momentum_dtype == "bfloat16" else torch.float32)


class Adafusion(Optimizer):
    """Conv-aware factored optimizer with optional bf16 momentum.

    Args:
        params: parameters or param-group dicts.
        lr: learning rate.
        betas: ``(beta1, beta2)``. ``beta1=0`` disables momentum (minimum memory,
            Adafactor-like). ``beta1>0`` enables momentum (AdamW-like quality).
            ``betas[1]`` is ignored when ``decay_rate`` is set.
        eps: ``(eps1, eps2)``. ``eps1`` is added to ``grad**2`` before the
            factored reductions (HF Adafactor convention). ``eps2`` is currently
            unused (reserved).
        weight_decay: decoupled weight decay (folded into the per-step delta).
        clip_threshold: Adafactor RMS update clipping (``rms(update) <= thr``).
        decay_rate: HF Adafactor adaptive ``beta2_t = 1 - step**decay_rate``
            (typical ``-0.8``); ``betas[1]`` ignored when set.
        momentum_dtype: storage for the first-moment buffer when ``beta1>0`` —
            ``"bfloat16"`` (default; ~2 B/param), ``"float32"`` (4 B/param),
            ``"int8"`` (~1 B/param, per-row absmax quantized; Lion8bit-class
            memory but with the factored adaptive second moment), or ``"4bit"``
            (~0.5 B/param: signed linear 4-bit, two nibbles per byte, with a
            per-block absmax scale — block size ``momentum_4bit_block``). On real
            SDXL gradients block-128 4-bit matched int8's delta cosine vs fp32.
        momentum_4bit_block: block size (consecutive flattened elements sharing one
            absmax scale) for ``momentum_dtype="4bit"``. Default ``128``. Smaller
            blocks raise fidelity at the cost of more scale bytes
            (``4/block`` B/param); ``128`` adds ~0.03 B/param for a ~0.53 B/param
            total. ``0``/negative means whole-tensor (single scale).
        cautious: enable cautious masking (off by default; opt-in regularizer).
        bf16_method: weight-update strategy for low-precision params —
            ``"stochastic_rounding"`` (default), ``"kahan"`` (+2 B/param), or
            ``"none"``. No-op on fp32 params.
        factor_conv_as_matrix: deprecated and ignored. Conv-aware factoring
            (reshaping a 4-D conv kernel ``[out,in,kh,kw]`` to 2-D
            ``[out, in·kh·kw]`` before factoring the second moment) is now always
            on — the legacy ``False`` last-dims path left ~0.4 B/param of conv
            state for no quality gain and was removed. The argument is still
            accepted (so existing configs don't break) but has no effect.
        compile: deprecated and ignored. ``torch.compile`` of the per-tensor
            factored core measured neutral-to-negative across model sizes and is
            superseded by ``foreach`` batching; the argument is still accepted (so
            existing configs don't break) but has no effect.
        foreach: batch the step across parameters with multi-tensor (stacked) ops
            instead of a per-parameter Python loop. Default ``True``. Huge win when
            many tensors are stepped at once — LoRA/LoKr adapters (hundreds of tiny
            2-D tensors) *and* full fine-tunes (thousands of weights incl. all the
            1-D biases/norms). Params are bucketed by shape and each bucket steps
            as a few stacked kernels: ``ndim >= 2`` factored ``[N, R, C]``, ``ndim
            == 1`` non-factored ``[N, L]``. Matches the per-parameter path
            numerically (stochastic-rounding draws differ, unbiased either way);
            int8 momentum is also batched (per-row absmax dequant/EMA/requant on
            the stacked layout), as is 4bit (per-block absmax, packed nibbles). The
            rest (0-D scalars, kahan, fp16+SR,
            non-contiguous matrixized convs, single-param groups) transparently
            falls back to it. Set ``False`` to force the per-parameter path.
        foreach_batch_cutoff: per-tensor element count above which a weight is
            stepped by the per-parameter loop instead of being stacked. A
            **performance** knob, decoupled from VRAM: batching only pays off while
            launch overhead dominates (small tensors), so large weights loop. The
            default ``2_000_000`` is the middle of a flat optimum measured on SDXL
            and Cosmos full fine-tunes; raise it only if profiling your GPU shows a
            higher crossover. See ``docs/foreach-batching.md``.
        foreach_stack_budget: the **memory-safety** ceiling — max elements in a
            single stacked ``foreach`` chunk. ``None`` (default) adapts to
            currently-free VRAM each step (roomy card → bigger chunks, full card →
            smaller, OOM-safe). Pass an int to pin a fixed cap (reproducibility, or
            a hard ceiling on a shared GPU). Decoupled from ``foreach_batch_cutoff``
            so raising it never pulls large weights into stacking.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: tuple[float, float] = (1e-30, 1e-3),
        weight_decay: float = 0.0,
        *,
        clip_threshold: float = 1.0,
        decay_rate: float | None = None,
        momentum_dtype: MomentumDtype = "bfloat16",
        momentum_4bit_block: int = _FOURBIT_BLOCK,
        cautious: bool = False,
        bf16_method: str = "stochastic_rounding",
        factor_conv_as_matrix: bool = True,
        compile: bool = False,
        foreach: bool = True,
        foreach_batch_cutoff: int = _FOREACH_BATCH_CUTOFF,
        foreach_stack_budget: int | None = None,
    ) -> None:
        beta1, beta2 = float(betas[0]), float(betas[1])
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"betas[0] must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"betas[1] must be in [0, 1), got {beta2}")
        if lr < 0.0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if clip_threshold <= 0.0:
            raise ValueError(f"clip_threshold must be > 0, got {clip_threshold}")
        if momentum_dtype not in ("bfloat16", "float32", "int8", "4bit"):
            raise ValueError(
                f"momentum_dtype must be bfloat16/float32/int8/4bit, got {momentum_dtype!r}"
            )
        if bf16_method not in ("stochastic_rounding", "kahan", "none"):
            raise ValueError(f"bf16_method must be stochastic_rounding/kahan/none, got {bf16_method!r}")
        if foreach_batch_cutoff < 1:
            raise ValueError(f"foreach_batch_cutoff must be >= 1, got {foreach_batch_cutoff}")
        defaults = {
            "lr": lr,
            "betas": (beta1, beta2),
            "eps": (float(eps[0]), float(eps[1])),
            "weight_decay": weight_decay,
            "clip_threshold": clip_threshold,
            "decay_rate": decay_rate,
            "momentum_dtype": momentum_dtype,
            "momentum_4bit_block": momentum_4bit_block,
            "cautious": cautious,
            "bf16_method": bf16_method,
            "factor_conv_as_matrix": factor_conv_as_matrix,
        }
        super().__init__(params, defaults)
        # Multi-tensor (foreach) batching of the factored fast path. Collapses the
        # per-parameter Python loop + per-tensor kernel launches into a handful of
        # stacked-tensor ops per (shape, dtype) bucket — the decisive win when many
        # small weights are trained at once (LoRA/LoKr adapters). Numerically
        # matches the per-parameter path; stochastic-rounding draws differ
        # (unbiased either way). Anything it doesn't cover falls back per-param.
        self._foreach = foreach
        # Performance cutoff: weights larger than this loop instead of stacking
        # (batching only helps while launch overhead dominates). Decoupled from
        # the VRAM-safety chunk budget below.
        self._foreach_batch_cutoff = foreach_batch_cutoff
        # Memory-safety ceiling: max elements per stacked chunk. None -> adaptive
        # to free VRAM (see _foreach_budget); an int forces a fixed cap.
        self._foreach_stack_budget = foreach_stack_budget
        # One momentum codec per dtype string (the codec is stateless beyond the
        # dtype). Encapsulates every dequant→EMA→requant detail so the three step
        # functions stay dtype-agnostic.
        self._codecs: dict[str, _MomentumCodec] = {}

    def _codec(self, group: dict[str, Any]) -> _MomentumCodec:
        md = group["momentum_dtype"]
        codec = self._codecs.get(md)
        if codec is None:
            codec = self._codecs[md] = _make_codec(md)
        return codec

    @torch.no_grad()
    def _init_state(self, p: Tensor, state: dict[str, Any], group: dict[str, Any]) -> None:
        state["step"] = 0
        grad = p.grad
        factored = p.ndim >= 2
        if factored:
            # ndim==2 is already its own matrix; ndim>2 (conv) reshapes to
            # [out, in·kh·kw] before factoring (the conv-aware fix, always on).
            gv = grad if p.ndim == 2 else grad.reshape(grad.shape[0], -1)
            row_shape = gv.shape[:-1]
            col_shape = gv.shape[:-2] + gv.shape[-1:]
            state["row"] = torch.zeros(row_shape, dtype=torch.float32, device=p.device)
            state["col"] = torch.zeros(col_shape, dtype=torch.float32, device=p.device)
        else:
            state["v"] = torch.zeros_like(grad, dtype=torch.float32)
        if group["betas"][0] > 0:
            self._codec(group).init_state(state, grad, group)
        if _is_low_precision(p) and group["bf16_method"] == "kahan":
            state["shift"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            for p in params:
                if p.grad.is_sparse:
                    raise RuntimeError("Adafusion does not support sparse gradients")
            if self._foreach and self._group_foreach_eligible(group):
                chunk_budget = self._foreach_budget(params[0].device)
                # Effective cutoff = the performance threshold, lowered only if the
                # memory budget can't fit two of a tensor in a chunk (so batching
                # would be a wasteful stack-of-1). Roomy card -> cutoff wins;
                # constrained card -> safety wins.
                cutoff = min(self._foreach_batch_cutoff, chunk_budget // 2)
                fast: list[Tensor] = []
                slow: list[Tensor] = []
                for p in params:
                    (fast if self._param_foreach_eligible(p, group, cutoff) else slow).append(p)
                if len(fast) >= 2:
                    self._step_foreach(fast, group, chunk_budget)
                    for p in slow:
                        self._step_one_param(p, group)
                else:
                    for p in params:
                        self._step_one_param(p, group)
            else:
                for p in params:
                    self._step_one_param(p, group)
        return loss

    # ----------------------------------------------------------------- foreach
    def _foreach_budget(self, device: torch.device) -> int:
        """Max elements per stacked chunk.

        An explicit ``foreach_stack_budget`` int is returned verbatim. Otherwise the
        chunk is ``min(adaptive_to_free_VRAM, 4 * batch_cutoff)``:

        * the VRAM term shrinks the chunk when a big model already fills the card
          (OOM safety) and grows it on a roomy card;
        * the ``4 * batch_cutoff`` cap stops over-stacking — beyond a few
          cutoff-sized tensors, stacking medium weights just adds copy bandwidth and
          is slower (measured on full FT). Tying the cap to the cutoff keeps a single
          performance knob, and a roomy card no longer over-stacks.
        """
        if self._foreach_stack_budget is not None:
            return self._foreach_stack_budget
        cap = 4 * self._foreach_batch_cutoff
        if device.type == "cuda":
            free_bytes = torch.cuda.mem_get_info(device)[0]
            adaptive = int(free_bytes * _STACK_SAFETY_FRACTION / _STACK_BYTES_PER_ELEM)
            return max(_MIN_STACK_ELEMS, min(adaptive, cap))
        return min(_DEFAULT_STACK_ELEMS, cap)

    @staticmethod
    def _group_foreach_eligible(group: dict[str, Any]) -> bool:
        """Group-level options the batched fast path supports.

        ``decay_rate`` (adaptive beta2) is supported: the buckets compute a
        per-param step-dependent beta2 vector, so it goes through foreach too.
        """
        return (
            group["clip_threshold"] > 0          # clip always applied in the batched path
            and group["bf16_method"] != "kahan"  # kahan needs a per-param shift buffer
        )

    @staticmethod
    def _param_foreach_eligible(p: Tensor, group: dict[str, Any], cutoff: int) -> bool:
        """Per-parameter shapes/dtypes the batched fast path can stack.

        Both branches are covered: ``ndim >= 2`` uses the factored bucket, ``ndim
        == 1`` (biases/norms — the bulk of a full fine-tune) uses the non-factored
        bucket. Only 0-D scalars and the awkward dtype/contiguity cases fall back.

        ``cutoff`` is the effective per-tensor size limit (performance threshold,
        possibly lowered by the memory budget) — bigger weights loop.
        """
        if p.ndim == 0:                          # 0-D scalars -> per-param path
            return False
        if p.numel() > cutoff:
            # Compute/bandwidth-bound: the per-tensor launch overhead is noise for
            # it, so looping is as fast and skips the stack/copy traffic.
            return False
        if (
            group["bf16_method"] == "stochastic_rounding"
            and _is_low_precision(p)
            and p.dtype != torch.bfloat16        # fp16+SR is unsupported -> per-param (raises)
        ):
            return False
        if p.ndim > 2:
            # Matrixized conv writes back through a reshaped view -> needs contiguity.
            return p.data.is_contiguous() and p.grad.is_contiguous()
        return True

    @torch.no_grad()
    def _step_foreach(self, params: list[Tensor], group: dict[str, Any], budget: int) -> None:
        """Batched step for many params at once.

        Params are bucketed so each bucket can be stacked into a single tensor and
        stepped with a handful of kernels — element-for-element the same math as
        :meth:`_step_one_param`:

        * ``ndim >= 2`` -> factored bucket, keyed by effective 2-D shape ``[N, R, C]``.
        * ``ndim == 1`` (biases/norms) -> non-factored bucket, keyed by length ``[N, L]``.
        """
        beta1, beta2 = group["betas"]
        eps1, _eps2 = group["eps"]
        lr, clip = group["lr"], group["clip_threshold"]
        wd = group["weight_decay"]
        decay_rate = group["decay_rate"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]
        codec = self._codec(group)

        factored_buckets: dict[tuple[Any, ...], list[Tensor]] = {}
        flat_buckets: dict[tuple[Any, ...], list[Tensor]] = {}
        for p in params:
            state = self.state[p]
            if "step" not in state:
                self._init_state(p, state, group)
            state["step"] += 1
            g = p.grad
            if g.ndim >= 2:
                matrixize = g.ndim > 2  # conv kernels always reshape to 2-D before factoring
                eff = (g.shape[0], g.numel() // g.shape[0]) if matrixize else tuple(g.shape)
                factored_buckets.setdefault((eff, p.dtype, matrixize), []).append(p)
            else:  # ndim == 1
                flat_buckets.setdefault((g.shape[0], p.dtype), []).append(p)

        for (eff, _dtype, matrixize), plist in factored_buckets.items():
            step = max(1, budget // max(eff[0] * eff[1], 1))
            for i in range(0, len(plist), step):
                self._factored_bucket(
                    plist[i:i + step], eff, matrixize,
                    beta1, beta2, eps1, lr, clip, wd, cautious, bf16_method, codec, decay_rate,
                )
        for (length, _dtype), plist in flat_buckets.items():
            step = max(1, budget // max(length, 1))
            for i in range(0, len(plist), step):
                self._nonfactored_bucket(
                    plist[i:i + step], length,
                    beta1, beta2, eps1, lr, clip, wd, cautious, bf16_method, codec, decay_rate,
                )

    def _one_minus_beta2_vec(
        self, plist: list[Tensor], beta2: float, decay_rate: float | None
    ) -> float | Tensor:
        """The ``1 - beta2`` lerp weight for a bucket's second-moment EMA.

        Fixed beta2 -> the scalar ``1 - beta2`` (unchanged behaviour). Adaptive
        beta2 (``decay_rate`` set) -> a per-param fp32 vector ``[N]`` of ``1 -
        beta2_t``, where ``step`` is each param-state's own (already-incremented)
        counter. A vector (not a scalar from a shared step) stays correct if grads
        were skipped and steps desynced.

        Bit-exact-parity note: the per-param path computes ``beta2_t = 1.0 -
        step**decay_rate`` (a Python float) and then lerps with ``1.0 - beta2_t``.
        ``1.0 - (1.0 - x)`` is not generally ``x`` in floating point, so we replay
        the *same* two-step round-trip here (per element, as Python floats) rather
        than the algebraically-simplified ``step**decay_rate``.
        """
        if decay_rate is None:
            return 1.0 - beta2
        omb = [
            1.0 - (1.0 - float(self.state[p]["step"]) ** decay_rate) for p in plist
        ]
        return torch.tensor(omb, dtype=torch.float32, device=plist[0].grad.device)

    @torch.no_grad()
    def _factored_bucket(
        self,
        plist: list[Tensor],
        eff: tuple[int, int],
        matrixize: bool,
        beta1: float,
        beta2: float,
        eps1: float,
        lr: float,
        clip: float,
        wd: float,
        cautious: bool,
        bf16_method: str,
        codec: _MomentumCodec,
        decay_rate: float | None = None,
    ) -> None:
        R, C = eff  # noqa: N806 — matrix dims (stacked tensor is [N, R, C])
        N = len(plist)  # noqa: N806

        def mat(t: Tensor) -> Tensor:
            return t.view(R, C) if matrixize else t

        rows = [self.state[p]["row"] for p in plist]
        cols = [self.state[p]["col"] for p in plist]

        grad = torch.stack([mat(p.grad).float() for p in plist])          # [N, R, C]
        row = torch.stack(rows)                                           # [N, R]
        col = torch.stack(cols)                                           # [N, C]

        # Second-moment EMA weight. Fixed beta2 is a scalar; adaptive (decay_rate)
        # is a per-param vector beta2_t = 1 - step**decay_rate computed from each
        # param's own step counter (params normally share a step, but a vector is
        # the safe form against grad=None desync). 1 - beta2 broadcasts as [N,1,1].
        one_minus_beta2 = self._one_minus_beta2_vec(plist, beta2, decay_rate)
        # row/col are [N, R]/[N, C], so a per-param weight broadcasts as [N, 1].
        omb = one_minus_beta2 if isinstance(one_minus_beta2, float) else one_minus_beta2.view(N, 1)

        # Factored second-moment EMA (HF eps placement: eps1 before the means).
        grad_sq = grad * grad
        if eps1 > 0:
            grad_sq = grad_sq.add_(eps1)
        row.lerp_(grad_sq.mean(dim=-1), omb)
        col.lerp_(grad_sq.mean(dim=-2), omb)
        torch._foreach_copy_(rows, list(row.unbind(0)))
        torch._foreach_copy_(cols, list(col.unbind(0)))

        # Reconstruct 1/sqrt(v_hat) = r_factor * c_factor, then clip and scale.
        r_factor = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)  # [N, R, 1]
        c_factor = col.rsqrt().unsqueeze(-2)                                       # [N, 1, C]
        update = grad.mul(r_factor).mul_(c_factor)                                 # [N, R, C]
        rms = update.reshape(N, -1).norm(2, dim=1) / math.sqrt(R * C)              # per-slice RMS
        update.div_(rms.div_(clip).clamp_(min=1.0).view(N, 1, 1))
        update.mul_(lr)

        if beta1 > 0:
            # The codec owns every dtype's dequant → fp32 EMA → requant detail; this
            # block is identical for fp32/bf16/int8/4bit (and to _step_one_param).
            states = [self.state[p] for p in plist]
            delta = codec.ema_stacked(states, update, mat, (R, C), beta1)  # [N, R, C]
        else:
            delta = update

        if wd != 0:
            p_fp32 = torch.stack([mat(p.data).float() for p in plist])
            delta = delta.add_(p_fp32, alpha=lr * wd)

        if cautious:
            mask = (delta * grad > 0).to(delta.dtype)
            denom = mask.reshape(N, -1).mean(dim=1).clamp_(min=1e-8).view(N, 1, 1)
            delta = delta.mul_(mask).div_(denom)

        # Subtract delta from the (matrixized) weights, batched, then scatter back.
        pviews = [mat(p.data) for p in plist]
        weights = torch.stack(pviews)                                     # [N, R, C], param dtype
        self._apply_subtract_batched(weights, delta, bf16_method)
        torch._foreach_copy_(pviews, list(weights.unbind(0)))

    @torch.no_grad()
    def _nonfactored_bucket(
        self,
        plist: list[Tensor],
        length: int,
        beta1: float,
        beta2: float,
        eps1: float,
        lr: float,
        clip: float,
        wd: float,
        cautious: bool,
        bf16_method: str,
        codec: _MomentumCodec,
        decay_rate: float | None = None,
    ) -> None:
        """Non-factored update (full per-coordinate second moment) for 1-D params.

        The bulk of a full fine-tune is biases and norm weights. Their update is
        the plain Adam-style ``grad / sqrt(v)`` — no row/col factoring — so a
        bucket of equal-length 1-D tensors stacks to ``[N, L]`` and steps as a few
        kernels. Mirrors the ``not factored`` branch of :meth:`_step_one_param`.
        """
        N = len(plist)  # noqa: N806 — matrix dim (stacked tensor is [N, L])
        vs = [self.state[p]["v"] for p in plist]                          # each [L], fp32

        grad = torch.stack([p.grad.float() for p in plist])               # [N, L]
        v = torch.stack(vs)                                               # [N, L]

        # Per-param second-moment EMA weight (scalar for fixed beta2, [N,1] vector
        # for adaptive beta2_t = 1 - step**decay_rate; see _one_minus_beta2_vec).
        one_minus_beta2 = self._one_minus_beta2_vec(plist, beta2, decay_rate)
        omb = one_minus_beta2 if isinstance(one_minus_beta2, float) else one_minus_beta2.view(N, 1)

        grad_sq = grad * grad
        if eps1 > 0:
            grad_sq = grad_sq.add_(eps1)
        v.lerp_(grad_sq, omb)
        torch._foreach_copy_(vs, list(v.unbind(0)))

        update = grad.mul(v.rsqrt())                                      # [N, L]
        rms = update.norm(2, dim=1) / math.sqrt(length)                   # per-slice RMS
        update.div_(rms.div_(clip).clamp_(min=1.0).view(N, 1))
        update.mul_(lr)

        if beta1 > 0:
            # Same codec entry point as the factored bucket; mat is identity here and
            # the effective per-param shape is the 1-D length (so int8 reduces the
            # whole L axis to one scalar scale, 4bit blocks over L).
            states = [self.state[p] for p in plist]
            delta = codec.ema_stacked(states, update, lambda t: t, (length,), beta1)  # [N, L]
        else:
            delta = update

        if wd != 0:
            p_fp32 = torch.stack([p.data.float() for p in plist])
            delta = delta.add_(p_fp32, alpha=lr * wd)

        if cautious:
            mask = (delta * grad > 0).to(delta.dtype)
            denom = mask.mean(dim=1).clamp_(min=1e-8).view(N, 1)
            delta = delta.mul_(mask).div_(denom)

        pviews = [p.data for p in plist]
        weights = torch.stack(pviews)                                     # [N, L], param dtype
        self._apply_subtract_batched(weights, delta, bf16_method)
        torch._foreach_copy_(pviews, list(weights.unbind(0)))

    @staticmethod
    def _apply_subtract_batched(weights: Tensor, delta_fp32: Tensor, bf16_method: str) -> None:
        """Stacked counterpart of :meth:`_apply_subtract` (no kahan/fp16 here)."""
        if (
            _is_low_precision(weights)
            and bf16_method == "stochastic_rounding"
            and weights.dtype == torch.bfloat16
        ):
            add_stochastic_(weights, delta_fp32, alpha=-1.0)
        else:
            weights.sub_(delta_fp32.to(weights.dtype))

    @torch.no_grad()
    def _step_one_param(self, p: Tensor, group: dict[str, Any]) -> None:
        beta1, beta2_fixed = group["betas"]
        eps1, _eps2 = group["eps"]
        lr, clip = group["lr"], group["clip_threshold"]
        wd, decay_rate = group["weight_decay"], group["decay_rate"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]

        state = self.state[p]
        if "step" not in state:
            self._init_state(p, state, group)
        state["step"] += 1
        beta2 = 1.0 - state["step"] ** decay_rate if decay_rate is not None else beta2_fixed

        grad_fp32 = p.grad if p.grad.dtype == torch.float32 else p.grad.float()
        ndim = grad_fp32.ndim
        factored = ndim >= 2

        if factored:
            matrixize = ndim > 2  # conv kernels always reshape to 2-D before factoring
            gv = grad_fp32.reshape(grad_fp32.shape[0], -1) if matrixize else grad_fp32
            update_factored_state(gv, state["row"], state["col"], beta2, eps1)
            r_factor, c_factor = factored_inv_sqrt_factors(state["row"], state["col"])
            update = gv.mul(r_factor).mul_(c_factor)
            if matrixize:
                update = update.view_as(grad_fp32)
        else:
            v = state["v"]
            grad_sq = grad_fp32 * grad_fp32
            if eps1 > 0:
                grad_sq.add_(eps1)
            v.lerp_(grad_sq, 1.0 - beta2)
            update = grad_fp32.mul(v.rsqrt())

        if clip > 0:
            update.div_((_rms(update) / clip).clamp_(min=1.0))
        update.mul_(lr)

        # Single codec call owns dequant → fp32 EMA → requant for every dtype.
        delta = self._codec(group).ema_one(state, update, beta1) if beta1 > 0 else update

        if wd != 0:
            p_fp32 = p.data if p.dtype == torch.float32 else p.data.float()
            delta = delta.add_(p_fp32, alpha=lr * wd)

        if cautious:
            mask = (delta * grad_fp32 > 0).to(delta.dtype)
            delta = delta.mul_(mask).div_(mask.mean().clamp_(min=1e-8))

        self._apply_subtract(p, delta, state, bf16_method)

    @staticmethod
    def _apply_subtract(p: Tensor, delta_fp32: Tensor, state: dict[str, Any], bf16_method: str) -> None:
        low = _is_low_precision(p)
        if low and bf16_method == "kahan":
            shift = state["shift"]
            shift.sub_(delta_fp32.to(p.dtype))
            p_before = p.detach().clone()
            p.add_(shift)
            shift.add_(p_before.sub_(p))
        elif low and bf16_method == "stochastic_rounding" and p.dtype == torch.bfloat16:
            add_stochastic_(p.data, delta_fp32, alpha=-1.0)
        else:
            p.data.sub_(delta_fp32.to(p.dtype))


def _is_low_precision(t: Tensor) -> bool:
    return t.dtype in _LOW_PRECISION
