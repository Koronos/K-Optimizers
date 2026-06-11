"""Adakaon — a conv-aware factored optimizer aimed at AdamW quality at
Adafactor memory, for bf16 diffusion fine-tuning.

Adakaon is the flagship of the library and the optimizer that most fully exercises
the **kaon** shared backend (factored second moment, quantized momentum codec,
stochastic rounding, foreach, cautious) — hence the name. Every other optimizer in
the package reuses pieces of Adakaon's machinery. (Formerly named *Adafusion*.)

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
* **Cautious masking** (Liang et al. 2024): zero the update coordinates whose
  sign disagrees with the gradient, renormalized to keep the step size. **On by
  default** — measured ~1.4% lower held-out val loss with momentum (paired
  t=-4.07); a literal no-op without momentum. Set ``cautious=False`` for
  no-momentum configs.

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

from kaon._backend import (
    FOREACH_BATCH_CUTOFF,
    cautious_batched_,
    cautious_one_,
    centralize_grads_,
    foreach_budget,
    is_low_precision,
    rms,
    subtract_batched_,
    subtract_one_,
)
from kaon._factored import factored_inv_sqrt_factors, update_factored_state
from kaon._momentum_codec import (
    _FOURBIT_BLOCK,
    _dequant_4bit,
    _dequant_4bit_stacked,
    _FloatCodec,
    _FourBitCodec,
    _Int8Codec,
    _make_codec,
    _MomentumCodec,
    _pack_nibbles,
    _quant_4bit,
    _quant_4bit_stacked,
    _quant_int8,
    _quant_int8_stacked,
    _unpack_nibbles,
    load_state_dict_preserving_dtypes,
)

__all__ = ["Adakaon"]

# Re-exported codec internals (kept importable from ``kaon.adakaon`` for
# backwards compatibility with existing tests/benchmarks). The implementations
# now live in :mod:`kaon._momentum_codec`, shared with KProdigy.
_ = (
    _dequant_4bit, _dequant_4bit_stacked, _FloatCodec, _FourBitCodec, _Int8Codec,
    _MomentumCodec, _pack_nibbles, _quant_4bit, _quant_4bit_stacked, _quant_int8,
    _quant_int8_stacked, _unpack_nibbles,
)

MomentumDtype = Literal["bfloat16", "float32", "int8", "4bit"]

# Diagnostic kernel-routing override (env-gated, zero cost when unset): a comma list of
# fused subsets to force onto the native path — e.g. KAON_FUSED_DISABLE="one_block" or
# "big,one_dim". Used to bisect which Triton kernel a real-training divergence lives in
# (the 2026-06-10 Nekaon NaN hunt); harmless to leave in.
import os as _os  # noqa: E402

_FUSED_DISABLE = frozenset(
    s.strip() for s in _os.environ.get("KAON_FUSED_DISABLE", "").split(",") if s.strip()
)
_PROBE_LOG_PATH = _os.environ.get("KAON_PROBE_LOG")

# Stacking a foreach bucket allocates several transient copies of the stacked
# tensor (grad fp32, the reconstruction, the SR intermediate, ...), so an unbounded
# bucket of large weights can OOM a full fine-tune — which would undercut
# Adakaon's whole memory story. We therefore cap the per-chunk element count and
# split bigger buckets. The cap is **adaptive to free VRAM** rather than a fixed
# constant: a card with lots of headroom batches whole buckets (and even stacks
# large weights), while a constrained card shrinks the chunk and stays safe. The
# budget is `free_bytes * SAFETY_FRACTION / BYTES_PER_ELEM`; the divisor accounts
# for the ~handful of simultaneous transient copies a chunk touches at peak.
# Peak transient bytes per stacked element. This is a property of the optimizer's
# intermediate tensors, NOT of the model: measured byte-for-byte identical on SDXL
# and Cosmos shapes. It depends only on the path and config — 2-D factored 24 B
# (common) / 38 B (momentum+wd+cautious), 1-D non-factored 28 B / 42 B (it also
# stacks the full second-moment state). 48 = worst measured (42.1) + margin.
_STACK_BYTES_PER_ELEM = 48

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



class Adakaon(Optimizer):
    """Conv-aware factored optimizer with optional bf16 momentum.

    Args:
        params: parameters or param-group dicts.
        lr: learning rate.
        betas: ``(beta1, beta2)``. ``beta1=0`` disables momentum (minimum memory,
            Adafactor-like). ``beta1>0`` enables momentum (AdamW-like quality).
        eps: ``(eps1, eps2)``. ``eps1`` is added to ``grad**2`` before the
            factored reductions (HF Adafactor convention). ``eps2`` is currently
            unused (reserved).
        weight_decay: decoupled weight decay (folded into the per-step delta).
        clip_threshold: Adafactor RMS update clipping (``rms(update) <= thr``).
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
        cautious: cautious masking (Liang et al. 2024) — zero the update
            coordinates whose sign disagrees with the gradient. **On by default**:
            it improves convergence when momentum is on (``beta1>0``) and is a
            literal no-op without momentum (the mask is all-ones — verified). Turn
            it off for no-momentum configs to skip the then-useless masking op.
        bf16_method: weight-update strategy for low-precision params —
            ``"stochastic_rounding"`` (default), ``"kahan"`` (+2 B/param), or
            ``"none"``. No-op on fp32 params.
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

    Note: Adakaon deliberately exposes **no** ``compile`` flag. A whole-step
    ``torch.compile`` was measured ~neutral on most shapes here and a slight loss on
    trivial steps (Adakaon's step has little fusable elementwise math), so it is
    not worth the API surface — Adakaon stays lean. The flag lives on
    :class:`~kaon.adamuon.AdaMuon`, whose heavy Newton-Schulz math it actually
    speeds up.
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
        momentum_dtype: MomentumDtype = "bfloat16",
        momentum_4bit_block: int = _FOURBIT_BLOCK,
        cautious: bool = True,
        gradient_centralization: bool = True,
        bf16_method: str = "stochastic_rounding",
        foreach: bool = True,
        foreach_batch_cutoff: int = FOREACH_BATCH_CUTOFF,
        foreach_stack_budget: int | None = None,
        fused: bool = False,
        fused_tile_cap: int | None = None,
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
            "momentum_dtype": momentum_dtype,
            "momentum_4bit_block": momentum_4bit_block,
            "cautious": cautious,
            "gradient_centralization": gradient_centralization,
            "bf16_method": bf16_method,
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
        # Optional Triton-fused step (same math + state, faster on GPU). Eligible 2-D weights run
        # through the fused kernels (one-block tile / chunked big-tensor); everything else falls back
        # to the native path below, in-place on the SAME state, so fused/non-fused interoperate and
        # resume from each other's checkpoints.
        self._fused = bool(fused)
        # When True, the many-same-shape big regime (>tile_cap) runs the batched chunked kernel; set
        # False to revert to the batched-native-foreach path (the A/B baseline). See _fused_big.
        self._fused_big_batched = True
        # EXPERIMENTAL (candidate #4): fuse the batched-big reductions into Triton (grad via pointer
        # array, no [N,R,C] stack, GC in-kernel). Default False until the A/B confirms a win. See
        # _chunked_reductions_fused and docs/FUSED_REDUCTIONS_DESIGN.md.
        self._fused_reductions = True
        self._t = 0
        self._fused_part: dict[int, tuple] = {}          # group id -> cached (ids, one_block, big, one_dim, native)
        self._fused_ob_caches: dict[int, Any] = {}       # group id -> PointerArrayCache (one-block)
        self._fused_od_caches: dict[int, Any] = {}       # group id -> OneDimPointerCache (1-D)
        if self._fused:
            from kaon._fused_triton import HAS_TRITON, TILE_CAP
            if not HAS_TRITON:
                raise RuntimeError("Adakaon(fused=True) requires Triton (a GPU-only optional dependency)")
            self._fused_tile_cap = TILE_CAP if fused_tile_cap is None else fused_tile_cap

    def _codec(self, group: dict[str, Any]) -> _MomentumCodec:
        md = group["momentum_dtype"]
        codec = self._codecs.get(md)
        if codec is None:
            codec = self._codecs[md] = _make_codec(md)
        return codec

    @torch.no_grad()
    def _init_state(self, p: Tensor, state: dict[str, Any], group: dict[str, Any]) -> None:
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
        if is_low_precision(p) and group["bf16_method"] == "kahan":
            state["shift"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self._fused:
            return self._fused_step(loss)
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            for p in params:
                if p.grad.is_sparse:
                    raise RuntimeError("Adakaon does not support sparse gradients")
            if group["gradient_centralization"]:
                centralize_grads_(params)
            self._native_dispatch(params, group)
        return loss

    @torch.no_grad()
    def _native_dispatch(self, params: list[Tensor], group: dict[str, Any]) -> None:
        """The native (non-fused) step over ``params`` — foreach batching where eligible, else
        per-param. Gradient Centralization is the caller's responsibility (done per-subset)."""
        if not params:
            return
        if self._foreach and self._group_foreach_eligible(group):
            chunk_budget = foreach_budget(self._foreach_stack_budget, self._foreach_batch_cutoff, _STACK_BYTES_PER_ELEM, params[0].device)
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

    # ----------------------------------------------------------------- fused (Triton) step
    @torch.no_grad()
    def _fused_step(self, loss: Any) -> Any:
        """Triton-fused step: eligible 2-D weights run through the one-block / chunked kernels
        (same math + state as native); everything else falls back to :meth:`_native_dispatch`."""
        import kaon._fused_triton as ft

        self._t += 1
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            for p in params:
                if p.grad.is_sparse:
                    raise RuntimeError("Adakaon does not support sparse gradients")
            one_block, big, one_dim, native = self._fused_partition(group, params, ft)
            if native:  # GC for the native subset (fused subsets centralize in-kernel / in-reductions)
                if group["gradient_centralization"]:
                    centralize_grads_(native)
                self._native_dispatch(native, group)
            if one_block:
                self._fused_one_block(one_block, group, ft)
            if big:
                self._fused_big(big, group, ft)
            if one_dim:
                self._fused_one_dim(one_dim, group, ft)
        return loss

    @torch.no_grad()
    def _fused_big(self, big: list[Tensor], group: dict[str, Any], ft: Any) -> None:
        """Dispatch the >tile-cap ("big") 2-D weights. The per-tensor fused-chunked kernel is
        launch-bound (~7 kernels/tensor); for the many same-shape big factors a real LoKr run has
        (e.g. 236x 512x512) we run the **batched chunked kernel** (``_chunked_step_batched``) —
        the whole same-shape bucket in ~2 launches, p/m written in place via a pointer array. A lone
        big tensor has no batch to amortize, so it keeps the per-tensor fused-chunked kernel. Same
        math + state either way. ``self._fused_big_batched=False`` reverts the multi-tensor case to
        the batched-native-foreach path (the A/B baseline)."""
        if len(big) >= 2 and not self._fused_big_batched:
            if group["gradient_centralization"]:
                centralize_grads_(big)
            self._native_dispatch(big, group)
            return
        # Group by EXACT shape; same-shape buckets of >=2 take the batched chunked kernel, lone
        # tensors take the per-tensor chunked kernel.
        by_shape: dict[tuple[int, int], list[Tensor]] = {}
        for p in big:
            by_shape.setdefault(tuple(p.shape), []).append(p)
        for plist in by_shape.values():
            if len(plist) >= 2:
                self._chunked_step_batched(plist, group, ft)
            else:
                self._chunked_step(plist[0], group, ft)

    def _fused_partition(self, group: dict[str, Any], params: list[Tensor], ft: Any) -> tuple:
        """Split a group's params into (one-block, chunked-big, one-dim, native), cached per param-set."""
        gid = id(group)
        ids = tuple(id(p) for p in params)
        cached = self._fused_part.get(gid)
        if cached is not None and cached[0] == ids:
            return cached[1], cached[2], cached[3], cached[4]
        md, bf16m, cap = group["momentum_dtype"], group["bf16_method"], self._fused_tile_cap
        one_block: list[Tensor] = []
        big: list[Tensor] = []
        one_dim: list[Tensor] = []
        native: list[Tensor] = []
        momentum = group["betas"][0] > 0  # the kernels assume a momentum buffer; beta1==0 -> native
        float_mom = md in ("bfloat16", "float32")  # the 1-D kernel handles only fp32/bf16 momentum
        for p in params:
            # bf16 params need stochastic rounding (the kernel's only bf16 write); kahan/none -> native
            bf_ok = (p.dtype != torch.bfloat16) or (bf16m == "stochastic_rounding")
            # ndim>2 (conv) is matrixized to (out, in*kh*kw); the reshape needs a contiguous grad, and
            # only fp32/bf16 momentum (quant's per-row requant would reshape the conv state) -> native.
            conv_ok = p.ndim <= 2 or (p.grad is not None and p.grad.is_contiguous() and float_mom)
            two_d = momentum and bf_ok and conv_ok and p.ndim >= 2 and p.is_cuda and p.is_contiguous() \
                and p.dtype in (torch.float32, torch.bfloat16)
            ok = momentum and bf_ok and conv_ok and ft.fused_eligible(p, cap)
            if ok and md == "4bit" and ft.eff_2d(p)[1] % 2 != 0:
                ok = False                                  # one-block 4bit needs even C
            if ok:
                one_block.append(p)
            elif two_d and ft.next_pow2_tile(*ft.eff_2d(p))[0] * ft.next_pow2_tile(*ft.eff_2d(p))[1] > cap:
                big.append(p)
            elif momentum and bf_ok and float_mom and ft.fused_1d_eligible(p, cap):
                one_dim.append(p)
            else:
                native.append(p)
        if _FUSED_DISABLE:  # diagnostic routing override (see module note)
            if "one_block" in _FUSED_DISABLE:
                native += one_block; one_block = []
            if "big" in _FUSED_DISABLE:
                native += big; big = []
            if "one_dim" in _FUSED_DISABLE:
                native += one_dim; one_dim = []
        if _PROBE_LOG_PATH:  # routing census + grad-contiguity audit (probe runs only)
            noncontig = [tuple(p.shape) for p in params if p.grad is not None and not p.grad.is_contiguous()]
            with open(_PROBE_LOG_PATH, "a") as fh:  # noqa: SIM115 — diagnostics only
                fh.write(
                    f"[census] one_block={len(one_block)} big={len(big)} one_dim={len(one_dim)} "
                    f"native={len(native)} noncontig_grads={noncontig[:8]} disable={sorted(_FUSED_DISABLE)}\n"
                )
        self._fused_part[gid] = (ids, one_block, big, one_dim, native)
        return one_block, big, one_dim, native

    def _fused_one_block(self, plist: list[Tensor], group: dict[str, Any], ft: Any) -> None:
        """Launch the one-block pointer-array kernel over the eligible small 2-D weights."""
        for p in plist:
            st = self.state[p]
            if not st:
                self._init_state(p, st, group)
        ids = tuple(id(p) for p in plist)
        gid = id(group)
        cache = self._fused_ob_caches.get(gid)
        if cache is None or cache.ids != ids:
            cache = ft.PointerArrayCache(plist, lambda p: self.state[p], None)
            self._fused_ob_caches[gid] = cache
        cache.refresh_grads()
        b1, b2 = group["betas"]
        lr, eps1 = group["lr"], group["eps"][0]
        clip, wd = group["clip_threshold"], group["weight_decay"]
        cautious, gc = group["cautious"], group["gradient_centralization"]
        for bk in cache.buckets:
            lanes = bk["BR"] * bk["BC"]
            ft._adakaon_tile_kernel[(len(bk["plist"]),)](
                bk["g_addr"], bk["p_addr"], bk["m_addr"], bk["mscale_addr"], bk["row_addr"], bk["col_addr"],
                bk["Rs"], bk["Cs"], lr, b1, b2, eps1, clip, wd, self._t,
                LOWP=bk["lowp"], MOM=bk["mom"], CAUTIOUS=cautious, WD=wd != 0, GC=gc, SR=bk["lowp"],
                BR=bk["BR"], BC=bk["BC"], num_warps=ft.warps_for(lanes),
            )

    def _fused_one_dim(self, plist: list[Tensor], group: dict[str, Any], ft: Any) -> None:
        """One-block non-factored kernel over the eligible 1-D weights (biases / norm scales). GC is a
        no-op on 1-D (``centralize_grads_`` skips ndim<2), so it never enters this path."""
        for p in plist:
            st = self.state[p]
            if not st:
                self._init_state(p, st, group)
        ids = tuple(id(p) for p in plist)
        gid = id(group)
        cache = self._fused_od_caches.get(gid)
        if cache is None or cache.ids != ids:
            cache = ft.OneDimPointerCache(plist, lambda p: self.state[p])
            self._fused_od_caches[gid] = cache
        cache.refresh_grads()
        b1, b2 = group["betas"]
        lr, eps1 = group["lr"], group["eps"][0]
        clip, wd = group["clip_threshold"], group["weight_decay"]
        cautious = group["cautious"]
        for bk in cache.buckets:
            ft._adam_1d_kernel[(len(bk["plist"]),)](
                bk["g_addr"], bk["p_addr"], bk["m_addr"], bk["v_addr"], bk["Ls"],
                lr, b1, b2, eps1, clip, wd, self._t,
                LOWP=bk["lowp"], MOM=bk["mom"], MOMENTUM=bk["momentum"], CAUTIOUS=cautious,
                WD=wd != 0, SR=bk["lowp"], BL=bk["BL"], num_warps=ft.warps_for(bk["BL"]),
            )

    def _chunked_reductions(self, p: Tensor, group: dict[str, Any], st: dict[str, Any]) -> tuple:
        """Shared torch part of a big-tensor step: GC + row/col EMA + rms (matvec, no [R,C] temp).
        ``ndim>2`` convs are matrixized to ``(out, in*kh*kw)`` (the row/col state's shape)."""
        n = p.numel()
        R = p.shape[0]  # noqa: N806
        b2, eps1 = group["betas"][1], group["eps"][0]
        clip, lr = group["clip_threshold"], group["lr"]
        g = p.grad.float().reshape(R, n // R)
        if group["gradient_centralization"]:
            g = g - g.mean(dim=1, keepdim=True)
        g = g.contiguous()
        gsq = g * g
        st["row"].lerp_(gsq.mean(1).add_(eps1), 1.0 - b2)
        st["col"].lerp_(gsq.mean(0).add_(eps1), 1.0 - b2)
        r = st["row"].div(st["row"].mean()).rsqrt_()
        c = st["col"].rsqrt()
        rms = (((r * r) * gsq.matmul(c * c)).sum() / n).sqrt_()
        return g, r, c, lr / float(rms.div_(clip).clamp_(min=1.0))

    def _chunked_step(self, p: Tensor, group: dict[str, Any], ft: Any) -> None:
        """One big 2-D tensor via the chunked kernels; int8/4bit momentum through the codec (dequant
        -> fp32 temp -> kernels -> requant) so the weight update uses the exact pre-requant momentum."""
        st = self.state[p]
        if not st:
            self._init_state(p, st, group)
        R, C = p.shape[0], p.numel() // p.shape[0]  # noqa: N806 — matrix dims (conv -> matrixized)
        n = R * C
        md, b1 = group["momentum_dtype"], group["betas"][0]
        lr, wd, cautious = group["lr"], group["weight_decay"], group["cautious"]
        sr = (p.dtype == torch.bfloat16) and (group["bf16_method"] == "stochastic_rounding")
        g, r, c, inv_rms_lr = self._chunked_reductions(p, group, st)
        quant = md in ("int8", "4bit")
        if quant:
            m_fp32 = self._codec(group).dequant_one(st, torch.empty(R, C, device=p.device)).reshape(R, C)
            mf = m_fp32.reshape(-1)
        else:
            mf = st["m"].reshape(-1)
        keep = torch.zeros(1, dtype=torch.int32, device=p.device)
        gf, pf = g.reshape(-1), p.reshape(-1)
        grid = ((n + 1023) // 1024,)
        ft._chunked_mom[grid](gf, mf, pf, r, c, keep, C, n, inv_rms_lr, lr * wd, b1,
                              CAUTIOUS=cautious, WD=wd != 0, BLOCK=1024)
        if quant:
            if md == "int8":
                st["m"], st["m_scale"] = _quant_int8(m_fp32)
            else:
                st["m"], st["m_scale"], _ = _quant_4bit(m_fp32.reshape(-1), st["m_block"])
        inv_mean = 1.0 / max(keep.item() / n, 1e-8) if cautious else 1.0
        ft._chunked_apply[grid](gf, mf, pf, n, inv_mean, lr * wd, self._t,
                                CAUTIOUS=cautious, WD=wd != 0, SR=sr, BLOCK=1024)

    # ----------------------------------------------- batched chunked (many same-shape big tensors)
    @torch.no_grad()
    def _chunked_reductions_batched(self, plist: list[Tensor], group: dict[str, Any]) -> tuple:
        """Stacked torch reductions for a same-shape big bucket: GC (on the fp32 copy) + row/col EMA
        + per-tensor rms (matvec, no [N,R,C] beyond grad/gsq). Returns the stacked fp32 grad ``[N,n]``,
        the stacked r/c factors ``[N,R]``/``[N,C]`` (contiguous), and ``inv_rms_lr`` ``[N]`` — the same
        quantities ``_chunked_reductions`` returns per tensor. Mirrors that math exactly (eps1 added to
        the row/col means; rms uses raw gsq)."""
        b2, eps1 = group["betas"][1], group["eps"][0]
        clip, lr = group["clip_threshold"], group["lr"]
        N = len(plist)  # noqa: N806
        R, C = plist[0].shape[0], plist[0].numel() // plist[0].shape[0]  # noqa: N806 — conv -> matrixized
        n = R * C
        states = [self.state[p] for p in plist]
        g = torch.stack([p.grad.float().reshape(R, C) for p in plist])     # [N, R, C]
        if group["gradient_centralization"]:
            g.sub_(g.mean(dim=-1, keepdim=True))                          # GC on the fp32 copy
        gsq = g * g                                                       # raw (eps1 goes on the means)
        row = torch.stack([s["row"] for s in states])                    # [N, R]
        col = torch.stack([s["col"] for s in states])                    # [N, C]
        row.lerp_(gsq.mean(dim=-1).add_(eps1), 1.0 - b2)
        col.lerp_(gsq.mean(dim=-2).add_(eps1), 1.0 - b2)
        torch._foreach_copy_([s["row"] for s in states], list(row.unbind(0)))
        torch._foreach_copy_([s["col"] for s in states], list(col.unbind(0)))
        r = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_()             # [N, R]
        c = col.rsqrt()                                                  # [N, C]
        # rms per tensor via matvec: sqrt( sum_i r_i^2 * (gsq @ c^2)_i / n )  (no [N,R,C] temp)
        rms = (r * r).mul_(torch.bmm(gsq, (c * c).unsqueeze(-1)).squeeze(-1)).sum(-1).div_(n).sqrt_()
        inv_rms_lr = lr / rms.div_(clip).clamp_(min=1.0)                 # [N]
        return g.reshape(N, n), r.contiguous(), c.contiguous(), inv_rms_lr.contiguous()

    @torch.no_grad()
    def _chunked_step_batched(self, plist: list[Tensor], group: dict[str, Any], ft: Any) -> None:
        """A bucket of >=2 same-shape big 2-D tensors via the batched chunked kernels (~2 launches).
        fp32/bf16 momentum is read/written in place via the m pointer array; int8/4bit is dequant'd to
        a stacked fp32 temp, stepped on the temp, requant'd between passes (the ``_chunked_step``
        precedent), so the weight update uses the exact pre-requant momentum."""
        for p in plist:
            st = self.state[p]
            if not st:
                self._init_state(p, st, group)
        N = len(plist)  # noqa: N806
        R, C = plist[0].shape[0], plist[0].numel() // plist[0].shape[0]  # noqa: N806 — conv -> matrixized
        n = R * C
        dev = plist[0].device
        md, b1 = group["momentum_dtype"], group["betas"][0]
        lr, wd, cautious = group["lr"], group["weight_decay"], group["cautious"]
        gc = group["gradient_centralization"]
        lowp = plist[0].dtype == torch.bfloat16
        sr = lowp and (group["bf16_method"] == "stochastic_rounding")
        states = [self.state[p] for p in plist]

        # Reductions: fused (grad via pointer array, no [N,R,C] stack — candidate #4) or torch.
        fused_red = self._fused_reductions
        if fused_red:
            g_addr, rowmean, r, c, inv_rms_lr = self._chunked_reductions_fused(plist, group, ft, R, C, n, lowp)
        else:
            g, r, c, inv_rms_lr = self._chunked_reductions_batched(plist, group)

        quant = md in ("int8", "4bit")
        if quant:  # dequant whole bucket to a stacked fp32 temp; kernel m pointers index its slices
            temp = torch.empty(N, R, C, dtype=torch.float32, device=dev)
            for i, st in enumerate(states):
                temp[i] = self._codec(group).dequant_one(st, temp[i]).reshape(R, C)  # temp[i] = shape template
            m_addr = ft.ptr_array(list(temp), dev)
            mom = ft.MOM_FP32
        else:
            m_addr = ft.ptr_array([st["m"] for st in states], dev)
            mom = ft.MOM_BF16 if md == "bfloat16" else ft.MOM_FP32
        p_addr = ft.ptr_array(plist, dev)
        keep = torch.zeros(N, dtype=torch.int32, device=dev)
        K = (n + 1023) // 1024  # noqa: N806
        grid = (N * K,)
        if fused_red:
            ft._chunked_mom_batched_g[grid](
                g_addr, rowmean, m_addr, p_addr, r, c, keep, inv_rms_lr, lr * wd, b1, R, C, n, K,
                LOWP=lowp, MOM=mom, GC=gc, CAUTIOUS=cautious, WD=wd != 0, BLOCK=1024,
            )
        else:
            ft._chunked_mom_batched[grid](
                g, m_addr, p_addr, r, c, keep, inv_rms_lr, lr * wd, b1, R, C, n, K,
                LOWP=lowp, MOM=mom, CAUTIOUS=cautious, WD=wd != 0, BLOCK=1024,
            )
        if quant:  # requant the updated fp32 temp back into per-tensor storage (apply reads the temp)
            for i, st in enumerate(states):
                if md == "int8":
                    st["m"], st["m_scale"] = _quant_int8(temp[i])
                else:
                    st["m"], st["m_scale"], _ = _quant_4bit(temp[i].reshape(-1), st["m_block"])
        inv_mean = (1.0 / (keep.float() / n).clamp_(min=1e-8)) if cautious else torch.ones(N, device=dev)
        if fused_red:
            ft._chunked_apply_batched_g[grid](
                g_addr, rowmean, m_addr, p_addr, inv_mean, lr * wd, self._t, R, C, n, K,
                LOWP=lowp, MOM=mom, GC=gc, CAUTIOUS=cautious, WD=wd != 0, SR=sr, BLOCK=1024,
            )
        else:
            ft._chunked_apply_batched[grid](
                g, m_addr, p_addr, inv_mean, lr * wd, self._t, n, K,
                LOWP=lowp, MOM=mom, CAUTIOUS=cautious, WD=wd != 0, SR=sr, BLOCK=1024,
            )

    @torch.no_grad()
    def _chunked_reductions_fused(self, plist, group, ft, R, C, n, lowp):  # noqa: N803
        """Candidate #4: row/col EMA factors + inv_rms_lr via Triton reduction kernels reading grad
        from a pointer array (NO [N,R,C] stack; GC in-kernel). Returns (g_addr, rowmean, r, c,
        inv_rms_lr) — the mom/apply ``_g`` kernels re-read grad via g_addr and GC via rowmean."""
        b2, eps1 = group["betas"][1], group["eps"][0]
        clip, lr = group["clip_threshold"], group["lr"]
        gc = group["gradient_centralization"]
        N = len(plist)  # noqa: N806
        dev = plist[0].device
        states = [self.state[p] for p in plist]
        g_addr = ft.ptr_array([p.grad for p in plist], dev)
        BR, BC, RB = ft.reduction_tile(R, C)  # noqa: N806
        rowmean = torch.empty(N * R, dtype=torch.float32, device=dev)
        rowsum = torch.empty(N * R, dtype=torch.float32, device=dev)
        colsum = torch.zeros(N * C, dtype=torch.float32, device=dev)  # atomic target
        ft._reduce_rowcol[(N * RB,)](
            g_addr, rowmean, rowsum, colsum, R, C, RB,
            LOWP=lowp, GC=gc, BR=BR, BC=BC, num_warps=ft.warps_for(BR * BC),
        )
        # cheap [N,R]/[N,C] EMA + factors (no [N,R,C] anywhere)
        rowsum = rowsum.view(N, R)
        colsum = colsum.view(N, C)
        row = torch.stack([s["row"] for s in states])
        col = torch.stack([s["col"] for s in states])
        row.lerp_(rowsum.div(C).add_(eps1), 1.0 - b2)
        col.lerp_(colsum.div(R).add_(eps1), 1.0 - b2)
        torch._foreach_copy_([s["row"] for s in states], list(row.unbind(0)))
        torch._foreach_copy_([s["col"] for s in states], list(col.unbind(0)))
        r = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_().contiguous()
        c = col.rsqrt().contiguous()
        rms = torch.zeros(N, dtype=torch.float32, device=dev)
        ft._reduce_rms[(N * RB,)](
            g_addr, rowmean, r, c, rms, R, C, RB,
            LOWP=lowp, GC=gc, BR=BR, BC=BC, num_warps=ft.warps_for(BR * BC),
        )
        inv_rms_lr = lr / rms.div_(n).sqrt_().div_(clip).clamp_(min=1.0)
        return g_addr, rowmean, r, c, inv_rms_lr

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state, preserving the quantized first moment's stored dtype.

        torch's default ``load_state_dict`` upcasts every state tensor to the
        param's dtype (fp32), which would silently inflate a bf16/int8/4bit
        ``momentum_dtype`` back to fp32 on resume — losing the memory the codec
        was chosen to save and breaking bit-exact resume. Delegate to the shared
        helper that restores each tensor to how it was checkpointed.
        """
        load_state_dict_preserving_dtypes(self, state_dict)

    # ----------------------------------------------------------------- foreach

    @staticmethod
    def _group_foreach_eligible(group: dict[str, Any]) -> bool:
        """Group-level options the batched fast path supports."""
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
            and is_low_precision(p)
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
        cautious, bf16_method = group["cautious"], group["bf16_method"]
        codec = self._codec(group)

        factored_buckets: dict[tuple[Any, ...], list[Tensor]] = {}
        flat_buckets: dict[tuple[Any, ...], list[Tensor]] = {}
        for p in params:
            state = self.state[p]
            if not state:
                self._init_state(p, state, group)
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
                    beta1, beta2, eps1, lr, clip, wd, cautious, bf16_method, codec,
                )
        for (length, _dtype), plist in flat_buckets.items():
            step = max(1, budget // max(length, 1))
            for i in range(0, len(plist), step):
                self._nonfactored_bucket(
                    plist[i:i + step], length,
                    beta1, beta2, eps1, lr, clip, wd, cautious, bf16_method, codec,
                )

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

        # Second-moment EMA weight (fixed beta2). row/col are [N, R]/[N, C], so the
        # scalar broadcasts cleanly.
        omb = 1.0 - beta2

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
            delta = cautious_batched_(delta, grad)

        # Subtract delta from the (matrixized) weights, batched.
        subtract_batched_([mat(p.data) for p in plist], delta, bf16_method)

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

        # Second-moment EMA weight (fixed beta2).
        omb = 1.0 - beta2

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
            delta = cautious_batched_(delta, grad)

        subtract_batched_([p.data for p in plist], delta, bf16_method)

    @torch.no_grad()
    def _step_one_param(self, p: Tensor, group: dict[str, Any]) -> None:
        beta1, beta2 = group["betas"]
        eps1, _eps2 = group["eps"]
        lr, clip = group["lr"], group["clip_threshold"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]

        state = self.state[p]
        if not state:
            self._init_state(p, state, group)

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
            update.div_((rms(update) / clip).clamp_(min=1.0))
        update.mul_(lr)

        # Single codec call owns dequant → fp32 EMA → requant for every dtype.
        delta = self._codec(group).ema_one(state, update, beta1) if beta1 > 0 else update

        if wd != 0:
            p_fp32 = p.data if p.dtype == torch.float32 else p.data.float()
            delta = delta.add_(p_fp32, alpha=lr * wd)

        if cautious:
            delta = cautious_one_(delta, grad_fp32)

        subtract_one_(p, delta, state, bf16_method)


