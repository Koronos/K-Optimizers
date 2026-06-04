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

from koptim._compiled import _get_compiled_factored_update
from koptim._factored import factored_inv_sqrt_factors, update_factored_state
from koptim._stochastic_rounding import add_stochastic_

__all__ = ["Adafusion"]

_LOW_PRECISION = (torch.bfloat16, torch.float16)
MomentumDtype = Literal["bfloat16", "float32", "int8"]

# Upper bound on elements in a single stacked foreach bucket. Stacking a bucket
# allocates a few transient fp32 copies (grad, update, the SR intermediate), so an
# unbounded bucket of large weights can OOM during a full fine-tune — which would
# undercut Adafusion's whole memory story. Chunking caps that transient to roughly
# a few * this * 4 bytes, independent of model size. Tiny adapter tensors still
# batch hundreds-at-once; only large-weight buckets split (their per-tensor compute
# dwarfs the kernel-launch overhead anyway, so the speedup barely changes).
_MAX_STACK_ELEMS = 4_000_000


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
            ``"bfloat16"`` (default; ~2 B/param), ``"float32"`` (4 B/param), or
            ``"int8"`` (~1 B/param, per-row absmax quantized; Lion8bit-class
            memory but with the factored adaptive second moment).
        cautious: enable cautious masking (off by default; opt-in regularizer).
        bf16_method: weight-update strategy for low-precision params —
            ``"stochastic_rounding"`` (default), ``"kahan"`` (+2 B/param), or
            ``"none"``. No-op on fp32 params.
        factor_conv_as_matrix: reshape 4-D conv kernels to 2-D before factoring
            (the conv-aware fix). Default ``True``; set ``False`` for the legacy
            last-dims behaviour.
        compile: ``torch.compile`` the factored core. Big win on LARGE 2-D
            weights (transformer/DiT 2048x2048+, ~+30% measured, closes the gap
            to AdamW), neutral-to-negative on many small weights; off by default.
            Only the fixed-beta2 (``decay_rate=None``) factored path with
            ``clip_threshold > 0`` is compiled. Needs a torch.compile backend.
        foreach: batch the step across parameters with multi-tensor (stacked) ops
            instead of a per-parameter Python loop. Default ``True``. Huge win when
            many tensors are stepped at once — LoRA/LoKr adapters (hundreds of tiny
            2-D tensors) *and* full fine-tunes (thousands of weights incl. all the
            1-D biases/norms). Params are bucketed by shape and each bucket steps
            as a few stacked kernels: ``ndim >= 2`` factored ``[N, R, C]``, ``ndim
            == 1`` non-factored ``[N, L]``. Matches the per-parameter path
            numerically (stochastic-rounding draws differ, unbiased either way);
            the rest (0-D scalars, int8 momentum, kahan, fp16+SR, non-contiguous
            matrixized convs, single-param groups) transparently falls back to it.
            For eligible params it supersedes ``compile`` (no per-tensor graph
            needed). Set ``False`` to force the per-parameter path.
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
        cautious: bool = False,
        bf16_method: str = "stochastic_rounding",
        factor_conv_as_matrix: bool = True,
        compile: bool = False,
        foreach: bool = True,
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
        if momentum_dtype not in ("bfloat16", "float32", "int8"):
            raise ValueError(f"momentum_dtype must be bfloat16/float32/int8, got {momentum_dtype!r}")
        if bf16_method not in ("stochastic_rounding", "kahan", "none"):
            raise ValueError(f"bf16_method must be stochastic_rounding/kahan/none, got {bf16_method!r}")
        defaults = {
            "lr": lr,
            "betas": (beta1, beta2),
            "eps": (float(eps[0]), float(eps[1])),
            "weight_decay": weight_decay,
            "clip_threshold": clip_threshold,
            "decay_rate": decay_rate,
            "momentum_dtype": momentum_dtype,
            "cautious": cautious,
            "bf16_method": bf16_method,
            "factor_conv_as_matrix": factor_conv_as_matrix,
        }
        super().__init__(params, defaults)
        # Optional torch.compile of the factored core. Big win on LARGE 2-D
        # weights (e.g. transformer/DiT 2048x2048+, ~+30%), neutral-to-negative
        # on many small weights. Only the fixed-beta2 (decay_rate=None) factored
        # path with clip>0 is routed through it.
        self._factored_fn = _get_compiled_factored_update() if compile else None
        # Multi-tensor (foreach) batching of the factored fast path. Collapses the
        # per-parameter Python loop + per-tensor kernel launches into a handful of
        # stacked-tensor ops per (shape, dtype) bucket — the decisive win when many
        # small weights are trained at once (LoRA/LoKr adapters). Numerically
        # matches the per-parameter path; stochastic-rounding draws differ
        # (unbiased either way). Anything it doesn't cover falls back per-param.
        self._foreach = foreach

    @torch.no_grad()
    def _init_state(self, p: Tensor, state: dict[str, Any], group: dict[str, Any]) -> None:
        state["step"] = 0
        grad = p.grad
        factored = p.ndim >= 2
        if factored:
            gv = grad if (p.ndim == 2 or not group["factor_conv_as_matrix"]) else grad.reshape(grad.shape[0], -1)
            row_shape = gv.shape[:-1]
            col_shape = gv.shape[:-2] + gv.shape[-1:]
            state["row"] = torch.zeros(row_shape, dtype=torch.float32, device=p.device)
            state["col"] = torch.zeros(col_shape, dtype=torch.float32, device=p.device)
        else:
            state["v"] = torch.zeros_like(grad, dtype=torch.float32)
        if group["betas"][0] > 0:
            md = group["momentum_dtype"]
            if md == "int8":
                state["m"] = torch.zeros_like(grad, dtype=torch.int8)
                state["m_scale"] = torch.ones(
                    (grad.shape[0],) + (1,) * (grad.ndim - 1) if grad.ndim >= 2 else (),
                    dtype=torch.float32, device=p.device,
                )
            else:
                state["m"] = torch.zeros_like(grad, dtype=torch.bfloat16 if md == "bfloat16" else torch.float32)
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
                fast: list[Tensor] = []
                slow: list[Tensor] = []
                for p in params:
                    (fast if self._param_foreach_eligible(p, group) else slow).append(p)
                if len(fast) >= 2:
                    self._step_foreach(fast, group)
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
    @staticmethod
    def _group_foreach_eligible(group: dict[str, Any]) -> bool:
        """Group-level options the batched fast path supports."""
        return (
            group["decay_rate"] is None          # fixed beta2 (no step-dependent decay)
            and group["clip_threshold"] > 0      # clip always applied (matches compiled path)
            and group["momentum_dtype"] != "int8"  # int8 requant is per-row, kept per-param
            and group["bf16_method"] != "kahan"  # kahan needs a per-param shift buffer
        )

    @staticmethod
    def _param_foreach_eligible(p: Tensor, group: dict[str, Any]) -> bool:
        """Per-parameter shapes/dtypes the batched fast path can stack.

        Both branches are covered: ``ndim >= 2`` uses the factored bucket, ``ndim
        == 1`` (biases/norms — the bulk of a full fine-tune) uses the non-factored
        bucket. Only 0-D scalars and the awkward dtype/contiguity cases fall back.
        """
        if p.ndim == 0:                          # 0-D scalars -> per-param path
            return False
        if p.numel() > _MAX_STACK_ELEMS // 2:
            # Large weights can't share a stack (chunk size would be 1) and are
            # compute/bandwidth-bound anyway — the per-tensor launch overhead is
            # noise for them, so loop them and skip the stack/copy overhead.
            return False
        if (
            group["bf16_method"] == "stochastic_rounding"
            and _is_low_precision(p)
            and p.dtype != torch.bfloat16        # fp16+SR is unsupported -> per-param (raises)
        ):
            return False
        if p.ndim > 2 and group["factor_conv_as_matrix"]:
            # Matrixized conv writes back through a reshaped view -> needs contiguity.
            if not (p.data.is_contiguous() and p.grad.is_contiguous()):
                return False
        return True

    @torch.no_grad()
    def _step_foreach(self, params: list[Tensor], group: dict[str, Any]) -> None:
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
        reshape_conv = group["factor_conv_as_matrix"]

        factored_buckets: dict[tuple[Any, ...], list[Tensor]] = {}
        flat_buckets: dict[tuple[Any, ...], list[Tensor]] = {}
        for p in params:
            state = self.state[p]
            if "step" not in state:
                self._init_state(p, state, group)
            state["step"] += 1
            g = p.grad
            if g.ndim >= 2:
                matrixize = g.ndim > 2 and reshape_conv
                eff = (g.shape[0], g.numel() // g.shape[0]) if matrixize else tuple(g.shape)
                factored_buckets.setdefault((eff, p.dtype, matrixize), []).append(p)
            else:  # ndim == 1
                flat_buckets.setdefault((g.shape[0], p.dtype), []).append(p)

        for (eff, _dtype, matrixize), plist in factored_buckets.items():
            step = max(1, _MAX_STACK_ELEMS // max(eff[0] * eff[1], 1))
            for i in range(0, len(plist), step):
                self._factored_bucket(
                    plist[i:i + step], eff, matrixize,
                    beta1, beta2, eps1, lr, clip, wd, cautious, bf16_method,
                )
        for (length, _dtype), plist in flat_buckets.items():
            step = max(1, _MAX_STACK_ELEMS // max(length, 1))
            for i in range(0, len(plist), step):
                self._nonfactored_bucket(
                    plist[i:i + step], length,
                    beta1, beta2, eps1, lr, clip, wd, cautious, bf16_method,
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
    ) -> None:
        R, C = eff
        N = len(plist)

        def mat(t: Tensor) -> Tensor:
            return t.view(R, C) if matrixize else t

        rows = [self.state[p]["row"] for p in plist]
        cols = [self.state[p]["col"] for p in plist]

        grad = torch.stack([mat(p.grad).float() for p in plist])          # [N, R, C]
        row = torch.stack(rows)                                           # [N, R]
        col = torch.stack(cols)                                           # [N, C]

        # Factored second-moment EMA (HF eps placement: eps1 before the means).
        grad_sq = grad * grad
        if eps1 > 0:
            grad_sq = grad_sq.add_(eps1)
        row.lerp_(grad_sq.mean(dim=-1), 1.0 - beta2)
        col.lerp_(grad_sq.mean(dim=-2), 1.0 - beta2)
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
            ms = [self.state[p]["m"] for p in plist]
            mom = torch.stack([mat(m) for m in ms])                       # [N, R, C], momentum dtype
            mom.lerp_(update.to(mom.dtype), 1.0 - beta1)
            torch._foreach_copy_([mat(m) for m in ms], list(mom.unbind(0)))
            delta = mom.float()
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
    ) -> None:
        """Non-factored update (full per-coordinate second moment) for 1-D params.

        The bulk of a full fine-tune is biases and norm weights. Their update is
        the plain Adam-style ``grad / sqrt(v)`` — no row/col factoring — so a
        bucket of equal-length 1-D tensors stacks to ``[N, L]`` and steps as a few
        kernels. Mirrors the ``not factored`` branch of :meth:`_step_one_param`.
        """
        N = len(plist)
        vs = [self.state[p]["v"] for p in plist]                          # each [L], fp32

        grad = torch.stack([p.grad.float() for p in plist])               # [N, L]
        v = torch.stack(vs)                                               # [N, L]

        grad_sq = grad * grad
        if eps1 > 0:
            grad_sq = grad_sq.add_(eps1)
        v.lerp_(grad_sq, 1.0 - beta2)
        torch._foreach_copy_(vs, list(v.unbind(0)))

        update = grad.mul(v.rsqrt())                                      # [N, L]
        rms = update.norm(2, dim=1) / math.sqrt(length)                   # per-slice RMS
        update.div_(rms.div_(clip).clamp_(min=1.0).view(N, 1))
        update.mul_(lr)

        if beta1 > 0:
            ms = [self.state[p]["m"] for p in plist]
            mom = torch.stack(ms)                                         # [N, L], momentum dtype
            mom.lerp_(update.to(mom.dtype), 1.0 - beta1)
            torch._foreach_copy_(ms, list(mom.unbind(0)))
            delta = mom.float()
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
        reshape_conv = group["factor_conv_as_matrix"]

        state = self.state[p]
        if "step" not in state:
            self._init_state(p, state, group)
        state["step"] += 1
        beta2 = 1.0 - state["step"] ** decay_rate if decay_rate is not None else beta2_fixed

        grad_fp32 = p.grad if p.grad.dtype == torch.float32 else p.grad.float()
        ndim = grad_fp32.ndim
        factored = ndim >= 2

        update_is_clipped = False
        if factored:
            matrixize = ndim > 2 and reshape_conv
            gv = grad_fp32.reshape(grad_fp32.shape[0], -1) if matrixize else grad_fp32
            if self._factored_fn is not None and clip > 0 and decay_rate is None:
                # Compiled EMA + reconstruction + clip in one fused graph.
                update = self._factored_fn(gv, state["row"], state["col"], beta2, eps1, clip)
                update_is_clipped = True
            else:
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

        if clip > 0 and not update_is_clipped:
            update.div_((_rms(update) / clip).clamp_(min=1.0))
        update.mul_(lr)

        if beta1 > 0:
            if state["m"].dtype == torch.int8:
                m = state["m"].float() * state["m_scale"]   # dequant
                m.lerp_(update, 1.0 - beta1)
                delta = m.clone()
                state["m"], state["m_scale"] = _quant_int8(m)  # requant
            else:
                m = state["m"]
                m.lerp_(update.to(m.dtype), 1.0 - beta1)
                delta = m.float() if m.dtype != torch.float32 else m.clone()
        else:
            delta = update

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
