"""Gemini — AdEMAMix's two-EMA update fused onto koptim's memory-efficient backend.

Gemini (**provisional code name**) marries **AdEMAMix** (Pagliardini, Ablin &
Grangier, *The AdEMAMix Optimizer: Better, Faster, Older*, arXiv:2409.03137,
ICLR 2025) to the precision and memory machinery already proven in
:class:`~koptim.adafusion.Adafusion`. AdEMAMix's thesis is that a *single*
first-moment EMA cannot be both responsive to recent gradients and faithful to
old ones, so it keeps **two**: a fast EMA (``beta1`` ~ 0.9) and a very slow EMA
(``beta3`` ~ 0.9999) whose gradients stay relevant for *tens of thousands* of
steps, mixed into the step by a coefficient ``alpha`` (~5-10). A standard Adam
second moment normalizes the result. It is a separate class — Adafusion and
Liofusion are left byte-for-byte unchanged so the three can be A/B'd cleanly.

**The reference update (kozistr ``pytorch_optimizer`` AdEMAMix, full / not the
"simplified" variant), per parameter:**

.. code-block:: text

    alpha_t  = schedule_alpha(t, step, alpha)          # ramp 0 -> alpha
    beta3_t  = schedule_beta3(t, step, beta1, beta3)   # ramp beta1 -> beta3
    m1  = beta1  * m1 + (1 - beta1 ) * g               # FAST first-moment EMA
    m2  = beta3_t* m2 + (1 - beta3_t) * g              # SLOW first-moment EMA
    v   = beta2  * v  + (1 - beta2 ) * g^2             # second moment
    bc1 = 1 - beta1 ** step                            # m1 bias correction
    bc2 = sqrt(1 - beta2 ** step)                      # v  bias correction
    de_nom    = sqrt(v) / bc2 + eps
    update    = (m1 + alpha_t * m2) / de_nom
    p        -= (lr / bc1) * update                    # + decoupled weight decay

The slow EMA ``m2`` is **deliberately not** bias-corrected: it starts near zero
and is ramped in by the ``beta3``/``alpha`` warmup, so an early bias correction
would over-amplify a near-empty buffer (this matches the paper and the kozistr
reference).

**The koptim fusion (what Gemini changes vs the dense reference):**

* The second moment ``v`` uses Adafusion's **conv-aware factored** store (row+col
  EMAs of ``g^2``; ``ndim>2`` conv kernels reshape to ``[out, in*kh*kw]`` first),
  so ``v`` costs ~0 state. 1-D params keep a full per-coordinate ``v``.
  Reconstruction yields ``1/sqrt(v_hat)`` directly; the ``1/bc2`` factor is folded
  in as a multiply by ``bc2`` (``bc2/sqrt(v)``). ⚠️ One consequence: in the
  factored branch the additive ``eps`` of the dense ``sqrt(v)/bc2 + eps``
  denominator is **not** representable, so Gemini uses Adafusion's HF-Adafactor
  ``eps1`` convention instead (``eps1`` added to ``g^2`` *before* the factored
  means). The exposed ``eps`` is mapped to that ``eps1`` floor. The 1-D
  non-factored branch uses the same ``eps1`` floor for consistency. This is the
  one numerical deviation from the dense reference; everything else matches.

* The two first moments ``m1`` and ``m2`` are each stored through the shared
  **momentum codec** (``bfloat16`` ~2 B/param, ``int8`` ~1 B/param, ``4bit``
  ~0.5 B/param). TWO quantized buffers are the only memory cost over Adam's one;
  with the factored ``v`` the optimizer-state floor stays Adafactor-class. Gemini
  does the **raw-gradient** EMAs itself (dequant -> fp32 lerp -> requant), the
  same way Liofusion does — the codec's ``ema_*`` helpers do an Adam-style
  *momentum-of-the-update* and so cannot be reused verbatim here (AdEMAMix's EMAs
  are of the bare gradient, before the second-moment normalization). We reuse the
  codec by feeding it ``update := grad`` and ``beta1 := the momentum's beta``, so
  its ``m.lerp_(update, 1-beta1)`` IS the raw-gradient EMA we want.

* **Stochastic-rounding bf16 weight update** (``add_stochastic_``; ``kahan`` /
  ``none`` also available), **cautious masking** (Liang et al. 2024; on by
  default), **foreach batching** (bit-exact vs the per-parameter path, now over
  *two* momenta), 1-D handling and the **dtype-safe checkpoint**
  (:func:`load_state_dict_preserving_dtypes`, which now preserves ``m1``, ``m2``
  *and* the factored ``row``/``col`` of ``v``).

**Warmup schedulers.** ``alpha`` and ``beta3`` warmups share the paper's /
kozistr's schedule shapes but are exposed as two independent horizons
``alpha_warmup`` / ``beta3_warmup`` (steps). ``alpha`` ramps linearly ``0 ->
alpha``; ``beta3`` ramps ``beta1 -> beta3`` in log-(1-beta) space. If both are
``None`` (the default) the schedules are constant (``alpha_t = alpha``,
``beta3_t = beta3``) — i.e. no warmup, which the paper notes is fine for short
runs but recommends enabling (e.g. set both to the total step count) for long
ones to avoid early instability from a suddenly-large ``alpha``. Passing only one
enables just that ramp.

**lr is Adam-scale**, but note the slow EMA inflates the effective step, so
AdEMAMix typically wants a slightly smaller ``lr`` and/or larger weight decay
than plain Adam at the same ``alpha``.

⚠️ **Caveat for small-data diffusion fine-tuning.** AdEMAMix's headline property
is that it *slows forgetting* — old gradients keep pulling the weights. On a
large pretraining corpus that is the win. On a *small* fine-tuning set this same
property may **entrench memorization** of the training images (the slow EMA keeps
re-applying the same few gradients for tens of thousands of steps), hurting
generalization. Whether Gemini helps or hurts here is an open question to be
settled by the train/val gap + FID test; treat the slow EMA as a knob to tune,
not a free win, on small datasets.

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
from koptim._momentum_codec import (
    _FOURBIT_BLOCK,
    _make_codec,
    _MomentumCodec,
    load_state_dict_preserving_dtypes,
)
from koptim._stochastic_rounding import add_stochastic_

__all__ = ["Gemini"]

_LOW_PRECISION = (torch.bfloat16, torch.float16)
MomentumDtype = Literal["bfloat16", "float32", "int8", "4bit"]

# Mirror Adafusion's batching/memory knobs (see adafusion.py for the rationale).
_STACK_SAFETY_FRACTION = 0.10
_STACK_BYTES_PER_ELEM = 64  # a touch higher than Adafusion (48): two momenta stacked
_MIN_STACK_ELEMS = 262_144
_DEFAULT_STACK_ELEMS = 64_000_000
_FOREACH_BATCH_CUTOFF = 2_000_000


def _is_low_precision(t: Tensor) -> bool:
    return t.dtype in _LOW_PRECISION


def _rms(t: Tensor) -> Tensor:
    return t.norm(2) / math.sqrt(max(t.numel(), 1))


def schedule_alpha(t_warmup: float | None, step: int, alpha: float) -> float:
    """Linear ``alpha`` warmup ``0 -> alpha`` over ``t_warmup`` steps (paper / kozistr).

    ``None`` -> constant ``alpha`` (no warmup). Matches kozistr's
    ``min(step * alpha / t, alpha)``.
    """
    if t_warmup is None:
        return alpha
    return min(step * alpha / t_warmup, alpha)


def schedule_beta3(t_warmup: float | None, step: int, beta1: float, beta3: float) -> float:
    """``beta3`` warmup ``beta1 -> beta3`` in log-(1-beta) space (paper / kozistr).

    ``None`` -> constant ``beta3``. Matches kozistr's interpolation
    ``exp(log b1 * log b3 / ((1 - s) log b3 + s log b1))`` capped at ``beta3``,
    with ``s = step / t_warmup``.
    """
    if t_warmup is None:
        return beta3
    log_b1, log_b3 = math.log(beta1), math.log(beta3)
    s = step / t_warmup
    return min(
        math.exp(log_b1 * log_b3 / ((1.0 - s) * log_b3 + s * log_b1)),
        beta3,
    )


class _PrefixedState:
    """A thin mutable-mapping proxy that namespaces the codec's keys under a prefix.

    The shared momentum codecs in :mod:`koptim._momentum_codec` read and write a
    fixed key set (``m``, ``m_scale``, ``m_numel``, ``m_block``) on a per-param
    ``state`` dict. Gemini stores TWO independent momenta, so each is given its
    own ``<prefix>_<key>`` namespace inside the real param state, and the codec is
    handed this proxy so it operates on the right buffers unmodified. Only the
    handful of mapping ops the codecs use are implemented.
    """

    __slots__ = ("_state", "_prefix")

    def __init__(self, state: dict[str, Any], prefix: str) -> None:
        self._state = state
        self._prefix = prefix

    def _key(self, key: str) -> str:
        return f"{self._prefix}_{key}"

    def __getitem__(self, key: str) -> Any:
        return self._state[self._key(key)]

    def __setitem__(self, key: str, value: Any) -> None:
        self._state[self._key(key)] = value

    def __contains__(self, key: str) -> bool:
        return self._key(key) in self._state

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(self._key(key), default)


class Gemini(Optimizer):
    """AdEMAMix (two first-moment EMAs) on koptim's factored/quantized backend.

    Args:
        params: parameters or param-group dicts.
        lr: learning rate (Adam-scale, but see the module docstring — the slow EMA
            inflates the effective step, so a touch smaller / more WD often helps).
        betas: ``(beta1, beta2, beta3)``. ``beta1`` is the **fast** first-moment
            EMA decay (~0.9), ``beta2`` the second-moment decay (~0.999), ``beta3``
            the **slow** first-moment EMA decay (~0.9999 — gradients stay relevant
            for tens of thousands of steps). AdEMAMix's defaults
            ``(0.9, 0.999, 0.9999)``.
        alpha: mixing coefficient for the slow EMA (~5-10). The step uses
            ``m1 + alpha_t * m2``. ``alpha=0`` collapses to plain Adam on the
            factored backend.
        eps: stability floor. Used as Adafactor's ``eps1`` — added to ``g^2``
            before the factored second-moment reductions — NOT as the dense
            reference's additive denominator ``eps`` (not representable in the
            factored inv-sqrt form; see the module docstring).
        weight_decay: decoupled (AdamW-style) weight decay, folded into the
            per-step delta as ``lr * wd * p``.
        alpha_warmup: horizon (in steps) for the linear ``alpha`` ramp ``0 ->
            alpha``. ``None`` (default) -> constant ``alpha`` (no warmup).
        beta3_warmup: horizon (in steps) for the ``beta3`` ramp ``beta1 -> beta3``
            in log-(1-beta) space. ``None`` (default) -> constant ``beta3``. The
            paper recommends enabling both (commonly set to the total step count)
            for long runs to avoid early instability; short runs can leave them off.
        clip_threshold: optional Adafactor RMS update clipping (``rms(update) <=
            thr``). ``0`` (default) disables it — AdEMAMix has no such clip, so it
            is off by default to stay faithful; enable it (e.g. ``1.0``) for extra
            stability on the factored backend.
        cautious: cautious masking (Liang et al. 2024) — zero the update
            coordinates whose sign disagrees with the gradient, then rescale the
            survivors to preserve the mean step magnitude. **On by default.**
        momentum_dtype: storage for **both** first-moment buffers (``m1``, ``m2``)
            — ``"bfloat16"`` (default, ~2 B/param each), ``"float32"`` (4 B/param),
            ``"int8"`` (~1 B/param, per-row absmax), or ``"4bit"`` (~0.5 B/param,
            per-block absmax, nibble-packed). Same layout as Adafusion's first
            moment, so checkpoints resume bit-exactly.
        momentum_4bit_block: block size for ``momentum_dtype="4bit"``. Default
            ``128``. ``0``/negative -> whole-tensor (one scale).
        bf16_method: weight-update strategy for low-precision params —
            ``"stochastic_rounding"`` (default), ``"kahan"`` (+2 B/param), or
            ``"none"``. No-op on fp32 params.
        foreach: batch the step across parameters with stacked multi-tensor ops
            instead of a per-parameter Python loop. Default ``True``. Numerically
            matches the per-parameter path (over BOTH momenta);
            stochastic-rounding draws differ (unbiased either way). 0-D scalars,
            kahan, and fp16+SR fall back to the per-parameter path.
        foreach_batch_cutoff: per-tensor element count above which a weight loops
            instead of stacking (performance knob; default ``2_000_000``).
        foreach_stack_budget: max elements per stacked chunk. ``None`` (default)
            adapts to free VRAM; an int pins a fixed cap.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 5.0,
        eps: float = 1e-30,
        weight_decay: float = 0.0,
        *,
        alpha_warmup: float | None = None,
        beta3_warmup: float | None = None,
        clip_threshold: float = 0.0,
        cautious: bool = True,
        momentum_dtype: MomentumDtype = "bfloat16",
        momentum_4bit_block: int = _FOURBIT_BLOCK,
        bf16_method: str = "stochastic_rounding",
        foreach: bool = True,
        foreach_batch_cutoff: int = _FOREACH_BATCH_CUTOFF,
        foreach_stack_budget: int | None = None,
    ) -> None:
        beta1, beta2, beta3 = float(betas[0]), float(betas[1]), float(betas[2])
        for i, b in enumerate((beta1, beta2, beta3)):
            if not 0.0 <= b < 1.0:
                raise ValueError(f"betas[{i}] must be in [0, 1), got {b}")
        if lr < 0.0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if alpha < 0.0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        if clip_threshold < 0.0:
            raise ValueError(f"clip_threshold must be >= 0, got {clip_threshold}")
        if alpha_warmup is not None and alpha_warmup <= 0.0:
            raise ValueError(f"alpha_warmup must be > 0 or None, got {alpha_warmup}")
        if beta3_warmup is not None and beta3_warmup <= 0.0:
            raise ValueError(f"beta3_warmup must be > 0 or None, got {beta3_warmup}")
        if momentum_dtype not in ("bfloat16", "float32", "int8", "4bit"):
            raise ValueError(
                f"momentum_dtype must be bfloat16/float32/int8/4bit, got {momentum_dtype!r}"
            )
        if bf16_method not in ("stochastic_rounding", "kahan", "none"):
            raise ValueError(
                f"bf16_method must be stochastic_rounding/kahan/none, got {bf16_method!r}"
            )
        if foreach_batch_cutoff < 1:
            raise ValueError(f"foreach_batch_cutoff must be >= 1, got {foreach_batch_cutoff}")
        defaults = {
            "lr": lr,
            "betas": (beta1, beta2, beta3),
            "alpha": alpha,
            "eps": float(eps),
            "weight_decay": weight_decay,
            "alpha_warmup": alpha_warmup,
            "beta3_warmup": beta3_warmup,
            "clip_threshold": clip_threshold,
            "cautious": cautious,
            "momentum_dtype": momentum_dtype,
            "momentum_4bit_block": momentum_4bit_block,
            "bf16_method": bf16_method,
        }
        super().__init__(params, defaults)
        self._foreach = foreach
        self._foreach_batch_cutoff = foreach_batch_cutoff
        self._foreach_stack_budget = foreach_stack_budget
        # One momentum codec per dtype string (stateless beyond the dtype); shared
        # by m1 and m2 since they use the same storage layout.
        self._codecs: dict[str, _MomentumCodec] = {}

    def _codec(self, group: dict[str, Any]) -> _MomentumCodec:
        md = group["momentum_dtype"]
        codec = self._codecs.get(md)
        if codec is None:
            codec = self._codecs[md] = _make_codec(md)
        return codec

    @staticmethod
    def _codec_view(state: dict[str, Any], prefix: str) -> _PrefixedState:
        """A dict-like proxy mapping the codec's keys onto ``<prefix>_*`` of ``state``.

        Lets the shared codec be reused verbatim for two independent momenta.
        """
        return _PrefixedState(state, prefix)

    # ------------------------------------------------------------------- state
    @torch.no_grad()
    def _init_state(self, p: Tensor, state: dict[str, Any], group: dict[str, Any]) -> None:
        """Allocate v (factored row/col or full), the two momenta, and the step."""
        grad = p.grad
        state["step"] = 0
        factored = p.ndim >= 2
        if factored:
            gv = grad if p.ndim == 2 else grad.reshape(grad.shape[0], -1)
            row_shape = gv.shape[:-1]
            col_shape = gv.shape[:-2] + gv.shape[-1:]
            state["row"] = torch.zeros(row_shape, dtype=torch.float32, device=p.device)
            state["col"] = torch.zeros(col_shape, dtype=torch.float32, device=p.device)
            # Codec layout follows the matrixized gradient (so int8 per-row scales
            # align with the factored v's rows). For ndim==2 gv IS grad.
            codec_like = gv
        else:
            state["v"] = torch.zeros_like(grad, dtype=torch.float32)
            codec_like = grad
        codec = self._codec(group)
        codec.init_state(self._codec_view(state, "m1"), codec_like, group)
        codec.init_state(self._codec_view(state, "m2"), codec_like, group)
        if _is_low_precision(p) and group["bf16_method"] == "kahan":
            state["shift"] = torch.zeros_like(p)

    # -------------------------------------------------------------------- step
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
                    raise RuntimeError("Gemini does not support sparse gradients")
            if self._foreach and self._group_foreach_eligible(group):
                chunk_budget = self._foreach_budget(params[0].device)
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

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state, preserving the quantized dtype of BOTH momenta.

        torch's default ``load_state_dict`` upcasts every state tensor to the
        param dtype (fp32), which would inflate the bf16/int8/4bit ``m1`` and
        ``m2`` (and their scale/packed buffers) back to fp32 on resume. The shared
        helper walks every tensor in the per-param state generically, so the two
        namespaced momenta and the factored ``row``/``col`` of ``v`` are all
        restored to how they were checkpointed.
        """
        load_state_dict_preserving_dtypes(self, state_dict)

    # ----------------------------------------------------------------- foreach
    def _foreach_budget(self, device: torch.device) -> int:
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
        return group["bf16_method"] != "kahan"  # kahan needs a per-param shift buffer

    @staticmethod
    def _param_foreach_eligible(p: Tensor, group: dict[str, Any], cutoff: int) -> bool:
        if p.ndim == 0 or p.numel() > cutoff:
            return False
        if (
            group["bf16_method"] == "stochastic_rounding"
            and _is_low_precision(p)
            and p.dtype != torch.bfloat16  # fp16+SR is unsupported -> per-param (raises)
        ):
            return False
        if p.ndim > 2:
            # Matrixized conv writes back through a reshaped view -> needs contiguity.
            return p.data.is_contiguous() and p.grad.is_contiguous()
        return True

    @torch.no_grad()
    def _step_foreach(self, params: list[Tensor], group: dict[str, Any], budget: int) -> None:
        """Batched step: bucket by effective shape, then stack each bucket.

        ``ndim >= 2`` -> factored bucket keyed by effective 2-D shape ``[N, R, C]``;
        ``ndim == 1`` -> non-factored bucket keyed by length ``[N, L]``. Mirrors
        :meth:`_step_one_param` element-for-element (modulo SR draws).
        """
        factored_buckets: dict[tuple[Any, ...], list[Tensor]] = {}
        flat_buckets: dict[tuple[Any, ...], list[Tensor]] = {}
        for p in params:
            state = self.state[p]
            if not state:
                self._init_state(p, state, group)
            g = p.grad
            if g.ndim >= 2:
                matrixize = g.ndim > 2
                eff = (g.shape[0], g.numel() // g.shape[0]) if matrixize else tuple(g.shape)
                factored_buckets.setdefault((eff, p.dtype, matrixize), []).append(p)
            else:
                flat_buckets.setdefault((g.shape[0], p.dtype), []).append(p)

        for (eff, _dtype, matrixize), plist in factored_buckets.items():
            chunk = max(1, budget // max(eff[0] * eff[1], 1))
            for i in range(0, len(plist), chunk):
                self._factored_bucket(plist[i:i + chunk], eff, matrixize, group)
        for (length, _dtype), plist in flat_buckets.items():
            chunk = max(1, budget // max(length, 1))
            for i in range(0, len(plist), chunk):
                self._nonfactored_bucket(plist[i:i + chunk], length, group)

    # ------------------------------------------------------ shared math helper
    def _schedules(self, group: dict[str, Any], step: int) -> tuple[float, float, float, float]:
        """Per-step ``(alpha_t, beta3_t, bias_correction1, bias_correction2_sq)``."""
        beta1, beta2, beta3 = group["betas"]
        alpha_t = schedule_alpha(group["alpha_warmup"], step, group["alpha"])
        beta3_t = schedule_beta3(group["beta3_warmup"], step, beta1, beta3)
        bc1 = 1.0 - beta1 ** step
        bc2_sq = math.sqrt(1.0 - beta2 ** step)
        return alpha_t, beta3_t, bc1, bc2_sq

    def _ema_stacked(
        self,
        codec: _MomentumCodec,
        states: list[dict[str, Any]],
        prefix: str,
        grad: Tensor,
        mat: Any,
        eff: tuple[int, ...],
        beta: float,
    ) -> Tensor:
        """Stacked raw-gradient EMA ``m <- beta*m + (1-beta)*g`` via the codec.

        The codec's ``ema_stacked`` does ``m.lerp_(update, 1-beta1)`` and returns
        the fp32 result — exactly the EMA we want with ``update := grad`` and
        ``beta1 := beta``. Each momentum reads/writes its own prefixed sub-state.
        """
        views = [self._codec_view(s, prefix) for s in states]
        return codec.ema_stacked(views, grad, mat, eff, beta)

    @torch.no_grad()
    def _factored_bucket(
        self, plist: list[Tensor], eff: tuple[int, int], matrixize: bool, group: dict[str, Any]
    ) -> None:
        R, C = eff  # noqa: N806
        N = len(plist)  # noqa: N806
        beta1, beta2, _beta3 = group["betas"]
        eps1 = group["eps"]
        lr, clip, wd = group["lr"], group["clip_threshold"], group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]
        codec = self._codec(group)
        states = [self.state[p] for p in plist]
        step = states[0]["step"] + 1
        for s in states:
            s["step"] = step
        alpha_t, beta3_t, bc1, bc2_sq = self._schedules(group, step)

        def mat(t: Tensor) -> Tensor:
            return t.view(R, C) if matrixize else t

        grad = torch.stack([mat(p.grad).float() for p in plist])          # [N, R, C]

        # Factored second-moment EMA (HF eps1 placement).
        rows = [s["row"] for s in states]
        cols = [s["col"] for s in states]
        row = torch.stack(rows)                                           # [N, R]
        col = torch.stack(cols)                                           # [N, C]
        grad_sq = grad * grad
        if eps1 > 0:
            grad_sq = grad_sq.add_(eps1)
        omb2 = 1.0 - beta2
        row.lerp_(grad_sq.mean(dim=-1), omb2)
        col.lerp_(grad_sq.mean(dim=-2), omb2)
        torch._foreach_copy_(rows, list(row.unbind(0)))
        torch._foreach_copy_(cols, list(col.unbind(0)))
        # 1/sqrt(v_hat) = r_factor * c_factor.
        r_factor = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)   # [N, R, 1]
        c_factor = col.rsqrt().unsqueeze(-2)                                        # [N, 1, C]

        # Two raw-gradient first-moment EMAs (codec-stored), in fp32.
        m1 = self._ema_stacked(codec, states, "m1", grad, mat, (R, C), beta1)
        m2 = self._ema_stacked(codec, states, "m2", grad, mat, (R, C), beta3_t)

        # AdEMAMix numerator, then normalize (with bc2 folded into the denom).
        update = m1.add_(m2, alpha=alpha_t)                               # m1 + alpha_t*m2
        update = update.mul_(r_factor).mul_(c_factor).mul_(bc2_sq)        # / de_nom
        if clip > 0:
            rms = update.reshape(N, -1).norm(2, dim=1) / math.sqrt(R * C)
            update.div_(rms.div_(clip).clamp_(min=1.0).view(N, 1, 1))
        delta = update.mul_(lr / bc1)

        if wd != 0:
            p_fp32 = torch.stack([mat(p.data).float() for p in plist])
            delta = delta.add_(p_fp32, alpha=lr * wd)

        if cautious:
            mask = (delta * grad > 0).to(delta.dtype)
            denom = mask.reshape(N, -1).mean(dim=1).clamp_(min=1e-8).view(N, 1, 1)
            delta = delta.mul_(mask).div_(denom)

        pviews = [mat(p.data) for p in plist]
        weights = torch.stack(pviews)                                     # [N, R, C]
        self._apply_subtract_batched(weights, delta, bf16_method)
        torch._foreach_copy_(pviews, list(weights.unbind(0)))

    @torch.no_grad()
    def _nonfactored_bucket(
        self, plist: list[Tensor], length: int, group: dict[str, Any]
    ) -> None:
        N = len(plist)  # noqa: N806
        beta1, beta2, _beta3 = group["betas"]
        eps1 = group["eps"]
        lr, clip, wd = group["lr"], group["clip_threshold"], group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]
        codec = self._codec(group)
        states = [self.state[p] for p in plist]
        step = states[0]["step"] + 1
        for s in states:
            s["step"] = step
        alpha_t, beta3_t, bc1, bc2_sq = self._schedules(group, step)

        grad = torch.stack([p.grad.float() for p in plist])               # [N, L]
        vs = [s["v"] for s in states]
        v = torch.stack(vs)                                               # [N, L]
        grad_sq = grad * grad
        if eps1 > 0:
            grad_sq = grad_sq.add_(eps1)
        v.lerp_(grad_sq, 1.0 - beta2)
        torch._foreach_copy_(vs, list(v.unbind(0)))
        inv = v.rsqrt()                                                   # 1/sqrt(v)

        def ident(t: Tensor) -> Tensor:
            return t

        m1 = self._ema_stacked(codec, states, "m1", grad, ident, (length,), beta1)
        m2 = self._ema_stacked(codec, states, "m2", grad, ident, (length,), beta3_t)

        update = m1.add_(m2, alpha=alpha_t).mul_(inv).mul_(bc2_sq)        # (m1+a*m2)/de_nom
        if clip > 0:
            rms = update.norm(2, dim=1) / math.sqrt(length)
            update.div_(rms.div_(clip).clamp_(min=1.0).view(N, 1))
        delta = update.mul_(lr / bc1)

        if wd != 0:
            p_fp32 = torch.stack([p.data.float() for p in plist])
            delta = delta.add_(p_fp32, alpha=lr * wd)

        if cautious:
            mask = (delta * grad > 0).to(delta.dtype)
            denom = mask.mean(dim=1).clamp_(min=1e-8).view(N, 1)
            delta = delta.mul_(mask).div_(denom)

        pviews = [p.data for p in plist]
        weights = torch.stack(pviews)
        self._apply_subtract_batched(weights, delta, bf16_method)
        torch._foreach_copy_(pviews, list(weights.unbind(0)))

    @staticmethod
    def _apply_subtract_batched(weights: Tensor, delta_fp32: Tensor, bf16_method: str) -> None:
        if (
            _is_low_precision(weights)
            and bf16_method == "stochastic_rounding"
            and weights.dtype == torch.bfloat16
        ):
            add_stochastic_(weights, delta_fp32, alpha=-1.0)
        else:
            weights.sub_(delta_fp32.to(weights.dtype))

    # ---------------------------------------------------------- per-parameter
    @torch.no_grad()
    def _step_one_param(self, p: Tensor, group: dict[str, Any]) -> None:
        beta1, beta2, _beta3 = group["betas"]
        eps1 = group["eps"]
        lr, clip, wd = group["lr"], group["clip_threshold"], group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]
        codec = self._codec(group)

        state = self.state[p]
        if not state:
            self._init_state(p, state, group)
        state["step"] += 1
        step = state["step"]
        alpha_t, beta3_t, bc1, bc2_sq = self._schedules(group, step)

        grad = p.grad if p.grad.dtype == torch.float32 else p.grad.float()
        ndim = grad.ndim
        factored = ndim >= 2

        if factored:
            matrixize = ndim > 2
            gv = grad.reshape(grad.shape[0], -1) if matrixize else grad
            update_factored_state(gv, state["row"], state["col"], beta2, eps1)
            r_factor, c_factor = factored_inv_sqrt_factors(state["row"], state["col"])
            # Raw-gradient first-moment EMAs on the matrixized gradient (codec row
            # layout then aligns with the factored v's rows).
            m1 = codec.ema_one(self._codec_view(state, "m1"), gv, beta1)
            m2 = codec.ema_one(self._codec_view(state, "m2"), gv, beta3_t)
            update = m1.add_(m2, alpha=alpha_t)               # (R, C) effective
            update = update.mul_(r_factor).mul_(c_factor).mul_(bc2_sq)
            if matrixize:
                update = update.view_as(grad)
        else:
            v = state["v"]
            grad_sq = grad * grad
            if eps1 > 0:
                grad_sq.add_(eps1)
            v.lerp_(grad_sq, 1.0 - beta2)
            inv = v.rsqrt()
            m1 = codec.ema_one(self._codec_view(state, "m1"), grad, beta1)
            m2 = codec.ema_one(self._codec_view(state, "m2"), grad, beta3_t)
            update = m1.add_(m2, alpha=alpha_t).mul_(inv).mul_(bc2_sq)

        if clip > 0:
            update.div_((_rms(update) / clip).clamp_(min=1.0))
        delta = update.mul_(lr / bc1)

        if wd != 0:
            p_fp32 = p.data if p.dtype == torch.float32 else p.data.float()
            delta = delta.add_(p_fp32, alpha=lr * wd)

        if cautious:
            mask = (delta * grad > 0).to(delta.dtype)
            delta = delta.mul_(mask).div_(mask.mean().clamp_(min=1e-8))

        self._apply_subtract(p, delta, state, bf16_method)

    @staticmethod
    def _apply_subtract(
        p: Tensor, delta_fp32: Tensor, state: dict[str, Any], bf16_method: str
    ) -> None:
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
