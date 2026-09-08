"""AdaPNM — Adam + Positive-Negative Momentum on kaon's memory backend.

AdaPNM — the adaptive variant of **Positive-Negative Momentum** (Xie et al. 2021,
*Positive-Negative Momentum: Manipulating Stochastic Gradient Noise to Improve
Generalization*, ICML 2021, arXiv:2103.17182) — implemented on top of the
precision and memory machinery already proven in
:class:`~kaon.adakaon.Adakaon`. It is the **generalization-bucket**
optimizer: an *implicit regularizer* that improves flat-minima / train-val-gap
behaviour **without** the extra forward-backward of SAM. Keeping it a separate
class lets it be A/B'd against Adakaon cleanly (Adakaon is left byte-for-byte
unchanged).

(Developed under the provisional code name *Janus*.)

**The idea (PNM / AdaPNM).** Vanilla momentum averages away the stochastic
gradient noise. PNM instead maintains **two** momentum buffers and, on each step,
feeds the gradient to only **one** of them (alternating which), then forms the
update direction as a *positive-negative mix*

.. code-block:: text

    pn = ((1 + beta0) * m_pos  -  beta0 * m_neg) / noise_norm
    noise_norm = sqrt((1 + beta0)^2 + beta0^2)

The ``(1 + beta0)`` / ``-beta0`` coefficients **amplify the momentum signal and
enlarge the variance of the injected gradient noise** in a controlled way; the
larger, more isotropic noise is what biases SGD toward flatter minima (better
generalization). ``noise_norm`` renormalizes so the *effective* step magnitude is
preserved (the two coefficients have squared-norm ``(1+beta0)^2 + beta0^2``).
AdaPNM divides that pos-neg numerator by the Adam second-moment denominator
``sqrt(v_hat) + eps`` — so it fits the factored-``v`` framework directly.

**The exact update (matches kozistr ``pytorch_optimizer`` ``AdaPNM``, defaults
``ams_bound=False`` here — see below):**

.. code-block:: text

    # per group, step t (1-indexed); alternate which buffer is "positive":
    t odd : (m_pos, m_neg) = (exp_avg,      neg_exp_avg)
    t even: (m_pos, m_neg) = (neg_exp_avg,  exp_avg)

    beta1_sq = beta1 ** 2                      # NOTE: momentum decay is beta1^2
    m_pos   = beta1_sq * m_pos + (1 - beta1_sq) * grad     # only m_pos sees grad
    v       = beta2 * v + (1 - beta2) * grad^2             # Adam second moment

    bc1     = 1 - beta1 ** t                   # bias corrections use beta1 (not ^2),
    bc2_sq  = sqrt(1 - beta2 ** t)             #   matching kozistr exactly
    denom   = sqrt(v_hat) + eps                # v_hat = v / bc2 (AdaPNM folds bc2 in)
    pn      = ((1 + beta0) * m_pos - beta0 * m_neg) / noise_norm
    p      -= (lr / bc1) * pn / denom

Two subtleties carried over verbatim from kozistr: (1) the **first-moment decay
is ``beta1**2``** (their ``beta1_p2``), while the **bias correction uses
``beta1``** (their ``debias(beta1, step)``); (2) only the *positive* buffer is
EMA-updated each step — the negative buffer is the *stale* (one-step-old, because
of the alternation) momentum that gets subtracted. ``beta0`` is kozistr's
``beta3`` (the pos-neg coefficient); their default ``beta3 = 1.0`` gives
``pn = (2*m_pos - m_neg)/sqrt(5)``.

**The factored second moment.** ``v`` reuses Adakaon's backend exactly:
``ndim >= 2`` weights factor ``v`` into row+column EMAs (conv kernels matrixized
to ``[out, in*kh*kw]`` first), ``ndim == 1`` keeps a full per-coordinate ``v``.
The denominator is Adakaon's ``r_factor * c_factor`` reconstruction of
``1/sqrt(v_hat)`` with the same RMS-clip; ``eps`` is added Adafactor-style via the
factored ``eps1`` (the kozistr scalar ``eps`` on the denominator has no factored
analogue, so it is exposed as ``eps`` and applied on the non-factored path /
folded into ``eps1`` — documented under ``eps``). Bias correction ``bc2`` is
applied by scaling the reconstructed inverse-denominator.

**AMSGrad / ``ams_bound``.** kozistr's AdaPNM defaults ``ams_bound=True`` (a
running element-wise max of ``v``). A *factored* ``v`` has no materialized matrix
to take a max over (``max`` of two rank-1 reconstructions is not rank-1), so
AMSBound cannot be applied to 2-D weights without giving up the factoring that is
the whole memory story. AdaPNM therefore **defaults ``ams_bound=False``** and, when
enabled, applies the running max only on the **non-factored (1-D) path** (where
``v`` is full); on factored weights it is silently a no-op. This is the one
deliberate deviation from the kozistr default, made for the factored backend; the
1-D path then matches kozistr's AdaPNM exactly.

**Cautious masking and the pos-neg direction.** Cautious (Liang et al. 2024) zeroes
the update coordinates whose sign disagrees with the *current* gradient
(``delta * g <= 0``), rescaling survivors to preserve the mean magnitude — the same
semantics Adakaon/Lion use, applied here to the **final** ``delta`` (the
pos-neg-mixed, denominator-divided, WD-folded step) against the raw gradient. Note
the tension: PNM's amplified ``-beta0 * m_neg`` term is *designed* to let the step
oppose the instantaneous gradient on noisy coordinates (that is the
noise-manipulation mechanism). Cautious masking removes exactly those coordinates,
so it partially damps the implicit regularizer. We keep ``cautious=True`` by default
for consistency with the rest of kaon and because the rescale preserves step
size, but **this is the knob to ablate first** when measuring AdaPNM's
generalization benefit — try ``cautious=False`` to let the pos-neg mechanism run
unmasked. Like Adakaon, with momentum effectively always on here the mask is not
a no-op.

**What is reused vs new.** Reused from Adakaon's backend: the factored
second-moment helpers (:mod:`kaon._factored`), the momentum **storage layout**
and quant/dequant primitives in :mod:`kaon._momentum_codec` (int8 per-row
absmax; 4-bit per-block absmax, nibble-packed), the stochastic-rounding bf16
weight update (:func:`kaon._stochastic_rounding.add_stochastic_`),
``load_state_dict_preserving_dtypes`` for dtype-safe checkpoint resume, and the
bucketed foreach batching pattern. New here: **two** momentum buffers with the
PNM alternation + positive-negative mixing and ``noise_norm`` renormalization, the
``beta1**2`` first-moment decay with ``beta1`` bias correction, and the
read-it-yourself EMA. AdaPNM uses the shared codec's ``dequant_*`` and
``store_*`` entry points around its raw-gradient EMA on the positive buffer.

It is a standard ``torch.optim.Optimizer`` with a single per-parameter step, so it
drops into per-parameter / gradient-release training loops unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, Literal

import torch
from torch import Tensor
from torch.optim import Optimizer

from kaon._autolr import DEFAULT_FUSE_REL, AutoLRMixin
from kaon._backend import (
    FOREACH_BATCH_CUTOFF,
    cautious_batched_,
    cautious_one_,
    centralize_grads_,
    flat_view,
    foreach_budget,
    is_low_precision,
    rms,
    subtract_batched_,
    subtract_one_,
)
from kaon._factored import factored_inv_sqrt_factors, update_factored_state
from kaon._foreach_plan import WatchedStateMixin, state_generation
from kaon._momentum_codec import (
    _FOURBIT_BLOCK,
    _make_codec,
    fourbit_block_size,
    load_state_dict_preserving_dtypes,
    warn_if_4bit_high_beta1,
)

# The fused routing safeguards below are optimizer-agnostic and already proven on Adakaon (audit
# batch A): the per-step non-contiguous-grad demotion, and the shape x dtype x DEVICE bucketing
# the big route's pointer arrays need. Reused rather than duplicated — ``kaon.adakaon`` imports
# nothing from here, so there is no cycle, and a divergence between two copies of this logic is
# exactly the class of bug the audit found.
from kaon.adakaon import _demote_non_contiguous_grads, _same_shape_device_buckets

__all__ = ["AdaPNM"]

MomentumDtype = Literal["bfloat16", "float32", "int8", "4bit"]

# Performance / memory knobs mirror Adakaon (see that module for the rationale).
_STACK_BYTES_PER_ELEM = 64  # two momenta + factored v: a touch above Adakaon's 48

#: ``dict.__setitem__``, i.e. a per-param state write that SKIPS the state-identity
#: hook (:class:`kaon._foreach_plan.WatchedParamState`). Legal for scalar bookkeeping
#: keys ONLY — never for a tensor buffer. See :meth:`AdaPNM._prepare_param_steps`.
_set_unwatched = dict.__setitem__


def _rms_clip_one_(u: Tensor, clip: float) -> Tensor:
    """Adafactor RMS clip on a single normalized update: ``rms(u) <= clip``. In place."""
    if clip > 0.0:
        u.div_((rms(u) / clip).clamp_(min=1.0))
    return u


def _rms_clip_batched_(u: Tensor, clip: float) -> Tensor:
    """Per-slice RMS clip on a stacked ``[N, *shape]`` normalized update. In place."""
    if clip > 0.0:
        n = u.shape[0]
        per = max(u[0].numel(), 1) if n else 1
        rms = u.reshape(n, -1).norm(2, dim=1) / math.sqrt(per)
        u.div_(rms.div_(clip).clamp_(min=1.0).view(n, *([1] * (u.ndim - 1))))
    return u


# --------------------------------------------------------------------- diagnostics
# Env-gated, zero-overhead-when-off stability probe used to diagnose the real-training
# NaN divergence. Set KAON_PROBE_LOG=/path/to/log to enable. It records, per group step:
#   * the worst factored denominator multiplier  max(r_factor * c_factor * bc2_sq)  and which
#     param produced it (the suspected blowup channel: a near-zero col-EMA -> huge rsqrt), and
#   * the FIRST non-finite parameter, with its shape, fused routing (one-block/big/native),
#     and the offending channel's row/col/momentum stats — i.e. the exact culprit, not a guess.
import os  # noqa: E402

_PROBE_LOG = os.environ.get("KAON_PROBE_LOG")
_PROBE_EVERY = int(os.environ.get("KAON_PROBE_EVERY", "25"))


def _probe_write(line: str) -> None:
    with open(_PROBE_LOG, "a") as fh:  # noqa: SIM115 — short append, diagnostics only
        fh.write(line + "\n")


@torch.no_grad()
def _probe_group(opt: AdaPNM, group: dict[str, Any]) -> None:
    """Inspect every factored param after a step: worst denom multiplier + first non-finite."""
    step = group["step"]
    routing = _probe_routing(opt, group)
    worst_mult, worst_shape = 0.0, None
    for p in group["params"]:
        st = opt.state.get(p)
        if not st or "col" not in st:
            continue
        bc2_sq = opt._coeffs(group, st.get("step", step))["bc2_sq"]
        col = st["col"]
        row = st["row"]
        cfac_max = col.clamp_min(1e-30).rsqrt().max().item()
        rfac_max = row.div(row.mean().clamp_min(1e-30)).clamp_min(1e-30).rsqrt().max().item()
        mult = rfac_max * cfac_max * bc2_sq
        if mult > worst_mult:
            worst_mult, worst_shape = mult, tuple(p.shape)
        if not torch.isfinite(p).all():
            _probe_write(
                f"[NONFINITE] step={step} shape={tuple(p.shape)} route={routing.get(id(p),'?')} "
                f"col_min={col.min().item():.3e} col_max={col.max().item():.3e} "
                f"cfac_max={cfac_max:.3e} rfac_max={rfac_max:.3e} denom_mult={mult:.3e} "
                f"p_absmax={p.detach().abs().float().amax().item():.3e} "
                f"grad_absmax={(p.grad.detach().abs().float().amax().item() if p.grad is not None else float('nan')):.3e}"
            )
    if step == 1 or step % _PROBE_EVERY == 0:
        _probe_write(f"[denom] step={step} worst_mult={worst_mult:.3e} shape={worst_shape}")


def _native_reason(p: Tensor, md: str, bf16m: str, cap: int, ft: Any) -> str:
    """Why does this param miss the fused path? (the 'falling to native' census)."""
    if p.ndim != 2:
        return f"ndim={p.ndim}(1d/conv)"
    if not p.is_cuda:
        return "not_cuda"
    if not p.is_contiguous():
        return "non_contiguous"
    if p.dtype not in (torch.float32, torch.bfloat16):
        return f"dtype={p.dtype}"
    if p.dtype == torch.bfloat16 and bf16m != "stochastic_rounding":
        return f"bf16_method={bf16m}"
    if md == "4bit" and p.shape[1] % 2 != 0:
        return "4bit_odd_C"
    br, bc = ft.next_pow2_tile(*p.shape)
    if br * bc > cap:
        return f"tile>{cap}({br}x{bc})"
    return "unknown"


def _probe_census(one_block: list, big: list, native: list, md: str, bf16m: str, cap: int, ft: Any) -> None:
    from collections import Counter
    reasons = Counter(_native_reason(p, md, bf16m, cap, ft) for p in native)
    shapes_nat = Counter(tuple(p.shape) for p in native)
    _probe_write(
        f"[census] one_block={len(one_block)} big={len(big)} native={len(native)} "
        f"native_reasons={dict(reasons)} native_shapes={dict(shapes_nat)}"
    )
    _probe_write(f"[census] one_block_shapes={dict(Counter(tuple(p.shape) for p in one_block))}")
    _probe_write(f"[census] big_shapes={dict(Counter(tuple(p.shape) for p in big))}")


def _probe_routing(opt: AdaPNM, group: dict[str, Any]) -> dict[int, str]:
    """Map id(param) -> 'one_block' | 'big' | 'native' from the cached fused partition."""
    out: dict[int, str] = {}
    if not getattr(opt, "_fused", False):
        return out
    cached = opt._fused_part.get(id(group))
    if cached is None:
        return out
    one_block, big, one_dim, native = cached[-4:]   # leading fields are cache keys
    for p in one_block:
        out[id(p)] = "one_block"
    for p in big:
        out[id(p)] = "big"
    for p in one_dim:
        out[id(p)] = "one_dim"
    for p in native:
        out[id(p)] = "native"
    return out


class AdaPNM(AutoLRMixin, WatchedStateMixin, Optimizer):
    """AdaPNM (Adam + Positive-Negative Momentum) on Adakaon's memory backend.

    Args:
        params: parameters or param-group dicts.
        lr: learning rate.
        betas: ``(beta1, beta2)``. ``beta1`` is the first-moment decay — note the
            actual EMA decay is ``beta1**2`` (matching kozistr's AdaPNM), while the
            bias correction uses ``beta1``. ``beta2`` is the (factored) second-moment
            decay. Default ``(0.8, 0.999)``. **``beta1`` is the loss↔gap dial**: on
            the synthetic proxy the loss is U-shaped in ``beta1`` and bottoms at
            ``0.8`` while the train–val gap stays low; ``0.9`` (the usual Adam value)
            is measurably worse here, and below ``~0.7`` the gap climbs with no loss
            gain. ``beta2=0.999`` is the sweet spot. Raise ``beta1`` toward ``0.95``
            for more regularization (lower gap, higher loss).
        beta0: the **positive-negative momentum coefficient** (kozistr's ``beta3``).
            The update direction is ``((1+beta0)*m_pos - beta0*m_neg)/noise_norm``
            with ``noise_norm = sqrt((1+beta0)^2 + beta0^2)``. ``beta0`` must be in
            ``[0, 1]``. ``beta0=0`` collapses to plain (debiased) Adam-momentum (the
            PNM noise-injection is then off — measurably worse on the proxy, so PNM is
            load-bearing); ``beta0=1`` is the canonical PNM ``(2*m_pos-m_neg)/sqrt(5)``.
            Default ``0.5`` (the measured sweet spot — best loss/gap on the proxy).
        eps: term added to the second-moment denominator for stability. On the
            non-factored (1-D) path it is added to ``sqrt(v_hat)`` exactly as
            kozistr does. On the factored path it is folded into the Adafactor
            ``eps1`` (added to ``grad**2`` before the row/col reductions).
        clip_threshold: Adafactor-style RMS clip on the (v_hat-normalized) update —
            ``rms(pn / sqrt(v_hat)) <= clip_threshold`` before the lr scale, exactly as
            Adakaon. **On by default (``1.0``).** This is the stability guard for the
            factored denominator: a near-zero ``col`` EMA makes ``c_factor = rsqrt(col)``
            explode (~1e4), so a fresh gradient on that channel produces an unbounded
            step → NaN. The clip bounds that runaway (measured: real Cosmos LoKr
            training diverged to NaN without it). ``0`` disables the clip (the original
            unclamped PNM update — diverges on real diffusion training, kept only for
            ablation). Set looser (e.g. ``> 1``) to recover more of the raw PNM step if
            a generalization measurement shows the clip costs gap.
        weight_decay: decoupled (AdamW-style) weight decay. Applied multiplicatively
            ``p *= (1 - lr*weight_decay)`` *before* the moment updates, matching
            kozistr's ``weight_decouple=True`` default (not folded into the cautious
            delta — so cautious does not gate weight decay, unlike Adakaon).
        cautious: cautious masking (Liang et al. 2024) on the final pos-neg step vs
            the gradient. **On by default.** See the class docstring: it interacts
            with — and partially damps — PNM's noise-manipulation mechanism; ablate
            it first when measuring generalization.
        ams_bound: AMSGrad-style running max of ``v``. **Off by default** (kozistr's
            AdaPNM defaults it on, but a factored ``v`` cannot be max'd). When on, it
            applies only to the non-factored 1-D path; a no-op on factored weights.
        momentum_dtype: storage for **both** momentum buffers — ``"bfloat16"``
            (default, ~2 B/param each), ``"float32"`` (4 B/param each), ``"int8"``
            (~1 B/param each, per-row absmax), or ``"4bit"`` (~0.5 B/param each,
            per-block absmax, nibble-packed). Same layout as Adakaon's first
            moment, so checkpoints resume bit-exactly via ``load_state_dict``. Note
            AdaPNM carries *two* momenta, so its momentum floor is ~2x a
            single-momentum optimizer at the same dtype (the price of PNM).
        momentum_4bit_block: block size for ``momentum_dtype="4bit"``. Default
            ``128``. ``0``/negative means whole-tensor.
        bf16_method: weight-update strategy for low-precision params —
            ``"stochastic_rounding"`` (default), ``"kahan"`` (+2 B/param), or
            ``"none"``. No-op on fp32 params.
        foreach: batch the step across parameters with stacked multi-tensor ops.
            Default ``True``. Numerically matches the per-parameter path
            (stochastic-rounding draws differ, unbiased either way). 0-D scalars,
            kahan, and fp16+SR fall back to the per-parameter path.
        foreach_batch_cutoff: per-tensor element count above which a weight loops
            instead of stacking (a performance knob; default ``2_000_000``).
        foreach_stack_budget: max elements per stacked chunk. ``None`` (default)
            adapts to free VRAM; an int pins a fixed cap.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.8, 0.999),
        beta0: float = 0.5,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        clip_threshold: float = 1.0,
        cautious: bool = True,
        gradient_centralization: bool = True,
        ams_bound: bool = False,
        momentum_dtype: MomentumDtype = "bfloat16",
        momentum_4bit_block: int = _FOURBIT_BLOCK,
        bf16_method: str = "stochastic_rounding",
        foreach: bool = True,
        foreach_batch_cutoff: int = FOREACH_BATCH_CUTOFF,
        foreach_stack_budget: int | None = None,
        fused: bool = False,
        fused_tile_cap: int | None = None,
        auto_lr: bool = False,
        auto_lr_scale: float = 1.0,
        auto_lr_fuse_rel: float = DEFAULT_FUSE_REL,
        auto_lr_d0: float | None = None,
    ) -> None:
        beta1, beta2 = float(betas[0]), float(betas[1])
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"betas[0] must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"betas[1] must be in [0, 1), got {beta2}")
        if not 0.0 <= beta0 <= 1.0:
            raise ValueError(f"beta0 must be in [0, 1], got {beta0}")
        if lr < 0.0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if eps < 0.0:
            raise ValueError(f"eps must be >= 0, got {eps}")
        if clip_threshold < 0.0:
            raise ValueError(f"clip_threshold must be >= 0, got {clip_threshold}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
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
        warn_if_4bit_high_beta1(beta1, momentum_dtype)  # beta1 = betas[0], the momentum EMA decay
        defaults = {
            "lr": lr,
            "betas": (beta1, beta2),
            "beta0": float(beta0),
            "eps": float(eps),
            "clip_threshold": float(clip_threshold),
            "weight_decay": weight_decay,
            "cautious": cautious,
            "gradient_centralization": gradient_centralization,
            "ams_bound": ams_bound,
            "momentum_dtype": momentum_dtype,
            "momentum_4bit_block": momentum_4bit_block,
            "bf16_method": bf16_method,
            "step": 0,
        }
        super().__init__(params, defaults)
        # ``self.state`` reports every change of state IDENTITY, so no cross-step cache can
        # keep stepping a retired momentum / second moment (see
        # ``kaon._foreach_plan.WatchedState``). Zero cost per step.
        self._install_state_watch()
        self._foreach = foreach
        self._foreach_batch_cutoff = foreach_batch_cutoff
        self._foreach_stack_budget = foreach_stack_budget
        # Optional Triton-fused step (same math + state). Eligible 2-D weights run through the fused
        # PNM kernels (one-block / chunked); everything else falls back to the native path, in-place
        # on the SAME state, so fused and non-fused interoperate and resume from each other.
        self._fused = bool(fused)
        # When True, the many-same-shape big regime (>tile_cap) runs the batched chunked kernel; set
        # False to revert to the batched-native-foreach path (the A/B baseline). See _fused_big.
        self._fused_big_batched = True
        # EXPERIMENTAL (candidate #4): fuse the batched-big reductions into Triton (grad via pointer
        # array, no [N,R,C] stack, GC in-kernel). Default False until the A/B confirms a win.
        self._fused_reductions = True
        # group id -> (param witness, state-identity generation, one_block, big, one_dim,
        # native). The two leading fields are the cache KEY; read the routes off the END.
        self._fused_part: dict[int, tuple] = {}
        self._fused_ob_caches: dict[tuple[int, int], Any] = {}
        self._fused_od_caches: dict[tuple[int, int], Any] = {}
        self._fused_big_caches: dict[tuple, Any] = {}
        # (group id, lag) -> (the ``big`` list the buckets were split from, the buckets)
        self._fused_big_buckets: dict[tuple[int, int], tuple[list, list]] = {}
        self._fused_demoted: dict[int, tuple] = {}
        if self._fused:
            from kaon._fused_triton import HAS_TRITON, TILE_CAP
            if not HAS_TRITON:
                raise RuntimeError("AdaPNM(fused=True) requires Triton (a GPU-only optional dependency)")
            # 2-D one_block/chunked crossover ONLY; the ndim<=1 ceiling is ft.TILE_CAP_1D.
            self._fused_tile_cap = TILE_CAP if fused_tile_cap is None else fused_tile_cap

        # Composable parameter-free LR (continuous Mechanic) via AutoLRMixin. off -> zero overhead.
        self._init_autolr(auto_lr, auto_lr_scale, auto_lr_fuse_rel, auto_lr_d0)

    def _invalidate_fused_caches(self) -> None:
        """Drop every host-side cache that may retain pointers into optimizer state."""
        self._fused_part.clear()
        self._fused_ob_caches.clear()
        self._fused_od_caches.clear()
        self._fused_big_caches.clear()
        self._fused_big_buckets.clear()
        self._fused_demoted.clear()

    def _autolr_reset_base_state(self) -> None:
        """Reset AdaPNM's base optimizer after an AutoLR rollback/contact."""
        self.state.clear()
        for group in self.param_groups:
            group["step"] = 0
        self._invalidate_fused_caches()

    # ------------------------------------------------------------------- state
    @torch.no_grad()
    def _alloc_momentum(
        self, prefix: str, grad: Tensor, state: dict[str, Any], group: dict[str, Any]
    ) -> None:
        """Allocate one momentum buffer (keys ``f"{prefix}"``, ``f"{prefix}_scale"`` …).

        Storage layout matches :mod:`kaon._momentum_codec` exactly (per-row int8
        scale; per-block 4-bit scale, zero == nibble 8) so the two momenta resume
        bit-exactly via ``load_state_dict_preserving_dtypes``.
        """
        md = group["momentum_dtype"]
        if md in ("bfloat16", "float32"):
            dtype = torch.bfloat16 if md == "bfloat16" else torch.float32
            state[prefix] = torch.zeros_like(grad, dtype=dtype)
        elif md == "int8":
            state[prefix] = torch.zeros_like(grad, dtype=torch.int8)
            state[f"{prefix}_scale"] = torch.ones(
                (grad.shape[0],) + (1,) * (grad.ndim - 1) if grad.ndim >= 2 else (),
                dtype=torch.float32, device=grad.device,
            )
        else:  # 4bit
            numel = grad.numel()
            bs = fourbit_block_size(grad, group)
            nblocks = (numel + bs - 1) // bs
            state[prefix] = torch.full(
                ((numel + 1) // 2,), 0x88, dtype=torch.uint8, device=grad.device
            )
            state[f"{prefix}_scale"] = torch.ones(nblocks, dtype=torch.float32, device=grad.device)
            state[f"{prefix}_numel"] = numel
            state[f"{prefix}_block"] = bs

    @torch.no_grad()
    def _init_state(self, p: Tensor, state: dict[str, Any], group: dict[str, Any]) -> None:
        grad = p.grad
        state["step"] = 0
        factored = p.ndim >= 2
        if factored:
            gv = grad if p.ndim == 2 else grad.reshape(grad.shape[0], -1)
            row_shape = gv.shape[:-1]
            col_shape = gv.shape[:-2] + gv.shape[-1:]
            state["row"] = torch.zeros(row_shape, dtype=torch.float32, device=p.device)
            state["col"] = torch.zeros(col_shape, dtype=torch.float32, device=p.device)
        else:
            state["v"] = torch.zeros_like(grad, dtype=torch.float32)
            if group["ams_bound"]:
                state["max_v"] = torch.zeros_like(grad, dtype=torch.float32)
        # Two momenta (pos / neg), each through the shared codec layout.
        self._alloc_momentum("m_pos", grad, state, group)
        self._alloc_momentum("m_neg", grad, state, group)
        if is_low_precision(p) and group["bf16_method"] == "kahan":
            state["shift"] = torch.zeros_like(p)

    # -------------------------------------------------- momentum read / write
    @staticmethod
    def _codec_state(state: dict[str, Any], prefix: str) -> dict[str, Any]:
        """Alias an AdaPNM-prefixed momentum as the shared codec's ``m`` keys."""
        alias = {"m": state[prefix]}
        for suffix in ("scale", "numel", "block"):
            key = f"{prefix}_{suffix}"
            if key in state:
                alias[f"m_{suffix}"] = state[key]
        return alias

    @staticmethod
    def _dequant_one(state: dict[str, Any], prefix: str, md: str, like: Tensor) -> Tensor:
        alias = AdaPNM._codec_state(state, prefix)
        return _make_codec(md).dequant_one(alias, like).reshape_as(like)

    @staticmethod
    def _store_one(state: dict[str, Any], prefix: str, md: str, m_fp32: Tensor) -> None:
        _make_codec(md).store_one(AdaPNM._codec_state(state, prefix), m_fp32)

    @staticmethod
    def _dequant_stacked(
        states: list[dict[str, Any]], prefix: str, md: str, shape: tuple[int, ...]
    ) -> Tensor:
        aliases = [AdaPNM._codec_state(state, prefix) for state in states]
        mat = lambda tensor: tensor.reshape(shape)  # noqa: E731
        return _make_codec(md).dequant_stacked(aliases, mat, shape)

    @staticmethod
    def _store_stacked(
        states: list[dict[str, Any]], prefix: str, md: str, m_fp32: Tensor
    ) -> None:
        aliases = [AdaPNM._codec_state(state, prefix) for state in states]
        _make_codec(md).store_stacked(aliases, m_fp32)

    def _prepare_param_steps(self, params: list[Tensor], group: dict[str, Any]) -> None:
        """Initialize state and advance each parameter's bias-correction counter.

        The counter is written through ``dict.__setitem__``, NOT ``state[...] = ...``, and
        that is a MEASURED choice. This is the only per-step state write anywhere in kaon
        and it runs once per parameter, so it is the whole per-step cost of the state
        identity guard on AdaPNM (Adakaon writes no state per step and pays nothing).
        Paired against the pre-guard form on the 428-param bag, 31 pairs of 400 reps,
        ``benchmarks/fused/bench_state_witness.py --case writes``. RANGES SPAN TWO MACHINES
        (the ordering and the conclusion were identical on both; the absolutes differ by
        ~2x, so treat the percentages of the same machine's own step as the number):

            through the hook (``state["step"] += 1``)   +56 … +107 µs   1.1 … 1.7% of the step
            ``dict.__setitem__`` (SHIPPED)              +18.5 … +48 µs  0.35 … 0.9%

        of which the whole remainder is the unbound-METHOD CALL, not the dict being a
        subclass: the same call form on a PLAIN dict costs the same, and a subclass read or
        C-level write is bit-for-bit the same slot as ``dict``'s. So that is the floor for
        any per-param write once the hook exists; the only cheaper option is not having the
        hook, which drops the ``state[p]["m"] = ...`` / ``state[p].clear()`` coverage the
        guard exists for. On the slower machine it is still over the 0.5% budget, which is
        reported rather than papered over.

        Bypassing the hook is sound *for this key and only for this key*: ``step`` is an
        int, it is not in :data:`~kaon._foreach_plan.WATCHED_STATE_KEYS`, and no cache
        anywhere bakes it, so the hook would have nothing to report.
        ``tests/test_state_identity_witness.py`` pins that invariant, and a TENSOR buffer
        must never be written this way.
        """
        set_raw = _set_unwatched            # LOAD_FAST in the per-param loop
        for p in params:
            state = self.state[p]
            if not state:
                self._init_state(p, state, group)
            elif "step" not in state:
                # The group counter was already advanced for this update. A legacy
                # state with moments therefore finished the preceding group step.
                set_raw(state, "step", group["step"] - 1)
            set_raw(state, "step", state["step"] + 1)

    @staticmethod
    def _pos_neg_prefixes(step: int) -> tuple[str, str]:
        """Which stored buffer plays positive / negative this step (the alternation).

        On odd steps ``m_pos`` is positive; on even steps the roles swap, so the
        buffer that received the gradient last step becomes the (stale) negative
        momentum that is subtracted. ``step`` is 1-indexed (incremented before use).
        """
        return ("m_pos", "m_neg") if step % 2 == 1 else ("m_neg", "m_pos")

    # -------------------------------------------------------------------- step
    @torch.no_grad()
    def _step_impl(self, closure: Any = None) -> Any:
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
                    raise RuntimeError("AdaPNM does not support sparse gradients")
            if not params:
                continue
            group["step"] += 1
            self._prepare_param_steps(params, group)
            if group["gradient_centralization"]:
                centralize_grads_(params)
            self._native_dispatch(params, group)
            if _PROBE_LOG:
                _probe_group(self, group)
        return loss

    @torch.no_grad()
    def _native_dispatch(self, params: list[Tensor], group: dict[str, Any]) -> None:
        """The native (non-fused) step over ``params`` — foreach where eligible, else per-param.
        Gradient Centralization is the caller's responsibility (done per-subset)."""
        if not params:
            return
        if self._foreach and self._group_foreach_eligible(group):
            chunk_budget = foreach_budget(self._foreach_stack_budget, self._foreach_batch_cutoff, _STACK_BYTES_PER_ELEM, params[0].device)
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

    # ----------------------------------------------------------- fused (Triton) step
    @torch.no_grad()
    def _fused_step(self, loss: Any) -> Any:
        import kaon._fused_triton as ft

        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            for p in params:
                if p.grad.is_sparse:
                    raise RuntimeError("AdaPNM does not support sparse gradients")
            if not params:
                continue
            group["step"] += 1
            self._prepare_param_steps(params, group)
            pos_pref, neg_pref = self._pos_neg_prefixes(group["step"])
            parts = self._fused_partition(group, params, ft)
            # Routing is cached; grad CONTIGUITY is not cacheable (fresh tensor every backward).
            one_block, big, one_dim, native = self._fused_demote(id(group), parts)
            if native:
                if group["gradient_centralization"]:
                    centralize_grads_(native)
                self._native_dispatch(native, group)
            group_step = group["step"]
            one_block_buckets = self._local_step_buckets(one_block, group_step)
            big_buckets = self._local_step_buckets(big, group_step)
            one_dim_buckets = self._local_step_buckets(one_dim, group_step)
            gid = id(group)
            self._fused_ob_caches = self._prune_lag_caches(
                self._fused_ob_caches, gid, set(one_block_buckets))
            self._fused_od_caches = self._prune_lag_caches(
                self._fused_od_caches, gid, set(one_dim_buckets))
            self._fused_big_caches = self._prune_lag_caches(
                self._fused_big_caches, gid, set(big_buckets))
            self._fused_big_buckets = self._prune_lag_caches(
                self._fused_big_buckets, gid, set(big_buckets))
            for lag, plist in one_block_buckets.items():
                c = self._coeffs(group, group_step - lag)
                self._fused_one_block(plist, group, ft, lag, c)
            for lag, plist in big_buckets.items():
                self._fused_big(
                    plist,
                    group,
                    ft,
                    self._coeffs(group, group_step - lag),
                    pos_pref,
                    neg_pref,
                    lag,
                )
            for lag, plist in one_dim_buckets.items():
                self._fused_one_dim(
                    plist,
                    group,
                    ft,
                    lag,
                    self._coeffs(group, group_step - lag),
                    pos_pref,
                    neg_pref,
                )
            if _PROBE_LOG:
                _probe_group(self, group)
        return loss

    @staticmethod
    def _prune_lag_caches(caches: dict, gid: int, active: set[int]) -> dict:
        """Drop this group's pointer caches for lags that no longer have a bucket.

        The fused caches are keyed ``(id(group), lag)``, and a lag disappears as soon as every
        parameter that was behind catches up — without this the dict grows one entry per lag ever
        seen, each pinning a bucket's index tensors (and, through them, device memory).
        """
        return {k: c for k, c in caches.items() if k[0] != gid or k[1] in active}

    def _big_shape_buckets(
        self, gid: int, lag: int, big: list[Tensor]
    ) -> list[list[Tensor]]:
        """``big`` split into same-shape/dtype/device buckets, memoized per (group, lag).

        Rebuilding these lists every step is what made the witness expensive rather than
        what cost the time: a fresh list means :class:`~kaon._fused_triton.BigPnmCache`
        cannot recognize it (``built_from`` is identity), so every step fell through to
        the full ``stale`` compare — a :func:`~kaon._fused_triton.param_witness` tuple
        over the bucket, once per big shape bucket, on top of the sweep
        :meth:`_fused_partition` already ran for the whole group. Same fix, same
        contract and the same argument as ``Adakaon._big_shape_buckets``: the list
        IDENTITY of ``big`` is as strong as recomparing the witness, because ``big``
        only survives while the partition's own witness (ids + ``data_ptr``s +
        contiguity over the group's whole param set) held AND
        :meth:`~kaon.adakaon.Adakaon._fused_demote`'s demoted set and
        :meth:`_local_step_buckets`' lag partition were unchanged. The ``lag`` is in the
        key because a mixed-lag group hands each lag its own sub-list; the entries are
        pruned with the pointer caches by :meth:`_prune_lag_caches`.
        """
        cached = self._fused_big_buckets.get((gid, lag))
        if cached is not None and cached[0] is big:
            return cached[1]
        buckets = list(_same_shape_device_buckets(big).values())
        self._fused_big_buckets[gid, lag] = (big, buckets)
        return buckets

    def _local_step_buckets(
        self, params: list[Tensor], group_step: int
    ) -> dict[int, list[Tensor]]:
        """Bucket parameters by stable lag behind the group's absolute step.

        When every param shares one lag — the overwhelmingly common case, a group whose params all
        got a gradient — the CALLER'S list object is handed straight back instead of a fresh copy.
        That is what lets the pointer caches revalidate by identity (``_WitnessedCache.built_from``,
        O(1)) rather than rebuilding the witness tuple per bucket per step: the partition has
        already revalidated ids + data_ptrs + contiguity for the whole group this step, and any
        change there produces fresh route lists. A genuinely mixed-lag group still gets fresh
        sub-lists and pays the full ``stale`` compare.
        """
        buckets: dict[int, list[Tensor]] = {}
        for p in params:
            state = self.state[p]
            assert state and state.get("step", 0) >= 1, (
                "AdaPNM parameter state must be prepared before fused bucketing"
            )
            lag = group_step - state["step"]
            assert lag >= 0, "AdaPNM parameter step cannot exceed its group step"
            buckets.setdefault(lag, []).append(p)
        if len(buckets) == 1:
            return {next(iter(buckets)): params}
        return buckets

    def _fused_partition(self, group: dict[str, Any], params: list[Tensor], ft: Any) -> tuple:
        """Split a group's params into (one-block, chunked-big, one-dim, native), cached per param-set.

        Keyed on :func:`kaon._fused_triton.param_witness` — ids, ``data_ptr``s and contiguity —
        because each is a routing input the partition (and every pointer array derived from it)
        bakes in, and ``p.data = ...`` can change any of them while the Parameter object stays the
        same. The id-only key kept dispatching a stale plan at memory the optimizer no longer owns:
        an external EMA, a ``.to(dtype/device)`` or Rengu-Flow's block-swap offloader rebinds the
        SAME parameter to fresh storage, and the cached ``pos``/``neg``/``p``/``grad`` arrays then
        wrote the whole step into the retired buffer. Same guard (and the same shape-rebind limit)
        as ``Adakaon._fused_partition``.

        SECOND KEY FIELD — the STATE-identity generation
        (:class:`~kaon._foreach_plan.WatchedState`). No parameter field can see a state buffer
        being retired (``del opt.state[p]``, ``opt.state[p].clear()``,
        ``opt.state[p]["m_pos"] = ...``), and the cached ``pos``/``neg``/``row``/``col`` tables
        then keep writing it. The lag bucketing happened to cover the two whole-state cases
        already (a re-initialised state restarts at ``step == 1``, so its lag moves and
        ``_prune_lag_caches`` drops the old key), but nothing covered a single buffer being
        rebound — measured writing the retired ``row`` on the one-block route. One integer
        compare here covers every fused route, for the same reason as in
        ``Adakaon._fused_partition``: a rebuild hands back fresh route lists and every
        downstream memo and ``_WitnessedCache`` revalidates by list identity.

        Grad properties deliberately stay OUT of this key: a gradient is a new tensor every
        backward, so its contiguity is re-checked per step in
        :func:`kaon.adakaon._demote_non_contiguous_grads` rather than frozen into the routing.
        """
        gid = id(group)
        witness = ft.param_witness(params)
        gen = state_generation(self.state)
        cached = self._fused_part.get(gid)
        if cached is not None and cached[1] == gen and cached[0] == witness:
            return cached[2], cached[3], cached[4], cached[5]
        md, bf16m, cap = group["momentum_dtype"], group["bf16_method"], self._fused_tile_cap
        float_mom = md in ("bfloat16", "float32")  # the 1-D kernel handles only fp32/bf16 momentum
        no_ams = not group["ams_bound"]            # ams_bound 1-D (full max_v) -> native
        one_block: list[Tensor] = []
        big: list[Tensor] = []
        one_dim: list[Tensor] = []
        native: list[Tensor] = []
        for p in params:
            bf_ok = (p.dtype != torch.bfloat16) or (bf16m == "stochastic_rounding")
            # ndim>2 (conv) is matrixized to (out, in*kh*kw); needs fp32/bf16 momentum (quant's
            # per-row requant would reshape the conv state) -> else native. The matrixized
            # write-back also needs a contiguous GRAD, enforced per step by
            # _demote_non_contiguous_grads (below) and NOT here, because the grad changes every
            # backward and this partition is cached across steps.
            conv_ok = p.ndim <= 2 or float_mom
            ok = bf_ok and conv_ok and ft.fused_eligible(p, cap)
            if ok and md == "4bit":
                if ft.eff_2d(p)[1] % 2 != 0:
                    ok = False                              # one-block 4bit needs even C
                elif self._fourbit_block(p, group) != min(_FOURBIT_BLOCK, p.numel()):
                    # ``_adapnm_tile_kernel`` hardcodes its absmax block at ``BLK = min(R*C, 128)``
                    # for BOTH momenta's dequant AND the positive's requant. Under any other
                    # ``momentum_4bit_block`` that does not merely read the wrong scales: the
                    # requant addresses ceil(numel/128) blocks against an ``m_pos_scale`` /
                    # ``m_neg_scale`` sized for the REAL block count — measured as 32 floats
                    # written past the end of each of a (64,128) weight's four scale buffers at
                    # block=256. Route it to the native / chunked path, which honours the stored
                    # ``m_pos_block``. Making the block a constexpr would multiply the JIT
                    # variants; the 1-D route needs no guard (quant 1-D is native anyway).
                    ok = False
            big_ok = (bf_ok and conv_ok and p.ndim >= 2 and p.is_cuda and p.is_contiguous()
                      and p.dtype in (torch.float32, torch.bfloat16)
                      and ft.next_pow2_tile(*ft.eff_2d(p))[0] * ft.next_pow2_tile(*ft.eff_2d(p))[1] > cap)
            if ok:
                one_block.append(p)
            elif big_ok:
                big.append(p)
            # No cap argument: 1-D is bounded by ft.TILE_CAP_1D, not the 2-D crossover ``cap``
            # (above the 1-D cap a tensor falls to native, not to the chunked kernel).
            elif bf_ok and float_mom and no_ams and ft.fused_1d_eligible(p):
                one_dim.append(p)
            else:
                native.append(p)
        self._fused_part[gid] = (witness, gen, one_block, big, one_dim, native)
        if _PROBE_LOG:
            _probe_census(one_block, big, native, md, bf16m, cap, ft)
        return one_block, big, one_dim, native

    def _fourbit_block(self, p: Tensor, group: dict[str, Any]) -> int:
        """This param's 4-bit absmax block size: the one already in state when there is one (a
        checkpoint can carry a layout the current group setting would not produce), else the one
        :meth:`_init_state` is about to pick.

        The two momenta are always allocated with the same block, so the ``min`` is belt and
        braces — but it is the SAME reduction :class:`~kaon._fused_triton.AdaPnmCache` applies to
        the two scale capacities it validates, and the routing guard has to be at least as strict
        as the capacity check or the cache would raise on a tensor the routing let through.
        """
        st = self.state.get(p)
        if st and "m_pos_block" in st:
            return min(st["m_pos_block"], st.get("m_neg_block", st["m_pos_block"]))
        return fourbit_block_size(p.grad, group)

    def _fused_demote(self, gid: int, parts: tuple) -> tuple:
        """This step's routing, with any non-contiguous-grad tensor moved to the native subset.

        Thin memo over :func:`kaon.adakaon._demote_non_contiguous_grads`. Every fused kernel reads
        the gradient as ``base + ri*C + ci`` (or ``base + offs``) straight off ``grad.data_ptr()``,
        so a transposed (``grad = x.t()``) or strided (``grad = buf[::2]``) gradient has the right
        shape and the wrong layout and the kernel silently steps the wrong numbers — measured at
        ~1e-2 against native on all three routes, with no error raised anywhere.

        The demotion has to build fresh route lists, and a fresh list means every downstream
        pointer cache re-validates; reuse the lists while both the partition (identity of its four
        lists) and the demoted SET are unchanged. The contiguity sweep itself still runs every
        step, since that is what detects the change.
        """
        demoted = tuple(id(p) for sub in parts[:3] for p in sub if not p.grad.is_contiguous())
        if not demoted:
            self._fused_demoted.pop(gid, None)
            return parts
        cached = self._fused_demoted.get(gid)
        if (cached is not None and cached[0] == demoted
                and all(a is b for a, b in zip(cached[1], parts, strict=True))):
            return cached[2]
        out = _demote_non_contiguous_grads(*parts)
        self._fused_demoted[gid] = (demoted, parts, out)
        return out

    def _fused_one_block(
        self,
        plist: list[Tensor],
        group: dict[str, Any],
        ft: Any,
        lag: int,
        c: dict,
    ) -> None:
        for p in plist:
            st = self.state[p]
            assert st and st.get("step", 0) >= 1, (
                "AdaPNM parameter state must be prepared before fused one-block step"
            )
        key = (id(group), lag)
        cache = self._fused_ob_caches.get(key)
        # ``revalidate``: list identity first (``_fused_partition`` compared ids, data_ptrs AND
        # contiguity across the whole group this step and, in the single-lag case,
        # ``_local_step_buckets`` hands back that very list, so identity is exactly as strong as
        # recomparing the tuples and skips a second witness sweep), falling back to the full
        # witness compare — which is the only thing that can tell a ``p.data`` rebind (it keeps
        # every id) from a harmless re-bucketing — and ADOPTING the fresh list when nothing
        # moved. That adoption is the whole point: ``not built_from(...) and stale(...)`` left
        # ``src`` one generation behind and brought the per-bucket sweep back permanently (see
        # ``_WitnessedCache.revalidate``).
        gen = state_generation(self.state)
        if cache is None or not cache.revalidate(plist, gen):
            cache = ft.AdaPnmCache(plist, lambda p: self.state[p], gen=gen)
            self._fused_ob_caches[key] = cache
        cache.refresh_grads()
        odd = group["step"] % 2 == 1
        lr, wd, eps1 = group["lr"], group["weight_decay"], group["eps"]
        cautious, gc = group["cautious"], group["gradient_centralization"]
        clip = group["clip_threshold"]
        sc, inv_noise = c["bc2_sq"] * c["step_size"], 1.0 / c["noise_norm"]
        clip_eff = clip * c["step_size"]        # rms(upd) <= clip*step_size == rms(pn/sqrt(v_hat)) <= clip
        for bk in cache.buckets:
            # which physical buffer plays positive this step (alternation): the m_pos slot if odd.
            if odd:
                kpos, kneg, kposc, knegc = bk["pos_addr"], bk["neg_addr"], bk["posc_addr"], bk["negc_addr"]
            else:
                kpos, kneg, kposc, knegc = bk["neg_addr"], bk["pos_addr"], bk["negc_addr"], bk["posc_addr"]
            lanes = bk["BR"] * bk["BC"]
            # A bucket's index arrays live on ITS device (AdaPnmCache buckets by device), and a
            # Triton launch goes to the CURRENT device, not to the one the arguments came from.
            # A group spanning cuda:0 and cuda:1 would otherwise launch every bucket on whichever
            # device happened to be current. PLAUSIBLE, not verified: one GPU on this machine.
            with torch.cuda.device(bk["dev"]):
                ft._adapnm_tile_kernel[(len(bk["plist"]),)](
                    bk["g_addr"], bk["p_addr"], kpos, kneg, kposc, knegc, bk["row_addr"],
                    bk["col_addr"], bk["Rs"], bk["Cs"], bk["mscale_n"],
                    c["beta1_sq"], c["beta0"], inv_noise, c["beta2"], sc, lr * wd, eps1,
                    clip_eff, group["step"], LOWP=bk["lowp"], MOM=bk["mom"], CAUTIOUS=cautious,
                    WD=wd != 0, GC=gc, SR=bk["lowp"], CLIP=clip > 0.0, BR=bk["BR"], BC=bk["BC"],
                    num_warps=ft.warps_for(lanes),
                )

    def _fused_one_dim(
        self,
        plist: list[Tensor],
        group: dict[str, Any],
        ft: Any,
        lag: int,
        c: dict,
        pos_pref: str,
        neg_pref: str,
    ) -> None:
        """One-block non-factored kernel over eligible 1-D weights (biases / norm scales). GC is a no-op
        on 1-D. fp32/bf16 momenta only; ams_bound and quant route to native (excluded in the partition)."""
        for p in plist:
            st = self.state[p]
            assert st and st.get("step", 0) >= 1, (
                "AdaPNM parameter state must be prepared before fused one-dim step"
            )
        key = (id(group), lag)
        cache = self._fused_od_caches.get(key)
        gen = state_generation(self.state)
        if cache is None or not cache.revalidate(plist, gen):            # see _fused_one_block
            cache = ft.OneDimPnmCache(plist, lambda p: self.state[p], gen=gen)
            self._fused_od_caches[key] = cache
        cache.refresh_grads()
        odd = group["step"] % 2 == 1
        lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
        cautious, clip = group["cautious"], group["clip_threshold"]
        inv_noise = 1.0 / c["noise_norm"]
        for bk in cache.buckets:
            # which physical buffer plays positive this step (alternation): the m_pos slot if odd.
            kpos, kneg = (bk["pos_addr"], bk["neg_addr"]) if odd else (bk["neg_addr"], bk["pos_addr"])
            with torch.cuda.device(bk["dev"]):         # see _fused_one_block on the device scope
                ft._adapnm_1d_kernel[(len(bk["plist"]),)](
                    bk["g_addr"], bk["p_addr"], kpos, kneg, bk["v_addr"], bk["Ls"],
                    c["beta1_sq"], c["beta0"], inv_noise, c["beta2"], c["step_size"], c["bc2_sq"],
                    eps, lr * wd, clip, group["step"], LOWP=bk["lowp"], MOM=bk["mom"],
                    CAUTIOUS=cautious, WD=wd != 0, CLIP=clip > 0.0, SR=bk["lowp"], BL=bk["BL"],
                    num_warps=ft.warps_for(bk["BL"]),
                )

    @torch.no_grad()
    def _fused_big(self, big: list[Tensor], group: dict[str, Any], ft: Any, c: dict,
                   pos_pref: str, neg_pref: str, lag: int = 0) -> None:
        """Dispatch the >tile-cap ("big") 2-D factors.

        The per-tensor fused-chunked kernel is launch-bound (~8 kernels/tensor): for the many
        same-shape big factors a real LoKr run has (e.g. 236x 512x512), the **batched native
        foreach** path is ~5x faster (measured 14ms vs 69ms) because it stacks same-shape tensors
        and amortizes launches. A lone big tensor has no batch to amortize, so it keeps the
        fused-chunked kernel. Same math + state either way (both RMS-clip), so they interoperate.
        """
        if len(big) >= 2 and not self._fused_big_batched:
            if group["gradient_centralization"]:
                centralize_grads_(big)
            self._native_dispatch(big, group)
            return
        # Group by EXACT shape, dtype AND DEVICE: a bucket is launched as one grid against pointer
        # arrays built on ``plist[0].device``, so two CUDA devices sharing a shape would run the
        # second one's tensors against index tensors from the first.
        for plist in self._big_shape_buckets(id(group), lag, big):
            # One device scope per bucket, covering every launch inside the chunked steps (see
            # _fused_one_block): the bucket's pointer arrays and scratch live on plist[0].device,
            # and a Triton launch targets the CURRENT device. PLAUSIBLE, not verified — one GPU here.
            with torch.cuda.device(plist[0].device):
                if len(plist) >= 2:
                    self._chunked_step_batched(plist, group, ft, c, pos_pref, neg_pref, lag)
                else:
                    self._chunked_step(plist[0], group, ft, c, pos_pref, neg_pref)

    def _chunked_reductions(self, p: Tensor, group: dict[str, Any], st: dict[str, Any]) -> tuple:
        b2, eps1 = group["betas"][1], group["eps"]
        g = p.grad.float().reshape(p.shape[0], p.numel() // p.shape[0])  # conv -> matrixized (out, in*kh*kw)
        if group["gradient_centralization"]:
            g = g - g.mean(dim=1, keepdim=True)
        g = g.contiguous()
        gsq = g * g
        st["row"].lerp_(gsq.mean(1).add_(eps1), 1.0 - b2)
        st["col"].lerp_(gsq.mean(0).add_(eps1), 1.0 - b2)
        return g, st["row"].div(st["row"].mean()).rsqrt_(), st["col"].rsqrt()

    def _chunked_step(self, p: Tensor, group: dict[str, Any], ft: Any, c: dict,
                      pos_pref: str, neg_pref: str) -> None:
        st = self.state[p]
        assert st and st.get("step", 0) >= 1, (
            "AdaPNM parameter state must be prepared before fused chunked step"
        )
        R, C = p.shape[0], p.numel() // p.shape[0]  # noqa: N806 — matrix dims (conv -> matrixized)
        n = R * C
        md = group["momentum_dtype"]
        lr, wd, cautious = group["lr"], group["weight_decay"], group["cautious"]
        sr = (p.dtype == torch.bfloat16) and (group["bf16_method"] == "stochastic_rounding")
        g, r, cfac = self._chunked_reductions(p, group, st)
        m_pos = self._dequant_one(st, pos_pref, md, g).reshape(R, C)   # fp32 temp (EMA'd in-kernel)
        m_neg = self._dequant_one(st, neg_pref, md, g).reshape(R, C)   # fp32 temp (read-only)
        sc, inv_noise = c["bc2_sq"] * c["step_size"], 1.0 / c["noise_norm"]
        keep = torch.zeros(1, dtype=torch.int32, device=p.device)
        gf, posf, negf, pf = g.reshape(-1), m_pos.reshape(-1), m_neg.reshape(-1), p.reshape(-1)
        grid = ((n + 1023) // 1024,)
        ft._adapnm_chunked_mom[grid](gf, posf, negf, pf, r, cfac, keep, C, n, c["beta1_sq"], c["beta0"],
                                     inv_noise, sc, lr * wd, CAUTIOUS=cautious, WD=wd != 0, BLOCK=1024)
        self._store_one(st, pos_pref, md, m_pos)                       # requant/store updated positive
        # Adafactor RMS-clip on the v_hat-normalized update U = pn * r * c * bc2_sq (== delta/step_size):
        # bound rms(U) <= clip by folding 1/max(rms(U)/clip, 1) into the lr-scale `sc` (one [R,C] temp).
        sc_apply = sc
        clip = group["clip_threshold"]
        if clip > 0.0:
            u = m_pos.mul(1.0 + c["beta0"]).sub_(m_neg, alpha=c["beta0"])   # pn numerator (fresh temp)
            u.mul_(r.reshape(R, 1)).mul_(cfac.reshape(1, C))
            rms_u = float(u.norm()) * inv_noise * c["bc2_sq"] / math.sqrt(n)
            sc_apply = sc / max(rms_u / clip, 1.0)
        inv_mean = 1.0 / max(keep.item() / n, 1e-8) if cautious else 1.0
        ft._adapnm_chunked_apply[grid](gf, posf, negf, pf, r, cfac, C, n, c["beta0"], inv_noise, sc_apply,
                                       lr * wd, inv_mean, group["step"], CAUTIOUS=cautious, WD=wd != 0,
                                       SR=sr, BLOCK=1024)

    # ----------------------------------------------- batched chunked (many same-shape big tensors)
    @torch.no_grad()
    def _chunked_reductions_batched(self, plist: list[Tensor], group: dict[str, Any]) -> tuple:
        """Stacked AdaPNM reductions for a same-shape big bucket: GC (on the fp32 copy) + row/col EMA.
        Returns the stacked fp32 grad ``[N, R, C]`` and the stacked r/c factors ``[N, R]``/``[N, C]``
        (contiguous). Mirrors :meth:`_chunked_reductions` per tensor (eps1 on the means; no rms — the
        AdaPNM clip is computed separately, folded into ``sc`` in :meth:`_chunked_step_batched`)."""
        b2, eps1 = group["betas"][1], group["eps"]
        states = [self.state[p] for p in plist]
        R, C = plist[0].shape[0], plist[0].numel() // plist[0].shape[0]    # noqa: N806 — conv -> matrixized
        g = torch.stack([p.grad.reshape(R, C) for p in plist]).float()     # [N, R, C]
        if group["gradient_centralization"]:
            g.sub_(g.mean(dim=-1, keepdim=True))
        gsq = g * g
        row = torch.stack([s["row"] for s in states])                    # [N, R]
        col = torch.stack([s["col"] for s in states])                    # [N, C]
        row.lerp_(gsq.mean(dim=-1).add_(eps1), 1.0 - b2)
        col.lerp_(gsq.mean(dim=-2).add_(eps1), 1.0 - b2)
        torch._foreach_copy_([s["row"] for s in states], list(row.unbind(0)))
        torch._foreach_copy_([s["col"] for s in states], list(col.unbind(0)))
        r = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_()             # [N, R]
        c = col.rsqrt()                                                  # [N, C]
        return g, r.contiguous(), c.contiguous()

    @torch.no_grad()
    def _chunked_step_batched(self, plist: list[Tensor], group: dict[str, Any], ft: Any, c: dict,
                              pos_pref: str, neg_pref: str, lag: int = 0) -> None:
        """A bucket of >=2 same-shape big 2-D tensors via the batched AdaPNM chunked kernels (~2
        launches). fp32/bf16 momenta are read/written in place via the pos/neg pointer arrays (no
        temps); int8/4bit dequant to fp32 temps, step on them, requant the positive between passes.
        The Adafactor RMS-clip's per-tensor sum-of-squares is accumulated in-kernel (no torch momentum
        temp in the float case), turned into ``sc_apply[N]`` between passes.

        The pointer arrays and the reduction scratch come from a :class:`~kaon._fused_triton.
        BigPnmCache` keyed ``(id(group), lag, shape, dtype, device)`` and revalidated by witness:
        they used to be rebuilt from a fresh ``torch.tensor([...])`` on EVERY step (grad, p, both
        momenta, rowmean/rowsum/colsum/keep/rms) while the one-block and 1-D routes have cached
        theirs since 0.7.9."""
        for p in plist:
            st = self.state[p]
            assert st and st.get("step", 0) >= 1, (
                "AdaPNM parameter state must be prepared before batched chunked step"
            )
        N = len(plist)  # noqa: N806
        R, C = plist[0].shape[0], plist[0].numel() // plist[0].shape[0]  # noqa: N806 — conv -> matrixized
        n = R * C
        dev = plist[0].device
        md = group["momentum_dtype"]
        lr, wd, cautious = group["lr"], group["weight_decay"], group["cautious"]
        clip = group["clip_threshold"]
        gc = group["gradient_centralization"]
        lowp = plist[0].dtype == torch.bfloat16
        sr = lowp and (group["bf16_method"] == "stochastic_rounding")
        states = [self.state[p] for p in plist]
        cache_key = (id(group), lag, tuple(plist[0].shape), plist[0].dtype, dev)
        cache = self._fused_big_caches.get(cache_key)
        # ``revalidate``'s identity path now actually HITS: ``plist`` comes from
        # :meth:`_big_shape_buckets`, which hands back the same list object while the
        # partition's witness holds. Its witness fallback covers the step after the memo
        # hands out a fresh list (a re-bucketing that moved nothing) and rebinds the cache
        # onto it, so the identity path keeps working from the next step on.
        gen = state_generation(self.state)
        if cache is None or not cache.revalidate(plist, gen):
            cache = ft.BigPnmCache(plist, lambda p: self.state[p], R, C, gen=gen)  # see _fused_one_block
            self._fused_big_caches[cache_key] = cache
        cache.refresh_grads()
        fused_red = self._fused_reductions
        if fused_red:  # grad via pointer array, no [N,R,C] stack (candidate #4); rowmean carries GC
            g_addr, rowmean, r, cfac = self._chunked_reductions_fused(
                plist, group, ft, R, C, lowp, cache
            )
        else:
            g, r, cfac = self._chunked_reductions_batched(plist, group)   # g [N,R,C], r [N,R], c [N,C]
            gf = g.reshape(-1)
        sc, inv_noise = c["bc2_sq"] * c["step_size"], 1.0 / c["noise_norm"]

        quant = md in ("int8", "4bit")
        if quant:  # dequant both momenta to stacked fp32 temps; arrays point at the temp slices
            pos_temp = torch.empty(N, R, C, dtype=torch.float32, device=dev)
            neg_temp = torch.empty(N, R, C, dtype=torch.float32, device=dev)
            for i, st in enumerate(states):
                pos_temp[i] = self._dequant_one(st, pos_pref, md, pos_temp[i])
                neg_temp[i] = self._dequant_one(st, neg_pref, md, neg_temp[i])
            pos_addr = ft.ptr_array(list(pos_temp), dev)
            neg_addr = ft.ptr_array(list(neg_temp), dev)
            mom = ft.MOM_FP32
        else:      # fp32/bf16: cached pointer arrays to the stored buffers (EMA positive in place)
            pos_addr, neg_addr = cache.momenta(pos_pref == "m_pos")
            mom = ft.MOM_BF16 if md == "bfloat16" else ft.MOM_FP32
        p_addr = cache.p_addr
        keep = cache.keep.zero_()
        rms_acc = cache.rms_acc.zero_()
        K = (n + 1023) // 1024  # noqa: N806
        grid = (N * K,)
        if fused_red:
            ft._adapnm_chunked_mom_batched_g[grid](
                g_addr, rowmean, pos_addr, neg_addr, r, cfac, keep, rms_acc, R, C, n, K,
                c["beta1_sq"], c["beta0"], inv_noise,
                LOWP=lowp, MOM=mom, GC=gc, CAUTIOUS=cautious, CLIP=clip > 0.0, BLOCK=1024,
            )
        else:
            ft._adapnm_chunked_mom_batched[grid](
                gf, pos_addr, neg_addr, r, cfac, keep, rms_acc, R, C, n, K, c["beta1_sq"], c["beta0"],
                inv_noise, MOM=mom, CAUTIOUS=cautious, CLIP=clip > 0.0, BLOCK=1024,
            )
        if quant:  # requant the updated positive temp back into storage (apply reads the temp)
            for i, st in enumerate(states):
                self._store_one(st, pos_pref, md, pos_temp[i])
        # Per-tensor Adafactor RMS-clip: rms_u = bc2_sq * sqrt(rms_acc / n); fold into sc_apply[N].
        if clip > 0.0:
            rms_u = rms_acc.div_(n).sqrt_().mul_(c["bc2_sq"])             # [N]
            sc_apply = (sc / rms_u.div_(clip).clamp_(min=1.0)).contiguous()
        else:
            sc_apply = torch.full((N,), sc, dtype=torch.float32, device=dev)
        inv_mean = (1.0 / (keep.float() / n).clamp_(min=1e-8)) if cautious else torch.ones(N, device=dev)
        if fused_red:
            ft._adapnm_chunked_apply_batched_g[grid](
                g_addr, rowmean, pos_addr, neg_addr, p_addr, r, cfac, sc_apply, inv_mean, R, C, n, K,
                c["beta0"], inv_noise, lr * wd, group["step"],
                LOWP=lowp, MOM=mom, GC=gc, CAUTIOUS=cautious, WD=wd != 0, SR=sr, BLOCK=1024,
            )
        else:
            ft._adapnm_chunked_apply_batched[grid](
                gf, pos_addr, neg_addr, p_addr, r, cfac, sc_apply, inv_mean, R, C, n, K, c["beta0"],
                inv_noise, lr * wd, group["step"], LOWP=lowp, MOM=mom, CAUTIOUS=cautious, WD=wd != 0,
                SR=sr, BLOCK=1024,
            )

    @torch.no_grad()
    def _chunked_reductions_fused(self, plist, group, ft, R, C, lowp, cache):  # noqa: N803
        """Candidate #4 for AdaPNM: row/col EMA factors via the Triton reduction kernel reading grad
        from a pointer array (no [N,R,C] stack; GC in-kernel). No rms here — AdaPNM's clip is computed
        in the mom kernel. Returns (g_addr, rowmean, r_factor[N,R], c_factor[N,C]).

        ``cache`` supplies the grad pointer array and the three scratch buffers; they were
        reallocated here on every step, which is exactly what the cache exists to avoid."""
        b2, eps1 = group["betas"][1], group["eps"]
        N = len(plist)  # noqa: N806
        states = [self.state[p] for p in plist]
        g_addr = cache.g_addr
        BR, BC, RB = ft.reduction_tile(R, C)  # noqa: N806
        rowmean = cache.rowmean
        rowsum = cache.rowsum
        colsum = cache.colsum.zero_()  # atomic target
        ft._reduce_rowcol[(N * RB,)](
            g_addr, rowmean, rowsum, colsum, R, C, RB,
            LOWP=lowp, GC=group["gradient_centralization"], BR=BR, BC=BC, num_warps=ft.warps_for(BR * BC),
        )
        rowsum = rowsum.view(N, R)
        colsum = colsum.view(N, C)
        row = torch.stack([s["row"] for s in states])
        col = torch.stack([s["col"] for s in states])
        row.lerp_(rowsum.div(C).add_(eps1), 1.0 - b2)
        col.lerp_(colsum.div(R).add_(eps1), 1.0 - b2)
        torch._foreach_copy_([s["row"] for s in states], list(row.unbind(0)))
        torch._foreach_copy_([s["col"] for s in states], list(col.unbind(0)))
        r = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_().contiguous()
        cfac = col.rsqrt().contiguous()
        return g_addr, rowmean, r, cfac

    def state_dict(self) -> dict[str, Any]:
        """Base state + the auto_lr tuner blob (via AutoLRMixin) when auto_lr is on."""
        return self._autolr_state_dict(super().state_dict())

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state, preserving both quantized momenta's stored dtype.

        torch's default ``load_state_dict`` upcasts every state tensor to the
        param's dtype (fp32), which would silently inflate bf16/int8/4bit momenta
        back to fp32 on resume. Delegate to the shared helper that restores each
        tensor to how it was checkpointed.

        It also **replaces** each ``param_groups`` dict with the checkpoint's (only
        ``params`` is carried over), so a checkpoint written by an older kaon has no
        entry for a hyperparameter added since — reading it would raise ``KeyError``
        on the first step. Backfill any key the checkpoint predates from
        ``self.defaults``; keys the checkpoint *does* carry win, so a resumed run
        keeps its own tuning.
        """
        self._autolr_load(state_dict, lambda sd: load_state_dict_preserving_dtypes(self, sd))
        for group in self.param_groups:
            for key, value in self.defaults.items():
                group.setdefault(key, value)
        self._invalidate_fused_caches()

    # ----------------------------------------------------------- coefficients
    @staticmethod
    def _coeffs(group: dict[str, Any], step: int | None = None) -> dict[str, float]:
        """Scalar coefficients; bias corrections use the parameter-local step."""
        beta1, beta2 = group["betas"]
        beta0 = group["beta0"]
        step = group["step"] if step is None else step
        assert step >= 1, "AdaPNM coefficients require a prepared 1-indexed step"
        beta1_sq = beta1 * beta1
        noise_norm = math.sqrt((1.0 + beta0) ** 2 + beta0 ** 2)
        bc1 = 1.0 - beta1 ** step          # bias correction uses beta1, NOT beta1^2
        bc2_sq = math.sqrt(1.0 - beta2 ** step)
        return {
            "beta1_sq": beta1_sq,
            "beta2": beta2,
            "beta0": beta0,
            "noise_norm": noise_norm,
            "bc1": bc1,
            "bc2_sq": bc2_sq,
            "step_size": group["lr"] / bc1,
        }

    # ----------------------------------------------------------------- foreach

    @staticmethod
    def _group_foreach_eligible(group: dict[str, Any]) -> bool:
        return group["bf16_method"] != "kahan"  # kahan needs a per-param shift buffer

    @staticmethod
    def _param_foreach_eligible(p: Tensor, group: dict[str, Any], cutoff: int) -> bool:
        # 0-D scalars are NOT excluded: they ride the non-factored bucket as
        # length-1 rows (see kaon._backend.flat_view). Only the per-tensor size cap
        # and the awkward dtype/contiguity cases fall back to the per-param loop.
        if p.numel() > cutoff:
            return False
        if (
            group["bf16_method"] == "stochastic_rounding"
            and is_low_precision(p)
            and p.dtype != torch.bfloat16  # fp16+SR unsupported -> per-param (raises)
        ):
            return False
        if p.ndim > 2:
            # Matrixized conv writes back through a reshaped view -> needs contiguity.
            return p.data.is_contiguous() and p.grad.is_contiguous()
        return True

    @torch.no_grad()
    def _step_foreach(self, params: list[Tensor], group: dict[str, Any], budget: int) -> None:
        """Batched step. Factored (ndim>=2) and non-factored (ndim<=1) buckets, by shape.

        0-D scalars ride the non-factored bucket keyed by ``numel() == 1``, sharing it
        with real shape-(1,) params."""
        md = group["momentum_dtype"]
        pos, neg = self._pos_neg_prefixes(group["step"])
        group_step = group["step"]
        assert group_step >= 1, "AdaPNM foreach step requires an advanced group step"

        factored_buckets: dict[tuple[Any, ...], list[Tensor]] = {}
        flat_buckets: dict[tuple[Any, ...], list[Tensor]] = {}
        for p in params:
            state = self.state[p]
            assert state and state.get("step", 0) >= 1, (
                "AdaPNM parameter state must be prepared before foreach step"
            )
            lag = group_step - state["step"]
            assert lag >= 0, "AdaPNM parameter step cannot exceed its group step"
            g = p.grad
            # The DEVICE belongs in both keys: a bucket is stepped with ``torch.stack`` /
            # ``_foreach_*`` over its members, so a CPU and a CUDA weight of the same shape landing
            # in one bucket raised "Expected all tensors to be on the same device" and took the
            # WHOLE step down. Same fix (and same reason) as Adakaon's foreach plan.
            if g.ndim >= 2:
                matrixize = g.ndim > 2
                eff = (g.shape[0], g.numel() // g.shape[0]) if matrixize else tuple(g.shape)
                key = (eff, p.dtype, p.device, matrixize, lag)
                factored_buckets.setdefault(key, []).append(p)
            else:  # ndim <= 1 — 0-D scalars ride as length 1 (numel == shape[0] for 1-D)
                key = (g.numel(), p.dtype, p.device, lag)
                flat_buckets.setdefault(key, []).append(p)

        for (eff, _dtype, _dev, matrixize, lag), plist in factored_buckets.items():
            c = self._coeffs(group, group_step - lag)
            stepn = max(1, budget // max(eff[0] * eff[1], 1))
            for i in range(0, len(plist), stepn):
                self._factored_bucket(plist[i:i + stepn], eff, matrixize, md, pos, neg, c, group)
        for (length, _dtype, _dev, lag), plist in flat_buckets.items():
            c = self._coeffs(group, group_step - lag)
            stepn = max(1, budget // max(length, 1))
            for i in range(0, len(plist), stepn):
                self._nonfactored_bucket(plist[i:i + stepn], length, md, pos, neg, c, group)

    @torch.no_grad()
    def _factored_bucket(
        self,
        plist: list[Tensor],
        eff: tuple[int, int],
        matrixize: bool,
        md: str,
        pos: str,
        neg: str,
        c: dict[str, float],
        group: dict[str, Any],
    ) -> None:
        R, C = eff  # noqa: N806
        eps1 = group["eps"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]

        def mat(t: Tensor) -> Tensor:
            return t.view(R, C) if matrixize else t

        states = [self.state[p] for p in plist]
        rows = [s["row"] for s in states]
        cols = [s["col"] for s in states]

        grad = torch.stack([mat(p.grad) for p in plist]).float()          # [N, R, C]
        row = torch.stack(rows)                                           # [N, R]
        col = torch.stack(cols)                                           # [N, C]

        # Decoupled weight decay BEFORE moment updates (kozistr order): p *= (1 - lr*wd).
        if wd != 0:
            self._apply_decoupled_wd_batched(plist, mat, group["lr"] * wd)

        # Factored second-moment EMA (HF eps1 placement).
        omb2 = 1.0 - c["beta2"]
        grad_sq = grad * grad
        if eps1 > 0:
            grad_sq = grad_sq.add_(eps1)
        row.lerp_(grad_sq.mean(dim=-1), omb2)
        col.lerp_(grad_sq.mean(dim=-2), omb2)
        torch._foreach_copy_(rows, list(row.unbind(0)))
        torch._foreach_copy_(cols, list(col.unbind(0)))

        r_factor = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)  # [N, R, 1]
        c_factor = col.rsqrt().unsqueeze(-2)                                       # [N, 1, C]
        inv_denom = (r_factor * c_factor).mul_(c["bc2_sq"])                        # 1/sqrt(v_hat)

        # Positive-negative momentum mixing (read both, EMA only the positive).
        pn = self._pn_stacked(states, pos, neg, md, (R, C), grad, c)               # [N, R, C]

        update = _rms_clip_batched_(pn.mul_(inv_denom), group["clip_threshold"])   # rms(u)<=clip
        delta = update.mul_(c["step_size"])                                        # full step

        if cautious:
            delta = cautious_batched_(delta, grad)

        subtract_batched_([mat(p.data) for p in plist], delta, bf16_method)

    @torch.no_grad()
    def _nonfactored_bucket(
        self,
        plist: list[Tensor],
        length: int,
        md: str,
        pos: str,
        neg: str,
        c: dict[str, float],
        group: dict[str, Any],
    ) -> None:
        """Non-factored (full per-coordinate ``v``) update for ``ndim <= 1`` params.

        0-D scalars share the ``L == 1`` bucket with shape-``(1,)`` params as length-1
        **views** (:func:`~kaon._backend.flat_view`) of the same storage, so ``v`` /
        ``max_v``, both momenta and the weight subtract reach the original 0-D
        tensors. At ``L == 1`` the batched RMS clip ``norm(dim=1)/sqrt(1)`` is the
        per-param ``rms()`` of a scalar, so the clip matches :meth:`_step_one_param`.
        """
        eps1 = group["eps"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]
        ams_bound = group["ams_bound"]

        states = [self.state[p] for p in plist]
        vs = [flat_view(s["v"]) for s in states]

        grad = torch.stack([flat_view(p.grad) for p in plist]).float()    # [N, L]
        v = torch.stack(vs)                                               # [N, L]

        if wd != 0:
            self._apply_decoupled_wd_batched(plist, flat_view, group["lr"] * wd)

        # Full per-coordinate second moment (1-D). eps here goes on the denominator
        # (kozistr), NOT folded into grad^2; eps1==eps for the 1-D path.
        v.mul_(c["beta2"]).addcmul_(grad, grad, value=1.0 - c["beta2"])
        torch._foreach_copy_(vs, list(v.unbind(0)))

        if ams_bound:
            max_vs = [flat_view(s["max_v"]) for s in states]
            max_v = torch.stack(max_vs)
            torch.maximum(max_v, v, out=max_v)
            torch._foreach_copy_(max_vs, list(max_v.unbind(0)))
            de_nom = max_v.add(1e-15).sqrt_().add_(eps1)
        else:
            de_nom = v.add(1e-15).sqrt_().add_(eps1)
        de_nom.div_(c["bc2_sq"])                                          # v_hat denom

        pn = self._pn_stacked(states, pos, neg, md, (length,), grad, c)   # [N, L]
        update = _rms_clip_batched_(pn.div_(de_nom), group["clip_threshold"])
        delta = update.mul_(c["step_size"])

        if cautious:
            delta = cautious_batched_(delta, grad)

        subtract_batched_([flat_view(p.data) for p in plist], delta, bf16_method)

    def _pn_stacked(
        self,
        states: list[dict[str, Any]],
        pos: str,
        neg: str,
        md: str,
        shape: tuple[int, ...],
        grad: Tensor,
        c: dict[str, float],
    ) -> Tensor:
        """Stacked positive-negative momentum numerator (EMA the positive buffer).

        ``grad`` is the stacked, reshaped, post-WD gradient (``[N, *shape]``). Reads
        both momenta as fp32, EMA-updates only the positive buffer with the raw
        gradient (decay ``beta1**2``), stores it back, and returns the renormalized
        ``((1+beta0)*m_pos - beta0*m_neg)/noise_norm``.
        """
        n = grad.shape[0]
        m_pos = self._dequant_stacked(states, pos, md, shape).reshape((n, *shape))
        m_neg = self._dequant_stacked(states, neg, md, shape).reshape((n, *shape))
        m_pos.mul_(c["beta1_sq"]).add_(grad, alpha=1.0 - c["beta1_sq"])
        self._store_stacked(states, pos, md, m_pos.reshape((n, *shape)))
        # m_pos is a fresh stacked tensor (torch.stack copies) and is already stored, so
        # the pos-neg mix can run in-place on it — no extra [N, *shape] allocation.
        pn = m_pos.mul_(1.0 + c["beta0"]).add_(m_neg, alpha=-c["beta0"]).mul_(1.0 / c["noise_norm"])
        return pn

    @torch.no_grad()
    def _apply_decoupled_wd_batched(self, plist: list[Tensor], mat: Any, factor: float) -> None:
        """In-place decoupled WD ``p *= (1 - factor)`` on the (matrixized) weights."""
        scale = 1.0 - factor
        torch._foreach_mul_([mat(p.data) for p in plist], scale)

    # ---------------------------------------------------------- per-parameter
    @torch.no_grad()
    def _step_one_param(self, p: Tensor, group: dict[str, Any]) -> None:
        state = self.state[p]
        assert state and state.get("step", 0) >= 1, (
            "AdaPNM parameter state must be prepared before per-parameter step"
        )
        c = self._coeffs(group, state["step"])
        md = group["momentum_dtype"]
        eps1 = group["eps"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]
        ams_bound = group["ams_bound"]
        pos, neg = self._pos_neg_prefixes(group["step"])

        grad = p.grad if p.grad.dtype == torch.float32 else p.grad.float()
        ndim = grad.ndim
        factored = ndim >= 2

        # Decoupled weight decay BEFORE the moment updates (kozistr order).
        if wd != 0:
            p.data.mul_(1.0 - group["lr"] * wd)

        if factored:
            matrixize = ndim > 2
            gv = grad.reshape(grad.shape[0], -1) if matrixize else grad
            update_factored_state(gv, state["row"], state["col"], c["beta2"], eps1)
            r_factor, c_factor = factored_inv_sqrt_factors(state["row"], state["col"])
            inv_denom = (r_factor * c_factor).mul_(c["bc2_sq"])           # 1/sqrt(v_hat)
            pn = self._pn_one(state, pos, neg, md, gv, c)
            update = _rms_clip_one_(pn.mul_(inv_denom), group["clip_threshold"])
            delta = update.mul_(c["step_size"])
            if matrixize:
                delta = delta.reshape_as(grad)
        else:
            v = state["v"]
            v.mul_(c["beta2"]).addcmul_(grad, grad, value=1.0 - c["beta2"])
            if ams_bound:
                max_v = state["max_v"]
                torch.maximum(max_v, v, out=max_v)
                de_nom = max_v.add(1e-15).sqrt_().add_(eps1)
            else:
                de_nom = v.add(1e-15).sqrt_().add_(eps1)
            de_nom.div_(c["bc2_sq"])
            pn = self._pn_one(state, pos, neg, md, grad, c)
            update = _rms_clip_one_(pn.div_(de_nom), group["clip_threshold"])
            delta = update.mul_(c["step_size"])

        if cautious:
            delta = cautious_one_(delta, grad)

        subtract_one_(p, delta, state, bf16_method)

    def _pn_one(
        self,
        state: dict[str, Any],
        pos: str,
        neg: str,
        md: str,
        grad: Tensor,
        c: dict[str, float],
    ) -> Tensor:
        """Per-param pos-neg numerator (EMA the positive buffer with the raw grad)."""
        m_pos = self._dequant_one(state, pos, md, grad)
        m_neg = self._dequant_one(state, neg, md, grad)
        m_pos.mul_(c["beta1_sq"]).add_(grad, alpha=1.0 - c["beta1_sq"])
        self._store_one(state, pos, md, m_pos)
        return m_pos.mul(1.0 + c["beta0"]).add_(m_neg, alpha=-c["beta0"]).mul_(1.0 / c["noise_norm"])

