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
``beta1**2`` first-moment decay with ``beta1`` bias correction, and the read-it-
yourself EMA (the shared codec's ``ema_*`` helpers do an Adam *momentum-of-update*
and update a *single* buffer, so they cannot be reused verbatim; AdaPNM uses the
codec's storage + *read-only* ``_dequant``/requant primitives and runs the
raw-gradient EMA on the positive buffer itself).

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

from kaon._factored import factored_inv_sqrt_factors, update_factored_state
from kaon._momentum_codec import (
    _FOURBIT_BLOCK,
    _dequant_4bit,
    _dequant_4bit_stacked,
    _quant_4bit,
    _quant_4bit_stacked,
    _quant_int8,
    _quant_int8_stacked,
    load_state_dict_preserving_dtypes,
)
from kaon._stochastic_rounding import add_stochastic_

__all__ = ["AdaPNM"]

_LOW_PRECISION = (torch.bfloat16, torch.float16)
MomentumDtype = Literal["bfloat16", "float32", "int8", "4bit"]

# Performance / memory knobs mirror Adakaon (see that module for the rationale).
_FOREACH_BATCH_CUTOFF = 2_000_000
_DEFAULT_STACK_ELEMS = 64_000_000
_STACK_SAFETY_FRACTION = 0.10
_STACK_BYTES_PER_ELEM = 64  # two momenta + factored v: a touch above Adakaon's 48
_MIN_STACK_ELEMS = 262_144


def _rms(t: Tensor) -> Tensor:
    return t.norm(2) / math.sqrt(max(t.numel(), 1))


def _is_low_precision(t: Tensor) -> bool:
    return t.dtype in _LOW_PRECISION


class AdaPNM(Optimizer):
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
        cautious: bool = True,
        ams_bound: bool = False,
        momentum_dtype: MomentumDtype = "bfloat16",
        momentum_4bit_block: int = _FOURBIT_BLOCK,
        bf16_method: str = "stochastic_rounding",
        foreach: bool = True,
        foreach_batch_cutoff: int = _FOREACH_BATCH_CUTOFF,
        foreach_stack_budget: int | None = None,
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
        defaults = {
            "lr": lr,
            "betas": (beta1, beta2),
            "beta0": float(beta0),
            "eps": float(eps),
            "weight_decay": weight_decay,
            "cautious": cautious,
            "ams_bound": ams_bound,
            "momentum_dtype": momentum_dtype,
            "momentum_4bit_block": momentum_4bit_block,
            "bf16_method": bf16_method,
            "step": 0,
        }
        super().__init__(params, defaults)
        self._foreach = foreach
        self._foreach_batch_cutoff = foreach_batch_cutoff
        self._foreach_stack_budget = foreach_stack_budget

    # ------------------------------------------------------------------- state
    @staticmethod
    def _block_size(grad: Tensor, group: dict[str, Any]) -> int:
        bs = group["momentum_4bit_block"]
        numel = grad.numel()
        return numel if bs <= 0 else (min(bs, numel) if numel > 0 else 1)

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
            bs = self._block_size(grad, group)
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
        if _is_low_precision(p) and group["bf16_method"] == "kahan":
            state["shift"] = torch.zeros_like(p)

    # -------------------------------------------------- momentum read / write
    @staticmethod
    def _dequant_one(state: dict[str, Any], prefix: str, md: str, like: Tensor) -> Tensor:
        """Read one stored momentum back as a fresh fp32 tensor shaped like ``like``.

        Buffers are stored in the param's original shape; conv kernels are
        matrixized at use-site so ``like`` may be the ``[R, C]`` view — reshape to
        match (the per-row int8 scale's row grouping is preserved across the
        reshape because dim-0 is unchanged).
        """
        if md in ("bfloat16", "float32"):
            return state[prefix].float().reshape_as(like)
        if md == "int8":
            return state[prefix].float().mul_(state[f"{prefix}_scale"]).reshape_as(like)
        m = _dequant_4bit(
            state[prefix], state[f"{prefix}_scale"], state[f"{prefix}_numel"], state[f"{prefix}_block"]
        )
        return m.view_as(like)

    @staticmethod
    def _store_one(state: dict[str, Any], prefix: str, md: str, m_fp32: Tensor) -> None:
        """Write an updated fp32 momentum back into the configured storage layout.

        ``m_fp32`` may be the matrixized ``[R, C]`` view; the int8 per-row scale and
        the float buffer both reduce/store over the param's *original* shape, so
        reshape back first (dim-0 — the int8 row axis — is preserved).
        """
        if md in ("bfloat16", "float32"):
            tgt = state[prefix]
            tgt.copy_(m_fp32.reshape(tgt.shape))
        elif md == "int8":
            m_orig = m_fp32.reshape(state[prefix].shape)
            state[prefix], state[f"{prefix}_scale"] = _quant_int8(m_orig)
        else:  # 4bit
            packed, scale, _ = _quant_4bit(m_fp32, state[f"{prefix}_block"])
            state[prefix], state[f"{prefix}_scale"] = packed, scale

    @staticmethod
    def _dequant_stacked(
        states: list[dict[str, Any]], prefix: str, md: str, shape: tuple[int, ...]
    ) -> Tensor:
        """Stacked fp32 momentum ``[N, *shape]`` from per-param storage (see Lion)."""
        n = len(states)
        per = 1
        for d in shape:
            per *= d
        if md in ("bfloat16", "float32"):
            # Buffers are stored in the param's original shape; reshape to the
            # effective (matrixized [R, C] / flat [L]) view so stacking aligns.
            return torch.stack([s[prefix].reshape(shape) for s in states]).float()
        if md == "int8":
            row = shape[0] if len(shape) >= 2 else 1
            rest = max(per // row, 1)
            m = torch.stack([s[prefix].reshape(row, rest) for s in states]).float()  # [N, R, rest]
            scale = torch.stack([s[f"{prefix}_scale"].reshape(row, 1) for s in states])  # [N, R, 1]
            return m.mul_(scale).reshape((n, *shape))
        packed = torch.stack([s[prefix] for s in states])
        sc = torch.stack([s[f"{prefix}_scale"] for s in states])
        bs = states[0][f"{prefix}_block"]
        return _dequant_4bit_stacked(packed, sc, per, bs).reshape((n, *shape))

    @staticmethod
    def _store_stacked(
        states: list[dict[str, Any]], prefix: str, md: str, m_fp32: Tensor
    ) -> None:
        """Write stacked fp32 momentum ``[N, *shape]`` back into per-param storage."""
        n = m_fp32.shape[0]
        shape = tuple(m_fp32.shape[1:])
        per = 1
        for d in shape:
            per *= d
        if md in ("bfloat16", "float32"):
            ms = [s[prefix].reshape(shape) for s in states]
            torch._foreach_copy_(ms, list(m_fp32.unbind(0)))
        elif md == "int8":
            row = shape[0] if len(shape) >= 2 else 1
            rest = max(per // row, 1)
            q, new_scale = _quant_int8_stacked(m_fp32.reshape(n, row, rest))  # [N,R,rest]->[N,R,1]
            torch._foreach_copy_(
                [s[prefix].reshape(row, rest) for s in states], list(q.unbind(0))
            )
            for s, sc in zip(states, new_scale.unbind(0), strict=True):  # sc: [R, 1]
                s[f"{prefix}_scale"] = sc.reshape(row, 1) if len(shape) >= 2 else sc.reshape(1)
        else:  # 4bit
            bs = states[0][f"{prefix}_block"]
            new_packed, new_scale = _quant_4bit_stacked(m_fp32.reshape(n, per), bs)
            torch._foreach_copy_([s[prefix] for s in states], list(new_packed.unbind(0)))
            for s, sc in zip(states, new_scale.unbind(0), strict=True):
                s[f"{prefix}_scale"].copy_(sc)

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
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            for p in params:
                if p.grad.is_sparse:
                    raise RuntimeError("AdaPNM does not support sparse gradients")
            group["step"] += 1
            if not params:
                continue
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
        """Restore state, preserving both quantized momenta's stored dtype.

        torch's default ``load_state_dict`` upcasts every state tensor to the
        param's dtype (fp32), which would silently inflate bf16/int8/4bit momenta
        back to fp32 on resume. Delegate to the shared helper that restores each
        tensor to how it was checkpointed.
        """
        load_state_dict_preserving_dtypes(self, state_dict)

    # ----------------------------------------------------------- coefficients
    @staticmethod
    def _coeffs(group: dict[str, Any]) -> dict[str, float]:
        """All per-step scalar coefficients (shared by the per-param and foreach paths)."""
        beta1, beta2 = group["betas"]
        beta0 = group["beta0"]
        step = group["step"]
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
            and p.dtype != torch.bfloat16  # fp16+SR unsupported -> per-param (raises)
        ):
            return False
        if p.ndim > 2:
            # Matrixized conv writes back through a reshaped view -> needs contiguity.
            return p.data.is_contiguous() and p.grad.is_contiguous()
        return True

    @torch.no_grad()
    def _step_foreach(self, params: list[Tensor], group: dict[str, Any], budget: int) -> None:
        """Batched step. Factored (ndim>=2) and non-factored (ndim==1) buckets, by shape."""
        c = self._coeffs(group)
        md = group["momentum_dtype"]
        pos, neg = self._pos_neg_prefixes(group["step"])

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
            stepn = max(1, budget // max(eff[0] * eff[1], 1))
            for i in range(0, len(plist), stepn):
                self._factored_bucket(plist[i:i + stepn], eff, matrixize, md, pos, neg, c, group)
        for (length, _dtype), plist in flat_buckets.items():
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
        N = len(plist)  # noqa: N806
        eps1 = group["eps"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]

        def mat(t: Tensor) -> Tensor:
            return t.view(R, C) if matrixize else t

        states = [self.state[p] for p in plist]
        rows = [s["row"] for s in states]
        cols = [s["col"] for s in states]

        grad = torch.stack([mat(p.grad).float() for p in plist])          # [N, R, C]
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

        delta = pn.mul_(inv_denom).mul_(c["step_size"])                            # full step

        if cautious:
            mask = (delta * grad > 0).to(delta.dtype)
            denom = mask.reshape(N, -1).mean(dim=1).clamp_(min=1e-8).view(N, 1, 1)
            delta = delta.mul_(mask).div_(denom)

        self._write_back([mat(p.data) for p in plist], delta, bf16_method)

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
        N = len(plist)  # noqa: N806
        eps1 = group["eps"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]
        ams_bound = group["ams_bound"]

        states = [self.state[p] for p in plist]
        vs = [s["v"] for s in states]

        grad = torch.stack([p.grad.float() for p in plist])               # [N, L]
        v = torch.stack(vs)                                               # [N, L]

        if wd != 0:
            self._apply_decoupled_wd_batched(plist, lambda t: t, group["lr"] * wd)

        # Full per-coordinate second moment (1-D). eps here goes on the denominator
        # (kozistr), NOT folded into grad^2; eps1==eps for the 1-D path.
        v.mul_(c["beta2"]).addcmul_(grad, grad, value=1.0 - c["beta2"])
        torch._foreach_copy_(vs, list(v.unbind(0)))

        if ams_bound:
            max_vs = [s["max_v"] for s in states]
            max_v = torch.stack(max_vs)
            torch.maximum(max_v, v, out=max_v)
            torch._foreach_copy_(max_vs, list(max_v.unbind(0)))
            de_nom = max_v.add(1e-15).sqrt_().add_(eps1)
        else:
            de_nom = v.add(1e-15).sqrt_().add_(eps1)
        de_nom.div_(c["bc2_sq"])                                          # v_hat denom

        pn = self._pn_stacked(states, pos, neg, md, (length,), grad, c)   # [N, L]
        delta = pn.div_(de_nom).mul_(c["step_size"])

        if cautious:
            mask = (delta * grad > 0).to(delta.dtype)
            denom = mask.mean(dim=1).clamp_(min=1e-8).view(N, 1)
            delta = delta.mul_(mask).div_(denom)

        self._write_back([p.data for p in plist], delta, bf16_method)

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

    @staticmethod
    def _write_back(pviews: list[Tensor], delta: Tensor, bf16_method: str) -> None:
        """Apply ``p -= delta`` to a bucket of (matrixized) param views ``[N, *shape]``.

        Only the bf16 + stochastic-rounding path needs a materialized stacked-weights
        tensor (``add_stochastic_`` works on the stack); every other case subtracts the
        delta slices straight into the param views with ``_foreach_sub_`` — skipping the
        stack-weights allocation *and* the copy-back, which is pure overhead in the
        many-tiny-tensor (LoRA) regime.
        """
        p0 = pviews[0]
        if (
            _is_low_precision(p0)
            and bf16_method == "stochastic_rounding"
            and p0.dtype == torch.bfloat16
        ):
            weights = torch.stack(pviews)
            add_stochastic_(weights, delta, alpha=-1.0)
            torch._foreach_copy_(pviews, list(weights.unbind(0)))
        elif p0.dtype == delta.dtype:
            torch._foreach_sub_(pviews, list(delta.unbind(0)))
        else:
            torch._foreach_sub_(pviews, [d.to(p0.dtype) for d in delta.unbind(0)])

    # ---------------------------------------------------------- per-parameter
    @torch.no_grad()
    def _step_one_param(self, p: Tensor, group: dict[str, Any]) -> None:
        c = self._coeffs(group)
        md = group["momentum_dtype"]
        eps1 = group["eps"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]
        ams_bound = group["ams_bound"]
        pos, neg = self._pos_neg_prefixes(group["step"])

        state = self.state[p]
        if not state:
            self._init_state(p, state, group)

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
            delta = pn.mul_(inv_denom).mul_(c["step_size"])
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
            delta = pn.div_(de_nom).mul_(c["step_size"])

        if cautious:
            mask = (delta * grad > 0).to(delta.dtype)
            delta = delta.mul_(mask).div_(mask.mean().clamp_(min=1e-8))

        self._apply_subtract(p, delta, state, bf16_method)

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
