"""ADOPT — Modified Adam that converges with any beta2 — on kaon's memory backend.

ADOPT (Taniguchi et al. 2024, *ADOPT: Modified Adam Can Converge with Any
:math:`\\beta_2` with the Optimal Rate*, NeurIPS 2024, arXiv:2411.02853),
implemented on top of the precision and memory machinery proven in
:class:`~kaon.adakaon.Adakaon` / :class:`~kaon.adapnm.AdaPNM`. ADOPT is the
**"any-beta2"** optimizer: it removes the dependence of Adam's convergence on the
second-moment decay, so a very large ``beta2`` (its signature default ``0.9999``)
is not just tolerated but *preferred*.

**The two fixes vs Adam (and why the ordering is load-bearing).**
Adam's update normalizes the *current* gradient ``g_t`` by a second moment
``v_t`` that itself includes ``g_t`` — that correlation is what breaks Adam's
convergence for some ``beta2``. ADOPT fixes it with two changes:

1. **v lags by one step.** The normalizer ``v`` reflects grads up to ``t-1``
   only; ``g_t`` is folded into ``v`` *after* it has been used. So the
   normalization of the current update is **independent of ``g_t``**.
2. **Normalize before the momentum EMA.** ADOPT normalizes the gradient by
   ``sqrt(v)`` *first*, then takes the first-moment EMA of the *normalized*
   gradient (Adam takes the EMA of the raw gradient and normalizes the result).

The official per-step ordering (matching ``iShohei220/adopt`` ``src/adopt/adopt.py``
and the kozistr ``pytorch_optimizer`` port), with the internal step 0-indexed:

.. code-block:: text

    # step 0 (first .step() call): initialize v and DO NOT update p
    v = g_0 ** 2            # raw square, no beta2 EMA, no bias correction
    # (decoupled weight decay is also skipped on step 0)

    # step t >= 1, with v holding the second moment from grads up to t-1:
    p          *= (1 - lr * weight_decay)          # decoupled (AdamW) WD, skipped at t=0
    denom       = clamp(sqrt(v), min=eps)          # eps is a FLOOR (max(sqrt v, eps))
    normed_grad = g_t / denom                      # normalize by the PRE-update v
    normed_grad = clamp(normed_grad, -c_t, +c_t)   # c_t = step ** 0.25  (Algorithm 2 clip)
    m           = beta1 * m + (1 - beta1) * normed_grad
    p          -= lr * m
    v           = beta2 * v + (1 - beta2) * g_t ** 2   # fold g_t in AFTER using it

There is **no bias correction** on either moment (the step-0 ``v`` init and the
v-lag are what make this correct — the paper's whole point), unlike Adam/AdaPNM.

**The clip (which arXiv revision).** The first arXiv revision normalized with a
plain ``sqrt(v) + eps`` denominator; the revised paper's **Algorithm 2** (the
practical version, and the current official default) adds the per-step clip
``clamp(g/max(sqrt(v),eps), c_t)`` with ``c_t = step ** 0.25`` (``step`` is the
0-indexed counter, so the very first *updating* step uses ``c_1 = 1``). This
implementation follows **Algorithm 2** and matches the official default
(``clip_lambda = lambda step: step ** 0.25``); set ``clip=False`` for the
revision-1 (unclipped) behaviour.

**The factored second moment.** ``v`` reuses Adakaon's backend: ``ndim >= 2``
weights factor ``v`` into row+column EMAs (conv kernels matrixized to
``[out, in*kh*kw]`` first), ``ndim == 1`` keeps a full per-coordinate ``v``. The
denominator ``1/sqrt(v)`` is the ``r_factor * c_factor`` reconstruction
(:func:`kaon._factored.factored_inv_sqrt_factors`). ADOPT's ``eps`` is a *floor*
on ``sqrt(v)`` — i.e. a **cap** ``1/eps`` on the reconstructed inverse-denominator
``1/sqrt(v)`` — which is how it is applied on both paths (the 1-D path could clamp
``sqrt(v)`` directly; capping ``1/sqrt(v)`` at ``1/eps`` is the same operation and
is what the factored path must do since it never materializes ``sqrt(v)``). The
factored ``eps1`` (Adafactor's pre-reduction stabilizer) is left at **0** to match
the official, which adds no eps inside ``g**2``.

**The v-lag in BOTH paths (the parity-critical part).** ADOPT's distinctive
ordering means the second moment used to normalize ``g_t`` must be read **before**
``g_t`` is folded in. The per-param path reads ``v`` (or the factored
reconstruction), forms the update, and only then calls the v-EMA. The foreach path
does the **same read-before-write**: it reads the stacked factors / full ``v``,
computes ``inv_denom`` from the *pre-update* state, builds the whole delta, and
only afterwards EMAs ``v`` with ``g_t**2`` and writes it back. Getting this
ordering identical in both paths is what makes ``foreach == per-param`` bit-exact;
the parity test exercises >= 6 steps so the v-lag and the ``step**0.25`` clip
schedule are both covered.

**What is reused vs new.** Reused from Adakaon/AdaPNM: the factored second-moment
helpers (:mod:`kaon._factored`), the quantized-momentum storage + read/requant
primitives in :mod:`kaon._momentum_codec`, the bf16-correct weight write
(:mod:`kaon._backend`), Gradient Centralization, cautious masking,
``load_state_dict_preserving_dtypes`` and the bucketed foreach batching. New here:
the **v-lag** (read-old-v / fold-new-v-after), the **normalize-then-EMA** ordering
(the momentum EMA is of the *normalized, clipped* gradient — so the shared codec's
Adam-style ``ema_*`` helpers, which EMA the raw update, are used with the
*normalized* gradient as their ``update``), the step-0 init/skip, and the
``step**0.25`` clip schedule.

It is a standard ``torch.optim.Optimizer`` with a single per-parameter step, so it
drops into per-parameter / gradient-release training loops unchanged.
"""

from __future__ import annotations

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
    subtract_batched_,
    subtract_one_,
)
from kaon._autolr import AutoLRMixin
from kaon._factored import factored_inv_sqrt_factors, update_factored_state
from kaon._momentum_codec import (
    _FOURBIT_BLOCK,
    _make_codec,
    load_state_dict_preserving_dtypes,
)

__all__ = ["ADOPT"]

MomentumDtype = Literal["bfloat16", "float32", "int8", "4bit"]

# Performance / memory knobs mirror Adakaon (one momentum + factored v).
_STACK_BYTES_PER_ELEM = 48


class ADOPT(AutoLRMixin, Optimizer):
    """ADOPT (any-beta2 modified Adam) on Adakaon's memory backend.

    Args:
        params: parameters or param-group dicts.
        lr: learning rate.
        betas: ``(beta1, beta2)``. ``beta1`` is the first-moment EMA decay (of the
            *normalized* gradient); ``beta2`` is the (factored) second-moment decay.
            Default ``(0.9, 0.9999)`` — ADOPT's signature: a very high ``beta2`` is
            preferred (the paper's whole point is convergence is insensitive to it).
        eps: a **floor** on ``sqrt(v)`` in the normalizer (``denom = max(sqrt(v),
            eps)``), equivalently a cap of ``1/eps`` on the reconstructed
            ``1/sqrt(v)``. Default ``1e-6`` (the official default). Applied on both
            the factored and 1-D paths; the factored Adafactor ``eps1`` is left at 0
            to match the official (no eps inside ``g**2``).
        weight_decay: decoupled (AdamW-style) weight decay, ``p *= (1 - lr*wd)``,
            applied *before* the moment ops and **skipped on the step-0 init**
            (matching the official ``decouple=True`` path).
        clip: enable ADOPT's Algorithm-2 per-step clip of the normalized gradient to
            ``[-c_t, +c_t]`` with ``c_t = step ** 0.25`` (``step`` 0-indexed, so the
            first updating step clips at 1.0). **On by default** (the official
            current default). ``False`` recovers the unclipped revision-1 behaviour.
        cautious: cautious masking (Liang et al. 2024) on the final step vs the
            gradient. **On by default**, consistent with the rest of kaon.
        gradient_centralization: subtract the per-output-row gradient mean on
            ``ndim>=2`` weights before the step (Yong et al. 2020). **On by default**
            (kaon-wide); pin ``False`` for an exact match to the reference ADOPT math.
        momentum_dtype: storage for the first moment — ``"bfloat16"`` (default),
            ``"float32"``, ``"int8"`` (per-row absmax) or ``"4bit"`` (per-block
            absmax, nibble-packed). Same layout as Adakaon, so checkpoints resume
            bit-exactly via ``load_state_dict``.
        momentum_4bit_block: block size for ``momentum_dtype="4bit"`` (default 128;
            ``0``/negative means whole-tensor).
        bf16_method: low-precision weight-update strategy —
            ``"stochastic_rounding"`` (default), ``"kahan"`` (+2 B/param), ``"none"``.
        foreach: batch the step across params with stacked multi-tensor ops. Default
            ``True``; numerically matches the per-parameter path (the v-lag read/write
            ordering is identical in both — see the module docstring).
        foreach_batch_cutoff: per-tensor element cap above which a weight loops
            instead of stacking (default ``2_000_000``).
        foreach_stack_budget: max elements per stacked chunk. ``None`` (default)
            adapts to free VRAM; an int pins a fixed cap.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.9999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        *,
        clip: bool = True,
        cautious: bool = True,
        gradient_centralization: bool = True,
        momentum_dtype: MomentumDtype = "bfloat16",
        momentum_4bit_block: int = _FOURBIT_BLOCK,
        bf16_method: str = "stochastic_rounding",
        foreach: bool = True,
        foreach_batch_cutoff: int = FOREACH_BATCH_CUTOFF,
        foreach_stack_budget: int | None = None,
        auto_lr: bool = False,
        auto_lr_scale: float = 1.0,
        auto_lr_fuse_rel: float = 20.0,
        auto_lr_d0: float | None = None,
    ) -> None:
        beta1, beta2 = float(betas[0]), float(betas[1])
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"betas[0] must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"betas[1] must be in [0, 1), got {beta2}")
        if lr < 0.0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
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
            "eps": float(eps),
            "weight_decay": weight_decay,
            "clip": clip,
            "cautious": cautious,
            "gradient_centralization": gradient_centralization,
            "momentum_dtype": momentum_dtype,
            "momentum_4bit_block": momentum_4bit_block,
            "bf16_method": bf16_method,
            "step": 0,
        }
        super().__init__(params, defaults)
        self._codec = _make_codec(momentum_dtype)
        self._foreach = foreach
        self._foreach_batch_cutoff = foreach_batch_cutoff
        self._foreach_stack_budget = foreach_stack_budget

        # Composable parameter-free LR (update-space DoWG) via AutoLRMixin. off -> zero overhead.
        self._init_autolr(auto_lr, auto_lr_scale, auto_lr_fuse_rel, auto_lr_d0)

    # ------------------------------------------------------------------- state
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
        self._codec.init_state(state, grad, group)
        if is_low_precision(p) and group["bf16_method"] == "kahan":
            state["shift"] = torch.zeros_like(p)

    # -------------------------------------------------------------------- step
    @torch.no_grad()
    def _step_impl(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            for p in params:
                if p.grad.is_sparse:
                    raise RuntimeError("ADOPT does not support sparse gradients")
            group["step"] += 1
            if not params:
                continue
            if group["gradient_centralization"]:
                centralize_grads_(params)
            if self._foreach and self._group_foreach_eligible(group):
                chunk_budget = foreach_budget(
                    self._foreach_stack_budget, self._foreach_batch_cutoff,
                    _STACK_BYTES_PER_ELEM, params[0].device,
                )
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

    def state_dict(self) -> dict[str, Any]:
        """Base state + the auto_lr tuner blob (via AutoLRMixin) when auto_lr is on."""
        return self._autolr_state_dict(super().state_dict())

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state, preserving the quantized momentum's stored dtype.

        torch's default ``load_state_dict`` upcasts every state tensor to the
        param's dtype (fp32), which would silently inflate bf16/int8/4bit momentum
        back to fp32 on resume. Delegate to the shared dtype-preserving helper.
        """
        self._autolr_load(state_dict, lambda sd: load_state_dict_preserving_dtypes(self, sd))

    # ----------------------------------------------------------- coefficients
    @staticmethod
    def _coeffs(group: dict[str, Any]) -> dict[str, float]:
        """Per-step scalar coefficients shared by the per-param and foreach paths.

        ``group["step"]`` is 1 on the first ``.step()`` call; the official ADOPT
        counter is 0-indexed, so ``ostep = step - 1`` is the official step. The
        ``ostep == 0`` step only initializes ``v`` and skips the param update; the
        clip uses ``c_t = ostep ** 0.25`` (the first updating step, ``ostep == 1``,
        clips at 1.0).
        """
        beta1, beta2 = group["betas"]
        ostep = group["step"] - 1
        clip = ostep ** 0.25 if group["clip"] else None
        return {
            "beta1": beta1,
            "beta2": beta2,
            "lr": group["lr"],
            "eps": group["eps"],
            "ostep": ostep,
            "clip": clip,
        }

    # ----------------------------------------------------------------- foreach
    @staticmethod
    def _group_foreach_eligible(group: dict[str, Any]) -> bool:
        return group["bf16_method"] != "kahan"  # kahan needs a per-param shift buffer

    @staticmethod
    def _param_foreach_eligible(p: Tensor, group: dict[str, Any], cutoff: int) -> bool:
        if p.ndim == 0 or p.numel() > cutoff:
            return False
        if (
            group["bf16_method"] == "stochastic_rounding"
            and is_low_precision(p)
            and p.dtype != torch.bfloat16  # fp16+SR unsupported -> per-param (raises)
        ):
            return False
        if p.ndim > 2:
            return p.data.is_contiguous() and p.grad.is_contiguous()
        return True

    @torch.no_grad()
    def _step_foreach(self, params: list[Tensor], group: dict[str, Any], budget: int) -> None:
        """Batched step. Factored (ndim>=2) and non-factored (ndim==1) buckets, by shape."""
        c = self._coeffs(group)

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
                self._factored_bucket(plist[i:i + stepn], eff, matrixize, c, group)
        for (length, _dtype), plist in flat_buckets.items():
            stepn = max(1, budget // max(length, 1))
            for i in range(0, len(plist), stepn):
                self._nonfactored_bucket(plist[i:i + stepn], length, c, group)

    @torch.no_grad()
    def _factored_bucket(
        self,
        plist: list[Tensor],
        eff: tuple[int, int],
        matrixize: bool,
        c: dict[str, float],
        group: dict[str, Any],
    ) -> None:
        R, C = eff  # noqa: N806
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

        if c["ostep"] == 0:
            # Step-0 init: v = g_0^2 (no EMA, no WD, no param update). Mirror the
            # factored row/col reductions of g^2 (eps1 = 0, like the official).
            grad_sq = grad * grad
            torch._foreach_copy_(rows, list(grad_sq.mean(dim=-1).unbind(0)))
            torch._foreach_copy_(cols, list(grad_sq.mean(dim=-2).unbind(0)))
            return

        # Decoupled weight decay BEFORE the moment ops.
        if wd != 0:
            torch._foreach_mul_([mat(p.data) for p in plist], 1.0 - group["lr"] * wd)

        # --- normalize by the PRE-update (lagged) v ---
        r_factor = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)  # [N, R, 1]
        c_factor = col.rsqrt().unsqueeze(-2)                                       # [N, 1, C]
        inv_denom = (r_factor * c_factor).clamp_(max=1.0 / c["eps"])               # 1/max(sqrt v, eps)
        normed = grad * inv_denom
        if c["clip"] is not None:
            normed.clamp_(-c["clip"], c["clip"])

        # --- momentum EMA of the NORMALIZED grad, then p -= lr * m ---
        m = self._codec.ema_stacked(states, normed, mat, (R, C), c["beta1"])       # [N, R, C]
        delta = m.mul_(c["lr"])

        if cautious:
            delta = cautious_batched_(delta, grad)
        subtract_batched_([mat(p.data) for p in plist], delta, bf16_method)

        # --- fold g_t into v AFTER it has been used (the v-lag) ---
        grad_sq = grad * grad
        omb2 = 1.0 - c["beta2"]
        row.lerp_(grad_sq.mean(dim=-1), omb2)
        col.lerp_(grad_sq.mean(dim=-2), omb2)
        torch._foreach_copy_(rows, list(row.unbind(0)))
        torch._foreach_copy_(cols, list(col.unbind(0)))

    @torch.no_grad()
    def _nonfactored_bucket(
        self,
        plist: list[Tensor],
        length: int,
        c: dict[str, float],
        group: dict[str, Any],
    ) -> None:
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]

        states = [self.state[p] for p in plist]
        vs = [s["v"] for s in states]

        grad = torch.stack([p.grad.float() for p in plist])               # [N, L]
        v = torch.stack(vs)                                               # [N, L]

        if c["ostep"] == 0:
            torch._foreach_copy_(vs, list((grad * grad).unbind(0)))       # v = g_0^2
            return

        if wd != 0:
            torch._foreach_mul_([p.data for p in plist], 1.0 - group["lr"] * wd)

        # normalize by the PRE-update v: denom = max(sqrt(v), eps).
        denom = v.sqrt().clamp_(min=c["eps"])
        normed = grad / denom
        if c["clip"] is not None:
            normed.clamp_(-c["clip"], c["clip"])

        m = self._codec.ema_stacked(states, normed, lambda t: t, (length,), c["beta1"])  # [N, L]
        delta = m.mul_(c["lr"])

        if cautious:
            delta = cautious_batched_(delta, grad)
        subtract_batched_([p.data for p in plist], delta, bf16_method)

        # fold g_t into v AFTER use.
        v.mul_(c["beta2"]).addcmul_(grad, grad, value=1.0 - c["beta2"])
        torch._foreach_copy_(vs, list(v.unbind(0)))

    # ---------------------------------------------------------- per-parameter
    @torch.no_grad()
    def _step_one_param(self, p: Tensor, group: dict[str, Any]) -> None:
        c = self._coeffs(group)
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]

        state = self.state[p]
        if not state:
            self._init_state(p, state, group)

        grad = p.grad if p.grad.dtype == torch.float32 else p.grad.float()
        ndim = grad.ndim
        factored = ndim >= 2

        if c["ostep"] == 0:
            # Step-0 init: v = g_0^2, no param update, no WD.
            if factored:
                matrixize = ndim > 2
                gv = grad.reshape(grad.shape[0], -1) if matrixize else grad
                grad_sq = gv * gv
                state["row"].copy_(grad_sq.mean(dim=-1))
                state["col"].copy_(grad_sq.mean(dim=-2))
            else:
                state["v"].copy_(grad * grad)
            return

        # Decoupled weight decay BEFORE the moment ops.
        if wd != 0:
            p.data.mul_(1.0 - group["lr"] * wd)

        if factored:
            matrixize = ndim > 2
            gv = grad.reshape(grad.shape[0], -1) if matrixize else grad
            # normalize by the PRE-update factored v (read BEFORE the EMA update).
            r_factor, c_factor = factored_inv_sqrt_factors(state["row"], state["col"])
            inv_denom = (r_factor * c_factor).clamp_(max=1.0 / c["eps"])
            normed = gv * inv_denom
            if c["clip"] is not None:
                normed.clamp_(-c["clip"], c["clip"])
            # Codec stores momentum in the param's ORIGINAL shape; reshape the
            # matrixized normed grad back before the EMA (matches Adakaon).
            if matrixize:
                normed = normed.view_as(grad)
            m = self._codec.ema_one(state, normed, c["beta1"])
            delta = m.mul_(c["lr"])
            # fold g_t into v AFTER use (eps1 = 0 to match the official).
            update_factored_state(gv, state["row"], state["col"], c["beta2"], 0.0)
        else:
            v = state["v"]
            denom = v.sqrt().clamp_(min=c["eps"])
            normed = grad / denom
            if c["clip"] is not None:
                normed.clamp_(-c["clip"], c["clip"])
            m = self._codec.ema_one(state, normed, c["beta1"])
            delta = m.mul_(c["lr"])
            # fold g_t into v AFTER use.
            v.mul_(c["beta2"]).addcmul_(grad, grad, value=1.0 - c["beta2"])

        if cautious:
            delta = cautious_one_(delta, grad)
        subtract_one_(p, delta, state, bf16_method)
