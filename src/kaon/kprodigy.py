"""KProdigy — a memory-efficient Prodigy (parameter-free D-adaptation) optimizer.

Prodigy (Mishchenko & Defazio, 2023, arXiv:2306.06101) estimates the distance
``D`` to the solution on the fly and uses it as the effective learning rate, so
you train at ``lr=1.0`` and the optimizer finds the scale itself.

This is a clean reimplementation aimed at kaon's thesis — *memory-efficient
bf16 diffusion fine-tuning* — fixing the issues that plagued the original
``KProdigy`` research repo (whose shipped defaults, ``d_update_freq=5`` and
``use_bias_correction=True``, starved the D-bootstrap and made the effective LR
fail to rise). The D-estimation math here matches the reference Prodigy bit for
bit (with ``gradient_centralization=False`` — the default GC is a gradient
preprocessor on top, a measured held-out-loss win); the *enhancements* are
orthogonal memory savings:

* **bf16 / int8 / 4bit first moment** (``momentum_dtype``) — like ``Adakaon``.
* **factored second moment** (``second_moment="factored"``) — Adafactor-style
  row+column EMA, ~0 state on convs/attention. Experimental: it uses the
  current-``d`` convention for the second moment (the historical-``d`` scaling
  cannot be factored), so it is a small approximation during D-growth — measure
  before trusting it on a new setup.
* **stochastic-rounding bf16 weight updates** (``bf16_method``) — no Kahan
  buffer, no CPU offload, bf16-correct steps.
* **sliced D statistics** (``slice_p``) — compute the ``s``/``p0`` D-estimation
  buffers on every ``p``-th element (~0.3% D error at ``slice_p=11`` for ~11x
  less D-state).

**Engine.** KProdigy is a two-pass optimizer: pass 1 accumulates the global D
statistics + computes the new D, pass 2 applies the weight update. The D math
(pass 1) is the validated reference Prodigy; the *update backend* (pass 2) reuses
``Adakaon``'s full engine — its **foreach batching**, its **momentum codec**
(float32/bfloat16/int8/4bit), **cautious** masking, **conv-aware matrixized
factoring**, and **stochastic-rounding** bf16 weights — with Prodigy's effective
learning rate (``lr × D``) folded into the update. Set ``foreach=False`` for the
per-parameter path.

Memory at ``beta1=0`` (no momentum), ``second_moment="factored"``, ``slice_p=11``
is well under AdamW; even the full-precision default (bf16 momentum + full fp32
second moment) is ~6 B/param vs AdamW's 8.

Unlike ``Adakaon``/``AdaMuon``, Prodigy needs a **global reduction over all
parameters** each step (the D estimate), so it is a two-pass optimizer and does
**not** support the per-parameter / gradient-release loop. Use it as a normal
``optimizer.step()`` optimizer.

Based on Prodigy by Konstantin Mishchenko and Aaron Defazio
(https://github.com/konstmish/prodigy), MIT licensed.
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
    subtract_batched_,
    subtract_one_,
)
from kaon._factored import factored_inv_sqrt_factors, update_factored_state
from kaon._momentum_codec import (
    _FOURBIT_BLOCK,
    _make_codec,
    _MomentumCodec,
    _quant_int8,
    load_state_dict_preserving_dtypes,
)

__all__ = ["KProdigy"]

MomentumDtype = Literal["bfloat16", "float32", "int8", "4bit"]
SecondMoment = Literal["full", "factored"]

# Reuse Adakaon's adaptive-VRAM foreach stacking constants so a KProdigy chunk
# budget never OOMs a full fine-tune (same engine, same safety story).
_STACK_BYTES_PER_ELEM = 48


def _rms(t: Tensor) -> Tensor:
    return t.norm(2) / math.sqrt(max(t.numel(), 1))


class KProdigy(Optimizer):
    """Memory-efficient Prodigy with parameter-free D-adaptation.

    Args:
        params: parameters or param-group dicts.
        lr: learning-rate multiplier. Leave at ``1.0`` — Prodigy adapts the
            scale via ``D``. (For SDXL, ``D`` is the effective LR; it typically
            wants to reach ~1e-4..2e-4.)
        betas: ``(beta1, beta2)``. ``beta1=0`` disables the momentum buffer
            (minimum memory). Default ``(0.9, 0.999)``.
        beta3: D-adaptation EMA coefficient. ``None`` -> ``sqrt(beta2)``.
        eps: denominator floor for numerical stability.
        weight_decay: weight decay (L2 / decoupled).
        decouple: AdamW-style decoupled weight decay (default ``True``).
        use_bias_correction: Adam bias correction. **Default ``False``** — the
            original KProdigy repo defaulted this to ``True`` and it damaged the
            D-bootstrap and convergence; keep it off unless measured otherwise.
        safeguard_warmup: remove ``lr`` from the D-denominator during warmup.
        d0: initial D estimate. Default ``1e-6``.
        d_coef: coefficient on the D estimate (the main tuning knob if D rises
            too slowly / too fast). Default ``1.0``.
        growth_rate: cap on per-step multiplicative D growth. Default ``inf``.
        d_update_freq: update D every N steps. **Default ``1``** (exact). Values
            > 1 trade D accuracy for speed and *starve the D-bootstrap* — the
            original repo's ``5`` is why the LR failed to rise.
        slice_p: compute D statistics on every ``p``-th element (memory). ``1``
            is exact; ``11`` is ~0.3% D error for ~11x less D-state.
        independent_d: separate D per param group (essential for SDXL UNet+TE so
            one component does not burn the other). ``None`` -> auto: on when
            there is more than one param group.
        momentum_dtype: first-moment storage — ``"bfloat16"`` (default, ~2
            B/param), ``"float32"`` (4 B/param), ``"int8"`` (~1 B/param,
            per-row absmax), or ``"4bit"`` (~0.5 B/param, signed linear 4-bit
            with a per-block absmax scale; block size ``momentum_4bit_block``).
        momentum_4bit_block: block size for ``momentum_dtype="4bit"`` (default
            128). ``0``/negative means whole-tensor (single scale).
        second_moment: ``"full"`` (default; fp32, exact) or ``"factored"``
            (Adafactor row+col, ~0 state on >=2-D weights; experimental).
        eps_factored: ``eps1`` added to ``grad**2`` before the factored
            reductions (HF Adafactor convention). Only used when factored.
        cautious: cautious masking (Liang et al. 2024) — zero the update
            coordinates whose sign disagrees with the gradient, renormalized to
            keep the step size. Default ``False``. A no-op when ``beta1=0`` (the
            numerator is the raw grad, so the mask is all-ones).
        bf16_method: weight-update strategy for low-precision params —
            ``"stochastic_rounding"`` (default), ``"kahan"`` (+2 B/param), or
            ``"none"``. No-op on fp32 params.
        factor_conv_as_matrix: reshape 4-D conv kernels to 2-D before factoring
            (the conv-aware fix). Default ``True``.
        foreach: batch the pass-2 update across parameters with multi-tensor
            (stacked) ops — Adakaon's engine — instead of a per-parameter
            Python loop. Default ``True``. The D-estimation pass-1 is always
            per-parameter (it is a global reduction); only the update backend is
            batched. Set ``False`` to force the per-parameter update path.
        foreach_batch_cutoff: per-tensor element count above which a weight is
            updated by the per-parameter loop instead of being stacked (a
            performance knob; default 2_000_000, mirroring Adakaon).
        foreach_stack_budget: max elements in a single stacked foreach chunk.
            ``None`` (default) adapts to free VRAM; an int pins a fixed cap.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1.0,
        betas: tuple[float, float] = (0.9, 0.999),
        beta3: float | None = None,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        decouple: bool = True,
        use_bias_correction: bool = False,
        safeguard_warmup: bool = False,
        d0: float = 1e-6,
        d_coef: float = 1.0,
        growth_rate: float = float("inf"),
        d_update_freq: int = 1,
        slice_p: int = 1,
        independent_d: bool | None = None,
        momentum_dtype: MomentumDtype = "bfloat16",
        momentum_4bit_block: int = _FOURBIT_BLOCK,
        second_moment: SecondMoment = "full",
        eps_factored: float = 1e-30,
        cautious: bool = False,
        gradient_centralization: bool = True,
        bf16_method: str = "stochastic_rounding",
        factor_conv_as_matrix: bool = True,
        foreach: bool = True,
        foreach_batch_cutoff: int = FOREACH_BATCH_CUTOFF,
        foreach_stack_budget: int | None = None,
    ) -> None:
        beta1, beta2 = float(betas[0]), float(betas[1])
        if not d0 > 0.0:
            raise ValueError(f"d0 must be > 0, got {d0}")
        if not lr > 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not eps > 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"betas[0] must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"betas[1] must be in [0, 1), got {beta2}")
        if d_update_freq < 1:
            raise ValueError(f"d_update_freq must be >= 1, got {d_update_freq}")
        if slice_p < 1:
            raise ValueError(f"slice_p must be >= 1, got {slice_p}")
        if momentum_dtype not in ("bfloat16", "float32", "int8", "4bit"):
            raise ValueError(
                f"momentum_dtype must be bfloat16/float32/int8/4bit, got {momentum_dtype!r}"
            )
        if second_moment not in ("full", "factored"):
            raise ValueError(f"second_moment must be full/factored, got {second_moment!r}")
        if bf16_method not in ("stochastic_rounding", "kahan", "none"):
            raise ValueError(f"bf16_method must be stochastic_rounding/kahan/none, got {bf16_method!r}")
        if foreach_batch_cutoff < 1:
            raise ValueError(f"foreach_batch_cutoff must be >= 1, got {foreach_batch_cutoff}")

        defaults = {
            "lr": lr,
            "betas": (beta1, beta2),
            "beta3": beta3,
            "eps": eps,
            "weight_decay": weight_decay,
            "decouple": decouple,
            "use_bias_correction": use_bias_correction,
            "safeguard_warmup": safeguard_warmup,
            "d": d0,
            "d0": d0,
            "d_max": d0,
            "d_numerator": 0.0,
            "d_coef": d_coef,
            "growth_rate": growth_rate,
            "d_update_freq": d_update_freq,
            "slice_p": slice_p,
            "k": 0,
            "momentum_dtype": momentum_dtype,
            "momentum_4bit_block": momentum_4bit_block,
            "second_moment": second_moment,
            "eps_factored": eps_factored,
            "cautious": cautious,
            "gradient_centralization": gradient_centralization,
            "bf16_method": bf16_method,
            "factor_conv_as_matrix": factor_conv_as_matrix,
        }
        self.d0 = d0
        super().__init__(params, defaults)
        # Auto: independent D when the user gave more than one param group
        # (e.g. SDXL UNet + Text Encoder), unless explicitly overridden.
        self._independent_d = (len(self.param_groups) > 1) if independent_d is None else independent_d
        # Adakaon-engine foreach knobs for the pass-2 update backend.
        self._foreach = foreach
        self._foreach_batch_cutoff = foreach_batch_cutoff
        self._foreach_stack_budget = foreach_stack_budget
        # Internal switch to batch pass-1 (the D-estimation reduction + moment
        # EMAs) as well as pass-2. Always True in normal use; exposed only so
        # benchmarks can isolate the pass-1 batching speedup. The result is
        # bit-identical either way, so this never affects numerics.
        self._foreach_pass1 = foreach
        # One momentum codec per dtype string, shared with Adakaon. The codec
        # owns storage + dequant; KProdigy does its own d-scaled EMA in pass 1
        # and reads the momentum back (dequant) in pass 2.
        self._codecs: dict[str, _MomentumCodec] = {}

    def get_d(self) -> float:
        """Current D estimate (effective learning rate) of the first group."""
        return float(self.param_groups[0].get("d", self.d0))

    def _codec(self, group: dict[str, Any]) -> _MomentumCodec:
        md = group["momentum_dtype"]
        codec = self._codecs.get(md)
        if codec is None:
            codec = self._codecs[md] = _make_codec(md)
        return codec

    # -- state -------------------------------------------------------------

    @torch.no_grad()
    def _init_state(self, p: Tensor, state: dict[str, Any], group: dict[str, Any]) -> None:
        beta1 = group["betas"][0]
        slice_p = group["slice_p"]
        sliced = p.flatten()[::slice_p]

        state["step"] = 0
        state["s"] = torch.zeros_like(sliced, dtype=torch.float32)
        # p0: reference point for the D estimate. fp32 (sliced -> small).
        if sliced.norm() > 0:
            state["p0"] = sliced.detach().float().clone()
        else:
            state["p0"] = torch.zeros((), device=p.device, dtype=torch.float32)

        if beta1 > 0:
            # Momentum storage is owned by the shared codec (float/bf16/int8/4bit),
            # exactly as Adakaon allocates it.
            self._codec(group).init_state(state, p, group)

        if group["second_moment"] == "factored" and p.ndim >= 2:
            gv = p if (p.ndim == 2 or not group["factor_conv_as_matrix"]) else p.reshape(p.shape[0], -1)
            state["row"] = torch.zeros(gv.shape[:-1], dtype=torch.float32, device=p.device)
            state["col"] = torch.zeros(gv.shape[:-2] + gv.shape[-1:], dtype=torch.float32, device=p.device)
        else:
            # Full second moment (also the fallback for 1-D params under factored).
            state["v"] = torch.zeros_like(p, dtype=torch.float32)

        if is_low_precision(p) and group["bf16_method"] == "kahan":
            state["shift"] = torch.zeros_like(p)

    # -- step --------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # A "scope" is a set of groups sharing one D estimate. Independent-D
        # gives each group its own scope; otherwise all groups share one.
        if self._independent_d:
            scopes = [[g] for g in self.param_groups]
        else:
            scopes = [list(self.param_groups)]

        for scope in scopes:
            self._step_scope(scope)
        return loss

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state, preserving the quantized first moment's stored dtype.

        torch's default ``load_state_dict`` upcasts every state tensor to the
        param's dtype (fp32), which would silently inflate a bf16/int8/4bit
        ``momentum_dtype`` back to fp32 on resume — losing the memory the codec
        was chosen to save and breaking bit-exact resume. Delegate to the shared
        helper that restores each tensor to how it was checkpointed. (Prodigy's
        ``d``/``step``/``s``/``p0`` bookkeeping rides along in the same state.)
        """
        load_state_dict_preserving_dtypes(self, state_dict)

    @torch.no_grad()
    def _step_scope(self, groups: list[dict[str, Any]]) -> None:
        lead = groups[0]
        if lead["gradient_centralization"]:  # before pass-1 so the D-stats see the centralized grad
            centralize_grads_([p for g in groups for p in g["params"] if p.grad is not None])
        beta1, beta2 = lead["betas"]
        beta3 = lead["beta3"] if lead["beta3"] is not None else math.sqrt(beta2)
        k = lead["k"]
        d = lead["d"]
        d_max = lead["d_max"]
        d0 = lead["d0"]
        d_coef = lead["d_coef"]
        growth_rate = lead["growth_rate"]
        slice_p = lead["slice_p"]
        safeguard_warmup = lead["safeguard_warmup"]
        d_update_freq = lead["d_update_freq"]
        # When groups share one D, only differing lr of 0 is allowed (a frozen
        # component); the active lr drives D. (Independent-D groups each use
        # their own lr, so this is a single-group scope there.)
        lr = max(g["lr"] for g in groups)

        if lead["use_bias_correction"]:
            bias_correction = ((1 - beta2 ** (k + 1)) ** 0.5) / (1 - beta1 ** (k + 1))
        else:
            bias_correction = 1.0
        dlr = d * lr * bias_correction

        should_update_d = (k % d_update_freq) == 0
        d_over_d0 = d / d0

        # ---- pass 1: D estimate + moment EMAs --------------------------------
        # UNCHANGED Prodigy D-estimation: global reduction over all params, the
        # d-scaled momentum EMA, and the (full / factored) second-moment EMA.
        # The foreach path batches every elementwise op with torch.stack /
        # torch._foreach_*, but keeps the *scalar* numerator/denominator fold in
        # the original per-parameter order so the D trajectory is bit-identical.
        d_numerator = lead["d_numerator"] * beta3
        pass1_ctx = {
            "beta1": beta1, "beta2": beta2, "beta3": beta3, "d": d, "dlr": dlr,
            "d_over_d0": d_over_d0, "slice_p": slice_p,
            "safeguard_warmup": safeguard_warmup, "should_update_d": should_update_d,
            "lr": lr,
        }
        if self._foreach_pass1:
            delta_numerator, d_denom, device_seen = self._pass1_foreach(groups, pass1_ctx)
        else:
            delta_numerator, d_denom, device_seen = self._pass1_per_param(groups, pass1_ctx)

        # ---- D update --------------------------------------------------------
        if should_update_d and lr > 0.0:
            denom_val = float(d_denom.item()) if device_seen is not None else 0.0
            if denom_val > 0.0:
                global_num = d_numerator + float(delta_numerator.item())
                d_hat = d_coef * global_num / denom_val
                if d == d0:
                    d = max(d, d_hat)
                d_max = max(d_max, d_hat)
                d = min(d_max, d * growth_rate)
                for group in groups:
                    group["d_numerator"] = global_num
                    group["d"] = d
                    group["d_max"] = d_max
                    group["d_hat"] = d_hat
        # NOTE: the parameter update below keeps the ``dlr`` computed at the top
        # of the step (the *old* d). Reference Prodigy applies the new d only on
        # the next step, and the momentum/second-moment were scaled by the old d,
        # so the d-cancellation in the Adam ratio stays consistent.

        # ---- pass 2: apply updates (Adakaon engine) -----------------------
        for group in groups:
            params = [p for p in group["params"] if p.grad is not None]
            for p in params:
                self.state[p]["step"] += 1
            self._apply_updates(params, group, d, dlr)
            group["k"] = group["k"] + 1

    # -- pass-1: D estimate + moment EMAs (per-param reference path) --------

    def _collect_grad(
        self, p: Tensor, group: dict[str, Any]
    ) -> Tensor:
        """fp32 gradient with non-decoupled weight decay folded in (pass-1 input)."""
        grad = p.grad
        grad_fp32 = grad if grad.dtype == torch.float32 else grad.float()
        if group["weight_decay"] != 0 and not group["decouple"]:
            grad_fp32 = grad_fp32.add(p.detach().float(), alpha=group["weight_decay"])
        return grad_fp32

    @torch.no_grad()
    def _pass1_per_param(
        self, groups: list[dict[str, Any]], ctx: dict[str, Any]
    ) -> tuple[Tensor, Tensor, torch.device | None]:
        """Reference (per-parameter) pass 1. Bit-exact original behaviour."""
        beta1, beta2, beta3 = ctx["beta1"], ctx["beta2"], ctx["beta3"]
        d, dlr, d_over_d0 = ctx["d"], ctx["dlr"], ctx["d_over_d0"]
        slice_p = ctx["slice_p"]
        safeguard_warmup = ctx["safeguard_warmup"]
        should_update_d = ctx["should_update_d"]
        lr = ctx["lr"]

        delta_numerator = torch.zeros((), dtype=torch.float32)
        d_denom = torch.zeros((), dtype=torch.float32)
        device_seen = None

        for group in groups:
            group_lr = group["lr"]
            if group_lr not in (lr, 0.0):
                raise RuntimeError(
                    "KProdigy: groups sharing one D estimate must use the same lr "
                    "(or 0 for a frozen group). Use independent_d=True for per-group lr."
                )
            codec = self._codec(group)
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("KProdigy does not support sparse gradients")
                state = self.state[p]
                if "step" not in state:
                    self._init_state(p, state, group)
                device_seen = p.device

                grad_fp32 = self._collect_grad(p, group)

                if group_lr > 0.0 and should_update_d:
                    s = state["s"]
                    p0 = state["p0"]
                    sliced_g = grad_fp32.flatten()[::slice_p]
                    sliced_p = p.detach().float().flatten()[::slice_p]
                    # numerator term: <grad, p0 - p>, scaled. (a*b).sum avoids
                    # torch.dot (a cuBLAS gemv path that SIGFPEs on some setups).
                    delta_numerator = delta_numerator.to(p.device) if delta_numerator.device != p.device else delta_numerator
                    delta_numerator = delta_numerator + (d_over_d0 * dlr) * (sliced_g * (p0 - sliced_p)).sum()
                    alpha_s = (d_over_d0 * d) if safeguard_warmup else (d_over_d0 * dlr)
                    s.mul_(beta3).add_(sliced_g, alpha=alpha_s)
                    d_denom = d_denom.to(p.device) if d_denom.device != p.device else d_denom
                    d_denom = d_denom + s.abs().sum()

                # First moment EMA, scaled by current d (reference convention).
                if beta1 > 0:
                    self._update_momentum(codec, state, grad_fp32, beta1, d, group["momentum_dtype"])

                # Second moment EMA.
                if "v" in state:
                    state["v"].mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=d * d * (1 - beta2))
                else:
                    gv = self._matrixize(grad_fp32, group)
                    update_factored_state(gv, state["row"], state["col"], beta2, group["eps_factored"])

        return delta_numerator, d_denom, device_seen

    # -- pass-1: foreach (stacked / bucketed) batched path -----------------

    @torch.no_grad()
    def _pass1_foreach(
        self, groups: list[dict[str, Any]], ctx: dict[str, Any]
    ) -> tuple[Tensor, Tensor, torch.device | None]:
        """Batched pass 1: every elementwise op (the D-estimation inner products
        and ``s`` buffer, the d-scaled momentum EMA, and the full/factored
        second-moment EMA) is computed with stacked / ``torch._foreach_*`` kernels,
        bucketed by effective shape the same way pass 2 / Adakaon are.

        The *scalar* numerator/denominator accumulation is still folded in the
        exact original per-parameter order (a cheap left-fold over an ``[N]``
        vector of per-param partials), so the D trajectory is bit-identical to
        :meth:`_pass1_per_param`.
        """
        beta1, beta2, beta3 = ctx["beta1"], ctx["beta2"], ctx["beta3"]
        d, dlr, d_over_d0 = ctx["d"], ctx["dlr"], ctx["d_over_d0"]
        slice_p = ctx["slice_p"]
        safeguard_warmup = ctx["safeguard_warmup"]
        should_update_d = ctx["should_update_d"]
        lr = ctx["lr"]

        # -- collection (original order) ----------------------------------
        # records[i] = (p, grad_fp32, state, group, do_d) in iteration order.
        records: list[tuple[Tensor, Tensor, dict[str, Any], dict[str, Any], bool]] = []
        device_seen = None
        for group in groups:
            group_lr = group["lr"]
            if group_lr not in (lr, 0.0):
                raise RuntimeError(
                    "KProdigy: groups sharing one D estimate must use the same lr "
                    "(or 0 for a frozen group). Use independent_d=True for per-group lr."
                )
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("KProdigy does not support sparse gradients")
                state = self.state[p]
                if "step" not in state:
                    self._init_state(p, state, group)
                device_seen = p.device
                grad_fp32 = self._collect_grad(p, group)
                do_d = group_lr > 0.0 and should_update_d
                records.append((p, grad_fp32, state, group, do_d))

        if not records:
            return (
                torch.zeros((), dtype=torch.float32),
                torch.zeros((), dtype=torch.float32),
                device_seen,
            )

        n = len(records)
        # Per-param scalar partials, kept in record order for the ordered fold.
        num_partials = torch.zeros(n, dtype=torch.float32, device=device_seen)
        denom_partials = torch.zeros(n, dtype=torch.float32, device=device_seen)

        # -- (a) batched D-estimation reduction + s buffer ----------------
        # Bucket the D-stat params by sliced length. Each param contributes a
        # numerator partial <g, p0-p> and a denom partial |s|.sum after the s EMA.
        d_buckets: dict[int, list[int]] = {}
        for i, (_p, _g, _state, _grp, do_d) in enumerate(records):
            if do_d:
                d_buckets.setdefault(records[i][2]["s"].numel(), []).append(i)

        for _len, idxs in d_buckets.items():
            # Stack the sliced grad and (p0 - p) for this bucket.
            sliced_g = torch.stack([
                records[i][1].flatten()[::slice_p] for i in idxs
            ])                                                            # [B, L]
            sliced_p = torch.stack([
                records[i][0].detach().float().flatten()[::slice_p] for i in idxs
            ])                                                            # [B, L]
            p0 = torch.stack([records[i][2]["p0"].expand_as(sliced_p[0]) for i in idxs])
            # numerator partials: <g, p0 - p> per param (tail reduction is
            # bit-identical to a per-tensor .sum()).
            num = (sliced_g * (p0 - sliced_p)).sum(dim=-1)               # [B]
            num_partials[idxs] = num.to(num_partials.dtype)
            # s EMA: s <- beta3*s + alpha_s * g (stacked, bit-identical).
            alpha_s = (d_over_d0 * d) if safeguard_warmup else (d_over_d0 * dlr)
            s_stack = torch.stack([records[i][2]["s"] for i in idxs])    # [B, L]
            s_stack.mul_(beta3).add_(sliced_g, alpha=alpha_s)
            torch._foreach_copy_(
                [records[i][2]["s"] for i in idxs], list(s_stack.unbind(0))
            )
            denom_partials[idxs] = s_stack.abs().sum(dim=-1).to(denom_partials.dtype)

        # -- (b) batched first-moment (momentum) EMA ----------------------
        if beta1 > 0:
            self._pass1_momentum_foreach(records, beta1, d)

        # -- (b) batched second-moment EMA --------------------------------
        self._pass1_second_moment_foreach(records, beta2, d)

        # -- ordered scalar fold (bit-identical accumulation) -------------
        # Reconstruct the EXACT original per-param add sequence:
        #   delta_numerator += (d_over_d0 * dlr) * <g, p0-p>_i
        #   d_denom         += |s_i|.sum
        # in record (group->params) order. This is a cheap O(N) left-fold over
        # the [N] partials (N = #params), not over weight elements.
        const = d_over_d0 * dlr
        delta_numerator = torch.zeros((), dtype=torch.float32, device=device_seen)
        d_denom = torch.zeros((), dtype=torch.float32, device=device_seen)
        scaled_num = num_partials * const
        for i in range(n):
            if records[i][4]:  # do_d
                delta_numerator = delta_numerator + scaled_num[i]
                d_denom = d_denom + denom_partials[i]
        return delta_numerator, d_denom, device_seen

    @torch.no_grad()
    def _pass1_momentum_foreach(
        self, records: list[tuple[Tensor, Tensor, dict[str, Any], dict[str, Any], bool]],
        beta1: float, d: float,
    ) -> None:
        """Batched d-scaled first-moment EMA, bucketed by (momentum_dtype, shape).

        Each dtype reproduces the exact per-param arithmetic of
        :meth:`_update_momentum` (mul_/add_ for float/bf16/int8 with the native
        dim-0 row scale; the shared codec's stacked lerp for 4bit), so the stored
        momentum is bit-identical to the per-param path.
        """
        buckets: dict[tuple[str, tuple[int, ...]], list[int]] = {}
        for i, (p, _g, _state, group, _do_d) in enumerate(records):
            md = group["momentum_dtype"]
            buckets.setdefault((md, tuple(p.shape)), []).append(i)

        for (md, _shape), idxs in buckets.items():
            grads = torch.stack([records[i][1] for i in idxs])          # [B, *shape]
            states = [records[i][2] for i in idxs]
            group = records[idxs[0]][3]
            if md == "bfloat16":
                # bf16 storage EMAs *in bf16*: whether stacking the [*shape] buffers
                # into a [B, *shape] tensor changes the in-place add's rounding is
                # shape-dependent (it does for some shapes at the denormal-scale d0
                # bootstrap), so it cannot be guaranteed bit-identical. The bf16
                # momentum EMA therefore stays a per-tensor loop (the exact original
                # op) — bit-identical by construction. The D-reduction and the
                # second moment, the actual step-cost dominators, are still batched.
                target = d * (1 - beta1)
                for j, s in zip(idxs, states, strict=True):
                    m = s["m"]
                    m.mul_(beta1).add_(records[j][1].to(m.dtype), alpha=target)
            elif md == "float32":
                # fp32 storage EMA is bit-identical stacked (no denormal ambiguity).
                target = d * (1 - beta1)
                ms = [s["m"] for s in states]
                m_stack = torch.stack(ms)                               # [B, *shape]
                m_stack.mul_(beta1).add_(grads.to(m_stack.dtype), alpha=target)
                torch._foreach_copy_(ms, list(m_stack.unbind(0)))
            elif md == "int8":
                # int8 EMA runs in fp32 (dequant -> EMA -> requant), which IS
                # bit-identical stacked. The per-param scale follows _quant_int8:
                # a per dim-0 row scale for ndim>=2 (reduce trailing dims), a single
                # scalar for ndim<=1 (reduce the whole row). The stacked layout is
                # [B, *shape]; reduce all dims after B except dim-0-row.
                target = d * (1 - beta1)
                ndim = records[idxs[0]][0].ndim
                m_stack = torch.stack([s["m"] for s in states]).float()  # [B, *shape]
                # Broadcast each per-param scale ([R,1...] or scalar) under the
                # leading batch dim.
                if ndim >= 2:
                    reduce_dims = tuple(range(2, ndim + 1))
                    scales = torch.stack([s["m_scale"] for s in states])  # [B, R, 1...]
                else:
                    reduce_dims = (1,)
                    scales = torch.stack(
                        [s["m_scale"].reshape(1) for s in states]
                    ).reshape(len(states), 1)                            # [B, 1]
                m_stack.mul_(scales)
                m_stack.mul_(beta1).add_(grads, alpha=target)
                absmax = m_stack.abs().amax(dim=reduce_dims, keepdim=True).clamp_(min=1e-12)
                new_scale = absmax / 127.0
                q = (m_stack / new_scale).round_().clamp_(-127, 127).to(torch.int8)
                torch._foreach_copy_([s["m"] for s in states], list(q.unbind(0)))
                for s, sc in zip(states, new_scale.unbind(0), strict=True):
                    s["m_scale"].copy_(sc.reshape(s["m_scale"].shape))
            else:  # 4bit — shared codec stacked lerp (bit-identical to ema_one)
                upd = grads if d == 1.0 else grads.mul(d)
                self._codec(group).ema_stacked(states, upd, lambda t: t, tuple(records[idxs[0]][0].shape), beta1)

    @torch.no_grad()
    def _pass1_second_moment_foreach(
        self, records: list[tuple[Tensor, Tensor, dict[str, Any], dict[str, Any], bool]],
        beta2: float, d: float,
    ) -> None:
        """Batched second-moment EMA: full ``v`` bucketed by shape, factored
        row/col bucketed by matrixized effective shape. Bit-identical to the
        per-param addcmul / Adafactor reductions."""
        full_buckets: dict[tuple[int, ...], list[int]] = {}
        fac_buckets: dict[tuple[Any, ...], list[int]] = {}
        for i, (p, _g, state, group, _do_d) in enumerate(records):
            if "v" in state:
                full_buckets.setdefault(tuple(p.shape), []).append(i)
            else:
                matrixize = p.ndim > 2 and group["factor_conv_as_matrix"]
                eff = (p.shape[0], p.numel() // p.shape[0]) if matrixize else tuple(p.shape)
                fac_buckets.setdefault((eff, matrixize), []).append(i)

        v_value = d * d * (1 - beta2)
        for _shape, idxs in full_buckets.items():
            grads = torch.stack([records[i][1] for i in idxs])
            v_stack = torch.stack([records[i][2]["v"] for i in idxs])
            v_stack.mul_(beta2).addcmul_(grads, grads, value=v_value)
            torch._foreach_copy_([records[i][2]["v"] for i in idxs], list(v_stack.unbind(0)))

        for (eff, matrixize), idxs in fac_buckets.items():
            group = records[idxs[0]][3]
            R, C = eff  # noqa: N806
            grads = torch.stack([
                (records[i][1].reshape(R, C) if matrixize else records[i][1]) for i in idxs
            ])                                                          # [B, R, C]
            rows = torch.stack([records[i][2]["row"] for i in idxs])    # [B, R]
            cols = torch.stack([records[i][2]["col"] for i in idxs])    # [B, C]
            eps1 = group["eps_factored"]
            grad_sq = grads.pow(2)
            if eps1 > 0:
                grad_sq.add_(eps1)
            rows.lerp_(grad_sq.mean(dim=-1), 1.0 - beta2)
            cols.lerp_(grad_sq.mean(dim=-2), 1.0 - beta2)
            torch._foreach_copy_([records[i][2]["row"] for i in idxs], list(rows.unbind(0)))
            torch._foreach_copy_([records[i][2]["col"] for i in idxs], list(cols.unbind(0)))

    # -- pass-2 update backend (Adakaon engine) --------------------------

    def _apply_updates(self, params: list[Tensor], group: dict[str, Any], d: float, dlr: float) -> None:
        """Apply the Prodigy weight update for every param in ``group``.

        Routes to the batched foreach engine (Adakaon's bucketing) when
        eligible; falls back per-param otherwise. The Prodigy-specific math
        (d-scaled denominator, dlr scaling, decoupled WD, eps floor) is identical
        in both paths.
        """
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
                self._update_foreach(fast, group, d, dlr, chunk_budget)
                for p in slow:
                    self._update_one_param(p, group, d, dlr)
            else:
                for p in params:
                    self._update_one_param(p, group, d, dlr)
        else:
            for p in params:
                self._update_one_param(p, group, d, dlr)

    # -- foreach eligibility / budget (mirrors Adakaon) ------------------


    @staticmethod
    def _group_foreach_eligible(group: dict[str, Any]) -> bool:
        # kahan needs a per-param shift buffer -> per-param path.
        return group["bf16_method"] != "kahan"

    def _param_foreach_eligible(self, p: Tensor, group: dict[str, Any], cutoff: int) -> bool:
        if p.ndim == 0:
            return False
        if p.numel() > cutoff:
            return False
        if (
            group["bf16_method"] == "stochastic_rounding"
            and is_low_precision(p)
            and p.dtype != torch.bfloat16
        ):
            return False
        # factored conv kernels (ndim>2) are matrixized into a view -> contiguity.
        factored = group["second_moment"] == "factored" and p.ndim >= 2 and group["factor_conv_as_matrix"]
        if p.ndim > 2 and factored:
            return p.data.is_contiguous() and p.grad.is_contiguous()
        return True

    # -- momentum EMA (pass 1; D-relevant, kept numerically as before) -----

    @staticmethod
    def _update_momentum(
        codec: _MomentumCodec, state: dict[str, Any], grad_fp32: Tensor, beta1: float, d: float, md: str
    ) -> None:
        """EMA ``m <- beta1*m + (1-beta1)*d*grad`` in the momentum dtype.

        float/bf16/int8 keep the *exact* arithmetic of the original KProdigy
        (mul_/add_) so the D-validated behaviour is byte-identical; 4bit uses the
        shared codec's dequant/EMA/requant.
        """
        target = d * (1 - beta1)
        if md == "int8":
            m = state["m"].float().mul_(state["m_scale"])
            m.mul_(beta1).add_(grad_fp32, alpha=target)
            state["m"], state["m_scale"] = _quant_int8(m)
        elif md == "4bit":
            # m <- beta1*m + (1-beta1)*d*grad == m.lerp_(d*grad, 1-beta1)
            codec.ema_one(state, grad_fp32 if d == 1.0 else grad_fp32.mul(d), beta1)
        else:
            m = state["m"]
            m.mul_(beta1).add_(grad_fp32.to(m.dtype), alpha=target)

    # -- shape helpers (conv-aware factoring) ------------------------------

    @staticmethod
    def _matrixize(t: Tensor, group: dict[str, Any]) -> Tensor:
        if t.ndim > 2 and group["factor_conv_as_matrix"]:
            return t.reshape(t.shape[0], -1)
        return t

    @staticmethod
    def _unmatrixize(t: Tensor, like: Tensor, group: dict[str, Any]) -> Tensor:
        if like.ndim > 2 and group["factor_conv_as_matrix"]:
            return t.view_as(like)
        return t

    # -- per-param update --------------------------------------------------

    @torch.no_grad()
    def _update_one_param(self, p: Tensor, group: dict[str, Any], d: float, dlr: float) -> None:
        eps = group["eps"]
        decay = group["weight_decay"]
        decouple = group["decouple"]
        cautious = group["cautious"]
        bf16_method = group["bf16_method"]
        state = self.state[p]

        grad = p.grad
        grad_fp32 = grad if grad.dtype == torch.float32 else grad.float()

        # numerator: d-scaled momentum (beta1>0) or the raw grad (beta1=0).
        if group["betas"][0] > 0:
            numer = self._codec(group).dequant_one(state, grad_fp32)
        else:
            numer = grad_fp32.clone()

        # denom = sqrt(second moment) floored at d*eps; the O(d) denominator
        # cancels the O(d) numerator (d-scaled momentum / d-scaled grad).
        if "v" in state:
            denom = state["v"].sqrt().clamp_(min=d * eps)
            delta = numer.div_(denom)
        else:
            r_factor, c_factor = factored_inv_sqrt_factors(state["row"], state["col"])
            inv_denom = r_factor.mul(c_factor).div_(d).clamp_(max=1.0 / (d * eps))
            inv_denom = self._unmatrixize(inv_denom, grad_fp32, group)
            delta = numer.mul_(inv_denom)

        delta.mul_(dlr)

        if decay != 0 and decouple:
            p_fp32 = p.detach().float()
            delta = delta.add_(p_fp32, alpha=decay * dlr)

        if cautious:
            delta = cautious_one_(delta, grad_fp32)

        subtract_one_(p, delta, state, bf16_method)

    # -- foreach update (Adakaon bucketing) ------------------------------

    @torch.no_grad()
    def _update_foreach(
        self, params: list[Tensor], group: dict[str, Any], d: float, dlr: float, budget: int
    ) -> None:
        """Batched pass-2 update: bucket params by effective shape and step each
        bucket with a handful of stacked kernels (Adakaon's foreach engine),
        applying Prodigy's d-scaled update math."""
        factored_buckets: dict[tuple[Any, ...], list[Tensor]] = {}
        flat_buckets: dict[tuple[Any, ...], list[Tensor]] = {}
        full_buckets: dict[tuple[Any, ...], list[Tensor]] = {}
        for p in params:
            state = self.state[p]
            g = p.grad
            if "v" in state:
                # Full second moment: bucket by exact shape; update is plain
                # grad/sqrt(v) with no factoring.
                full_buckets.setdefault((tuple(g.shape), p.dtype), []).append(p)
            elif g.ndim >= 2:
                matrixize = g.ndim > 2 and group["factor_conv_as_matrix"]
                eff = (g.shape[0], g.numel() // g.shape[0]) if matrixize else tuple(g.shape)
                factored_buckets.setdefault((eff, p.dtype, matrixize), []).append(p)
            else:
                flat_buckets.setdefault((g.shape[0], p.dtype), []).append(p)

        for (eff, _dt, matrixize), plist in factored_buckets.items():
            step = max(1, budget // max(eff[0] * eff[1], 1))
            for i in range(0, len(plist), step):
                self._factored_bucket(plist[i:i + step], eff, matrixize, group, d, dlr)
        for (shape, _dt), plist in full_buckets.items():
            per = 1
            for s in shape:
                per *= s
            step = max(1, budget // max(per, 1))
            for i in range(0, len(plist), step):
                self._full_bucket(plist[i:i + step], shape, group, d, dlr)
        for (length, _dt), plist in flat_buckets.items():
            step = max(1, budget // max(length, 1))
            for i in range(0, len(plist), step):
                self._flat_full_bucket(plist[i:i + step], length, group, d, dlr)

    def _numer_stacked(
        self, plist: list[Tensor], group: dict[str, Any], mat: Any, eff: tuple[int, ...]
    ) -> Tensor:
        """Stacked numerator ``[N, *eff]``: d-scaled momentum (beta1>0) or grad."""
        if group["betas"][0] > 0:
            states = [self.state[p] for p in plist]
            return self._codec(group).dequant_stacked(states, mat, eff)
        return torch.stack([mat(p.grad).float() for p in plist])

    @torch.no_grad()
    def _factored_bucket(
        self, plist: list[Tensor], eff: tuple[int, int], matrixize: bool,
        group: dict[str, Any], d: float, dlr: float,
    ) -> None:
        R, C = eff  # noqa: N806
        eps = group["eps"]
        decay = group["weight_decay"]
        decouple = group["decouple"]
        cautious = group["cautious"]
        bf16_method = group["bf16_method"]

        def mat(t: Tensor) -> Tensor:
            return t.view(R, C) if matrixize else t

        grad = torch.stack([mat(p.grad).float() for p in plist])              # [N, R, C]
        rows = torch.stack([self.state[p]["row"] for p in plist])            # [N, R]
        cols = torch.stack([self.state[p]["col"] for p in plist])            # [N, C]

        r_factor = rows.div(rows.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)  # [N, R, 1]
        c_factor = cols.rsqrt().unsqueeze(-2)                                        # [N, 1, C]
        inv_denom = (r_factor * c_factor).div_(d).clamp_(max=1.0 / (d * eps))        # [N, R, C]

        numer = self._numer_stacked(plist, group, mat, (R, C))               # [N, R, C]
        delta = numer.mul_(inv_denom).mul_(dlr)

        if decay != 0 and decouple:
            p_fp32 = torch.stack([mat(p.data).float() for p in plist])
            delta = delta.add_(p_fp32, alpha=decay * dlr)

        if cautious:
            delta = cautious_batched_(delta, grad)

        subtract_batched_([mat(p.data) for p in plist], delta, bf16_method)

    @torch.no_grad()
    def _full_bucket(
        self, plist: list[Tensor], shape: tuple[int, ...], group: dict[str, Any], d: float, dlr: float
    ) -> None:
        """Batched update for params using the FULL second moment (any shape)."""
        eps = group["eps"]
        decay = group["weight_decay"]
        decouple = group["decouple"]
        cautious = group["cautious"]
        bf16_method = group["bf16_method"]

        grad = torch.stack([p.grad.float() for p in plist])                  # [N, *shape]
        v = torch.stack([self.state[p]["v"] for p in plist])
        denom = v.sqrt().clamp_(min=d * eps)

        numer = self._numer_stacked(plist, group, lambda t: t, tuple(shape))
        delta = numer.div_(denom).mul_(dlr)

        if decay != 0 and decouple:
            p_fp32 = torch.stack([p.data.float() for p in plist])
            delta = delta.add_(p_fp32, alpha=decay * dlr)

        if cautious:
            delta = cautious_batched_(delta, grad)

        subtract_batched_([p.data for p in plist], delta, bf16_method)

    @torch.no_grad()
    def _flat_full_bucket(
        self, plist: list[Tensor], length: int, group: dict[str, Any], d: float, dlr: float
    ) -> None:
        """Batched update for 1-D params under factored second_moment (they use the
        full ``v`` fallback). Same math as :meth:`_full_bucket` for a [N, L] stack."""
        self._full_bucket(plist, (length,), group, d, dlr)

    # -- weight update -----------------------------------------------------


