"""KProdigy — a memory-efficient Prodigy (parameter-free D-adaptation) optimizer.

Prodigy (Mishchenko & Defazio, 2023, arXiv:2306.06101) estimates the distance
``D`` to the solution on the fly and uses it as the effective learning rate, so
you train at ``lr=1.0`` and the optimizer finds the scale itself.

This is a clean reimplementation aimed at koptim's thesis — *memory-efficient
bf16 diffusion fine-tuning* — fixing the issues that plagued the original
``KProdigy`` research repo (whose shipped defaults, ``d_update_freq=5`` and
``use_bias_correction=True``, starved the D-bootstrap and made the effective LR
fail to rise). The D-estimation math here matches the reference Prodigy bit for
bit at the defaults; the *enhancements* are orthogonal memory savings:

* **bf16 / int8 first moment** (``momentum_dtype``) — like ``Adafusion``.
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

Memory at ``beta1=0`` (no momentum), ``second_moment="factored"``, ``slice_p=11``
is well under AdamW; even the full-precision default (bf16 momentum + full fp32
second moment) is ~6 B/param vs AdamW's 8.

Unlike ``Adafusion``/``Muon``, Prodigy needs a **global reduction over all
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

from koptim._factored import factored_inv_sqrt_factors, update_factored_state
from koptim._stochastic_rounding import add_stochastic_

__all__ = ["KProdigy"]

_LOW_PRECISION = (torch.bfloat16, torch.float16)
MomentumDtype = Literal["bfloat16", "float32", "int8"]
SecondMoment = Literal["full", "factored"]


def _is_low_precision(t: Tensor) -> bool:
    return t.dtype in _LOW_PRECISION


def _quant_int8(m_fp32: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize a momentum tensor to int8 with a per-row (dim-0) absmax scale."""
    dims = tuple(range(1, m_fp32.ndim)) if m_fp32.ndim >= 2 else ()
    absmax = m_fp32.abs().amax(dim=dims, keepdim=True).clamp_(min=1e-12)
    scale = absmax / 127.0
    q = (m_fp32 / scale).round_().clamp_(-127, 127).to(torch.int8)
    return q, scale


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
            B/param), ``"float32"`` (4 B/param), or ``"int8"`` (~1 B/param).
        second_moment: ``"full"`` (default; fp32, exact) or ``"factored"``
            (Adafactor row+col, ~0 state on >=2-D weights; experimental).
        eps_factored: ``eps1`` added to ``grad**2`` before the factored
            reductions (HF Adafactor convention). Only used when factored.
        bf16_method: weight-update strategy for low-precision params —
            ``"stochastic_rounding"`` (default), ``"kahan"`` (+2 B/param), or
            ``"none"``. No-op on fp32 params.
        factor_conv_as_matrix: reshape 4-D conv kernels to 2-D before factoring
            (the conv-aware fix). Default ``True``.
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
        second_moment: SecondMoment = "full",
        eps_factored: float = 1e-30,
        bf16_method: str = "stochastic_rounding",
        factor_conv_as_matrix: bool = True,
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
        if momentum_dtype not in ("bfloat16", "float32", "int8"):
            raise ValueError(f"momentum_dtype must be bfloat16/float32/int8, got {momentum_dtype!r}")
        if second_moment not in ("full", "factored"):
            raise ValueError(f"second_moment must be full/factored, got {second_moment!r}")
        if bf16_method not in ("stochastic_rounding", "kahan", "none"):
            raise ValueError(f"bf16_method must be stochastic_rounding/kahan/none, got {bf16_method!r}")

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
            "second_moment": second_moment,
            "eps_factored": eps_factored,
            "bf16_method": bf16_method,
            "factor_conv_as_matrix": factor_conv_as_matrix,
        }
        self.d0 = d0
        super().__init__(params, defaults)
        # Auto: independent D when the user gave more than one param group
        # (e.g. SDXL UNet + Text Encoder), unless explicitly overridden.
        self._independent_d = (len(self.param_groups) > 1) if independent_d is None else independent_d

    def get_d(self) -> float:
        """Current D estimate (effective learning rate) of the first group."""
        return float(self.param_groups[0].get("d", self.d0))

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
            md = group["momentum_dtype"]
            if md == "int8":
                state["m"] = torch.zeros_like(p, dtype=torch.int8)
                state["m_scale"] = torch.ones(
                    (p.shape[0],) + (1,) * (p.ndim - 1) if p.ndim >= 2 else (),
                    dtype=torch.float32, device=p.device,
                )
            else:
                state["m"] = torch.zeros_like(p, dtype=torch.bfloat16 if md == "bfloat16" else torch.float32)

        if group["second_moment"] == "factored" and p.ndim >= 2:
            gv = p if (p.ndim == 2 or not group["factor_conv_as_matrix"]) else p.reshape(p.shape[0], -1)
            state["row"] = torch.zeros(gv.shape[:-1], dtype=torch.float32, device=p.device)
            state["col"] = torch.zeros(gv.shape[:-2] + gv.shape[-1:], dtype=torch.float32, device=p.device)
        else:
            # Full second moment (also the fallback for 1-D params under factored).
            state["v"] = torch.zeros_like(p, dtype=torch.float32)

        if _is_low_precision(p) and group["bf16_method"] == "kahan":
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

    @torch.no_grad()
    def _step_scope(self, groups: list[dict[str, Any]]) -> None:
        lead = groups[0]
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
        d_numerator = lead["d_numerator"] * beta3
        delta_numerator = torch.zeros((), dtype=torch.float32)
        d_denom = torch.zeros((), dtype=torch.float32)
        device_seen = None

        for group in groups:
            decouple = group["decouple"]
            decay = group["weight_decay"]
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

                grad = p.grad
                grad_fp32 = grad if grad.dtype == torch.float32 else grad.float()
                if decay != 0 and not decouple:
                    grad_fp32 = grad_fp32.add(p.detach().float(), alpha=decay)

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
                    self._update_momentum(state, grad_fp32, beta1, d, group["momentum_dtype"])

                # Second moment EMA.
                if "v" in state:
                    state["v"].mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=d * d * (1 - beta2))
                else:
                    gv = self._matrixize(grad_fp32, group)
                    update_factored_state(gv, state["row"], state["col"], beta2, group["eps_factored"])

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

        # ---- pass 2: apply updates ------------------------------------------
        for group in groups:
            eps = group["eps"]
            decay = group["weight_decay"]
            decouple = group["decouple"]
            bf16_method = group["bf16_method"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                state["step"] += 1

                grad = p.grad
                grad_fp32 = grad if grad.dtype == torch.float32 else grad.float()

                # denom = sqrt(second moment) floored at d*eps. Both the full and
                # factored second moments reconstruct an O(d) denominator, so the
                # O(d) numerator (d-scaled momentum / d-scaled grad) cancels.
                # numerator: d-scaled momentum (beta1>0) or the raw grad
                # (beta1=0, matching reference Prodigy — there the d cancels in
                # the Adam ratio, so beta1=0 is RMSprop-like at lr=1).
                if group["betas"][0] > 0:
                    numer = self._momentum_value(state, grad_fp32, group, d)
                else:
                    numer = grad_fp32.clone()

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
                    delta = delta.add_(p.detach().float(), alpha=decay * dlr)

                self._apply_subtract(p, delta, state, bf16_method)
            group["k"] = group["k"] + 1

    # -- momentum helpers --------------------------------------------------

    @staticmethod
    def _update_momentum(state: dict[str, Any], grad_fp32: Tensor, beta1: float, d: float, md: str) -> None:
        """EMA ``m <- beta1*m + (1-beta1)*d*grad`` in the momentum dtype."""
        target = d * (1 - beta1)
        if md == "int8":
            m = state["m"].float().mul_(state["m_scale"])
            m.mul_(beta1).add_(grad_fp32, alpha=target)
            state["m"], state["m_scale"] = _quant_int8(m)
        else:
            m = state["m"]
            m.mul_(beta1).add_(grad_fp32.to(m.dtype), alpha=target)

    @staticmethod
    def _momentum_value(state: dict[str, Any], grad_fp32: Tensor, group: dict[str, Any], d: float) -> Tensor:
        """Return the fp32 first-moment numerator for the update."""
        md = group["momentum_dtype"]
        if md == "int8":
            return state["m"].float().mul_(state["m_scale"])
        # Must be a fresh fp32 tensor: the caller mutates it in place, and
        # ``.float()`` is a no-op (no copy) when the buffer is already fp32.
        m = state["m"]
        return m.float() if m.dtype != torch.float32 else m.clone()

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

    # -- weight update -----------------------------------------------------

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
