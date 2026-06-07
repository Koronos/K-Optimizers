"""Muon — orthogonalized-momentum optimizer with an AdamW fallback.

Muon (Jordan et al.; scaled for production by MoonshotAI's *Muon is Scalable for
LLM Training*, arXiv:2502.16982) takes the SGD-momentum update for a 2-D weight
and **orthogonalizes** it with a Newton-Schulz iteration before applying it. That
replaces Adam's per-coordinate second moment with a much cheaper structural
normalization, and unlike Lion's *sign* it preserves the update's geometry — so
it keeps fine detail that sign-based optimizers lose.

State cost is **one momentum buffer** (no second moment): roughly half of AdamW,
though more than a factored optimizer like :class:`Compactor` in its no-momentum
mode. Muon's pitch is *quality at momentum-only memory*, not absolute minimum
memory.

This implementation is a **minimal-config hybrid**: it auto-routes each parameter
by rank — Muon for ``ndim >= 2`` (matrices, plus flattened 4-D conv kernels),
plain AdamW for ``ndim < 2`` (biases, norm scales). So a caller only has to set
``lr`` (and optionally ``adamw_lr`` for the tiny 1-D group). It is a standard
``torch.optim.Optimizer`` and works one-parameter-at-a-time, so it drops into
per-parameter / gradient-release training loops unchanged.

References:
    * https://kellerjordan.github.io/posts/muon/
    * MoonshotAI/Moonlight (`adjusted_lr = lr * 0.2 * sqrt(max(A, B))`, decoupled
      weight decay) — the scaling that lets Muon run without per-layer tuning.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from kaon._stochastic_rounding import add_stochastic_

__all__ = ["Muon"]

_LOW_PRECISION = (torch.bfloat16, torch.float16)


def zeropower_via_newtonschulz5(grad: Tensor, steps: int) -> Tensor:
    """Newton-Schulz quintic iteration: approximate the orthogonal factor of ``grad``.

    Returns ``U`` (≈ ``U @ V.T`` of ``grad = U S V.T``) in bf16. Runs in bf16 for
    speed/memory — the iteration is robust to it. ``grad`` must be 2-D.
    """
    assert grad.ndim == 2, "Newton-Schulz expects a 2-D matrix"
    a, b, c = 3.4445, -4.7750, 2.0315
    x = grad.bfloat16()
    transposed = x.size(0) > x.size(1)
    if transposed:  # iterate on the smaller inner dimension
        x = x.mT
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        aa = x @ x.mT
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x
    if transposed:
        x = x.mT
    return x


class Muon(Optimizer):
    """Orthogonalized-momentum optimizer (2-D weights) with AdamW fallback (1-D).

    Args:
        params: parameters or param-group dicts.
        lr: Muon learning rate (for ``ndim >= 2`` weights). Muon LRs are larger
            than Adam's; ``~2e-2`` is a typical starting point.
        momentum: SGD-momentum coefficient for the Muon buffer.
        nesterov: use Nesterov momentum (recommended).
        ns_steps: Newton-Schulz iteration steps.
        weight_decay: decoupled weight decay for the Muon group.
        adamw_lr: learning rate for the 1-D AdamW fallback group (biases / norms).
            Kept separate because those want an Adam-scale LR, not the Muon LR.
        adamw_betas, adamw_eps, adamw_weight_decay: standard AdamW knobs for the
            fallback group.
        bf16_method: how to apply the update to low-precision (bf16) weights:
            ``"stochastic_rounding"`` (default; no extra state, recovers sub-ULP
            updates) or ``"none"`` (plain bf16 add). fp32 weights ignore this.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 2e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        *,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
        adamw_weight_decay: float = 0.0,
        bf16_method: str = "stochastic_rounding",
        momentum_dtype: str = "float32",
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be >= 1, got {ns_steps}")
        if bf16_method not in ("stochastic_rounding", "none"):
            raise ValueError(f"bf16_method must be 'stochastic_rounding' or 'none', got {bf16_method!r}")
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "weight_decay": weight_decay,
            "adamw_lr": adamw_lr,
            "adamw_betas": adamw_betas,
            "adamw_eps": adamw_eps,
            "adamw_weight_decay": adamw_weight_decay,
            "bf16_method": bf16_method,
            "momentum_dtype": torch.bfloat16 if momentum_dtype == "bfloat16" else torch.float32,
        }
        super().__init__(params, defaults)

    @staticmethod
    def _apply_update(p: Tensor, update_fp32: Tensor, lr: float, bf16_method: str) -> None:
        """``p -= lr * update_fp32`` honouring the bf16 strategy."""
        if p.dtype in _LOW_PRECISION and bf16_method == "stochastic_rounding" and p.dtype == torch.bfloat16:
            add_stochastic_(p.data, update_fp32, alpha=-lr)
        else:
            p.data.add_(update_fp32.to(p.dtype), alpha=-lr)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                if p.ndim >= 2:
                    self._muon_step(p, group)
                else:
                    self._adamw_step(p, group)
        return loss

    @torch.no_grad()
    def _muon_step(self, p: Tensor, group: dict[str, Any]) -> None:
        grad = p.grad
        state = self.state[p]
        if "momentum_buffer" not in state:
            # bf16 momentum keeps Muon at ~2 B/param (Newton-Schulz runs in bf16
            # anyway); fp32 is the safe default for long runs.
            state["momentum_buffer"] = torch.zeros_like(p, dtype=group["momentum_dtype"])
        buf = state["momentum_buffer"]
        momentum = group["momentum"]
        buf.mul_(momentum).add_(grad.to(buf.dtype))
        g = grad.to(buf.dtype).add(buf, alpha=momentum) if group["nesterov"] else buf

        mat = g if g.ndim == 2 else g.reshape(g.shape[0], -1)
        u = zeropower_via_newtonschulz5(mat, group["ns_steps"]).to(torch.float32)
        if g.ndim != 2:
            u = u.view_as(g)

        rows, cols = mat.shape
        adjusted_lr = group["lr"] * 0.2 * math.sqrt(max(rows, cols))

        wd = group["weight_decay"]
        if wd > 0:
            p.data.mul_(1.0 - group["lr"] * wd)
        self._apply_update(p, u, adjusted_lr, group["bf16_method"])

    @torch.no_grad()
    def _adamw_step(self, p: Tensor, group: dict[str, Any]) -> None:
        grad = p.grad.float()
        state = self.state[p]
        if "step" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(grad)
            state["exp_avg_sq"] = torch.zeros_like(grad)
        state["step"] += 1
        beta1, beta2 = group["adamw_betas"]
        eps = group["adamw_eps"]
        lr = group["adamw_lr"]
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        bias_c1 = 1.0 - beta1 ** state["step"]
        bias_c2 = 1.0 - beta2 ** state["step"]
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_c2)).add_(eps)
        update = exp_avg / denom  # fp32 step direction
        wd = group["adamw_weight_decay"]
        if wd > 0:
            p.data.mul_(1.0 - lr * wd)
        self._apply_update(p, update, lr / bias_c1, group["bf16_method"])
