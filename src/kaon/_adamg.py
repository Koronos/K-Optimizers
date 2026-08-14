"""Unexported AdamG prototype for zero-configuration evaluation.

The implementation follows Algorithm 2 of "Towards Reliability of
Parameter-free Optimization".  ``cap`` remains a diagnostic constructor
argument so the benchmark can determine whether the paper's default of one is
actually insensitive across problem families.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor


class AdamG(torch.optim.Optimizer):
    """Research-only AdamG implementation; intentionally not package-exported."""

    def __init__(
        self,
        params: Iterable[Any],
        *,
        cap: float = 1.0,
        betas: tuple[float, float, float] = (0.95, 0.999, 0.95),
        eps: float = 1e-8,
    ) -> None:
        if math.isnan(cap) or not cap > 0.0:
            raise ValueError(f"cap must be > 0, got {cap}")
        if any(not 0.0 <= beta < 1.0 for beta in betas):
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        if eps < 0.0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        super().__init__(params, {"cap": cap, "betas": betas, "eps": eps, "step": 0})

    @staticmethod
    def _golden_numerator(value: Tensor) -> Tensor:
        return value.pow(0.24).mul_(0.2)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            group["step"] += 1
            step = int(group["step"])
            beta1, beta2, beta3 = group["betas"]
            correction1 = 1.0 - beta1**step
            correction2 = 1.0 - beta2**step
            step_size = min(float(group["cap"]), 1.0 / math.sqrt(step))
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("AdamG does not support sparse gradients")
                if not bool(torch.isfinite(grad).all()):
                    raise FloatingPointError("AdamG gradients must be finite")
                state = self.state[param]
                if not state:
                    state["m"] = torch.zeros_like(param)
                    state["v"] = torch.zeros_like(param)
                    state["r"] = torch.zeros_like(param)
                m, v, r = state["m"], state["v"], state["r"]
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                r.mul_(beta3).add_(self._golden_numerator(v), alpha=1.0 - beta3)
                m.mul_(beta1).addcmul_(r, grad, value=1.0 - beta1)
                denominator = (v / correction2).sqrt_().add_(group["eps"])
                update = (m / correction1) / denominator
                param.add_(update, alpha=-step_size)
        return loss

    def get_d(self) -> float:
        if not self.param_groups:
            return 0.0
        group = self.param_groups[0]
        step = int(group["step"])
        return min(float(group["cap"]), 1.0 / math.sqrt(max(step, 1)))
