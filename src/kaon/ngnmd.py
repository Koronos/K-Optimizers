"""Reference NGN-MDv1 implementation for Kaon experiments.

This is the loss-aware, diagonal momentum variant from Islamov et al., NeurIPS
2025.  ``lr`` is the paper's step-size hyperparameter ``c``.  The optimizer
requires the non-negative stochastic loss from the same mini-batch whose
gradients are currently stored in ``param.grad``::

    loss.backward()
    optimizer.step(loss=loss.detach())

The initial implementation intentionally keeps fp32 diagonal second moment and
heavy-ball velocity buffers.  It is a mathematical reference used to decide
whether NGN-MD is worth porting onto Kaon's factored/quantized backend.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from kaon._backend import is_low_precision
from kaon._stochastic_rounding import add_stochastic_

__all__ = ["NGNMD"]


class NGNMD(Optimizer):
    """NGN-MDv1 with heavy-ball momentum and RMSprop diagonal preconditioning."""

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        beta1, beta2 = (float(value) for value in betas)
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError(f"lr (NGN c) must be finite and > 0, got {lr}")
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(f"eps must be finite and > 0, got {eps}")
        if not math.isfinite(weight_decay) or weight_decay < 0.0:
            raise ValueError(f"weight_decay must be finite and >= 0, got {weight_decay}")
        super().__init__(params, {
            "lr": float(lr), "betas": (beta1, beta2), "eps": float(eps),
            "weight_decay": float(weight_decay),
        })
        self._last_effective_lrs = [float(lr) for _ in self.param_groups]

    @torch.no_grad()
    def step(self, closure: Any = None, *, loss: Tensor | float | None = None) -> Any:
        closure_loss = None
        if closure is not None:
            with torch.enable_grad():
                closure_loss = closure()
            if loss is None:
                loss = closure_loss
        if loss is None:
            raise ValueError("NGNMD.step requires the non-negative mini-batch loss")
        loss_value = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        if not math.isfinite(loss_value) or loss_value < 0.0:
            raise ValueError(f"NGNMD requires a finite non-negative loss, got {loss_value}")

        for group_index, group in enumerate(self.param_groups):
            beta1, beta2 = group["betas"]
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            metric_norm: Tensor | None = None
            for param in params:
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("NGNMD does not support sparse gradients")
                grad_f = grad.detach().float()
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["v"] = torch.zeros_like(grad_f)
                    state["velocity"] = torch.zeros_like(grad_f)
                state["step"] += 1
                state["v"].mul_(beta2).addcmul_(grad_f, grad_f, value=1.0 - beta2)
                bias = 1.0 - beta2 ** state["step"]
                diagonal = state["v"].div(bias).sqrt_().add_(group["eps"])
                contribution = grad_f.square().div_(diagonal).sum()
                metric_norm = contribution if metric_norm is None else metric_norm + contribution

            norm_value = float(metric_norm)
            c = group["lr"]
            # Paper Eq. (3), written without division by a possibly tiny loss:
            #   gamma = c / (1 + c * ||g||^2_D^-1 / (2 f))
            #         = 2 c f / (2 f + c * ||g||^2_D^-1).
            denominator = 2.0 * loss_value + c * norm_value
            gamma = (
                c
                if denominator == 0.0
                else (2.0 * c * loss_value / denominator if math.isfinite(denominator) else 0.0)
            )
            self._last_effective_lrs[group_index] = gamma

            for param in params:
                grad_f = param.grad.detach().float()
                state = self.state[param]
                bias = 1.0 - beta2 ** state["step"]
                direction = grad_f.div(state["v"].div(bias).sqrt_().add_(group["eps"]))
                velocity = state["velocity"]
                velocity.mul_(beta1).add_(direction, alpha=(1.0 - beta1) * gamma)
                if group["weight_decay"]:
                    param.mul_(1.0 - c * group["weight_decay"])
                if is_low_precision(param):
                    add_stochastic_(param, velocity, alpha=-1.0)
                else:
                    param.sub_(velocity)
        return closure_loss

    def get_effective_lr(self, group: int = 0) -> float:
        return self._last_effective_lrs[group]
