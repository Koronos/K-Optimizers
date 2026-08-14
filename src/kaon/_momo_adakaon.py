"""Experimental, unexported MoMo-Adakaon integration.

The class in this module exists to validate the algorithm before replacing the
public ``auto_lr`` path.  It intentionally supports only the mathematically
auditable native FP32/single-group configuration.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor

from kaon._momo_model import MomoModel, MomoStep
from kaon.adakaon import (
    Adakaon,
    factored_inv_sqrt_factors,
    rms,
    subtract_one_,
    update_factored_state,
)


class MomoAdakaon(Adakaon):
    """Research-only MoMo controller coupled to Adakaon's preconditioner.

    ``cap`` is deliberately required: the benchmark must determine whether a
    broad universal region exists.  This class is not exported from ``kaon`` and
    is not a proposed user API.
    """

    def __init__(
        self,
        params: Iterable[Any],
        *,
        cap: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        estimate_lower_bound: bool = True,
    ) -> None:
        super().__init__(
            params,
            lr=1.0,
            betas=betas,
            eps=eps,
            weight_decay=0.0,
            clip_threshold=clip_threshold,
            momentum_dtype="float32",
            cautious=False,
            gradient_centralization=False,
            bf16_method="none",
            foreach=False,
            fused=False,
            auto_lr=False,
        )
        if len(self.param_groups) != 1:
            raise ValueError("experimental MomoAdakaon supports exactly one parameter group")
        if any(
            param.dtype != torch.float32
            for group in self.param_groups
            for param in group["params"]
        ):
            raise ValueError("experimental MomoAdakaon currently supports FP32 parameters only")
        self._momo_model = MomoModel(
            beta=float(betas[0]),
            cap=cap,
            lower_bound=0.0,
            estimate_lower_bound=estimate_lower_bound,
        )
        self._momo_last: MomoStep | None = None

    @torch.no_grad()
    def step(self, closure: Any = None, *, loss: Tensor | None = None) -> Any:
        if (closure is None) == (loss is None):
            raise ValueError("pass exactly one of loss or closure to MomoAdakaon.step")
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not isinstance(loss, Tensor) or loss.numel() != 1:
            raise TypeError("loss must be a scalar tensor from the same batch as the gradients")
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise FloatingPointError("loss must be finite")

        group = self.param_groups[0]
        beta1, beta2 = group["betas"]
        eps1, _ = group["eps"]
        clip = group["clip_threshold"]
        active = [param for param in group["params"] if param.grad is not None]
        if not active:
            self._momo_last = self._momo_model.update(
                loss=loss_value,
                grad_dot_param=0.0,
                avg_grad_dot_param=0.0,
                denominator=0.0,
            )
            return loss

        directions: list[tuple[Tensor, Tensor]] = []
        grad_dot_param = 0.0
        avg_grad_dot_param = 0.0
        denominator = 0.0

        for param in active:
            grad = param.grad
            if grad.is_sparse:
                raise RuntimeError("MomoAdakaon does not support sparse gradients")
            if not bool(torch.isfinite(grad).all()):
                raise FloatingPointError("gradients must be finite")
            state = self.state[param]
            if not state:
                self._init_state(param, state, group)

            grad_fp32 = grad if grad.dtype == torch.float32 else grad.float()
            if grad_fp32.ndim >= 2:
                matrixized = grad_fp32.reshape(grad_fp32.shape[0], -1)
                update_factored_state(matrixized, state["row"], state["col"], beta2, eps1)
            else:
                squared = grad_fp32 * grad_fp32
                if eps1 > 0.0:
                    squared.add_(eps1)
                state["v"].lerp_(squared, 1.0 - beta2)

            if beta1 > 0.0:
                avg_grad = self._codec(group).ema_one(state, grad_fp32, beta1)
            else:
                avg_grad = grad_fp32

            if grad_fp32.ndim >= 2:
                avg_matrixized = avg_grad.reshape(avg_grad.shape[0], -1)
                row_factor, col_factor = factored_inv_sqrt_factors(
                    state["row"], state["col"]
                )
                direction = avg_matrixized.mul(row_factor).mul_(col_factor).view_as(avg_grad)
            else:
                direction = avg_grad.mul(state["v"].rsqrt())
            direction.div_((rms(direction) / clip).clamp_(min=1.0))

            param_fp32 = param.detach()
            grad_dot_param += float(torch.sum(grad_fp32 * param_fp32))
            avg_grad_dot_param += float(torch.sum(avg_grad * param_fp32))
            denominator += float(torch.sum(avg_grad * direction))
            directions.append((param, direction))

        if denominator < 0.0 and denominator > -1e-10:
            denominator = 0.0
        self._momo_last = self._momo_model.update(
            loss=loss_value,
            grad_dot_param=grad_dot_param,
            avg_grad_dot_param=avg_grad_dot_param,
            denominator=denominator,
        )
        tau = self._momo_last.tau
        for param, direction in directions:
            state = self.state[param]
            subtract_one_(param, direction.mul_(tau), state, group["bf16_method"])
        self._t += 1
        return loss

    def get_d(self) -> float:
        return self._momo_model.last_tau

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["_momo_model"] = self._momo_model.state_dict()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        copied = dict(state_dict)
        model_state = copied.pop("_momo_model", None)
        if model_state is None:
            raise ValueError("MomoAdakaon checkpoint is missing _momo_model")
        super().load_state_dict(copied)
        self._momo_model.load_state_dict(model_state)
        self._momo_last = None
