"""Scalar model used by the experimental MoMo-Adakaon path.

This module deliberately contains no optimizer plumbing.  It implements the
global scalar recurrence from MoMo so that the recurrence can be tested against
the reference implementation before it is connected to Adakaon's tensor paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class MomoStep:
    """Diagnostics returned for one model update."""

    tau: float
    tau_model: float
    rho: float
    numerator: float
    denominator: float
    lower_bound: float
    cap_active: bool


class MomoModel:
    """Global MoMo recurrence for a single homogeneous parameter group.

    ``grad_dot_param`` is ``<g_k, x_k>``, ``avg_grad_dot_param`` is
    ``<d_k, x_k>``, and ``denominator`` is ``<d_k, D_k^-1 d_k>`` after any
    scalar RMS clipping applied to the direction.  The caller owns all tensor
    state and must provide values from the same loss/gradient sample.

    ``cap`` remains explicit here because the experiment must establish whether
    a wide, model-independent cap exists.  It is not a proposed public option.
    """

    _VERSION = 1

    def __init__(
        self,
        *,
        beta: float,
        cap: float,
        lower_bound: float = 0.0,
        estimate_lower_bound: bool = True,
    ) -> None:
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        if math.isnan(cap) or not cap > 0.0:
            raise ValueError(f"cap must be > 0, got {cap}")
        if not math.isfinite(lower_bound):
            raise ValueError("lower_bound must be finite")
        self.beta = float(beta)
        self.cap = float(cap)
        self.initial_lower_bound = float(lower_bound)
        self.lower_bound = float(lower_bound)
        self.estimate_lower_bound = bool(estimate_lower_bound)
        self.step_count = 0
        self.loss_average = 0.0
        self.gamma = 0.0
        self.last_tau = 0.0

    def update(
        self,
        *,
        loss: float,
        grad_dot_param: float,
        avg_grad_dot_param: float,
        denominator: float,
        weight_decay: float = 0.0,
    ) -> MomoStep:
        """Consume one coherent loss/gradient model and return its step scale."""
        values = (loss, grad_dot_param, avg_grad_dot_param, denominator, weight_decay)
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("MoMo inputs must all be finite")
        if denominator < 0.0:
            raise ValueError(f"MoMo denominator must be non-negative, got {denominator}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")

        self.step_count += 1
        beta = self.beta
        self.loss_average = beta * self.loss_average + (1.0 - beta) * loss
        self.gamma = beta * self.gamma + (1.0 - beta) * grad_dot_param
        rho = 1.0 - beta**self.step_count
        if math.isinf(self.cap) and weight_decay > 0.0:
            raise ValueError("an uncapped MoMo model cannot define capped weight decay")
        regularizer = 1.0 if weight_decay == 0.0 else 1.0 + self.cap * weight_decay

        if self.estimate_lower_bound:
            model_cap = (
                regularizer * self.loss_average
                + avg_grad_dot_param
                - regularizer * self.gamma
            )
            bound_cap = regularizer * rho * self.lower_bound
            if model_cap < bound_cap:
                candidate = model_cap / (2.0 * regularizer * rho)
                self.lower_bound = max(self.initial_lower_bound, candidate)

        numerator = (
            regularizer * (self.loss_average - rho * self.lower_bound)
            + avg_grad_dot_param
            - regularizer * self.gamma
        )
        tau_model = (
            0.0 if denominator == 0.0 or numerator <= 0.0 else numerator / denominator
        )
        tau_cap = self.cap / rho
        tau = min(tau_cap, tau_model)

        if self.estimate_lower_bound:
            h = self.loss_average + avg_grad_dot_param - self.gamma
            candidate = (h - 0.5 * tau * denominator) / rho
            self.lower_bound = max(self.initial_lower_bound, candidate)

        self.last_tau = tau
        return MomoStep(
            tau=tau,
            tau_model=tau_model,
            rho=rho,
            numerator=numerator,
            denominator=denominator,
            lower_bound=self.lower_bound,
            cap_active=tau_model > tau_cap,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self._VERSION,
            "beta": self.beta,
            "cap": self.cap,
            "initial_lower_bound": self.initial_lower_bound,
            "lower_bound": self.lower_bound,
            "estimate_lower_bound": self.estimate_lower_bound,
            "step_count": self.step_count,
            "loss_average": self.loss_average,
            "gamma": self.gamma,
            "last_tau": self.last_tau,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != self._VERSION:
            raise ValueError(f"unsupported MoMo model state version: {state.get('version')!r}")
        if float(state["beta"]) != self.beta or float(state["cap"]) != self.cap:
            raise ValueError("MoMo checkpoint beta/cap does not match the configured model")
        self.initial_lower_bound = float(state["initial_lower_bound"])
        self.lower_bound = float(state["lower_bound"])
        self.estimate_lower_bound = bool(state["estimate_lower_bound"])
        self.step_count = int(state["step_count"])
        self.loss_average = float(state["loss_average"])
        self.gamma = float(state["gamma"])
        self.last_tau = float(state["last_tau"])
