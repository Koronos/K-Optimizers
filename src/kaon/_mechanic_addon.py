"""Private research wrapper for the Mechanic learning-rate tuner.

This module is deliberately not exported.  It is an A/B prototype, not the implementation
behind ``auto_lr=True``.  The scalar update follows the reference Optax implementation of
Mechanic: six exponentially weighted bettors, an anchored base trajectory, and the
memory-saving reconstruction ``delta_prev = (x0 - x) / (sum(s) + eps)``.

PyTorch optimizers mutate parameters in-place whereas Optax returns their unit updates.  The
wrapper therefore retains a temporary pre-step snapshot while the inner optimizer runs.  Its
persistent overhead is one anchor copy; the temporary snapshot is an explicit prototype
trade-off to keep the wrapper generic.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

_BETAS = tuple(1.0 - 0.1**k for k in range(1, 7))
_EPS = 1e-8
_S_INIT = 1e-6
_STATE_KEY = "_mechanic_addon"


@dataclass(frozen=True)
class MechanicStats:
    """Latest scalar diagnostics consumed by research batteries."""

    scale: float
    h: float
    guarded: bool
    candidate_scale: float
    guard_events: int
    nonfinite_steps: int


class MechanicAddon(torch.optim.Optimizer):
    """Experimental, private Mechanic wrapper around an already-built optimizer.

    ``guard`` is the sole research switch.  With it enabled, negative raw Mechanic feedback
    bisects the previously applied scale.  Since the next favorable observation can grow the
    scale again, the guard is reversible and does not introduce a freeze or horizon.
    """

    def __init__(self, inner: torch.optim.Optimizer, *, guard: bool = False) -> None:
        # Do not call Optimizer.__init__: the inner optimizer owns these groups and its state.
        self.inner = inner
        self.param_groups = inner.param_groups
        self.defaults = inner.defaults
        self.guard = bool(guard)
        self._count = 0
        self._r = [0.0] * len(_BETAS)
        self._m = [0.0] * len(_BETAS)
        self._v = [0.0] * len(_BETAS)
        self._s = [_S_INIT] * len(_BETAS)
        self._x0: list[Tensor] | None = None
        self._last_h = 0.0
        self._last_candidate_scale = self.get_scale()
        self._last_guarded = False
        self._guard_events = 0
        self._nonfinite_steps = 0
        self._log_every = max(0, int(os.environ.get("KAON_MECHANIC_LOG_EVERY", "0")))
        self._force_unit_lr()

    @property
    def state(self) -> dict[Any, Any]:  # type: ignore[override]
        return self.inner.state

    def _params(self) -> list[Tensor]:
        return [p for group in self.param_groups for p in group["params"]]

    def _force_unit_lr(self) -> None:
        for group in self.param_groups:
            group["lr"] = 1.0

    def get_scale(self) -> float:
        return float(sum(self._s))

    @property
    def last_stats(self) -> MechanicStats:
        return MechanicStats(
            scale=self.get_scale(),
            h=self._last_h,
            guarded=self._last_guarded,
            candidate_scale=self._last_candidate_scale,
            guard_events=self._guard_events,
            nonfinite_steps=self._nonfinite_steps,
        )

    def zero_grad(self, set_to_none: bool = True) -> None:  # noqa: FBT001, FBT002
        self.inner.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        self.inner.add_param_group(param_group)
        self.param_groups = self.inner.param_groups
        self._force_unit_lr()
        # A changed parameter topology invalidates the single anchored trajectory.
        self._x0 = None
        self._count = 0

    @staticmethod
    def _all_finite(params: list[Tensor]) -> bool:
        return all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params)

    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None) -> Any:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        params = self._params()
        if not self._all_finite(params):
            self._nonfinite_steps += 1
            return loss

        if self._x0 is None:
            # Match Optax's first-step refresh: construction may precede weight loading.
            self._x0 = [p.detach().clone() for p in params]

        scale_prev = self.get_scale()
        if not math.isfinite(scale_prev) or scale_prev <= 0.0:
            self._nonfinite_steps += 1
            raise FloatingPointError("Mechanic scale is non-finite or non-positive")
        delta_prev: list[Tensor] = []
        h = 0.0
        for p, x0 in zip(params, self._x0, strict=True):
            delta = (x0 - p) / (scale_prev + _EPS)
            delta_prev.append(delta)
            if p.grad is not None:
                h += float(torch.sum(p.grad.detach().float() * delta.float()).item())
        if not math.isfinite(h):
            self._nonfinite_steps += 1
            raise FloatingPointError("Mechanic feedback overflowed before the inner optimizer step")

        # The inner optimizer is the virtual BASE and must emit its unit-LR update.  Capture
        # pre-step values because PyTorch applies that update in-place.
        before = [p.detach().clone() for p in params]
        self._force_unit_lr()
        self.inner.step()

        if not all(bool(torch.isfinite(p).all()) for p in params):
            for p, old in zip(params, before, strict=True):
                p.copy_(old)
            self._nonfinite_steps += 1
            raise FloatingPointError(
                "inner optimizer produced non-finite parameters; parameters were restored, "
                "but its internal state may have advanced and the run must not continue"
            )

        old_m = self._m
        clipped_h = [min(max(h, -m), m) for m in old_m]
        new_m = [max(beta * m, abs(h) + _EPS) for beta, m in zip(_BETAS, old_m, strict=True)]
        new_v = [
            beta * beta * v + h * h
            for beta, v in zip(_BETAS, self._v, strict=True)
        ]
        new_r = [
            beta * r + h_clip * s
            for beta, r, h_clip, s in zip(
                _BETAS, self._r, clipped_h, self._s, strict=True
            )
        ]
        candidate = [
            ((_S_INIT / len(_BETAS)) * m + max(r, 0.0)) / (math.sqrt(v) + _EPS)
            for m, r, v in zip(new_m, new_r, new_v, strict=True)
        ]

        candidate_sum = sum(candidate)
        scalar_state = (*new_m, *new_v, *new_r, *candidate, candidate_sum)
        if not all(math.isfinite(value) for value in scalar_state) or candidate_sum <= 0.0:
            for p, old in zip(params, before, strict=True):
                p.copy_(old)
            self._nonfinite_steps += 1
            raise FloatingPointError(
                "Mechanic scalar state became invalid; parameters were restored, but the "
                "inner optimizer state advanced and the run must not continue"
            )
        applied_sum = candidate_sum
        guarded = False
        if self.guard and h < 0.0:
            applied_sum = min(candidate_sum, 0.5 * scale_prev)
            self._guard_events += 1
            guarded = True
        if applied_sum != candidate_sum:
            if candidate_sum > 0.0:
                ratio = applied_sum / candidate_sum
                candidate = [s * ratio for s in candidate]
            else:
                candidate = [applied_sum / len(candidate)] * len(candidate)

        # Optax: delta = delta_prev - new_neg_updates; x = x0 - sum(s) * delta.
        for p, x0, old, previous_delta in zip(
            params, self._x0, before, delta_prev, strict=True
        ):
            base_update = p - old
            delta = previous_delta - base_update
            p.copy_(x0 - applied_sum * delta)

        if not all(bool(torch.isfinite(p).all()) for p in params):
            for p, old in zip(params, before, strict=True):
                p.copy_(old)
            self._nonfinite_steps += 1
            raise FloatingPointError(
                "Mechanic reconstruction produced non-finite parameters; parameters were "
                "restored, but the inner optimizer state advanced and the run must not continue"
            )

        self._m = new_m
        self._v = new_v
        self._r = new_r
        self._s = candidate
        self._last_h = h
        self._last_candidate_scale = candidate_sum
        self._last_guarded = guarded
        self._count += 1
        if self._log_every and self._count % self._log_every == 0:
            print(
                "[kaon.mechanic] "
                f"step={self._count} scale={applied_sum:.8g} h={h:.8g} "
                f"nonfinite_steps={self._nonfinite_steps}",
                flush=True,
            )
        return loss

    def state_dict(self) -> dict[str, Any]:
        result = deepcopy(self.inner.state_dict())
        result[_STATE_KEY] = {
            "version": 1,
            "guard": self.guard,
            "count": self._count,
            "r": list(self._r),
            "m": list(self._m),
            "v": list(self._v),
            "s": list(self._s),
            "x0": None if self._x0 is None else [x.detach().clone() for x in self._x0],
            "last_h": self._last_h,
            "last_candidate_scale": self._last_candidate_scale,
            "last_guarded": self._last_guarded,
            "guard_events": self._guard_events,
            "nonfinite_steps": self._nonfinite_steps,
        }
        return result

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        copied = deepcopy(state_dict)
        addon = copied.pop(_STATE_KEY)
        if int(addon.get("version", 0)) != 1:
            raise ValueError("unsupported MechanicAddon checkpoint version")
        if bool(addon["guard"]) != self.guard:
            raise ValueError("MechanicAddon guard mode differs from checkpoint")
        self.inner.load_state_dict(copied)
        self.param_groups = self.inner.param_groups
        self._force_unit_lr()
        self._count = int(addon["count"])
        self._r = [float(x) for x in addon["r"]]
        self._m = [float(x) for x in addon["m"]]
        self._v = [float(x) for x in addon["v"]]
        self._s = [float(x) for x in addon["s"]]
        self._last_h = float(addon["last_h"])
        self._last_candidate_scale = float(addon.get("last_candidate_scale", sum(self._s)))
        self._last_guarded = bool(addon.get("last_guarded", False))
        self._guard_events = int(addon["guard_events"])
        self._nonfinite_steps = int(addon["nonfinite_steps"])
        saved_x0 = addon["x0"]
        if saved_x0 is None:
            self._x0 = None
        else:
            params = self._params()
            if len(saved_x0) != len(params):
                raise ValueError("MechanicAddon checkpoint parameter count differs")
            self._x0 = [
                x.to(device=p.device, dtype=p.dtype).clone()
                for x, p in zip(saved_x0, params, strict=True)
            ]
