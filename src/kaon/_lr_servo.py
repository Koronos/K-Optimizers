"""A small, continuous learning-rate *corrector* for momentum optimizers.

Unlike :mod:`kaon._autolr`, this controller does not try to discover a learning
rate from an arbitrary seed.  It treats the user supplied LR as a useful local
prior and follows the hypergradient sign already present in the optimizer:

``<gradient_t, update_direction_(t-1)>``

Positive alignment means the preceding downhill direction is still useful and
permits a slightly larger step.  Negative alignment is the characteristic
overshoot/oscillation signal and reduces the step.  The signal is normalized,
smoothed, and applied in log space.  A fixed 0.25x--4x trust region makes the
contract explicit and prevents a local corrector from becoming an unbounded LR
search again.

The implementation reuses Kaon's existing momentum buffer.  It allocates no
persistent tensor per parameter; only five Python scalars are kept per parameter
group.  Quantized momentum is dequantized transiently through ``CodecBuffer``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from kaon._wrappers import CodecBuffer

__all__ = ["LRServo", "LRServoMixin"]

_MIN_SCALE = 0.25
_MAX_SCALE = 4.0
_EMA_BETA = 0.95
_TARGET_ALIGNMENT = 0.30
_DEAD_ZONE = 0.05
_SAMPLE_EVERY = 4
_GAIN = 0.025 * _SAMPLE_EVERY
_EPS = 1e-30
_STATE_VERSION = 1
_REDUCTION_CHUNK_ELEMENTS = 262_144


class LRServo:
    """Local LR controller attached to an optimizer with Kaon momentum state."""

    def __init__(self, opt: torch.optim.Optimizer) -> None:
        self.opt = opt
        self._groups = [
            {
                "base_lr": float(group["lr"]),
                "log_scale": 0.0,
                "ema": 0.0,
                "mass": 0.0,
                "steps": 0,
                "calls": 0,
                "bound": None,
            }
            for group in opt.param_groups
        ]
        if any(not math.isfinite(s["base_lr"]) or s["base_lr"] <= 0.0 for s in self._groups):
            raise ValueError("lr_servo requires a finite lr > 0 in every parameter group")

    def _momentum_owner(self) -> Any:
        owner_hook = getattr(self.opt, "_momentum_owner", None)
        return owner_hook() if owner_hook is not None else self.opt

    @torch.no_grad()
    def _alignment(self, group: dict[str, Any]) -> float | None:
        owner_state = self._momentum_owner().state
        dot: Tensor | None = None
        grad_sq: Tensor | None = None
        momentum_sq: Tensor | None = None
        buckets: dict[tuple[Any, ...], tuple[list[Tensor], list[dict[str, Any]]]] = {}
        for param in group["params"]:
            grad = param.grad
            state = owner_state.get(param)
            if grad is None or not state or "m" not in state:
                continue
            key = (tuple(param.shape), param.dtype, grad.dtype, group["momentum_dtype"])
            params, states = buckets.setdefault(key, ([], []))
            params.append(param)
            states.append(state)
        for (shape, _pdtype, _gdtype, momentum_dtype), (params, states) in buckets.items():
            per_tensor = max(math.prod(shape), 1)
            chunk_size = max(1, _REDUCTION_CHUNK_ELEMENTS // per_tensor)
            for start in range(0, len(params), chunk_size):
                pchunk = params[start : start + chunk_size]
                schunk = states[start : start + chunk_size]
                # Batching is essential in the many-small-tensor LoRA regime:
                # one stack/reduction replaces thousands of scalar kernel launches.
                # The fixed element cap bounds all transient fp32 stacks to a few MiB.
                grads = torch.stack([param.grad.detach() for param in pchunk]).float()
                momentum = CodecBuffer.read_stacked(schunk, "m", momentum_dtype, shape)
                d = (grads * momentum).sum()
                g2 = (grads * grads).sum()
                m2 = (momentum * momentum).sum()
                dot = d if dot is None else dot + d
                grad_sq = g2 if grad_sq is None else grad_sq + g2
                momentum_sq = m2 if momentum_sq is None else momentum_sq + m2
        if dot is None or grad_sq is None or momentum_sq is None:
            return None
        values = torch.stack((dot, grad_sq, momentum_sq)).cpu()
        d, g2, m2 = (float(value) for value in values)
        if not all(math.isfinite(value) for value in (d, g2, m2)) or g2 <= 0.0 or m2 <= 0.0:
            return None
        return max(-1.0, min(1.0, d / (math.sqrt(g2 * m2) + _EPS)))

    def _set_lr(self, index: int) -> None:
        state = self._groups[index]
        self.opt.param_groups[index]["lr"] = state["base_lr"] * math.exp(state["log_scale"])

    @torch.no_grad()
    def before_step(self) -> None:
        """Consume current gradients and set the LR used by the upcoming base step."""
        lo = math.log(_MIN_SCALE)
        hi = math.log(_MAX_SCALE)
        for index, (group, state) in enumerate(zip(self.opt.param_groups, self._groups, strict=True)):
            # Reassert ownership if a trainer/scheduler overwrote group["lr"].
            self._set_lr(index)
            state["calls"] += 1
            if state["calls"] % _SAMPLE_EVERY:
                continue
            alignment = self._alignment(group)
            if alignment is None:
                continue
            state["ema"] = _EMA_BETA * state["ema"] + (1.0 - _EMA_BETA) * alignment
            state["mass"] = _EMA_BETA * state["mass"] + (1.0 - _EMA_BETA)
            state["steps"] += 1
            mean = state["ema"] / max(state["mass"], _EPS)
            # Do not chase the stability edge (mean == 0).  Keeping a positive
            # alignment reserve makes the controller back away *before* clear
            # alternating-gradient overshoot appears, which is especially
            # important for Lion's long-memory raw-gradient EMA.
            error = mean - _TARGET_ALIGNMENT
            correction = math.copysign(max(abs(error) - _DEAD_ZONE, 0.0), error)
            proposed = state["log_scale"] + _GAIN * correction
            state["log_scale"] = min(hi, max(lo, proposed))
            if state["log_scale"] <= lo + 1e-12:
                state["bound"] = "lower"
            elif state["log_scale"] >= hi - 1e-12:
                state["bound"] = "upper"
            else:
                state["bound"] = None
            self._set_lr(index)

    def scales(self) -> list[float]:
        return [math.exp(state["log_scale"]) for state in self._groups]

    def state_blob(self) -> dict[str, Any]:
        return {"version": _STATE_VERSION, "groups": [dict(state) for state in self._groups]}

    def load_blob(self, blob: dict[str, Any]) -> None:
        if int(blob.get("version", 0)) != _STATE_VERSION:
            raise ValueError("unsupported LR servo checkpoint version")
        saved = blob.get("groups")
        if not isinstance(saved, list) or len(saved) != len(self._groups):
            raise ValueError("LR servo checkpoint parameter-group topology differs")
        lo, hi = math.log(_MIN_SCALE), math.log(_MAX_SCALE)
        restored = []
        for item in saved:
            state = {
                "base_lr": float(item["base_lr"]),
                "log_scale": float(item["log_scale"]),
                "ema": float(item["ema"]),
                "mass": float(item["mass"]),
                "steps": int(item["steps"]),
                "calls": int(item.get("calls", item["steps"] * _SAMPLE_EVERY)),
                "bound": item.get("bound"),
            }
            if (
                not all(math.isfinite(state[key]) for key in ("base_lr", "log_scale", "ema", "mass"))
                or state["base_lr"] <= 0.0
                or not lo - 1e-12 <= state["log_scale"] <= hi + 1e-12
                or not 0.0 <= state["mass"] <= 1.0 + 1e-12
                or state["steps"] < 0
                or state["calls"] < state["steps"]
                or state["bound"] not in (None, "lower", "upper")
            ):
                raise ValueError("LR servo checkpoint contains invalid controller state")
            restored.append(state)
        self._groups = restored
        for index in range(len(self._groups)):
            self._set_lr(index)


class LRServoMixin:
    """Composable step router for Kaon optimizers with an ``m`` momentum buffer."""

    _lr_servo: LRServo | None

    def _init_lr_servo(self, enabled: bool) -> None:
        if enabled and getattr(self, "_autolr", None) is not None:
            raise ValueError("auto_lr and lr_servo are mutually exclusive")
        self._lr_servo = LRServo(self) if enabled else None  # type: ignore[arg-type]

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        if self._lr_servo is not None:
            self._lr_servo.before_step()
        return self._step_impl(closure)

    def _step_impl(self, closure: Any = None) -> Any:
        raise NotImplementedError("optimizer using LRServoMixin must provide _step_impl")

    def get_lr_servo_scale(self) -> float:
        """Current multiplier for the first parameter group (1.0 when disabled)."""
        return self._lr_servo.scales()[0] if self._lr_servo is not None else 1.0

    def _lr_servo_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        if self._lr_servo is not None:
            state_dict["_lr_servo"] = self._lr_servo.state_blob()
        return state_dict

    def _lr_servo_restore(self, blob: dict[str, Any] | None) -> None:
        if self._lr_servo is None:
            return
        if blob is not None:
            self._lr_servo.load_blob(blob)
        else:
            self._lr_servo = LRServo(self)  # type: ignore[arg-type]

    def state_dict(self) -> dict[str, Any]:
        return self._lr_servo_state_dict(super().state_dict())  # type: ignore[misc]

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        copied = dict(state_dict)
        blob = copied.pop("_lr_servo", None)
        super().load_state_dict(copied)  # type: ignore[misc]
        self._lr_servo_restore(blob)
