"""Small passive telemetry records shared by Kaon optimizer experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["AdakaonStepTelemetry"]


def _cosine(dot: float, left_sq: float, right_sq: float) -> float | None:
    denom = math.sqrt(max(left_sq, 0.0) * max(right_sq, 0.0))
    if denom == 0.0 or not math.isfinite(denom):
        return None
    return dot / denom


@dataclass(frozen=True, slots=True)
class AdakaonStepTelemetry:
    """One native/foreach Adakaon step reduced to optimizer-wide scalars.

    ``direction`` is the learned direction after preconditioning, RMS clipping,
    momentum and (when enabled) cautious masking, but before both learning-rate
    scaling and weight decay.  ``prev_direction`` uses the existing momentum
    buffer when momentum is enabled.  With fp32/bf16 momentum it is therefore the
    exact previously applied learned direction only when ``cautious=False``;
    int8/4bit expose the codec-quantized approximation.  With cautious enabled it
    is also an explicitly documented proxy because Adakaon does not retain the old
    cautious mask.  Momentum-free telemetry keeps the exact previous direction
    only while a hook is installed.
    """

    step: int
    active_numel: int
    previous_numel: int
    grad_norm_sq: float
    direction_norm_sq: float
    grad_direction_dot: float
    prev_direction_norm_sq: float
    grad_prev_direction_dot: float
    decay_direction_norm_sq: float
    finite: bool

    @property
    def grad_direction_cosine(self) -> float | None:
        return _cosine(
            self.grad_direction_dot,
            self.grad_norm_sq,
            self.direction_norm_sq,
        )

    @property
    def grad_prev_direction_cosine(self) -> float | None:
        return _cosine(
            self.grad_prev_direction_dot,
            self.grad_norm_sq,
            self.prev_direction_norm_sq,
        )


class _AdakaonTelemetryAccumulator:
    """Device-local scalar accumulation; synchronizes only when finalized."""

    _NAMES = (
        "grad_norm_sq",
        "direction_norm_sq",
        "grad_direction_dot",
        "prev_direction_norm_sq",
        "grad_prev_direction_dot",
        "decay_direction_norm_sq",
    )

    def __init__(self, step: int) -> None:
        self.step = step
        self.active_numel = 0
        self.previous_numel = 0
        self._by_device: dict[torch.device, dict[str, Tensor]] = {}
        self._finite: dict[torch.device, Tensor] = {}

    def observe(
        self,
        grad: Tensor,
        direction: Tensor,
        prev_direction: Tensor,
        decay_direction: Tensor | None,
        *,
        previous_numel: int,
    ) -> None:
        grad_f = grad.float()
        direction_f = direction.float()
        prev_f = prev_direction.float()
        decay_f = decay_direction.float() if decay_direction is not None else None
        values = {
            "grad_norm_sq": grad_f.square().sum(),
            "direction_norm_sq": direction_f.square().sum(),
            "grad_direction_dot": (grad_f * direction_f).sum(),
            "prev_direction_norm_sq": prev_f.square().sum(),
            "grad_prev_direction_dot": (grad_f * prev_f).sum(),
            "decay_direction_norm_sq": (
                decay_f.square().sum()
                if decay_f is not None
                else torch.zeros((), dtype=torch.float32, device=grad.device)
            ),
        }
        sums = self._by_device.setdefault(grad.device, {})
        for name, value in values.items():
            sums[name] = value if name not in sums else sums[name] + value
        tensors = [grad_f, direction_f, prev_f]
        if decay_f is not None:
            tensors.append(decay_f)
        finite = torch.stack([torch.isfinite(t).all() for t in tensors]).all()
        self._finite[grad.device] = (
            finite if grad.device not in self._finite else self._finite[grad.device] & finite
        )
        self.active_numel += grad.numel()
        self.previous_numel += previous_numel

    def finish(self) -> AdakaonStepTelemetry:
        reduced = {
            name: sum(float(sums.get(name, 0.0)) for sums in self._by_device.values())
            for name in self._NAMES
        }
        finite = all(bool(value) for value in self._finite.values())
        return AdakaonStepTelemetry(
            step=self.step,
            active_numel=self.active_numel,
            previous_numel=self.previous_numel,
            finite=finite,
            **reduced,
        )
