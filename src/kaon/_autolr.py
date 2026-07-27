"""Compatibility quarantine for Kaon's retired AutoLR experiment.

The autonomous controller could silently overshoot a workload's safe learning
rate and damage a training run. Its implementation remains recoverable from
Git history and the research branches, but production hosts now fail closed
before the first optimizer step when ``auto_lr=True`` is requested.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch

__all__ = ["AutoLRMixin", "AutoLRTuner", "DEFAULT_FUSE_REL"]

# Retained only for source-compatible constructor signatures while quarantined.
DEFAULT_FUSE_REL: float = 20.0

_QUARANTINE_MESSAGE = (
    "auto_lr=True is disabled: Kaon's experimental AutoLR controller could "
    "overshoot the safe learning rate and damage training. Choose an explicit "
    "lr instead. The experiment is documented in docs/EXPERIMENTS_GRAVEYARD.md."
)


class AutoLRTuner:
    """Retired public symbol retained to produce a safe, actionable error."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError(_QUARANTINE_MESSAGE)


class AutoLRMixin:
    """Compatibility plumbing for optimizer classes that formerly hosted AutoLR."""

    _autolr: None

    def _init_autolr(
        self,
        auto_lr: bool,
        scale: float,
        fuse_rel: float,
        d0: float | None = None,
    ) -> None:
        del scale, fuse_rel, d0
        if auto_lr:
            raise RuntimeError(_QUARANTINE_MESSAGE)
        self._autolr = None

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        return self._step_impl(closure)

    def _step_impl(self, closure: Any = None) -> Any:
        raise NotImplementedError("optimizer using AutoLRMixin must provide _step_impl")

    def _autolr_reset_base_state(self) -> None:
        self.state.clear()  # type: ignore[attr-defined]

    def get_d(self) -> float:
        """Return the configured LR; retained for compatibility with old callers."""
        return float(self.param_groups[0]["lr"])  # type: ignore[attr-defined]

    def report_loss(self, loss: Any) -> None:
        """Retired compatibility no-op."""
        del loss
        warnings.warn(
            "optimizer.report_loss() is retired with AutoLR and has no effect",
            DeprecationWarning,
            stacklevel=2,
        )

    def is_frozen(self) -> bool:
        return False

    def _autolr_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        return state_dict

    def _autolr_load(self, state_dict: dict[str, Any], inner_load: Any) -> None:
        copied = dict(state_dict)
        legacy_blob = copied.pop("_autolr", None)
        if legacy_blob is not None:
            warnings.warn(
                "Ignoring retired AutoLR state while loading this checkpoint; "
                "the base optimizer state is preserved",
                RuntimeWarning,
                stacklevel=2,
            )
        inner_load(copied)
