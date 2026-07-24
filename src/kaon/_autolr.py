"""Autonomous, composable learning-rate discovery for Kaon optimizers.

``AutoLRTuner`` measures updates in the base optimizer's own update space and
runs a continuous DoWG estimate.  A geometric ramp accelerates discovery until
the first stability contact.  Contacts are detected from gradients alone by an
instantaneous EMA spike guard and a fixed-reference, windowed level guard.  No
loss, closure result, scheduler, or trainer decision participates in discovery.

The tuner owns the effective learning rate while adapting.  It freezes after a
confirmed stability edge, when it reaches its conservative fuse, or after 192
adapting steps.  Its only persistent tensor cost is one exact parameter snapshot,
which is released at freeze.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import torch
from torch import Tensor

__all__ = ["AutoLRMixin", "AutoLRTuner", "DEFAULT_FUSE_REL"]

# Default for the hosts' ``auto_lr_fuse_rel`` kwarg.
DEFAULT_FUSE_REL: float = 20.0

_SEED_REL: float = 1e-6
_FUSE_REF_REL: float = 3e-3
_EPS: float = 1e-30

# Instantaneous stability guard.
_SPIKE_RATIO: float = 5.0
_EMA_BETA: float = 0.9
_EMA_WARMUP: int = 3

# Contact policy.
_EDGE_BAND: float = 2.0
_BACKOFF: float = 0.5

# Discovery ramp, active only before the first contact.
_RAMP_GROWTH: float = 1.1

# Fixed-reference level guard.  The baseline never follows the trajectory, so a
# gradual increase cannot "cook" the detector as it can a moving EMA.
_LEVEL_BASE_STEPS: int = 8
_LEVEL_WINDOW: int = 8
_LEVEL_RATIO: float = 2.5
_LEVEL_NMAD: float = 4.0

_ADAPT_MAX_STEPS: int = 192
_D0_FUSE_HEADROOM: float = 4.0


def _median(vals: list[float]) -> float:
    ordered = sorted(vals)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


class AutoLRTuner:
    """Update-space DoWG tuner attached to a Kaon optimizer host.

    The host exposes ``param_groups``, ``state``, ``_step_impl()`` and the
    internal ``_autolr_reset_base_state()`` hook supplied by :class:`AutoLRMixin`.
    Fused hosts override the reset hook to invalidate pointer caches as well.
    """

    def __init__(
        self,
        opt: torch.optim.Optimizer,
        *,
        scale: float,
        fuse_rel: float,
        d0: float | None = None,
    ) -> None:
        if not scale > 0.0:
            raise ValueError(f"auto_lr_scale must be > 0, got {scale}")
        if not fuse_rel > 0.0:
            raise ValueError(f"auto_lr_fuse_rel must be > 0, got {fuse_rel}")
        if d0 is not None and not d0 > 0.0:
            raise ValueError(
                "auto_lr_d0 must be > 0 (or None for the data-relative seed), "
                f"got {d0}"
            )
        self.opt = opt

        # The tuner owns group["lr"].  A trainer default such as 1e-4 must not
        # silently scale the autonomous result.
        small = [float(group["lr"]) for group in opt.param_groups if group["lr"] < 1.0]
        if small:
            warnings.warn(
                "auto_lr=True discovers the learning rate itself; the base lr "
                f"({small[0]:g}) is ignored. Use auto_lr_scale for an explicit "
                "multiplier, or set lr=1.0 to silence this warning.",
                stacklevel=3,
            )

        self._scale = float(scale)
        self._fuse_rel = float(fuse_rel)
        self._d0 = float(d0) if d0 is not None else None

        self.S: float | None = None
        self._seed: float | None = None
        self._fuse: float | None = None
        self._v = _EPS
        self._rbar = 0.0
        self._t = 0
        self._x0: dict[Tensor, Tensor] = {}

        self._gema: float | None = None
        self._edge: float | None = None
        self._contacts = 0
        self._nan_run = 0
        self._nonfinite_backed_off = False
        self._nonfinite_warned = False
        self._level_base: list[float] = []
        self._level_window: list[float] = []
        self._report_loss_warned = False

        self.frozen = False
        self.frozen_lr: float | None = None
        self.freeze_reason: str | None = None

    # -- compatibility -----------------------------------------------------
    def report_loss(self, loss: Any) -> None:
        """Deprecated compatibility no-op; AutoLR no longer consumes loss."""
        del loss
        if not self._report_loss_warned:
            warnings.warn(
                "optimizer.report_loss() is deprecated and ignored: auto_lr is fully "
                "autonomous as of kaon 0.7.4.",
                DeprecationWarning,
                stacklevel=3,
            )
            self._report_loss_warned = True

    # -- introspection -----------------------------------------------------
    @property
    def _ramp_on(self) -> bool:
        return self._edge is None

    def get_d(self) -> float:
        if self.frozen:
            return float(self.frozen_lr)  # type: ignore[arg-type]
        return float(self.S) if self.S is not None else 0.0

    # -- discovery ---------------------------------------------------------
    def _initialize(self) -> None:
        """Seed the tuner and snapshot every currently trainable parameter."""
        trainable = [
            p
            for group in self.opt.param_groups
            for p in group["params"]
            if p.requires_grad
        ]
        sq = 0.0
        count = 0
        for param in trainable:
            param_f = param.detach().float()
            sq += float((param_f * param_f).sum())
            count += param_f.numel()
        rms = math.sqrt(sq / max(count, 1)) if count else 0.0

        requested = self._d0 if self._d0 is not None else max(_SEED_REL * rms, 1e-10)
        # d0 may retain a small amount of compatibility headroom, but it cannot
        # inflate its own ceiling without bound.  The old ``fuse_rel * d0`` rule
        # made the protection disappear exactly when it was needed most (and
        # could freeze a damaged run at 20*d0).
        base_fuse = max(self._fuse_rel * _FUSE_REF_REL * rms, 1e-12)
        requested_fuse = max(base_fuse, self._fuse_rel * requested)
        self._fuse = min(requested_fuse, _D0_FUSE_HEADROOM * base_fuse)
        self.S = min(requested, self._fuse)
        self._seed = self.S
        if self._d0 is not None and requested > self._fuse:
            warnings.warn(
                f"auto_lr_d0={requested:g} exceeds the autonomous safety fuse "
                f"({self._fuse:g}) and was clamped to it.",
                stacklevel=3,
            )
        self._x0 = {param: param.detach().clone() for param in trainable}

    @staticmethod
    def _global_norm(tensors: list[Tensor]) -> float:
        norms = torch._foreach_norm(tensors)
        if len({norm.dtype for norm in norms}) > 1:
            norms = [norm.float() for norm in norms]
        return math.sqrt(float(torch.stack(norms).float().square_().sum()))

    def _level_is_hot(self, grad_norm: float) -> bool:
        """Detect a sustained shift from the first eight finite grad norms."""
        log_norm = math.log(max(grad_norm, _EPS))
        if len(self._level_base) < _LEVEL_BASE_STEPS:
            self._level_base.append(log_norm)
            return False

        self._level_window.append(log_norm)
        if len(self._level_window) > _LEVEL_WINDOW:
            del self._level_window[0]
        if len(self._level_window) < _LEVEL_WINDOW:
            return False

        baseline = _median(self._level_base)
        raw_mad = _median([abs(value - baseline) for value in self._level_base])
        robust_mad = 1.4826 * raw_mad
        threshold = baseline + max(
            math.log(_LEVEL_RATIO),
            _LEVEL_NMAD * robust_mad,
        )
        return _median(self._level_window) > threshold

    def _complete_adapting_step(self) -> None:
        """Advance the hard budget and freeze at a terminal autonomous bound."""
        self._t += 1
        if self.frozen:
            return
        if self._fuse is not None and self.S is not None and self._fuse * (1 - 1e-12) <= self.S:
            self.S = self._fuse
            self._do_freeze("fuse_bound")
        elif self._t >= _ADAPT_MAX_STEPS:
            self._do_freeze("budget_bound")

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        """Take one autonomous adapting step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        params = [
            p
            for group in self.opt.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not params:
            return loss

        if self.S is None:
            self._initialize()

        grad_norm = self._global_norm([p.grad for p in params])
        if not math.isfinite(grad_norm):
            # A persistent poison source is not repaired by repeatedly shrinking LR.
            # Roll back/back off once per consecutive run and skip every poisoned step.
            self._nan_run += 1
            if not self._nonfinite_backed_off:
                self._nonfinite_backoff()
            elif not self._nonfinite_warned:
                warnings.warn(
                    "auto_lr: gradients are non-finite again; the LR has already been "
                    "backed off once, so this is not treated as another LR contact. Check "
                    "the data, loss, and precision; steps are skipped until gradients "
                    "become finite.",
                    stacklevel=2,
                )
                self._nonfinite_warned = True
            self._complete_adapting_step()
            return loss
        self._nan_run = 0

        level_hot = self._level_is_hot(grad_norm)
        spike = (
            self._gema is not None
            and self._gema > 0.0
            and self._t >= _EMA_WARMUP
            and grad_norm > _SPIKE_RATIO * self._gema
        )
        if self._gema is None or self._gema <= 0.0:
            self._gema = grad_norm
        elif level_hot or spike:
            if self._ramp_on:
                # The gradient belongs to the pre-rollback parameters, so it cannot
                # safely be applied after restoring x0 and resetting base state.
                self._edge_contact(step_after=False)
                self._complete_adapting_step()
                return loss

            frozen_now = self._edge_contact(step_after=True)
            self._gema = grad_norm
            self._complete_adapting_step()
            if frozen_now:
                return loss
            # A non-confirming contact re-anchors DoWG and then applies this finite
            # gradient at the backed-off LR.
        else:
            self._gema = _EMA_BETA * self._gema + (1.0 - _EMA_BETA) * grad_norm

        s_prev = float(self.S)
        for group in self.opt.param_groups:
            group["lr"] = s_prev

        previous = [
            p.detach().float() if p.dtype != torch.float32 else p.detach().clone()
            for p in params
        ]
        self.opt._step_impl()  # type: ignore[attr-defined]

        current = [p.detach().float() for p in params]
        delta = torch._foreach_sub(current, previous)
        update_norm_sq = float(torch.stack(torch._foreach_norm(delta)).square_().sum())
        update_norm_sq /= s_prev * s_prev + _EPS

        if update_norm_sq <= 0.0:
            self._complete_adapting_step()
            return loss

        # Distance is global over all snapshotted trainable parameters, including
        # ones without a gradient on this particular step.
        tracked = list(self._x0)
        displacement = torch._foreach_sub(
            [p.detach().float() for p in tracked],
            [self._x0[p].float() for p in tracked],
        )
        distance = self._global_norm(displacement)
        self._rbar = max(self._rbar, distance)
        radius_sq = self._rbar * self._rbar
        self._v += radius_sq * update_norm_sq

        new_s = (radius_sq / math.sqrt(self._v)) * self._scale
        if self._ramp_on:
            new_s = max(new_s, s_prev * _RAMP_GROWTH)
        if self._fuse is not None:
            new_s = min(new_s, self._fuse)
        self.S = max(new_s, 1e-12)

        self._complete_adapting_step()
        return loss

    # -- contacts and freeze ----------------------------------------------
    def _nonfinite_backoff(self) -> None:
        """Perform the single lifetime rollback allowed for poisoned gradients."""
        contact = float(self.S)  # type: ignore[arg-type]
        self.S = max(contact * _BACKOFF, 1e-12)
        self._contacts += 1
        self._nonfinite_backed_off = True
        if self._edge is None:
            self._edge = contact
        self._level_window.clear()
        for param, reference in self._x0.items():
            param.copy_(reference)
        self.opt._autolr_reset_base_state()  # type: ignore[attr-defined]
        self._rbar = 0.0
        self._v = _EPS

    def _edge_contact(self, *, step_after: bool) -> bool:
        """Back off at a stability contact and freeze on a comparable repeat."""
        contact = float(self.S)  # type: ignore[arg-type]
        self.S = max(contact * _BACKOFF, 1e-12)
        self._contacts += 1
        comparable = (
            self._edge is not None
            and self._edge / _EDGE_BAND <= contact <= self._edge * _EDGE_BAND
        )
        if comparable:
            self._do_freeze("edge_confirmed")
            if step_after:
                self.opt._step_impl()  # type: ignore[attr-defined]
            return True

        first_contact = self._edge is None
        self._edge = contact
        self._level_window.clear()

        if first_contact:
            # The exploratory trajectory is discarded exactly, together with every
            # piece of base state or fused pointer metadata derived from it.
            for param, reference in self._x0.items():
                param.copy_(reference)
            self.opt._autolr_reset_base_state()  # type: ignore[attr-defined]
        else:
            # A non-comparable later contact starts a new local DoWG epoch without
            # allocating another reference copy.
            for param in self._x0:
                self._x0[param].copy_(param)

        self._rbar = 0.0
        self._v = _EPS
        return False

    def _do_freeze(self, reason: str) -> None:
        """Lock the effective LR and release the parameter reference buffers."""
        self.frozen_lr = float(self.S) if self.S is not None else 0.0
        for group in self.opt.param_groups:
            group["lr"] = self.frozen_lr
        self._x0.clear()
        self.freeze_reason = reason
        self.frozen = True

    # -- checkpointing -----------------------------------------------------
    def state_blob(self) -> dict[str, Any]:
        """Return tuner state with references stored in stable parameter order."""
        params = [p for group in self.opt.param_groups for p in group["params"]]
        index = {param: i for i, param in enumerate(params)}
        x0_list: list[Tensor | None] = [None] * len(params)
        for param, reference in self._x0.items():
            x0_list[index[param]] = reference.detach().clone()
        return {
            "version": 2,
            "S": self.S,
            "seed": self._seed,
            "fuse": self._fuse,
            "v": self._v,
            "rbar": self._rbar,
            "t": self._t,
            "x0": x0_list,
            "gema": self._gema,
            "edge": self._edge,
            "contacts": self._contacts,
            "nan_run": self._nan_run,
            "nonfinite_backed_off": self._nonfinite_backed_off,
            "nonfinite_warned": self._nonfinite_warned,
            "level_base": list(self._level_base),
            "level_window": list(self._level_window),
            "frozen": self.frozen,
            "frozen_lr": self.frozen_lr,
            "freeze_reason": self.freeze_reason,
        }

    def load_blob(self, blob: dict[str, Any]) -> None:
        """Restore 0.7.4 state or migrate a 0.7.3 blob.

        The obsolete 0.7.3 ``probe`` member is intentionally ignored.  All fields
        shared with continuous DoWG remain usable, while new detector state starts
        empty when it is absent.
        """
        params = [p for group in self.opt.param_groups for p in group["params"]]
        self.S = blob.get("S")
        self._seed = blob.get("seed")
        self._fuse = blob.get("fuse")
        self._v = float(blob.get("v", _EPS))
        self._rbar = float(blob.get("rbar", 0.0))
        self._t = int(blob.get("t", 0))
        self._x0 = {}
        for index, reference in enumerate(blob.get("x0", [])):
            if reference is not None and index < len(params):
                self._x0[params[index]] = reference.to(
                    device=params[index].device,
                    dtype=params[index].dtype,
                )
        self._gema = blob.get("gema")
        self._edge = blob.get("edge")
        self._contacts = int(blob.get("contacts", 1 if self._edge is not None else 0))
        self._nan_run = int(blob.get("nan_run", 0))
        self._nonfinite_backed_off = bool(blob.get("nonfinite_backed_off", False))
        self._nonfinite_warned = bool(blob.get("nonfinite_warned", False))
        self._level_base = [float(value) for value in blob.get("level_base", [])]
        self._level_window = [float(value) for value in blob.get("level_window", [])]
        self.frozen = bool(blob.get("frozen", False))
        self.frozen_lr = blob.get("frozen_lr")
        self.freeze_reason = blob.get("freeze_reason")
        if isinstance(blob.get("probe"), dict) and not self.frozen:
            # A 0.7.3 loss-probe checkpoint may contain parameters and optimizer
            # moments from a half-finished ladder rung.  Discard that trial and
            # restart autonomous discovery from its exact pre-probe snapshot.
            with torch.no_grad():
                for param, reference in self._x0.items():
                    param.copy_(reference)
            self.opt._autolr_reset_base_state()  # type: ignore[attr-defined]
            self.S = self._seed if self._seed is not None else self.S
            self._v = _EPS
            self._rbar = 0.0
            self._t = 0
            self._gema = None
            self._edge = None
            self._contacts = 0
            self._nan_run = 0
            self._nonfinite_backed_off = False
            self._nonfinite_warned = False
            self._level_base = []
            self._level_window = []
        if self.frozen:
            for group in self.opt.param_groups:
                group["lr"] = self.frozen_lr


class AutoLRMixin:
    """Attach autonomous AutoLR to a Kaon optimizer with shared plumbing."""

    _autolr: AutoLRTuner | None

    def _init_autolr(
        self,
        auto_lr: bool,
        scale: float,
        fuse_rel: float,
        d0: float | None = None,
    ) -> None:
        self._autolr = (
            AutoLRTuner(self, scale=scale, fuse_rel=fuse_rel, d0=d0)  # type: ignore[arg-type]
            if auto_lr
            else None
        )

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        tuner = self._autolr
        if tuner is not None:
            if not tuner.frozen:
                return tuner.step(closure)
            for group in self.param_groups:  # type: ignore[attr-defined]
                group["lr"] = tuner.frozen_lr
        return self._step_impl(closure)

    def _step_impl(self, closure: Any = None) -> Any:
        raise NotImplementedError("optimizer using AutoLRMixin must provide _step_impl")

    def _autolr_reset_base_state(self) -> None:
        """Reset base state after rollback; fused hosts also invalidate caches."""
        self.state.clear()  # type: ignore[attr-defined]

    def get_d(self) -> float:
        if self._autolr is not None:
            return self._autolr.get_d()
        return float(self.param_groups[0]["lr"])  # type: ignore[attr-defined]

    def report_loss(self, loss: Any) -> None:
        """Deprecated compatibility no-op when AutoLR is enabled."""
        if self._autolr is not None:
            self._autolr.report_loss(loss)

    def is_frozen(self) -> bool:
        return self._autolr is not None and self._autolr.frozen

    def _autolr_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        if self._autolr is not None:
            state_dict["_autolr"] = self._autolr.state_blob()
        return state_dict

    def _autolr_load(self, state_dict: dict[str, Any], inner_load: Any) -> None:
        copied = dict(state_dict)
        blob = copied.pop("_autolr", None)
        inner_load(copied)
        if self._autolr is not None and blob is not None:
            self._autolr.load_blob(blob)
