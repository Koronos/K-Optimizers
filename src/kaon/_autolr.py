"""Continuous, trainer-independent Mechanic step-size adaptation for Kaon.

The controller treats the host optimizer as a *unit-update oracle*.  Its six
exponentially-discounted bettors continuously rescale the anchored trajectory;
there is no range-test phase, freeze, horizon, loss signal, or trainer decision.
Hosts with a live parameter view (MSAM/Nekaon) expose three private lifecycle
hooks so the anchor always follows the true iterate rather than the lookahead.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import torch
from torch import Tensor

__all__ = ["AutoLRMixin", "AutoLRTuner", "DEFAULT_FUSE_REL"]

DEFAULT_FUSE_REL: float = 20.0

_BETAS = tuple(1.0 - 0.1**k for k in range(1, 7))
_EPS = 1e-8
_S_INIT = 1e-6
_STATE_VERSION = 3
_REDUCTION_CHUNK_ELEMENTS = 2_000_000


class AutoLRTuner:
    """Mechanic controller attached directly to a Kaon optimizer host."""

    def __init__(
        self,
        opt: torch.optim.Optimizer,
        *,
        scale: float,
        fuse_rel: float,
        d0: float | None = None,
    ) -> None:
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"auto_lr_scale must be finite and > 0, got {scale}")
        if not math.isfinite(fuse_rel) or fuse_rel <= 0.0:
            raise ValueError(f"auto_lr_fuse_rel must be finite and > 0, got {fuse_rel}")
        if d0 is not None and (not math.isfinite(d0) or d0 <= 0.0):
            raise ValueError(f"auto_lr_d0 must be finite and > 0 or None, got {d0}")
        self.opt = opt

        small = [float(group["lr"]) for group in opt.param_groups if group["lr"] < 1.0]
        if small:
            warnings.warn(
                "auto_lr=True discovers the learning rate itself; the base lr "
                f"({small[0]:g}) is ignored. Use auto_lr_scale for an explicit "
                "multiplier, or set lr=1.0 to silence this warning.",
                stacklevel=3,
            )

        self._scale = float(scale)
        # Retained in state/API for source compatibility. Mechanic has no fuse.
        self._fuse_rel = float(fuse_rel)
        self._d0 = float(d0) if d0 is not None else None
        if self._d0 is not None:
            warnings.warn(
                "auto_lr_d0 is deprecated and ignored by continuous Mechanic; "
                "the safe seed is fixed. Use auto_lr_scale only when an explicit "
                "global multiplier is intended.",
                stacklevel=3,
            )
        # A user-provided d0 must never turn the safe-start controller into an
        # accidentally high-LR controller. This seed is an algorithm invariant.
        self._seed = _S_INIT

        self._r = [0.0] * len(_BETAS)
        self._m = [0.0] * len(_BETAS)
        self._v = [0.0] * len(_BETAS)
        self._s = [self._seed / len(_BETAS)] * len(_BETAS)
        self._x0: dict[Tensor, Tensor] = {}
        # The ideal normalized trajectory must not be inferred back from bf16
        # weights: a safe 1e-6 seed is usually below one bf16 ULP.  Keeping it in
        # fp32 lets the controller receive feedback even while the materialized
        # model weights temporarily round back to their anchor.
        self._delta: dict[Tensor, Tensor] = {}
        self._t = 0
        self._last_h = 0.0
        self._nonfinite_steps = 0
        self._nonfinite_warned = False
        self._report_loss_warned = False

        # Compatibility attributes: continuous Mechanic never freezes.
        self.frozen = False
        self.frozen_lr: float | None = None
        self.freeze_reason: str | None = None
        self.S = self._applied_scale()
        self._set_group_lr(self.S)

    def _params(self) -> list[Tensor]:
        return [p for group in self.opt.param_groups for p in group["params"] if p.requires_grad]

    def _applied_scale(self) -> float:
        return self._scale * float(sum(self._s))

    def _set_group_lr(self, value: float) -> None:
        for group in self.opt.param_groups:
            group["lr"] = value

    def _host_hook(self, name: str, *args: Any) -> None:
        hook = getattr(self.opt, name, None)
        if hook is not None:
            hook(*args)

    def report_loss(self, loss: Any) -> None:
        """Deprecated compatibility no-op; Mechanic never consumes loss."""
        del loss
        if not self._report_loss_warned:
            warnings.warn(
                "optimizer.report_loss() is deprecated and ignored: auto_lr is fully autonomous.",
                DeprecationWarning,
                stacklevel=3,
            )
            self._report_loss_warned = True

    def get_d(self) -> float:
        return float(self.S)

    @staticmethod
    def _all_finite(tensors: list[Tensor]) -> bool:
        """Check many tensors with multi-tensor kernels and one sync per device."""
        by_device_dtype: dict[tuple[torch.device, torch.dtype], list[Tensor]] = {}
        for tensor in tensors:
            by_device_dtype.setdefault((tensor.device, tensor.dtype), []).append(tensor)
        norms_by_device: dict[torch.device, list[Tensor]] = {}
        for (device, _dtype), items in by_device_dtype.items():
            norms = torch._foreach_norm(items)
            norms_by_device.setdefault(device, []).extend(norms)
        return all(
            bool(torch.isfinite(torch.stack(norms)).all())
            for norms in norms_by_device.values()
        )

    @staticmethod
    def _feedback_sum(params: list[Tensor], delta: dict[Tensor, Tensor]) -> float:
        """Compute global <gradient, trajectory> in shape buckets."""
        by_device: dict[torch.device, list[Tensor]] = {}
        active = [p for p in params if p.grad is not None]
        for chunk in AutoLRTuner._shape_batches(active):
            if len(chunk) == 1 and chunk[0].numel() > _REDUCTION_CHUNK_ELEMENTS:
                p = chunk[0]
                term = torch.vdot(p.grad.detach().float().reshape(-1), delta[p].reshape(-1))
                by_device.setdefault(p.device, []).append(term)
                continue
            gradients = torch.stack([p.grad.detach() for p in chunk]).float()
            trajectory = torch.stack([delta[p] for p in chunk])
            by_device.setdefault(chunk[0].device, []).append(
                torch.vdot(gradients.reshape(-1), trajectory.reshape(-1))
            )
        return sum(float(torch.stack(terms).sum()) for terms in by_device.values())

    @staticmethod
    def _shape_batches(params: list[Tensor]) -> list[list[Tensor]]:
        buckets: dict[tuple[torch.device, torch.dtype, tuple[int, ...]], list[Tensor]] = {}
        for p in params:
            key = (p.device, p.dtype, tuple(p.shape))
            buckets.setdefault(key, []).append(p)
        result: list[list[Tensor]] = []
        for (_device, _dtype, shape), bucket in buckets.items():
            per_tensor = max(math.prod(shape), 1)
            chunk_size = max(_REDUCTION_CHUNK_ELEMENTS // per_tensor, 1)
            result.extend(
                bucket[start : start + chunk_size]
                for start in range(0, len(bucket), chunk_size)
            )
        return result

    def _materialize(
        self, params: list[Tensor], delta: dict[Tensor, Tensor], scale: float
    ) -> None:
        for batch in self._shape_batches(params):
            if len(batch) == 1 and batch[0].numel() > _REDUCTION_CHUNK_ELEMENTS:
                p = batch[0]
                value = delta[p].clone().mul_(-scale).add_(self._x0[p])
                p.copy_(value.to(dtype=p.dtype))
                continue
            anchors = torch.stack([self._x0[p] for p in batch])
            values = torch.stack([delta[p] for p in batch])
            values.mul_(-scale).add_(anchors)
            values = values.to(dtype=batch[0].dtype)
            torch._foreach_copy_([p.data for p in batch], list(values.unbind(0)))

    def _warn_nonfinite(self) -> None:
        if not self._nonfinite_warned:
            warnings.warn(
                "auto_lr: non-finite gradients were skipped without advancing the "
                "optimizer or Mechanic state.",
                stacklevel=3,
            )
            self._nonfinite_warned = True

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        params = self._params()
        active = [p for p in params if p.grad is not None]
        if not active:
            return loss
        # Check before declimbing a live MSAM/Nekaon view: a skipped poisoned
        # step must leave that view and every state tensor exactly untouched.
        if not self._all_finite([p.grad for p in active]):
            self._nonfinite_steps += 1
            self._warn_nonfinite()
            return loss

        scale_prev = self._applied_scale()
        self._host_hook("_autolr_prepare_true")
        if not self._x0:
            self._x0 = {p: p.detach().clone() for p in params}
            self._delta = {p: torch.zeros_like(p, dtype=torch.float32) for p in params}

        h = self._feedback_sum(params, self._delta)
        if not math.isfinite(h):
            self._nonfinite_steps += 1
            self._warn_nonfinite()
            self._host_hook("_autolr_restore_live", scale_prev)
            return loss

        # Materialize the controller's ideal true iterate before asking the base
        # optimizer for a unit update.  This also normalizes any bf16 round-trip
        # noise introduced while removing an MSAM/Nekaon live climb.
        self._materialize(params, self._delta, scale_prev)

        self._set_group_lr(1.0)
        self.opt._step_impl()  # type: ignore[attr-defined]
        # A live-view host now carries a newly-created *unit-scale* climb. Remove
        # it before reading the virtual optimizer's true output.
        self._host_hook("_autolr_after_virtual_step")

        if not self._all_finite(params):
            self._materialize(params, self._delta, scale_prev)
            self._set_group_lr(scale_prev)
            raise FloatingPointError(
                "base optimizer produced non-finite parameters during the unit update; "
                "parameters were restored, but optimizer state advanced"
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
            ((self._seed / len(_BETAS)) * m + max(r, 0.0)) / (math.sqrt(v) + _EPS)
            for m, r, v in zip(new_m, new_r, new_v, strict=True)
        ]
        if not all(math.isfinite(value) and value >= 0.0 for value in candidate):
            self._materialize(params, self._delta, scale_prev)
            self._set_group_lr(scale_prev)
            raise FloatingPointError("Mechanic produced a non-finite scale; optimizer state advanced")

        new_scale = self._scale * float(sum(candidate))
        new_delta: dict[Tensor, Tensor] = {}
        for batch in self._shape_batches(params):
            if len(batch) == 1 and batch[0].numel() > _REDUCTION_CHUNK_ELEMENTS:
                p = batch[0]
                trajectories = self._delta[p].clone()
                old_representable = (
                    self._delta[p].clone().mul_(-scale_prev).add_(self._x0[p]).to(dtype=p.dtype)
                )
                trajectories.add_(old_representable).sub_(p.detach())
                new_delta[p] = trajectories
                continue
            anchors = torch.stack([self._x0[p] for p in batch])
            trajectories = torch.stack([self._delta[p] for p in batch])
            old_representable = trajectories.mul(-scale_prev).add_(anchors).to(dtype=batch[0].dtype)
            current = torch.stack([p.detach() for p in batch])
            trajectories.add_(old_representable).sub_(current)
            new_delta.update(zip(batch, trajectories.unbind(0), strict=True))

        self._materialize(params, new_delta, new_scale)

        if not self._all_finite(params):
            self._materialize(params, self._delta, scale_prev)
            self._set_group_lr(scale_prev)
            raise FloatingPointError(
                "Mechanic reconstruction produced non-finite parameters; parameters were "
                "restored, but optimizer state advanced"
            )

        self._m, self._v, self._r, self._s = new_m, new_v, new_r, candidate
        self._delta = new_delta
        self.S = new_scale

        self._last_h = h
        self._t += 1
        self._set_group_lr(self.S)
        # Rebuild the live lookahead from the refreshed unit-direction momentum,
        # scaled by the exact Mechanic value used for the true reconstruction.
        self._host_hook("_autolr_restore_live", self.S)
        return loss

    def state_blob(self) -> dict[str, Any]:
        params = self._params()
        return {
            "version": _STATE_VERSION,
            "algorithm": "mechanic",
            "scale": self._scale,
            "fuse_rel": self._fuse_rel,
            "d0": self._d0,
            "seed": self._seed,
            "r": list(self._r),
            "m": list(self._m),
            "v": list(self._v),
            "s": list(self._s),
            "x0": [self._x0[p].detach().clone() if p in self._x0 else None for p in params],
            "delta": [
                self._delta[p].detach().clone() if p in self._delta else None for p in params
            ],
            "t": self._t,
            "last_h": self._last_h,
            "nonfinite_steps": self._nonfinite_steps,
            "nonfinite_warned": self._nonfinite_warned,
        }

    def load_blob(self, blob: dict[str, Any]) -> None:
        version = int(blob.get("version", 0))
        if version != _STATE_VERSION or blob.get("algorithm") != "mechanic":
            raise ValueError(
                "AutoLR checkpoint uses the retired probe/DoWG controller and cannot be "
                "safely migrated to continuous Mechanic; restart AutoLR from an unadapted checkpoint"
            )
        params = self._params()
        saved_x0 = blob.get("x0")
        if not isinstance(saved_x0, list) or len(saved_x0) != len(params):
            raise ValueError("AutoLR Mechanic checkpoint parameter topology differs")
        saved_delta = blob.get("delta")
        if not isinstance(saved_delta, list) or len(saved_delta) != len(params):
            raise ValueError("AutoLR Mechanic checkpoint trajectory topology differs")
        for name in ("r", "m", "v", "s"):
            values = blob.get(name)
            if not isinstance(values, list) or len(values) != len(_BETAS):
                raise ValueError(f"AutoLR Mechanic checkpoint has invalid {name!r} state")
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"AutoLR Mechanic checkpoint has non-finite {name!r} state")

        scale = float(blob["scale"])
        fuse_rel = float(blob.get("fuse_rel", self._fuse_rel))
        d0 = blob.get("d0")
        d0 = None if d0 is None else float(d0)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("AutoLR Mechanic checkpoint has invalid scale")
        if not math.isfinite(fuse_rel) or fuse_rel <= 0.0:
            raise ValueError("AutoLR Mechanic checkpoint has invalid fuse compatibility value")
        if d0 is not None and (not math.isfinite(d0) or d0 <= 0.0):
            raise ValueError("AutoLR Mechanic checkpoint has invalid d0 compatibility value")
        if any(float(value) < 0.0 for value in blob["m"]):
            raise ValueError("AutoLR Mechanic checkpoint has negative m state")
        if any(float(value) < 0.0 for value in blob["v"]):
            raise ValueError("AutoLR Mechanic checkpoint has negative v state")
        if (
            any(float(value) < 0.0 for value in blob["s"])
            or sum(float(value) for value in blob["s"]) <= 0.0
        ):
            raise ValueError("AutoLR Mechanic checkpoint has invalid s state")
        applied_scale = scale * sum(float(value) for value in blob["s"])
        if not math.isfinite(applied_scale) or applied_scale <= 0.0:
            raise ValueError("AutoLR Mechanic checkpoint has invalid applied scale")

        step = int(blob.get("t", 0))
        last_h = float(blob.get("last_h", 0.0))
        nonfinite_steps = int(blob.get("nonfinite_steps", 0))
        if step < 0 or nonfinite_steps < 0 or not math.isfinite(last_h):
            raise ValueError("AutoLR Mechanic checkpoint has invalid counters/feedback")

        for p, anchor, trajectory in zip(params, saved_x0, saved_delta, strict=True):
            if (anchor is None) != (trajectory is None):
                raise ValueError("AutoLR Mechanic checkpoint anchor/trajectory state differs")
            if anchor is None:
                if step > 0:
                    raise ValueError("AutoLR Mechanic checkpoint is missing an active trajectory")
                continue
            if not torch.is_tensor(anchor) or not torch.is_tensor(trajectory):
                raise ValueError("AutoLR Mechanic checkpoint trajectory entries must be tensors")
            if anchor.shape != p.shape or trajectory.shape != p.shape:
                raise ValueError("AutoLR Mechanic checkpoint tensor topology differs")
            if not bool(torch.isfinite(anchor).all()) or not bool(torch.isfinite(trajectory).all()):
                raise ValueError("AutoLR Mechanic checkpoint contains non-finite tensors")

        saved_seed = float(blob.get("seed", _S_INIT))
        if saved_seed != _S_INIT:
            raise ValueError("AutoLR Mechanic checkpoint has an unsafe/noncanonical seed")
        self._scale = scale
        self._fuse_rel = fuse_rel
        self._d0 = d0
        self._seed = _S_INIT
        self._r = [float(value) for value in blob["r"]]
        self._m = [float(value) for value in blob["m"]]
        self._v = [float(value) for value in blob["v"]]
        self._s = [float(value) for value in blob["s"]]
        self._x0 = {
            p: value.to(device=p.device, dtype=p.dtype).clone()
            for p, value in zip(params, saved_x0, strict=True)
            if value is not None
        }
        self._delta = {
            p: value.to(device=p.device, dtype=torch.float32).clone()
            for p, value in zip(params, saved_delta, strict=True)
            if value is not None
        }
        if self._x0.keys() != self._delta.keys():
            raise ValueError("AutoLR Mechanic checkpoint anchor/trajectory state differs")
        self._t = step
        self._last_h = last_h
        self._nonfinite_steps = nonfinite_steps
        self._nonfinite_warned = bool(blob.get("nonfinite_warned", False))
        self.S = self._applied_scale()
        self._set_group_lr(self.S)


class AutoLRMixin:
    """Attach continuous Mechanic AutoLR to a Kaon optimizer."""

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
        if self._autolr is not None:
            return self._autolr.step(closure)
        return self._step_impl(closure)

    def _step_impl(self, closure: Any = None) -> Any:
        raise NotImplementedError("optimizer using AutoLRMixin must provide _step_impl")

    def _autolr_reset_base_state(self) -> None:
        """Legacy host hook retained for external subclasses; Mechanic does not rollback."""
        self.state.clear()  # type: ignore[attr-defined]

    def get_d(self) -> float:
        if self._autolr is not None:
            return self._autolr.get_d()
        return float(self.param_groups[0]["lr"])  # type: ignore[attr-defined]

    def report_loss(self, loss: Any) -> None:
        if self._autolr is not None:
            self._autolr.report_loss(loss)

    def is_frozen(self) -> bool:
        return False

    def state_dict(self) -> dict[str, Any]:
        """Serialize both the host optimizer and autonomous controller state."""
        return self._autolr_state_dict(super().state_dict())  # type: ignore[misc]

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore both the host optimizer and autonomous controller state."""
        self._autolr_load(state_dict, super().load_state_dict)  # type: ignore[misc]

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
        after_load = getattr(self, "_autolr_after_load", None)
        if after_load is not None:
            after_load()
