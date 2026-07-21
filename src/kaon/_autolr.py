"""``AutoLRTuner`` — composable parameter-free learning rate via **update-space
DoWG** (Distance-over-Weighted-Gradients; Khaled, Mishchenko & Takáč, NeurIPS
2023). Attaches to any kaon base optimizer through the ``auto_lr=True`` flag and
discovers the LR with no per-model tuning.

Why DoWG and not coin-betting (the previous Mechanic tuner, now purged): on
kaon's normalize-then-momentum base the base update is ~unit-RMS (scale-invariant),
which makes coin-betting's travel-distance anchor degenerate — the scale ran away
to a hard cap and the "discovered" LR was just ``cap_rel·RMS(p)`` (measured LARS in
disguise; the cap was load-bearing). DoWG genuinely discovers: on a two-sided seed
sweep the effective LR converges *seed-independently* to the same value and matches
/ beats hand tuning, with no Adam-form coupling and far below Prodigy's per-param
state.

Mechanism (measured in the base's OWN update space, so it composes over any base
update form — normalize-then-momentum, Adam-form, sign, orthogonalized):

    each step the base applies ``S_t · u_t`` (``u_t`` = the base's unit-scale update);
    r̄_t = max_i ‖x_i − x_0‖                (running-max distance from the start point)
    v_t = v_{t-1} + r̄_t² · ‖u_t‖²          (distance-weighted accumulator)
    S_{t+1} = r̄_t² / √v_t                   (clamped to a LOOSE safety fuse)

The estimate is intrinsically bounded (no coin-betting wealth runaway). The fuse
(``fuse_rel × seed``, default 100×) is a safety rail, **not** load-bearing — DoWG
discovers well below it — its only job is to stop an absurd seed from blowing up.

Seed & drift. A data-relative seed (``3e-3 · RMS(p)``) starts the LR *below* the
operating scale by construction, i.e. in the sane "from-below" regime where
distance-ratio methods discover cleanly (from above they cannot recover — but the
data-relative seed never starts there). On a non-converging fine-tune the distance
from ``x_0`` keeps growing, so ``S`` drifts up slowly; ``auto_lr_freeze`` (an int
step count) locks the discovered value after the warmup. Robustification attempts
that avoid the freeze were measured and rejected: an EMA denominator *accelerates*
the runaway, a growth cap is insufficient, and a moving-reference window is
unstable (collapse/explode). The freeze is the honest, cheap answer.

State: one per-parameter reference buffer ``x_0`` (bf16, freed at freeze) plus a
few scalars. The base runs at ``lr = S`` throughout (incremental), so — unlike a
lr=1-then-rescale wrapper — **freeze needs no momentum fold**: at freeze the base
is already stepping at the frozen LR, so from then on ``step()`` is byte-for-byte
the plain base at ``lr = S``.

CUDA note: this env SIGFPEs on ``torch.dot`` for CUDA tensors, so inner products
are ``(a * b).sum()``.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import torch
from torch import Tensor

__all__ = ["AutoLRTuner"]

_SEED_REL: float = 3e-3   # data-relative seed = _SEED_REL * RMS(params); below the operating LR
_EPS: float = 1e-30


class AutoLRTuner:
    """Update-space DoWG LR tuner attached to a host optimizer ``opt``.

    ``opt`` must expose ``param_groups``, a per-param ``state`` mapping, and an
    ``_step_impl(closure=None)`` that performs one base update in place at the
    current ``group["lr"]``.
    """

    def __init__(
        self,
        opt: torch.optim.Optimizer,
        *,
        freeze: int | None,
        scale: float,
        fuse_rel: float,
    ) -> None:
        if freeze is not None and (not isinstance(freeze, int) or freeze < 1):
            raise ValueError(f"auto_lr_freeze must be None or an int >= 1, got {freeze!r}")
        if not scale > 0.0:
            raise ValueError(f"auto_lr_scale must be > 0, got {scale}")
        if not fuse_rel > 0.0:
            raise ValueError(f"auto_lr_fuse_rel must be > 0, got {fuse_rel}")
        self.opt = opt

        # The tuner owns the LR. A leftover/prefilled base lr < 1 (e.g. renga-flow's
        # 1e-4) is an accident — ignore it with a one-time warning.
        small = [float(g["lr"]) for g in opt.param_groups if g["lr"] < 1.0]
        if small:
            warnings.warn(
                f"auto_lr=True discovers the learning rate itself; the base lr "
                f"({small[0]:g}) is ignored. Use auto_lr_scale for an explicit "
                f"multiplier, or set lr=1.0 to silence this warning.",
                stacklevel=3,
            )

        self._freeze_at = freeze
        self._scale = float(scale)
        self._fuse_rel = float(fuse_rel)

        self.S: float | None = None      # effective LR; seeded on the first step
        self._fuse: float | None = None  # absolute fuse cap; = fuse_rel * seed
        self._v = _EPS                   # DoWG distance-weighted accumulator
        self._rbar = 0.0                 # running-max distance from x0
        self._t = 0                      # adapting-step counter
        self._x0: dict[Tensor, Tensor] = {}  # per-param reference (freed at freeze)

        self.frozen = False
        self.frozen_lr: float | None = None

    # -- introspection -----------------------------------------------------
    def get_d(self) -> float:
        """Discovered effective LR (or the frozen LR)."""
        if self.frozen:
            return float(self.frozen_lr)  # type: ignore[arg-type]
        return float(self.S) if self.S is not None else 0.0

    # -- step --------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        """One adapting step. Assumes not frozen (the host routes frozen -> _step_impl)."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        params = [p for g in self.opt.param_groups for p in g["params"] if p.grad is not None]
        if not params:
            return loss

        # First step: data-relative seed (below the operating LR) + fuse + x0 refs.
        if self.S is None:
            sq = 0.0
            cnt = 0
            for p in params:
                pf = p.detach().float()
                sq = sq + float((pf * pf).sum())
                cnt += pf.numel()
            rms = (sq / max(cnt, 1)) ** 0.5 if cnt else 0.0
            self.S = max(_SEED_REL * rms, 1e-8)
            self._fuse = self._fuse_rel * self.S
            for p in params:
                self._x0[p] = p.detach().to(torch.bfloat16).clone()

        s_prev = float(self.S)
        # Incremental step: the base applies S·u at group["lr"]=S. Forced each step
        # so an external harness rewriting group["lr"] can't clobber the discovered LR.
        for g in self.opt.param_groups:
            g["lr"] = s_prev
        prev = [p.detach().clone() for p in params]
        self.opt._step_impl()  # type: ignore[attr-defined]

        # ‖u_t‖² (unit-update norm²) from the applied displacement, and the
        # running-max distance from x0. Both are plain reductions over the params.
        un2 = 0.0
        dist2 = 0.0
        for p, pv in zip(params, prev, strict=True):
            d = (p.detach() - pv).float()
            un2 += float((d * d).sum())
            diff = p.detach().float() - self._x0[p].float()
            dist2 += float((diff * diff).sum())
        un2 = un2 / (s_prev * s_prev + _EPS)  # divide out S² -> unit-update norm²

        dist = math.sqrt(dist2)
        if dist > self._rbar:
            self._rbar = dist
        r2 = self._rbar * self._rbar
        self._v += r2 * un2
        new_s = (r2 / math.sqrt(self._v)) * self._scale
        if self._fuse is not None and new_s > self._fuse:  # loose safety fuse (not load-bearing)
            new_s = self._fuse
        self.S = max(new_s, 1e-12)

        self._t += 1
        if self._freeze_at is not None and self._t >= self._freeze_at:
            self._do_freeze()
        return loss

    # -- freeze ------------------------------------------------------------
    def _do_freeze(self) -> None:
        """Lock the discovered LR and free the reference buffers.

        The base already stepped at ``lr = S`` every step (incremental), so no
        momentum fold is needed — post-freeze the base simply keeps running at the
        frozen ``S`` and is byte-for-byte the plain base from here on.
        """
        self.frozen_lr = float(self.S) if self.S is not None else 0.0
        for g in self.opt.param_groups:
            g["lr"] = self.frozen_lr
        self._x0.clear()  # free the per-param reference buffers
        self.frozen = True

    # -- checkpointing (blob merged into the host's state_dict) -------------
    def state_blob(self) -> dict[str, Any]:
        """Serializable tuner state. ``x0`` refs are stored in stable param order."""
        params = [p for g in self.opt.param_groups for p in g["params"]]
        index = {p: i for i, p in enumerate(params)}
        x0_list: list[Tensor | None] = [None] * len(params)
        for p, r in self._x0.items():
            x0_list[index[p]] = r.detach().clone()
        return {
            "S": self.S,
            "fuse": self._fuse,
            "v": self._v,
            "rbar": self._rbar,
            "t": self._t,
            "x0": x0_list,
            "frozen": self.frozen,
            "frozen_lr": self.frozen_lr,
        }

    def load_blob(self, blob: dict[str, Any]) -> None:
        """Restore tuner state produced by :meth:`state_blob`."""
        params = [p for g in self.opt.param_groups for p in g["params"]]
        self.S = blob["S"]
        self._fuse = blob["fuse"]
        self._v = float(blob["v"])
        self._rbar = float(blob["rbar"])
        self._t = int(blob["t"])
        self._x0 = {}
        for i, r in enumerate(blob["x0"]):
            if r is not None:
                self._x0[params[i]] = r.to(params[i].device)
        self.frozen = bool(blob["frozen"])
        self.frozen_lr = blob["frozen_lr"]
