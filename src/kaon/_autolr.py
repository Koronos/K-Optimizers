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
from ``x_0`` keeps growing, so ``S`` drifts up slowly, and freezing is *necessary*
(never-freezing was measured to let the drift diverge on some seeds). ``auto_lr_freeze``
locks the discovered value: ``"auto"`` (default) freezes once ``S`` has grown
``_FREEZE_GROWTH×`` over its seed — a dimensionless, reparametrization-invariant trigger
(an absolute step count breaks when batch/step-rate/run-length change; the held-out-loss
plateau is broad so the exact ratio is non-critical, verified consistent across
seeds/model-size/momentum); an ``int`` freezes after N steps; ``None`` never freezes.
Robustification attempts that avoid the freeze entirely were measured and rejected: an
EMA denominator *accelerates* the runaway, a growth cap is insufficient, a moving-reference
window is unstable (collapse/explode), and no purely-internal constant-free STOP exists
(``log S ∝ log t`` is scale-free), but the freeze POINT sits on a broad loss plateau so a
dimensionless ratio trigger is enough.

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

__all__ = ["AutoLRMixin", "AutoLRTuner"]

_SEED_REL: float = 3e-3   # data-relative seed = _SEED_REL * RMS(params); below the operating LR
_EPS: float = 1e-30
# auto_lr_freeze="auto": freeze once the discovered LR has grown this many x over its
# data-relative seed — an "order of magnitude of LR growth from a below-optimum seed"
# brings you into the operating band. This is a DIMENSIONLESS, reparametrization-invariant
# trigger (unlike an absolute step count, which breaks when batch size / step rate / the
# training-length change). Non-critical: the held-out-loss plateau is broad, so any ratio in
# ~[6, 22] lands in it (measured); 10 sits mid-plateau. Verified consistent across seeds /
# model size / momentum, and SAFER than never-freezing (which can let the drift diverge).
_FREEZE_GROWTH: float = 10.0


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
        freeze: int | str | None,
        scale: float,
        fuse_rel: float,
    ) -> None:
        if freeze != "auto" and freeze is not None and (not isinstance(freeze, int) or freeze < 1):
            raise ValueError(f"auto_lr_freeze must be 'auto', None, or an int >= 1, got {freeze!r}")
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
        self._seed: float | None = None  # the data-relative seed (for the "auto" growth-ratio freeze)
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
            self._seed = self.S
            self._fuse = self._fuse_rel * self.S
            for p in params:
                self._x0[p] = p.detach().to(torch.bfloat16).clone()

        s_prev = float(self.S)
        # Incremental step: the base applies S·u at group["lr"]=S. Forced each step
        # so an external harness rewriting group["lr"] can't clobber the discovered LR.
        for g in self.opt.param_groups:
            g["lr"] = s_prev
        # fp32 pre-step snapshot. fp32 avoids bf16 catastrophic cancellation on the
        # small step / early displacement. ``.float()`` gives a safe fp32 copy for a
        # low-precision param; an fp32 param needs ``.clone()`` (``.float()`` would
        # alias the param the base is about to mutate).
        prev_f = [
            p.detach().float() if p.dtype != torch.float32 else p.detach().clone()
            for p in params
        ]
        self.opt._step_impl()  # type: ignore[attr-defined]

        # ‖u_t‖² (unit-update norm²) and dist² = ‖x−x0‖², batched with
        # torch._foreach_* so the cost is a handful of kernel launches + two syncs,
        # not ~8 launches + 2 syncs *per tensor* (the launch-bound blowup on
        # many-small-tensor LoRA/LoKr fleets). ``.float()`` is a no-op on fp32 params.
        cur_f = [p.detach().float() for p in params]
        x0_f = [self._x0[p].float() for p in params]
        du = torch._foreach_sub(cur_f, prev_f)          # Δp
        ddist = torch._foreach_sub(cur_f, x0_f)         # x − x0
        un2 = float(torch.stack(torch._foreach_norm(du)).square_().sum())
        dist2 = float(torch.stack(torch._foreach_norm(ddist)).square_().sum())
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
        if self._freeze_at == "auto":
            # Dimensionless growth-ratio trigger: freeze once the LR has grown _FREEZE_GROWTH×
            # over its (below-optimum) seed — reparametrization-invariant, no absolute step count.
            if self.S >= _FREEZE_GROWTH * self._seed:  # type: ignore[operator]
                self._do_freeze()
        elif self._freeze_at is not None:  # int step count
            if self._t >= self._freeze_at:
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
            "seed": self._seed,
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
        self._seed = blob.get("seed")
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


class AutoLRMixin:
    """Turns an ``auto_lr`` flag into the composable DoWG tuner, so each base optimizer only
    hooks up in ~4 lines. Recipe: inherit this FIRST (``class Foo(AutoLRMixin, Optimizer)``),
    call ``self._init_autolr(...)`` at the end of ``__init__``, expose the base's one-step
    update as ``_step_impl(closure=None)`` (rename the old ``step`` for a flat optimizer, or
    delegate to the wrapped ``step`` for a wrapper), and route ``state_dict`` / ``load_state_dict``
    through the two helpers here. The step router (with the freeze-time harness-clobber guard),
    ``get_d`` / ``is_frozen``, and the checkpoint plumbing are shared here — no per-optimizer copy.
    """

    _autolr: "AutoLRTuner | None"

    def _init_autolr(self, auto_lr: bool, freeze: int | str | None, scale: float, fuse_rel: float) -> None:
        """Call at the END of ``__init__`` (after ``super().__init__`` / param_groups exist)."""
        self._autolr = (
            AutoLRTuner(self, freeze=freeze, scale=scale, fuse_rel=fuse_rel)  # type: ignore[arg-type]
            if auto_lr
            else None
        )

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        # Adapting -> the DoWG tuner drives the step at the discovered lr=S. Frozen -> plain base
        # at S, imposed EACH step so an external harness rewriting group["lr"] (renga/kohya
        # schedulers, the control battery) can't clobber it. Off -> step == _step_impl (zero overhead).
        t = self._autolr
        if t is not None:
            if not t.frozen:
                return t.step(closure)
            for group in self.param_groups:  # type: ignore[attr-defined]
                group["lr"] = t.frozen_lr
        return self._step_impl(closure)

    def _step_impl(self, closure: Any = None) -> Any:
        raise NotImplementedError("optimizer using AutoLRMixin must provide _step_impl")

    def get_d(self) -> float:
        """Discovered effective LR under ``auto_lr`` (else the plain group lr)."""
        if self._autolr is not None:
            return self._autolr.get_d()
        return float(self.param_groups[0]["lr"])  # type: ignore[attr-defined]

    def is_frozen(self) -> bool:
        """Whether ``auto_lr`` has frozen to a fixed LR (False when auto_lr is off)."""
        return self._autolr is not None and self._autolr.frozen

    # -- checkpoint helpers (the host's state_dict/load_state_dict call these) --
    def _autolr_state_dict(self, sd: dict[str, Any]) -> dict[str, Any]:
        """``def state_dict(self): return self._autolr_state_dict(super().state_dict())``"""
        if self._autolr is not None:
            sd["_autolr"] = self._autolr.state_blob()
        return sd

    def _autolr_load(self, state_dict: dict[str, Any], inner_load: Any) -> None:
        """``def load_state_dict(self, sd): self._autolr_load(sd, lambda s: <base restore>(s))``"""
        sd = dict(state_dict)
        blob = sd.pop("_autolr", None)
        inner_load(sd)
        if self._autolr is not None and blob is not None:
            self._autolr.load_blob(blob)
