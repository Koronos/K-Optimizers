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
(``fuse_rel × 3e-3·RMS(p)``, default ceiling ``0.06·RMS(p)``) is a LOOSE prior
ceiling with an honest, measured role: in **sharp-edge** regimes (real adapter
fine-tunes à la Anima — instability spikes) the edge guard governs far below it,
and when the problem's true scale is lower still, from-below discovery governs;
but in **mushy-edge** regimes (smooth proxy-like landscapes where over-LR degrades
loss WITHOUT grad spikes) DoWG-with-momentum's own fixed point overshoots the loss
optimum, no internal signal marks it (measured: grad-norm level is flat in LR;
instantaneous-cosine and cumulative-coherence corrections both fail), and the
ceiling is what stops the slow drift — there, it IS load-bearing by design
(``fuse_rel=20`` measured best on both proxy configs; the old 100 let momentum
drift to ``0.3·RMS``, te +0.011). Caveat, honestly flagged: a mushy-edge problem
whose optimum sits far below ``0.06·RMS`` would over-shoot — no such case measured
yet; the sharp/low cases that exist are covered by the guard/discovery.

Seed. DoWG can only discover **from below** (from above the running-max distance
ratchets the estimate *up* while the training destabilizes — a positive feedback it
cannot exit). The seed therefore sits DECADES below any plausible operating LR:
``1e-6 · RMS(p)``. This is safe by construction (the previous ``3e-3·RMS`` "operating
scale minus a bit" seed landed 7.6× ABOVE the needed LR on a real Anima LoRA — weight
RMS carries no information about an adapter's stability edge, which is set by the
frozen base network and the adapter parametrization, not by the adapter's init scale).
The climb is a **geometric ramp floor** (``S ≥ S_prev × _RAMP_GROWTH`` per update step,
≥1 decade per ~24 steps) active until the FIRST edge contact: bare DoWG's climb is
diffusive under real noisy gradients (measured ~250 steps/decade on a real Anima LoRA
— useless as a warmup), so the floor supplies the pace and the edge guard supplies the
stop — an online LR-range test, and a free warmup. ``auto_lr_d0`` optionally replaces
the data-relative seed with an explicit starting LR; with the guard either direction
is safe (too high → backoff walks it down; too low → the ramp climbs it fast).

Stability-edge guard & freeze. A grad-norm EMA watches for instability: a spike
(``> _SPIKE_RATIO ×`` the EMA) means the current ``S`` touched the model's stability
edge → back off (``S × _BACKOFF``), re-anchor the DoWG accumulators (``x0 := x``,
``r̄ = 0``, ``v = ε``; the restart is self-consistent — the first post-anchor estimate
reproduces the backed-off ``S``), and remember the contact level. The step itself is
NOT skipped: the base update is normalized (bounded magnitude), so stepping at the
backed-off ``S`` is safe and lets the params recover — skipping could deadlock
(damaged params → high grads → eternal spike, no progress). The EMA resets to the
spiked level (hysteresis: an elevated aftermath doesn't re-trigger in a chain).

The freeze is **automatic and internal — there is no knob**: it fires on the
**second contact within ``_EDGE_BAND×`` of the first** — a confirmed edge → lock at
``edge × _BACKOFF``, just below the edge, the safe side of the measured asymmetry
(overshoot is a cliff, undershoot is mild). An isolated spike (bad batch) never
freezes: without a repeat contact ``S`` simply resumes climbing — the false positive
self-corrects. This is an online LR-range-test, and it decouples the freeze from the
seed entirely (the previous user-facing ``auto_lr_freeze`` growth-ratio trigger
silently required the seed to land within one decade below the optimum — no
data-relative rule achieves that across architectures — and contradicted the whole
point of *automatic* discovery: the system now decides by itself when discovery is
done). On a non-converging fine-tune the distance from ``x_0`` keeps growing, so
``S`` drifts up slowly; the drift eventually meets the edge and the guard converts
it into a freeze — never-freezing without the guard was measured to diverge on some
seeds. Freezing (rather than adapting forever) also frees the reference buffers.

State: one per-parameter reference buffer ``x_0`` (the param's own dtype — an exact
snapshot; a bf16 copy of fp32 params was measured to inject a quantization-noise
floor of ``~4e-3·RMS(p)`` into the first distance estimate, silently catapulting the
decades-low seed straight back into the danger zone — freed at freeze) plus a
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

_SEED_REL: float = 1e-6   # data-relative seed = _SEED_REL * RMS(params); DECADES below any operating LR
_FUSE_REF_REL: float = 3e-3  # fuse anchor = fuse_rel * _FUSE_REF_REL * RMS(params) (the typical operating scale)
_EPS: float = 1e-30
# Stability-edge guard: a grad-norm spike > _SPIKE_RATIO × its EMA marks contact with the
# model's stability edge (healthy batch-to-batch variation is <2×; real instability is 10–50×).
_SPIKE_RATIO: float = 5.0
_EMA_BETA: float = 0.9     # grad-norm EMA horizon ~10 steps
_EMA_WARMUP: int = 3       # steps of EMA seeding before the guard arms
_EDGE_BAND: float = 2.0    # a repeat contact within this factor confirms the edge -> freeze
_BACKOFF: float = 0.5      # S multiplier on each edge contact; the freeze locks at edge*_BACKOFF
# Geometric ramp floor, active until the FIRST edge contact: S >= S_prev * _RAMP_GROWTH each
# real update step (>=1 decade per ~24 steps). Real (noisy) gradients make the bare DoWG climb
# DIFFUSIVE — measured ~250 steps/decade on a real Anima LoRA, useless as a warmup — while the
# coherent-regime simulation promised ~4. The floor restores the fast climb; it is safe because
# the edge guard is the stop (this is an online LR-range test: grow exponentially until the
# model protests, back off, hand over to DoWG). Never re-enabled after a contact.
_RAMP_GROWTH: float = 1.1


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
        scale: float,
        fuse_rel: float,
        d0: float | None = None,
    ) -> None:
        if not scale > 0.0:
            raise ValueError(f"auto_lr_scale must be > 0, got {scale}")
        if not fuse_rel > 0.0:
            raise ValueError(f"auto_lr_fuse_rel must be > 0, got {fuse_rel}")
        if d0 is not None and not d0 > 0.0:
            raise ValueError(f"auto_lr_d0 must be > 0 (or None for the data-relative seed), got {d0}")
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

        self._scale = float(scale)
        self._fuse_rel = float(fuse_rel)
        self._d0 = float(d0) if d0 is not None else None

        self.S: float | None = None      # effective LR; seeded on the first step
        self._ramp_on = True             # geometric ramp floor; off at the first edge contact
        self._seed: float | None = None  # the data-relative seed
        self._fuse: float | None = None  # absolute fuse cap; = fuse_rel * _FUSE_REF_REL * RMS(p)
        self._v = _EPS                   # DoWG distance-weighted accumulator
        self._rbar = 0.0                 # running-max distance from x0
        self._t = 0                      # adapting-step counter
        self._x0: dict[Tensor, Tensor] = {}  # per-param reference (freed at freeze)
        self._gema: float | None = None  # grad-norm EMA (the stability-edge signal)
        self._edge: float | None = None  # S at the last edge contact (None = no contact yet)
        self._nan_run = 0                # consecutive non-finite-grad steps (bounds the skip path)

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

        # First step: decades-low data-relative seed + fuse + x0 refs.
        if self.S is None:
            sq = 0.0
            cnt = 0
            for p in params:
                pf = p.detach().float()
                sq = sq + float((pf * pf).sum())
                cnt += pf.numel()
            rms = (sq / max(cnt, 1)) ** 0.5 if cnt else 0.0
            # Explicit d0 wins over the data-relative seed. With the edge guard either
            # direction is safe: too high -> backoff walks it down; too low -> the ramp
            # floor climbs it fast.
            self.S = self._d0 if self._d0 is not None else max(_SEED_REL * rms, 1e-10)
            self._seed = self.S
            self._fuse = max(self._fuse_rel * _FUSE_REF_REL * rms, self._fuse_rel * self.S)
            # x0 in the param's OWN dtype: an exact snapshot. (bf16-compressing fp32
            # params injects a ~4e-3·RMS(p) quantization-noise floor into the first
            # distance estimate — measured to clobber the decades-low seed.)
            for p in params:
                self._x0[p] = p.detach().clone()

        # Stability-edge guard: grad-norm spike vs its EMA.
        gnorms = torch._foreach_norm([p.grad for p in params])
        gn = float(torch.stack([n.float() for n in gnorms]).square_().sum()) ** 0.5
        if not math.isfinite(gn):
            # inf/nan gradient: the grads are poison — never step (stepping would spread
            # the poison into the base's own state). Only the FIRST of a consecutive run
            # counts as an edge contact (backoff/rollback); repeats skip without shrinking
            # S further — if the grads are non-finite regardless of LR, the LR is not the
            # problem, and melting S to 1e-12 would just stall training silently.
            self._nan_run += 1
            if self._nan_run == 1:
                self._edge_contact(params, step_after=False)
            elif self._nan_run == 3:
                warnings.warn(
                    "auto_lr: gradients are non-finite for 3+ consecutive steps; the LR "
                    "has already been backed off, so this is not an LR problem — check "
                    "the data/loss/precision. Steps are skipped while grads stay non-finite.",
                    stacklevel=2,
                )
            self._t += 1
            return loss
        self._nan_run = 0
        if self._gema is None or self._gema <= 0.0:
            self._gema = gn
        elif self._t >= _EMA_WARMUP and gn > _SPIKE_RATIO * self._gema:
            if self._ramp_on:
                # Rollback path: params are restored to x0, so the spiked gradient is
                # stale (computed at a state that no longer exists) — skip the step and
                # keep the EMA (the healthy pre-spike baseline is still the right one).
                self._edge_contact(params, step_after=False)
                self._t += 1
                return loss
            frozen_now = self._edge_contact(params, step_after=True)
            self._gema = gn  # hysteresis: the elevated aftermath must not re-trigger in a chain
            self._t += 1
            if frozen_now:
                return loss
            # fall through: step at the backed-off S with freshly re-anchored accumulators
        else:
            self._gema = _EMA_BETA * self._gema + (1.0 - _EMA_BETA) * gn

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

        if un2 <= 0.0:
            # The base produced NO update this step (e.g. ADOPT's v-lag warmup no-op, which
            # only initializes the second moment). Feeding un2=0 into the ratio leaves v ~ 0,
            # so S = r̄²/√v would blow up to the fuse (and r̄ would pick up the bf16 x0
            # quantization error as spurious "distance"). Skip the estimate — keep S at its
            # seed until the base actually moves.
            self._t += 1
            return loss

        dist = math.sqrt(dist2)
        if dist > self._rbar:
            self._rbar = dist
        r2 = self._rbar * self._rbar
        self._v += r2 * un2
        new_s = (r2 / math.sqrt(self._v)) * self._scale
        if self._ramp_on:
            # Geometric ramp floor until the first edge contact (see _RAMP_GROWTH).
            new_s = max(new_s, s_prev * _RAMP_GROWTH)
        if self._fuse is not None and new_s > self._fuse:
            new_s = self._fuse
        self.S = max(new_s, 1e-12)

        self._t += 1
        return loss

    # -- stability-edge contact --------------------------------------------
    def _edge_contact(self, params: list[Tensor], *, step_after: bool) -> bool:
        """Back off from a stability-edge contact; freeze if the edge is confirmed.

        Returns True if this contact froze the LR. A repeat contact within
        ``_EDGE_BAND×`` of the recorded one confirms the edge → lock at
        ``edge × _BACKOFF``, the safe side of the overshoot cliff. A first
        or far-away contact records the level, backs off, and re-anchors the DoWG
        accumulators — the restart is self-consistent (the first post-anchor
        estimate reproduces the backed-off ``S``), so ``S`` resumes climbing and an
        isolated bad-batch spike self-corrects instead of freezing.
        """
        contact = float(self.S)  # type: ignore[arg-type]
        if self._ramp_on:
            # Range-test rollback: all pre-contact progress happened at sub-operating
            # LRs (negligible), and x0 IS the exact pre-ramp snapshot — restore it so
            # the overshoot leaves NO trace on the params. The base's own EMAs are not
            # rolled back (composability boundary); the spike gradient is never applied
            # and the contaminated EMA content decays within tens of steps.
            self._ramp_on = False
            self._edge = contact
            self.S = max(contact * _BACKOFF, 1e-12)
            for p, ref in self._x0.items():
                p.data.copy_(ref)
            self._rbar = 0.0
            self._v = _EPS
            return False
        if self._edge is not None and contact <= _EDGE_BAND * self._edge:
            self.S = max(contact * _BACKOFF, 1e-12)
            self._do_freeze()
            if step_after:
                self.opt._step_impl()  # type: ignore[attr-defined]
            return True
        self._edge = contact
        self.S = max(contact * _BACKOFF, 1e-12)
        for p in params:
            self._x0[p] = p.detach().clone()
        self._rbar = 0.0
        self._v = _EPS
        return False

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
            "gema": self._gema,
            "edge": self._edge,
            "ramp_on": self._ramp_on,
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
        self._gema = blob.get("gema")
        self._edge = blob.get("edge")
        self._ramp_on = bool(blob.get("ramp_on", blob.get("edge") is None))
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

    def _init_autolr(self, auto_lr: bool, scale: float, fuse_rel: float, d0: float | None = None) -> None:
        """Call at the END of ``__init__`` (after ``super().__init__`` / param_groups exist)."""
        self._autolr = (
            AutoLRTuner(self, scale=scale, fuse_rel=fuse_rel, d0=d0)  # type: ignore[arg-type]
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
