"""``AutoLRTuner`` — composable parameter-free learning rate. Attaches to any kaon
base optimizer through the ``auto_lr=True`` flag and discovers the LR with no
per-model tuning.

**Two modes, chosen automatically by what the tuner can see** (decided at the
first step, per an /experts-meeting synthesis — 4 independent experts + web):

1. **Loss-driven range test (primary; active whenever a loss is visible** — via a
   step closure, or the trainer calling ``optimizer.report_loss(loss)`` each step,
   a one-line trainer change). The panel's unanimous finding: every gradient-only
   signal family is structurally insufficient — travel-distance estimators
   (DoWG/Prodigy/D-adaptation/Mechanic) are one-directional ratchets that cannot
   come down from above, and gradient-MAGNITUDE guards are information-theoretically
   blind on mushy landscapes (measured: grad-norm level flat across a full LR decade
   on the proxy) and get "boiled-frog"-ed by gradual ramps on sharp ones (measured on
   a real Anima run: norms 0.5→27 over 30 steps, zero spike triggers, model
   destroyed). Loss is the one cheap, model-independent "did this step help" signal
   — degradation at the edge is a 10-100σ event vs batch noise.
   Mechanism (fastai lr_finder pattern, inside the optimizer, non-destructive,
   ADAPTIVE coarse-to-fine): an exponential LR ladder of independent windows —
   each restores the exact ``x0`` snapshot AND clears the base optimizer state
   (a clean trial), runs real steps at the window LR, and is judged by its MEDIAN
   batch loss against a POOLED baseline with an SE-of-the-median threshold
   (statistically scaled: longer windows are automatically finer detectors, and
   only provably-clean windows join the pool — pooling everything that merely
   passes was measured to creep the baseline upward and erode the detector).
   Three phases:
   - CLIMB: coarse half-decade rungs (8 steps each); when a rung's grad-norm
     median heats past 2× the pooled gn baseline, the growth factor refines
     (3.16→1.78→1.33) — a PACE signal only (a wrong "warm" costs extra rungs,
     never correctness; the loss judgments stay authoritative). Non-finite
     loss/grads or a single 3×-baseline loss fail a window instantly.
   - BISECT: after the first fail, log-bisection of the [pass, fail] bracket
     down to ~1.33× — binary search on a monotone property, optimal per step.
   - CONFIRM: a SEQUENTIAL long test (SPRT-style) at the candidate — keeps
     sampling until the evidence decides. Fails the moment the running median
     crosses the SE-scaled threshold for the current n (bad candidates die in
     ~15-80 steps); passes only by surviving ``_PROBE_CONFIRM_CAP`` samples
     (~+19% sensitivity at Anima-like batch noise, MAD ~25% of the median —
     a fixed 32-step window was measured to pass a burning LR by a hair: telling
     +20-25% slow burn from that noise takes ~100 samples, which is statistics,
     not tuning). On failure it steps down and re-confirms (fine steps after a
     tight bracket; coarse after a ceiling top-out) — bounded tries.
   Then params restore to ``x0`` once more, base state clears, and training
   starts byte-clean at the confirmed LR. Cost ~200-400 steps typical (bad
   candidates cheap, the winner pays the full cap once); the d0-above-the-edge
   case descends first (bidirectional). For quality margins BELOW what any
   loss statistic can detect (human-preview-level burn), ``auto_lr_scale``
   (e.g. 0.5) applies a global post-discovery multiplier — a preference, not
   per-model calibration.

2. **Continuous update-space DoWG + stability-edge guard (fallback; no loss
   visible — e.g. kohya)** — the pre-existing mechanism, unchanged, documented
   below. Known limits (measured, accepted for the fallback): discovery-from-below
   only, magnitude guard blind on mushy edges, fuse ceiling load-bearing there.

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

__all__ = ["AutoLRMixin", "AutoLRTuner", "DEFAULT_FUSE_REL"]

# Default for the hosts' ``auto_lr_fuse_rel`` kwarg (single source of truth — the
# measured recalibrations of this ceiling must not be an 8-file edit).
DEFAULT_FUSE_REL: float = 20.0

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

# -- loss-driven range-test (probe) constants --------------------------------------
# Adaptive coarse-to-fine ladder: coarse growth far from the edge, refined near it
# (warmth-paced), log-bisection of the [pass, fail] bracket, then a LONG confirmation
# window at the candidate that catches slow degradation an 8-step rung cannot see.
_PROBE_STEPS: int = 8            # measured losses per short rung
_PROBE_CONFIRM_MIN: int = 16     # confirmation: samples before the sequential fail test arms
_PROBE_CONFIRM_CAP: int = 96     # confirmation: samples to PASS (sensitivity ~ +19% at Anima-like noise)
_PROBE_FACTOR: float = 10.0 ** 0.5    # coarse ladder spacing (half-decade)
_PROBE_FACTOR_MIN: float = 10.0 ** 0.125  # finest grid / bracket resolution (~1.33x)
_PROBE_MARGIN: float = 0.25      # short-rung relative fail margin (fast, coarse test)
_PROBE_CONFIRM_MARGIN: float = 0.10  # confirmation relative fail margin (long, fine test)
_PROBE_NSIG: float = 4.0         # z on the standard error of the window median
_PROBE_ABORT: float = 3.0        # any single loss > this × baseline median aborts the window
_PROBE_WARM: float = 2.0         # rung grad-norm median > this × baseline gn median = "warm"
_PROBE_CONFIRM_MAX: int = 8      # bounded step-downs during confirmation


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def _probe_thresh(base: list[float], margin: float, n: int) -> float:
    """Fail threshold for a window median of ``n`` samples vs the pooled baseline.

    Statistically scaled, not calibrated: the noise term is the standard error of
    the median (1.2533·σ/√n with σ from the MAD), so LONGER windows are
    automatically FINER detectors, and the pooled baseline tightens the estimate
    as the probe runs. The relative ``margin`` is a floor against hyper-sensitivity
    once the SE becomes tiny.
    """
    med = _median(base)
    mad = _median([abs(v - med) for v in base]) or 0.05 * abs(med)
    se = 1.2533 * 1.4826 * mad / math.sqrt(max(n, 1))
    return med + max(margin * abs(med), _PROBE_NSIG * se)


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
        self._seed: float | None = None  # the data-relative seed
        self._fuse: float | None = None  # absolute fuse cap; = fuse_rel * _FUSE_REF_REL * RMS(p)
        self._v = _EPS                   # DoWG distance-weighted accumulator
        self._rbar = 0.0                 # running-max distance from x0
        self._t = 0                      # adapting-step counter
        self._x0: dict[Tensor, Tensor] = {}  # per-param reference (freed at freeze)
        self._gema: float | None = None  # grad-norm EMA (the stability-edge signal)
        self._edge: float | None = None  # S at the last edge contact (None = no contact yet)
        self._nan_run = 0                # consecutive non-finite-grad steps (bounds the skip path)

        # Loss-driven range test. None = mode undecided (first step decides: a visible
        # loss -> probe; none -> continuous fallback). dict = probe running. False = off.
        self._probe: dict[str, Any] | bool | None = None
        self._pending_loss: float | None = None

        self.frozen = False
        self.frozen_lr: float | None = None

    # -- loss channel -------------------------------------------------------
    def report_loss(self, loss: Any) -> None:
        """Give the tuner this step's training loss (float or 0-dim tensor).

        Call once per step BEFORE ``optimizer.step()``. Enables the loss-driven
        range test (the primary discovery mode); without it the tuner falls back
        to the continuous gradient-only estimator.
        """
        if loss is None:
            return
        v = float(loss.detach()) if isinstance(loss, Tensor) else float(loss)
        self._pending_loss = v

    # -- introspection -----------------------------------------------------
    @property
    def _ramp_on(self) -> bool:
        """Ramp (range-test) phase = no edge contact yet. Derived, never stored."""
        return self._edge is None

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
            if loss is not None:
                self.report_loss(loss)  # a closure loss enables the probe like report_loss does

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

        # Mode decision: a visible loss -> loss-driven range test (primary); none within
        # the first few steps -> continuous gradient-only fallback. The grace window
        # exists because trainers that report the PREVIOUS step's loss (e.g. DeepSpeed
        # train_batch, where the loss is only known after the internal step) have
        # nothing to report at step 0.
        if self._probe is None:
            if self._pending_loss is not None:
                self._probe = {"phase": "climb", "lr": float(self.S),
                               "factor": _PROBE_FACTOR, "losses": [], "gns": [],
                               "base": [], "gbase": [], "passed": None, "fail": None,
                               "descend": False, "tries": 0,
                               "restore": False, "skip1": False}
            elif self._t >= 3:
                self._probe = False
        if isinstance(self._probe, dict):
            return self._probe_step(loss)
        if self._probe is False:
            self._pending_loss = None  # continuous mode ignores late-arriving losses

        # Stability-edge guard: grad-norm spike vs its EMA.
        gnorms = torch._foreach_norm([p.grad for p in params])
        if len({n.dtype for n in gnorms}) > 1:  # rare mixed-dtype fleet; stack needs one dtype
            gnorms = [n.float() for n in gnorms]
        gn = float(torch.stack(gnorms).float().square_().sum()) ** 0.5
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
                # (step_after is moot here: a ramp contact can never be a freeze.)
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

    # -- loss-driven range test (probe mode) --------------------------------
    def _probe_step(self, loss: Any) -> Any:
        """One step of the adaptive ladder. Phases: climb → bisect → confirm.

        Loss timing: the loss reported at step t was computed BEFORE this step's
        update (forward → backward → step), so after a restore the very next
        reported loss already reflects the restored params — no staleness. The
        one skipped update per rung is the judgment step itself (its gradient
        was computed at the params being discarded).
        """
        pr = self._probe
        lv = self._pending_loss
        self._pending_loss = None
        if pr["restore"]:  # only set on mid-rung checkpoint resume: redo the rung cleanly
            self._probe_restore()
            pr["restore"] = False
            pr["losses"] = []
            pr["gns"] = []
            pr["skip1"] = True
            self._t += 1
            return loss

        if lv is not None and pr["skip1"]:
            # First loss after a restore is dropped: under a report-after-step trainer
            # (DeepSpeed train_batch) it was computed at the DISCARDED rung's params
            # (one-step lag); under a closure it is merely the untouched-x0 loss.
            pr["skip1"] = False
            lv = None

        baselined = len(pr["base"]) >= 2 * _PROBE_STEPS
        bmed = _median(pr["base"]) if baselined else None
        finite = lv is not None and math.isfinite(lv)
        fail = lv is not None and not finite  # non-finite loss: instant window failure
        if finite:
            pr["losses"].append(lv)
            if bmed is not None and lv > _PROBE_ABORT * bmed:
                fail = True  # catastrophic mid-window; no need to finish it
            elif (pr["phase"] == "confirm" and baselined
                  and len(pr["losses"]) >= _PROBE_CONFIRM_MIN):
                # SEQUENTIAL confirmation (SPRT-style): to tell a +20-25% slow burn
                # from Anima-like batch noise (MAD ~25% of the median) takes ~100
                # samples, not 32 — a fixed short window passes burning LRs by a
                # hair (measured on a real run). So keep sampling until the evidence
                # decides: fail the moment the running median crosses the threshold
                # for the CURRENT n (bad candidates die in ~15-80 steps); pass only
                # by surviving the full cap. The 4-sigma z keeps the repeated
                # peeking honest.
                fail = _median(pr["losses"]) > _probe_thresh(
                    pr["base"], _PROBE_CONFIRM_MARGIN, len(pr["losses"])
                )
        target = _PROBE_CONFIRM_CAP if pr["phase"] == "confirm" else _PROBE_STEPS
        if not fail and len(pr["losses"]) < target:
            # mid-window: a real update at the window LR (+ grad-norm sample for warmth)
            gnorms = torch._foreach_norm([p.grad for g_ in self.opt.param_groups
                                          for p in g_["params"] if p.grad is not None])
            if len({n.dtype for n in gnorms}) > 1:
                gnorms = [n.float() for n in gnorms]
            gn = float(torch.stack(gnorms).float().square_().sum()) ** 0.5
            if math.isfinite(gn):
                pr["gns"].append(gn)
            else:
                fail = True  # non-finite gradients: the window is over the edge
        if not fail and len(pr["losses"]) < target:
            self.S = pr["lr"]
            for g in self.opt.param_groups:
                g["lr"] = pr["lr"]
            self.opt._step_impl()  # type: ignore[attr-defined]
            self._t += 1
            return loss

        # Window complete (or aborted): judge it against the POOLED baseline with an
        # SE-scaled threshold (longer windows = automatically finer detection).
        warm = False
        if not fail and baselined:
            margin = _PROBE_CONFIRM_MARGIN if pr["phase"] == "confirm" else _PROBE_MARGIN
            fail = _median(pr["losses"]) > _probe_thresh(pr["base"], margin, len(pr["losses"]))
        if not fail:
            if pr["gbase"] and pr["gns"]:
                warm = _median(pr["gns"]) > _PROBE_WARM * _median(pr["gbase"])
            # Pool only windows that are PROVABLY clean (margin 0: within the pure
            # SE band of the baseline), which is stricter than merely passing.
            # Pooling everything that passes would let mild elevations creep the
            # baseline upward and erode the detector (measured in tests).
            provably_clean = (not baselined) or (
                pr["losses"]
                and _median(pr["losses"]) <= _probe_thresh(pr["base"], 0.0, len(pr["losses"]))
            )
            if not warm and provably_clean:
                # Every such window is a draw from the same healthy regime (each starts
                # from the same x0), so it grows the pooled baseline for free.
                pr["base"].extend(pr["losses"])
                pr["gbase"].extend(pr["gns"])

        if pr["phase"] == "confirm":
            if fail:
                # Slow burn detected at the candidate: step down and re-confirm. The
                # step size is contextual — fine (x1.33) after a tight bisected
                # bracket, coarse (the ladder factor) after a ceiling top-out where
                # nothing below has been measured at long horizon.
                pr["tries"] += 1
                if pr["tries"] >= _PROBE_CONFIRM_MAX:
                    return self._probe_finish(pr["lr"] / pr.get("cstep", _PROBE_FACTOR_MIN), loss)
                pr["lr"] = max(pr["lr"] / pr.get("cstep", _PROBE_FACTOR_MIN), 1e-12)
            else:
                return self._probe_finish(pr["lr"], loss)  # confirmed clean at long horizon
        elif fail:
            if pr["passed"] is None:
                # Seed/d0 already above the edge: descend until something passes.
                pr["descend"] = True
                pr["fail"] = pr["lr"]
                pr["lr"] = max(pr["lr"] / (pr["factor"] * pr["factor"]), 1e-12)
            else:
                pr["fail"] = pr["lr"]
                pr["phase"] = "bisect"
                pr["lr"] = math.sqrt(pr["passed"] * pr["fail"])  # log-midpoint of the bracket
        else:
            pr["passed"] = pr["lr"]
            if pr["phase"] == "bisect" or pr["descend"]:
                # A (pass, fail) bracket exists: bisect it down to the fine grid.
                pr["phase"] = "bisect"
                if pr["fail"] is not None and pr["fail"] / pr["passed"] > _PROBE_FACTOR_MIN * 1.05:
                    pr["lr"] = math.sqrt(pr["passed"] * pr["fail"])
                else:
                    pr["phase"] = "confirm"
                    pr["cstep"] = _PROBE_FACTOR_MIN  # tight bracket -> fine descent
                    pr["lr"] = pr["passed"]
            else:  # climb
                if warm:
                    # Grad norms are heating up: the edge is near — refine the growth
                    # factor. Purely a PACE signal: wrong "warm" costs extra rungs,
                    # never correctness (the loss judgments stay authoritative).
                    pr["factor"] = max(math.sqrt(pr["factor"]), _PROBE_FACTOR_MIN)
                nxt = pr["lr"] * pr["factor"]
                if self._fuse is not None and nxt > self._fuse:
                    # Ladder topped out with no degradation (mushy regime): confirm the
                    # ceiling at long horizon — same role the fuse plays in the fallback.
                    pr["phase"] = "confirm"
                    pr["cstep"] = pr["factor"]  # nothing measured below -> coarse descent
                else:
                    pr["lr"] = nxt

        # Start the next window: clean trial from the exact snapshot.
        self._probe_restore()
        pr["losses"] = []
        pr["gns"] = []
        pr["skip1"] = True
        self.S = pr["lr"]  # so get_d()/trainer logs show the ladder live
        self._t += 1
        return loss  # judgment step: no update (its gradient belongs to discarded params)

    def _probe_restore(self) -> None:
        """Params back to the exact x0 snapshot + base optimizer state wiped (lazy re-init)."""
        for p, ref in self._x0.items():
            p.data.copy_(ref)
        self.opt.state.clear()

    def _probe_finish(self, chosen: float, loss: Any) -> Any:
        """Lock the discovered LR: restore x0, wipe base state, freeze. Training now
        starts byte-clean at ``chosen`` — the probe leaves no trace on the model."""
        self._probe_restore()
        self._probe = False
        self.S = max(chosen * self._scale, 1e-12)
        self._do_freeze()
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
        self.S = max(contact * _BACKOFF, 1e-12)
        if self._edge is not None and contact <= _EDGE_BAND * self._edge:
            self._do_freeze()
            if step_after:
                self.opt._step_impl()  # type: ignore[attr-defined]
            return True
        ramp = self._edge is None  # first-ever contact = the range-test phase ends here
        self._edge = contact
        # The one principled difference between the phases is the x0 handling:
        # - ramp contact ROLLS BACK to x0 (all pre-contact progress happened at
        #   sub-operating LRs — negligible — and x0 IS the exact pre-ramp snapshot,
        #   so the overshoot leaves NO trace on the params; the base's own EMAs are
        #   not rolled back — composability boundary — the spike gradient is never
        #   applied and the contamination decays within tens of steps);
        # - a post-ramp (drift-level) contact re-anchors x0 FORWARD to the current
        #   params and keeps going.
        if ramp:
            for p, ref in self._x0.items():
                p.data.copy_(ref)
        else:
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
            "edge": self._edge,  # also carries the ramp phase (_ramp_on == edge is None)
            "probe": dict(self._probe) if isinstance(self._probe, dict) else self._probe,
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
        self._probe = blob.get("probe")
        if isinstance(self._probe, dict):
            # Mid-probe resume: the checkpoint params are mid-rung — redo the current
            # rung from a clean restore rather than trusting a half-measured one.
            self._probe = dict(self._probe)
            self._probe["restore"] = True
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

    _autolr: AutoLRTuner | None

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

    def report_loss(self, loss: Any) -> None:
        """Feed this step's training loss to ``auto_lr`` (no-op when off/frozen).

        Call once per step BEFORE ``step()``. Enables the loss-driven range test —
        the primary, model-independent discovery mode. One line in the trainer:
        ``optimizer.report_loss(loss)`` right after computing the loss.
        """
        t = self._autolr
        if t is not None and not t.frozen:
            t.report_loss(loss)

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
