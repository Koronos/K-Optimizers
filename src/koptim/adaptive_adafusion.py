"""AdaptiveAdafusion — a parameter-free learning rate on top of *Adafusion's*
update rule, via a Mechanic-style online scale tuner, with a **freeze-to-free**
handoff that turns it into plain Adafusion after warmup.

(Formerly ``AdafusionProdigy`` — a misnomer, it is Mechanic, not Prodigy. The old
name remains importable as a back-compat alias.)

Motivation
----------
A matched-effective-LR ablation found that ``KProdigy`` (Prodigy's Adam-form
D-adaptation) converges ~2x worse than ``Adafusion`` on the mini pixel-DDPM, and
the gap isolates entirely to **first-moment placement relative to the √v
normalization**:

* KProdigy / Adam / Prodigy: ``delta = ema(d·g) / √v`` — momentum of the *raw*
  gradient, then normalize.
* Adafusion / Adafactor-with-momentum: ``delta = ema(clip(g / √v))`` —
  normalize + RMS-clip first, *then* momentum.

Adafusion's ordering is the better one. We want a *parameter-free* optimizer that
keeps Adafusion's update verbatim but auto-discovers the learning rate. Prodigy's
D-estimator is derived for the Adam form, so it does not transplant cleanly onto
the normalize-then-momentum update.

Mechanic (Cutkosky, Defazio & Mehta, NeurIPS 2023, arXiv:2306.00144) solves this
the clean way: it is an online learning-rate **tuner that wraps an arbitrary base
optimizer** and learns a *scalar* multiplier ``s`` on the base update by
coin-betting / reward maximisation. It is update-agnostic *by construction* — it
only ever sees the gradient and the base optimizer's update vector
``Delta = sum_t u_t``, never how ``u_t`` was formed.

Design (mirrors the reference ``mechanize`` wrapper, arXiv:2306.00144 Alg. 1)
----------------------------------------------------------------------------
The base optimizer is an internal :class:`~koptim.adafusion.Adafusion` at
``lr=1``. Each step (while adapting): snapshot ``p``, run the base step to get
``u_t = p_after - p_before``, accumulate ``Delta_t`` (stored, or recomputed from
``(p - ref)/sum(s)`` when ``store_delta=False``), form the Mechanic gradient
``h_t = <Delta_t, g_t + decay_t>`` summed over params, run the per-beta scalar
tuner to get ``s``, and set ``p = ref + sum(s) * Delta_t``.

The discovered effective LR is ``sum(s)``; read it with :meth:`get_d`.

Memory
------
The only irreducible per-parameter cost over plain Adafusion is **``ref`` (one
extra copy of the weights, in param dtype)** — matching the reference Mechanic
("at minimum one extra slot of memory"). ``store_delta`` adds a second per-param
buffer; the reference *defaults it off* and reconstructs ``Delta`` on the fly,
reporting "negligible effect" — so we default ``store_delta=False`` too.

Freeze-to-free (``lr_freeze``)
------------------------------
The headline feature. Mechanic's scale converges to a stable operating LR; once
it has, the per-step wrapper overhead (snapshot + grad clone + Delta passes) and
the ``ref``/``delta`` buffers are pure waste. ``lr_freeze`` ends adaptation:

* ``None`` — never freeze (plain Mechanic-tuned Adafusion).
* ``int N`` — freeze after ``N`` steps.
* ``"auto"`` — freeze when ``sum(s)`` plateaus: relative change below
  ``lr_freeze_tol`` for ``lr_freeze_patience`` consecutive steps.

On freeze we record ``S = sum(s)``, **set the base Adafusion's ``lr`` to ``S``**,
**free ``ref``/``delta``/tuner scalars**, and route every subsequent ``step()``
straight to ``base.step()``. After freeze the optimizer **is** the inner plain
Adafusion at ``lr=S`` — same memory (ref/delta gone), same speed (no wrapper
passes), same update — *by construction*, because ``step()`` literally calls
``base.step()``.

Bit-exactness of the *handoff* (does ``lr=S`` reproduce the pre-freeze
``p = ref + S·Delta`` extrapolation?) depends on the momentum setting:

* **``beta1 == 0`` (the default ``adafusion_betas=(0.0, 0.999)``):** the update is
  ``delta = update`` with ``update`` linear in ``lr`` (RMS-clip + cautious mask are
  computed pre-``lr``; the second-moment EMA is ``lr``-independent). So
  ``base.step(lr=1)`` then ``p = ref + S·Δstep`` equals ``base.step(lr=S)``
  **bit-exact** (fp32; up to unbiased bf16/SR rounding otherwise). Freeze is exact.
* **``beta1 > 0``:** the first-moment EMA accumulates the *``lr``-scaled* update
  (``ema(lr·update)``), so the buffer built at ``lr=1`` during warmup carries a
  different scale than one built at ``lr=S``. Folding ``S`` into ``lr`` at the
  freeze boundary thus introduces a one-time EMA-scale change. The post-freeze
  trajectory is still genuine Adafusion(lr=S) going forward (and converges fine),
  it is simply not bit-identical to the pre-freeze extrapolation. For a
  truly seamless freeze use the default ``beta1 == 0``.

CUDA note: this env SIGFPEs on ``torch.dot`` for CUDA tensors, so every inner
product is ``(a * b).sum()``.

Based on Mechanic by Ashok Cutkosky, Aaron Defazio & Harsh Mehta
(https://github.com/optimizedlearning/mechanic), and Adafusion's update engine.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import torch
from torch import Tensor

from koptim.adafusion import Adafusion

__all__ = ["AdaptiveAdafusion", "AdafusionProdigy"]

# Mechanic's default tuner betas (arXiv:2306.00144, Alg. 1): n=6 parallel
# coin-betting tuners with geometrically spaced recency horizons, summed.
_DEFAULT_BETAS: tuple[float, ...] = (0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999)


class AdaptiveAdafusion(torch.optim.Optimizer):
    """Parameter-free LR on Adafusion's update via a Mechanic scale tuner, with a
    freeze-to-pure-Adafusion handoff.

    Args:
        params: parameters or param-group dicts.
        lr: outer multiplier on the discovered scale. **Leave at 1.0** — Mechanic
            finds the scale. (Kept so a schedule can still be layered on top.)
        s_init: Mechanic's initial scale seed. The paper default is ``1e-8``; on
            short fine-tuning runs the bootstrap is slow there (short-horizon bias).
            Default ``"auto"`` — a data-relative LARS-style seed set on the first
            step from the global param RMS (Adafusion's ``lr=1`` update is unit-RMS,
            so the trust ratio ``||p||/||u_lr1|| == RMS(p)``); the seed is
            ``s_init_rel * RMS(p)``, landing the initial effective LR at a
            data-relative scale instead of ramping there over ~100 steps. Pass a
            float to pin a fixed seed (e.g. ``1e-8`` for very long runs).
        s_init_rel: the multiplier on the param RMS for the ``s_init="auto"`` seed.
            Default ``3e-3``. Larger seeds higher (refines down); smaller seeds lower.
        scale_floor_frac: floor the effective ``sum(s)`` at ``scale_floor_frac`` times
            its running max, so the discovered LR cannot collapse/stall once it has
            grown. Default ``0.5``. Set ``0`` to disable the floor.
        scale_cap: hard cap on the effective ``sum(s)`` (the discovered LR). Default
            ``"auto"`` — set on the first step to ``scale_cap_rel`` times the
            data-relative seed, so the ceiling tracks the problem's LR scale. **This
            is the load-bearing stability fix**: on short DDPM horizons the Mechanic
            scale is prone to a large transient spike (measured: it tries to run to
            ~1e-2 while the operating LR is ~5e-4); the cap converts that spike into a
            robust ceiling near the operating LR, removing the seed-dependent
            divergence. Pass a fixed float to pin a manual ceiling, or ``None`` to
            disable (reproduces the unstable iteration-1 behavior).
        scale_cap_rel: multiplier on the seed for the ``scale_cap="auto"`` ceiling.
            Default ``6.0`` (lands the cap at ~6x the seed, near the swept-optimal LR
            on the measured DDPM). Lower for a more conservative ceiling; this is the
            main knob for the auto cap.
        betas: tuner recency horizons (the 6 Mechanic betas), summed into the LR.
        s_decay: Mechanic's ``lambda`` weight-decay-esque tuner term (default
            ``0.01``). Set ``0`` to disable.
        eps: tuner numerical floor.
        store_delta: store the per-parameter displacement ``Delta`` from ``ref``.
            Default ``False`` (matches the reference Mechanic; recomputes
            ``Delta = (p - ref) / sum(s)`` on the fly, saving 1x param bytes with
            negligible effect). ``True`` keeps an explicit fp32 ``Delta`` buffer
            (marginally more accurate in the first few steps).
        lr_freeze: when to stop adapting and become plain Adafusion at the frozen
            LR. ``None`` (default) never freezes. ``int N`` freezes after ``N``
            steps. ``"auto"`` freezes on a ``sum(s)`` plateau (see ``lr_freeze_tol``
            / ``lr_freeze_patience``). On freeze, ``ref``/``delta``/tuner scalars
            are freed and the inner Adafusion runs at ``lr = sum(s)_frozen`` — from
            then on it is byte-for-byte and speed-for-speed plain Adafusion.
        lr_freeze_tol: for ``lr_freeze="auto"`` — a step counts as a plateau when
            the relative change of ``sum(s)`` vs the previous step is below this.
            Default ``0.02``.
        lr_freeze_patience: for ``lr_freeze="auto"`` — consecutive plateau steps
            required to trigger the freeze. Default ``50``.
        lr_freeze_max_frac: for ``lr_freeze="auto"`` — a step only counts toward the
            plateau when ``sum(s)`` is also at least this fraction of its running max,
            so a transient DIP (sum(s) flat but well below its peak) does not trigger
            an early freeze. Default ``0.9``. Set ``0`` to disable the near-max guard.
        adafusion_betas: the inner Adafusion's ``(beta1, beta2)`` momentum /
            second-moment EMAs. Distinct from this class's tuner ``betas`` (which is
            shadowed by the keyword above, so this is the *only* way to set the base
            momentum). Default ``(0.0, 0.999)`` — **no first-moment EMA**, which is
            both Eduardo's minimum-VRAM config AND the regime where freeze is
            *bit-exact* (with ``beta1 > 0`` the momentum buffer accumulates the
            ``lr``-scaled update, so folding ``S`` into ``lr`` at freeze introduces a
            one-time EMA-scale change at the boundary — the trajectory is still plain
            Adafusion(lr=S) going forward, just not bit-identical to the pre-freeze
            ``ref + S·Delta`` extrapolation; see module docstring).
        **adafusion_kwargs: forwarded verbatim to the internal
            :class:`~koptim.adafusion.Adafusion` base (``clip_threshold``,
            ``cautious``, ``momentum_dtype``, ``bf16_method``, ``foreach``,
            ``weight_decay``...). NB: pass momentum betas via ``adafusion_betas``,
            not here — ``betas`` is the tuner's.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1.0,
        *,
        s_init: float | Literal["auto"] = "auto",
        s_init_rel: float = 3e-3,
        betas: tuple[float, ...] = _DEFAULT_BETAS,
        s_decay: float = 0.01,
        eps: float = 1e-8,
        store_delta: bool = False,
        scale_floor_frac: float = 0.5,
        scale_cap: float | Literal["auto"] | None = "auto",
        scale_cap_rel: float = 6.0,
        lr_freeze: int | Literal["auto"] | None = None,
        lr_freeze_tol: float = 0.02,
        lr_freeze_patience: int = 50,
        lr_freeze_max_frac: float = 0.9,
        adafusion_betas: tuple[float, float] = (0.0, 0.999),
        **adafusion_kwargs: Any,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if s_init != "auto" and not s_init > 0.0:
            raise ValueError(f"s_init must be > 0 or 'auto', got {s_init!r}")
        if not s_init_rel > 0.0:
            raise ValueError(f"s_init_rel must be > 0, got {s_init_rel}")
        if not all(0.0 < b < 1.0 for b in betas):
            raise ValueError(f"all tuner betas must be in (0, 1), got {betas}")
        if s_decay < 0.0:
            raise ValueError(f"s_decay must be >= 0, got {s_decay}")
        if not 0.0 <= scale_floor_frac < 1.0:
            raise ValueError(f"scale_floor_frac must be in [0, 1), got {scale_floor_frac}")
        if scale_cap is not None and scale_cap != "auto" and not scale_cap > 0.0:
            raise ValueError(f"scale_cap must be > 0, 'auto', or None, got {scale_cap!r}")
        if not scale_cap_rel > 0.0:
            raise ValueError(f"scale_cap_rel must be > 0, got {scale_cap_rel}")
        if not 0.0 <= lr_freeze_max_frac <= 1.0:
            raise ValueError(f"lr_freeze_max_frac must be in [0, 1], got {lr_freeze_max_frac}")
        if (
            lr_freeze is not None
            and lr_freeze != "auto"
            and (not isinstance(lr_freeze, int) or lr_freeze < 1)
        ):
            raise ValueError(f"lr_freeze must be None, 'auto', or an int >= 1, got {lr_freeze!r}")
        if not lr_freeze_tol > 0.0:
            raise ValueError(f"lr_freeze_tol must be > 0, got {lr_freeze_tol}")
        if lr_freeze_patience < 1:
            raise ValueError(f"lr_freeze_patience must be >= 1, got {lr_freeze_patience}")
        # We deliberately do NOT call super().__init__ with our own defaults: this
        # class owns no per-parameter optimizer state beyond ref/delta, which we
        # manage in self._mech. The param_groups are the base Adafusion's.
        param_list = list(params)
        # The base optimizer holds the param_groups and does the real update at
        # lr=1.0 (the Mechanic scale is applied by THIS class, on top) until freeze,
        # after which its lr is set to the frozen scale and it runs alone.
        self.base = Adafusion(param_list, lr=1.0, betas=adafusion_betas, **adafusion_kwargs)
        self.param_groups = self.base.param_groups
        self.state = self.base.state  # share so opt.state introspection sees both
        self.defaults = self.base.defaults

        self._lr = float(lr)
        self._betas = torch.tensor(betas, dtype=torch.float32)
        # s_init: a fixed float, or "auto" (data-relative LARS-style seed set on the
        # first step from the param RMS — Adafusion's lr=1 update is unit-RMS, so the
        # trust ratio ||p||/||u_lr1|| == RMS(p); seed = s_init_rel * RMS(p)).
        self._s_init_auto = s_init == "auto"
        self._s_init = 0.0 if self._s_init_auto else float(s_init)
        self._s_init_rel = float(s_init_rel)
        self._s_decay = float(s_decay)
        self._eps = float(eps)
        self._store_delta = store_delta

        # Scale floor/cap: clamp the effective sum(s) to
        # [scale_floor_frac * running_max, scale_cap]. Floor (relative to the running
        # max) stops a collapse/stall; cap stops a runaway spike.
        self._scale_floor_frac = float(scale_floor_frac)
        # scale_cap: a fixed float, None (no cap), or "auto" — set on the first step
        # to scale_cap_rel * (data-relative seed), so the cap tracks the problem's LR
        # scale (the seed is itself data-relative). The Mechanic scale on a short
        # horizon is prone to a large transient spike; the cap converts that into a
        # robust ceiling near the operating LR.
        self._scale_cap_auto = scale_cap == "auto"
        self._scale_cap = None if scale_cap in (None, "auto") else float(scale_cap)
        self._scale_cap_rel = float(scale_cap_rel)
        self._s_sum_max = 0.0  # running max of the effective sum(s)

        # Freeze ("become Adafusion") config + bookkeeping.
        self._lr_freeze = lr_freeze
        self._lr_freeze_tol = float(lr_freeze_tol)
        self._lr_freeze_patience = int(lr_freeze_patience)
        self._lr_freeze_max_frac = float(lr_freeze_max_frac)
        self._frozen = False
        self._frozen_lr: float | None = None
        self._plateau_count = 0
        self._prev_s_sum: float | None = None
        # The last *effective* sum(s) actually applied to params (after floor/cap).
        # Used at freeze so the folded base lr matches the trajectory the model walks.
        self._last_eff_s_sum: float = 0.0

        # Tuner state (lives on CPU until first step pins it to the param device).
        n = len(betas)
        self._mech: dict[str, Any] = {
            "s": torch.zeros(n, dtype=torch.float32),
            "v": torch.zeros(n, dtype=torch.float32),          # sum of squared h
            "reward": torch.zeros(n, dtype=torch.float32),     # r
            # m floored at 1e-6 (matches the reference impl), so the wealth seed is
            # non-degenerate even if the first-step inner product happens to be ~0.
            "max_product": torch.full((n,), 1e-6, dtype=torch.float32),  # m
            "iter": 0,
            "ref": {},     # p -> reference (start) value
            "delta": {},   # p -> accumulated base-update displacement (store_delta)
        }

    # -- introspection -----------------------------------------------------
    def get_d(self) -> float:
        """Discovered effective learning rate (the floor/cap-clamped ``lr * sum(s)``
        actually applied to params, or the frozen LR)."""
        if self._frozen:
            return float(self._frozen_lr)  # type: ignore[arg-type]
        if self._mech["iter"] == 0:
            return self._lr * float(self._mech["s"].sum().item())
        return self._last_eff_s_sum

    def get_s(self) -> torch.Tensor:
        """Per-beta scale vector (CPU copy)."""
        return self._mech["s"].detach().cpu().clone()

    def is_frozen(self) -> bool:
        """Whether the LR has frozen (now running as plain Adafusion)."""
        return self._frozen

    @property
    def frozen_lr(self) -> float | None:
        """The frozen effective LR, or ``None`` if still adapting."""
        return self._frozen_lr

    def zero_grad(self, set_to_none: bool = True) -> None:  # noqa: FBT001, FBT002
        self.base.zero_grad(set_to_none=set_to_none)

    # -- freeze ------------------------------------------------------------
    def _freeze(self) -> None:
        """Stop adapting: fold ``sum(s)`` into the base lr and free Mechanic state.

        Adafusion's update is linear in ``lr``, so running the base at
        ``lr = sum(s)`` reproduces the frozen ``p = ref + sum(s)·Delta`` trajectory
        exactly (up to unbiased bf16/SR rounding) — see module docstring. We freeze
        at the *effective* (floor/cap-clamped) sum(s) the model is actually walking
        at, so the handoff stays exact even when the floor/cap engaged on the last
        step.
        """
        s_sum = max(self._last_eff_s_sum, 0.0)
        self._frozen_lr = s_sum
        for group in self.param_groups:
            group["lr"] = s_sum
        # Free the wrapper's per-param buffers and tuner scalars — post-freeze the
        # optimizer is byte-for-byte plain Adafusion.
        self._mech["ref"].clear()
        self._mech["delta"].clear()
        for k in ("s", "v", "reward", "max_product"):
            self._mech[k] = torch.zeros(0)
        self._frozen = True

    def _maybe_freeze(self, s_sum: float, iters_done: int) -> None:
        """Decide whether to freeze AFTER a tuner step at the current ``sum(s)``."""
        if self._lr_freeze is None or self._frozen:
            return
        if self._lr_freeze == "auto":
            prev = self._prev_s_sum
            # A step counts toward the plateau only if sum(s) is BOTH flat (rel change
            # below tol) AND near its running max (>= max_frac * running_max). The
            # near-max guard prevents an early freeze on a transient DIP (iter-1: the
            # plain flatness test froze on noisy transients before the scale settled).
            near_max = s_sum >= self._lr_freeze_max_frac * self._s_sum_max
            if prev is not None and prev > 0.0:
                rel = abs(s_sum - prev) / prev
                if rel < self._lr_freeze_tol and near_max:
                    self._plateau_count += 1
                else:
                    self._plateau_count = 0
            self._prev_s_sum = s_sum
            if self._plateau_count >= self._lr_freeze_patience:
                self._freeze()
        else:  # int N
            if iters_done >= int(self._lr_freeze):
                self._freeze()

    # -- step --------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:  # noqa: C901
        # Post-freeze: pure Adafusion, single update, no wrapper overhead.
        if self._frozen:
            return self.base.step(closure)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        mech = self._mech
        betas = self._betas
        first = mech["iter"] == 0

        # 1) snapshot params + clone grads (base step may mutate grads, e.g. wd),
        #    and lazily allocate ref/delta on first sight of each param.
        prev: dict[Tensor, Tensor] = {}
        grads: dict[Tensor, Tensor] = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grads[p] = p.grad.detach().clone()
                prev[p] = p.detach().clone()
                if p not in mech["ref"]:
                    mech["ref"][p] = p.detach().clone()
                    if self._store_delta:
                        mech["delta"][p] = torch.zeros_like(p, dtype=torch.float32)

        if not grads:
            return loss

        # Pin tuner scalars to the param device on first real step.
        if first:
            dev = next(iter(grads)).device
            for k in ("s", "v", "reward", "max_product"):
                mech[k] = mech[k].to(dev)
            betas = self._betas = betas.to(dev)
            # Data-relative (LARS-style) seed: Adafusion's lr=1 update is unit-RMS,
            # so the trust ratio ||p|| / ||u_lr1|| == RMS(p). Seeding s_init from the
            # global param RMS lands the *initial* effective LR at a data-relative
            # scale instead of a fixed 1e-8/1e-4 that has to ramp there over ~100
            # steps (short-horizon bias). The tuner then refines from a good seed.
            if self._s_init_auto:
                sq = 0.0
                cnt = 0
                for p in prev:
                    pf = prev[p].float()
                    sq = sq + (pf * pf).sum()
                    cnt += pf.numel()
                rms = float((sq / max(cnt, 1)) ** 0.5) if cnt else 0.0
                self._s_init = max(self._s_init_rel * rms, 1e-8)
            # Auto cap: a fixed multiple of the seed (which is itself data-relative
            # under s_init="auto"), so the ceiling tracks the problem's LR scale.
            # Needs the seed, so it runs after the seed is resolved.
            if self._scale_cap_auto:
                self._scale_cap = self._scale_cap_rel * max(self._s_init, 1e-8)

        # 2) run the base Adafusion step (the real normalize-then-momentum update
        #    at lr=1). u_t = p_after - p_before is the base update vector.
        self.base.step()

        s = mech["s"]
        s_sum = float(s.sum().item())
        # For the store_delta=False reconstruction we must undo the *effective*
        # (floor/cap-clamped) scale actually applied last step, which equals
        # _last_eff_s_sum (== raw sum(s) when no clamp engaged). On the very first
        # step ref==prev so the displacement is 0 regardless of the denominator.
        prev_eff_s_sum = self._last_eff_s_sum

        # 3) global norms for the Mechanic weight-decay-esque term (paper Alg.1
        #    line 10). (a*b).sum() everywhere (CUDA SIGFPE on torch.dot here).
        grad_sq = 0.0
        param_sq = 0.0
        if self._s_decay != 0.0:
            for p, g in grads.items():
                gf = g.float()
                grad_sq = grad_sq + (gf * gf).sum()
                pf = p.detach().float()
                param_sq = param_sq + (pf * pf).sum()
            grad_norm = torch.sqrt(grad_sq) if torch.is_tensor(grad_sq) else torch.tensor(0.0)
            param_norm = torch.sqrt(param_sq) if torch.is_tensor(param_sq) else torch.tensor(0.0)

        # 4) form Delta_t per param and h = <Delta_t, g + decay> summed over params.
        deltas: dict[Tensor, Tensor] = {}
        inner = None
        for group in self.param_groups:
            for p in group["params"]:
                if p not in grads:
                    continue
                u = (p.detach() - prev[p]).float()  # base update u_t
                ref = mech["ref"][p]
                if self._store_delta:
                    delta = mech["delta"][p]
                    delta.add_(u)
                else:
                    delta = (prev[p].float() - ref.float()) / (prev_eff_s_sum + self._eps)
                    delta = delta.add_(u)
                deltas[p] = delta
                g = grads[p].float()
                if self._s_decay != 0.0:
                    decay = (
                        self._s_decay * p.detach().float() * s_sum
                        * grad_norm / (param_norm + self._eps)
                    )
                    contrib = (delta * (g + decay)).sum()
                else:
                    contrib = (delta * g).sum()
                inner = contrib if inner is None else inner + contrib

        h = inner  # scalar tensor on device

        # 5) Mechanic tuner recurrences (arXiv:2306.00144 Alg. 1, lines 11-16).
        m = mech["max_product"]
        v = mech["v"]
        reward = mech["reward"]
        m.copy_(torch.maximum(betas * m, torch.abs(h)))
        v.mul_(betas * betas).add_(h * h)
        reward.mul_(betas).sub_(s * h)
        reward.clamp_(min=0.0)
        wealth = self._s_init * m / betas.numel() + reward
        s.copy_(wealth / (torch.sqrt(v) + self._eps))

        # 6) set p = ref + lr * sum(s) * Delta_t, with a floor/cap on the effective
        #    scale: clamp sum(s) to [floor_frac * running_max, cap]. The running-max
        #    floor stops a collapse/stall (seed-0); the cap stops a runaway spike
        #    (seed-1). The floor only engages once the scale has grown (running max),
        #    so it never inflates the legitimate early bootstrap.
        new_s_sum = max(float(s.sum().item()) * self._lr, 0.0)
        if self._scale_cap is not None and new_s_sum > self._scale_cap:
            new_s_sum = self._scale_cap
        self._s_sum_max = max(self._s_sum_max, new_s_sum)
        floor = self._scale_floor_frac * self._s_sum_max
        if new_s_sum < floor:
            new_s_sum = floor
        self._last_eff_s_sum = new_s_sum
        for p, delta in deltas.items():
            ref = mech["ref"][p]
            p.copy_((ref.float() + new_s_sum * delta).to(p.dtype))

        mech["iter"] += 1
        # 7) decide whether to freeze (and become plain Adafusion) for next step.
        self._maybe_freeze(new_s_sum, mech["iter"])
        return loss

    # -- state dict (delegate to base; tuner state is small) ---------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "base": self.base.state_dict(),
            "frozen": self._frozen,
            "frozen_lr": self._frozen_lr,
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.base.load_state_dict(sd["base"])
        self._frozen = sd.get("frozen", False)
        self._frozen_lr = sd.get("frozen_lr")


# Back-compat alias: the optimizer was shipped as ``AdafusionProdigy`` (a misnomer
# — it is Mechanic, not Prodigy). Keep the old name importable.
AdafusionProdigy = AdaptiveAdafusion
