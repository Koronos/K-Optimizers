"""``MechanicTuner`` — a reusable Mechanic online LR tuner (Cutkosky, Defazio &
Mehta, NeurIPS 2023, arXiv:2306.00144) as a *composable* attachment for kaon
optimizers, exposed through the ``auto_lr=True`` flag.

This is the engine extracted from :class:`~kaon.autokaon.Autokaon` so any kaon
base optimizer can auto-discover its learning rate without a separate wrapper
class. The base optimizer keeps its own ``param_groups`` / ``state`` / ``_codec``;
the tuner only ever snapshots ``p``, drives one base update (``opt._step_impl``),
measures the update vector ``Delta = p_after - p_before`` and the Mechanic
gradient ``h = <Delta, g + decay>``, runs the scalar coin-betting recurrences,
and rewrites ``p = ref + sum(s)·Delta``. It is update-agnostic *by construction*
— it never sees how the base update was formed.

Design notes specific to the flag (vs the old ``Autokaon`` class):

* **``lr`` is not the LR under ``auto_lr``.** The tuner owns the scale. A base
  ``lr < 1`` (e.g. renga-flow's prefilled ``1e-4``) is *ignored* with a one-time
  warning — never silently used as a ~1e-4 multiplier that would kill training.
  Use ``auto_lr_scale`` for an explicit multiplier. The base's ``group["lr"]`` is
  forced to ``1.0`` during adaptation (so the base update is unit-scaled), then
  set to the discovered ``S`` at freeze.
* **Freeze keeps the tuner scalars (the hedge).** Only the per-parameter ``ref``
  buffer — the single expensive slot — is freed on freeze. The 24 tuner floats
  (6 betas × 4 buffers) are kept so a future signal-triggered re-arm is a *warm*
  resume (no cold re-bootstrap transient), and a frozen checkpoint carries the
  tuner history. Freezing still hands the base back at ``lr=S``: same memory
  (ref gone), same speed (no wrapper passes).

The seed / cap / floor / tuner-betas that iteration-3 validated as generalizing
are frozen here as module constants — the only user knobs are ``auto_lr``,
``auto_lr_freeze``, ``auto_lr_scale`` and (advanced) ``auto_lr_cap_rel``.

CUDA note: this env SIGFPEs on ``torch.dot`` for CUDA tensors, so every inner
product is ``(a * b).sum()``.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import Tensor

__all__ = ["MechanicTuner"]

# Mechanic's default tuner betas (arXiv:2306.00144, Alg. 1): n=6 parallel
# coin-betting tuners with geometrically spaced recency horizons, summed.
_DEFAULT_BETAS: tuple[float, ...] = (0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999)

# Internal constants (iteration-3 validated these generalize; changing them is a
# source edit, not a call-site option) — mirror kaon.autokaon.
_S_INIT_REL: float = 3e-3          # multiplier on global param RMS for the data-relative seed
_SCALE_FLOOR_FRAC: float = 0.5     # floor sum(s) at this fraction of its running max
_LR_FREEZE_TOL: float = 0.02       # "auto" plateau: rel sum(s) change below this counts as flat
_LR_FREEZE_PATIENCE: int = 50      # "auto" plateau: consecutive flat steps to freeze
_LR_FREEZE_MAX_FRAC: float = 0.9   # "auto": a flat step counts only when sum(s) >= this * running max
_S_DECAY: float = 0.01             # Mechanic's lambda tuner weight-decay term
_EPS: float = 1e-8                 # tuner numerical floor


class MechanicTuner:
    """Mechanic scalar-LR tuner attached to a host optimizer ``opt``.

    ``opt`` must expose ``param_groups``, a per-param ``state`` mapping, and an
    ``_step_impl(closure=None)`` that performs one base update in place. Optionally
    ``_codec(group)`` (for the momentum fold on freeze); absent codec -> lr-only
    freeze fold.
    """

    def __init__(
        self,
        opt: torch.optim.Optimizer,
        *,
        lr_freeze: int | str | None,
        scale: float,
        cap_rel: float,
    ) -> None:
        if lr_freeze is not None and lr_freeze != "auto" and (not isinstance(lr_freeze, int) or lr_freeze < 1):
            raise ValueError(f"auto_lr_freeze must be None, 'auto', or an int >= 1, got {lr_freeze!r}")
        if not scale > 0.0:
            raise ValueError(f"auto_lr_scale must be > 0, got {scale}")
        if not cap_rel > 0.0:
            raise ValueError(f"auto_lr_cap_rel must be > 0, got {cap_rel}")
        self.opt = opt

        # lr<1 safety: the tuner owns the LR. A leftover/prefilled small lr is an
        # accident (it would shrink the discovered LR ~lr×) -> ignore it, warn once.
        small = [float(g["lr"]) for g in opt.param_groups if g["lr"] < 1.0]
        if small:
            warnings.warn(
                f"auto_lr=True discovers the learning rate itself; the base lr "
                f"({small[0]:g}) is ignored. Use auto_lr_scale for an explicit "
                f"multiplier, or set lr=1.0 to silence this warning.",
                stacklevel=3,
            )
        # Force the base to step at unit lr during adaptation (so the base update
        # vector is unit-scaled and Mechanic's sum(s) is the effective LR).
        for group in opt.param_groups:
            group["lr"] = 1.0

        self._scale = float(scale)
        self._cap_rel = float(cap_rel)
        self._lr_freeze = lr_freeze
        self._s_decay = _S_DECAY
        self._eps = _EPS
        self._s_init_rel = _S_INIT_REL
        self._scale_floor_frac = _SCALE_FLOOR_FRAC
        self._lr_freeze_tol = _LR_FREEZE_TOL
        self._lr_freeze_patience = _LR_FREEZE_PATIENCE
        self._lr_freeze_max_frac = _LR_FREEZE_MAX_FRAC

        self._betas = torch.tensor(_DEFAULT_BETAS, dtype=torch.float32)
        self._s_init = 0.0            # resolved on the first step (data-relative)
        self._scale_cap: float | None = None  # resolved on the first step
        self._s_sum_max = 0.0
        self._last_eff_s_sum = 0.0

        self.frozen = False
        self.frozen_lr: float | None = None
        self._plateau_count = 0
        self._prev_s_sum: float | None = None

        n = len(_DEFAULT_BETAS)
        self._mech: dict[str, Any] = {
            "s": torch.zeros(n, dtype=torch.float32),
            "v": torch.zeros(n, dtype=torch.float32),
            "reward": torch.zeros(n, dtype=torch.float32),
            "max_product": torch.full((n,), 1e-6, dtype=torch.float32),
            "iter": 0,
            "ref": {},
        }

    # -- introspection -----------------------------------------------------
    def get_d(self) -> float:
        """Discovered effective learning rate (floor/cap-clamped, or the frozen LR)."""
        if self.frozen:
            return float(self.frozen_lr)  # type: ignore[arg-type]
        if self._mech["iter"] == 0:
            return self._scale * float(self._mech["s"].sum().item())
        return self._last_eff_s_sum

    def get_s(self) -> Tensor:
        """Per-beta scale vector (CPU copy)."""
        return self._mech["s"].detach().cpu().clone()

    # -- step --------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        """One adapting step. Assumes not frozen (the host routes frozen -> _step_impl)."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        mech = self._mech
        betas = self._betas
        first = mech["iter"] == 0

        # 1) snapshot params + clone grads (base step may mutate grads), lazily
        #    allocate ref on first sight of each param.
        prev: dict[Tensor, Tensor] = {}
        grads: dict[Tensor, Tensor] = {}
        for group in self.opt.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grads[p] = p.grad.detach().clone()
                prev[p] = p.detach().clone()
                if p not in mech["ref"]:
                    mech["ref"][p] = p.detach().clone()
        if not grads:
            return loss

        if first:
            dev = next(iter(grads)).device
            for k in ("s", "v", "reward", "max_product"):
                mech[k] = mech[k].to(dev)
            betas = self._betas = betas.to(dev)
            # Data-relative (LARS-style) seed: the base lr=1 update is ~unit-RMS,
            # so the trust ratio ||p|| / ||u_lr1|| ~ RMS(p). Seed s_init from the
            # global param RMS so the initial effective LR lands at a data-relative
            # scale instead of ramping from a fixed 1e-8 over ~100 steps.
            sq = 0.0
            cnt = 0
            for p in prev:
                pf = prev[p].float()
                sq = sq + (pf * pf).sum()
                cnt += pf.numel()
            rms = float((sq / max(cnt, 1)) ** 0.5) if cnt else 0.0
            self._s_init = max(self._s_init_rel * rms, 1e-8)
            # Auto cap: a fixed multiple of the (data-relative) seed, so the ceiling
            # tracks the problem's LR scale. Load-bearing on short horizons.
            self._scale_cap = self._cap_rel * max(self._s_init, 1e-8)

        # 2) base update at lr=1. u_t = p_after - prev is the base update vector.
        self.opt._step_impl()  # type: ignore[attr-defined]

        s = mech["s"]
        s_sum = float(s.sum().item())
        # Undo the *effective* (floor/cap-clamped) scale applied last step for the
        # on-the-fly Delta reconstruction. On the first step ref==prev so the
        # displacement is 0 regardless of the denominator.
        prev_eff_s_sum = self._last_eff_s_sum

        # 3) global norms for the Mechanic weight-decay-esque term (Alg.1 line 10).
        grad_norm = param_norm = None
        if self._s_decay != 0.0:
            grad_sq: Any = 0.0
            param_sq: Any = 0.0
            for p, g in grads.items():
                gf = g.float()
                grad_sq = grad_sq + (gf * gf).sum()
                pf = p.detach().float()
                param_sq = param_sq + (pf * pf).sum()
            grad_norm = torch.sqrt(grad_sq) if torch.is_tensor(grad_sq) else torch.tensor(0.0)
            param_norm = torch.sqrt(param_sq) if torch.is_tensor(param_sq) else torch.tensor(0.0)

        # 4) form Delta_t per param and h = <Delta_t, g + decay> summed over params.
        deltas: dict[Tensor, Tensor] = {}
        inner: Tensor | None = None
        for group in self.opt.param_groups:
            for p in group["params"]:
                if p not in grads:
                    continue
                u = (p.detach() - prev[p]).float()  # base update u_t
                ref = mech["ref"][p]
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

        # 6) set p = ref + scale * sum(s) * Delta_t, clamped to [floor, cap].
        new_s_sum = max(float(s.sum().item()) * self._scale, 0.0)
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
        # 7) decide whether to freeze (and become plain base) for next step.
        self._maybe_freeze(new_s_sum, mech["iter"])
        return loss

    # -- freeze ------------------------------------------------------------
    def _freeze(self) -> None:
        """Fold ``sum(s)`` into the base lr, free only ``ref``, keep tuner scalars.

        The base update is linear in ``lr`` (RMS-clip is pre-lr, the second-moment
        EMA is lr-independent), so running the base at ``lr=S`` reproduces the
        frozen ``p = ref + S·Delta`` trajectory (up to unbiased bf16/SR rounding).
        With momentum (``beta1>0``) the first-moment EMA breaks that linearity, so
        we fold ``S`` into the stored momentum too via the codec (bit-exact for
        fp32/int8/4bit, rounding-exact for bf16); ``beta1=0`` skips it.

        HEDGE: only ``ref`` (the one per-parameter buffer) is freed; the 24 tuner
        floats are kept so a future re-arm is a warm resume and checkpoints carry
        the tuner history.
        """
        s_sum = max(self._last_eff_s_sum, 0.0)
        self.frozen_lr = s_sum
        if s_sum != 1.0 and hasattr(self.opt, "_codec"):
            for group in self.opt.param_groups:
                codec = self.opt._codec(group)  # type: ignore[attr-defined]
                for p in group["params"]:
                    state = self.opt.state.get(p)
                    if state is not None and "m" in state:
                        codec.scale_(state, s_sum)
        for group in self.opt.param_groups:
            group["lr"] = s_sum
        self._mech["ref"].clear()  # free the big per-param buffer; keep the scalars
        self.frozen = True

    def _maybe_freeze(self, s_sum: float, iters_done: int) -> None:
        """Decide whether to freeze AFTER a tuner step at the current ``sum(s)``."""
        if self._lr_freeze is None or self.frozen:
            return
        if self._lr_freeze == "auto":
            prev = self._prev_s_sum
            # A step counts toward the plateau only if sum(s) is BOTH flat (rel
            # change below tol) AND near its running max (guards against an early
            # freeze on a transient dip before the scale settles).
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

    # -- checkpointing (blob merged into the host's state_dict) -------------
    def state_blob(self) -> dict[str, Any]:
        """Serializable tuner state. ``ref`` is stored in stable param order (not
        keyed by Tensor identity, which does not survive (de)serialization)."""
        params = [p for g in self.opt.param_groups for p in g["params"]]
        index = {p: i for i, p in enumerate(params)}
        ref_list: list[Tensor | None] = [None] * len(params)
        for p, r in self._mech["ref"].items():
            ref_list[index[p]] = r.detach().clone()
        return {
            "s": self._mech["s"].detach().clone(),
            "v": self._mech["v"].detach().clone(),
            "reward": self._mech["reward"].detach().clone(),
            "max_product": self._mech["max_product"].detach().clone(),
            "iter": self._mech["iter"],
            "ref": ref_list,
            "frozen": self.frozen,
            "frozen_lr": self.frozen_lr,
            "s_init": self._s_init,
            "scale_cap": self._scale_cap,
            "s_sum_max": self._s_sum_max,
            "last_eff_s_sum": self._last_eff_s_sum,
            "plateau_count": self._plateau_count,
            "prev_s_sum": self._prev_s_sum,
        }

    def load_blob(self, blob: dict[str, Any]) -> None:
        """Restore tuner state produced by :meth:`state_blob`."""
        params = [p for g in self.opt.param_groups for p in g["params"]]
        dev = params[0].device if params else torch.device("cpu")
        self._mech["s"] = blob["s"].to(dev)
        self._mech["v"] = blob["v"].to(dev)
        self._mech["reward"] = blob["reward"].to(dev)
        self._mech["max_product"] = blob["max_product"].to(dev)
        self._mech["iter"] = int(blob["iter"])
        ref: dict[Tensor, Tensor] = {}
        for i, r in enumerate(blob["ref"]):
            if r is not None:
                ref[params[i]] = r.to(params[i].device)
        self._mech["ref"] = ref
        self._betas = self._betas.to(dev)
        self.frozen = bool(blob["frozen"])
        self.frozen_lr = blob["frozen_lr"]
        self._s_init = float(blob["s_init"])
        self._scale_cap = blob["scale_cap"]
        self._s_sum_max = float(blob["s_sum_max"])
        self._last_eff_s_sum = float(blob["last_eff_s_sum"])
        self._plateau_count = int(blob["plateau_count"])
        self._prev_s_sum = blob["prev_s_sum"]
