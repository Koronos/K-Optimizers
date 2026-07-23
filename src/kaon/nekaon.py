"""Nekaon — Adakaon + negative momentum-lookahead (the in-house flat-minima flagship).

Nekaon is :class:`~kaon.adakaon.Adakaon` with one structural addition: between steps,
the live weights are displaced **k optimizer-steps ahead** along the smoothed update
direction, so every gradient is evaluated at the *anticipated* point (extragradient /
Nesterov-style), while the update itself lands on the true iterate:

.. code-block:: text

    # end of step t (inside opt.step(), after the Adakaon update):
    w_live <- w + k * m_t          # m = EMA of the lr-scaled, preconditioned,
    # loop: forward/backward        #     RMS-clipped update (Adakaon's momentum,
    # start of step t+1:            #     which points downhill)
    w_live <- w_live - k * m_t     # exact removal (m unchanged in between)
    adakaon_step(grad_at_lookahead)

Why this exists (measured on the control battery, 2026-06-10 campaign):

* It is the **SAM-problem solver**: SAM's flat-minima bias costs a second
  forward/backward per step (~2x the GEMM phase). Nekaon's perturbation needs **no
  extra pass and no extra memory** — the momentum buffer already exists, and the
  perturbation is recomputed from it on removal.
* The **negative** (downhill, Nesterov-like) direction measurably beats the SAM-like
  uphill climb on BOTH loss and train-val gap (probes ``MSAM+`` vs ``MSAM-``), and the
  mechanism owns both field records: best constant-LR loss (with rich momentum) and
  best constant-LR gap (with lean momentum) — ``k`` is the dial between them.
* The lookahead is **measured in steps, not weight units** (the structural fix over a
  fixed SAM/MSAM radius ``rho``): ``e = k * m`` self-scales with the LR (and any
  schedule), with the per-coordinate ``1/sqrt(v)`` metric, and with the model's weight
  scale. Calibration on the proxy: the best fixed radius translated to the SAME
  ``k ~ 1.7`` at beta1=0.2 and 0.9 — the step-unit formulation is the invariant, so it
  transfers across models without re-tuning (validated by holding ``k`` fixed under
  LR x0.5 / x2).
* ``weight_decay=0.1`` is the default: measured as a frontier-mover (improves loss AND
  gap together) on two bases, never harmful — an include-free-win.

Like Lookahead / Schedule-Free / MSAM, the live weights between steps are NOT the true
iterate: call :meth:`eval` before sampling / validation / checkpointing and
:meth:`train` to resume (always checkpoint in eval mode).

Implementation: a thin preset over :class:`~kaon.msam.MSAM` (``norm="none"``,
``rho=-k``) wrapping Adakaon — one tested mechanism, one code path. ``k=0`` is exactly
Adakaon.

Low-VRAM mode (``low_vram_above``): on a full fine-tune, most bytes live in a few big
weight matrices, while the lookahead's loss/gap win is measured to matter most on the
many-small-tensor regime (biases, norms, LoRA/LoKr factors) — exactly where kaon's
foreach kernels already dominate on speed too. Setting ``low_vram_above`` routes tensors
bigger than that (in ``numel()``) into a momentum-free group (``betas=(0, betas[1])``,
``lr * low_vram_lr_ratio``): Adakaon's own per-group gate (``betas[0] > 0``) skips
allocating a momentum buffer for that group, and MSAM's lookahead only climbs params
that HAVE one — so the big group transparently gets plain momentum-free Adakaon, no
separate optimizer needed. Default ``None`` is the original behavior (momentum + lookahead everywhere), replacing
the standalone ``NekaonAlloc`` PoC — cuts optimizer-state memory from 0.56 to ~0.05
B/param (LoRA/LoKr adapters are too small in absolute bytes for this to matter either
way; it's a full-fine-tune lever).

``low_vram_lr_ratio`` and ``low_vram_above`` defaults are MEASURED, not guessed (2026-07-03
sweep, proxy `C=128`/`N=1400`; caught and fixed a battery/profiler bug along the way where
the schedule loop overwrote every param group's lr to the SAME value every step, silently
defeating any per-group lr ratio — the fix preserves each group's own base lr through the
schedule). At threshold=65536: ratio sweep te 0.0884/0.0806/**0.0789**/0.0801/0.0812 for
ratio 0.1/0.25/**0.5**/0.75/1.0 — ``0.5`` is a genuine U-shaped minimum, not a flat/inert
knob. Surprising result vs plain Nekaon (no split, te 0.0846/gap +0.0058): the split at
``ratio=0.5`` **improves loss** (0.0789) at a gap cost (+0.0102) — this is a real loss<->gap
trade, not a strict quality regression, so the "low-VRAM mode" name undersells it; it may be
worth trying even when memory isn't the constraint. If gap matters more than loss,
``ratio=0.75`` (te 0.0801, gap +0.0073) is the pick. Threshold sweep at ratio=0.5: 32768 and
65536 tie for best loss (both 0.0789/+0.0102); 8192 is close behind with a tighter gap
(0.0795/+0.0087); 262144/2_000_000 (~no split on this proxy model) degrade back toward the
no-split number.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from kaon._autolr import DEFAULT_FUSE_REL, AutoLRMixin
from kaon.msam import MSAM

__all__ = ["Nekaon"]


class Nekaon(AutoLRMixin, MSAM):
    """Adakaon + k-step negative momentum-lookahead (zero-cost flat-minima bias).

    Args:
        params: parameters or param-group dicts.
        lr: learning rate (forwarded to the inner Adakaon).
        k: lookahead distance in **optimizer steps** (dimensionless, self-scaling;
            ``0`` disables the mechanism — plain Adakaon). The loss<->gap dial:
            larger ``k`` tightens the train-val gap, smaller ``k`` favors raw loss.
        betas: ``(beta1, beta2)`` of the inner Adakaon. **``beta1`` is the regime
            knob**: ``0.5`` (default) is the balanced mode that dominates the prior
            best loss+gap combo; ``0.2`` is the anti-memorization mode (field-record
            const-LR gap, the small-data LoRA pick); ``0.9`` is the fidelity mode
            (near-field-best const-LR loss; the lookahead is ~neutral there).
            ``beta1`` must be > 0 — the lookahead direction IS the momentum.
        weight_decay: decoupled weight decay. Default ``0.1`` (measured
            frontier-mover: improves loss and gap together on this backend).
        momentum_dtype: storage for the momentum buffer the lookahead rides.
            Default ``"4bit"`` (~0.56 B/param measured — the fine-tuning pick):
            the battery shows the quantized momentum carries the mechanism with
            no measurable loss vs bf16 (cte 0.0802 vs 0.0806, cgap +0.0066 vs
            +0.0056, both within the proxy's noise and under the gap target).
            ``"int8"`` (~1.04 B/param) and ``"bfloat16"`` (~2.03 B/param) are the
            higher-fidelity stops of the same measured-flat dial.
        low_vram_above: tensors with ``numel() > low_vram_above`` skip momentum and
            the lookahead (plain Adakaon at ``lr * low_vram_lr_ratio``) — see "Low-VRAM
            mode" above. Default ``None`` disables this (momentum + lookahead on every
            tensor, the original behavior). Not compatible with passing your own
            param-group dicts in ``params`` (this splits a flat param list itself).
        low_vram_lr_ratio: the low-VRAM group's LR relative to ``lr``, when
            ``low_vram_above`` is set. Default ``0.5``, matching the registry's
            precedent that momentum-free Adakaon wants roughly half of Nekaon's LR.
        **adakaon_kwargs: forwarded verbatim to the inner
            :class:`~kaon.adakaon.Adakaon` (``eps``, ``cautious``,
            ``gradient_centralization``, ``foreach``, ...).
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        k: float = 1.5,
        betas: tuple[float, float] = (0.5, 0.999),
        weight_decay: float = 0.1,
        momentum_dtype: str = "4bit",
        *,
        low_vram_above: int | None = None,
        low_vram_lr_ratio: float = 0.5,
        auto_lr: bool = False,
        auto_lr_scale: float = 1.0,
        auto_lr_fuse_rel: float = DEFAULT_FUSE_REL,
        auto_lr_d0: float | None = None,
        **adakaon_kwargs: Any,
    ) -> None:
        if k < 0.0:
            raise ValueError(f"k must be >= 0 (lookahead steps), got {k}")
        if betas[0] <= 0.0:
            raise ValueError("Nekaon requires betas[0] > 0 (the lookahead rides the momentum)")
        if low_vram_above is not None:
            if low_vram_above < 0:
                raise ValueError(f"low_vram_above must be >= 0, got {low_vram_above}")
            params = list(params)
            if params and isinstance(params[0], dict):
                raise TypeError(
                    "low_vram_above does not support param-group dicts — pass a flat "
                    "parameter iterable; Nekaon builds its own small/big groups by tensor size."
                )
            small = [p for p in params if p.numel() <= low_vram_above]
            big = [p for p in params if p.numel() > low_vram_above]
            if not small and not big:
                raise ValueError("Nekaon got no parameters")
            groups: list[dict[str, Any]] = []
            if small:
                groups.append({"params": small})
            if big:
                groups.append({"params": big, "lr": lr * low_vram_lr_ratio, "betas": (0.0, betas[1])})
            params = groups
        super().__init__(
            params,
            rho=-float(k),
            norm="none",
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            momentum_dtype=momentum_dtype,
            **adakaon_kwargs,
        )
        self.k = float(k)
        if auto_lr and low_vram_above is not None:
            raise ValueError(
                "Nekaon(auto_lr=True) is incompatible with low_vram_above: auto_lr forces one "
                "discovered LR on all groups each step, which would clobber the per-group "
                "low-VRAM lr ratio. Use one or the other."
            )
        # Composable parameter-free LR (update-space DoWG) via AutoLRMixin. off -> zero overhead.
        self._init_autolr(auto_lr, auto_lr_scale, auto_lr_fuse_rel, auto_lr_d0)

    # step() is the AutoLRMixin router; _step_impl is the full Nekaon step (SAM declimb ->
    # inner base -> climb) — DoWG measures the net displacement, so it composes over the lookahead.
    def _step_impl(self, closure: Any = None) -> Any:
        return MSAM.step(self, closure)  # explicit: super() would hit AutoLRMixin.step (the router)

    def state_dict(self) -> dict[str, Any]:
        return self._autolr_state_dict(super().state_dict())

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._autolr_load(state_dict, lambda sd: MSAM.load_state_dict(self, sd))
