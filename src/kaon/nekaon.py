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
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from kaon.msam import MSAM

__all__ = ["Nekaon"]


class Nekaon(MSAM):
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
        **adakaon_kwargs: Any,
    ) -> None:
        if k < 0.0:
            raise ValueError(f"k must be >= 0 (lookahead steps), got {k}")
        if betas[0] <= 0.0:
            raise ValueError("Nekaon requires betas[0] > 0 (the lookahead rides the momentum)")
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
