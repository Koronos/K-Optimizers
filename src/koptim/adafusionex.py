"""AdafusionEx — Adafusion's exact update plus the *generalization techniques*
our diffusion campaign found to actually move held-out loss, kept as a separate,
opt-in class so it can be A/B'd against stable :class:`~koptim.adafusion.Adafusion`.

Why this exists
---------------
On small-data diffusion LoRA, **train loss is the wrong objective** — minimizing
it is largely memorization, and the train↔val *gap* is the real signal. A
deep-research pass over the diffusion-*measured* literature found that the wins
which transfer are **optimizer-level techniques**, not a fancier base update:

1. **Weight averaging / EMA.** A shadow exponential moving average of the weights
   is, empirically, the single most reliable generalization knob for diffusion
   fine-tuning (it is standard in essentially every strong diffusion training
   recipe). It costs one extra weight-sized buffer and nothing at train time but
   a cheap lerp.
2. **Flat-minima seeking (SAM family).** Perturbing toward the local worst case
   before taking the step biases optimization toward flat minima that generalize
   better. The memory-free variant **MSAM** (Momentum-SAM, Becker et al.,
   arXiv:2401.12033) perturbs along the *existing momentum buffer* rather than the
   fresh gradient, so it needs **zero extra state**.

AdafusionEx is the **vehicle** for running those techniques on top of Adafusion's
already-regularizing factored update, without destabilizing the shipped
Adafusion. With ``ema_decay=0.0`` and ``sam_mode=None`` it is **bit-comparable to
Adafusion** (the A/B baseline): ``step()`` is literally ``Adafusion.step()`` plus
no-ops.

Techniques & citations
----------------------
* **EMA of weights** — diffusers' ``EMAModel`` API shape (``store`` / ``copy_to`` /
  ``restore``) is mirrored so it drops into existing eval/sampling loops.
* **Post-hoc EMA** (optional) — EDM2, Karras et al., *Analyzing and Improving the
  Training Dynamics of Diffusion Models*, arXiv:2312.02696. Instead of committing
  to one decay during training, periodic **parameter snapshots** are stored so an
  EMA profile (effective length) can be reconstructed *after* training. We keep
  the cheap version: store snapshots; reconstruct a decayed average on request.
  (The full EDM2 power-function least-squares solve over two parallel EMAs is out
  of scope for v1 — see :meth:`reconstruct_posthoc_ema`.)
* **MSAM** — Momentum-SAM, Becker, Altrichter & Igel, arXiv:2401.12033. Perturb
  ``p -> p - rho * m/||m||`` along the (Adafusion) momentum, recompute the
  gradient there, then take the normal Adafusion step from the *un-perturbed*
  point. Zero extra memory (reuses the momentum buffer).
* **Caveat — Bi-LoRA**, arXiv:2508.19564: naive SAM-on-LoRA is *weak* (the two
  low-rank factors let the adversarial perturbation be cancelled), and Bi-LoRA
  argues for a bilevel adversary instead. So flat-minima mode here is **OFF by
  default** and explicitly experimental; treat any LoRA SAM result as suspect
  until measured against the EMA-only baseline.

Composition
-----------
:class:`AdafusionEx` **subclasses** :class:`~koptim.adafusion.Adafusion` and
reuses its update *verbatim* — ``step()`` calls ``super().step()`` for the core
factored move and only layers EMA / SAM bookkeeping around it. Nothing in
``adafusion.py`` is touched, so the inner update can never silently diverge from
the shipped optimizer.

EMA math
--------
Per tracked parameter ``p`` with shadow ``e``::

    e <- decay * e + (1 - decay) * p          (after the optimizer step)

with an optional warmup that ramps the effective decay so early, noisy weights
are not over-trusted (diffusers convention)::

    decay_t = min(decay, (1 + step) / (10 + step))

``store()`` snapshots the live params, ``copy_to()`` writes the EMA (or any given
params) into the live model for eval, ``restore()`` puts the snapshot back.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any, Literal

import torch
from torch import Tensor

from koptim.adafusion import Adafusion, MomentumDtype

__all__ = ["AdafusionEx"]

EmaDtype = Literal["bfloat16", "float32"]
SamMode = Literal["msam"]

_EMA_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


class AdafusionEx(Adafusion):
    """Adafusion + first-class weight EMA + optional MSAM flat-minima mode.

    With ``ema_decay=0.0`` and ``sam_mode=None`` this is Adafusion (bit-comparable
    on CPU fp32): the update is inherited verbatim and the added machinery is a
    no-op. See the module docstring for the techniques and citations.

    Args:
        params: parameters or param-group dicts (forwarded to Adafusion).
        ema_decay: EMA decay for the weight shadow. ``0.0`` (default) disables EMA
            entirely (no buffer allocated). A typical diffusion value is ``0.999``
            – ``0.9999``.
        ema_dtype: storage dtype for the EMA shadow — ``"bfloat16"`` (default,
            ~2 B/param) or ``"float32"`` (4 B/param, exact). The EMA math is always
            done in fp32 then cast back, so bf16 storage is the only precision loss.
        ema_warmup: ramp the effective decay early via
            ``min(ema_decay, (1+t)/(10+t))`` so the shadow is not dominated by the
            first noisy steps (diffusers convention). Default ``True``.
        posthoc_ema: if ``True``, also retain periodic parameter snapshots so an EMA
            profile can be chosen *after* training (EDM2, arXiv:2312.02696). Costs a
            stored copy of the params every ``posthoc_interval`` steps (CPU by
            default to spare VRAM). Independent of ``ema_decay``.
        posthoc_interval: steps between post-hoc snapshots (default ``100``).
        posthoc_device: device for stored snapshots (default ``"cpu"``).
        sam_mode: flat-minima mode. ``None`` (default) → off. ``"msam"`` →
            Momentum-SAM (arXiv:2401.12033): zero-extra-memory perturbation along
            the momentum buffer. Requires ``betas[0] > 0`` (a momentum buffer must
            exist) and a cooperating training loop (see :meth:`first_step` /
            :meth:`second_step` or ``step(closure)``). **Experimental**; see the
            Bi-LoRA caveat (arXiv:2508.19564) in the module docstring.
        sam_rho: MSAM perturbation radius. ``0.0`` (default) → perturbation is a
            no-op even if ``sam_mode="msam"`` (so it stays bit-comparable). Typical
            values are ``0.1`` – ``1.0`` (MSAM tolerates larger rho than vanilla
            SAM because the momentum direction is smoother).
        **adafusion_kwargs: every Adafusion option (``lr``, ``betas``, ``eps``,
            ``weight_decay``, ``clip_threshold``, ``momentum_dtype``, ``cautious``,
            ``bf16_method``, ``foreach`` ...) forwards through unchanged.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: tuple[float, float] = (1e-30, 1e-3),
        weight_decay: float = 0.0,
        *,
        clip_threshold: float = 1.0,
        momentum_dtype: MomentumDtype = "bfloat16",
        cautious: bool = True,
        bf16_method: str = "stochastic_rounding",
        foreach: bool = True,
        ema_decay: float = 0.0,
        ema_dtype: EmaDtype = "bfloat16",
        ema_warmup: bool = True,
        posthoc_ema: bool = False,
        posthoc_interval: int = 100,
        posthoc_device: str | torch.device = "cpu",
        sam_mode: SamMode | None = None,
        sam_rho: float = 0.0,
        **adafusion_kwargs: Any,
    ) -> None:
        if not 0.0 <= ema_decay <= 1.0:
            raise ValueError(f"ema_decay must be in [0, 1], got {ema_decay}")
        if ema_dtype not in _EMA_DTYPES:
            raise ValueError(f"ema_dtype must be bfloat16/float32, got {ema_dtype!r}")
        if posthoc_interval < 1:
            raise ValueError(f"posthoc_interval must be >= 1, got {posthoc_interval}")
        if sam_mode not in (None, "msam"):
            raise ValueError(f"sam_mode must be None or 'msam', got {sam_mode!r}")
        if sam_rho < 0.0:
            raise ValueError(f"sam_rho must be >= 0, got {sam_rho}")
        if sam_mode == "msam" and betas[0] <= 0.0:
            raise ValueError("sam_mode='msam' needs betas[0] > 0 (it perturbs along momentum)")

        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            clip_threshold=clip_threshold,
            momentum_dtype=momentum_dtype,
            cautious=cautious,
            bf16_method=bf16_method,
            foreach=foreach,
            **adafusion_kwargs,
        )

        self.ema_decay = float(ema_decay)
        self._ema_dtype = _EMA_DTYPES[ema_dtype]
        self.ema_warmup = bool(ema_warmup)
        self.ema_step = 0  # number of EMA updates applied (drives the warmup ramp)

        self.posthoc_ema = bool(posthoc_ema)
        self.posthoc_interval = int(posthoc_interval)
        self.posthoc_device = torch.device(posthoc_device)
        #: list of (ema_step, [fp32 snapshot per param]) — see reconstruct_posthoc_ema.
        self._posthoc_snapshots: list[tuple[int, list[Tensor]]] = []

        self.sam_mode = sam_mode
        self.sam_rho = float(sam_rho)
        self._sam_perturbed = False  # guards first_step/second_step ordering

        #: EMA shadow per param, seeded from the initial params (so the average
        #: starts at the pre-training weights, matching diffusers / the closed-form
        #: ``e_0 = p_0``). Empty when EMA is disabled.
        self._ema_shadow: dict[Tensor, Tensor] = {}
        if self.ema_enabled:
            self._ensure_shadow()
        #: store()/restore() backup of the live params.
        self._param_backup: dict[Tensor, Tensor] = {}

    # ------------------------------------------------------------------ params
    def _all_params(self) -> list[Tensor]:
        return [p for group in self.param_groups for p in group["params"]]

    @property
    def ema_enabled(self) -> bool:
        return self.ema_decay > 0.0

    # --------------------------------------------------------------------- EMA
    @torch.no_grad()
    def _ensure_shadow(self) -> None:
        for p in self._all_params():
            if p not in self._ema_shadow:
                self._ema_shadow[p] = p.detach().to(self._ema_dtype).clone()

    def _effective_decay(self) -> float:
        if not self.ema_warmup:
            return self.ema_decay
        warm = (1 + self.ema_step) / (10 + self.ema_step)
        return min(self.ema_decay, warm)

    @torch.no_grad()
    def update_ema(self) -> None:
        """Advance the weight EMA one step: ``e <- d*e + (1-d)*p``.

        Called automatically at the end of :meth:`step` when ``ema_decay > 0``. The
        lerp is computed in the shadow's dtype (``lerp_`` upcasts internally) and the
        result lands back in ``ema_dtype``, so bf16 storage is the only loss. A
        no-op when EMA is disabled.
        """
        if not self.ema_enabled:
            return
        self._ensure_shadow()
        decay = self._effective_decay()
        for p in self._all_params():
            e = self._ema_shadow[p]
            e.lerp_(p.detach().to(e.dtype), 1.0 - decay)
        self.ema_step += 1
        if self.posthoc_ema and (self.ema_step % self.posthoc_interval == 0):
            self._take_posthoc_snapshot()

    @torch.no_grad()
    def store(self, params: Iterable[Tensor] | None = None) -> None:
        """Snapshot the live params so :meth:`restore` can put them back.

        Mirrors diffusers' ``EMAModel.store``. Typical use: ``store()`` →
        ``copy_to()`` (eval on EMA weights) → ``restore()``.
        """
        plist = list(params) if params is not None else self._all_params()
        self._param_backup = {p: p.detach().clone() for p in plist}

    @torch.no_grad()
    def copy_to(self, params: Iterable[Tensor] | None = None) -> None:
        """Copy the EMA shadow into the live params (for eval/sampling).

        Mirrors diffusers' ``EMAModel.copy_to``. Raises if EMA was never built.
        """
        if not self._ema_shadow:
            raise RuntimeError("no EMA shadow to copy (ema_decay=0 or no step taken)")
        plist = list(params) if params is not None else self._all_params()
        for p in plist:
            e = self._ema_shadow.get(p)
            if e is not None:
                p.data.copy_(e.to(p.dtype))

    @torch.no_grad()
    def restore(self, params: Iterable[Tensor] | None = None) -> None:
        """Restore params saved by :meth:`store` (undo a :meth:`copy_to`)."""
        if not self._param_backup:
            raise RuntimeError("call store() before restore()")
        plist = list(params) if params is not None else list(self._param_backup)
        for p in plist:
            backup = self._param_backup.get(p)
            if backup is not None:
                p.data.copy_(backup)
        self._param_backup = {}

    # --------------------------------------------------------- post-hoc EMA
    @torch.no_grad()
    def _take_posthoc_snapshot(self) -> None:
        snap = [p.detach().to(self.posthoc_device, torch.float32).clone() for p in self._all_params()]
        self._posthoc_snapshots.append((self.ema_step, snap))

    @torch.no_grad()
    def reconstruct_posthoc_ema(self, decay: float) -> list[Tensor]:
        """Reconstruct a weight EMA *after* training from stored snapshots.

        Post-hoc EMA (EDM2, arXiv:2312.02696): rather than commit to one decay
        during training, snapshots were retained so a profile can be picked later.
        This is the **cheap** reconstruction — a decayed weighted average of the
        stored snapshots with weight ``decay**(t_last - t_i)`` per snapshot — not
        EDM2's full power-function least-squares fit over two parallel EMAs (that
        needs the dual-EMA recording, out of scope for v1).

        Returns one fp32 tensor per param (in param order); does **not** mutate the
        live model. ``posthoc_ema=True`` must have been set.
        """
        if not self.posthoc_ema:
            raise RuntimeError("posthoc_ema=False; no snapshots were recorded")
        if not self._posthoc_snapshots:
            raise RuntimeError("no post-hoc snapshots recorded yet")
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        t_last = self._posthoc_snapshots[-1][0]
        weights = [decay ** (t_last - t) for t, _ in self._posthoc_snapshots]
        wsum = sum(weights)
        n_params = len(self._posthoc_snapshots[0][1])
        out: list[Tensor] = []
        for j in range(n_params):
            acc = torch.zeros_like(self._posthoc_snapshots[0][1][j])
            for w, (_, snap) in zip(weights, self._posthoc_snapshots, strict=True):
                acc.add_(snap[j], alpha=w)
            out.append(acc.div_(wsum))
        return out

    # --------------------------------------------------------------- MSAM/SAM
    @torch.no_grad()
    def _momentum_dirs(self) -> tuple[dict[Tensor, Tensor], float]:
        """Dequantized fp32 momentum per param + the global L2 norm of all of them.

        The direction MSAM perturbs along. Uses the codec so int8/4bit momentum is
        dequantized transparently. Params without a momentum buffer yet (e.g. before
        the first step) are skipped — perturbation is then a no-op for them.
        """
        dirs: dict[Tensor, Tensor] = {}
        sq = 0.0
        for group in self.param_groups:
            beta1 = group["betas"][0]
            if beta1 <= 0:
                continue
            codec = self._codec(group)
            for p in group["params"]:
                state = self.state.get(p)
                if not state or "m" not in state:
                    continue
                d = codec.dequant_one(state, p)
                dirs[p] = d
                sq += float((d * d).sum())  # (a*b).sum(): torch.dot SIGFPEs on CUDA here
        return dirs, sq ** 0.5

    @torch.no_grad()
    def first_step(self) -> None:
        """MSAM: perturb params along the momentum direction (zero extra memory).

        Moves ``p -> p - rho * m/||m||`` (global-norm normalized, as in MSAM,
        arXiv:2401.12033). The training loop must then recompute the loss+grad at
        this perturbed point before calling :meth:`second_step`. A no-op (other than
        an ordering flag) when ``sam_mode`` is not ``"msam"`` or ``sam_rho == 0``.
        """
        if self.sam_mode == "msam" and self.sam_rho != 0.0:
            dirs, norm = self._momentum_dirs()
            if norm > 0:
                scale = self.sam_rho / (norm + 1e-12)
                for p, d in dirs.items():
                    p.data.sub_(d.to(p.dtype), alpha=scale)
        self._sam_perturbed = True

    @torch.no_grad()
    def second_step(self, closure: Any = None) -> Any:
        """MSAM: un-perturb, then take the normal Adafusion step.

        Restores ``p`` to the un-perturbed point (undoing :meth:`first_step`) and
        runs the standard Adafusion update with the gradient that was recomputed at
        the perturbed point. Must be preceded by :meth:`first_step`. The momentum
        buffer is unchanged between the two calls, so the same direction/scale that
        :meth:`first_step` subtracted is added back exactly.
        """
        if not self._sam_perturbed:
            raise RuntimeError("call first_step() before second_step()")
        if self.sam_mode == "msam" and self.sam_rho != 0.0:
            dirs, norm = self._momentum_dirs()
            if norm > 0:
                scale = self.sam_rho / (norm + 1e-12)
                for p, d in dirs.items():
                    p.data.add_(d.to(p.dtype), alpha=scale)
        self._sam_perturbed = False
        return self.step(closure)

    # -------------------------------------------------------------------- step
    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        """Adafusion's update verbatim, then advance the weight EMA.

        The core factored move is :meth:`Adafusion.step` unchanged; only the EMA
        update is layered on afterward. With ``ema_decay=0`` this is exactly
        Adafusion (the EMA call returns immediately).
        """
        loss = super().step(closure)
        self.update_ema()
        return loss

    # ------------------------------------------------------------ checkpointing
    def state_dict(self) -> dict[str, Any]:
        """Adafusion's state plus the EMA shadow / post-hoc snapshots.

        The EMA shadow and snapshots are stored in **stable param order** (Tensor
        identity does not survive (de)serialization), exactly like Autofusion's
        ``ref`` handling. The shadow is cloned so a live optimizer that keeps
        training cannot corrupt a separately-loaded copy.
        """
        params = self._all_params()
        index = {p: i for i, p in enumerate(params)}
        shadow_list: list[Tensor | None] = [None] * len(params)
        for p, e in self._ema_shadow.items():
            if p in index:
                shadow_list[index[p]] = e.detach().clone()
        return {
            "adafusion": copy.deepcopy(super().state_dict()),
            "ema_shadow": shadow_list,
            "ema_step": self.ema_step,
            "ema_decay": self.ema_decay,
            "posthoc_snapshots": [
                (t, [s.detach().clone() for s in snap]) for t, snap in self._posthoc_snapshots
            ],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore Adafusion state (dtype-preserving) plus the EMA shadow."""
        super().load_state_dict(state_dict["adafusion"])
        params = self._all_params()
        self._ema_shadow = {}
        for i, e in enumerate(state_dict.get("ema_shadow", [])):
            if e is not None and i < len(params):
                p = params[i]
                self._ema_shadow[p] = e.to(device=p.device)
        self.ema_step = int(state_dict.get("ema_step", 0))
        if "ema_decay" in state_dict:
            self.ema_decay = float(state_dict["ema_decay"])
        self._posthoc_snapshots = [
            (t, [s.clone() for s in snap]) for t, snap in state_dict.get("posthoc_snapshots", [])
        ]
        self._param_backup = {}
