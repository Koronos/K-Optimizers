"""SAM — Sharpness-Aware Minimization (Foret et al. 2021, arXiv:2010.01412).

A meta-optimizer that **wraps a base optimizer** (here :class:`~kaon.adakaon.Adakaon`,
the kaon flagship) and replaces its gradient with one computed at a *worst-case nearby
point* in weight space. The intuition (and the reason it lives in a diffusion
fine-tuning library): minimizing the loss in a small *neighborhood* steers training
toward **flat minima**, which generalize better — i.e. it targets the train-val GAP, the
objective that actually matters here, not low train loss. The published technique flagged
by the deep-research as the strongest single lever on the generalization Pareto frontier.

The price is ~2× compute: every optimizer step needs **two** forward/backward passes
(one at ``w`` to find the perturbation, one at the perturbed ``w + e(w)`` for the real
update). That is acceptable in this library, where sample quality / gap dominates per-step
speed.

The two-pass step (standard SAM)::

    # pass 1: g = grad of loss at w
    e(w) = rho * g / (||g||_2 + eps)     # ascend to the worst-case nearby point
    w <- w + e(w)                        # "climb"
    # pass 2: g~ = grad of loss at w + e(w)   (zero_grad, backward again, SAME batch)
    w <- w - e(w)                        # restore the original w
    base_opt.step()  using g~            # the BASE optimizer steps with the perturbed grad

``||g||_2`` is the **global** L2 norm over *all* params (standard SAM). With
``adaptive=True`` (ASAM, Kwon et al. 2021, arXiv:2102.11600) the perturbation and the norm
are scaled per-weight by ``|w|`` (norm uses ``|w|·g``; the perturbation uses ``w²·g``),
which makes the sharpness measure scale-invariant.

API — the training loop must drive the two passes (this is NOT a drop-in
``torch.optim.Optimizer`` like the rest of kaon; cf. Schedule-Free's train/eval methods,
which the loop must also call):

    loss = batch_loss(...); loss.backward()
    opt.first_step(zero_grad=True)        # climb to w+e, store e/old_p, zero grad
    loss2 = batch_loss(... SAME batch/noise ...); loss2.backward()
    opt.second_step(zero_grad=True)       # restore w, run base_opt.step() with g~

or, equivalently, a single ``opt.step(closure)`` where ``closure`` recomputes
loss+backward at the perturbed point::

    def closure():
        opt.zero_grad(); loss = batch_loss(...); loss.backward(); return loss
    loss = batch_loss(...); loss.backward()   # first pass grad must already be present
    opt.step(closure)

**bf16-correctness.** The climb ``w += e`` is written through the kaon stochastic-rounding
primitive (``add_stochastic_``) so a bf16 / SR weight is not corrupted by truncation during
the perturbation. The restore is **exact** for every dtype — ``first_step`` snapshots the
pre-climb weight (``old_p``) and ``second_step`` copies it back — so the climb→restore
round-trip leaves a low-precision weight bit-identical when the base step is skipped (no
drift), which a naive ``w += e; w -= e`` with two independent SR draws would *not*
guarantee. The base optimizer then performs its own bf16-correct write at ``w``.

**Memory.** SAM adds no *persistent* optimizer state of its own. During a step it holds one
weight-sized snapshot per param (``old_p``, freed/overwritten each step) — i.e. peak extra
≈ 1× the trainable weights for the duration of the step, on top of whatever the base
optimizer keeps. (The perturbation ``e`` itself is materialized transiently per-param and
not retained.)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from kaon._stochastic_rounding import add_stochastic_

__all__ = ["SAM"]


class SAM(Optimizer):
    """Sharpness-Aware Minimization wrapping a base kaon optimizer.

    Args:
        params: parameters or param-group dicts (shared with the base optimizer).
        base_optimizer: the optimizer **class** to wrap (default
            :class:`~kaon.adakaon.Adakaon`). It is instantiated internally over the same
            param groups; its kwargs are forwarded via ``**kwargs``.
        rho: neighborhood radius — the L2 size of the ascent step ``e(w)``. Default
            ``0.05`` (the SAM paper default).
        adaptive: ``True`` enables ASAM (per-weight ``|w|`` scaling of both the norm and the
            perturbation), making the sharpness measure scale-invariant. Default ``False``
            (standard SAM).
        eps: numerical floor added to the global gradient norm before dividing. Default
            ``1e-12`` (matches the official ``davda54/sam``).
        **kwargs: forwarded verbatim to ``base_optimizer`` (e.g. ``lr``, ``betas``,
            ``cautious``, ``momentum_dtype``, ``bf16_method``, ``foreach``,
            ``gradient_centralization``).

    The loop must call ``first_step`` then (recompute grad) then ``second_step``, or the
    closure form ``step(closure)``. See the module docstring for the exact sequence.
    """

    def __init__(
        self,
        params: Iterable[Any],
        base_optimizer: type[Optimizer] | None = None,
        rho: float = 0.05,
        adaptive: bool = False,
        eps: float = 1e-12,
        **kwargs: Any,
    ) -> None:
        if rho < 0.0:
            raise ValueError(f"rho must be >= 0, got {rho}")
        if eps < 0.0:
            raise ValueError(f"eps must be >= 0, got {eps}")
        if base_optimizer is None:
            # Default base = the kaon flagship. Imported lazily to avoid a module-load
            # cycle (adakaon -> _backend -> ... never imports sam, but keep it lean).
            from kaon.adakaon import Adakaon

            base_optimizer = Adakaon

        self.rho = float(rho)
        self.adaptive = bool(adaptive)
        # SAM's own eps is an *instance* attribute, deliberately NOT a per-group default:
        # the base optimizer (Adakaon) also has an ``eps`` key (a tuple), and torch's
        # ``Optimizer.__init__`` fills only *missing* defaults into pre-existing param
        # groups — so a scalar ``eps`` planted by SAM would shadow Adakaon's tuple and
        # break its step. ``rho``/``adaptive`` are SAM-only and safe to keep per-group
        # (so a param-group override of rho works).
        self.eps = float(eps)

        defaults = {"rho": float(rho), "adaptive": bool(adaptive), **kwargs}
        super().__init__(params, defaults)

        # Build the inner base optimizer over the SAME param groups, then alias our
        # param_groups to its so the two stay in lock-step (lr schedules etc. drive both).
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    # ------------------------------------------------------------------ norm
    @torch.no_grad()
    def _grad_norm(self) -> Tensor:
        """Global L2 norm of the gradient over all params, ``sqrt(sum_i ||g_i||^2)``.

        Computed via ``(g*g).sum()`` rather than ``torch.dot``/``.norm()`` — on this GPU
        ``torch.dot`` SIGFPEs, and ``Tensor.norm`` dispatches to a dot for contiguous
        tensors. With ``adaptive=True`` each gradient is scaled by ``|w|`` first (ASAM).
        """
        sq_sum: Tensor | None = None
        for group in self.param_groups:
            adaptive = group["adaptive"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if adaptive:
                    g = p.abs() * g
                s = (g * g).sum()
                sq_sum = s if sq_sum is None else sq_sum + s
        if sq_sum is None:
            # No grads at all — return a scalar zero on a sensible device.
            dev = self.param_groups[0]["params"][0].device
            return torch.zeros((), device=dev)
        return sq_sum.sqrt()

    # ------------------------------------------------------------------ pass 1
    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Pass 1: climb to ``w + e(w)`` and snapshot ``w`` for the restore.

        Requires ``p.grad`` already populated (the loop did the first backward). For each
        param: ``e = scale * (w^2 if adaptive else 1) * g`` with
        ``scale = rho / (global_grad_norm + eps)``; the climb ``w += e`` is bf16-correct.
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            adaptive = group["adaptive"]
            scale = group["rho"] / (grad_norm + self.eps)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Exact restore for any dtype: snapshot the pre-climb weight.
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p.device)
                if adaptive:
                    e_w = e_w * (p.data * p.data)
                # bf16-correct climb: stochastic-round the perturbation into the weight
                # (no-op fast path for fp32; SR bit-trick for bf16) instead of a truncating
                # ``p.add_(e_w)`` that would drop sub-ULP perturbation on low-precision weights.
                add_stochastic_(p.data, e_w, alpha=1.0)
        if zero_grad:
            self.zero_grad()

    # ------------------------------------------------------------------ pass 2
    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Pass 2: restore the original ``w`` (exactly), then run the base optimizer step.

        The base optimizer reads ``p.grad`` — which the loop recomputed at ``w + e(w)``
        between the two calls — and performs its own (bf16-correct) update at the restored
        ``w``. The restore is an exact ``copy_`` of the pre-climb snapshot, so no climb
        rounding leaks into the final weights.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                old_p = self.state[p].pop("old_p", None)
                if old_p is not None:
                    p.data.copy_(old_p)
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    # ------------------------------------------------------------------ combined
    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None) -> Any:  # type: ignore[override]
        """Run a full SAM step. ``closure`` must recompute loss+backward at the perturbed
        point (it is called between ``first_step`` and ``second_step``).

        The first-pass gradient (at ``w``) must already be present on entry — call
        ``loss.backward()`` before ``step``, exactly as the manual two-pass loop does.
        Returns the closure's value (typically the perturbed-point loss).
        """
        if closure is None:
            raise RuntimeError(
                "SAM.step requires a closure that recomputes loss+backward at the perturbed "
                "point; or drive first_step()/second_step() manually (see module docstring)."
            )
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        loss = closure()
        self.second_step()
        return loss

    # ------------------------------------------------------------------ state
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        # Keep the inner optimizer pointed at the (restored) shared param groups.
        self.base_optimizer.param_groups = self.param_groups
