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

from kaon._backend import foreach_budget
from kaon._stochastic_rounding import add_stochastic_
from kaon._wrappers import WrapsInnerOptimizer

__all__ = ["SAM"]

# Stacked climb transient: weight + grad + perturbation (+ fp32 intermediates).
_STACK_BYTES_PER_ELEM = 32


class SAM(WrapsInnerOptimizer, Optimizer):
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
        # break its step. (WrapsInnerOptimizer's separate state also keeps SAM's transient
        # snapshot out of the inner Adam state.) ``rho``/``adaptive`` are SAM-only and safe
        # to keep per-group, so a param-group override of rho works.
        self.eps = float(eps)

        # WrapsInnerOptimizer builds nothing — we build the inner base optimizer and bind it
        # (shared param_groups, separate per-param wrapper state, delegated zero_grad /
        # state_dict). ``base_optimizer`` stays as a public alias of the bound ``inner``.
        self._bind_inner(base_optimizer(params, **kwargs), state_key="sam")
        self.base_optimizer = self.inner
        # SAM's OWN per-group hyperparameters (not part of the inner optimizer's
        # ``defaults``, per the ``eps`` note above). Kept as ``self.defaults`` — mirroring
        # the attribute every plain ``torch.optim.Optimizer`` carries — purely so
        # ``load_state_dict`` can backfill a key an older checkpoint predates the same
        # way every other kaon optimizer does.
        self.defaults: dict[str, Any] = {"rho": float(rho), "adaptive": bool(adaptive)}
        for group in self.param_groups:
            for key, value in self.defaults.items():
                group.setdefault(key, value)

    # ------------------------------------------------------------------ norm
    @torch.no_grad()
    def _grad_norm(self) -> Tensor:
        """Global L2 norm of the gradient over all params, ``sqrt(sum_i ||g_i||^2)``.

        Batched via ``torch._foreach_norm(..., dtype=torch.float32)``: every per-tensor
        norm is ACCUMULATED and returned in fp32 even for bf16 grads (without the
        ``dtype`` the per-tensor norms came back as bf16 scalars — 8 mantissa bits, up to
        ~2e-3 relative error — before any ``.float()`` could help), then stacked on one
        device and reduced in fp32. The returned norm, and so ``scale = rho / (norm +
        eps)``, is fp32 whatever the weight/grad dtype. With ``adaptive=True`` each
        gradient is scaled by ``|w|`` first (ASAM); that product is materialized one
        stack-budget chunk at a time, never for every param at once (+1x the weights).
        Params may live on different devices: the per-tensor norms are moved to the
        first one's device before the final reduction.
        """
        norms: list[Tensor] = []
        plain: list[Tensor] = []
        for group in self.param_groups:
            with_grad = [p for p in group["params"] if p.grad is not None]
            if not with_grad:
                continue
            if not group["adaptive"]:
                plain.extend(p.grad for p in with_grad)
                continue
            budget = self._chunk_budget(with_grad)
            chunk: list[Tensor] = []
            size = 0
            for p in with_grad:
                if chunk and size + p.numel() > budget:
                    norms.extend(self._adaptive_norms(chunk))
                    chunk, size = [], 0
                chunk.append(p)
                size += p.numel()
            norms.extend(self._adaptive_norms(chunk))
        if plain:
            norms = list(torch._foreach_norm(plain, 2, dtype=torch.float32)) + norms  # type: ignore[attr-defined]
        if not norms:
            # No grad anywhere: nothing will climb, so the device is immaterial (and the
            # first group's param list may itself be empty).
            return torch.zeros(())
        dev = norms[0].device
        return torch.linalg.vector_norm(torch.stack([n.to(dev) for n in norms]))

    @staticmethod
    def _adaptive_norms(chunk: list[Tensor]) -> list[Tensor]:
        """fp32 per-tensor norms of ASAM's ``|w| * g`` for one bounded chunk."""
        return list(torch._foreach_norm([p.abs() * p.grad for p in chunk], 2,  # type: ignore[attr-defined]
                                        dtype=torch.float32))

    @staticmethod
    def _bucket_params(params: list[Tensor]) -> dict[tuple[Any, ...], list[Tensor]]:
        buckets: dict[tuple[Any, ...], list[Tensor]] = {}
        for p in params:
            buckets.setdefault((tuple(p.shape), p.dtype, p.device), []).append(p)
        return buckets

    def _chunk_budget(self, plist: list[Tensor]) -> int:
        return foreach_budget(
            self._foreach_stack_budget,
            self._foreach_batch_cutoff,
            _STACK_BYTES_PER_ELEM,
            plist[0].device,
        )

    @torch.no_grad()
    def _climb_chunk(
        self,
        plist: list[Tensor],
        *,
        scale: Tensor,
        adaptive: bool,
    ) -> None:
        """Climb one same-(shape, dtype, device) chunk: ``w += e`` and snapshot ``old_p``.

        Every route computes ``e = (g * scale) [* (w * w)]`` with the same ops in the same
        order and writes it with the same primitive, so all three are bit-identical to the
        historical stack -> clone -> add -> copy-back (including the stochastic-rounding
        stream: the torch SR draw depends only on the element count and logical order).
        What differs is the transient: the historical route held a weight stack, its
        clone, a grad stack and ``e`` (~4-5x the chunk) and paid a copy-back.

        * fp32 weights + fp32 grads: no stack at all — ``_foreach`` ops in place on the
          live weights; only the ``old_p`` snapshot and ``e`` are allocated.
        * a single contiguous low-precision weight (a unique shape, or one above the stack
          budget — e.g. a 3072x3072 DiT MLP): stacking one tensor is pure overhead, so it
          climbs in place.
        * otherwise (bf16 buckets of N >= 2) one stacked SR write, with ``e`` computed in
          place in the grad stack.
        """
        s = scale.to(plist[0].device)
        pdata = [p.data for p in plist]
        grads = [p.grad for p in plist]
        if plist[0].dtype == torch.float32 and all(g.dtype == torch.float32 for g in grads):
            olds = [pdata[0].clone()] if len(pdata) == 1 else list(torch.stack(pdata).unbind(0))
            e = torch._foreach_mul(grads, s)  # type: ignore[attr-defined]
            if adaptive:
                torch._foreach_mul_(e, torch._foreach_mul(pdata, pdata))  # type: ignore[attr-defined]
            torch._foreach_add_(pdata, e)  # type: ignore[attr-defined]
        elif len(pdata) == 1 and pdata[0].is_contiguous():
            w = pdata[0]
            olds = [w.clone()]
            e1 = grads[0] * s
            if adaptive:
                e1 = e1 * (w * w)
            add_stochastic_(w, e1, alpha=1.0, sr=self.sr_stream)
        else:
            weights = torch.stack(pdata)
            olds = list(weights.clone().unbind(0))
            e_w = torch.stack(grads).mul_(s)
            if adaptive:
                ww = weights * weights
                # in place only when it cannot narrow (bf16 grads on fp32 weights promote)
                e_w = e_w.mul_(ww) if e_w.dtype == ww.dtype else e_w * ww
            add_stochastic_(weights, e_w, alpha=1.0, sr=self.sr_stream)
            torch._foreach_copy_(pdata, list(weights.unbind(0)))  # type: ignore[attr-defined]
        for p, old in zip(plist, olds, strict=True):
            self.state[p]["old_p"] = old

    # ------------------------------------------------------------------ pass 1
    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Pass 1: climb to ``w + e(w)`` and snapshot ``w`` for the restore.

        Requires ``p.grad`` already populated (the loop did the first backward). For each
        param: ``e = scale * (w^2 if adaptive else 1) * g`` with
        ``scale = rho / (global_grad_norm + eps)``; the climb ``w += e`` is bf16-correct.

        A previous ``first_step`` whose ``second_step`` never ran (an aborted step: the
        perturbed forward was skipped, raised, or produced a non-finite loss) is undone
        FIRST — its ``old_p`` snapshot is copied back — so this climb starts from, and
        snapshots, the true weights. Without that the new ``old_p`` was the perturbed
        weight and the first climb stayed in the model for good (measured 1.9e-2 max
        drift in fp32). Restoring rather than raising keeps the "skip this batch and
        carry on" loop pattern working; the true weights can never be lost.
        """
        self._restore_climb()
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            adaptive = group["adaptive"]
            scale = group["rho"] / (grad_norm + self.eps)
            with_grad = [p for p in group["params"] if p.grad is not None]
            for plist in self._bucket_params(with_grad).values():
                budget = self._chunk_budget(plist)
                n_per = max(1, budget // max(p.numel() for p in plist))
                for i in range(0, len(plist), n_per):
                    self._climb_chunk(plist[i:i + n_per], scale=scale, adaptive=adaptive)
        self._climbed = True
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
        self._restore_climb(force=True)
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def _restore_climb(self, force: bool = False) -> None:
        """Copy every pending ``old_p`` snapshot back into its weight (exact) and drop it.

        ``first_step`` only walks the params when a climb is known to be pending (the
        flag, which ``load_state_dict`` re-derives from the restored state), so the
        normal ``first_step``/``second_step`` cycle pays nothing extra; ``second_step``
        always walks them, as it always has."""
        if not (force or getattr(self, "_climbed", False)):
            return
        state = self.state
        for group in self.param_groups:
            for p in group["params"]:
                st = state.get(p)
                old_p = st.pop("old_p", None) if st else None
                if old_p is not None:
                    p.data.copy_(old_p)
        self._climbed = False

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
        """Restore the inner base optimizer (dtype-preserving) via the wrapper mixin.

        WrapsInnerOptimizer's ``state_dict`` already saves the inner optimizer's full
        state (SAM keeps no persistent state of its own), so this round-trips the base
        optimizer's momentum/factored state — unlike the previous bare ``super()`` call.

        ``inner.load_state_dict`` backfills the INNER optimizer's own defaults (every
        kaon optimizer's ``load_state_dict`` does this now); backfill SAM's own
        ``rho``/``adaptive`` here too, since those live in the same shared
        ``param_groups`` dict but are not part of the inner optimizer's ``defaults``.
        """
        self._load_wrapped(state_dict, lambda inner, sd: inner.load_state_dict(sd))
        self.base_optimizer = self.inner
        # A checkpoint taken between first_step and second_step carries old_p snapshots.
        self._climbed = any("old_p" in st for st in self.state.values())
        for group in self.param_groups:
            for key, value in self.defaults.items():
                group.setdefault(key, value)
