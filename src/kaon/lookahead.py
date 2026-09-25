"""Lookahead (Zhang et al. 2019) wrapping kaon's Adakaon as the fast optimizer.

Lookahead (Zhang, Lucas, Ba, Hinton 2019, *Lookahead Optimizer: k steps forward,
1 step back*, arXiv:1907.08610) is a **wrapper** around a fast inner optimizer. It
keeps a copy of *slow weights* ``phi``. The inner optimizer advances the live
*fast weights* ``theta`` normally; every ``k`` steps a **sync** pulls the slow
weights a fraction ``alpha`` toward the fast weights and resets the fast weights
to the new slow weights:

.. code-block:: text

    every step:    inner_opt.step()                 # theta advances normally
    every k steps: phi   <- phi + alpha*(theta - phi)   # slow weights interpolate
                   theta <- phi                         # reset fast weights to slow

The slow weights ``phi`` are the kept/evaluated sequence (a weight-averaging
regularizer; at every sync ``theta == phi``). This sync rule is the
paper's / `kozistr/pytorch_optimizer`'s convention with ``alpha`` weighting the
*pull toward fast* (``alpha=0.5``, ``k=5/6`` are the paper defaults). The
official ``michaelrzhang/lookahead`` repo writes the *same* interpolation as
``theta = a*theta + (1-a)*phi`` with ``a=0.8`` — i.e. its ``a`` is ``1 - alpha``
here (its 0.8 == this 0.2 pull). We follow the paper/kozistr convention.

Design on the kaon backend
--------------------------
The fast optimizer is :class:`~kaon.adakaon.Adakaon` (the kaon flagship). Lookahead
*builds* an inner ``Adakaon`` from the forwarded kwargs and only adds the slow-weight
machinery, so it inherits all of Adakaon's memory efficiency. The wrapper composes the
three shared wrapper mixins (:mod:`kaon._wrappers`):

* :class:`~kaon._wrappers.WrapsInnerOptimizer` — the inner-optimizer delegation (shared
  ``param_groups``, a separate per-param ``state``, ``state_dict`` merge, ``zero_grad``).
* :class:`~kaon._wrappers.CodecBuffer` — the slow-weight ``phi`` storage through the codec
  at ``slow_dtype`` (~2 B/param bf16, ~1 B int8, ~0.5 B 4bit; fp32 is bit-exact).
* :class:`~kaon._wrappers.TrainEvalWeights` — the ``train()``/``eval()`` swap bookkeeping;
  Lookahead supplies the two view hooks (eval = back up ``theta`` + show ``phi``).

train() / eval()
----------------
The weights you want to **evaluate** are the slow ``phi``, but between syncs the live
weights are the fast ``theta``. :meth:`eval` swaps the live buffer to ``phi`` (saving
``theta``), :meth:`train` swaps it back to the exact pre-eval ``theta``. **Default is
train.** Bracket sampling/validation with ``eval()``/``train()``::

    opt = Lookahead(model.parameters(), lr=1e-3, k=5)
    opt.train()
    for batch in loader:
        opt.zero_grad(); loss(model(batch)).backward(); opt.step()
    opt.eval()       # p.data now holds the slow weights phi
    sample_or_checkpoint(model)
    opt.train()      # back to the fast weights for more training

(:meth:`train` / :meth:`eval` are idempotent — a no-op if already in that mode.)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import torch
from torch import Tensor
from torch.optim import Optimizer

from kaon._backend import (
    foreach_budget,
    per_param_only_bf16_method,
    subtract_batched_,
    subtract_one_,
)
from kaon._compact_kahan import RESIDUAL_KEY, is_compact_kahan
from kaon._momentum_codec import _FOURBIT_BLOCK
from kaon._wrappers import CodecBuffer, TrainEvalWeights, WrapsInnerOptimizer
from kaon.adakaon import Adakaon

__all__ = ["Lookahead"]

SlowDtype = Literal["bfloat16", "float32", "int8", "4bit"]

# sync foreach transient: theta + phi + delta (fp32 intermediates).
_SYNC_STACK_BYTES_PER_ELEM = 48


class Lookahead(WrapsInnerOptimizer, TrainEvalWeights, Optimizer):
    """Lookahead (Zhang et al. 2019) wrapping :class:`~kaon.adakaon.Adakaon`.

    Builds an inner ``Adakaon`` (the *fast* optimizer) from the forwarded kwargs and adds the
    slow-weight ``phi`` machinery. The live parameters hold the fast weights ``theta`` in
    **train** mode and the slow weights ``phi`` in **eval** mode; call :meth:`train` before
    training and :meth:`eval` before sampling / checkpointing (default is train).

    Args:
        params: parameters or param-group dicts.
        lr: learning rate of the inner Adakaon (forwarded).
        k: sync period — run ``k`` inner steps, then sync (default ``5``).
        alpha: slow-weight interpolation factor ``phi += alpha*(theta - phi)`` (default
            ``0.5``; ``1.0`` recovers the plain inner optimizer, ``0.0`` freezes the slow
            weights). Paper/kozistr convention (``alpha`` weights the pull *toward* fast).
        slow_dtype: storage dtype for ``phi`` through the codec — ``"bfloat16"`` (default,
            ~2 B/param), ``"float32"`` (4 B, bit-exact), ``"int8"`` (~1 B) or ``"4bit"``
            (~0.5 B). ``phi`` is read+written once per ``k`` steps, so a quantized choice
            injects only a per-sync requant error.
        slow_4bit_block: block size for ``slow_dtype="4bit"`` (default ``128``).
        **adakaon_kwargs: forwarded verbatim to the inner :class:`Adakaon` (``betas``,
            ``eps``, ``weight_decay``, ``clip_threshold``, ``momentum_dtype``, ``cautious``,
            ``gradient_centralization``, ``bf16_method``, ``foreach``, ...).
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        k: int = 5,
        alpha: float = 0.5,
        *,
        slow_dtype: SlowDtype = "bfloat16",
        slow_4bit_block: int = _FOURBIT_BLOCK,
        **adakaon_kwargs: Any,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if slow_dtype not in ("bfloat16", "float32", "int8", "4bit"):
            raise ValueError(f"slow_dtype must be bfloat16/float32/int8/4bit, got {slow_dtype!r}")

        # Build the inner (fast) optimizer; WrapsInnerOptimizer shares its param_groups,
        # mirrors its foreach toggles, and owns the wrapper's separate per-param state.
        self._bind_inner(Adakaon(params, lr=lr, **adakaon_kwargs), state_key="lookahead")
        # Lookahead's OWN per-group keys (not part of the inner Adakaon's ``defaults``).
        # Kept as ``self.defaults`` — mirroring the attribute every plain
        # ``torch.optim.Optimizer`` carries — purely so ``load_state_dict`` can backfill
        # a key an older checkpoint predates the same way every other kaon optimizer does.
        self.defaults: dict[str, Any] = {
            "k": k,
            "alpha": alpha,
            "slow_dtype": slow_dtype,
            "slow_4bit_block": slow_4bit_block,
            "la_step": 0,        # inner steps since last sync
            "train_mode": True,  # live buffer holds theta
        }
        for group in self.param_groups:
            for key, value in self.defaults.items():
                group.setdefault(key, value)

    # ====================================================== train/eval view hooks
    def _to_eval_view(self, p: Tensor, st: dict[str, Any], group: dict[str, Any]) -> None:
        """eval: back up the fast ``theta`` and show the slow ``phi`` (codec-exact)."""
        if "phi" in st:
            st["backup"] = p.detach().clone()
            p.data.copy_(CodecBuffer.read(st, "phi", group["slow_dtype"], p).to(p.dtype))

    def _to_train_view(self, p: Tensor, st: dict[str, Any], group: dict[str, Any]) -> None:
        """train: restore the byte-exact pre-eval ``theta`` saved by :meth:`eval`."""
        if "backup" in st:
            p.data.copy_(st.pop("backup"))

    # ============================================================================ step
    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        self._require_train_mode("Lookahead")
        # 0) snapshot phi BEFORE the fast step on the first step each param is seen, so phi
        #    starts at the pre-update weights (== the paper's phi_0 = theta_0), not theta_k.
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "phi" not in st:
                    CodecBuffer.alloc(st, "phi", p, group["slow_dtype"], group["slow_4bit_block"])
        # 1) fast step: theta advances via the inner Adakaon.
        loss = self.inner.step(closure)
        # 2) per group, count the step and sync every k.
        for group in self.param_groups:
            group["la_step"] += 1
            if group["la_step"] >= group["k"]:
                group["la_step"] = 0
                self._sync(group)
        return loss

    @torch.no_grad()
    def _sync(self, group: dict[str, Any]) -> None:
        """Slow-weight sync: ``phi += alpha*(theta - phi)`` then ``theta <- phi``.

        ``phi`` is updated in fp32 through the codec (dequant -> lerp -> requant); the live
        ``theta`` is then written to the new ``phi`` via the bf16-correct subtract
        (``theta -= theta - phi_new``), so on bf16 params the reset is SR/kahan-correct.

        ``bf16_method="kahan"`` keeps the whole sync on the per-parameter path: the kahan
        write needs the param's compensation buffer, which only ``subtract_one_`` handles
        (``subtract_batched_`` has no kahan branch at all and would silently downgrade the
        reset to a plain narrowing cast).
        """
        alpha = group["alpha"]
        md = group["slow_dtype"]
        bf16_method = group["bf16_method"]
        synced = [p for p in group["params"] if "phi" in self.state[p]]
        if not synced:
            return
        if self._foreach and not per_param_only_bf16_method(bf16_method):
            self._sync_foreach(synced, group, alpha, md, bf16_method)
        else:
            for p in synced:
                self._sync_one(p, group, alpha, md, bf16_method)

    @torch.no_grad()
    def _sync_one(
        self, p: Tensor, group: dict[str, Any], alpha: float, md: str, bf16_method: str
    ) -> None:
        st = self.state[p]
        theta = p.detach().clone().float()          # fast weights (clone: not a p alias)
        phi = CodecBuffer.read(st, "phi", md, p)    # slow weights, fresh fp32
        phi.lerp_(theta, alpha)                     # phi += alpha*(theta - phi)
        CodecBuffer.write(st, "phi", md, phi)
        delta = theta.sub_(phi)                     # theta - phi_new
        # theta <- phi, through the INNER's per-param state. The only thing the writer
        # reads from it is the kahan compensation buffer ``shift``, and that buffer belongs
        # to the WEIGHT, not to whoever writes it: the inner step and this sync both write
        # the same ``p``, so the dropped bits of both have to accumulate in ONE buffer.
        # A ``shift`` of the wrapper's own would split the residue in half and, worse,
        # carry compensation for a theta the ``theta <- phi`` reset has already overwritten
        # into the next inner step. The wrapper's state has no ``shift`` at all (it
        # allocates ``phi``/``backup``), which is why passing it raised ``KeyError``.
        subtract_one_(p, delta, self.inner.state[p], bf16_method, sr=self.sr_stream)

    @torch.no_grad()
    def _sync_foreach(
        self, params: list[Tensor], group: dict[str, Any], alpha: float, md: str, bf16_method: str
    ) -> None:
        chunk_budget = foreach_budget(
            self._foreach_stack_budget,
            self._foreach_batch_cutoff,
            _SYNC_STACK_BYTES_PER_ELEM + (1 if is_compact_kahan(bf16_method) else 0),  # + residual stack
            params[0].device,
        )
        buckets: dict[tuple[Any, ...], list[Tensor]] = {}
        for p in params:
            buckets.setdefault((tuple(p.shape), p.dtype), []).append(p)
        for (shape, _dtype), plist in buckets.items():
            n_per = max(1, chunk_budget // max(p.numel() for p in plist))
            for i in range(0, len(plist), n_per):
                chunk = plist[i:i + n_per]
                states = [self.state[p] for p in chunk]
                theta = torch.stack([p.detach() for p in chunk]).float()        # [N, *shape]
                phi = CodecBuffer.read_stacked(states, "phi", md, shape)        # [N, *shape]
                phi.lerp_(theta, alpha)
                CodecBuffer.write_stacked(states, "phi", md, phi)
                delta = theta.sub_(phi)                                         # theta - phi_new
                # kahan8: the residual belongs to the WEIGHT, so it is the inner's ``kahan_lo``
                # (same reasoning as ``_sync_one`` handing over the inner's state for ``shift``).
                # Buckets are keyed by dtype, so the chunk is uniformly bf16 or not: only a bf16
                # chunk carries (and needs) the residual; an fp32 chunk under kahan8 has none.
                comp = (
                    [self.inner.state[p][RESIDUAL_KEY] for p in chunk]
                    if is_compact_kahan(bf16_method) and chunk[0].dtype == torch.bfloat16 else None
                )
                subtract_batched_([p.data for p in chunk], delta, bf16_method, sr=self.sr_stream,
                                  comp=comp)

    # ================================================================= state_dict glue
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the inner Adakaon (through ITS OWN loader) and the phi buffers.

        Delegating to ``inner.load_state_dict`` — the same thing SAM and MSAM do — is
        what makes a resume through this wrapper equivalent to one through a bare
        Adakaon. It still gets the dtype-exact restore this used to call directly
        (``Adakaon.load_state_dict`` delegates to ``load_state_dict_preserving_dtypes``),
        and additionally the three things only the inner's loader does: consume
        ``_adakaon_meta`` (restore ``_t``, the Triton kernels' stochastic-rounding seed
        counter, and migrate a pre-0.7.11 lr-scaled momentum to direction units),
        back-fill the inner's own group defaults, and — load-bearing —
        ``_invalidate_fused_caches``: the loader REPLACES every state tensor, so the
        fused pointer tables and foreach plans must not survive it. Calling the
        preserving helper directly skipped all three, leaking one plan per load under a
        dead ``id(group)`` and restarting the fused SR seed at 0 on every resume.

        Backfill Lookahead's own per-group keys (``k``/``alpha``/``slow_dtype``/
        ``slow_4bit_block``/``la_step``/``train_mode``) the checkpoint predates, from
        ``self.defaults`` — same rationale as every other kaon ``load_state_dict``.
        """
        self._load_wrapped(state_dict, lambda inner, sd: inner.load_state_dict(sd))
        for group in self.param_groups:
            for key, value in self.defaults.items():
                group.setdefault(key, value)
