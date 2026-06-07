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
The fast optimizer is :class:`~kaon.adakaon.Adakaon` (the kaon flagship —
factored quantized second moment, optional quantized first moment, foreach,
cautious, gradient centralization). ``Lookahead`` *builds* an inner ``Adakaon``
from the forwarded kwargs and only adds the slow-weight machinery, so it inherits
all of Adakaon's memory efficiency. The only extra state is **one** ``phi``
buffer per parameter, stored through the shared first-moment **codec**
(:mod:`kaon._momentum_codec`) at ``slow_dtype`` — ``"bfloat16"`` (default,
~2 B/param), ``"int8"`` (~1 B/param) or ``"4bit"`` (~0.5 B/param) — so it adds
exactly one momentum-sized buffer on top of Adakaon's factored state.

train() / eval()
----------------
Like Schedule-Free, the weights you want to **evaluate** are the slow ``phi``,
but between syncs the live weights are the fast ``theta``. :meth:`eval` swaps the
live buffer to ``phi`` (saving ``theta``), :meth:`train` swaps it back to the
exact pre-eval ``theta``. **Default is train.** This is an in-place swap (mirrors
``ScheduleFree``); the harness brackets sampling/validation with
``eval()``/``train()`` so the optimizer is measured at the slow weights::

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

from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Literal

import torch
from torch import Tensor
from torch.optim import Optimizer

from kaon._backend import subtract_batched_, subtract_one_
from kaon._momentum_codec import (
    _FOURBIT_BLOCK,
    _dequant_4bit,
    _dequant_4bit_stacked,
    _quant_4bit,
    _quant_4bit_stacked,
    _quant_int8,
    _quant_int8_stacked,
    load_state_dict_preserving_dtypes,
)
from kaon.adakaon import Adakaon

__all__ = ["Lookahead"]

SlowDtype = Literal["bfloat16", "float32", "int8", "4bit"]


class Lookahead(Optimizer):
    """Lookahead (Zhang et al. 2019) wrapping :class:`~kaon.adakaon.Adakaon`.

    Builds an inner ``Adakaon`` (the *fast* optimizer) from the forwarded kwargs
    and adds the slow-weight ``phi`` machinery. The live parameters hold the fast
    weights ``theta`` in **train** mode and the slow weights ``phi`` in **eval**
    mode; call :meth:`train` before training and :meth:`eval` before sampling /
    checkpointing (default is train). See the module docstring.

    Args:
        params: parameters or param-group dicts.
        lr: learning rate of the inner Adakaon (forwarded).
        k: sync period — run ``k`` inner steps, then sync the slow weights
            (default ``5``; the paper uses ``5`` or ``6``).
        alpha: slow-weight interpolation factor ``phi += alpha*(theta - phi)``
            (default ``0.5``; ``1.0`` recovers the plain inner optimizer, ``0.0``
            freezes the slow weights). This is the paper/kozistr convention where
            ``alpha`` weights the pull *toward* the fast weights.
        slow_dtype: storage dtype for the ``phi`` buffer through the shared codec —
            ``"bfloat16"`` (default, ~2 B/param), ``"float32"`` (4 B/param),
            ``"int8"`` (~1 B/param) or ``"4bit"`` (~0.5 B/param). ``phi`` is the
            kept/averaged weight and is read+written only once per ``k`` steps, so
            a quantized choice injects only a per-sync requant error; ``"float32"``
            is bit-exact.
        slow_4bit_block: block size for ``slow_dtype="4bit"`` (default ``128``).
        **adakaon_kwargs: forwarded verbatim to the inner :class:`Adakaon`
            (``betas``, ``eps``, ``weight_decay``, ``clip_threshold``,
            ``momentum_dtype``, ``cautious``, ``gradient_centralization``,
            ``bf16_method``, ``foreach``, ...).
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
            raise ValueError(
                f"slow_dtype must be bfloat16/float32/int8/4bit, got {slow_dtype!r}"
            )

        # The inner (fast) optimizer owns the param_groups and the per-param Adam
        # state. Lookahead is a thin wrapper that drives it and keeps phi.
        self.inner = Adakaon(params, lr=lr, **adakaon_kwargs)

        # Share the inner optimizer's param_groups so generic Optimizer machinery
        # (zero_grad, .params iteration, schedulers reading group["lr"]) operates
        # on the live weights, and stash the Lookahead config on each group.
        self.param_groups = self.inner.param_groups
        # Match the inner optimizer's foreach toggle for the sync path.
        self._foreach = self.inner._foreach
        self._foreach_stack_budget = self.inner._foreach_stack_budget
        self._foreach_batch_cutoff = self.inner._foreach_batch_cutoff
        for group in self.param_groups:
            group.setdefault("k", k)
            group.setdefault("alpha", alpha)
            group.setdefault("slow_dtype", slow_dtype)
            group.setdefault("slow_4bit_block", slow_4bit_block)
            group.setdefault("la_step", 0)        # inner steps since last sync
            group.setdefault("train_mode", True)  # live buffer holds theta

    # Lookahead keeps its own per-param state (the phi buffer + the eval backup);
    # the inner Adam state lives in self.inner.state. Using a separate dict avoids
    # colliding with the inner optimizer's keys.
    @property
    def state(self) -> dict[Any, dict[str, Any]]:  # type: ignore[override]
        if not hasattr(self, "_la_state"):
            self._la_state: dict[Any, dict[str, Any]] = defaultdict(dict)
        return self._la_state

    # ============================================================ phi codec storage
    # phi is a full-size copy of the weights, stored through the shared first-moment
    # codec layout so a configured int8/4bit slow_dtype keeps it compact and resumes
    # bit-exactly. Mirrors ScheduleFree's `z` storage (one buffer).
    @staticmethod
    def _block_size(t: Tensor, group: dict[str, Any]) -> int:
        bs = group["slow_4bit_block"]
        numel = t.numel()
        return numel if bs <= 0 else (min(bs, numel) if numel > 0 else 1)

    @torch.no_grad()
    def _alloc_phi(self, p: Tensor, state: dict[str, Any], group: dict[str, Any]) -> None:
        """Allocate the slow-weight ``phi`` buffer, copy-initialized from ``p``."""
        md = group["slow_dtype"]
        src = p.detach()
        if md in ("bfloat16", "float32"):
            dtype = torch.bfloat16 if md == "bfloat16" else torch.float32
            state["phi"] = src.to(dtype).clone()
        elif md == "int8":
            q, scale = _quant_int8(src.float())
            state["phi"], state["phi_scale"] = q, scale
        else:  # 4bit
            bs = self._block_size(p, group)
            packed, scale, _ = _quant_4bit(src.float(), bs)
            state["phi"], state["phi_scale"] = packed, scale
            state["phi_numel"] = p.numel()
            state["phi_block"] = bs

    @staticmethod
    def _dequant_phi(state: dict[str, Any], md: str, like: Tensor) -> Tensor:
        """Read ``phi`` back as a fresh fp32 tensor shaped like ``like``."""
        if md in ("bfloat16", "float32"):
            return state["phi"].float().reshape_as(like)
        if md == "int8":
            codes = state["phi"]
            row = codes.shape[0] if codes.ndim >= 2 else 1
            scale = state["phi_scale"].reshape(row, 1) if codes.ndim >= 2 else state["phi_scale"]
            return codes.float().reshape(row, -1).mul_(scale).reshape_as(like)
        m = _dequant_4bit(state["phi"], state["phi_scale"], state["phi_numel"], state["phi_block"])
        return m.view_as(like)

    @staticmethod
    def _store_phi(state: dict[str, Any], md: str, m_fp32: Tensor) -> None:
        """Write an updated fp32 ``phi`` back into the configured storage."""
        if md in ("bfloat16", "float32"):
            state["phi"].copy_(m_fp32.reshape(state["phi"].shape))
        elif md == "int8":
            state["phi"], state["phi_scale"] = _quant_int8(m_fp32.reshape(state["phi"].shape))
        else:  # 4bit
            packed, scale, _ = _quant_4bit(m_fp32, state["phi_block"])
            state["phi"], state["phi_scale"] = packed, scale

    @staticmethod
    def _dequant_phi_stacked(
        states: list[dict[str, Any]], md: str, shape: tuple[int, ...]
    ) -> Tensor:
        """Stacked fp32 ``phi`` ``[N, *shape]`` from per-param storage."""
        n = len(states)
        per = 1
        for d in shape:
            per *= d
        if md in ("bfloat16", "float32"):
            return torch.stack([s["phi"].reshape(shape) for s in states]).float()
        if md == "int8":
            row = shape[0] if len(shape) >= 2 else 1
            rest = max(per // row, 1)
            m = torch.stack([s["phi"].reshape(row, rest) for s in states]).float()
            scale = torch.stack([s["phi_scale"].reshape(row, 1) for s in states])
            return m.mul_(scale).reshape((n, *shape))
        packed = torch.stack([s["phi"] for s in states])
        sc = torch.stack([s["phi_scale"] for s in states])
        bs = states[0]["phi_block"]
        return _dequant_4bit_stacked(packed, sc, per, bs).reshape((n, *shape))

    @staticmethod
    def _store_phi_stacked(
        states: list[dict[str, Any]], md: str, m_fp32: Tensor
    ) -> None:
        """Write a stacked fp32 ``phi`` ``[N, *shape]`` back into per-param storage."""
        n = m_fp32.shape[0]
        shape = tuple(m_fp32.shape[1:])
        per = 1
        for d in shape:
            per *= d
        if md in ("bfloat16", "float32"):
            torch._foreach_copy_(
                [s["phi"].reshape(shape) for s in states], list(m_fp32.unbind(0))
            )
        elif md == "int8":
            row = shape[0] if len(shape) >= 2 else 1
            rest = max(per // row, 1)
            q, new_scale = _quant_int8_stacked(m_fp32.reshape(n, row, rest))
            torch._foreach_copy_(
                [s["phi"].reshape(row, rest) for s in states], list(q.unbind(0))
            )
            for s, sc in zip(states, new_scale.unbind(0), strict=True):
                s["phi_scale"] = sc.reshape(row, 1) if len(shape) >= 2 else sc.reshape(1)
        else:  # 4bit
            bs = states[0]["phi_block"]
            new_packed, new_scale = _quant_4bit_stacked(m_fp32.reshape(n, per), bs)
            torch._foreach_copy_([s["phi"] for s in states], list(new_packed.unbind(0)))
            for s, sc in zip(states, new_scale.unbind(0), strict=True):
                s["phi_scale"].copy_(sc)

    # ====================================================================== train/eval
    @torch.no_grad()
    def eval(self) -> None:
        """Switch the live buffer from the fast ``theta`` to the slow ``phi``.

        Saves ``theta`` (in a backup) and writes ``phi`` into ``p.data`` so
        sampling / validation / checkpointing sees the kept slow weights.
        Idempotent: a no-op if already in eval mode. ``phi`` is exact (the codec
        round-trips the stored representation), so the swap is lossless for the
        eval view; :meth:`train` restores the byte-exact pre-eval ``theta``.
        """
        for group in self.param_groups:
            if not group["train_mode"]:
                continue
            md = group["slow_dtype"]
            for p in group["params"]:
                st = self.state.get(p)
                if st and "phi" in st:
                    st["backup"] = p.detach().clone()
                    p.data.copy_(self._dequant_phi(st, md, p).to(p.dtype))
            group["train_mode"] = False

    @torch.no_grad()
    def train(self) -> None:
        """Restore the live buffer from the slow ``phi`` back to the fast ``theta``.

        Copies the backup saved by :meth:`eval` back into ``p.data`` (exact inverse
        of :meth:`eval`). Idempotent: a no-op if already in train mode.
        """
        for group in self.param_groups:
            if group["train_mode"]:
                continue
            for p in group["params"]:
                st = self.state.get(p)
                if st and "backup" in st:
                    p.data.copy_(st.pop("backup"))
            group["train_mode"] = True

    # ============================================================================ step
    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        if not self.param_groups[0]["train_mode"]:
            raise RuntimeError(
                "Lookahead.step() called outside train mode. Call optimizer.train() "
                "before the training step (and optimizer.eval() before sampling / "
                "checkpointing). See the Lookahead docstring."
            )
        # 0) snapshot phi BEFORE the fast step on the first step each param is seen,
        #    so phi starts at the pre-update weights (== the paper's phi_0 = theta_0),
        #    not at theta_k after k steps (which would make the first sync a no-op).
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "phi" not in st:
                    self._alloc_phi(p, st, group)
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

        ``phi`` is updated in fp32 through the codec (dequant -> lerp -> requant);
        the live weights ``theta`` are then written to the new ``phi`` via the
        bf16-correct subtract (``theta -= (theta - phi_new)``), so on bf16 params
        the reset is stochastic-rounding/kahan-correct rather than a naive cast.
        """
        alpha = group["alpha"]
        md = group["slow_dtype"]
        bf16_method = group["bf16_method"]
        # phi was snapshotted in step() before the fast step; sync every param that
        # has a phi buffer (i.e. has been stepped at least once).
        synced = [p for p in group["params"] if "phi" in self.state[p]]
        if not synced:
            return

        if self._foreach and bf16_method != "kahan":
            self._sync_foreach(synced, group, alpha, md, bf16_method)
        else:
            for p in synced:
                self._sync_one(p, group, alpha, md, bf16_method)

    @torch.no_grad()
    def _sync_one(
        self, p: Tensor, group: dict[str, Any], alpha: float, md: str, bf16_method: str
    ) -> None:
        st = self.state[p]
        theta = p.detach().clone().float()              # fast weights (clone: not p alias)
        phi = self._dequant_phi(st, md, p)              # slow weights, fp32
        # phi += alpha*(theta - phi)
        phi.lerp_(theta, alpha)
        self._store_phi(st, md, phi)
        # theta <- phi : write via bf16-correct subtract (theta -= theta - phi).
        delta = theta.sub_(phi)
        subtract_one_(p, delta, st, bf16_method)

    @torch.no_grad()
    def _sync_foreach(
        self, params: list[Tensor], group: dict[str, Any], alpha: float, md: str, bf16_method: str
    ) -> None:
        # Bucket by (effective shape, dtype) so each bucket stacks to one tensor.
        buckets: dict[tuple[Any, ...], list[Tensor]] = {}
        for p in params:
            buckets.setdefault((tuple(p.shape), p.dtype), []).append(p)
        for (shape, _dtype), plist in buckets.items():
            states = [self.state[p] for p in plist]
            theta = torch.stack([p.detach().float() for p in plist])        # [N, *shape]
            phi = self._dequant_phi_stacked(states, md, shape)              # [N, *shape]
            phi.lerp_(theta, alpha)
            self._store_phi_stacked(states, md, phi)
            delta = theta.sub_(phi)                                         # theta - phi_new
            subtract_batched_([p.data for p in plist], delta, bf16_method)

    # ================================================================= state_dict glue
    def state_dict(self) -> dict[str, Any]:
        """Combined state: the inner Adakaon's state + Lookahead's phi/config.

        The inner optimizer's ``state_dict`` carries the param-groups (and so the
        Lookahead per-group config stashed there) plus the inner Adam state. The
        Lookahead per-param ``phi`` buffers are saved alongside, keyed by the same
        flattened param index torch uses, so they round-trip dtype-exactly.
        """
        inner = self.inner.state_dict()
        params = [p for group in self.param_groups for p in group["params"]]
        idx = {p: i for i, p in enumerate(params)}
        la = {idx[p]: st for p, st in self.state.items() if p in idx}
        inner["lookahead"] = la
        return inner

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the inner optimizer (dtype-preserving) and the phi buffers."""
        sd = dict(state_dict)
        la = sd.pop("lookahead", {})
        load_state_dict_preserving_dtypes(self.inner, sd)
        # Re-bind param_groups to the freshly-loaded inner groups.
        self.param_groups = self.inner.param_groups
        params = [p for group in self.param_groups for p in group["params"]]
        self._la_state = defaultdict(dict)
        for i, p in enumerate(params):
            st = la.get(i)
            if st is None and str(i) in la:  # JSON-ish int->str key drift
                st = la[str(i)]
            if st is not None:
                self.state[p] = st

    def zero_grad(self, set_to_none: bool = True) -> None:  # noqa: FBT001, FBT002
        self.inner.zero_grad(set_to_none=set_to_none)
