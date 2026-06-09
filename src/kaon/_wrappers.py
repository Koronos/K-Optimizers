"""Shared mixins for *wrapper* optimizers (Lookahead, Schedule-Free, SAM, future EMA/SWA).

Several kaon optimizers are **wrappers**: they keep an alternate set of weights (slow /
averaged) and/or delegate the actual step to an inner base optimizer. They all kept
re-implementing the same three concerns — and kept re-hitting the same footguns. This
module factors those concerns into three small, single-responsibility mixins so the
wrappers compose them instead of duplicating ~120 lines each:

* :class:`CodecBuffer` — store a full-size fp32-logical tensor per parameter through the
  shared momentum codec (bf16 / int8 / 4bit / fp32), per-param **and** stacked, resuming
  dtype-exactly. **Owns the fresh-fp32 read contract** (``read``/``read_stacked`` ALWAYS
  return an independent fp32 tensor), which kills the ``.float()``-aliases-storage bug that
  bit 6 of the campaign's candidates.

* :class:`TrainEvalWeights` — the per-group ``train_mode`` flag plumbing, idempotent
  :meth:`~TrainEvalWeights.train` / :meth:`~TrainEvalWeights.eval` that swap ``p.data``
  between the training-view and an eval-view via two subclass hooks, and a ``step()`` guard.
  The swap *math* is the subclass's hooks; this owns the bookkeeping.

* :class:`WrapsInnerOptimizer` — the delegation boilerplate for a wrapper that builds an
  inner base optimizer: shared ``param_groups``, a separate per-param ``state`` dict (so the
  wrapper's buffers never collide with the inner Adam state), a ``state_dict`` /
  ``load_state_dict`` merge under a namespaced key, and ``zero_grad`` delegation.

These are deliberately orthogonal: a wrapper picks the ones it needs. Lookahead uses all
three; SAM needs only :class:`WrapsInnerOptimizer`; Schedule-Free needs
:class:`CodecBuffer` + :class:`TrainEvalWeights` (its swap is a closed-form lerp, so its
hooks don't materialize an explicit backup). The four recurring footguns —
fp32-aliasing, hyperparameter/``eps`` namespace collisions, non-bf16-correct swaps, and
foreach↔per-param parity — are owned here once, so each wrapper stays small and clean.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch
from torch import Tensor

from kaon._momentum_codec import (
    _dequant_4bit,
    _dequant_4bit_stacked,
    _quant_4bit,
    _quant_4bit_stacked,
    _quant_int8,
    _quant_int8_stacked,
)

__all__ = ["CodecBuffer", "TrainEvalWeights", "WrapsInnerOptimizer"]

FullDtype = ("bfloat16", "float32", "int8", "4bit")


class CodecBuffer:
    """Per-parameter full-size buffer stored through the shared momentum codec.

    A buffer (named by ``key``) is a full-size copy of a weight-shaped tensor stored at a
    configurable ``dtype`` — ``"bfloat16"`` (~2 B/param), ``"float32"`` (4 B, bit-exact),
    ``"int8"`` (~1 B) or ``"4bit"`` (~0.5 B). Used for Lookahead's slow weights ``phi`` and
    Schedule-Free's iterate buffer ``z`` (and ``exp_avg``). All methods are static and take
    the per-param ``state`` dict + the ``key`` (so a multi-buffer optimizer reuses it with
    several keys), keeping the codec layout identical to the hand-rolled versions they replace.

    The companion scale / block metadata live under ``f"{key}_scale"``, ``f"{key}_numel"``,
    ``f"{key}_block"`` — exactly the layout :func:`kaon._momentum_codec.load_state_dict_preserving_dtypes`
    already preserves on resume.
    """

    @staticmethod
    def block_size(t: Tensor, block: int) -> int:
        numel = t.numel()
        return numel if block <= 0 else (min(block, numel) if numel > 0 else 1)

    @staticmethod
    @torch.no_grad()
    def alloc(state: dict[str, Any], key: str, src: Tensor, dtype: str, block: int) -> None:
        """Allocate ``key``, copy-initialized from ``src`` (pass ``zeros_like`` for zero-init)."""
        s = src.detach()
        if dtype in ("bfloat16", "float32"):
            d = torch.bfloat16 if dtype == "bfloat16" else torch.float32
            state[key] = s.to(d).clone()
        elif dtype == "int8":
            state[key], state[f"{key}_scale"] = _quant_int8(s.float())
        else:  # 4bit
            bs = CodecBuffer.block_size(src, block)
            packed, scale, _ = _quant_4bit(s.float(), bs)
            state[key], state[f"{key}_scale"] = packed, scale
            state[f"{key}_numel"] = src.numel()
            state[f"{key}_block"] = bs

    @staticmethod
    def read(state: dict[str, Any], key: str, dtype: str, like: Tensor) -> Tensor:
        """Read ``key`` back as a FRESH fp32 tensor shaped like ``like``.

        The fresh-fp32 guarantee is load-bearing: for an fp32-stored buffer ``t.float()``
        returns ``t`` itself, so an in-place op on the result would silently corrupt the
        stored buffer. We clone in exactly that case (no extra copy for bf16/int8/4bit,
        whose dequant already allocates)."""
        if dtype in ("bfloat16", "float32"):
            t = state[key]
            out = t.float()
            if out is t:  # fp32 buffer -> .float() is a no-op alias; clone to stay safe
                out = out.clone()
            return out.reshape_as(like)
        if dtype == "int8":
            codes = state[key]
            row = codes.shape[0] if codes.ndim >= 2 else 1
            scale = state[f"{key}_scale"].reshape(row, 1) if codes.ndim >= 2 else state[f"{key}_scale"]
            return codes.float().reshape(row, -1).mul_(scale).reshape_as(like)
        m = _dequant_4bit(state[key], state[f"{key}_scale"], state[f"{key}_numel"], state[f"{key}_block"])
        return m.view_as(like)

    @staticmethod
    def write(state: dict[str, Any], key: str, dtype: str, value_fp32: Tensor) -> None:
        """Write an updated fp32 buffer back into the configured storage."""
        if dtype in ("bfloat16", "float32"):
            state[key].copy_(value_fp32.reshape(state[key].shape))
        elif dtype == "int8":
            state[key], state[f"{key}_scale"] = _quant_int8(value_fp32.reshape(state[key].shape))
        else:  # 4bit
            packed, scale, _ = _quant_4bit(value_fp32, state[f"{key}_block"])
            state[key], state[f"{key}_scale"] = packed, scale

    @staticmethod
    def read_stacked(
        states: list[dict[str, Any]], key: str, dtype: str, shape: tuple[int, ...]
    ) -> Tensor:
        """Stacked fp32 buffer ``[N, *shape]`` from per-param storage (always a fresh tensor)."""
        n = len(states)
        per = math.prod(shape)
        if dtype in ("bfloat16", "float32"):
            return torch.stack([s[key].reshape(shape) for s in states]).float()
        if dtype == "int8":
            row = shape[0] if len(shape) >= 2 else 1
            rest = max(per // row, 1)
            m = torch.stack([s[key].reshape(row, rest) for s in states]).float()
            scale = torch.stack([s[f"{key}_scale"].reshape(row, 1) for s in states])
            return m.mul_(scale).reshape((n, *shape))
        packed = torch.stack([s[key] for s in states])
        sc = torch.stack([s[f"{key}_scale"] for s in states])
        bs = states[0][f"{key}_block"]
        return _dequant_4bit_stacked(packed, sc, per, bs).reshape((n, *shape))

    @staticmethod
    def write_stacked(
        states: list[dict[str, Any]], key: str, dtype: str, value_fp32: Tensor
    ) -> None:
        """Write a stacked fp32 buffer ``[N, *shape]`` back into per-param storage."""
        n = value_fp32.shape[0]
        shape = tuple(value_fp32.shape[1:])
        per = math.prod(shape)
        if dtype in ("bfloat16", "float32"):
            torch._foreach_copy_([s[key].reshape(shape) for s in states], list(value_fp32.unbind(0)))
        elif dtype == "int8":
            row = shape[0] if len(shape) >= 2 else 1
            rest = max(per // row, 1)
            q, new_scale = _quant_int8_stacked(value_fp32.reshape(n, row, rest))
            torch._foreach_copy_([s[key].reshape(row, rest) for s in states], list(q.unbind(0)))
            for s, sc in zip(states, new_scale.unbind(0), strict=True):
                s[f"{key}_scale"] = sc.reshape(row, 1) if len(shape) >= 2 else sc.reshape(1)
        else:  # 4bit
            bs = states[0][f"{key}_block"]
            new_packed, new_scale = _quant_4bit_stacked(value_fp32.reshape(n, per), bs)
            torch._foreach_copy_([s[key] for s in states], list(new_packed.unbind(0)))
            for s, sc in zip(states, new_scale.unbind(0), strict=True):
                s[f"{key}_scale"].copy_(sc)


class TrainEvalWeights:
    """Mixin: ``train()`` / ``eval()`` that swap ``p.data`` between a training-view and an
    eval-view, for optimizers whose kept/evaluated weights differ from the live training
    weights (Lookahead's slow ``phi``, Schedule-Free's averaged ``x``).

    This owns only the **plumbing**: a per-group ``train_mode`` flag, idempotent swaps that
    iterate the groups, and a ``step()`` guard. The actual swap is delegated to two hooks the
    subclass implements (which may no-op when the buffer for ``p`` does not exist yet):

    * ``_to_eval_view(p, state, group)``  — make ``p.data`` the eval weights (save what's needed).
    * ``_to_train_view(p, state, group)`` — restore ``p.data`` to the training weights.

    Requires the host to expose ``param_groups`` (with a ``"train_mode"`` key per group, set
    up at construction) and a ``state`` mapping. Default mode is train.
    """

    @torch.no_grad()
    def eval(self) -> None:  # noqa: A003 - mirrors the established optimizer.eval() API
        for group in self.param_groups:
            if not group["train_mode"]:
                continue
            for p in group["params"]:
                st = self.state.get(p)
                if st is not None:
                    self._to_eval_view(p, st, group)
            group["train_mode"] = False

    @torch.no_grad()
    def train(self) -> None:
        for group in self.param_groups:
            if group["train_mode"]:
                continue
            for p in group["params"]:
                st = self.state.get(p)
                if st is not None:
                    self._to_train_view(p, st, group)
            group["train_mode"] = True

    def _require_train_mode(self, who: str) -> None:
        if not self.param_groups[0]["train_mode"]:
            raise RuntimeError(
                f"{who}.step() called outside train mode. Call optimizer.train() before the "
                "training step (and optimizer.eval() before sampling / checkpointing)."
            )

    # subclasses implement these:
    def _to_eval_view(self, p: Tensor, state: dict[str, Any], group: dict[str, Any]) -> None:
        raise NotImplementedError

    def _to_train_view(self, p: Tensor, state: dict[str, Any], group: dict[str, Any]) -> None:
        raise NotImplementedError


class WrapsInnerOptimizer:
    """Mixin: delegation boilerplate for a wrapper that drives an inner base optimizer.

    The inner optimizer owns the ``param_groups`` and the per-param base state (factored /
    momentum); the wrapper shares those ``param_groups`` (so ``zero_grad``, LR schedulers and
    ``.params`` iteration all hit the live weights) and keeps its OWN per-param ``state`` in a
    separate ``defaultdict`` (so its buffers never collide with the inner Adam state — the
    same namespacing that avoids the ``eps`` key collision SAM hit). ``state_dict`` /
    ``load_state_dict`` merge the wrapper's per-param state under a namespaced key, keyed by
    torch's flattened param index so they round-trip dtype-exactly.

    Call :meth:`_bind_inner` from ``__init__`` after building ``self.inner``.
    """

    def _bind_inner(self, inner: Any, *, state_key: str) -> None:
        self.inner = inner
        self._wrap_state_key = state_key
        self.param_groups = inner.param_groups
        # Mirror the inner optimizer's foreach toggles for any batched wrapper path.
        self._foreach = getattr(inner, "_foreach", True)
        self._foreach_stack_budget = getattr(inner, "_foreach_stack_budget", None)
        self._foreach_batch_cutoff = getattr(inner, "_foreach_batch_cutoff", 2_000_000)

    @property
    def state(self) -> dict[Any, dict[str, Any]]:  # type: ignore[override]
        if not hasattr(self, "_wrap_state"):
            self._wrap_state: dict[Any, dict[str, Any]] = defaultdict(dict)
        return self._wrap_state

    def zero_grad(self, set_to_none: bool = True) -> None:  # noqa: FBT001, FBT002
        self.inner.zero_grad(set_to_none=set_to_none)

    def _flat_params(self) -> list[Tensor]:
        return [p for group in self.param_groups for p in group["params"]]

    def state_dict(self) -> dict[str, Any]:
        """Inner optimizer's state_dict + the wrapper's per-param state under ``state_key``."""
        inner = self.inner.state_dict()
        idx = {p: i for i, p in enumerate(self._flat_params())}
        inner[self._wrap_state_key] = {idx[p]: st for p, st in self.state.items() if p in idx}
        return inner

    def _load_wrapped(self, state_dict: dict[str, Any], inner_loader: Any) -> None:
        """Restore the inner optimizer (via ``inner_loader``) and the wrapper's per-param state."""
        sd = dict(state_dict)
        wrapped = sd.pop(self._wrap_state_key, {})
        inner_loader(self.inner, sd)
        self.param_groups = self.inner.param_groups
        self._wrap_state = defaultdict(dict)
        for i, p in enumerate(self._flat_params()):
            st = wrapped.get(i)
            if st is None and str(i) in wrapped:  # int->str key drift (JSON round-trips)
                st = wrapped[str(i)]
            if st is not None:
                self.state[p] = st
