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

A wrapper that writes weights itself (Lookahead's ``phi`` sync, SAM's climb) is a *second*
owner of stochastic-rounding noise on top of the inner optimizer's weight write, so
:class:`WrapsInnerOptimizer` carries its own :class:`kaon._backend.SRSeedState` stream and
persists it under a namespaced key — for the same reason its per-param state is namespaced.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch
from torch import Tensor

from kaon._backend import SRSeedState
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

# Base of every wrapper's stochastic-rounding checkpoint key. The live key appends the
# wrapper's own ``state_key`` (see ``WrapsInnerOptimizer._bind_inner``); the bare name
# stays readable so checkpoints written while the key was shared still resume.
_WRAP_SR_KEY = "_sr_wrap_meta"


def _onto_device(value: Any, device: torch.device) -> Any:
    """Move every tensor inside ``value`` to ``device``, **preserving dtype exactly**.

    The device/dtype policy of :func:`kaon._momentum_codec.load_state_dict_preserving_dtypes`,
    applied to a restored per-parameter state entry: ``.to(device=...)`` only — never a
    dtype cast. Casting would silently rewrite a wrapper buffer's declared storage (a bf16
    ``phi`` widened to fp32, an int8 code tensor turned into floats, a 4-bit nibble
    payload destroyed) and its companion fp32 scales.

    Returns ``value`` **itself** whenever nothing has to move, so a checkpoint already on
    the parameter's device costs no allocation, no copy and no bit of drift. Containers
    are walked (torch's own loader does the same) and likewise rebuilt only if a tensor
    inside them actually moved.
    """
    if isinstance(value, Tensor):
        return value if value.device == device else value.to(device=device)
    if isinstance(value, dict):
        moved = {k: _onto_device(v, device) for k, v in value.items()}
        return value if all(moved[k] is v for k, v in value.items()) else moved
    if type(value) in (list, tuple):  # exact types only: a namedtuple/subclass is left alone
        moved_seq = [_onto_device(v, device) for v in value]
        if all(m is v for m, v in zip(moved_seq, value, strict=True)):
            return value
        return moved_seq if isinstance(value, list) else tuple(moved_seq)
    return value


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
            # In place: pointer caches over these buffers must never dangle (see adakaon requant).
            q, sc = _quant_int8(value_fp32.reshape(state[key].shape))
            state[key].copy_(q)
            state[f"{key}_scale"].copy_(sc.reshape(state[f"{key}_scale"].shape))
        else:  # 4bit
            packed, scale, _ = _quant_4bit(value_fp32, state[f"{key}_block"])
            state[key].copy_(packed)
            state[f"{key}_scale"].copy_(scale)

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
                # In place (not reassignment): pointer caches reference these scale tensors.
                s[f"{key}_scale"].copy_(sc.reshape(s[f"{key}_scale"].shape))
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


class WrapsInnerOptimizer(SRSeedState):
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

    # Namespaced away from the inner optimizer's own ``_sr_meta``, which the inner's
    # ``state_dict`` has already written into the same dict by the time we add ours.
    # ``_bind_inner`` narrows it further, PER WRAPPER; this bare name is only ever a
    # placeholder for a wrapper that never bound an inner optimizer.
    SR_META_KEY = _WRAP_SR_KEY

    def _bind_inner(self, inner: Any, *, state_key: str) -> None:
        self.inner = inner
        self._wrap_state_key = state_key
        # One SR key per wrapper, derived from the same ``state_key`` that already
        # namespaces its per-param state. A single shared key is wrong as soon as wrappers
        # NEST — ``SAM(base_optimizer=Lookahead, ...)`` is public API and makes three noise
        # owners — because the outer ``state_dict`` then overwrites the intermediate one's
        # position and the resume silently continues from the wrong draw (measured 2.34e-2
        # on bf16 weights). Instance attribute on purpose: it shadows the class default,
        # which stays reachable as the read fallback.
        self.SR_META_KEY = f"{_WRAP_SR_KEY}_{state_key}"
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

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Add a group through the INNER optimizer, which owns ``param_groups`` (shared with
        this wrapper), their defaults and their validation (e.g. the fp16 / bf16_method
        checks). ``torch.optim.Optimizer.add_param_group`` on the wrapper itself read
        ``self.defaults`` — which a wrapper does not have unless it keeps its OWN per-group
        keys — and failed with a bare ``AttributeError`` (``Nekaon(...).add_param_group``).
        A wrapper that does keep own keys (Lookahead's ``k``/``alpha``/...) gets them
        back-filled on the new group from its ``defaults``, exactly as at construction."""
        self.inner.add_param_group(param_group)
        own = self.__dict__.get("defaults")
        if own:
            for key, value in own.items():
                self.param_groups[-1].setdefault(key, value)

    def _flat_params(self) -> list[Tensor]:
        return [p for group in self.param_groups for p in group["params"]]

    def state_dict(self) -> dict[str, Any]:
        """Inner optimizer's state_dict + the wrapper's per-param state under ``state_key``.

        The wrapper's own SR noise stream rides along under a namespaced key too: the inner
        optimizer's ``state_dict`` already carries ITS stream under ``_sr_meta``, and the
        two are different owners (Lookahead's ``phi`` sync writes through the same shared
        kernel as the inner weight write, from its own position in the noise space).
        """
        inner = self.inner.state_dict()
        idx = {p: i for i, p in enumerate(self._flat_params())}
        inner[self._wrap_state_key] = {idx[p]: st for p, st in self.state.items() if p in idx}
        return self._sr_save(inner)

    def _load_wrapped(self, state_dict: dict[str, Any], inner_loader: Any) -> None:
        """Restore the inner optimizer (via ``inner_loader``) and the wrapper's per-param state.

        ``inner_loader(inner, sd)`` MUST be the inner optimizer's own ``load_state_dict``
        (``lambda inner, sd: inner.load_state_dict(sd)``), never a lower-level loader such
        as :func:`kaon._momentum_codec.load_state_dict_preserving_dtypes`. A kaon
        optimizer's ``load_state_dict`` is not a thin shell over torch's: it also consumes
        its own ``_..._meta`` blob (Adakaon's fused SR seed counter and its pre-0.7.11
        momentum-unit migration), back-fills group keys the checkpoint predates from its
        ``defaults``, and — load-bearing — drops every host-side cache holding pointers or
        views into the state tensors the load just REPLACED (``_invalidate_fused_caches`` /
        ``_clear_foreach_plans``). Bypassing it leaks one cache entry per load under a dead
        ``id(group)`` and desynchronises the fused path's noise stream on resume — which is
        exactly what Lookahead did until it was switched to delegate here.

        The wrapper's own per-param state is restored **on the parameter's device** (via
        :func:`_onto_device`), at the storage dtype the checkpoint carries — the same
        device policy ``torch.optim.Optimizer.load_state_dict`` applies to the inner
        optimizer's state, and the reason the inner half of a resume always worked. This
        used to install the checkpoint's dicts verbatim, so the near-universal consumer
        idiom ``torch.load(path, map_location="cpu")`` left Lookahead's ``phi`` (and its
        scales) on the CPU under CUDA parameters and the next sync died with *"Expected
        all tensors to be on the same device"*. Nothing is copied when the state already
        sits on the right device, so a same-device resume stays free and bit-identical.
        """
        sd = dict(state_dict)
        wrapped = sd.pop(self._wrap_state_key, {})
        # The wrapper's OWN noise stream (its ``phi`` sync / climb writes); the inner's
        # rides in ``_sr_meta`` and is restored by the inner's loader below. Only THIS
        # wrapper's key is consumed, so a nested inner wrapper never picks up a position
        # meant for the outer one. The un-namespaced ``_sr_wrap_meta`` of the (unreleased)
        # intermediate layout is dropped rather than read: guessing which level of a nested
        # stack it belonged to is worse than resuming that one blob from draw 0.
        self._sr_load(sd)
        for key in (self.SR_META_KEY, _WRAP_SR_KEY):
            sd.pop(key, None)
        inner_loader(self.inner, sd)
        self.param_groups = self.inner.param_groups
        self._wrap_state = defaultdict(dict)
        for i, p in enumerate(self._flat_params()):
            st = wrapped.get(i)
            if st is None and str(i) in wrapped:  # int->str key drift (JSON round-trips)
                st = wrapped[str(i)]
            if st is not None:
                for key, value in list(st.items()):  # device follows the param, dtype intact
                    moved = _onto_device(value, p.device)
                    if moved is not value:
                        st[key] = moved
                self.state[p] = st
