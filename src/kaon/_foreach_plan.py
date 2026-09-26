"""Shared foreach bucketing + cached view plan for kaon's batched optimizers.

Every optimizer with a ``foreach`` path used to carry its own copy of the same twenty
lines: partition a group's params into *factored* (``ndim >= 2``, stacked ``[N, R, C]``)
and *non-factored* (``ndim <= 1``, stacked ``[N, L]``) buckets keyed by effective shape
and dtype, split each bucket by the per-chunk element budget, and then — inside every
bucket body, **on every step** — rebuild the derived view lists it works through
(``[mat(p.data) for p in plist]``, ``[flat_view(s["v"]) for s in states]``, the momentum
reshapes). On a 428-tensor bag that rebuild alone was 860-1300 ``aten::view`` /
``reshape`` / ``select`` calls and 12-17 ms of CPU dispatch per step, for lists that are
pure functions of tensors the optimizer **already owns**.

This module holds that machinery once. :class:`ForeachPlanMixin` gives an optimizer a
``_foreach_chunks(params, group, budget)`` call that returns ready-made
:class:`ForeachChunk` objects; what each optimizer needs cached is declared once, as a
:class:`ForeachSpec` class attribute.

Caching views pins **no memory**: every cached tensor is a view of a param or of a state
buffer the optimizer holds anyway. Gradients are the deliberate exception — see
:meth:`ForeachChunk.grad_stack`.

Every batched optimizer in kaon now shares this module, Adakaon included: this design was
extracted FROM Adakaon and folded back into it once the shared version had settled. Adakaon
is also the only user whose fused Triton routing sits next to the native plan; the two stay
consistent because ``Adakaon._fused_partition`` keys on the same witness CONTRACT —
:func:`kaon._fused_triton.param_witness`, this module's three fields plus an optional strides
field under ``ft.SHAPE_WITNESS`` — and ``Adakaon._invalidate_fused_caches`` drops both sets of
caches in one call.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping
from itertools import count
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from kaon._backend import decode_value, ensure_residuals, flat_view
from kaon._compact_kahan import (
    RESIDUAL_KEY,
    is_compact_kahan,
    residual_bits,
    residual_bits_of,
    residual_dtype,
)

if TYPE_CHECKING:  # annotations only — ``_momentum_codec`` does not import this module,
    # so a runtime import would not cycle either; it is deferred to keep the import
    # graph of the NATIVE path free of the codec.
    from kaon._momentum_codec import _MomentumCodec, _StackedViews

# Unbound, so the per-step staleness witness is a C-level ``map()`` instead of a genexpr.
_DATA_PTR = Tensor.data_ptr
_IS_CONTIG = Tensor.is_contiguous


def _identity(t: Tensor) -> Tensor:
    """``mat`` for buckets whose effective layout *is* the tensor (no view needed)."""
    return t


def param_witness(plist: list[Tensor]) -> tuple:
    """Per-step staleness witness for a cached plan over ``plist``: ``(ids, data_ptrs,
    contiguity)``, one flat tuple each.

    Each field is load-bearing and none implies another. ``id`` catches a changed param
    set (including the foreach/per-param split moving a param out of the fast list).
    ``data_ptr`` catches ``p.data = <fresh storage>`` — an external EMA, a
    ``.to(dtype/device)``, an offloader's block swap — and, through it, every dtype and
    device change. Contiguity catches ``p.data = p.data.t()``, which on a SQUARE weight
    keeps id, pointer AND shape and moves only the strides, while the cached view would
    keep stepping the pre-transpose layout.

    LIMIT — a rebind that CHANGES THE SHAPE (``p.data = p.data.view(...)``) is not
    supported and deliberately not watched: the factored second moment is bound to the
    effective 2-D shape, and there is no meaningful migration of an EMA onto a different
    factorization. The stale bucketing makes the next step raise a size mismatch, which
    is the intended outcome. THIS WITNESS IS FIXED AT THREE FIELDS, and that is a property of
    the native path, not a compromise: the plan re-stacks by effective shape every step, so a
    shape-changing rebind raises out of ``torch.stack`` on the very next step and a fourth field
    would add nothing but cost. ``kaon._fused_triton.param_witness`` holds this exact contract in
    its default configuration and grows an optional FOURTH field (per-param strides) under
    ``ft.SHAPE_WITNESS``, because the fused path freezes its whole geometry at plan-build time and
    has to be told; that field is defined THERE, once, and never here. Adakaon's fused ROUTING
    (``Adakaon._fused_partition``) therefore keys on the fused witness — through the ``ft`` module
    it is already handed, so nothing imports Triton at module scope — which also keeps its key
    exactly as strong as the pointer caches' own. This one is what the NATIVE plan keys on, and it
    has to keep working in a build without Triton.
    """
    return (tuple(map(id, plist)), tuple(map(_DATA_PTR, plist)), tuple(map(_IS_CONTIG, plist)))


# ============================================================ state-identity watch
# :func:`param_witness` is the whole staleness story for the PARAMETERS and says nothing
# about the STATE. Every cross-step cache in kaon bakes state buffers — the fused pointer
# tables (``m``/``m_scale``/``row``/``col``/``v`` addresses, frozen at build) and this
# module's :attr:`ForeachChunk.state_views` (views of the same buffers) — and revalidates
# them against the params alone. So a state buffer swapped out from under a cache while
# every param stood still left the tables addressing a RETIRED tensor, and the step wrote
# it: confirmed on all four fused routes and on the native plan for ``del opt.state[p]``,
# ``opt.state[p].clear()`` and ``opt.state[p]["m"] = ...``.
#
# WHY A COUNTER AND NOT A WITNESS FIELD. A state witness is another per-param host sweep,
# and the three param fields already cost 3.0-3.6% of the 428-param step (see
# ``kaon._fused_triton.SHAPE_WITNESS``); one state-tensor field measured +19-25 µs there,
# ~1.2% of the step, against a 0.5% budget — and it would still be blind to
# ``state[p].clear()`` followed by a refill (same dict, same everything, new buffers).
# Making the change UNMISSABLE instead costs nothing per step: ``self.state`` is a
# :class:`WatchedState` whose per-param dicts count every rebinding of a baked key, and a
# cache holds the count it was built at. Nothing in a steady-state step rebinds one
# (AdaPNM's per-param ``state["step"] += 1`` is not a baked key), so the counter never
# moves and no cache is ever rebuilt for it — asserted in
# ``tests/test_state_identity_witness.py``, which is what keeps the cost at zero.
#
# The CELL is per mapping (each :class:`WatchedState` owns one, shared with its per-param
# dicts); the VALUES come from one process-wide counter and are therefore never reused. Both
# halves matter. Per-mapping is what keeps two optimizers in the same process from
# invalidating each other's caches — a shared cell would make every AdaPNM state write bump
# Adakaon's number too. Globally unique values are what close ABA: a reinstalled watch
# (``__setattr__``, or ``__setstate__`` after ``load_state_dict`` / unpickling / ``deepcopy``)
# starts from a number no cache anywhere can already be holding, instead of restarting at 0
# and colliding with a cache built by the mapping it replaced.
_next_epoch = count(1).__next__
_MISSING = object()

#: The ``self.state`` keys whose VALUE a cross-step cache bakes. A rebinding of any of them
#: retires a buffer some pointer table or cached view may still be addressing; everything
#: else in a state dict is either a scalar (``step``, ``m_numel``, ``m_block``) or is read
#: fresh every step, so rebinding it invalidates nothing and must stay free.
WATCHED_STATE_KEYS = frozenset({
    "m", "m_scale",                                  # momentum (every codec) + its scales
    "m_pos", "m_neg", "m_pos_scale", "m_neg_scale",  # AdaPNM's two momenta
    "row", "col",                                    # factored second moment
    "v", "max_v",                                    # non-factored second moment (+ AMSGrad)
    "shift",                                         # Kahan compensation (bf16, legacy)
    "kahan_lo",                                      # compact Kahan residual (kahan8/kahan16)
})


class WatchedParamState(dict):
    """One parameter's optimizer state, counting every change of buffer IDENTITY.

    Only a rebinding of a :data:`WATCHED_STATE_KEYS` entry to a DIFFERENT object counts:
    an in-place write (the codec's requant contract, ``row.lerp_``, ``copy_``) keeps every
    cached pointer valid and must not invalidate anything, and neither must a scalar
    counter. Storing the very same object back is likewise a no-op.

    Serialises as a PLAIN dict (:meth:`__reduce__`). ``Optimizer.state_dict`` hands these
    dicts to the caller by reference, so a checkpoint would otherwise embed this class and
    its generation cell — unloadable without kaon, and meaningless once loaded.
    """

    __slots__ = ("_gen",)

    def __init__(self, items: Any = (), *, gen: list[int]) -> None:
        # ``gen`` is KEYWORD-ONLY on purpose. With it first and positional,
        # ``WatchedParamState({"m": ...})`` — the shape every dict subclass is expected to
        # take, and the shape ``dict.__init__`` itself takes — bound the MAPPING to
        # ``_gen`` and left the instance with a generation cell that is a dict: every later
        # bump then wrote ``self._gen[0] = ...`` into that dict and the state's counter was
        # silently dead. Keyword-only makes that call a ``TypeError`` instead.
        dict.__init__(self, items)
        if type(gen) is not list:
            raise TypeError(
                f"WatchedParamState needs the mapping's generation CELL (a one-element "
                f"list), got {type(gen).__name__} — see WatchedState.gen"
            )
        self._gen = gen

    # -- mutation: bump the generation when a baked buffer is retired ------------
    def __setitem__(self, key: str, value: Any) -> None:
        # Only a key that was ALREADY THERE can retire anything: a cache dereferences
        # ``state["m"]`` / ``state["row"]`` at build time, so a key that was absent was in
        # no cache. That is what makes ``_init_state`` free — a parameter's first step
        # populates five watched keys and moves the generation zero times, so neither the
        # first step nor the one after it pays a rebuild for having allocated state.
        if key in WATCHED_STATE_KEYS:
            old = dict.get(self, key, _MISSING)
            if old is not _MISSING and old is not value:
                self._gen[0] = _next_epoch()
        dict.__setitem__(self, key, value)

    def __delitem__(self, key: str) -> None:
        if key in WATCHED_STATE_KEYS:
            self._gen[0] = _next_epoch()
        dict.__delitem__(self, key)

    def clear(self) -> None:
        if self:
            self._gen[0] = _next_epoch()
        dict.clear(self)

    def pop(self, key: str, *default: Any) -> Any:
        if key in WATCHED_STATE_KEYS and key in self:
            self._gen[0] = _next_epoch()
        return dict.pop(self, key, *default)

    def popitem(self) -> tuple[str, Any]:
        self._gen[0] = _next_epoch()
        return dict.popitem(self)

    def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        # ``dict.update`` is C and would bypass ``__setitem__`` entirely.
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def __ior__(self, other: Any) -> WatchedParamState:  # type: ignore[misc]
        # ``st |= {...}`` lands on ``nb_inplace_or``, a DIFFERENT C slot from
        # ``mp_ass_subscript`` and from ``update`` — so hooking those two left it wide
        # open: ``opt.state[p] |= {"m": fresh}`` retired the buffer, moved nothing, and the
        # next step wrote the retired tensor (measured 4/4 on every fused route, on the
        # foreach plan and through MSAM / Nekaon / Lookahead / SAM). It must mutate in
        # place and return ``self``, which is what ``dict.__ior__`` does.
        self.update(other)
        return self

    def setdefault(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        # ``dict.setdefault`` is C and would bypass ``__setitem__``. Inserting a key that
        # was absent retires nothing, so this never bumps — it is here for the write path,
        # not for the generation.
        if key in self:
            return dict.__getitem__(self, key)
        self[key] = default
        return default

    def __reduce__(self) -> tuple:
        return (dict, (list(self.items()),))


class WatchedState(defaultdict):
    """``parameter -> per-param state``, with a generation every cache can compare against.

    ``self.gen`` is a one-element list shared with every :class:`WatchedParamState` this
    mapping owns, so ONE integer read tells a cache whether any state buffer anywhere in
    the optimizer has been retired since it was built. A cache that keys on it needs no
    per-step sweep at all.

    Materialising an EMPTY state (``__missing__``, i.e. a parameter's first step) does NOT
    move the generation: no cache can hold an empty state — the pointer tables dereference
    ``state["row"]``/``state["m"]`` at build and ``check_state_geometry`` skips a state
    that is not there — so there is nothing to invalidate, and bumping would cost every new
    parameter an extra full rebuild on its second step.
    """

    __slots__ = ("gen",)

    def __init__(self, items: Iterable[tuple[Any, Any]] = ()) -> None:
        # ``default_factory`` is kept at ``dict`` for anything that introspects it; the
        # ``__missing__`` override below is what actually builds a per-param state.
        super().__init__(dict)
        self.gen = [_next_epoch()]
        for key, value in items:
            dict.__setitem__(self, key, self._adopt(value))

    def _adopt(self, state: Any) -> Any:
        """A per-param state, watched by THIS mapping's generation cell."""
        if type(state) is WatchedParamState and state._gen is self.gen:
            return state
        return WatchedParamState(state, gen=self.gen)

    def __missing__(self, key: Any) -> WatchedParamState:
        state = WatchedParamState(gen=self.gen)
        dict.__setitem__(self, key, state)      # no bump — see the class docstring
        return state

    def __setitem__(self, key: Any, value: Any) -> None:
        self.gen[0] = _next_epoch()
        dict.__setitem__(self, key, self._adopt(value))

    def __delitem__(self, key: Any) -> None:
        self.gen[0] = _next_epoch()
        dict.__delitem__(self, key)

    def clear(self) -> None:
        if self:
            self.gen[0] = _next_epoch()
        dict.clear(self)

    def pop(self, key: Any, *default: Any) -> Any:
        if key in self:
            self.gen[0] = _next_epoch()
        return dict.pop(self, key, *default)

    def popitem(self) -> tuple[Any, Any]:
        self.gen[0] = _next_epoch()
        return dict.popitem(self)

    def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def __ior__(self, other: Any) -> WatchedState:  # type: ignore[misc]
        # See :meth:`WatchedParamState.__ior__`. On this mapping the leak had a second
        # half: ``opt.state |= {p: {...}}`` also left the incoming plain dict UNADOPTED, so
        # the mapping looked watched while one of its entries was not and every later
        # ``state[p]["m"] = ...`` through it was invisible.
        self.update(other)
        return self

    def setdefault(self, key: Any, default: Any = None) -> Any:  # type: ignore[override]
        if key in self:
            return dict.__getitem__(self, key)
        self[key] = default
        return dict.__getitem__(self, key)

    def __reduce__(self) -> tuple:
        return (_plain_state, (list(self.items()),))

    # ``defaultdict.copy``/``__copy__`` reconstruct as ``type(self)(default_factory, self)``,
    # which this ``__init__`` does not take (and a shallow copy sharing the generation cell
    # would be a trap anyway: a write through the copy would invalidate the ORIGINAL's
    # caches). Hand back the plain mapping, exactly as :meth:`__reduce__` does.
    def copy(self) -> defaultdict:
        return _plain_state(list(self.items()))

    __copy__ = copy


def _plain_state(items: list[tuple[Any, Any]]) -> defaultdict:
    """Unpickle/deepcopy target for :class:`WatchedState` — a plain ``defaultdict(dict)``.

    A watched mapping is REINSTALLED by ``WatchedStateMixin.__setstate__`` on the way in, so
    nothing is lost by keeping the serialised form free of kaon's classes.
    """
    return defaultdict(dict, items)


#: What :func:`state_generation` reports for an optimizer whose state is not watched — a
#: constant, so a cache built by one keeps validating for free.
_NO_GENERATION = 0


def state_generation(state: Any) -> int:
    """``state``'s identity generation, or :data:`_NO_GENERATION` if it is not watched.

    One ``getattr`` plus one list index per call, and every caller is once per group per
    step — this is the whole per-step cost of the state-identity guard.
    """
    gen = getattr(state, "gen", None)
    return _NO_GENERATION if gen is None else gen[0]


def _as_watched(value: Any) -> Any:
    """``value`` as a :class:`WatchedState`, if it is a mapping at all (idempotent)."""
    if type(value) is WatchedState:
        return value
    items = getattr(value, "items", None)
    return WatchedState(items()) if callable(items) else value


class WatchedStateMixin:
    """Make ``self.state`` a :class:`WatchedState` and keep it one, on EVERY route in.

    There are three, and each one needed its own answer:

    * **Assignment** — ``opt.state = defaultdict(dict)``, which user code and
      ``torch.optim.Optimizer.__init__`` both do. Caught by :meth:`__setattr__`, which
      re-wraps whatever mapping is handed in. Without it that assignment reopened the
      blind spot *silently and permanently*: the generation went from N to 0, which
      invalidated every cache exactly once (so the next step looked fine) and then left a
      counter-less mapping behind, after which a ``state[p]["m"] = …`` was invisible again
      — measured 4/4 retired buffers written on the step after.

      ``__setattr__`` and not a ``state`` PROPERTY on purpose: a property getter is a
      Python-level call on every ``self.state`` READ, and there are far more of those than
      the per-parameter loops suggest — counted by installing a counting property, an
      AdaPNM fused step on the 428-parameter bag performs 1687-1884 (two machines) and an
      Adakaon fused step 631, against 1 on an Adakaon native step. A property access
      measures 21-42 ns more than a plain instance attribute (73 vs 31 ns here), so that is
      +40…+71 µs/step for AdaPNM — one to two orders of magnitude above the 0.13-0.80 µs the
      whole guard costs. Interception belongs on the WRITE, which the same step does 1-4
      times.
    * **``__dict__`` writes** — ``Optimizer.__setstate__`` does ``self.__dict__.update(...)``
      and so bypasses ``__setattr__`` entirely. That is the path ``load_state_dict``,
      unpickling and ``deepcopy`` all funnel through, which is why :meth:`__setstate__`
      reinstalls afterwards.
    * **Construction** — the optimizer's own ``__init__`` calls
      :meth:`_install_state_watch` once, so the watch is in place before the first step
      even if a future torch stops assigning ``self.state`` at all.

    A reinstalled mapping carries a fresh generation value, which is itself the correct
    outcome: a load replaces every state tensor, and a number no cache can have recorded
    is exactly the invalidation that wants.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "state":
            value = _as_watched(value)
        super().__setattr__(name, value)

    def _install_state_watch(self) -> None:
        """Wrap ``self.state`` (idempotent)."""
        current = self.state                                # type: ignore[attr-defined]
        if type(current) is WatchedState:
            return
        self.state = _as_watched(current)                   # type: ignore[attr-defined]

    def _state_generation(self) -> int:
        """The state-identity generation, or :data:`_NO_GENERATION` when unwatched."""
        return state_generation(self.state)                 # type: ignore[attr-defined]

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)                         # type: ignore[misc]
        self._install_state_watch()


class ForeachSpec:
    """What one optimizer needs its cached chunks to carry. Built once, per class.

    ``factored_state`` / ``flat_state`` name the ``self.state`` keys whose per-param
    tensors the bucket bodies stack and write back through. Factored keys are taken raw
    (``state["row"]``, ``state["col"]`` are already in their own layout); flat keys go
    through :func:`~kaon._backend.flat_view`, which is what admits 0-D params into the
    ``L == 1`` bucket as length-1 views.

    ``extra_key(state, group)`` is the optimizer-specific component of the bucket key —
    AdaBelief / AdamP / ADOPT bucket by the per-parameter step so every slice of a bucket
    shares one bias correction, AdaMuon by its ``t`` when ``bias_correction`` is on. It is
    read once per param per step and its *value* is handed back on
    :attr:`ForeachChunk.key`; the cached plan survives the value changing as long as the
    **partition** it induces does not (see :meth:`ForeachPlanMixin._foreach_chunks`).

    ``key_major`` orders the buckets by that key first (all of key 0, factored then flat,
    then all of key 1, ...) instead of all-factored-then-all-flat. It exists to preserve
    ADOPT's pre-refactor bucket order exactly: bucket order is numerically irrelevant on
    its own (buckets touch disjoint params and disjoint state), but it decides the order
    stochastic-rounding draws are consumed in, so reordering would move bf16+SR weights.

    ``single_alias`` reproduces AdaMuon's ``_stack_fp32``: a one-element bucket is
    ``unsqueeze``d instead of stacked, aliasing the param's storage rather than copying it.

    The last four flags exist for the same reason as ``key_major``: an optimizer that
    bucketed differently *before* it moved onto this module has to keep doing so, because
    the partition and the order of the buckets decide how the stochastic-rounding draws
    are consumed and therefore reach bf16 weights (see the ``key_major`` note above).
    They are declarative on purpose — the bucketing an optimizer is pinned to should be
    readable from its class, not reverse-engineered from an override.

    ``matrixize`` (default on) reshapes an ``ndim > 2`` weight to its 2-D effective shape
    ``[R, C]``. Turn it OFF for an optimizer whose ``ndim >= 2`` state is per-coordinate in
    the param's own layout rather than factored: the bucket's ``eff`` is then the raw
    shape (any rank), :attr:`ForeachChunk.view` is the identity, and nothing needs the
    weight to be contiguous. KProdigy's ``second_moment="full"`` is that case.

    ``raw_shape_key`` adds ``tuple(g.shape)`` to the factored-family bucket key, so two
    conv kernels that matrixize to the *same* ``[R, C]`` (e.g. ``(16,8,3,3)`` and
    ``(16,24,3,1)``) stay in separate buckets. Only Lion needs it: its update is fully
    per-coordinate, so it bucketed by exact shape and never merged them.

    ``scalar_bucket`` keeps 0-D params out of the ``L == 1`` bucket they would otherwise
    share with real shape-``(1,)`` params (both are one element). Lion and KProdigy
    bucketed by exact shape and so kept them apart; the merge is a (small) win they do
    not get, in exchange for their bf16+SR weights not moving.

    ``insertion_order`` emits the buckets in first-appearance order across both families
    instead of all-factored-then-all-flat. Lion and KProdigy's full-second-moment path
    each kept ONE bucket dict keyed by shape, so that is the order they stepped in.

    HISTORY — a ``momentum_cache(group)`` flag used to ask a chunk to prebuild the
    identity-keyed ``{state["m"]: view(state["m"])}`` lookup the momentum codec called
    per param per step. It is gone: every optimizer on this plan now hands the codec
    :meth:`ForeachChunk.momentum_views`, so the codec's stacked path never calls
    ``view`` at all, and a dict keyed on ``Tensor.__hash__`` (a Python-level call in
    torch) was a strictly worse cache of the very same views.
    """

    __slots__ = ("extra_key", "factored_state", "flat_state", "insertion_order",
                 "key_major", "matrixize", "raw_shape_key", "scalar_bucket",
                 "single_alias")

    def __init__(
        self,
        *,
        factored_state: tuple[str, ...] = (),
        flat_state: tuple[str, ...] = (),
        extra_key: Callable[[dict[str, Any], dict[str, Any]], Hashable] | None = None,
        key_major: bool = False,
        single_alias: bool = False,
        matrixize: bool = True,
        raw_shape_key: bool = False,
        scalar_bucket: bool = False,
        insertion_order: bool = False,
    ) -> None:
        self.factored_state = factored_state
        self.flat_state = flat_state
        self.extra_key = extra_key
        self.key_major = key_major
        self.single_alias = single_alias
        self.matrixize = matrixize
        self.raw_shape_key = raw_shape_key
        self.scalar_bucket = scalar_bucket
        self.insertion_order = insertion_order


class ForeachChunk:
    """Cached derived views for ONE stacked chunk of a foreach path.

    A chunk owns the params and state dicts of one bucket slice plus everything the
    bucket body would otherwise rebuild each step:

    * :attr:`view` — the bucket's effective-layout callback (``mat``): ``t.view(R, C)``
      for a matrixized conv bucket, :func:`~kaon._backend.flat_view` for a bucket that
      admits 0-D params, identity otherwise.
    * :attr:`pviews` — ``[view(p.data) for p in plist]``, the list the weight decay,
      the projection stack and the final subtract all walk.
    * :attr:`cviews` — the same views over ``state["kahan_lo"]`` (the compact-Kahan
      residual bytes) when the bucket carries them, else ``None``; handed to
      :func:`kaon._backend.subtract_batched_` as ``comp=``.
    * :attr:`state_views` — one list per key named in the spec, in spec order.
    * :meth:`momentum_views` — the codec's own stacked-path view lists, built on first
      use and kept for the chunk's lifetime. :attr:`view` is also what the codec's
      ``mat`` argument is for: it is the *uncached* fallback the codec keeps for a
      layout :meth:`momentum_views` declined, and nothing else calls it.

    Gradient views are deliberately **not** cached: a retained view of ``p.grad`` keeps
    the previous step's gradient storage alive (``set_to_none=True`` allocates a fresh
    grad every backward), which would add a whole gradient set to peak memory — an
    unacceptable trade for optimizers whose pitch is memory. :meth:`grad_stack` avoids
    the per-param views a different way instead.

    Staleness is the caller's job — see :meth:`ForeachPlanMixin._foreach_chunks`.
    """

    __slots__ = ("cviews", "eff", "grad_reshape", "grad_uniform", "key", "key_index",
                 "length", "matrixize", "momentum_view_cache", "n", "plist", "pviews",
                 "single_alias", "state_views", "states", "view")

    def __init__(
        self,
        plist: list[Tensor],
        states: list[dict[str, Any]],
        spec: ForeachSpec,
        eff: tuple[int, int] | None,
        matrixize: bool,
        length: int,
        key_index: int,
        cached: bool,
    ) -> None:
        self.plist, self.states = plist, states
        self.eff, self.matrixize, self.length = eff, matrixize, length
        self.key_index, self.key = key_index, None
        self.n = n = len(plist)
        self.single_alias = spec.single_alias
        # ``grad_uniform``: the bucket's grads all share an ndim, so they stack raw and the
        # STACK gets reshaped once (``grad_reshape``, ``None`` when it already lands in
        # ``[N, *eff]``) instead of building N per-param views.
        self.grad_uniform = cached
        self.grad_reshape: tuple[int, ...] | None = None
        if eff is not None:                                   # factored bucket
            if matrixize:                                     # conv [N,O,I,kh,kw] -> [N,R,C]
                R, C = eff  # noqa: N806 — matrix dims (the stacked tensor is [N, R, C])
                self.view = lambda t: t.view(R, C)
                self.grad_reshape = (n, R, C)
                # A matrixized bucket is keyed on the EFFECTIVE shape, so it can hold
                # kernels whose raw shapes differ — (16,8,3,3) and (16,24,3,1) both
                # matrixize to (16,72). Those cannot be raw-stacked (``torch.stack``
                # wants one size), so such a bucket keeps the per-param view path.
                self.grad_uniform = cached and len({p.shape for p in plist}) == 1
            else:
                # ``eff`` is already the tensor's own layout — 2-D for a matrix bucket,
                # and any rank at all under ``ForeachSpec(matrixize=False)``.
                self.view = _identity
            keys = spec.factored_state
            self.state_views = tuple([s[k] for s in states] for k in keys)
        else:                                                 # non-factored bucket
            ndims = {p.ndim for p in plist}
            self.view = _identity if ndims == {1} else flat_view
            self.grad_uniform = cached and len(ndims) == 1
            if ndims == {0}:                                  # 0-D bag: [N] -> [N, 1]
                self.grad_reshape = (n, 1)
            keys = spec.flat_state
            view = self.view
            self.state_views = tuple([view(s[k]) for s in states] for k in keys)
        self.pviews = [self.view(p.data) for p in plist]
        view = self.view
        # ALL states, not states[0]: after a mid-run switch to kahan8 a chunk can mix a param
        # with a residual (fresh state) and one without (older state), and after a switch
        # between kahan8 and kahan16 one residual width with the other. None here lets the
        # plan's ensure_residuals hook allocate / convert them and rebuild the views; a
        # non-None ``cviews`` is therefore always of ONE dtype, so the hook checks [0] only.
        self.cviews = (
            [view(s["kahan_lo"]) for s in states]
            if states and all("kahan_lo" in s for s in states)
            and len({s["kahan_lo"].dtype for s in states}) == 1 else None
        )
        self.momentum_view_cache: tuple[_MomentumCodec, _StackedViews | None] | None = None

    def momentum_views(self, codec: _MomentumCodec) -> _StackedViews | None:
        """This chunk's cached view lists for ``codec``'s stacked path, or ``None``.

        The momentum codec's stacked entry points (``ema_stacked`` / ``store_stacked``
        / ``dequant_stacked``) walk the same per-param lists as the rest of a bucket
        body — ``mat(state["m"])``, the per-row ``state["m_scale"]`` views, the
        write-back targets — and rebuilt them on every step. This asks the codec to
        build them ONCE (``codec.stacked_views``) and hands the result to every call as
        ``views=``; ``None`` means the codec declined the layout (a non-contiguous
        buffer) and the caller keeps its uncached path.

        Keyed on the codec *instance*, so a group whose ``momentum_dtype`` changes
        cannot read another codec's layout (optimizers memoize one codec per dtype).
        Otherwise the lifetime is the chunk's: the lists alias ``state["m"]`` /
        ``state["m_scale"]`` exactly as :attr:`state_views` aliases ``row``/``col``/``v``,
        so they go stale on the same events and are dropped by the same plan
        invalidation (:meth:`ForeachPlanMixin._foreach_chunks` and
        :meth:`ForeachPlanMixin._clear_foreach_plans` — ``load_state_dict`` *replaces*
        the state tensors). An in-place requant, 4-bit's included, keeps them valid.
        """
        cache = self.momentum_view_cache
        if cache is None or cache[0] is not codec:
            eff = self.eff if self.eff is not None else (self.length,)
            self.momentum_view_cache = cache = (
                codec, codec.stacked_views(self.states, self.view, eff)
            )
        return cache[1]

    # KNOWN CEILING (measured on a 448x 0-D bag, after this cache): the residue is three
    # ``stack`` + three ``unbind`` per step (~1344 ``aten::select``). Only ONE of each pair
    # lives here (the flat state stack and its write-back); the other two are inside
    # ``_momentum_codec.ema_stacked`` and ``_backend.subtract_batched_``.
    #
    # NO VIEW CACHE CAN REMOVE THEM — not this module's, not the codec's. They select
    # into tensors that are FRESHLY ALLOCATED every step (the stacked update, the delta),
    # so there is no cross-step view to hold: what the caches here alias is state the
    # optimizer owns *across* steps. Removing the residue needs a persistent stacked
    # ``cat(out=)`` scratch buffer with cached unbind slices, which would buy only the
    # local pair (~10% of the remaining step) while pinning a stacked fp32 buffer per
    # bucket for the process's lifetime — a bad trade for optimizers whose pitch is
    # memory, and it would feed back into the free-VRAM-adaptive chunk budget. The two
    # outside would additionally need the codec / the backend to hand that buffer back.
    # Follow-up work, deliberately not attempted; see "Known ceiling" in
    # docs/foreach-batching.md.
    def grad_stack(self) -> Tensor:
        """This step's stacked fp32 gradient ``[N, *eff]``. Never cached — see the class
        docstring.

        ``torch.stack`` always writes a contiguous output, so stacking the **raw** grads
        and reshaping the *stack* once is element-for-element the same buffer as stacking
        N per-param reshapes — one ``view`` per step instead of N. That covers every
        bucket whose params share an ``ndim`` (all convs, all 1-D, all 0-D); only a bucket
        that genuinely mixes 0-D with shape-``(1,)`` params still builds per-param views.
        """
        if self.grad_uniform:
            raws = [p.grad for p in self.plist]
            g = raws[0].unsqueeze(0) if self.single_alias and self.n == 1 else torch.stack(raws)
            if self.grad_reshape is not None:
                g = g.view(self.grad_reshape)
            return g.float()
        view = self.view
        views = [view(p.grad) for p in self.plist]
        if self.single_alias and self.n == 1:
            return views[0].unsqueeze(0).float()
        return torch.stack(views).float()

    def param_stack(self) -> Tensor:
        """The stacked fp32 weights ``[N, *eff]`` (from the cached :attr:`pviews`)."""
        if self.single_alias and self.n == 1:
            return self.pviews[0].unsqueeze(0).float()
        return torch.stack(self.pviews).float()

    def value_stack(self, bf16_method: str) -> Tensor:
        """The stacked fp32 VALUE of the weights ``[N, *eff]`` for a term that reads them
        (weight decay): :meth:`param_stack`, except that a bf16 bucket under ``kahan8`` /
        ``kahan16`` decodes ``(weight, residual)`` — the stacked twin of
        :func:`kaon._backend.weight_value`. The residual views are the ones the plan already
        normalized for this step's write (:attr:`cviews`), decoded with the width they are
        stored in."""
        return self.value_and_stacks(bf16_method)[0]

    def value_and_stacks(self, bf16_method: str) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        """:meth:`value_stack`, plus — for a decoded compact-Kahan bucket — the stacked
        ``(weights, residuals)`` it decoded from, which the caller hands to the weight write
        (:func:`kaon._backend.subtract_batched_` ``stacked=``) so the same bucket is not
        stacked twice per step (``None`` otherwise). Valid until the weights are written.
        The decode is ONE Triton launch on CUDA (:func:`kaon._backend.decode_value`); the
        torch reference's ~10 integer kernels over the stack were a +30-80% self-CUDA
        regression of the foreach kahan8/kahan16 step (retime-016)."""
        cv = self.cviews
        if (cv is None or not is_compact_kahan(bf16_method)
                or self.pviews[0].dtype != torch.bfloat16):
            return self.param_stack(), None
        if self.single_alias and self.n == 1:
            w, lo = self.pviews[0].unsqueeze(0), cv[0].unsqueeze(0)
        else:
            w, lo = torch.stack(self.pviews), torch.stack(cv)
        return decode_value(w, lo, residual_bits_of(lo)), (w, lo)


class ForeachPlan:
    """One param group's bucketing plus the per-chunk view caches.

    ``buckets`` preserves the order the uncached code stepped in. ``chunks`` is that
    bucketing split by the current memory budget; the split is re-derived only when a
    bucket's chunk length actually changes, because the adaptive VRAM budget wobbles every
    step while ``budget // size`` almost never does.

    ``signature`` is the *partition* the spec's extra key induced when the plan was built
    — each param's key mapped to the index of the key's first appearance. It stays put
    while every param advances in lockstep (the normal case) even though the key VALUES
    change every step; :meth:`refresh` re-hangs the current values on the chunks.

    ``gen`` is the state-identity generation the plan was built at (see
    :class:`WatchedState`). ``witness`` covers the params; ``gen`` covers the STATE, whose
    buffers :attr:`ForeachChunk.state_views` aliases and which no param field can see —
    ``del opt.state[p]`` moves nothing else, and the chunk then steps the retired
    ``row``/``col``/``v`` (and never runs ``_init_state`` on the fresh state at all).
    """

    __slots__ = ("buckets", "chunks", "gen", "signature", "steps", "witness")

    def __init__(self, witness: tuple, signature: tuple[int, ...], buckets: list[tuple],
                 gen: int = _NO_GENERATION) -> None:
        self.witness, self.signature, self.buckets = witness, signature, buckets
        self.gen = gen
        self.chunks: list[ForeachChunk] | None = None
        self.steps: tuple[int, ...] = ()

    def rechunk(self, budget: int, spec: ForeachSpec, cached: bool) -> list[ForeachChunk]:
        steps = tuple(max(1, budget // size) for size, *_ in self.buckets)
        if self.chunks is not None and steps == self.steps:
            return self.chunks
        self.steps = steps
        self.chunks = chunks = [
            ForeachChunk(plist[i:i + n], states[i:i + n], spec,
                         eff, matrixize, length, ki, cached)
            for n, (_size, plist, states, eff, matrixize, length, ki) in zip(
                steps, self.buckets, strict=True)
            for i in range(0, len(plist), n)
        ]
        return chunks

    def refresh(self, values: list[Hashable]) -> None:
        """Hang this step's extra-key values on the (possibly cached) chunks."""
        for chunk in self.chunks:  # type: ignore[union-attr]
            chunk.key = values[chunk.key_index]


def _canonical_keys(
    params: list[Tensor], group: dict[str, Any], state: Mapping[Any, dict[str, Any]],
    extra_key: Callable[[dict[str, Any], dict[str, Any]], Hashable],
) -> tuple[list[int], list[Hashable]]:
    """``(per-param first-appearance index, distinct key values)`` for ``extra_key``."""
    seen: dict[Hashable, int] = {}
    values: list[Hashable] = []
    indices = []
    for p in params:
        k = extra_key(state[p], group)
        i = seen.get(k)
        if i is None:
            i = seen[k] = len(values)
            values.append(k)
        indices.append(i)
    return indices, values


class ForeachPlanMixin:
    """Bucketing, chunking and cached views for an optimizer's ``foreach`` path.

    Subclasses set :attr:`_FOREACH_SPEC` and call
    :meth:`_foreach_chunks` from their ``_step_foreach``. They must also drop the cache on
    the events a witness cannot see — ``load_state_dict`` (it *replaces* the state
    tensors the views alias) and an AutoLR base-state reset — via
    :meth:`_clear_foreach_plans`, and drop a single group's plan when that group falls
    back to the per-parameter loop, via :meth:`_drop_foreach_plan`.

    Set ``_foreach_cache_enabled = False`` on an instance to drop the cross-step cache:
    the plan and every view list are then rebuilt on every step. It is numerically a
    no-op either way; it exists as the A/B arm the speedup is measured against. It is
    NOT a byte-for-byte reproduction of the pre-plan code's *work*: a chunk still builds
    each list once per step where the old bucket bodies rebuilt one per use site, so the
    off arm lands between the old code and the cached one.
    """

    _FOREACH_SPEC: ForeachSpec = ForeachSpec()
    _foreach_cache_enabled: bool = True

    def _foreach_spec(self, group: dict[str, Any]) -> ForeachSpec:
        """The spec that governs ``group``'s bucketing. Normally the class attribute.

        Overridable because one optimizer's bucketing is a per-GROUP property: KProdigy's
        ``second_moment`` decides whether an ``ndim >= 2`` weight carries a factored
        ``row``/``col`` pair or a full per-coordinate ``v``, and those need different
        state keys and a different effective layout. ``second_moment`` is fixed for the
        life of a group (the state was allocated from it), so a cached plan can never
        outlive the spec that built it.
        """
        return self._FOREACH_SPEC

    @property
    def _foreach_plans(self) -> dict[int, ForeachPlan]:
        """group id -> :class:`ForeachPlan`. Lazy: ``Optimizer.__init__`` routes the
        constructor's groups through ``add_param_group`` before a subclass ``__init__``
        body ever runs."""
        plans = self.__dict__.get("_foreach_plan_cache")
        if plans is None:
            plans = self.__dict__["_foreach_plan_cache"] = {}
        return plans

    def _clear_foreach_plans(self) -> None:
        """Drop every cached plan (checkpoint load, state reset, new param group)."""
        plans = self.__dict__.get("_foreach_plan_cache")
        if plans:
            plans.clear()

    def _drop_foreach_plan(self, group: dict[str, Any]) -> None:
        """Drop one group's plan — used when the group steps per-parameter instead, so a
        cached plan only ever describes a group the foreach path actually stepped."""
        plans = self.__dict__.get("_foreach_plan_cache")
        if plans:
            plans.pop(id(group), None)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Add a group and drop the cached plans.

        ``id(group)`` is the cache key and CPython reuses ids of dead objects; a group
        added after one was dropped could otherwise land on a stale plan. Clearing is
        also what keeps the cache from growing without bound across repeated
        ``load_state_dict`` calls, which replace ``param_groups`` wholesale.
        """
        super().add_param_group(param_group)
        self._clear_foreach_plans()

    def _foreach_chunks(
        self, params: list[Tensor], group: dict[str, Any], budget: int,
    ) -> list[ForeachChunk]:
        """This step's chunks for ``group``, from the cached plan when it is still valid.

        The plan is rebuilt whenever :func:`param_witness` moves (the param set, the
        storage a param points at — which also covers dtype and device — or its
        contiguity), whenever the STATE-identity generation moves (a state buffer this
        plan's :attr:`ForeachChunk.state_views` alias was retired — see
        :class:`WatchedState`; a constant for an optimizer whose state is not watched, so
        this costs one integer compare), or whenever the spec's extra key stops inducing
        the same partition (a param that skipped a step and fell out of lockstep).
        Everything else that can invalidate it is event-driven:
        :meth:`_clear_foreach_plans` and :meth:`_drop_foreach_plan`.
        """
        spec = self._foreach_spec(group)
        cached = self._foreach_cache_enabled
        witness = param_witness(params)
        gen = state_generation(self.state)
        plan = self._foreach_plans.get(id(group)) if cached else None
        values: list[Hashable] = []
        if plan is not None:
            if plan.witness != witness or plan.gen != gen:
                plan = None
            elif spec.extra_key is not None:
                indices, values = _canonical_keys(params, group, self.state, spec.extra_key)
                if tuple(indices) != plan.signature:
                    plan = None
        if plan is None:
            plan, values = self._build_foreach_plan(params, group, witness, spec)
            # Read AFTER the build, which is free either way: ``_build_foreach_plan``
            # allocates state for any param that has none yet, and allocating it does NOT
            # move the counter (populating an absent key retires nothing — see
            # ``WatchedParamState.__setitem__``; verified 0 movement across 12
            # optimizer x route x bag combinations). Taking it here rather than reusing the
            # value read above is simply the tighter thing to do: it is the generation the
            # plan's views were actually built against, so no future write inside the build
            # can leave the plan claiming a generation older than its own contents.
            plan.gen = state_generation(self.state)
            if cached:
                self._foreach_plans[id(group)] = plan
        chunks = plan.rechunk(budget, spec, cached)
        if spec.extra_key is not None:
            plan.refresh(values)
        method = group.get("bf16_method", "")
        if is_compact_kahan(method):
            # A group switched to kahan8/kahan16 after its plan/state existed: give every bf16
            # chunk its residual views now (allocating zero residuals, with a one-time
            # warning), instead of letting the batched writer refuse the bucket. Allocating a
            # NEW key does not move the state generation, so the plan itself stays valid. A
            # switch BETWEEN the two widths converts the residuals (new tensors: a watched
            # rebinding, so the plan rebuilds next step; this step's views are rebuilt here).
            bits = residual_bits(method)
            want = residual_dtype(bits)
            for chunk in chunks:
                cv = chunk.cviews
                if (cv is None or cv[0].dtype != want) and chunk.plist[0].dtype == torch.bfloat16:
                    ensure_residuals(chunk.plist, chunk.states, bits)
                    chunk.cviews = [chunk.view(s[RESIDUAL_KEY]) for s in chunk.states]
        return chunks

    def _build_foreach_plan(
        self, params: list[Tensor], group: dict[str, Any], witness: tuple, spec: ForeachSpec,
    ) -> tuple[ForeachPlan, list[Hashable]]:
        """Bucket ``params`` so each bucket stacks into one tensor, and init their state.

        * ``ndim >= 2`` -> factored bucket, keyed by effective 2-D shape ``[N, R, C]``
          (or by the raw shape, any rank, under ``ForeachSpec(matrixize=False)``).
        * ``ndim <= 1`` (biases/norms, 0-D scalars) -> non-factored bucket, keyed by
          element count ``[N, L]`` (a 0-D scalar is a length-1 row, sharing the ``L == 1``
          bucket with shape-``(1,)`` params unless the spec sets ``scalar_bucket``).

        DEVICE is part of every bucket key: a bucket is stacked with ``torch.stack``,
        which refuses to mix devices, so a group holding a CPU and a CUDA weight of the
        same shape used to crash the whole step rather than step each on its own device.

        Buckets come out of ONE insertion-ordered dict, so ``insertion_order`` is the
        natural order and the default (all factored buckets, then all flat, each in
        first-appearance order) is a *stable* sort of it by family.
        """
        state_map = self.state
        init_state = self._init_state
        extra_key = spec.extra_key
        seen: dict[Hashable, int] = {}
        values: list[Hashable] = []
        signature: list[int] = []
        # key -> (size, plist, states, eff, matrixize, length, ki) — the tuple
        # ``ForeachPlan.rechunk`` consumes, with the two lists filled in place.
        found: dict[tuple[Any, ...], tuple] = {}
        for p in params:
            state = state_map[p]
            if not state:
                init_state(p, state, group)
            if extra_key is None:
                ki = 0
            else:
                k = extra_key(state, group)
                ki = seen.get(k)
                if ki is None:
                    ki = seen[k] = len(values)
                    values.append(k)
                signature.append(ki)
            g = p.grad
            if g.ndim >= 2:
                # conv kernels reshape to 2-D before factoring; ``matrixize=False``
                # specs keep every rank in the tensor's own layout instead.
                matrixize = spec.matrixize and g.ndim > 2
                eff = ((g.shape[0], g.numel() // g.shape[0]) if matrixize
                       else tuple(g.shape))
                key: tuple[Any, ...] = (0, eff, p.dtype, p.device, matrixize, ki,
                                        tuple(g.shape) if spec.raw_shape_key else None)
                entry = found.get(key)
                if entry is None:
                    entry = found[key] = (max(math.prod(eff), 1), [], [], eff,
                                          matrixize, 0, ki)
            else:  # ndim <= 1 — 0-D scalars ride as length 1 (numel == shape[0] for 1-D)
                length = g.numel()
                key = (1, length, p.dtype, p.device, ki,
                       spec.scalar_bucket and g.ndim == 0)
                entry = found.get(key)
                if entry is None:
                    entry = found[key] = (max(length, 1), [], [], None, False, length, ki)
            entry[1].append(p)
            entry[2].append(state)
        buckets = list(found.values())
        if not spec.insertion_order:
            buckets.sort(key=lambda b: b[3] is None)  # factored family first (stable)
        if spec.key_major:
            # Stable sort: key 0's factored buckets, then its flat ones, then key 1's, ...
            buckets.sort(key=lambda b: b[6])
        return ForeachPlan(witness, tuple(signature), buckets), values
