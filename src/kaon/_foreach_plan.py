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
consistent because ``Adakaon._fused_partition`` keys on :func:`param_witness` from here and
``Adakaon._invalidate_fused_caches`` drops both sets of caches in one call.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Mapping
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from kaon._backend import flat_view

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
    is the intended outcome. Same contract as ``kaon._fused_triton.param_witness``, kept
    separate from it because this one guards the NATIVE path, which must work in a build
    without Triton and must not import the Triton module at module scope. Adakaon's fused
    ROUTING (``_fused_partition``) calls this one for that reason; the pointer-array caches
    inside ``_fused_triton`` use their own copy.
    """
    return (tuple(map(id, plist)), tuple(map(_DATA_PTR, plist)), tuple(map(_IS_CONTIG, plist)))


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

    ``momentum_cache(group)`` says whether a chunk should prebuild the ``mat`` lookup the
    momentum codec calls per param per step; ``None`` for optimizers that do not hand a
    ``mat`` callback to the codec. **Only Adakaon sets it**, because it is the one
    optimizer that still hands the codec a bare ``mat``; everywhere else the codec's
    stacked path takes its per-param view lists from
    :meth:`ForeachChunk.momentum_views`, so it never calls ``mat``, and a dict keyed on
    ``Tensor.__hash__`` would be a strictly worse cache of the same views.

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
    """

    __slots__ = ("extra_key", "factored_state", "flat_state", "insertion_order",
                 "key_major", "matrixize", "momentum_cache", "raw_shape_key",
                 "scalar_bucket", "single_alias")

    def __init__(
        self,
        *,
        factored_state: tuple[str, ...] = (),
        flat_state: tuple[str, ...] = (),
        extra_key: Callable[[dict[str, Any], dict[str, Any]], Hashable] | None = None,
        key_major: bool = False,
        momentum_cache: Callable[[dict[str, Any]], bool] | None = None,
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
        self.momentum_cache = momentum_cache
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
    * :attr:`state_views` — one list per key named in the spec, in spec order.
    * :attr:`mat` — what to hand the momentum codec. Usually :attr:`view`; when the view
      is a real reshape, an identity-keyed lookup of prebuilt momentum views is cheaper,
      because the codec calls it once (float) or twice (int8) per param per step. Only
      the *uncached* codec path calls it — see :meth:`momentum_views`.
    * :meth:`momentum_views` — the codec's own stacked-path view lists, built on first
      use and kept for the chunk's lifetime.

    Gradient views are deliberately **not** cached: a retained view of ``p.grad`` keeps
    the previous step's gradient storage alive (``set_to_none=True`` allocates a fresh
    grad every backward), which would add a whole gradient set to peak memory — an
    unacceptable trade for optimizers whose pitch is memory. :meth:`grad_stack` avoids
    the per-param views a different way instead.

    Staleness is the caller's job — see :meth:`ForeachPlanMixin._foreach_chunks`.
    """

    __slots__ = ("eff", "grad_reshape", "grad_uniform", "key", "key_index", "length",
                 "mat", "matrixize", "momentum_view_cache", "n", "plist", "pviews",
                 "single_alias", "state_views", "states", "view")

    def __init__(
        self,
        plist: list[Tensor],
        states: list[dict[str, Any]],
        spec: ForeachSpec,
        group: dict[str, Any],
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
        # An identity-keyed lookup of prebuilt views only pays when there is a view to
        # save — ``Tensor.__hash__`` is a Python-level call in torch, so a dict hit is
        # *more* expensive than ``_identity``. 4bit is excluded by the optimizer's
        # ``momentum_cache`` predicate: its ``m`` is a packed byte string, not a momentum
        # in the effective layout (``view`` would raise), and its codec never calls ``mat``.
        self.mat = self.view
        if (cached and self.view is not _identity and spec.momentum_cache is not None
                and spec.momentum_cache(group)):
            self.mat = {s["m"]: self.view(s["m"]) for s in states}.__getitem__
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
    # ``_momentum_codec.ema_stacked`` and ``_backend.subtract_batched_``. Persistent
    # ``cat(out=)`` scratch buffers with cached unbind slices would therefore buy only the
    # local pair (~10% of the remaining step) while pinning a stacked fp32 buffer per bucket
    # for the process's lifetime — a bad trade for optimizers whose pitch is memory, and it
    # would feed back into the free-VRAM-adaptive chunk budget. Removing the other two needs
    # the codec to hand back a reusable buffer: follow-up work.
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
    """

    __slots__ = ("buckets", "chunks", "signature", "steps", "witness")

    def __init__(self, witness: tuple, signature: tuple[int, ...], buckets: list[tuple]) -> None:
        self.witness, self.signature, self.buckets = witness, signature, buckets
        self.chunks: list[ForeachChunk] | None = None
        self.steps: tuple[int, ...] = ()

    def rechunk(self, budget: int, spec: ForeachSpec, group: dict[str, Any],
                cached: bool) -> list[ForeachChunk]:
        steps = tuple(max(1, budget // size) for size, *_ in self.buckets)
        if self.chunks is not None and steps == self.steps:
            return self.chunks
        self.steps = steps
        self.chunks = chunks = [
            ForeachChunk(plist[i:i + n], states[i:i + n], spec, group,
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
        contiguity) or whenever the spec's extra key stops inducing the same partition
        (a param that skipped a step and fell out of lockstep). Everything else that can
        invalidate it is event-driven: :meth:`_clear_foreach_plans` and
        :meth:`_drop_foreach_plan`.
        """
        spec = self._foreach_spec(group)
        cached = self._foreach_cache_enabled
        witness = param_witness(params)
        plan = self._foreach_plans.get(id(group)) if cached else None
        values: list[Hashable] = []
        if plan is not None:
            if plan.witness != witness:
                plan = None
            elif spec.extra_key is not None:
                indices, values = _canonical_keys(params, group, self.state, spec.extra_key)
                if tuple(indices) != plan.signature:
                    plan = None
        if plan is None:
            plan, values = self._build_foreach_plan(params, group, witness, spec)
            if cached:
                self._foreach_plans[id(group)] = plan
        chunks = plan.rechunk(budget, spec, group, cached)
        if spec.extra_key is not None:
            plan.refresh(values)
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
