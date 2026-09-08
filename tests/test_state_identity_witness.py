"""State-identity staleness: a cross-step cache must never step a RETIRED state tensor.

Every cross-step cache in kaon is validated against a PARAMETER witness — ids,
``data_ptr``s, contiguity (:func:`kaon._foreach_plan.param_witness`). None of those
fields observes ``self.state``, so a state buffer could be swapped out from under a
cache while every parameter stood still, and the next step wrote the buffer the
optimizer had already let go of:

* ``del opt.state[p]`` — the recovery ``check_state_geometry`` documents for a refused
  rebind. Confirmed on the fused big / one-block / 1-D / 0-D routes AND on the native
  foreach plan (``ForeachChunk.state_views``): the retired ``m``/``row``/``col``/``v``
  were written, byte for byte identically with and without the caches.
* ``opt.state[p].clear()`` — same, and worse: the very same dict is refilled with FRESH
  buffers by the next ``_init_state``, so the optimizer holds both generations at once.
* ``opt.state[p]["m"] = ...`` / ``["row"] = ...`` — an external EMA, a partial
  ``load_state_dict`` that bypasses the optimizer's own loader, a checkpoint tool that
  casts one buffer, a requant that does not follow the in-place codec contract.

The guard is :class:`kaon._foreach_plan.WatchedState`: ``self.state`` counts every change
of state IDENTITY, and every cross-step cache carries that count. Cost is zero per step
(nothing in a steady-state step rebinds a watched key) — which is itself asserted here,
because a future per-step rebind would silently turn every step into a cache rebuild.

The per-parameter path is the reference throughout: it re-reads ``self.state[p]`` on every
step and therefore never had the blind spot.
"""
from __future__ import annotations

import copy
import io
import pickle

import pytest
import torch

from kaon import Adakaon, AdaPNM
from kaon._foreach_plan import WATCHED_STATE_KEYS, WatchedParamState, WatchedState
from kaon._fused_triton import HAS_TRITON

CUDA = torch.cuda.is_available()
FUSED_OK = HAS_TRITON and CUDA
requires_fused = pytest.mark.skipif(
    not FUSED_OK, reason="Triton fused kernels require CUDA + Triton"
)

_CFG = dict(lr=2e-3, betas=(0.9, 0.999), momentum_dtype="float32", weight_decay=0.02)
_ADAKAON_CFG = dict(_CFG, cautious=True, gradient_centralization=True)

# One representative bag per fused route (routing is asserted, not assumed).
_ROUTES = {
    "one_block": [(8, 16)] * 4,
    "one_dim": [(32,)] * 3,
    "zero_dim": [()] * 3,
    "big": [(512, 512)] * 2,
}
# Which state keys each route's cached pointer tables / views actually bake.
_MUTATIONS = ("del", "clear", "reassign_m", "reassign_second_moment")


# --------------------------------------------------------------------------- helpers
def _bag(shapes, dtype=torch.float32, seed=0, device="cpu"):
    g = torch.Generator(device=device).manual_seed(seed)
    return [torch.randn(s, generator=g, device=device, dtype=dtype).requires_grad_(True)
            for s in shapes]


def _clone(ps):
    return [p.detach().clone().requires_grad_(True) for p in ps]


def _second_moment_keys(state):
    return [k for k in ("row", "v") if k in state]


def _momentum_keys(state):
    return [k for k in ("m", "m_pos") if k in state]


def _mutate(opt, params, kind):
    """Apply ``kind`` to every param's state; return the (tensor, snapshot) sentinels.

    Only the buffers the mutation actually RETIRES are snapshotted — a live buffer the
    step is supposed to advance would otherwise read as a violation.
    """
    retired = []
    for p in params:
        st = opt.state[p]
        if kind in ("del", "clear"):
            keys = [k for k, v in st.items() if torch.is_tensor(v)]
        elif kind == "reassign_m":
            keys = _momentum_keys(st)
        else:
            keys = _second_moment_keys(st)
        assert keys, f"{kind}: nothing to retire in state keys {sorted(st)}"
        for k in keys:
            retired.append((k, st[k], st[k].detach().clone()))
        if kind == "del":
            del opt.state[p]
        elif kind == "clear":
            st.clear()
        else:
            for k in keys:
                st[k] = st[k].detach().clone()
    return retired


def _assert_untouched(retired, what):
    dirty = [k for k, live, snap in retired if not torch.equal(live, snap)]
    assert not dirty, (
        f"{what}: the step wrote {len(dirty)}/{len(retired)} RETIRED state buffers "
        f"({sorted(set(dirty))}) — a cross-step cache is still addressing them"
    )


def _drive(pairs, steps, seed, device, mutate=None):
    """Step every (params, opt) pair on IDENTICAL grads; run ``mutate(step, pairs)`` first."""
    gen = torch.Generator().manual_seed(seed)     # CPU: this only draws the per-step seed
    for step in range(steps):
        out = mutate(step, pairs) if mutate is not None else None
        draw = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=gen).item())
        for plist, opt in pairs:
            g = torch.Generator(device=device).manual_seed(draw)
            for p in plist:
                p.grad = torch.randn(tuple(p.shape), generator=g, device=device, dtype=p.dtype)
            opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    return out


def _maxdiff(pa, pb):
    return max((a.detach().float() - b.detach().float()).abs().max().item()
               for a, b in zip(pa, pb, strict=True))


def _routes_of(opt):
    ob, big, od, nat = [], [], [], []
    for entry in opt._fused_part.values():
        o, b, d, n = entry[-4:]
        ob += o
        big += b
        od += d
        nat += n
    return ob, big, od, nat


def _assert_route(opt, route):
    ob, big, od, nat = _routes_of(opt)
    assert not nat, f"{route}: {len(nat)} params fell to the native path"
    if route == "one_block":
        assert ob and not big and not od
    elif route == "big":
        assert big and not ob and not od
    else:
        assert od and not ob and not big


def _mutation_case(cls, cfg, shapes, kind, device, arm, steps=4, cut=2, **arm_kw):
    """``arm`` (fused / foreach) vs the per-parameter reference, with ``kind`` applied at ``cut``.

    Returns ``(params_arm, params_ref, retired_sentinels, opt_arm)``.
    """
    pa = _bag(shapes, seed=3, device=device)
    pr = _clone(pa)
    oa = cls(pa, **cfg, **arm_kw)
    ref = cls(pr, fused=False, foreach=False, **cfg)
    box: list = []

    def mutate(step, _pairs):
        if step != cut:
            return
        box.append(_mutate(oa, pa, kind))
        _mutate(ref, pr, kind)

    _drive([(pa, oa), (pr, ref)], steps, 11, device, mutate=mutate)
    assert box, f"{arm}: the mutation hook never ran"
    return pa, pr, box[0], oa


# ============================================================ 1. the watch primitives
def test_watched_state_creates_watched_per_param_dicts():
    st = WatchedState()
    key = torch.zeros(2)
    inner = st[key]
    assert isinstance(inner, WatchedParamState)
    assert inner is st[key]


def test_creating_an_empty_state_does_not_move_the_generation():
    """An EMPTY per-param state is in no cache, so materialising one must be free.

    Otherwise the first step of every new parameter would cost an extra full rebuild.
    """
    st = WatchedState()
    before = st.gen[0]
    for _ in range(4):
        st[torch.zeros(2)]
    assert st.gen[0] == before


@pytest.mark.parametrize("key", sorted(WATCHED_STATE_KEYS))
def test_rebinding_a_watched_key_moves_the_generation(key):
    st = WatchedState()
    inner = st[torch.zeros(2)]
    inner[key] = torch.zeros(3)
    gen = st.gen[0]
    inner[key] = torch.zeros(3)
    assert st.gen[0] > gen, f"rebinding state[{key!r}] must move the generation"


def test_writing_the_same_object_back_does_not_move_the_generation():
    st = WatchedState()
    inner = st[torch.zeros(2)]
    buf = torch.zeros(3)
    inner["m"] = buf
    gen = st.gen[0]
    inner["m"] = buf                      # an idempotent re-store retires nothing
    assert st.gen[0] == gen


@pytest.mark.parametrize("write", ["setitem", "setdefault", "update"])
def test_populating_an_absent_key_does_not_move_the_generation(write):
    """A key that was ABSENT was in no cache — so ``_init_state`` must cost nothing.

    Five watched keys are written on a parameter's first step; bumping there would
    charge every new parameter a full cache rebuild on the step after.
    """
    st = WatchedState()
    inner = st[torch.zeros(2)]
    gen = st.gen[0]
    if write == "setitem":
        inner["m"] = torch.zeros(3)
    elif write == "setdefault":
        inner.setdefault("m", torch.zeros(3))
    else:
        inner.update({"m": torch.zeros(3), "row": torch.zeros(2)})
    assert st.gen[0] == gen
    assert "m" in inner


def test_scalar_bookkeeping_does_not_move_the_generation():
    """``state["step"] += 1`` (AdaPNM, every param, every step) must not invalidate."""
    st = WatchedState()
    inner = st[torch.zeros(2)]
    inner["step"] = 0
    gen = st.gen[0]
    for _ in range(100):
        inner["step"] += 1
    assert st.gen[0] == gen


def test_the_unwatched_write_bypass_only_covers_scalar_keys():
    """``AdaPNM._prepare_param_steps`` writes ``step`` through ``dict.__setitem__``.

    That skips the identity hook, which is sound only while ``step`` is outside
    :data:`WATCHED_STATE_KEYS` — the write is 428 Python calls per step on the reference bag
    (+96 µs, 1.06% of its fused step) and had to come off the hot path. If a future change
    ever routes a BAKED key through the bypass, this is the guard that says so.
    """
    from kaon.adapnm import _set_unwatched

    assert _set_unwatched is dict.__setitem__
    assert "step" not in WATCHED_STATE_KEYS
    # Every key the bypass is used for, spelled out.
    assert {"step"}.isdisjoint(WATCHED_STATE_KEYS)


@pytest.mark.parametrize("drop", ["del", "clear", "pop", "popitem", "outer_del",
                                  "outer_clear", "outer_pop", "outer_set", "update"])
def test_every_removal_route_moves_the_generation(drop):
    st = WatchedState()
    key = torch.zeros(2)
    inner = st[key]
    inner["m"] = torch.zeros(3)
    gen = st.gen[0]
    if drop == "del":
        del inner["m"]
    elif drop == "clear":
        inner.clear()
    elif drop == "pop":
        inner.pop("m")
    elif drop == "popitem":
        inner.popitem()
    elif drop == "update":
        inner.update({"m": torch.zeros(3)})
    elif drop == "outer_del":
        del st[key]
    elif drop == "outer_clear":
        st.clear()
    elif drop == "outer_pop":
        st.pop(key)
    else:
        st[key] = {"m": torch.zeros(3)}
    assert st.gen[0] > gen, f"{drop} must move the generation"


def test_outer_setitem_adopts_a_plain_dict():
    """A plain dict handed in through ``opt.state[p] = ...`` must be watched from then on."""
    st = WatchedState()
    key = torch.zeros(2)
    st[key] = {"m": torch.zeros(3)}
    assert isinstance(st[key], WatchedParamState)
    gen = st.gen[0]
    st[key]["m"] = torch.zeros(3)
    assert st.gen[0] > gen


@pytest.mark.parametrize("how", ["method", "copy_module"])
def test_shallow_copying_the_state_mapping_yields_a_plain_mapping(how):
    """``defaultdict.copy`` reconstructs as ``type(self)(default_factory, self)``.

    That signature does not exist here, and a shallow copy sharing the generation cell
    would be a trap besides: a write through the copy would invalidate the ORIGINAL's
    caches.
    """
    st = WatchedState()
    st[torch.zeros(2)]["m"] = torch.zeros(3)
    back = st.copy() if how == "method" else copy.copy(st)
    assert not isinstance(back, WatchedState)
    assert isinstance(back, dict) and len(back) == 1


@pytest.mark.parametrize("dump", ["pickle", "deepcopy"])
def test_watched_dicts_serialise_as_plain_dicts(dump):
    """``torch.save(opt.state_dict())`` pickles the INNER dicts by reference.

    They must come back as ordinary dicts (no class to import, no generation cell
    riding along into the checkpoint).
    """
    st = WatchedState()
    key = torch.zeros(2)
    st[key]["m"] = torch.arange(3.0)
    st[key]["step"] = 7
    for obj in (st, st[key]):
        back = (pickle.loads(pickle.dumps(obj)) if dump == "pickle"
                else copy.deepcopy(obj))
        assert type(back) in (dict, type(back)) and isinstance(back, dict)
        assert not isinstance(back, (WatchedState, WatchedParamState)), (
            f"{type(obj).__name__} survived {dump} as a watched dict"
        )


# ============================================================ 2. Adakaon, native paths
@pytest.mark.parametrize("kind", _MUTATIONS)
def test_foreach_plan_drops_a_retired_state_buffer(kind):
    """The native foreach plan caches ``row``/``col``/``v`` VIEWS per chunk."""
    pa, pr, retired, opt = _mutation_case(
        Adakaon, _ADAKAON_CFG, [(8, 16)] * 3 + [(32,)] * 2, kind, "cpu", "foreach",
        foreach=True,
    )
    _assert_untouched(retired, f"foreach/{kind}")
    d = _maxdiff(pa, pr)
    assert d < 1e-5, f"foreach/{kind}: max|Δp| vs the per-param reference = {d:.2e}"


def test_foreach_plan_is_reused_while_the_state_stands_still():
    """The generation check must not cost a rebuild per step (the whole point of (c))."""
    pl = _bag([(8, 16)] * 3 + [(32,)] * 2, seed=5)
    opt = Adakaon(pl, foreach=True, **_ADAKAON_CFG)
    _drive([(pl, opt)], 3, 17, "cpu")
    plans = dict(opt._foreach_plans)
    chunks = {gid: plan.chunks for gid, plan in plans.items()}
    assert chunks, "no foreach plan was cached"
    _drive([(pl, opt)], 3, 19, "cpu")
    assert {gid: p.chunks for gid, p in opt._foreach_plans.items()} == chunks, (
        "the foreach plan was rebuilt on a steady-state step"
    )


def test_state_generation_stands_still_across_steady_state_steps():
    pl = _bag([(8, 16)] * 3 + [(32,)] * 2, seed=5)
    opt = Adakaon(pl, foreach=True, **_ADAKAON_CFG)
    _drive([(pl, opt)], 3, 17, "cpu")
    gen = opt.state.gen[0]
    _drive([(pl, opt)], 5, 19, "cpu")
    assert opt.state.gen[0] == gen, (
        "a steady-state step rebinds a watched state key — every step is now a full "
        "cache rebuild"
    )


# ============================================================ 3. Adakaon, fused routes
@requires_fused
@pytest.mark.parametrize("route", list(_ROUTES))
@pytest.mark.parametrize("kind", _MUTATIONS)
def test_fused_route_drops_a_retired_state_buffer(route, kind):
    pa, pr, retired, opt = _mutation_case(
        Adakaon, _ADAKAON_CFG, _ROUTES[route], kind, "cuda", f"fused/{route}",
        fused=True,
    )
    _assert_route(opt, route)
    _assert_untouched(retired, f"fused/{route}/{kind}")
    d = _maxdiff(pa, pr)
    assert d < 1e-5, f"fused/{route}/{kind}: max|Δp| vs the per-param reference = {d:.2e}"


@requires_fused
@pytest.mark.parametrize("kind", ["del", "clear"])
def test_fused_deterministic_big_bucket_drops_a_retired_state_buffer(kind):
    pa, pr, retired, opt = _mutation_case(
        Adakaon, _ADAKAON_CFG, [(512, 512)] * 2, kind, "cuda", "fused/big-det",
        fused=True, deterministic_reductions=True,
    )
    _assert_untouched(retired, f"fused/big-det/{kind}")
    d = _maxdiff(pa, pr)
    assert d < 1e-5, f"fused/big-det/{kind}: max|Δp| = {d:.2e}"


@requires_fused
def test_fused_partition_is_reused_while_the_state_stands_still():
    pl = _bag(_ROUTES["one_block"] + _ROUTES["one_dim"], seed=5, device="cuda")
    opt = Adakaon(pl, fused=True, **_ADAKAON_CFG)
    _drive([(pl, opt)], 3, 17, "cuda")
    parts = {gid: entry[-4:] for gid, entry in opt._fused_part.items()}
    caches = (dict(opt._fused_ob_caches), dict(opt._fused_od_caches))
    _drive([(pl, opt)], 3, 19, "cuda")
    assert {gid: e[-4:] for gid, e in opt._fused_part.items()} == parts, (
        "the fused partition was rebuilt on a steady-state step"
    )
    assert (dict(opt._fused_ob_caches), dict(opt._fused_od_caches)) == caches, (
        "a fused pointer cache was rebuilt on a steady-state step"
    )


@requires_fused
def test_del_state_after_a_refused_shape_rebind_recovers():
    """``check_state_geometry``'s documented RECOVERY, on a bag whose caches are BUILT.

    The message tells the user to ``del opt.state[p]`` and carry on; before the state
    watch that was safe only because the refusing cache had never been built.
    """
    pl = _bag([(16, 64)] * 3, seed=7, device="cuda")
    opt = Adakaon(pl, fused=True, **_ADAKAON_CFG)
    _drive([(pl, opt)], 3, 21, "cuda")
    for p in pl:
        p.data = p.data.view(64, 16).clone()
    for p in pl:
        p.grad = torch.randn(64, 16, device="cuda")
    with pytest.raises(RuntimeError, match="does not describe their current SHAPE"):
        opt.step()
    retired = [(k, t, t.detach().clone())
               for p in pl for k, t in opt.state[p].items() if torch.is_tensor(t)]
    for p in pl:
        del opt.state[p]
    for p in pl:
        p.grad = torch.randn(64, 16, device="cuda")
    opt.step()
    torch.cuda.synchronize()
    _assert_untouched(retired, "recovery after a refused shape rebind")
    for p in pl:
        assert torch.isfinite(p.detach()).all()


# ============================================================ 4. AdaPNM
@requires_fused
@pytest.mark.parametrize("route", ["one_block", "one_dim", "zero_dim", "big"])
@pytest.mark.parametrize("kind", _MUTATIONS)
def test_adapnm_fused_route_drops_a_retired_state_buffer(route, kind):
    pa, pr, retired, opt = _mutation_case(
        AdaPNM, _CFG, _ROUTES[route], kind, "cuda", f"adapnm/{route}", fused=True,
    )
    _assert_untouched(retired, f"adapnm/{route}/{kind}")
    d = _maxdiff(pa, pr)
    assert d < 1e-5, f"adapnm/{route}/{kind}: max|Δp| = {d:.2e}"


@requires_fused
def test_adapnm_fused_partition_is_reused_while_the_state_stands_still():
    pl = _bag(_ROUTES["one_block"], seed=5, device="cuda")
    opt = AdaPNM(pl, fused=True, **_CFG)
    _drive([(pl, opt)], 3, 17, "cuda")
    caches = dict(opt._fused_ob_caches)
    parts = {gid: e[-4:] for gid, e in opt._fused_part.items()}
    _drive([(pl, opt)], 3, 19, "cuda")
    assert dict(opt._fused_ob_caches) == caches, "AdaPNM rebuilt a pointer cache per step"
    assert {gid: e[-4:] for gid, e in opt._fused_part.items()} == parts


# ============================================================ 5. wrappers
@requires_fused
@pytest.mark.parametrize("kind", _MUTATIONS)
def test_msam_drops_a_retired_inner_momentum(kind):
    """MSAM caches ``m``/``m_scale`` pointer ARRAYS keyed on the inner state dicts."""
    from kaon import MSAM

    pl = _bag([(8, 16)] * 4, seed=9, device="cuda")
    opt = MSAM(pl, rho=0.1, fused=True, **_ADAKAON_CFG)
    inner = opt.inner
    _drive([(pl, opt)], 3, 23, "cuda")
    retired = _mutate(inner, pl, kind)
    for p in pl:
        p.grad = torch.randn(tuple(p.shape), device="cuda")
    opt.step()
    torch.cuda.synchronize()
    _assert_untouched(retired, f"msam/{kind}")
    for p in pl:
        assert torch.isfinite(p.detach()).all()


@requires_fused
@pytest.mark.parametrize("kind", ["del", "reassign_m"])
def test_nekaon_drops_a_retired_inner_momentum(kind):
    from kaon import Nekaon

    pl = _bag([(8, 16)] * 4, seed=9, device="cuda")
    opt = Nekaon(pl, k=1.5, fused=True, **_ADAKAON_CFG)
    inner = opt.inner
    _drive([(pl, opt)], 3, 25, "cuda")
    retired = _mutate(inner, pl, kind)
    for p in pl:
        p.grad = torch.randn(tuple(p.shape), device="cuda")
    opt.step()
    torch.cuda.synchronize()
    _assert_untouched(retired, f"nekaon/{kind}")
    for p in pl:
        assert torch.isfinite(p.detach()).all()


# ============================================================ 6. checkpoint plumbing
@pytest.mark.parametrize("cls,cfg", [(Adakaon, _ADAKAON_CFG), (AdaPNM, _CFG)])
def test_load_state_dict_reinstalls_the_watch(cls, cfg):
    """``Optimizer.load_state_dict`` REPLACES ``self.state`` with a plain defaultdict."""
    pl = _bag([(8, 16)] * 2, seed=11)
    opt = cls(pl, **cfg)
    _drive([(pl, opt)], 2, 27, "cpu")
    buf = io.BytesIO()
    torch.save(opt.state_dict(), buf)
    buf.seek(0)
    opt.load_state_dict(torch.load(buf, weights_only=False))
    assert isinstance(opt.state, WatchedState), (
        f"{cls.__name__}.load_state_dict left an unwatched state mapping"
    )
    for p in pl:
        assert isinstance(opt.state[p], WatchedParamState)
    _drive([(pl, opt)], 2, 29, "cpu")


@pytest.mark.parametrize("cls,cfg", [(Adakaon, _ADAKAON_CFG), (AdaPNM, _CFG)])
def test_deepcopy_keeps_the_watch(cls, cfg):
    pl = _bag([(8, 16)] * 2, seed=13)
    opt = cls(pl, **cfg)
    _drive([(pl, opt)], 2, 31, "cpu")
    twin = copy.deepcopy(opt)
    assert isinstance(twin.state, WatchedState)
    for p in twin.param_groups[0]["params"]:
        assert isinstance(twin.state[p], WatchedParamState)


@pytest.mark.parametrize("cls,cfg", [(Adakaon, _ADAKAON_CFG), (AdaPNM, _CFG)])
def test_state_dict_carries_plain_dicts(cls, cfg):
    """A checkpoint must not embed the watch classes (unpicklable in a plain-torch load)."""
    pl = _bag([(8, 16)] * 2, seed=15)
    opt = cls(pl, **cfg)
    _drive([(pl, opt)], 2, 33, "cpu")
    blob = pickle.dumps(opt.state_dict())
    assert b"WatchedParamState" not in blob and b"WatchedState" not in blob
    back = pickle.loads(blob)
    for entry in back["state"].values():
        assert type(entry) is dict
