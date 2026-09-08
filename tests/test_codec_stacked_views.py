"""Tests for the momentum codecs' cached stacked-view lists (``stacked_views``).

The stacked codec paths (``ema_stacked`` / ``store_stacked`` / ``dequant_stacked``)
rebuilt, on every step, lists that are pure functions of tensors the optimizer already
owns: ``[mat(s["m"]) for s in states]``, ``[s["m_scale"].view(rowshape) ...]``, the
write-back targets. :meth:`_MomentumCodec.stacked_views` precomputes them once and the
callers hand them back through ``views=``. Three properties have to hold:

* **Numerically invisible.** ``views=`` must be bit-identical to the uncached call, for
  every ``momentum_dtype`` and every bucket layout (2-D, matrixized conv, 1-D, 0-D).
* **Aliasing, not copying.** Every cached entry must be a *view* of the live
  ``state["m"]`` / ``state["m_scale"]`` — it pins no memory and the write-backs reach
  the real buffers (the storage-identity contract).
* **Never stale.** A cached view of a buffer the optimizer no longer owns would step
  detached memory: the layouts it cannot alias must decline the cache, an ``eff``
  mismatch must be ignored, and the per-chunk cache must be rebuilt when the codec
  changes and dropped with the plan on ``load_state_dict``.
"""

from __future__ import annotations

import io

import pytest
import torch

from kaon import ADOPT, AdaBelief, Adakaon, AdamP, AdaMuon
from kaon._backend import flat_view
from kaon._foreach_plan import ForeachChunk
from kaon._momentum_codec import _make_codec, _StackedViews

MDS = ["float32", "bfloat16", "int8", "4bit"]

# (bucket name, param shape, effective shape, effective-layout view callback)
LAYOUTS = {
    "2d": ((4, 3), (4, 3), lambda t: t),
    "conv": ((2, 2, 3, 3), (2, 18), lambda t: t.view(2, 18)),
    "1d": ((5,), (5,), lambda t: t),
    "0d": ((), (1,), flat_view),
}
LAYOUT_NAMES = list(LAYOUTS)

GROUP = {"momentum_4bit_block": 8}


def make_states(md, shape, n=3, *, seed=0):
    """``n`` fresh codec states for params of ``shape``, plus the codec."""
    torch.manual_seed(seed)
    codec = _make_codec(md)
    states = []
    for _ in range(n):
        state = {}
        codec.init_state(state, torch.zeros(shape), GROUP)
        states.append(state)
    return codec, states


def state_snapshot(states):
    return [
        {k: v.clone() for k, v in s.items() if torch.is_tensor(v)}
        for s in states
    ]


def assert_states_equal(a, b):
    for i, (sa, sb) in enumerate(zip(a, b, strict=True)):
        assert sa.keys() == sb.keys()
        for k in sa:
            assert torch.equal(sa[k], sb[k]), f"state {i} key {k!r} differs"


# ------------------------------------------------------- numerically invisible
@pytest.mark.parametrize("md", MDS)
@pytest.mark.parametrize("layout", LAYOUT_NAMES)
def test_ema_stacked_with_views_matches_uncached(md, layout):
    """4 steps of the cached and uncached ``ema_stacked`` must agree bit-for-bit."""
    shape, eff, view = LAYOUTS[layout]
    codec_a, states_a = make_states(md, shape)
    codec_b, states_b = make_states(md, shape)
    views = codec_b.stacked_views(states_b, view, eff)
    assert views is not None, "contiguous state must be cacheable"
    g = torch.Generator().manual_seed(3)
    for _ in range(4):
        upd = torch.randn((3, *eff), generator=g)
        da = codec_a.ema_stacked(states_a, upd.clone(), view, eff, 0.9)
        db = codec_b.ema_stacked(states_b, upd.clone(), view, eff, 0.9, views=views)
        assert torch.equal(da, db), "delta differs"
        assert_states_equal(state_snapshot(states_a), state_snapshot(states_b))


@pytest.mark.parametrize("md", MDS)
@pytest.mark.parametrize("layout", LAYOUT_NAMES)
def test_store_stacked_with_views_matches_uncached(md, layout):
    shape, eff, view = LAYOUTS[layout]
    codec_a, states_a = make_states(md, shape)
    codec_b, states_b = make_states(md, shape)
    views = codec_b.stacked_views(states_b, view, eff)
    g = torch.Generator().manual_seed(4)
    for _ in range(4):
        m = torch.randn((3, *eff), generator=g)
        codec_a.store_stacked(states_a, m.clone())
        codec_b.store_stacked(states_b, m.clone(), views=views)
        assert_states_equal(state_snapshot(states_a), state_snapshot(states_b))


@pytest.mark.parametrize("md", MDS)
@pytest.mark.parametrize("layout", LAYOUT_NAMES)
def test_dequant_stacked_with_views_matches_uncached(md, layout):
    shape, eff, view = LAYOUTS[layout]
    codec_a, states_a = make_states(md, shape)
    codec_b, states_b = make_states(md, shape)
    views = codec_b.stacked_views(states_b, view, eff)
    g = torch.Generator().manual_seed(5)
    for _ in range(4):
        m = torch.randn((3, *eff), generator=g)
        codec_a.store_stacked(states_a, m.clone())
        codec_b.store_stacked(states_b, m.clone(), views=views)
        da = codec_a.dequant_stacked(states_a, view, eff)
        db = codec_b.dequant_stacked(states_b, view, eff, views=views)
        assert torch.equal(da, db), "dequant differs"


# ------------------------------------------------------ aliasing, not copying
@pytest.mark.parametrize("md", MDS)
@pytest.mark.parametrize("layout", LAYOUT_NAMES)
def test_cached_views_alias_the_state_buffers(md, layout):
    """Every cached entry shares storage with the live state — pins no memory, and the
    write-backs reach the real buffer (a reshape-COPY would silently detach them)."""
    shape, eff, view = LAYOUTS[layout]
    codec, states = make_states(md, shape)
    views = codec.stacked_views(states, view, eff)
    for entry, state in zip(views.m, states, strict=True):
        assert entry.data_ptr() == state["m"].data_ptr()
    for entry, state in zip(views.store, states, strict=True):
        assert entry.data_ptr() == state["m"].data_ptr()
    if md in ("int8", "4bit"):
        for entry, state in zip(views.scale, states, strict=True):
            assert entry.data_ptr() == state["m_scale"].data_ptr()
    else:
        assert views.scale is None


@pytest.mark.parametrize("md", MDS)
def test_cached_views_see_an_in_place_state_write(md):
    """A requant that writes IN PLACE (4-bit's included) leaves the cache valid."""
    codec, states = make_states(md, (4, 3))
    views = codec.stacked_views(states, lambda t: t, (4, 3))
    m = torch.full((3, 4, 3), 0.5)
    codec.store_stacked(states, m, views=views)
    before = [v.clone() for v in views.m]
    codec.store_stacked(states, torch.full((3, 4, 3), -0.25), views=views)
    assert any(not torch.equal(a, b) for a, b in zip(before, views.m, strict=True))
    for entry, state in zip(views.m, states, strict=True):
        assert entry.data_ptr() == state["m"].data_ptr()


# -------------------------------------------------------------- never stale
def strided_like(t):
    """A non-contiguous tensor holding ``t``'s values (every other element of a
    double-width buffer). Works for the packed 1-D 4-bit buffer too."""
    wide = t.new_zeros(tuple(t.shape[:-1]) + (t.shape[-1] * 2,))
    out = wide[..., ::2]
    out.copy_(t)
    assert not out.is_contiguous()
    return out


@pytest.mark.parametrize("md", MDS)
def test_non_contiguous_momentum_declines_the_cache(md):
    """A layout the cache cannot alias must return ``None`` (the caller then takes the
    codec's uncached / per-param fallback), never a detached reshape-copy."""
    codec, states = make_states(md, (4, 3))
    for s in states:
        s["m"] = strided_like(s["m"])
    assert codec.stacked_views(states, lambda t: t, (4, 3)) is None


@pytest.mark.parametrize("md", MDS)
def test_non_contiguous_scale_declines_the_cache(md):
    """The scale views must alias too — a ``reshape`` copy would freeze the scale."""
    codec, states = make_states(md, (4, 3))
    if "m_scale" not in states[0]:
        pytest.skip("float codec stores no scale")
    for s in states:
        s["m_scale"] = strided_like(s["m_scale"])
    assert codec.stacked_views(states, lambda t: t, (4, 3)) is None


def exercise_all_three(codec, states, ref_codec, ref_states, eff, stale, *, rounds=3):
    """Run every stacked entry point with ``stale`` views and demand the plain,
    view-less reference — same deltas, same reads, same state.

    The *read* is checked first on purpose: it is the assertion that discriminates by
    **value** rather than by exception, so the guard's removal is caught even where the
    write-back happens to be shape-compatible.
    """
    g = torch.Generator().manual_seed(6)
    for _ in range(rounds):
        assert torch.equal(
            codec.dequant_stacked(states, lambda t: t, eff, views=stale),
            ref_codec.dequant_stacked(ref_states, lambda t: t, eff),
        ), "dequant_stacked used the stale views"
        m = torch.randn((3, *eff), generator=g)
        codec.store_stacked(states, m.clone(), views=stale)
        ref_codec.store_stacked(ref_states, m.clone())
        assert_states_equal(state_snapshot(states), state_snapshot(ref_states))
        upd = torch.randn((3, *eff), generator=g)
        da = codec.ema_stacked(states, upd.clone(), lambda t: t, eff, 0.9, views=stale)
        db = ref_codec.ema_stacked(ref_states, upd.clone(), lambda t: t, eff, 0.9)
        assert torch.equal(da, db), "ema_stacked used the stale views"
        assert_states_equal(state_snapshot(states), state_snapshot(ref_states))


@pytest.mark.parametrize("md", MDS)
def test_views_from_another_bucket_are_ignored(md):
    """The hand-off the ``eff`` guard exists for: views built by a **different** chunk.

    ``other_states`` holds the same element count in a different effective layout
    (``(2, 6)`` vs ``(4, 3)``), so its cached lists are shape-compatible enough to be
    used by accident. Without the guard the layout-bearing codecs read back the wrong
    shape and every codec *writes into the other bucket's buffers*, leaving these
    states untouched — which is what the state assertions catch for 4-bit, whose view
    lists are the raw packed buffers and carry no layout of their own.
    """
    codec, states = make_states(md, (4, 3))
    _other_codec, other_states = make_states(md, (2, 6), seed=1)
    stale = codec.stacked_views(other_states, lambda t: t.view(2, 6), (2, 6))
    assert stale is not None and stale.eff == (2, 6)
    ref_codec, ref_states = make_states(md, (4, 3))
    exercise_all_three(codec, states, ref_codec, ref_states, (4, 3), stale)


# 4-bit is absent on purpose: its cached lists are the raw packed ``m`` / ``m_scale``
# with no effective layout, and ``per`` comes from the ``eff`` ARGUMENT, so views built
# for another ``eff`` of the SAME buffers are indistinguishable from the right ones.
# The cross-bucket case above is what covers 4-bit's guard.
@pytest.mark.parametrize("md", ["float32", "bfloat16", "int8"])
def test_views_built_for_another_eff_of_the_same_states_are_ignored(md):
    """A views object over the same buffers in a different effective layout.

    Nothing can be written to the wrong *place* here, so the kill is purely by value:
    the cached lists are ``[1, 1]`` views where the call works in ``[1]``, and without
    the guard the read comes back ``[N, 1, 1]`` instead of ``[N, 1]``.
    """
    codec, states = make_states(md, (1,))
    stale = codec.stacked_views(states, lambda t: t.view(1, 1), (1, 1))
    assert stale is not None and stale.eff == (1, 1)
    ref_codec, ref_states = make_states(md, (1,))
    exercise_all_three(codec, states, ref_codec, ref_states, (1,), stale)


@pytest.mark.parametrize("md", MDS)
def test_hand_crafted_eff_mismatch_is_ignored(md):
    """The guard is a plain tuple compare, so a bogus ``eff`` alone must disable the
    cache even when the lists themselves are the right ones (defense in depth)."""
    codec, states = make_states(md, (4, 3))
    real = codec.stacked_views(states, lambda t: t, (4, 3))
    stale = _StackedViews((9, 9), real.m, real.scale, real.store)
    ref_codec, ref_states = make_states(md, (4, 3))
    exercise_all_three(codec, states, ref_codec, ref_states, (4, 3), stale)


# -------------------------------------------------------- ForeachChunk cache
OPTIMIZERS = {"AdaBelief": AdaBelief, "AdamP": AdamP, "ADOPT": ADOPT,
              "AdaMuon": AdaMuon, "Adakaon": Adakaon}
NAMES = list(OPTIMIZERS)


def build(name, bag, **over):
    kwargs = {"lr": 1e-2, "weight_decay": 0.01, "momentum_dtype": "float32"}
    kwargs.update(over)
    return OPTIMIZERS[name](bag, **kwargs)


def set_grads(bag, step):
    g = torch.Generator().manual_seed(1000 + step)
    for p in bag:
        p.grad = torch.randn(p.shape, generator=g).mul_(0.1)


@pytest.mark.parametrize("name", NAMES)
def test_chunk_momentum_views_are_built_once_and_reused(name):
    torch.manual_seed(11)
    bag = [torch.nn.Parameter(torch.randn(4, 3)) for _ in range(3)]
    opt = build(name, bag)
    set_grads(bag, 1)
    opt.step()
    (chunk,) = next(iter(opt._foreach_plans.values())).chunks
    assert isinstance(chunk, ForeachChunk)
    codec = _make_codec("float32")
    first = chunk.momentum_views(codec)
    assert first is not None
    assert chunk.momentum_views(codec) is first
    other = _make_codec("float32")  # same class, different instance
    assert chunk.momentum_views(other) is not first
    assert chunk.momentum_views(codec) is not first  # rebuilt for the original codec


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("md", MDS)
def test_step_after_load_state_dict_moves_the_live_momentum(name, md):
    """``load_state_dict`` REPLACES ``state["m"]``; the next foreach step must write the
    new buffer, not a cached view of the freed one."""
    torch.manual_seed(11)
    bag = [torch.nn.Parameter(torch.randn(4, 3)) for _ in range(3)]
    bag += [torch.nn.Parameter(torch.randn(())) for _ in range(2)]
    opt = build(name, bag, momentum_dtype=md)
    set_grads(bag, 1)
    opt.step()
    buf = io.BytesIO()
    torch.save(opt.state_dict(), buf)
    buf.seek(0)
    opt.load_state_dict(torch.load(buf, weights_only=False))
    live = [opt.state[p]["m"] for p in bag]
    before = [m.clone() for m in live]
    set_grads(bag, 2)
    opt.step()
    assert any(not torch.equal(a, b.clone()) for a, b in zip(before, live, strict=True)), (
        "the step did not reach the reloaded momentum buffers"
    )


# ------------------------------------------------- Adakaon: the EMA entry point
# Adakaon is the only shared-plan optimizer that reaches the codec through
# ``ema_stacked`` (the others ``dequant``/``store``), and it was the last one still
# rebuilding those lists every step. The three assertions below are what "it is wired
# to the cache" means, as opposed to "the cache exists": the views are BUILT once per
# chunk (not per step), they are HANDED to every call, and the codec therefore never
# falls back to the per-param ``mat`` sweep.


def spy_codec(opt, md, group_defaults=None):
    """Install a counting codec on ``opt`` for ``md`` and return its call log."""
    codec = _make_codec(md)
    log = {"views": 0, "handed": [], "mat": 0}
    real_views, real_ema = codec.stacked_views, codec.ema_stacked

    def stacked_views(states, view, eff):
        log["views"] += 1
        return real_views(states, view, eff)

    def ema_stacked(states, update, mat, eff, beta1, views=None):
        log["handed"].append(views)

        def counting_mat(t):
            log["mat"] += 1
            return mat(t)

        return real_ema(states, update, counting_mat, eff, beta1, views=views)

    codec.stacked_views = stacked_views
    codec.ema_stacked = ema_stacked
    opt._codecs[md] = codec
    return log


def adakaon_bag():
    """A bag that produces exactly two Adakaon buckets: factored (4,3) and flat 0-D."""
    torch.manual_seed(19)
    return ([torch.nn.Parameter(torch.randn(4, 3)) for _ in range(3)]
            + [torch.nn.Parameter(torch.randn(())) for _ in range(2)])


@pytest.mark.parametrize("md", MDS)
def test_adakaon_hands_the_codec_its_cached_views(md):
    """Both Adakaon bucket bodies must pass ``views=`` on every step, from one build."""
    bag = adakaon_bag()
    opt = Adakaon(bag, lr=1e-2, weight_decay=0.01, momentum_dtype=md,
                  momentum_4bit_block=8)
    log = spy_codec(opt, md)
    for step in range(4):
        set_grads(bag, step)
        opt.step()

    assert len(log["handed"]) == 8, "expected 2 buckets x 4 steps of ema_stacked calls"
    assert all(v is not None for v in log["handed"]), (
        f"{md}: the codec was called with views=None ({log['handed'].count(None)}/8)"
    )
    assert log["views"] == 2, (
        f"{md}: stacked_views ran {log['views']} times for 2 chunks over 4 steps"
    )
    assert log["mat"] == 0, (
        f"{md}: the codec still rebuilt {log['mat']} per-param views through mat()"
    )
    # One object per chunk, reused: two distinct views objects across the eight calls.
    assert len({id(v) for v in log["handed"]}) == 2


@pytest.mark.parametrize("md", MDS)
def test_adakaon_load_state_dict_drops_the_cached_views(md):
    """``load_state_dict`` replaces ``state["m"]``; the chunk's views must be rebuilt."""
    bag = adakaon_bag()
    opt = Adakaon(bag, lr=1e-2, momentum_dtype=md, momentum_4bit_block=8)
    log = spy_codec(opt, md)
    for step in range(2):
        set_grads(bag, step)
        opt.step()
    assert log["views"] == 2

    buf = io.BytesIO()
    torch.save(opt.state_dict(), buf)
    buf.seek(0)
    opt.load_state_dict(torch.load(buf, weights_only=False))
    log = spy_codec(opt, md)          # a fresh log; the codec is re-looked-up either way
    set_grads(bag, 2)
    opt.step()

    assert log["views"] == 2, "the plan (and its views) survived a load_state_dict"
    assert all(v is not None for v in log["handed"])
    # The live (reloaded) buffers are what the step must reach.
    live = [opt.state[p]["m"] for p in bag]
    snapshot = [m.clone() for m in live]
    set_grads(bag, 3)
    opt.step()
    assert any(not torch.equal(a, b) for a, b in zip(snapshot, live, strict=True)), (
        "the step did not reach the reloaded momentum buffers"
    )
