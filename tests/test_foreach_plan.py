"""Tests for the shared foreach bucketing/view plan (``kaon._foreach_plan``).

The plan caches, per param group, the bucketing AND every derived view a bucket body
walks (param views, factored ``row``/``col``, the non-factored state view, and — via
``ForeachChunk.momentum_views``, covered in ``test_codec_stacked_views.py`` — the
momentum codec's stacked view lists). Two properties have to hold for that to be safe:

* **Numerically invisible.** The cache is a pure host-side optimization: a run with
  ``_foreach_cache_enabled = False`` must land on bit-identical weights and state.
* **Never stale.** A cached view of a tensor the optimizer no longer owns would step
  detached memory. Six things can invalidate a plan and every one is covered here:
  a changed param set, a ``p.data`` rebind (fresh storage *or* a transpose that only
  moves strides), ``load_state_dict``, an AutoLR base-state reset, ``add_param_group``,
  and a re-chunk when the stack budget moves.

Plus the one thing the plan must NOT cache: gradient views. A retained view of
``p.grad`` keeps the previous step's gradient storage alive, which would add a whole
gradient set to peak memory.
"""

from __future__ import annotations

import gc
import io
import weakref

import pytest
import torch

from kaon import (
    ADOPT,
    AdaBelief,
    Adakaon,
    AdamP,
    AdaMuon,
    ScheduleFree,
    reseed_stochastic_rounding,
)
from kaon._foreach_plan import ForeachChunk, ForeachPlanMixin, param_witness

OPTIMIZERS = {
    "Adakaon": (Adakaon, {}),
    "AdaMuon": (AdaMuon, {"bias_correction": True}),
    "AdaBelief": (AdaBelief, {}),
    "AdamP": (AdamP, {}),
    "ADOPT": (ADOPT, {}),
    "ScheduleFree": (ScheduleFree, {}),
}
NAMES = list(OPTIMIZERS)
AUTOLR = ["Adakaon", "AdaMuon", "AdaBelief", "AdamP", "ADOPT"]  # ScheduleFree: no AutoLR mixin
NO_EXTRA_KEY = ["Adakaon", "ScheduleFree"]        # coefficients are per group, not per param
EXTRA_KEY = [n for n in NAMES if n not in NO_EXTRA_KEY]


def make_bag(dtype=torch.float32, *, seed=0):
    """0-D / (1,) / 1-D / 2-D / conv — one of every bucket shape, at least two each."""
    g = torch.Generator().manual_seed(seed)
    shapes = [(), (), (1,), (5,), (5,), (4, 3), (4, 3), (3, 5), (2, 2, 3, 3), (2, 2, 3, 3)]
    return [torch.nn.Parameter(torch.randn(s, generator=g).to(dtype)) for s in shapes]


def set_grads(bag, step, *, skip=()):
    g = torch.Generator().manual_seed(1000 + step)
    for i, p in enumerate(bag):
        if i in skip:
            p.grad = None
            continue
        raw = torch.randn(p.shape, generator=g)
        p.grad = raw.to(device=p.device, dtype=p.dtype).mul_(0.1)


def build(name, bag, **over):
    cls, extra = OPTIMIZERS[name]
    kwargs = {"lr": 1e-2, "weight_decay": 0.01, "momentum_dtype": "float32", **extra}
    kwargs.update(over)
    return cls(bag, **kwargs)


def run(name, *, cache=True, steps=4, dtype=torch.float32, skip_map=None, **over):
    torch.manual_seed(11)
    # SR draws come from a module-owned generator that only re-syncs when the global
    # seed CHANGES; re-seeding with the same value inside one process would otherwise
    # let the second arm continue the first arm's stream.
    reseed_stochastic_rounding()
    bag = make_bag(dtype)
    opt = build(name, bag, **over)
    opt._foreach_cache_enabled = cache
    for step in range(1, steps + 1):
        set_grads(bag, step, skip=(skip_map or {}).get(step, ()))
        opt.step()
    return bag, opt


def snapshot(bag, opt):
    out = []
    for p in bag:
        out.append(p.detach().float().clone())
        for _, v in sorted(opt.state[p].items()):
            out.append(v.detach().float().clone() if isinstance(v, torch.Tensor)
                       else torch.tensor(float(v)))
    return out


def assert_same(a, b):
    assert len(a) == len(b)
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        assert torch.equal(x, y), f"tensor {i} differs (max {(x - y).abs().max()})"


def only_plan(opt):
    plans = opt._foreach_plans
    assert len(plans) == 1, f"expected one cached plan, got {len(plans)}"
    return next(iter(plans.values()))


# --------------------------------------------------------------- numerically invisible
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cache_toggle_is_bit_identical(name, dtype):
    """``_foreach_cache_enabled`` is a pure host-side switch: same bits either way."""
    on = snapshot(*run(name, cache=True, dtype=dtype))
    off = snapshot(*run(name, cache=False, dtype=dtype))
    assert_same(on, off)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("md", ["float32", "bfloat16", "int8", "4bit"])
def test_cache_toggle_bit_identical_every_momentum_dtype(name, md):
    """Including the quantized codecs, whose ``mat`` callback the plan prebuilds."""
    on = snapshot(*run(name, cache=True, momentum_dtype=md))
    off = snapshot(*run(name, cache=False, momentum_dtype=md))
    assert_same(on, off)


# ------------------------------------------------------------------------- reuse
@pytest.mark.parametrize("name", NAMES)
def test_plan_and_chunks_are_reused_across_steps(name):
    """A steady param set reuses the plan object AND its chunk list — the whole point.

    The extra bucket key (per-param step / ``t``) changes value every step; only its
    *partition* has to hold, so the plan must survive it.
    """
    torch.manual_seed(11)
    bag = make_bag()
    opt = build(name, bag)
    set_grads(bag, 1)
    opt.step()
    plan = only_plan(opt)
    chunks = plan.chunks
    for step in (2, 3, 4):
        set_grads(bag, step)
        opt.step()
        assert only_plan(opt) is plan
        assert plan.chunks is chunks


@pytest.mark.parametrize("name", NAMES)
def test_disabled_cache_stores_nothing(name):
    _bag, opt = run(name, cache=False)
    assert opt._foreach_plans == {}


# ------------------------------------------------------------------- bucket keys
@pytest.mark.parametrize("name", NAMES)
def test_scalar_and_shape_one_share_a_bucket(name):
    """0-D params ride the ``L == 1`` bucket as length-1 views of the same storage."""
    torch.manual_seed(11)
    bag = [torch.nn.Parameter(torch.randn(())), torch.nn.Parameter(torch.randn(1))]
    opt = build(name, bag)
    set_grads(bag, 1)
    opt.step()
    chunks = only_plan(opt).chunks
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.eff is None and chunk.length == 1 and chunk.n == 2
    # the cached param view of the 0-D param must alias its storage, not copy it
    assert chunk.pviews[0].shape == (1,)
    assert chunk.pviews[0].data_ptr() == bag[0].data.data_ptr()


@pytest.mark.parametrize("name", NAMES)
def test_dtype_splits_buckets(name):
    torch.manual_seed(11)
    bag = [torch.nn.Parameter(torch.randn(4, 3)),
           torch.nn.Parameter(torch.randn(4, 3).bfloat16())]
    opt = build(name, bag, bf16_method="none")
    set_grads(bag, 1)
    opt.step()
    assert len(only_plan(opt).chunks) == 2


@pytest.mark.parametrize("name", NAMES)
def test_conv_bucket_is_matrixized(name):
    torch.manual_seed(11)
    bag = [torch.nn.Parameter(torch.randn(2, 2, 3, 3)) for _ in range(2)]
    opt = build(name, bag)
    set_grads(bag, 1)
    opt.step()
    chunk = only_plan(opt).chunks[0]
    assert chunk.matrixize and chunk.eff == (2, 18)
    assert chunk.pviews[0].shape == (2, 18)
    assert chunk.grad_stack().shape == (2, 2, 18)


# ------------------------------------------------------- invalidation 1: param set
@pytest.mark.parametrize("name", NAMES)
def test_param_set_change_rebuilds_plan(name):
    """A param that stops producing gradients leaves the fast list -> new witness."""
    torch.manual_seed(11)
    bag = make_bag()
    opt = build(name, bag)
    set_grads(bag, 1)
    opt.step()
    plan = only_plan(opt)
    set_grads(bag, 2, skip=(0, 5))
    opt.step()
    new = only_plan(opt)
    assert new is not plan
    assert len(new.buckets) != len(plan.buckets) or new.witness != plan.witness


# -------------------------------------------------- invalidation 2: p.data rebind
@pytest.mark.parametrize("name", NAMES)
def test_data_rebind_rebuilds_plan_and_steps_the_new_storage(name):
    """``p.data = <fresh storage>`` moves the pointer; the plan must follow it."""
    torch.manual_seed(11)
    bag = make_bag()
    opt = build(name, bag)
    set_grads(bag, 1)
    opt.step()
    plan = only_plan(opt)

    fresh = bag[5].data.clone().add_(1.0)
    bag[5].data = fresh
    before = fresh.clone()
    set_grads(bag, 2)
    opt.step()
    assert only_plan(opt) is not plan
    assert not torch.equal(bag[5].data, before)      # the NEW storage got the update
    assert bag[5].data.data_ptr() == fresh.data_ptr()


@pytest.mark.parametrize("name", NAMES)
def test_transpose_rebind_rebuilds_plan(name):
    """A square weight transposed in place keeps id, pointer AND shape — only the
    strides move, which is exactly what the witness's contiguity field is for."""
    torch.manual_seed(11)
    bag = [torch.nn.Parameter(torch.randn(4, 4)) for _ in range(2)]
    opt = build(name, bag)
    set_grads(bag, 1)
    opt.step()
    plan = only_plan(opt)
    bag[0].data = bag[0].data.t()
    set_grads(bag, 2)
    opt.step()
    assert only_plan(opt) is not plan


# ----------------------------------------------- invalidation 3: load_state_dict
@pytest.mark.parametrize("name", NAMES)
def test_load_state_dict_drops_the_plan(name):
    """The loader REPLACES the state tensors the views alias (and every group dict)."""
    torch.manual_seed(11)
    bag = make_bag()
    opt = build(name, bag)
    set_grads(bag, 1)
    opt.step()
    assert opt._foreach_plans
    buf = io.BytesIO()
    torch.save(opt.state_dict(), buf)
    buf.seek(0)
    opt.load_state_dict(torch.load(buf, weights_only=False))
    assert opt._foreach_plans == {}
    set_grads(bag, 2)
    opt.step()  # rebuilds cleanly against the restored buffers


@pytest.mark.parametrize("name", NAMES)
def test_resume_through_checkpoint_matches_uninterrupted_run(name):
    """The cached plan must not make a resumed run diverge from a straight one."""
    straight = snapshot(*run(name, steps=4))

    torch.manual_seed(11)
    bag = make_bag()
    opt = build(name, bag)
    for step in (1, 2):
        set_grads(bag, step)
        opt.step()
    buf = io.BytesIO()
    torch.save(opt.state_dict(), buf)
    buf.seek(0)
    opt2 = build(name, bag)
    opt2.load_state_dict(torch.load(buf, weights_only=False))
    for step in (3, 4):
        set_grads(bag, step)
        opt2.step()
    assert_same(straight, snapshot(bag, opt2))


# --------------------------------------------- invalidation 4: AutoLR state reset
@pytest.mark.parametrize("name", AUTOLR)
def test_autolr_reset_drops_the_plan(name):
    torch.manual_seed(11)
    bag = make_bag()
    opt = build(name, bag)
    set_grads(bag, 1)
    opt.step()
    assert opt._foreach_plans
    opt._autolr_reset_base_state()
    assert opt._foreach_plans == {}
    assert not opt.state
    set_grads(bag, 2)
    opt.step()  # state and plan are both reallocated


# ---------------------------------------------- invalidation 5: add_param_group
@pytest.mark.parametrize("name", NAMES)
def test_add_param_group_drops_the_plans(name):
    torch.manual_seed(11)
    bag = make_bag()
    opt = build(name, bag)
    set_grads(bag, 1)
    opt.step()
    assert opt._foreach_plans
    extra = [torch.nn.Parameter(torch.randn(4, 3)) for _ in range(2)]
    opt.add_param_group({"params": extra})
    assert opt._foreach_plans == {}
    for p in extra:
        p.grad = torch.randn(p.shape).mul_(0.1)
    set_grads(bag, 2)
    opt.step()
    assert len(opt._foreach_plans) == 2  # one plan per group


# ------------------------------------------------- invalidation 6: budget re-chunk
@pytest.mark.parametrize("name", NAMES)
def test_budget_change_rechunks_in_place(name):
    """A smaller stack budget splits the buckets further; the PLAN survives, its
    chunk list does not."""
    torch.manual_seed(11)
    bag = [torch.nn.Parameter(torch.randn(4, 3)) for _ in range(6)]
    opt = build(name, bag)
    set_grads(bag, 1)
    opt.step()
    plan = only_plan(opt)
    assert len(plan.chunks) == 1            # the whole bucket fits one chunk
    # 24 elements per chunk = two 4x3 params; the per-tensor cutoff (budget // 2 = 12)
    # still admits them, so this exercises the re-chunk and not the fallback.
    opt._foreach_stack_budget = 24
    set_grads(bag, 2)
    opt.step()
    assert only_plan(opt) is plan           # same bucketing...
    assert len(plan.chunks) == 3            # ...re-chunked
    assert all(c.n == 2 for c in plan.chunks)


# ----------------------------------------- the per-parameter fallback drops a plan
@pytest.mark.parametrize("name", NAMES)
def test_per_param_fallback_drops_the_plan(name):
    """A cached plan only ever describes a group the foreach path actually stepped."""
    torch.manual_seed(11)
    bag = [torch.nn.Parameter(torch.randn(4, 3)) for _ in range(3)]
    opt = build(name, bag)
    set_grads(bag, 1)
    opt.step()
    assert opt._foreach_plans
    opt._foreach_batch_cutoff = 1           # every param is now too big to stack
    set_grads(bag, 2)
    opt.step()
    assert opt._foreach_plans == {}


# ------------------------------------------------------- gradients are NOT cached
@pytest.mark.parametrize("name", NAMES)
def test_plan_retains_no_gradient(name):
    """``set_to_none=True`` must actually free every step's gradients: a cached grad
    view would pin a whole gradient set for the process's lifetime.

    The witnesses start at step **1**, the step the plan is built on. A chunk that
    cached ``[p.grad for p in plist]`` in its constructor would pin exactly that step's
    gradients and nothing later, so refs taken from step 2 onwards would never see it.
    Every step's refs are re-checked after every ``zero_grad`` for the same reason.
    """
    torch.manual_seed(11)
    bag = make_bag()
    opt = build(name, bag)
    refs: list[tuple[int, list]] = []
    for step in (1, 2, 3):
        set_grads(bag, step)
        refs.append((step, [weakref.ref(p.grad) for p in bag]))
        opt.step()
        assert opt._foreach_plans                   # a plan IS cached...
        opt.zero_grad(set_to_none=True)
        gc.collect()
        for seen, srefs in refs:
            alive = [i for i, r in enumerate(srefs) if r() is not None]
            assert not alive, (
                f"after step {step}, the plan still pins step {seen}'s "
                f"gradients for params {alive}"
            )


@pytest.mark.parametrize("name", NAMES)
def test_chunks_hold_no_grad_attribute(name):
    """Belt and braces: nothing on a chunk is a view of ``p.grad``."""
    torch.manual_seed(11)
    bag = make_bag()
    opt = build(name, bag)
    set_grads(bag, 1)
    opt.step()
    grad_ptrs = {p.grad.data_ptr() for p in bag}
    for chunk in only_plan(opt).chunks:
        cached = list(chunk.pviews) + [t for lst in chunk.state_views for t in lst]
        assert not (grad_ptrs & {t.data_ptr() for t in cached})


# ------------------------------------------------------- the extra key's partition
@pytest.mark.parametrize("name", EXTRA_KEY)
def test_extra_key_partition_change_rebuilds_the_plan(name):
    """Params that fall out of lockstep must land in separate buckets.

    Withholding a grad leaves those params one step behind for good; the plan's
    partition signature moves and the bucketing is rebuilt with more buckets.
    """
    torch.manual_seed(11)
    bag = [torch.nn.Parameter(torch.randn(4, 3)) for _ in range(4)]
    opt = build(name, bag)
    for step in (1, 2):
        set_grads(bag, step)
        opt.step()
    plan = only_plan(opt)
    assert len(plan.buckets) == 1               # one shape, one step -> one bucket
    set_grads(bag, 3, skip=(0, 1))
    opt.step()                                  # params 0,1 stay a step behind
    set_grads(bag, 4)
    opt.step()
    split = only_plan(opt)
    assert split is not plan
    assert len(split.buckets) == 2              # same shape, two per-param clocks
    assert len({c.key for c in split.chunks}) == 2
    # ...and once split, the two clocks advance in lockstep again: the partition holds,
    # so the plan (and its views) must be REUSED even though every key VALUE moves.
    for step in (5, 6):
        set_grads(bag, step)
        opt.step()
        assert only_plan(opt) is split
    assert len({c.key for c in split.chunks}) == 2


@pytest.mark.parametrize("name", NO_EXTRA_KEY)
def test_no_extra_key_optimizers_bucket_without_a_clock(name):
    """ScheduleFree and Adakaon compute their coefficients once per group per step
    (neither carries a bias correction), so their bucketing must not depend on any
    per-parameter clock."""
    assert OPTIMIZERS[name][0]._FOREACH_SPEC.extra_key is None


# ------------------------------------------------------------------ plumbing bits
def test_param_witness_fields_are_independent():
    a = torch.randn(4, 4)
    b = torch.randn(4, 4)
    base = param_witness([a, b])
    assert param_witness([a, b]) == base
    assert param_witness([b, a]) != base                 # ids
    assert param_witness([a, b.clone()]) != base         # data_ptr
    assert param_witness([a, b.t()]) != base             # contiguity only


def test_mixin_is_in_every_optimizer_mro():
    for cls, _ in OPTIMIZERS.values():
        assert issubclass(cls, ForeachPlanMixin), cls.__name__
        assert cls._FOREACH_SPEC.factored_state == ("row", "col")
        assert len(cls._FOREACH_SPEC.flat_state) == 1


@pytest.mark.parametrize("name", NAMES)
def test_chunk_state_views_alias_the_state_buffers(name):
    """The cached state views must write THROUGH to ``self.state``, 0-D included."""
    torch.manual_seed(11)
    bag = [torch.nn.Parameter(torch.randn(())) for _ in range(2)]
    opt = build(name, bag)
    set_grads(bag, 1)
    opt.step()
    chunk = only_plan(opt).chunks[0]
    assert isinstance(chunk, ForeachChunk)
    (views,) = chunk.state_views
    key = OPTIMIZERS[name][0]._FOREACH_SPEC.flat_state[0]
    for view, state in zip(views, chunk.states, strict=True):
        assert view.shape == (1,)
        assert view.data_ptr() == state[key].data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("name", NAMES)
def test_device_is_part_of_the_bucket_key(name):
    """``torch.stack`` refuses to mix devices, so a group holding a CPU and a CUDA
    weight of the same shape must step them in separate buckets, not crash."""
    torch.manual_seed(11)
    bag = [torch.nn.Parameter(torch.randn(4, 3)),
           torch.nn.Parameter(torch.randn(4, 3, device="cuda")),
           torch.nn.Parameter(torch.randn(4, 3)),
           torch.nn.Parameter(torch.randn(4, 3, device="cuda"))]
    opt = build(name, bag)
    set_grads(bag, 1)
    opt.step()
    chunks = only_plan(opt).chunks
    assert len(chunks) == 2
    assert {c.plist[0].device.type for c in chunks} == {"cpu", "cuda"}

# ================================================================== bucket ORDER
# Bucket order is numerically inert on its own — buckets touch disjoint params and
# disjoint state — but it decides the order the stochastic-rounding draws are consumed
# in, so it is part of the wire-visible contract for bf16+SR weights. ADOPT stepped one
# per-parameter clock at a time before the refactor (``ForeachSpec(key_major=True)``
# reproduces that); the others stepped all factored buckets first, then all flat.
# Nothing else in the suite pins that down, so these two tests do.
def _fragmented(name, *, steps=4):
    """A bag whose per-parameter clocks split: params 0 and 2 skip one gradient."""
    torch.manual_seed(11)
    bag = [torch.nn.Parameter(torch.randn(4, 3)), torch.nn.Parameter(torch.randn(4, 3)),
           torch.nn.Parameter(torch.randn(())), torch.nn.Parameter(torch.randn(()))]
    opt = build(name, bag)
    for step in range(1, steps + 1):
        set_grads(bag, step, skip=(0, 2) if step == 2 else ())
        opt.step()
    return bag, opt


@pytest.mark.parametrize("name", EXTRA_KEY)
def test_bucket_order_is_anchored(name):
    """The exact ``(key, factored?)`` sequence of the chunk list, per optimizer."""
    _bag, opt = _fragmented(name)
    chunks = only_plan(opt).chunks
    assert len(chunks) == 4                 # (4,3) x 2 clocks, L==1 x 2 clocks
    kinds = [c.eff is not None for c in chunks]
    keys = [c.key for c in chunks]
    if name == "ADOPT":
        # key_major: one per-parameter clock at a time, factored before flat inside it.
        assert kinds == [True, False, True, False]
        assert keys[0] == keys[1] and keys[2] == keys[3] and keys[0] != keys[2]
        assert keys[0] < keys[2]            # the first-seen clock is the lagging one
    else:
        # all factored buckets first, then all flat, each in first-seen key order.
        assert kinds == [True, True, False, False]
        assert keys[0] != keys[1] and keys[0] == keys[2] and keys[1] == keys[3]
        assert keys[0] < keys[1]


@pytest.mark.parametrize("name", NO_EXTRA_KEY)
def test_bucket_order_is_anchored_without_an_extra_key(name):
    """No per-parameter clock at all: one bucket per shape, factored before flat."""
    _bag, opt = _fragmented(name)
    chunks = only_plan(opt).chunks
    assert [(c.key, c.eff is not None) for c in chunks] == [(None, True), (None, False)]


# ==================================================== frozen bf16 + SR trajectories
# The order test above pins the chunk list; this one pins its observable consequence,
# against numbers captured from the pre-refactor tree. bf16 weights with stochastic
# rounding are the one place bucket order reaches the weights: every chunk draws its
# rounding noise from the module-owned SR generator, so reordering the chunks (or
# re-chunking them) changes the draw sizes and moves the last bit of every bf16 weight.
#
# TO REGENERATE — only when a change to the *math* is intended, never to turn a red
# test green. Export the reference tree and re-run this scenario against it:
#
#     git archive <ref> | tar -x -C /tmp/ref
#     cd /tmp/ref && PYTHONPATH=src python -c "$(cat <<'EOF'
#     import torch, kaon
#     ... paste _frozen_sr_run here, minus the `_foreach_plans` assert ...
#     for n, c, e in (("ADOPT", kaon.ADOPT, {}), ("AdaBelief", kaon.AdaBelief, {}),
#                     ("AdaMuon", kaon.AdaMuon, {"bias_correction": True})):
#         print(n, _frozen_sr_run(c, **e))
#     EOF
#     )"
#
# The values are bf16 bit patterns (uint16), so the comparison is exact and free of
# float-repr ambiguity. Captured on CPU: the SR draws come from a CPU MT19937 seeded
# from the global initial seed, reproducible across runs for a given torch version.
_FROZEN_SR_SHAPES = [(4, 3), (4, 3), (), ()]
_FROZEN_SR_LAGGING = (0, 2)          # no gradient on step 2 -> one step behind for good
_FROZEN_SR_BITS = {
    # captured from ee8f871 (pre-refactor), torch 2.12.0+cu130, CPU
    "ADOPT": [49193, 15725, 16296, 49037, 15984, 16006, 48992, 16243, 49040, 16305,
              48801, 48908, 15914, 48793, 49037, 16315, 48873, 16205, 48717, 49024,
              16064, 16151, 49208, 48644, 48829, 49108],
    "AdaBelief": [49195, 15711, 16298, 49042, 15971, 16023, 48996, 16248, 49042, 16307,
                  48822, 48903, 15945, 48811, 49035, 16307, 48836, 16197, 48735, 49021,
                  16066, 16161, 49210, 48563, 48819, 49104],
    "AdaMuon": [49194, 15733, 16297, 49039, 15981, 16009, 48993, 16245, 49041, 16305,
                48802, 48907, 15910, 48798, 49036, 16315, 48871, 16204, 48698, 49024,
                16064, 16150, 49209, 48648, 48830, 49108],
    # captured from 54bd887 (kaon 0.7.12), i.e. from Adakaon's own pre-migration
    # ``_ForeachPlan``: this vector is the wire-visible half of the shared-plan migration.
    "Adakaon": [49194, 15723, 16296, 49039, 15981, 16007, 48992, 16244, 49040, 16304,
                48803, 48908, 15915, 48796, 49036, 16315, 48870, 16204, 48710, 49024,
                16060, 16154, 49209, 48638, 48829, 49108],
}


def _frozen_sr_run(cls, *, cache=True, **kwargs):
    """Five steps of bf16 + stochastic rounding over a bag with split clocks."""
    torch.manual_seed(0x51D)
    # The SR generator only re-syncs when the GLOBAL seed changes, so re-seeding with
    # the same value inside one process would continue the previous run's stream.
    reseed_stochastic_rounding()
    g = torch.Generator().manual_seed(4242)
    params = [torch.nn.Parameter(torch.randn(s, generator=g).bfloat16())
              for s in _FROZEN_SR_SHAPES]
    opt = cls(params, lr=1e-2, weight_decay=0.01, momentum_dtype="bfloat16",
              bf16_method="stochastic_rounding", **kwargs)
    opt._foreach_cache_enabled = cache
    for step in range(1, 6):
        gg = torch.Generator().manual_seed(500 + step)
        for i, p in enumerate(params):
            if step == 2 and i in _FROZEN_SR_LAGGING:
                p.grad = None
                continue
            p.grad = (torch.randn(p.shape, generator=gg) * 0.05).bfloat16()
        opt.step()
    if cache:
        assert opt._foreach_plans, "the frozen vector must come from the cached path"
    return [int(b) for p in params
            for b in p.detach().view(torch.uint16).reshape(-1).tolist()]


@pytest.mark.parametrize("name", ["ADOPT", "AdaBelief", "AdaMuon", "Adakaon"])
@pytest.mark.parametrize("cache", [True, False])
def test_frozen_bf16_sr_vector_matches_the_pre_refactor_tree(name, cache):
    """bf16 weights, bit for bit, against values captured from ee8f871 — with the
    host-side cache on and off, since neither may move a single draw."""
    cls, extra = OPTIMIZERS[name]
    assert _frozen_sr_run(cls, cache=cache, **extra) == _FROZEN_SR_BITS[name]
