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
    KProdigy,
    Lion,
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
    "Lion": (Lion, {}),
    # KProdigy's bucketing depends on ``second_moment`` (its ``ndim >= 2`` state is a
    # factored row/col pair or a full per-coordinate ``v``), so BOTH of its specs get the
    # whole battery. ``d0`` is raised off the 1e-6 default so the update is not orders of
    # magnitude below a bf16 ULP and the frozen-SR scenarios actually move weights.
    "KProdigy": (KProdigy, {"lr": 1.0, "d0": 1e-2}),
    "KProdigyFactored": (KProdigy, {"lr": 1.0, "d0": 1e-2, "second_moment": "factored"}),
}
NAMES = list(OPTIMIZERS)
# ScheduleFree / KProdigy: no AutoLR mixin.
AUTOLR = ["Adakaon", "AdaMuon", "AdaBelief", "AdamP", "ADOPT", "Lion"]
# Coefficients are per group, not per param -> no per-parameter clock in the bucket key.
NO_EXTRA_KEY = ["Adakaon", "ScheduleFree", "Lion", "KProdigy", "KProdigyFactored"]
EXTRA_KEY = [n for n in NAMES if n not in NO_EXTRA_KEY]
# Lion and KProdigy bucketed by exact shape before they moved onto the shared plan, so
# they keep 0-D params out of the ``L == 1`` bucket (``ForeachSpec.scalar_bucket``) —
# that partition is what their frozen bf16+SR vectors are anchored to.
SCALAR_SPLIT = ["Lion", "KProdigy", "KProdigyFactored"]
SCALAR_MERGED = [n for n in NAMES if n not in SCALAR_SPLIT]
# ``KProdigy(second_moment="full")`` keeps an ``ndim > 2`` weight in its own layout
# (``ForeachSpec(matrixize=False)``): its ``v`` is per-coordinate, not factored.
NO_MATRIXIZE = ["KProdigy"]
MATRIXIZE = [n for n in NAMES if n not in NO_MATRIXIZE]
# Optimizers whose factored bucket key IS the effective shape, so two conv kernels that
# matrixize to the same ``[R, C]`` share a bucket. Lion is the exception
# (``ForeachSpec.raw_shape_key``) and ``KProdigy(second_moment="full")`` never matrixizes.
MERGE_EFF = [n for n in MATRIXIZE if n != "Lion"]
# Optimizers whose non-factored bucket walks exactly one cached state view. Lion's only
# state IS the momentum, which it reads through the codec's own cached view lists.
NO_FLAT_STATE = ["Lion"]
FLAT_STATE = [n for n in NAMES if n not in NO_FLAT_STATE]
# KProdigy does not support a param group spread over several devices AT ALL, and not
# because of the pass-2 bucketing this module owns: its pass 1 is a GLOBAL reduction that
# stacks the sliced gradients and folds one scalar pair, so it raises on the first
# cross-device stack long before a bucket is built. Out of scope here.
MIXED_DEVICE = [n for n in NAMES if not n.startswith("KProdigy")]


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
@pytest.mark.parametrize("name", SCALAR_MERGED)
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


@pytest.mark.parametrize("name", SCALAR_SPLIT)
def test_scalar_bucket_keeps_0d_apart_from_shape_one(name):
    """Lion / KProdigy bucketed by exact shape, so 0-D and ``(1,)`` stay separate.

    Two ``L == 1`` buckets instead of one is a (small) missed merge; it is the price of
    their bf16+SR draw order not moving, which the frozen vectors below pin down. The
    0-D param still rides its bucket as a length-1 *view* of its own storage.
    """
    torch.manual_seed(11)
    bag = [torch.nn.Parameter(torch.randn(())), torch.nn.Parameter(torch.randn(1))]
    opt = build(name, bag)
    set_grads(bag, 1)
    opt.step()
    chunks = only_plan(opt).chunks
    assert len(chunks) == 2
    assert [c.n for c in chunks] == [1, 1]
    assert all(c.eff is None and c.length == 1 for c in chunks)
    scalar = chunks[0]
    assert scalar.pviews[0].shape == (1,)
    assert scalar.pviews[0].data_ptr() == bag[0].data.data_ptr()


@pytest.mark.parametrize("name", NAMES)
def test_dtype_splits_buckets(name):
    torch.manual_seed(11)
    bag = [torch.nn.Parameter(torch.randn(4, 3)),
           torch.nn.Parameter(torch.randn(4, 3).bfloat16())]
    opt = build(name, bag, bf16_method="none")
    set_grads(bag, 1)
    opt.step()
    assert len(only_plan(opt).chunks) == 2


@pytest.mark.parametrize("name", MATRIXIZE)
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


@pytest.mark.parametrize("name", MERGE_EFF)
def test_matrixized_bucket_may_mix_raw_conv_shapes(name):
    """A matrixized bucket is keyed on ``eff``, so it can hold different raw shapes.

    ``(2,2,3,3)`` and ``(2,6,1,3)`` both matrixize to ``(2,18)`` and therefore share a
    bucket. Regression: ``grad_stack``'s fast path stacked the RAW gradients (one
    ``view`` of the stack instead of N per-param views), which ``torch.stack`` rejects
    when the sizes differ — every shared-plan optimizer raised
    ``stack expects each tensor to be equal size`` on such a bucket. The fast path now
    requires a common raw shape and falls back to per-param views otherwise.

    The reference arm is compared to a **tolerance**, not bit-exactly, and deliberately
    so: the batched and per-parameter paths are not bit-identical for every optimizer
    here (AdaBelief and ScheduleFree were not before this module either — a pre-existing
    property of their reductions, not something this bucket changes). What the assertion
    pins is that a mixed-raw-shape bucket computes the *right* update, which is the half
    the crash was hiding.
    """
    torch.manual_seed(11)
    shapes = [(2, 2, 3, 3), (2, 6, 1, 3)]
    bag = [torch.nn.Parameter(torch.randn(s)) for s in shapes]
    ref = [torch.nn.Parameter(p.detach().clone()) for p in bag]
    opt = build(name, bag)
    opt_ref = build(name, ref)
    opt_ref._foreach_batch_cutoff = 1          # force the per-parameter reference path
    for step in (1, 2, 3):
        g = torch.Generator().manual_seed(1000 + step)
        for p, r in zip(bag, ref, strict=True):
            raw = torch.randn(p.shape, generator=g).mul_(0.1)
            p.grad, r.grad = raw.clone(), raw.clone()
        opt.step()
        opt_ref.step()
    chunk = only_plan(opt).chunks[0]
    assert chunk.n == 2 and chunk.eff == (2, 18) and not chunk.grad_uniform
    for p, r in zip(bag, ref, strict=True):
        torch.testing.assert_close(p.detach(), r.detach(), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("name", NO_MATRIXIZE)
def test_conv_bucket_keeps_its_own_layout_without_matrixize(name):
    """``ForeachSpec(matrixize=False)``: ``eff`` is the raw shape, of any rank.

    ``KProdigy(second_moment="full")`` keeps a per-coordinate ``v`` shaped like the
    weight, so there is nothing to matrixize *for* — and not reshaping is also what lets
    it keep stacking a channels_last conv, which a ``view`` into ``[R, C]`` cannot.
    """
    torch.manual_seed(11)
    bag = [torch.nn.Parameter(torch.randn(2, 2, 3, 3)) for _ in range(2)]
    opt = build(name, bag)
    set_grads(bag, 1)
    opt.step()
    chunk = only_plan(opt).chunks[0]
    assert not chunk.matrixize and chunk.eff == (2, 2, 3, 3)
    assert chunk.pviews[0].shape == (2, 2, 3, 3)
    assert chunk.pviews[0].data_ptr() == bag[0].data.data_ptr()
    assert chunk.grad_stack().shape == (2, 2, 2, 3, 3)


@pytest.mark.parametrize("name", NO_MATRIXIZE)
def test_non_contiguous_conv_still_batches_without_matrixize(name):
    """A channels_last conv has no ``[R, C]`` view, so a matrixizing bucket would raise.

    ``matrixize=False`` steps it in place instead — which is what the pre-plan
    ``_full_bucket`` did, and why this configuration must not start falling back.
    """
    torch.manual_seed(11)
    bag = [torch.nn.Parameter(torch.randn(2, 2, 3, 3).to(memory_format=torch.channels_last))
           for _ in range(2)]
    for p in bag:
        assert not p.data.is_contiguous()
    opt = build(name, bag)
    set_grads(bag, 1)
    opt.step()
    chunk = only_plan(opt).chunks[0]
    assert chunk.n == 2 and chunk.eff == (2, 2, 3, 3)


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


@pytest.mark.parametrize("name", NAMES)
def test_group_level_fallback_drops_the_plan(name):
    """The *group*-level rejection has to drop the plan too, not just the per-param one.

    A param group is a mutable dict and a scheduler can flip a hyperparameter mid-run.
    Every optimizer here refuses the batched path for ``bf16_method="kahan"`` (it needs a
    per-parameter compensation buffer), so flipping it in place moves the whole group to
    the per-parameter loop — and a cached plan must only ever describe a group the
    foreach path actually stepped. fp32 params on purpose: kahan is a no-op for them
    (``subtract_one_`` only compensates low-precision weights), so the per-parameter
    step works without the ``shift`` buffer the group never allocated.
    """
    torch.manual_seed(11)
    bag = [torch.nn.Parameter(torch.randn(4, 3)) for _ in range(3)]
    opt = build(name, bag)
    set_grads(bag, 1)
    opt.step()
    assert opt._foreach_plans
    for group in opt.param_groups:
        group["bf16_method"] = "kahan"
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


# The state keys each optimizer's bucket bodies walk. Everything factored keeps an
# Adafactor row/col pair; the exceptions are the two optimizers whose ``ndim >= 2``
# second moment is not factored at all (KProdigy's ``second_moment="full"`` ``v``) or
# absent (Lion has no second moment, only the codec-owned momentum).
STATE_KEYS = {
    "Adakaon": (("row", "col"), 1),
    "AdaMuon": (("row", "col"), 1),
    "AdaBelief": (("row", "col"), 1),
    "AdamP": (("row", "col"), 1),
    "ADOPT": (("row", "col"), 1),
    "ScheduleFree": (("row", "col"), 1),
    "Lion": ((), 0),
    "KProdigy": (("v",), 1),
    "KProdigyFactored": (("row", "col"), 1),
}


@pytest.mark.parametrize("name", NAMES)
def test_spec_declares_the_state_keys_its_buckets_walk(name):
    factored, n_flat = STATE_KEYS[name]
    spec = build(name, [torch.nn.Parameter(torch.randn(4, 3))])._foreach_spec(
        {"second_moment": OPTIMIZERS[name][1].get("second_moment", "full")})
    assert spec.factored_state == factored
    assert len(spec.flat_state) == n_flat


@pytest.mark.parametrize("name", FLAT_STATE)
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
    key = opt._foreach_spec(opt.param_groups[0]).flat_state[0]
    for view, state in zip(views, chunk.states, strict=True):
        assert view.shape == (1,)
        assert view.data_ptr() == state[key].data_ptr()


@pytest.mark.parametrize("name", NO_FLAT_STATE)
def test_lion_caches_only_param_and_codec_views(name):
    """Lion declares no state keys: its single buffer is the codec-owned momentum, whose
    cached view lists live on ``chunk.momentum_views(codec)``, not ``state_views``."""
    torch.manual_seed(11)
    bag = [torch.nn.Parameter(torch.randn(())) for _ in range(2)]
    opt = build(name, bag, momentum_dtype="int8")
    set_grads(bag, 1)
    opt.step()
    chunk = only_plan(opt).chunks[0]
    assert chunk.state_views == ()
    views = chunk.momentum_views(opt._codec("int8"))
    assert views is not None and views.eff == (1,)
    for m, state in zip(views.m, chunk.states, strict=True):
        assert m.data_ptr() == state["m"].data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("name", MIXED_DEVICE)
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


# =================================== Lion / KProdigy: the bucketing they are pinned to
# Both carried their OWN bucketing before this module: one dict keyed by the parameter's
# exact shape, stepped in first-appearance order. That partition and that order are
# wire-visible through the stochastic-rounding draws (a bucket draws its rounding noise
# in one shot, sized by the bucket), so the migration had to keep them: hence
# ``ForeachSpec(scalar_bucket=..., raw_shape_key=..., insertion_order=..., matrixize=...)``.
# The bag below is built so every one of those flags matters — read the comments on
# ``_PINNED_BAG`` — and the two tests are the partition/order anchor and its
# observable consequence.
#
# ``(2, 2, 3, 3)`` and ``(2, 6, 1, 3)`` matrixize to the SAME ``(2, 18)``: the canonical
# plan merges them, Lion never did (``raw_shape_key``). ``()`` before ``(1,)`` and both
# before any matrix: the canonical plan would emit all matrices first and merge the two
# one-element buckets (``insertion_order``, ``scalar_bucket``).
_PINNED_BAG = [(), (4, 3), (1,), (2, 2, 3, 3), (5,), (2, 6, 1, 3), (4, 3)]
_PINNED_LAGGING = (0, 3)          # no gradient on step 2

# (name, factored?, n) per chunk, in order.
_PINNED_ORDER = {
    "Lion": [(None, False, 1), ((4, 3), True, 2), (None, False, 1),
             ((2, 18), True, 1), (None, False, 1), ((2, 18), True, 1)],
    # second_moment="full": nothing is factored, the conv keeps its own rank-4 layout,
    # and the order is still first-appearance (one dict, pre-plan).
    "KProdigy": [(None, False, 1), ((4, 3), True, 2), (None, False, 1),
                 ((2, 2, 3, 3), True, 1), (None, False, 1), ((2, 6, 1, 3), True, 1)],
    # second_moment="factored": row/col for ndim>=2 and a full ``v`` for ndim<=1, which
    # is the canonical family split, so these DO come out factored-first. It is also the
    # one variant that MERGES the two convs (its factored state lives in the ``[R, C]``
    # layout, so ``eff`` is the whole key) — as its pre-plan bucketing already did.
    "KProdigyFactored": [((4, 3), True, 2), ((2, 18), True, 2),
                         (None, False, 1), (None, False, 1), (None, False, 1)],
}


@pytest.mark.parametrize("name", SCALAR_SPLIT)
def test_pinned_bucket_partition_and_order(name):
    """The exact chunk list — eff, family and size — Lion / KProdigy stepped pre-plan."""
    torch.manual_seed(11)
    bag = [torch.nn.Parameter(torch.randn(s)) for s in _PINNED_BAG]
    opt = build(name, bag)
    set_grads(bag, 1)
    opt.step()
    chunks = only_plan(opt).chunks
    got = [(c.eff, c.eff is not None, c.n) for c in chunks]
    assert got == _PINNED_ORDER[name]


_PINNED_SR_BITS = {
    ("Lion", "bfloat16"): [
        49192, 15723, 16293, 49039, 15973, 16022, 48999, 16242, 49044, 16308, 48796,
        48915, 15919, 48798, 48816, 16299, 49025, 48914, 48764, 15970, 49063, 48971,
        16132, 48759, 16137, 48885, 16355, 48853, 49124, 15933, 48881, 15849, 16188,
        15861, 49056, 16365, 16339, 49074, 16027, 49068, 16248, 48866, 48942, 48432,
        49078, 16141, 48692, 48921, 48832, 16065, 15939, 15872, 49100, 16149, 49089,
        16045, 49019, 16020, 49095, 16086, 48955, 48941, 15941, 49123, 16217, 16288,
        16343, 48888, 48760, 16234, 15882, 48882, 16157, 48865, 16166, 49167, 49004,
        49092, 48951, 16120, 16012, 16271, 49013, 48298, 16292, 48970, 48893, 16135,
        16187, 16208, 49168, 16249, 49184, 16259, 48819, 48945, 16177, 49134, 16312,
        16187, 49086, 48878, 15951],
    ("Lion", "int8"): [
        49192, 15723, 16293, 49039, 15973, 16022, 48999, 16242, 49044, 16308, 48796,
        48915, 15919, 48798, 48816, 16299, 49025, 48914, 48764, 15970, 49063, 48971,
        16132, 48759, 16137, 48885, 16355, 48853, 49124, 15933, 48881, 15849, 16188,
        15861, 49056, 16365, 16339, 49074, 16027, 49068, 16248, 48866, 48942, 48432,
        49078, 16141, 48692, 48921, 48832, 16065, 15939, 15872, 49100, 16149, 49089,
        16045, 49019, 16020, 49095, 16086, 48955, 48941, 15941, 49123, 16217, 16288,
        16343, 48888, 48760, 16234, 15882, 48882, 16157, 48865, 16166, 49167, 49004,
        49092, 48951, 16120, 16012, 16271, 49013, 48298, 16292, 48970, 48893, 16135,
        16187, 16208, 49168, 16249, 49184, 16259, 48819, 48945, 16177, 49134, 16312,
        16187, 49086, 48878, 15951],
    ("Lion", "4bit"): [
        49192, 15723, 16293, 49039, 15973, 16022, 48999, 16242, 49044, 16308, 48796,
        48915, 15919, 48798, 48816, 16299, 49025, 48914, 48764, 15971, 49063, 48971,
        16132, 48759, 16137, 48885, 16355, 48853, 49124, 15932, 48882, 15849, 16188,
        15861, 49056, 16365, 16338, 49074, 16032, 49068, 16248, 48866, 48942, 48434,
        49078, 16141, 48692, 48921, 48832, 16066, 15939, 15872, 49100, 16149, 49089,
        16044, 49019, 16020, 49095, 16086, 48955, 48941, 15940, 49123, 16217, 16288,
        16343, 48888, 48760, 16237, 15883, 48881, 16157, 48866, 16167, 49167, 49004,
        49092, 48952, 16120, 16012, 16271, 49013, 48300, 16292, 48970, 48893, 16135,
        16187, 16208, 49168, 16249, 49184, 16259, 48819, 48945, 16177, 49134, 16312,
        16187, 49086, 48878, 15951],
    ("KProdigy", "bfloat16"): [
        49193, 15834, 16296, 49050, 15787, 16062, 49000, 16230, 49048, 16322, 48787,
        48903, 15844, 48831, 48881, 16297, 49005, 48911, 48809, 15833, 49073, 48938,
        16158, 48828, 16169, 48831, 16371, 48887, 49111, 15932, 48916, 15382, 16167,
        15929, 49042, 16361, 16352, 49075, 15961, 49054, 16257, 48802, 48977, 15754,
        49072, 16159, 48749, 48953, 48811, 16053, 16046, 48299, 49087, 16155, 49071,
        16012, 49030, 15930, 49084, 16116, 48932, 48973, 16053, 49105, 16214, 16297,
        16327, 48903, 48787, 16200, 15890, 48930, 16202, 48812, 16156, 49158, 49024,
        49074, 48979, 16136, 16034, 16247, 49038, 48617, 16313, 48989, 48839, 16086,
        16151, 16234, 49159, 16275, 49194, 16260, 48722, 48938, 16152, 49143, 16319,
        16193, 49099, 48793, 15946],
    ("KProdigy", "int8"): [
        49193, 15833, 16296, 49051, 15786, 16062, 49000, 16230, 49048, 16322, 48787,
        48903, 15844, 48831, 48881, 16297, 49005, 48911, 48807, 15835, 49073, 48938,
        16158, 48828, 16169, 48831, 16371, 48887, 49111, 15933, 48916, 15397, 16165,
        15929, 49042, 16361, 16351, 49075, 15962, 49054, 16257, 48802, 48977, 15753,
        49072, 16159, 48749, 48953, 48812, 16053, 16047, 48307, 49087, 16155, 49071,
        16013, 49030, 15928, 49084, 16116, 48932, 48973, 16052, 49105, 16214, 16297,
        16328, 48902, 48788, 16200, 15890, 48930, 16202, 48812, 16155, 49158, 49024,
        49074, 48980, 16136, 16034, 16246, 49038, 48617, 16313, 48988, 48839, 16091,
        16150, 16232, 49159, 16275, 49194, 16260, 48722, 48938, 16152, 49143, 16319,
        16193, 49099, 48793, 15946],
    ("KProdigy", "4bit"): [
        49193, 15791, 16296, 49050, 15765, 16049, 49005, 16227, 49047, 16322, 48753,
        48905, 15850, 48831, 48880, 16296, 48999, 48914, 48814, 15802, 49071, 48939,
        16159, 48832, 16171, 48831, 16368, 48881, 49107, 15932, 48919, 15708, 16172,
        15898, 49046, 16361, 16338, 49076, 15986, 49057, 16256, 48797, 48976, 48307,
        49068, 16155, 48756, 48951, 48794, 16037, 16043, 48303, 49088, 16155, 49071,
        16000, 49031, 15914, 49082, 16108, 48933, 48964, 16067, 49103, 16214, 16294,
        16326, 48906, 48787, 16196, 15809, 48938, 16195, 48814, 16155, 49157, 49013,
        49076, 48966, 16139, 16041, 16246, 49040, 48640, 16311, 48997, 48843, 16067,
        16143, 16210, 49160, 16278, 49192, 16263, 48720, 48935, 16148, 49141, 16323,
        16194, 49099, 48799, 15929],
    ("KProdigyFactored", "bfloat16"): [
        49192, 15890, 16296, 49047, 15543, 16066, 49000, 16237, 49048, 16322, 48781,
        48905, 15829, 48830, 48894, 16299, 49008, 48911, 48821, 15841, 49074, 48944,
        16159, 48824, 16169, 48821, 16375, 48886, 49120, 15930, 48917, 15749, 16176,
        15892, 49039, 16359, 16338, 49073, 15972, 49052, 16260, 48796, 48977, 15239,
        49076, 16160, 48772, 48954, 48798, 16031, 16046, 48303, 49088, 16156, 49072,
        16016, 49027, 15985, 49082, 16128, 48928, 48959, 16049, 49105, 16213, 16292,
        16323, 48895, 48775, 16198, 15858, 48943, 16210, 48810, 16162, 49156, 49022,
        49076, 48970, 16143, 16032, 16250, 49034, 48634, 16308, 48994, 48837, 16061,
        16144, 16219, 49160, 16276, 49197, 16252, 48750, 48935, 16149, 49142, 16322,
        16192, 49102, 48819, 15944],
    ("KProdigyFactored", "int8"): [
        49192, 15891, 16296, 49047, 15537, 16066, 48999, 16237, 49048, 16321, 48781,
        48905, 15830, 48830, 48894, 16298, 49008, 48911, 48821, 15841, 49074, 48944,
        16159, 48825, 16169, 48821, 16375, 48885, 49120, 15930, 48917, 15750, 16176,
        15893, 49039, 16359, 16337, 49073, 15973, 49052, 16260, 48797, 48978, 15241,
        49076, 16160, 48772, 48954, 48797, 16030, 16046, 48310, 49088, 16156, 49072,
        16016, 49027, 15985, 49082, 16129, 48928, 48959, 16049, 49105, 16213, 16292,
        16324, 48893, 48775, 16198, 15860, 48943, 16210, 48810, 16162, 49156, 49022,
        49076, 48971, 16143, 16034, 16250, 49034, 48634, 16308, 48994, 48837, 16063,
        16144, 16219, 49160, 16276, 49197, 16252, 48750, 48935, 16149, 49142, 16322,
        16192, 49102, 48819, 15944],
    ("KProdigyFactored", "4bit"): [
        49192, 15879, 16297, 49047, 48033, 16057, 49004, 16233, 49048, 16324, 48774,
        48907, 15839, 48830, 48894, 16296, 49002, 48912, 48827, 15813, 49073, 48943,
        16160, 48829, 16171, 48820, 16372, 48879, 49118, 15928, 48921, 15797, 16176,
        15875, 49042, 16359, 16335, 49076, 15993, 49054, 16259, 48793, 48977, 48302,
        49072, 16154, 48777, 48952, 48793, 16021, 16043, 48304, 49089, 16154, 49071,
        16004, 49030, 15977, 49076, 16120, 48929, 48954, 16062, 49103, 16213, 16291,
        16322, 48900, 48775, 16195, 15766, 48954, 16202, 48811, 16160, 49156, 49013,
        49076, 48965, 16148, 16040, 16254, 49035, 48648, 16306, 48997, 48841, 16058,
        16137, 16210, 49161, 16278, 49197, 16253, 48745, 48932, 16145, 49142, 16323,
        16194, 49102, 48823, 15929],
}


def _pinned_sr_run(name, md, *, cache=True):
    """Five steps of bf16 + stochastic rounding over ``_PINNED_BAG``."""
    torch.manual_seed(0x51D)
    reseed_stochastic_rounding()
    g = torch.Generator().manual_seed(4242)
    params = [torch.nn.Parameter(torch.randn(s, generator=g).bfloat16())
              for s in _PINNED_BAG]
    cls, extra = OPTIMIZERS[name]
    kwargs = {"lr": 1e-2, "weight_decay": 0.01, "momentum_dtype": md,
              "bf16_method": "stochastic_rounding", **extra}
    opt = cls(params, **kwargs)
    opt._foreach_cache_enabled = cache
    for step in range(1, 6):
        gg = torch.Generator().manual_seed(500 + step)
        for i, p in enumerate(params):
            if step == 2 and i in _PINNED_LAGGING:
                p.grad = None
                continue
            p.grad = (torch.randn(p.shape, generator=gg) * 0.05).bfloat16()
        opt.step()
    if cache:
        assert opt._foreach_plans, "the frozen vector must come from the cached path"
    return [int(b) for p in params
            for b in p.detach().view(torch.uint16).reshape(-1).tolist()]


@pytest.mark.parametrize("md", ["bfloat16", "int8", "4bit"])
@pytest.mark.parametrize("name", SCALAR_SPLIT)
@pytest.mark.parametrize("cache", [True, False])
def test_pinned_bf16_sr_vector_matches_the_pre_plan_tree(name, md, cache):
    """bf16 weights, bit for bit, against values captured from f88face (pre-migration).

    Sensitivity check that justifies this being the anchor: on this bag the foreach and
    per-parameter paths disagree on 59-74 of the 103 bf16 words, so the draw order the
    bucketing induces is very much visible here.
    """
    assert _pinned_sr_run(name, md, cache=cache) == _PINNED_SR_BITS[(name, md)]


# ---------------------------------------------------------- state-identity watch
# The plan's ``state_views`` alias the buffers a spec names, so a REBINDING of one of those
# keys has to invalidate it. That is what ``WatchedState`` + ``WATCHED_STATE_KEYS`` do — but
# only if (a) every key a spec bakes is in the watched set and (b) the optimizer's ``state``
# is actually a ``WatchedState``. Until 0.7.18 (a) missed AdaBelief's ``"s"`` and (b) held
# for Adakaon / AdaPNM only: every other foreach optimizer kept a plain ``defaultdict``, so
# ``opt.state[p]["row"] = ...`` (or ``["s"]``, ``["v"]``, ``["m"]``) left the plan stepping
# the RETIRED tensor. Measured 2/2 ``s``, 3/3 ``row`` on AdaBelief and 2/2 ``v`` on AdamP.
def _foreach_specs():
    """Every ``ForeachSpec`` any kaon optimizer class declares, by ``(class, attr)``."""
    import inspect

    import kaon
    from kaon._foreach_plan import ForeachSpec

    found = {}
    for _, cls in inspect.getmembers(kaon, inspect.isclass):
        if not issubclass(cls, ForeachPlanMixin):
            continue
        for klass in cls.__mro__:
            for attr, value in vars(klass).items():
                if isinstance(value, ForeachSpec):
                    found[(klass.__name__, attr)] = value
    return found


def test_every_spec_state_key_is_watched():
    """STRUCTURAL: a key a spec bakes into cached views must be one a rebinding of is seen."""
    from kaon._foreach_plan import WATCHED_STATE_KEYS

    specs = _foreach_specs()
    # Sanity: the scan found the real specs (a silent empty scan would pass vacuously).
    assert ("AdaBelief", "_FOREACH_SPEC") in specs
    assert ("KProdigy", "_FOREACH_SPEC_FACTORED") in specs
    missing = {
        where: sorted(set(spec.factored_state + spec.flat_state) - WATCHED_STATE_KEYS)
        for where, spec in specs.items()
    }
    missing = {k: v for k, v in missing.items() if v}
    assert not missing, f"spec state keys missing from WATCHED_STATE_KEYS: {missing}"


@pytest.mark.parametrize("name", NAMES)
def test_foreach_optimizer_state_is_watched(name):
    from kaon._foreach_plan import WatchedState

    _bag, opt = run(name, steps=1)
    assert type(opt.state) is WatchedState, (
        f"{name}.state is a {type(opt.state).__name__}: a state rebinding is invisible "
        "to its cached foreach plan"
    )


def _spec_keys(name, opt, p):
    spec = opt._foreach_spec(opt.param_groups[0])
    st = opt.state[p]
    keys = spec.factored_state if p.ndim >= 2 else spec.flat_state
    return [k for k in (*keys, "m", "m_scale") if k in st and torch.is_tensor(st[k])]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("md", ["float32", "int8"])
def test_rebinding_a_baked_state_key_drops_the_plan(name, md):
    """Swap every baked buffer for an equal-valued CLONE mid-run: the trajectory must be
    bit-identical to the uninterrupted run and the retired buffers must stay untouched."""
    ref = snapshot(*run(name, momentum_dtype=md))

    torch.manual_seed(11)
    reseed_stochastic_rounding()
    bag = make_bag()
    opt = build(name, bag, momentum_dtype=md)
    retired = []
    for step in range(1, 5):
        set_grads(bag, step)
        if step == 3:
            for p in bag:
                st = opt.state[p]
                for k in _spec_keys(name, opt, p):
                    old = st[k]
                    st[k] = old.clone()
                    retired.append((k, old, old.clone()))
        opt.step()
    assert retired, "nothing was rebound"
    dirty = sorted({k for k, live, snap in retired if not torch.equal(live, snap)})
    assert not dirty, f"{name}: the step wrote RETIRED state buffers {dirty}"
    assert_same(snapshot(bag, opt), ref)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("md", ["float32", "bfloat16", "int8", "4bit"])
def test_steady_state_steps_do_not_move_the_state_generation(name, md):
    """Nothing in a steady-state step may rebind a watched key: that would turn every step
    into a full plan rebuild (the watch is only free because the counter never moves)."""
    _bag, opt = run(name, steps=2, momentum_dtype=md)
    gen = opt.state.gen[0]
    plan = only_plan(opt)
    bag = _bag
    for step in (3, 4, 5):
        set_grads(bag, step)
        opt.step()
    assert opt.state.gen[0] == gen, f"{name}/{md}: a steady-state step rebinds a watched key"
    assert only_plan(opt) is plan
