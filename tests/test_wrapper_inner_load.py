"""Loading a checkpoint THROUGH a wrapper must leave the inner optimizer exactly as
``inner.load_state_dict`` would have left it.

kaon's wrappers (:class:`~kaon.lookahead.Lookahead`, :class:`~kaon.sam.SAM`,
:class:`~kaon.msam.MSAM`, :class:`~kaon.nekaon.Nekaon`) share ``param_groups`` with an
inner base optimizer and restore it through
:meth:`kaon._wrappers.WrapsInnerOptimizer._load_wrapped`. The inner's own
``load_state_dict`` is not a thin wrapper around torch's: for Adakaon it also

* drops every host-side cache holding pointers/views into the state tensors torch just
  **replaced** (``_invalidate_fused_caches``: the fused pointer caches and the foreach
  plans) — the same invalidation ``tests/test_foreach_plan.py`` pins for the standalone
  optimizers;
* consumes ``_adakaon_meta`` — restoring ``_t`` (the seed counter the Triton kernels use
  for their bf16 stochastic-rounding writes) and migrating a pre-0.7.11 lr-scaled
  momentum to direction units;
* back-fills a group key the checkpoint predates from the inner's ``defaults``.

Lookahead used to hand ``load_state_dict_preserving_dtypes`` straight to
``_load_wrapped``, skipping all three (it kept only the dtype preservation, which is one
*part* of what the inner's loader does). SAM and MSAM already delegated to
``inner.load_state_dict``; the structural tests below cover all three so the pattern
stays fixed.
"""

from __future__ import annotations

import copy
import io

import pytest
import torch

import kaon
from kaon._fused_triton import HAS_TRITON
from kaon.adakaon import Adakaon
from kaon.lookahead import Lookahead
from kaon.msam import MSAM
from kaon.sam import SAM

# Every host-side cache ``Adakaon._invalidate_fused_caches`` drops. A survivor holds views
# or raw pointer tables into state tensors the loader replaced, under a dead ``id(group)``.
_INNER_CACHES = (
    "_fused_part",
    "_fused_demoted",
    "_fused_ob_caches",
    "_fused_od_caches",
    "_fused_big_caches",
    "_foreach_plans",
)


def _cache_sizes(inner: Adakaon) -> dict[str, int]:
    return {name: len(getattr(inner, name)) for name in _INNER_CACHES}


def _make_params(shapes, *, dtype=torch.float32, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    return [
        torch.nn.Parameter(torch.randn(*s, generator=g).to(device=device, dtype=dtype))
        for s in shapes
    ]


def _grad_seq(params, steps, *, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [
        [torch.randn(*p.shape, generator=g) * 0.1 for p in params]
        for _ in range(steps)
    ]


def _apply(params, grads):
    for p, g in zip(params, grads, strict=True):
        p.grad = g.clone().to(device=p.device, dtype=p.dtype)


def _roundtrip(state_dict, map_location=None):
    """Save/load through torch so a test sees a real checkpoint, not the live tensors.

    ``map_location`` mirrors what the consumer actually writes: Rengu-Flow (and most
    training loops) load every checkpoint with ``map_location="cpu"`` and let the
    optimizer put the state back where the parameters live.
    """
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    buf.seek(0)
    return torch.load(buf, map_location=map_location, weights_only=False)


# ------------------------------------------------- (a) inner caches are invalidated

def test_lookahead_load_drops_the_inner_foreach_plan():
    """The loader REPLACES the inner's state tensors (and every group dict), so a plan
    cached under the old ``id(group)`` both dangles and leaks."""
    params = _make_params([(6, 6), (5,)])
    opt = Lookahead(params, lr=1e-2, k=2, momentum_dtype="float32", foreach=True, fused=False)
    for grads in _grad_seq(params, 2):
        _apply(params, grads)
        opt.step()
    assert opt.inner._foreach_plans

    opt.load_state_dict(_roundtrip(opt.state_dict()))
    assert _cache_sizes(opt.inner) == dict.fromkeys(_INNER_CACHES, 0)

    _apply(params, _grad_seq(params, 1, seed=2)[0])
    opt.step()  # rebuilds cleanly against the restored buffers, no orphan left behind
    assert len(opt.inner._foreach_plans) == 1


@pytest.mark.parametrize("name", ["SAM", "MSAM"])
def test_wrapper_load_drops_the_inner_foreach_plan(name):
    """Same invariant for the wrappers that already delegated — a regression guard."""
    params = _make_params([(6, 6), (5,)])
    kwargs = dict(lr=1e-2, momentum_dtype="float32", foreach=True, fused=False)
    opt = SAM(params, **kwargs) if name == "SAM" else MSAM(params, **kwargs)
    if name == "MSAM":
        opt.train()
    for grads in _grad_seq(params, 2):
        _apply(params, grads)
        if name == "SAM":
            opt.first_step()
            _apply(params, grads)
            opt.second_step()
        else:
            opt.step()
    assert opt.inner._foreach_plans

    if name == "MSAM":
        opt.eval()  # MSAM checkpoints must be taken on the unperturbed weights
    opt.load_state_dict(_roundtrip(opt.state_dict()))
    assert _cache_sizes(opt.inner) == dict.fromkeys(_INNER_CACHES, 0)


@pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()), reason="fused caches need CUDA + Triton"
)
def test_lookahead_load_drops_the_inner_fused_pointer_caches():
    """The fused path caches raw pointer tables; those must not survive a wrapper load."""
    params = _make_params([(64, 64), (64,)], dtype=torch.bfloat16, device="cuda")
    opt = Lookahead(params, lr=1e-2, k=2, momentum_dtype="bfloat16", fused=True)
    for grads in _grad_seq(params, 2):
        _apply(params, grads)
        opt.step()
    assert sum(_cache_sizes(opt.inner).values()) > 0

    opt.load_state_dict(_roundtrip(opt.state_dict()))
    assert _cache_sizes(opt.inner) == dict.fromkeys(_INNER_CACHES, 0)


# ------------------------------------------- _adakaon_meta reaches the inner's loader

def test_lookahead_load_restores_the_inner_fused_step_counter():
    """``_adakaon_meta['fused_step']`` is the Triton SR seed counter. Dropping it restarts
    the bf16 write noise at 0 on resume, diverging from an uninterrupted run."""
    params = _make_params([(6, 6)])
    opt = Lookahead(params, lr=1e-2, k=2, momentum_dtype="float32")
    _apply(params, _grad_seq(params, 1)[0])
    opt.step()
    sd = copy.deepcopy(opt.state_dict())
    sd["_adakaon_meta"]["fused_step"] = 7

    opt2 = Lookahead(_make_params([(6, 6)], seed=9), lr=1e-2, k=2, momentum_dtype="float32")
    opt2.load_state_dict(sd)
    assert opt2.inner._t == 7


def test_lookahead_load_rejects_a_negative_fused_step_counter():
    """Same validation a direct ``Adakaon.load_state_dict`` applies."""
    params = _make_params([(6, 6)])
    opt = Lookahead(params, lr=1e-2, k=2, momentum_dtype="float32")
    _apply(params, _grad_seq(params, 1)[0])
    opt.step()
    sd = copy.deepcopy(opt.state_dict())
    sd["_adakaon_meta"]["fused_step"] = -1

    with pytest.raises(ValueError, match="invalid fused step counter"):
        opt.load_state_dict(sd)


def test_lookahead_load_migrates_pre_0_7_11_momentum_units():
    """A checkpoint whose momentum still folds lr in must be rescaled to direction units
    — identically to what a direct ``Adakaon.load_state_dict`` does with it."""
    lr = 0.25
    params = _make_params([(6, 6)])
    opt = Lookahead(params, lr=lr, k=2, momentum_dtype="float32")
    _apply(params, _grad_seq(params, 1)[0])
    opt.step()
    sd = copy.deepcopy(opt.state_dict())
    sd["_adakaon_meta"]["momentum_units"] = 1
    checkpointed_m = sd["state"][0]["m"].clone()

    wrapped = Lookahead(_make_params([(6, 6)], seed=9), lr=lr, k=2, momentum_dtype="float32")
    wrapped.load_state_dict(copy.deepcopy(sd))

    direct_params = _make_params([(6, 6)], seed=9)
    direct = Adakaon(direct_params, lr=lr, momentum_dtype="float32")
    inner_sd = copy.deepcopy(sd)
    inner_sd.pop("lookahead")
    direct.load_state_dict(inner_sd)

    got = wrapped.inner.state[wrapped.param_groups[0]["params"][0]]["m"]
    torch.testing.assert_close(got, direct.state[direct_params[0]]["m"], rtol=0, atol=0)
    # ... and the migration actually did something (guards a vacuous comparison).
    assert not torch.equal(got, checkpointed_m)


# ------------------------------------------- (c) the inner's own defaults back-fill

def test_lookahead_load_backfills_an_inner_group_default():
    """``cautious_wd`` is back-filled by ``Adakaon.load_state_dict``; a checkpoint that
    predates it must resume through the wrapper too, not die on the first step."""
    params = _make_params([(6, 6)])
    opt = Lookahead(params, lr=1e-2, k=1, momentum_dtype="float32", cautious=True)
    _apply(params, _grad_seq(params, 1)[0])
    opt.step()
    sd = copy.deepcopy(opt.state_dict())
    for pg in sd["param_groups"]:
        pg.pop("cautious_wd", None)

    params2 = _make_params([(6, 6)], seed=9)
    opt2 = Lookahead(params2, lr=1e-2, k=1, momentum_dtype="float32", cautious=True)
    opt2.load_state_dict(sd)

    assert opt2.param_groups[0]["cautious_wd"] == opt2.inner.defaults["cautious_wd"]
    _apply(params2, _grad_seq(params2, 1, seed=3)[0])
    opt2.step()  # must not raise KeyError


def test_lookahead_load_leaves_the_inner_like_a_direct_load():
    """End-to-end parity: the restored groups, inner state and counters must match what
    ``Adakaon.load_state_dict`` produces from the same checkpoint."""
    shapes = [(6, 6), (5,)]
    params = _make_params(shapes)
    opt = Lookahead(params, lr=1e-2, k=2, momentum_dtype="int8")
    for grads in _grad_seq(params, 3):
        _apply(params, grads)
        opt.step()
    sd = copy.deepcopy(opt.state_dict())
    for pg in sd["param_groups"]:
        pg.pop("cautious_wd", None)

    wrapped = Lookahead(_make_params(shapes, seed=9), lr=1e-2, k=2, momentum_dtype="int8")
    wrapped.load_state_dict(copy.deepcopy(sd))

    direct_params = _make_params(shapes, seed=9)
    direct = Adakaon(direct_params, lr=1e-2, momentum_dtype="int8")
    inner_sd = copy.deepcopy(sd)
    inner_sd.pop("lookahead")
    direct.load_state_dict(inner_sd)

    assert wrapped.inner._t == direct._t
    # Drop ``params`` (different tensors) and Lookahead's own keys — the checkpoint's group
    # dicts carry those, so the bare Adakaon gets them restored verbatim as inert extras.
    skip = {"params", *wrapped.defaults}
    for gw, gd in zip(wrapped.param_groups, direct.param_groups, strict=True):
        assert {k: v for k, v in gw.items() if k not in skip} == {
            k: v for k, v in gd.items() if k not in skip
        }
    for pw, pd in zip(wrapped.param_groups[0]["params"], direct_params, strict=True):
        sw, sd_direct = wrapped.inner.state[pw], direct.state[pd]
        assert set(sw) == set(sd_direct)
        for key, ref in sd_direct.items():
            if torch.is_tensor(ref):
                assert sw[key].dtype == ref.dtype, key
                assert torch.equal(sw[key], ref), key
            else:
                assert sw[key] == ref, key


# ------------------------------------------------------- (b) numerical resume parity

_ARMS = [
    pytest.param("cpu", False, torch.float32, id="cpu-native-fp32params"),
    pytest.param("cuda", False, torch.bfloat16, id="cuda-native-bf16params"),
    pytest.param("cuda", True, torch.bfloat16, id="cuda-fused-bf16params"),
]
_MOMENTUM = ["float32", "bfloat16", "int8", "4bit"]
_SHAPES = [(64, 32), (48, 16), (64,)]
_WARMUP, _TAIL = 4, 3


def _snapshot(params, opt):
    """Weights + the inner's per-param state + the wrapper's own (``phi``) state."""
    out: list = [p.detach().float().cpu().clone() for p in params]
    for p in params:
        for store in (opt.inner.state[p], opt.state[p]):
            for _key, value in sorted(store.items()):
                out.append(value.detach().cpu().clone() if torch.is_tensor(value) else value)
    return out


def _assert_same(a, b, what):
    assert len(a) == len(b)
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        if torch.is_tensor(x):
            assert x.dtype == y.dtype, f"{what}[{i}] dtype"
            assert torch.equal(x, y), f"{what}[{i}] values"
        else:
            assert x == y, f"{what}[{i}]"


@pytest.mark.parametrize("momentum_dtype", _MOMENTUM)
@pytest.mark.parametrize("device, fused, param_dtype", _ARMS)
def test_resume_through_lookahead_is_bit_identical(device, fused, param_dtype, momentum_dtype):
    """A resume through the wrapper — into a fresh optimizer AND into the same (warm,
    plan-cached) one — must land bit-for-bit on the uninterrupted run, and on each other.

    The three arms run **sequentially** and each re-seeds kaon's stochastic-rounding
    streams first, so every arm draws the same SR noise at the same step index (those
    streams are global and advance per draw; interleaving the arms would desynchronise
    them for a reason unrelated to what is under test).
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if fused and not HAS_TRITON:
        pytest.skip("Triton not available")

    kwargs = dict(lr=1e-2, k=2, alpha=0.5, momentum_dtype=momentum_dtype, fused=fused)
    grads = _grad_seq(_make_params(_SHAPES), _WARMUP + _TAIL, seed=5)

    def fresh():
        torch.manual_seed(0x5EED)
        kaon.reseed_stochastic_rounding()
        params = _make_params(_SHAPES, dtype=param_dtype, device=device, seed=3)
        return params, Lookahead(params, **kwargs)

    def advance(params, opt, window):
        for gs in window:
            _apply(params, gs)
            opt.step()

    # reference: never saves, never loads.
    ref_params, ref_opt = fresh()
    advance(ref_params, ref_opt, grads)
    reference = _snapshot(ref_params, ref_opt)

    # arm 1: resume into a FRESH Lookahead (cold caches).
    cold_params, cold_opt = fresh()
    advance(cold_params, cold_opt, grads[:_WARMUP])
    sd = _roundtrip(cold_opt.state_dict())
    cold_opt = Lookahead(cold_params, **kwargs)
    cold_opt.load_state_dict(sd)
    advance(cold_params, cold_opt, grads[_WARMUP:])
    cold = _snapshot(cold_params, cold_opt)
    _assert_same(reference, cold, "cold-resume")

    # arm 2: resume into the SAME Lookahead, whose plans / pointer caches are already warm.
    warm_params, warm_opt = fresh()
    advance(warm_params, warm_opt, grads[:_WARMUP])
    warm_opt.load_state_dict(_roundtrip(warm_opt.state_dict()))
    advance(warm_params, warm_opt, grads[_WARMUP:])
    warm = _snapshot(warm_params, warm_opt)
    _assert_same(reference, warm, "warm-resume")
    _assert_same(cold, warm, "cold-vs-warm")


# ---------------------- (c) map_location: the wrapper's own state follows its parameter

# ``torch.optim.Optimizer.load_state_dict`` puts every per-param state tensor on the
# param's device, so the INNER optimizer always came back correct even from a checkpoint
# loaded with ``map_location="cpu"``. The wrapper's own per-param state does not go
# through that loader — ``_load_wrapped`` used to install the checkpoint's dicts verbatim,
# leaving Lookahead's ``phi`` (and its scales) on the CPU under CUDA parameters until the
# next sync raised "Expected all tensors to be on the same device". These pin the mixin's
# contract: device follows the parameter, storage dtype is preserved exactly.

_SLOW = ["float32", "bfloat16", "int8", "4bit"]


def _wrapper_state_tensors(opt):
    """``{(param index, key): tensor}`` for every tensor in the WRAPPER's own state."""
    return {
        (i, key): value
        for i, p in enumerate(opt._flat_params())
        for key, value in opt.state[p].items()
        if torch.is_tensor(value)
    }


def _assert_state_on_param_devices(opt, what):
    params = opt._flat_params()
    tensors = _wrapper_state_tensors(opt)
    assert tensors, f"{what}: no wrapper state to check — the test is not exercising anything"
    for (i, key), value in tensors.items():
        assert value.device == params[i].device, (
            f"{what}: wrapper state [{i}][{key!r}] is on {value.device}, "
            f"but its parameter lives on {params[i].device}"
        )


@pytest.mark.parametrize("param_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("slow_dtype", _SLOW)
def test_lookahead_load_from_a_cpu_mapped_checkpoint_syncs(slow_dtype, param_dtype):
    """The consumer's idiom: ``torch.load(path, map_location="cpu")`` into CUDA params.

    ``phi`` (plus ``phi_scale`` at int8/4bit) must come back on the parameter's device, at
    its declared storage dtype, and the next sync must run instead of raising.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shapes = [(16, 8), (8,)]
    params = _make_params(shapes, dtype=param_dtype, device="cuda")
    kwargs = dict(lr=1e-2, k=2, slow_dtype=slow_dtype, momentum_dtype="int8", fused=False)
    opt = Lookahead(params, **kwargs)
    for grads in _grad_seq(params, 3):
        _apply(params, grads)
        opt.step()
    stored = {k: (t.dtype, t.shape) for k, t in _wrapper_state_tensors(opt).items()}

    reloaded = Lookahead(params, **kwargs)
    reloaded.load_state_dict(_roundtrip(opt.state_dict(), map_location="cpu"))

    _assert_state_on_param_devices(reloaded, "cpu-mapped load")
    assert {k: (t.dtype, t.shape) for k, t in _wrapper_state_tensors(reloaded).items()} == stored

    for grads in _grad_seq(params, 2, seed=7):  # crosses a k=2 sync
        _apply(params, grads)
        reloaded.step()


@pytest.mark.parametrize("slow_dtype", _SLOW)
def test_lookahead_load_with_explicit_cuda_map_location_still_works(slow_dtype):
    """A checkpoint mapped straight onto the param's device must stay a no-op: nothing to
    move, so nothing is copied or reallocated."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    params = _make_params([(16, 8), (8,)], dtype=torch.bfloat16, device="cuda")
    kwargs = dict(lr=1e-2, k=2, slow_dtype=slow_dtype, fused=False)
    opt = Lookahead(params, **kwargs)
    for grads in _grad_seq(params, 3):
        _apply(params, grads)
        opt.step()

    sd = _roundtrip(opt.state_dict(), map_location=torch.device("cuda", 0))
    from_checkpoint = {
        (int(i), key): value
        for i, st in sd["lookahead"].items()
        for key, value in st.items()
        if torch.is_tensor(value)
    }
    reloaded = Lookahead(params, **kwargs)
    reloaded.load_state_dict(sd)

    _assert_state_on_param_devices(reloaded, "cuda-mapped load")
    # Already on the right device: the very tensors the checkpoint carried are installed,
    # not copies of them — no allocation on the load path when nothing has to move.
    for key, restored in _wrapper_state_tensors(reloaded).items():
        assert restored is from_checkpoint[key], f"{key} was needlessly reallocated"

    _apply(params, _grad_seq(params, 1, seed=7)[0])
    reloaded.step()


@pytest.mark.parametrize("slow_dtype", _SLOW)
def test_lookahead_load_of_a_cuda_checkpoint_into_cpu_params_syncs(slow_dtype):
    """The inverse direction: a CUDA checkpoint restored under CPU parameters (an offload
    resume, or a GPU run continued on the CPU). The state must come DOWN to the params."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    shapes = [(16, 8), (8,)]
    kwargs = dict(lr=1e-2, k=2, slow_dtype=slow_dtype, fused=False)
    cuda_params = _make_params(shapes, device="cuda")
    cuda_opt = Lookahead(cuda_params, **kwargs)
    for grads in _grad_seq(cuda_params, 3):
        _apply(cuda_params, grads)
        cuda_opt.step()
    sd = _roundtrip(cuda_opt.state_dict())  # no map_location: comes back on CUDA

    cpu_params = _make_params(shapes, device="cpu")
    cpu_opt = Lookahead(cpu_params, **kwargs)
    cpu_opt.load_state_dict(sd)

    _assert_state_on_param_devices(cpu_opt, "cuda checkpoint into cpu params")
    for grads in _grad_seq(cpu_params, 2, seed=7):
        _apply(cpu_params, grads)
        cpu_opt.step()


def test_lookahead_eval_backup_lands_on_the_param_device():
    """A checkpoint taken in eval mode (what the docs prescribe) carries the saved fast
    weights ``backup``; the ``train()`` after the resume restores them, so they migrate."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    params = _make_params([(16, 8), (8,)], device="cuda")
    kwargs = dict(lr=1e-2, k=2, slow_dtype="bfloat16", fused=False)
    opt = Lookahead(params, **kwargs)
    for grads in _grad_seq(params, 3):
        _apply(params, grads)
        opt.step()
    opt.eval()
    assert any(key == "backup" for _i, key in _wrapper_state_tensors(opt))

    reloaded = Lookahead(params, **kwargs)
    reloaded.load_state_dict(_roundtrip(opt.state_dict(), map_location="cpu"))
    _assert_state_on_param_devices(reloaded, "eval-mode checkpoint")


def test_sam_load_from_a_cpu_mapped_checkpoint_keeps_its_state_on_the_params():
    """SAM keeps no state ACROSS steps, but a checkpoint taken between its two passes
    carries ``old_p``. The mixin's contract is wrapper-agnostic, so pin it here too."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    params = _make_params([(16, 8), (8,)], device="cuda")
    kwargs = dict(lr=1e-2, fused=False)
    opt = SAM(params, **kwargs)
    _apply(params, _grad_seq(params, 1)[0])
    opt.first_step()
    assert any(key == "old_p" for _i, key in _wrapper_state_tensors(opt))

    reloaded = SAM(params, **kwargs)
    reloaded.load_state_dict(_roundtrip(opt.state_dict(), map_location="cpu"))
    _assert_state_on_param_devices(reloaded, "SAM mid-step checkpoint")


@pytest.mark.parametrize("momentum_dtype", _MOMENTUM)
@pytest.mark.parametrize("slow_dtype", _SLOW)
def test_cpu_mapped_resume_through_lookahead_is_bit_identical(slow_dtype, momentum_dtype):
    """A resume from a ``map_location="cpu"`` checkpoint must land bit-for-bit on the
    uninterrupted run — a device move changes no bit, so the only thing this can catch is
    the migration doing something *else* (a dtype cast, a requant, a re-scaled buffer).

    Same protocol as :func:`test_resume_through_lookahead_is_bit_identical`: the arms run
    sequentially and each re-seeds kaon's global stochastic-rounding streams first, so
    both draw the same SR noise at the same step index.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    kwargs = dict(
        lr=1e-2, k=2, alpha=0.5, slow_dtype=slow_dtype,
        momentum_dtype=momentum_dtype, fused=False,
    )
    grads = _grad_seq(_make_params(_SHAPES), _WARMUP + _TAIL, seed=5)

    def fresh():
        torch.manual_seed(0x5EED)
        kaon.reseed_stochastic_rounding()
        params = _make_params(_SHAPES, dtype=torch.bfloat16, device="cuda", seed=3)
        return params, Lookahead(params, **kwargs)

    def advance(params, opt, window):
        for gs in window:
            _apply(params, gs)
            opt.step()

    ref_params, ref_opt = fresh()
    advance(ref_params, ref_opt, grads)
    reference = _snapshot(ref_params, ref_opt)

    res_params, res_opt = fresh()
    advance(res_params, res_opt, grads[:_WARMUP])
    sd = _roundtrip(res_opt.state_dict(), map_location="cpu")
    res_opt = Lookahead(res_params, **kwargs)
    res_opt.load_state_dict(sd)
    advance(res_params, res_opt, grads[_WARMUP:])
    _assert_same(reference, _snapshot(res_params, res_opt), "cpu-mapped resume")
