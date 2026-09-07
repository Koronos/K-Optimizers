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


def _roundtrip(state_dict):
    """Save/load through torch so a test sees a real checkpoint, not the live tensors."""
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    buf.seek(0)
    return torch.load(buf, weights_only=False)


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
