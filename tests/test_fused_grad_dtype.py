"""A gradient whose dtype is not the param's (fp32 grad on a bf16 weight) must not reach a fused kernel.

Every fused kernel types the gradient pointer by the PARAM's dtype (``LOWP`` casts it to bf16
exactly when the weight is bf16), so an fp32 ``.grad`` on a bf16 weight — reachable with
``p.grad_dtype = None`` in torch 2.12, or from a trainer that keeps fp32 grads — was read as
bf16 pairs of fp32 words: NaN/garbage weights from the first step, on every fused route
(one-block, big, 1-D), every ``bf16_method`` and every fused optimizer (Adakaon, Nekaon's inner
Adakaon, AdaPNM). The fix demotes such a param to the native path FOR THAT STEP, in the same
per-step sweep that demotes non-contiguous grads; the native path upcasts any grad to fp32.

What is pinned here:

1. fused + fp32 grad == ``fused=False`` + the same fp32 grad, BIT FOR BIT (weights and the
   compact-Kahan residual), per optimizer x ``bf16_method`` x route. Bit-exactness is
   possible because a demoted bag runs the very same native code the ``fused=False`` optimizer
   runs, and both SR noise streams are pinned to the same id (:class:`SRStream`).
2. Only the offending tensor leaves the fused route; its neighbours stay fused.
3. A grad that changes dtype between steps re-routes every step (the demotion memo keys on
   the demoted SET, never on a stale partition), and the trajectory matches native.
4. The NON-fused paths (per-param and foreach, including a foreach bucket whose grads mix
   bf16 and fp32) read an fp32 grad losslessly: ``kahan16`` bf16 weights reproduce the
   fp32-weight run on the upcast grads bit for bit on the same route — a lossy downcast of the
   fp32 grads would break that equality.
"""

from __future__ import annotations

import pytest
import torch

from kaon import Adakaon, AdaPNM, Nekaon, decode_weights
from kaon._fused_triton import HAS_TRITON
from kaon._stochastic_rounding import SRStream

pytestmark = pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()),
    reason="Triton fused kernel requires CUDA + Triton",
)

DEV = "cuda"

_OPTS = {"Adakaon": Adakaon, "Nekaon": Nekaon, "AdaPNM": AdaPNM}
_METHODS = ["stochastic_rounding", "kahan8", "kahan16"]
# One bag per fused route; the routing is asserted, not assumed.
_ROUTES = {
    "one_block": [(16, 8)] * 3,
    "big": [(512, 512)] * 2,
    "big_lone": [(1024, 1200)],
    "one_dim": [(64,)] * 3,
}
_CFG = dict(lr=1e-3, weight_decay=0.02)


# ----------------------------------------------------------------- helpers
def _allow_any_grad_dtype(p: torch.Tensor) -> None:
    """Let ``p`` hold a grad of another dtype (torch >= 2.12 enforces ``grad_dtype``)."""
    if hasattr(p, "grad_dtype"):
        p.grad_dtype = None
    try:
        p.grad = torch.zeros(p.shape, device=p.device, dtype=torch.float32)
    except (RuntimeError, TypeError) as exc:  # older torch: a mismatched grad is refused
        pytest.skip(f"this torch cannot hold an fp32 grad on a bf16 param: {exc}")
    p.grad = None


def _bag(shapes, seed, dtype=torch.bfloat16):
    g = torch.Generator(device=DEV).manual_seed(seed)
    ps = [torch.nn.Parameter(torch.randn(s, generator=g, device=DEV).to(dtype)) for s in shapes]
    for p in ps:
        _allow_any_grad_dtype(p)
    return ps


def _clone(ps, dtype=None):
    out = [torch.nn.Parameter(p.detach().clone().to(dtype or p.dtype)) for p in ps]
    for p in out:
        _allow_any_grad_dtype(p)
    return out


def _pin_sr(opt) -> None:
    """Pin every SR owner in ``opt`` (wrapper and inner) to the same noise id, so two optimizers
    built side by side draw the same stochastic-rounding noise."""
    for o in (opt, getattr(opt, "inner", None)):
        if o is not None:
            o.sr_stream = SRStream(7)


def _make(name, params, fused, method, foreach=True, **extra):
    cfg = {**_CFG, **extra}
    opt = _OPTS[name](params, fused=fused, foreach=foreach, bf16_method=method, **cfg)
    _pin_sr(opt)
    return opt


def _fused_owner(opt):
    """The optimizer that holds the fused partition (Nekaon delegates to its inner Adakaon)."""
    return getattr(opt, "inner", opt)


def _parts(opt):
    ob, big, od, nat = [], [], [], []
    for entry in _fused_owner(opt)._fused_part.values():
        o, b, d, n = entry[-4:]
        ob += o
        big += b
        od += d
        nat += n
    return ob, big, od, nat


def _demoted_ids(opt) -> set[int]:
    """Params the per-step demotion moved to native on the LAST step."""
    out: set[int] = set()
    for demoted, _parts_in, _out in _fused_owner(opt)._fused_demoted.values():
        out |= set(demoted)
    return out


def _route_of(opt, p) -> str:
    ob, big, od, nat = _parts(opt)
    for name, sub in (("one_block", ob), ("big", big), ("one_dim", od), ("native", nat)):
        if any(q is p for q in sub):
            return name
    raise AssertionError("param not in the fused partition")


def _drive(pairs, steps, seed, dtype_of):
    """Step every (params, optimizer) pair on IDENTICAL gradients. ``dtype_of(step, i)`` is the
    dtype of param ``i``'s grad at ``step``; values are drawn in fp32 and cast, so a bf16 grad
    and its fp32 twin (``.float()`` of it) are the same numbers."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    for step in range(steps):
        raw = [torch.randn(tuple(p.shape), generator=gen, device=DEV) for p in pairs[0][0]]
        for plist, opt in pairs:
            for i, (p, g) in enumerate(zip(plist, raw, strict=True)):
                p.grad = g.to(dtype_of(step, i, p), copy=True)  # never alias across pairs
            opt.step()
    torch.cuda.synchronize()


def _assert_same_bits(pa, pb, oa, ob, what):
    for i, (a, b) in enumerate(zip(pa, pb, strict=True)):
        assert torch.isfinite(a).all(), f"{what}: param {i} is not finite"
        assert torch.equal(a.detach(), b.detach()), (
            f"{what}: param {i} differs, max|d|="
            f"{(a.detach().float() - b.detach().float()).abs().max().item():.3e}")
        ra = _fused_owner(oa).state[a].get("kahan_lo")
        rb = _fused_owner(ob).state[b].get("kahan_lo")
        assert (ra is None) == (rb is None), f"{what}: residual present on one side only"
        if ra is not None:
            assert torch.equal(ra, rb), f"{what}: kahan residual of param {i} differs"


def _fp32_grads(step, i, p):
    return torch.float32


# ----------------------------------------------------------------- 1. fused == native
@pytest.mark.parametrize("route", list(_ROUTES))
@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("name", list(_OPTS))
def test_fp32_grad_on_bf16_param_matches_native(name, method, route):
    """The repro: fp32 grads on bf16 weights under ``fused=True``. Must be finite AND the same
    bits as ``fused=False`` on the same grads — the param is demoted, not stepped by a kernel
    reading its fp32 words as bf16."""
    shapes = _ROUTES[route]
    pv = _bag(shapes, seed=11)
    pn = _clone(pv)
    ov = _make(name, pv, True, method)
    on = _make(name, pn, False, method)
    _drive([(pv, ov), (pn, on)], 3, seed=13, dtype_of=_fp32_grads)
    # The partition still ROUTES the params to the fused kernel (it only sees the param) —
    # the per-step grad sweep is what keeps them off it. AdaPNM fuses bf16 weights only under
    # SR, so its compact-Kahan cases are native by partition (never affected; still pinned).
    if name == "AdaPNM" and method != "stochastic_rounding":
        assert all(_route_of(ov, p) == "native" for p in pv)
    else:
        expected = "big" if route.startswith("big") else route
        assert all(_route_of(ov, p) == expected for p in pv), [_route_of(ov, p) for p in pv]
        assert _demoted_ids(ov) == {id(p) for p in pv}
    _assert_same_bits(pv, pn, ov, on, f"{name}/{method}/{route}")


@pytest.mark.parametrize("route", list(_ROUTES))
@pytest.mark.parametrize("name", list(_OPTS))
def test_bf16_grad_on_fp32_param_matches_native(name, route):
    """The mirror case: a bf16 grad on an fp32 weight would be read as fp32 words spanning two
    bf16 values (and past the end of the buffer). Same demotion, same bits as native."""
    pv = _bag(_ROUTES[route], seed=47, dtype=torch.float32)
    pn = _clone(pv)
    ov = _make(name, pv, True, "stochastic_rounding")
    on = _make(name, pn, False, "stochastic_rounding")
    _drive([(pv, ov), (pn, on)], 3, seed=53, dtype_of=lambda s, i, p: torch.bfloat16)
    expected = "big" if route.startswith("big") else route
    assert all(_route_of(ov, p) == expected for p in pv), [_route_of(ov, p) for p in pv]
    assert _demoted_ids(ov) == {id(p) for p in pv}
    _assert_same_bits(pv, pn, ov, on, f"{name}/fp32-param/{route}")


# ----------------------------------------------------------------- 2. only the offender moves
@pytest.mark.parametrize("name", list(_OPTS))
def test_one_fp32_grad_demotes_only_its_own_param(name):
    """One fp32 grad in a bag of bf16 grads: that param alone leaves the fused route, its
    neighbours keep it, and every weight stays finite."""
    pv = _bag([(16, 8)] * 3, seed=17)
    ov = _make(name, pv, True, "stochastic_rounding")

    def dtype_of(step, i, p):
        return torch.float32 if i == 1 else torch.bfloat16

    _drive([(pv, ov)], 3, seed=19, dtype_of=dtype_of)
    assert _demoted_ids(ov) == {id(pv[1])}
    ((_, _, out),) = _fused_owner(ov)._fused_demoted.values()
    one_block, _big, _od, native = out
    assert [id(p) for p in one_block] == [id(pv[0]), id(pv[2])]
    assert [id(p) for p in native] == [id(pv[1])]
    assert all(torch.isfinite(p).all() for p in pv)


def test_demoted_param_steps_exactly_like_a_lone_native_param():
    """The demoted param runs the native per-param step with the same grads — bit for bit what
    a lone ``fused=False`` optimizer over that param computes (per-param state, no coupling)."""
    pv = _bag([(16, 8)] * 3, seed=23)
    lone = _clone([pv[1]])
    ov = _make("Adakaon", pv, True, "kahan16")
    ol = _make("Adakaon", lone, False, "kahan16")
    gen = torch.Generator(device=DEV).manual_seed(29)
    for _ in range(3):
        raw = [torch.randn((16, 8), generator=gen, device=DEV) for _ in pv]
        for i, p in enumerate(pv):
            p.grad = raw[i] if i == 1 else raw[i].bfloat16()
        lone[0].grad = raw[1].clone()
        ov.step()
        ol.step()
    _assert_same_bits([pv[1]], lone, ov, ol, "demoted vs lone native")


# ----------------------------------------------------------------- 3. dtype changes per step
_ALTERNATING = [
    # (optimizer, param dtype, bf16_method, grad dtype on EVEN steps, grad dtype on ODD steps)
    ("Adakaon", torch.bfloat16, "kahan16", torch.float32, torch.bfloat16),
    ("Nekaon", torch.bfloat16, "kahan16", torch.float32, torch.bfloat16),
    # AdaPNM fuses bf16 weights only under SR, whose noise differs between kernel and native;
    # fp32 weights with alternating bf16/fp32 grads exercise the same re-routing exactly.
    ("AdaPNM", torch.float32, "stochastic_rounding", torch.bfloat16, torch.float32),
    ("Adakaon", torch.float32, "stochastic_rounding", torch.bfloat16, torch.float32),
]


@pytest.mark.parametrize("name,pdtype,method,even,odd", _ALTERNATING,
                         ids=[f"{c[0]}-{str(c[1])[6:]}" for c in _ALTERNATING])
def test_grad_dtype_changing_between_steps_reroutes_and_matches_native(
        name, pdtype, method, even, odd):
    """Mismatched -> matching -> mismatched ... grad dtypes on consecutive steps, starting
    MISMATCHED on the step that builds (and caches) the partition. Each step must route by THAT
    step's grad: demoted on the mismatched steps, back on the fused kernel on the matching ones
    (the memo must neither keep a param native nor hand back a stale demotion). Compared on the
    fp32 trajectory (``kahan16`` decoded, or the fp32 weights): fused and native steps are the
    same math and differ only by fp32 rounding, so the bound is fp32-rounding sized; the bug
    produced NaN/garbage (|d| >> 1)."""
    pv = _bag([(16, 8)] * 3, seed=31, dtype=pdtype)
    pn = _clone(pv)
    ov = _make(name, pv, True, method)
    on = _make(name, pn, False, method)
    for step in range(6):
        dt = even if step % 2 == 0 else odd
        _drive([(pv, ov), (pn, on)], 1, seed=37 + step, dtype_of=lambda _s, i, p, d=dt: d)
        want = {id(p) for p in pv} if step % 2 == 0 else set()
        assert _demoted_ids(ov) == want, f"step {step}: wrong demotion"
        assert all(torch.isfinite(p).all() for p in pv), f"step {step}: non-finite weights"
    for o in (ov, on):
        if hasattr(o, "eval"):
            o.eval()
    dv, dn = decode_weights(ov), decode_weights(on)
    for a, b in zip(pv, pn, strict=True):
        d = (dv[a] - dn[b]).abs().max().item()
        assert d < 1e-6, f"{name}: max|d|={d:.2e} across alternating grad dtypes"


# ----------------------------------------------------------------- 4. native paths are lossless
_NATIVE_CASES = [("Adakaon", True), ("Adakaon", False), ("Nekaon", True), ("Nekaon", False),
                 ("AdaPNM", False)]


@pytest.mark.parametrize("foreach", [True, False], ids=["foreach", "per_param"])
@pytest.mark.parametrize("name,gc", _NATIVE_CASES,
                         ids=[f"{n}-{'gc' if g else 'nogc'}" for n, g in _NATIVE_CASES])
def test_native_paths_read_fp32_grads_losslessly(name, gc, foreach):
    """``fused=False``: a bucket of bf16 weights whose grads MIX bf16 and fp32. ``kahan16``
    reproduces the fp32-weight run bit for bit on the same route, given the same grad VALUES
    (the fp32 run gets the exact upcast of each bf16 grad). If the foreach stack downcast the
    fp32 grads to bf16, or broke on the mixed stack, this would fail.

    With GC on this also pins the foreach GC decision: it used to be taken per bucket from the
    FIRST param's GRAD dtype, so in a mixed bucket the bf16-grad rows got no GC at all (3.6e-4
    off the fp32 run on Adakaon). ``weight_decay=0`` and AdaPNM without GC: AdaPNM's decay and
    GC still read the bf16 weight / centralize a bf16 grad in bf16 (CHANGELOG 0.7.16
    follow-up), which breaks the kahan16/fp32 identity for reasons unrelated to fp32 grads."""
    shapes = [(16, 8)] * 3 + [(64,)] * 3
    pb = _bag(shapes, seed=41)
    pf = _clone(pb, torch.float32)
    cfg = dict(foreach=foreach, weight_decay=0.0, gradient_centralization=gc)
    ob = _make(name, pb, False, "kahan16", **cfg)
    of = _make(name, pf, False, "kahan16", **cfg)

    def dtype_of(step, i, p):
        if p.dtype == torch.float32:
            return torch.float32
        return torch.float32 if (i + step) % 2 == 0 else torch.bfloat16

    gen = torch.Generator(device=DEV).manual_seed(43)
    for step in range(4):
        raw = [torch.randn(s, generator=gen, device=DEV) for s in shapes]
        for i, (p, q, g) in enumerate(zip(pb, pf, raw, strict=True)):
            p.grad = g.to(dtype_of(step, i, p))
            q.grad = p.grad.to(torch.float32, copy=True)  # same VALUES, never aliased
        ob.step()
        of.step()
    torch.cuda.synchronize()
    for o in (ob, of):
        if hasattr(o, "eval"):
            o.eval()
    dec = decode_weights(ob)
    for i, (a, b) in enumerate(zip(pb, pf, strict=True)):
        assert torch.isfinite(dec[a]).all(), f"{name}: param {i} not finite"
        assert torch.equal(dec[a], b.detach()), (
            f"{name}/{'foreach' if foreach else 'per-param'}: param {i} differs from the fp32 "
            f"run, max|d|={(dec[a] - b.detach()).abs().max().item():.3e}")
