"""Gradient Centralization must not annihilate a fan-in-1 gradient.

GC (Yong et al. 2020) subtracts, per output row, the gradient's mean over the *fan-in*
dims (every dim but dim 0). For a weight whose fan-in is a single element — a rank-1
LoRA up-projection ``(out, 1)``, a one-input 1x1 conv ``(out, 1, 1, 1)`` — that mean
**is** the element, so ``g - mean(g)`` is identically zero and the parameter freezes,
silently, on every route. The fix is semantic, not numerical: GC is *undefined* on a
one-element row, so it is SKIPPED there.

The predicate has exactly one definition, :func:`kaon._backend.gc_applies`, because GC
is implemented at nine host sites plus sixteen Triton kernels and a native-only skip
would make fused and native disagree. This file pins:

* the predicate itself, and that every host site routes through it;
* that fan-in >= 2 is untouched, bit for bit (the no-regression half);
* that fan-in-1 params actually TRAIN, for every GC-capable optimizer, on both routes,
  at every momentum dtype;
* fused<->native parity on ``(out, 1)`` / ``(out, 1, 1, 1)`` for Adakaon and AdaPNM,
  across the one-block, lone-big and batched-big fused routes — parity is what a
  half-applied fix breaks, and it was *trivially* green before the fix (both routes
  froze the param), so it only has teeth next to the "trains" tests above.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import warnings

import pytest
import torch

import kaon
from kaon._backend import centralize_grads_, gc_applies

pytestmark = pytest.mark.filterwarnings("ignore")

_MOMENTUM_DTYPES = ("float32", "bfloat16", "int8", "4bit")

# ndim>=2 shapes whose fan-in (numel // shape[0]) is exactly 1 — what GC used to zero.
_FANIN1_SHAPES = [(5, 1), (1, 1), (129, 1), (5, 1, 1, 1), (1, 1, 1, 1)]

# GC applies to these: real fan-in, ndim>=2.
_HEALTHY_SHAPES = [(1, 5), (5, 2), (4, 3), (5, 2, 1, 1), (3, 1, 2, 2), (6, 1, 1, 3)]

# GC never touched these (ndim<2) and still must not.
_LOW_RANK_SHAPES = [(), (1,), (5,)]

# Every optimizer that ships GC on by default.
_GC_DEFAULT_ON = ["Adakaon", "AdaBelief", "AdamP", "AdaPNM", "KProdigy", "Lion", "ADOPT",
                  "ScheduleFree"]
# ...plus the two that accept it but default it off / delegate it.
_GC_OPT_IN = ["AdaMuon", "Nekaon"]


# --------------------------------------------------------------- the predicate itself
@pytest.mark.parametrize("shape", _FANIN1_SHAPES + _LOW_RANK_SHAPES)
def test_gc_applies_is_false_where_gc_is_meaningless(shape) -> None:
    """ndim<2 (no fan-in dims) and fan-in==1 (a one-element row) are both skips."""
    assert gc_applies(shape) is False


@pytest.mark.parametrize("shape", _HEALTHY_SHAPES)
def test_gc_applies_is_true_for_real_fan_in(shape) -> None:
    assert gc_applies(shape) is True


def test_gc_applies_skips_empty_fan_in() -> None:
    """A zero-width fan-in falls on the skip side of the same ``fan_in > 1`` comparison.

    Nothing stronger is claimed, and nothing stronger is assertable: the pre-fix code was
    already harmless on ``(5, 0)`` because the NaN mean it computed was subtracted into a
    zero-element destination and wrote nothing. Any assertion about that grad's *values*
    is vacuously true on an empty tensor, so what is pinned is the predicate plus the fact
    that an empty weight still steps through ``centralize_grads_`` without raising.
    """
    assert gc_applies((5, 0)) is False
    assert gc_applies((0, 3)) is True         # fan-in 3; empty only in the output dim
    p = torch.nn.Parameter(torch.zeros(5, 0))
    p.grad = torch.zeros(5, 0)
    centralize_grads_([p])
    assert p.grad.numel() == 0


def test_gc_applies_accepts_torch_size_and_tuple() -> None:
    assert gc_applies(torch.empty(4, 1).shape) is False
    assert gc_applies(torch.empty(4, 3).shape) is True


def test_bucket_gc_ok_refuses_a_mixed_bucket() -> None:
    """A bucket whose members disagree on the predicate has NO correct constexpr.

    ``GC`` is one ``tl.constexpr`` per launch, so a mixed bucket would either freeze its
    fan-in-1 members or strip GC from its healthy ones. No bucket key in use produces one
    (the big routes key on exact shape; the one-block routes key on the padded tile, where
    ``BC == 1`` iff ``C == 1``), so this can only be reached by a future change to a bucket
    key — which is exactly why it must raise instead of silently picking a side. Called
    directly, because no supported configuration can build such a bucket.
    """
    import kaon._fused_triton as ft

    assert ft.bucket_gc_ok([torch.zeros(5, 2), torch.zeros(5, 3)]) is True
    assert ft.bucket_gc_ok([torch.zeros(5, 1), torch.zeros(9, 1, 1, 1)]) is False
    with pytest.raises(RuntimeError, match="Gradient Centralization"):
        ft.bucket_gc_ok([torch.zeros(5, 1), torch.zeros(5, 2)])
    with pytest.raises(RuntimeError, match="gc_applies"):
        ft.bucket_gc_ok([torch.zeros(5, 2), torch.zeros(5, 1, 1, 1)])


def _code_lines(fn) -> list[str]:
    """``fn``'s EXECUTABLE source, via ``ast.unparse`` — no comments, no docstring.

    Both would make every check below pass on a mention rather than a use, and the fix's
    own prose names ``gc_applies`` at sites that (correctly) receive the resolved flag
    from a caller instead of computing it. Round-tripping through the AST is what makes
    the distinction reliable.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    body = tree.body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return [ln for stmt in body for ln in ast.unparse(stmt).splitlines()]


def test_gc_applies_is_the_only_definition() -> None:
    """Every host-side GC decision must resolve through ``gc_applies``.

    GC lives at nine host sites feeding sixteen Triton kernels; a fix applied to fewer
    than all of them makes fused and native diverge, which is exactly why the bug shipped
    as a known issue instead of a one-line patch. Each site is checked for the *mechanism*
    it is supposed to use, over code with comments stripped, so a future site that
    open-codes ``C > 1`` or ``numel // shape[0]`` fails here.
    """
    import kaon._fused_triton as ft
    from kaon import _backend
    from kaon.adakaon import Adakaon
    from kaon.adapnm import AdaPNM

    # site -> a token that must appear in its CODE (not its comments)
    sites = {
        # computes the predicate itself
        _backend.centralize_grads_: "gc_applies(",
        Adakaon._chunked_reductions: "gc_applies(",
        AdaPNM._chunked_reductions: "gc_applies(",
        # reads the per-bucket flag the cache resolved
        Adakaon._fused_one_block: "bk['gc_ok']",
        AdaPNM._fused_one_block: "bk['gc_ok']",
        Adakaon._chunked_step_batched: "cache.gc_ok",
        AdaPNM._chunked_step_batched: "cache.gc_ok",
        # the gc_ok flags themselves
        ft.bucket_gc_ok: "gc_applies(",
        ft.PointerArrayCache.__init__: "bucket_gc_ok(",
        ft.AdaPnmCache.__init__: "bucket_gc_ok(",
        ft.BigPointerCache.__init__: "bucket_gc_ok(",
        ft.BigPnmCache.__init__: "bucket_gc_ok(",
    }
    missing = [
        f"{fn.__qualname__} (expected {token!r} in its code)"
        for fn, token in sites.items()
        if not any(token in line for line in _code_lines(fn))
    ]
    assert not missing, "GC sites not routed through gc_applies: " + "; ".join(missing)

    # The sites that are HANDED the resolved flag must not re-read the group dict for it:
    # the batched-big reductions and the mom/apply kernels have to agree bit for bit (they
    # share the kernel-written ``rowmean``), so exactly one place may decide.
    for fn in (Adakaon._chunked_reductions_batched, Adakaon._chunked_reductions_fused,
               Adakaon._chunked_step_batched_nomom, AdaPNM._chunked_reductions_batched,
               AdaPNM._chunked_reductions_fused):
        assert "gc" in inspect.signature(fn).parameters, fn.__qualname__
        lines = _code_lines(fn)
        # The only legitimate read left is the guard in front of a ``centralize_grads_``
        # native fallback, which applies the predicate itself.
        leaked = [
            ln for i, ln in enumerate(lines)
            if "group['gradient_centralization']" in ln
            and "centralize_grads_" not in "".join(lines[i:i + 2])
        ]
        assert not leaked, f"{fn.__qualname__} re-derives GC from the group: {leaked}"


def test_no_gc_launch_uses_the_raw_group_flag() -> None:
    """No ``GC=`` kernel argument may be the raw ``gradient_centralization`` flag.

    The Triton kernels take GC as a ``tl.constexpr``, so this is the last line where the
    shape predicate can still be applied. Whitelists the two shapes the resolved flag can
    take — a local already ANDed with the cache's ``gc_ok``, or an explicit per-bucket AND.
    """
    import pathlib
    root = pathlib.Path(kaon.__file__).parent
    allowed = ("gc,", 'gc and bk["gc_ok"],')
    offenders, launch_sites = [], 0
    for path in (root / "adakaon.py", root / "adapnm.py"):   # the only GC launch sites
        for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.split("#", 1)[0]
            if "GC=" not in line or "``" in line:            # skip prose
                continue
            launch_sites += 1
            if not line.split("GC=", 1)[1].startswith(allowed):
                offenders.append(f"{path.name}:{i}: {raw.strip()}")
    assert launch_sites >= 14, f"only found {launch_sites} GC launch args — scan drifted"
    assert not offenders, "GC launch args not resolved per bucket:\n" + "\n".join(offenders)


# ----------------------------------------------------- centralize_grads_: what changed
def _reference_centralize(g: torch.Tensor) -> torch.Tensor:
    """The pre-fix expression, verbatim — the bit-identity reference for fan-in >= 2."""
    return g - g.mean(dim=tuple(range(1, g.ndim)), keepdim=True)


@pytest.mark.parametrize("shape", _FANIN1_SHAPES + _LOW_RANK_SHAPES)
@pytest.mark.parametrize("n", [1, 3])
def test_centralize_leaves_fanin_1_bit_untouched(shape, n) -> None:
    """A skipped grad must come back BITWISE unchanged (n=3 also covers the stacked route)."""
    gen = torch.Generator().manual_seed(4)
    params = [torch.nn.Parameter(torch.randn(shape, generator=gen)) for _ in range(n)]
    for p in params:
        p.grad = torch.randn(shape, generator=gen)
    before = [p.grad.clone() for p in params]
    centralize_grads_(params)
    for p, g0 in zip(params, before, strict=True):
        assert torch.equal(p.grad.view(torch.int32), g0.view(torch.int32))
    if shape in _FANIN1_SHAPES:
        assert all(torch.count_nonzero(p.grad) > 0 for p in params)


@pytest.mark.parametrize("shape", _HEALTHY_SHAPES)
@pytest.mark.parametrize("n", [1, 2, 5])
def test_centralize_fanin_ge2_is_bit_identical(shape, n) -> None:
    """The no-regression half: GC on a real fan-in is unchanged, bit for bit.

    ``n`` spans the lone-shape in-place branch and the stacked branch, because the fix
    filters at the BUCKET level and a mistake there would reshuffle the stack.
    """
    gen = torch.Generator().manual_seed(9)
    params = [torch.nn.Parameter(torch.randn(shape, generator=gen)) for _ in range(n)]
    for p in params:
        p.grad = torch.randn(shape, generator=gen)
    want = [_reference_centralize(p.grad) for p in params]
    centralize_grads_(params)
    for p, w in zip(params, want, strict=True):
        assert torch.equal(p.grad.view(torch.int32), w.view(torch.int32))


def test_centralize_mixed_bag_does_not_disturb_the_healthy_shapes() -> None:
    """Fan-in-1 params dropped from the plan must not change any other bucket's result."""
    gen = torch.Generator().manual_seed(11)
    shapes = [(5, 1), (5, 2), (5, 1), (5, 2), (1, 1), (4, 3), (3,), (5, 1, 1, 1), (3, 1, 2, 2)]
    params = [torch.nn.Parameter(torch.randn(s, generator=gen)) for s in shapes]
    for p in params:
        p.grad = torch.randn(p.shape, generator=gen)
    want = [
        p.grad.clone() if not gc_applies(p.shape) else _reference_centralize(p.grad)
        for p in params
    ]
    centralize_grads_(params)
    for p, w in zip(params, want, strict=True):
        assert torch.equal(p.grad.view(torch.int32), w.view(torch.int32)), tuple(p.shape)


# ------------------------------------------------------- fan-in 1 actually trains
def _bag(shape, n=3, seed=7, device="cpu", dtype=torch.float32):
    gen = torch.Generator().manual_seed(seed)
    return [
        torch.nn.Parameter(torch.randn(shape, generator=gen).to(device=device, dtype=dtype))
        for _ in range(n)
    ]


def _drive(opt, params, steps=6, seed=500, scale=0.07) -> None:
    dev = params[0].device
    gen = torch.Generator(device=dev).manual_seed(seed)
    for _ in range(steps):
        for p in params:
            p.grad = torch.randn(p.shape, generator=gen, device=dev).mul_(scale).to(p.dtype)
        opt.step()


def _moved(name, shape, *, gc, device="cpu", **kw) -> list[bool]:
    """Which of a 3-param fan-in-1 bag actually moved over 6 steps."""
    torch.manual_seed(23)
    kaon.reseed_stochastic_rounding()
    params = _bag(shape, 3, device=device)
    before = [p.detach().clone() for p in params]
    cls = getattr(kaon, name)
    opt = cls(params, lr=1e-2, weight_decay=0.0, gradient_centralization=gc, **kw)
    _drive(opt, params)
    assert all(torch.isfinite(p).all() for p in params), f"{name} {shape} went non-finite"
    return [not torch.equal(p.detach(), p0) for p, p0 in zip(params, before, strict=True)]


@pytest.mark.parametrize("name", _GC_DEFAULT_ON + _GC_OPT_IN)
@pytest.mark.parametrize("shape", [(5, 1), (1, 1), (5, 1, 1, 1)])
@pytest.mark.parametrize("foreach", [True, False])
@pytest.mark.parametrize("momentum_dtype", _MOMENTUM_DTYPES)
def test_fanin_1_trains_with_gc_native(name, shape, foreach, momentum_dtype) -> None:
    """With GC on, a fan-in-1 weight must move exactly as much as it does with GC off.

    The reference is the SAME run with ``gradient_centralization=False``, not "it moves",
    because a couple of optimizer/shape pairs are degenerate for reasons that have nothing
    to do with GC: ``AdaMuon`` on a ``(1, 1)`` weight leaves one of the three params
    untouched with GC on *and* off (measured — its Newton-Schulz orthogonalization on a
    1x1 matrix), and hard-coding "all three move" would have blamed GC for it. The control
    also has to move *something*, so a fix that froze both runs cannot pass.
    """
    kw = dict(foreach=foreach, momentum_dtype=momentum_dtype)
    on = _moved(name, shape, gc=True, **kw)
    off = _moved(name, shape, gc=False, **kw)
    assert any(off), f"{name} {shape}: the GC-off control froze too — bad test fixture"
    assert on == off, (
        f"{name} {shape}: GC changes which params train (gc-on {on} vs gc-off {off})"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("name", ["Adakaon", "AdaPNM"])
@pytest.mark.parametrize("shape", [(5, 1), (1, 1), (5, 1, 1, 1), (129, 1), (9000, 1)])
@pytest.mark.parametrize("momentum_dtype", _MOMENTUM_DTYPES)
def test_fanin_1_trains_with_gc_fused(name, shape, momentum_dtype) -> None:
    """Same, on the Triton routes. ``(9000, 1)`` is above TILE_CAP -> the big/chunked path."""
    on = _moved(name, shape, gc=True, device="cuda", fused=True,
                momentum_dtype=momentum_dtype)
    off = _moved(name, shape, gc=False, device="cuda", fused=True,
                 momentum_dtype=momentum_dtype)
    assert any(off), f"{name} {shape}: the GC-off control froze too — bad test fixture"
    assert on == off, (
        f"{name} fused {shape}: GC changes which params train ({on} vs {off})"
    )


def _gc_on_off_run(name, *, gc, foreach, shape=(6, 1), steps=6):
    cls = getattr(kaon, name)
    params = _bag(shape, 3, seed=41)
    opt = cls(params, lr=1e-2, weight_decay=0.013, foreach=foreach,
              momentum_dtype="float32", gradient_centralization=gc)
    gen = torch.Generator().manual_seed(31)
    for _ in range(steps):
        for p in params:
            p.grad = torch.randn(p.shape, generator=gen) * 0.07
        opt.step()
    return [p.detach().clone() for p in params]


@pytest.mark.parametrize("name", _GC_DEFAULT_ON)
@pytest.mark.parametrize("foreach", [True, False])
def test_fanin_1_matches_gc_off_native(name, foreach) -> None:
    """GC-on and GC-off must be the SAME run, BIT FOR BIT, for a fan-in-1 param.

    Not merely "it moves": skipping GC has to be a true no-op there, i.e. the optimizer sees
    the raw gradient. This is the discriminating test against a fix that, say, centralized
    over dim 0 instead, or that scaled the gradient rather than leaving it alone.

    ``_gc_on_off_run`` pins ``momentum_dtype="float32"``, which makes bit equality the right
    assertion for all eight optimizers including ``ScheduleFree``: its run-to-run
    irreproducibility (noted under Known in the CHANGELOG) comes from the *default*
    ``bfloat16`` momentum, whose stochastically-rounded ``z`` writes draw from a global noise
    stream that is not reseeded between runs — measured at fp32/int8/4bit momentum the repeat
    noise floor is exactly 0, and only ``bfloat16`` moves (7.8e-3 on this bag). The repeat run
    below asserts that floor in-test rather than trusting this paragraph.
    """
    kw = dict(foreach=foreach)
    on = _gc_on_off_run(name, gc=True, **kw)
    off = _gc_on_off_run(name, gc=False, **kw)
    off2 = _gc_on_off_run(name, gc=False, **kw)
    for a, b in zip(off, off2, strict=True):
        assert torch.equal(a, b), f"{name}: not reproducible at fp32 momentum — bad fixture"
    for a, b in zip(on, off, strict=True):
        assert torch.equal(a, b), f"{name}: GC still perturbs a fan-in-1 param"


# ------------------------------------------------------- fused <-> native parity
_PARITY_SHAPES = [(5, 1), (1, 1), (129, 1), (5, 1, 1, 1), (64, 1, 1, 1)]


def _run(cls, shapes, *, device, seed=17, steps=5, **kw):
    gen = torch.Generator().manual_seed(seed)
    params = [
        torch.nn.Parameter(torch.randn(s, generator=gen).to(device)) for s in shapes
    ]
    torch.manual_seed(5)
    kaon.reseed_stochastic_rounding()
    opt = cls(params, lr=1e-2, **kw)
    ggen = torch.Generator().manual_seed(77)
    for _ in range(steps):
        for p in params:
            p.grad = (torch.randn(p.shape, generator=ggen) * 0.07).to(device)
        opt.step()
    return params, opt


def _assert_same(a_params, a_opt, b_params, b_opt, *, atol, rtol, label) -> None:
    for pa, pb in zip(a_params, b_params, strict=True):
        torch.testing.assert_close(
            pa.detach().float().cpu(), pb.detach().float().cpu(),
            atol=atol, rtol=rtol, msg=lambda m, s=tuple(pa.shape): f"{label} {s}: {m}",
        )
        sa, sb = a_opt.state[pa], b_opt.state[pb]
        for k in sorted(set(sa) & set(sb)):
            if torch.is_tensor(sa[k]) and sa[k].is_floating_point():
                torch.testing.assert_close(
                    sa[k].float().cpu(), sb[k].float().cpu(), atol=atol, rtol=rtol,
                    msg=lambda m, s=tuple(pa.shape), k=k: f"{label} {s} state[{k}]: {m}",
                )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("name", ["Adakaon", "AdaPNM"])
@pytest.mark.parametrize("momentum_dtype", _MOMENTUM_DTYPES)
@pytest.mark.parametrize("weight_decay", [0.0, 0.013])
def test_fused_matches_native_on_fanin_1_one_block(name, momentum_dtype, weight_decay) -> None:
    """One-block fused route vs native on the fan-in-1 shapes, incl. repeats so the
    tile buckets really hold >= 2 tensors."""
    cls = getattr(kaon, name)
    shapes = _PARITY_SHAPES + _PARITY_SHAPES + [(5, 2), (7, 3)]
    kw = dict(weight_decay=weight_decay, momentum_dtype=momentum_dtype, bf16_method="none")
    fp, fo = _run(cls, shapes, device="cuda", fused=True, **kw)
    np_, no = _run(cls, shapes, device="cuda", fused=False, **kw)
    _assert_same(fp, fo, np_, no, atol=2e-5, rtol=2e-4, label=f"{name} one-block")


def _worst_parity(cls, shapes, *, fanin, **kw) -> float:
    """Max |fused - native| over weights AND float state, for ``shapes`` widened to ``fanin``."""
    wide = [(s[0], fanin) + tuple(s[2:]) for s in shapes]
    fp, fo = _run(cls, wide, device="cuda", fused=True, **kw)
    np_, no = _run(cls, wide, device="cuda", fused=False, **kw)
    worst = 0.0
    for pa, pb in zip(fp, np_, strict=True):
        worst = max(worst, float((pa.detach() - pb.detach()).abs().max()))
        sa, sb = fo.state[pa], no.state[pb]
        for k in sorted(set(sa) & set(sb)):
            if torch.is_tensor(sa[k]) and sa[k].is_floating_point():
                worst = max(worst, float((sa[k].float() - sb[k].float()).abs().max()))
    return worst


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("name", ["Adakaon", "AdaPNM"])
@pytest.mark.parametrize("momentum_dtype", ["float32", "bfloat16"])
def test_fused_matches_native_on_fanin_1_big(name, momentum_dtype) -> None:
    """The >TILE_CAP ("big") fan-in-1 route: a LONE tensor (per-tensor chunked kernel) and a
    same-shape BUCKET (the batched chunked kernel), each against native.

    Judged AGAINST A CONTROL rather than a hand-picked tolerance. The big routes do not
    reproduce native to the bit at every momentum dtype and never did: with ``bfloat16``
    momentum the two disagree by ~5e-5 (Adakaon) / ~2e-4 (AdaPNM) on ``(9000, C)`` weights,
    and that is the SAME divergence for ``C == 1`` and ``C == 3`` — measured on this branch
    and identical on ea46330 for the healthy fan-ins. A fixed atol here would either have to
    swallow that (and stop discriminating) or fail on a difference GC has nothing to do with.
    So the assertion is: the fan-in-1 route's divergence is no worse than the fan-in-3
    route's, which is exactly the claim "GC's per-bucket skip did not degrade parity".
    """
    cls = getattr(kaon, name)
    kw = dict(weight_decay=0.011, momentum_dtype=momentum_dtype, bf16_method="none")
    if name == "Adakaon":
        kw["deterministic_reductions"] = True
    for shapes, label in (([(9000, 1)], "lone-big"),
                          ([(9000, 1), (9000, 1), (12000, 1)], "batched-big")):
        degenerate = _worst_parity(cls, shapes, fanin=1, **kw)
        control = _worst_parity(cls, shapes, fanin=3, **kw)
        assert degenerate <= max(4.0 * control, 1e-6), (
            f"{name} {label} {momentum_dtype}: fan-in-1 fused<->native divergence "
            f"{degenerate:.3e} exceeds the fan-in-3 control {control:.3e}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("name", ["Adakaon", "AdaPNM"])
def test_fused_mixed_fanin_bag_matches_native(name) -> None:
    """A realistic mixed bag — fan-in-1, healthy 2-D, convs, 1-D, big — in ONE group.

    This is what a per-bucket predicate has to get right: the same group holds buckets
    where GC applies and buckets where it does not, and the one-block tile buckets are
    keyed on the PADDED tile, not the exact shape.
    """
    cls = getattr(kaon, name)
    shapes = [(5, 1), (5, 1), (5, 2), (5, 3), (5, 4), (8, 1, 1, 1), (8, 2, 1, 1),
              (1, 1), (129, 1), (16,), (), (9000, 1), (9000, 1), (300, 40)]
    kw = dict(weight_decay=0.007, momentum_dtype="float32", bf16_method="none")
    extra = dict(deterministic_reductions=True) if name == "Adakaon" else {}
    fp, fo = _run(cls, shapes, device="cuda", fused=True, **kw, **extra)
    np_, no = _run(cls, shapes, device="cuda", fused=False, **kw)
    _assert_same(fp, fo, np_, no, atol=3e-5, rtol=3e-4, label=f"{name} mixed")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("name", ["Adakaon", "AdaPNM"])
def test_fused_fanin_1_matches_gc_off(name) -> None:
    """Fused, GC on vs GC off, on fan-in-1 shapes: the skip must be a true no-op there."""
    cls = getattr(kaon, name)
    shapes = [(5, 1), (129, 1), (5, 1, 1, 1), (9000, 1)]
    kw = dict(weight_decay=0.011, momentum_dtype="float32", bf16_method="none", fused=True)
    extra = dict(deterministic_reductions=True) if name == "Adakaon" else {}
    on, oon = _run(cls, shapes, device="cuda", gradient_centralization=True, **kw, **extra)
    off, ooff = _run(cls, shapes, device="cuda", gradient_centralization=False, **kw, **extra)
    _assert_same(on, oon, off, ooff, atol=0, rtol=0, label=f"{name} fused gc-on vs gc-off")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flipping_gc_midrun_rebuilds_only_the_buckets_it_moves() -> None:
    """A scheduler flipping ``gradient_centralization`` mid-run must rebuild the caches
    whose EFFECTIVE flag moved — and only those.

    ``BigPointerCache`` aliases ``rowmean`` onto ``rowsum`` when GC is off, so its validity
    includes the flag; the caller compares against ``group_flag and cache.gc_ok``. That
    makes a fan-in-1 bucket's cache stable across the flip (its effective flag is already
    ``False`` and cannot change), while a healthy bucket's must be rebuilt. Both halves are
    asserted, because getting the second one wrong is a silent 1.1e-3 divergence and getting
    the first one wrong would rebuild the pointer arrays of every degenerate bucket on every
    flip. ``tests/test_fused_safety.py::test_big_bucket_memo_is_invalidated[gc_flipped]``
    covers the healthy-only bag; this adds the mixed one.
    """
    shapes = [(512, 512), (512, 512), (9000, 1), (9000, 1)]
    cfg = dict(lr=1e-3, weight_decay=0.011, momentum_dtype="float32", bf16_method="none")
    gen = torch.Generator().manual_seed(211)
    pv = [torch.nn.Parameter(torch.randn(s, generator=gen).cuda()) for s in shapes]
    pn = [torch.nn.Parameter(p.detach().clone()) for p in pv]
    ov = kaon.Adakaon(pv, fused=True, deterministic_reductions=True, **cfg)
    on = kaon.Adakaon(pn, **cfg)

    def drive(steps):
        g = torch.Generator().manual_seed(223)
        for _ in range(steps):
            for a, b in zip(pv, pn, strict=True):
                grad = (torch.randn(a.shape, generator=g) * 0.05).cuda()
                a.grad, b.grad = grad, grad.clone()
            ov.step()
            on.step()

    drive(2)
    warm = dict(ov._fused_big_caches)
    healthy = [k for k in warm if k[1] == (512, 512)]
    degenerate = [k for k in warm if k[1] == (9000, 1)]
    assert len(healthy) == 1 and len(degenerate) == 1, list(warm)
    assert warm[healthy[0]].gc_ok is True
    assert warm[degenerate[0]].gc_ok is False
    assert warm[healthy[0]].gc is True and warm[degenerate[0]].gc is False

    # STEADY STATE, GC still on: neither cache may be rebuilt. This is the cost half —
    # comparing the cache's stored flag against the raw group flag instead of the effective
    # one leaves the fan-in-1 bucket permanently mismatched (stored False, group True) and
    # reallocates its pointer arrays and scratch on every single step.
    drive(3)
    before = dict(ov._fused_big_caches)
    for key, label in ((healthy[0], "GC-applicable"), (degenerate[0], "fan-in-1")):
        assert before[key] is warm[key], (
            f"the {label} big bucket's cache is rebuilt every step with GC unchanged"
        )

    for g in ov.param_groups + on.param_groups:      # the mid-run flip
        g["gradient_centralization"] = False
    drive(3)

    after = ov._fused_big_caches
    assert after[healthy[0]] is not before[healthy[0]], (
        "the GC-applicable bucket kept a cache whose rowmean alias is now wrong"
    )
    assert after[degenerate[0]] is before[degenerate[0]], (
        "the fan-in-1 bucket was rebuilt although its effective GC flag never moved"
    )
    assert after[healthy[0]].gc is False
    d = max(float((a.detach() - b.detach()).abs().max())
            for a, b in zip(pv, pn, strict=True))
    assert d < 1e-4, f"fused drifted from native across the flip: {d:.2e}"


# ------------------------------------------------------ does GC still help fan-in >= 2?
def test_gc_still_centralizes_a_real_conv_bucket() -> None:
    """Sanity that the skip is narrow: a 1x1 conv with 2 input channels is still GC'd."""
    p = torch.nn.Parameter(torch.randn(4, 2, 1, 1))
    p.grad = torch.randn(4, 2, 1, 1)
    centralize_grads_([p])
    assert torch.allclose(
        p.grad.sum(dim=(1, 2, 3)), torch.zeros(4), atol=1e-6
    ), "GC no longer zero-means a real fan-in"


# ---------------------------------------------------------- synthetic quality check
def _fit_rank1_up(name: str, gc: bool, *, steps=600, seed=0) -> tuple[float, bool]:
    """LoRA-shaped least squares over the ``(hidden, 1)`` up-projection alone.

    ``y = h @ up_true.T`` with ``h = x @ down.T`` and ``down`` a FIXED ``(1, d)`` buffer, so
    the only trainable weight is the rank-1 up-projection ``up`` — the exact shape GC used to
    freeze. The problem is then convex in ``up`` and the GC-off arm drives the loss to ~0,
    which is what gives the GC-on arm something to be measured against.

    ``down`` is deliberately NOT a parameter: GC on a single-row ``(1, d)`` weight constrains
    its update to the zero-mean subspace, which is GC working as designed (measured: it alone
    plateaus the joint fit at ~1.4 for AdaBelief) and would drown out the effect under test.

    Returns ``(final loss, whether ``up`` moved at all)``.
    """
    g = torch.Generator().manual_seed(seed)
    n, d, hidden = 512, 8, 6
    x = torch.randn(n, d, generator=g)
    down = torch.randn(1, d, generator=g)                                 # fixed projection
    h = x @ down.t()                                                      # [n, 1]
    up_true = torch.randn(hidden, 1, generator=g)
    y = h @ up_true.t()
    up = torch.nn.Parameter(torch.randn(hidden, 1, generator=g) * 0.3)    # (hidden, 1): fan-in 1
    up0 = up.detach().clone()
    opt = getattr(kaon, name)([up], lr=0.02, weight_decay=0.0, gradient_centralization=gc)
    loss = torch.tensor(float("nan"))
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = ((h @ up.t()) - y).pow(2).mean()
        loss.backward()
        opt.step()
    return float(loss), not torch.equal(up.detach(), up0)


@pytest.mark.parametrize("name", ["Adakaon", "AdaPNM", "Lion", "AdaBelief", "ADOPT"])
def test_gc_no_longer_costs_the_fanin_1_factor_its_convergence(name) -> None:
    """The quality half: with GC on, the rank-1 up-projection now actually converges.

    There is no dataset in this repo, so this is a synthetic stand-in for the real regime
    (a rank-1 LoRA adapter). Before the fix ``up`` was pinned at its initialization and the
    loss stayed at whatever its random init gave (~7 here) while the GC-off arm reached ~0;
    after it the two arms land in the same neighbourhood — GC is inert on that factor now,
    instead of fatal.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss_on, moved_on = _fit_rank1_up(name, gc=True)
        loss_off, moved_off = _fit_rank1_up(name, gc=False)
    assert moved_off, f"{name}: GC-off control never moved `up` — bad fixture"
    assert moved_on, f"{name}: `up` still frozen with GC on"
    assert loss_off < 0.05, f"{name}: GC-off control did not converge ({loss_off:.4g})"
    assert loss_on < 10.0 * loss_off + 1e-3, (
        f"{name}: GC-on loss {loss_on:.4g} vs GC-off {loss_off:.4g}"
    )
