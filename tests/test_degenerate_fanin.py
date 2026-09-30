"""ADOPT's factored second moment must not manufacture NaN from a finite gradient.

The reconstruction of the Adafactor factored ``v`` divides the row stats by their own
mean. ``exp_avg_sq_row`` is non-negative, so that mean is zero only when the whole row
vector is zero — which needs ``eps1 == 0``. **With the shipped defaults ADOPT is the
only optimizer here that reaches that**: it passes ``eps1 == 0`` on purpose
(``adopt.py``, matching the official implementation, which adds no eps inside
``g ** 2``), while every other factored optimizer defaults to a positive ``eps1`` that
floors the row/col means. (It is *not* structurally impossible for the others —
``AdaBelief``, ``AdamP``, ``AdaPNM``, ``ScheduleFree`` and ``Adakaon`` all accept
``eps=0.0`` from the constructor, and then a zero gradient NaNs them too, on 0.7.12 and
on this branch alike. Only ``ADOPT`` and ``KProdigy`` validate ``eps > 0``. Fixing that
class of misconfiguration is a separate job — see the note on ``floor`` below.)

For ADOPT, an all-zero gradient left ``row == 0`` and the reconstruction computed
``0 / 0`` — a NaN that then survived the ``clamp(max=1/eps)`` cap and poisoned ``m``
and the weights. Repo policy: NaNs that come *from the gradient* propagate, but a NaN
the optimizer invents out of a finite gradient is a bug.

**The fix is opt-in, and that is load-bearing.** The floor on the row-mean divisor
lives behind ``factored_inv_sqrt_factors(..., floor=...)``, defaulting to ``0.0`` (no
clamp at all), and **only ADOPT passes it**. A floor turns ``0 / 0`` into
``rsqrt(0) = +inf``, which is only *useful* to a caller that caps the reconstructed
inverse-denominator afterwards — ADOPT and KProdigy do, the other six do not. For a
non-capping caller a subnormal row mean is a legitimate finite update, and flooring the
divisor would move it instead of rescuing a NaN. So the default has to be inert, which
``test_factored_default_floor_is_inert`` pins bit-for-bit against the pre-fix formula.

**How this was found (and the separate defect that triggers it).** Gradient
Centralization subtracts the per-output-row mean over the fan-in dims; for a weight
whose fan-in is a single element (a rank-1 LoRA up-projection ``(out, 1)``, a
``(out, 1, 1, 1)`` conv) that mean *is* the element, so the centralized gradient was
identically zero. Every GC-enabled optimizer therefore saw ``g == 0`` on those shapes
and silently froze the parameter — and ADOPT, alone, turned the freeze into NaN. The
freeze was a real second defect but a *separate* one, fixed in its own batch: GC lives
at nine host sites feeding sixteen Triton kernels and they all had to move together or
fused and native diverge. It is now ONE predicate, ``kaon._backend.gc_applies``,
evaluated per shape/tile bucket; the two tests at the bottom of this file were strict
xfails until then, and ``tests/test_gc_fanin_1.py`` carries the full matrix.
"""

from __future__ import annotations

import warnings

import pytest
import torch

import kaon
from kaon import ADOPT
from kaon._backend import centralize_grads_
from kaon._factored import (
    _MIN_NORMAL,
    factored_inv_sqrt_factors,
    zero_safe_inv_sqrt_factors,
)

_MOMENTUM_DTYPES = ("float32", "bfloat16", "int8", "4bit")

# 2-D+ shapes whose fan-in (numel // shape[0]) is exactly 1 — the shapes GC used to zero.
_FANIN1_SHAPES = [(5, 1), (1, 1), (129, 1), (5, 1, 1, 1), (1, 1, 1, 1)]

# Shapes with a real fan-in, plus the 0-D / 1-D params GC never touches.
_HEALTHY_SHAPES = [(), (1,), (5,), (1, 5), (5, 2)]


# ------------------------------------------------- unit: the reconstruction / floor
def _reference_factors(row: torch.Tensor, col: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The 0.7.12 expression, verbatim — the bit-identity reference for ``floor=0.0``."""
    r_factor = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
    c_factor = col.rsqrt().unsqueeze(-2)
    return r_factor, c_factor


def _row_means_grid() -> list[torch.Tensor]:
    """Row vectors spanning normal down to subnormal magnitudes, plus the odd cases.

    The subnormal end is the whole point: that is the range where a too-large floor
    (``1e-8``, say) would silently reshape a *finite* update, so the default has to be
    provably inert there and not merely "inert for sane values".
    """
    scales = [1e3, 1.0, 1e-8, 1e-20, 1e-30, 1e-38, 5e-39, 1e-42, 1e-45]
    rows = [torch.tensor([1.0, 4.0, 9.0, 0.25]) * s for s in scales]
    rows += [
        torch.tensor([0.0, 0.0, 1e-20, 0.0]),          # zero entries, positive mean
        torch.tensor([-0.0, -0.0, -0.0, -0.0]),        # signed zero -> -inf/nan territory
        torch.tensor([1.0, float("nan"), 4.0, 2.0]),   # gradient NaN must pass through
        torch.tensor([1.0, float("inf"), 4.0, 2.0]),
    ]
    return rows


@pytest.mark.parametrize("idx", range(len(_row_means_grid())))
def test_factored_default_floor_is_inert(idx) -> None:
    """``floor=0.0`` must be bit-identical to the pre-fix expression, everywhere.

    Covers subnormal row means, exact and signed zeros, NaN and inf — i.e. exactly the
    inputs where a floor could change a result. This is what makes the ``floor`` kwarg
    safe to leave defaulted for the six callers that do not cap the reconstruction, and
    it kills any mutant that raises the default above ``0.0``.
    """
    row = _row_means_grid()[idx]
    col = torch.tensor([0.5, 2.0, 1e-30])
    got_r, got_c = factored_inv_sqrt_factors(row.clone(), col.clone())
    want_r, want_c = _reference_factors(row.clone(), col.clone())
    assert got_r.dtype == want_r.dtype and got_r.shape == want_r.shape
    # bitwise, so a NaN payload or a -0.0/+0.0 flip counts as a difference
    assert torch.equal(got_r.view(torch.int32), want_r.view(torch.int32)), (
        f"row={row.tolist()}\n got={got_r.flatten().tolist()}\nwant={want_r.flatten().tolist()}"
    )
    assert torch.equal(got_c.view(torch.int32), want_c.view(torch.int32))


@pytest.mark.parametrize("mean_scale", [1e-20, 1e-30, 1e-37, 1e-38])
def test_factored_floor_does_not_bite_above_min_normal(mean_scale) -> None:
    """``floor=_MIN_NORMAL`` must be inert for any row mean above the smallest normal.

    These means all sit strictly between ``_MIN_NORMAL`` (1.18e-38) and ``1e-8``, so a
    mutant that inflates the floor to ``1e-8`` clamps here and diverges, while the real
    floor does nothing. This is the discriminating case for the floor's *value*.
    """
    row = torch.tensor([1.0, 4.0, 9.0, 0.25]) * mean_scale
    assert _MIN_NORMAL < row.mean().item() < 1e-8
    col = torch.tensor([1.0, 4.0]) * mean_scale
    got_r, _ = factored_inv_sqrt_factors(row.clone(), col.clone(), floor=_MIN_NORMAL)
    want_r, _ = _reference_factors(row.clone(), col.clone())
    assert torch.equal(got_r, want_r), (
        f"floor bit at mean={row.mean().item():.3e}: got={got_r.flatten().tolist()} "
        f"want={want_r.flatten().tolist()}"
    )


def test_factored_floor_saturates_all_zero_row_instead_of_nan() -> None:
    """With ``floor>0`` an all-zero factored ``v`` reconstructs to ``+inf``, not NaN.

    ``+inf`` is what ADOPT's ``clamp(max=1/eps)`` needs in order to land on the same
    denominator the non-factored path uses for ``v == 0``.
    """
    row = torch.zeros(5)
    col = torch.zeros(3)
    r_factor, c_factor = factored_inv_sqrt_factors(row, col, floor=_MIN_NORMAL)
    inv_denom = r_factor * c_factor
    assert not inv_denom.isnan().any(), f"0/0 leaked into the reconstruction: {inv_denom}"
    cap = 1.0 / 1e-6
    assert torch.equal(inv_denom.clamp(max=cap), torch.full((5, 3), cap))


def test_factored_without_floor_still_nans_on_all_zero_row() -> None:
    """The default really is the old behaviour: no floor, no rescue.

    Pinned so nobody "helpfully" turns the floor on by default for the six callers
    that cannot use it (see the module docstring).
    """
    r_factor, c_factor = factored_inv_sqrt_factors(torch.zeros(5), torch.zeros(3))
    assert (r_factor * c_factor).isnan().any()


def test_factored_factors_zero_column_only() -> None:
    """A single all-zero column (row stats still positive) must not produce NaN.

    This half already worked — ``rsqrt(0)`` is ``+inf``, which the caller's cap
    collapses — and is pinned so the fix is not mistaken for having changed it.
    """
    row = torch.tensor([1.0, 4.0, 9.0])
    col = torch.tensor([1.0, 0.0])
    r_factor, c_factor = factored_inv_sqrt_factors(row, col)
    inv_denom = r_factor * c_factor
    assert not inv_denom.isnan().any()
    assert inv_denom[:, 1].isinf().all()  # then capped by the caller's clamp


def test_factored_factors_propagate_nan_row() -> None:
    """A NaN that came from the gradient still propagates, floor or no floor."""
    row = torch.tensor([1.0, float("nan"), 4.0])
    col = torch.tensor([1.0, 2.0])
    for floor in (0.0, _MIN_NORMAL):
        r_factor, c_factor = factored_inv_sqrt_factors(row.clone(), col.clone(), floor=floor)
        assert (r_factor * c_factor).isnan().any()


# ------------------------------- unit: the eps1 == 0 reconstruction of the non-capping callers
# ``zero_safe_inv_sqrt_factors`` is what AdaBelief, AdamP, AdaMuon and ScheduleFree call when
# ``eps1 == 0`` (they do not cap the reconstruction, so ADOPT's floor would only trade the NaN
# for an inf). One helper for the four; these pin its contract.
@pytest.mark.parametrize("idx", range(8))  # the scaled rows down to 1e-42 (all entries > 0)
def test_zero_safe_factors_match_the_plain_ones_on_nonzero_stats(idx) -> None:
    """Every stat > 0 (normal OR subnormal) gets the plain reconstruction, bit for bit — no
    floor reshapes a finite update, so ``eps1 == 0`` stays the exact math wherever the
    math is finite."""
    row = _row_means_grid()[idx]
    col = torch.tensor([0.5, 2.0, 1e-30, 1e-44])
    got_r, got_c = zero_safe_inv_sqrt_factors(row.clone(), col.clone())
    want_r, want_c = _reference_factors(row.clone(), col.clone())
    assert torch.isfinite(want_r).all() and torch.isfinite(want_c).all()
    assert torch.equal(got_r.view(torch.int32), want_r.view(torch.int32))
    assert torch.equal(got_c.view(torch.int32), want_c.view(torch.int32))


def test_zero_safe_factors_zero_stats_give_a_zero_factor() -> None:
    """An exactly-zero row / column statistic gives factor 0 (update 0 there), for an
    all-zero row vector (``0/0`` in the mean) too — never inf, never NaN, and never a
    bounded-but-huge factor: with ``g**2`` underflowed to 0 the first moment can still be
    ~1e-25 (bf16 grads reach it), and a ``rsqrt(_MIN_NORMAL) = 2**63`` per factor made that
    a ~1e13 step."""
    r, c = zero_safe_inv_sqrt_factors(torch.tensor([1.0, 0.0, 4.0]), torch.tensor([0.0, 2.0]))
    assert r.flatten().tolist()[1] == 0.0 and c.flatten().tolist()[0] == 0.0
    assert torch.isfinite(r).all() and torch.isfinite(c).all()
    r, c = zero_safe_inv_sqrt_factors(torch.zeros(2, 5), torch.zeros(2, 3))
    assert r.shape == (2, 5, 1) and c.shape == (2, 1, 3)
    assert not r.any() and not c.any()
    m = torch.full((5, 3), 1e-25)                   # tiny first moment where v underflowed
    assert not (m * r[0] * c[0]).any()


def test_zero_safe_factors_underflowed_row_mean_is_a_zero_row() -> None:
    """Nonzero subnormal row stats whose MEAN rounds to 0: the row/mean ratio is undefined,
    so the row factor is 0 (no step), not ``rsqrt(row)`` against a substitute divisor
    (~1e22: a giant step on a row whose gradient is ~1e-23)."""
    row = torch.tensor([1e-45, 0.0, 0.0, 0.0])
    assert row.mean().item() == 0.0
    r, _ = zero_safe_inv_sqrt_factors(row, torch.ones(2))
    assert not r.any()


def test_zero_safe_factors_propagate_gradient_nan() -> None:
    row = torch.tensor([1.0, float("nan"), 4.0])
    col = torch.tensor([1.0, float("nan")])
    r, c = zero_safe_inv_sqrt_factors(row, col)
    assert r.isnan().all() and c.flatten().isnan().tolist() == [False, True]


# --------------------------------------------------------------- ADOPT: no NaN
def _bag(shape, n: int = 3, seed: int = 7) -> list[torch.nn.Parameter]:
    """``n`` same-shape params.

    Never fewer than 2: ``adopt.py``'s ``if len(fast) >= 2`` sends a lone parameter
    down the per-param loop even with ``foreach=True``, so a single-param bag would
    silently test the same route twice.
    """
    g = torch.Generator().manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(shape, generator=g)) for _ in range(n)]


def _drive(opt, params, steps: int = 6, *, seed: int = 500, scale: float = 0.07) -> None:
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        for p in params:
            p.grad = torch.randn(p.shape, generator=gen).mul_(scale)
        opt.step()


def _all_finite(opt, params) -> bool:
    if not all(torch.isfinite(p).all() for p in params):
        return False
    return all(
        torch.isfinite(b).all()
        for p in params
        for b in opt.state[p].values()
        if torch.is_tensor(b) and b.is_floating_point()
    )


def test_foreach_bag_actually_takes_the_batched_route() -> None:
    """Guard the guard: prove a >=2 bag of fan-in-1 params reaches the batched bucket.

    Without this, every ``foreach=True`` parametrisation below could be silently
    running the per-param code and the inlined floor in ``ADOPT._factored_bucket``
    would be untested.
    """
    looped: list[torch.Tensor] = []
    orig = ADOPT._step_one_param

    def spy(self, p, *args, **kwargs):
        looped.append(p)
        return orig(self, p, *args, **kwargs)

    params = _bag((5, 1), 3)
    opt = ADOPT(params, lr=1e-2, foreach=True)
    ADOPT._step_one_param = spy
    try:
        _drive(opt, params, steps=2)
    finally:
        ADOPT._step_one_param = orig
    assert not looped, f"{len(looped)} params fell back to the per-param loop"


@pytest.mark.parametrize("shape", _FANIN1_SHAPES)
@pytest.mark.parametrize("foreach", [True, False])
@pytest.mark.parametrize("momentum_dtype", _MOMENTUM_DTYPES)
def test_adopt_fanin_1_stays_finite(shape, foreach, momentum_dtype) -> None:
    """The reported bug: ADOPT NaN'd on ``(R, 1)``-style shapes, on both routes.

    GC zeroes the gradient of a fan-in-1 param, ADOPT's ``eps1 == 0`` leaves the
    factored ``v`` at exactly zero, and the reconstruction divided ``0 / 0``. Bags of
    3 so ``foreach=True`` genuinely exercises ``_factored_bucket``.
    """
    torch.manual_seed(23)
    params = _bag(shape, 3)
    opt = ADOPT(
        params, lr=1e-2, weight_decay=0.0, momentum_dtype=momentum_dtype, foreach=foreach
    )
    _drive(opt, params)
    assert _all_finite(opt, params), f"non-finite param or state for shape {shape}"


@pytest.mark.parametrize("shape", _FANIN1_SHAPES)
@pytest.mark.parametrize("foreach", [True, False])
@pytest.mark.parametrize("momentum_dtype", _MOMENTUM_DTYPES)
def test_adopt_fanin_1_trains_without_gc(shape, foreach, momentum_dtype) -> None:
    """With GC off, a fan-in-1 param gets a real gradient and must actually train.

    Separates the two defects: this proves ADOPT's *factored* path is healthy on
    these shapes once GC is not nulling the gradient (before the GC fix, GC on froze it
    for every optimizer in the repo — see the two tests at the bottom of this file).
    """
    torch.manual_seed(23)
    params = _bag(shape, 3)
    before = [p.detach().clone() for p in params]
    opt = ADOPT(
        params, lr=1e-2, weight_decay=0.0, momentum_dtype=momentum_dtype,
        foreach=foreach, gradient_centralization=False,
    )
    _drive(opt, params)
    assert _all_finite(opt, params)
    for p, p0 in zip(params, before, strict=True):
        assert not torch.equal(p.detach(), p0), f"param frozen for shape {shape}"


@pytest.mark.parametrize("shape", _FANIN1_SHAPES + _HEALTHY_SHAPES)
@pytest.mark.parametrize("gradient_centralization", [True, False])
def test_adopt_zero_grad_is_finite(shape, gradient_centralization) -> None:
    """An exactly-zero (finite!) gradient must never make ADOPT emit NaN.

    The reconstruction bug on its own, independent of GC and of the shape: a dead
    branch, a frozen slice or a masked loss term hands the optimizer a genuine
    ``g == 0``, and with ``eps1 == 0`` the row/col stats stay at exactly zero.
    """
    params = _bag(shape, 3)
    opt = ADOPT(params, lr=1e-2, gradient_centralization=gradient_centralization)
    for _ in range(4):
        for p in params:
            p.grad = torch.zeros(shape)
        opt.step()
    assert _all_finite(opt, params)


def test_adopt_zero_v_normalizer_matches_nonfactored() -> None:
    """With ``v == 0`` the factored path must floor exactly like the 1-D path.

    Step 0 sees a zero gradient (so ``v`` is all zeros), step 1 sees a real one. The
    1-D path computes ``g / max(sqrt(0), eps) = g / eps`` and then clips at 1.0; the
    factored path caps ``1/sqrt(v)`` at ``1/eps``, which is the same number. So a
    ``(R, 1)``-shaped run and a flat ``(R,)`` run of the same values must take the
    same step — this is why the fix is a *floor*, not a NaN scrub.
    """
    g1 = torch.tensor([0.03, -0.07, 0.11, -0.002, 0.5])
    init = torch.tensor([0.5, -0.25, 1.0, 0.125, -0.75])

    p2d = torch.nn.Parameter(init.reshape(5, 1).clone())
    o2d = ADOPT([p2d], lr=1e-2, gradient_centralization=False, momentum_dtype="float32")
    p1d = torch.nn.Parameter(init.clone())
    o1d = ADOPT([p1d], lr=1e-2, gradient_centralization=False, momentum_dtype="float32")
    for g in (torch.zeros(5), g1):
        p2d.grad = g.reshape(5, 1).clone()
        p1d.grad = g.clone()
        o2d.step()
        o1d.step()
    torch.testing.assert_close(p2d.detach().reshape(5), p1d.detach(), rtol=0, atol=0)


@pytest.mark.parametrize("foreach", [True, False])
def test_adopt_nan_grad_still_propagates(foreach) -> None:
    """Repo policy: a NaN coming from the gradient is NOT masked by the new floor."""
    params = _bag((5, 1), 3)
    opt = ADOPT(params, lr=1e-2, foreach=foreach, gradient_centralization=False)
    gen = torch.Generator().manual_seed(3)
    for p in params:
        p.grad = torch.randn(p.shape, generator=gen) * 0.05
    opt.step()
    for p in params:
        p.grad = torch.full(p.shape, float("nan"))
    opt.step()
    for p in params:
        p.grad = torch.randn(p.shape, generator=gen) * 0.05
    opt.step()
    for p in params:
        assert p.isnan().any(), "a NaN gradient must reach the weights"


# --------------------------------------------------------------- ADOPT: parity
def _degenerate_bag(seed: int = 11) -> list[torch.nn.Parameter]:
    """Fan-in-1 shapes, with repeats so the batched path really batches."""
    g = torch.Generator().manual_seed(seed)
    shapes = [(5, 1), (5, 1), (1, 1), (1, 1), (129, 1), (5, 1, 1, 1), (1, 1, 1, 1), (1, 5), (5, 2)]
    return [torch.nn.Parameter(torch.randn(s, generator=g)) for s in shapes]


@pytest.mark.parametrize("momentum_dtype", _MOMENTUM_DTYPES)
@pytest.mark.parametrize("weight_decay", [0.0, 0.013])
@pytest.mark.parametrize("gradient_centralization", [True, False])
def test_adopt_foreach_matches_per_param_on_degenerate_shapes(
    momentum_dtype, weight_decay, gradient_centralization
) -> None:
    """foreach and per-param must agree element-for-element on fan-in-1 shapes.

    The floor lives in two places — the ``floor=`` argument ADOPT passes to
    ``kaon._factored`` on the per-param route, and an inlined copy in
    ``ADOPT._factored_bucket`` for the batched one — so the routes have to be checked
    against each other, not just for finiteness.
    """
    pa = _degenerate_bag()
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    cfg = dict(
        lr=1e-2, weight_decay=weight_decay, momentum_dtype=momentum_dtype,
        bf16_method="none", gradient_centralization=gradient_centralization,
    )
    oa = ADOPT(pa, foreach=True, **cfg)
    ob = ADOPT(pb, foreach=False, **cfg)
    gg = torch.Generator().manual_seed(7)
    for _ in range(7):
        for a, b in zip(pa, pb, strict=True):
            grad = torch.randn(a.shape, generator=gg) * 0.05
            a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb, strict=True):
        assert torch.isfinite(a).all()
        assert torch.equal(a, b)


# ------------------------------------------------- every optimizer: finiteness
_ALL_OPTIMIZERS = [
    "Adakaon",
    "AdaBelief",
    "AdamP",
    "AdaPNM",
    "AdaMuon",
    "KProdigy",
    "Lion",
    "ADOPT",
    "ScheduleFree",
    "Nekaon",
]


@pytest.mark.parametrize("name", _ALL_OPTIMIZERS)
@pytest.mark.parametrize("shape", [(5, 1), (1, 1), (5, 1, 1, 1)])
@pytest.mark.parametrize("foreach", [True, False])
def test_no_optimizer_manufactures_nan_on_fanin_1(name, shape, foreach) -> None:
    """No optimizer may turn a finite gradient on a fan-in-1 param into NaN.

    Only ADOPT ever did with the shipped defaults (``eps1 == 0``); the rest default to
    a positive ``eps1`` that floors the row/col means. Pinned across the whole family
    so a future optimizer that adopts ``eps1 == 0`` cannot reintroduce this quietly.
    """
    cls = getattr(kaon, name)
    torch.manual_seed(23)
    kaon.reseed_stochastic_rounding()
    params = _bag(shape, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = cls(params, lr=1e-2, weight_decay=0.0, foreach=foreach)
        _drive(opt, params)
    assert _all_finite(opt, params), f"{name} produced non-finite values for {shape}"


# ------------------------------------------- the GC half of the defect (fixed, pinned)
# GC used to zero every fan-in-1 gradient at nine host sites plus sixteen Triton kernels,
# so a native-only skip made fused and native disagree. It is now ONE predicate,
# ``kaon._backend.gc_applies``, evaluated per shape/tile bucket and handed to the kernels
# as the ``GC`` ``tl.constexpr``. The two tests below were strict xfails until then;
# ``tests/test_gc_fanin_1.py`` carries the full matrix (every optimizer, every route,
# every momentum dtype, fused<->native parity, and fan-in >= 2 bit-identity).


@pytest.mark.parametrize("shape", _FANIN1_SHAPES)
def test_gc_leaves_fanin_1_grads_usable(shape) -> None:
    """GC must not annihilate a fan-in-1 gradient.

    ``g - mean_fanin(g)`` over a one-element fan-in is identically zero, so GC destroyed
    the update signal instead of decorrelating it. GC is skipped there now.
    """
    p = torch.nn.Parameter(torch.randn(shape))
    p.grad = torch.randn(shape)
    centralize_grads_([p])
    assert torch.count_nonzero(p.grad) > 0


@pytest.mark.parametrize("name", ["Adakaon", "AdaPNM"])
@pytest.mark.parametrize("fused", [False, True])
def test_gc_freezes_fanin_1_params(name, fused) -> None:
    """A ``(out, 1)`` weight must train under GC.

    Covers the fused route as well as the native one, because GC is reimplemented
    inside the Triton path and in both ``_chunked_reductions*`` copies — a fix that
    only touched ``centralize_grads_`` would pass this natively and keep failing
    fused, which is exactly the divergence to avoid.
    """
    if fused and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda" if fused else "cpu"
    cls = getattr(kaon, name)
    torch.manual_seed(23)
    kaon.reseed_stochastic_rounding()
    params = [p.to(device) for p in _bag((64, 1), 3)]
    params = [torch.nn.Parameter(p.detach()) for p in params]
    before = [p.detach().clone() for p in params]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = cls(params, lr=1e-2, weight_decay=0.0, fused=fused)
        gen = torch.Generator(device=device).manual_seed(500)
        for _ in range(6):
            for p in params:
                p.grad = torch.randn(p.shape, generator=gen, device=device) * 0.07
            opt.step()
    assert all(
        not torch.equal(p.detach(), p0) for p, p0 in zip(params, before, strict=True)
    )
