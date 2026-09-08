"""ADOPT's factored second moment must not manufacture NaN from a finite gradient.

The reconstruction of the Adafactor factored ``v`` divides the row stats by their own
mean. ``exp_avg_sq_row`` is non-negative, so that mean is zero only when the whole row
vector is zero — which needs ``eps1 == 0``. **ADOPT is the only optimizer in the repo
that passes ``eps1 == 0``** (``adopt.py``, to match the official implementation, which
adds no eps inside ``g ** 2``); every other factored optimizer floors the row/col means
at a positive ``eps1``, so their means are provably positive and they never hit this.
For ADOPT, an all-zero gradient left ``row == 0`` and the reconstruction computed
``0 / 0`` — a NaN that then survived the ``clamp(max=1/eps)`` cap and poisoned ``m``
and the weights. Repo policy: NaNs that come *from the gradient* propagate, but a NaN
the optimizer invents out of a finite gradient is a bug.

The fix makes the degenerate case agree with what the non-factored (1-D) path already
did for ``v == 0``: floor ``denom`` at ``eps``, i.e. cap ``1/sqrt(v)`` at ``1/eps``.
``test_adopt_zero_v_normalizer_matches_nonfactored`` is the proof of that equivalence.

**How this was found (and the separate defect that triggers it).** Gradient
Centralization subtracts the per-output-row mean over the fan-in dims; for a weight
whose fan-in is a single element (a rank-1 LoRA up-projection ``(out, 1)``, a
``(out, 1, 1, 1)`` conv) that mean *is* the element, so the centralized gradient is
identically zero. Every GC-enabled optimizer therefore sees ``g == 0`` on those shapes
and silently freezes the parameter — and ADOPT, alone, turned the freeze into NaN. The
freeze is a real second defect but a *separate* one: fixing it means changing GC in the
Triton kernel and both ``_chunked_reductions_batched`` copies as well as
``_backend.centralize_grads_``, or fused and native diverge (the parity contract in
``docs/FUSED_REDUCTIONS_DESIGN.md`` and the pins in
``tests/test_fused_safety.py::test_degenerate_2d_shapes_match_native``). It is pinned
as-is at the bottom of this file so the follow-up has a witness to flip.
"""

from __future__ import annotations

import warnings

import pytest
import torch

import kaon
from kaon import ADOPT
from kaon._backend import centralize_grads_
from kaon._factored import factored_inv_sqrt_factors

_MOMENTUM_DTYPES = ("float32", "bfloat16", "int8", "4bit")

# 2-D+ shapes whose fan-in (numel // shape[0]) is exactly 1 — the shapes GC zeroes.
_FANIN1_SHAPES = [(5, 1), (1, 1), (129, 1), (5, 1, 1, 1), (1, 1, 1, 1)]

# Shapes with a real fan-in, plus the 0-D / 1-D params GC never touches.
_HEALTHY_SHAPES = [(), (1,), (5,), (1, 5), (5, 2)]


# ------------------------------------------------------- unit: the reconstruction
def test_factored_factors_all_zero_row_saturates_instead_of_nan() -> None:
    """An all-zero factored ``v`` must reconstruct to the ``1/eps`` cap, not NaN.

    ``row == 0`` means "no second-moment signal yet"; the non-factored path maps that
    to ``denom = eps``, i.e. an inverse denominator of exactly ``1/eps``.
    """
    row = torch.zeros(5)
    col = torch.zeros(3)
    r_factor, c_factor = factored_inv_sqrt_factors(row, col)
    inv_denom = r_factor * c_factor
    assert not inv_denom.isnan().any(), f"0/0 leaked into the reconstruction: {inv_denom}"
    cap = 1.0 / 1e-6
    assert torch.equal(inv_denom.clamp(max=cap), torch.full((5, 3), cap))


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


def test_factored_factors_nonzero_unchanged() -> None:
    """The healthy case is untouched: exactly ``rsqrt(row/mean(row)) * rsqrt(col)``."""
    row = torch.tensor([0.25, 1.0, 4.0])
    col = torch.tensor([0.5, 2.0])
    r_factor, c_factor = factored_inv_sqrt_factors(row, col)
    want_r = (row / row.mean()).rsqrt().unsqueeze(-1)
    want_c = col.rsqrt().unsqueeze(-2)
    assert torch.equal(r_factor, want_r)
    assert torch.equal(c_factor, want_c)


def test_factored_factors_propagate_nan_row() -> None:
    """A NaN that came from the gradient still propagates (repo policy)."""
    row = torch.tensor([1.0, float("nan"), 4.0])
    col = torch.tensor([1.0, 2.0])
    r_factor, c_factor = factored_inv_sqrt_factors(row, col)
    assert (r_factor * c_factor).isnan().any()


# --------------------------------------------------------------- ADOPT: no NaN
def _drive(opt, p, steps: int = 6, *, seed: int = 500, scale: float = 0.07) -> None:
    for s in range(steps):
        g = torch.randn(p.shape, generator=torch.Generator().manual_seed(seed + s))
        p.grad = g.mul_(scale)
        opt.step()


def _state_is_finite(opt, p) -> bool:
    return all(
        torch.isfinite(b).all()
        for b in opt.state[p].values()
        if torch.is_tensor(b) and b.is_floating_point()
    )


@pytest.mark.parametrize("shape", _FANIN1_SHAPES)
@pytest.mark.parametrize("foreach", [True, False])
@pytest.mark.parametrize("momentum_dtype", _MOMENTUM_DTYPES)
def test_adopt_fanin_1_stays_finite(shape, foreach, momentum_dtype) -> None:
    """The reported bug: ADOPT NaN'd on ``(R, 1)``-style shapes, on both routes.

    GC zeroes the gradient of a fan-in-1 param, ADOPT's ``eps1 == 0`` leaves the
    factored ``v`` at exactly zero, and the reconstruction divided ``0 / 0``.
    """
    torch.manual_seed(23)
    p = torch.nn.Parameter(torch.randn(shape, generator=torch.Generator().manual_seed(7)))
    opt = ADOPT([p], lr=1e-2, weight_decay=0.0, momentum_dtype=momentum_dtype, foreach=foreach)
    _drive(opt, p)
    assert torch.isfinite(p).all(), f"non-finite param for shape {shape}"
    assert _state_is_finite(opt, p), f"non-finite optimizer state for shape {shape}"


@pytest.mark.parametrize("shape", _FANIN1_SHAPES)
@pytest.mark.parametrize("foreach", [True, False])
@pytest.mark.parametrize("momentum_dtype", _MOMENTUM_DTYPES)
def test_adopt_fanin_1_trains_without_gc(shape, foreach, momentum_dtype) -> None:
    """With GC off, a fan-in-1 param gets a real gradient and must actually train.

    Separates the two defects: this proves ADOPT's *factored* path is healthy on
    these shapes once GC is not nulling the gradient (with GC on the param is frozen
    for every optimizer in the repo — see the pin at the bottom of this file).
    """
    torch.manual_seed(23)
    p = torch.nn.Parameter(torch.randn(shape, generator=torch.Generator().manual_seed(7)))
    p0 = p.detach().clone()
    opt = ADOPT(
        [p], lr=1e-2, weight_decay=0.0, momentum_dtype=momentum_dtype,
        foreach=foreach, gradient_centralization=False,
    )
    _drive(opt, p)
    assert torch.isfinite(p).all()
    assert _state_is_finite(opt, p)
    assert not torch.equal(p.detach(), p0), f"param frozen for shape {shape}"


@pytest.mark.parametrize("shape", _FANIN1_SHAPES + _HEALTHY_SHAPES)
@pytest.mark.parametrize("gradient_centralization", [True, False])
def test_adopt_zero_grad_is_finite(shape, gradient_centralization) -> None:
    """An exactly-zero (finite!) gradient must never make ADOPT emit NaN.

    The reconstruction bug on its own, independent of GC and of the shape: a dead
    branch, a frozen slice or a masked loss term hands the optimizer a genuine
    ``g == 0``, and with ``eps1 == 0`` the row/col stats stay at exactly zero.
    """
    p = torch.nn.Parameter(torch.randn(shape, generator=torch.Generator().manual_seed(7)))
    opt = ADOPT([p], lr=1e-2, gradient_centralization=gradient_centralization)
    for _ in range(4):
        p.grad = torch.zeros(shape)
        opt.step()
    assert torch.isfinite(p).all()
    assert _state_is_finite(opt, p)


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
    p = torch.nn.Parameter(torch.randn(5, 1))
    opt = ADOPT([p], lr=1e-2, foreach=foreach, gradient_centralization=False)
    p.grad = torch.randn(5, 1) * 0.05
    opt.step()
    p.grad = torch.full((5, 1), float("nan"))
    opt.step()
    p.grad = torch.randn(5, 1) * 0.05
    opt.step()
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

    The floor lives in two places — ``kaon._factored`` for the per-param path and an
    inlined copy in ``ADOPT._factored_bucket`` for the batched one — so the routes
    have to be checked against each other, not just for finiteness.
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

    Only ADOPT ever did (``eps1 == 0``); the rest floor the row/col means at a
    positive ``eps1``. Pinned across the whole family so a future optimizer that
    adopts ``eps1 == 0`` cannot reintroduce this quietly.
    """
    cls = getattr(kaon, name)
    torch.manual_seed(23)
    kaon.reseed_stochastic_rounding()
    p = torch.nn.Parameter(torch.randn(shape, generator=torch.Generator().manual_seed(7)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = cls([p], lr=1e-2, weight_decay=0.0, foreach=foreach)
        _drive(opt, p)
    assert torch.isfinite(p).all(), f"{name} produced a non-finite param for {shape}"
    assert _state_is_finite(opt, p), f"{name} produced non-finite state for {shape}"


# ---------------------------------------------------- known limitation (pinned)
@pytest.mark.parametrize("shape", _FANIN1_SHAPES)
def test_gc_zeroes_fanin_1_grads_known_limitation(shape) -> None:
    """**Pinned defect, not desired behaviour.** GC destroys a fan-in-1 gradient.

    ``g - mean_fanin(g)`` over a one-element fan-in is identically zero, so every
    GC-enabled optimizer freezes these params (rank-1 LoRA up-projections among
    them). Skipping them in :func:`kaon._backend.centralize_grads_` alone is *not*
    the fix: the Triton path centralizes in-kernel and both
    ``_chunked_reductions_batched`` copies (``adakaon.py``, ``adapnm.py``) do it in
    torch, so a native-only skip makes fused and native disagree — measured, it
    breaks ``tests/test_fused_safety.py::test_degenerate_2d_shapes_match_native``
    and ``tests/test_adapnm_fused.py::test_pnm_extreme_aspect_shapes_compile_and_match_native``.
    The real fix has to change all four GC sites together; this pin is the witness
    to flip when it does.
    """
    p = torch.nn.Parameter(torch.randn(shape))
    p.grad = torch.randn(shape)
    centralize_grads_([p])
    assert torch.count_nonzero(p.grad) == 0, (
        "GC no longer zeroes fan-in-1 gradients — if that is intentional, the fused "
        "in-kernel GC and both _chunked_reductions_batched copies must change too, and "
        "this test should be replaced by the positive assertion."
    )
