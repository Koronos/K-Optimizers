"""Tests for the Orphan optimizer (ADOPT fused with koptim's backend).

Orphan implements the ADOPT update (arXiv:2411.02853): normalize the gradient by
the **previous-step** second moment ``v_{t-1}``, take the first-moment EMA of the
normalized gradient, step, then update ``v_t``. The reference below is an
independent numpy reimplementation of that exact ordering; the tests assert
Orphan's fp32 per-param path matches it, that the ``v_{t-1}``-not-``v_t`` ordering
holds, and that the shared-backend features (momentum dtypes, foreach batching,
cautious masking, checkpoint dtype round-trip) behave.
"""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
import torch

from koptim import Orphan


def _ref_adopt_1d(
    p: np.ndarray,
    grads: list[np.ndarray],
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    cautious: bool,
    clip: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent numpy ADOPT for a 1-D (full second moment) param.

    Mirrors Orphan's non-factored per-param path: ADOPT's exact
    ``max(sqrt(v), eps)`` denom clamp, ``step**0.25`` clip, EMA-of-normalized-grad,
    decoupled WD folded into the delta, cautious on ``delta * g``, ``v`` updated
    after the step. Returns ``(new_p, m, v)``.
    """
    m = np.zeros_like(p)
    v = np.zeros_like(p)
    for t, g in enumerate(grads, start=1):
        if t == 1:
            v = g * g  # ADOPT step 1: seed v, no weight step
            continue
        denom = np.maximum(np.sqrt(v), eps)
        normed = g / denom
        if clip:
            c = math.pow(t, 0.25)
            normed = np.clip(normed, -c, c)
        m = m + (1.0 - beta1) * (normed - m)  # lerp(m, normed, 1-beta1)
        delta = m + weight_decay * p
        if cautious:
            mask = (delta * g > 0).astype(delta.dtype)
            denom_c = max(mask.mean(), 1e-8)
            delta = delta * mask / denom_c
        p = p - lr * delta
        v = beta2 * v + (1.0 - beta2) * (g * g)  # AFTER the step
    return p, m, v


# --------------------------------------------------------------------- smoke


def test_construct_and_step():
    """Construct and step on tiny CPU tensors (2-D + 1-D + conv)."""
    params = [
        torch.nn.Parameter(torch.randn(8, 4)),
        torch.nn.Parameter(torch.randn(5)),
        torch.nn.Parameter(torch.randn(3, 2, 3, 3)),
    ]
    opt = Orphan(params, lr=1e-3)
    for _ in range(3):
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
    for p in params:
        assert torch.isfinite(p).all()


def test_invalid_args():
    p = [torch.nn.Parameter(torch.randn(4))]
    with pytest.raises(ValueError):
        Orphan(p, betas=(1.0, 0.9))
    with pytest.raises(ValueError):
        Orphan(p, lr=-1.0)
    with pytest.raises(ValueError):
        Orphan(p, momentum_dtype="fp8")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Orphan(p, clip_lambda="sometimes")  # type: ignore[arg-type]


# ------------------------------------------------------- numpy-reference parity


@pytest.mark.parametrize("cautious", [False, True])
@pytest.mark.parametrize("weight_decay", [0.0, 0.05])
@pytest.mark.parametrize("clip", [False, True])
def test_matches_numpy_reference_1d(cautious, weight_decay, clip):
    """The fp32 non-factored (1-D) path matches an independent numpy ADOPT."""
    lr, b1, b2, eps = 0.01, 0.9, 0.9999, 1e-6
    p = torch.nn.Parameter(torch.randn(40, dtype=torch.float64).float())
    p0 = p.detach().numpy().copy().astype(np.float64)  # capture initial weights
    opt = Orphan(
        [p], lr=lr, betas=(b1, b2), eps=(eps, 1e-3), weight_decay=weight_decay,
        clip_lambda="default" if clip else None,
        momentum_dtype="float32", cautious=cautious, foreach=False,
    )
    gg = torch.Generator().manual_seed(7)
    grads = [torch.randn(40, generator=gg) for _ in range(15)]
    for g in grads:
        p.grad = g.clone()
        opt.step()
    pr, mr, vr = _ref_adopt_1d(
        p0, [g.numpy().astype(np.float64) for g in grads],
        lr, b1, b2, eps, weight_decay, cautious, clip,
    )
    torch.testing.assert_close(
        p.detach().double(), torch.from_numpy(pr), rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        opt.state[p]["m"].double(), torch.from_numpy(mr), rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        opt.state[p]["v"].double(), torch.from_numpy(vr), rtol=1e-5, atol=1e-6
    )


def test_seed_step_takes_no_weight_step():
    """ADOPT step 1 only seeds the second moment; the weights must not move."""
    p = torch.nn.Parameter(torch.randn(6, 5))
    p0 = p.detach().clone()
    opt = Orphan([p], lr=1.0, foreach=False)
    p.grad = torch.randn_like(p)
    opt.step()
    torch.testing.assert_close(p.detach(), p0)  # unchanged on the seed step
    assert opt.state[p]["step"] == 1
    # Second step DOES move the weights.
    p.grad = torch.randn_like(p)
    opt.step()
    assert not torch.allclose(p.detach(), p0)


def test_normalizes_by_v_prev_not_v_current():
    """ADOPT divides by v_{t-1} (excludes the current g), not v_t.

    We construct a 1-D param so v is the full per-coordinate second moment, run two
    steps, and check that on the second step the normalization used v from after
    step 1 (== g1**2) — NOT a v that already folded in g2.
    """
    lr, b1, b2, eps = 0.1, 0.9, 0.9999, 1e-6
    p = torch.nn.Parameter(torch.zeros(3))
    opt = Orphan(
        [p], lr=lr, betas=(b1, b2), eps=(eps, 1e-3),
        clip_lambda=None, cautious=False, momentum_dtype="float32", foreach=False,
    )
    g1 = torch.tensor([2.0, -3.0, 0.5])
    g2 = torch.tensor([1.0, 1.0, 1.0])
    p.grad = g1.clone()
    opt.step()  # seed v = g1**2, no weight step
    v_after_seed = opt.state[p]["v"].clone()
    torch.testing.assert_close(v_after_seed, g1 * g1)
    p.grad = g2.clone()
    opt.step()
    # The update on step 2 must have normalized by v_{t-1} = g1**2 (NOT a v that
    # includes g2). Reconstruct the expected delta and compare the weight move.
    denom = torch.sqrt(g1 * g1).clamp_(min=eps)
    normed = g2 / denom
    m = (1.0 - b1) * normed                  # m starts at 0
    expected_p = -lr * m                      # p started at 0, no WD/cautious
    torch.testing.assert_close(p.detach(), expected_p, rtol=1e-5, atol=1e-6)
    # And v has now advanced to include g2.
    expected_v = b2 * (g1 * g1) + (1.0 - b2) * (g2 * g2)
    torch.testing.assert_close(opt.state[p]["v"], expected_v, rtol=1e-5, atol=1e-6)


# ----------------------------------------------------- momentum-dtype variants


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
def test_momentum_dtype_variants_construct_and_step(momentum_dtype):
    """Each momentum dtype constructs, seeds, steps, and stays finite."""
    params = [
        torch.nn.Parameter(torch.randn(8, 6)),
        torch.nn.Parameter(torch.randn(7)),
    ]
    opt = Orphan(params, lr=1e-3, momentum_dtype=momentum_dtype, foreach=False)
    for _ in range(4):
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
    for p in params:
        assert torch.isfinite(p).all()
    # The stored first moment is in the configured layout.
    m = opt.state[params[0]]["m"]
    if momentum_dtype == "bfloat16":
        assert m.dtype == torch.bfloat16
    elif momentum_dtype == "float32":
        assert m.dtype == torch.float32
    elif momentum_dtype == "int8":
        assert m.dtype == torch.int8
    else:
        assert m.dtype == torch.uint8  # nibble-packed


# ------------------------------------------------------ foreach == per-param


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
@pytest.mark.parametrize("cautious", [False, True])
def test_foreach_matches_per_param(momentum_dtype, cautious):
    """The batched path is bit-exact vs the per-parameter path (fp32 weights).

    Weights are fp32 so stochastic rounding is a no-op and the two paths must agree
    exactly. Mixes 2-D, 1-D and conv params so both factored and non-factored
    buckets are exercised.
    """
    def make():
        torch.manual_seed(123)
        return [
            torch.nn.Parameter(torch.randn(8, 6)),
            torch.nn.Parameter(torch.randn(8, 6)),   # same shape -> shares a bucket
            torch.nn.Parameter(torch.randn(5)),
            torch.nn.Parameter(torch.randn(5)),
            torch.nn.Parameter(torch.randn(4, 3, 3, 3)),
        ]

    pa = make()
    pb = make()
    kw = {"lr": 1e-2, "momentum_dtype": momentum_dtype, "cautious": cautious,
          "weight_decay": 0.03}
    oa = Orphan(pa, foreach=False, **kw)
    ob = Orphan(pb, foreach=True, **kw)

    gg = torch.Generator().manual_seed(99)
    for _ in range(6):
        gs = [torch.randn(p.shape, generator=gg) for p in pa]
        for p, g in zip(pa, gs, strict=True):
            p.grad = g.clone()
        for p, g in zip(pb, gs, strict=True):
            p.grad = g.clone()
        oa.step()
        ob.step()

    for a, b in zip(pa, pb, strict=True):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


# -------------------------------------------------------------- cautious


def test_cautious_masks_disagreeing_coords():
    """Cautious zeroes coords where the update disagrees with the gradient sign."""
    lr, b1 = 0.1, 0.9
    p = torch.nn.Parameter(torch.zeros(4))
    opt = Orphan(
        [p], lr=lr, betas=(b1, 0.9999), clip_lambda=None, cautious=True,
        momentum_dtype="float32", foreach=False, eps=(1e-6, 1e-3),
    )
    # Seed (no weight step), then build a strong positive momentum over several
    # steps so m points firmly positive on every coordinate.
    p.grad = torch.ones(4)
    opt.step()  # seed v
    for _ in range(8):
        p.grad = torch.ones(4)
        opt.step()
    assert (opt.state[p]["m"] > 0).all()  # momentum is positive everywhere
    p_before = p.detach().clone()
    # A gradient that disagrees (negative) on coords 0,1 and still agrees on 2,3.
    p.grad = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    opt.step()
    moved = (p.detach() - p_before).abs()
    # Disagreeing coords (m>0 vs g<0) are masked to exactly zero; agreeing coords
    # move -> cautious is actually filtering.
    assert moved[0] == 0 and moved[1] == 0
    assert moved[2] > 0 and moved[3] > 0


def test_cautious_off_moves_all():
    """Without cautious, no coordinate is force-zeroed by the mask."""
    p = torch.nn.Parameter(torch.zeros(4))
    opt = Orphan(
        [p], lr=0.1, clip_lambda=None, cautious=False,
        momentum_dtype="float32", foreach=False,
    )
    p.grad = torch.ones(4)
    opt.step()  # seed
    p.grad = torch.tensor([1.0, -1.0, 1.0, -1.0])
    opt.step()
    # All coords moved (m == (1-b1)*normed, all nonzero).
    assert (p.detach() != 0).all()


# ----------------------------------------------------- checkpoint round-trip


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
def test_checkpoint_dtype_round_trip(momentum_dtype):
    """Resume preserves the quantized momentum dtype and reproduces the next step."""
    def make():
        torch.manual_seed(5)
        return [
            torch.nn.Parameter(torch.randn(8, 6)),
            torch.nn.Parameter(torch.randn(5)),
        ]

    pa = make()
    oa = Orphan(pa, lr=1e-2, momentum_dtype=momentum_dtype, foreach=False)
    gg = torch.Generator().manual_seed(11)
    saved_grads = []
    for _ in range(3):
        gs = [torch.randn(p.shape, generator=gg) for p in pa]
        saved_grads.append(gs)
        for p, g in zip(pa, gs, strict=True):
            p.grad = g.clone()
        oa.step()

    # Serialize and reload into a fresh optimizer/params.
    buf = io.BytesIO()
    torch.save({"opt": oa.state_dict(), "params": [p.detach() for p in pa]}, buf)
    buf.seek(0)
    ckpt = torch.load(buf, weights_only=False)

    pb = [torch.nn.Parameter(w.clone()) for w in ckpt["params"]]
    ob = Orphan(pb, lr=1e-2, momentum_dtype=momentum_dtype, foreach=False)
    ob.load_state_dict(ckpt["opt"])

    # Stored momentum dtype must be preserved (not upcast to fp32).
    for p in pb:
        m = ob.state[p]["m"]
        expected = {
            "bfloat16": torch.bfloat16, "float32": torch.float32,
            "int8": torch.int8, "4bit": torch.uint8,
        }[momentum_dtype]
        assert m.dtype == expected

    # One more identical step on both must agree exactly (fp32 weights).
    gs = [torch.randn(p.shape, generator=gg) for p in pa]
    for p, g in zip(pa, gs, strict=True):
        p.grad = g.clone()
    for p, g in zip(pb, gs, strict=True):
        p.grad = g.clone()
    oa.step()
    ob.step()
    for a, b in zip(pa, pb, strict=True):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)
