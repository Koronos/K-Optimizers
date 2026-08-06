"""Tests for the MSAM climb's numerical contract on low-precision weights.

The removal RECOMPUTES the perturbation instead of restoring a snapshot (that is what
buys MSAM its zero extra state), so the +e / -e pair must round identically in both
directions or the weights random-walk. Measured before the fix: 19% relative L2 drift
after 4000 climb/removal cycles on bf16 weights, growing as sqrt(N); fp32 was exact.

Covers:
  1. the round trip is exact on bf16 as well as fp32, on the torch and Triton paths
  2. the climb is still applied where bf16 can represent it (the fix must not silently
     disable the mechanism)
  3. the inert-lookahead warning fires when the displacement is too small to do
     anything, and stays quiet during an LR warmup and when the mechanism works
  4. a checkpoint saved in train mode is rejected instead of silently baking one
     perturbation into the weights per resume
  5. the fused plan notices a weight whose storage was rebound by something external
"""

from __future__ import annotations

import copy
import warnings

import pytest
import torch

from kaon import MSAM, Nekaon


def _nekaon(params, lr=1e-4, k=1.5, **kw):
    return Nekaon(params, lr=lr, k=k, betas=(0.5, 0.999), weight_decay=0.0,
                  gradient_centralization=False, **kw)


def _spin(opt, params, steps=5, seed=11):
    for _ in range(steps):
        g = torch.Generator(device=params[0].device).manual_seed(seed)
        for p in params:
            p.grad = torch.randn(p.shape, generator=g, dtype=p.dtype, device=p.device)
        opt.step()


def _round_trip_drift(dtype, cycles=200, device="cpu", **kw):
    """Drift of the eval-view weights over `cycles` climb/removal pairs, state frozen."""
    g = torch.Generator(device=device).manual_seed(0)
    p = ((torch.randn(64, 64, generator=g, device=device) * 0.02)
         .to(dtype).requires_grad_(True))
    opt = _nekaon([p], **kw)
    _spin(opt, [p])
    opt.eval()
    ref = p.detach().float().clone()
    opt.train()
    applied = float((p.detach().float() - ref).abs().mean())
    for _ in range(cycles):
        opt.eval()
        opt.train()
    opt.eval()
    return float((p.detach().float() - ref).norm()), applied


# --------------------------------------------------------------------------- 1
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_climb_round_trip_is_exact(dtype):
    """+e then -e must land back on the same stored value, at any weight precision."""
    drift, _ = _round_trip_drift(dtype)
    assert drift == 0.0, f"{dtype} weights drifted {drift:.3e} over 200 climb cycles"


@pytest.mark.parametrize("momentum_dtype", ["4bit", "int8", "bfloat16", "float32"])
def test_climb_round_trip_exact_for_every_codec(momentum_dtype):
    drift, _ = _round_trip_drift(torch.bfloat16, momentum_dtype=momentum_dtype)
    assert drift == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused climb needs CUDA")
def test_climb_round_trip_exact_on_the_fused_path():
    drift, _ = _round_trip_drift(torch.bfloat16, device="cuda", momentum_dtype="4bit")
    assert drift == 0.0


@pytest.mark.parametrize("norm,rho", [("global", 0.3), ("tensor", 0.3), ("none", -1.5)])
def test_every_msam_norm_mode_round_trips(norm, rho):
    g = torch.Generator().manual_seed(0)
    p = ((torch.randn(64, 64, generator=g) * 0.02).to(torch.bfloat16)).requires_grad_(True)
    opt = MSAM([p], rho=rho, norm=norm, lr=1e-4, betas=(0.5, 0.999), weight_decay=0.0,
               gradient_centralization=False)
    _spin(opt, [p])
    opt.eval()
    ref = p.detach().float().clone()
    for _ in range(200):
        opt.train()
        opt.eval()
    assert float((p.detach().float() - ref).norm()) == 0.0


# --------------------------------------------------------------------------- 2
def test_exact_round_trip_does_not_disable_the_climb():
    """Round-to-nearest must still apply a climb bf16 can represent (regression guard:
    a 'fix' that simply stopped perturbing would also show zero drift)."""
    _, applied_bf16 = _round_trip_drift(torch.bfloat16, cycles=0)
    _, applied_fp32 = _round_trip_drift(torch.float32, cycles=0)
    assert applied_bf16 > 0.5 * applied_fp32, (
        f"bf16 climb collapsed: {applied_bf16:.3e} vs fp32 {applied_fp32:.3e}"
    )


# --------------------------------------------------------------------------- 3
def _warnings_over(lr, steps=80, dtype=torch.bfloat16, k=1.5):
    g = torch.Generator().manual_seed(0)
    p = ((torch.randn(64, 64, generator=g) * 0.02).to(dtype)).requires_grad_(True)
    opt = _nekaon([p], lr=lr, k=k)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _spin(opt, [p], steps=steps)
    return [str(w.message) for w in caught]


def test_inert_lookahead_warns_at_tiny_lr():
    msgs = _warnings_over(1e-8)
    assert len(msgs) == 1 and "lookahead" in msgs[0]


def test_inert_lookahead_warns_on_fp32_too():
    """Inertness is about displacement size, not representability — fp32 gets it too."""
    msgs = _warnings_over(6e-7, dtype=torch.float32)
    assert len(msgs) == 1 and "inert" in msgs[0]


def test_no_warning_when_the_mechanism_works():
    assert _warnings_over(1e-2) == []


def test_no_warning_before_the_patience_window():
    """An LR warmup legitimately starts near zero; the warning must not trip on it."""
    assert _warnings_over(1e-8, steps=MSAM._INERT_PATIENCE - 5) == []


def test_no_warning_when_the_mechanism_is_off():
    assert _warnings_over(1e-8, k=0.0) == []


# --------------------------------------------------------------------------- 4
def _fresh_like(p):
    return _nekaon([p.detach().clone().requires_grad_(True)])


def test_train_mode_checkpoint_is_rejected():
    g = torch.Generator().manual_seed(0)
    p = (torch.randn(16, 16, generator=g) * 0.02).requires_grad_(True)
    opt = _nekaon([p])
    _spin(opt, [p])
    bad = copy.deepcopy(opt.state_dict())          # saved WITHOUT eval() — weights carry e
    assert bad["_msam_meta"]["train_mode"] is True
    with pytest.raises(ValueError, match="train mode"):
        _fresh_like(p).load_state_dict(bad)


def test_eval_mode_checkpoint_still_loads():
    g = torch.Generator().manual_seed(0)
    p = (torch.randn(16, 16, generator=g) * 0.02).requires_grad_(True)
    opt = _nekaon([p])
    _spin(opt, [p])
    opt.eval()
    good = copy.deepcopy(opt.state_dict())
    assert good["_msam_meta"]["train_mode"] is False
    _fresh_like(p).load_state_dict(good)           # must not raise


def test_checkpoints_written_before_the_marker_still_load():
    g = torch.Generator().manual_seed(0)
    p = (torch.randn(16, 16, generator=g) * 0.02).requires_grad_(True)
    opt = _nekaon([p])
    _spin(opt, [p])
    opt.eval()
    legacy = copy.deepcopy(opt.state_dict())
    legacy["_msam_meta"] = {"axpy_seed": 3}        # no train_mode key
    _fresh_like(p).load_state_dict(legacy)         # must not raise


# --------------------------------------------------------------------------- 5
@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused plan needs CUDA")
def test_fused_plan_detects_a_relocated_weight():
    """An external EMA / .to() / reshard rebinds p.data; the cached pointer table must
    not keep writing to the old address."""
    p = ((torch.randn(128, 128, device="cuda") * 0.02)).requires_grad_(True)
    opt = _nekaon([p], momentum_dtype="4bit")
    _spin(opt, [p])
    assert opt._axpy_cache is not None
    assert opt._plan_addrs_valid(opt._axpy_cache)
    p.data = p.data.clone()                        # storage moves
    assert not opt._plan_addrs_valid(opt._axpy_cache)
    opt.eval()
    true_w = p.detach().clone()
    opt.train()
    assert float((p.detach() - true_w).abs().max()) > 0, "climb missed the moved weight"
