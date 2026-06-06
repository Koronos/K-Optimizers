"""Tests for AdafusionEx — Adafusion + weight EMA (+ optional MSAM).

The headline contract: with ``ema_decay=0`` and ``sam_mode=None`` AdafusionEx is
**bit-comparable to Adafusion** (so it is a clean A/B baseline). Plus the EMA math,
store/copy_to/restore round-trip, and the MSAM rho=0 / un-perturb identity.
"""

from __future__ import annotations

import copy

import torch

from koptim import Adafusion, AdafusionEx

from .conftest import train_steps


def _make_mlp() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 8),
    )


def _clone_model(model: torch.nn.Module) -> torch.nn.Module:
    return copy.deepcopy(model)


# --------------------------------------------------------------------- A/B parity
def _assert_models_bit_equal(a: torch.nn.Module, b: torch.nn.Module) -> None:
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb), "params diverged between Adafusion and AdafusionEx"


def test_matches_adafusion_when_techniques_off_momentum():
    """ema_decay=0, sam off -> identical trajectory to Adafusion (fp32, beta1>0)."""
    torch.manual_seed(0xC0DE)
    m_ref = _make_mlp()
    m_ex = _clone_model(m_ref)
    kw = dict(lr=3e-3, betas=(0.9, 0.999), weight_decay=0.01)
    opt_ref = Adafusion(m_ref.parameters(), **kw)
    opt_ex = AdafusionEx(m_ex.parameters(), ema_decay=0.0, sam_mode=None, **kw)

    x = torch.randn(8, 16)
    y = torch.randn(8, 8)
    train_steps(m_ref, opt_ref, [(x, y)] * 25)
    train_steps(m_ex, opt_ex, [(x, y)] * 25)
    _assert_models_bit_equal(m_ref, m_ex)


def test_matches_adafusion_when_techniques_off_no_momentum():
    """Same parity check with beta1=0 (minimum-VRAM / Adafactor-like config)."""
    torch.manual_seed(0xC0DE)
    m_ref = _make_mlp()
    m_ex = _clone_model(m_ref)
    kw = dict(lr=3e-3, betas=(0.0, 0.999), cautious=False)
    opt_ref = Adafusion(m_ref.parameters(), **kw)
    opt_ex = AdafusionEx(m_ex.parameters(), **kw)

    x = torch.randn(8, 16)
    y = torch.randn(8, 8)
    train_steps(m_ref, opt_ref, [(x, y)] * 25)
    train_steps(m_ex, opt_ex, [(x, y)] * 25)
    _assert_models_bit_equal(m_ref, m_ex)


def test_conv_parity_with_adafusion():
    """A conv kernel (4-D, matrixized path) also matches Adafusion bit-for-bit."""
    torch.manual_seed(0)
    net_ref = torch.nn.Sequential(
        torch.nn.Conv2d(4, 8, 3, padding=1), torch.nn.GELU(),
        torch.nn.Conv2d(8, 4, 3, padding=1),
    )
    net_ex = _clone_model(net_ref)
    opt_ref = Adafusion(net_ref.parameters(), lr=1e-3, betas=(0.9, 0.999))
    opt_ex = AdafusionEx(net_ex.parameters(), lr=1e-3, betas=(0.9, 0.999))
    x = torch.randn(2, 4, 8, 8)
    y = torch.randn(2, 4, 8, 8)
    train_steps(net_ref, opt_ref, [(x, y)] * 15)
    train_steps(net_ex, opt_ex, [(x, y)] * 15)
    _assert_models_bit_equal(net_ref, net_ex)


# ------------------------------------------------------------------- EMA math
def test_ema_matches_reference_decay_math():
    """The shadow tracks the closed-form EMA (fp32 storage, warmup off)."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(10, 10))
    decay = 0.9
    opt = AdafusionEx([p], lr=1e-2, betas=(0.9, 0.999),
                      ema_decay=decay, ema_dtype="float32", ema_warmup=False)

    ref = p.detach().clone()  # reference shadow, initialized to the params
    for _ in range(20):
        p.grad = torch.randn_like(p)
        opt.step()
        # Mirror lerp_'s formula exactly: start + w*(end - start) (vs the
        # algebraically-equal d*ref + (1-d)*p, which rounds slightly differently
        # in fp32 and would drift over many steps).
        ref = ref + (1.0 - decay) * (p.detach() - ref)

    shadow = opt._ema_shadow[p]
    assert torch.allclose(shadow, ref, atol=1e-6), "EMA shadow diverged from reference math"


def test_ema_warmup_ramps_decay():
    """With warmup, early effective decay is the (1+t)/(10+t) ramp, capped at ema_decay."""
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = AdafusionEx([p], ema_decay=0.999, ema_warmup=True)
    # ema_step starts at 0 -> warm = 1/10 = 0.1 < 0.999.
    assert abs(opt._effective_decay() - 0.1) < 1e-9
    opt.ema_step = 90  # warm = 91/100 = 0.91
    assert abs(opt._effective_decay() - 0.91) < 1e-9
    opt.ema_step = 100_000  # warm ~ 1.0 -> capped at ema_decay
    assert abs(opt._effective_decay() - 0.999) < 1e-9


def test_ema_does_not_touch_live_params():
    """update_ema must never mutate the live weights (only the shadow)."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = AdafusionEx([p], lr=1e-2, betas=(0.9, 0.999), ema_decay=0.99, ema_dtype="float32")
    p.grad = torch.randn_like(p)
    opt.step()
    before = p.detach().clone()
    opt.update_ema()  # extra manual EMA tick
    opt.update_ema()
    assert torch.equal(p.detach(), before), "EMA corrupted the live params"


def test_ema_disabled_allocates_nothing():
    """ema_decay=0 -> no shadow buffer ever created."""
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = AdafusionEx([p], lr=1e-2, betas=(0.9, 0.999), ema_decay=0.0)
    p.grad = torch.randn_like(p)
    opt.step()
    assert opt._ema_shadow == {}


# ----------------------------------------------------- store / copy_to / restore
def test_store_copy_restore_roundtrip():
    """copy_to swaps EMA in; restore puts the exact live params back."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = AdafusionEx([p], lr=1e-2, betas=(0.9, 0.999), ema_decay=0.5, ema_dtype="float32")
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()

    live = p.detach().clone()
    shadow = opt._ema_shadow[p].clone()
    assert not torch.equal(live, shadow), "shadow should differ from live after training"

    opt.store()
    opt.copy_to()
    assert torch.allclose(p.detach(), shadow), "copy_to did not install the EMA weights"

    opt.restore()
    assert torch.equal(p.detach(), live), "restore did not recover the live weights"


# ------------------------------------------------------------------- post-hoc EMA
def test_posthoc_snapshots_and_reconstruct():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(6, 6))
    opt = AdafusionEx([p], lr=1e-2, betas=(0.9, 0.999),
                      ema_decay=0.99, posthoc_ema=True, posthoc_interval=2)
    for _ in range(10):
        p.grad = torch.randn_like(p)
        opt.step()
    # 10 steps, interval 2 -> 5 snapshots.
    assert len(opt._posthoc_snapshots) == 5
    out = opt.reconstruct_posthoc_ema(decay=0.9)
    assert len(out) == 1
    assert out[0].shape == p.shape
    assert torch.isfinite(out[0]).all()
    # decay=0 -> only the last snapshot survives.
    last = opt._posthoc_snapshots[-1][1][0]
    out0 = opt.reconstruct_posthoc_ema(decay=0.0)
    assert torch.allclose(out0[0], last)


# ------------------------------------------------------------------- MSAM / SAM
def test_msam_rho_zero_is_bit_comparable_to_adafusion():
    """sam_mode='msam', rho=0 -> still identical to plain Adafusion."""
    torch.manual_seed(0xC0DE)
    m_ref = _make_mlp()
    m_ex = _clone_model(m_ref)
    kw = dict(lr=3e-3, betas=(0.9, 0.999))
    opt_ref = Adafusion(m_ref.parameters(), **kw)
    opt_ex = AdafusionEx(m_ex.parameters(), sam_mode="msam", sam_rho=0.0, **kw)

    x = torch.randn(8, 16)
    y = torch.randn(8, 8)
    for _ in range(15):
        opt_ref.zero_grad()
        (m_ref(x) - y).pow(2).mean().backward()
        opt_ref.step()

        opt_ex.zero_grad()
        opt_ex.first_step()  # rho=0 -> no perturbation
        (m_ex(x) - y).pow(2).mean().backward()
        opt_ex.second_step()
    _assert_models_bit_equal(m_ref, m_ex)


def test_msam_perturb_unperturb_is_identity_at_step_boundary():
    """first_step + (no grad change) + un-perturb leaves params unchanged.

    We perturb, then manually add the same direction back via the second_step
    un-perturb path *without* taking a real step, isolating the perturb/un-perturb
    pair. Verified for rho>0 (a real perturbation that must cancel).
    """
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = AdafusionEx([p], lr=1e-2, betas=(0.9, 0.999), sam_mode="msam", sam_rho=0.5)
    # Build a momentum buffer first.
    for _ in range(3):
        p.grad = torch.randn_like(p)
        opt.step()

    before = p.detach().clone()
    opt.first_step()
    assert not torch.allclose(p.detach(), before), "rho>0 should actually perturb"
    # Un-perturb manually (mirror second_step's add-back, without stepping).
    dirs, norm = opt._momentum_dirs()
    scale = opt.sam_rho / (norm + 1e-12)
    for q, d in dirs.items():
        q.data.add_(d.to(q.dtype), alpha=scale)
    assert torch.allclose(p.detach(), before, atol=1e-5), "perturb/un-perturb did not cancel"


def test_msam_requires_momentum():
    p = torch.nn.Parameter(torch.randn(4, 4))
    try:
        AdafusionEx([p], betas=(0.0, 0.999), sam_mode="msam", sam_rho=0.1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for sam_mode='msam' with beta1=0")


# ------------------------------------------------------------------- checkpoint
def test_state_dict_roundtrip_preserves_ema():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = AdafusionEx([p], lr=1e-2, betas=(0.9, 0.999), ema_decay=0.9, ema_dtype="bfloat16")
    for _ in range(6):
        p.grad = torch.randn_like(p)
        opt.step()
    sd = copy.deepcopy(opt.state_dict())
    shadow_before = opt._ema_shadow[p].clone()
    step_before = opt.ema_step

    p2 = torch.nn.Parameter(p.detach().clone())
    opt2 = AdafusionEx([p2], lr=1e-2, betas=(0.9, 0.999), ema_decay=0.9, ema_dtype="bfloat16")
    opt2.load_state_dict(sd)
    assert opt2.ema_step == step_before
    assert torch.equal(opt2._ema_shadow[p2], shadow_before)
    assert opt2._ema_shadow[p2].dtype == torch.bfloat16  # dtype preserved


def test_state_dict_does_not_alias_shadow():
    """A checkpoint snapshot must not alias the live shadow (continue-training safe)."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = AdafusionEx([p], lr=1e-2, betas=(0.9, 0.999), ema_decay=0.9, ema_dtype="float32")
    p.grad = torch.randn_like(p)
    opt.step()
    sd = opt.state_dict()
    saved = sd["ema_shadow"][0].clone()
    # Keep training the original; the saved snapshot must not move.
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()
    assert torch.equal(sd["ema_shadow"][0], saved), "state_dict aliased the live shadow"
