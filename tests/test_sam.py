"""Tests for SAM — Sharpness-Aware Minimization wrapping a base kaon optimizer.

Covers:
  1. Mechanics — after ``first_step`` each param moved by exactly ``rho * g / ||g||``
     (global norm), with ``old_p`` snapshotted; ``second_step`` restores ``w`` before the
     base step and the net update equals the base optimizer applied to the perturbed grad.
  2. ``step(closure)`` equals manual ``first_step`` + (closure recompute) + ``second_step``.
  3. bf16-correctness — climb+restore round-trip leaves a bf16 weight bit-identical when
     the base step is skipped (no drift).
  4. ASAM (``adaptive=True``) basic mechanics: per-weight ``w^2`` scaling of the
     perturbation and ``|w|`` scaling of the norm.
"""

from __future__ import annotations

import math

import torch

from kaon import SAM, Adakaon


def _global_grad_norm(params, adaptive=False):
    """Reference global L2 grad norm via (g*g).sum() (no torch.dot — SIGFPE-safe)."""
    sq = 0.0
    for p in params:
        g = p.grad
        if adaptive:
            g = p.abs() * g
        sq += float((g * g).sum().detach())
    return math.sqrt(sq)


def _quadratic_params(seed=0):
    """Two params (a 2-D matrix + a 1-D bias) with deterministic grads attached."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(4, 3, generator=g, dtype=torch.float32).requires_grad_(True)
    b = torch.randn(5, generator=g, dtype=torch.float32).requires_grad_(True)
    return [w, b]


def _attach_grads(params, seed=1):
    g = torch.Generator().manual_seed(seed)
    for p in params:
        p.grad = torch.randn(p.shape, generator=g, dtype=p.dtype)


# --------------------------------------------------------------------------- 1
def test_first_step_perturbation_exact():
    """After first_step, each param moved by exactly rho * g / (||g|| + eps) (global)."""
    rho = 0.05
    params = _quadratic_params()
    _attach_grads(params)
    w0 = [p.data.clone() for p in params]
    grads = [p.grad.clone() for p in params]

    gn = _global_grad_norm(params)
    opt = SAM(params, Adakaon, rho=rho, lr=1e-3)
    opt.first_step(zero_grad=False)

    for p, w_init, g in zip(params, w0, grads, strict=False):
        expected_e = rho / (gn + opt.eps) * g
        delta = p.data - w_init
        assert torch.allclose(delta, expected_e, atol=1e-6, rtol=1e-5)
        # old_p snapshot is the pre-climb weight.
        assert torch.allclose(opt.state[p]["old_p"], w_init, atol=0, rtol=0)


def test_second_step_restores_then_base_steps():
    """second_step restores w exactly, then the base optimizer steps with the grad
    present at call time (the perturbed grad). Net result == Adakaon applied at w to g~."""
    rho = 0.05
    params = _quadratic_params()
    _attach_grads(params, seed=1)
    w0 = [p.data.clone() for p in params]

    opt = SAM(params, Adakaon, rho=rho, lr=1e-3, betas=(0.9, 0.999), cautious=True)
    opt.first_step(zero_grad=True)

    # Simulate the second backward: attach the "perturbed-point" gradient.
    _attach_grads(params, seed=2)
    g_tilde = [p.grad.clone() for p in params]

    # Reference: a fresh Adakaon with identical config stepping from w0 on g_tilde.
    ref_params = [w.clone().detach().requires_grad_(True) for w in w0]
    for rp, g in zip(ref_params, g_tilde, strict=False):
        rp.grad = g.clone()
    ref_opt = Adakaon(ref_params, lr=1e-3, betas=(0.9, 0.999), cautious=True)
    ref_opt.step()

    opt.second_step(zero_grad=False)

    for p, rp in zip(params, ref_params, strict=False):
        assert torch.allclose(p.data, rp.data, atol=1e-6, rtol=1e-5), (
            "SAM second_step must restore w then apply the base step to the perturbed grad"
        )


# --------------------------------------------------------------------------- 2
def test_step_closure_equals_manual_two_pass():
    """step(closure) == first_step + (closure recompute) + second_step."""
    rho = 0.05
    # --- manual two-pass ---
    pm = _quadratic_params()
    _attach_grads(pm, seed=1)
    opt_m = SAM(pm, Adakaon, rho=rho, lr=1e-3, betas=(0.9, 0.999))
    opt_m.first_step(zero_grad=True)
    _attach_grads(pm, seed=2)  # the recomputed perturbed-point grad
    opt_m.second_step(zero_grad=False)

    # --- closure form, same grads ---
    pc = _quadratic_params()
    _attach_grads(pc, seed=1)
    opt_c = SAM(pc, Adakaon, rho=rho, lr=1e-3, betas=(0.9, 0.999))

    def closure():
        # Mimic "zero_grad; recompute loss; backward" by directly attaching the same
        # perturbed-point grad the manual path used.
        _attach_grads(pc, seed=2)
        return torch.tensor(0.0)

    opt_c.step(closure)

    for a, b in zip(pm, pc, strict=False):
        assert torch.allclose(a.data, b.data, atol=0, rtol=0)


# --------------------------------------------------------------------------- 3
def test_bf16_climb_restore_roundtrip_no_drift():
    """climb + restore on a bf16 weight leaves it bit-identical when the base step is
    skipped — the restore is exact, so no SR climb rounding leaks into the weight."""
    rho = 0.1
    g = torch.Generator().manual_seed(7)
    w = torch.randn(8, 6, generator=g).to(torch.bfloat16).requires_grad_(True)
    w.grad = torch.randn(8, 6, generator=g).to(torch.bfloat16)
    w_before = w.data.clone()

    opt = SAM([w], Adakaon, rho=rho, lr=1e-3, bf16_method="stochastic_rounding")
    opt.first_step(zero_grad=False)
    # The climb must have actually moved the weight (sanity: rho > 0, grad nonzero).
    assert not torch.equal(w.data, w_before)
    # Now zero the grad so the base step is a no-op (no grad -> base skips it),
    # then restore: weight must return to exactly w_before.
    w.grad.zero_()
    opt.second_step(zero_grad=False)
    assert torch.equal(w.data, w_before), "climb/restore round-trip drifted a bf16 weight"


def test_bf16_climb_is_stochastic_rounded():
    """The bf16 climb goes through add_stochastic_ (not a truncating add): a sub-ULP
    perturbation must have a nonzero chance of moving the weight across many draws."""
    g = torch.Generator().manual_seed(3)
    base = torch.randn(2000, generator=g).to(torch.bfloat16)
    moved_any = False
    for s in range(8):
        w = base.clone().requires_grad_(True)
        # Tiny grad so rho*g/||g|| is well below the bf16 ULP for most coords.
        w.grad = (torch.randn(2000, generator=torch.Generator().manual_seed(100 + s)) * 1e-3).to(torch.bfloat16)
        w0 = w.data.clone()
        opt = SAM([w], Adakaon, rho=0.01, lr=1e-3, bf16_method="stochastic_rounding")
        opt.first_step(zero_grad=False)
        if not torch.equal(w.data, w0):
            moved_any = True
            break
    assert moved_any, "bf16 climb never moved the weight — SR not applied?"


# --------------------------------------------------------------------------- 4
def test_adaptive_asam_mechanics():
    """ASAM: norm uses |w|*g, perturbation uses w^2 * g * scale."""
    rho = 0.05
    params = _quadratic_params(seed=4)
    _attach_grads(params, seed=5)
    w0 = [p.data.clone() for p in params]
    grads = [p.grad.clone() for p in params]

    gn = _global_grad_norm(params, adaptive=True)
    opt = SAM(params, Adakaon, rho=rho, adaptive=True, lr=1e-3)
    opt.first_step(zero_grad=False)

    for p, w_init, gr in zip(params, w0, grads, strict=False):
        scale = rho / (gn + opt.eps)
        expected_e = scale * (w_init * w_init) * gr
        delta = p.data - w_init
        assert torch.allclose(delta, expected_e, atol=1e-6, rtol=1e-5)


def test_step_without_closure_raises():
    params = _quadratic_params()
    _attach_grads(params)
    opt = SAM(params, Adakaon, lr=1e-3)
    try:
        opt.step()
    except RuntimeError:
        return
    raise AssertionError("SAM.step() without a closure must raise")
