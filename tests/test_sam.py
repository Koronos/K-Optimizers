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

import pytest
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


def test_second_step_restores_param_without_second_grad():
    """If a param had grad in first_step but None in second_step, restore w and drop old_p."""
    rho = 0.05
    w = torch.nn.Parameter(torch.randn(4, 3))
    b = torch.nn.Parameter(torch.randn(5))
    params = [w, b]
    _attach_grads(params, seed=10)
    b0 = b.data.clone()

    opt = SAM(params, Adakaon, rho=rho, lr=1e-3)
    opt.first_step(zero_grad=True)

    w.grad = torch.randn_like(w)
    b.grad = None

    opt.second_step(zero_grad=False)

    assert torch.equal(b.data, b0)
    assert "old_p" not in opt.state[b]


def _manual_first_step_fp32(params, rho, eps, adaptive=False):
    """Reference per-parameter first_step for fp32 checks (fp32 accumulation order)."""
    gn = _global_grad_norm(params, adaptive=adaptive)
    scale = torch.tensor(rho / (gn + eps), dtype=torch.float32)
    for p in params:
        if p.grad is None:
            continue
        e = p.grad.float() * scale
        if adaptive:
            e = e * (p.data.float() * p.data.float())
        p.data.add_(e.to(p.dtype))


def test_first_step_foreach_fp32_bit_identical():
    """Batched first_step matches a manual per-parameter climb in fp32."""
    rho = 0.05
    eps = 1e-12
    for adaptive in (False, True):
        params = _quadratic_params(seed=20 if adaptive else 21)
        ref = _quadratic_params(seed=20 if adaptive else 21)
        _attach_grads(params, seed=30)
        _attach_grads(ref, seed=30)

        opt = SAM(params, Adakaon, rho=rho, adaptive=adaptive, eps=eps, lr=1e-3)
        opt.first_step(zero_grad=False)
        _manual_first_step_fp32(ref, rho, eps, adaptive=adaptive)

        for p, rp in zip(params, ref, strict=True):
            torch.testing.assert_close(p.data, rp.data, rtol=1e-6, atol=1e-6)


def test_first_step_chunking_matches_per_param(monkeypatch):
    """Forced foreach budget splits the climb into chunks without changing the result."""
    from kaon import sam as sam_mod

    def tiny_budget(_stack_budget, _cutoff, _bytes_per, _device):
        return 2 * 4 * 3  # two (4, 3) tensors per chunk

    monkeypatch.setattr(sam_mod, "foreach_budget", tiny_budget)

    rho = 0.05
    eps = 1e-12
    g = torch.Generator().manual_seed(40)
    params = [
        torch.nn.Parameter(torch.randn(4, 3, generator=g, dtype=torch.float32).requires_grad_(True))
        for _ in range(8)
    ]
    ref = [
        torch.nn.Parameter(p.detach().clone().requires_grad_(True))
        for p in params
    ]
    _attach_grads(params, seed=41)
    _attach_grads(ref, seed=41)

    opt = SAM(params, Adakaon, rho=rho, eps=eps, lr=1e-3)
    opt.first_step(zero_grad=False)
    _manual_first_step_fp32(ref, rho, eps)

    for p, rp in zip(params, ref, strict=True):
        torch.testing.assert_close(p.data, rp.data, rtol=1e-6, atol=1e-6)


# ------------------------------------------------------- 0.7.18 audit: norm precision
def test_grad_norm_of_bf16_grads_is_fp32_accurate():
    """The per-tensor norms are reduced in fp32: ``_foreach_norm`` of bf16 grads used to
    return bf16 scalars (rounded to 8 mantissa bits) BEFORE the ``.float()``, so the
    global norm carried up to ~2e-3 relative error."""
    g = torch.Generator().manual_seed(5)
    params = []
    for n in (50_000, 3_001, 777):
        p = torch.randn(n, generator=g).to(torch.bfloat16).requires_grad_(True)
        p.grad = torch.randn(n, generator=g).to(torch.bfloat16)
        params.append(p)
    opt = SAM(params, Adakaon, lr=1e-3)
    got = opt._grad_norm()
    assert got.dtype == torch.float32
    ref = math.sqrt(sum(float(p.grad.double().pow(2).sum()) for p in params))
    assert abs(float(got) - ref) / ref < 1e-5


def test_grad_norm_adaptive_is_chunked_and_exact(monkeypatch):
    """ASAM's ``|w|*g`` is materialized per chunk, never for all params at once — and the
    chunked norm equals the reference."""
    from kaon import sam as sam_mod

    g = torch.Generator().manual_seed(9)
    params = [torch.randn(4, 3, generator=g).requires_grad_(True) for _ in range(7)]
    params.append(torch.randn(5, generator=g).requires_grad_(True))
    _attach_grads(params, seed=10)
    ref = _global_grad_norm(params, adaptive=True)
    opt = SAM(params, Adakaon, adaptive=True, lr=1e-3)
    full = float(opt._grad_norm())

    def tiny_budget(_stack_budget, _cutoff, _bytes_per, _device):
        return 2 * 4 * 3

    monkeypatch.setattr(sam_mod, "foreach_budget", tiny_budget)
    chunked = float(opt._grad_norm())
    assert abs(chunked - ref) / ref < 1e-6
    assert abs(chunked - full) / ref < 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_grad_norm_adaptive_peak_is_one_chunk():
    """ASAM's norm held ``|w|*g`` for EVERY param at once (+1x the weights); with a
    one-tensor chunk budget the transient must stay around one tensor."""
    n, k = 1 << 20, 8
    params = [torch.randn(n, device="cuda").requires_grad_(True) for _ in range(k)]
    for p in params:
        p.grad = torch.randn(n, device="cuda")
    opt = SAM(params, Adakaon, adaptive=True, lr=1e-3, foreach_stack_budget=n)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    opt._grad_norm()
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - base
    assert extra < 2.5 * 4 * n, f"adaptive norm transient {extra / (4 * n):.2f}x one tensor"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_grad_norm_params_on_different_devices():
    """``torch.stack`` of per-tensor norms living on different devices raised; the norms
    are brought to one device first."""
    a = torch.randn(6, 4).requires_grad_(True)
    b = torch.randn(6, 4, device="cuda").requires_grad_(True)
    a.grad = torch.randn(6, 4)
    b.grad = torch.randn(6, 4, device="cuda")
    opt = SAM([a, b], Adakaon, lr=1e-3, rho=0.05)
    ref = math.sqrt(float(a.grad.double().pow(2).sum()) + float(b.grad.double().pow(2).sum()))
    assert abs(float(opt._grad_norm()) - ref) / ref < 1e-6
    a0, b0 = a.detach().clone(), b.detach().clone()
    opt.first_step()
    assert not torch.equal(a.detach(), a0) and not torch.equal(b.detach(), b0)


# ------------------------------------------------ 0.7.18 audit: groups / double climb
def test_groups_without_grads_are_skipped():
    a = torch.randn(4, 3).requires_grad_(True)
    b = torch.randn(4, 3).requires_grad_(True)
    opt = SAM([{"params": [a]}, {"params": [b]}], Adakaon, lr=1e-2, rho=0.05)
    b0 = b.detach().clone()
    a.grad = torch.randn(4, 3)
    opt.first_step()
    a.grad = torch.randn(4, 3)
    opt.second_step()
    assert torch.equal(b.detach(), b0)


def test_first_step_without_any_grad_and_empty_first_group():
    b = torch.randn(3, 3).requires_grad_(True)
    opt = SAM([{"params": []}, {"params": [b]}], Adakaon, lr=1e-3)
    b0 = b.detach().clone()
    opt.first_step()     # no grad anywhere; the first group has no params at all
    opt.second_step()
    assert torch.equal(b.detach(), b0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_repeated_first_step_restores_before_climbing(dtype):
    """An aborted SAM step (``first_step`` with no ``second_step`` — e.g. the perturbed
    forward was skipped) followed by another ``first_step`` must not snapshot the
    PERTURBED weights as ``old_p``: that baked the first climb into the weights for good
    (1.9e-2 max drift in fp32). The second ``first_step`` restores first, so the restore
    after it lands on the original weights exactly."""
    g = torch.Generator().manual_seed(3)
    w = torch.randn(16, 16, generator=g).to(dtype).requires_grad_(True)
    w0 = w.detach().clone()
    opt = SAM([w], Adakaon, lr=1e-3, rho=0.1)
    w.grad = torch.randn(16, 16, generator=g).to(dtype)
    opt.first_step()                  # climb 1, never completed
    w.grad = torch.randn(16, 16, generator=g).to(dtype)
    g2 = w.grad.clone()
    opt.first_step()                  # climb 2 must start from w0
    if dtype == torch.float32:
        scale = torch.tensor(0.1 / (float(g2.double().norm()) + 1e-12), dtype=torch.float32)
        torch.testing.assert_close(w.detach(), w0 + g2 * scale, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(opt.state[w]["old_p"], w0, rtol=0, atol=0)
    w.grad = None                     # base step is a no-op without a grad
    opt.second_step()
    assert torch.equal(w.detach(), w0)


def test_pending_climb_survives_a_checkpoint():
    """A checkpoint taken between first_step and second_step carries old_p; a resumed
    SAM whose next call is first_step must restore it before climbing again."""
    import copy

    w = torch.randn(6, 5).requires_grad_(True)
    w0 = w.detach().clone()
    opt = SAM([w], Adakaon, lr=1e-3, rho=0.1)
    w.grad = torch.randn(6, 5)
    opt.first_step()
    sd = copy.deepcopy(opt.state_dict())
    opt2 = SAM([w], Adakaon, lr=1e-3, rho=0.1)
    opt2.load_state_dict(sd)
    w.grad = torch.randn(6, 5)
    opt2.first_step()
    torch.testing.assert_close(opt2.state[w]["old_p"], w0, rtol=0, atol=0)
    w.grad = None
    opt2.second_step()
    assert torch.equal(w.detach(), w0)


def test_add_param_group_after_construction():
    a = torch.randn(4, 3).requires_grad_(True)
    c = torch.randn(5).requires_grad_(True)
    opt = SAM([a], Adakaon, lr=1e-3, rho=0.07)
    opt.add_param_group({"params": [c], "lr": 5e-3})
    assert opt.param_groups[-1]["rho"] == 0.07 and opt.param_groups[-1]["adaptive"] is False
    assert opt.base_optimizer.param_groups is opt.param_groups
    c0 = c.detach().clone()
    for p in (a, c):
        p.grad = torch.randn_like(p)
    opt.first_step(zero_grad=True)
    for p in (a, c):
        p.grad = torch.randn_like(p)
    opt.second_step()
    assert not torch.equal(c.detach(), c0)
    assert "old_p" not in opt.state[c]


@pytest.mark.filterwarnings("ignore:Detected call of `lr_scheduler.step")
def test_torch_lr_scheduler_drives_the_inner_lr():
    """A torch LR scheduler built on the SAM wrapper must change the lr the BASE optimizer
    actually steps with (shared ``param_groups``)."""
    torch.manual_seed(0)
    deltas = []
    w = torch.randn(8, 8).requires_grad_(True)
    opt = SAM([w], Adakaon, lr=1e-2, rho=0.0, betas=(0.0, 0.999), cautious=False,
              gradient_centralization=False)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.1)
    g = torch.randn(8, 8)
    for _ in range(2):
        before = w.detach().clone()
        w.grad = g.clone()
        opt.first_step()
        w.grad = g.clone()
        opt.second_step()
        sched.step()
        deltas.append(float((w.detach() - before).abs().max()))
    assert opt.base_optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert deltas[1] < 0.5 * deltas[0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_element_params_climb_and_restore(dtype):
    """Buckets of zero-element params divided the stack budget by ``numel == 0``."""
    ps = [torch.randn(s).to(dtype).requires_grad_(True) for s in [(4, 3), (0, 3), (0, 3)]]
    opt = SAM(ps, Adakaon, lr=1e-3, rho=0.05)
    w0 = [p.detach().clone() for p in ps]
    for p in ps:
        p.grad = torch.randn(p.shape).to(dtype)
    opt.first_step()
    for p in ps:
        p.grad = None
    opt.second_step()
    for a, p in zip(w0, ps, strict=True):
        assert torch.equal(a, p.detach())


def test_reload_after_aborted_first_step_keeps_the_reloaded_weights():
    """An aborted first_step, then the trainer reloads the weights (in place, e.g.
    ``load_state_dict``): the next first_step must NOT clobber them with the stale
    pre-climb snapshot."""
    w = torch.randn(6, 5).requires_grad_(True)
    opt = SAM([w], Adakaon, lr=1e-3, rho=0.1)
    w.grad = torch.randn(6, 5)
    opt.first_step()                                   # aborted
    reloaded = torch.randn(6, 5)
    with torch.no_grad():
        w.copy_(reloaded)                              # what load_state_dict does
    w.grad = torch.randn(6, 5)
    with pytest.warns(UserWarning, match="modified in place"):
        opt.first_step()
    torch.testing.assert_close(opt.state[w]["old_p"], reloaded, rtol=0, atol=0)
    w.grad = None
    opt.second_step()
    assert torch.equal(w.detach(), reloaded)


def test_normal_cycle_does_not_warn_about_reloads(recwarn):
    w = torch.randn(6, 5).requires_grad_(True)
    opt = SAM([w], Adakaon, lr=1e-3, rho=0.1)
    for _ in range(3):
        w.grad = torch.randn(6, 5)
        opt.first_step(zero_grad=True)
        w.grad = torch.randn(6, 5)
        opt.second_step()
    w.grad = torch.randn(6, 5)
    opt.first_step()
    w.grad = torch.randn(6, 5)
    opt.first_step()                                   # aborted + retry, no reload
    assert not [r for r in recwarn if "modified in place" in str(r.message)]
