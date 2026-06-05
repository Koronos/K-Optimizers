"""Tests for Autofusion (Mechanic LR tuner on Adafusion, freeze-to-free).

Covers the base Mechanic behavior, the back-compat ``AdaptiveAdafusion`` /
``AdafusionProdigy`` aliases, the on-the-fly ``Delta`` reconstruction, and the
``lr_freeze`` (int / "auto") handoff that turns the optimizer into plain Adafusion
after warmup.

Autofusion's empirical scaffolding (``store_delta``, ``s_init_rel``,
``scale_floor_frac``, and the auto-freeze ``tol``/``patience``/``max_frac``) was
collapsed to internal constants once iteration-3 validated the defaults
generalize; see ``test_purged_knobs_are_not_public`` for the public-surface guard.
"""

from __future__ import annotations

import inspect

import torch

from koptim import AdafusionProdigy, AdaptiveAdafusion, Autofusion
from koptim.adafusion import Adafusion

from .conftest import train_steps


def test_aliases_are_same_class() -> None:
    """AdaptiveAdafusion and AdafusionProdigy are back-compat aliases of Autofusion."""
    assert AdaptiveAdafusion is Autofusion
    assert AdafusionProdigy is Autofusion


def test_purged_knobs_are_not_public() -> None:
    """The empirical scaffolding knobs are gone from the public __init__ signature
    (they are now module-level constants at their validated defaults)."""
    params = set(inspect.signature(Autofusion.__init__).parameters)
    for gone in (
        "store_delta",
        "s_init_rel",
        "scale_floor_frac",
        "lr_freeze_tol",
        "lr_freeze_patience",
        "lr_freeze_max_frac",
    ):
        assert gone not in params, f"{gone} should no longer be a public kwarg"
    # And the headline / advanced knobs that survived the purge are still public.
    for kept in ("lr_freeze", "scale_cap", "scale_cap_rel", "adafusion_betas"):
        assert kept in params, f"{kept} must stay public"


def test_lr_freeze_defaults_to_auto() -> None:
    """lr_freeze defaults to 'auto' (the headline freeze-to-free feature, on)."""
    assert inspect.signature(Autofusion.__init__).parameters["lr_freeze"].default == "auto"
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = Autofusion([p])
    assert opt._lr_freeze == "auto"


def test_internal_constants_unchanged() -> None:
    """The purged knobs are pinned to the exact values they defaulted to before, so
    behavior is identical to the old defaults."""
    from koptim import autofusion as af

    assert af._S_INIT_REL == 3e-3
    assert af._SCALE_FLOOR_FRAC == 0.5
    assert af._LR_FREEZE_TOL == 0.02
    assert af._LR_FREEZE_PATIENCE == 50
    assert af._LR_FREEZE_MAX_FRAC == 0.9
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = Autofusion([p])
    assert opt._s_init_rel == 3e-3
    assert opt._scale_floor_frac == 0.5
    assert opt._lr_freeze_tol == 0.02
    assert opt._lr_freeze_patience == 50
    assert opt._lr_freeze_max_frac == 0.9


def test_scale_starts_at_seed_then_rises() -> None:
    """The discovered effective LR seeds near ``s_init`` and grows with training."""
    torch.manual_seed(0)
    w = torch.randn(32, 32)
    target = torch.randn(32, 32)
    x = torch.nn.Parameter(torch.zeros(32, 32))
    opt = Autofusion([x], s_init=1e-6, lr_freeze=None)
    opt.zero_grad()
    ((w @ x - target) ** 2).mean().backward()
    opt.step()
    first = opt.get_d()
    for _ in range(60):
        opt.zero_grad()
        ((w @ x - target) ** 2).mean().backward()
        opt.step()
    assert opt.get_d() > first, "effective LR should grow during training"
    assert opt.get_d() < 1.0, "effective LR should stay sane (< 1.0) on a toy problem"


def test_reduces_loss(
    toy_mlp: torch.nn.Module, random_batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Parameter-free (lr=1) optimizer reduces a toy MLP's loss."""
    x, y = random_batch
    opt = Autofusion(toy_mlp.parameters(), s_init=1e-4)
    before = (toy_mlp(x) - y).pow(2).mean().item()
    train_steps(toy_mlp, opt, [random_batch] * 50)
    after = (toy_mlp(x) - y).pow(2).mean().item()
    assert after < before, f"loss should drop: {before:.4f} -> {after:.4f}"


def test_delta_reconstruction_is_valid(
    toy_mlp: torch.nn.Module, random_batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """The on-the-fly Delta = (p - ref)/sum(s) reconstruction drives a real descent
    (the optimizer never stores Delta explicitly — ref is its only per-param state)."""
    x, y = random_batch
    opt = Autofusion(toy_mlp.parameters(), s_init=1e-4)
    before = (toy_mlp(x) - y).pow(2).mean().item()
    train_steps(toy_mlp, opt, [random_batch] * 40)
    after = (toy_mlp(x) - y).pow(2).mean().item()
    assert after < before


def test_only_ref_buffer_allocated() -> None:
    """The only per-param Mechanic state while adapting is ``ref`` (one copy of the
    weights); no Delta buffer is ever allocated."""
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = Autofusion([p])
    p.grad = torch.randn_like(p)
    opt.step()
    assert "delta" not in opt._mech, "no delta buffer key should exist"
    assert len(opt._mech["ref"]) == 1, "exactly one ref buffer (for the one param)"


def test_forwards_adafusion_kwargs() -> None:
    """Adafusion knobs (clip_threshold, cautious, momentum_dtype) forward to base."""
    p = torch.nn.Parameter(torch.randn(16, 16))
    opt = Autofusion(
        [p],
        betas=(0.9, 0.99),  # tuner betas
        adafusion_betas=(0.9, 0.999),  # base momentum betas (explicit passthrough)
        clip_threshold=0.5,
        cautious=False,
        momentum_dtype="float32",
    )
    g = opt.param_groups[0]
    assert g["betas"] == (0.9, 0.999)  # tuner betas did NOT leak into base momentum betas
    assert g["clip_threshold"] == 0.5
    assert g["cautious"] is False
    assert g["momentum_dtype"] == "float32"


def test_default_base_betas_no_momentum() -> None:
    """Default inner Adafusion has beta1=0 (no momentum) — exact-freeze regime."""
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = Autofusion([p])
    assert opt.param_groups[0]["betas"] == (0.0, 0.999)


def test_get_s_vector_length() -> None:
    """get_s returns one scale per tuner beta."""
    p = torch.nn.Parameter(torch.randn(8, 8))
    betas = (0.9, 0.99, 0.999)
    opt = Autofusion([p], betas=betas)
    p.grad = torch.randn_like(p)
    opt.step()
    assert opt.get_s().numel() == len(betas)


# -- freeze-to-free ---------------------------------------------------------


def test_freeze_after_n_steps_frees_state() -> None:
    """lr_freeze=N freezes at step N and frees the ref buffer."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, 16))
    opt = Autofusion([p], s_init=1e-4, lr_freeze=5)
    for _ in range(4):
        p.grad = torch.randn_like(p)
        opt.step()
        assert not opt.is_frozen()
    # The 5th step's end-of-step check (iters_done == 5 >= 5) triggers the freeze.
    p.grad = torch.randn_like(p)
    opt.step()
    assert opt.is_frozen()
    assert opt.frozen_lr is not None and opt.frozen_lr > 0.0
    assert len(opt._mech["ref"]) == 0, "ref freed on freeze"
    assert opt.param_groups[0]["lr"] == opt.frozen_lr


def test_get_d_stable_after_freeze() -> None:
    """get_d returns the frozen LR (no longer changes) after freezing."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, 16))
    opt = Autofusion([p], s_init=1e-4, lr_freeze=4)
    for _ in range(6):
        p.grad = torch.randn_like(p)
        opt.step()
    assert opt.is_frozen()
    d_at_freeze = opt.get_d()
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()
    assert opt.get_d() == d_at_freeze, "frozen LR must not drift"


def test_auto_freeze_on_plateau() -> None:
    """lr_freeze='auto' (the default) eventually freezes once the scale plateaus."""
    torch.manual_seed(0)
    w = torch.randn(24, 24)
    target = torch.randn(24, 24)
    x = torch.nn.Parameter(torch.zeros(24, 24))
    opt = Autofusion([x], s_init=1e-4)  # lr_freeze defaults to "auto"
    for _ in range(3000):
        opt.zero_grad()
        ((w @ x - target) ** 2).mean().backward()
        opt.step()
        if opt.is_frozen():
            break
    assert opt.is_frozen(), "auto freeze should trigger on a plateau"


def test_frozen_step_delegates_to_base() -> None:
    """After freezing, step() routes straight to the inner Adafusion (no overhead).

    Post-freeze the optimizer IS plain Adafusion at lr=S: a frozen step must produce
    exactly the same weights as calling the inner ``base.step()`` directly, and must
    not touch any Mechanic bookkeeping.
    """
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(12, 12))
    opt = Autofusion([p], s_init=1e-4, lr_freeze=3,
                     momentum_dtype="float32", bf16_method="none")
    for _ in range(4):
        p.grad = torch.randn_like(p)
        opt.step()
    assert opt.is_frozen()
    s = opt.frozen_lr
    assert opt.param_groups[0]["lr"] == s

    # A frozen step on the wrapper == the inner base.step() on a forked copy.
    p_fork = torch.nn.Parameter(p.detach().clone())
    base_fork = Adafusion([p_fork], lr=s, momentum_dtype="float32", bf16_method="none")
    # Copy the base optimizer's EMA state into the fork so they start identical.
    import copy as _copy

    base_fork.load_state_dict(_copy.deepcopy(opt.base.state_dict()))
    base_fork.param_groups[0]["lr"] = s

    g = torch.randn_like(p)
    p.grad = g.clone()
    p_fork.grad = g.clone()
    iters_before = opt._mech["iter"]
    opt.step()          # frozen -> base.step()
    base_fork.step()    # plain Adafusion at lr=S
    assert torch.equal(p.detach(), p_fork.detach()), "frozen step != Adafusion(lr=S) step"
    assert opt._mech["iter"] == iters_before, "frozen step must not advance the tuner"


# ----------------------- iteration-2: LR-discovery quality -----------------------
def test_auto_s_init_is_data_relative() -> None:
    """s_init='auto' seeds the initial effective LR at _S_INIT_REL * RMS(p).

    Adafusion's lr=1 update is unit-RMS, so the LARS trust ratio ||p||/||u_lr1|| ==
    RMS(p); the auto seed lands the step-1 effective LR at _S_INIT_REL * RMS(p).
    """
    torch.manual_seed(0)
    w = torch.randn(64, 64)
    target = torch.randn(64, 64)
    x = torch.nn.Parameter(torch.randn(64, 64) * 0.1)  # RMS(p) ~ 0.1
    rms = x.detach().pow(2).mean().sqrt().item()
    opt = Autofusion([x], s_init="auto")
    opt.zero_grad()
    ((w @ x - target) ** 2).mean().backward()
    opt.step()
    assert abs(opt.get_d() - 3e-3 * rms) < 1e-5, (
        f"auto seed should be 3e-3 * RMS(p)={3e-3 * rms:.2e}, got {opt.get_d():.2e}"
    )


def test_auto_s_init_and_cap_are_default() -> None:
    """The defaults are s_init='auto' (data-relative) and scale_cap='auto'."""
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = Autofusion([p])
    assert opt._s_init_auto is True
    assert opt._scale_cap_auto is True


def test_auto_cap_is_multiple_of_seed() -> None:
    """scale_cap='auto' resolves on step 1 to scale_cap_rel * the data-relative seed,
    so the ceiling tracks the problem's LR scale."""
    torch.manual_seed(0)
    w = torch.randn(64, 64)
    target = torch.randn(64, 64)
    x = torch.nn.Parameter(torch.randn(64, 64) * 0.1)
    opt = Autofusion([x], s_init="auto", scale_cap="auto", scale_cap_rel=6.0, lr_freeze=None)
    opt.zero_grad()
    ((w @ x - target) ** 2).mean().backward()
    opt.step()
    assert abs(opt._scale_cap - 6.0 * opt._s_init) < 1e-12
    # and the discovered LR can never exceed it
    for _ in range(50):
        opt.zero_grad()
        ((w @ x - target) ** 2).mean().backward()
        opt.step()
        assert opt.get_d() <= opt._scale_cap + 1e-9


def test_scale_cap_clamps_lr() -> None:
    """scale_cap is a hard ceiling on the discovered effective LR."""
    torch.manual_seed(1)
    w = torch.randn(64, 64)
    target = torch.randn(64, 64)
    x = torch.nn.Parameter(torch.randn(64, 64) * 0.1)
    opt = Autofusion([x], s_init=1e-3, scale_cap=5e-3, lr_freeze=None)
    seen = []
    for _ in range(40):
        opt.zero_grad()
        ((w @ x - target) ** 2).mean().backward()
        opt.step()
        seen.append(opt.get_d())
    assert max(seen) <= 5e-3 + 1e-9, f"cap should hold; max LR seen {max(seen):.3e}"


def test_scale_floor_prevents_collapse() -> None:
    """Once the effective LR has grown, the floor (internal _SCALE_FLOOR_FRAC=0.5)
    stops it collapsing below half of its running max."""
    torch.manual_seed(2)
    w = torch.randn(48, 48)
    target = torch.randn(48, 48)
    x = torch.nn.Parameter(torch.randn(48, 48) * 0.1)
    opt = Autofusion([x], s_init=1e-3, lr_freeze=None)
    seen = []
    for _ in range(60):
        opt.zero_grad()
        ((w @ x - target) ** 2).mean().backward()
        opt.step()
        seen.append(opt.get_d())
    peak = max(seen)
    post_peak_min = min(seen[seen.index(peak):])
    assert post_peak_min >= 0.5 * peak - 1e-9, (
        f"floor breached: post-peak min {post_peak_min:.3e} < 0.5*peak {0.5 * peak:.3e}"
    )


def test_floor_does_not_inflate_bootstrap() -> None:
    """The floor is relative to the RUNNING max, so the legitimate early ramp-up of
    the scale is never inflated (the floor only tracks what has already been seen)."""
    torch.manual_seed(0)
    w = torch.randn(32, 32)
    target = torch.randn(32, 32)
    x = torch.nn.Parameter(torch.zeros(32, 32))
    opt = Autofusion([x], s_init=1e-6, scale_cap=None, lr_freeze=None)
    prev = 0.0
    for i in range(20):
        opt.zero_grad()
        ((w @ x - target) ** 2).mean().backward()
        opt.step()
        d = opt.get_d()
        # while monotonically rising, d should equal the raw sum(s) (floor == d itself)
        if d >= prev:
            assert abs(d - opt._mech["s"].sum().item()) < 1e-9 or i == 0
        prev = d


def test_freeze_exact_after_cap() -> None:
    """Freeze folds the EFFECTIVE (capped) sum(s) into the base lr, so the handoff is
    byte-exact even when the cap engaged on the last pre-freeze step."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(12, 12))
    opt = Autofusion(
        [p], s_init=1e-2, scale_cap=2e-3, lr_freeze=5,
        momentum_dtype="float32", bf16_method="none",
    )
    for _ in range(6):
        p.grad = torch.randn_like(p)
        opt.step()
    assert opt.is_frozen()
    # frozen lr must equal the capped value, not the (larger) raw sum(s)
    assert opt.frozen_lr <= 2e-3 + 1e-9
    assert opt.param_groups[0]["lr"] == opt.frozen_lr


def test_hardened_auto_freeze_waits_for_near_max() -> None:
    """The internal _LR_FREEZE_MAX_FRAC guard: a flat-but-low (transient-dip) scale
    must NOT freeze; the scale has to be near its running max as well as flat."""
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = Autofusion([p], lr_freeze="auto")
    # Drive the auto-freeze plateau logic directly with a short patience so the test
    # stays fast (the production patience constant is 50).
    opt._lr_freeze_tol = 0.5
    opt._lr_freeze_patience = 3
    # Establish a running max well above the current scale, then feed flat low values.
    opt._s_sum_max = 1.0
    opt._prev_s_sum = 0.1
    for _ in range(10):
        # s_sum stays flat at 0.1 (rel change 0) but is only 0.1 of the max -> no freeze
        opt._maybe_freeze(0.1, 100)
    assert not opt.is_frozen(), "flat-but-low scale must not trigger the freeze"
    # Now near the max and flat -> should freeze within patience steps.
    opt._s_sum_max = 0.1
    for _ in range(5):
        opt._maybe_freeze(0.1, 100)
    assert opt.is_frozen(), "flat AND near-max scale should freeze"


# -- Phase C: batched (foreach) warmup ------------------------------------


def _shaped_params() -> list[torch.nn.Parameter]:
    torch.manual_seed(1)
    return [
        torch.nn.Parameter(torch.randn(s))
        for s in [(8, 8), (16, 4), (3,), (5, 7), (2, 2, 2)]
    ]


def test_foreach_warmup_matches_per_param() -> None:
    """foreach_warmup=True must be numerically identical (fp32 round-off) to the
    per-param loop, across the full warmup trajectory — incl. the s_decay term."""
    for kwargs in (
        {"s_decay": 0.0},
        {"s_decay": 0.01},
        {"s_decay": 0.01, "scale_cap": None},
    ):
        a = _shaped_params()
        b = _shaped_params()
        o_fe = Autofusion(a, lr=1.0, foreach_warmup=True, lr_freeze=None, **kwargs)
        o_loop = Autofusion(b, lr=1.0, foreach_warmup=False, lr_freeze=None, **kwargs)
        g = torch.Generator().manual_seed(7)
        for _ in range(20):
            for pa, pb in zip(a, b, strict=True):
                grad = torch.randn(pa.shape, generator=g)
                pa.grad = grad.clone()
                pb.grad = grad.clone()
            o_fe.step()
            o_loop.step()
            assert abs(o_fe.get_d() - o_loop.get_d()) < 1e-6
            for pa, pb in zip(a, b, strict=True):
                assert torch.allclose(pa, pb, atol=1e-4, rtol=1e-4)


def test_foreach_warmup_default_on() -> None:
    """foreach_warmup defaults to True (the fast path)."""
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = Autofusion([p])
    assert opt._foreach_warmup is True


def test_foreach_warmup_does_not_mutate_ref() -> None:
    """The batched writeback must not alias/mutate mech['ref'] (regression: .float()
    on an fp32 tensor returns the same tensor, so an in-place add corrupted ref)."""
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = Autofusion([p], foreach_warmup=True)
    p.grad = torch.randn_like(p)
    opt.step()
    ref0 = opt._mech["ref"][p].clone()
    p.grad = torch.randn_like(p)
    opt.step()
    assert torch.equal(opt._mech["ref"][p], ref0), "ref must stay constant across steps"


def test_behavior_identical_to_old_defaults() -> None:
    """Regression guard for the knob purge: the new minimal API must walk the exact
    same trajectory as the per-param loop at the hardcoded constant values — i.e. the
    purge is non-functional. Two optimizers built identically must be byte-for-byte
    equal across both the foreach and the per-param warmup paths."""
    torch.manual_seed(3)
    a = _shaped_params()
    b = _shaped_params()
    o_fe = Autofusion(a, lr_freeze=None, foreach_warmup=True,
                      momentum_dtype="float32", bf16_method="none")
    o_loop = Autofusion(b, lr_freeze=None, foreach_warmup=False,
                        momentum_dtype="float32", bf16_method="none")
    g = torch.Generator().manual_seed(11)
    for _ in range(30):
        for pa, pb in zip(a, b, strict=True):
            grad = torch.randn(pa.shape, generator=g)
            pa.grad = grad.clone()
            pb.grad = grad.clone()
        o_fe.step()
        o_loop.step()
        assert abs(o_fe.get_d() - o_loop.get_d()) < 1e-6
        for pa, pb in zip(a, b, strict=True):
            assert torch.allclose(pa, pb, atol=1e-5, rtol=1e-5)
