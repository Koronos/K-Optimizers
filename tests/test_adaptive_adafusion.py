"""Tests for AdaptiveAdafusion (Mechanic LR tuner on Adafusion, freeze-to-free).

Covers the base Mechanic behavior, the back-compat ``AdafusionProdigy`` alias,
``store_delta`` agreement, and the new ``lr_freeze`` (int / "auto") handoff that
turns the optimizer into plain Adafusion after warmup.
"""

from __future__ import annotations

import copy

import torch

from koptim import AdafusionProdigy, AdaptiveAdafusion
from koptim.adafusion import Adafusion

from .conftest import train_steps


def test_alias_is_same_class() -> None:
    """AdafusionProdigy is kept importable as a back-compat alias."""
    assert AdafusionProdigy is AdaptiveAdafusion


def test_scale_starts_at_seed_then_rises() -> None:
    """The discovered effective LR seeds near ``s_init`` and grows with training."""
    torch.manual_seed(0)
    w = torch.randn(32, 32)
    target = torch.randn(32, 32)
    x = torch.nn.Parameter(torch.zeros(32, 32))
    opt = AdaptiveAdafusion([x], s_init=1e-6, store_delta=True)
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
    opt = AdaptiveAdafusion(toy_mlp.parameters(), s_init=1e-4)
    before = (toy_mlp(x) - y).pow(2).mean().item()
    train_steps(toy_mlp, opt, [random_batch] * 50)
    after = (toy_mlp(x) - y).pow(2).mean().item()
    assert after < before, f"loss should drop: {before:.4f} -> {after:.4f}"


def test_store_delta_modes_agree(
    toy_mlp: torch.nn.Module, random_batch: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """store_delta True/False reach comparable loss (the on-the-fly Delta is valid)."""
    x, y = random_batch
    m1 = toy_mlp
    m2 = copy.deepcopy(toy_mlp)
    o1 = AdaptiveAdafusion(m1.parameters(), s_init=1e-4, store_delta=True)
    o2 = AdaptiveAdafusion(m2.parameters(), s_init=1e-4, store_delta=False)
    train_steps(m1, o1, [random_batch] * 40)
    train_steps(m2, o2, [random_batch] * 40)
    l1 = (m1(x) - y).pow(2).mean().item()
    l2 = (m2(x) - y).pow(2).mean().item()
    assert abs(l1 - l2) < 0.2 * max(l1, l2) + 1e-6, f"modes diverged: {l1} vs {l2}"


def test_store_delta_defaults_off() -> None:
    """store_delta defaults to False (reference Mechanic default; ref-only memory)."""
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = AdaptiveAdafusion([p])
    p.grad = torch.randn_like(p)
    opt.step()
    assert len(opt._mech["delta"]) == 0, "no delta buffer should be allocated by default"


def test_forwards_adafusion_kwargs() -> None:
    """Adafusion knobs (clip_threshold, cautious, momentum_dtype) forward to base."""
    p = torch.nn.Parameter(torch.randn(16, 16))
    opt = AdaptiveAdafusion(
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
    opt = AdaptiveAdafusion([p])
    assert opt.param_groups[0]["betas"] == (0.0, 0.999)


def test_get_s_vector_length() -> None:
    """get_s returns one scale per tuner beta."""
    p = torch.nn.Parameter(torch.randn(8, 8))
    betas = (0.9, 0.99, 0.999)
    opt = AdaptiveAdafusion([p], betas=betas)
    p.grad = torch.randn_like(p)
    opt.step()
    assert opt.get_s().numel() == len(betas)


# -- freeze-to-free ---------------------------------------------------------


def test_freeze_after_n_steps_frees_state() -> None:
    """lr_freeze=N freezes at step N and frees ref/delta buffers."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, 16))
    opt = AdaptiveAdafusion([p], s_init=1e-4, store_delta=True, lr_freeze=5)
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
    assert len(opt._mech["delta"]) == 0, "delta freed on freeze"
    assert opt.param_groups[0]["lr"] == opt.frozen_lr


def test_get_d_stable_after_freeze() -> None:
    """get_d returns the frozen LR (no longer changes) after freezing."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, 16))
    opt = AdaptiveAdafusion([p], s_init=1e-4, lr_freeze=4)
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
    """lr_freeze='auto' eventually freezes once the scale plateaus."""
    torch.manual_seed(0)
    w = torch.randn(24, 24)
    target = torch.randn(24, 24)
    x = torch.nn.Parameter(torch.zeros(24, 24))
    opt = AdaptiveAdafusion(
        [x], s_init=1e-4, lr_freeze="auto", lr_freeze_tol=0.05, lr_freeze_patience=10
    )
    for _ in range(400):
        opt.zero_grad()
        ((w @ x - target) ** 2).mean().backward()
        opt.step()
        if opt.is_frozen():
            break
    assert opt.is_frozen(), "auto freeze should trigger on a plateau within 400 steps"


def test_frozen_step_delegates_to_base() -> None:
    """After freezing, step() routes straight to the inner Adafusion (no overhead).

    Post-freeze the optimizer IS plain Adafusion at lr=S: a frozen step must produce
    exactly the same weights as calling the inner ``base.step()`` directly, and must
    not touch any Mechanic bookkeeping.
    """
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(12, 12))
    opt = AdaptiveAdafusion([p], s_init=1e-4, lr_freeze=3,
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
