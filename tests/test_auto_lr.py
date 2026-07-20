"""Tests for the ``auto_lr`` flag (Mechanic tuner) on Adakaon — the first base.

CPU/fp32. Validates: off == plain, LR discovery, the lr<1 ignore+warn safety,
freeze-to-free + the tuner-scalar hedge, numerical parity with the standalone
``Autokaon`` class, and checkpoint round-trip.
"""

from __future__ import annotations

import copy
import warnings

import torch

from kaon import Adakaon, Autokaon


def _tiny_problem(seed: int = 0) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4))
    x = torch.randn(32, 8)
    y = torch.randn(32, 4)
    return model, x, y


def _closure(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, opt: torch.optim.Optimizer):
    def c():
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        return loss
    return c


def test_off_is_plain() -> None:
    model, x, y = _tiny_problem()
    opt = Adakaon(model.parameters(), lr=1e-3)
    assert opt._mech_tuner is None
    assert not opt.is_frozen()
    assert opt.get_d() == 1e-3
    opt.step(_closure(model, x, y, opt))  # no crash, plain path


def test_discovers_and_reduces_loss() -> None:
    model, x, y = _tiny_problem()
    opt = Adakaon(model.parameters(), betas=(0.0, 0.999), auto_lr=True, auto_lr_freeze=None)
    c = _closure(model, x, y, opt)
    with torch.no_grad():
        start_loss = float(torch.nn.functional.mse_loss(model(x), y))
    d0 = opt.get_d()
    for _ in range(120):
        opt.step(c)
    d_end = opt.get_d()
    with torch.no_grad():
        end_loss = float(torch.nn.functional.mse_loss(model(x), y))
    assert d_end > d0 * 5, f"D should rise from the seed ({d0:g} -> {d_end:g})"
    assert end_loss < 0.5 * start_loss, f"loss should fall ({start_loss:g} -> {end_loss:g})"


def test_lr_below_one_warns_and_is_ignored() -> None:
    model, x, y = _tiny_problem()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        opt = Adakaon(model.parameters(), lr=1e-4, betas=(0.0, 0.999), auto_lr=True, auto_lr_freeze=None)
        assert any("ignored" in str(wi.message) for wi in w), "a leftover lr<1 must warn"
    # The base lr must have been forced to 1.0 (tuner owns the scale).
    assert opt.param_groups[0]["lr"] == 1.0
    c = _closure(model, x, y, opt)
    with torch.no_grad():
        start = float(torch.nn.functional.mse_loss(model(x), y))
    for _ in range(120):
        opt.step(c)
    with torch.no_grad():
        end = float(torch.nn.functional.mse_loss(model(x), y))
    # If lr=1e-4 had been used as a multiplier, the effective LR would be ~1e-4×
    # the discovered scale and the loss would barely move. It must train normally.
    assert end < 0.5 * start, f"lr<1 must be ignored, not shrink the LR ({start:g} -> {end:g})"


def test_freeze_after_n_frees_ref_keeps_scalars() -> None:
    model, x, y = _tiny_problem()
    opt = Adakaon(model.parameters(), betas=(0.0, 0.999), auto_lr=True, auto_lr_freeze=8)
    c = _closure(model, x, y, opt)
    for _ in range(8):
        opt.step(c)
    assert opt.is_frozen()
    tuner = opt._mech_tuner
    # HEDGE: ref (the big per-param buffer) is freed, tuner scalars are KEPT.
    assert tuner._mech["ref"] == {}, "ref must be freed on freeze"
    assert tuner._mech["s"].numel() == 6, "tuner scalars must be kept (hedge)"
    assert tuner.frozen_lr is not None and tuner.frozen_lr > 0
    # group lr folded to the discovered S; post-freeze step is plain base.
    assert opt.param_groups[0]["lr"] == tuner.frozen_lr
    d_frozen = opt.get_d()
    opt.step(c)
    assert opt.get_d() == d_frozen, "frozen LR must not move"


def test_parity_with_autokaon() -> None:
    # Identical init + identical data => Adakaon(auto_lr=True) must match the
    # standalone Autokaon (same Mechanic core), per-param path both sides.
    model_a, x, y = _tiny_problem(seed=3)
    model_b = copy.deepcopy(model_a)

    opt_a = Adakaon(model_a.parameters(), betas=(0.0, 0.999), auto_lr=True, auto_lr_freeze=None)
    opt_b = Autokaon(model_b.parameters(), lr_freeze=None, foreach_warmup=False)  # adakaon_betas default (0.0,0.999)

    ca = _closure(model_a, x, y, opt_a)
    cb = _closure(model_b, x, y, opt_b)
    for _ in range(60):
        opt_a.step(ca)
        opt_b.step(cb)

    assert abs(opt_a.get_d() - opt_b.get_d()) < 1e-6 * max(opt_b.get_d(), 1e-8), \
        f"discovered LR must match Autokaon ({opt_a.get_d():g} vs {opt_b.get_d():g})"
    for pa, pb in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert torch.allclose(pa, pb, rtol=1e-4, atol=1e-6)


def test_survives_harness_lr_clobber() -> None:
    # External trainers (renga-flow, kohya, the control battery) rewrite
    # group["lr"] every step. auto_lr must impose its own effective LR each
    # iteration — a one-time fold into group["lr"] gets clobbered, and a frozen
    # base run at the harness's lr (here 1.0, ~2000x too hot) diverges.
    model, x, y = _tiny_problem(seed=7)
    opt = Adakaon(model.parameters(), betas=(0.0, 0.999), auto_lr=True, auto_lr_freeze=10)
    c = _closure(model, x, y, opt)
    with torch.no_grad():
        start = float(torch.nn.functional.mse_loss(model(x), y))
    for _ in range(150):
        for g in opt.param_groups:      # the harness clobber, every step
            g["lr"] = 1.0
        opt.step(c)
    assert opt.is_frozen()
    with torch.no_grad():
        end = float(torch.nn.functional.mse_loss(model(x), y))
    assert end < 0.5 * start, f"must survive per-step lr clobber, not diverge ({start:g} -> {end:g})"


def test_state_dict_resume() -> None:
    model, x, y = _tiny_problem(seed=5)
    opt = Adakaon(model.parameters(), betas=(0.0, 0.999), auto_lr=True, auto_lr_freeze=None)
    c = _closure(model, x, y, opt)
    for _ in range(40):
        opt.step(c)
    sd = copy.deepcopy(opt.state_dict())
    d_at_save = opt.get_d()

    # Fresh optimizer over the SAME model, resume mid-warmup.
    opt2 = Adakaon(model.parameters(), betas=(0.0, 0.999), auto_lr=True, auto_lr_freeze=None)
    opt2.load_state_dict(sd)
    assert abs(opt2.get_d() - d_at_save) < 1e-6 * max(d_at_save, 1e-8)
    assert opt2._mech_tuner._mech["iter"] == 40
    # continuing must not throw and must keep adapting
    opt2.step(c)
    assert opt2._mech_tuner._mech["iter"] == 41
