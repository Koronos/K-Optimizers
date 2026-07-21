"""Tests for the ``auto_lr`` flag — composable parameter-free LR via update-space
DoWG (``kaon._autolr.AutoLRTuner``), on Adakaon (the first base).

CPU/fp32. Validates: off == plain, LR discovery + loss reduction, the lr<1
ignore+warn safety, freeze (frees the x0 refs, delegates to the base), survival of
a per-step harness lr clobber, and checkpoint round-trip.
"""

from __future__ import annotations

import copy
import warnings

import torch

from kaon import Adakaon


def _tiny_problem(seed: int = 0) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4))
    x = torch.randn(32, 8)
    y = torch.randn(32, 4)
    return model, x, y


def _closure(model, x, y, opt):
    def c():
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        return loss
    return c


def test_off_is_plain() -> None:
    model, x, y = _tiny_problem()
    opt = Adakaon(model.parameters(), lr=1e-3)
    assert opt._autolr is None
    assert not opt.is_frozen()
    assert opt.get_d() == 1e-3
    opt.step(_closure(model, x, y, opt))  # no crash, plain path


def test_discovers_and_reduces_loss() -> None:
    model, x, y = _tiny_problem()
    opt = Adakaon(model.parameters(), betas=(0.0, 0.999), auto_lr=True, auto_lr_freeze=None)
    c = _closure(model, x, y, opt)
    with torch.no_grad():
        start_loss = float(torch.nn.functional.mse_loss(model(x), y))
    opt.step(c)
    d0 = opt.get_d()  # the data-relative seed after the first step
    for _ in range(160):
        opt.step(c)
    d_end = opt.get_d()
    with torch.no_grad():
        end_loss = float(torch.nn.functional.mse_loss(model(x), y))
    assert d_end > d0, f"discovered LR should rise from the seed ({d0:g} -> {d_end:g})"
    assert end_loss < 0.6 * start_loss, f"loss should fall ({start_loss:g} -> {end_loss:g})"


def test_lr_below_one_warns_and_is_ignored() -> None:
    model, x, y = _tiny_problem()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        opt = Adakaon(model.parameters(), lr=1e-4, betas=(0.0, 0.999), auto_lr=True, auto_lr_freeze=None)
        c = _closure(model, x, y, opt)
        opt.step(c)  # the tuner allocates + warns on the first real step
        assert any("ignored" in str(wi.message) for wi in w), "a leftover lr<1 must warn"
    with torch.no_grad():
        start = float(torch.nn.functional.mse_loss(model(x), y))
    for _ in range(160):
        opt.step(c)
    with torch.no_grad():
        end = float(torch.nn.functional.mse_loss(model(x), y))
    # If lr=1e-4 had been honored, the effective LR would be ~1e-4× the discovered
    # scale and the loss would barely move. It must train normally.
    assert end < 0.6 * start, f"lr<1 must be ignored, not shrink the LR ({start:g} -> {end:g})"


def test_freeze_after_n_frees_refs() -> None:
    model, x, y = _tiny_problem()
    opt = Adakaon(model.parameters(), betas=(0.0, 0.999), auto_lr=True, auto_lr_freeze=8)
    c = _closure(model, x, y, opt)
    for _ in range(8):
        opt.step(c)
    assert opt.is_frozen()
    tuner = opt._autolr
    assert tuner._x0 == {}, "the per-param reference buffers must be freed on freeze"
    assert tuner.frozen_lr is not None and tuner.frozen_lr > 0
    assert opt.param_groups[0]["lr"] == tuner.frozen_lr, "group lr folded to the discovered S"
    d_frozen = opt.get_d()
    opt.step(c)
    assert opt.get_d() == d_frozen, "frozen LR must not move"


def test_auto_freeze_is_growth_ratio() -> None:
    # auto_lr_freeze="auto" (the default) freezes when the discovered LR has grown
    # _FREEZE_GROWTH× over its data-relative seed — a dimensionless, reparametrization-
    # invariant trigger (not an absolute step count).
    from kaon._autolr import _FREEZE_GROWTH
    model, x, y = _tiny_problem()
    opt = Adakaon(model.parameters(), betas=(0.0, 0.999), auto_lr=True, auto_lr_freeze="auto")
    c = _closure(model, x, y, opt)
    for _ in range(600):
        opt.step(c)
        if opt.is_frozen():
            break
    assert opt.is_frozen(), "auto freeze should fire once S grows ~10x over the seed"
    t = opt._autolr
    # frozen at ~ _FREEZE_GROWTH × seed (the growth ratio; may slightly overshoot in one step)
    assert t.frozen_lr >= _FREEZE_GROWTH * t._seed * 0.9
    assert t.frozen_lr <= _FREEZE_GROWTH * t._seed * 2.0


def test_survives_harness_lr_clobber() -> None:
    # External trainers (renga-flow, kohya, the control battery) rewrite
    # group["lr"] every step. auto_lr must impose its own LR each iteration — both
    # while adapting and after freeze — or the base steps at the harness's lr and
    # diverges.
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
    assert end < 0.6 * start, f"must survive per-step lr clobber, not diverge ({start:g} -> {end:g})"


def test_state_dict_resume() -> None:
    model, x, y = _tiny_problem(seed=5)
    opt = Adakaon(model.parameters(), betas=(0.0, 0.999), auto_lr=True, auto_lr_freeze=None)
    c = _closure(model, x, y, opt)
    for _ in range(40):
        opt.step(c)
    sd = copy.deepcopy(opt.state_dict())
    d_at_save = opt.get_d()

    opt2 = Adakaon(model.parameters(), betas=(0.0, 0.999), auto_lr=True, auto_lr_freeze=None)
    opt2.load_state_dict(sd)
    assert abs(opt2.get_d() - d_at_save) < 1e-9 * max(d_at_save, 1e-9)
    assert opt2._autolr._t == 40
    opt2.step(c)  # continues without throwing
    assert opt2._autolr._t == 41
