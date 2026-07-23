"""Tests for the ``auto_lr`` flag — composable parameter-free LR via update-space
DoWG (``kaon._autolr.AutoLRTuner``), on Adakaon (the first base).

CPU/fp32. Validates: off == plain, LR discovery + loss reduction from the
decades-low seed, the lr<1 ignore+warn safety, the stability-edge guard (spike
backoff + isolated-spike self-correction + confirmed-edge automatic freeze +
from-above recovery + nan-grad path), survival of a per-step harness lr clobber,
and checkpoint round-trip. There is deliberately NO freeze knob: the tuner
decides by itself when discovery is done (confirmed edge contact).
"""

from __future__ import annotations

import copy
import math
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


def _manual_step(model, x, y, opt, grad_factor: float = 1.0) -> float:
    """One step with direct grad access (to inject spikes)."""
    opt.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    if grad_factor != 1.0:
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(grad_factor)
    opt.step()
    return float(loss.detach())


def test_off_is_plain() -> None:
    model, x, y = _tiny_problem()
    opt = Adakaon(model.parameters(), lr=1e-3)
    assert opt._autolr is None
    assert not opt.is_frozen()
    assert opt.get_d() == 1e-3
    opt.step(_closure(model, x, y, opt))  # no crash, plain path


def test_discovers_and_reduces_loss() -> None:
    model, x, y = _tiny_problem()
    opt = Adakaon(model.parameters(), betas=(0.0, 0.999), auto_lr=True)
    c = _closure(model, x, y, opt)
    with torch.no_grad():
        start_loss = float(torch.nn.functional.mse_loss(model(x), y))
    opt.step(c)
    d0 = opt.get_d()  # the data-relative seed after the first step
    for _ in range(300):
        opt.step(c)
    d_end = opt.get_d()
    with torch.no_grad():
        end_loss = float(torch.nn.functional.mse_loss(model(x), y))
    assert d_end > 10 * d0, f"discovered LR should climb decades from the seed ({d0:g} -> {d_end:g})"
    assert end_loss < 0.6 * start_loss, f"loss should fall ({start_loss:g} -> {end_loss:g})"


def test_lr_below_one_warns_and_is_ignored() -> None:
    model, x, y = _tiny_problem()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        opt = Adakaon(model.parameters(), lr=1e-4, betas=(0.0, 0.999), auto_lr=True)
        c = _closure(model, x, y, opt)
        opt.step(c)  # the tuner allocates + warns on the first real step
        assert any("ignored" in str(wi.message) for wi in w), "a leftover lr<1 must warn"
    with torch.no_grad():
        start = float(torch.nn.functional.mse_loss(model(x), y))
    for _ in range(300):
        opt.step(c)
    with torch.no_grad():
        end = float(torch.nn.functional.mse_loss(model(x), y))
    # If lr=1e-4 had been honored, the effective LR would be ~1e-4× the discovered
    # scale and the loss would barely move. It must train normally.
    assert end < 0.6 * start, f"lr<1 must be ignored, not shrink the LR ({start:g} -> {end:g})"


def test_spike_backs_off_and_self_corrects() -> None:
    # An isolated grad spike (bad batch) = one edge contact: S halves, the DoWG
    # accumulators re-anchor, and — without a confirming second contact — S resumes
    # climbing. No freeze.
    model, x, y = _tiny_problem(seed=3)
    opt = Adakaon(model.parameters(), betas=(0.0, 0.999), auto_lr=True)
    for _ in range(30):
        _manual_step(model, x, y, opt)
    t = opt._autolr
    s_before = t.S
    _manual_step(model, x, y, opt, grad_factor=50.0)
    assert t._edge == s_before, "the contact level must be recorded"
    assert not opt.is_frozen(), "a single spike must not freeze"
    # the self-consistent restart re-derives ~the backed-off S on the contact step
    assert s_before > t.S, f"S must back off from the contact ({s_before:g} -> {t.S:g})"
    assert 0.2 * s_before < t.S, f"backoff should be ~x0.5, not a collapse ({s_before:g} -> {t.S:g})"
    s_backed = t.S
    for _ in range(60):
        _manual_step(model, x, y, opt)
    assert not opt.is_frozen(), "an isolated spike must self-correct, not freeze"
    assert 1.2 * s_backed < t.S, f"S must resume climbing after the false positive ({s_backed:g} -> {t.S:g})"


def test_confirmed_edge_freezes_below_edge() -> None:
    # A second contact within the edge band confirms the edge: the tuner freezes
    # ITSELF at edge*backoff (just BELOW the edge — the safe side of the overshoot
    # cliff) and frees the reference buffers. No knob involved.
    from kaon._autolr import _BACKOFF
    model, x, y = _tiny_problem(seed=4)
    opt = Adakaon(model.parameters(), betas=(0.0, 0.999), auto_lr=True)
    for _ in range(30):
        _manual_step(model, x, y, opt)
    t = opt._autolr
    t._edge = t.S  # simulate a prior contact at the current level
    s_contact = t.S
    _manual_step(model, x, y, opt, grad_factor=50.0)
    assert opt.is_frozen(), "a repeat contact within the band must freeze"
    assert math.isclose(t.frozen_lr, s_contact * _BACKOFF, rel_tol=1e-6), (
        f"freeze must lock just below the edge ({s_contact:g} -> {t.frozen_lr:g})"
    )
    assert t._x0 == {}, "the per-param reference buffers must be freed on freeze"
    assert opt.param_groups[0]["lr"] == t.frozen_lr, "group lr folded to the frozen S"
    d_frozen = opt.get_d()
    _manual_step(model, x, y, opt)
    assert opt.get_d() == d_frozen, "frozen LR must not move"


def test_recovers_from_far_above() -> None:
    # The Anima failure mode: the LR lands far ABOVE the stability edge. The guard
    # must convert the blowup into backoffs — losses stay finite and the LR comes
    # back down instead of ratcheting up with the divergence.
    model, x, y = _tiny_problem(seed=5)
    opt = Adakaon(model.parameters(), betas=(0.0, 0.999), auto_lr=True)
    for _ in range(10):
        _manual_step(model, x, y, opt)
    t = opt._autolr
    t.S = 1.0  # hurl the LR far above the edge for this problem
    losses = [_manual_step(model, x, y, opt) for _ in range(300)]
    assert all(math.isfinite(v) for v in losses), "training must not diverge to inf/nan"
    assert opt.get_d() < 0.1, f"the LR must come back down from far above (got {opt.get_d():g})"
    assert losses[-1] < losses[0], "training must actually recover, not just survive"
    for p in model.parameters():
        assert torch.isfinite(p).all(), "params must stay finite through the recovery"


def test_nonfinite_grads_back_off_without_stepping() -> None:
    # inf/nan grads: unambiguous contact, but stepping would poison the base state.
    model, x, y = _tiny_problem(seed=6)
    opt = Adakaon(model.parameters(), betas=(0.0, 0.999), auto_lr=True)
    for _ in range(20):
        _manual_step(model, x, y, opt)
    t = opt._autolr
    s_before = t.S
    before = [p.detach().clone() for p in model.parameters()]
    opt.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p.grad.fill_(float("inf"))
    opt.step()
    assert s_before > t.S, "non-finite grads must back the LR off"
    for p, b in zip(model.parameters(), before):
        assert torch.equal(p, b), "the poison gradient must not be applied"
    _manual_step(model, x, y, opt)  # and training continues normally


def test_survives_harness_lr_clobber_while_adapting() -> None:
    # External trainers (renga-flow, kohya, the control battery) rewrite
    # group["lr"] every step. auto_lr must impose its own LR each iteration or the
    # base steps at the harness's lr and diverges.
    model, x, y = _tiny_problem(seed=7)
    opt = Adakaon(model.parameters(), betas=(0.0, 0.999), auto_lr=True)
    c = _closure(model, x, y, opt)
    with torch.no_grad():
        start = float(torch.nn.functional.mse_loss(model(x), y))
    for _ in range(300):
        for g in opt.param_groups:      # the harness clobber, every step
            g["lr"] = 1.0
        opt.step(c)
    with torch.no_grad():
        end = float(torch.nn.functional.mse_loss(model(x), y))
    assert end < 0.6 * start, f"must survive per-step lr clobber, not diverge ({start:g} -> {end:g})"


def test_survives_harness_lr_clobber_when_frozen() -> None:
    model, x, y = _tiny_problem(seed=7)
    opt = Adakaon(model.parameters(), betas=(0.0, 0.999), auto_lr=True)
    for _ in range(30):
        _manual_step(model, x, y, opt)
    t = opt._autolr
    t._edge = t.S
    _manual_step(model, x, y, opt, grad_factor=50.0)  # confirmed edge -> frozen
    assert opt.is_frozen()
    frozen = t.frozen_lr
    for g in opt.param_groups:
        g["lr"] = 1.0                   # the harness clobber
    _manual_step(model, x, y, opt)
    assert opt.param_groups[0]["lr"] == frozen, "frozen LR must be re-imposed over the clobber"


def test_state_dict_resume() -> None:
    model, x, y = _tiny_problem(seed=5)
    opt = Adakaon(model.parameters(), betas=(0.0, 0.999), auto_lr=True)
    c = _closure(model, x, y, opt)
    for _ in range(40):
        opt.step(c)
    sd = copy.deepcopy(opt.state_dict())
    d_at_save = opt.get_d()

    opt2 = Adakaon(model.parameters(), betas=(0.0, 0.999), auto_lr=True)
    opt2.load_state_dict(sd)
    assert abs(opt2.get_d() - d_at_save) < 1e-9 * max(d_at_save, 1e-9)
    assert opt2._autolr._t == 40
    assert opt2._autolr._gema == opt._autolr._gema, "the edge-guard EMA must round-trip"
    assert opt2._autolr._edge == opt._autolr._edge
    opt2.step(c)  # continues without throwing
    assert opt2._autolr._t == 41
