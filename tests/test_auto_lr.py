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


def test_ramp_floor_guarantees_climb() -> None:
    # Bare DoWG's climb is diffusive under noisy gradients (~250 steps/decade measured
    # on a real LoRA). Until the first edge contact, the geometric ramp floor must
    # guarantee >= _RAMP_GROWTH per update step regardless of the gradient statistics.
    from kaon._autolr import _RAMP_GROWTH
    model, x, y = _tiny_problem(seed=8)
    opt = Adakaon(model.parameters(), betas=(0.0, 0.999), auto_lr=True)
    _manual_step(model, x, y, opt)
    seed = opt._autolr._seed
    n = 40
    for _ in range(n):
        _manual_step(model, x, y, opt)
    if opt._autolr._edge is None and not opt.is_frozen():  # floor only promised pre-contact
        floor = seed * (_RAMP_GROWTH ** n) * 0.9
        assert opt.get_d() >= min(floor, opt._autolr._fuse), (
            f"ramp floor must guarantee the climb ({opt.get_d():g} < {floor:g})"
        )


def test_ramp_floor_off_after_contact() -> None:
    model, x, y = _tiny_problem(seed=8)
    opt = Adakaon(model.parameters(), betas=(0.0, 0.999), auto_lr=True)
    for _ in range(20):
        _manual_step(model, x, y, opt)
    t = opt._autolr
    assert t._ramp_on
    _manual_step(model, x, y, opt, grad_factor=50.0)  # first edge contact
    assert not t._ramp_on, "the range-test ramp must end at the first contact, for good"


def test_d0_overrides_seed() -> None:
    # auto_lr_d0 replaces the data-relative seed with an explicit starting LR.
    model, x, y = _tiny_problem(seed=9)
    opt = Adakaon(model.parameters(), betas=(0.0, 0.999), auto_lr=True, auto_lr_d0=5e-4)
    _manual_step(model, x, y, opt)
    t = opt._autolr
    assert t._seed == 5e-4
    assert opt.get_d() >= 1e-4, f"S must start at ~d0, not the data-relative seed ({opt.get_d():g})"


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
    t._edge = t.S       # simulate a prior contact at the current level
    t._ramp_on = False  # (a real prior contact also ends the ramp phase)
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


def test_ramp_contact_rolls_back_to_x0() -> None:
    # During the ramp (range-test) phase, an edge contact restores the exact x0
    # snapshot: the overshoot leaves NO trace on the params, the spike gradient is
    # never applied, and the grad-norm EMA keeps its healthy pre-spike baseline.
    model, x, y = _tiny_problem(seed=6)
    orig = [p.detach().clone() for p in model.parameters()]  # == x0 (taken pre-first-update)
    opt = Adakaon(model.parameters(), betas=(0.0, 0.999), auto_lr=True)
    for _ in range(20):
        _manual_step(model, x, y, opt)
    t = opt._autolr
    assert t._ramp_on
    s_before = t.S
    gema_before = t._gema
    _manual_step(model, x, y, opt, grad_factor=50.0)
    assert t.S == s_before * 0.5, "contact must back the LR off"
    assert not t._ramp_on
    assert t._edge == s_before
    for p, o in zip(model.parameters(), orig):
        assert torch.equal(p, o), "ramp-phase contact must restore the exact x0 snapshot"
    assert t._gema == gema_before, "the healthy pre-spike EMA must be kept (that state was restored)"
    _manual_step(model, x, y, opt)  # and training continues normally


def test_nonfinite_grads_roll_back_without_stepping() -> None:
    # inf/nan grads during the ramp: same rollback path — restore x0, never step.
    model, x, y = _tiny_problem(seed=6)
    orig = [p.detach().clone() for p in model.parameters()]
    opt = Adakaon(model.parameters(), betas=(0.0, 0.999), auto_lr=True)
    for _ in range(20):
        _manual_step(model, x, y, opt)
    t = opt._autolr
    s_before = t.S
    opt.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p.grad.fill_(float("inf"))
    opt.step()
    assert s_before > t.S, "non-finite grads must back the LR off"
    for p, o in zip(model.parameters(), orig):
        assert torch.equal(p, o), "the poison gradient must not be applied (params back at x0)"
    _manual_step(model, x, y, opt)  # and training continues normally


def test_nan_run_is_bounded() -> None:
    # A RUN of non-finite grads must not melt S (only the first counts as a contact)
    # and must warn once it is clearly not an LR problem; a finite step resets the run.
    model, x, y = _tiny_problem(seed=6)
    opt = Adakaon(model.parameters(), betas=(0.0, 0.999), auto_lr=True)
    for _ in range(20):
        _manual_step(model, x, y, opt)
    t = opt._autolr
    s_before = t.S

    def _nan_step():
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(x), y).backward()
        with torch.no_grad():
            for p in model.parameters():
                p.grad.fill_(float("nan"))
        opt.step()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _nan_step()
        s_after_first = t.S
        assert s_after_first == s_before * 0.5, "first nan of a run = one backoff"
        _nan_step()
        _nan_step()
        assert t.S == s_after_first, "repeat nans must NOT keep shrinking S"
        assert any("non-finite" in str(wi.message) for wi in w), "a nan run must warn"
    _manual_step(model, x, y, opt)  # finite step resumes training
    assert t._nan_run == 0, "a finite gradient must reset the run"


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
    t._ramp_on = False
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
    assert opt2._autolr._ramp_on == opt._autolr._ramp_on, "the ramp phase must round-trip"
    opt2.step(c)  # continues without throwing
    assert opt2._autolr._t == 41
