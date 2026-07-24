"""CPU tests for autonomous AutoLR.

The deterministic toy host makes detector and rollback behavior observable
without involving a trainer or a loss signal.  One public Adakaon smoke test
keeps the integration path covered as well.
"""

from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Iterable

import pytest
import torch

from kaon import Adakaon
from kaon._autolr import _ADAPT_MAX_STEPS, _BACKOFF, AutoLRMixin


class _ToyOptimizer(AutoLRMixin, torch.optim.Optimizer):
    """Small stateful SGD host used to inspect AutoLR transactions."""

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        *,
        auto_lr: bool = True,
        auto_lr_scale: float = 1.0,
        auto_lr_fuse_rel: float = 20.0,
        auto_lr_d0: float | None = 1e-3,
    ) -> None:
        super().__init__(params, {"lr": 1.0})
        self.reset_calls = 0
        self._init_autolr(
            auto_lr,
            auto_lr_scale,
            auto_lr_fuse_rel,
            auto_lr_d0,
        )

    def _step_impl(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                momentum = state.setdefault("momentum", torch.zeros_like(param))
                momentum.mul_(0.5).add_(param.grad)
                state["steps"] = state.get("steps", 0) + 1
                param.add_(momentum, alpha=-float(group["lr"]))
        return loss

    def _autolr_reset_base_state(self) -> None:
        self.reset_calls += 1
        AutoLRMixin._autolr_reset_base_state(self)

    def state_dict(self):
        return self._autolr_state_dict(super().state_dict())

    def load_state_dict(self, state_dict):
        self._autolr_load(
            state_dict,
            lambda state: torch.optim.Optimizer.load_state_dict(self, state),
        )


def _make(
    values: tuple[float, ...] = (1.0,),
    **kwargs,
) -> tuple[list[torch.nn.Parameter], _ToyOptimizer]:
    params = [torch.nn.Parameter(torch.tensor([value], dtype=torch.float32)) for value in values]
    return params, _ToyOptimizer(params, **kwargs)


def _step(opt: _ToyOptimizer, grads: tuple[float | None, ...]) -> None:
    params = [param for group in opt.param_groups for param in group["params"]]
    assert len(params) == len(grads)
    for param, grad in zip(params, grads):
        param.grad = None if grad is None else torch.full_like(param, grad)
    opt.step()


def _first_spike(opt: _ToyOptimizer, *, healthy_steps: int = 4) -> float:
    for _ in range(healthy_steps):
        _step(opt, (1.0,))
    contact = float(opt._autolr.S)
    _step(opt, (10.0,))
    return contact


def test_discovery_runs_without_loss_or_report_loss() -> None:
    _, opt = _make()
    for _ in range(12):
        _step(opt, (1.0,))
    assert opt._autolr._seed == pytest.approx(1e-3)
    assert opt.get_d() > opt._autolr._seed
    assert opt._autolr._level_base == pytest.approx([0.0] * 8)


def test_report_loss_is_a_warn_once_noop() -> None:
    params_a, opt_a = _make()
    params_b, opt_b = _make()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for index in range(12):
            opt_a.report_loss(float(index))
            opt_a.report_loss(float("nan"))
            _step(opt_a, (1.0,))
            _step(opt_b, (1.0,))

    deprecations = [item for item in caught if issubclass(item.category, DeprecationWarning)]
    assert len(deprecations) == 1
    assert torch.equal(params_a[0], params_b[0])
    assert opt_a.get_d() == opt_b.get_d()
    assert opt_a._autolr.state_blob() == opt_b._autolr.state_blob()


def test_snapshots_parameters_that_receive_gradients_late() -> None:
    params, opt = _make((1.0, 2.0))
    initial_late = params[1].detach().clone()
    _step(opt, (1.0, None))
    assert params[1] in opt._autolr._x0
    assert torch.equal(opt._autolr._x0[params[1]], initial_late)

    for _ in range(3):
        _step(opt, (1.0, None))
    _step(opt, (1.0, 1.0))
    assert not torch.equal(params[1], initial_late)
    _step(opt, (10.0, 10.0))
    assert torch.equal(params[1], initial_late)


def test_abrupt_spike_rolls_back_parameters_and_base_state_exactly() -> None:
    params, opt = _make()
    initial = params[0].detach().clone()
    contact = _first_spike(opt)

    assert opt._autolr._edge == contact
    assert opt.get_d() == pytest.approx(contact * _BACKOFF)
    assert torch.equal(params[0], initial)
    assert len(opt.state) == 0
    assert opt.reset_calls == 1
    assert opt._autolr._rbar == 0.0
    assert opt._autolr._contacts == 1
    assert not opt.is_frozen()


def test_fixed_baseline_detects_gradual_growth() -> None:
    params, opt = _make(auto_lr_d0=1e-6, auto_lr_fuse_rel=1e12)
    initial = params[0].detach().clone()
    for _ in range(8):
        _step(opt, (1.0,))
    for _ in range(7):
        _step(opt, (3.0,))
        assert opt._autolr._edge is None
    _step(opt, (3.0,))

    assert opt._autolr._edge is not None
    assert opt._autolr._contacts == 1
    assert opt._autolr._level_base == pytest.approx([0.0] * 8)
    assert opt._autolr._level_window == []
    assert torch.equal(params[0], initial)


def test_second_comparable_contact_freezes_below_edge() -> None:
    params, opt = _make()
    initial = params[0].detach().clone()
    edge = _first_spike(opt)
    _step(opt, (10.0,))

    assert opt.is_frozen()
    assert opt._autolr.freeze_reason == "edge_confirmed"
    assert opt.get_d() == pytest.approx(edge * _BACKOFF * _BACKOFF)
    assert opt._autolr._contacts == 2
    assert opt._autolr._x0 == {}
    assert not torch.equal(params[0], initial), "the confirming finite step runs at the safe LR"


def test_no_edge_freezes_at_fuse() -> None:
    _, opt = _make(auto_lr_fuse_rel=1.0)
    for _ in range(64):
        _step(opt, (1.0,))
        if opt.is_frozen():
            break
    assert opt.is_frozen()
    assert opt._autolr.freeze_reason == "fuse_bound"
    assert opt.get_d() == pytest.approx(opt._autolr._fuse)
    assert opt._autolr._t < _ADAPT_MAX_STEPS


def test_high_d0_has_bounded_headroom_and_cannot_bypass_safety_fuse() -> None:
    _, reference = _make(auto_lr_d0=None)
    _step(reference, (1.0,))
    expected_fuse = reference._autolr._fuse

    _, high = _make(auto_lr_d0=10.0)
    with pytest.warns(UserWarning, match="clamped"):
        _step(high, (1.0,))
    assert high._autolr._fuse == pytest.approx(4.0 * expected_fuse)
    assert high._autolr._seed == pytest.approx(4.0 * expected_fuse)
    assert high.get_d() == pytest.approx(4.0 * expected_fuse)
    assert high.is_frozen()
    assert high._autolr.freeze_reason == "fuse_bound"


def test_no_edge_freezes_at_192_step_budget() -> None:
    _, opt = _make(auto_lr_d0=1e-12, auto_lr_fuse_rel=1e30)
    for _ in range(_ADAPT_MAX_STEPS):
        _step(opt, (1.0,))
    assert opt.is_frozen()
    assert opt._autolr._t == _ADAPT_MAX_STEPS
    assert opt._autolr.freeze_reason == "budget_bound"


def test_persistent_nonfinite_gradients_back_off_once_and_skip() -> None:
    params, opt = _make()
    for _ in range(4):
        _step(opt, (1.0,))
    before = float(opt._autolr.S)
    initial = torch.tensor([1.0])

    with pytest.warns(UserWarning, match="non-finite"):
        for _ in range(4):
            _step(opt, (float("nan"),))

    assert opt.get_d() == pytest.approx(before * _BACKOFF)
    assert opt._autolr._contacts == 1
    assert opt.reset_calls == 1
    assert torch.equal(params[0], initial)
    assert len(opt.state) == 0

    _step(opt, (1.0,))
    assert opt._autolr._nan_run == 0
    _step(opt, (float("inf"),))
    assert opt.get_d() == pytest.approx(before * _BACKOFF)
    assert opt._autolr._contacts == 1, "later poison must not create an LR ratchet"


def test_auto_lr_scale_applies_to_each_dowg_estimate_without_bypassing_fuse() -> None:
    _, normal = _make(auto_lr_scale=1.0, auto_lr_fuse_rel=1.0)
    _, doubled = _make(auto_lr_scale=2.0, auto_lr_fuse_rel=1.0)
    _step(normal, (1.0,))
    _step(doubled, (1.0,))
    assert doubled.get_d() > normal.get_d(), "scale must affect the live DoWG estimate"

    for opt in (normal, doubled):
        for _ in range(64):
            if opt.is_frozen():
                break
            _step(opt, (1.0,))
        assert opt.is_frozen()
        assert opt._autolr.freeze_reason == "fuse_bound"
        assert opt.get_d() == pytest.approx(opt._autolr._fuse)


def test_harness_lr_overwrite_cannot_change_adapting_or_frozen_lr() -> None:
    params_a, guarded = _make()
    params_b, reference = _make()
    for _ in range(4):
        guarded.param_groups[0]["lr"] = 99.0
        _step(guarded, (1.0,))
        _step(reference, (1.0,))
    assert torch.equal(params_a[0], params_b[0])

    _step(guarded, (10.0,))
    _step(guarded, (10.0,))
    assert guarded.is_frozen()
    frozen = guarded.get_d()
    guarded.param_groups[0]["lr"] = 99.0
    _step(guarded, (1.0,))
    assert guarded.param_groups[0]["lr"] == frozen


def test_074_checkpoint_round_trip_preserves_detector_and_freeze_reason() -> None:
    params, opt = _make(auto_lr_d0=1e-6, auto_lr_fuse_rel=1e12)
    for _ in range(8):
        _step(opt, (1.0,))
    for _ in range(3):
        _step(opt, (2.0,))

    state = copy.deepcopy(opt.state_dict())
    params2, resumed = _make(auto_lr_d0=1e-6, auto_lr_fuse_rel=1e12)
    params2[0].data.copy_(params[0])
    resumed.load_state_dict(state)
    assert resumed._autolr._level_base == opt._autolr._level_base
    assert resumed._autolr._level_window == opt._autolr._level_window
    assert resumed._autolr._contacts == opt._autolr._contacts
    assert resumed._autolr._t == opt._autolr._t

    for _ in range(5):
        _step(opt, (2.0,))
        _step(resumed, (2.0,))
    assert torch.equal(params2[0], params[0])
    assert resumed._autolr.state_blob() == opt._autolr.state_blob()

    _step(opt, (10.0,))
    _step(resumed, (10.0,))
    assert resumed._autolr.freeze_reason == opt._autolr.freeze_reason


def test_073_checkpoint_load_ignores_legacy_probe_state() -> None:
    params, opt = _make()
    for _ in range(5):
        _step(opt, (1.0,))
    state = copy.deepcopy(opt.state_dict())
    blob = state["_autolr"]
    for key in (
        "version",
        "contacts",
        "nan_run",
        "nonfinite_backed_off",
        "nonfinite_warned",
        "level_base",
        "level_window",
        "freeze_reason",
    ):
        blob.pop(key)
    blob["probe"] = {
        "phase": "bisect",
        "lr": 0.123,
        "losses": [99.0],
    }

    params2, resumed = _make()
    params2[0].data.copy_(params[0])
    resumed.load_state_dict(state)
    assert not hasattr(resumed._autolr, "_probe")
    assert resumed.get_d() == resumed._autolr._seed
    assert resumed._autolr._t == 0
    assert torch.equal(params2[0], resumed._autolr._x0[params2[0]])
    assert len(resumed.state) == 0
    assert resumed.reset_calls == 1
    assert resumed._autolr._level_base == []
    assert resumed._autolr._level_window == []
    _step(resumed, (1.0,))


def test_frozen_checkpoint_reimposes_lr_on_load_and_next_step() -> None:
    _, opt = _make()
    _first_spike(opt)
    _step(opt, (10.0,))
    state = copy.deepcopy(opt.state_dict())

    _, resumed = _make()
    resumed.load_state_dict(state)
    assert resumed.is_frozen()
    assert resumed._autolr.freeze_reason == "edge_confirmed"
    assert resumed.param_groups[0]["lr"] == resumed.get_d()
    resumed.param_groups[0]["lr"] = 99.0
    _step(resumed, (1.0,))
    assert resumed.param_groups[0]["lr"] == resumed.get_d()


def test_off_path_is_plain_optimizer() -> None:
    params, opt = _make(auto_lr=False)
    opt.param_groups[0]["lr"] = 0.1
    _step(opt, (1.0,))
    assert opt._autolr is None
    assert not opt.is_frozen()
    assert opt.get_d() == 0.1
    assert params[0].item() == pytest.approx(0.9)


def test_adakaon_autonomous_cpu_smoke() -> None:
    torch.manual_seed(7)
    model = torch.nn.Linear(4, 2)
    inputs = torch.randn(8, 4)
    targets = torch.randn(8, 2)
    opt = Adakaon(model.parameters(), betas=(0.0, 0.999), auto_lr=True)

    initial_d = None
    for _ in range(12):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        opt.step()
        if initial_d is None:
            initial_d = opt.get_d()

    assert math.isfinite(opt.get_d())
    assert opt.get_d() > initial_d
