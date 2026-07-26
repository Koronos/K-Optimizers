"""CPU tests for continuous Mechanic AutoLR."""

from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Iterable

import pytest
import torch

from kaon import Adakaon, AutoLRMixin, Lion, Nekaon


class _ToyOptimizer(AutoLRMixin, torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        *,
        auto_lr: bool = True,
        auto_lr_scale: float = 1.0,
        auto_lr_fuse_rel: float = 20.0,
        auto_lr_d0: float | None = None,
    ) -> None:
        super().__init__(params, {"lr": 1.0})
        self._init_autolr(auto_lr, auto_lr_scale, auto_lr_fuse_rel, auto_lr_d0)

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

def _make(values=(1.0,), **kwargs):
    params = [torch.nn.Parameter(torch.tensor([value], dtype=torch.float32)) for value in values]
    return params, _ToyOptimizer(params, **kwargs)


def _step(opt: _ToyOptimizer, grads) -> None:
    params = [p for group in opt.param_groups for p in group["params"]]
    for param, grad in zip(params, grads, strict=True):
        param.grad = None if grad is None else torch.full_like(param, grad)
    opt.step()


def _assert_tuners_equal(left, right) -> None:
    assert left.get_d() == right.get_d()
    for name in ("_r", "_m", "_v", "_s"):
        assert getattr(left._autolr, name) == getattr(right._autolr, name)
    assert left._autolr._t == right._autolr._t
    assert left._autolr._last_h == right._autolr._last_h
    for lp, rp in zip(left._autolr._params(), right._autolr._params(), strict=True):
        assert torch.equal(left._autolr._x0[lp], right._autolr._x0[rp])
        assert torch.equal(left._autolr._delta[lp], right._autolr._delta[rp])


def test_first_step_is_seed_scaled_unit_update() -> None:
    params, opt = _make()
    _step(opt, (1.0,))
    assert 1.0 - params[0].item() == pytest.approx(1e-6, rel=0.05)
    assert opt.get_d() == pytest.approx(1e-6)
    assert opt.param_groups[0]["lr"] == opt.get_d()


@pytest.mark.parametrize(
    ("optimizer_cls", "betas", "extra"),
    [
        (Adakaon, (0.9, 0.999), {}),
        (Nekaon, (0.5, 0.999), {"k": 1.5}),
        (Lion, (0.9, 0.99), {}),
    ],
)
def test_first_autolr_update_matches_same_fixed_lr_with_cautious_weight_decay(
    optimizer_cls, betas, extra
) -> None:
    torch.manual_seed(11)
    auto_param = torch.nn.Parameter(torch.randn(8, 8))
    fixed_param = torch.nn.Parameter(auto_param.detach().clone())
    shared = {
        "betas": betas,
        "weight_decay": 0.1,
        "cautious": True,
        "gradient_centralization": True,
        "momentum_dtype": "float32",
        "foreach": False,
        **extra,
    }
    auto = optimizer_cls([auto_param], lr=1.0, auto_lr=True, **shared)
    fixed = optimizer_cls([fixed_param], lr=auto.get_d(), auto_lr=False, **shared)
    grad = torch.randn_like(auto_param)
    auto_param.grad = grad.clone()
    fixed_param.grad = grad.clone()
    auto.step()
    fixed.step()
    for optimizer in (auto, fixed):
        eval_fn = getattr(optimizer, "eval", None)
        if eval_fn is not None:
            eval_fn()
    torch.testing.assert_close(auto_param, fixed_param, rtol=0.0, atol=0.0)


def test_continuous_feedback_grows_without_freeze_or_horizon() -> None:
    _, opt = _make()
    initial = opt.get_d()
    for _ in range(300):
        _step(opt, (1.0,))
    assert opt.get_d() > initial
    assert opt._autolr._t == 300
    assert not opt.is_frozen()
    assert opt._autolr.freeze_reason is None


def test_auto_lr_scale_is_explicit_multiplier() -> None:
    _, normal = _make(auto_lr_scale=1.0)
    _, triple = _make(auto_lr_scale=3.0)
    assert triple.get_d() == pytest.approx(3.0 * normal.get_d())
    _step(normal, (1.0,))
    _step(triple, (1.0,))
    assert triple.get_d() == pytest.approx(3.0 * normal.get_d())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"auto_lr_scale": float("inf")},
        {"auto_lr_scale": float("nan")},
        {"auto_lr_fuse_rel": float("inf")},
        {"auto_lr_d0": float("inf")},
    ],
)
def test_constructor_rejects_nonfinite_compatibility_values(kwargs) -> None:
    with pytest.raises(ValueError, match="finite and > 0"):
        _make(**kwargs)


def test_d0_extremes_warn_and_produce_identical_trajectories() -> None:
    with pytest.warns(UserWarning, match="deprecated and ignored"):
        params_low, low = _make(auto_lr_d0=1e-30)
    with pytest.warns(UserWarning, match="deprecated and ignored"):
        params_high, high = _make(auto_lr_d0=1e6)
    for grad in (1.0, 0.5, -0.25, 0.75, 0.2):
        _step(low, (grad,))
        _step(high, (grad,))
    assert torch.equal(params_low[0], params_high[0])
    _assert_tuners_equal(low, high)


def test_fuse_rel_is_accepted_but_does_not_freeze_or_cap() -> None:
    _, opt = _make(auto_lr_fuse_rel=1e-9)
    for _ in range(40):
        _step(opt, (1.0,))
    assert opt.get_d() > 1e-6
    assert not opt.is_frozen()


def test_report_loss_is_warn_once_and_trajectory_noop() -> None:
    params_a, a = _make()
    params_b, b = _make()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(8):
            a.report_loss(99.0)
            a.report_loss(float("nan"))
            _step(a, (1.0,))
            _step(b, (1.0,))
    assert sum(issubclass(w.category, DeprecationWarning) for w in caught) == 1
    assert torch.equal(params_a[0], params_b[0])
    _assert_tuners_equal(a, b)


def test_anchor_includes_parameter_receiving_gradient_late() -> None:
    params, opt = _make((1.0, 2.0))
    _step(opt, (1.0, None))
    assert torch.equal(opt._autolr._x0[params[1]], torch.tensor([2.0]))
    _step(opt, (1.0, 1.0))
    assert params[1].item() < 2.0


def test_nonfinite_gradients_skip_without_touching_live_or_base_state() -> None:
    params, opt = _make()
    for _ in range(4):
        _step(opt, (1.0,))
    before_param = params[0].detach().clone()
    before_momentum = opt.state[params[0]]["momentum"].clone()
    before_scale = opt.get_d()
    before_t = opt._autolr._t
    with pytest.warns(UserWarning, match="non-finite"):
        _step(opt, (float("nan"),))
    _step(opt, (float("inf"),))
    assert torch.equal(params[0], before_param)
    assert torch.equal(opt.state[params[0]]["momentum"], before_momentum)
    assert opt.get_d() == before_scale
    assert opt._autolr._t == before_t
    assert opt._autolr._nonfinite_steps == 2


def test_harness_lr_overwrite_is_ignored() -> None:
    params_a, a = _make()
    params_b, b = _make()
    for _ in range(6):
        a.param_groups[0]["lr"] = 99.0
        _step(a, (1.0,))
        _step(b, (1.0,))
    assert torch.equal(params_a[0], params_b[0])
    _assert_tuners_equal(a, b)


def test_checkpoint_round_trip_resumes_exactly() -> None:
    params, opt = _make()
    for grad in (1.0, 0.5, -0.25, 0.75):
        _step(opt, (grad,))
    state = copy.deepcopy(opt.state_dict())
    params2, resumed = _make()
    params2[0].data.copy_(params[0])
    resumed.load_state_dict(state)
    _assert_tuners_equal(opt, resumed)

    for grad in (0.2, -0.4, 1.1):
        _step(opt, (grad,))
        _step(resumed, (grad,))
    assert torch.equal(params[0], params2[0])
    _assert_tuners_equal(opt, resumed)


def test_old_dowg_checkpoint_fails_closed() -> None:
    _, opt = _make()
    state = opt.state_dict()
    state["_autolr"] = {"version": 2, "S": 1e-3, "x0": []}
    _, resumed = _make()
    with pytest.raises(ValueError, match="retired probe/DoWG"):
        resumed.load_state_dict(state)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("scale", float("nan"), "invalid scale"),
        ("v", [-1.0] * 6, "negative v"),
        ("s", [0.0] * 6, "invalid s"),
    ],
)
def test_corrupt_mechanic_scalar_checkpoint_fails_closed(field, value, match) -> None:
    _, opt = _make()
    _step(opt, (1.0,))
    state = copy.deepcopy(opt.state_dict())
    state["_autolr"][field] = value
    _, resumed = _make()
    with pytest.raises(ValueError, match=match):
        resumed.load_state_dict(state)


def test_corrupt_mechanic_trajectory_checkpoint_fails_closed() -> None:
    _, opt = _make()
    _step(opt, (1.0,))
    state = copy.deepcopy(opt.state_dict())
    state["_autolr"]["delta"][0].fill_(float("nan"))
    _, resumed = _make()
    with pytest.raises(ValueError, match="non-finite tensors"):
        resumed.load_state_dict(state)


def test_checkpoint_rejects_overflowed_applied_scale() -> None:
    _, opt = _make()
    state = copy.deepcopy(opt.state_dict())
    state["_autolr"]["scale"] = 1e308
    state["_autolr"]["s"] = [1.0] * 6
    _, resumed = _make()
    with pytest.raises(ValueError, match="invalid applied scale"):
        resumed.load_state_dict(state)


def test_oversized_tensor_uses_unstacked_memory_path(monkeypatch) -> None:
    param = torch.nn.Parameter(torch.zeros(2_000_001))
    opt = _ToyOptimizer([param])
    original_stack = torch.stack

    def guarded_stack(tensors, *args, **kwargs):
        values = list(tensors)
        assert not any(value.numel() > 2_000_000 for value in values)
        return original_stack(values, *args, **kwargs)

    monkeypatch.setattr(torch, "stack", guarded_stack)
    param.grad = torch.ones_like(param)
    opt.step()
    assert torch.isfinite(param).all()


def test_off_path_is_plain_optimizer() -> None:
    params, opt = _make(auto_lr=False)
    opt.param_groups[0]["lr"] = 0.1
    _step(opt, (1.0,))
    assert params[0].item() == pytest.approx(0.9)
    assert opt.get_d() == 0.1
    assert not opt.is_frozen()


def test_adakaon_autonomous_cpu_smoke() -> None:
    torch.manual_seed(7)
    model = torch.nn.Linear(4, 2)
    inputs = torch.randn(8, 4)
    targets = torch.randn(8, 2)
    opt = Adakaon(
        model.parameters(),
        betas=(0.0, 0.999),
        auto_lr=True,
    )
    initial = opt.get_d()
    for _ in range(20):
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(inputs), targets).backward()
        opt.step()
    assert math.isfinite(opt.get_d())
    assert opt.get_d() > initial
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_adakaon_bf16_safe_seed_does_not_quantize_feedback_to_zero() -> None:
    param = torch.ones(8, 8, dtype=torch.bfloat16, requires_grad=True)
    opt = Adakaon(
        [param],
        lr=1.0,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        momentum_dtype="bfloat16",
        bf16_method="stochastic_rounding",
        cautious=False,
        gradient_centralization=False,
        foreach=False,
        auto_lr=True,
    )
    for _ in range(64):
        param.grad = torch.ones_like(param)
        opt.step()
    assert opt.get_d() > 1e-6
    assert opt._autolr._last_h != 0.0
    assert any(bool(delta.abs().max() > 0) for delta in opt._autolr._delta.values())
