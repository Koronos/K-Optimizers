"""Tests for the bounded momentum LR corrector."""

from __future__ import annotations

import copy
import math

import pytest
import torch

from kaon import Adakaon, Lion, Nekaon
from kaon._fused_triton import HAS_TRITON


def _kwargs(cls):
    common = dict(
        lr_servo=True,
        momentum_dtype="float32",
        foreach=False,
        cautious=False,
        gradient_centralization=False,
        weight_decay=0.0,
    )
    if cls is Lion:
        common["betas"] = (0.9, 0.9)
    elif cls is Nekaon:
        common.update(betas=(0.5, 0.9), k=1.5)
    else:
        common["betas"] = (0.5, 0.9)
    return common


@pytest.mark.parametrize("cls", [Adakaon, Nekaon, Lion])
def test_servo_moves_up_on_alignment_without_parameter_state(cls) -> None:
    p = torch.nn.Parameter(torch.ones(8))
    opt = cls([p], lr=1e-3, **_kwargs(cls))
    for _ in range(4):
        p.grad = torch.ones_like(p)
        opt.step()
    assert opt.get_lr_servo_scale() > 1.0
    assert "_lr_servo" in opt.state_dict()
    owner = opt._momentum_owner() if cls is Nekaon else opt
    assert all("lr_servo" not in key for state in owner.state.values() for key in state)


@pytest.mark.parametrize("cls", [Adakaon, Nekaon, Lion])
def test_servo_moves_down_on_overshoot_alignment(cls) -> None:
    p = torch.nn.Parameter(torch.ones(8))
    opt = cls([p], lr=1e-3, **_kwargs(cls))
    for _ in range(3):
        p.grad = torch.ones_like(p)
        opt.step()
    p.grad = -torch.ones_like(p)
    opt.step()
    assert opt.get_lr_servo_scale() < 1.0


@pytest.mark.parametrize("cls", [Adakaon, Nekaon, Lion])
def test_auto_lr_and_servo_are_mutually_exclusive(cls) -> None:
    p = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError, match="mutually exclusive"):
        cls([p], auto_lr=True, **_kwargs(cls))


def test_servo_owns_lr_and_checkpoint_resumes_exactly() -> None:
    p = torch.nn.Parameter(torch.ones(4))
    opt = Lion([p], lr=1e-3, **_kwargs(Lion))
    for sign in (1.0, 1.0, -1.0, 1.0):
        p.grad = torch.full_like(p, sign)
        opt.step()
    saved = copy.deepcopy(opt.state_dict())
    q = torch.nn.Parameter(p.detach().clone())
    resumed = Lion([q], lr=9.0, **_kwargs(Lion))
    resumed.load_state_dict(saved)
    assert resumed.get_lr_servo_scale() == opt.get_lr_servo_scale()
    resumed.param_groups[0]["lr"] = 99.0
    for sign in (-1.0, 1.0, 1.0):
        p.grad = torch.full_like(p, sign)
        q.grad = torch.full_like(q, sign)
        opt.step()
        resumed.step()
    torch.testing.assert_close(q, p, rtol=0.0, atol=0.0)
    assert resumed.param_groups[0]["lr"] == opt.param_groups[0]["lr"]


def test_late_gradient_joins_without_extra_buffer() -> None:
    p = torch.nn.Parameter(torch.ones(4))
    late = torch.nn.Parameter(torch.ones(4))
    opt = Adakaon([p, late], lr=1e-3, **_kwargs(Adakaon))
    p.grad = torch.ones_like(p)
    opt.step()
    p.grad = torch.ones_like(p)
    late.grad = torch.ones_like(late)
    opt.step()
    assert "m" in opt.state[late]
    assert math.isfinite(opt.get_lr_servo_scale())


def test_servo_checkpoint_rejects_corruption() -> None:
    p = torch.nn.Parameter(torch.ones(2))
    opt = Lion([p], lr=1e-3, **_kwargs(Lion))
    saved = opt.state_dict()
    saved["_lr_servo"]["groups"][0]["log_scale"] = float("nan")
    with pytest.raises(ValueError, match="invalid controller state"):
        opt.load_state_dict(saved)


def test_old_checkpoint_uses_saved_lr_as_servo_prior() -> None:
    p = torch.nn.Parameter(torch.ones(2))
    fixed = Lion([p], lr=3e-4, lr_servo=False, **{k: v for k, v in _kwargs(Lion).items() if k != "lr_servo"})
    saved = fixed.state_dict()
    q = torch.nn.Parameter(torch.ones(2))
    servo = Lion([q], lr=9e-3, **_kwargs(Lion))
    servo.load_state_dict(saved)
    assert servo.param_groups[0]["lr"] == pytest.approx(3e-4)
    assert servo.get_lr_servo_scale() == 1.0


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_TRITON,
    reason="fused LR servo parity requires CUDA + Triton",
)
@pytest.mark.parametrize("cls", [Adakaon, Nekaon])
def test_fused_and_native_servo_remain_equivalent(cls) -> None:
    torch.manual_seed(8)
    a = torch.nn.Parameter(torch.randn(8, 16, device="cuda"))
    b = torch.nn.Parameter(a.detach().clone())
    kwargs = _kwargs(cls)
    native = cls([a], lr=2e-3, fused=False, **kwargs)
    fused = cls([b], lr=2e-3, fused=True, **kwargs)
    for _ in range(12):
        grad = torch.randn_like(a)
        a.grad = grad.clone()
        b.grad = grad.clone()
        native.step()
        fused.step()
    for opt in (native, fused):
        eval_fn = getattr(opt, "eval", None)
        if callable(eval_fn):
            eval_fn()
    torch.testing.assert_close(b, a, rtol=2e-5, atol=2e-5)
    assert fused.get_lr_servo_scale() == pytest.approx(native.get_lr_servo_scale(), rel=2e-5)
