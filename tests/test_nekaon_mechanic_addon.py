"""Integration audit for the private external Mechanic wrapper and Nekaon.

The strict xfails are executable blocker specifications.  They must turn XPASS only after the
external wrapper preserves Nekaon's true/live-weight lifecycle; constructor compatibility alone
is not sufficient.
"""

from __future__ import annotations

import copy

import pytest
import torch

from kaon import Nekaon
from kaon._mechanic_addon import MechanicAddon


def _nekaon(p: torch.Tensor) -> Nekaon:
    return Nekaon(
        [p],
        lr=1e-3,
        k=1.5,
        betas=(0.5, 0.999),
        weight_decay=0.0,
        momentum_dtype="bfloat16",
        cautious=False,
        foreach=False,
    )


def _shim_wrap(p: torch.Tensor, *, guard: bool = True) -> tuple[Nekaon, MechanicAddon]:
    inner = _nekaon(p)
    # Audit-only shim: the public composition currently fails before this point.
    inner.defaults = inner.inner.defaults
    return inner, MechanicAddon(inner, guard=guard)


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="MSAM/Nekaon does not expose Optimizer.defaults to the external wrapper",
)
def test_public_wrapper_construction_is_supported():
    p = torch.tensor([1.0], requires_grad=True)
    MechanicAddon(_nekaon(p), guard=True)


@pytest.mark.xfail(
    strict=True,
    reason="MechanicAddon does not delegate Nekaon's eval/train weight views",
)
def test_wrapper_delegates_eval_and_train():
    p = torch.tensor([1.0], requires_grad=True)
    _inner, opt = _shim_wrap(p)
    assert callable(opt.eval)
    assert callable(opt.train)


@pytest.mark.xfail(
    strict=True,
    reason="outer rescaling leaves Nekaon's cached unit-LR climb unscaled",
)
def test_eval_true_iterate_stays_on_the_mechanic_scaled_trajectory():
    p = torch.tensor([1.0], requires_grad=True)
    initial = p.detach().clone()
    inner, opt = _shim_wrap(p)
    p.grad = torch.ones_like(p)
    opt.step()
    live_distance = float((p.detach() - initial).abs().max())

    inner.eval()
    true_distance = float((p.detach() - initial).abs().max())
    inner.train()

    # A scale of ~1e-6 cannot legitimately hide a O(1) true-iterate displacement.
    assert true_distance <= max(10.0 * live_distance, 1e-4)


def test_audit_shim_forces_shared_base_lr_to_one_and_toggles_exactly():
    p = torch.tensor([1.0], requires_grad=True)
    inner, opt = _shim_wrap(p)
    assert opt.param_groups is inner.param_groups
    assert all(group["lr"] == 1.0 for group in opt.param_groups)
    p.grad = torch.ones_like(p)
    opt.step()
    live = p.detach().clone()
    inner.eval()
    inner.train()
    torch.testing.assert_close(p, live, rtol=0.0, atol=1e-7)


@pytest.mark.xfail(
    strict=True,
    reason="checkpointing through the outer wrapper has no supported eval/train lifecycle",
)
def test_checkpoint_resume_preserves_true_and_live_views():
    p = torch.tensor([1.0], requires_grad=True)
    inner, opt = _shim_wrap(p)
    p.grad = torch.ones_like(p)
    opt.step()
    opt.eval()
    saved_param = p.detach().clone()
    saved_state = copy.deepcopy(opt.state_dict())

    q = saved_param.clone().requires_grad_(True)
    _inner2, resumed = _shim_wrap(q)
    resumed.load_state_dict(saved_state)
    resumed.train()
    assert torch.isfinite(q).all()
