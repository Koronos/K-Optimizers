"""Safety tests for the quarantined AutoLR API."""

from __future__ import annotations

import copy

import pytest
import torch

from kaon import ADOPT, AdaBelief, Adakaon, AdamP, AdaMuon, AdaPNM, Lion, Nekaon


@pytest.mark.parametrize(
    "optimizer_cls",
    [Adakaon, AdaPNM, AdaBelief, AdamP, ADOPT, AdaMuon, Lion, Nekaon],
)
def test_auto_lr_fails_closed_before_training(optimizer_cls) -> None:
    param = torch.nn.Parameter(torch.ones(2, 2))
    original = param.detach().clone()
    with pytest.raises(RuntimeError, match="auto_lr=True is disabled"):
        optimizer_cls([param], auto_lr=True)
    torch.testing.assert_close(param, original)


def test_normal_optimizer_path_is_unchanged() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    opt = Adakaon([param], lr=1e-3, betas=(0.0, 0.999))
    param.grad = torch.ones_like(param)
    opt.step()
    assert param.item() < 1.0
    assert opt.get_d() == pytest.approx(1e-3)
    assert not opt.is_frozen()


def test_report_loss_is_a_deprecated_noop() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    opt = Adakaon([param], lr=1e-3)
    before = copy.deepcopy(opt.state_dict())
    with pytest.warns(DeprecationWarning, match="retired"):
        opt.report_loss(torch.tensor(123.0))
    assert opt.state_dict() == before


def test_legacy_checkpoint_drops_only_autolr_blob() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    source = Adakaon([param], lr=2e-3)
    legacy = copy.deepcopy(source.state_dict())
    legacy["_autolr"] = {"format": 2, "frozen_lr": 9.9}

    restored_param = torch.nn.Parameter(torch.tensor([1.0]))
    restored = Adakaon([restored_param], lr=1.0)
    with pytest.warns(RuntimeWarning, match="Ignoring retired AutoLR state"):
        restored.load_state_dict(legacy)
    assert restored.param_groups[0]["lr"] == pytest.approx(2e-3)
