"""Tests for Nekaon — Adakaon + k-step negative momentum-lookahead.

Nekaon is a tested preset over the MSAM wrapper (``norm="none"``, ``rho=-k``), so the
mechanism's own coverage lives in ``test_msam.py``; this file pins the preset contract.
"""

from __future__ import annotations

import torch

from kaon import Adakaon, Nekaon
from kaon.msam import MSAM


def _params(seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(4, 3, generator=g).requires_grad_(True),
            torch.randn(5, generator=g).requires_grad_(True)]


def _run(opt_params, opt, seeds=(1, 2, 3)):
    for seed in seeds:
        g = torch.Generator().manual_seed(seed)
        for p in opt_params:
            p.grad = torch.randn(p.shape, generator=g)
        opt.step()


def test_nekaon_is_the_msam_preset():
    """Nekaon(k) is bit-identical to MSAM(rho=-k, norm='none', wd=0.1) over Adakaon."""
    pa, pb = _params(), _params()
    a = Nekaon(pa, lr=1e-3, k=1.5, betas=(0.9, 0.999), momentum_dtype="float32", foreach=False)
    b = MSAM(pb, rho=-1.5, norm="none", lr=1e-3, weight_decay=0.1,
             betas=(0.9, 0.999), momentum_dtype="float32", foreach=False)
    _run(pa, a)
    _run(pb, b)
    for x, y in zip(pa, pb, strict=True):
        assert torch.equal(x.data, y.data)


def test_k_zero_is_plain_adakaon():
    """k=0 disables the lookahead — bit-identical to bare Adakaon (same wd)."""
    pa, pb = _params(), _params()
    a = Nekaon(pa, lr=1e-3, k=0.0, betas=(0.9, 0.999), momentum_dtype="float32", foreach=False)
    b = Adakaon(pb, lr=1e-3, weight_decay=0.1, betas=(0.9, 0.999),
                momentum_dtype="float32", foreach=False)
    _run(pa, a)
    _run(pb, b)
    for x, y in zip(pa, pb, strict=True):
        assert torch.equal(x.data, y.data)


def test_rejects_no_momentum_and_negative_k():
    try:
        Nekaon(_params(), betas=(0.0, 0.999))
        raise AssertionError("beta1=0 must be rejected")
    except ValueError:
        pass
    try:
        Nekaon(_params(), k=-1.0)
        raise AssertionError("k<0 must be rejected")
    except ValueError:
        pass


def test_eval_shows_true_weights_and_train_restores():
    pa = _params()
    opt = Nekaon(pa, lr=1e-3, k=1.5, betas=(0.9, 0.999), momentum_dtype="float32", foreach=False)
    _run(pa, opt, seeds=(1,))
    live = [p.data.clone() for p in pa]
    opt.eval()
    assert any(not torch.equal(a, w) for a, w in zip(pa, live, strict=True))  # was perturbed
    opt.train()
    for a, w in zip(pa, live, strict=True):
        assert torch.equal(a.data, w)


def test_low_vram_above_splits_momentum_by_tensor_size():
    """Tensors over the threshold route to a momentum-free (and lookahead-free) group —
    replaces the standalone NekaonAlloc PoC with one optimizer instance."""
    big, small = _params()  # (4,3)=numel 12 > threshold; (5,)=numel 5 <= threshold
    opt = Nekaon([big, small], lr=1e-3, k=1.5, betas=(0.9, 0.999),
                 momentum_dtype="float32", foreach=False, low_vram_above=10)
    _run([big, small], opt, seeds=(1,))
    state = opt.base_optimizer.state
    assert "m" in state[small]
    assert "m" not in state[big]
    big_group = next(g for g in opt.param_groups if any(p is big for p in g["params"]))
    assert big_group["lr"] == 1e-3 * 0.5
    assert big_group["betas"] == (0.0, 0.999)


def test_low_vram_above_none_is_unaffected():
    """Default (no low_vram_above) keeps momentum on every tensor, same as the base preset."""
    pa, pb = _params(), _params()
    a = Nekaon(pa, lr=1e-3, k=1.5, betas=(0.9, 0.999), momentum_dtype="float32", foreach=False)
    b = Nekaon(pb, lr=1e-3, k=1.5, betas=(0.9, 0.999), momentum_dtype="float32", foreach=False,
               low_vram_above=None)
    _run(pa, a)
    _run(pb, b)
    for x, y in zip(pa, pb, strict=True):
        assert torch.equal(x.data, y.data)


def test_low_vram_above_rejects_param_group_dicts():
    try:
        Nekaon([{"params": _params()}], low_vram_above=10)
        raise AssertionError("param-group dicts must be rejected with low_vram_above")
    except TypeError:
        pass
