"""``load_state_dict`` param-group backfill — kaon 0.7.11 audit, mechanical batch.

``torch.optim.Optimizer.load_state_dict`` **replaces** each ``param_groups`` dict with
the checkpoint's (only ``params`` is carried over from the live optimizer), so any
hyperparameter added to an optimizer after a checkpoint was written simply vanishes
from the resumed group — the next ``step()`` then dies with ``KeyError``. This first
surfaced in AdaMuon with ``bias_correction`` (see ``test_adamuon.py``); the fix there
(backfill any key the checkpoint predates from ``self.defaults``, keeping whatever the
checkpoint *does* carry) is now applied to every other kaon optimizer that defines its
own ``load_state_dict``.

Covers, parametrized by optimizer:

* a checkpoint missing an existing group key resumes without ``KeyError`` and the
  resumed group falls back to the (fresh instance's) default for that key;
* a checkpoint that *does* carry a non-default value for that key is not clobbered by
  the backfill.

Plus two dedicated tests for :class:`~kaon.sam.SAM` (which drives ``first_step``/
``second_step`` instead of a plain ``step()``) and :class:`~kaon.lookahead.Lookahead`'s
own per-group keys (``k``/``alpha``/... are not part of the wrapped Adakaon's
``defaults``, only of the wrapper's own ``self.defaults`` added by this batch).

MSAM and Nekaon need no changes here: both keep their own hyperparameters (``rho``,
``norm``) as instance attributes, not per-group keys, and fully delegate
``load_state_dict`` to the inner optimizer — so there is nothing of their own to
backfill (see ``src/kaon/msam.py`` / ``src/kaon/nekaon.py``).
"""

from __future__ import annotations

import pytest
import torch

from kaon import ADOPT, AdaBelief, AdamP, AdaPNM, KProdigy, Lion, Lookahead, ScheduleFree
from kaon.sam import SAM

# (name, optimizer class, base kwargs, group key to probe, non-default value for it)
_STANDALONE_SPECS = [
    pytest.param("ADOPT", ADOPT, dict(lr=1e-2), "weight_decay", 0.07, id="ADOPT"),
    pytest.param("AdaBelief", AdaBelief, dict(lr=1e-2), "weight_decay", 0.07, id="AdaBelief"),
    pytest.param("AdamP", AdamP, dict(lr=1e-2), "weight_decay", 0.07, id="AdamP"),
    pytest.param("KProdigy", KProdigy, dict(lr=1.0), "weight_decay", 0.07, id="KProdigy"),
    pytest.param("AdaPNM", AdaPNM, dict(lr=1e-2), "weight_decay", 0.07, id="AdaPNM"),
    pytest.param("Lion", Lion, dict(lr=1e-3), "weight_decay", 0.07, id="Lion"),
    pytest.param("ScheduleFree", ScheduleFree, dict(lr=1e-2), "weight_decay", 0.07, id="ScheduleFree"),
    # k=1 forces a sync on every step, so a missing/backfilled ``alpha`` is exercised
    # immediately (Lookahead's default k=5 would only touch it every 5th step).
    pytest.param("Lookahead", Lookahead, dict(lr=1e-2, k=1), "alpha", 0.9, id="Lookahead"),
]


@pytest.mark.parametrize("name, cls, base_kwargs, key, alt_value", _STANDALONE_SPECS)
def test_load_state_dict_backfills_missing_group_key(name, cls, base_kwargs, key, alt_value):
    """A checkpoint predating ``key`` resumes without ``KeyError`` and gets the default."""
    p = torch.nn.Parameter(torch.randn(6, 6))
    opt = cls([p], **base_kwargs)
    p.grad = torch.randn(6, 6)
    opt.step()
    sd = opt.state_dict()
    for pg in sd["param_groups"]:
        pg.pop(key, None)

    q = torch.nn.Parameter(torch.randn(6, 6))
    opt2 = cls([q], **base_kwargs)
    opt2.load_state_dict(sd)  # must not raise

    g = opt2.param_groups[0]
    assert g[key] == opt2.defaults[key]

    q.grad = torch.randn(6, 6)
    opt2.step()  # the first post-resume step must not raise KeyError either


@pytest.mark.parametrize("name, cls, base_kwargs, key, alt_value", _STANDALONE_SPECS)
def test_load_state_dict_keeps_checkpoint_value(name, cls, base_kwargs, key, alt_value):
    """The defaults backfill must not clobber a value the checkpoint *does* carry."""
    kwargs_a = {**base_kwargs, key: alt_value}
    p = torch.nn.Parameter(torch.randn(6, 6))
    opt = cls([p], **kwargs_a)
    p.grad = torch.randn(6, 6)
    opt.step()
    sd = opt.state_dict()

    q = torch.nn.Parameter(torch.randn(6, 6))
    opt2 = cls([q], **base_kwargs)  # constructed with the OLD/default value
    opt2.load_state_dict(sd)

    assert opt2.param_groups[0][key] == alt_value


# ------------------------------------------------------------------------- SAM

def test_sam_load_state_dict_backfills_missing_rho():
    """SAM's own ``rho``/``adaptive`` live in the shared groups but not in the inner
    optimizer's ``defaults`` — this batch gives SAM its own ``self.defaults`` so they
    backfill the same way as everything else."""
    p = torch.nn.Parameter(torch.randn(6, 6))
    opt = SAM([p], rho=0.05, lr=1e-2)
    p.grad = torch.randn(6, 6)
    opt.first_step()
    p.grad = torch.randn(6, 6)
    opt.second_step()
    sd = opt.state_dict()
    for pg in sd["param_groups"]:
        pg.pop("rho", None)
        pg.pop("adaptive", None)

    q = torch.nn.Parameter(torch.randn(6, 6))
    opt2 = SAM([q], rho=0.05, lr=1e-2)
    opt2.load_state_dict(sd)  # must not raise

    g = opt2.param_groups[0]
    assert g["rho"] == 0.05
    assert g["adaptive"] is False

    q.grad = torch.randn(6, 6)
    opt2.first_step()
    q.grad = torch.randn(6, 6)
    opt2.second_step()  # must not raise KeyError


def test_sam_load_state_dict_keeps_checkpoint_rho():
    p = torch.nn.Parameter(torch.randn(6, 6))
    opt = SAM([p], rho=0.2, lr=1e-2)
    p.grad = torch.randn(6, 6)
    opt.first_step()
    p.grad = torch.randn(6, 6)
    opt.second_step()
    sd = opt.state_dict()

    q = torch.nn.Parameter(torch.randn(6, 6))
    opt2 = SAM([q], rho=0.05, lr=1e-2)  # different default
    opt2.load_state_dict(sd)

    assert opt2.param_groups[0]["rho"] == 0.2


# ------------------------------------------------------------------------- parity

@pytest.mark.parametrize("name, cls, base_kwargs, key, alt_value", _STANDALONE_SPECS)
def test_resume_from_full_checkpoint_is_bit_identical(name, cls, base_kwargs, key, alt_value):
    """A normal resume (no key missing) must be bit-identical to an uninterrupted run.

    Guards against the backfill loop itself perturbing anything when every key is
    already present (``setdefault`` must be a true no-op on the happy path).

    Momentum forced to ``float32`` storage: bf16 momentum otherwise goes through
    kaon's stochastic-rounding codec, whose *global* generator advances once per
    write and would drift between the two runs for a reason unrelated to this
    fix (unquantized bit-exactness is a stronger check for what's under test here
    anyway).
    """
    kwargs = {**base_kwargs, "momentum_dtype": "float32"}
    torch.manual_seed(0)
    base = torch.randn(6, 6)
    grads = [torch.randn(6, 6) * 0.1 for _ in range(6)]

    control = torch.nn.Parameter(base.clone())
    opt_c = cls([control], **kwargs)
    resumed = torch.nn.Parameter(base.clone())
    opt_r = cls([resumed], **kwargs)

    for g in grads[:3]:
        control.grad = g.clone()
        resumed.grad = g.clone()
        opt_c.step()
        opt_r.step()

    sd = opt_r.state_dict()
    opt_r2 = cls([resumed], **kwargs)
    opt_r2.load_state_dict(sd)

    for g in grads[3:]:
        control.grad = g.clone()
        resumed.grad = g.clone()
        opt_c.step()
        opt_r2.step()

    torch.testing.assert_close(control.detach(), resumed.detach(), rtol=0, atol=0)
