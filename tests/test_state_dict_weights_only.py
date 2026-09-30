"""Every kaon optimizer's checkpoint loads under ``torch.load``'s default ``weights_only=True``.

``Optimizer.state_dict`` hands the per-parameter state dicts out BY REFERENCE, and once
``self.state`` is a ``WatchedState`` those are ``WatchedParamState`` objects, which pickle
through a ``GLOBAL builtins.dict`` opcode the weights-only unpickler refuses ("Unsupported
global: GLOBAL dict"). It broke Adakaon / AdaPNM / Nekaon / Lookahead checkpoints and, once
every foreach optimizer got the watch (0.7.18), all of them. ``state_dict()`` now emits
plain per-param dicts (``kaon._foreach_plan.plain_state_dict``).
"""
from __future__ import annotations

import warnings

import pytest
import torch

import kaon
from kaon._foreach_plan import WatchedState

CLASSES = ["ADOPT", "AdaBelief", "AdaMuon", "AdaPNM", "Adakaon", "AdamP", "KProdigy", "Lion",
           "Lookahead", "MSAM", "Nekaon", "Rakaon", "SAM", "ScheduleFree"]


def _stepped(name):
    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(s)) for s in [(4, 3), (4, 3), (5,)]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = getattr(kaon, name)(params, lr=1e-2)
        if hasattr(opt, "train"):
            opt.train()

        def closure():
            opt.zero_grad()
            loss = sum((p * p).sum() for p in params)
            loss.backward()
            return loss

        for _ in range(2):
            closure()
            if name == "SAM":
                opt.step(closure)
            else:
                opt.step()
        if name in ("Nekaon", "MSAM"):
            opt.eval()
    return params, opt


def _watched_states(opt):
    out = []
    while opt is not None:
        if type(getattr(opt, "state", None)) is WatchedState:
            out.append(opt.state)
        opt = getattr(opt, "inner", None)
    return out


@pytest.mark.parametrize("weights_only", [None, True, False], ids=["default", "wo", "full"])
@pytest.mark.parametrize("name", CLASSES)
def test_state_dict_round_trips_through_torch_load(name, weights_only, tmp_path):
    _params, opt = _stepped(name)
    watched_before = len(_watched_states(opt))
    path = tmp_path / "ck.pt"
    torch.save({"opt": opt.state_dict()}, path)
    kw = {} if weights_only is None else {"weights_only": weights_only}
    sd = torch.load(path, **kw)
    opt.load_state_dict(sd["opt"])
    # The live state is still watched after the load (and the checkpoint was plain dicts).
    assert len(_watched_states(opt)) == watched_before
    for st in sd["opt"].get("state", {}).values():
        assert type(st) is dict


def test_state_dict_keeps_the_tensor_references():
    """The plain copy is shallow: no tensor is cloned by ``state_dict()``."""
    params, opt = _stepped("AdaBelief")
    sd = opt.state_dict()
    live = opt.state[params[0]]
    saved = sd["state"][0]
    assert type(saved) is dict
    for k, v in live.items():
        if torch.is_tensor(v):
            assert saved[k] is v
