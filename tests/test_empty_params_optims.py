"""Zero-element parameters (``(5, 0)``, ``(0, 5)``, ``(0,)``) in ADOPT, AdaBelief and AdamP.

A zero-element weight has nothing to update, but it used to reach the step: AdamP's projection
divided by its fan-in / ``sqrt(0)`` (ZeroDivisionError), ``amax`` over an empty dim raised,
and the int8 / 4bit momentum codecs failed on the empty rows in all three. They are now left
out of the step like a parameter without a gradient; the other parameters are unaffected.
"""
from __future__ import annotations

import pytest
import torch

from kaon import ADOPT, AdaBelief, AdamP

CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])


def _run(cls, md, foreach, dev, empty_shape, with_empty, steps=3):
    torch.manual_seed(0)
    real = [torch.nn.Parameter(torch.randn(s, device=dev)) for s in [(4, 3), (4, 3), (3,)]]
    empties = ([torch.nn.Parameter(torch.randn(empty_shape, device=dev)) for _ in range(2)]
               if with_empty else [])
    opt = cls(real + empties, lr=1e-2, weight_decay=0.1, momentum_dtype=md, foreach=foreach)
    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        for p in real + empties:
            p.grad = torch.randn(p.shape, generator=g).to(dev)
        opt.step()
    return real, empties, opt


@pytest.mark.parametrize("cls", [ADOPT, AdaBelief, AdamP])
@pytest.mark.parametrize("md", ["float32", "bfloat16", "int8", "4bit"])
@pytest.mark.parametrize("empty_shape", [(5, 0), (0, 5), (0,)])
@pytest.mark.parametrize("foreach", [False, True])
@pytest.mark.parametrize("dev", DEVICES)
def test_empty_params_are_skipped(cls, md, empty_shape, foreach, dev):
    real, empties, opt = _run(cls, md, foreach, dev, empty_shape, with_empty=True)
    ref, _, _ = _run(cls, md, foreach, dev, empty_shape, with_empty=False)
    for a, b in zip(real, ref, strict=True):
        assert torch.equal(a, b)
    for p in empties:
        assert p.numel() == 0 and not opt.state[p]
