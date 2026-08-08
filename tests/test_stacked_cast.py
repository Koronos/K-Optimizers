"""Regression guard for the per-parameter cast anti-pattern in the foreach buckets.

The batched paths build their stacked fp32 working tensors as
``torch.stack([...]).float()`` — one widening cast for the whole bucket. Writing it
the other way round, ``torch.stack([x.float() for x in xs])``, is numerically
identical (bf16/fp16 -> fp32 is exact widening) but launches **one cast kernel per
parameter**, which dominates the launch-bound regime foreach batching exists to fix
(measured 2026-08-07: 901 of 949 CUDA launches in a non-factored Adakaon bucket step;
removing it was ~3-4x on the scalar/mixed micro-benchmarks).

The property under test is structural, not numeric: the number of dtype-conversion
kernels a bucket step issues must not grow with the bucket size. Counting
``aten::_to_copy`` through a dispatch mode makes that exact and device-independent —
the anti-pattern is equally visible on CPU, so this needs no GPU.
"""

from __future__ import annotations

import linecache
import os
import sys
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from kaon import (
    ADOPT,
    AdaBelief,
    Adakaon,
    AdamP,
    AdaMuon,
    AdaPNM,
    KProdigy,
    Lion,
    Lookahead,
    ScheduleFree,
)

# Lookahead only stacks on a sync step, so make every step one.
OPTIMIZERS: list[tuple[type, dict[str, Any]]] = [
    (Adakaon, {}),
    (AdaMuon, {}),
    (ADOPT, {}),
    (AdaPNM, {}),
    (Lion, {}),
    (KProdigy, {}),
    (AdaBelief, {}),
    (AdamP, {}),
    (ScheduleFree, {}),
    (Lookahead, {"k": 1}),
]


def _stacking_call_site() -> bool:
    """True if the innermost ``kaon`` frame on the stack is executing a ``torch.stack`` line.

    Optimizers legitimately cast per param *outside* the batched stacking sites (KProdigy's
    d-estimation bookkeeping, the per-param fallback path). Attributing each cast to its
    call site keeps this test aimed at the stacking sites only, instead of turning into a
    brittle total-kernel budget.
    """
    frame = sys._getframe(2)
    while frame is not None:
        name = frame.f_code.co_filename
        if f"kaon{os.sep}" in name and "_python_dispatch" not in name:
            line = linecache.getline(name, frame.f_lineno)
            # A multi-line stack(...) reports the executing element line, so scan a small
            # window back to the opening call.
            window = "".join(
                linecache.getline(name, n) for n in range(max(1, frame.f_lineno - 3), frame.f_lineno + 1)
            )
            return "torch.stack" in line or "torch.stack" in window
        frame = frame.f_back
    return False


class _CastCounter(TorchDispatchMode):
    """Counts dtype-conversion kernels (``aten::_to_copy``) raised from a ``torch.stack`` site."""

    def __init__(self) -> None:
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001, ANN204
        if func is torch.ops.aten._to_copy.default and _stacking_call_site():
            self.count += 1
        return func(*args, **(kwargs or {}))


def _casts_per_step(cls: type, extra: dict[str, Any], n_params: int, shape: tuple[int, ...]) -> int:
    """Dtype-conversion kernels issued by one steady-state ``step()`` of an ``n_params`` bucket.

    bf16 params + stochastic rounding is deliberate: it is the one write-back branch
    that does not itself cast per param, so any growth with ``n_params`` comes from the
    stacking sites this test guards. The first step is discarded (state init).
    """
    bag = [nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16)) for _ in range(n_params)]
    opt = cls(bag, lr=1e-3, bf16_method="stochastic_rounding", weight_decay=0.01,
              foreach=True, **extra)
    for _ in range(2):
        for p in bag:
            p.grad = torch.randn(shape, dtype=torch.bfloat16)
        opt.step()
    for p in bag:
        p.grad = torch.randn(shape, dtype=torch.bfloat16)
    with _CastCounter() as counter:
        opt.step()
    return counter.count


@pytest.mark.parametrize("shape", [(), (8,), (4, 6)], ids=["scalar0d", "flat1d", "factored2d"])
@pytest.mark.parametrize(("cls", "extra"), OPTIMIZERS, ids=[c.__name__ for c, _ in OPTIMIZERS])
def test_bucket_casts_do_not_scale_with_bucket_size(
    cls: type, extra: dict[str, Any], shape: tuple[int, ...]
) -> None:
    """A 4x larger bucket must not issue more cast kernels — casts are per bucket, not per param."""
    small = _casts_per_step(cls, extra, 8, shape)
    large = _casts_per_step(cls, extra, 32, shape)
    assert small > 0, f"{cls.__name__} {shape}: no stacked cast observed — test no longer covers it"
    assert large == small, (
        f"{cls.__name__} shape={shape}: cast kernels at torch.stack sites grew with bucket "
        f"size ({small} at N=8 -> {large} at N=32) — a per-parameter "
        "torch.stack([x.float() for x in xs]) has crept back in"
    )
