"""Shared pytest fixtures and helpers for the adakaon test suite."""

from __future__ import annotations

import sys
from collections.abc import Iterable

import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    """Seed all relevant RNGs at the start of each test for reproducibility."""
    torch.manual_seed(0xC0DE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0xC0DE)


@pytest.fixture
def toy_mlp() -> torch.nn.Module:
    """A tiny two-layer MLP used by smoke tests.

    Shapes are chosen so that the factored path (>= 2-D weights) and the
    1-D fallback path (biases) both get exercised.
    """
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 8),
    )


@pytest.fixture
def random_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """A deterministic input/target pair compatible with ``toy_mlp``."""
    x = torch.randn(4, 16)
    y = torch.randn(4, 8)
    return x, y


def train_steps(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> None:
    """Run ``opt`` on ``model`` for one optimization step per batch.

    Uses MSE loss. The model and optimizer are mutated in place.
    """
    for x, y in batches:
        opt.zero_grad()
        (model(x) - y).pow(2).mean().backward()
        opt.step()


def skip_if_no_cuda() -> None:
    """Skip the current test if CUDA is unavailable."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def skip_if_missing(module_name: str) -> None:
    """Skip the current test if an optional dependency is missing."""
    pytest.importorskip(module_name)


def python_opcodes_executed(fn, *args, **kwargs):
    """Count the bytecode instructions executed inside Python frames during ``fn(*args)``.

    A deterministic, load-independent stand-in for "how much Python actually runs per
    call" — the thing wall-clock witness benchmarks were trying to measure. Used to lock
    the pointer-witness scans at C level: ``tuple(map(Tensor.data_ptr, plist))`` runs its
    per-element loop inside ``map``/``tuple``, so the executed-opcode count does not move
    with ``len(plist)`` at all; a generator expression, a list comprehension and a plain
    ``for`` loop each execute ~15 opcodes PER element, so the count grows linearly.

    Counting *opcodes* rather than Python *frames* is what makes this catch every rewrite:
    a genexpr does create a frame per resume, but a ``<listcomp>`` creates exactly one
    frame however long the list is, and an inlined ``for`` loop creates none — both would
    slip past a frame counter, and neither slips past this.

    ``fn`` runs under ``sys.settrace`` with opcode tracing on, which is one to two orders
    of magnitude slower than normal: keep ``fn`` cheap, and never time anything from
    inside. Any previously installed tracer is restored on the way out.
    """
    count = 0

    def _local(frame, event, arg):
        nonlocal count
        if event == "opcode":
            count += 1
        return _local

    def _global(frame, event, arg):
        if event == "call":
            frame.f_trace_opcodes = True
            return _local
        return None

    previous = sys.gettrace()
    sys.settrace(_global)
    try:
        fn(*args, **kwargs)
    finally:
        sys.settrace(previous)
    return count


def assert_scan_runs_in_c(call, small_arg, big_arg, what, slack=8):
    """Assert ``call`` executes the same amount of Python bytecode for a small and a large
    input — i.e. that its per-element loop runs in C (``map``) and not in Python.

    ``slack`` allows a handful of opcodes of constant overhead (a size-dependent branch
    that does not itself loop); any real Python-level scan overshoots it by two orders of
    magnitude, so the bound is not a tuning knob. No wall clock is involved, so this
    cannot flake under CPU contention.
    """
    small = python_opcodes_executed(call, small_arg)
    big = python_opcodes_executed(call, big_arg)
    assert abs(big - small) <= slack, (
        f"{what} executes {big} Python opcodes on the large input vs {small} on the small "
        "one — the count scales with the input, so the C-level `map` scan has been "
        f"replaced by a Python loop (generator expression, list comprehension or `for`)"
    )
    return small
