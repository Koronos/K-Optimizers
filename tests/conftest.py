"""Shared pytest fixtures and helpers for the adafusion test suite."""

from __future__ import annotations

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
