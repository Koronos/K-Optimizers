"""The factored second-moment core, pulled out so ``torch.compile`` can fuse it.

``Adafusion(compile=True)`` routes the fixed-beta2 factored path (EMA +
reconstruction + RMS clip) through a single dynamic-shape compiled graph, which
Inductor fuses into a few kernels. Big win on large 2-D weights
(transformer/DiT 2048x2048+), neutral-to-negative on many small weights.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from koptim._factored import factored_inv_sqrt_factors, update_factored_state


def _rms(t: Tensor) -> Tensor:
    """Adafactor's root-mean-square of a tensor: ``||t||_2 / sqrt(N)``."""
    return t.norm(2) / math.sqrt(max(t.numel(), 1))


def _factored_update(
    grad_fp32: Tensor,
    exp_avg_sq_row: Tensor,
    exp_avg_sq_col: Tensor,
    beta2: float,
    eps1: float,
    clip_threshold: float,
) -> Tensor:
    """Factored second-moment EMA + reconstruction + RMS clip, as one function.

    Requires ``clip_threshold > 0`` (the compiled path is only taken when set).
    """
    update_factored_state(grad_fp32, exp_avg_sq_row, exp_avg_sq_col, beta2, eps1)
    r_factor, c_factor = factored_inv_sqrt_factors(exp_avg_sq_row, exp_avg_sq_col)
    update = grad_fp32.mul(r_factor).mul_(c_factor)
    divisor = (_rms(update) / clip_threshold).clamp_(min=1.0)
    update.div_(divisor)
    return update


_COMPILED_FACTORED_UPDATE: Any = None


def _get_compiled_factored_update() -> Any:
    """Lazily build (once, process-wide) the ``torch.compile``'d factored update.

    ``dynamic=True`` compiles a single graph that serves every weight shape, so a
    per-parameter / gradient-release setup (thousands of optimizers) shares one
    compiled artifact instead of recompiling per shape.
    """
    global _COMPILED_FACTORED_UPDATE
    if _COMPILED_FACTORED_UPDATE is None:
        _COMPILED_FACTORED_UPDATE = torch.compile(_factored_update, dynamic=True)
    return _COMPILED_FACTORED_UPDATE
