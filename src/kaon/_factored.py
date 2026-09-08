"""Pure helper functions for the factored second moment (Adafactor).

The factored second moment stores per-row and per-column running means of
``grad ** 2`` instead of the full matrix. For a tensor of shape ``(R, C)``
this uses ``R + C`` floats instead of ``R * C`` — typically a 1000x+ reduction.

We match the **eps placement of Hugging Face Transformers' Adafactor**
(``eps1`` is added to ``grad ** 2`` *before* the row/column mean reductions)
so we can numerically check our implementation against that reference.

References:
    * Shazeer, N., & Stern, M. (2018). *Adafactor: Adaptive Learning Rates
      with Sublinear Memory Cost.* arXiv:1804.04235.
    * ``transformers.Adafactor`` (Hugging Face Transformers).
    * ``tdrussell/diffusion-pipe`` ``optimizers/generic_optim.py``
      (``second_moment_type == "factored"`` branch).
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["update_factored_state", "factored_inv_sqrt_factors"]

# Floor for the row-mean divisor in the reconstruction. The smallest *normal* fp32,
# so it is a no-op for every representable positive mean and only bites on an exact
# zero (see factored_inv_sqrt_factors).
_MIN_NORMAL = torch.finfo(torch.float32).tiny


def update_factored_state(
    grad: Tensor,
    exp_avg_sq_row: Tensor,
    exp_avg_sq_col: Tensor,
    beta2: float,
    eps1: float,
) -> None:
    """In-place EMA update of factored row and column second-moment stats.

    Adds ``eps1`` to ``grad ** 2`` *before* taking the row/column means
    (matching Hugging Face's Adafactor). The state tensors are updated as::

        exp_avg_sq_row <- lerp(exp_avg_sq_row, mean(g^2 + eps1, dim=-1), 1 - beta2)
        exp_avg_sq_col <- lerp(exp_avg_sq_col, mean(g^2 + eps1, dim=-2), 1 - beta2)

    Args:
        grad: Gradient tensor of shape ``(..., R, C)``. Must have ``ndim >= 2``.
        exp_avg_sq_row: Row statistics, shape ``(..., R)``. Updated in place.
        exp_avg_sq_col: Column statistics, shape ``(..., C)``. Updated in place.
        beta2: Second-moment EMA decay coefficient.
        eps1: Small positive constant added to ``grad ** 2`` for stability.
    """
    grad_sq = grad.pow(2)
    if eps1 > 0:
        grad_sq.add_(eps1)
    exp_avg_sq_row.lerp_(grad_sq.mean(dim=-1), 1.0 - beta2)
    exp_avg_sq_col.lerp_(grad_sq.mean(dim=-2), 1.0 - beta2)


def factored_inv_sqrt_factors(
    exp_avg_sq_row: Tensor,
    exp_avg_sq_col: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the row and column factors of ``1 / sqrt(v_hat)``.

    Splitting the reconstruction into its two factors lets the caller chain
    a broadcasting multiply ``grad * r * c`` and avoid materialising the
    full ``(R, C)`` reconstructed denominator. Mathematically::

        inv_sqrt_v_hat[..., r, c] = r_factor[..., r, 0] * c_factor[..., 0, c]
        r_factor = rsqrt(exp_avg_sq_row / mean(exp_avg_sq_row, dim=-1))  unsqueezed at -1
        c_factor = rsqrt(exp_avg_sq_col)                                  unsqueezed at -2

    **The all-zero second moment.** ``exp_avg_sq_row`` is non-negative, so its mean
    is zero only when the whole row vector is zero — which happens when the folded
    gradients were all exactly zero *and* the caller uses ``eps1 == 0`` (ADOPT, to
    match the official implementation). ``row / mean(row)`` would then be ``0 / 0``,
    and the NaN survives the caller's ``clamp(max=1/eps)`` and poisons the weights.
    Flooring the divisor at the smallest normal fp32 turns that case into
    ``rsqrt(0) = +inf``, which the caller's cap collapses to ``1 / eps`` — exactly
    what the non-factored path produces for ``v == 0``
    (``denom = max(sqrt(0), eps) = eps``). The floor is a no-op for every positive
    mean: a mean small enough to be affected already puts the reconstruction far
    above any sane ``1 / eps`` cap, so the result is unchanged there too. NaNs that
    came *from the gradient* still propagate (``clamp`` keeps them).

    Args:
        exp_avg_sq_row: Row statistics, shape ``(..., R)``.
        exp_avg_sq_col: Column statistics, shape ``(..., C)``.

    Returns:
        ``(r_factor, c_factor)`` with shapes ``(..., R, 1)`` and ``(..., 1, C)``.
    """
    r_factor = (
        exp_avg_sq_row
        .div(exp_avg_sq_row.mean(dim=-1, keepdim=True).clamp_min_(_MIN_NORMAL))
        .rsqrt_()
        .unsqueeze(-1)
    )
    c_factor = exp_avg_sq_col.rsqrt().unsqueeze(-2)
    return r_factor, c_factor
