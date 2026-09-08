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

# The floor ADOPT passes to ``factored_inv_sqrt_factors``: the smallest *normal* fp32.
# Deliberately the smallest value that rescues an exact zero, so it perturbs the least
# possible amount of the subnormal range; anything larger (1e-8, say) would silently
# reshape real updates whose row mean is merely very small. Only meaningful for a
# caller that caps the reconstruction afterwards — see factored_inv_sqrt_factors.
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
    floor: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Return the row and column factors of ``1 / sqrt(v_hat)``.

    Splitting the reconstruction into its two factors lets the caller chain
    a broadcasting multiply ``grad * r * c`` and avoid materialising the
    full ``(R, C)`` reconstructed denominator. Mathematically::

        inv_sqrt_v_hat[..., r, c] = r_factor[..., r, 0] * c_factor[..., 0, c]
        r_factor = rsqrt(exp_avg_sq_row / mean(exp_avg_sq_row, dim=-1))  unsqueezed at -1
        c_factor = rsqrt(exp_avg_sq_col)                                  unsqueezed at -2

    **The all-zero second moment, and why the floor is opt-in.**
    ``exp_avg_sq_row`` is non-negative, so its mean is zero only when the whole row
    vector is zero — which needs the caller's ``eps1 == 0``, since ``eps1`` is added
    to ``grad ** 2`` before the reductions. ``row / mean(row)`` is then ``0 / 0``, and
    that NaN poisons everything downstream. ``floor > 0`` clamps the divisor from
    below, turning the case into ``rsqrt(0) = +inf``.

    ``+inf`` is only *useful* to a caller that caps the reconstructed
    inverse-denominator afterwards: ADOPT and KProdigy do
    (``clamp(max=1/eps)``), so for them the degenerate case collapses to ``1 / eps``
    — exactly what a non-factored ``denom = max(sqrt(0), eps)`` yields. The other
    callers (Adakaon, AdaPNM, AdaBelief, AdamP, AdaMuon, ScheduleFree) do **not**
    cap, so for them a floor would be neither inert nor helpful: with a subnormal
    row mean — reachable with a huge ``beta2`` and tiny gradients — clamping the
    divisor would visibly move a *finite* update instead of rescuing a NaN. That is
    why this is a keyword with a ``0.0`` default rather than an unconditional guard,
    and why **only ADOPT passes it** (``eps1 == 0`` is unique to ADOPT here).

    With the ``0.0`` default no clamp runs at all, so the expression is bit-identical
    to the 0.7.12 one, including its treatment of ``-0.0`` and NaN. NaNs that came
    *from the gradient* propagate either way (``clamp`` keeps them).

    Args:
        exp_avg_sq_row: Row statistics, shape ``(..., R)``.
        exp_avg_sq_col: Column statistics, shape ``(..., C)``.
        floor: Lower bound for the row-mean divisor. ``0.0`` (default) applies no
            bound at all. Pass a positive value **only** if you cap the
            reconstruction afterwards; see :data:`_MIN_NORMAL`.

    Returns:
        ``(r_factor, c_factor)`` with shapes ``(..., R, 1)`` and ``(..., 1, C)``.
    """
    row_mean = exp_avg_sq_row.mean(dim=-1, keepdim=True)
    if floor > 0.0:
        row_mean.clamp_min_(floor)
    r_factor = exp_avg_sq_row.div(row_mean).rsqrt_().unsqueeze(-1)
    c_factor = exp_avg_sq_col.rsqrt().unsqueeze(-2)
    return r_factor, c_factor
