"""Shared per-step primitives reused by every kaon optimizer's update.

One implementation each — so a fix or a perf change lands everywhere at once — for the
cross-cutting pieces that used to be copy-pasted into each optimizer:

* the low-precision dtype check,
* the bf16-correct weight write (``p -= delta``), per-param and batched (foreach),
* cautious masking (Liang et al. 2024), per-param and batched.

All are bit-exact with the per-optimizer copies they replaced (same arithmetic); the
``foreach == per-param`` parity tests across every optimizer/dtype are the proof.
"""
from __future__ import annotations

import torch
from torch import Tensor

from kaon._stochastic_rounding import add_stochastic_

__all__ = [
    "LOW_PRECISION",
    "cautious_batched_",
    "cautious_one_",
    "is_low_precision",
    "subtract_batched_",
    "subtract_one_",
]

LOW_PRECISION = (torch.bfloat16, torch.float16)


def is_low_precision(t: Tensor) -> bool:
    return t.dtype in LOW_PRECISION


# ----------------------------- weight write: p -= delta -----------------------------
@torch.no_grad()
def subtract_one_(p: Tensor, delta_fp32: Tensor, state: dict, bf16_method: str) -> None:
    """Per-parameter ``p -= delta`` with the configured bf16 handling.

    ``kahan`` keeps a per-param compensation buffer (``state['shift']``); ``stochastic_
    rounding`` does the unbiased bf16 round; otherwise a plain cast-and-subtract.
    """
    low = is_low_precision(p)
    if low and bf16_method == "kahan":
        shift = state["shift"]
        shift.sub_(delta_fp32.to(p.dtype))
        p_before = p.detach().clone()
        p.add_(shift)
        shift.add_(p_before.sub_(p))
    elif low and bf16_method == "stochastic_rounding" and p.dtype == torch.bfloat16:
        add_stochastic_(p.data, delta_fp32, alpha=-1.0)
    else:
        p.data.sub_(delta_fp32.to(p.dtype))


@torch.no_grad()
def subtract_batched_(pviews: list[Tensor], delta: Tensor, bf16_method: str) -> None:
    """In-place ``p -= delta`` over a foreach bucket of (matrixized) param views.

    ``pviews`` is the list of N same-shape param views (each ``[*shape]``); ``delta`` is
    the stacked fp32 step ``[N, *shape]`` (row i applies to ``pviews[i]``).

    Only the **bf16 + stochastic-rounding** case needs a materialized stacked-weights
    tensor (``add_stochastic_`` operates on the stack). Every other case — notably the
    fp32 regime, including LoRA's many-tiny-tensor buckets — subtracts the delta slices
    straight into the param views with ``_foreach_sub_``, skipping *both* the stack-weights
    allocation and the copy-back, which are pure overhead in the launch-bound regime.
    """
    p0 = pviews[0]
    if p0.dtype == torch.bfloat16 and bf16_method == "stochastic_rounding":
        weights = torch.stack(pviews)
        add_stochastic_(weights, delta, alpha=-1.0)
        torch._foreach_copy_(pviews, list(weights.unbind(0)))
    elif p0.dtype == delta.dtype:
        torch._foreach_sub_(pviews, list(delta.unbind(0)))
    else:
        torch._foreach_sub_(pviews, [d.to(p0.dtype) for d in delta.unbind(0)])


# ----------------------------- cautious masking -----------------------------
@torch.no_grad()
def cautious_batched_(delta: Tensor, grad: Tensor) -> Tensor:
    """Cautious masking on a foreach bucket ``[N, *shape]``: zero the update coordinates
    whose sign disagrees with the gradient (``delta*grad <= 0``) and rescale the survivors
    by their per-slice surviving fraction so the mean step magnitude is preserved. Modifies
    and returns ``delta``."""
    mask = (delta * grad > 0).to(delta.dtype)
    n = delta.shape[0]
    denom = mask.reshape(n, -1).mean(dim=1).clamp_(min=1e-8).view(n, *([1] * (delta.ndim - 1)))
    return delta.mul_(mask).div_(denom)


@torch.no_grad()
def cautious_one_(delta: Tensor, grad: Tensor) -> Tensor:
    """Per-parameter cautious masking (scalar rescale). Modifies and returns ``delta``."""
    mask = (delta * grad > 0).to(delta.dtype)
    return delta.mul_(mask).div_(mask.mean().clamp_(min=1e-8))
