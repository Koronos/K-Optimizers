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
    "DEFAULT_STACK_ELEMS",
    "FOREACH_BATCH_CUTOFF",
    "LOW_PRECISION",
    "MIN_STACK_ELEMS",
    "STACK_SAFETY_FRACTION",
    "cautious_batched_",
    "cautious_one_",
    "centralize_grads_",
    "foreach_budget",
    "is_low_precision",
    "subtract_batched_",
    "subtract_one_",
]

LOW_PRECISION = (torch.bfloat16, torch.float16)


def is_low_precision(t: Tensor) -> bool:
    return t.dtype in LOW_PRECISION


# ----------------------------- foreach budget -----------------------------
# Shared foreach-batching knobs (the per-optimizer ``bytes_per_elem`` differs and is
# passed in; everything else is identical across optimizers). See docs/foreach-batching.md.
FOREACH_BATCH_CUTOFF = 2_000_000   # per-tensor element cap above which a weight loops (perf)
STACK_SAFETY_FRACTION = 0.10       # use at most ~10% of currently-free VRAM per stacked chunk
MIN_STACK_ELEMS = 262_144          # still batch small tensors even under memory pressure
DEFAULT_STACK_ELEMS = 64_000_000   # CPU / unknown device: no VRAM limit to respect


def foreach_budget(stack_budget: int | None, batch_cutoff: int, bytes_per_elem: int,
                   device: torch.device) -> int:
    """Max elements per stacked chunk for the foreach path.

    An explicit ``stack_budget`` is returned verbatim. Otherwise the chunk is
    ``min(adaptive_to_free_VRAM, 4 * batch_cutoff)``: the VRAM term shrinks the chunk when a
    big model already fills the card (OOM safety) and grows it on a roomy one; the
    ``4 * batch_cutoff`` cap stops over-stacking (beyond a few cutoff-sized tensors, stacking
    medium weights just adds copy bandwidth). ``bytes_per_elem`` is the optimizer's stacked
    working-set estimate per element (more momenta -> larger).
    """
    if stack_budget is not None:
        return stack_budget
    cap = 4 * batch_cutoff
    if device.type == "cuda":
        free_bytes = torch.cuda.mem_get_info(device)[0]
        adaptive = int(free_bytes * STACK_SAFETY_FRACTION / bytes_per_elem)
        return max(MIN_STACK_ELEMS, min(adaptive, cap))
    return min(DEFAULT_STACK_ELEMS, cap)


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


# ----------------------------- gradient preprocessing -----------------------------
@torch.no_grad()
def centralize_grads_(params: list[Tensor]) -> None:
    """Gradient Centralization (Yong et al. 2020, arXiv:2004.01461), in place.

    For every ``ndim >= 2`` weight, subtract the gradient's mean over the fan-in dims (all
    dims except the output channel, dim 0) per output row. A **zero-state** gradient
    preprocessor applied at the top of the step, before the optimizer reads ``p.grad``;
    1-D params (biases / norm scales) are left untouched.

    Measured (proxy, gap lens, 3 seeds-pairs): a free held-out-loss win (~-0.003..-0.006) for
    the factored-Adam and sign optimizers (Adakaon, Lion, AdaPNM, KProdigy); neutral-to-negative
    for the orthogonalized AdaMuon, so it is per-optimizer opt-out (``gradient_centralization``).

    **Batched by shape** so the LoRA many-tiny-tensor regime stays fast: a naive per-param Python
    loop here added ~1024 kernel launches/step on a 512-adapter bag (3x slower). Same-shape grads
    are stacked and centralized in a handful of ops; lone shapes go in place.
    """
    by_shape: dict[tuple[int, ...], list[Tensor]] = {}
    for p in params:
        g = p.grad
        if g is not None and g.ndim >= 2:
            by_shape.setdefault(tuple(g.shape), []).append(g)
    for grads in by_shape.values():
        if len(grads) == 1:
            g = grads[0]
            g.sub_(g.mean(dim=tuple(range(1, g.ndim)), keepdim=True))
        else:
            # Stack so the per-param means are one reduction, centralize the stack, scatter
            # back. (A broadcasting ``_foreach_sub_`` falls to a slow path and is ~2x worse.)
            gs = torch.stack(grads)  # [N, *shape]; fan-in dims are 2..end (dim 1 = output row)
            gs.sub_(gs.mean(dim=tuple(range(2, gs.ndim)), keepdim=True))
            torch._foreach_copy_(grads, list(gs.unbind(0)))
