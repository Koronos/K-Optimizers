"""Decoupled (AdamW) weight decay folded into the fp32 step, for ADOPT / AdaBelief / AdamP.

These optimizers used to decay the weight in place, ``p *= (1 - lr*wd)``, before their
update. On a bf16 weight that multiply rounds to nearest in bf16, outside stochastic rounding
and outside the compact-Kahan residual, so at a diffusion-typical ``lr*wd ~ 1e-5`` the factor
rounds every coordinate back to itself: the decay silently did nothing (``p = 1`` stays
exactly ``1`` for as long as you train). The helpers here fold it into the fp32 delta instead,
``delta += lr*wd*p``, so it is written by the same SR / Kahan write as the update — Adakaon's
placement (``kaon.adakaon``).

Algebraically ``p*(1 - a) - d == p - (d + a*p)``; on fp32 weights the two differ only by
rounding. The term READS the weight, so under ``kahan8`` / ``kahan16`` it reads the decoded
full value (:func:`kaon._backend.weight_value`): that is what makes a ``kahan16`` run the
fp32-weight run bit for bit with decay on.

The callers add it AFTER the cautious mask, because the in-place decay they replace was
outside the mask too (every coordinate decays by the same ``lr*wd``) — the placement
Adakaon calls ``cautious_wd="full"``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from kaon._backend import weight_value
from kaon._compact_kahan import is_compact_kahan
from kaon._foreach_plan import ForeachChunk

__all__ = ["decay_batched_", "decay_one_"]


def _decoded(p: Tensor, bf16_method: str) -> bool:
    return p.dtype == torch.bfloat16 and is_compact_kahan(bf16_method)


@torch.no_grad()
def decay_one_(delta: Tensor, p: Tensor, state: dict, bf16_method: str, alpha: float,
               ratio: Tensor | None = None, value: Tensor | None = None) -> Tensor:
    """``delta += alpha * [ratio *] value(p)`` in place (per-parameter path).

    A plain bf16 / fp16 weight is added as is (the fp32 ``add_`` widens it exactly, without
    the fp32 copy ``weight_value`` would make); only a compact-Kahan bf16 weight is decoded.
    ``ratio`` (a 0-D tensor, AdamP's ``wd_ratio``) goes through ``addcmul_`` so the scalar
    never has to be synchronized to the host. ``value`` is the weight value when the caller
    already read it (AdamP's projection)."""
    if value is None:
        value = weight_value(p, state, bf16_method) if _decoded(p, bf16_method) else p.data
    if ratio is None:
        return delta.add_(value, alpha=alpha)
    return delta.addcmul_(value, ratio, value=alpha)


@torch.no_grad()
def decay_batched_(delta: Tensor, chunk: ForeachChunk, bf16_method: str, alpha: float,
                   ratio: Tensor | None = None,
                   value: Tensor | None = None) -> tuple[Tensor, Tensor] | None:
    """``delta[i] += alpha * [ratio[i] *] value(p_i)`` in place over a foreach bucket.

    Returns the stacked ``(weights, residuals)`` a compact-Kahan decode built, for
    :func:`kaon._backend.subtract_batched_` ``stacked=`` (``None`` otherwise). ``value`` is
    the stacked weight value when the caller already has one (AdamP's projection stack).

    fp32 buckets add the param views straight into the delta slices (``_foreach_add_``): no
    stacked copy of the weights at all, which is what the in-place ``_foreach_mul_`` it
    replaces cost too. Other dtypes stack once in their own dtype (half the bytes of an fp32
    stack) and let the fp32 add widen them."""
    stacks = None
    if value is None:
        p0 = chunk.pviews[0]
        if _decoded(p0, bf16_method) and chunk.cviews is not None:
            value, stacks = chunk.value_and_stacks(bf16_method)
        elif ratio is None and p0.dtype == delta.dtype:
            torch._foreach_add_(list(delta.unbind(0)), chunk.pviews, alpha=alpha)
            return None
        else:
            value = torch.stack(chunk.pviews)
    if ratio is None:
        delta.add_(value, alpha=alpha)
    else:
        delta.addcmul_(value, ratio, value=alpha)
    return stacks
