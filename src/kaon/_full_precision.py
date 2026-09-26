"""Export the FULL-precision weights a compact-Kahan optimizer is tracking.

Under ``bf16_method="kahan8"`` / ``"kahan16"`` a bf16 parameter holds only the nearest bf16
to the value the optimizer is integrating; the rest lives in the optimizer's
``state['kahan_lo']`` residual (:mod:`kaon._compact_kahan`). ``model.state_dict()`` therefore
saves the rounded half: fine for bf16 inference, lossy for a checkpoint meant to continue
training elsewhere or to be served in fp32. These helpers decode ``(bf16, residual)`` back
into fp32 — for ``kahan16`` bit for bit the fp32 master, for ``kahan8`` the value to
``ulp/256`` — for any kaon optimizer (and wrapper chain) using either method.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from kaon._compact_kahan import RESIDUAL_KEY, decode, is_compact_kahan, residual_bits_of

__all__ = ["decode_weights", "full_precision_state_dict"]


def _chain(optimizer: Any) -> list[Any]:
    """``optimizer`` and every optimizer it wraps (``.inner``), outermost first."""
    out = [optimizer]
    while hasattr(out[-1], "inner"):
        out.append(out[-1].inner)
    return out


def _check_view(chain: list[Any]) -> None:
    """Refuse a live view the residual does not belong to (see :func:`decode_weights`)."""
    for opt in chain:
        # MSAM / Nekaon: in train mode the parameters carry the lookahead perturbation.
        if getattr(opt, "_train_mode", False) and getattr(opt, "_has_e", False):
            raise RuntimeError(
                f"decode_weights: {type(opt).__name__} is in train mode, so the parameters "
                "carry the lookahead perturbation; call optimizer.eval() first (and "
                "optimizer.train() afterwards)."
            )
        # Lookahead (TrainEvalWeights): the eval view shows the slow weights ``phi``; the
        # residual belongs to the fast weights it replaced.
        groups = getattr(opt, "param_groups", None) or []
        if groups and "train_mode" in groups[0] and not groups[0]["train_mode"]:
            raise RuntimeError(
                f"decode_weights: {type(opt).__name__} is in eval mode, which shows the slow "
                "weights (stored at slow_dtype; there is no residual for them to decode) — "
                "save model.state_dict() in eval mode as usual, or call decode_weights in "
                "train mode for the fast weights."
            )


@torch.no_grad()
def decode_weights(optimizer: Any) -> dict[Tensor, Tensor]:
    """The full-precision value of every parameter ``optimizer`` steps, as fp32.

    Returns ``{param: fp32 tensor}`` (fresh tensors, detached, same device). A bf16
    parameter with a compact-Kahan residual (``bf16_method="kahan8"`` / ``"kahan16"``) is
    decoded — ``kahan16``: exactly the fp32 value an fp32-weight run would hold;
    ``kahan8``: to ``ulp/256`` — and every other parameter is its own value upcast to fp32
    (so the dict is a complete fp32 export whatever the method). Works through wrappers
    (Nekaon / MSAM / Lookahead / SAM): the residual is looked up at whichever level owns
    it (the inner optimizer that writes the weight).

    Call it on the TRUE weights: after ``optimizer.eval()`` for Nekaon / MSAM (in train
    mode the parameters sit at the lookahead point — refused with an error), in train
    mode for Lookahead (its eval view shows the slow weights, which have no residual —
    also refused). Load the optimizer state AFTER the model weights when resuming: a
    residual written for one bf16 pattern is meaningless on another.

    Typical use, an fp32 checkpoint of a model trained in bf16::

        opt.eval()
        torch.save(kaon.full_precision_state_dict(model, opt), "model_fp32.pt")
        opt.train()
    """
    chain = _chain(optimizer)
    _check_view(chain)
    out: dict[Tensor, Tensor] = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            lo = None
            if p.dtype == torch.bfloat16:
                for opt in chain:
                    st = opt.state.get(p) if hasattr(opt, "state") else None
                    if st and RESIDUAL_KEY in st:
                        lo = st[RESIDUAL_KEY]
                        break
            # A residual is only maintained while the group's method is compact Kahan:
            # after a switch back to SR / none it is stale and must not be added.
            if lo is not None and is_compact_kahan(group.get("bf16_method", "")):
                out[p] = decode(p.detach(), lo, residual_bits_of(lo))
            else:
                out[p] = p.detach().float().clone()
    return out


@torch.no_grad()
def full_precision_state_dict(module: torch.nn.Module, optimizer: Any) -> dict[str, Any]:
    """``module.state_dict()`` with every parameter ``optimizer`` steps replaced by its
    fp32 full-precision value (:func:`decode_weights`); buffers and parameters the optimizer
    does not own are kept as they are. Same view rules as :func:`decode_weights`."""
    values = decode_weights(optimizer)
    sd = module.state_dict(keep_vars=True)
    out: dict[str, Any] = {}
    for name, t in sd.items():
        v = values.get(t) if isinstance(t, Tensor) else None
        out[name] = v if v is not None else (t.detach() if isinstance(t, Tensor) else t)
    return out
