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

import warnings
from collections.abc import Iterator
from typing import Any

import torch
from torch import Tensor

from kaon._backend import decode_value
from kaon._compact_kahan import RESIDUAL_KEY, decode, is_compact_kahan, residual_bits_of

__all__ = ["decode_weights", "full_precision_state_dict"]


def _chain(optimizer: Any) -> list[Any]:
    """``optimizer`` and every optimizer it wraps (``.inner``), outermost first."""
    out = [optimizer]
    while hasattr(out[-1], "inner"):
        out.append(out[-1].inner)
    return out


def _check_view(chain: list[Any]) -> None:
    """Refuse (or flag) a live view the export would get wrong (see :func:`decode_weights`)."""
    for opt in chain:
        # MSAM / Nekaon: in train mode the parameters carry the lookahead perturbation.
        if getattr(opt, "_train_mode", False) and getattr(opt, "_has_e", False):
            raise RuntimeError(
                f"decode_weights: {type(opt).__name__} is in train mode, so the parameters "
                "carry the lookahead perturbation; call optimizer.eval() first (and "
                "optimizer.train() afterwards)."
            )
        groups = getattr(opt, "param_groups", None) or []
        if not groups or "train_mode" not in groups[0]:
            continue
        train = groups[0]["train_mode"]
        if "slow_dtype" in groups[0]:
            # Lookahead: its eval view shows the slow weights ``phi`` (stored at slow_dtype,
            # no residual); the inner's residual belongs to the fast weights it replaced.
            if not train:
                raise RuntimeError(
                    f"decode_weights: {type(opt).__name__} is in eval mode, which shows the "
                    "slow weights phi (stored at slow_dtype — there is no residual to decode "
                    "for them); save model.state_dict() in eval mode for phi, or call "
                    "decode_weights in train mode for the fast weights' full value."
                )
        elif train:
            # ScheduleFree (TrainEvalWeights, no compact Kahan): the train view is the
            # gradient point y, not the iterate x a checkpoint wants. Nothing to decode;
            # the export is the live view upcast — flag it rather than refuse.
            warnings.warn(
                f"decode_weights: {type(opt).__name__} is in train mode, so the parameters "
                "hold its training view (ScheduleFree: y), not the evaluation weights; call "
                "optimizer.eval() before exporting.",
                stacklevel=3,
            )


#: Elements per CPU decode chunk (1 Mi). The torch reference :func:`decode` holds ~4.25x its
#: input in int32/bool scratch at peak (``w``, ``q``, the carry mask and product); decoding
#: chunk by chunk into ONE preallocated fp32 output bounds that scratch to ~17 MB whatever
#: the tensor. Measured process peak over the call (output included): 4608x1152 (5.3 M)
#: 66.5 -> 13.9 MiB, 21 M elements 425.9 -> 102.9 MiB. 4 Mi chunks were WORSE than no
#: chunking on the 5.3 M DiT MLP (78.8 vs 66.5 MiB): the chunk must be well below the tensor.
_DECODE_CHUNK = 1 << 20


def _decode_lowmem(p: Tensor, lo: Tensor, bits: int) -> Tensor:
    """:func:`decode` with bounded scratch; bit-identical (the decode is elementwise).

    CUDA: :func:`kaon._backend.decode_value` — the single-launch Triton kernel the foreach
    path already uses, writing straight into its fp32 output with NO scratch (the torch
    reference it falls back to, e.g. without Triton, is the old peak). CPU: the torch
    reference, chunk by chunk, into a preallocated output.
    """
    if p.is_cuda:
        return decode_value(p, lo, bits)
    n = p.numel()
    if n <= _DECODE_CHUNK or not (p.is_contiguous() and lo.is_contiguous()):
        return decode(p, lo, bits)
    out = torch.empty(p.shape, dtype=torch.float32, device=p.device)
    of, pf, lf = out.view(-1), p.view(-1), lo.view(-1)
    for i in range(0, n, _DECODE_CHUNK):
        j = min(i + _DECODE_CHUNK, n)
        of[i:j] = decode(pf[i:j], lf[i:j], bits)
    return out


def _decoded(p: Tensor, lo: Tensor | None, method: str) -> Tensor:
    """fp32 full value of one param, on ``p``'s device (a fresh tensor)."""
    # A residual is only maintained while the group's method is compact Kahan: after a
    # switch back to SR / none it is stale and must not be added.
    if lo is not None and is_compact_kahan(method):
        return _decode_lowmem(p.detach(), lo, residual_bits_of(lo))
    return p.detach().float().clone()


def _iter_decoded(optimizer: Any, device: Any) -> Iterator[tuple[Tensor, Tensor]]:
    """``(param, fp32 value on device)`` one param at a time: the decode's temporaries and
    the fp32 value exist on the param's device for ONE tensor at a time, then move."""
    chain = _chain(optimizer)
    _check_view(chain)
    for group in optimizer.param_groups:
        method = group.get("bf16_method", "")
        for p in group["params"]:
            lo = None
            if p.dtype == torch.bfloat16:
                for opt in chain:
                    st = opt.state.get(p) if hasattr(opt, "state") else None
                    if st and RESIDUAL_KEY in st:
                        lo = st[RESIDUAL_KEY]
                        break
            v = _decoded(p, lo, method)
            if device is not None and v.device != torch.device(device):
                v = v.to(device)
            yield p, v


@torch.no_grad()
def decode_weights(optimizer: Any, device: Any = None) -> dict[Tensor, Tensor]:
    """The full-precision value of every parameter ``optimizer`` steps, as fp32.

    Returns ``{param: fp32 tensor}`` (fresh, detached). A bf16 parameter with a
    compact-Kahan residual (``bf16_method="kahan8"`` / ``"kahan16"``) is decoded —
    ``kahan16``: exactly the fp32 value an fp32-weight run would hold; ``kahan8``: to
    ``ulp/256`` — and every other parameter is its own value upcast to fp32 (so the dict is
    a complete fp32 export whatever the method). Works for every kaon optimizer that
    supports kahan8/kahan16 and through wrappers (Nekaon / MSAM / Lookahead / SAM): the
    residual is looked up at whichever level owns it. (ScheduleFree does not support compact
    Kahan; for it this is the live view upcast.)

    ``device``: where the fp32 values go. ``None`` (default) keeps each on its param's device
    — the WHOLE model in fp32 there, 4 B/param: a 2.6B-param model on an 8 GB GPU does not fit.
    Pass ``device="cpu"`` to stream: each tensor is decoded on its device and moved before
    the next one, so the device peak is one tensor (its fp32 value + the decode's scratch).
    :func:`full_precision_state_dict` defaults to ``"cpu"``.

    Call it on the TRUE weights: after ``optimizer.eval()`` for Nekaon / MSAM (in train mode
    the parameters sit at the lookahead point — refused), in train mode for Lookahead (its
    eval view shows the slow weights, which have no residual — refused), after ``eval()``
    for ScheduleFree (its train view is ``y`` — warns). Load the optimizer state AFTER the
    model weights when resuming: a residual written for one bf16 pattern is meaningless on
    another.

    Typical use, an fp32 checkpoint of a model trained in bf16::

        opt.eval()
        torch.save(kaon.full_precision_state_dict(model, opt), "model_fp32.pt")  # on CPU
        opt.train()
    """
    return dict(_iter_decoded(optimizer, device))


@torch.no_grad()
def full_precision_state_dict(module: torch.nn.Module, optimizer: Any,
                              device: Any = "cpu") -> dict[str, Any]:
    """``module.state_dict()`` with every parameter ``optimizer`` steps replaced by its
    fp32 full-precision value (:func:`decode_weights`); buffers and parameters the optimizer
    does not own are kept as they are (not moved). Same view rules as :func:`decode_weights`.

    ``device`` (default ``"cpu"``): where the fp32 values go. Streamed one tensor at a time,
    so the GPU peak is a single tensor's decode, never the model in fp32. ``None`` keeps
    them on the params' devices (the whole fp32 model there at once).

    Raises ``ValueError`` if none of the optimizer's parameters is a parameter of ``module``
    (a different model instance, or a ``module`` whose tensors were re-created): the result
    would otherwise silently be the bf16 state dict.
    """
    sd = module.state_dict(keep_vars=True)
    by_id = {id(t): name for name, t in sd.items() if isinstance(t, Tensor)}
    out: dict[str, Any] = {name: (t.detach() if isinstance(t, Tensor) else t)
                           for name, t in sd.items()}
    found = n_opt = 0
    for p, v in _iter_decoded(optimizer, device):
        n_opt += 1
        name = by_id.get(id(p))
        if name is not None:
            out[name] = v
            found += 1
    if n_opt and not found:
        raise ValueError(
            "full_precision_state_dict: none of the optimizer's parameters is a parameter of "
            "this module (another model instance, or tensors re-created after the optimizer "
            "was built) — the result would be the unchanged bf16 state dict."
        )
    return out
