"""MARS — Make vAriance Reduction Shine (the MARS-AdamW instance) on kaon's backend.

MARS (Yuan, Liu, Wu, Zhuang, Gu 2024, *MARS: Unleashing the Power of Variance
Reduction for Training Large Models*, arXiv:2411.10438) is a variance-reduction
preconditioned optimizer. The MARS-AdamW instance keeps an AdamW-shaped update but
feeds AdamW a **variance-reduced "corrected" gradient** ``c_t`` instead of the raw
gradient. ``c_t`` is a STORM/SVRG-style correction that adds a scaled gradient
*difference* (current minus previous) to the current gradient, cancelling part of
the stochastic-gradient noise and giving the second-moment estimate a cleaner
signal. This implementation rebuilds MARS-AdamW on the precision and memory
machinery proven in :class:`~kaon.adakaon.Adakaon` / :class:`~kaon.adapnm.AdaPNM`.

**The exact update (matches kozistr ``pytorch_optimizer`` ``MARS`` with
``mars_type="adamw"``, ``optimize_1d=False``):**

.. code-block:: text

    # 2-D / conv ("matrix") params — the variance-reduction path:
    c_t = (g_t - g_{t-1}) * (gamma * beta1/(1-beta1)) + g_t      # corrected gradient
    if ||c_t||_2 > 1:  c_t = c_t / ||c_t||_2                     # global-norm clip
    m   = beta1 * m + (1 - beta1) * c_t                          # first moment of c_t
    v   = beta2 * v + (1 - beta2) * c_t^2                        # second moment of c_t
    bc1 = 1 - beta1^t ,  bc2 = 1 - beta2^t
    denom = (sqrt(v) + eps) / sqrt(bc2) * bc1
    p  -= lr * m / denom                                        # == lr/bc1 * m / ((sqrt(v)+eps)/sqrt(bc2))
    g_{t-1} <- g_t                                              # store raw grad for next step

    # 1-D params (biases / norm scales) — plain AdamW on the RAW gradient
    # (optimize_1d=False -> NO variance-reduction correction, NO clipping):
    m   = beta1 * m + (1 - beta1) * g_t
    v   = beta2 * v + (1 - beta2) * g_t^2
    p  -= lr/bc1 * m / ((sqrt(v)+eps)/sqrt(bc2))

Decoupled (AdamW) weight decay ``p *= (1 - lr*weight_decay)`` is applied before the
moment updates (it only touches ``p``, so the order is immaterial). Cautious masking
(Liang et al. 2024), when on, is applied to the final step against the **raw**
gradient (kozistr masks ``m`` against ``grad`` before dividing by the denominator;
since the denominator is positive that yields the same sign mask — kaon's shared
``cautious_*`` rescale differs only in the survivor-rescale constant).

**The kaon twist.**

* Both persistent per-param buffers — the first moment ``m`` and the previous
  gradient ``g_{t-1}`` — live in the shared quantized momentum codec
  (:mod:`kaon._momentum_codec`), exactly like AdaPNM's two momenta. So at
  ``momentum_dtype="int8"`` MARS stores ~2 B/param of persistent state (one byte
  each) and at ``"4bit"`` ~1 B/param, plus the factored ``v``.
* The second moment ``v`` (of ``c_t^2`` on the matrix path, of ``g_t^2`` on the 1-D
  path) reuses Adakaon's **factored** machinery for ``ndim >= 2`` weights (conv
  kernels matrixized to ``[out, in*kh*kw]``); ``ndim == 1`` keeps a full
  per-coordinate ``v``.

**The g_{t-1} init.** On the very first step there is no previous gradient. kaon
seeds ``g_{t-1} = g_t`` so the difference term vanishes and ``c_t = g_t`` (the clean
"no correction yet" start). NOTE this differs from kozistr, which seeds
``last_grad = 0`` so its first ``c_t = g_t * (1 + gamma*beta1/(1-beta1))`` (then
clipped); the kaon init is the documented requirement for this port and is the
mathematically cleaner choice. From step 2 on the two are identical.

**What is reused vs new.** Reused: the factored second-moment helpers
(:mod:`kaon._factored`), the momentum storage layout and quant/dequant primitives
(:mod:`kaon._momentum_codec`), the bf16-correct weight write
(``subtract_one_``/``subtract_batched_``), shared ``cautious_*`` masking and
``centralize_grads_`` (Gradient Centralization), ``load_state_dict_preserving_dtypes``
for dtype-safe resume, and the bucketed foreach batching pattern (``foreach_budget``).
New here: the variance-reduction corrected gradient ``c_t`` with the
``gamma*beta1/(1-beta1)`` scaling and the global-norm clip, the *two-buffer*
read-it-yourself EMA where one buffer is the first moment and the other is the
verbatim previous gradient, and the matrix-vs-1-D split (variance reduction only on
``ndim >= 2``).

It is a standard ``torch.optim.Optimizer`` with a single per-parameter step, so it
drops into per-parameter / gradient-release training loops unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, Literal

import torch
from torch import Tensor
from torch.optim import Optimizer

from kaon._backend import (
    FOREACH_BATCH_CUTOFF,
    cautious_batched_,
    cautious_one_,
    centralize_grads_,
    foreach_budget,
    is_low_precision,
    subtract_batched_,
    subtract_one_,
)
from kaon._factored import factored_inv_sqrt_factors, update_factored_state
from kaon._momentum_codec import (
    _FOURBIT_BLOCK,
    _dequant_4bit,
    _dequant_4bit_stacked,
    _quant_4bit,
    _quant_4bit_stacked,
    _quant_int8,
    _quant_int8_stacked,
    load_state_dict_preserving_dtypes,
)

__all__ = ["MARS"]

MomentumDtype = Literal["bfloat16", "float32", "int8", "4bit"]

# Performance / memory knob mirrors AdaPNM: two codec buffers (m + last_grad) + factored v.
_STACK_BYTES_PER_ELEM = 64


class MARS(Optimizer):
    """MARS-AdamW (variance-reduction) on Adakaon's memory backend.

    Args:
        params: parameters or param-group dicts.
        lr: learning rate. Default ``3e-3`` (the MARS paper / kozistr default — MARS
            tolerates a larger LR than AdamW thanks to the variance reduction).
        betas: ``(beta1, beta2)``. ``beta1`` is the first-moment decay (and drives the
            correction scaling ``gamma*beta1/(1-beta1)``); ``beta2`` the (factored)
            second-moment decay. Default ``(0.9, 0.999)`` (kaon AdamW-scale default;
            the MARS paper uses ``(0.95, 0.99)``).
        eps: term added to the second-moment denominator ``sqrt(v)``. On the factored
            path it is also folded into the Adafactor ``eps1`` (added to the squared
            corrected gradient before the row/col reductions).
        weight_decay: decoupled (AdamW-style) weight decay, ``p *= (1 - lr*wd)``,
            applied before the moment updates.
        gamma: the variance-reduction scaling. The corrected gradient is
            ``c_t = g_t + gamma*(beta1/(1-beta1))*(g_t - g_{t-1})``. ``gamma=0`` turns
            the correction off (MARS collapses to AdamW on the matrix path). Default
            ``0.025`` (paper / kozistr default).
        mars_clip: clip the corrected gradient by its global L2 norm to 1 when
            ``||c_t|| > 1``. **On by default**, matching the reference. Only affects
            the matrix (variance-reduction) path.
        cautious: cautious masking (Liang et al. 2024) on the final step vs the raw
            gradient. **On by default.**
        gradient_centralization: Gradient Centralization (Yong et al. 2020) on
            ``ndim >= 2`` grads before the step. **On by default.**
        momentum_dtype: storage for **both** persistent buffers (the first moment and
            the previous gradient) — ``"bfloat16"`` (default), ``"float32"``,
            ``"int8"`` (per-row absmax), or ``"4bit"`` (per-block absmax, packed).
            Same layout as Adakaon's first moment, so checkpoints resume bit-exactly.
            MARS carries *two* such buffers, so its floor is ~2x a single-momentum
            optimizer at the same dtype.
        momentum_4bit_block: block size for ``momentum_dtype="4bit"``. Default ``128``.
        bf16_method: weight-update strategy for low-precision params —
            ``"stochastic_rounding"`` (default), ``"kahan"`` (+2 B/param), or
            ``"none"``.
        foreach: batch the step across parameters with stacked multi-tensor ops.
            Default ``True``. Numerically matches the per-parameter path.
        foreach_batch_cutoff: per-tensor element count above which a weight loops
            instead of stacking (default ``2_000_000``).
        foreach_stack_budget: max elements per stacked chunk. ``None`` (default)
            adapts to free VRAM; an int pins a fixed cap.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 3e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        gamma: float = 0.025,
        mars_clip: bool = True,
        cautious: bool = True,
        gradient_centralization: bool = True,
        momentum_dtype: MomentumDtype = "bfloat16",
        momentum_4bit_block: int = _FOURBIT_BLOCK,
        bf16_method: str = "stochastic_rounding",
        foreach: bool = True,
        foreach_batch_cutoff: int = FOREACH_BATCH_CUTOFF,
        foreach_stack_budget: int | None = None,
    ) -> None:
        beta1, beta2 = float(betas[0]), float(betas[1])
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"betas[0] must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"betas[1] must be in [0, 1), got {beta2}")
        if lr < 0.0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if eps < 0.0:
            raise ValueError(f"eps must be >= 0, got {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        if gamma < 0.0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if momentum_dtype not in ("bfloat16", "float32", "int8", "4bit"):
            raise ValueError(
                f"momentum_dtype must be bfloat16/float32/int8/4bit, got {momentum_dtype!r}"
            )
        if bf16_method not in ("stochastic_rounding", "kahan", "none"):
            raise ValueError(
                f"bf16_method must be stochastic_rounding/kahan/none, got {bf16_method!r}"
            )
        if foreach_batch_cutoff < 1:
            raise ValueError(f"foreach_batch_cutoff must be >= 1, got {foreach_batch_cutoff}")
        defaults = {
            "lr": lr,
            "betas": (beta1, beta2),
            "eps": float(eps),
            "weight_decay": weight_decay,
            "gamma": float(gamma),
            "mars_clip": mars_clip,
            "cautious": cautious,
            "gradient_centralization": gradient_centralization,
            "momentum_dtype": momentum_dtype,
            "momentum_4bit_block": momentum_4bit_block,
            "bf16_method": bf16_method,
            "step": 0,
        }
        super().__init__(params, defaults)
        self._foreach = foreach
        self._foreach_batch_cutoff = foreach_batch_cutoff
        self._foreach_stack_budget = foreach_stack_budget

    # ------------------------------------------------------------------- state
    @staticmethod
    def _block_size(grad: Tensor, group: dict[str, Any]) -> int:
        bs = group["momentum_4bit_block"]
        numel = grad.numel()
        return numel if bs <= 0 else (min(bs, numel) if numel > 0 else 1)

    @torch.no_grad()
    def _alloc_momentum(
        self, prefix: str, grad: Tensor, state: dict[str, Any], group: dict[str, Any]
    ) -> None:
        """Allocate one codec buffer (``prefix``, ``prefix_scale`` …); layout == codec."""
        md = group["momentum_dtype"]
        if md in ("bfloat16", "float32"):
            dtype = torch.bfloat16 if md == "bfloat16" else torch.float32
            state[prefix] = torch.zeros_like(grad, dtype=dtype)
        elif md == "int8":
            state[prefix] = torch.zeros_like(grad, dtype=torch.int8)
            state[f"{prefix}_scale"] = torch.ones(
                (grad.shape[0],) + (1,) * (grad.ndim - 1) if grad.ndim >= 2 else (),
                dtype=torch.float32, device=grad.device,
            )
        else:  # 4bit
            numel = grad.numel()
            bs = self._block_size(grad, group)
            nblocks = (numel + bs - 1) // bs
            state[prefix] = torch.full(
                ((numel + 1) // 2,), 0x88, dtype=torch.uint8, device=grad.device
            )
            state[f"{prefix}_scale"] = torch.ones(nblocks, dtype=torch.float32, device=grad.device)
            state[f"{prefix}_numel"] = numel
            state[f"{prefix}_block"] = bs

    @torch.no_grad()
    def _init_state(self, p: Tensor, state: dict[str, Any], group: dict[str, Any]) -> None:
        grad = p.grad
        factored = p.ndim >= 2
        if factored:
            gv = grad if p.ndim == 2 else grad.reshape(grad.shape[0], -1)
            row_shape = gv.shape[:-1]
            col_shape = gv.shape[:-2] + gv.shape[-1:]
            state["row"] = torch.zeros(row_shape, dtype=torch.float32, device=p.device)
            state["col"] = torch.zeros(col_shape, dtype=torch.float32, device=p.device)
        else:
            state["v"] = torch.zeros_like(grad, dtype=torch.float32)
        # First moment (m) and previous gradient (g_{t-1}), both via the codec layout.
        self._alloc_momentum("exp_avg", grad, state, group)
        self._alloc_momentum("last_grad", grad, state, group)
        # Seed last_grad = g_t on the first step so c_t = g_t (no correction yet).
        self._store_one(state, "last_grad", group["momentum_dtype"], grad.float())
        if is_low_precision(p) and group["bf16_method"] == "kahan":
            state["shift"] = torch.zeros_like(p)

    # -------------------------------------------------- codec read / write
    @staticmethod
    def _dequant_one(state: dict[str, Any], prefix: str, md: str, like: Tensor) -> Tensor:
        """Read one stored buffer back as a fresh fp32 tensor shaped like ``like``."""
        if md in ("bfloat16", "float32"):
            return state[prefix].float().reshape_as(like)
        if md == "int8":
            return state[prefix].float().mul_(state[f"{prefix}_scale"]).reshape_as(like)
        m = _dequant_4bit(
            state[prefix], state[f"{prefix}_scale"], state[f"{prefix}_numel"], state[f"{prefix}_block"]
        )
        return m.view_as(like)

    @staticmethod
    def _store_one(state: dict[str, Any], prefix: str, md: str, m_fp32: Tensor) -> None:
        """Write an updated fp32 buffer back into the configured storage layout."""
        if md in ("bfloat16", "float32"):
            tgt = state[prefix]
            tgt.copy_(m_fp32.reshape(tgt.shape))
        elif md == "int8":
            m_orig = m_fp32.reshape(state[prefix].shape)
            state[prefix], state[f"{prefix}_scale"] = _quant_int8(m_orig)
        else:  # 4bit
            packed, scale, _ = _quant_4bit(m_fp32, state[f"{prefix}_block"])
            state[prefix], state[f"{prefix}_scale"] = packed, scale

    @staticmethod
    def _dequant_stacked(
        states: list[dict[str, Any]], prefix: str, md: str, shape: tuple[int, ...]
    ) -> Tensor:
        """Stacked fp32 buffer ``[N, *shape]`` from per-param storage."""
        n = len(states)
        per = 1
        for d in shape:
            per *= d
        if md in ("bfloat16", "float32"):
            return torch.stack([s[prefix].reshape(shape) for s in states]).float()
        if md == "int8":
            row = shape[0] if len(shape) >= 2 else 1
            rest = max(per // row, 1)
            m = torch.stack([s[prefix].reshape(row, rest) for s in states]).float()  # [N, R, rest]
            scale = torch.stack([s[f"{prefix}_scale"].reshape(row, 1) for s in states])  # [N, R, 1]
            return m.mul_(scale).reshape((n, *shape))
        packed = torch.stack([s[prefix] for s in states])
        sc = torch.stack([s[f"{prefix}_scale"] for s in states])
        bs = states[0][f"{prefix}_block"]
        return _dequant_4bit_stacked(packed, sc, per, bs).reshape((n, *shape))

    @staticmethod
    def _store_stacked(
        states: list[dict[str, Any]], prefix: str, md: str, m_fp32: Tensor
    ) -> None:
        """Write stacked fp32 buffer ``[N, *shape]`` back into per-param storage."""
        n = m_fp32.shape[0]
        shape = tuple(m_fp32.shape[1:])
        per = 1
        for d in shape:
            per *= d
        if md in ("bfloat16", "float32"):
            ms = [s[prefix].reshape(shape) for s in states]
            torch._foreach_copy_(ms, list(m_fp32.unbind(0)))
        elif md == "int8":
            row = shape[0] if len(shape) >= 2 else 1
            rest = max(per // row, 1)
            q, new_scale = _quant_int8_stacked(m_fp32.reshape(n, row, rest))  # [N,R,rest]->[N,R,1]
            torch._foreach_copy_(
                [s[prefix].reshape(row, rest) for s in states], list(q.unbind(0))
            )
            for s, sc in zip(states, new_scale.unbind(0), strict=True):  # sc: [R, 1]
                s[f"{prefix}_scale"] = sc.reshape(row, 1) if len(shape) >= 2 else sc.reshape(1)
        else:  # 4bit
            bs = states[0][f"{prefix}_block"]
            new_packed, new_scale = _quant_4bit_stacked(m_fp32.reshape(n, per), bs)
            torch._foreach_copy_([s[prefix] for s in states], list(new_packed.unbind(0)))
            for s, sc in zip(states, new_scale.unbind(0), strict=True):
                s[f"{prefix}_scale"].copy_(sc)

    # -------------------------------------------------------------------- step
    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            for p in params:
                if p.grad.is_sparse:
                    raise RuntimeError("MARS does not support sparse gradients")
            group["step"] += 1
            if not params:
                continue
            if group["gradient_centralization"]:
                centralize_grads_(params)
            if self._foreach and self._group_foreach_eligible(group):
                chunk_budget = foreach_budget(
                    self._foreach_stack_budget, self._foreach_batch_cutoff,
                    _STACK_BYTES_PER_ELEM, params[0].device,
                )
                cutoff = min(self._foreach_batch_cutoff, chunk_budget // 2)
                fast: list[Tensor] = []
                slow: list[Tensor] = []
                for p in params:
                    (fast if self._param_foreach_eligible(p, group, cutoff) else slow).append(p)
                if len(fast) >= 2:
                    self._step_foreach(fast, group, chunk_budget)
                    for p in slow:
                        self._step_one_param(p, group)
                else:
                    for p in params:
                        self._step_one_param(p, group)
            else:
                for p in params:
                    self._step_one_param(p, group)
        return loss

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state, preserving both codec buffers' stored dtype on resume."""
        load_state_dict_preserving_dtypes(self, state_dict)

    # ----------------------------------------------------------- coefficients
    @staticmethod
    def _coeffs(group: dict[str, Any]) -> dict[str, float]:
        """Per-step scalar coefficients (shared by the per-param and foreach paths)."""
        beta1, beta2 = group["betas"]
        gamma = group["gamma"]
        step = group["step"]
        bc1 = 1.0 - beta1 ** step
        bc2_sq = math.sqrt(1.0 - beta2 ** step)
        return {
            "beta1": beta1,
            "beta2": beta2,
            "corr": gamma * (beta1 / (1.0 - beta1)),  # gamma * beta1/(1-beta1)
            "bc1": bc1,
            "bc2_sq": bc2_sq,
            "step_size": group["lr"] / bc1,
        }

    # ----------------------------------------------------------------- foreach
    @staticmethod
    def _group_foreach_eligible(group: dict[str, Any]) -> bool:
        return group["bf16_method"] != "kahan"  # kahan needs a per-param shift buffer

    @staticmethod
    def _param_foreach_eligible(p: Tensor, group: dict[str, Any], cutoff: int) -> bool:
        if p.ndim == 0 or p.numel() > cutoff:
            return False
        if (
            group["bf16_method"] == "stochastic_rounding"
            and is_low_precision(p)
            and p.dtype != torch.bfloat16  # fp16+SR unsupported -> per-param (raises)
        ):
            return False
        if p.ndim > 2:
            # Matrixized conv writes back through a reshaped view -> needs contiguity.
            return p.data.is_contiguous() and p.grad.is_contiguous()
        return True

    @torch.no_grad()
    def _step_foreach(self, params: list[Tensor], group: dict[str, Any], budget: int) -> None:
        """Batched step. Factored (ndim>=2, variance reduction) and 1-D (plain AdamW) buckets."""
        c = self._coeffs(group)
        md = group["momentum_dtype"]

        factored_buckets: dict[tuple[Any, ...], list[Tensor]] = {}
        flat_buckets: dict[tuple[Any, ...], list[Tensor]] = {}
        for p in params:
            state = self.state[p]
            if not state:
                self._init_state(p, state, group)
            g = p.grad
            if g.ndim >= 2:
                matrixize = g.ndim > 2
                eff = (g.shape[0], g.numel() // g.shape[0]) if matrixize else tuple(g.shape)
                factored_buckets.setdefault((eff, p.dtype, matrixize), []).append(p)
            else:
                flat_buckets.setdefault((g.shape[0], p.dtype), []).append(p)

        for (eff, _dtype, matrixize), plist in factored_buckets.items():
            stepn = max(1, budget // max(eff[0] * eff[1], 1))
            for i in range(0, len(plist), stepn):
                self._factored_bucket(plist[i:i + stepn], eff, matrixize, md, c, group)
        for (length, _dtype), plist in flat_buckets.items():
            stepn = max(1, budget // max(length, 1))
            for i in range(0, len(plist), stepn):
                self._nonfactored_bucket(plist[i:i + stepn], length, md, c, group)

    @torch.no_grad()
    def _factored_bucket(
        self,
        plist: list[Tensor],
        eff: tuple[int, int],
        matrixize: bool,
        md: str,
        c: dict[str, float],
        group: dict[str, Any],
    ) -> None:
        R, C = eff  # noqa: N806
        eps = group["eps"]
        wd = group["weight_decay"]
        cautious, bf16_method, mars_clip = group["cautious"], group["bf16_method"], group["mars_clip"]

        def mat(t: Tensor) -> Tensor:
            return t.view(R, C) if matrixize else t

        states = [self.state[p] for p in plist]
        n = len(plist)
        rows = [s["row"] for s in states]
        cols = [s["col"] for s in states]

        grad = torch.stack([mat(p.grad).float() for p in plist])          # [N, R, C]

        if wd != 0:
            self._apply_decoupled_wd_batched(plist, mat, group["lr"] * wd)

        # Variance-reduction corrected gradient c_t (matrix path).
        last = self._dequant_stacked(states, "last_grad", md, (R, C))      # [N, R, C]
        c_t = grad.sub(last).mul_(c["corr"]).add_(grad)                    # g + corr*(g - g_prev)
        if mars_clip:
            norm = c_t.reshape(n, -1).norm(dim=1).clamp_(min=1.0)         # ||c_t|| or 1
            c_t.div_(norm.view(n, 1, 1))
        # Store raw grad as next step's previous gradient.
        self._store_stacked(states, "last_grad", md, grad)

        # Factored second moment of c_t (HF eps1 placement; eps folded into eps1).
        omb2 = 1.0 - c["beta2"]
        ct_sq = c_t * c_t
        if eps > 0:
            ct_sq = ct_sq.add_(eps)
        row = torch.stack(rows)
        col = torch.stack(cols)
        row.lerp_(ct_sq.mean(dim=-1), omb2)
        col.lerp_(ct_sq.mean(dim=-2), omb2)
        torch._foreach_copy_(rows, list(row.unbind(0)))
        torch._foreach_copy_(cols, list(col.unbind(0)))

        r_factor = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)  # [N, R, 1]
        c_factor = col.rsqrt().unsqueeze(-2)                                       # [N, 1, C]
        inv_denom = (r_factor * c_factor).mul_(c["bc2_sq"])                        # 1/sqrt(v_hat)

        # First moment EMA of c_t.
        m = self._dequant_stacked(states, "exp_avg", md, (R, C))
        m.mul_(c["beta1"]).add_(c_t, alpha=1.0 - c["beta1"])
        self._store_stacked(states, "exp_avg", md, m)

        delta = m.mul_(inv_denom).mul_(c["step_size"])                            # full step

        if cautious:
            delta = cautious_batched_(delta, grad)

        subtract_batched_([mat(p.data) for p in plist], delta, bf16_method)

    @torch.no_grad()
    def _nonfactored_bucket(
        self,
        plist: list[Tensor],
        length: int,
        md: str,
        c: dict[str, float],
        group: dict[str, Any],
    ) -> None:
        eps = group["eps"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]

        states = [self.state[p] for p in plist]
        vs = [s["v"] for s in states]

        grad = torch.stack([p.grad.float() for p in plist])               # [N, L]

        if wd != 0:
            self._apply_decoupled_wd_batched(plist, lambda t: t, group["lr"] * wd)

        # 1-D path: plain AdamW on the RAW gradient (no correction, no clip).
        v = torch.stack(vs)
        v.mul_(c["beta2"]).addcmul_(grad, grad, value=1.0 - c["beta2"])
        torch._foreach_copy_(vs, list(v.unbind(0)))
        de_nom = v.add(1e-15).sqrt_().add_(eps)
        de_nom.div_(c["bc2_sq"])

        m = self._dequant_stacked(states, "exp_avg", md, (length,))
        m.mul_(c["beta1"]).add_(grad, alpha=1.0 - c["beta1"])
        self._store_stacked(states, "exp_avg", md, m)

        delta = m.div_(de_nom).mul_(c["step_size"])

        if cautious:
            delta = cautious_batched_(delta, grad)

        subtract_batched_([p.data for p in plist], delta, bf16_method)

    @torch.no_grad()
    def _apply_decoupled_wd_batched(self, plist: list[Tensor], mat: Any, factor: float) -> None:
        """In-place decoupled WD ``p *= (1 - factor)`` on the (matrixized) weights."""
        torch._foreach_mul_([mat(p.data) for p in plist], 1.0 - factor)

    # ---------------------------------------------------------- per-parameter
    @torch.no_grad()
    def _step_one_param(self, p: Tensor, group: dict[str, Any]) -> None:
        c = self._coeffs(group)
        md = group["momentum_dtype"]
        eps = group["eps"]
        wd = group["weight_decay"]
        cautious, bf16_method, mars_clip = group["cautious"], group["bf16_method"], group["mars_clip"]

        state = self.state[p]
        if not state:
            self._init_state(p, state, group)

        grad = p.grad if p.grad.dtype == torch.float32 else p.grad.float()
        ndim = grad.ndim
        factored = ndim >= 2

        if wd != 0:
            p.data.mul_(1.0 - group["lr"] * wd)

        if factored:
            matrixize = ndim > 2
            gv = grad.reshape(grad.shape[0], -1) if matrixize else grad

            # Variance-reduction corrected gradient c_t.
            last = self._dequant_one(state, "last_grad", md, gv)
            c_t = gv.sub(last).mul_(c["corr"]).add_(gv)
            if mars_clip:
                norm = c_t.norm()
                if norm > 1.0:
                    c_t.div_(norm)
            self._store_one(state, "last_grad", md, gv)

            update_factored_state(c_t, state["row"], state["col"], c["beta2"], eps)
            r_factor, c_factor = factored_inv_sqrt_factors(state["row"], state["col"])
            inv_denom = (r_factor * c_factor).mul_(c["bc2_sq"])           # 1/sqrt(v_hat)

            m = self._dequant_one(state, "exp_avg", md, gv)
            m.mul_(c["beta1"]).add_(c_t, alpha=1.0 - c["beta1"])
            self._store_one(state, "exp_avg", md, m)

            # m may alias the stored fp32 buffer (``.float()`` is a no-op on fp32), so
            # build the step out-of-place to avoid corrupting the stored EMA.
            delta = m.mul(inv_denom).mul_(c["step_size"])
            if matrixize:
                delta = delta.reshape_as(grad)
        else:
            # 1-D path: plain AdamW on the raw gradient.
            v = state["v"]
            v.mul_(c["beta2"]).addcmul_(grad, grad, value=1.0 - c["beta2"])
            de_nom = v.add(1e-15).sqrt_().add_(eps)
            de_nom.div_(c["bc2_sq"])

            m = self._dequant_one(state, "exp_avg", md, grad)
            m.mul_(c["beta1"]).add_(grad, alpha=1.0 - c["beta1"])
            self._store_one(state, "exp_avg", md, m)

            # See the factored branch: avoid in-place on a possibly-aliased buffer.
            delta = m.div(de_nom).mul_(c["step_size"])

        if cautious:
            delta = cautious_one_(delta, grad)

        subtract_one_(p, delta, state, bf16_method)
