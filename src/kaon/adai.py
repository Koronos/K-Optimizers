"""Adai — Adaptive Inertia (Xie et al. 2022) on kaon's memory backend.

Adai (Xie, Wang, Zhang, Sato, Sugiyama, *Adaptive Inertia: Disentangling the
Effects of Adaptive Learning Rate and Momentum*, ICML 2022, arXiv:2006.15815)
**disentangles** the two ingredients Adam fuses. Adam couples an adaptive
*learning rate* (``1/sqrt(v)``) with a *fixed* momentum ``beta1``; Adai keeps an
Adam-style second moment ``v`` only to derive a **per-parameter, per-step
adaptive momentum** ``beta1_t`` (the "inertia") and then steps **along the
momentum itself** — there is no ``1/sqrt(v)`` shrink on the update. The inertia
is normalized by the *global* mean of the (bias-corrected) second moment across
the whole parameter group, so flat / low-curvature coordinates (small ``v``) get
*more* momentum (inertia closer to 1, to accelerate) and sharp coordinates (large
``v``) get *less* (to avoid overshooting). This adaptive inertia is what biases
Adai toward flat minima — its generalization story.

**The exact update (matches the official zeke-xie/Adai ``adai.py``).** Per group,
step ``t`` (1-indexed); ``beta0`` = ``betas[0]`` (the inertia coefficient, NOT a
fixed first-moment decay), ``beta2`` = ``betas[1]``:

.. code-block:: text

    # ---- pass 1 (GLOBAL reduction over the whole param group) ----
    v   = beta2*v + (1-beta2)*g^2                 # Adam second moment (factored on >=2D)
    v_hat_sum   += sum(v) / (1 - beta2^t)         # accumulated over ALL params
    param_size  += numel
    v_mean = v_hat_sum / param_size               # a single SCALAR for the group

    # ---- pass 2 (per parameter) ----
    v_hat   = v / (1 - beta2^t)                    # per-coordinate
    beta1_t = clamp(1 - (beta0/v_mean) * v_hat, 0, 1 - eps)   # per-COORDINATE inertia
    beta1_prod *= beta1_t                          # per-coordinate running product
    m   = beta1_t*m + (1 - beta1_t)*g              # inertia-weighted first moment
    m_hat = m / (1 - beta1_prod)                   # per-coordinate bias correction
    p  -= lr * m_hat                                # step ALONG the momentum (no 1/sqrt(v))

Weight decay is the AdaiW (decoupled) form ``p *= (1 - lr*wd)`` by default
(``decoupled=True``); ``decoupled=False`` is the L2 form ``g += wd*p`` (the
official default — see the discrepancy note below). The decay is applied in pass
1, before the ``v`` update, matching the official order.

**The global ``v_mean`` is Adai's signature and the awkward part for a per-tensor /
foreach optimizer.** It forces a *two-pass* step (exactly like KProdigy's
D-estimate): pass 1 must accumulate ``sum(v_hat)`` and the total element count
across every parameter in the group *before* any weight moves; pass 2 then uses
that one scalar for the per-coordinate inertia. We mirror KProdigy's structure: a
per-param scalar partial ``sum(v_hat)_i`` is computed identically in both the
per-param and foreach paths and **folded in parameter-iteration order**, so the
global ``v_mean`` — and therefore every subsequent step — is *bit-identical*
between the two paths (the ``foreach == per-param`` parity test is the proof).

**Factored second moment and ``v_mean``.** ``ndim >= 2`` weights factor ``v``
into Adafactor row+column EMAs (conv kernels matrixized to ``[out, in*kh*kw]``);
``ndim == 1`` keeps a full per-coordinate ``v``. The official Adai sums the *full*
``exp_avg_sq``. With a factored ``v`` we never materialize the full matrix, so the
global sum uses the **rank-1 reconstruction** ``v_recon = (row / mean(row)) ⊗
col`` (the same reconstruction the denominator factors come from), whose sum is the
cheap outer-product identity ``sum_r(row_r/mean(row)) * sum_c(col_c)``. This is an
*approximation* of the true per-coordinate ``v`` sum (factored ``v`` is rank-1);
it is exact for genuinely rank-1 curvature and a low-rank approximation otherwise.
The per-coordinate ``v_hat`` used for the inertia in pass 2 is the same
reconstruction. Document/measure before trusting the factored path for a
generalization claim; the 1-D path is exact.

**Momentum storage.** ``m`` rides the shared codec storage (bf16/int8/4bit), read
back via the codec's dequant/requant primitives (Adai runs its own inertia EMA, so
the codec's Adam ``ema_*`` helpers do not apply — same pattern as AdaPNM/KProdigy).
``beta1_prod`` is a per-coordinate running product in ``(0, 1)`` that must NOT be
quantized: it is multiplied every step and a small relative error compounds, so it
is kept **fp32** always (it is the cheap buffer — one fp32 tensor per param,
unavoidable for the bias correction). A ``.clone()`` guards the fp32 momentum read
on the per-param path (reading a stored fp32 buffer via ``.float()`` returns the
*same* tensor; an in-place EMA would corrupt stored state and diverge from foreach).

It is a standard ``torch.optim.Optimizer``. Because of the global ``v_mean``
reduction it is a two-pass optimizer and does **not** support the per-parameter /
gradient-release loop (like KProdigy); use it as a normal ``optimizer.step()``.
"""

from __future__ import annotations

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
from kaon._factored import update_factored_state
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

__all__ = ["Adai"]

MomentumDtype = Literal["bfloat16", "float32", "int8", "4bit"]

# Stored momentum + two fp32 buffers in flight (beta1_prod + reconstructed inertia);
# a touch above Adakaon's 48.
_STACK_BYTES_PER_ELEM = 56


class Adai(Optimizer):
    """Adai (Adaptive Inertia) on kaon's factored / quantized backend.

    Args:
        params: parameters or param-group dicts.
        lr: learning rate. **Adai's LR scale is SGD-like** (the update is the
            inertia-momentum, NOT divided by ``sqrt(v)``), so it is typically much
            larger than an Adam LR — the paper uses values around ``1``. Default
            ``1.0``.
        betas: ``(beta0, beta2)``. ``beta0`` is the **inertia coefficient** (the
            knob that sets how strongly the per-coordinate momentum reacts to the
            global-normalized second moment), NOT a fixed first-moment decay.
            ``beta2`` is the second-moment EMA decay. Default ``(0.1, 0.99)``
            (the official Adai defaults).
        eps: the inertia clamp ceiling is ``1 - eps`` (the maximum per-coordinate
            momentum). Adai uses a comparatively large ``eps`` — default ``1e-3``.
            Also floors the factored ``v`` reductions (Adafactor ``eps1``).
        weight_decay: weight decay. With ``decoupled=True`` (default) it is the
            AdaiW form ``p *= (1 - lr*wd)``; with ``decoupled=False`` it is L2
            ``g += wd*p`` (the official Adai default). Applied in pass 1, before
            the ``v`` update (official order).
        decoupled: AdamW-style decoupled weight decay. **Default ``True``** (the
            AdaiW variant). The official ``adai.py`` defaults this ``False`` (L2);
            we default to decoupled to match kaon's house style and the rest of the
            library — pass ``decoupled=False`` for the official L2 behaviour.
        cautious: cautious masking (Liang et al. 2024) on the final inertia step vs
            the gradient. Default ``True``; pin ``False`` to match the bare Adai
            math.
        gradient_centralization: centralize ``>=2``-D gradients before the step
            (Yong et al. 2020). Default ``True``; pin ``False`` for bare Adai.
        momentum_dtype: storage for the inertia first moment ``m`` —
            ``"bfloat16"`` (default), ``"float32"``, ``"int8"`` (per-row absmax),
            or ``"4bit"`` (per-block absmax, nibble-packed). ``beta1_prod`` is
            always fp32 (a compounding running product — see the module docstring).
        momentum_4bit_block: block size for ``momentum_dtype="4bit"`` (default
            ``128``). ``0``/negative means whole-tensor.
        bf16_method: weight-update strategy for low-precision params —
            ``"stochastic_rounding"`` (default), ``"kahan"`` (+2 B/param), or
            ``"none"``.
        foreach: batch the step across parameters with stacked multi-tensor ops.
            Default ``True``. Numerically (bit-)identical to the per-parameter path,
            INCLUDING the global ``v_mean`` reduction.
        foreach_batch_cutoff: per-tensor element count above which a weight loops
            instead of stacking (a performance knob; default ``2_000_000``).
        foreach_stack_budget: max elements per stacked chunk. ``None`` (default)
            adapts to free VRAM; an int pins a fixed cap.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1.0,
        betas: tuple[float, float] = (0.1, 0.99),
        eps: float = 1e-3,
        weight_decay: float = 0.0,
        *,
        decoupled: bool = True,
        cautious: bool = True,
        gradient_centralization: bool = True,
        momentum_dtype: MomentumDtype = "bfloat16",
        momentum_4bit_block: int = _FOURBIT_BLOCK,
        bf16_method: str = "stochastic_rounding",
        foreach: bool = True,
        foreach_batch_cutoff: int = FOREACH_BATCH_CUTOFF,
        foreach_stack_budget: int | None = None,
    ) -> None:
        beta0, beta2 = float(betas[0]), float(betas[1])
        if lr < 0.0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if beta0 < 0.0:
            raise ValueError(f"betas[0] (beta0) must be >= 0, got {beta0}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"betas[1] (beta2) must be in [0, 1), got {beta2}")
        if not 0.0 < eps < 1.0:
            raise ValueError(f"eps must be in (0, 1), got {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
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
            "betas": (beta0, beta2),
            "eps": float(eps),
            "weight_decay": weight_decay,
            "decoupled": decoupled,
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
    def _alloc_momentum(self, grad: Tensor, state: dict[str, Any], group: dict[str, Any]) -> None:
        """Allocate the inertia first moment ``m`` in the configured codec layout.

        Storage matches :mod:`kaon._momentum_codec` exactly (per-row int8 scale;
        per-block 4-bit scale, zero == nibble 8) so it resumes bit-exactly.
        """
        md = group["momentum_dtype"]
        if md in ("bfloat16", "float32"):
            dtype = torch.bfloat16 if md == "bfloat16" else torch.float32
            state["m"] = torch.zeros_like(grad, dtype=dtype)
        elif md == "int8":
            state["m"] = torch.zeros_like(grad, dtype=torch.int8)
            state["m_scale"] = torch.ones(
                (grad.shape[0],) + (1,) * (grad.ndim - 1) if grad.ndim >= 2 else (),
                dtype=torch.float32, device=grad.device,
            )
        else:  # 4bit
            numel = grad.numel()
            bs = self._block_size(grad, group)
            nblocks = (numel + bs - 1) // bs
            state["m"] = torch.full(
                ((numel + 1) // 2,), 0x88, dtype=torch.uint8, device=grad.device
            )
            state["m_scale"] = torch.ones(nblocks, dtype=torch.float32, device=grad.device)
            state["m_numel"] = numel
            state["m_block"] = bs

    @torch.no_grad()
    def _init_state(self, p: Tensor, state: dict[str, Any], group: dict[str, Any]) -> None:
        grad = p.grad
        state["step"] = 0
        factored = p.ndim >= 2
        if factored:
            gv = grad if p.ndim == 2 else grad.reshape(grad.shape[0], -1)
            state["row"] = torch.zeros(gv.shape[:-1], dtype=torch.float32, device=p.device)
            state["col"] = torch.zeros(gv.shape[:-2] + gv.shape[-1:], dtype=torch.float32, device=p.device)
        else:
            state["v"] = torch.zeros_like(grad, dtype=torch.float32)
        self._alloc_momentum(grad, state, group)
        # beta1_prod: per-coordinate running product of the adaptive beta1, used for
        # the bias correction m_hat = m / (1 - beta1_prod). ALWAYS fp32 — it compounds.
        state["beta1_prod"] = torch.ones_like(grad, dtype=torch.float32)
        if is_low_precision(p) and group["bf16_method"] == "kahan":
            state["shift"] = torch.zeros_like(p)

    # -------------------------------------------------- momentum read / write
    @staticmethod
    def _dequant_one(state: dict[str, Any], md: str, like: Tensor) -> Tensor:
        """Read ``m`` back as a fresh fp32 tensor shaped like ``like`` (clones on fp32)."""
        if md in ("bfloat16", "float32"):
            # .float() on an fp32 buffer returns the SAME tensor; clone so the EMA
            # never corrupts stored state (the fp32 footgun).
            m = state["m"]
            return (m.float() if m.dtype != torch.float32 else m.clone()).reshape_as(like)
        if md == "int8":
            return state["m"].float().mul_(state["m_scale"]).reshape_as(like)
        m = _dequant_4bit(state["m"], state["m_scale"], state["m_numel"], state["m_block"])
        return m.view_as(like)

    @staticmethod
    def _store_one(state: dict[str, Any], md: str, m_fp32: Tensor) -> None:
        """Write the updated fp32 ``m`` back into the configured storage layout."""
        if md in ("bfloat16", "float32"):
            tgt = state["m"]
            tgt.copy_(m_fp32.reshape(tgt.shape))
        elif md == "int8":
            m_orig = m_fp32.reshape(state["m"].shape)
            state["m"], state["m_scale"] = _quant_int8(m_orig)
        else:  # 4bit
            packed, scale, _ = _quant_4bit(m_fp32, state["m_block"])
            state["m"], state["m_scale"] = packed, scale

    @staticmethod
    def _dequant_stacked(
        states: list[dict[str, Any]], md: str, shape: tuple[int, ...]
    ) -> Tensor:
        """Stacked fp32 ``m`` ``[N, *shape]`` from per-param storage."""
        n = len(states)
        per = 1
        for d in shape:
            per *= d
        if md in ("bfloat16", "float32"):
            return torch.stack([s["m"].reshape(shape) for s in states]).float()
        if md == "int8":
            row = shape[0] if len(shape) >= 2 else 1
            rest = max(per // row, 1)
            m = torch.stack([s["m"].reshape(row, rest) for s in states]).float()  # [N, R, rest]
            scale = torch.stack([s["m_scale"].reshape(row, 1) for s in states])   # [N, R, 1]
            return m.mul_(scale).reshape((n, *shape))
        packed = torch.stack([s["m"] for s in states])
        sc = torch.stack([s["m_scale"] for s in states])
        bs = states[0]["m_block"]
        return _dequant_4bit_stacked(packed, sc, per, bs).reshape((n, *shape))

    @staticmethod
    def _store_stacked(states: list[dict[str, Any]], md: str, m_fp32: Tensor) -> None:
        """Write stacked fp32 ``m`` ``[N, *shape]`` back into per-param storage."""
        n = m_fp32.shape[0]
        shape = tuple(m_fp32.shape[1:])
        per = 1
        for d in shape:
            per *= d
        if md in ("bfloat16", "float32"):
            ms = [s["m"].reshape(shape) for s in states]
            torch._foreach_copy_(ms, list(m_fp32.unbind(0)))
        elif md == "int8":
            row = shape[0] if len(shape) >= 2 else 1
            rest = max(per // row, 1)
            q, new_scale = _quant_int8_stacked(m_fp32.reshape(n, row, rest))  # [N,R,rest]->[N,R,1]
            torch._foreach_copy_([s["m"].reshape(row, rest) for s in states], list(q.unbind(0)))
            for s, sc in zip(states, new_scale.unbind(0), strict=True):  # sc: [R, 1]
                s["m_scale"] = sc.reshape(row, 1) if len(shape) >= 2 else sc.reshape(1)
        else:  # 4bit
            bs = states[0]["m_block"]
            new_packed, new_scale = _quant_4bit_stacked(m_fp32.reshape(n, per), bs)
            torch._foreach_copy_([s["m"] for s in states], list(new_packed.unbind(0)))
            for s, sc in zip(states, new_scale.unbind(0), strict=True):
                s["m_scale"].copy_(sc)

    # -------------------------------------------------------------------- step
    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Each param group is its own v_mean scope. The official Adai reduces over
        # the whole optimizer, but kaon groups carry independent hyperparameters, so
        # a per-group reduction is the natural, self-consistent boundary; a
        # single-group setup matches the official exactly.
        for group in self.param_groups:
            self._step_group(group)
        return loss

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state, preserving the quantized momentum's stored dtype.

        torch's default ``load_state_dict`` upcasts every state tensor to the
        param's dtype (fp32), inflating bf16/int8/4bit ``m`` back to fp32 on resume.
        Delegate to the shared helper that restores each tensor to how it was saved.
        """
        load_state_dict_preserving_dtypes(self, state_dict)

    @torch.no_grad()
    def _step_group(self, group: dict[str, Any]) -> None:
        params = [p for p in group["params"] if p.grad is not None]
        for p in params:
            if p.grad.is_sparse:
                raise RuntimeError("Adai does not support sparse gradients")
        group["step"] += 1
        if not params:
            return
        if group["gradient_centralization"]:
            centralize_grads_(params)

        beta0, beta2 = group["betas"]
        step = group["step"]
        bc2 = 1.0 - beta2 ** step

        for p in params:
            state = self.state[p]
            if "step" not in state:
                self._init_state(p, state, group)

        # ---- pass 1: weight decay + v EMA + GLOBAL v_hat-sum reduction ----
        # The per-param scalar sum(v_hat)_i is accumulated in parameter-iteration
        # order in BOTH paths (pass 1 is always this loop), so v_mean is bit-identical.
        v_hat_sum = torch.zeros((), dtype=torch.float32, device=params[0].device)
        param_size = 0
        eps1 = group["eps"]
        wd = group["weight_decay"]
        decoupled = group["decoupled"]

        for p in params:
            grad = p.grad if p.grad.dtype == torch.float32 else p.grad.float()
            # Weight decay BEFORE the v update (official order).
            if wd != 0:
                if decoupled:
                    p.data.mul_(1.0 - group["lr"] * wd)
                else:
                    grad = grad.add(p.detach().float(), alpha=wd)
                    # Stash the WD-folded grad so pass 2 uses the same gradient the
                    # v update saw (and avoids re-centralizing / recomputing it).
                    self.state[p]["_wd_grad"] = grad
            state = self.state[p]
            if "v" in state:  # 1-D full second moment
                v = state["v"]
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                v_hat_sum = v_hat_sum + v.sum() / bc2
            else:  # factored
                matrixize = grad.ndim > 2
                gv = grad.reshape(grad.shape[0], -1) if matrixize else grad
                update_factored_state(gv, state["row"], state["col"], beta2, eps1)
                # sum of the rank-1 reconstruction (row/mean(row)) ⊗ col:
                #   sum_rc v_recon = sum_r(row_r / mean(row)) * sum_c(col_c).
                row, col = state["row"], state["col"]
                row_norm = row.div(row.mean(dim=-1, keepdim=True))           # row / mean(row)
                v_recon_sum = (row_norm.sum(dim=-1) * col.sum(dim=-1)).sum()
                v_hat_sum = v_hat_sum + v_recon_sum / bc2
            param_size += p.numel()

        v_mean = v_hat_sum / max(param_size, 1)

        # ---- pass 2: per-coordinate inertia + step ----
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
                self._pass2_foreach(fast, group, beta0, bc2, v_mean, chunk_budget)
                for p in slow:
                    self._pass2_one_param(p, group, beta0, bc2, v_mean)
            else:
                for p in params:
                    self._pass2_one_param(p, group, beta0, bc2, v_mean)
        else:
            for p in params:
                self._pass2_one_param(p, group, beta0, bc2, v_mean)

        # Clean the pass-1 grad stash.
        if wd != 0 and not decoupled:
            for p in params:
                self.state[p].pop("_wd_grad", None)

    # ----------------------------------------------------------- foreach gates
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
            return p.data.is_contiguous() and p.grad.is_contiguous()
        return True

    def _grad_fp32(self, p: Tensor, group: dict[str, Any]) -> Tensor:
        """The fp32 gradient pass 2 should use (the WD-folded grad from pass 1 if any)."""
        state = self.state[p]
        if "_wd_grad" in state:
            return state["_wd_grad"]
        return p.grad if p.grad.dtype == torch.float32 else p.grad.float()

    # ----------------------------------------------------- inertia primitives
    @staticmethod
    def _inertia(v_hat: Tensor, v_mean: Tensor, beta0: float, eps: float) -> Tensor:
        """beta1 = clamp(1 - (beta0/v_mean)*v_hat, 0, 1-eps). Consumes ``v_hat``."""
        return (1.0 - v_hat.mul_(beta0 / v_mean)).clamp_(0.0, 1.0 - eps)

    # ---------------------------------------------------------- per-parameter
    @torch.no_grad()
    def _pass2_one_param(
        self, p: Tensor, group: dict[str, Any], beta0: float, bc2: float, v_mean: Tensor,
    ) -> None:
        md = group["momentum_dtype"]
        eps = group["eps"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]
        lr = group["lr"]
        state = self.state[p]
        grad = self._grad_fp32(p, group)
        ndim = grad.ndim

        if ndim >= 2:  # factored
            matrixize = ndim > 2
            gv = grad.reshape(grad.shape[0], -1) if matrixize else grad
            row, col = state["row"], state["col"]
            row_norm = row.div(row.mean(dim=-1, keepdim=True))             # row / mean(row)
            v_hat = (row_norm.unsqueeze(-1) * col.unsqueeze(-2)).div_(bc2)  # [R, C]
            beta1 = self._inertia(v_hat, v_mean, beta0, eps)
            delta = self._inertia_step(state, md, gv, beta1, lr)
            if matrixize:
                delta = delta.reshape_as(grad)
        else:  # 1-D full second moment
            v_hat = state["v"].div(bc2)
            beta1 = self._inertia(v_hat, v_mean, beta0, eps)
            delta = self._inertia_step(state, md, grad, beta1, lr)

        if cautious:
            delta = cautious_one_(delta, grad)
        subtract_one_(p, delta, state, bf16_method)

    def _inertia_step(
        self, state: dict[str, Any], md: str, grad: Tensor, beta1: Tensor, lr: float
    ) -> Tensor:
        """Per-coordinate inertia EMA + bias-corrected step ``lr * m_hat``.

        ``m = beta1*m + (1-beta1)*g``; ``beta1_prod *= beta1``;
        ``m_hat = m / (1 - beta1_prod)``; returns ``lr * m_hat`` (caller subtracts).
        ``grad`` is the (matrixized) view; ``beta1`` has the same shape.
        """
        m = self._dequant_one(state, md, grad)                       # fp32, cloned
        m.mul_(beta1).addcmul_(1.0 - beta1, grad)                    # inertia EMA
        self._store_one(state, md, m)
        bp = state["beta1_prod"].reshape_as(grad)
        bp.mul_(beta1)
        m_hat = m.div_(1.0 - bp)
        return m_hat.mul_(lr)

    # ----------------------------------------------------------------- foreach
    @torch.no_grad()
    def _pass2_foreach(
        self, params: list[Tensor], group: dict[str, Any], beta0: float,
        bc2: float, v_mean: Tensor, budget: int,
    ) -> None:
        md = group["momentum_dtype"]
        factored_buckets: dict[tuple[Any, ...], list[Tensor]] = {}
        flat_buckets: dict[tuple[Any, ...], list[Tensor]] = {}
        for p in params:
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
                self._factored_bucket(
                    plist[i:i + stepn], eff, matrixize, md, beta0, bc2, v_mean, group
                )
        for (length, _dtype), plist in flat_buckets.items():
            stepn = max(1, budget // max(length, 1))
            for i in range(0, len(plist), stepn):
                self._flat_bucket(plist[i:i + stepn], length, md, beta0, bc2, v_mean, group)

    @torch.no_grad()
    def _factored_bucket(
        self, plist: list[Tensor], eff: tuple[int, int], matrixize: bool, md: str,
        beta0: float, bc2: float, v_mean: Tensor, group: dict[str, Any],
    ) -> None:
        R, C = eff  # noqa: N806
        eps = group["eps"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]
        lr = group["lr"]

        def mat(t: Tensor) -> Tensor:
            return t.view(R, C) if matrixize else t

        states = [self.state[p] for p in plist]
        grad = torch.stack([mat(self._grad_fp32(p, group)) for p in plist])     # [N, R, C]
        row = torch.stack([s["row"] for s in states])                           # [N, R]
        col = torch.stack([s["col"] for s in states])                           # [N, C]

        row_norm = row.div(row.mean(dim=-1, keepdim=True))                      # [N, R]
        v_hat = (row_norm.unsqueeze(-1) * col.unsqueeze(-2)).div_(bc2)          # [N, R, C]
        beta1 = self._inertia(v_hat, v_mean, beta0, eps)                        # [N, R, C]

        m = self._dequant_stacked(states, md, (R, C))                          # [N, R, C]
        m.mul_(beta1).addcmul_(1.0 - beta1, grad)
        self._store_stacked(states, md, m)

        bp = torch.stack([s["beta1_prod"].reshape(R, C) for s in states])       # [N, R, C]
        bp.mul_(beta1)
        torch._foreach_copy_(
            [s["beta1_prod"].reshape(R, C) for s in states], list(bp.unbind(0))
        )
        delta = m.div_(1.0 - bp).mul_(lr)

        if cautious:
            delta = cautious_batched_(delta, grad)
        subtract_batched_([mat(p.data) for p in plist], delta, bf16_method)

    @torch.no_grad()
    def _flat_bucket(
        self, plist: list[Tensor], length: int, md: str, beta0: float, bc2: float,
        v_mean: Tensor, group: dict[str, Any],
    ) -> None:
        eps = group["eps"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]
        lr = group["lr"]
        states = [self.state[p] for p in plist]

        grad = torch.stack([self._grad_fp32(p, group) for p in plist])          # [N, L]
        v = torch.stack([s["v"] for s in states])                               # [N, L]
        v_hat = v.div(bc2)
        beta1 = self._inertia(v_hat, v_mean, beta0, eps)                        # [N, L]

        m = self._dequant_stacked(states, md, (length,))                       # [N, L]
        m.mul_(beta1).addcmul_(1.0 - beta1, grad)
        self._store_stacked(states, md, m)

        bp = torch.stack([s["beta1_prod"] for s in states])                     # [N, L]
        bp.mul_(beta1)
        torch._foreach_copy_([s["beta1_prod"] for s in states], list(bp.unbind(0)))
        delta = m.div_(1.0 - bp).mul_(lr)

        if cautious:
            delta = cautious_batched_(delta, grad)
        subtract_batched_([p.data for p in plist], delta, bf16_method)
