"""Schedule-Free AdamW on kaon's memory backend.

Schedule-Free learning (Defazio et al. 2024, *The Road Less Scheduled*,
arXiv:2405.15682) replaces a learning-rate **schedule** with online **iterate
averaging**, so the optimizer needs no decay schedule and no knowledge of the
total step count to land near the schedule-tuned optimum. It is implemented here
on the precision + memory machinery proven in
:class:`~kaon.adakaon.Adakaon` / :class:`~kaon.adapnm.AdaPNM`: the
factored quantized second moment (``_factored``), the quantized first-moment
**storage** codec (``_momentum_codec``) reused to store the full-size ``z``
sequence, bf16-correct weight write-back (``_backend.subtract_*``), cautious
masking, and gradient centralization.

The three sequences
-------------------
Schedule-Free maintains three *logical* weight sequences (Defazio's notation):

* ``z`` — the SGD-/Adam-like base sequence (full-size; the only extra full-size
  state this optimizer keeps).
* ``x`` — the running **average** of the ``z`` iterates; the sequence you keep,
  evaluate, sample and checkpoint.
* ``y`` — the **interpolation point** ``y = beta1*x + (1 - beta1)*z`` at which the
  gradient is evaluated. (This is the official ``facebookresearch/schedule_free``
  convention, where ``beta1`` weights ``x``; Defazio's paper writes the same
  interpolation with the roles of the coefficients swapped — the code matches the
  official repo exactly.)

Crucially, only ``z`` (and the second moment) is *stored*. ``x`` is never
materialized: the parameter buffer ``p.data`` itself carries ``y`` during
training and is converted to ``x`` (and back) in place by :meth:`eval` /
:meth:`train`. This is the closed-form swap from the official
``facebookresearch/schedule_free`` repo, carried over exactly.

The update (matches the official ``AdamWScheduleFree``)
-------------------------------------------------------
Per step ``k`` (0-indexed internally; ``t = k + 1``), with the gradient ``g``
evaluated at ``y`` (so ``p.data`` must hold ``y`` — call :meth:`train` first):

.. code-block:: text

    sched      = (k+1)/warmup_steps  if k < warmup_steps else 1.0
    lr_t       = lr * sched
    lr_max     = max(lr_max, lr_t)
    weight     = (k+1)**r * lr_max**weight_lr_power
    weight_sum += weight
    ckp1       = weight / weight_sum                      # the 1/t-style avg weight

    v          = beta2*v + (1-beta2)*g^2                  # Adam 2nd moment (factored)
    denom      = sqrt(v / (1-beta2^t)) + eps
    d          = g / denom                                # "grad_normalized"
    d         += weight_decay * y                         # decoupled WD, evaluated at y

    y          = (1-ckp1)*y + ckp1*z                      # average toward z (forms x in-place)
    y         += d * (lr_t*(beta1*(1-ckp1) - 1))          # the y-update
    z         -= lr_t * d                                 # the z step

``x`` (what :meth:`eval` exposes) is implied by ``y`` and ``z`` via
``x = (y - (1-beta1)*z)/beta1``; equivalently ``y = beta1*x + (1-beta1)*z``. The
two ``lerp`` swaps are exactly:

.. code-block:: text

    eval():   p.lerp_(z, 1 - 1/beta1)     # y -> x
    train():  p.lerp_(z, 1 - beta1)       # x -> y

These are inverses (``train(eval(y)) == y`` up to fp round-off), which the
round-trip test exercises.

``beta1`` here is the **Schedule-Free interpolation momentum** (default ``0.9``),
NOT an Adam first moment — the default has no first-moment buffer at all
(``inner_momentum=0``), so the only adaptive state is the factored ``v``. Setting
``inner_momentum > 0`` (a recommended ``0.9``) adds an AdamW-style first moment
``exp_avg`` and feeds ``exp_avg/bias_correction1`` (instead of the raw ``g``) into
``d``; that costs one extra full-size buffer (stored through the same codec as
``z``).

What is reused vs new
---------------------
Reused from Adakaon/AdaPNM: the factored second-moment helpers
(:mod:`kaon._factored`), the first-moment storage codec
(:mod:`kaon._momentum_codec`) — here repurposed to store the full-size ``z``
(and optional ``exp_avg``) buffers at the configured ``momentum_dtype`` —
``load_state_dict_preserving_dtypes`` for dtype-safe resume, the
stochastic-rounding bf16 weight write (:func:`kaon._backend.subtract_*`),
cautious masking, gradient centralization, and the bucketed foreach pattern.
New here: the three-sequence ``z``/``x``/``y`` recurrence, the ``c_t`` (``ckp1``)
polynomial-weighted averaging, and the in-place :meth:`train` / :meth:`eval`
``y <-> x`` swap.

Required call pattern
---------------------
``.train()`` puts ``p.data`` in the ``y``-view (gradient/training view);
``.eval()`` puts it in the ``x``-view (the kept/averaged weights). **Default state
is train.** Call ``.train()`` before each training step's forward/backward and
``.eval()`` before sampling / validation / checkpointing::

    opt = ScheduleFree(model.parameters(), lr=2e-3)
    opt.train()
    for batch in loader:
        opt.zero_grad(); loss(model(batch)).backward(); opt.step()
    opt.eval()       # p.data now holds x
    sample_or_checkpoint(model)
    opt.train()      # back to y for more training

(Calling :meth:`train` / :meth:`eval` is idempotent — a no-op if already in that
mode — so it is safe to bracket liberally.)
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
from kaon._stochastic_rounding import add_stochastic_

__all__ = ["ScheduleFree"]

MomentumDtype = Literal["bfloat16", "float32", "int8", "4bit"]

# One full-size z (and optional exp_avg) + factored v; mirrors AdaPNM's two-momenta
# working-set estimate closely enough for the foreach budget heuristic.
_STACK_BYTES_PER_ELEM = 48


class ScheduleFree(Optimizer):
    """Schedule-Free AdamW (Defazio et al. 2024) on kaon's memory backend.

    The model's parameter buffer holds ``y`` (the interpolation point) in **train**
    mode and ``x`` (the averaged, kept weights) in **eval** mode. You MUST call
    :meth:`train` before training steps and :meth:`eval` before sampling /
    checkpointing (default mode is train). See the module docstring.

    Args:
        params: parameters or param-group dicts.
        lr: base learning rate (default ``2.5e-3``, the official Schedule-Free
            default; Schedule-Free typically wants a *higher* constant LR than a
            scheduled AdamW).
        betas: ``(beta1, beta2)``. ``beta1`` is the **Schedule-Free interpolation
            momentum** (the ``y <-> x <-> z`` mixing coefficient, default ``0.9``) —
            it is *not* an Adam first moment. ``beta2`` is the (factored)
            second-moment decay (default ``0.999``).
        eps: term added to ``sqrt(v_hat)`` in the denominator (non-factored path);
            folded into the Adafactor ``eps1`` on the factored path.
        weight_decay: decoupled (AdamW-style) weight decay, evaluated at ``y`` and
            folded into the normalized gradient ``d`` (matching the official repo).
        warmup_steps: linear LR warmup over this many steps (default ``0``). The
            Schedule-Free replacement for a warmup schedule.
        r: polynomial power in the iterate-average weighting ``weight = t**r *
            lr_max**weight_lr_power`` (default ``0.0``).
        weight_lr_power: LR power in the average weighting (default ``2.0``). With
            ``r=0`` and constant LR this makes ``ckp1 = 1/t`` (uniform averaging).
        inner_momentum: optional AdamW first-moment beta (default ``0.0`` = off, no
            buffer). ``0.9`` is the recommended on-value; adds one full-size buffer
            (stored at ``momentum_dtype``).
        cautious: cautious masking (Liang et al. 2024) on the normalized gradient
            ``d`` vs the raw gradient. On by default.
        gradient_centralization: Gradient Centralization (Yong et al. 2020) on
            ``ndim>=2`` grads. On by default (pin ``False`` for reference parity).
        momentum_dtype: storage dtype for the full-size ``z`` (and optional
            ``exp_avg``) buffers — ``"bfloat16"`` (default), ``"float32"``,
            ``"int8"`` or ``"4bit"``. **Note:** ``z`` is read+written every step and
            participates in the *exact* iterate average, so quantizing it injects a
            small per-step requant error; ``"float32"`` is the bit-exact choice (the
            reference test uses it), ``"bfloat16"`` is the memory-friendly default
            mirroring how kaon stores weights.
        momentum_4bit_block: block size for ``momentum_dtype="4bit"`` (default
            ``128``).
        bf16_method: low-precision weight-write strategy for ``z`` writes and the
            ``y`` write-back — ``"stochastic_rounding"`` (default), ``"kahan"`` or
            ``"none"``.
        foreach: batch the step with multi-tensor ops (default ``True``); numerically
            equal to the per-param path.
        foreach_batch_cutoff: per-tensor element cap above which a weight loops
            (default ``2_000_000``).
        foreach_stack_budget: max elements per stacked chunk (``None`` adapts to VRAM).
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 2.5e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        inner_momentum: float = 0.0,
        *,
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
        if not 0.0 < beta1 < 1.0:
            raise ValueError(f"betas[0] must be in (0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"betas[1] must be in [0, 1), got {beta2}")
        if not 0.0 <= inner_momentum < 1.0:
            raise ValueError(f"inner_momentum must be in [0, 1), got {inner_momentum}")
        if lr < 0.0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if eps < 0.0:
            raise ValueError(f"eps must be >= 0, got {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
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
            "warmup_steps": warmup_steps,
            "r": float(r),
            "weight_lr_power": float(weight_lr_power),
            "inner_momentum": float(inner_momentum),
            "cautious": cautious,
            "gradient_centralization": gradient_centralization,
            "momentum_dtype": momentum_dtype,
            "momentum_4bit_block": momentum_4bit_block,
            "bf16_method": bf16_method,
            "step": 0,            # = official k (0-indexed step count taken so far)
            "weight_sum": 0.0,
            "lr_max": -1.0,
            "train_mode": True,   # default to train (p.data holds y)
        }
        super().__init__(params, defaults)
        self._foreach = foreach
        self._foreach_batch_cutoff = foreach_batch_cutoff
        self._foreach_stack_budget = foreach_stack_budget

    # =================================================================== train/eval
    @torch.no_grad()
    def eval(self) -> None:
        """Switch ``p.data`` from the ``y`` (train) view to the ``x`` (averaged) view.

        Call before sampling / validation / checkpointing. In-place closed-form
        ``p <- p + (1 - 1/beta1)*(z - p)`` (== ``y -> x``). Idempotent: a no-op if
        already in eval mode.
        """
        for group in self.param_groups:
            if not group["train_mode"]:
                continue
            beta1, _ = group["betas"]
            for p in group["params"]:
                state = self.state.get(p)
                if state and "z" in state:
                    z = self._dequant_z(state, group["momentum_dtype"], p)
                    p.lerp_(z.to(p.dtype), weight=1.0 - 1.0 / beta1)
            group["train_mode"] = False

    @torch.no_grad()
    def train(self) -> None:
        """Switch ``p.data`` from the ``x`` (eval) view back to the ``y`` (train) view.

        Call before training steps. In-place closed-form ``p <- p + (1 - beta1)*(z - p)``
        (== ``x -> y``), the exact inverse of :meth:`eval`. Idempotent: a no-op if
        already in train mode.
        """
        for group in self.param_groups:
            if group["train_mode"]:
                continue
            beta1, _ = group["betas"]
            for p in group["params"]:
                state = self.state.get(p)
                if state and "z" in state:
                    z = self._dequant_z(state, group["momentum_dtype"], p)
                    p.lerp_(z.to(p.dtype), weight=1.0 - beta1)
            group["train_mode"] = True

    # ===================================================================== z storage
    # z (and the optional inner-momentum exp_avg) are full-size; they are stored
    # through the shared first-moment codec layout so a configured int8/4bit
    # momentum_dtype keeps z compact and resumes bit-exactly. The per-param and
    # stacked read/write helpers mirror AdaPNM's (one buffer instead of two).
    @staticmethod
    def _block_size(grad: Tensor, group: dict[str, Any]) -> int:
        bs = group["momentum_4bit_block"]
        numel = grad.numel()
        return numel if bs <= 0 else (min(bs, numel) if numel > 0 else 1)

    @torch.no_grad()
    def _alloc_full(
        self, prefix: str, src: Tensor, state: dict[str, Any], group: dict[str, Any], *, copy: bool
    ) -> None:
        """Allocate a full-size buffer (``z`` or ``exp_avg``) in the codec layout.

        ``copy=True`` initializes the (float) buffer from ``src`` (used for ``z``,
        which starts at ``x0 == p``); ``copy=False`` zero-initializes (``exp_avg``).
        The quantized layouts always start at zero (the +8 nibble / unit scale), so
        a copy-init is only honored for the float codecs.
        """
        md = group["momentum_dtype"]
        if md in ("bfloat16", "float32"):
            dtype = torch.bfloat16 if md == "bfloat16" else torch.float32
            if copy:
                state[prefix] = src.detach().to(dtype).clone()
            else:
                state[prefix] = torch.zeros_like(src, dtype=dtype)
        elif md == "int8":
            if copy:
                q, scale = _quant_int8(src.detach().float())
                state[prefix], state[f"{prefix}_scale"] = q, scale
            else:
                state[prefix] = torch.zeros_like(src, dtype=torch.int8)
                state[f"{prefix}_scale"] = torch.ones(
                    (src.shape[0],) + (1,) * (src.ndim - 1) if src.ndim >= 2 else (),
                    dtype=torch.float32, device=src.device,
                )
        else:  # 4bit
            numel = src.numel()
            bs = self._block_size(src, group)
            nblocks = (numel + bs - 1) // bs
            if copy:
                packed, scale, _ = _quant_4bit(src.detach().float(), bs)
                state[prefix], state[f"{prefix}_scale"] = packed, scale
            else:
                state[prefix] = torch.full(
                    ((numel + 1) // 2,), 0x88, dtype=torch.uint8, device=src.device
                )
                state[f"{prefix}_scale"] = torch.ones(
                    nblocks, dtype=torch.float32, device=src.device
                )
            state[f"{prefix}_numel"] = numel
            state[f"{prefix}_block"] = bs

    @torch.no_grad()
    def _init_state(self, p: Tensor, state: dict[str, Any], group: dict[str, Any]) -> None:
        grad = p.grad
        factored = p.ndim >= 2
        if factored:
            gv = grad if p.ndim == 2 else grad.reshape(grad.shape[0], -1)
            state["row"] = torch.zeros(gv.shape[:-1], dtype=torch.float32, device=p.device)
            state["col"] = torch.zeros(
                gv.shape[:-2] + gv.shape[-1:], dtype=torch.float32, device=p.device
            )
        else:
            state["v"] = torch.zeros_like(grad, dtype=torch.float32)
        # z starts at x0 == the current parameter (which == y0 == x0 at k=0).
        self._alloc_full("z", p, state, group, copy=True)
        if group["inner_momentum"] != 0:
            self._alloc_full("exp_avg", grad, state, group, copy=False)
        if is_low_precision(p) and group["bf16_method"] == "kahan":
            state["shift"] = torch.zeros_like(p)
            state["shift_z"] = torch.zeros_like(p)

    @staticmethod
    def _dequant_full(state: dict[str, Any], prefix: str, md: str, like: Tensor) -> Tensor:
        """Read a stored full-size buffer back as a fresh fp32 tensor shaped like ``like``.

        The int8 per-row scale may have been stored with a matrixized ``[row, 1]``
        shape (foreach path) or the original ``[row, 1, ...]`` shape (per-param
        path); both encode the same per-row (dim-0) values, so dequant by reshaping
        the codes to ``[row, -1]`` and broadcasting a flattened ``[row, 1]`` scale —
        agnostic to which shape was stored.
        """
        if md in ("bfloat16", "float32"):
            return state[prefix].float().reshape_as(like)
        if md == "int8":
            codes = state[prefix]
            row = codes.shape[0] if codes.ndim >= 2 else 1
            scale = state[f"{prefix}_scale"].reshape(row, 1) if codes.ndim >= 2 else state[f"{prefix}_scale"]
            return codes.float().reshape(row, -1).mul_(scale).reshape_as(like)
        m = _dequant_4bit(
            state[prefix], state[f"{prefix}_scale"],
            state[f"{prefix}_numel"], state[f"{prefix}_block"],
        )
        return m.view_as(like)

    def _dequant_z(self, state: dict[str, Any], md: str, like: Tensor) -> Tensor:
        return self._dequant_full(state, "z", md, like)

    @staticmethod
    def _store_full(state: dict[str, Any], prefix: str, md: str, m_fp32: Tensor) -> None:
        """Write an updated fp32 full-size buffer back into the configured storage."""
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
    def _dequant_full_stacked(
        states: list[dict[str, Any]], prefix: str, md: str, shape: tuple[int, ...]
    ) -> Tensor:
        """Stacked fp32 full-size buffer ``[N, *shape]`` from per-param storage."""
        n = len(states)
        per = 1
        for d in shape:
            per *= d
        if md in ("bfloat16", "float32"):
            return torch.stack([s[prefix].reshape(shape) for s in states]).float()
        if md == "int8":
            row = shape[0] if len(shape) >= 2 else 1
            rest = max(per // row, 1)
            m = torch.stack([s[prefix].reshape(row, rest) for s in states]).float()
            scale = torch.stack([s[f"{prefix}_scale"].reshape(row, 1) for s in states])
            return m.mul_(scale).reshape((n, *shape))
        packed = torch.stack([s[prefix] for s in states])
        sc = torch.stack([s[f"{prefix}_scale"] for s in states])
        bs = states[0][f"{prefix}_block"]
        return _dequant_4bit_stacked(packed, sc, per, bs).reshape((n, *shape))

    @staticmethod
    def _store_full_stacked(
        states: list[dict[str, Any]], prefix: str, md: str, m_fp32: Tensor
    ) -> None:
        """Write a stacked fp32 full-size buffer ``[N, *shape]`` back into per-param storage."""
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
            q, new_scale = _quant_int8_stacked(m_fp32.reshape(n, row, rest))
            torch._foreach_copy_(
                [s[prefix].reshape(row, rest) for s in states], list(q.unbind(0))
            )
            for s, sc in zip(states, new_scale.unbind(0), strict=True):
                s[f"{prefix}_scale"] = sc.reshape(row, 1) if len(shape) >= 2 else sc.reshape(1)
        else:  # 4bit
            bs = states[0][f"{prefix}_block"]
            new_packed, new_scale = _quant_4bit_stacked(m_fp32.reshape(n, per), bs)
            torch._foreach_copy_([s[prefix] for s in states], list(new_packed.unbind(0)))
            for s, sc in zip(states, new_scale.unbind(0), strict=True):
                s[f"{prefix}_scale"].copy_(sc)

    # ============================================================================ step
    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        if not self.param_groups[0]["train_mode"]:
            raise RuntimeError(
                "ScheduleFree.step() called outside train mode. Call optimizer.train() "
                "before the training step (and optimizer.eval() before sampling / "
                "checkpointing). See the ScheduleFree docstring."
            )
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            for p in params:
                if p.grad.is_sparse:
                    raise RuntimeError("ScheduleFree does not support sparse gradients")
            if not params:
                group["step"] += 1
                continue
            if group["gradient_centralization"]:
                centralize_grads_(params)
            # Compute the per-step coefficients ONCE (advances lr_max / weight_sum
            # exactly once per step), then thread them through both code paths.
            c = self._coeffs(group)
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
                    self._step_foreach(fast, group, c, chunk_budget)
                    for p in slow:
                        self._step_one_param(p, group, c)
                else:
                    for p in params:
                        self._step_one_param(p, group, c)
            else:
                for p in params:
                    self._step_one_param(p, group, c)
            group["step"] += 1
        return loss

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state, preserving the stored dtype of ``z`` (and ``exp_avg``)."""
        load_state_dict_preserving_dtypes(self, state_dict)

    # ----------------------------------------------------------------- coefficients
    @staticmethod
    def _coeffs(group: dict[str, Any]) -> dict[str, float]:
        """All per-step scalar coefficients (shared by per-param and foreach paths).

        ``step`` is the official ``k`` (0-indexed, BEFORE this step's increment), so
        ``t = step + 1``. Mutates ``group['lr_max']`` and ``group['weight_sum']``,
        so it MUST be called exactly once per step (``step()`` calls it once per
        group and threads the result through both the foreach and per-param paths).
        """
        beta1, beta2 = group["betas"]
        k = group["step"]
        t = k + 1
        warmup = group["warmup_steps"]
        sched = (t / warmup) if (warmup > 0 and k < warmup) else 1.0
        lr_t = group["lr"] * sched
        lr_max = group["lr_max"] = max(lr_t, group["lr_max"])
        weight = (t ** group["r"]) * (lr_max ** group["weight_lr_power"])
        weight_sum = group["weight_sum"] = group["weight_sum"] + weight
        ckp1 = weight / weight_sum if weight_sum != 0 else 0.0
        inner = group["inner_momentum"]
        bc2_sq = math.sqrt(1.0 - beta2 ** t)
        bc1 = (1.0 - inner ** t) if inner != 0 else 1.0
        return {
            "beta1": beta1,
            "beta2": beta2,
            "inner": inner,
            "bc1": bc1,
            "bc2_sq": bc2_sq,
            "ckp1": ckp1,
            "lr_t": lr_t,
            # coefficient on d in the y-update: lr_t*(beta1*(1-ckp1) - 1)
            "y_d_coef": lr_t * (beta1 * (1.0 - ckp1) - 1.0),
        }

    # --------------------------------------------------------------------- normalized d
    def _normalized_d_one(
        self, state: dict[str, Any], md: str, grad: Tensor, inv_denom: Tensor, c: dict[str, float]
    ) -> Tensor:
        """Per-param ``d = grad_normalized``: ``g/denom`` (or the inner-momentum form).

        ``inv_denom`` is ``1/denom`` (already includes ``bc2`` and eps placement).
        With inner momentum, EMA ``exp_avg`` with ``inner`` and use
        ``(exp_avg/bc1) * inv_denom``.
        """
        if c["inner"] != 0:
            exp_avg = self._dequant_full(state, "exp_avg", md, grad)
            exp_avg.mul_(c["inner"]).add_(grad, alpha=1.0 - c["inner"])
            self._store_full(state, "exp_avg", md, exp_avg)
            # ``.div`` (not ``.div_``) -> fresh tensor; the float codec's dequant may
            # alias the stored buffer, which must NOT be scaled by bc1/inv_denom.
            return exp_avg.div(c["bc1"]).mul_(inv_denom)
        return grad.mul(inv_denom)

    def _normalized_d_stacked(
        self, states: list[dict[str, Any]], md: str, grad: Tensor,
        inv_denom: Tensor, shape: tuple[int, ...], c: dict[str, float],
    ) -> Tensor:
        """Stacked ``d = grad_normalized`` ``[N, *shape]`` (see :meth:`_normalized_d_one`)."""
        if c["inner"] != 0:
            n = grad.shape[0]
            exp_avg = self._dequant_full_stacked(states, "exp_avg", md, shape).reshape((n, *shape))
            exp_avg.mul_(c["inner"]).add_(grad, alpha=1.0 - c["inner"])
            self._store_full_stacked(states, "exp_avg", md, exp_avg.reshape((n, *shape)))
            # ``.div`` -> fresh tensor (the float codec's stacked dequant returns a
            # fresh stack here, but keep it consistent with the per-param path).
            return exp_avg.div(c["bc1"]).mul_(inv_denom)
        return grad.mul(inv_denom)

    # ============================================================== foreach eligibility
    @staticmethod
    def _group_foreach_eligible(group: dict[str, Any]) -> bool:
        return group["bf16_method"] != "kahan"  # kahan needs per-param shift buffers

    @staticmethod
    def _param_foreach_eligible(p: Tensor, group: dict[str, Any], cutoff: int) -> bool:
        if p.ndim == 0 or p.numel() > cutoff:
            return False
        if (
            group["bf16_method"] == "stochastic_rounding"
            and is_low_precision(p)
            and p.dtype != torch.bfloat16
        ):
            return False
        if p.ndim > 2:
            return p.data.is_contiguous() and p.grad.is_contiguous()
        return True

    # ===================================================================== foreach path
    @torch.no_grad()
    def _step_foreach(
        self, params: list[Tensor], group: dict[str, Any], c: dict[str, float], budget: int
    ) -> None:
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
        self, plist: list[Tensor], eff: tuple[int, int], matrixize: bool,
        md: str, c: dict[str, float], group: dict[str, Any],
    ) -> None:
        R, C = eff  # noqa: N806
        eps1 = group["eps"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]

        def mat(t: Tensor) -> Tensor:
            return t.view(R, C) if matrixize else t

        states = [self.state[p] for p in plist]
        rows = [s["row"] for s in states]
        cols = [s["col"] for s in states]

        grad = torch.stack([mat(p.grad).float() for p in plist])          # [N, R, C]
        row = torch.stack(rows)                                           # [N, R]
        col = torch.stack(cols)                                           # [N, C]

        # Factored second-moment EMA (HF eps1 placement).
        omb2 = 1.0 - c["beta2"]
        grad_sq = grad * grad
        if eps1 > 0:
            grad_sq = grad_sq.add_(eps1)
        row.lerp_(grad_sq.mean(dim=-1), omb2)
        col.lerp_(grad_sq.mean(dim=-2), omb2)
        torch._foreach_copy_(rows, list(row.unbind(0)))
        torch._foreach_copy_(cols, list(col.unbind(0)))

        r_factor = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)  # [N, R, 1]
        c_factor = col.rsqrt().unsqueeze(-2)                                       # [N, 1, C]
        inv_denom = (r_factor * c_factor).mul_(c["bc2_sq"])                        # 1/sqrt(v_hat)

        d = self._normalized_d_stacked(states, md, grad, inv_denom, (R, C), c)     # [N, R, C]

        if wd != 0:
            d.add_(torch.stack([mat(p.data).float() for p in plist]), alpha=wd)    # at y

        if cautious:
            d = cautious_batched_(d, grad)

        z = self._dequant_full_stacked(states, "z", md, (R, C))                    # [N, R, C]
        ys = [mat(p.data) for p in plist]

        # y <- (1-ckp1)*y + ckp1*z, then y += d * y_d_coef ; z -= lr_t*d
        self._lerp_then_add_batched(ys, z, d, c["ckp1"], c["y_d_coef"], bf16_method)
        z.sub_(d, alpha=c["lr_t"])
        self._store_full_stacked(states, "z", md, z)

    @torch.no_grad()
    def _nonfactored_bucket(
        self, plist: list[Tensor], length: int, md: str,
        c: dict[str, float], group: dict[str, Any],
    ) -> None:
        eps1 = group["eps"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]

        states = [self.state[p] for p in plist]
        vs = [s["v"] for s in states]

        grad = torch.stack([p.grad.float() for p in plist])               # [N, L]
        v = torch.stack(vs)

        v.mul_(c["beta2"]).addcmul_(grad, grad, value=1.0 - c["beta2"])
        torch._foreach_copy_(vs, list(v.unbind(0)))

        denom = v.add(1e-15).sqrt_().div_(c["bc2_sq"]).add_(eps1)          # sqrt(v_hat)+eps
        inv_denom = denom.reciprocal_()

        d = self._normalized_d_stacked(states, md, grad, inv_denom, (length,), c)  # [N, L]

        if wd != 0:
            d.add_(torch.stack([p.data.float() for p in plist]), alpha=wd)

        if cautious:
            d = cautious_batched_(d, grad)

        z = self._dequant_full_stacked(states, "z", md, (length,))
        ys = [p.data for p in plist]
        self._lerp_then_add_batched(ys, z, d, c["ckp1"], c["y_d_coef"], bf16_method)
        z.sub_(d, alpha=c["lr_t"])
        self._store_full_stacked(states, "z", md, z)

    @torch.no_grad()
    def _lerp_then_add_batched(
        self, yviews: list[Tensor], z: Tensor, d: Tensor,
        ckp1: float, y_d_coef: float, bf16_method: str,
    ) -> None:
        """In-place ``y <- (1-ckp1)*y + ckp1*z + y_d_coef*d`` over a foreach bucket.

        ``y`` is the (matrixized) param view; ``z`` and ``d`` are stacked fp32
        ``[N, *shape]``. The whole y-update is expressed as a single subtract of
        ``delta = ckp1*(y - z) - y_d_coef*d`` so it can flow through the bf16-correct
        ``subtract_batched_`` write-back (kahan is handled on the per-param path).
        """
        # y_new = (1-ckp1)*y + ckp1*z + y_d_coef*d ; y_new = y - delta
        # => delta = ckp1*(y - z) - y_d_coef*d
        ystack = torch.stack([yv.float() for yv in yviews])
        delta = ystack.sub_(z).mul_(ckp1).sub_(d, alpha=y_d_coef)
        subtract_batched_(yviews, delta, bf16_method)

    # ===================================================================== per-param path
    @torch.no_grad()
    def _step_one_param(self, p: Tensor, group: dict[str, Any], c: dict[str, float]) -> None:
        md = group["momentum_dtype"]
        eps1 = group["eps"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]

        state = self.state[p]
        if not state:
            self._init_state(p, state, group)

        grad = p.grad if p.grad.dtype == torch.float32 else p.grad.float()
        ndim = grad.ndim
        factored = ndim >= 2

        if factored:
            matrixize = ndim > 2
            gv = grad.reshape(grad.shape[0], -1) if matrixize else grad
            update_factored_state(gv, state["row"], state["col"], c["beta2"], eps1)
            r_factor, c_factor = factored_inv_sqrt_factors(state["row"], state["col"])
            inv_denom = (r_factor * c_factor).mul_(c["bc2_sq"])            # 1/sqrt(v_hat)
            d = self._normalized_d_one(state, md, gv, inv_denom, c)        # [R, C]
            if matrixize:
                d = d.reshape_as(grad)
                gv = grad
        else:
            v = state["v"]
            v.mul_(c["beta2"]).addcmul_(grad, grad, value=1.0 - c["beta2"])
            denom = v.add(1e-15).sqrt_().div_(c["bc2_sq"]).add_(eps1)
            inv_denom = denom.reciprocal_()
            d = self._normalized_d_one(state, md, grad, inv_denom, c)
            gv = grad

        if wd != 0:
            d.add_(p.data.float(), alpha=wd)                              # decoupled WD at y

        if cautious:
            d = cautious_one_(d, gv)

        z = self._dequant_z(state, md, p)                                 # fp32, param shape

        # y-update: y <- (1-ckp1)*y + ckp1*z + y_d_coef*d, via a single bf16-correct write.
        # delta = ckp1*(y - z) - y_d_coef*d  (y_new = y - delta). The exact same op
        # order as the foreach path (``(y-z).mul_(ckp1).sub_(d, y_d_coef)``) so the two
        # paths are bit-identical. ``p.detach().clone().float()`` avoids the fp32
        # ``.float()`` alias that would mutate the weight in place.
        y_fp32 = p.detach().clone().float()
        delta = y_fp32.sub_(z).mul_(c["ckp1"]).sub_(d, alpha=c["y_d_coef"])
        self._subtract_y(p, delta, state, bf16_method)

        # z step: z -= lr_t * d
        z.sub_(d, alpha=c["lr_t"])
        self._store_full(state, "z", md, z)

    @torch.no_grad()
    def _subtract_y(self, p: Tensor, delta_fp32: Tensor, state: dict[str, Any], bf16_method: str) -> None:
        """``p -= delta`` (the y write-back) with bf16-correct handling.

        Mirrors :func:`kaon._backend.subtract_one_` but uses ``shift`` for the y
        kahan buffer (``z`` has its own ``shift_z``).
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
