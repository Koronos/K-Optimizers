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
polynomial-weighted averaging, the in-place :meth:`train` / :meth:`eval`
``y <-> x`` swap, and the stochastically-rounded ``z`` write-back that a bf16-stored
``z`` needs in order to move at all (:meth:`ScheduleFree._store_z`).

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

Reproducibility (read this before comparing two runs)
-----------------------------------------------------
**With the default ``momentum_dtype="bfloat16"`` a ScheduleFree run is stochastic even
when the model is fp32.** ``z`` is stored in bf16 and its write-back therefore has to be
stochastically rounded — a round-to-nearest write would freeze the sequence outright
(:meth:`ScheduleFree._store_z`) — so this is the one optimizer in kaon that draws SR noise
without a single low-precision *weight* in the model. Every other kaon optimizer reaches
the SR write only through a bf16/fp16 parameter, which is why an fp32 run of any of them
has a noise floor of exactly zero and this one does not.

Two consequences, both of them expected behaviour rather than bugs:

* **Re-running inside one process needs kaon's own reseed.** ``torch.manual_seed(s)``
  alone is not enough, because re-seeding the global RNG to the *same* value is not
  observable through it and because an SR stream's identity is handed out in order of
  first draw (see :class:`kaon._stochastic_rounding.SRStream`): the second run claims a
  different stream and rounds ``z`` differently, so the weights part ways at the
  **second** step by about one bf16 grid step of ``z``. Call
  ``kaon.reseed_stochastic_rounding()`` after ``torch.manual_seed`` in a sweep, a test or
  any two-run comparison — then ScheduleFree is bit-identical against itself on every
  device, on both the foreach and per-parameter paths, and for every parameter / momentum
  dtype (``tests/test_schedulefree_determinism.py``).
* **A fully deterministic trajectory means a ``z`` that is not bf16.**
  ``momentum_dtype="float32"`` (the exact choice, 4 B/param for ``z``), ``"int8"`` or
  ``"4bit"`` all write ``z`` round-to-nearest, draw no noise at all, claim no stream, and
  reproduce under a bare ``torch.manual_seed``. The quantized two trade that determinism
  against ``z``'s own stall risk at small ``lr*d`` (see ``momentum_dtype`` below), so
  ``"float32"`` is the one to pick when reproducibility is the requirement.

The noise is unbiased either way, and the stream position is checkpointed, so a *resume*
is bit-exact regardless (``tests/test_sr_seed_checkpoint.py``).
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
    SRSeedState,
    _sr_write_,
    cautious_batched_,
    cautious_one_,
    centralize_grads_,
    foreach_budget,
    is_low_precision,
    subtract_batched_,
)
from kaon._factored import _MIN_NORMAL, factored_inv_sqrt_factors, update_factored_state
from kaon._foreach_plan import ForeachChunk, ForeachPlanMixin, ForeachSpec
from kaon._momentum_codec import (
    _FOURBIT_BLOCK,
    _quant_4bit,
    _quant_int8,
    fourbit_block_size,
    load_state_dict_preserving_dtypes,
)
from kaon._stochastic_rounding import SRStream
from kaon._wrappers import CodecBuffer, TrainEvalWeights

__all__ = ["ScheduleFree"]

MomentumDtype = Literal["bfloat16", "float32", "int8", "4bit"]

# One full-size z (and optional exp_avg) + factored v; mirrors AdaPNM's two-momenta
# working-set estimate closely enough for the foreach budget heuristic.
_STACK_BYTES_PER_ELEM = 48


def _zero_safe_inv_sqrt_factors(row: Tensor, col: Tensor) -> tuple[Tensor, Tensor]:
    """``factored_inv_sqrt_factors`` for ``eps1 == 0``, where a stat can be exactly zero.

    With no ``eps1`` a weight whose gradient history is all zero (step 1 of a layer
    behind a zero-init gate, LoRA ``A`` behind a zero-init ``B``) has ``row == 0``, so
    ``row / mean(row)`` is ``0/0``; an all-zero row or column gives ``rsqrt(0) = inf``,
    and the zero gradient there turns ``0 * inf`` into NaN. Nothing downstream caps the
    reconstruction, so ADOPT's lone divisor floor would only trade the NaN for an
    ``inf``. Flooring the divisor AND both rsqrt arguments at the smallest normal fp32
    bounds each factor by ``rsqrt(_MIN_NORMAL) = 2**63`` (their product stays finite),
    so a zero-stat coordinate gets a zero update. A stat is below ``_MIN_NORMAL`` only
    when it is zero or subnormal, so every normal-range value is untouched; NaNs from the
    gradient still propagate (``clamp`` keeps them). Only the ``eps1 == 0`` branch calls
    this, so any ``eps1 > 0`` run stays bit-identical to the plain reconstruction.
    """
    row_mean = row.mean(dim=-1, keepdim=True).clamp_min_(_MIN_NORMAL)
    r_factor = row.div(row_mean).clamp_min_(_MIN_NORMAL).rsqrt_().unsqueeze(-1)
    c_factor = col.clamp_min(_MIN_NORMAL).rsqrt_().unsqueeze(-2)
    return r_factor, c_factor


class ScheduleFree(TrainEvalWeights, ForeachPlanMixin, SRSeedState, Optimizer):
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
            Skipped where the fan-in is 1 (``(out, 1)``, ``(out, 1, 1, 1)``), where it
            would zero the gradient rather than centralize it — see
            ``kaon._backend.gc_applies``.
        momentum_dtype: storage dtype for the full-size ``z`` (and optional
            ``exp_avg``) buffers — ``"bfloat16"`` (default), ``"float32"``,
            ``"int8"`` or ``"4bit"``. **This is not a "small per-step error" knob.**
            ``z`` is read, stepped by ``lr*d`` and written back every step, and that
            step is routinely *smaller than one quantum of its own storage*: at
            ``|z| ~ 1`` a bf16 ULP is ~8e-3 (and an int8 absmax quantum ~8e-3 too)
            while ``lr*d`` is ~1e-3 or less. A round-to-nearest write-back then
            returns the OLD value, so ``z`` stops moving altogether and the iterate
            average keeps averaging a frozen sequence — the failure is a stalled
            ``z``, not a bounded rounding error. ``"bfloat16"`` therefore writes
            ``z`` stochastically rounded through :func:`kaon._backend._sr_write_`
            (unbiased, so sub-quantum steps survive *in expectation*; see
            :meth:`_store_z`), which makes it the memory-friendly default.
            ``"float32"`` is the exact choice (the reference test uses it).
            ``"int8"`` / ``"4bit"`` still requant ``z`` with round-to-nearest and are
            therefore still exposed to the stall at small ``lr*d``; pick them only
            when the memory saving outweighs that.
            **It is also the reproducibility knob.** Because a bf16 ``z`` rounds
            stochastically on every step, the default makes the run stochastic *even for
            an fp32 model* — the only optimizer in kaon that does — so two runs in one
            process need ``kaon.reseed_stochastic_rounding()`` between them to land on
            the same bits, and a trajectory that is deterministic under a bare
            ``torch.manual_seed`` needs a non-bf16 ``z`` (``"float32"`` for the exact
            one). See "Reproducibility" in the module docstring.
        momentum_4bit_block: block size for ``momentum_dtype="4bit"`` (default
            ``128``).
        bf16_method: low-precision **weight**-write strategy for the ``y`` write-back
            only — ``"stochastic_rounding"`` (default), ``"kahan"`` or ``"none"``.
            It does NOT govern ``z``: bf16 ``z`` always uses stochastic rounding and
            never has a Kahan/shift buffer (see :meth:`_store_z`).
        foreach: batch the step with multi-tensor ops (default ``True``). Numerically
            **equal** to the per-param path — bit-for-bit — for ``momentum_dtype``
            ``"float32"`` / ``"int8"`` / ``"4bit"``, which keep that contract unchanged.
            A bf16 ``z`` is the exception: its write-back is stochastically rounded and
            the batched path draws its noise once per stacked bucket while the per-param
            path draws once per tensor, so the two then follow the *same law* rather than
            the same bits (equivalence in expectation; the divergence per write is one
            bf16 grid step of ``z``).
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
    # train() / eval() come from TrainEvalWeights (the flag plumbing + step() guard, run under
    # no_grad); the swap math is the two hooks below. The model's buffer holds y (train) and x
    # (eval); both are a closed-form lerp toward z (x is never materialized), exact inverses.
    def _to_eval_view(self, p: Tensor, state: dict[str, Any], group: dict[str, Any]) -> None:
        """y -> x : ``p <- p + (1 - 1/beta1)*(z - p)``."""
        if "z" in state:
            beta1, _ = group["betas"]
            z = CodecBuffer.read(state, "z", group["momentum_dtype"], p)
            p.lerp_(z.to(p.dtype), weight=1.0 - 1.0 / beta1)

    def _to_train_view(self, p: Tensor, state: dict[str, Any], group: dict[str, Any]) -> None:
        """x -> y : ``p <- p + (1 - beta1)*(z - p)`` (inverse of eval)."""
        if "z" in state:
            beta1, _ = group["betas"]
            z = CodecBuffer.read(state, "z", group["momentum_dtype"], p)
            p.lerp_(z.to(p.dtype), weight=1.0 - beta1)

    # ===================================================================== z storage
    # z (and the optional inner-momentum exp_avg) are full-size; they are stored
    # through the shared first-moment codec layout so a configured int8/4bit
    # momentum_dtype keeps z compact and resumes bit-exactly. The per-param and
    # stacked read/write helpers mirror AdaPNM's (one buffer instead of two).
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
            bs = fourbit_block_size(src, group)
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

    # The read/write of the full-size buffers go through the shared CodecBuffer
    # (kaon._wrappers) — the same codec storage Lookahead's phi uses, byte-identical to the
    # hand-rolled versions these replace (CodecBuffer additionally guarantees a fresh fp32 on
    # read, which is harmless here since z is written straight back). Only `_alloc_full` above
    # stays local: it also zero-initializes the quantized exp_avg, which must NOT route through
    # a quantize-of-zeros.
    @staticmethod
    def _dequant_full(state: dict[str, Any], prefix: str, md: str, like: Tensor) -> Tensor:
        return CodecBuffer.read(state, prefix, md, like)

    def _dequant_z(self, state: dict[str, Any], md: str, like: Tensor) -> Tensor:
        return CodecBuffer.read(state, "z", md, like)

    @staticmethod
    def _store_full(state: dict[str, Any], prefix: str, md: str, m_fp32: Tensor) -> None:
        CodecBuffer.write(state, prefix, md, m_fp32)

    @staticmethod
    def _dequant_full_stacked(
        states: list[dict[str, Any]], prefix: str, md: str, shape: tuple[int, ...]
    ) -> Tensor:
        return CodecBuffer.read_stacked(states, prefix, md, shape)

    @staticmethod
    def _store_full_stacked(
        states: list[dict[str, Any]], prefix: str, md: str, m_fp32: Tensor
    ) -> None:
        CodecBuffer.write_stacked(states, prefix, md, m_fp32)

    # z's write-back is NOT the plain codec write the other full-size buffers use: at
    # ``momentum_dtype="bfloat16"`` it has to be stochastically rounded, otherwise the
    # z-sequence freezes (see :meth:`_store_z`). The gate is on the *storage dtype of z*
    # only — not on ``bf16_method`` (which governs the weight write-back) and not on the
    # weights' dtype: an fp32 model with a bf16 ``z`` has exactly the same stall.
    @staticmethod
    def _store_z(state: dict[str, Any], md: str, z_fp32: Tensor, sr: SRStream) -> None:
        """Write the updated fp32 ``z`` into its storage; bf16 storage rounds stochastically.

        A bf16 ``z`` cannot use the codec's round-to-nearest ``copy_``: the per-step
        z-step is ``lr_t*d``, which for any realistic LR sits *below* the bf16 ULP of
        ``z`` (~8e-3 at ``|z| ~ 1``), so RNE writes the OLD value back and ``z`` never
        moves — the iterate average then averages a frozen sequence.
        Stochastic rounding rounds up with probability equal to the fractional
        distance, so those sub-ULP steps survive in expectation.

        The write is phrased as the increment ``z_new - z_stored`` (``z_stored`` is a
        bf16 value, hence exact in fp32, so the increment reconstructs ``z_new``) which
        is what lets it go through the shared primitive and keeps the identity of
        ``state["z"]`` — the SR write copies into it, never replaces it.

        The SR itself goes through :func:`kaon._backend._sr_write_`, the same entry point
        every bf16 *weight* write in the library uses: one Triton launch with no temporary
        on CUDA, the torch ``add_stochastic_`` reference everywhere else. Calling
        ``add_stochastic_`` directly here cost ~7 extra kernels and two z-sized fp32/int32
        temporaries per bucket — measured 0.73x the step and +98 MiB of transient on the
        UNet/DiT bag before this went through the shared primitive.
        """
        if md != "bfloat16":
            CodecBuffer.write(state, "z", md, z_fp32)
            return
        buf = state["z"]
        # ``z_new - z_stored`` in ONE promoting kernel (fp32 - bf16 -> fp32). The old
        # ``buf.float().neg_().add_(...)`` spelling allocated the same single temporary
        # but took three passes over it; the rounded result is bit-identical.
        delta = z_fp32.reshape(buf.shape) - buf
        _sr_write_(buf, delta, 1.0, sr=sr)

    @staticmethod
    def _store_z_stacked(
        states: list[dict[str, Any]], md: str, z_fp32: Tensor, sr: SRStream
    ) -> None:
        """Batched :meth:`_store_z` over a foreach bucket's stacked ``z`` ``[N, *shape]``.

        One noise draw over the whole stack (not one per param) followed by a single
        ``_foreach_copy_`` into the per-param storages — the same shape of work
        :func:`kaon._backend.subtract_batched_` does for bf16 weights, and through the
        same :func:`kaon._backend._sr_write_` entry point (one Triton launch, no
        temporary, on CUDA). Since the draws differ from the per-param path's, a bf16
        ``z`` makes the two paths agree in expectation instead of bit-for-bit (the
        quantized and fp32 codecs stay exact).

        ``N == 1`` — every bucket of a big-unique-shape model (UNet/DiT) — skips the
        stack entirely and rounds straight into the parameter's own ``z``: a one-tensor
        ``torch.stack`` is a full copy, and with the copy-back that is two z-sized
        transfers per bucket for nothing. The arithmetic and the number of drawn
        elements are unchanged.
        """
        if md != "bfloat16":
            CodecBuffer.write_stacked(states, "z", md, z_fp32)
            return
        shape = tuple(z_fp32.shape[1:])
        # Same aliasing contract as CodecBuffer.write_stacked: z is a contiguous clone
        # (and a same-shape reshape is a view anyway), so these views reach the storage.
        bufs = [s["z"].reshape(shape) for s in states]
        single = len(bufs) == 1
        # [N, *shape], bf16 — a VIEW of the sole storage when N == 1, else a stacked copy.
        stacked = bufs[0].unsqueeze(0) if single else torch.stack(bufs)
        delta = z_fp32 - stacked                    # one promoting kernel; see _store_z
        _sr_write_(stacked, delta, 1.0, sr=sr)
        if not single:
            torch._foreach_copy_(bufs, list(stacked.unbind(0)))

    # ============================================================================ step
    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        self._require_train_mode("ScheduleFree")  # from TrainEvalWeights
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
                # A group with no gradients this iteration is a COMPLETE no-op: `step`
                # must NOT advance. `step` is the k that feeds `_coeffs` (t**r weighting,
                # the bias corrections) while `lr_max` / `weight_sum` only advance inside
                # `_coeffs`, i.e. only on steps this group actually took. Bumping `step`
                # here desynchronizes the three, so a group that starts getting grads late
                # (frozen / conditionally-active params) resumes with the wrong t.
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
                    self._drop_foreach_plan(group)
                    for p in params:
                        self._step_one_param(p, group, c)
            else:
                # Per-parameter fallback for the whole group: drop any cached plan for it,
                # so a cached plan only ever describes a group the foreach path stepped.
                self._drop_foreach_plan(group)
                for p in params:
                    self._step_one_param(p, group, c)
            group["step"] += 1
        return loss

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state, preserving the stored dtype of ``z`` (and ``exp_avg``).

        It also **replaces** each ``param_groups`` dict with the checkpoint's (only
        ``params`` is carried over), so a checkpoint written by an older kaon has no
        entry for a hyperparameter added since — reading it would raise ``KeyError``
        on the first step. Backfill any key the checkpoint predates from
        ``self.defaults``; keys the checkpoint *does* carry win, so a resumed run
        keeps its own tuning.
        """
        load_state_dict_preserving_dtypes(self, state_dict)
        for group in self.param_groups:
            for key, value in self.defaults.items():
                group.setdefault(key, value)
        # The loader REPLACES state["v"] / state["z"] (and every param_groups dict), so a
        # cached foreach plan would keep stepping detached buffers under a dead group id.
        self._clear_foreach_plans()

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
    # Bucketing, chunking and the cached view plan live in kaon._foreach_plan. ``row`` /
    # ``col`` (factored) and ``v`` (non-factored) are the state buffers the bucket bodies
    # stack and write back through. No extra bucket key: ``_coeffs`` is computed ONCE per
    # group per step in ``_step_impl`` (it advances ``lr_max`` / ``weight_sum``) and
    # threaded through every bucket, so nothing here depends on a per-parameter clock. ``z``
    # and ``exp_avg`` go through the codec-buffer helpers, which take ``states`` rather than
    # a ``mat`` callback, so there is no codec view cache to prebuild.
    # ``single_alias``: a one-parameter bucket — every bucket of a big-unique-shape model
    # (UNet/DiT), where each weight owns its shape — ``unsqueeze``es instead of stacking, so
    # ``grad_stack`` / ``param_stack`` widen straight from the parameter's storage instead of
    # copying it to a same-dtype stack first. Both stacks are read-only in
    # ``_factored_bucket`` / ``_nonfactored_bucket`` (the y-update makes its own mutable
    # stack in ``_lerp_then_add_batched``), which is the precondition for aliasing them, and
    # ``_param_foreach_eligible`` already rejects an ``ndim > 2`` param whose data or grad is
    # non-contiguous — the case where the matrixizing ``view`` on the unsqueezed tensor would
    # not be expressible. Bit-identical: stacking one tensor is a copy, and the widening cast
    # that follows it is elementwise. Same contract AdaMuon has had since 0.7.11.
    _FOREACH_SPEC = ForeachSpec(
        factored_state=("row", "col"), flat_state=("v",), single_alias=True,
    )

    @staticmethod
    def _group_foreach_eligible(group: dict[str, Any]) -> bool:
        return group["bf16_method"] != "kahan"  # kahan needs per-param shift buffers

    @staticmethod
    def _param_foreach_eligible(p: Tensor, group: dict[str, Any], cutoff: int) -> bool:
        # 0-D scalars are NOT excluded: they ride the non-factored bucket as
        # length-1 rows (see kaon._backend.flat_view). Only the per-tensor size cap
        # and the awkward dtype/contiguity cases fall back to the per-param loop.
        if p.numel() > cutoff:
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
        for chunk in self._foreach_chunks(params, group, budget):
            bucket = self._factored_bucket if chunk.eff is not None else self._nonfactored_bucket
            bucket(chunk, md, c, group)

    @torch.no_grad()
    def _factored_bucket(
        self, chunk: ForeachChunk, md: str, c: dict[str, float], group: dict[str, Any],
    ) -> None:
        R, C = chunk.eff  # noqa: N806
        eps1 = group["eps"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]

        states = chunk.states
        rows, cols = chunk.state_views

        grad = chunk.grad_stack()                                         # [N, R, C]
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

        if eps1 > 0:
            r_factor = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)  # [N, R, 1]
            c_factor = col.rsqrt().unsqueeze(-2)                                       # [N, 1, C]
        else:
            r_factor, c_factor = _zero_safe_inv_sqrt_factors(row, col)
        inv_denom = (r_factor * c_factor).mul_(c["bc2_sq"])                        # 1/sqrt(v_hat)

        d = self._normalized_d_stacked(states, md, grad, inv_denom, (R, C), c)     # [N, R, C]

        if wd != 0:
            d.add_(chunk.param_stack(), alpha=wd)                                  # at y

        if cautious:
            d = cautious_batched_(d, grad)

        z = self._dequant_full_stacked(states, "z", md, (R, C))                    # [N, R, C]
        ys = chunk.pviews

        # y <- (1-ckp1)*y + ckp1*z, then y += d * y_d_coef ; z -= lr_t*d
        self._lerp_then_add_batched(ys, z, d, c["ckp1"], c["y_d_coef"], bf16_method)
        z.sub_(d, alpha=c["lr_t"])
        self._store_z_stacked(states, md, z, self.sr_stream)

    @torch.no_grad()
    def _nonfactored_bucket(
        self, chunk: ForeachChunk, md: str, c: dict[str, float], group: dict[str, Any],
    ) -> None:
        """Non-factored (full per-coordinate ``v``) update for ``ndim <= 1`` params.

        0-D scalars share the ``L == 1`` bucket with shape-``(1,)`` params as length-1
        **views** (:func:`~kaon._backend.flat_view`) of the same storage, so ``v``, the
        ``z`` / ``exp_avg`` codec write-backs and the y write all reach the original
        0-D tensors. Those views are the cached plan's (``chunk.state_views`` /
        ``chunk.pviews``), not rebuilt per param per step.
        """
        eps1 = group["eps"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]

        length = chunk.length
        states = chunk.states
        (vs,) = chunk.state_views

        grad = chunk.grad_stack()                                         # [N, L]
        v = torch.stack(vs)

        v.mul_(c["beta2"]).addcmul_(grad, grad, value=1.0 - c["beta2"])
        torch._foreach_copy_(vs, list(v.unbind(0)))

        denom = v.add(1e-15).sqrt_().div_(c["bc2_sq"]).add_(eps1)          # sqrt(v_hat)+eps
        inv_denom = denom.reciprocal_()

        d = self._normalized_d_stacked(states, md, grad, inv_denom, (length,), c)  # [N, L]

        if wd != 0:
            d.add_(chunk.param_stack(), alpha=wd)

        if cautious:
            d = cautious_batched_(d, grad)

        z = self._dequant_full_stacked(states, "z", md, (length,))
        ys = chunk.pviews
        self._lerp_then_add_batched(ys, z, d, c["ckp1"], c["y_d_coef"], bf16_method)
        z.sub_(d, alpha=c["lr_t"])
        self._store_z_stacked(states, md, z, self.sr_stream)

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
        ystack = torch.stack(yviews).float()
        delta = ystack.sub_(z).mul_(ckp1).sub_(d, alpha=y_d_coef)
        subtract_batched_(yviews, delta, bf16_method, sr=self.sr_stream)

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
            if eps1 > 0:
                r_factor, c_factor = factored_inv_sqrt_factors(state["row"], state["col"])
            else:
                r_factor, c_factor = _zero_safe_inv_sqrt_factors(state["row"], state["col"])
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
        self._store_z(state, md, z, self.sr_stream)

    @torch.no_grad()
    def _subtract_y(
        self,
        p: Tensor,
        delta_fp32: Tensor,
        state: dict[str, Any],
        bf16_method: str,
    ) -> None:
        """``p -= delta`` (the y write-back) with bf16-correct handling.

        Mirrors :func:`kaon._backend.subtract_one_` — including its SR entry point
        :func:`kaon._backend._sr_write_` (Triton on CUDA, the torch reference
        elsewhere); ``shift`` belongs only to y. A bf16 z always uses stochastic
        rounding and has no Kahan/shift buffer.

        The SR branch is reached by every param above ``foreach_batch_cutoff`` even in a
        ``foreach=True`` run, so calling ``add_stochastic_`` here left the big weights of
        a UNet/DiT bag on the torch path while the batched buckets already used Triton —
        measured +42 MiB of transient per step for the two weight-sized temporaries it
        draws (an fp32 upcast and an int32 noise tensor).
        """
        low = is_low_precision(p)
        if low and bf16_method == "kahan":
            shift = state["shift"]
            shift.sub_(delta_fp32.to(p.dtype))
            p_before = p.detach().clone()
            p.add_(shift)
            shift.add_(p_before.sub_(p))
        elif low and bf16_method == "stochastic_rounding" and p.dtype == torch.bfloat16:
            _sr_write_(p.data, delta_fp32, -1.0, sr=self.sr_stream)
        else:
            p.data.sub_(delta_fp32.to(p.dtype))
