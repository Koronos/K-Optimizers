"""MSAM — Momentum-SAM (Becker et al. 2024, arXiv:2401.12033) on the kaon backend.

Sharpness-Aware Minimization **without the second forward/backward**. Standard
:class:`~kaon.sam.SAM` buys its flat-minima bias with ~2x compute: every step needs an
extra full forward/backward at the perturbed point ``w + rho * g/||g||`` — on a diffusion
DiT that doubles the GEMM phase, the most expensive part of the step. MSAM's observation:
the **momentum buffer is already an estimate of the expected gradient**, so perturbing
along the *momentum* direction instead of the instantaneous gradient needs **no extra
pass at all** — the perturbation is applied to the live weights at the *end* of step
``t``, the training loop's normal forward/backward then computes the gradient *at the
perturbed point*, and step ``t+1`` removes the perturbation before the base optimizer
updates the unperturbed weights:

.. code-block:: text

    # end of step t   (inside opt.step(), after the base update):
    w <- w + rho * m_t / ||m_t||        # climb along momentum (uphill estimate)
    # training loop: forward/backward   -> grad is evaluated AT the climbed point
    # start of step t+1 (inside opt.step()):
    w <- w - rho * m_t / ||m_t||        # exact same e (m unchanged in between)
    base.step()                          # base optimizer updates the TRUE weights

``||m||`` is the global L2 norm over all momenta (the SAM convention for ``||g||``).
The perturbation is **recomputed from the stored momentum** on removal, so MSAM keeps
**zero extra persistent state** — the whole flat-minima mechanism is memory-free, which
is the point of putting it on the kaon backend.

Sign of ``rho``. The base optimizer's momentum is an EMA of the (v-normalized)
*update*, which points along the gradient — i.e. uphill. ``rho > 0``
climbs uphill (the SAM-like direction); ``rho < 0`` probes the Nesterov-like downhill
lookahead instead. Both are exposed because the right sign is an empirical question
(measured on the control battery, not assumed).

train() / eval(). Between steps the live weights deliberately carry the perturbation —
that is the mechanism — so **sampling / validation / checkpointing must use**
:meth:`eval` (removes the perturbation) **and** :meth:`train` (restores it) around the
measurement, exactly like Lookahead / Schedule-Free. Always checkpoint in eval mode: a
checkpoint saved in train mode stores perturbed weights, and a fresh MSAM cannot know to
remove that perturbation on resume.

bf16 note. The climb/restore round-trip uses **round-to-nearest, not stochastic
rounding** — deliberately. SR exists so an *accumulating* update smaller than the
weight's ulp is not lost; the climb accumulates nothing (it is applied and removed one
step later), so its rounding error has no signal to preserve, only a random walk to
contribute. Two independent SR draws do not cancel: measured 19% relative L2 drift after
4000 climb/removal cycles on bf16 weights, growing as sqrt(N) (fp32 weights: exactly
zero, the round trip is exact to floating-point addition). Round-to-nearest makes the
+e/-e pair land back on the same bf16 value whenever the residual is under half an ulp —
measured drift exactly zero. The cost is that a climb below half an ulp is not applied at
all; see the inert-lookahead warning in :meth:`_warn_if_inert`, which fires in exactly
that regime. SAM and Lookahead sidestep all of this by keeping an exact weight snapshot;
MSAM deliberately trades that snapshot away for zero memory, which is what makes the
removal a *recompute* rather than a *copy*.
"""

from __future__ import annotations

import operator
import warnings
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from kaon._backend import _ck_write_, ensure_residuals
from kaon._compact_kahan import RESIDUAL_KEY, is_compact_kahan, residual_bits, residual_bits_of
from kaon._foreach_plan import state_generation
from kaon._wrappers import CodecBuffer, WrapsInnerOptimizer

__all__ = ["MSAM"]

# Unbound accessors for the fused plan's pointer witness (:meth:`MSAM._plan_addrs_valid`).
# The witness re-reads every cached address twice per step, so its whole cost is Python
# call overhead: hoisting these lets the scan be a ``tuple(map(...))`` — the loop runs
# inside ``map``/``tuple`` in C — instead of a generator expression, which pays a frame
# resume plus a ``LOAD_METHOD``/dict subscript per parameter. Measured on a 428-parameter
# bag (300 conv + 128 scalars): 114 -> 95 us per validation at ``momentum_dtype="4bit"``
# (three address tables) and 89 -> 62 us at ``"bfloat16"`` (two).
_data_ptr = Tensor.data_ptr
_get_m = operator.itemgetter("m")
_get_m_scale = operator.itemgetter("m_scale")
_get_lo = operator.itemgetter(RESIDUAL_KEY)

# The fused climb's Philox seed. The kernels draw ``tl.rand(seed + t, offs)`` (``t`` = the
# parameter's slot in the launch), and the inner fused step seeds with its step counter
# (``self._t + t``): a bare launch counter here would walk the SAME small integers and reuse
# the inner write's uniform on the same element (correlated residual rounding between the
# climb and the step). The counter is therefore hashed away from the small integers: a salt
# plus a golden-ratio stride, masked to the kernels' non-negative int32 seed range.
_CLIMB_SEED_SALT = 0x2545F491
_CLIMB_SEED_STRIDE = 0x9E3779B1
_CLIMB_SEED_MASK = 0x7FFFFFFF


def _climb_seed(counter: int) -> int:
    """Philox seed of the fused climb launch number ``counter`` (see the constants above)."""
    return (_CLIMB_SEED_SALT + counter * _CLIMB_SEED_STRIDE) & _CLIMB_SEED_MASK


def _ck_climb(group: dict[str, Any], plist: list[Tensor], states: list[dict[str, Any]]) -> int:
    """The residual width a bucket's climb decodes with (8 / 16), or 0 for a plain climb.

    Keyed on the GROUP's ``bf16_method`` (and bf16 weights with every residual present), not
    on the mere presence of ``kahan_lo``: a group switched from ``kahan8`` back to SR keeps a
    stale residual in its state that the writer no longer maintains, and a climb that decoded
    it would inject that stale residual into the weights.

    The WIDTH is the stored residual's (its dtype), not the method's: right after a switch
    between ``kahan8`` and ``kahan16`` the removal at the top of the step must decode the
    residual the climb encoded; the inner step converts it afterwards and the next climb
    uses the new width. A bucket whose residuals are of mixed widths (a conversion caught
    half-way) reads as 0 here rather than decoding some of them wrongly; the CLIMB then
    normalizes it (:func:`_ck_ready`), so it is never left on the plain climb.
    """
    if plist[0].dtype != torch.bfloat16 or not is_compact_kahan(group.get("bf16_method", "")):
        return 0
    lo0 = states[0].get(RESIDUAL_KEY)
    if lo0 is None:
        return 0
    dt = lo0.dtype
    for st in states:
        lo = st.get(RESIDUAL_KEY)
        if lo is None or lo.dtype != dt:
            return 0
    return residual_bits_of(lo0)


def _ck_ready(group: dict[str, Any], plist: list[Tensor], states: list[dict[str, Any]],
              climb: bool) -> int:
    """:func:`_ck_climb`, and on the CLIMB (``climb=True``) first bring a compact-Kahan bf16
    bucket to the group's width: residuals missing (a param that had state before a switch
    to kahan8/kahan16) are allocated and residuals of the other width converted
    (:func:`kaon._backend.ensure_residuals`, one warning). Without it a bucket holding a param
    the inner never steps (no grad: it keeps the uint8 residual it had when the group was
    switched to kahan16) stays mixed forever and climbs on the bare bf16 every step — the
    coherent sub-ulp loss §4b of ``docs/research/compact-kahan.md`` measures.

    Only the climb converts: the REMOVAL at the top of a step must decode the residual the
    previous climb encoded, and the climb after a normalization always leaves the bucket
    uniform, so the next removal reads one width. O(1) extra per bucket in steady state (the
    width check is :func:`_ck_climb`'s own scan).
    """
    ck = _ck_climb(group, plist, states)
    if climb and plist[0].dtype == torch.bfloat16:
        method = group.get("bf16_method", "")
        if is_compact_kahan(method):
            want = residual_bits(method)
            if ck != want:
                ensure_residuals(plist, states, want)
                ck = want
    return ck

# Env-gated divergence probe (zero overhead when unset; same env var as AdaPNM's probe).
# Set KAON_PROBE_LOG=/path/to/log to record, per step, the FIRST non-finite tensor and the
# PHASE it appeared in — the discriminating fact for a real-training NaN:
#   [GRAD]  non-finite gradient on entry      -> NaN entered via the forward/backward at the
#           perturbed point (suspect the batch/bucket/precision, not the optimizer state)
#   [STATE] weights/momentum non-finite after the inner step -> inner optimizer channel
#   [CLIMB] weights non-finite after the climb -> the perturbation itself wrote it
import os  # noqa: E402

_PROBE_LOG = os.environ.get("KAON_PROBE_LOG")


class MSAM(WrapsInnerOptimizer, Optimizer):
    """Momentum-SAM wrapping a base kaon optimizer (default :class:`~kaon.adakaon.Adakaon`).

    Args:
        params: parameters or param-group dicts (shared with the base optimizer).
        base_optimizer: the optimizer **class** to wrap (default Adakaon). Instantiated
            internally over the same param groups; ``**kwargs`` are forwarded. The base
            must keep a kaon-codec momentum buffer (``betas[0] > 0``) — the perturbation
            direction is read from it.
        rho: perturbation radius (L2 size of the climb). ``rho > 0`` climbs along
            the momentum (uphill, SAM-like); ``rho < 0`` probes downhill (Nesterov-like).
            ``0`` disables MSAM (the wrapper becomes a transparent passthrough).
        norm: ``"global"`` (default) normalizes the climb by the global momentum norm
            (one radius for the whole net, the SAM/MSAM convention — needs a cross-param
            reduction); ``"tensor"`` gives every param its own radius ``rho`` normalized
            by its own momentum norm (layerwise — no global sync, so the perturbation can
            fuse into a single batched pass); ``"none"`` applies the **raw** momentum
            converted to step units, ``e = rho * lr * m`` — since a kaon momentum is the
            EMA of the *preconditioned, RMS-clipped update direction*, this makes ``rho``
            a **lookahead measured in optimizer steps** at the current lr
            ("perturb to where ~rho more steps would land"). Unlike a
            fixed weight-space radius, that is dimensionless and self-scaling: it tracks
            the LR (and any schedule), the per-coordinate ``1/sqrt(v)`` metric, and the
            model's weight scale by construction — the transfer-robust formulation.
        eps: numerical floor on the momentum norm before dividing.
        inert_check_interval: sample the inactivity warning every this many steps
            (default 10, formerly every step). The heuristic observes at most the
            first 200 climbs. Set 1 for the old diagnostic cadence; updates and
            perturbations are unaffected. Transient changes between samples may
            not be observed.
        **kwargs: forwarded verbatim to ``base_optimizer`` (e.g. ``lr``, ``betas``,
            ``cautious``, ``momentum_dtype``, ``gradient_centralization``, ``foreach``).
    """

    def __init__(
        self,
        params: Iterable[Any],
        base_optimizer: type[Optimizer] | None = None,
        rho: float = 0.3,
        norm: str = "global",
        eps: float = 1e-12,
        inert_check_interval: int = 10,
        **kwargs: Any,
    ) -> None:
        if eps < 0.0:
            raise ValueError(f"eps must be >= 0, got {eps}")
        if isinstance(inert_check_interval, bool) or not isinstance(inert_check_interval, int) or inert_check_interval < 1:
            raise ValueError("inert_check_interval must be a positive integer")
        self.inert_check_interval = inert_check_interval
        if norm not in ("global", "tensor", "none"):
            raise ValueError(f"norm must be 'global', 'tensor' or 'none', got {norm!r}")
        self.norm = norm
        if base_optimizer is None:
            from kaon.adakaon import Adakaon

            base_optimizer = Adakaon
        self.rho = float(rho)
        self.eps = float(eps)
        self._bind_inner(base_optimizer(params, **kwargs), state_key="msam")
        self.base_optimizer = self.inner
        # Dual-momentum bases (AdaPNM: m_pos/m_neg) never allocate a single `m`
        # buffer; without this guard MSAM's climb would silently find zero targets
        # and become a no-op wrapper. rho=0 is a documented transparent passthrough.
        if self.rho != 0.0:
            from kaon.adapnm import AdaPNM

            owner = self._momentum_owner()
            if isinstance(owner, AdaPNM):
                raise TypeError(
                    f"MSAM cannot wrap {type(owner).__name__}: it keeps dual momentum "
                    f"(m_pos/m_neg), not a single `m` buffer that the climb reads. "
                    f"Use a base with a kaon-codec first moment (Adakaon, Lion, …), "
                    f"or set rho=0 for a passthrough."
                )
        # Live weights carry the perturbation only while (training mode AND a momentum
        # exists). eval()/train() toggle the mode; _has_e tracks whether a perturbation
        # is currently defined (false until the first inner step populates momentum).
        self._train_mode = True
        self._has_e = False
        self._mnorm = 0.0  # global ||m|| cached at climb time (m is unchanged until the next inner step)
        self._axpy_cache: dict[str, Any] | None = None  # Triton 4bit fast-path pointer arrays
        self._axpy_seed = 0                             # SR seed counter for the fused axpy
        self._eclamp: dict[int, float] = {}             # per-group climb bound, frozen per cycle
        self._estep_scale: dict[int, float] = {}        # direction -> step units, frozen per cycle
        self._e_scale = 1.0                             # exact scale frozen with the live climb
        # Param groups and state dictionaries are stable between checkpoint loads.
        # Cache the momentum census/buckets so Nekaon's two perturbation passes do
        # not rescan and regroup hundreds of adapters every step.  A late-gradient
        # parameter changes owner-state size and invalidates the cache lazily.
        self._momentum_cache_key: tuple[Any, ...] | None = None
        self._momentum_cache: list[tuple[Tensor, dict[str, Any], str, dict[str, Any]]] = []
        self._bucket_cache: list[
            tuple[list[Tensor], list[dict[str, Any]], str, tuple[int, ...], dict[str, Any]]
        ] | None = None
        self._inert_streak = 0                          # consecutive climbs too small to do anything
        self._inert_checks = 0                          # bounded: the check reads weights
        self._inert_warned = False
        self._no_m_warned = False                       # beta1=0 / no-`m` base warned once

    # ------------------------------------------------------------- perturbation
    def _momentum_owner(self) -> Any:
        """The innermost optimizer that owns the ``m`` codec buffers. Lets MSAM wrap
        another wrapper (e.g. ``MSAM(base_optimizer=Lookahead, ...)``) — wrapper state
        dicts hold their own buffers (phi/z), not the momentum, so unwrap to the base."""
        owner = self.inner
        while hasattr(owner, "inner"):
            owner = owner.inner
        return owner

    def _momentum_params(self) -> list[tuple[Tensor, dict[str, Any], str, dict[str, Any]]]:
        """Every (param, inner_state, momentum_dtype, group) that has a momentum buffer."""
        owner = self._momentum_owner()
        owner_state = owner.state
        # ``state_generation`` is the base's state-IDENTITY counter (a constant for a base
        # that does not watch its state, so this costs one ``getattr``). The other three
        # fields see a state dict being added or removed; none of them sees a per-param
        # state being emptied or a single buffer being rebound. ``opt.state[p].clear()``
        # left this cache holding the emptied dicts and ``_apply``'s stacked read raised a
        # bare ``KeyError: 'm'`` from inside the wrapper; a rebound ``st["m"]`` was caught
        # only on the FUSED plan (``_plan_addrs_valid`` re-reads the dicts) and only after
        # the bucket regrouping had already been skipped.
        key = (
            id(owner_state),
            len(owner_state),
            state_generation(owner_state),
            tuple(len(group["params"]) for group in self.param_groups),
        )
        if key == self._momentum_cache_key:
            return self._momentum_cache
        out = []
        for group in self.param_groups:
            md = group["momentum_dtype"]
            for p in group["params"]:
                st = owner_state.get(p)
                if st and "m" in st:
                    out.append((p, st, md, group))
        self._momentum_cache_key = key
        self._momentum_cache = out
        self._bucket_cache = None
        self._axpy_cache = None
        return self._momentum_cache

    def _buckets(self) -> list[tuple[list[Tensor], list[dict[str, Any]], str, tuple[int, ...], dict[str, Any]]]:
        """Group the momentum-carrying params by (shape, dtype, momentum_dtype, group) so
        the perturbation reads/applies as a few stacked ops instead of a per-param loop
        (the per-param dequant x2 per step made the 512-tiny-tensor LoRA regime ~10x
        slower). Buckets never mix param groups: the per-element climb bound is a
        per-group quantity (it reads the group's lr / clip_threshold)."""
        momentum = self._momentum_params()
        if self._bucket_cache is not None:
            return self._bucket_cache
        by_key: dict[tuple[Any, ...], tuple[list[Tensor], list[dict[str, Any]], dict[str, Any]]] = {}
        for p, st, md, group in momentum:
            key = (tuple(p.shape), p.dtype, md, id(group))
            plist, states, _g = by_key.setdefault(key, ([], [], group))
            plist.append(p)
            states.append(st)
        self._bucket_cache = [
            (plist, states, key[2], key[0], g)
            for key, (plist, states, g) in by_key.items()
        ]
        return self._bucket_cache

    def _climb_bound(self, group: dict[str, Any], sign: float) -> float:
        """Per-element cap on the climb: ``|e_i| <= |rho| * clip_threshold * lr``.

        The stability guard for ``norm="none"`` (the same failure channel AdaPNM's
        ``clip_threshold`` closed): Adakaon's RMS clip bounds the update's *RMS*, not its
        per-element max, so a near-zero factored col-EMA concentrates ~sqrt(n)*lr spikes
        on a few coordinates; the momentum accumulates them and the lookahead would hold
        the weights displaced k-fold along the spike between steps (and the 4-bit codec
        smears a spike over its 128-block neighbours) — measured NaN on a real Cosmos
        LoKr run at step ~406. The cap says: no coordinate may be displaced further than
        ``k`` maximum-allowed update steps. Inactive in the normal regime (typical
        ``|m_i| ~ lr``), it bites exactly on the runaway channel.

        The bound is FROZEN at climb time (keyed per group) and reused for the removal /
        eval / train swaps — an LR-scheduler change between steps must not change the
        ``e`` being removed. ``step()`` clears the cache after each removal."""
        gid = id(group)
        if sign > 0 and gid not in self._eclamp:
            self._eclamp[gid] = abs(self.rho) * group.get("clip_threshold", 1.0) * group["lr"]
        return self._eclamp[gid]

    def _climb_step_scale(self, group: dict[str, Any], sign: float) -> float:
        """Convert the stored momentum to OPTIMIZER-STEP units for a ``norm="none"`` climb.

        Inner optimizers that store an LR-independent direction (Adakaon since
        0.7.11 — marked ``_momentum_is_unscaled``) need a ``* lr`` so ``rho`` keeps
        meaning "lookahead in optimizer steps"; owners with lr-scaled momentum keep
        the historical 1.0. FROZEN at climb time per group (like ``_climb_bound``):
        the removal / eval / train swaps must undo the SAME e even if a scheduler
        moved lr between them. ``step()`` clears the cache after each removal."""
        gid = id(group)
        if sign > 0 and gid not in self._estep_scale:
            owner = self._momentum_owner()
            self._estep_scale[gid] = (
                float(group["lr"])
                if getattr(owner, "_momentum_is_unscaled", False)
                else 1.0
            )
        return self._estep_scale[gid]

    # Below this relative displacement the perturbed gradient is measurably the same as
    # the true one: on a real MLP with fp32 weights (so representation is not the limit),
    # |dw|/|w| = 2.3e-5 moved the gradient by 1.8e-4 (0.018%), while |dw|/|w| = 3.7e-3
    # moved it by 2.1%. A climb under ~1e-4 relative samples nothing and is pure cost.
    _INERT_REL = 1e-4
    # Warn after sampled conditions span this many climbs; this is a heuristic,
    # not a claim that every intervening step was inspected.
    _INERT_PATIENCE = 50
    # ...and stop looking after this many climbs either way: the check reads weights, so
    # leaving it armed for the whole run would cost bandwidth every step forever.
    _INERT_MAX_CHECKS = 4 * _INERT_PATIENCE
    # Weight magnitude is read from at most this many params per group — a mean over a
    # handful of tensors is a fine scale estimate and keeps the check off the hot path.
    _INERT_SAMPLE = 8

    @torch.no_grad()
    def _warn_if_inert(self) -> None:
        """Warn once when sampled weight scales suggest an ineffective climb.

        This periodically sampled heuristic compares a displacement bound with a
        representative ulp or relative weight scale. It does not inspect every
        coordinate or measure whether the perturbed gradient actually changes.
        """
        if self._inert_warned or self.rho == 0.0 or self._inert_checks >= self._INERT_MAX_CHECKS:
            return
        self._inert_checks += 1
        # This heuristic converts device reductions to Python scalars. Sample
        # periodically rather than synchronizing every optimizer step. Interval
        # 1 retains the old cadence; it never changes the actual perturbation.
        if self._inert_checks % self.inert_check_interval:
            return
        msg = None
        for group in self.param_groups:
            params = [p for p in group["params"] if p.numel()][: self._INERT_SAMPLE]
            if not params:
                continue
            # Displacement scale depends on the norm mode (sample ≤ _INERT_SAMPLE params):
            #   "none"   — per-coordinate |e| ≲ |rho| * lr * clip; compare to mean |w|
            #   "global" — L2 radius |rho| over the net; relative = |rho| / ||w||_2 (sample)
            #   "tensor" — per-tensor L2 radius |rho|; relative = mean_t |rho| / ||p_t||_2
            dtype = params[0].dtype
            if self.norm == "none":
                e = abs(self.rho) * group["lr"] * group.get("clip_threshold", 1.0)
                w = float(torch.stack([p.detach().abs().mean().float() for p in params]).mean())
                if w == 0.0:
                    continue
                half_ulp = 0.5 * torch.finfo(dtype).eps * w
                rel = e / w
                tip = "Raise lr or rho/k, or set rho/k=0 to drop the cost."
            elif self.norm == "global":
                # ||w||_2 over the sampled params (documented sample, not the full net).
                w_sq = sum(float(p.detach().float().pow(2).sum()) for p in params)
                w_norm = w_sq ** 0.5
                if w_norm == 0.0:
                    continue
                e = abs(self.rho)
                half_ulp = 0.5 * torch.finfo(dtype).eps * w_norm
                rel = e / w_norm
                tip = "Raise rho/k, or set rho/k=0 to drop the cost."
            else:  # "tensor"
                rels = []
                for p in params:
                    pn = float(p.detach().float().pow(2).sum()) ** 0.5
                    if pn > 0.0:
                        rels.append(abs(self.rho) / pn)
                if not rels:
                    continue
                rel = sum(rels) / len(rels)
                e = abs(self.rho)
                w_mean = float(torch.stack([p.detach().abs().mean().float() for p in params]).mean())
                half_ulp = 0.5 * torch.finfo(dtype).eps * max(w_mean, 1e-12)
                tip = "Raise rho/k, or set rho/k=0 to drop the cost."
            # A compact-Kahan climb goes through the DECODED value (_ck_ready): the sub-ulp
            # part of e is kept in the residual, so the resolution is ulp / 2**bits (kahan8:
            # ulp/256; kahan16: the fp32 master's), not the bare bf16 ulp.
            method = group.get("bf16_method", "")
            res_bits = (residual_bits(method)
                        if dtype == torch.bfloat16 and is_compact_kahan(method) else 0)
            half_ulp /= 1 << res_bits
            if dtype != torch.float32 and e < half_ulp:
                what = (f"half the {method} residual grid (ulp/{1 << res_bits})" if res_bits
                        else f"half a {dtype} ulp")
                msg = (
                    f"{type(self).__name__}: the lookahead displacement (<= {e:.2e}) is below "
                    f"{what} at the sampled mean weight scale ({half_ulp:.2e}); "
                    f"some perturbations may round to zero. "
                    f"Use fp32 weights for these parameters, or set rho/k=0 to drop the cost."
                )
            elif rel < self._INERT_REL:
                msg = (
                    f"{type(self).__name__}: the lookahead displaces the weights by only "
                    f"{rel:.1e} relative (norm={self.norm!r}, lr={group['lr']:.2e}); below "
                    f"~{self._INERT_REL:.0e} this heuristic flags potentially inert lookahead. "
                    f"It does not measure the gradient difference. {tip}"
                )
            if msg is not None:
                break
        if msg is None:
            self._inert_streak = 0
            return
        self._inert_streak += self.inert_check_interval
        if self._inert_streak >= self._INERT_PATIENCE:
            self._inert_warned = True
            warnings.warn(msg, stacklevel=3)

    @torch.no_grad()
    def _global_mnorm(self) -> float:
        """Global L2 norm over all momenta, via ``(m*m).sum()`` (``torch.dot`` is avoided
        deliberately — it SIGFPEs on some GPUs; see SAM._grad_norm)."""
        sq = None
        for _plist, states, md, shape, _group in self._buckets():
            m = CodecBuffer.read_stacked(states, "m", md, shape)
            s = (m * m).sum()
            sq = s if sq is None else sq + s
        return 0.0 if sq is None else float(sq.sqrt())

    @torch.no_grad()
    def _apply(self, sign: float, scale: float | None = None) -> None:
        """Add ``sign * rho * m / ||m||`` to every weight (bf16-correct write), bucketed.

        ``norm="global"`` uses the cached cross-param norm (one radius for the net);
        ``norm="tensor"`` rescales each slice by its own momentum norm (recomputed — m is
        unchanged between the climb and its removal, so the round trip is exact).
        ``norm="none"`` + 4-bit momentum on GPU takes the Triton fast path: the torch
        dequant (unpack -> scale -> stack -> axpy, several kernels + an fp32 temp) is
        the dominant perturbation cost at 4 bits; ``_axpy_4bit_batched`` does dequant +
        bf16-SR axpy in ONE launch per bucket (measured ~4.4 ms/step -> sub-ms on the
        C=128 proxy). Ineligible params (non-contiguous, exotic dtype) fall back here."""
        if sign > 0.0 and scale is not None:
            self._e_scale = float(scale)
        applied_scale = self._e_scale if scale is None else float(scale)
        leftover = (
            self._apply_fused(sign, applied_scale)
            if self.norm == "none"
            else None
        )
        for plist, states, md, shape, group in (self._buckets() if leftover is None else leftover):
            m = CodecBuffer.read_stacked(states, "m", md, shape)  # [N, *shape] fp32
            n = m.shape[0]
            if self.norm == "global":
                m.mul_(sign * self.rho * applied_scale / (self._mnorm + self.eps))
            elif self.norm == "tensor":  # per-tensor radius: rho * m_i / ||m_i|| per slice
                norms = (m * m).reshape(n, -1).sum(dim=1).sqrt_()  # no .norm(): dot SIGFPEs here
                scales = (sign * self.rho * applied_scale) / (norms + self.eps)
                m.mul_(scales.view(n, *([1] * (m.ndim - 1))))
            else:  # "none": raw momentum — rho is a lookahead in OPTIMIZER-STEP units
                m.mul_(sign * self.rho * applied_scale * self._climb_step_scale(group, sign))
                bound = self._climb_bound(group, sign)
                # NaN passes through clamp(): a non-finite momentum coordinate (e.g. a
                # 0*inf from a blown 4-bit block scale) must contribute ZERO climb, never
                # poison the weights. inf is mapped to +-bound by the clamp either way.
                torch.nan_to_num_(m, nan=0.0, posinf=bound, neginf=-bound)
                m.clamp_(-bound, bound)  # per-element stability cap (see _climb_bound)
            if plist[0].dtype == torch.float32:
                torch._foreach_add_([p.data for p in plist], list(m.unbind(0)))
            elif ck_bits := _ck_ready(group, plist, states, sign > 0.0):
                # Compact Kahan (kahan8/kahan16): perturb the DECODED value and re-encode, so
                # the climb/removal pair leaves the clean value intact to ~1/256 ulp
                # (kahan8, stochastic rounding of the residual: unbiased) or to fp32's own
                # rounding (kahan16, an exact split). Perturbing the bare bf16 instead
                # loses the sub-ulp part of ``e`` coherently every step (25 ulp / 300 steps
                # measured). One stacked write per bucket; the residual noise draws from
                # this wrapper's own stream, so it is checkpointed like the SR write's.
                weights = torch.stack([p.data for p in plist])
                lows = torch.stack([st[RESIDUAL_KEY] for st in states])
                _ck_write_(weights, lows, m, 1.0, ck_bits, sr=self.sr_stream)
                torch._foreach_copy_([p.data for p in plist], list(weights.unbind(0)))
                torch._foreach_copy_([st[RESIDUAL_KEY] for st in states], list(lows.unbind(0)))
            else:  # low-precision weights: round-to-nearest, deliberately NOT stochastic
                for p, m_i in zip(plist, m.unbind(0), strict=True):
                    p.data.copy_((p.data.float() + m_i).to(p.dtype))

    @torch.no_grad()
    def _apply_fused(self, sign: float, scale: float):
        """Triton fast path for a ``norm="none"`` momentum perturbation.

        Returns the list of torch-path leftover buckets, or ``None`` if Triton is
        unavailable (caller then runs the full torch path). Momentum buffers are
        requantized in place by the shared codec contract, so pointer arrays remain
        valid until state is reset — and ``_plan_addrs_valid`` rechecks them in case
        a non-codec writer still reassigned."""
        try:
            import kaon._fused_triton as ft
            if not ft.HAS_TRITON:
                return None
        except Exception:  # noqa: BLE001 — optional dependency; torch path is always correct
            return None
        # Momentum storage has stable identity for the lifetime of optimizer
        # state when writers follow the codec's in-place store contract.  Reuse
        # the complete dispatch plan until _momentum_params observes a state-size
        # change, load resets the cache, or a witness pointer moves. Validating
        # the plan is a cheap O(N) data_ptr pass (2-3 ints per param); rebuilding
        # it is the expensive part (ptr_array allocation + bucket regrouping).
        self._momentum_params()  # O(1) cache-key check; catches late-gradient state growth
        cache = self._axpy_cache
        if cache is not None and cache["scale"] == scale and self._plan_addrs_valid(cache):
            self._launch_fused(cache, sign, scale, ft)
            return cache["leftover"]
        eligible: dict[tuple[Any, ...], tuple[list[Tensor], list[dict[str, Any]], dict[str, Any]]] = {}
        leftover: dict[tuple[Any, ...], tuple[list[Tensor], list[dict[str, Any]], str, tuple[int, ...], dict[str, Any]]] = {}
        for plist, states, md, shape, group in self._buckets():
            ok = md in ("float32", "bfloat16", "int8", "4bit") and all(
                p.is_cuda and p.data.is_contiguous() and p.dtype in (torch.float32, torch.bfloat16)
                for p in plist
            )
            # compact Kahan: the kernel needs every state's residual (contiguous, same numel)
            ck = _ck_ready(group, plist, states, sign > 0.0)
            if ck and not all(st[RESIDUAL_KEY].is_contiguous() for st in states):
                ok = False
            if ok:
                block = states[0].get("m_block", 1)
                row_width = plist[0].numel() // plist[0].shape[0] if plist[0].ndim >= 2 else plist[0].numel()
                key = (
                    plist[0].numel(), plist[0].dtype, md, block, row_width, id(group), ck
                )
                lp, ls, _g = eligible.setdefault(key, ([], [], group))
                lp.extend(plist)
                ls.extend(states)
            else:
                leftover[(shape, md, id(group))] = (plist, states, md, shape, group)
        buckets = []
        for (n, dtype, md, block, row_width, _gid, ck), (plist, states, group) in eligible.items():
            dev = plist[0].device
            mom = {
                "float32": ft.MOM_FP32,
                "bfloat16": ft.MOM_BF16,
                "int8": ft.MOM_INT8,
                "4bit": ft.MOM_4BIT,
            }[md]
            m_addr = ft.ptr_array([st["m"] for st in states], dev)
            # Witness via the STATE DICTS (stable objects) + address tuples, not via
            # retained Tensor refs: a base that reassigns st["m"] keeps the old Tensor
            # alive in a witness field (its data_ptr never changes) while the dict
            # already points at a new buffer — the plan would stay falsely valid.
            buckets.append(dict(
                p_addr=ft.ptr_array(plist, dev),
                c_addr=ft.ptr_array([st[RESIDUAL_KEY] for st in states], dev) if ck else None,
                ck=ck,                                   # residual width (8/16), 0 = plain
                plist=plist,
                states=states,
                p_addrs=tuple(map(_data_ptr, plist)),
                c_addrs=tuple(map(_data_ptr, map(_get_lo, states))) if ck else None,
                m_addrs=tuple(map(_data_ptr, map(_get_m, states))),
                sc_addrs=(
                    tuple(map(_data_ptr, map(_get_m_scale, states)))
                    if md in ("int8", "4bit") else None
                ),
                m_addr=m_addr,
                sc_addr=(
                    ft.ptr_array([st["m_scale"] for st in states], dev)
                    if md in ("int8", "4bit") else m_addr
                ),
                n=n, K=(n + 1023) // 1024, N=len(plist), block=block,
                row_width=row_width, mom=mom, lowp=dtype == torch.bfloat16,
                group=group,
            ))
        cache = self._axpy_cache = {
            "scale": scale,
            "buckets": buckets,
            "leftover": list(leftover.values()),
        }
        self._launch_fused(cache, sign, scale, ft)
        return cache["leftover"]

    @staticmethod
    def _plan_addrs_valid(cache: dict[str, Any]) -> bool:
        """True while every cached weight / momentum pointer still matches live storage.

        Re-reads ``data_ptr()`` from the param list and from ``states[*]["m"]`` /
        ``["m_scale"]`` (the dicts are stable; the tensors they name may be replaced).

        This runs on EVERY ``_apply`` — both the removal at the top of the step and the
        climb at the end — because each one launches a kernel that dereferences the cached
        ``p_addr`` / ``m_addr`` / ``sc_addr`` device tables. Checking only one of the two
        would leave a window in which a rebind (the inner optimizer between them; user code
        — ``.to()``, an EMA swap, a resharding — between steps and between
        ``eval()``/``train()``) makes the very next launch read or write freed CUDA memory.
        The check is therefore kept complete and made *cheap* instead: the scan is
        ``tuple(map(...))`` over module-level accessors (see ``_data_ptr`` / ``_get_m``),
        so the per-parameter loop runs in C. Nothing here allocates or touches the device;
        the whole cost is ``N`` unbound-method calls per witness.
        """
        for bk in cache["buckets"]:
            if tuple(map(_data_ptr, bk["plist"])) != bk["p_addrs"]:
                return False
            states = bk["states"]
            # bf16_method switched since the plan was built (either way), or a compact-Kahan
            # bucket was planned plain (residuals missing / of mixed widths): rebuild. The
            # climb's rebuild normalizes the bucket (_ck_ready), so this settles after at most
            # one removal + one climb. O(1) per bucket unless the method and the plan
            # disagree. A kahan8 <-> kahan16 conversion replaces the residual tensors: the
            # c_addrs check below.
            if (bk.get("lowp")
                    and bool(bk["ck"]) != is_compact_kahan(bk["group"].get("bf16_method", ""))):
                return False
            c_addrs = bk.get("c_addrs")
            if c_addrs is not None and (
                any(RESIDUAL_KEY not in st for st in states)
                or tuple(map(_data_ptr, map(_get_lo, states))) != c_addrs
            ):
                return False
            if tuple(map(_data_ptr, map(_get_m, states))) != bk["m_addrs"]:
                return False
            sc_addrs = bk["sc_addrs"]
            if sc_addrs is not None and (
                tuple(map(_data_ptr, map(_get_m_scale, states))) != sc_addrs
            ):
                return False
        return True

    def _launch_fused(self, cache: dict[str, Any], sign: float, scale: float, ft: Any) -> None:
        """Launch a cached fused perturbation plan."""
        self._axpy_seed += 1
        for bk in cache["buckets"]:
            alpha = sign * self.rho * scale * self._climb_step_scale(bk["group"], sign)
            ft._axpy_momentum_batched[(bk["N"] * bk["K"],)](
                bk["p_addr"], bk["c_addr"] if bk["c_addr"] is not None else bk["p_addr"],
                bk["m_addr"], bk["sc_addr"], alpha,
                self._climb_bound(bk["group"], sign), bk["n"], bk["K"], bk["row_width"],
                _climb_seed(self._axpy_seed), MOM=bk["mom"], FBLOCK=bk["block"],
                # SR=False: round-to-nearest, matching the torch path. See the bf16 note.
                # CK (kahan8/kahan16): decoded-value climb — see the kernel's doc.
                LOWP=bk["lowp"], SR=False, BLOCK=1024, CK=bk["ck"],
            )

    # --------------------------------------------------------------- train/eval
    @torch.no_grad()
    def eval(self) -> None:  # noqa: A003 — mirrors the optimizer.eval() API (Lookahead/SF)
        """Remove the perturbation (it lives on the inner TRAIN weights, so unperturb
        first), then chain into a wrapped inner's own eval view (e.g. Lookahead's phi)."""
        if self._train_mode and self._has_e:
            self._apply(-1.0)
        self._train_mode = False
        if hasattr(self.inner, "eval"):
            self.inner.eval()

    @torch.no_grad()
    def train(self) -> None:
        """Chain the inner back to its train view first, then restore the perturbation
        on top of it (momentum is unchanged in between, so the climb re-lands exactly)."""
        if hasattr(self.inner, "train"):
            self.inner.train()
        if not self._train_mode and self._has_e:
            self._apply(+1.0)
        elif not self._train_mode:
            # A checkpoint is saved in eval/true view and cannot serialize an
            # already-applied climb. Rebuild it from the restored momentum.
            self._restore_live_view()
        self._train_mode = True

    def _restore_live_view(self) -> None:
        """Rebuild the ordinary unit-scale live view after loading a checkpoint."""
        if self.rho == 0.0:
            return
        if self.norm == "global":
            self._mnorm = self._global_mnorm()
            climb = self._mnorm > 0.0
        else:
            climb = bool(self._momentum_params())
        if climb:
            self._apply(+1.0, scale=1.0)
            self._has_e = True

    # ------------------------------------------------------------------- probe
    @torch.no_grad()
    def _probe(self, phase: str) -> None:
        """Log the first non-finite tensor for ``phase`` (see module docstring). Probe-only."""
        self._probe_step = getattr(self, "_probe_step", 0)
        if phase == "GRAD" and not getattr(self, "_probe_dumped", False):
            # Rolling 1-step snapshot of the full optimizer state (cpu): the NaN is BORN
            # inside a step, so the replayable forensics need the PRE-step state. Cheap on
            # an adapter run (a few MB); probe-only.
            self._probe_snap = {
                id(p): {
                    "grad": p.grad.detach().cpu() if p.grad is not None else None,
                    "p": p.data.detach().cpu(),
                    "state": {k: (v.detach().cpu().clone() if torch.is_tensor(v) else v) for k, v in st.items()},
                }
                for p, st, _md, _g in self._momentum_params()
            }
        for p, st, md, _group in self._momentum_params():
            bad = None
            if phase == "GRAD" and p.grad is not None and not torch.isfinite(p.grad).all():
                bad = f"grad absmax={p.grad.abs().max().item():.3e}"
            elif not torch.isfinite(p.data).all():
                bad = f"weight absmax={p.data.detach().abs().max().item():.3e}"
            elif phase == "STATE":
                m = CodecBuffer.read(st, "m", md, p)
                if not torch.isfinite(m).all():
                    sc = st.get("m_scale")
                    bad = (f"momentum (codec {md}; scale absmax="
                           f"{sc.abs().max().item():.3e})" if sc is not None else f"momentum ({md})")
                    # FORENSICS (first occurrence only): full stats + a replayable dump of the
                    # offending tensor's grad/state, so the exact step can be re-run offline.
                    if not getattr(self, "_probe_dumped", False):
                        self._probe_dumped = True
                        row, col = st.get("row"), st.get("col")
                        nan_rows = int(torch.isnan(m).any(dim=-1).sum()) if m.ndim >= 2 else -1
                        with open(_PROBE_LOG, "a") as fh:  # noqa: SIM115
                            fh.write(
                                f"[FORENSICS] step={self._probe_step} shape={tuple(p.shape)} "
                                f"m: nan={int(torch.isnan(m).sum())} inf={int(torch.isinf(m).sum())} "
                                f"nan_rows={nan_rows} | "
                                f"row[min={row.min().item():.3e},max={row.max().item():.3e}] "
                                f"col[min={col.min().item():.3e},max={col.max().item():.3e}] | "
                                f"grad absmax={p.grad.abs().max().item():.3e} "
                                f"p absmax={p.data.abs().max().item():.3e} dtype={p.dtype}\n"
                                if row is not None and col is not None and p.grad is not None else
                                f"[FORENSICS] step={self._probe_step} shape={tuple(p.shape)} (1-D or no grad)\n"
                            )
                        dump = {
                            "shape": tuple(p.shape), "step": self._probe_step, "dtype": str(p.dtype),
                            "p": p.data.detach().cpu(), "grad": (p.grad.detach().cpu() if p.grad is not None else None),
                            "state": {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in st.items()},
                            # the PRE-step snapshot (grad/p/state at this step's GRAD phase) — replayable
                            "pre": getattr(self, "_probe_snap", {}).get(id(p)),
                        }
                        torch.save(dump, _PROBE_LOG + ".forensics.pt")
            if bad is not None:
                with open(_PROBE_LOG, "a") as fh:  # noqa: SIM115 — diagnostics only
                    fh.write(f"[{phase}] step={self._probe_step} shape={tuple(p.shape)} {bad}\n")
                return

    # --------------------------------------------------------------------- step
    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        if not self._train_mode:
            raise RuntimeError(
                "MSAM.step() called outside train mode. Call optimizer.train() before the "
                "training step (and optimizer.eval() before sampling / checkpointing)."
            )
        if _PROBE_LOG:
            self._probe_step = getattr(self, "_probe_step", 0) + 1
            self._probe("GRAD")
        # 1) remove the previous climb — the incoming p.grad was computed at the
        #    perturbed point (that is the SAM gradient); the update belongs at the base.
        if self._has_e:
            self._apply(-1.0)
            self._has_e = False
        self._eclamp.clear()  # next climb re-freezes the per-element bound at the CURRENT lr
        self._estep_scale.clear()
        # 2) base step at the true weights, with the perturbed-point gradient.
        loss = self.inner.step(closure)
        if _PROBE_LOG:
            self._probe("STATE")
        # 3) climb along the refreshed momentum for the NEXT forward/backward.
        if self.rho != 0.0:
            if self.norm == "global":
                self._mnorm = self._global_mnorm()
                climb = self._mnorm > 0.0
            else:  # tensor/none scale per slice inside _apply (zero-m slices no-op)
                climb = bool(self._momentum_params())
            if climb:
                self._warn_if_inert()
                self._apply(+1.0, scale=1.0)
                self._has_e = True
            elif not self._no_m_warned and self.inner.state and not self._momentum_params():
                # beta1=0 (Adakaon/KProdigy) never allocates `m`; MSAM would otherwise
                # stay a silent no-op. Only warn once the inner already has state —
                # a smoke step() with no gradients must not trip this.
                self._no_m_warned = True
                warnings.warn(
                    f"{type(self).__name__}: no first-moment buffer `m` found on the base "
                    f"optimizer (e.g. betas[0]=0). The lookahead has no effect; set rho/k=0 "
                    f"to drop the cost, or enable momentum on the base.",
                    stacklevel=2,
                )
        if _PROBE_LOG:
            self._probe("CLIMB")
        return loss

    # -------------------------------------------------------------- state_dict
    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        state_dict["_msam_meta"] = {
            "axpy_seed": self._axpy_seed,
            # Recorded so a checkpoint taken in train mode (weights carrying the climb,
            # which cannot be reconstructed on resume) fails loudly instead of silently
            # baking one perturbation into the weights per resume.
            "train_mode": self._train_mode,
        }
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the inner base optimizer (dtype-preserving — every kaon optimizer's own
        ``load_state_dict`` already delegates to the preserving helper; a wrapped wrapper
        restores its own buffers too). MSAM itself keeps no persistent per-param state
        (the perturbation is recomputed from momentum). Checkpoints must be saved in eval
        mode (unperturbed weights)."""
        copied = dict(state_dict)
        meta = copied.pop("_msam_meta", {})
        axpy_seed = int(meta.get("axpy_seed", 0))
        if axpy_seed < 0:
            raise ValueError("MSAM checkpoint has an invalid fused stochastic-rounding seed")
        if meta.get("train_mode", False):
            raise ValueError(
                "MSAM checkpoint was saved in train mode: the stored weights carry the "
                "lookahead perturbation, and a fresh optimizer cannot know to remove it — "
                "resuming would bake one perturbation into the weights per resume. Call "
                "optimizer.eval() before saving the checkpoint."
            )
        self._load_wrapped(copied, lambda inner, sd: inner.load_state_dict(sd))
        self.base_optimizer = self.inner
        # Checkpoints are required to contain the eval/true weights. Stay in
        # that view after loading so the caller's normal train() transition can
        # reconstruct the live perturbation before the first forward pass.
        self._train_mode = False
        self._has_e = False
        self._mnorm = 0.0
        self._eclamp.clear()
        self._estep_scale.clear()
        self._axpy_cache = None
        self._axpy_seed = axpy_seed
        self._e_scale = 1.0
        self._momentum_cache_key = None
        self._momentum_cache = []
        self._bucket_cache = None
        self.train()
