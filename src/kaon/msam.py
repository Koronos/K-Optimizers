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

Sign of ``rho``. The base optimizer's momentum is an EMA of the (lr-scaled,
v-normalized) *update*, which points along the gradient — i.e. uphill. ``rho > 0``
climbs uphill (the SAM-like direction); ``rho < 0`` probes the Nesterov-like downhill
lookahead instead. Both are exposed because the right sign is an empirical question
(measured on the control battery, not assumed).

train() / eval(). Between steps the live weights deliberately carry the perturbation —
that is the mechanism — so **sampling / validation / checkpointing must use**
:meth:`eval` (removes the perturbation) **and** :meth:`train` (restores it) around the
measurement, exactly like Lookahead / Schedule-Free. Always checkpoint in eval mode: a
checkpoint saved in train mode stores perturbed weights, and a fresh MSAM cannot know to
remove that perturbation on resume.

bf16 note. The climb/restore round-trip is two stochastic-rounding writes on bf16
weights (unbiased, but not bit-exact); on fp32 weights the round-trip is exact to
floating-point addition. SAM avoids this with a full weight snapshot per step; MSAM
deliberately trades that snapshot away for zero memory.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from kaon._stochastic_rounding import add_stochastic_
from kaon._wrappers import CodecBuffer, WrapsInnerOptimizer

__all__ = ["MSAM"]

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
            fuse into a single batched pass); ``"none"`` applies the **raw** momentum,
            ``e = rho * m`` — since a kaon momentum is the EMA of the *lr-scaled,
            preconditioned, RMS-clipped update*, this makes ``rho`` a **lookahead measured
            in optimizer steps** ("perturb to where ~rho more steps would land"). Unlike a
            fixed weight-space radius, that is dimensionless and self-scaling: it tracks
            the LR (and any schedule), the per-coordinate ``1/sqrt(v)`` metric, and the
            model's weight scale by construction — the transfer-robust formulation.
        eps: numerical floor on the momentum norm before dividing.
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
        **kwargs: Any,
    ) -> None:
        if eps < 0.0:
            raise ValueError(f"eps must be >= 0, got {eps}")
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
        # Live weights carry the perturbation only while (training mode AND a momentum
        # exists). eval()/train() toggle the mode; _has_e tracks whether a perturbation
        # is currently defined (false until the first inner step populates momentum).
        self._train_mode = True
        self._has_e = False
        self._mnorm = 0.0  # global ||m|| cached at climb time (m is unchanged until the next inner step)
        self._axpy_cache: dict[str, Any] | None = None  # Triton 4bit fast-path pointer arrays
        self._axpy_seed = 0                             # SR seed counter for the fused axpy
        self._eclamp: dict[int, float] = {}             # per-group climb bound, frozen per cycle

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
        out = []
        owner_state = self._momentum_owner().state
        for group in self.param_groups:
            md = group["momentum_dtype"]
            for p in group["params"]:
                st = owner_state.get(p)
                if st and "m" in st:
                    out.append((p, st, md, group))
        return out

    def _buckets(self) -> list[tuple[list[Tensor], list[dict[str, Any]], str, tuple[int, ...], dict[str, Any]]]:
        """Group the momentum-carrying params by (shape, dtype, momentum_dtype, group) so
        the perturbation reads/applies as a few stacked ops instead of a per-param loop
        (the per-param dequant x2 per step made the 512-tiny-tensor LoRA regime ~10x
        slower). Buckets never mix param groups: the per-element climb bound is a
        per-group quantity (it reads the group's lr / clip_threshold)."""
        by_key: dict[tuple[Any, ...], tuple[list[Tensor], list[dict[str, Any]], dict[str, Any]]] = {}
        for p, st, md, group in self._momentum_params():
            key = (tuple(p.shape), p.dtype, md, id(group))
            plist, states, _g = by_key.setdefault(key, ([], [], group))
            plist.append(p)
            states.append(st)
        return [(plist, states, key[2], key[0], g) for key, (plist, states, g) in by_key.items()]

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
    def _apply(self, sign: float) -> None:
        """Add ``sign * rho * m / ||m||`` to every weight (bf16-correct write), bucketed.

        ``norm="global"`` uses the cached cross-param norm (one radius for the net);
        ``norm="tensor"`` rescales each slice by its own momentum norm (recomputed — m is
        unchanged between the climb and its removal, so the round trip is exact).
        ``norm="none"`` + 4-bit momentum on GPU takes the Triton fast path: the torch
        dequant (unpack -> scale -> stack -> axpy, several kernels + an fp32 temp) is
        the dominant perturbation cost at 4 bits; ``_axpy_4bit_batched`` does dequant +
        bf16-SR axpy in ONE launch per bucket (measured ~4.4 ms/step -> sub-ms on the
        C=128 proxy). Ineligible params (non-contiguous, exotic dtype) fall back here."""
        leftover = self._apply_fused_4bit(sign) if self.norm == "none" else None
        for plist, states, md, shape, group in (self._buckets() if leftover is None else leftover):
            m = CodecBuffer.read_stacked(states, "m", md, shape)  # [N, *shape] fp32
            n = m.shape[0]
            if self.norm == "global":
                m.mul_(sign * self.rho / (self._mnorm + self.eps))
            elif self.norm == "tensor":  # per-tensor radius: rho * m_i / ||m_i|| per slice
                norms = (m * m).reshape(n, -1).sum(dim=1).sqrt_()  # no .norm(): dot SIGFPEs here
                scales = (sign * self.rho) / (norms + self.eps)
                m.mul_(scales.view(n, *([1] * (m.ndim - 1))))
            else:  # "none": raw momentum — rho is a lookahead in OPTIMIZER-STEP units
                m.mul_(sign * self.rho)
                bound = self._climb_bound(group, sign)
                # NaN passes through clamp(): a non-finite momentum coordinate (e.g. a
                # 0*inf from a blown 4-bit block scale) must contribute ZERO climb, never
                # poison the weights. inf is mapped to +-bound by the clamp either way.
                torch.nan_to_num_(m, nan=0.0, posinf=bound, neginf=-bound)
                m.clamp_(-bound, bound)  # per-element stability cap (see _climb_bound)
            if plist[0].dtype == torch.float32:
                torch._foreach_add_([p.data for p in plist], list(m.unbind(0)))
            else:  # low-precision weights: stochastic-rounding write per slice
                for p, m_i in zip(plist, m.unbind(0), strict=True):
                    add_stochastic_(p.data, m_i, alpha=1.0)

    @torch.no_grad()
    def _apply_fused_4bit(self, sign: float):
        """Triton fast path for the ``norm="none"`` 4-bit perturbation.

        Returns the list of torch-path leftover buckets, or ``None`` if Triton is
        unavailable (caller then runs the full torch path). The packed/scale buffers are
        REPLACED by every requant, so the pointer arrays are rebuilt whenever the buffer
        ids change (they stay valid across one climb -> removal/eval/train cycle)."""
        try:
            import kaon._fused_triton as ft
            if not ft.HAS_TRITON:
                return None
        except Exception:  # noqa: BLE001 — optional dependency; torch path is always correct
            return None
        eligible: dict[tuple[int, Any, int, int], tuple[list[Tensor], list[dict[str, Any]], dict[str, Any]]] = {}
        leftover: dict[tuple[Any, ...], tuple[list[Tensor], list[dict[str, Any]], str, tuple[int, ...], dict[str, Any]]] = {}
        for plist, states, md, shape, group in self._buckets():
            ok = md == "4bit" and all(
                p.is_cuda and p.data.is_contiguous() and p.dtype in (torch.float32, torch.bfloat16)
                for p in plist
            )
            if ok:
                key = (plist[0].numel(), plist[0].dtype, states[0]["m_block"], id(group))
                lp, ls, _g = eligible.setdefault(key, ([], [], group))
                lp.extend(plist)
                ls.extend(states)
            else:
                leftover[(shape, md, id(group))] = (plist, states, md, shape, group)
        ids = tuple(id(st["m"]) for _k, (_lp, ls, _g) in sorted(eligible.items(), key=lambda kv: kv[0][0]) for st in ls)
        cache = self._axpy_cache
        if cache is None or cache["ids"] != ids:
            buckets = []
            for (n, dtype, block, _gid), (plist, states, group) in eligible.items():
                dev = plist[0].device
                buckets.append(dict(
                    p_addr=ft.ptr_array(plist, dev),
                    pk_addr=ft.ptr_array([st["m"] for st in states], dev),
                    sc_addr=ft.ptr_array([st["m_scale"] for st in states], dev),
                    n=n, K=(n + 1023) // 1024, N=len(plist), block=block,
                    lowp=dtype == torch.bfloat16,
                    # frozen at climb time (cache rebuilds exactly once per climb — requant
                    # replaces the m buffers, changing the ids): the removal/eval/train
                    # swaps must subtract the SAME clamped e even if a scheduler moved lr.
                    bound=self._climb_bound(group, sign),
                ))
            cache = self._axpy_cache = {"ids": ids, "buckets": buckets}
        self._axpy_seed += 1
        alpha = sign * self.rho
        for bk in cache["buckets"]:
            ft._axpy_4bit_batched[(bk["N"] * bk["K"],)](
                bk["p_addr"], bk["pk_addr"], bk["sc_addr"], alpha, bk["bound"], bk["n"], bk["K"],
                self._axpy_seed, FBLOCK=bk["block"], LOWP=bk["lowp"], SR=bk["lowp"], BLOCK=1024,
            )
        return list(leftover.values())

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
        self._train_mode = True

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
                self._apply(+1.0)
                self._has_e = True
        if _PROBE_LOG:
            self._probe("CLIMB")
        return loss

    # -------------------------------------------------------------- state_dict
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the inner base optimizer (dtype-preserving — every kaon optimizer's own
        ``load_state_dict`` already delegates to the preserving helper; a wrapped wrapper
        restores its own buffers too). MSAM itself keeps no persistent per-param state
        (the perturbation is recomputed from momentum). Checkpoints must be saved in eval
        mode (unperturbed weights)."""
        self._load_wrapped(state_dict, lambda inner, sd: inner.load_state_dict(sd))
        self.base_optimizer = self.inner
        self._train_mode = True
        self._has_e = False
        self._mnorm = 0.0
        self._eclamp.clear()
        self._axpy_cache = None
