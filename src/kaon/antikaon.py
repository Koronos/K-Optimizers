"""Antikaon — momentum-free Adakaon + seeded Anti-PGD / RWP perturbation living in the weights.

Antikaon is :class:`~kaon.adakaon.Adakaon` (momentum-free by default) with one structural
addition: between steps the live weights carry a zero-mean random perturbation ``xi``, so
every gradient is evaluated at a *randomly perturbed* point, while the update lands on the
clean iterate ``z = w - xi``. On the clean iterate this is exactly Anti-PGD (Orvieto et al.
2022) / single-sample RWP (Li et al. 2024):

.. code-block:: text

    # live weights hold w_n = z_n + xi_n
    g_n     = grad L(w_n)                        # gradient AT the perturbed point
    row,col <- EMA(g_n^2)                        # v sees the perturbed gradient
    delta_n = clip_rms(P_n g_n) + wd * (w_n - xi_n)    # decay acts on z, not on w
    w_{n+1} = write(w_n - lr * delta_n + xi_{n+1} - xi_n)   # ONE combined write

whose implicit objective is ``L(z) + 1/2 Tr(Sigma * H(z))`` — a flat-minima bias — at
**zero persistent state** and **zero extra forward/backward passes**: ``xi_n`` is never
stored, it is regenerated from ``(noise_seed, parameter index, per-parameter step)`` and
the second-moment factors the step already keeps. Design note, derivations and bf16
simulations: ``docs/research/antikaon-design.md``; user guide: ``docs/antikaon.md``.

Noise law. ``xi = sigma * S * eps``:

* ``eps`` is Rademacher (``+-1``, default; Anti-PGD's theorem is stated for it, no tails)
  or Gaussian (``noise="gaussian"``).
* ``sigma = k_sigma * lr * clip_threshold`` — the radius is measured in **optimizer steps**
  (Nekaon's invariant): it self-scales with the LR and any schedule. ``sigma_ref="weight"``
  is the filter-wise ablation, ``sigma_i = max(k_weight * RMS_row_i(w), k_sigma*lr*clip)``
  (the floor is what keeps a zero-init LoRA ``B`` perturbed).
* ``S`` (``shape="v"``, default) is the scale-free ARWP shaping ``(mean(v_hat)/v_hat)^(1/4)``,
  a rank-1 product of the factored second moment, clamped to ``[1/s_cap, s_cap]`` — the
  covariance is then proportional to Adam's preconditioner and the regularizer is the
  preconditioned sharpness ``Tr(P*H)``. ``shape="none"`` gives ``S = 1`` (``Tr(H)``).
* ``antithetic=True`` pairs consecutive draws (``xi_{2m+1} = -xi_{2m}`` in the per-parameter
  noise index): the regularizer is unchanged while the first-order noise injected into ``z``
  cancels across each pair.

The combined write goes through the SAME writers Adakaon uses
(:func:`~kaon._backend.subtract_one_` / :func:`~kaon._backend.subtract_batched_`) with
``delta' = delta - (xi_{n+1} - xi_n) / lr``; every ``bf16_method`` they implement
(stochastic rounding, Kahan, compact Kahan ``kahan8``, plain) therefore works without
Antikaon knowing about it — see :meth:`Antikaon._combined_write_one` /
:meth:`Antikaon._combined_write_batched`, the only two places that touch the weight during
a step. Everything else that needs the weight's full value (weight decay, the
``sigma_ref="weight"`` radius, eval/train) goes through :meth:`Antikaon._read_clean` /
:meth:`Antikaon._write_clean` (and the stacked :meth:`Antikaon._clean_stack`), which know
each method's residual.

Like Nekaon / MSAM / Lookahead, the live weights between steps are NOT the clean iterate:
call :meth:`eval` before validation / sampling / checkpointing and :meth:`train` to resume
(always checkpoint in eval mode). ``step()`` in eval mode raises.

Paths: per-parameter and native foreach, bit-exact with each other on fp32 CPU params (on
CUDA they agree to Adakaon's own ~1-ulp foreach/per-param reduction parity). Noise is drawn
per parameter with a per-parameter seed, so bucket composition never changes it.

State: per parameter, ``noise_step`` (int, the live noise index) and ``noise_sigma`` (float,
the step radius frozen when that noise was installed — an LR schedule between the install
and the removal cannot desynchronize them); ``sigma_ref="weight"`` adds ``noise_sigma_rows``
(``R`` fp32). No per-element state: 0 B/param over momentum-free Adakaon. The
Triton ``fused`` path is **not** implemented for Antikaon yet: ``fused=True`` warns and runs
the native foreach path. Every noise quantity is computed by :meth:`Antikaon._noise`, the
single place a fused ``NOISE`` constexpr branch has to reproduce.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from typing import Any, Literal

import torch
from torch import Tensor

from kaon._backend import (
    cautious_batched_,
    cautious_one_,
    ensure_residuals,
    flat_view,
    rms,
    subtract_batched_,
    subtract_one_,
)
from kaon._compact_kahan import RESIDUAL_KEY, decode, encode_, is_compact_kahan, residual_bits
from kaon._factored import factored_inv_sqrt_factors, update_factored_state
from kaon._foreach_plan import ForeachChunk
from kaon._momentum_codec import _MomentumCodec
from kaon._stochastic_rounding import SRStream, _device_generator
from kaon.adakaon import Adakaon
from kaon.msam import MSAM

__all__ = ["Antikaon"]

NoiseLaw = Literal["rademacher", "gaussian"]
NoiseShape = Literal["v", "none"]
SigmaRef = Literal["step", "weight"]

# The noise stream identity recorded in checkpoints. "torch" = one torch.Generator draw per
# parameter per noise index. A future Triton NOISE branch draws from a different stream
# (tl.rand Philox) and must record a different name; see Antikaon.load_state_dict.
NOISE_BACKEND = "torch"


def _noise_backend(device_types: Iterable[str]) -> str:
    """``"torch-cuda"``, ``"torch-cpu"``, ... — a torch.Generator stream differs per device
    type, so the device is part of the noise identity (``"torch-cpu+cuda"`` if mixed)."""
    kinds = sorted(set(device_types))
    return f"{NOISE_BACKEND}-{'+'.join(kinds) if kinds else 'none'}"

_M64 = (1 << 64) - 1
_M63 = (1 << 63) - 1


def _splitmix64(x: int) -> int:
    """SplitMix64 finalizer on Python ints (Steele et al. 2014) — a cheap, well-mixed hash."""
    x = (x + 0x9E3779B97F4A7C15) & _M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M64
    return x ^ (x >> 31)


def noise_seed_for(noise_seed: int, pid: int, draw: int) -> int:
    """The torch.Generator seed of draw ``draw`` of parameter ``pid`` (63-bit, deterministic).

    Chained hashing, so ``(seed, pid, draw)`` triples never collide structurally the way a
    linear combination would. Public so a reference implementation (tests, diagnostics) can
    regenerate exactly the optimizer's ``eps``.
    """
    return _splitmix64(_splitmix64(_splitmix64(noise_seed & _M64) ^ pid) ^ draw) & _M63


class Antikaon(Adakaon):
    """Momentum-free Adakaon with a seeded Anti-PGD/RWP perturbation carried in the weights.

    Args:
        params: parameters or param-group dicts.
        lr: learning rate.
        k_sigma: perturbation radius in **optimizer steps** (``sigma = k_sigma * lr * clip``
            per-coordinate RMS before shaping). Dimensionless and LR-self-scaling. ``0``
            disables the mechanism (plain Adakaon, bit-exact) unless ``sigma_ref="weight"``
            with ``k_weight > 0``. Default ``5.0`` is the design prior, NOT a measured
            optimum (the planned sweep is ``{1.5, 5, 15}``).
        betas: ``(beta1, beta2)``. Default ``(0.0, 0.999)`` — momentum-free (the design
            point: 0.032 B/param). ``beta1 > 0`` is allowed and runs Adakaon's momentum codec
            unchanged; the perturbation does not depend on it.
        weight_decay: decoupled weight decay, applied to the CLEAN iterate ``z = w - xi``.
        shape: ``"v"`` (default) — noise std shaped by the factored second moment,
            ``(mean(v_hat)/v_hat)^(1/4)`` clamped to ``[1/s_cap, s_cap]``; ``"none"`` —
            isotropic.
        noise: ``"rademacher"`` (default, bounded: ``|xi| <= sigma*S``) or ``"gaussian"``.
        antithetic: pair consecutive draws with opposite signs (``xi_{2m+1} = -xi_{2m}`` in
            each parameter's noise index).
        sigma_ref: ``"step"`` (default; radius in optimizer steps) or ``"weight"``
            (ablation: ``sigma_i = max(k_weight * RMS_row_i(w), k_sigma * lr * clip)``,
            frozen per row at install time — ``R`` fp32 floats of state per tensor).
        k_weight: relative radius for ``sigma_ref="weight"`` (ignored otherwise).
        s_cap: clamp of the shaping factor (``>= 1``). Default ``4``.
        noise_seed: base seed of the noise stream. ``None`` (default) derives it from
            ``torch.initial_seed()``, so ``torch.manual_seed(s)`` before construction makes
            runs reproducible and different seeds give different noise. Saved in (and
            restored from) the state dict.
        cautious: forwarded to Adakaon. Default ``False`` (a no-op without momentum).
        fused: Antikaon has no Triton path yet — ``True`` warns and runs the native
            foreach path (same math).
        inert_check_interval: sample the inert-noise warning every this many steps (same
            heuristic and cadence as MSAM's inert-lookahead warning).
        **adakaon_kwargs: forwarded verbatim to :class:`~kaon.adakaon.Adakaon` (``eps``,
            ``clip_threshold``, ``bf16_method``, ``gradient_centralization``, ``foreach``, ...).

    Parameter identity: the noise of a parameter is keyed by its index in the flattened
    ``param_groups`` order (the same key ``state_dict`` uses). Adding a param group appends
    new indices; re-ordering parameters between runs changes the noise (not the correctness
    of an eval-mode checkpoint). Each parameter keeps its own noise index (``noise_step`` in
    its state), so a parameter that skips a step (``grad is None``) keeps its live ``xi``,
    and the radius of every installed ``xi`` is frozen with it (``noise_sigma``), so an LR
    schedule cannot desynchronize install and removal. The noise law (``k_sigma``, ``shape``,
    ``noise``, ``antithetic``, ``sigma_ref``, ``k_weight``, ``s_cap``) is read-only for the
    same reason: the removal regenerates ``xi`` with it.
    """

    # Reuse MSAM's inert-lookahead heuristic constants — one definition of "inert" in kaon.
    _INERT_REL = MSAM._INERT_REL
    _INERT_PATIENCE = MSAM._INERT_PATIENCE
    _INERT_MAX_CHECKS = MSAM._INERT_MAX_CHECKS
    _INERT_SAMPLE = MSAM._INERT_SAMPLE

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        k_sigma: float = 5.0,
        betas: tuple[float, float] = (0.0, 0.999),
        weight_decay: float = 0.0,
        *,
        shape: NoiseShape = "v",
        noise: NoiseLaw = "rademacher",
        antithetic: bool = False,
        sigma_ref: SigmaRef = "step",
        k_weight: float = 0.01,
        s_cap: float = 4.0,
        noise_seed: int | None = None,
        cautious: bool = False,
        fused: bool = False,
        inert_check_interval: int = 10,
        **adakaon_kwargs: Any,
    ) -> None:
        if not (math.isfinite(k_sigma) and k_sigma >= 0.0):
            raise ValueError(f"k_sigma must be finite and >= 0, got {k_sigma}")
        if not (math.isfinite(k_weight) and k_weight >= 0.0):
            raise ValueError(f"k_weight must be finite and >= 0, got {k_weight}")
        if not (math.isfinite(s_cap) and s_cap >= 1.0):
            raise ValueError(f"s_cap must be finite and >= 1, got {s_cap}")
        if shape not in ("v", "none"):
            raise ValueError(f"shape must be 'v' or 'none', got {shape!r}")
        if noise not in ("rademacher", "gaussian"):
            raise ValueError(f"noise must be 'rademacher' or 'gaussian', got {noise!r}")
        if sigma_ref not in ("step", "weight"):
            raise ValueError(f"sigma_ref must be 'step' or 'weight', got {sigma_ref!r}")
        if isinstance(inert_check_interval, bool) or not isinstance(inert_check_interval, int) \
                or inert_check_interval < 1:
            raise ValueError("inert_check_interval must be a positive integer")
        if fused:
            warnings.warn(
                "Antikaon has no Triton-fused path yet: fused=True runs the native foreach "
                "path (same math and state).",
                stacklevel=2,
            )
        # Set BEFORE super().__init__: Optimizer.__init__ routes every group through
        # add_param_group, which invalidates the parameter-index map.
        self._pid_map: dict[int, int] | None = None
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            cautious=cautious,
            fused=False,
            **adakaon_kwargs,
        )
        # The noise LAW is fixed at construction (read-only properties below): the removal of
        # xi_n regenerates it with the law that installed it, so changing any of these
        # between two steps would subtract a different xi than the one in the weights.
        self._k_sigma = float(k_sigma)
        self._k_weight = float(k_weight)
        self._s_cap = float(s_cap)
        self._shape = shape
        self._noise_law = noise
        self._antithetic = bool(antithetic)
        self._sigma_ref = sigma_ref
        self.noise_seed = int(torch.initial_seed() if noise_seed is None else noise_seed) & _M63
        self.inert_check_interval = inert_check_interval
        # The mechanism is off only when no radius can be non-zero; then every override
        # below delegates to Adakaon verbatim (k_sigma=0 IS Adakaon, bit for bit).
        self._noise_on = self.k_sigma > 0.0 or (sigma_ref == "weight" and self.k_weight > 0.0)
        # Live weights carry xi only in train mode. eval()/train() toggle it.
        self._train_mode = True
        self._noise_gens: dict[torch.device, torch.Generator] = {}
        self._inert_streak = 0
        self._inert_checks = 0
        self._inert_warned = False

    # ------------------------------------------------------------------ noise law (read-only)
    # Read-only on purpose: ``xi_n`` is never stored, its removal REGENERATES it, so the law
    # that removes it must be the law that installed it. Construct a new optimizer (and
    # checkpoint in eval mode) to change any of these.
    @property
    def k_sigma(self) -> float:
        return self._k_sigma

    @property
    def k_weight(self) -> float:
        return self._k_weight

    @property
    def s_cap(self) -> float:
        return self._s_cap

    @property
    def shape(self) -> str:
        return self._shape

    @property
    def noise(self) -> str:
        return self._noise_law

    @property
    def antithetic(self) -> bool:
        return self._antithetic

    @property
    def sigma_ref(self) -> str:
        return self._sigma_ref

    # ------------------------------------------------------------------ bookkeeping
    def add_param_group(self, param_group: dict[str, Any]) -> None:
        self._pid_map = None
        super().add_param_group(param_group)

    def _pids(self) -> dict[int, int]:
        """``id(param) -> index`` in the flattened ``param_groups`` order (cached)."""
        pmap = self._pid_map
        if pmap is None:
            pmap = self._pid_map = {
                id(p): i for i, p in enumerate(p for g in self.param_groups for p in g["params"])
            }
        return pmap

    def _gen(self, device: torch.device) -> torch.Generator:
        gens = self.__dict__.setdefault("_noise_gens", {})
        g = gens.get(device)
        if g is None:
            g = gens[device] = torch.Generator(device=device)
        return g

    def _sigma_step(self, lr: float, clip: float) -> float:
        """The step-unit radius ``k_sigma * lr * clip`` (frozen into the state at install)."""
        return self.k_sigma * lr * clip

    # ------------------------------------------------------------------ the noise
    def _draw_sign(self, k: int) -> tuple[int, float]:
        """Noise index ``k >= 1`` -> (``eps`` draw number, sign). Antithetic pairs share a draw."""
        j = k - 1
        if self.antithetic:
            return j >> 1, (-1.0 if j & 1 else 1.0)
        return j, 1.0

    @torch.no_grad()
    def _noise(
        self,
        pids: list[int],
        ks: list[int],
        scales: list[float],
        eff: tuple[int, ...],
        src: tuple[Tensor, ...],
        rows: Tensor | None,
        device: torch.device,
    ) -> Tensor:
        """``xi`` for N parameters, stacked ``[N, *eff]`` fp32. THE noise definition.

        ``ks[i]`` is parameter i's noise index (``0`` = no noise: that row is zero),
        ``scales[i]`` its frozen step radius (``1.0`` under ``sigma_ref="weight"``, where
        ``rows`` ``[N, R, 1]`` / ``[N, 1]`` carries the per-row radius), ``src`` the
        second-moment state the shaping reads — ``(row [N,R], col [N,C])`` factored or
        ``(v [N,L],)`` — at the values it had when this ``xi`` was installed.

        The per-parameter path calls this with ``N = 1`` and the foreach path with the
        whole chunk: one code path, so the two agree bit for bit wherever the elementwise
        kernels are layout-invariant (see :meth:`_shape_factor`). ``eps`` is
        drawn per parameter (its own seed, flat in the parameter's element order), so a
        parameter's noise never depends on which bucket or chunk it lands in.
        """
        n = len(pids)
        numel = math.prod(eff)
        eps = torch.empty((n, numel), dtype=torch.float32, device=device)
        signed: list[float] = []
        empty: list[int] = []
        gen = self._gen(eps.device)
        gaussian = self.noise == "gaussian"
        for i in range(n):
            k = ks[i]
            if k <= 0:
                # No live noise yet (a parameter stepping for the first time inside a chunk
                # of older ones): its second moment may still be all-zero, whose shaping is
                # 0/0 = NaN, and NaN * 0 is NaN — so the row is zeroed at the END instead.
                eps[i].zero_()
                signed.append(0.0)
                empty.append(i)
                continue
            draw, sign = self._draw_sign(k)
            gen.manual_seed(noise_seed_for(self.noise_seed, pids[i], draw))
            if gaussian:
                eps[i].normal_(generator=gen)
            else:
                eps[i].bernoulli_(0.5, generator=gen)
            signed.append(sign * scales[i])
        if not gaussian:
            eps.mul_(2.0).sub_(1.0)                     # {0, 1} -> {-1, +1}
        xi = eps.view(n, *eff)
        if self.shape == "v":
            xi.mul_(self._shape_factor(src))
        if rows is not None:
            xi.mul_(rows)
        first = signed[0]
        if all(s == first for s in signed):
            xi.mul_(first)
        else:  # mixed noise indices / radii inside one chunk (late-joining params)
            xi.mul_(torch.tensor(signed, dtype=torch.float32, device=xi.device)
                    .view(n, *([1] * len(eff))))
        for i in empty:
            xi[i].zero_()
        return xi

    def _shape_factor(self, src: tuple[Tensor, ...]) -> Tensor:
        """``S = clamp((mean(v_hat) / v_hat)^(1/4), 1/s_cap, s_cap)``, stacked.

        Factored: ``v_hat_ij = (row_i / mean(row)) * col_j`` and ``mean(v_hat) = mean(col)``,
        so ``S_ij = (row_i/mean(row))^(-1/4) * (col_j/mean(col))^(-1/4)`` — a rank-1 product
        (materialized here only because the clamp is per element). Non-factored:
        ``S_i = (v_i / mean(v))^(-1/4)``; a 0-D parameter (length-1 row) gets exactly 1.

        ``x^(-1/4)`` is spelled ``sqrt(rsqrt(x))``, not ``pow``: on CPU the vectorized and
        the scalar-tail ``pow`` kernels differ by an ulp, so a ``pow`` over a stacked
        ``[N, ...]`` bucket and over one parameter disagree depending on where the SIMD tail
        falls — which would break the foreach == per-param bit-parity. ``sqrt``/``rsqrt``
        are correctly rounded in both.
        """
        lo, hi = 1.0 / self.s_cap, self.s_cap
        if len(src) == 2:
            row, col = src
            a = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_().sqrt_()
            b = col.div(col.mean(dim=-1, keepdim=True)).rsqrt_().sqrt_()
            return (a.unsqueeze(-1) * b.unsqueeze(-2)).clamp_(lo, hi)
        (v,) = src
        return v.div(v.mean(dim=-1, keepdim=True)).rsqrt_().sqrt_().clamp_(lo, hi)

    def _weight_rows(self, p_eff: Tensor, xi_old: Tensor | None, sigma_step: float) -> Tensor:
        """``sigma_ref="weight"`` radius: ``max(k_weight * RMS_row(z), sigma_step)``.

        ``p_eff`` is ``[N, R, C]`` (factored -> ``[N, R, 1]``) or ``[N, L]`` (-> ``[N, 1]``).
        Read from the CLEAN iterate ``z = w - xi_n`` (the live weight minus the noise being
        removed, already in hand) at install time, and frozen in the state.
        """
        z = p_eff if xi_old is None else p_eff.sub(xi_old)
        rows = z.square().mean(dim=-1, keepdim=True).sqrt_().mul_(self.k_weight)
        return rows.clamp_(min=sigma_step)

    def _rows_of(self, states: list[dict[str, Any]], like: tuple[int, ...],
                 device: torch.device) -> Tensor | None:
        if self.sigma_ref != "weight":
            return None
        zero = None
        out = []
        for st in states:
            r = st.get("noise_sigma_rows")
            if r is None:
                if zero is None:
                    zero = torch.zeros(like, dtype=torch.float32, device=device)
                r = zero
            out.append(r)
        return torch.stack(out)

    def _noise_scale(self, sigma_step: float) -> float:
        return 1.0 if self.sigma_ref == "weight" else sigma_step

    @staticmethod
    def _rows_shape(eff: tuple[int, ...]) -> tuple[int, ...]:
        return (eff[0], 1) if len(eff) == 2 else (1,)

    # ------------------------------------------------------------------ the one write
    def _combined_write_one(self, p: Tensor, delta: Tensor, dxi: Tensor, state: dict[str, Any],
                            bf16_method: str, lr: float) -> None:
        """``p <- write(p - lr*delta + dxi)`` as ONE call into Adakaon's per-param writer.

        ``delta' = delta - dxi/lr`` makes the writer's own ``p -= lr*delta'`` the combined
        write, so whatever ``bf16_method`` the writer implements (SR, Kahan, compact Kahan
        ``kahan8``) applies to it unchanged. ``lr == 0`` passes ``-dxi`` with alpha 1.

        Every elementwise op on the stacked/per-param tensors here and in the decay avoids
        ``add_/sub_(..., alpha=)``: on CPU its SIMD body fuses ``a + alpha*b`` while the
        scalar tail rounds twice, so the result would depend on the bucket layout. Plain
        ``div``/``mul``/``sub`` round once everywhere, keeping foreach == per-param exact.
        """
        if lr != 0.0:
            delta.sub_(dxi.div_(lr))
            subtract_one_(p, delta, state, bf16_method, alpha=lr, sr=self.sr_stream)
        else:
            subtract_one_(p, dxi.neg_(), state, bf16_method, sr=self.sr_stream)

    def _combined_write_batched(self, chunk: ForeachChunk, delta: Tensor, dxi: Tensor,
                                bf16_method: str, lr: float) -> None:
        """Foreach twin of :meth:`_combined_write_one` (same identity, same writer family).

        ``chunk.cviews`` (the ``kahan_lo`` views; ``None`` outside ``kahan8``) goes to the
        writer exactly as Adakaon's own bucket write passes it.
        """
        if lr != 0.0:
            delta.sub_(dxi.div_(lr))
            subtract_batched_(chunk.pviews, delta, bf16_method, alpha=lr, sr=self.sr_stream,
                              comp=chunk.cviews)
        else:
            subtract_batched_(chunk.pviews, dxi.neg_(), bf16_method, sr=self.sr_stream,
                              comp=chunk.cviews)

    @staticmethod
    def _commit(states: list[dict[str, Any]], ks: list[int], sigma_step: float,
                rows_new: Tensor | None) -> None:
        for i, st in enumerate(states):
            st["noise_step"] = ks[i] + 1
            st["noise_sigma"] = sigma_step
            if rows_new is not None:
                st["noise_sigma_rows"] = rows_new[i].clone()

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def _step_impl(self, closure: Any = None) -> Any:
        if not self._train_mode:
            raise RuntimeError(
                "Antikaon.step() called outside train mode. Call optimizer.train() before the "
                "training step (and optimizer.eval() before validation / sampling / "
                "checkpointing)."
            )
        loss = super()._step_impl(closure)
        if self._noise_on:
            self._warn_if_inert()
        return loss

    @torch.no_grad()
    def _step_one_param(self, p: Tensor, group: dict[str, Any]) -> None:
        if not self._noise_on or p.numel() == 0:
            return super()._step_one_param(p, group)
        beta1, beta2 = group["betas"]
        eps1, _eps2 = group["eps"]
        lr, clip = group["lr"], group["clip_threshold"]
        wd = group["weight_decay"]
        cautious, bf16_method = group["cautious"], group["bf16_method"]

        state = self.state[p]
        if not state:
            self._init_state(p, state, group)
        pid = self._pids()[id(p)]
        k = state.get("noise_step", 0)

        grad_fp32 = p.grad if p.grad.dtype == torch.float32 else p.grad.float()
        ndim = grad_fp32.ndim
        factored = ndim >= 2
        dev = p.device
        if factored:
            matrixize = ndim > 2
            gv = grad_fp32.reshape(grad_fp32.shape[0], -1) if matrixize else grad_fp32
            eff: tuple[int, ...] = tuple(gv.shape)
            src: tuple[Tensor, ...] = (state["row"].unsqueeze(0), state["col"].unsqueeze(0))
        else:
            eff = (max(grad_fp32.numel(), 1),)
            src = (flat_view(state["v"]).unsqueeze(0),)

        # xi_n, from the second moment BEFORE this step's EMA (the values it was installed with).
        xi_old = None
        if k > 0:
            rows_old = self._rows_of([state], self._rows_shape(eff), dev)
            xi_old = self._noise([pid], [k], [self._noise_scale(state["noise_sigma"])],
                                 eff, src, rows_old, dev).view(p.shape)

        # ---- Adakaon's per-parameter update (identical to Adakaon._step_one_param) ----
        if factored:
            update_factored_state(gv, state["row"], state["col"], beta2, eps1)
            r_factor, c_factor = factored_inv_sqrt_factors(state["row"], state["col"])
            update = gv.mul(r_factor).mul_(c_factor)
            if matrixize:
                update = update.view_as(grad_fp32)
        else:
            v = state["v"]
            grad_sq = grad_fp32 * grad_fp32
            if eps1 > 0:
                grad_sq.add_(eps1)
            v.lerp_(grad_sq, 1.0 - beta2)
            update = grad_fp32.mul(v.rsqrt())
        if clip > 0:
            update.div_((rms(update) / clip).clamp_(min=1.0))
        delta = self._codec(group).ema_one(state, update, beta1) if beta1 > 0 else update

        # Weight decay on the CLEAN iterate z = w - xi_n (design §1.2 / §1.4c), where w is the
        # weight's full value — the compensated one under kahan / kahan8 (_read_clean).
        wd_full = wd != 0 and group["cautious_wd"] == "full"
        p_fp32 = None
        if wd != 0 or self.sigma_ref == "weight":
            p_fp32 = (p.data if p.dtype == torch.float32
                      else self._read_clean(p, state, bf16_method))
        if wd != 0 and not wd_full:
            delta = delta.add_(self._decay_term(p_fp32, xi_old, wd))
        if cautious:
            delta = cautious_one_(delta, grad_fp32)
        if wd_full:
            delta = delta.add_(self._decay_term(p_fp32, xi_old, wd))

        # xi_{n+1}, from the second moment just updated, frozen at the CURRENT lr.
        sigma_step = self._sigma_step(lr, clip)
        rows_new = (self._weight_rows(
            p_fp32.reshape(1, *eff),
            None if xi_old is None else xi_old.reshape(1, *eff), sigma_step)
            if self.sigma_ref == "weight" else None)
        dxi = self._noise([pid], [k + 1], [self._noise_scale(sigma_step)], eff, src,
                          rows_new, dev).view(p.shape)
        if xi_old is not None:
            dxi = dxi.sub_(xi_old)
        self._combined_write_one(p, delta, dxi, state, bf16_method, lr)
        self._commit([state], [k], sigma_step, rows_new)

    def _chunk_meta(self, chunk: ForeachChunk) -> tuple[list[int], list[int], list[float]]:
        pmap = self._pids()
        pids = [pmap[id(p)] for p in chunk.plist]
        ks = [st.get("noise_step", 0) for st in chunk.states]
        scales = [self._noise_scale(st.get("noise_sigma", 0.0)) for st in chunk.states]
        return pids, ks, scales

    @torch.no_grad()
    def _factored_bucket(
        self,
        chunk: ForeachChunk,
        beta1: float,
        beta2: float,
        eps1: float,
        lr: float,
        clip: float,
        wd: float,
        cautious: bool,
        wd_full: bool,
        bf16_method: str,
        codec: _MomentumCodec,
    ) -> None:
        if not self._noise_on:
            return super()._factored_bucket(chunk, beta1, beta2, eps1, lr, clip, wd, cautious,
                                            wd_full, bf16_method, codec)
        R, C = chunk.eff  # noqa: N806 — matrix dims (stacked tensor is [N, R, C])
        N = chunk.n  # noqa: N806
        rows, cols = chunk.state_views
        grad = chunk.grad_stack()                                         # [N, R, C]
        row = torch.stack(rows)                                           # [N, R]
        col = torch.stack(cols)                                           # [N, C]
        dev = grad.device
        eff = (R, C)
        pids, ks, scales = self._chunk_meta(chunk)
        xi_old = None
        if any(ks):
            xi_old = self._noise(pids, ks, scales, eff, (row, col),
                                 self._rows_of(chunk.states, (R, 1), dev), dev)

        # ---- Adakaon's factored bucket at its defaults (eps1 on the means, norm clip) ----
        omb = 1.0 - beta2
        grad_sq = grad * grad
        row_mean, col_mean = grad_sq.mean(dim=-1), grad_sq.mean(dim=-2)
        if eps1 > 0:
            row_mean = row_mean.add_(eps1)
            col_mean = col_mean.add_(eps1)
        row.lerp_(row_mean, omb)
        col.lerp_(col_mean, omb)
        torch._foreach_copy_(rows, list(row.unbind(0)))
        torch._foreach_copy_(cols, list(col.unbind(0)))
        r = row.div(row.mean(dim=-1, keepdim=True)).rsqrt_()                       # [N, R]
        c = col.rsqrt()                                                            # [N, C]
        update = grad.mul(r.unsqueeze(-1)).mul_(c.unsqueeze(-2))                   # [N, R, C]
        rms_ = update.reshape(N, -1).norm(2, dim=1) / math.sqrt(R * C)
        update.div_(rms_.div_(clip).clamp_(min=1.0).view(N, 1, 1))
        if beta1 > 0:
            delta = codec.ema_stacked(chunk.states, update, chunk.view, (R, C), beta1,
                                      views=chunk.momentum_views(codec))
        else:
            delta = update

        w_full = (self._clean_stack(chunk, bf16_method)
                  if wd != 0 or self.sigma_ref == "weight" else None)
        delta = self._decay_on_z(w_full, delta, grad, xi_old, wd, cautious, wd_full)

        sigma_step = self._sigma_step(lr, clip)
        rows_new = (self._weight_rows(w_full, xi_old, sigma_step)
                    if self.sigma_ref == "weight" else None)
        dxi = self._noise(pids, [k + 1 for k in ks], [self._noise_scale(sigma_step)] * N, eff,
                          (row, col), rows_new, dev)
        if xi_old is not None:
            dxi = dxi.sub_(xi_old)
        self._combined_write_batched(chunk, delta, dxi, bf16_method, lr)
        self._commit(chunk.states, ks, sigma_step, rows_new)

    @torch.no_grad()
    def _nonfactored_bucket(
        self,
        chunk: ForeachChunk,
        beta1: float,
        beta2: float,
        eps1: float,
        lr: float,
        clip: float,
        wd: float,
        cautious: bool,
        wd_full: bool,
        bf16_method: str,
        codec: _MomentumCodec,
    ) -> None:
        if not self._noise_on:
            return super()._nonfactored_bucket(chunk, beta1, beta2, eps1, lr, clip, wd,
                                               cautious, wd_full, bf16_method, codec)
        N = chunk.n  # noqa: N806
        length = chunk.length
        (vs,) = chunk.state_views
        grad = chunk.grad_stack()                                         # [N, L]
        v = torch.stack(vs)                                               # [N, L]
        dev = grad.device
        eff = (length,)
        pids, ks, scales = self._chunk_meta(chunk)
        xi_old = None
        if any(ks):
            xi_old = self._noise(pids, ks, scales, eff, (v,),
                                 self._rows_of(chunk.states, (1,), dev), dev)

        # ---- Adakaon's non-factored bucket ----
        omb = 1.0 - beta2
        grad_sq = grad * grad
        if eps1 > 0:
            grad_sq = grad_sq.add_(eps1)
        v.lerp_(grad_sq, omb)
        torch._foreach_copy_(vs, list(v.unbind(0)))
        update = grad.mul(v.rsqrt())                                      # [N, L]
        rms_ = update.norm(2, dim=1) / math.sqrt(length)
        update.div_(rms_.div_(clip).clamp_(min=1.0).view(N, 1))
        if beta1 > 0:
            delta = codec.ema_stacked(chunk.states, update, chunk.view, (length,), beta1,
                                      views=chunk.momentum_views(codec))
        else:
            delta = update

        w_full = (self._clean_stack(chunk, bf16_method)
                  if wd != 0 or self.sigma_ref == "weight" else None)
        delta = self._decay_on_z(w_full, delta, grad, xi_old, wd, cautious, wd_full)

        sigma_step = self._sigma_step(lr, clip)
        rows_new = (self._weight_rows(w_full, xi_old, sigma_step)
                    if self.sigma_ref == "weight" else None)
        dxi = self._noise(pids, [k + 1 for k in ks], [self._noise_scale(sigma_step)] * N, eff,
                          (v,), rows_new, dev)
        if xi_old is not None:
            dxi = dxi.sub_(xi_old)
        self._combined_write_batched(chunk, delta, dxi, bf16_method, lr)
        self._commit(chunk.states, ks, sigma_step, rows_new)

    @staticmethod
    def _clean_stack(chunk: ForeachChunk, bf16_method: str) -> Tensor:
        """The bucket's full-precision weights ``[N, *eff]`` fp32 — :meth:`_read_clean`,
        stacked. ``kahan8`` decodes ``(weights, kahan_lo)`` (exact and integer-only, so the
        stacked and the per-param decode agree bit for bit); legacy ``kahan`` never reaches
        the foreach path; SR / none / fp32 are the stored values."""
        if (chunk.cviews is not None and is_compact_kahan(bf16_method)
                and chunk.pviews[0].dtype == torch.bfloat16):
            return decode(torch.stack(chunk.pviews), torch.stack(chunk.cviews),
                          residual_bits(bf16_method))
        return chunk.param_stack()

    @staticmethod
    def _decay_on_z(w_full: Tensor | None, delta: Tensor, grad: Tensor, xi_old: Tensor | None,
                    wd: float, cautious: bool, wd_full: bool) -> Tensor:
        """Stacked ``delta += wd * (w - xi_n)`` around the cautious mask (Adakaon's order).
        ``w_full`` is :meth:`_clean_stack` (read only when ``wd != 0``)."""
        if wd != 0 and not wd_full:
            delta = delta.add_(Antikaon._decay_term(w_full, xi_old, wd))
        if cautious:
            delta = cautious_batched_(delta, grad)
        if wd_full:
            delta = delta.add_(Antikaon._decay_term(w_full, xi_old, wd))
        return delta

    @staticmethod
    def _decay_term(w: Tensor, xi_old: Tensor | None, wd: float) -> Tensor:
        """``wd * (w - xi_n)`` as a fresh tensor (``w`` may alias the live fp32 weight)."""
        z = w.mul(wd) if xi_old is None else w.sub(xi_old).mul_(wd)
        return z

    # ------------------------------------------------------------------ live noise / modes
    @torch.no_grad()
    def live_noise(self, p: Tensor) -> Tensor | None:
        """The perturbation ``xi`` this optimizer has installed on ``p`` (fp32, ``p.shape``).

        ``None`` before ``p``'s first step. Recomputed from the seed and the state (nothing
        is stored), so it is valid in either mode: in train mode the live weight is
        ``~ z + xi``, in eval mode it is ``~ z``. Diagnostic use: ``RMS(xi)/RMS(w)``.
        """
        st = self.state.get(p)
        if not st or st.get("noise_step", 0) <= 0:
            return None
        if p.ndim >= 2:
            eff: tuple[int, ...] = (p.shape[0], p.numel() // max(p.shape[0], 1))
            src: tuple[Tensor, ...] = (st["row"].unsqueeze(0), st["col"].unsqueeze(0))
        else:
            eff = (max(p.numel(), 1),)
            src = (flat_view(st["v"]).unsqueeze(0),)
        rows = self._rows_of([st], self._rows_shape(eff), p.device)
        xi = self._noise([self._pids()[id(p)]], [st["noise_step"]],
                         [self._noise_scale(st["noise_sigma"])], eff, src, rows, p.device)
        return xi.view(p.shape)

    # --- the clean-value accessors: the ONE place that knows how a writer stores a weight ---
    @staticmethod
    def _read_clean(p: Tensor, state: dict[str, Any], bf16_method: str) -> Tensor:
        """The weight's full-precision value, fp32, as a fresh tensor.

        Plain bf16/fp16 (SR, none): the stored value. ``kahan``: ``p + shift`` — the Kahan
        writer keeps the bits a narrowing write dropped in ``state["shift"]``, and they are
        part of the weight. ``kahan8``: the compensated value decoded from
        ``(p, state["kahan_lo"])`` (:func:`kaon._compact_kahan.decode`; bf16 weights, a
        missing residual reads as zero).
        """
        if p.dtype == torch.float32:
            return p.data.clone()
        if p.dtype == torch.bfloat16 and is_compact_kahan(bf16_method) and RESIDUAL_KEY in state:
            return decode(p.data, state[RESIDUAL_KEY], residual_bits(bf16_method))
        x = p.data.float()
        if bf16_method == "kahan" and "shift" in state:
            x.add_(state["shift"].float())
        return x

    @staticmethod
    def _write_clean(p: Tensor, state: dict[str, Any], bf16_method: str, x: Tensor,
                     sr: SRStream | None = None) -> None:
        """Store the fp32 value ``x`` (consumed) into the weight, keeping the residual
        wherever the writer has somewhere to keep it.

        SR / none: round-to-nearest (there is nowhere to keep the rest). ``kahan``: RTN plus
        ``shift = x - RTN(x)``. ``kahan8``: :func:`kaon._compact_kahan.encode_` with the
        residual's stochastic rounding — the ``kahan8`` writer's codec and noise law, drawn
        from ``sr`` (the owner's checkpointed stream, so a resume reproduces it); the stored
        bf16 is the nearest one to the kept value.
        """
        if p.dtype == torch.float32:
            p.data.copy_(x)
            return
        if p.dtype == torch.bfloat16 and is_compact_kahan(bf16_method):
            if RESIDUAL_KEY not in state:
                ensure_residuals([p], [state])   # method switched on mid-run
            bits = residual_bits(bf16_method)
            gen = _device_generator(x.device) if sr is None else sr.generator(x.device)
            noise = torch.randint(0, 1 << (16 - bits), x.shape, dtype=torch.int32,
                                  device=x.device, generator=gen)
            encode_(x.contiguous(), p.data, state[RESIDUAL_KEY], bits, noise)
            return
        p.data.copy_(x)
        if bf16_method == "kahan" and "shift" in state:
            state["shift"].copy_(x.sub_(p.data.float()))

    @torch.no_grad()
    def _shift_weights(self, sign: float) -> None:
        """``w <- w + sign * xi`` for every parameter carrying noise, on the CLEAN value.

        Round-to-nearest, like MSAM's climb round trip: the eval/train pair accumulates
        nothing, so SR would only add an independent random walk (design §3.1: error <= 1/2
        ulp(z), a <= 1 ulp return only where the subtraction crossed a binade). Under
        ``bf16_method="kahan"`` the pair goes through the compensated value ``p + shift``
        and leaves the RTN residual in ``shift``; under ``"kahan8"`` it goes through the
        decoded value and re-encodes it with the residual's stochastic rounding (unbiased,
        one ``ulp/256`` grid unit per write at most — what the MSAM/Nekaon kahan8 climb does).
        Either way (:meth:`_read_clean` / :meth:`_write_clean`) repeated eval/train cycles
        do not move the clean weight beyond the compensated grid.
        """
        for group in self.param_groups:
            method = group["bf16_method"]
            for p in group["params"]:
                xi = self.live_noise(p)
                if xi is None:
                    continue
                st = self.state[p]
                if p.dtype == torch.float32:
                    p.data.add_(xi, alpha=sign)
                    continue
                x = self._read_clean(p, st, method).add_(xi, alpha=sign)
                self._write_clean(p, st, method, x, self.sr_stream)

    @torch.no_grad()
    def eval(self) -> None:  # noqa: A003 — mirrors the optimizer.eval() API (MSAM/Lookahead/SF)
        """Remove the perturbation: the live weights become (the bf16 rounding of) ``z``."""
        if self._train_mode:
            self._shift_weights(-1.0)
        self._train_mode = False

    @torch.no_grad()
    def train(self) -> None:
        """Re-install the perturbation (recomputed from seed + state, so it is the same ``xi``)."""
        if not self._train_mode:
            self._shift_weights(+1.0)
        self._train_mode = True

    # ------------------------------------------------------------------ inert warning
    @torch.no_grad()
    def _warn_if_inert(self) -> None:
        """Warn once when the sampled perturbation radius is too small to do anything.

        MSAM's inert-lookahead heuristic (commit 265989f), same constants and cadence: a
        radius below half the stored weight's resolution at the sampled mean weight scale is
        not representable as designed; a radius under ``_INERT_REL`` of the weights moves the
        gradient by a negligible amount. Periodically sampled, bounded, off the hot path.

        The resolution depends on the writer. Plain bf16 (SR / none): half an ulp. With a
        compensated method the sub-ulp part of ``xi`` is not lost — it lives in the residual
        and the clean iterate stays exact to the residual grid — so the threshold is half of
        ``ulp/256`` (``kahan8``: its 8-bit residual; ``kahan``: its bf16 buffer holds the
        same order). The forward pass still sees the nearest bf16 to ``z + xi``: a sub-ulp
        ``xi`` reaches the gradient only through which coordinates it tips across a rounding
        boundary (on top of ``z``'s own rounding offset), not coordinate by coordinate as
        designed; that is a property of evaluating a bf16 model, not a loss of the
        clean iterate (``docs/antikaon.md``).
        """
        if self._inert_warned or self._inert_checks >= self._INERT_MAX_CHECKS:
            return
        self._inert_checks += 1
        if self._inert_checks % self.inert_check_interval:
            return
        msg = None
        for group in self.param_groups:
            params = [p for p in group["params"] if p.numel()][: self._INERT_SAMPLE]
            if not params:
                continue
            w = float(torch.stack([p.detach().abs().mean().float() for p in params]).mean())
            if w == 0.0:
                continue
            sigma = self._sigma_step(group["lr"], group["clip_threshold"])
            if self.sigma_ref == "weight":
                sigma = max(sigma, self.k_weight * w)
            dtype = params[0].dtype
            method = group["bf16_method"]
            # The resolution the weight's storage actually keeps: the dtype's ulp for plain
            # (SR / none) writes; a compensated method keeps what the ulp drops, so a sub-ulp
            # xi is NOT lost — it accumulates in the residual. kahan8 resolves ulp/2^bits;
            # legacy kahan's bf16 compensation buffer holds a residual <= ulp/2 to 8 bits
            # (<= ulp/2^9) — the same order, so the same ulp/2^8 floor is used for both.
            res_bits = (residual_bits(method) if is_compact_kahan(method)
                        else 8 if method == "kahan" else 0)
            half_res = 0.5 * torch.finfo(dtype).eps * w / (1 << res_bits)
            if dtype != torch.float32 and sigma < half_res:
                if res_bits:
                    how = (f"even bf16_method={method!r}, which keeps sub-ulp movement down to "
                           f"~ulp/{1 << res_bits}, stores it only as rounding")
                else:
                    how = ("with stochastic rounding it is realized only as {0, +-ulp} draws "
                           "(unbiased, but a larger covariance than designed, and a larger SR "
                           "walk of the clean iterate)")
                msg = (
                    f"{type(self).__name__}: the perturbation radius (~{sigma:.2e}) is below "
                    f"half the resolution of the stored weight at the sampled mean weight "
                    f"scale ({half_res:.2e}, {dtype}); {how}. Raise k_sigma, use "
                    f"bf16_method='kahan8', or keep these parameters in fp32."
                )
            elif sigma / w < self._INERT_REL:
                msg = (
                    f"{type(self).__name__}: the perturbation radius is only {sigma / w:.1e} "
                    f"of the sampled weight scale (lr={group['lr']:.2e}); below "
                    f"~{self._INERT_REL:.0e} this heuristic flags a potentially inert "
                    f"perturbation. Raise k_sigma, or set k_sigma=0 to drop the cost."
                )
            if msg is not None:
                break
        if msg is None:
            self._inert_streak = 0
            return
        self._inert_streak += self.inert_check_interval
        if self._inert_streak >= self._INERT_PATIENCE:
            self._inert_warned = True
            warnings.warn(msg, stacklevel=4)

    # ------------------------------------------------------------------ checkpoints
    def _backend_name(self) -> str:
        return _noise_backend(p.device.type for g in self.param_groups for p in g["params"])

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        state_dict["_antikaon_meta"] = {
            "noise_seed": self.noise_seed,
            "noise_backend": self._backend_name(),
            # A train-mode checkpoint holds z + xi; a fresh optimizer cannot know to remove
            # it, so load_state_dict refuses it (the MSAM contract).
            "train_mode": self._train_mode,
        }
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state and re-install the live perturbation (the checkpoint holds ``z``).

        The checkpoint must have been saved in eval mode. Load the optimizer AFTER the model
        weights (never before): this ends by calling :meth:`train`, which adds ``xi_n``
        (regenerated from the restored seed, noise indices, frozen radii and second moment)
        on top of the current weights; a model loaded afterwards would overwrite it.
        """
        copied = dict(state_dict)
        meta = copied.pop("_antikaon_meta", {})
        if meta.get("train_mode", False):
            raise ValueError(
                "Antikaon checkpoint was saved in train mode: the stored weights carry the "
                "perturbation, and a fresh optimizer cannot know to remove it — resuming "
                "would bake one perturbation into the weights per resume. Call "
                "optimizer.eval() before saving the checkpoint."
            )
        backend = meta.get("noise_backend")
        mine = self._backend_name()
        if backend is not None and backend != mine:
            # An eval-mode checkpoint is self-consistent under any stream: train() installs
            # this backend's xi and the next step removes that same xi. Only the continuation
            # of the noise sequence (bit-exact resume) is lost.
            warnings.warn(
                f"Antikaon checkpoint used noise backend {backend!r}, this run uses "
                f"{mine!r}: training continues correctly, but the noise sequence "
                f"differs from the original run.",
                stacklevel=2,
            )
        super().load_state_dict(copied)
        if "noise_seed" in meta:
            seed = int(meta["noise_seed"])
            if seed < 0:
                raise ValueError("Antikaon checkpoint has an invalid noise seed")
            self.noise_seed = seed & _M63
        self._pid_map = None
        self._train_mode = False
        self.train()
