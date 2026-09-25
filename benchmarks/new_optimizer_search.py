"""Cheap, isolated search for a post-Nekaon optimizer geometry.

This script deliberately keeps the candidates out of :mod:`kaon`: the native,
per-parameter reference math is enough to falsify a quality hypothesis before we
pay the complexity cost of foreach/Triton implementations.  The training loop is
the constant-LR arm of the control battery (same dataset, progressive resolution,
evaluation and seeds).

Examples::

    python benchmarks/new_optimizer_search.py --steps 600 --channels 40
    python benchmarks/new_optimizer_search.py --steps 2000 --channels 128 \
        --arms nekaon_b09,tangent_inverse:0.1 --seeds 0,1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path
from typing import Any

import torch

from kaon import MSAM, Adakaon, Nekaon
from kaon._backend import cautious_one_, rms, subtract_one_
from kaon._factored import factored_inv_sqrt_factors, update_factored_state

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


H = _load("new_optimizer_harness", REPO / "benchmarks/proxy/harness.py")
D = _load("new_optimizer_dataset", REPO / "benchmarks/proxy/dataset.py")
DEV = H.DEV


def seq_prog(n: int) -> list[int]:
    """Same 40/40/20 progressive-resolution sequence as the control battery."""
    tail = int(n * 0.2)
    first = (n - tail) // 2
    second = n - tail - first
    a = ([32, 64] * ((first // 2) + 1))[:first]
    b = ([48, 64] * ((second // 2) + 1))[:second]
    random.Random(123).shuffle(a)
    random.Random(124).shuffle(b)
    return a + b + [64] * tail


class GeometryKaon(Adakaon):
    """Adakaon reference core with one zero-state geometric exploration branch.

    ``tangent_inverse`` projects ``g * sqrt(v)`` onto the tangent plane of the
    current minibatch loss and adds it *after* cautious masking of the base step.
    ``tangent_fast`` does the same with the disagreement between the immediate
    adaptive direction and its long momentum EMA.  ``dual`` is the deliberately
    non-tangential inverse-adaptivity control.  ``switch_momentum`` and
    ``tensor_momentum`` retain beta=.9 only where the current adaptive direction
    agrees with its history; ``strength`` is their disagreement beta.

    All modes reuse Adakaon's existing first/factored-second moments.  ``foreach``
    is disabled because this is a mathematical reference, not a speed claim.
    """

    def __init__(
        self,
        params: Any,
        *,
        geometry: str,
        strength: float,
        ceiling_beta: float = 0.9,
        decay_boost: float = 0.0,
        coherence_decay: float = 0.0,
        switch_threshold: float = 0.0,
        lr: float = 1.2e-3,
        weight_decay: float = 0.3,
    ) -> None:
        if geometry not in {
            "tangent_inverse",
            "tangent_fast",
            "dual",
            "switch_momentum",
            "tensor_momentum",
            "ema_momentum",
            "ema_switch",
        }:
            raise ValueError(f"unknown geometry: {geometry}")
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        if not strength <= ceiling_beta < 1.0:
            raise ValueError("ceiling_beta must be in [strength, 1)")
        if decay_boost < 0.0:
            raise ValueError("decay_boost must be non-negative")
        if not 0.0 <= coherence_decay < 1.0:
            raise ValueError("coherence_decay must be in [0, 1)")
        if not 0.0 <= switch_threshold <= 1.0:
            raise ValueError("switch_threshold must be in [0, 1]")
        self.geometry = geometry
        self.strength = float(strength)
        self.decay_boost = float(decay_boost)
        self.coherence_decay = float(coherence_decay)
        self.switch_threshold = float(switch_threshold)
        super().__init__(
            params,
            lr=lr,
            betas=(float(ceiling_beta), 0.999),
            weight_decay=weight_decay,
            cautious=True,
            momentum_dtype="float32",
            foreach=False,
            fused=False,
            auto_lr=False,
        )

    @staticmethod
    def _tangent(source: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """Remove the source component parallel to ``grad`` (per parameter tensor)."""
        denom = grad.square().sum().clamp_min_(torch.finfo(torch.float32).tiny)
        coeff = source.mul(grad).sum().div_(denom)
        return source.sub(grad, alpha=float(coeff))

    @staticmethod
    def _match_rms(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Give ``source`` the RMS of ``reference`` without exploding a null tangent."""
        source_rms = rms(source)
        if not torch.isfinite(source_rms) or float(source_rms) <= 1e-20:
            return torch.zeros_like(source)
        return source.mul(rms(reference).div(source_rms))

    @torch.no_grad()
    def _step_one_param(self, p: torch.Tensor, group: dict[str, Any]) -> None:
        # strength=0 is an exact control for accidental pipeline differences.
        if self.strength == 0.0 and self.geometry in {
            "tangent_inverse", "tangent_fast", "dual"
        }:
            return super()._step_one_param(p, group)

        beta1, beta2 = group["betas"]
        eps1, _eps2 = group["eps"]
        lr, clip = group["lr"], group["clip_threshold"]
        wd = group["weight_decay"]

        state = self.state[p]
        if not state:
            self._init_state(p, state, group)

        grad = p.grad if p.grad.dtype == torch.float32 else p.grad.float()
        if grad.ndim >= 2:
            matrixized = grad.ndim > 2
            gv = grad.reshape(grad.shape[0], -1) if matrixized else grad
            update_factored_state(gv, state["row"], state["col"], beta2, eps1)
            r_factor, c_factor = factored_inv_sqrt_factors(state["row"], state["col"])
            adaptive = gv.mul(r_factor).mul_(c_factor)
            inverse = gv.mul(r_factor.reciprocal()).mul_(c_factor.reciprocal())
            if matrixized:
                adaptive = adaptive.view_as(grad)
                inverse = inverse.view_as(grad)
        else:
            v = state["v"]
            grad_sq = grad.square()
            if eps1 > 0:
                grad_sq.add_(eps1)
            v.lerp_(grad_sq, 1.0 - beta2)
            adaptive = grad.mul(v.rsqrt())
            inverse = grad.mul(v.sqrt())

        if clip > 0:
            adaptive.div_((rms(adaptive) / clip).clamp_(min=1.0))

        if self.geometry == "dual":
            # A schedule-free, fixed DualAdam-style control.  Matching RMS keeps
            # the comparison about geometry instead of raw step magnitude.
            inverse = self._match_rms(inverse, adaptive)
            adaptive.lerp_(inverse, self.strength)
            adaptive.div_((rms(adaptive) / clip).clamp_(min=1.0))

        immediate = adaptive.mul(lr)
        coherence_value = 1.0
        if self.geometry == "switch_momentum":
            momentum = state["m"]
            beta = torch.where(
                momentum.mul(immediate) > 0,
                torch.full_like(momentum, beta1),
                torch.full_like(momentum, self.strength),
            )
            momentum.mul_(beta).add_(immediate.mul(1.0 - beta))
            delta = momentum.clone()
        elif self.geometry in {"tensor_momentum", "ema_momentum", "ema_switch"}:
            momentum = state["m"]
            denom = momentum.square().sum().sqrt().mul(immediate.square().sum().sqrt()).add_(1e-12)
            coherence = momentum.mul(immediate).sum().div_(denom).clamp_(0.0, 1.0)
            coherence_value = float(coherence)
            if self.geometry in {"ema_momentum", "ema_switch"}:
                previous = float(state.get("coherence_ema", 1.0))
                coherence_value = (
                    self.coherence_decay * previous
                    + (1.0 - self.coherence_decay) * coherence_value
                )
                state["coherence_ema"] = coherence_value
            effective_beta = self.strength + (beta1 - self.strength) * float(coherence)
            if self.geometry == "ema_momentum":
                effective_beta = self.strength + (beta1 - self.strength) * coherence_value
            elif self.geometry == "ema_switch":
                switched = bool(state.get("coherence_switched", False))
                switched = switched or coherence_value < self.switch_threshold
                state["coherence_switched"] = switched
                effective_beta = self.strength if switched else beta1
            momentum.lerp_(immediate, 1.0 - effective_beta)
            delta = momentum.clone()
        else:
            delta = self._codec(group).ema_one(state, immediate, beta1)

        # Preserve Adakaon's convergence branch exactly: WD and cautious act on
        # the base delta.  The tangent branch is added afterwards because cautious
        # would otherwise erase precisely the sign-disagreeing tangent coordinates.
        if wd != 0:
            p_fp32 = p.data if p.dtype == torch.float32 else p.data.float()
            effective_wd = wd * (1.0 + self.decay_boost * (1.0 - coherence_value))
            delta.add_(p_fp32, alpha=lr * effective_wd)
        delta = cautious_one_(delta, grad)

        if self.geometry == "tangent_inverse":
            explorer = self._tangent(inverse, grad)
            explorer = self._match_rms(explorer, delta).mul_(self.strength)
            delta.add_(explorer)
        elif self.geometry == "tangent_fast":
            # Compare the instantaneous preconditioned step with the long EMA.
            # Use the momentum-only delta (before WD/cautious) as the reference;
            # reconstructing it is cheap and avoids another persistent buffer.
            momentum = self._codec(group).dequant_one(state, immediate)
            explorer = self._tangent(immediate.sub(momentum), grad)
            explorer = self._match_rms(explorer, delta).mul_(self.strength)
            delta.add_(explorer)

        subtract_one_(p, delta, state, group["bf16_method"])


class LookKaon(torch.optim.Optimizer):
    """LookSAM's periodic *true* sharpness direction on an Adakaon core.

    Every ``k`` steps the caller supplies a second gradient at the SAM-perturbed
    weights.  The component of that gradient orthogonal to the clean gradient is
    cached and reused between probes, following LookSAM.  Unlike the historical
    Nekaon experiments this is not a momentum proxy: the correction comes from an
    actual same-batch second backward.

    The reference stores ``gv`` in fp32.  A production Kaon version would pass it
    through the existing 4-bit codec; this class exists only to answer whether the
    quality improvement is present before optimizing its representation.
    """

    def __init__(
        self,
        params: Any,
        *,
        lr: float,
        rho: float,
        k: int,
        alpha: float,
    ) -> None:
        params = list(params)
        if rho <= 0.0:
            raise ValueError("rho must be positive")
        if k < 1:
            raise ValueError("k must be >= 1")
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        super().__init__(params, {"lr": lr, "rho": rho})
        self.inner = Adakaon(
            self.param_groups,
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.3,
            cautious=True,
            momentum_dtype="float32",
            foreach=False,
        )
        self.param_groups = self.inner.param_groups
        self.rho = float(rho)
        self.k = int(k)
        self.alpha = float(alpha)
        self._step = 0

    @property
    def needs_refresh(self) -> bool:
        return self._step % self.k == 0

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.inner.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def first_step(self) -> None:
        if not self.needs_refresh:
            raise RuntimeError("first_step called outside a refresh step")
        sq_sum = None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                term = p.grad.square().sum()
                sq_sum = term if sq_sum is None else sq_sum + term
        if sq_sum is None:
            return
        scale = self.rho / (sq_sum.sqrt() + 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                state["old_p"] = p.detach().clone()
                state["clean_grad"] = p.grad.detach().clone()
                p.add_(p.grad, alpha=float(scale))
        self.zero_grad()

    @torch.no_grad()
    def second_step(self) -> None:
        if not self.needs_refresh:
            raise RuntimeError("second_step called outside a refresh step")
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                old_p = state.pop("old_p", None)
                clean = state.pop("clean_grad", None)
                if old_p is None or clean is None or p.grad is None:
                    continue
                perturbed = p.grad
                clean_norm = clean.square().sum().sqrt()
                perturbed_norm = perturbed.square().sum().sqrt()
                denom = clean_norm.mul(perturbed_norm).add_(1e-12)
                cosine = clean.mul(perturbed).sum().div_(denom)
                # Component of the true SAM gradient orthogonal to the clean
                # gradient.  Per-tensor projection matches LookSAM's published
                # implementation and avoids a cross-device/global state buffer.
                unit_clean = clean.div(clean_norm.add(1e-12))
                gv = perturbed.sub(unit_clean.mul(perturbed_norm * cosine))
                state["gv"] = gv.detach().clone()
                p.copy_(old_p)
        self.inner.step()
        self._step += 1

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        if closure is not None:
            raise RuntimeError("the research harness drives LookKaon's two passes explicitly")
        if self.needs_refresh:
            raise RuntimeError("refresh step requires first_step/second_step")
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                gv = self.state[p].get("gv")
                if gv is None:
                    continue
                grad_norm = p.grad.square().sum().sqrt()
                gv_norm = gv.square().sum().sqrt()
                if float(gv_norm) > 1e-20:
                    p.grad.add_(gv, alpha=float(self.alpha * grad_norm / gv_norm))
        out = self.inner.step()
        self._step += 1
        return out


def make_optimizer(arm: str, params: list[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    if arm == "adakaon_b09":
        return Adakaon(
            params,
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.3,
            cautious=True,
            momentum_dtype="float32",
            foreach=False,
        )
    if arm == "nekaon_b09":
        return Nekaon(
            params,
            lr=lr,
            k=1.5,
            betas=(0.9, 0.999),
            weight_decay=0.3,
            momentum_dtype="4bit",
        )
    if arm == "nekaon_b07":
        return Nekaon(
            params,
            lr=lr,
            k=1.5,
            betas=(0.7, 0.999),
            weight_decay=0.3,
            momentum_dtype="4bit",
        )
    if arm == "nekaon_b05":
        return Nekaon(
            params,
            lr=lr,
            k=1.5,
            betas=(0.5, 0.999),
            weight_decay=0.3,
            momentum_dtype="4bit",
        )
    if arm.startswith("looksam:"):
        _name, k, rho, alpha = arm.split(":")
        return LookKaon(params, lr=lr, k=int(k), rho=float(rho), alpha=float(alpha))
    if arm.startswith("coherent_momentum:"):
        _name, floor, ceiling = arm.split(":")
        return GeometryKaon(
            params,
            geometry="tensor_momentum",
            strength=float(floor),
            ceiling_beta=float(ceiling),
            lr=lr,
        )
    if arm.startswith("coherent_decay:"):
        _name, floor, ceiling, weight_decay, boost = arm.split(":")
        return GeometryKaon(
            params,
            geometry="tensor_momentum",
            strength=float(floor),
            ceiling_beta=float(ceiling),
            weight_decay=float(weight_decay),
            decay_boost=float(boost),
            lr=lr,
        )
    if arm.startswith("ema_momentum:"):
        _name, floor, ceiling, coherence_decay = arm.split(":")
        return GeometryKaon(
            params,
            geometry="ema_momentum",
            strength=float(floor),
            ceiling_beta=float(ceiling),
            coherence_decay=float(coherence_decay),
            lr=lr,
        )
    if arm.startswith("ema_switch:"):
        _name, floor, ceiling, coherence_decay, threshold = arm.split(":")
        return GeometryKaon(
            params,
            geometry="ema_switch",
            strength=float(floor),
            ceiling_beta=float(ceiling),
            coherence_decay=float(coherence_decay),
            switch_threshold=float(threshold),
            lr=lr,
        )
    if arm.startswith("coherent_lookahead:"):
        _name, floor, ceiling, k = arm.split(":")

        def base_factory(inner_params: Any, **_kwargs: Any) -> GeometryKaon:
            return GeometryKaon(
                inner_params,
                geometry="tensor_momentum",
                strength=float(floor),
                ceiling_beta=float(ceiling),
                lr=lr,
            )

        return MSAM(
            params,
            base_optimizer=base_factory,
            rho=-float(k),
            norm="none",
        )
    geometry, strength = arm.split(":", 1)
    return GeometryKaon(params, geometry=geometry, strength=float(strength), lr=lr)


def evald(opt: torch.optim.Optimizer, fn: Any) -> Any:
    swap = hasattr(opt, "eval") and hasattr(opt, "train")
    if swap:
        opt.eval()
    out = fn()
    if swap:
        opt.train()
    return out


def run(
    arm: str,
    *,
    seed: int,
    channels: int,
    steps: int,
    batch_size: int,
    lr: float,
    data: dict[int, torch.Tensor],
    tr: list[int],
    te: list[int],
    ac: torch.Tensor,
) -> dict[str, float]:
    torch.manual_seed(seed)
    if DEV == "cuda":
        torch.cuda.manual_seed_all(seed)
    net = H.UNet(C=channels).to(DEV).to(H.DT)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = make_optimizer(arm, params, lr)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(seed + 12345)
    sequence = seq_prog(steps)
    pos = 0
    started = time.perf_counter()
    for resolution in sequence:
        idx = [tr[(pos + j) % len(tr)] for j in range(batch_size)]
        pos += batch_size
        opt.zero_grad()
        generator_state = gen.get_state()
        loss = H.batch_loss(
            net,
            data[resolution],
            torch.tensor(idx, device=DEV),
            ac,
            gen,
        )
        loss.backward()
        if isinstance(opt, LookKaon) and opt.needs_refresh:
            opt.first_step()
            # SAM requires exactly the same minibatch, timestep and diffusion
            # noise at the perturbed weights; rewind only this local generator.
            gen.set_state(generator_state)
            perturbed_loss = H.batch_loss(
                net,
                data[resolution],
                torch.tensor(idx, device=DEV),
                ac,
                gen,
            )
            perturbed_loss.backward()
            opt.second_step()
        else:
            opt.step()
    if DEV == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    train_loss, test_loss = evald(
        opt,
        lambda: (
            H.eval_loss(net, data[64], tr, ac),
            H.eval_loss(net, data[64], te, ac),
        ),
    )
    bpp = H.opt_state_bytes_per_param(opt, params)
    result = {
        "train": train_loss,
        "test": test_loss,
        "gap": test_loss - train_loss,
        "ms_step_reference": elapsed * 1000.0 / steps,
        "state_bpp": bpp,
    }
    coherence_values = [
        float(state["coherence_ema"])
        for state in opt.state.values()
        if "coherence_ema" in state
    ]
    if coherence_values:
        ordered = sorted(coherence_values)
        result.update(
            coherence_min=ordered[0],
            coherence_median=ordered[len(ordered) // 2],
            coherence_mean=sum(ordered) / len(ordered),
            coherence_max=ordered[-1],
        )
        result["coherence_switched_fraction"] = sum(
            bool(state.get("coherence_switched", False))
            for state in opt.state.values()
            if "coherence_ema" in state
        ) / len(coherence_values)
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--channels", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.2e-3)
    parser.add_argument("--seeds", default="0")
    parser.add_argument(
        "--arms",
        default=(
            "adakaon_b09,nekaon_b09,nekaon_b07,nekaon_b05,"
            "dual:0.1,dual:0.25,tangent_inverse:0.05,tangent_inverse:0.1,"
            "tangent_inverse:0.2,tangent_fast:0.05,tangent_fast:0.1,tangent_fast:0.2"
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(seed) for seed in args.seeds.split(",")]
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    dataset = D.build_proxy_dataset()
    data = {key: value.to(DEV).to(H.DT) for key, value in dataset["DATA"].items()}
    tr, te = dataset["TR"], dataset["TE"]
    ac = H.make_alphas()
    print(
        f"device={DEV} C={args.channels} N={args.steps} seeds={seeds} "
        f"lr={args.lr:g} arms={len(arms)}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    for arm in arms:
        for seed in seeds:
            metrics = run(
                arm,
                seed=seed,
                channels=args.channels,
                steps=args.steps,
                batch_size=args.batch_size,
                lr=args.lr,
                data=data,
                tr=tr,
                te=te,
                ac=ac,
            )
            row = {"arm": arm, "seed": seed, **metrics}
            rows.append(row)
            print(
                f"{arm:24s} seed={seed} test={metrics['test']:.5f} "
                f"gap={metrics['gap']:+.5f} train={metrics['train']:.5f} "
                f"ref_ms={metrics['ms_step_reference']:.2f} bpp={metrics['state_bpp']:.3f}",
                flush=True,
            )
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
