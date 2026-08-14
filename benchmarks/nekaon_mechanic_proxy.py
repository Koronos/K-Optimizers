"""Small paired proxy audit for constant-LR Nekaon vs external Mechanic(Nekaon).

``mechanic_shim`` crosses only the missing ``defaults`` attribute so later lifecycle defects
can be measured.  It is not a supported configuration and must never be reported as quality.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch

from kaon import Nekaon
from kaon._mechanic_addon import MechanicAddon

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = _load("nekaon_mechanic_harness", REPO / "benchmarks/proxy/harness.py")
D = _load("nekaon_mechanic_dataset", REPO / "benchmarks/proxy/dataset.py")


def _make(params, arm: str):
    inner = Nekaon(
        params,
        lr=1.2e-3,
        k=1.5,
        betas=(0.5, 0.999),
        weight_decay=0.1,
        momentum_dtype="bfloat16",
        cautious=True,
    )
    if arm == "constant":
        return inner
    if arm == "mechanic_public":
        return MechanicAddon(inner, guard=True)
    if arm == "mechanic_shim":
        inner.defaults = inner.inner.defaults
        return MechanicAddon(inner, guard=True)
    raise ValueError(arm)


def _view_owner(opt):
    return opt.inner if isinstance(opt, MechanicAddon) else opt


def _evaluate(opt, fn):
    owner = _view_owner(opt)
    owner.eval()
    try:
        return fn()
    finally:
        owner.train()


def run(seed: int, arm: str, steps: int) -> dict:
    torch.manual_seed(seed)
    dataset = D.build_proxy_dataset()
    data = {key: value.to(H.DEV).to(H.DT) for key, value in dataset["DATA"].items()}
    alphas = H.make_alphas()
    net = H.UNet(C=8).to(H.DEV).to(H.DT)
    params = [p for p in net.parameters() if p.requires_grad]
    try:
        opt = _make(params, arm)
    except Exception as exc:  # constructor failure is itself an audited result
        return {"seed": seed, "arm": arm, "status": "construct_failed", "error": repr(exc)}
    generator = torch.Generator(device=H.DEV).manual_seed(seed + 12345)
    initial = _evaluate(
        opt, lambda: H.eval_loss(net, data[64], dataset["TE"], alphas, reps=1)
    )
    position = 0
    peak = initial
    try:
        for step in range(steps):
            resolution = (32, 64, 48, 64)[step % 4]
            indices = [dataset["TR"][(position + j) % len(dataset["TR"])] for j in range(4)]
            position += 4
            opt.zero_grad()
            loss = H.batch_loss(
                net,
                data[resolution],
                torch.tensor(indices, device=H.DEV),
                alphas,
                generator,
            )
            loss.backward()
            opt.step()
            if (step + 1) % max(1, steps // 4) == 0:
                value = _evaluate(
                    opt, lambda: H.eval_loss(net, data[64], dataset["TE"], alphas, reps=1)
                )
                if not math.isfinite(value):
                    raise FloatingPointError("non-finite held-out loss")
                peak = max(peak, value)
        final = _evaluate(
            opt, lambda: H.eval_loss(net, data[64], dataset["TE"], alphas, reps=1)
        )
        result = {
            "seed": seed,
            "arm": arm,
            "status": "ok",
            "initial": initial,
            "final": final,
            "damage": peak / max(initial, 1e-12) - 1.0,
        }
        if isinstance(opt, MechanicAddon):
            result.update(
                scale=opt.last_stats.scale,
                h=opt.last_stats.h,
                guard_events=opt.last_stats.guard_events,
            )
        return result
    except Exception as exc:
        return {"seed": seed, "arm": arm, "status": "run_failed", "error": repr(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=HERE / "nekaon_mechanic_results.json")
    args = parser.parse_args()
    rows = [
        run(seed, arm, args.steps)
        for seed in range(args.seeds)
        for arm in ("constant", "mechanic_public", "mechanic_shim")
    ]
    args.output.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
