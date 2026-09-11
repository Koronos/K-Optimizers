"""Measure represented lookahead separately from accumulating base updates."""
import argparse
import json
import warnings
from pathlib import Path

import torch

from kaon import Nekaon


def run():
    rows = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for dtype in (torch.float32, torch.bfloat16):
        for magnitude in (1., .01, .001):
            for lr in (1e-4, 1e-5, 1e-6):
                torch.manual_seed(123)
                p = torch.nn.Parameter(torch.full((16, 32), magnitude, device=device, dtype=dtype))
                initial = p.detach().clone()
                opt = Nekaon([p], lr=lr, k=1.5, betas=(.5, .999), weight_decay=0,
                             momentum_dtype="4bit", cautious=False, gradient_centralization=False,
                             bf16_method="stochastic_rounding")
                with warnings.catch_warnings(record=True):
                    for _ in range(20):
                        p.grad = torch.linspace(-1, 1, p.numel(), device=device).reshape_as(p).to(dtype)
                        opt.step()
                    opt.eval()
                    true_weights = p.detach().clone()
                    opt.train()
                delta = p.detach().float() - true_weights.float()
                rows.append(dict(dtype=str(dtype), magnitude=magnitude, lr=lr,
                                 represented_lookahead_fraction=float((delta != 0).float().mean()),
                                 lookahead_rms=float(delta.square().mean().sqrt()),
                                 base_changed_fraction=float((true_weights != initial).float().mean())))
    return dict(device=device, torch=torch.__version__, steps=20, k=1.5,
                scope="Synthetic constant gradients, not a convergence or Anima gradient-effect measurement.",
                rows=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
