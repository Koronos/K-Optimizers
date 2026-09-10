"""CPU mechanism probe, not a diffusion convergence benchmark."""
import argparse
import json
from pathlib import Path

import torch

from kaon import Rakaon


def probe():
    torch.manual_seed(20260910)
    a0 = torch.randn(4, 48, dtype=torch.float64) / 8
    b0 = torch.randn(64, 4, dtype=torch.float64) / 8
    target = torch.randn(64, 48, dtype=torch.float64)
    initial = b0 @ a0
    rows = []
    for method in ("rakaon_isotropic", "undamped_gram_sgd_reference"):
        updates = []
        for scale in (1., .1, 10.):
            a = torch.nn.Parameter(a0 * scale)
            b = torch.nn.Parameter(b0 / scale)
            loss = (b @ a - target).square().mean()
            loss.backward()
            if method == "rakaon_isotropic":
                Rakaon([a, b], lr=.001, shrinkage=1., stochastic_rounding=False).step()
            else:
                # Full-rank factors only. Simultaneous updates from OLD factors.
                da = torch.linalg.solve(b.T @ b, a.grad)
                db = torch.linalg.solve(a @ a.T, b.grad.T).T
                with torch.no_grad():
                    a.add_(da, alpha=-.001)
                    b.add_(db, alpha=-.001)
            update = (b @ a - initial).detach()
            if not updates:
                reference = update
            updates.append(update)
            rows.append(dict(method=method, factor_scale=scale,
                             initial_product_error=float((b0 / scale @ (a0 * scale) - initial).norm()),
                             update_norm=float(update.norm()),
                             relative_update_difference=float((update - reference).norm() / reference.norm())))
    return dict(seed=20260910, shape=[64, 48], rank=4, lr=.001,
                scope="One step on squared matrix error. No diffusion or convergence claim.",
                caveat="Gram reference requires full-rank factors; standard zero-B LoRA initialization makes B.T@B singular. Damping changes invariance and needs separate study.",
                results=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
