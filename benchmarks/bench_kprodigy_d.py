"""Characterize KProdigy's D (effective-LR) trajectory.

This is the *measure-first* tool for deciding whether KProdigy needs a more
robust D-bootstrap. It runs entirely on CPU (the D math is device-independent)
on small synthetic problems, so it is deterministic and cheap.

It answers three questions:

1. **Defaults.** Do the original repo's defaults (``d_update_freq=5``,
   ``use_bias_correction=True``) starve the D-bootstrap vs the sane defaults?
2. **Memory variants.** How does ``second_moment="factored"`` (and bf16/int8
   momentum) shift the D trajectory vs the exact full-fp32 path?
3. **Dataset sensitivity.** Prodigy's pain point: D depends on the gradient
   scale / problem conditioning. We sweep a "difficulty" knob and report how
   much the final D and the steps-to-rise move — the evidence for/against
   investing in a D-robustness fix.

Run:  python benchmarks/bench_kprodigy_d.py
"""

from __future__ import annotations

import torch

from koptim import KProdigy

STEPS = 200


def _problem(scale: float = 1.0, cond: float = 1.0, n: int = 256, inp: int = 32, out: int = 64):
    """A linear regression. ``scale`` scales targets (gradient magnitude);
    ``cond`` stretches the input spectrum (conditioning / 'dataset shape')."""
    torch.manual_seed(0)
    w_true = torch.randn(out, inp)
    torch.manual_seed(7)
    x = torch.randn(n, inp)
    # stretch the input covariance: column j scaled by cond**(j/inp)
    spectrum = torch.tensor([cond ** (j / inp) for j in range(inp)])
    x = x * spectrum
    y = (x @ w_true.T) * scale
    return x, y


def _train(make_opt, x, y, steps=STEPS):
    lin = torch.nn.Linear(x.shape[1], y.shape[1], bias=False)
    opt = make_opt(lin)
    d_hist, loss_hist = [], []
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(lin(x), y)
        loss.backward()
        opt.step()
        d_hist.append(opt.get_d())
        loss_hist.append(loss.item())
    return d_hist, loss_hist


def _steps_to_rise(d_hist, d0, factor=10.0):
    """First step at which D exceeds factor*d0 (None if it never rises)."""
    for i, d in enumerate(d_hist):
        if d > factor * d0:
            return i
    return None


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def main():
    d0 = 1e-6

    section("1) DEFAULTS — does sparse-D / bias correction starve the bootstrap?")
    x, y = _problem()
    configs = {
        "sane (df=1, bias=off)":     dict(d_update_freq=1, use_bias_correction=False),
        "old default (df=5)":        dict(d_update_freq=5, use_bias_correction=False),
        "old default (bias=on)":     dict(d_update_freq=1, use_bias_correction=True),
        "old default (df=5+bias)":   dict(d_update_freq=5, use_bias_correction=True),
    }
    print(f"{'config':28s} {'D@10':>10} {'D@50':>10} {'D@end':>10} {'rise@':>6} {'loss@end':>10}")
    for name, cfg in configs.items():
        dh, lh = _train(lambda m, c=cfg: KProdigy(m.parameters(), lr=1.0, **c), x, y)
        r = _steps_to_rise(dh, d0)
        print(f"{name:28s} {dh[9]:10.2e} {dh[49]:10.2e} {dh[-1]:10.2e} {str(r):>6} {lh[-1]:10.2e}")

    section("2) MEMORY VARIANTS — D trajectory vs the exact full-fp32 path")
    variants = {
        "full + fp32 mom (exact)": dict(second_moment="full", momentum_dtype="float32"),
        "full + bf16 mom":         dict(second_moment="full", momentum_dtype="bfloat16"),
        "full + int8 mom":         dict(second_moment="full", momentum_dtype="int8"),
        "factored + bf16 mom":     dict(second_moment="factored", momentum_dtype="bfloat16"),
    }
    print(f"{'variant':28s} {'D@10':>10} {'D@50':>10} {'D@end':>10} {'loss@end':>10}")
    for name, cfg in variants.items():
        dh, lh = _train(lambda m, c=cfg: KProdigy(m.parameters(), lr=1.0, **c), x, y)
        print(f"{name:28s} {dh[9]:10.2e} {dh[49]:10.2e} {dh[-1]:10.2e} {lh[-1]:10.2e}")

    section("3) DATASET SENSITIVITY — how much does D move with grad scale / conditioning?")
    print("Prodigy's D is an estimate of the distance to the solution; it scales")
    print("with the problem. The question is how WILDLY it moves across setups.\n")
    print(f"{'grad scale':>10} {'cond':>6} {'D@end':>10} {'rise@':>6} {'loss@end':>10}")
    base_d = None
    for scale in (0.1, 1.0, 10.0):
        for cond in (1.0, 30.0):
            x, y = _problem(scale=scale, cond=cond)
            dh, lh = _train(lambda m: KProdigy(m.parameters(), lr=1.0), x, y)
            r = _steps_to_rise(dh, d0)
            if base_d is None:
                base_d = dh[-1]
            print(f"{scale:10.2f} {cond:6.0f} {dh[-1]:10.2e} {str(r):>6} {lh[-1]:10.2e}")
    print("\nSpread of D@end across setups is the 'dataset dependence' Eduardo hit.")
    print("A large spread / late 'rise@' motivates a D-robustness option")
    print("(e.g. better d0/d_coef autoscaling or a short D-warmup).")


if __name__ == "__main__":
    main()
