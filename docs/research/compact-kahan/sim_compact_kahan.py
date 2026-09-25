"""Compact-Kahan drift simulation (same harness as docs/research/antikaon-sim/sim_drift.py).

Live bf16 weights ``w`` track an fp32 iterate ``z`` under a prescribed update stream, with an
optional Antikaon-style perturbation carried IN the weights (``w_n = z_n + xi_n``, ONE write
per step: ``w <- round(w - delta_n + xi_{n+1} - xi_n)``). We recover ``z_rec = w_n - xi_n``
and compare with an fp32 reference on the same stream. No feedback (worst case).

Schemes (the weight write):
  fp32        exact reference
  sr          bf16 stochastic rounding, no state            (kaon default)
  kahan       bf16 compensation buffer, 2 B/param           (kaon bf16_method="kahan", legacy)
  kahan8-sr   bf16 + 8-bit fixed-point residual, SR on the residual grid   (1 B/param)  <- candidate
  kahan8-rn   same residual, round-half-away                               (1 B/param)
  kahan4-sr   bf16 + 4-bit residual, SR                                    (0.5 B/param)
  kahan4-rn   same, round-half-away
  kahan16     bf16 + 16-bit residual == an fp32 master weight split in two (2 B/param, exact)

Update stream: ``delta = lr * (0.1*c + noise_scale*0.99*nu)``, ``c`` in {-1,+1} redrawn every
50 steps, ``nu ~ N(0,1)`` (RMS ~ 1 like Adakaon's normalized update). ``noise_scale=0`` is the
pure-drift regime that exposes round-to-nearest stalls. Noise: ``xi_n = k*lr*s_n`` Rademacher,
regenerated from (seed, step) -> zero state.

Metrics (errors of ``z_rec - z_ref`` in units of ``ulp_ref = ulp_bf16(RMS(z_ref))``):
  bias +- se, std, max, rel L2 vs |z|, err/moved, and ``lost`` = fraction of the net movement
  that was NOT realized (projection of the error on the movement direction; 1.0 = fully stalled).
  ``err`` is measured on the value the scheme TRACKS (bf16 weight + its compensation, decoded);
  ``fwd_*`` on the bare bf16 weight the forward pass sees, which for every compensated scheme
  additionally carries the <= 1/2 ulp residual by construction. (The antikaon sim measured only
  the latter, so its "kahan+xi 0.49 ulp" is mostly that residual, not drift.)
"""
from __future__ import annotations

import json
import math
import sys
import time

import torch

import kaon  # noqa: F401
from kaon._compact_kahan import compensated_add_, decode
from kaon._stochastic_rounding import add_stochastic_

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D = 1 << 20 if dev.type == "cuda" else 1 << 16
N = 10_000 if dev.type == "cuda" else 2_000
print("kaon:", kaon.__file__, "| device:", dev, "| D:", D, "| N:", N, flush=True)

BITS = {"kahan8-sr": 8, "kahan8-rn": 8, "kahan4-sr": 4, "kahan4-rn": 4, "kahan16": 16}


def ulp_bf16(x: torch.Tensor) -> torch.Tensor:
    e = torch.floor(torch.log2(x.abs().clamp_min(1e-30)))
    return torch.exp2(e - 7)


def noise(seed: int, step: int) -> torch.Tensor:
    g = torch.Generator(device=dev)
    g.manual_seed((seed * 1_000_003 + step) & 0x7FFFFFFF)
    return torch.randint(0, 2, (D,), generator=g, device=dev, dtype=torch.float32).mul_(2).sub_(1)


def run(scheme: str, lr: float, k: float, noise_scale: float = 1.0, seed: int = 1234) -> dict:
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    w0 = torch.randn(D, generator=gen, device=dev) * 0.05
    c = torch.randint(0, 2, (D,), generator=gen, device=dev, dtype=torch.float32).mul_(2).sub_(1)
    z_ref = w0.clone()
    sigma = k * lr
    xi = sigma * noise(seed, 0) if k > 0 else torch.zeros(D, device=dev)
    w = (w0 + xi).clone() if scheme == "fp32" else (w0 + xi).to(torch.bfloat16)
    shift = torch.zeros(D, device=dev, dtype=torch.bfloat16) if scheme == "kahan" else None
    bits = BITS.get(scheme)
    lo = None
    if bits is not None:
        lo = torch.zeros(D, dtype=torch.uint8 if bits <= 8 else torch.int16, device=dev)
        # the initial bf16 cast of w0+xi loses bits; the residual starts at 0 like a real
        # optimizer's (the model is handed over as bf16), so the reference starts there too
    if scheme != "fp32":
        z_ref = w.float() - xi                   # every scheme starts from the same bf16 weights
    stochastic = scheme.endswith("-sr")

    for n in range(N):
        if n % 50 == 0:
            c = torch.randint(0, 2, (D,), generator=gen, device=dev, dtype=torch.float32).mul_(2).sub_(1)
        nu = torch.randn(D, generator=gen, device=dev)
        delta = lr * (0.1 * c + noise_scale * 0.99 * nu)
        z_ref.sub_(delta)
        xi_next = sigma * noise(seed, n + 1) if k > 0 else xi
        upd = (xi_next - xi) - delta
        if scheme == "fp32":
            w.add_(upd)
        elif scheme == "sr":
            add_stochastic_(w, upd, alpha=1.0)
        elif scheme == "kahan":
            shift.add_(upd.to(torch.bfloat16))
            p_before = w.clone()
            w.add_(shift)
            shift.add_(p_before.sub_(w))
        elif bits is not None:
            compensated_add_(w, lo, upd, 1.0, bits, None, stochastic)
        else:
            raise ValueError(scheme)
        xi = xi_next

    # What the optimizer TRACKS (the compensated value) vs what the forward pass SEES (bf16 w).
    if bits is not None:
        z_full = decode(w, lo, bits)
    elif scheme == "kahan":
        z_full = w.float() + shift.float()
    else:
        z_full = w.float()
    z_rec = z_full - xi
    err = z_rec - z_ref
    u_ref = float(ulp_bf16(z_ref.pow(2).mean().sqrt()))
    err_ulp = err / u_ref
    fwd_err_ulp = (w.float() - xi - z_ref) / u_ref
    move = z_ref - w0
    moved = move.norm()
    return dict(
        scheme=scheme, lr=lr, k=k, noise_scale=noise_scale, bits=bits,
        ulp_ref=u_ref, step_over_ulp=lr / u_ref, sigma_over_ulp=sigma / u_ref if k > 0 else 0.0,
        rel_l2_vs_z=float(err.norm() / z_ref.norm()),
        err_over_moved=float(err.norm() / moved),
        lost=float(-(err * move).sum() / (move * move).sum()),   # fraction of movement not realized
        bias_ulp=float(err_ulp.mean()),
        bias_se_ulp=float(err_ulp.std() / math.sqrt(D)),
        std_ulp=float(err_ulp.std()),
        max_ulp=float(err_ulp.abs().max()),
        fwd_std_ulp=float(fwd_err_ulp.std()),          # error the forward pass sees (bf16 w)
        fwd_bias_ulp=float(fwd_err_ulp.mean()),
        corr_err_c=float((err_ulp * c).mean() / err_ulp.std()) if float(err_ulp.std()) > 0 else 0.0,
    )


SCHEMES = ["sr", "kahan", "kahan8-sr", "kahan8-rn", "kahan4-sr", "kahan4-rn", "kahan16"]


def main() -> None:
    quick = "--quick" in sys.argv
    out = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = out[0] if out else "sim_compact_kahan.json"
    grid: list[tuple[str, float, float, float]] = []
    for lr in (1e-3, 1e-4, 1e-5):
        for k in (0.0, 4.0):
            for s in SCHEMES:
                grid.append((s, lr, k, 1.0))
    # pure-drift sub-grid regime: lr 1e-6 (0.004 ulp/step) with no update noise -> RTN stall test
    for s in SCHEMES:
        grid.append((s, 1e-6, 0.0, 0.0))
        grid.append((s, 1e-6, 4.0, 0.0))
    if quick:
        grid = [g for g in grid if g[1] == 1e-5]
    results = []
    t0 = time.time()
    for scheme, lr, k, ns in grid:
        r = run(scheme, lr, k, ns)
        results.append(r)
        print(json.dumps(r), flush=True)
        with open(path, "w") as fh:
            json.dump(results, fh, indent=1)
    print(f"elapsed {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
