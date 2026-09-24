"""bf16 drift simulation for the "noise kept in the live weights" design.

Live weights hold w_n = z_n + xi_n (bf16). Each step ONE write: w <- round(w - delta_n + xi_{n+1} - xi_n).
We recover z_n = w_n - xi_n (fp32) and compare against an fp32 reference z*_n (same delta stream).

Schemes:
  fp32        exact reference
  sr          bf16 + stochastic rounding, NO noise (plain SR training baseline)
  sr+xi       bf16 SR on the combined write (the hypothesis)
  rtn+xi      bf16 round-to-nearest on the combined write (subtractive-dither argument)
  2w-rtn      MSAM-style: remove xi_n (RTN), SR(-delta), apply xi_{n+1} (RTN)   [3 writes/step]
  2w-sr       same but every write SR (the measured 19%-drift bug)
  kahan+xi    combined write with a bf16 Kahan shift buffer (2 B/param), as kaon's bf16_method="kahan"

Update stream: delta = lr * (0.3*c_i + 0.95*nu), c_i in {-1,+1} fixed, nu ~ N(0,1)  -> RMS ~ 1 like Adakaon's u.
Noise: xi_n = k * lr * s_n, s_n Rademacher (default) or Gaussian; regenerated from (seed, step) -> zero state.
Only numerical checks. No timing.
"""
from __future__ import annotations

import json
import math
import sys
import time

import torch

import kaon  # noqa: F401  (must resolve into the worktree; printed below)
from kaon._stochastic_rounding import add_stochastic_

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D = 1 << 20 if dev.type == "cuda" else 1 << 17
N = 10_000 if dev.type == "cuda" else 3_000
print("kaon:", kaon.__file__, "| device:", dev, "| D:", D, "| N:", N, flush=True)


def ulp_bf16(x: torch.Tensor) -> torch.Tensor:
    """bf16 ulp of |x| (7 explicit mantissa bits): 2^(floor(log2|x|) - 7)."""
    e = torch.floor(torch.log2(x.abs().clamp_min(1e-30)))
    return torch.exp2(e - 7)


def noise(seed: int, step: int, kind: str) -> torch.Tensor:
    g = torch.Generator(device=dev)
    g.manual_seed((seed * 1_000_003 + step) & 0x7FFFFFFF)
    if kind == "rademacher":
        return torch.randint(0, 2, (D,), generator=g, device=dev, dtype=torch.float32).mul_(2).sub_(1)
    return torch.randn(D, generator=g, device=dev)


def run(scheme: str, lr: float, k: float, kind: str, seed: int = 1234, n_eval_cycles: int = 0) -> dict:
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    w0 = torch.randn(D, generator=gen, device=dev) * 0.05
    c = torch.randint(0, 2, (D,), generator=gen, device=dev, dtype=torch.float32).mul_(2).sub_(1)
    z_ref = w0.clone()                       # fp32 reference of the clean iterate
    sigma = k * lr
    xi = sigma * noise(seed, 0, kind) if k > 0 else torch.zeros(D, device=dev)
    if scheme == "fp32":
        w = (w0 + xi).clone()
    else:
        w = (w0 + xi).to(torch.bfloat16)      # live weights (carry xi_0)
    shift = torch.zeros(D, device=dev, dtype=torch.bfloat16) if scheme == "kahan+xi" else None
    sr = None  # kaon module generator (unbiased); its position is irrelevant for the statistics

    def write_sr(target: torch.Tensor, src: torch.Tensor) -> None:
        add_stochastic_(target, src, alpha=1.0, sr=sr)

    def write_rtn(target: torch.Tensor, src: torch.Tensor) -> None:
        target.copy_((target.float() + src).to(target.dtype))

    for n in range(N):
        if n % 50 == 0:  # piecewise-coherent drift: net movement stays at the weight scale (~0.1 at lr=1e-3)
            c = torch.randint(0, 2, (D,), generator=gen, device=dev, dtype=torch.float32).mul_(2).sub_(1)
        nu = torch.randn(D, generator=gen, device=dev)
        delta = lr * (0.1 * c + 0.99 * nu)
        z_ref.sub_(delta)
        xi_next = sigma * noise(seed, n + 1, kind) if k > 0 else xi
        dxi = xi_next - xi
        if scheme == "fp32":
            w.add_(dxi - delta)
        elif scheme == "sr":
            write_sr(w, -delta)
        elif scheme == "sr+xi":
            write_sr(w, dxi - delta)
        elif scheme == "rtn+xi":
            write_rtn(w, dxi - delta)
        elif scheme == "2w-rtn":
            write_rtn(w, -xi)
            write_sr(w, -delta)
            write_rtn(w, xi_next)
        elif scheme == "2w-sr":
            write_sr(w, -xi)
            write_sr(w, -delta)
            write_sr(w, xi_next)
        elif scheme == "kahan+xi":
            # kaon subtract_one_ kahan branch: shift -= delta(bf16); p += shift; shift += (p_before - p)
            shift.sub_((delta - dxi).to(torch.bfloat16))
            p_before = w.clone()
            w.add_(shift)
            shift.add_(p_before.sub_(w))
        else:
            raise ValueError(scheme)
        xi = xi_next

    z_rec = w.float() - xi
    err = z_rec - z_ref
    # Reference ulp: the bf16 ulp at the RMS weight magnitude (a per-coordinate ulp(z_ref) blows up
    # where z_ref crosses zero). Errors are reported in units of this one ulp.
    u_ref = float(ulp_bf16(z_ref.pow(2).mean().sqrt()))
    err_ulp = err / u_ref
    moved = (z_ref - w0).norm()
    # per-coordinate robust view: median of |err| / ulp(|z_ref| floored at the RMS/8)
    u_loc = ulp_bf16(torch.maximum(z_ref.abs(), z_ref.pow(2).mean().sqrt() / 8))
    out = dict(
        scheme=scheme, lr=lr, k=k, kind=kind,
        rms_w=float(z_ref.pow(2).mean().sqrt()), ulp_ref=u_ref,
        sigma_over_ulp=sigma / u_ref if k > 0 else 0.0,
        step_over_ulp=lr / u_ref,
        rel_l2_vs_z=float(err.norm() / z_ref.norm()),
        rel_l2_vs_moved=float(err.norm() / moved),
        bias_ulp=float(err_ulp.mean()),                    # mean signed error (should be ~0 if unbiased)
        bias_se_ulp=float(err_ulp.std() / math.sqrt(D)),   # standard error of that mean
        std_ulp=float(err_ulp.std()),
        median_abs_err_local_ulp=float((err.abs() / u_loc).median()),
        pred_std_sr=math.sqrt(N / 6.0), pred_std_rtn=math.sqrt(N / 12.0),
        max_ulp=float(err_ulp.abs().max()),
        # coherence check: correlation of error with the LAST drift direction c (a biased scheme shows it)
        corr_err_c=float((err_ulp * c).mean() / err_ulp.std()),
    )
    if n_eval_cycles and scheme != "fp32":
        # eval()/train() round trips: w_eval = RTN(w - xi); w_train = RTN(w_eval + xi)
        w_before = w.clone()
        for _ in range(n_eval_cycles):
            w_eval = (w.float() - xi).to(torch.bfloat16)
            w = (w_eval.float() + xi).to(torch.bfloat16)
        rt = (w.float() - w_before.float()) / u_ref
        out["evalcycle_drift_ulp_std"] = float(rt.std())
        out["evalcycle_drift_ulp_bias"] = float(rt.mean())
        out["evalcycle_frac_changed"] = float((rt != 0).float().mean())
        zc = w_before.float() - xi
        out["eval_view_err_local_ulp_max"] = float(((w_eval.float() - zc) / ulp_bf16(zc)).abs().max())
    return out


def main() -> None:
    results = []
    t0 = time.time()
    kinds = ["rademacher"]
    grid = []
    for lr in (1e-3, 1e-4, 1e-5):
        grid.append(("fp32", lr, 4.0, "rademacher"))
        grid.append(("sr", lr, 0.0, "rademacher"))
        for k in (0.5, 1.0, 4.0, 16.0):
            grid.append(("sr+xi", lr, k, "rademacher"))
            grid.append(("rtn+xi", lr, k, "rademacher"))
        for k in (1.0, 4.0):
            grid.append(("2w-rtn", lr, k, "rademacher"))
            grid.append(("2w-sr", lr, k, "rademacher"))
        grid.append(("kahan+xi", lr, 4.0, "rademacher"))
    grid.append(("sr+xi", 1e-4, 4.0, "gaussian"))
    grid.append(("rtn+xi", 1e-4, 4.0, "gaussian"))
    for scheme, lr, k, kind in grid:
        r = run(scheme, lr, k, kind, n_eval_cycles=100)
        results.append(r)
        print(json.dumps(r), flush=True)
    print(f"elapsed {time.time() - t0:.0f}s", flush=True)
    json.dump(results, open(sys.argv[1] if len(sys.argv) > 1 else "sim_drift.json", "w"), indent=1)


if __name__ == "__main__":
    main()
