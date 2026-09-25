"""Antikaon low-LR bf16 quality experiment — the regime the control battery never exercises.

The control battery (and ``benchmarks/antikaon_gate``) trains the proxy U-Net in fp32 at
``lr_const = 1.2e-3``, ~100x a real fine-tune LR, so every update is several bf16 ulps and
no ``bf16_method`` matters. This experiment keeps the SAME U-Net, dataset, loss and eval
(``benchmarks/control/battery.py`` + ``benchmarks/proxy``, loaded by file path exactly like
``run_gate.py``) but:

* **bf16 weights** (the whole U-Net in ``torch.bfloat16``; loss and eval MSE in fp32);
* **a real fine-tune LR** (constant, default ``1e-5``), where the typical update is a
  FRACTION of a bf16 ulp — the regime compact Kahan (``bf16_method="kahan8"``) exists for;
* **fine-tune protocol**: the U-Net is first pre-trained in fp32 at the battery's own
  ``lr_const`` with Adakaon-nomom (``--pretrain`` steps, cached per seed under ``cache/``),
  rounded ONCE to bf16, and every arm fine-tunes from that same bf16-representable start
  (the fp32 ceiling starts from the same values in fp32). Starting from random init at
  1e-5 would mostly measure how slowly a random net moves; a partly trained net at a
  fine-tune LR keeps improving measurably and is what low-LR bf16 training looks like.

Why ``lr = 1e-5`` (``--probe`` prints the evidence; README.md keeps the table): Adakaon's
update is RMS-clipped per tensor, so a coordinate moves ``lr * |u|`` with ``|u| ~ O(1)``; after
the 300-step pre-training the weights sit at |w| p10/p50/p90 = 0.004 / 0.023 / 0.053, where
one bf16 ulp is ``2^(floor(log2|w|) - 7)``. Measured per-coordinate ``lr*|u| / ulp`` on that
start (update directions of 10 fp32 steps), p10 / p50 / p90:

    lr 5e-6   0.010 / 0.030 / 0.16
    lr 1e-5   0.021 / 0.061 / 0.32     <- the 10-90 % band sits in the 0.01-0.3 ulp target
    lr 1.5e-5 0.031 / 0.091 / 0.48
    lr 2e-5   0.042 / 0.121 / 0.64
    lr 3e-5   0.063 / 0.181 / 0.97     (a tenth of the coordinates already move ~1 ulp/step)

so ``1e-5`` is the LR where the typical step is a small fraction of an ulp (median 0.06)
and almost none reaches half an ulp. Why 300 pre-training steps and 8000 fine-tune steps:
at 1e-5 a step moves ~1/120 of a battery step, so the fine-tune is ~70 battery-steps' worth
of progress; starting from the steeper early part of the curve (300 steps; the battery's
held-out loss still falls fast there) keeps that progress measurable against the arm-to-arm
differences, which are paired (same start, same data order, same seed).

Arms (``--arms``, comma-separated keys):
  ada-sr      Adakaon-nomom, bf16, stochastic rounding
  ada-k8      Adakaon-nomom, bf16, kahan8
  anti1.5-sr  Antikaon k_sigma=1.5, bf16, SR          anti1.5-k8  ... kahan8
  anti5-sr    Antikaon k_sigma=5,   bf16, SR          anti5-k8    ... kahan8
  nek-sr      Nekaon (registry defaults), bf16, SR    nek-k8      ... kahan8
  ada-fp32    Adakaon-nomom, fp32 weights — the ceiling (and the fp32 twin of ada-*)
  anti1.5-fp32, anti5-fp32, nek-fp32 — optional fp32 twins (``--twins``) so each rule's
              distance to fp32 is measured against ITS OWN fp32 run.

Optimizer configs mirror ``benchmarks/control/registry.py`` (``Adakaon-nomom``, ``Antikaon``,
``Nekaon``) with only ``bf16_method`` added; they are re-declared here because the registry
lambdas take no extra kwargs.

Metrics per arm and seed: held-out loss and train loss at the CLEAN weights (``evald``:
eval()/train() bracket, so Antikaon/Nekaon are scored without their perturbation), gap =
test - train, the test loss at the start of fine-tuning (so the learning is visible),
``test_full`` — the held-out loss of the same clean iterate at its FULL value in an fp32
copy of the net (for kahan8 the stored bf16 is the nearest one to it, which lags the
trajectory until it crosses a half-ulp boundary; SR's stored bf16 is an unbiased
rounding) —, which
Antikaon inert-noise warnings fired, and — cheap, from the final state — the distance of the
arm's clean full-precision iterate (``decode(p, kahan_lo)`` under kahan8) to the fp32 twin
of its rule (or to ``ada-fp32`` when no twin ran): RMS in bf16 ulps of the reference and
``||z - z_fp32|| / ||z_fp32 - z_start||`` (error relative to how far fp32 moved). Quality only:
NO timing is measured or reported.

Usage:
    python run_lowlr.py --smoke                       # a few steps, all arms: plumbing check
    python run_lowlr.py --probe                       # step/ulp table for candidate LRs
    python run_lowlr.py                               # full run (defaults below)
    python run_lowlr.py --seeds 0,1 --steps 8000 --lr 1e-5 --twins
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import warnings

import torch
import torch.nn.functional as F

from kaon import Adakaon, Antikaon, Nekaon
from kaon._compact_kahan import RESIDUAL_KEY, decode

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


CTRL = _load("control_battery", f"{REPO}/benchmarks/control/battery.py")
H = CTRL.H
D = CTRL.D
REG = CTRL.REG
DEV = CTRL.DEV

PRETRAIN_LR = REG.OPTIMIZERS["Adakaon-nomom"]["lr_const"]      # 1.2e-3, the battery's constant LR
DEFAULTS = dict(C=40, bs=8, pretrain=300, steps=8000, lr=1e-5, seeds=[0, 1])
SMOKE = dict(C=40, bs=8, pretrain=8, steps=6, lr=1e-5, seeds=[0])


# ----------------------------- model: the proxy U-Net at any weight dtype -----------------
class UNetLP(H.UNet):
    """``H.UNet`` whose forward casts its inputs to the weights' dtype and returns fp32.

    Identical to ``H.UNet.forward`` for fp32 weights (every cast is a no-op). For bf16 weights
    the harness would feed fp32 activations (the DDPM mixing and ``temb`` are fp32) into bf16
    convs; casting at the two entry points keeps the whole network in bf16 while the loss
    (``F.mse_loss`` against the fp32 noise in ``H.batch_loss``) stays fp32."""

    def forward(self, x, t):
        dt = self.inp.weight.dtype
        te = self.tmlp(H.temb(t, self.td).to(dt))
        h1 = self.d1(self.inp(x.to(dt)), te)
        h = self.mid(self.down(h1), te)
        h = self.u1(torch.cat([self.up(h), h1], 1), te)
        return self.out(F.silu(self.outn(h))).float()


# ----------------------------- arms ------------------------------------------------------
def _ada(method):
    return lambda p, lr: Adakaon(p, lr=lr, betas=(0.0, 0.999), cautious=False,
                                 momentum_dtype="bfloat16", bf16_method=method)


def _anti(k, method):
    return lambda p, lr: Antikaon(p, lr=lr, k_sigma=k, bf16_method=method)


def _nek(method):
    return lambda p, lr: Nekaon(p, lr=lr, k=1.5, betas=(0.5, 0.999), weight_decay=0.1,
                                momentum_dtype="4bit", bf16_method=method)


SR, K8 = "stochastic_rounding", "kahan8"
BF, FP = torch.bfloat16, torch.float32
# key -> (make, weight dtype, rule id: arms with the same rule share an fp32 twin)
ARMS = {
    "ada-sr": (_ada(SR), BF, "ada"),
    "ada-k8": (_ada(K8), BF, "ada"),
    "anti1.5-sr": (_anti(1.5, SR), BF, "anti1.5"),
    "anti1.5-k8": (_anti(1.5, K8), BF, "anti1.5"),
    "anti5-sr": (_anti(5.0, SR), BF, "anti5"),
    "anti5-k8": (_anti(5.0, K8), BF, "anti5"),
    "nek-sr": (_nek(SR), BF, "nek"),
    "nek-k8": (_nek(K8), BF, "nek"),
    "ada-fp32": (_ada(SR), FP, "ada"),
}
TWINS = {
    "anti1.5-fp32": (_anti(1.5, SR), FP, "anti1.5"),
    "anti5-fp32": (_anti(5.0, SR), FP, "anti5"),
    "nek-fp32": (_nek(SR), FP, "nek"),
}
ALL_ARMS = {**ARMS, **TWINS}


# ----------------------------- helpers ---------------------------------------------------
def full_values(opt, params):
    """The full-precision stored value of every param (fp32, detached): ``decode(p, lo)``
    when some optimizer in the wrapper chain keeps a kahan8 residual for it, else ``p``."""
    out = []
    for p in params:
        v = p.detach().float().clone()
        o = opt
        while o is not None:
            st = o.state.get(p) if hasattr(o, "state") else None
            if st and RESIDUAL_KEY in st and p.dtype == torch.bfloat16:
                v = decode(p.data, st[RESIDUAL_KEY])
                break
            o = getattr(o, "inner", None)
        out.append(v)
    return out


def clean_values(opt, params):
    """Full-precision CLEAN iterate: eval() (removes Antikaon's xi / MSAM's climb), read,
    train()."""
    return CTRL.evald(opt, lambda: full_values(opt, params))


def bf16_ulp(z):
    """Per-coordinate bf16 ulp (the probe's step/ulp ratio)."""
    e = torch.floor(torch.log2(z.abs().clamp_min(2.0 ** -126)))
    return torch.exp2(e - 7)


def distance(zs, refs, starts):
    """(RMS of (z - ref) in bf16 ulps, ||z - ref|| / ||ref - start||).

    The ulp is taken per TENSOR at the reference's RMS magnitude (``eps_bf16 * RMS(ref)``),
    as in ``tests/test_antikaon.py``: a per-coordinate ulp would let the near-zero
    coordinates, whose ulp is tiny, dominate the average."""
    eps = torch.finfo(torch.bfloat16).eps
    num_u, n, d2, m2 = 0.0, 0, 0.0, 0.0
    for z, r, s in zip(zs, refs, starts, strict=True):
        diff = z - r
        u = max(eps * float(r.pow(2).mean().sqrt()), 1e-30)
        num_u += float((diff / u).pow(2).sum())
        n += diff.numel()
        d2 += float((diff ** 2).sum())
        m2 += float(((r - s) ** 2).sum())
    return math.sqrt(num_u / max(n, 1)), math.sqrt(d2) / max(math.sqrt(m2), 1e-30)


def pretrained_state(seed, C, bs, n, data, tr, seq_fn):
    """fp32 Adakaon-nomom pre-training at the battery's constant LR, cached per config."""
    path = os.path.join(HERE, "cache", f"pretrain_C{C}_bs{bs}_n{n}_s{seed}.pt")
    if n > 0 and os.path.exists(path):
        return torch.load(path, map_location="cpu")
    torch.manual_seed(seed)
    if DEV == "cuda":
        torch.cuda.manual_seed_all(seed)
    net = UNetLP(C=C).to(DEV)
    if n > 0:
        opt = _ada(SR)([p for p in net.parameters() if p.requires_grad], PRETRAIN_LR)
        g = torch.Generator(device=DEV)
        g.manual_seed(seed + 777)
        ac = H.make_alphas()
        pos = 0
        for Rr in seq_fn(n):
            idx = [tr[(pos + j) % len(tr)] for j in range(bs)]
            pos += bs
            opt.zero_grad()
            H.batch_loss(net, data[Rr], torch.tensor(idx, device=DEV), ac, g).backward()
            opt.step()
    # Round ONCE to bf16: every arm (fp32 ceiling included) starts from these exact values.
    sd = {k: v.detach().to(torch.bfloat16).float().cpu() for k, v in net.state_dict().items()}
    if n > 0:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(sd, path)
    return sd


def finetune(key, *, start_sd, lr, seed, C, bs, n, data, tr, te, ac, seq):
    """One fine-tune run (the battery's const-LR loop, minus timing)."""
    make, dtype, _rule = ALL_ARMS[key]
    torch.manual_seed(seed)
    if DEV == "cuda":
        torch.cuda.manual_seed_all(seed)
    net = UNetLP(C=C)
    net.load_state_dict(start_sd)
    net = net.to(DEV).to(dtype)
    params = [p for p in net.parameters() if p.requires_grad]
    z0 = [p.detach().float().clone() for p in params]
    te0 = H.eval_loss(net, data[64], te, ac)
    g = torch.Generator(device=DEV)
    g.manual_seed(seed + 12345)
    fired = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opt = make(params, lr)
        pos = 0
        for Rr in seq:
            idx = [tr[(pos + j) % len(tr)] for j in range(bs)]
            pos += bs
            opt.zero_grad()
            H.batch_loss(net, data[Rr], torch.tensor(idx, device=DEV), ac, g).backward()
            opt.step()
    for w in caught:
        msg = str(w.message)
        if "perturbation radius" in msg or "lookahead" in msg.lower():
            fired.append(msg[:160])
    tr_loss, te_loss = CTRL.evald(
        opt, lambda: (H.eval_loss(net, data[64], tr, ac), H.eval_loss(net, data[64], te, ac)))
    z = clean_values(opt, params)
    # The same clean iterate evaluated in fp32 (the full value, not its bf16 rounding): a
    # kahan8 model's forward sees the NEAREST bf16 of z, which only moves once z crosses a
    # half-ulp boundary, while SR's forward sees an unbiased random rounding. test_full
    # separates the trajectory's quality from that final storage rounding.
    net32 = copy.deepcopy(net).float()
    with torch.no_grad():
        for q, v in zip((q for q in net32.parameters() if q.requires_grad), z, strict=True):
            q.copy_(v)
    te_full = H.eval_loss(net32, data[64], te, ac)
    del net32
    return dict(train=tr_loss, test=te_loss, gap=te_loss - tr_loss, test_start=te0,
                test_full=te_full,
                inert_warnings=sorted(set(fired)), clean=[t.cpu() for t in z],
                start=[t.cpu() for t in z0])


def mean(xs):
    return sum(xs) / len(xs)


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


# ----------------------------- probe: how many ulps is a step? ---------------------------
def probe(cfg, data, tr, lrs=(5e-6, 1e-5, 1.5e-5, 2e-5, 3e-5), steps=10):
    """Per-coordinate ``lr * |u| / ulp_bf16(w)`` on the pre-trained start, from the update
    directions of ``steps`` fp32 Adakaon-nomom steps (the update is lr-independent: it is the
    clipped preconditioned gradient). Prints quantiles over all coordinates and per tensor
    class (conv/linear weights vs 1-D gains/biases)."""
    seed = cfg["seeds"][0]
    sd = pretrained_state(seed, cfg["C"], cfg["bs"], cfg["pretrain"], data, tr, CTRL.seq_prog)
    net = UNetLP(C=cfg["C"])
    net.load_state_dict(sd)
    net = net.to(DEV)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = _ada(SR)(params, 1.0)        # lr 1 -> the weight change IS the update u
    g = torch.Generator(device=DEV)
    g.manual_seed(seed + 99)
    ac = H.make_alphas()
    us = [torch.zeros_like(p) for p in params]
    pos = 0
    for Rr in CTRL.seq_prog(steps):
        before = [p.detach().clone() for p in params]
        idx = [tr[(pos + j) % len(tr)] for j in range(cfg["bs"])]
        pos += cfg["bs"]
        opt.zero_grad()
        H.batch_loss(net, data[Rr], torch.tensor(idx, device=DEV), ac, g).backward()
        opt.step()
        for u, b, p in zip(us, before, params, strict=True):
            u.add_((b - p.detach()).abs() / steps)
            p.data.copy_(b)            # probe the update at the start point, do not move
    ulps = [bf16_ulp(p.detach()) for p in params]
    table = {}
    for lr in lrs:
        rows = {}
        for cls in ("all", "matrices", "vectors"):
            sel = [(u, q) for u, q, p in zip(us, ulps, params, strict=True)
                   if cls == "all" or (cls == "matrices") == (p.ndim >= 2)]
            r = torch.cat([(lr * u / q).flatten() for u, q in sel]).float().cpu()
            qs = torch.quantile(r[torch.randperm(r.numel())[:200000]],
                                torch.tensor([0.1, 0.5, 0.9]))
            rows[cls] = [round(float(x), 4) for x in qs]
        table[f"{lr:g}"] = rows
        print(f"lr={lr:g}: step/ulp quantiles (p10, p50, p90)  " +
              "  ".join(f"{c}={v}" for c, v in rows.items()), flush=True)
    mags = torch.cat([p.detach().abs().flatten() for p in params]).float().cpu()
    print("pretrained |w| quantiles (p10,p50,p90):",
          [round(float(x), 5) for x in torch.quantile(mags[torch.randperm(mags.numel())[:200000]],
                                                      torch.tensor([0.1, 0.5, 0.9]))])
    return table


# ----------------------------- main ------------------------------------------------------
def run(cfg, arm_keys, out_path):
    ds = D.build_proxy_dataset()
    fp = D.fingerprint(ds)
    data = {k: v.to(DEV) for k, v in ds["DATA"].items()}       # fp32; UNetLP casts
    tr, te = ds["TR"], ds["TE"]
    ac = H.make_alphas()
    seq = CTRL.seq_prog(cfg["steps"])
    results: dict[str, dict] = {}
    for seed in cfg["seeds"]:
        sd = pretrained_state(seed, cfg["C"], cfg["bs"], cfg["pretrain"], data, tr,
                              CTRL.seq_prog)
        runs = {}
        for key in arm_keys:
            r = finetune(key, start_sd=sd, lr=cfg["lr"], seed=seed, C=cfg["C"], bs=cfg["bs"],
                         n=cfg["steps"], data=data, tr=tr, te=te, ac=ac, seq=seq)
            runs[key] = r
            print(f"  [{key}] seed={seed} test={r['test']:.5f} (start {r['test_start']:.5f}, "
                  f"fp32 view {r['test_full']:.5f}) "
                  f"train={r['train']:.5f} gap={r['gap']:+.5f}"
                  + (f"  inert-warning x{len(r['inert_warnings'])}" if r["inert_warnings"] else ""),
                  flush=True)
        for key, r in runs.items():
            rule = ALL_ARMS[key][2]
            ref_key = next((k for k in runs if ALL_ARMS[k][1] == FP and ALL_ARMS[k][2] == rule
                            and k != key), "ada-fp32" if key != "ada-fp32" else None)
            d_ulp = d_rel = None
            if ref_key in runs:
                d_ulp, d_rel = distance(r["clean"], runs[ref_key]["clean"], r["start"])
            row = results.setdefault(key, dict(per_seed=[]))
            row["per_seed"].append(dict(
                seed=seed, test=r["test"], train=r["train"], gap=r["gap"],
                test_start=r["test_start"], test_full=r["test_full"], dist_ref=ref_key, dist_ulp=d_ulp, dist_rel=d_rel,
                inert_warnings=r["inert_warnings"]))
    for row in results.values():
        ps = row["per_seed"]
        for m in ("test", "gap", "train", "test_start", "test_full"):
            row[f"{m}_mean"] = mean([p[m] for p in ps])
            row[f"{m}_std"] = stdev([p[m] for p in ps])
        dus = [p["dist_ulp"] for p in ps if p["dist_ulp"] is not None]
        drs = [p["dist_rel"] for p in ps if p["dist_rel"] is not None]
        row["dist_ulp_mean"] = mean(dus) if dus else None
        row["dist_rel_mean"] = mean(drs) if drs else None
    print("\narm            test (+-sd)          gap (+-sd)         d_start   test_full  "
          "dist_ulp  dist_rel")
    for key, row in results.items():
        du = "-" if row["dist_ulp_mean"] is None else f"{row['dist_ulp_mean']:.3f}"
        dr = "-" if row["dist_rel_mean"] is None else f"{row['dist_rel_mean']:.3f}"
        print(f"{key:13s} {row['test_mean']:.5f}(+-{row['test_std']:.5f}) "
              f"{row['gap_mean']:+.5f}(+-{row['gap_std']:.5f}) "
              f"{row['test_mean'] - row['test_start_mean']:+.5f}  {row['test_full_mean']:.5f}  "
              f"{du:>8s}  {dr:>8s}", flush=True)
    manifest = dict(
        commit=os.popen("git rev-parse HEAD").read().strip(),
        branch=os.popen("git rev-parse --abbrev-ref HEAD").read().strip(),
        config=cfg, pretrain_lr=PRETRAIN_LR, schedule="const", device=DEV,
        dataset_fingerprint=fp, arms=arm_keys,
    )
    with open(out_path, "w") as f:
        json.dump(dict(manifest=manifest, results=results), f, indent=1)
    print(f"wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true", help="few steps, all arms (plumbing)")
    ap.add_argument("--probe", action="store_true", help="print the step/ulp table and exit")
    ap.add_argument("--arms", type=str, default=None, help="comma-separated arm keys")
    ap.add_argument("--twins", action="store_true", help="add the fp32 twins of anti*/nek")
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--pretrain", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--C", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    cfg = dict(SMOKE if a.smoke else DEFAULTS)
    for k in ("steps", "pretrain", "lr", "C"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    if a.seeds:
        cfg["seeds"] = [int(s) for s in a.seeds.split(",")]
    if a.probe:
        ds = D.build_proxy_dataset()
        probe(cfg, {k: v.to(DEV) for k, v in ds["DATA"].items()}, ds["TR"])
        return
    keys = [s.strip() for s in a.arms.split(",")] if a.arms else list(ARMS)
    if a.twins or a.smoke:
        keys += [k for k in TWINS if k not in keys]
    unknown = [k for k in keys if k not in ALL_ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {list(ALL_ARMS)}")
    out = a.out or os.path.join(HERE, "smoke.json" if a.smoke else "results.json")
    run(cfg, keys, out)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
