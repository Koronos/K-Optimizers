"""The optimizer PROFILER — the diagnostic probes we always run on a new optimizer to
discover *what it likes*. This is NOT the ranking battery (``battery.py``, one best config
vs the field); it explores configs for ONE optimizer and writes a profile to
``profiles/<Name>.md``.

It answers, by greedy coordinate search (each step fixes the previous winner, ranked by the
train–val GAP then loss):

  1. ideal LR        — sweep around the registry LR (loss-best vs gap-best may differ).
  2. schedule        — constant vs REX vs cosine vs linear: which decay does it want?
  3. warmup          — none / short / longer linear warmup: does it help?
  4. curriculum      — single-resolution vs the progressive curriculum: how much data-noise
                       regularization does it lean on?
  5. variants        — optimizer-specific constructor knobs declared in the registry's
                       optional ``variants`` field (e.g. cautious on/off, beta0, momentum dtype).

Run:  python profiler.py --opt AdaPNM
      python profiler.py --opt AdaPNM --quick

Same proxy + caveats as the battery: proxy LRs are ~100x real; the gap ranks objective
overfitting, not perceptual quality — confirm trends on a real LoRA.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import random

import torch

REPO = "/media/koronos/arca/repos/K-Optimizers"
HERE = f"{REPO}/benchmarks/control"


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


H = _load("harness", f"{REPO}/benchmarks/proxy/harness.py")
D = _load("dataset", f"{REPO}/benchmarks/proxy/dataset.py")
REG = _load("registry", f"{HERE}/registry.py")
DEV = H.DEV


def rex(p, d=0.9):
    z = 1 - p; return z / ((1 - d) + d * z)


def sched_base(name, p):
    if name == "const":  return 1.0
    if name == "rex":    return rex(p)
    if name == "cosine": return 0.5 * (1 + math.cos(math.pi * p))
    if name == "linear": return 1.0 - p
    raise ValueError(name)


def seq_prog(n):
    k = int(n * 0.2); m = (n - k) // 2; m2 = n - k - m
    a = ([32, 64] * ((m // 2) + 1))[:m]; random.Random(123).shuffle(a)
    b = ([48, 64] * ((m2 // 2) + 1))[:m2]; random.Random(124).shuffle(b)
    return a + b + [64] * k


def seq_single(n):
    return [64] * n


def run(make, lr, *, schedule, warm, seq, seed, data, tr, te, ac, C, bs, n):
    torch.manual_seed(seed)
    if DEV == "cuda":
        torch.cuda.manual_seed_all(seed)
    net = H.UNet(C=C).to(DEV).to(H.DT)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = make(params, lr)
    # Each group keeps ITS OWN base lr (relative to `lr`) through the schedule — a flat
    # overwrite to `lr * ...` would clobber any per-group lr ratio a multi-group optimizer
    # was constructed with (e.g. Nekaon's low_vram_lr_ratio), silently forcing every group
    # to the same lr regardless of what `make()` set up.
    base_lrs = [pg["lr"] for pg in opt.param_groups]
    g = torch.Generator(device=DEV); g.manual_seed(seed + 12345); pos = 0
    for it, Rr in enumerate(seq):
        p = it / n
        w = (p / warm) if (warm and p < warm) else 1.0
        for pg, base in zip(opt.param_groups, base_lrs):
            pg["lr"] = base * w * sched_base(schedule, p)
        idx = [tr[(pos + j) % len(tr)] for j in range(bs)]; pos += bs
        opt.zero_grad()
        H.batch_loss(net, data[Rr], torch.tensor(idx, device=DEV), ac, g).backward()
        opt.step()
    if DEV == "cuda":
        torch.cuda.synchronize()
    # Optimizers that keep an averaged / perturbed view (ScheduleFree's x, Lookahead's phi,
    # MSAM's unperturbed w — e.g. Nekaon) expose eval()/train(); bracket the measurement so
    # it scores the weights a real run would sample/checkpoint. Mirrors battery.py's evald().
    swap = hasattr(opt, "eval") and hasattr(opt, "train")
    if swap:
        opt.eval()
    tr_l = H.eval_loss(net, data[64], tr, ac); te_l = H.eval_loss(net, data[64], te, ac)
    if swap:
        opt.train()
    return te_l, te_l - tr_l  # (loss, gap)


def avg(make, lr, *, schedule, warm, seq, cfg, data, tr, te, ac):
    rs = [run(make, lr, schedule=schedule, warm=warm, seq=seq, seed=s, data=data, tr=tr, te=te,
              ac=ac, C=cfg["C"], bs=cfg["bs"], n=cfg["N"]) for s in range(cfg["seeds"])]
    return sum(r[0] for r in rs) / len(rs), sum(r[1] for r in rs) / len(rs)


def best_by_loss(results):
    """results: list of (label, loss, gap). Rank by HELD-OUT loss — it penalizes both
    underfitting and overfitting, so it avoids the degenerate 'underfit → tiny gap' pick.
    The gap is reported alongside as the overfit diagnostic."""
    return min(results, key=lambda r: r[1])


def tbl(header, rows, star=None):
    out = ["| " + " | ".join([*header, ""]) + " |",
           "|" + "|".join(["---"] * (len(header) + 1)) + "|"]
    for r in rows:
        mark = "⬅ likes" if star is not None and r[0] == star else ""
        out.append("| " + " | ".join([*(str(c) for c in r), mark]) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", required=True, help="optimizer name from registry.py")
    ap.add_argument("--quick", action="store_true")
    A = ap.parse_args()
    if A.opt not in REG.OPTIMIZERS:
        raise SystemExit(f"{A.opt} not in registry: {list(REG.OPTIMIZERS)}")
    spec = REG.OPTIMIZERS[A.opt]
    make, base_lr = spec["make"], spec["lr"]

    C = 96 if A.quick else 128
    N = 600 if A.quick else 1400
    SEEDS = 1 if A.quick else 2
    cfg = dict(C=C, N=N, seeds=SEEDS, bs=8)
    ds = D.build_proxy_dataset()
    data = {k: v.to(DEV).to(H.DT) for k, v in ds["DATA"].items()}
    tr, te = ds["TR"], ds["TE"]; ac = H.make_alphas()
    P = seq_prog(N); S1 = seq_single(N)
    print(f"PROFILE {A.opt}  C={C} N={N} seeds={SEEDS} (greedy; ranked by held-out loss)", flush=True)

    L = [f"# Profile — {A.opt}\n",
         f"> What this optimizer *likes*, found by [`profiler.py`](profiler.py) on the reproducible "
         f"proxy (`C={C}`, `N={N}`, {SEEDS} seed(s)). Greedy: each section fixes the previous winner; "
         f"ranked by **held-out loss** (which penalizes under- *and* over-fitting), with the "
         f"**train–val gap** shown as the overfit diagnostic. Proxy LRs are ~100x real — relative "
         f"knobs, not recommendations.\n",
         f"**Identity:** {spec['blurb']} · registry LR `{base_lr:.1e}`.\n",
         "> ⚠️ **Scale & objective.** This is a *short* run (mild overfitting), so the held-out-loss "
         "optimum favors **less regularization** (lower LR, lighter knobs, single-res). On real "
         "small-data LoRA overfitting bites far harder — there the **gap** column matters more, and "
         "the **registry's shipped config is gap-tuned** (usually *more* regularizing than the "
         "loss-optimum below). Read this as the optimizer's response *surface* — which way each knob "
         "pushes loss vs gap — not a ship-it config. The gap-ranked field view is "
         "[`RANKINGS.md`](RANKINGS.md).\n"]

    # 1. LR sweep (rex + prog)
    grid = [base_lr * f for f in (0.25, 0.5, 1.0, 2.0, 4.0)]
    res = []
    for lr in grid:
        lo, gp = avg(make, lr, schedule="rex", warm=0.0, seq=P, cfg=cfg, data=data, tr=tr, te=te, ac=ac)
        res.append((f"{lr:.1e}", lo, gp)); print(f"  lr {lr:.1e}: {lo:.4f}/{gp:+.4f}", flush=True)
    loss_lr = best_by_loss(res); gap_lr = min(res, key=lambda r: r[2])
    best_lr = float(loss_lr[0])
    L.append("## 1. Ideal LR (REX + progressive curriculum)\n")
    L.append(tbl(["lr", "loss", "gap"], [[a, f"{b:.4f}", f"{c:+.4f}"] for a, b, c in res], star=loss_lr[0]))
    L.append(f"\n**Likes:** lr ≈ `{loss_lr[0]}` (best held-out loss)"
             + (f"; if you want max regularization, `{gap_lr[0]}` gives the lowest gap (higher loss)"
                if gap_lr[0] != loss_lr[0] else "")
             + ". (LR optimum is ~resolution-invariant on this proxy.)\n")

    # 2. schedule (best_lr + prog)
    res = []
    for s in ("const", "rex", "cosine", "linear"):
        lo, gp = avg(make, best_lr, schedule=s, warm=0.0, seq=P, cfg=cfg, data=data, tr=tr, te=te, ac=ac)
        res.append((s, lo, gp)); print(f"  sched {s}: {lo:.4f}/{gp:+.4f}", flush=True)
    best_sched = best_by_loss(res)[0]
    gapmin_sched = min(res, key=lambda r: r[2])[0]
    L.append("\n## 2. Schedule (at the ideal LR)\n")
    L.append(tbl(["schedule", "loss", "gap"], [[a, f"{b:.4f}", f"{c:+.4f}"] for a, b, c in res], star=best_sched))
    note = " — and it is **happy on a constant LR** (resumable!)" if best_sched == "const" else ""
    extra = (f" The lowest *gap* is `{gapmin_sched}` (more regularizing)." if gapmin_sched != best_sched else "")
    L.append(f"\n**Likes:** `{best_sched}`{note}.{extra}\n")

    # 3. warmup (best_lr + best_sched)
    res = []
    for w in (0.0, 0.03, 0.08):
        lo, gp = avg(make, best_lr, schedule=best_sched, warm=w, seq=P, cfg=cfg, data=data, tr=tr, te=te, ac=ac)
        res.append((f"{int(w*100)}%", lo, gp)); print(f"  warmup {int(w*100)}%: {lo:.4f}/{gp:+.4f}", flush=True)
    best_warm = best_by_loss(res)[0]
    L.append("\n## 3. Warmup (linear, at the ideal LR + schedule)\n")
    L.append(tbl(["warmup", "loss", "gap"], [[a, f"{b:.4f}", f"{c:+.4f}"] for a, b, c in res], star=best_warm))
    helps = "no — skip it" if best_warm == "0%" else f"yes, ~{best_warm}"
    L.append(f"\n**Likes warmup:** {helps}.\n")

    # 4. curriculum dependence (best_lr + best_sched)
    res = []
    for label, seq in (("single-res", S1), ("progressive", P)):
        lo, gp = avg(make, best_lr, schedule=best_sched, warm=0.0, seq=seq, cfg=cfg, data=data, tr=tr, te=te, ac=ac)
        res.append((label, lo, gp)); print(f"  curric {label}: {lo:.4f}/{gp:+.4f}", flush=True)
    best_curr = best_by_loss(res)[0]
    dgap = dict((r[0], r[2]) for r in res)
    L.append("\n## 4. Resolution-curriculum dependence\n")
    L.append(tbl(["curriculum", "loss", "gap"], [[a, f"{b:.4f}", f"{c:+.4f}"] for a, b, c in res], star=best_curr))
    delta = dgap["progressive"] - dgap["single-res"]
    L.append(f"\n**Curriculum Δgap:** `{delta:+.4f}` (more negative = leans harder on the data-noise "
             "for its regularization; the curriculum helps every optimizer, so use it regardless).\n")

    # 5. registry-declared variants (optimizer-specific knobs) at best_lr + best_sched
    variants = spec.get("variants")
    best_v = None
    if variants:
        res = []
        for label, vmake in variants.items():
            lo, gp = avg(vmake, best_lr, schedule=best_sched, warm=0.0, seq=P, cfg=cfg, data=data, tr=tr, te=te, ac=ac)
            res.append((label, lo, gp)); print(f"  variant {label}: {lo:.4f}/{gp:+.4f}", flush=True)
        best_v = best_by_loss(res)[0]
        L.append("\n## 5. Optimizer-specific knobs\n")
        L.append(tbl(["variant", "loss", "gap"], [[a, f"{b:.4f}", f"{c:+.4f}"] for a, b, c in res], star=best_v))
        L.append(f"\n**Best knob (by held-out loss):** `{best_v}`.\n")

    L.append("\n---\n### TL;DR — what it likes\n")
    L.append(f"- **LR** ≈ `{loss_lr[0]}` (proxy scale) · **schedule** `{best_sched}` · "
             f"**warmup** {best_warm} · **curriculum** {best_curr} (Δgap `{delta:+.4f}`)"
             + (f" · **knob** `{best_v}`" if variants else "") + ".\n")

    os.makedirs(f"{HERE}/profiles", exist_ok=True)
    with open(f"{HERE}/profiles/{A.opt}.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote profiles/{A.opt}.md", flush=True)


if __name__ == "__main__":
    main()
