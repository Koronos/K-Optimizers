"""The control battery — one reproducible suite that scores EVERY optimizer in
``registry.py`` across the dimensions we care about, caches each optimizer's data in
``results.json``, and (re)generates the ranked tables in ``RANKINGS.md`` from that cache.

**Incremental by design.** Add a new optimizer to ``registry.py``, then run only it —
its data is measured, merged into the cache, and the whole ranking is regenerated against
everyone else. You never re-measure the field to add one contender.

    python battery.py                 # measure every registry optimizer, refresh rankings
    python battery.py --only AdaPNM   # measure just AdaPNM (+others, comma-sep), merge, re-rank
    python battery.py --new           # measure only optimizers missing from the cache
    python battery.py --render-only   # just rebuild RANKINGS.md from results.json (no training)
    python battery.py --quick         # smaller/faster settings (smoke; lives in its own cache)

Dimensions (each optimizer at its best config, on the reproducible proxy):
  1. per-iteration speed   — ms/step on the C=128 U-Net (full-FT-like) AND on a 512-tiny-tensor
                             adapter bag (LoRA-like, launch-bound — where foreach pays off).
  2. convergence speed     — steps to reach a common held-out target.
  3. time x quality        — wall-clock to that target = ms/step x steps.
  4. loss x generalization — final held-out loss and the train-val GAP (the real objective).
  5. memory                — measured optimizer-state bytes/param.
  6. continuity            — train-val gap at CONSTANT LR (no schedule, resumable) and its
                             change vs the scheduled gap.

Proxy LRs are ~100x real-training LRs (relative knobs, not recommendations). The metrics rank
objective overfitting/convergence, NOT perceptual fidelity — confirm on a real LoRA with FID.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import time

import torch

REPO = "/media/koronos/arca/repos/K-Optimizers"
HERE = f"{REPO}/benchmarks/control"


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


H = _load("harness", f"{REPO}/benchmarks/proxy/harness.py")
D = _load("dataset", f"{REPO}/benchmarks/proxy/dataset.py")
REG = _load("registry", f"{HERE}/registry.py")
OPTIMIZERS = REG.OPTIMIZERS
DEV = H.DEV


# ----------------------------- schedules & resolution sequences -----------------------------
def rex(p, d=0.9):
    z = 1 - p
    return z / ((1 - d) + d * z)


def seq_prog(n):
    """512+1024 -> 768+1024 -> 1024 (40/40/20) in proxy resolutions 32/48/64."""
    k = int(n * 0.2); m = (n - k) // 2; m2 = n - k - m
    a = ([32, 64] * ((m // 2) + 1))[:m]; random.Random(123).shuffle(a)
    b = ([48, 64] * ((m2 // 2) + 1))[:m2]; random.Random(124).shuffle(b)
    return a + b + [64] * k


# ----------------------------- one training run -----------------------------
def train(make, lr, *, schedule, seq, seed, data, tr, te, ac, channels, bs, n, checkpoints=0):
    torch.manual_seed(seed)
    if DEV == "cuda":
        torch.cuda.manual_seed_all(seed)
    net = H.UNet(C=channels).to(DEV).to(H.DT)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = make(params, lr)
    g = torch.Generator(device=DEV); g.manual_seed(seed + 12345)
    pos = 0; traj = []
    ckpt_every = max(1, n // checkpoints) if checkpoints else 0
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for it, Rr in enumerate(seq):
        mult = rex(it / n) if schedule == "rex" else 1.0
        for pg in opt.param_groups:
            pg["lr"] = lr * mult
        idx = [tr[(pos + j) % len(tr)] for j in range(bs)]; pos += bs
        opt.zero_grad()
        H.batch_loss(net, data[Rr], torch.tensor(idx, device=DEV), ac, g).backward()
        opt.step()
        if ckpt_every and (it + 1) % ckpt_every == 0:
            traj.append((it + 1, H.eval_loss(net, data[64], te, ac)))
    if DEV == "cuda":
        torch.cuda.synchronize()
    ms_step = (time.time() - t0) / n * 1000.0
    tr_loss = H.eval_loss(net, data[64], tr, ac)
    te_loss = H.eval_loss(net, data[64], te, ac)
    bpp = H.opt_state_bytes_per_param(opt, params)
    return dict(tr=tr_loss, te=te_loss, gap=te_loss - tr_loss, ms=ms_step, bpp=bpp, traj=traj)


def lora_step_ms(make, lr, reps=50, warmup=10):
    """Median ms to step a 512-tiny-tensor adapter bag (the launch-bound regime)."""
    params = H.lora_bag()
    opt = make(params, lr)
    for _ in range(warmup):
        opt.step()
    if DEV == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        if DEV == "cuda":
            torch.cuda.synchronize()
        t0 = time.time(); opt.step()
        if DEV == "cuda":
            torch.cuda.synchronize()
        ts.append((time.time() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


def mean(xs):
    return sum(xs) / len(xs)


# ----------------------------- measure ONE optimizer -----------------------------
def measure(name, spec, cfg, data, tr, te, ac, seqp):
    """Run the A (scheduled+prog), B (constant+prog), and lora-bag probes for one optimizer.
    Returns the cache entry (plain JSON-able dict, including the per-checkpoint trajectory)."""
    C, N, SEEDS = cfg["C"], cfg["N"], cfg["seeds"]
    As = [train(spec["make"], spec["lr"], schedule="rex", seq=seqp, seed=s,
                data=data, tr=tr, te=te, ac=ac, channels=C, bs=cfg["bs"], n=N, checkpoints=cfg["ckpt"])
          for s in range(SEEDS)]
    Bs = [train(spec["make"], spec["lr_const"], schedule="const", seq=seqp, seed=s,
                data=data, tr=tr, te=te, ac=ac, channels=C, bs=cfg["bs"], n=N)
          for s in range(SEEDS)]
    lms = lora_step_ms(spec["make"], spec["lr"])
    traj = [[As[0]["traj"][i][0], mean([a["traj"][i][1] for a in As])] for i in range(len(As[0]["traj"]))]
    return dict(
        te=mean([a["te"] for a in As]), gap=mean([a["gap"] for a in As]),
        tr=mean([a["tr"] for a in As]), ms=mean([a["ms"] for a in As]),
        bpp=mean([a["bpp"] for a in As]), cgap=mean([b["gap"] for b in Bs]),
        cte=mean([b["te"] for b in Bs]), lms=lms, traj=traj,
        family=spec["family"], blurb=spec["blurb"], sig=cfg["sig"],
    )


# ----------------------------- cache I/O -----------------------------
def store_path(quick):
    return f"{HERE}/results{'_quick' if quick else ''}.json"


def load_store(quick):
    p = store_path(quick)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def save_store(store, quick):
    with open(store_path(quick), "w") as f:
        json.dump(store, f, indent=1)


# ----------------------------- ranking helpers -----------------------------
def ranked(rows, key):
    order = sorted(rows, key=lambda n: rows[n][key])
    return {n: i + 1 for i, n in enumerate(order)}


def fmt_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# ----------------------------- render RANKINGS.md from the cache -----------------------------
def render(store, cfg, quick):
    # only rank entries measured at the active settings signature; flag the rest as stale
    active = {n: dict(m) for n, m in store.items() if m.get("sig") == cfg["sig"]}
    stale = [n for n, m in store.items() if m.get("sig") != cfg["sig"]]
    if len(active) < 2:
        print(f"render: only {len(active)} entries at sig {cfg['sig']} — need >=2 to rank.", flush=True)
        return
    # common convergence target = the worst optimizer's best held-out loss (everyone reaches it)
    T = max(min(v for _, v in m["traj"]) for m in active.values())
    for m in active.values():
        m["conv"] = next((s for s, v in m["traj"] if v <= T), m["traj"][-1][0])
        m["ttq"] = m["ms"] * m["conv"] / 1000.0
        m["dgap"] = m["cgap"] - m["gap"]

    rk = {k: ranked(active, k) for k in ("ms", "lms", "conv", "te", "gap", "bpp", "cgap")}
    comp = {n: mean([rk[k][n] for k in ("ms", "lms", "conv", "te", "gap", "bpp", "cgap")]) for n in active}

    def tbl(sortkey, header, cols):
        rows = [[f"{i+1}", n, *[c(active[n]) for c in cols]]
                for i, n in enumerate(sorted(active, key=lambda n: active[n][sortkey]))]
        return fmt_table(header, rows)

    L = ["# Optimizer control battery — rankings\n"]
    L.append(f"> Generated by [`battery.py`](battery.py) from [`results{'_quick' if quick else ''}.json`]"
             f"(results{'_quick' if quick else ''}.json). Settings: `C={cfg['C']}`, `N={cfg['N']}`, "
             f"{cfg['seeds']} seed(s), dataset fp `{cfg['fp'][:12]}`, REX d=0.9 + progressive-resolution "
             f"recipe. Each optimizer at its best config ([`registry.py`](registry.py)). **Lower is "
             f"better in every table.**\n")
    L.append("> Add a contender: drop it into `registry.py`, run `python battery.py --only <Name>`, and "
             "it joins every table. Proxy LRs are ~100x real-training LRs (relative knobs, not "
             "recommendations); metrics rank objective overfitting/convergence, not perceptual quality — "
             "confirm on a real LoRA with FID/KID.\n")

    L.append("## 🏁 Overall — but read this first\n")
    L.append("> **Mean rank is the wrong way to pick an optimizer.** These are *specialists*: a "
             "gap-champion that trades loss will rank low on loss/convergence (which are correlated) "
             "and look mediocre on the mean — yet be exactly right for small-data LoRA. The **🥇 wins** "
             "column shows where each one is rank #1; pick by the axis you care about, not the average.\n")
    friendly = {"ms": "iter-speed", "lms": "LoRA-speed", "conv": "convergence", "te": "loss",
                "gap": "generalization", "bpp": "memory", "cgap": "constant-LR"}
    wins = {n: [friendly[k] for k in friendly if rk[k][n] == 1] for n in active}
    rows = [[f"{i+1}", f"**{n}**", f"{comp[n]:.1f}", "🥇 " + ", ".join(wins[n]) if wins[n] else "—",
             active[n]["blurb"]]
            for i, n in enumerate(sorted(active, key=lambda n: comp[n]))]
    L.append(fmt_table(["#", "optimizer", "mean rank", "🥇 wins (rank 1)", "identity"], rows))

    L.append("## 🎯 Loss × generalization (scheduled, progressive curriculum)\n")
    L.append("The headline for small-data fine-tuning: rank by the **train–val gap**, not the loss.\n")
    L.append(tbl("gap", ["# (by gap)", "optimizer", "held-out loss", "train–val gap"],
                 [lambda m: f"{m['te']:.4f}", lambda m: f"{m['gap']:+.4f}"]))

    L.append(f"\n## ⏱️ Convergence speed & time×quality (target held-out loss ≤ {T:.4f})\n")
    L.append("`steps→target` = how fast it reaches the common quality bar; `time→target` folds in the "
             "per-step cost (the metric that actually matters in wall-clock).\n")
    L.append(tbl("ttq", ["# (by time×quality)", "optimizer", "steps→target", "ms/step", "time→target (s)"],
                 [lambda m: f"{m['conv']}", lambda m: f"{m['ms']:.1f}", lambda m: f"{m['ttq']:.2f}"]))

    L.append("\n## ⚡ Per-iteration speed\n")
    L.append("`ms/step` is the full-FT-like C=128 U-Net; `lora ms/step` is a 512-tiny-tensor adapter "
             "bag (launch-bound — where `foreach` batching pays off).\n")
    L.append(tbl("ms", ["# (by ms/step)", "optimizer", "ms/step (C=128)", "lora ms/step (512 tensors)"],
                 [lambda m: f"{m['ms']:.1f}", lambda m: f"{m['lms']:.2f}"]))

    L.append("\n## 💾 Memory (measured optimizer state)\n")
    L.append(tbl("bpp", ["# (by B/param)", "optimizer", "optimizer state (B/param)"],
                 [lambda m: f"{m['bpp']:.2f}"]))

    L.append("\n## 🔁 Continuity — robustness at constant LR (resumable, no schedule)\n")
    L.append("`const gap` is the train–val gap with **no scheduler**; `Δ vs sched` ≤ 0 means the "
             "optimizer *keeps* (or improves) its generalization without the decaying-schedule crutch "
             "— the property you want for open-ended / resumable runs.\n")
    L.append(tbl("cgap", ["# (by const gap)", "optimizer", "const-LR gap", "Δ vs scheduled"],
                 [lambda m: f"{m['cgap']:+.4f}", lambda m: f"{m['dgap']:+.4f}"]))

    if stale:
        L.append(f"\n> ⚠️ Not shown (measured at different settings — re-run to include): "
                 f"{', '.join(sorted(stale))}.\n")

    with open(f"{HERE}/RANKINGS.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"render: wrote RANKINGS.md ({len(active)} optimizers, target≤{T:.4f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None, help="comma-sep optimizer names to (re)measure")
    ap.add_argument("--new", action="store_true", help="measure only optimizers missing from the cache")
    ap.add_argument("--render-only", action="store_true", help="rebuild RANKINGS.md from cache; no training")
    ap.add_argument("--quick", action="store_true", help="smaller/faster settings (own cache)")
    A = ap.parse_args()

    C = 96 if A.quick else 128
    N = 800 if A.quick else 2000
    SEEDS = 1 if A.quick else 2
    ds = D.build_proxy_dataset()
    fp = D.fingerprint(ds)
    cfg = dict(C=C, N=N, seeds=SEEDS, bs=8, ckpt=16, fp=fp,
               sig=f"C{C}_N{N}_s{SEEDS}_{fp[:8]}")
    store = load_store(A.quick)

    if not A.render_only:
        data = {k: v.to(DEV).to(H.DT) for k, v in ds["DATA"].items()}
        tr, te = ds["TR"], ds["TE"]
        ac = H.make_alphas()
        seqp = seq_prog(N)
        if A.only:
            targets = [n.strip() for n in A.only.split(",")]
        elif A.new:
            targets = [n for n in OPTIMIZERS if store.get(n, {}).get("sig") != cfg["sig"]]
        else:
            targets = list(OPTIMIZERS)
        print(f"BATTERY sig={cfg['sig']} | measuring: {targets or '(none)'}", flush=True)
        for name in targets:
            if name not in OPTIMIZERS:
                print(f"  {name}: not in registry — skipped", flush=True); continue
            try:
                store[name] = measure(name, OPTIMIZERS[name], cfg, data, tr, te, ac, seqp)
                m = store[name]
                print(f"  {name:16s} te={m['te']:.4f} gap={m['gap']:+.4f} {m['ms']:.1f}ms "
                      f"{m['bpp']:.2f}B/p lora={m['lms']:.2f}ms const_gap={m['cgap']:+.4f}", flush=True)
                save_store(store, A.quick)  # incremental: persist after each optimizer
            except Exception as e:  # noqa: BLE001
                print(f"  {name:16s} FAILED: {type(e).__name__}: {e}", flush=True)

    render(store, cfg, A.quick)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
