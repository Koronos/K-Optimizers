"""The exploration battery — find the best *configuration* of ONE optimizer.

Where ``battery.py`` ranks optimizers against each other, this script sweeps the knobs of a single
optimizer (LR, scheduler, warmup, scheduler-shape) on the reproducible proxy and tells you how to set
them. It always reports BOTH **loss** (held-out te) and **generalization** (the train–val gap) so you
pick by the axis you care about — and it always sweeps **torch.AdamW** alongside as the reference, so
the LR answer comes out as a multiple of AdamW's optimum ("AdaPNM wants ~Nx AdamW's LR").

Like the benchmark battery it is **incremental + cached**: results dump to ``explore_results.json``;
re-running renders ``EXPLORE.md`` (per-optimizer tables, one per axis, sorted, the pick marked ⭐).

    python explore.py --opt AdaPNM            # full sweep for AdaPNM (+ AdamW reference)
    python explore.py --opt AdaPNM --only lr  # just the LR axis (axes: lr, scheduler, warmup)
    python explore.py --render-only           # rebuild EXPLORE.md from the cache, no training

Axes swept (each varies ONE knob around the base config — coordinate-style, so it stays tractable and
each axis's options are directly comparable):
  * **lr**        — LR ∈ {0.25, 0.5, 1, 2, 4}× the optimizer's proxy base LR (constant sched, no warmup)
  * **scheduler** — {constant, cosine, cosine-floor10, rex, linear} (at base LR, no warmup)
  * **warmup**    — {0, 50, 100, 200} linear-ramp steps (constant sched, base LR)

Proxy LRs are ~100x real-training LRs (relative knobs, not recommendations); the table's value is the
SHAPE of each curve and the AdamW-relative LR, not the absolute number. Confirm on a real LoRA + FID.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os

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
DEV = H.DEV


# ----------------------------- the optimizers we know how to explore -----------------------------
# make(params, lr) at the optimizer's best non-LR config; base_lr = the proxy LR the sweep centers on.
# AdamW is always swept too (the relative-LR reference). Add an entry to explore another optimizer.
def _kaon(cls, **kw):
    from kaon import ADOPT, AdaBelief, Adakaon, AdamP, AdaMuon, AdaPNM, Lion  # noqa: F401
    return {
        "Adakaon": lambda p, lr: Adakaon(p, lr=lr, betas=(0.9, 0.999), cautious=True, momentum_dtype="bfloat16"),
        "AdaPNM": lambda p, lr: AdaPNM(p, lr=lr, betas=(0.8, 0.999), beta0=0.5, cautious=True, momentum_dtype="bfloat16"),
        "Lion": lambda p, lr: Lion(p, lr=lr, betas=(0.95, 0.98), cautious=True, momentum_dtype="bfloat16"),
        "ADOPT": lambda p, lr: ADOPT(p, lr=lr, betas=(0.9, 0.9999), cautious=True, momentum_dtype="bfloat16"),
        "AdaBelief": lambda p, lr: AdaBelief(p, lr=lr, betas=(0.9, 0.999), cautious=True, momentum_dtype="bfloat16"),
        "AdamP": lambda p, lr: AdamP(p, lr=lr, weight_decay=0.05, cautious=True, momentum_dtype="bfloat16"),
        "AdaMuon": lambda p, lr: AdaMuon(p, lr=lr, betas=(0.95, 0.999), ns_steps=2, cautious=True, momentum_dtype="int8"),
    }[cls]


OPTS: dict[str, dict] = {
    "AdamW": dict(make=lambda p, lr: torch.optim.AdamW(p, lr=lr, betas=(0.9, 0.999)), base_lr=1.2e-3),
    "Adakaon": dict(make=_kaon("Adakaon"), base_lr=1.2e-3),
    "AdaPNM": dict(make=_kaon("AdaPNM"), base_lr=2.4e-3),
    "Lion": dict(make=_kaon("Lion"), base_lr=3e-4),
    "ADOPT": dict(make=_kaon("ADOPT"), base_lr=4e-3),
    "AdaBelief": dict(make=_kaon("AdaBelief"), base_lr=1e-3),
    "AdamP": dict(make=_kaon("AdamP"), base_lr=1e-3),
    "AdaMuon": dict(make=_kaon("AdaMuon"), base_lr=2.4e-3),
}

LR_MULTS = (0.25, 0.5, 1.0, 2.0, 4.0)
SCHEDULES = ("constant", "cosine", "cosine-floor10", "rex", "linear")
WARMUPS = (0, 50, 100, 200)


# ----------------------------- schedule shapes (warmup ∘ decay) -----------------------------
def _rex(p, d=0.9):
    z = 1 - p
    return z / ((1 - d) + d * z)


def sched_mult(name: str, it: int, n: int, warmup: int) -> float:
    """LR multiplier in [0,1]: a linear warmup ramp over the first ``warmup`` steps, times the decay."""
    p = it / n
    w = min(1.0, (it + 1) / warmup) if warmup > 0 and it < warmup else 1.0
    if name == "constant":
        base = 1.0
    elif name == "cosine":
        base = 0.5 * (1.0 + math.cos(math.pi * p))
    elif name == "cosine-floor10":
        base = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * p))   # cosine that floors at 10% LR
    elif name == "rex":
        base = _rex(p, 0.9)
    elif name == "linear":
        base = 1.0 - p
    else:
        raise ValueError(name)
    return w * base


# ----------------------------- one training run -----------------------------
def train(make, lr, *, scheduler, warmup, seq, seed, data, tr, te, ac, channels, bs, n):
    torch.manual_seed(seed)
    if DEV == "cuda":
        torch.cuda.manual_seed_all(seed)
    net = H.UNet(C=channels).to(DEV).to(H.DT)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = make(params, lr)
    g = torch.Generator(device=DEV)
    g.manual_seed(seed + 12345)
    pos = 0
    naninf = False
    for it, Rr in enumerate(seq):
        for pg in opt.param_groups:
            pg["lr"] = lr * sched_mult(scheduler, it, n, warmup)
        idx = [tr[(pos + j) % len(tr)] for j in range(bs)]
        pos += bs
        idxt = torch.tensor(idx, device=DEV)
        opt.zero_grad()
        loss = H.batch_loss(net, data[Rr], idxt, ac, g)
        loss.backward()
        if not torch.isfinite(loss):
            naninf = True
            break
        opt.step()
    trl = H.eval_loss(net, data[64], tr, ac)
    tel = H.eval_loss(net, data[64], te, ac)
    return dict(tr=trl, te=tel, gap=tel - trl, naninf=naninf)


def mean(xs):
    return sum(xs) / len(xs)


# ----------------------------- the per-axis config grid -----------------------------
def configs_for(opt_name: str, base_lr: float) -> list[dict]:
    """The coordinate sweep: vary ONE knob per config around (constant, base_lr, no-warmup)."""
    cfgs = []
    for m in LR_MULTS:                       # LR axis
        cfgs.append(dict(axis="lr", label=f"lr×{m:g}", scheduler="constant", lr=base_lr * m, warmup=0, lr_mult=m))
    for s in SCHEDULES:                      # scheduler axis (at base LR)
        cfgs.append(dict(axis="scheduler", label=s, scheduler=s, lr=base_lr, warmup=0, lr_mult=1.0))
    for w in WARMUPS:                        # warmup axis (constant, base LR)
        cfgs.append(dict(axis="warmup", label=f"warmup={w}", scheduler="constant", lr=base_lr, warmup=w, lr_mult=1.0))
    return cfgs


# ----------------------------- measure one optimizer's whole sweep -----------------------------
def measure_opt(opt_name, cfg_meta, data, tr, te, ac, seqp, settings, only_axis=None):
    base_lr = OPTS[opt_name]["base_lr"]
    make = OPTS[opt_name]["make"]
    out = {}
    for c in configs_for(opt_name, base_lr):
        if only_axis and c["axis"] != only_axis:
            continue
        rs = [train(make, c["lr"], scheduler=c["scheduler"], warmup=c["warmup"], seq=seqp, seed=s,
                    data=data, tr=tr, te=te, ac=ac, channels=cfg_meta["C"], bs=cfg_meta["bs"], n=cfg_meta["N"])
              for s in range(cfg_meta["seeds"])]
        out[c["label"]] = dict(
            axis=c["axis"], scheduler=c["scheduler"], lr=c["lr"], lr_mult=c["lr_mult"], warmup=c["warmup"],
            te=mean([r["te"] for r in rs]), gap=mean([r["gap"] for r in rs]), tr=mean([r["tr"] for r in rs]),
            naninf=any(r["naninf"] for r in rs), sig=settings,
        )
        m = out[c["label"]]
        print(f"  {opt_name:10s} {c['label']:14s} te={m['te']:.4f} gap={m['gap']:+.4f}"
              f"{'  NaN!' if m['naninf'] else ''}", flush=True)
    return out


# ----------------------------- cache + render -----------------------------
def store_path():
    return f"{HERE}/explore_results.json"


def load_store():
    return json.load(open(store_path())) if os.path.exists(store_path()) else {}


def save_store(store):
    with open(store_path(), "w") as f:
        json.dump(store, f, indent=1)


def _tbl(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def render(store, settings):
    L = ["# Optimizer configuration explorer\n"]
    L.append(f"> Generated by [`explore.py`](explore.py) from [`explore_results.json`](explore_results.json). "
             f"Settings: `{settings}`. Each row varies ONE knob; **both loss (held-out `te`) and "
             f"generalization (`gap` = te − tr) are shown — pick by the axis you care about** (the gap is "
             f"the small-data fine-tuning signal). ⭐ = best on the axis (by gap). Proxy LRs are ~100x "
             f"real-training LRs — read the SHAPE and the ×AdamW ratio, not the absolute number.\n")
    adamw = store.get("AdamW", {})
    adamw_best_lr = None
    if adamw:
        lr_rows = {k: v for k, v in adamw.items() if v["axis"] == "lr" and not v["naninf"]}
        if lr_rows:
            adamw_best_lr = min(lr_rows.values(), key=lambda v: v["gap"])["lr"]
    for opt_name, res in store.items():
        if opt_name == "AdamW" and len(store) > 1:
            continue  # AdamW is the reference; shown inline via the ×AdamW column
        L.append(f"\n## {opt_name}\n")
        for axis, title, note in (
            ("lr", "Learning rate", "best `te`/`gap` LR; the `×AdamW` column is this LR ÷ AdamW's best-gap LR"),
            ("scheduler", "Scheduler shape", "constant vs decaying — does it need a schedule, and which?"),
            ("warmup", "Warmup", "linear-ramp steps — does warmup help, hurt, or do nothing?"),
        ):
            rows_d = {k: v for k, v in res.items() if v["axis"] == axis}
            if not rows_d:
                continue
            best = min((v for v in rows_d.values() if not v["naninf"]), key=lambda v: v["gap"], default=None)
            L.append(f"### {title}\n_{note}_\n")
            header = ["", "te (loss)", "gap (generalization)"]
            if axis == "lr":
                header += ["LR", "×AdamW"]
            rows = []
            for label, v in sorted(rows_d.items(), key=lambda kv: kv[1]["gap"]):
                star = " ⭐" if v is best else ""
                nan = " ⚠️NaN" if v["naninf"] else ""
                row = [f"{label}{star}{nan}", f"{v['te']:.4f}", f"{v['gap']:+.4f}"]
                if axis == "lr":
                    ratio = f"{v['lr'] / adamw_best_lr:.2g}×" if adamw_best_lr else "—"
                    row += [f"{v['lr']:.1e}", ratio]
                rows.append(row)
            L.append(_tbl(rows, header) + "\n")
        # one-line recommendation
        rec = {}
        for axis in ("lr", "scheduler", "warmup"):
            rr = {k: v for k, v in res.items() if v["axis"] == axis and not v["naninf"]}
            if rr:
                rec[axis] = min(rr.items(), key=lambda kv: kv[1]["gap"])
        if rec:
            lr_pick = rec.get("lr")
            ratio = (f" (~{lr_pick[1]['lr'] / adamw_best_lr:.2g}× AdamW)"
                     if lr_pick and adamw_best_lr else "")
            parts = []
            if "scheduler" in rec:
                parts.append(f"scheduler **{rec['scheduler'][0]}**")
            if lr_pick:
                parts.append(f"LR **{lr_pick[0]}**{ratio}")
            if "warmup" in rec:
                parts.append(f"**{rec['warmup'][0]}**")
            L.append(f"> **Lowest-gap pick:** " + ", ".join(parts) + ".\n")
    with open(f"{HERE}/EXPLORE.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"render: wrote EXPLORE.md ({len(store)} optimizer(s))", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", type=str, default=None, help="optimizer name to explore (always +AdamW reference)")
    ap.add_argument("--only", type=str, default=None, help="single axis: lr | scheduler | warmup")
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--quick", action="store_true", help="smaller/faster (smoke)")
    A = ap.parse_args()

    C = 96 if A.quick else 128
    N = 800 if A.quick else 1500
    SEEDS = 2
    cfg_meta = dict(C=C, N=N, seeds=SEEDS, bs=8)
    settings = f"C{C}_N{N}_s{SEEDS}"
    store = load_store()

    if not A.render_only:
        ds = D.build_proxy_dataset()
        data = {k: v.to(DEV).to(H.DT) for k, v in ds["DATA"].items()}
        tr, te = ds["TR"], ds["TE"]
        ac = H.make_alphas()
        seqp = _load("battery", f"{HERE}/battery.py").seq_prog(N)
        targets = [A.opt] if A.opt else ["AdaPNM"]
        if "AdamW" not in targets:
            targets.append("AdamW")              # always sweep the relative-LR reference
        print(f"EXPLORE {settings} | optimizers: {targets} | axis: {A.only or 'all'}", flush=True)
        for name in targets:
            if name not in OPTS:
                print(f"  {name}: unknown (add to OPTS) — skipped", flush=True)
                continue
            merged = store.get(name, {})
            merged.update(measure_opt(name, cfg_meta, data, tr, te, ac, seqp, settings, only_axis=A.only))
            store[name] = merged
            save_store(store)

    render(store, settings)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
