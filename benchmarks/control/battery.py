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
    python battery.py --seeds 5       # override the seed count (enters the settings signature)
    python battery.py --out evidence  # cache/rankings -> results_evidence.json / RANKINGS_evidence.md

Every measured entry keeps the INDIVIDUAL per-seed runs (``per_seed``) behind each average, so the
quality tables can carry a 95% confidence interval instead of a bare point estimate. Measuring
aborts unless the laptop is on AC power (60 W vs 35 W GPU caps make timings incomparable); the
electrical state is stamped into every entry and into the rankings header.

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
import datetime
import importlib.util
import json
import os
import random
import shutil
import subprocess
import time

import torch

# Resolve paths relative to THIS file so the battery works in a git worktree (reads the worktree's
# registry/harness and writes its own results.json/RANKINGS.md) instead of always hitting main.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# Steps to skip before the ms/step clock starts (one-time JIT/cuDNN/CUDA-init warmup; see train()).
TIMING_WARMUP = 10


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
def evald(opt, fn):
    """Evaluate ``fn()`` at the optimizer's EVAL weights. Optimizers that keep an averaged /
    perturbed view (ScheduleFree's x, Lookahead's phi, MSAM's unperturbed w) expose
    ``eval()``/``train()``; bracket the measurement so they are scored on the weights a real
    run would sample/checkpoint. A plain optimizer (no ``eval`` attr) measures unchanged."""
    swap = hasattr(opt, "eval") and hasattr(opt, "train")
    if swap:
        opt.eval()
    out = fn()
    if swap:
        opt.train()
    return out


def train(make, lr, *, schedule, seq, seed, data, tr, te, ac, channels, bs, n, checkpoints=0):
    torch.manual_seed(seed)
    if DEV == "cuda":
        torch.cuda.manual_seed_all(seed)
    net = H.UNet(C=channels).to(DEV).to(H.DT)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = make(params, lr)
    # Preserve each group's own base lr (relative to `lr`) through the schedule — see the
    # matching fix/comment in profiler.py's run(); a flat `lr * mult` overwrite would
    # clobber any per-group lr ratio (e.g. Nekaon's low_vram_lr_ratio).
    base_lrs = [pg["lr"] for pg in opt.param_groups]
    g = torch.Generator(device=DEV); g.manual_seed(seed + 12345)
    pos = 0; traj = []
    ckpt_every = max(1, n // checkpoints) if checkpoints else 0
    # Time STEADY-STATE ms/step: start the clock after a short warmup so the one-time costs (Triton
    # JIT compilation of the fused kernels, cuDNN autotuning, lazy CUDA init) are NOT amortized into
    # the per-step number. Timing the whole loop made a fused optimizer look ~15% slower purely from
    # its first-step JIT spread over n — an artifact, not a real per-step regression. All n steps
    # still run (training is unchanged); only the timing window excludes the warmup.
    warm = min(TIMING_WARMUP, max(0, n - 1))
    t0 = None
    timed = 0
    for it, Rr in enumerate(seq):
        if it == warm:
            if DEV == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
        mult = rex(it / n) if schedule == "rex" else 1.0
        for pg, base in zip(opt.param_groups, base_lrs, strict=True):
            pg["lr"] = base * mult
        idx = [tr[(pos + j) % len(tr)] for j in range(bs)]; pos += bs
        opt.zero_grad()
        loss = H.batch_loss(net, data[Rr], torch.tensor(idx, device=DEV), ac, g)
        loss.backward()
        opt.step()
        if it >= warm:
            timed += 1
        if ckpt_every and (it + 1) % ckpt_every == 0:
            traj.append((it + 1, evald(opt, lambda: H.eval_loss(net, data[64], te, ac))))
    if DEV == "cuda":
        torch.cuda.synchronize()
    ms_step = (time.time() - t0) / max(timed, 1) * 1000.0
    tr_loss, te_loss = evald(
        opt, lambda: (H.eval_loss(net, data[64], tr, ac), H.eval_loss(net, data[64], te, ac))
    )
    bpp = H.opt_state_bytes_per_param(opt, params)
    # Optimizer-ONLY cost, probed after every metric above is computed (it steps the optimizer a few
    # more times, so it must never precede the evals). `ms` is fwd+bwd+step of the C=128 proxy and,
    # on a model this small, the optimizer dominates it; `opt_ms` isolates `opt.step()` so the
    # fraction is reportable. It never feeds back into `ms` -- that number is computed as before.
    opt_ms = opt_step_ms(opt)
    return dict(tr=tr_loss, te=te_loss, gap=te_loss - tr_loss, ms=ms_step, bpp=bpp,
                opt_ms=opt_ms, traj=traj)


def opt_step_ms(opt, reps=20, warmup=3):
    """Median ms of a bare ``opt.step()`` (sync-bracketed), reusing the grads and optimizer state
    ``train`` left behind -- real parameter shapes, real state, no fwd/bwd in the number."""
    for _ in range(warmup):
        opt.step()
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


def stdev(xs):
    """Sample standard deviation (n-1 denominator); 0.0 for a single point."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5


# Two-sided 95% Student-t critical values by degrees of freedom (n-1). Beyond the table we fall back
# to the normal approximation (1.96) -- past df=30 the difference is under 1.5%.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
        23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def ci95(xs):
    """Half-width of the 95% confidence interval of the MEAN of ``xs`` (Student-t, n-1 d.o.f.).

    Returns ``None`` for n < 2: one seed carries no interval, and neither do the legacy cache entries
    that only stored averages -- callers then render a bare mean, exactly as before.
    """
    xs = [x for x in (xs or []) if x is not None]
    n = len(xs)
    if n < 2:
        return None
    return _T95.get(n - 1, 1.96) * stdev(xs) / (n ** 0.5)


def entry_ci(m, key):
    """95% CI half-width for metric ``key`` of a cache entry; None if it has no per-seed data."""
    return ci95((m.get("per_seed") or {}).get(key))


def overlaps(a, b, key):
    """Do the two entries' 95% intervals for ``key`` overlap? False when either lacks per-seed data:
    a legacy entry makes no statistical claim, so it is never declared a tie."""
    ca, cb = entry_ci(a, key), entry_ci(b, key)
    if ca is None or cb is None:
        return False
    return a[key] - ca <= b[key] + cb and a[key] + ca >= b[key] - cb


# ----------------------------- run metadata & electrical state -----------------------------
def _run_cmd(cmd):
    """Best-effort capture of a short command's stdout; None if the tool is missing or it fails."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception:  # noqa: BLE001 -- metadata must never fail a measurement
        return None
    return (r.stdout.strip() or None) if r.returncode == 0 else None


def battery_status():
    """``Win32_Battery.BatteryStatus`` -- 2 means "running on AC". None when unreadable."""
    out = _run_cmd(["powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_Battery).BatteryStatus"])
    for tok in (out or "").split():
        try:
            return int(tok)
        except ValueError:
            continue
    return None


def power_state():
    """GPU identity + the electrical envelope the numbers are taken under.

    This laptop caps the GPU at 60 W on AC and 35 W on battery: the same optimizer measures far
    apart across that boundary, so timings from the two regimes must never be mixed.
    """
    st = battery_status()
    return dict(
        gpu=_run_cmd(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]),
        power=_run_cmd(["nvidia-smi",
                        "--query-gpu=power.limit,power.max_limit,power.default_limit,clocks.max.sm",
                        "--format=csv,noheader"]),
        battery_status=st,
        on_ac=(st == 2),
    )


def require_ac(ps=None):
    """Abort BEFORE measuring anything unless the laptop is on AC power (see ``power_state``)."""
    ps = power_state() if ps is None else ps
    if ps.get("on_ac"):
        return ps
    if os.environ.get("KAON_BATTERY_ALLOW_DC") == "1":
        print("WARNING: not on AC -- measuring anyway (KAON_BATTERY_ALLOW_DC=1). These timings are "
              "NOT comparable to AC-measured entries.", flush=True)
        return ps
    raise SystemExit(
        f"ABORT: the control battery must be measured on AC power "
        f"(Win32_Battery.BatteryStatus={ps.get('battery_status')}, 2 = AC). On battery the GPU runs "
        f"a 35 W cap instead of 60 W and every ms/step becomes incomparable. Plug in and re-run "
        f"(or set KAON_BATTERY_ALLOW_DC=1 to override knowingly)."
    )


def kaon_version():
    try:
        import kaon
        return getattr(kaon, "__version__", None)
    except Exception:  # noqa: BLE001
        return None


def run_metadata(ps=None):
    """Stamp WHEN / WITH WHAT / UNDER WHICH POWER an entry was measured, so a cache row can be
    audited long after the fact and mixed-epoch timings can be spotted."""
    ps = power_state() if ps is None else ps
    return dict(
        timestamp=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        kaon_version=kaon_version(),
        commit=_run_cmd(["git", "-C", REPO, "rev-parse", "--short", "HEAD"]),
        device=DEV,
        gpu=ps.get("gpu"),
        power=ps.get("power"),
        battery_status=ps.get("battery_status"),
        on_ac=ps.get("on_ac"),
    )


def _jsonsafe(v):
    """Coerce an optimizer hyperparameter to a JSON-able scalar/list (tuples->lists; enums/dtypes->str)."""
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, (tuple, list)):
        return [_jsonsafe(x) for x in v]
    return str(v)


def optimizer_config(make, lr, spec):
    """Self-describe HOW a result was generated: snapshot the actual hyperparameters of the built
    optimizer (its ``param_groups[0]`` + ``defaults``, minus the param tensors) so ``results.json`` is
    reproducible on its own — no need to keep the registry lambda around. Works for any optimizer
    (incl. wrappers like SAM/Lookahead) via introspection; never fails the run if it can't read one."""
    cfg = {"class": None, "lr": spec.get("lr"), "lr_const": spec.get("lr_const")}
    try:
        probe = make([torch.zeros(2, 2, device=DEV, requires_grad=True)], lr)
        cfg["class"] = type(probe).__name__
        src = {}
        src.update(getattr(probe, "defaults", {}) or {})
        if getattr(probe, "param_groups", None):
            src.update(probe.param_groups[0])
        for k, v in src.items():
            if k in ("params", "step"):  # tensors / mutable state, not config
                continue
            cfg[k] = _jsonsafe(v)
    except Exception as e:  # introspection is best-effort; a result is still valid without it
        cfg["introspect_error"] = f"{type(e).__name__}: {e}"
    return cfg


# ----------------------------- measure ONE optimizer -----------------------------
def measure(name, spec, cfg, data, tr, te, ac, seqp, meta=None):
    """Run the A (scheduled+prog), B (constant+prog), and lora-bag probes for one optimizer.

    Returns the cache entry (plain JSON-able dict, including the per-checkpoint trajectory). Besides
    the seed-averaged headline numbers every consumer reads, the entry keeps ``per_seed``: the
    INDIVIDUAL runs behind each average. Without them a 2-seed mean is a point estimate with no
    spread and no table can say whether a 0.001 difference is real -- see ``ci95`` and ``render``.
    """
    C, N, SEEDS = cfg["C"], cfg["N"], cfg["seeds"]
    As = [train(spec["make"], spec["lr"], schedule="rex", seq=seqp, seed=s,
                data=data, tr=tr, te=te, ac=ac, channels=C, bs=cfg["bs"], n=N, checkpoints=cfg["ckpt"])
          for s in range(SEEDS)]
    # Deliberately the SAME seeds as the REX phase: `train` seeds init, batch order and the
    # diffusion noise, so run B is run A with only the LR schedule changed. The A/B pair is
    # therefore paired, not two independent samples -- which is what makes `cgap` readable as
    # "what the constant-LR regime does to THIS optimizer" instead of seed noise.
    Bs = [train(spec["make"], spec["lr_const"], schedule="const", seq=seqp, seed=s,
                data=data, tr=tr, te=te, ac=ac, channels=C, bs=cfg["bs"], n=N)
          for s in range(SEEDS)]
    lms = lora_step_ms(spec["make"], spec["lr"])
    traj = [[As[0]["traj"][i][0], mean([a["traj"][i][1] for a in As])] for i in range(len(As[0]["traj"]))]
    per_seed = dict(
        seeds=list(range(SEEDS)),
        te=[a["te"] for a in As], gap=[a["gap"] for a in As], tr=[a["tr"] for a in As],
        ms=[a["ms"] for a in As], opt_ms=[a["opt_ms"] for a in As], bpp=[a["bpp"] for a in As],
        cte=[b["te"] for b in Bs], cgap=[b["gap"] for b in Bs], ctr=[b["tr"] for b in Bs],
        traj=[[[s, v] for s, v in a["traj"]] for a in As],
    )
    entry = dict(
        te=mean([a["te"] for a in As]), gap=mean([a["gap"] for a in As]),
        tr=mean([a["tr"] for a in As]), ms=mean([a["ms"] for a in As]),
        opt_ms=mean([a["opt_ms"] for a in As]),
        bpp=mean([a["bpp"] for a in As]), cgap=mean([b["gap"] for b in Bs]),
        cte=mean([b["te"] for b in Bs]), lms=lms, traj=traj, per_seed=per_seed,
        family=spec["family"], blurb=spec["blurb"], sig=cfg["sig"],
        config=optimizer_config(spec["make"], spec["lr"], spec),
    )
    if meta:
        entry["meta"] = meta
    return entry


# ----------------------------- cache I/O -----------------------------
def store_path(quick, tag=None):
    """Cache file for this run. A ``--out TAG`` run lives entirely on the side
    (``results_TAG.json``), so a fresh evidence battery never touches the historical cache."""
    if tag:
        return f"{HERE}/results_{tag}.json"
    return f"{HERE}/results{'_quick' if quick else ''}.json"


def rankings_path(tag=None):
    return f"{HERE}/RANKINGS_{tag}.md" if tag else f"{HERE}/RANKINGS.md"


def load_store(quick, tag=None):
    p = store_path(quick, tag)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def save_store(store, quick, tag=None):
    with open(store_path(quick, tag), "w") as f:
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
def render(store, cfg, quick, tag=None):
    # only rank entries measured at the active settings signature; flag the rest as stale
    active = {n: dict(m) for n, m in store.items() if m.get("sig") == cfg["sig"]}
    stale = [n for n, m in store.items() if m.get("sig") != cfg["sig"]]
    if len(active) < 2:
        print(f"render: only {len(active)} entries at sig {cfg['sig']} - need >=2 to rank.", flush=True)
        return
    # common convergence target = the worst optimizer's best held-out loss (everyone reaches it)
    T = max(min(v for _, v in m["traj"]) for m in active.values())
    for m in active.values():
        m["conv"] = next((s for s, v in m["traj"] if v <= T), m["traj"][-1][0])
        m["ttq"] = m["ms"] * m["conv"] / 1000.0
        m["dgap"] = m["cgap"] - m["gap"]

    rk = {k: ranked(active, k) for k in ("ms", "lms", "conv", "te", "gap", "bpp", "cgap")}
    comp = {n: mean([rk[k][n] for k in ("ms", "lms", "conv", "te", "gap", "bpp", "cgap")]) for n in active}

    def pm(m, key, prec=4, sign=False):
        """`mean ± CI95` when the entry kept its per-seed runs, else the bare mean (legacy format)."""
        v = m[key]
        txt = f"{v:+.{prec}f}" if sign else f"{v:.{prec}f}"
        c = entry_ci(m, key)
        return txt if c is None else f"{txt} ± {c:.{prec}f}"

    def tbl(sortkey, header, cols, tie_key=None):
        names = sorted(active, key=lambda n: active[n][sortkey])
        rows = []
        for i, n in enumerate(names):
            tie = bool(tie_key) and i > 0 and overlaps(active[names[0]], active[n], tie_key)
            rows.append([f"{i+1} ≈" if tie else f"{i+1}", n, *[c(active[n]) for c in cols]])
        return fmt_table(header, rows)

    # Entries measured before per-seed retention carry no interval: every ±/tie annotation below is
    # opt-in per row, so a cache of legacy entries renders byte-identical to what it always did.
    has_ci = any(entry_ci(m, "gap") is not None or entry_ci(m, "te") is not None
                 for m in active.values())
    metas = [m["meta"] for m in active.values() if m.get("meta")]
    store_name = os.path.basename(store_path(quick, tag))

    L = ["# Optimizer control battery — rankings\n"]
    L.append(f"> Generated by [`battery.py`](battery.py) from [`{store_name}`]"
             f"({store_name}). Settings: `C={cfg['C']}`, `N={cfg['N']}`, "
             f"{cfg['seeds']} seed(s), dataset fp `{cfg['fp'][:12]}`, REX d=0.9 + progressive-resolution "
             f"recipe. Each optimizer at its best config ([`registry.py`](registry.py)). **Lower is "
             f"better in every table.**\n")
    L.append("> Add a contender: drop it into `registry.py`, run `python battery.py --only <Name>`, and "
             "it joins every table. Proxy LRs are ~100x real-training LRs (relative knobs, not "
             "recommendations); metrics rank objective overfitting/convergence, not perceptual quality — "
             "confirm on a real LoRA with FID/KID.\n")

    if has_ci:
        L.append("> **Read the intervals, not the ranks.** Quality cells show `mean ± half-width of the "
                 "95% CI` over the seeds (Student-t, n-1 d.o.f.). A rank marked **≈** overlaps the "
                 "leader's interval: *tied within the CI*, not beaten. Rows measured before per-seed "
                 "retention show a bare mean and are never marked.\n")
    if metas:
        last = max(metas, key=lambda m: m.get("timestamp") or "")
        ac = "AC" if last.get("on_ac") else f"BATTERY (BatteryStatus={last.get('battery_status')})"
        L.append(f"> ⚡ Measured on **{ac}** — {last.get('gpu') or 'unknown GPU'}, "
                 f"`power.limit,power.max_limit,power.default_limit,clocks.max.sm = "
                 f"{last.get('power') or 'n/a'}`. kaon `{last.get('kaon_version') or '?'}` @ "
                 f"`{last.get('commit') or '?'}`, last entry {last.get('timestamp') or '?'}. "
                 f"**Timings taken under different power caps are not comparable** (this laptop runs "
                 f"60 W on AC vs 35 W on battery).\n")

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
                 [lambda m: pm(m, "te"), lambda m: pm(m, "gap", sign=True)], tie_key="gap"))

    L.append(f"\n## ⏱️ Convergence speed & time×quality (target held-out loss ≤ {T:.4f})\n")
    L.append("`steps→target` = how fast it reaches the common quality bar; `time→target` folds in the "
             "per-step cost (the metric that actually matters in wall-clock).\n")
    L.append(tbl("ttq", ["# (by time×quality)", "optimizer", "steps→target", "ms/step", "time→target (s)"],
                 [lambda m: f"{m['conv']}", lambda m: f"{m['ms']:.1f}", lambda m: f"{m['ttq']:.2f}"]))

    L.append("\n## ⚡ Per-iteration speed\n")
    speed_hdr = ["# (by ms/step)", "optimizer", "ms/step (C=128)", "lora ms/step (512 tensors)"]
    speed_cols = [lambda m: f"{m['ms']:.1f}", lambda m: f"{m['lms']:.2f}"]
    has_opt_ms = any(m.get("opt_ms") is not None for m in active.values())
    if has_opt_ms:
        speed_hdr.append("opt-only ms/step (% of ms)")
        speed_cols.append(lambda m: ("—" if m.get("opt_ms") is None or not m.get("ms") else
                                     f"{m['opt_ms']:.2f} ({100 * m['opt_ms'] / m['ms']:.0f}%)"))
    L.append("`ms/step` is the full-FT-like C=128 U-Net; `lora ms/step` is a 512-tiny-tensor adapter "
             "bag (launch-bound — where `foreach` batching pays off).\n")
    if has_opt_ms:
        L.append("`ms/step` is fwd+bwd+**step** — on a proxy this small the optimizer dominates it, so "
                 "`opt-only ms/step` (a bare sync-bracketed `opt.step()` on the same weights) is shown "
                 "with its share of `ms/step`: read a step-cost delta against THAT column.\n")
    L.append(tbl("ms", speed_hdr, speed_cols))

    L.append("\n## 💾 Memory (measured optimizer state)\n")
    L.append(tbl("bpp", ["# (by B/param)", "optimizer", "optimizer state (B/param)"],
                 [lambda m: f"{m['bpp']:.2f}"]))

    L.append("\n## 🔁 Continuity — robustness at constant LR (resumable, no schedule)\n")
    L.append("`const loss` is the held-out loss at **constant LR**; `const gap` is the train–val gap "
             "there; `Δ vs sched` ≤ 0 means the optimizer *keeps* (or improves) its generalization "
             "without the decaying-schedule crutch — the property you want for open-ended / resumable "
             "runs. **Read loss AND gap together**: a tight gap on a high loss is consistent underfitting, "
             "not quality (e.g. AdaPNM nails the gap but collapses on const-LR loss).\n")
    L.append(tbl("cgap", ["# (by const gap)", "optimizer", "const-LR loss", "const-LR gap", "Δ vs scheduled"],
                 [lambda m: pm(m, "cte"), lambda m: pm(m, "cgap", sign=True),
                  lambda m: f"{m['dgap']:+.4f}"], tie_key="cgap"))

    if stale:
        L.append(f"\n> ⚠️ Not shown (measured at different settings — re-run to include): "
                 f"{', '.join(sorted(stale))}.\n")

    out = rankings_path(tag)
    # explicit utf-8: the tables are full of emoji/en-dashes and a Windows default (cp1252) console
    # locale raises UnicodeEncodeError mid-write, truncating the file it was regenerating.
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"render: wrote {os.path.basename(out)} ({len(active)} optimizers, target<={T:.4f})", flush=True)


def build_cfg(quick, fp, seeds=None):
    """The settings of a run + the signature that decides which cached entries may be ranked
    TOGETHER. ``seeds`` is part of it: a 5-seed measurement is a different (stronger) claim than a
    2-seed one, so the two never share a table."""
    C = 96 if quick else 128
    N = 800 if quick else 2000
    SEEDS = seeds if seeds else (1 if quick else 2)
    return dict(C=C, N=N, seeds=SEEDS, bs=8, ckpt=16, fp=fp,
                # 'w10' = steady-state timing (warmup-excluded ms/step); bump if the timing changes so
                # cached entries re-measure under one consistent methodology.
                sig=f"C{C}_N{N}_s{SEEDS}_w{TIMING_WARMUP}_{fp[:8]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None, help="comma-sep optimizer names to (re)measure")
    ap.add_argument("--new", action="store_true", help="measure only optimizers missing from the cache")
    ap.add_argument("--render-only", action="store_true", help="rebuild RANKINGS.md from cache; no training")
    ap.add_argument("--quick", action="store_true", help="smaller/faster settings (own cache)")
    ap.add_argument("--seeds", type=int, default=None,
                    help="override the seed count (enters the settings signature)")
    ap.add_argument("--out", type=str, default=None,
                    help="tag: cache/rankings go to results_TAG.json / RANKINGS_TAG.md")
    A = ap.parse_args()

    if A.seeds is not None and A.seeds < 1:
        raise SystemExit("--seeds must be >= 1")
    ds = D.build_proxy_dataset()
    fp = D.fingerprint(ds)
    cfg = build_cfg(A.quick, fp, A.seeds)
    N = cfg["N"]
    store = load_store(A.quick, A.out)

    if not A.render_only:
        # Electrical gate BEFORE anything is measured: on battery this laptop caps the GPU at 35 W
        # instead of 60 W, so the resulting timings would silently contaminate the cache.
        meta = run_metadata(require_ac())
        data = {k: v.to(DEV).to(H.DT) for k, v in ds["DATA"].items()}
        tr, te = ds["TR"], ds["TE"]
        ac = H.make_alphas()
        seqp = seq_prog(N)
        if A.only:
            targets = [n.strip() for n in A.only.split(",")]
        elif A.new:
            targets = [n for n in OPTIMIZERS if store.get(n, {}).get("sig") != cfg["sig"]]
        else:
            # full run skips 'frozen' optimizers (e.g. external AdamW) already cached at this
            # settings signature -- they don't change with kaon edits. Re-measured only if
            # missing/stale; always re-measured when named via --only.
            targets = [n for n, spec in OPTIMIZERS.items()
                       if not (spec.get("frozen") and store.get(n, {}).get("sig") == cfg["sig"])]
        print(f"BATTERY sig={cfg['sig']} | measuring: {targets or '(none)'}", flush=True)
        for name in targets:
            if name not in OPTIMIZERS:
                print(f"  {name}: not in registry — skipped", flush=True); continue
            try:
                store[name] = measure(name, OPTIMIZERS[name], cfg, data, tr, te, ac, seqp, meta=meta)
                m = store[name]
                print(f"  {name:16s} te={m['te']:.4f} gap={m['gap']:+.4f} {m['ms']:.1f}ms "
                      f"{m['bpp']:.2f}B/p lora={m['lms']:.2f}ms const_gap={m['cgap']:+.4f}", flush=True)
                save_store(store, A.quick, A.out)  # incremental: persist after each optimizer
            except Exception as e:  # noqa: BLE001
                print(f"  {name:16s} FAILED: {type(e).__name__}: {e}", flush=True)

    render(store, cfg, A.quick, A.out)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
