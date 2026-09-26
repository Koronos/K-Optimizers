"""Low-LR bf16 quality experiment: the bf16 weight-write methods compared head to head.

The generalization of ``benchmarks/antikaon_lowlr/run_lowlr.py`` (same U-Net, dataset, loss,
eval and fine-tune protocol — see that file's header for the LR choice and the protocol's
rationale) that does NOT depend on Antikaon: two rules, every ``bf16_method``.

* **bf16 weights** (the whole U-Net in ``torch.bfloat16``; loss and eval MSE in fp32);
* **a fine-tune LR** (constant, default ``1e-5``: the median update is ~0.06 bf16 ulp);
* **fine-tune protocol**: fp32 pre-training at the battery's ``lr_const`` with Adakaon-nomom
  (``--pretrain`` steps, cached per seed under ``cache/``), rounded ONCE to bf16; every arm
  fine-tunes from that same bf16-representable start with the same data order.

Arms (``--arms``, comma-separated keys): ``{ada,nek}-{sr,kahan,k8,k8ld,k16,fp32}``

  ada-*   Adakaon-nomom (registry config: betas (0, 0.999), cautious off, bf16 momentum)
  nek-*   Nekaon (registry config: k 1.5, betas (0.5, 0.999), wd 0.1, 4-bit momentum)
  *-sr    bf16 weights, stochastic rounding (0 B/param)
  *-kahan bf16 weights, legacy ``kahan`` (bf16 ``shift`` buffer, +2 B/param, PER-PARAM path
          only — foreach/fused reject it, so its ms/step is the per-param loop's)
  *-k8    bf16 weights, ``kahan8`` (uint8 residual, +1 B/param, every path)
  *-k8ld  bf16 weights, ``kahan8ld`` (EXPERIMENTAL: kahan8's state with the residual rounded
          by a low-discrepancy dither instead of SR — ``docs/research/compact-kahan/lowdisc/``)
  *-k16   bf16 weights, ``kahan16`` (int16 residual, +2 B/param: an fp32 master split in two)
  *-fp32  fp32 weights — the ceiling and the reference each rule's distances are taken to

Metrics per arm and seed (the ``antikaon_lowlr`` set): held-out ``test`` and ``train`` loss at
the CLEAN weights (``evald``: eval()/train() bracket — Nekaon is scored without its climb),
``gap = test - train``, ``test_start``, ``test_full`` (the same clean iterate at its FULL value
in an fp32 copy of the net: ``decode(p, kahan_lo)`` for kahan8/kahan8ld/kahan16, ``p + shift`` for the
legacy kahan, ``p`` otherwise), and the distance of that full value to the rule's fp32 arm:
``dist_ulp`` (RMS in bf16 ulps of the reference, per tensor at its RMS magnitude) and
``dist_rel = ||z - z_fp32|| / ||z_fp32 - z_start||``.

**ms/step (ORIENTATIVE).** Each arm also reports the mean wall time of an active training
step (forward + backward + optimizer step; CUDA events, first ``WARMUP`` steps excluded) and
of the optimizer step alone (a third of the run under ``--smoke``). Arms run SERIALLY in one process on whatever machine this is, so
the numbers are only indicative (laptop GPU: power state, clocks, other load); a definitive
speed comparison needs a dedicated serial benchmark.

Usage (from the repo root, ``PYTHONPATH=src``):
    python benchmarks/lowlr_bf16/run_lowlr.py --smoke                  # plumbing: few steps, all arms
    python benchmarks/lowlr_bf16/run_lowlr.py                          # 12 arms x seeds 0,1, 8000 steps
    python benchmarks/lowlr_bf16/run_lowlr.py --lr 1e-5 --seeds 0,1 --steps 8000 --arms ada-k8,ada-k16,ada-fp32
    python benchmarks/lowlr_bf16/run_lowlr.py --arms ada-k8,ada-k8ld,ada-fp32,nek-k8,nek-k8ld,nek-fp32
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import time
import warnings

import torch
import torch.nn.functional as F

from kaon import Adakaon, Nekaon
from kaon._compact_kahan import RESIDUAL_KEY, decode, residual_bits_of

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
WARMUP = 20                    # steps excluded from ms/step (Triton compile, allocator warm-up)


# ----------------------------- model: the proxy U-Net at any weight dtype -----------------
class UNetLP(H.UNet):
    """``H.UNet`` whose forward casts its inputs to the weights' dtype and returns fp32
    (identical to ``H.UNet.forward`` for fp32 weights; see ``antikaon_lowlr``)."""

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


def _nek(method):
    return lambda p, lr: Nekaon(p, lr=lr, k=1.5, betas=(0.5, 0.999), weight_decay=0.1,
                                momentum_dtype="4bit", bf16_method=method)


BF, FP = torch.bfloat16, torch.float32
METHODS = {"sr": "stochastic_rounding", "kahan": "kahan", "k8": "kahan8", "k8ld": "kahan8ld",
           "k16": "kahan16"}
RULES = {"ada": _ada, "nek": _nek}
# key -> (make, weight dtype, rule id: arms of the same rule share the fp32 reference)
ARMS: dict[str, tuple] = {}
for _rule, _make in RULES.items():
    for _tag, _method in METHODS.items():
        ARMS[f"{_rule}-{_tag}"] = (_make(_method), BF, _rule)
    ARMS[f"{_rule}-fp32"] = (_make("stochastic_rounding"), FP, _rule)   # method is a no-op on fp32


# ----------------------------- helpers ---------------------------------------------------
def full_values(opt, params):
    """The full-precision stored value of every param (fp32, detached): ``decode(p, lo)``
    under kahan8/kahan16 (the residual's own width), ``p + shift`` under the legacy kahan,
    else ``p`` — looked up through the wrapper chain (Nekaon keeps it in its inner)."""
    out = []
    for p in params:
        v = p.detach().float().clone()
        o = opt
        while o is not None and p.dtype == torch.bfloat16:
            st = o.state.get(p) if hasattr(o, "state") else None
            if st and RESIDUAL_KEY in st:
                lo = st[RESIDUAL_KEY]
                v = decode(p.data, lo, residual_bits_of(lo))
                break
            if st and "shift" in st:
                v = p.detach().float() + st["shift"].float()
                break
            o = getattr(o, "inner", None)
        out.append(v)
    return out


def clean_values(opt, params):
    """Full-precision CLEAN iterate: eval() (removes Nekaon's climb), read, train()."""
    return CTRL.evald(opt, lambda: full_values(opt, params))


def distance(zs, refs, starts):
    """(RMS of (z - ref) in bf16 ulps of the reference's per-tensor RMS, ||z - ref|| /
    ||ref - start||) — as in ``antikaon_lowlr``."""
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
        opt = _ada("stochastic_rounding")([p for p in net.parameters() if p.requires_grad],
                                          PRETRAIN_LR)
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


class _StepTimer:
    """Mean ms of the active step and of the optimizer step, CUDA events (no per-step sync)
    or ``perf_counter`` on CPU; the first ``warmup`` steps are not recorded (``WARMUP``,
    or a third of a short run such as ``--smoke``)."""

    def __init__(self, steps):
        self.cuda = DEV == "cuda"
        self.warmup = min(WARMUP, max(1, steps // 3))
        self.rows: list = []

    def mark(self):
        if self.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e
        return time.perf_counter()

    def add(self, i, t0, t1, t2):
        if i >= self.warmup:
            self.rows.append((t0, t1, t2))

    def result(self):
        if not self.rows:
            return None, None
        if self.cuda:
            torch.cuda.synchronize()
            tot = [a.elapsed_time(c) for a, _b, c in self.rows]
            opt = [b.elapsed_time(c) for _a, b, c in self.rows]
        else:
            tot = [(c - a) * 1e3 for a, _b, c in self.rows]
            opt = [(c - b) * 1e3 for _a, b, c in self.rows]
        return sum(tot) / len(tot), sum(opt) / len(opt)


def finetune(key, *, start_sd, lr, seed, C, bs, data, tr, te, ac, seq):
    """One fine-tune run (the battery's const-LR loop) with an orientative step timer."""
    make, dtype, _rule = ARMS[key]
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
    timer = _StepTimer(len(seq))
    fired = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opt = make(params, lr)
        pos = 0
        for i, Rr in enumerate(seq):
            idx = [tr[(pos + j) % len(tr)] for j in range(bs)]
            pos += bs
            t0 = timer.mark()
            opt.zero_grad()
            H.batch_loss(net, data[Rr], torch.tensor(idx, device=DEV), ac, g).backward()
            t1 = timer.mark()
            opt.step()
            t2 = timer.mark()
            timer.add(i, t0, t1, t2)
    for w in caught:
        msg = str(w.message)
        if "below half" in msg or "lookahead" in msg.lower():
            fired.append(msg[:160])
    ms_step, ms_opt = timer.result()
    tr_loss, te_loss = CTRL.evald(
        opt, lambda: (H.eval_loss(net, data[64], tr, ac), H.eval_loss(net, data[64], te, ac)))
    z = clean_values(opt, params)
    # The same clean iterate evaluated in fp32 at its FULL value (see the module doc).
    net32 = copy.deepcopy(net).float()
    with torch.no_grad():
        for q, v in zip((q for q in net32.parameters() if q.requires_grad), z, strict=True):
            q.copy_(v)
    te_full = H.eval_loss(net32, data[64], te, ac)
    del net32
    return dict(train=tr_loss, test=te_loss, gap=te_loss - tr_loss, test_start=te0,
                test_full=te_full, ms_step=ms_step, ms_opt=ms_opt,
                inert_warnings=sorted(set(fired)), clean=[t.cpu() for t in z],
                start=[t.cpu() for t in z0])


def mean(xs):
    return sum(xs) / len(xs)


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _fmt(x, spec):
    return "-" if x is None else format(x, spec)


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
                         data=data, tr=tr, te=te, ac=ac, seq=seq)
            runs[key] = r
            print(f"  [{key}] seed={seed} test={r['test']:.5f} (start {r['test_start']:.5f}, "
                  f"fp32 view {r['test_full']:.5f}) train={r['train']:.5f} gap={r['gap']:+.5f} "
                  f"ms/step={_fmt(r['ms_step'], '.2f')} (opt {_fmt(r['ms_opt'], '.2f')}, "
                  f"orientative)"
                  + (f"  inert-warning x{len(r['inert_warnings'])}" if r["inert_warnings"] else ""),
                  flush=True)
        for key, r in runs.items():
            rule = ARMS[key][2]
            ref_key = f"{rule}-fp32"
            d_ulp = d_rel = None
            if ref_key in runs and ref_key != key:
                d_ulp, d_rel = distance(r["clean"], runs[ref_key]["clean"], r["start"])
            row = results.setdefault(key, dict(per_seed=[]))
            row["per_seed"].append(dict(
                seed=seed, test=r["test"], train=r["train"], gap=r["gap"],
                test_start=r["test_start"], test_full=r["test_full"],
                dist_ref=ref_key if d_ulp is not None else None, dist_ulp=d_ulp, dist_rel=d_rel,
                ms_step=r["ms_step"], ms_opt=r["ms_opt"], inert_warnings=r["inert_warnings"]))
    for row in results.values():
        ps = row["per_seed"]
        for m in ("test", "gap", "train", "test_start", "test_full"):
            row[f"{m}_mean"] = mean([p[m] for p in ps])
            row[f"{m}_std"] = stdev([p[m] for p in ps])
        for m in ("dist_ulp", "dist_rel", "ms_step", "ms_opt"):
            xs = [p[m] for p in ps if p[m] is not None]
            row[f"{m}_mean"] = mean(xs) if xs else None
    print("\narm         test (+-sd)          gap (+-sd)         d_start   test_full  "
          "dist_ulp  dist_rel  ms/step  ms/opt   (ms: orientative)")
    for key, row in results.items():
        print(f"{key:10s}  {row['test_mean']:.5f}(+-{row['test_std']:.5f}) "
              f"{row['gap_mean']:+.5f}(+-{row['gap_std']:.5f}) "
              f"{row['test_mean'] - row['test_start_mean']:+.5f}  {row['test_full_mean']:.5f}  "
              f"{_fmt(row['dist_ulp_mean'], '.3f'):>8s}  {_fmt(row['dist_rel_mean'], '.3f'):>8s}  "
              f"{_fmt(row['ms_step_mean'], '.2f'):>7s}  {_fmt(row['ms_opt_mean'], '.2f'):>6s}",
              flush=True)
    manifest = dict(
        commit=os.popen("git rev-parse HEAD").read().strip(),
        branch=os.popen("git rev-parse --abbrev-ref HEAD").read().strip(),
        config=cfg, pretrain_lr=PRETRAIN_LR, schedule="const", device=DEV,
        dataset_fingerprint=fp, arms=arm_keys, timing_warmup_steps=WARMUP,
        timing_note="ms/step and ms/opt are ORIENTATIVE (serial arms in one process, "
                    "no power/clock control)",
    )
    with open(out_path, "w") as f:
        json.dump(dict(manifest=manifest, results=results), f, indent=1)
    print(f"wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true", help="few steps, all arms (plumbing)")
    ap.add_argument("--arms", type=str, default=None,
                    help=f"comma-separated arm keys (default: all): {','.join(ARMS)}")
    ap.add_argument("--seeds", type=str, default=None, help="comma-separated, default 0,1")
    ap.add_argument("--steps", type=int, default=None, help="fine-tune steps, default 8000")
    ap.add_argument("--pretrain", type=int, default=None, help="fp32 pre-train steps, default 300")
    ap.add_argument("--lr", type=float, default=None, help="fine-tune LR, default 1e-5")
    ap.add_argument("--C", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    cfg = dict(SMOKE if a.smoke else DEFAULTS)
    for k in ("steps", "pretrain", "lr", "C"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    if a.seeds:
        cfg["seeds"] = [int(s) for s in a.seeds.split(",")]
    keys = [s.strip() for s in a.arms.split(",")] if a.arms else list(ARMS)
    unknown = [k for k in keys if k not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {list(ARMS)}")
    out = a.out or os.path.join(HERE, "smoke.json" if a.smoke else "results.json")
    run(cfg, keys, out)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
