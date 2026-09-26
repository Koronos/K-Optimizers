"""Render results/*.json (sim_lowdisc.py) and results/ema_bf16.json into markdown tables."""
from __future__ import annotations

import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["sr", "ld-phi", "ld-r2", "ld-sqrt2", "ld-third", "ld-phi-nohash",
         "kahan8-sr", "kahan8-rn", "kahan8-ld", "kahan8-ld-r2", "kahan4-sr", "kahan4-ld"]
REMEDIES = ["kahan8-sr", "kahan8-ld", "kahan8-ld-r2", "kahan8-ld-b64", "kahan8-ld-b256",
            "kahan8-ld-b1024", "kahan8-ld16", "kahan8-ld16-b256", "kahan8-ld-j32", "kahan8-ld16-j64"]
GROUPS = [("orig", "Existing sim streams (validation + figures to beat)"),
          ("coh", "1. Coherent: constant step, fixed sign"),
          ("noisy", "2. Noise-dominated: mu + sigma N(0,1), sigma = 10 mu"),
          ("adam", "3. Realistic: Adam-like (EMA beta1 0.9, sqrt v, beta2 0.999)"),
          ("multi", "4. Multiple writes per step (+xi, -xi, -delta; own n each)"),
          ("alias", "5. Aliasing: periodic update, P = 2, 3, 7")]


def fmt(x: float, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}" if abs(x) < 100 else f"{x:.1f}"


def load() -> dict[str, dict[tuple[str, int], dict]]:
    data: dict[str, dict] = {}
    for p in sorted(glob.glob(os.path.join(HERE, "results", "*.json"))):
        if os.path.basename(p).startswith("ema_bf16"):
            continue
        for r in json.load(open(p)):
            data.setdefault(r["regime"], {})[(r["scheme"], r["n"])] = r
    return data


def regime_table(reg: str, rows: dict) -> list[str]:
    s10 = rows[("sr", 10_000)]
    out = [f"#### `{reg}` — step RMS {s10['step_over_ulp']:.4f} ulp_ref at 10k (ulp_ref {s10['ulp_ref']:.3g})", "",
           "| scheme | bias ± se @10k | dir-bias ± se @10k | std @1k | std @3k | std @10k | 10k/1k | lost @10k | fwd std @10k |",
           "|---|---|---|---|---|---|---|---|---|"]
    for s in ORDER:
        if (s, 10_000) not in rows:
            continue
        a, b, c = rows[(s, 1_000)], rows[(s, 3_000)], rows[(s, 10_000)]
        ratio = c["std_ulp"] / a["std_ulp"] if a["std_ulp"] > 0 else float("nan")
        z = abs(c["bias_ulp"]) / c["bias_se_ulp"] if c["bias_se_ulp"] > 0 else 0.0
        flag = " **(>3 se)**" if z > 3 else ""
        zd = abs(c["dir_bias_ulp"]) / c["dir_bias_se_ulp"] if c["dir_bias_se_ulp"] > 0 else 0.0
        dflag = " **(>3 se)**" if zd > 3 else ""
        out.append(f"| {s} | {c['bias_ulp']:+.4f} ± {c['bias_se_ulp']:.4f}{flag} | "
                   f"{c['dir_bias_ulp']:+.4f} ± {c['dir_bias_se_ulp']:.4f}{dflag} | {fmt(a['std_ulp'])} | "
                   f"{fmt(b['std_ulp'])} | {fmt(c['std_ulp'])} | {fmt(ratio, 2)} | {c['lost']:+.4f} | "
                   f"{fmt(c['fwd_std_ulp'])} |")
    return out + [""]


def summary(data: dict) -> list[str]:
    regs = list(data)
    cols = ["sr", "ld-phi", "ld-r2", "ld-third", "kahan8-sr", "kahan8-rn", "kahan8-ld", "kahan4-sr", "kahan4-ld"]
    out = ["| regime | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for reg in regs:
        cells = []
        for s in cols:
            r = data[reg].get((s, 10_000))
            if r is None:
                cells.append("—")
                continue
            lost = f" ({100 * r['lost']:.0f} %)" if abs(r["lost"]) >= 0.01 else ""
            cells.append(fmt(r["std_ulp"]) + lost)
        out.append(f"| {reg} | " + " | ".join(cells) + " |")
    return out + ["", "Cells: std of the tracked-value error at 10k steps (ulp_ref); "
                  "`(x %)` = fraction of the net movement lost when |lost| >= 1 %.", ""]


def ema_tables() -> list[str]:
    rs = []
    for name in ("ema_bf16.json", "ema_bf16_b.json"):
        p = os.path.join(HERE, "results", name)
        if os.path.exists(p):
            rs += json.load(open(p))
    if not rs:
        return []
    out = ["| stream | beta | D | method | t | rel bias ± se | rel RMS err | magnitude ratio | stall (last 500) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rs:
        if r["method"] == "fp32":
            continue
        ratio = "—" if r["ratio_mean"] is None else f"{r['ratio_mean']:.3f}"
        stall = "—" if r["stall"] is None else f"{100 * r['stall']:.1f} %"
        out.append(f"| {r['stream']} | {r['beta']} | {r.get('size', '512x512')} | {r['method']} | {r['t']} | "
                   f"{r['rel_bias']:+.2e} ± {r['rel_bias_se']:.1e} | {r['rel_rms']:.2e} | {ratio} | {stall} |")
    return out + [""]


def long_tables() -> list[str]:
    files = sorted(glob.glob(os.path.join(HERE, "results_long", "*.json")))
    if not files:
        return []
    out = ["D = 2**16 (bias SE 4x the main grid's), 100 000 steps; same streams and seeds as the main grid.", ""]
    for p in files:
        rs = json.load(open(p))
        reg = rs[0]["regime"]
        ns = sorted({r["n"] for r in rs})
        by = {(r["scheme"], r["n"]): r for r in rs}
        last = ns[-1]
        out += [f"#### `{reg}` (long)", "",
                "| scheme | " + " | ".join(f"std @{n // 1000}k" for n in ns) +
                f" | {ns[-1] // 1000}k/{ns[-2] // 1000}k | dir-bias ± se @{last // 1000}k | lost @{last // 1000}k |",
                "|---" * (len(ns) + 4) + "|"]
        for sch in ORDER:
            if (sch, last) not in by:
                continue
            c = by[(sch, last)]
            ratio = c["std_ulp"] / by[(sch, ns[-2])]["std_ulp"]
            out.append(f"| {sch} | " + " | ".join(fmt(by[(sch, n)]["std_ulp"]) for n in ns) +
                       f" | {ratio:.2f} | {c['dir_bias_ulp']:+.4f} ± {c['dir_bias_se_ulp']:.4f} | {c['lost']:+.4f} |")
        out.append("")
    return out


def remedy_tables(dirname: str = "results_remedies", order: list[str] = REMEDIES) -> list[str]:
    """results_remedies*/*.json: std @100k per regime x variant, and each variant's worst case
    as the ratio to kahan8-sr (the shipped scheme) over every regime."""
    files = sorted(glob.glob(os.path.join(HERE, dirname, "*.json")))
    if not files:
        return []
    by_reg: dict[str, dict] = {}
    for p in files:
        rs = json.load(open(p))
        last = max(r["n"] for r in rs)
        by_reg[rs[0]["regime"]] = {r["scheme"]: r for r in rs if r["n"] == last}
    cols = [c for c in order if all(c in v for v in by_reg.values())]
    out = ["D = 2**16, 100 000 steps, same streams and seeds as `results_long`. Cells: std of the "
           "tracked-value error (ulp_ref) at 100k; **bold** = directional bias > 3 se; "
           "`(x)` = ratio to kahan8-sr.", "",
           "| regime | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    worst = {c: (0.0, "") for c in cols}
    for reg, rows in by_reg.items():
        ref = rows["kahan8-sr"]["std_ulp"]
        cells = []
        for c in cols:
            r = rows[c]
            ratio = r["std_ulp"] / ref
            if ratio > worst[c][0]:
                worst[c] = (ratio, reg)
            z = abs(r["dir_bias_ulp"]) / r["dir_bias_se_ulp"] if r["dir_bias_se_ulp"] > 0 else 0.0
            v = fmt(r["std_ulp"])
            cells.append((f"**{v}**" if z > 3 else v) + ("" if c == "kahan8-sr" else f" ({ratio:.2f})"))
        out.append(f"| {reg} | " + " | ".join(cells) + " |")
    out.append("| **worst ratio** | " + " | ".join(
        f"{worst[c][0]:.2f} ({worst[c][1]})" for c in cols) + " |")
    return out + [""]


def main() -> None:
    data = load()
    lines = ["# Low-discrepancy rounding — generated tables", "",
             "Generated by `make_tables.py` from `results/*.json`. Units: ulp_ref = bf16 ulp of RMS(z_ref).", "",
             "## Summary (std @10k)", ""] + summary(data)
    for prefix, title in GROUPS:
        regs = [r for r in data if r.startswith(prefix)]
        if not regs:
            continue
        lines += [f"## {title}", ""]
        for reg in regs:
            lines += regime_table(reg, data[reg])
    lt = long_tables()
    if lt:
        lines += ["## Long horizon (100k steps)", ""] + lt
    rt = remedy_tables()
    if rt:
        lines += ["## Remedies against the long-horizon aliasing (100k steps)", ""] + rt
    rt2 = remedy_tables("results_remedies2", ["kahan8-sr", "kahan8-ld-b256", "kahan8-ld-b256k"])
    if rt2:
        lines += ["### Second round: the Weyl increment re-drawn per block too (`b256k`)", ""] + rt2
    rt3 = remedy_tables("results_kaon", ["kahan8-sr", "kahan8-ld-b256", "kahan8-ld-kaon"])
    if rt3:
        lines += ["### The shipped `kahan8ld` noise (`kaon._compact_kahan.ld_noise`)", ""] + rt3
    ema = ema_tables()
    if ema:
        lines += ["## EMA stored in low precision (sim_ema_bf16.py)", ""] + ema
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "tables.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {path}: {len(data)} regimes")


if __name__ == "__main__":
    main()
