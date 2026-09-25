"""Merge stage1/stage2(+extra) JSON into results.json + RESULTS.md for the Antikaon gate.

Run once, after all stage JSONs exist, from this directory:
    python build_report.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

GATE_TEST = 0.0700
GATE_GAP = 0.0070

ARM_ORDER = [
    "A0 Adakaon-nomom", "A1 Nekaon (defaults)",
    "B0 k_sigma=0.5", "B1 k_sigma=1.5", "B2 k_sigma=5", "B3 k_sigma=15",
    "C1 shape=none", "C2 antithetic", "C3 sigma_ref=weight",
    "B2 lr x0.5", "B2 lr x2",
]


def load(name):
    with open(f"{HERE}/{name}") as f:
        return json.load(f)


def fmt(m):
    return f"{m['test_mean']:.5f} (+-{m['test_std']:.5f})", f"{m['gap_mean']:+.5f} (+-{m['gap_std']:.5f})"


def stage_table(stage):
    order = [n for n in ARM_ORDER if n in stage["results"]]
    rows = []
    for n in order:
        m = stage["results"][n]
        te, ga = fmt(m)
        seeds_str = "; ".join(f"s{p['seed']}: te={p['test']:.5f} gap={p['gap']:+.5f}"
                              for p in m["per_seed"])
        rows.append((n, te, ga, m["bpp"], seeds_str))
    return rows


def main():
    s1 = load("stage1.json")
    s2 = load("stage2.json")
    s2x = load("stage2_extra.json")
    # merge the extra k_sigma=0.5 arm into stage2's result set (same manifest settings)
    s2["results"].update(s2x["results"])

    merged = {"stage1": s1, "stage2": s2}
    with open(f"{HERE}/results.json", "w") as f:
        json.dump(merged, f, indent=1)

    m1, m2 = s1["manifest"], s2["manifest"]
    L = []
    L.append("# Antikaon quality gate — control battery, constant LR\n")
    L.append("Only quality (held-out loss, train-val gap) is scored; ms/step is NOT measured "
             "or reported (GPU shared with another job during this run).\n")
    L.append("## Manifest\n")
    L.append(f"- commit: `{m1['commit']}` (branch `{m1['branch']}`)")
    L.append(f"- dataset fingerprint (sha256, proxy/dataset.py): `{m1['dataset_fingerprint'][:16]}…` "
             f"(identical for stage 1 and stage 2 — fixed synthetic dataset, seed-independent)")
    L.append(f"- proxy weight dtype: `{m1['proxy_weight_dtype']}` (fp32 — the harness always "
             f"trains at fp32; Antikaon's `bf16_method` stochastic-rounding path is NOT "
             f"exercised by this gate, only its noise/shaping math)")
    L.append(f"- stage 1: C={m1['C']}, N={m1['N']}, seeds={m1['seeds']}, bs={m1['bs']}, "
             f"schedule={m1['schedule']}")
    L.append(f"- stage 2: C={m2['C']}, N={m2['N']}, seeds={m2['seeds']}, bs={m2['bs']}, "
             f"schedule={m2['schedule']}")
    L.append("- LR (constant, per arm): A0/A1/B*/C* = 1.2e-3 (registry `lr_const`); "
             "B2 lr x0.5 = 6e-4; B2 lr x2 = 2.4e-3")
    L.append(f"- gate reference: test <= {GATE_TEST} and gap <= {GATE_GAP}\n")

    for label, stage, prev_pass in (("Stage 1 (C=40, N=600, seed=0 — divergence/underfit screen)", s1, None),
                                     ("Stage 2 (C=40, N=2000, seeds=0,1 — late-overfit exposure)", s2, None)):
        L.append(f"## {label}\n")
        L.append("| arm | held-out loss (mean +- std) | train-val gap (mean +- std) | B/param | per-seed |")
        L.append("|---|---:|---:|---:|---|")
        for n, te, ga, bpp, seeds_str in stage_table(stage):
            L.append(f"| {n} | {te} | {ga} | {bpp:.3f} | {seeds_str} |")
        L.append("")

    # ---- gate verdict per arm (stage2 numbers) ----
    r2 = s2["results"]
    a0 = r2["A0 Adakaon-nomom"]
    a1 = r2["A1 Nekaon (defaults)"]
    L.append("## Gate verdict (stage 2, the long two-seed gate)\n")
    L.append(f"Reference corner: test <= {GATE_TEST}, gap <= {GATE_GAP}. Frontier-mover check "
             "(docs/research/antikaon-design.md §6): a B/C arm must beat A0 on BOTH axes by "
             "more than the two-seed spread, AND be non-dominated by A1 (loss<=A1 OR gap<=A1).\n")
    L.append("| arm | reaches test<=.0700 & gap<=.0070 | beats A0 both axes beyond spread | "
             "non-dominated by A1 | verdict |")
    L.append("|---|---|---|---|---|")
    for n in ARM_ORDER:
        if n not in r2 or n in ("A0 Adakaon-nomom", "A1 Nekaon (defaults)"):
            continue
        m = r2[n]
        corner = m["test_mean"] <= GATE_TEST and m["gap_mean"] <= GATE_GAP
        # "beyond spread": the mean difference vs A0 must exceed A0's own two-seed std on
        # that axis (a necessary, conservative bar with only 2 seeds/arm).
        beats_test = (a0["test_mean"] - m["test_mean"]) > a0["test_std"]
        beats_gap = (a0["gap_mean"] - m["gap_mean"]) > a0["gap_std"]
        beats_a0 = beats_test and beats_gap
        nondom_a1 = m["test_mean"] <= a1["test_mean"] or m["gap_mean"] <= a1["gap_mean"]
        verdict = "FRONTIER MOVER" if (beats_a0 and nondom_a1) else (
            "trades loss<->gap (dominated line)" if not beats_a0 else "non-dominated by A1 only")
        L.append(f"| {n} | {'yes' if corner else 'no'} | {'yes' if beats_a0 else 'no'} | "
                 f"{'yes' if nondom_a1 else 'no'} | {verdict} |")
    L.append("")

    with open(f"{HERE}/RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("wrote results.json and RESULTS.md")


if __name__ == "__main__":
    main()
