"""Aggregate the retime-016 ABBA battery JSONs into a per-arm md table.

Valid rounds (round 1 of B was discarded: wall 914.3s vs round 3's 644.6s, a 41.8%
deviation -- see report.md):
  A: 1_A_6201c1f, 4_A_6201c1f
  B: 2r_B_026621d, 3_B_026621d
"""
import json
from pathlib import Path

OUTDIR = Path(__file__).parent

A_TAGS = ["1_A_6201c1f", "4_A_6201c1f"]
B_TAGS = ["2r_B_026621d", "3_B_026621d"]


def load(tag):
    return json.loads((OUTDIR / tag / f"battery_{tag}.json").read_text())["results"]


A = [load(t) for t in A_TAGS]
B = [load(t) for t in B_TAGS]


def walk(results):
    """Yield (path_tuple, median_ms, q1_ms, q3_ms) for every arm in a results tree."""
    for shape, groups in results.items():
        for group, sub in groups.items():
            if group == "adakaon":
                for mom, arms in sub.items():
                    for arm, r in arms.items():
                        yield (shape, group, mom, arm), r
            else:
                for arm, r in sub.items():
                    yield (shape, group, "-", arm), r


def as_dict(results):
    return {k: v for k, v in walk(results)}


A_d = [as_dict(r) for r in A]
B_d = [as_dict(r) for r in B]

keys = list(A_d[0].keys())

rows = []
for k in keys:
    a_meds = [d[k]["median_ms"] for d in A_d]
    b_meds = [d[k]["median_ms"] for d in B_d]
    a_q1 = min(d[k]["q1_ms"] for d in A_d)
    a_q3 = max(d[k]["q3_ms"] for d in A_d)
    b_q1 = min(d[k]["q1_ms"] for d in B_d)
    b_q3 = max(d[k]["q3_ms"] for d in B_d)
    a_med = sorted(a_meds)[len(a_meds) // 2] if len(a_meds) % 2 else sum(sorted(a_meds)[len(a_meds)//2-1:len(a_meds)//2+1]) / 2
    b_med = sorted(b_meds)[len(b_meds) // 2] if len(b_meds) % 2 else sum(sorted(b_meds)[len(b_meds)//2-1:len(b_meds)//2+1]) / 2
    ratio = b_med / a_med if a_med else float("nan")
    # per-pairing ratios for consistency check: (round1 A vs round2r B), (round4 A vs round3 B)
    pair_ratios = [B_d[i][k]["median_ms"] / A_d[i][k]["median_ms"] for i in range(len(A_d))]
    consistent_reg = all(pr > 1.05 for pr in pair_ratios)
    rows.append({
        "arm": "/".join(k), "a_med": a_med, "a_q1": a_q1, "a_q3": a_q3,
        "b_med": b_med, "b_q1": b_q1, "b_q3": b_q3, "ratio": ratio,
        "pair_ratios": pair_ratios, "consistent_reg": consistent_reg,
    })

with open(OUTDIR / "arms_table.md", "w") as fh:
    fh.write("| arm (shape/opt/mom/mode_method) | A median ms [IQR] | B median ms [IQR] | ratio B/A | pair ratios | flag |\n")
    fh.write("|---|---|---|---|---|---|\n")
    for r in rows:
        flag = "**REGRESSION**" if r["consistent_reg"] else ("watch" if r["ratio"] > 1.05 else "")
        fh.write(f"| {r['arm']} | {r['a_med']:.4f} [{r['a_q1']:.4f},{r['a_q3']:.4f}] | "
                  f"{r['b_med']:.4f} [{r['b_q1']:.4f},{r['b_q3']:.4f}] | {r['ratio']:.3f} | "
                  f"{','.join(f'{p:.3f}' for p in r['pair_ratios'])} | {flag} |\n")

n_flagged = sum(1 for r in rows if r["consistent_reg"])
n_watch = sum(1 for r in rows if r["ratio"] > 1.05 and not r["consistent_reg"])
print(f"total arms: {len(rows)}, consistent regressions (ratio>1.05 both pairings): {n_flagged}, "
      f"watch (ratio>1.05 not consistent): {n_watch}")
for r in rows:
    if r["consistent_reg"]:
        print("REGRESSION:", r["arm"], "ratio=%.3f" % r["ratio"], "pairs=", r["pair_ratios"])
