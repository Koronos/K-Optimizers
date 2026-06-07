"""Render the control-battery rankings as charts (from results.json).

Produces two PNGs under ``benchmarks/control/plots/``:
  * ``dashboard.png``    — 6 horizontal-bar panels, one per battery dimension.
  * ``loss_vs_gap.png``  — the headline scatter (held-out loss vs train-val gap).
Colored by family (reference / in-house / candidate). Reads the cached numbers, so it
re-renders instantly without re-training. ``python plot_rankings.py``.
"""
from __future__ import annotations

import json
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = "/media/koronos/arca/repos/K-Optimizers/benchmarks/control"
# Group by what matters for the merge decision: already-SHIPPED (on main) vs NEW candidate vs
# reference vs DISCARDED. (The registry's `family` tags Lion/AdaPNM/AdaMuon "published" even
# though they ship.) Discarded optimizers keep their numbers in results_archive.json so the
# charts/README preserve them after they leave the live registry.
SHIPPED = {"Adakaon-nomom", "Adakaon-bf16", "Lion", "AdaPNM", "AdaMuon"}
# Wrappers run OVER Adakaon as their base optimizer — flagged with a "*" in the charts.
WRAPPERS = {"ScheduleFree", "Lookahead", "SAM"}
GRP_COLOR = {"reference": "#9aa0a6", "shipped": "#2c7fb8", "candidate": "#e6843c",
             "discarded": "#c0392b"}
GRP_LABEL = {"reference": "reference (torch.AdamW)", "shipped": "shipped (on main)",
             "candidate": "new candidate", "discarded": "discarded (archived)"}


def group_of(name, m):
    if m.get("_discarded"):
        return "discarded"
    if m.get("family") == "reference":
        return "reference"
    return "shipped" if name in SHIPPED else "candidate"


def disp(name):
    """Display name — wrappers get a '*' marking them as Adakaon-based."""
    return f"{name} *" if name in WRAPPERS else name


def load_active():
    with open(f"{HERE}/results.json") as f:
        store = json.load(f)
    archive = {}
    arch_path = f"{HERE}/results_archive.json"
    if os.path.exists(arch_path):
        with open(arch_path) as f:
            archive = json.load(f)
    sig = Counter(v.get("sig") for v in store.values()).most_common(1)[0][0]
    a = {}
    for n, m in {**archive, **store}.items():  # live overrides archive on name clashes
        if m.get("sig") != sig:
            continue
        d = dict(m)
        d["_discarded"] = (n in archive) and (n not in store)
        a[n] = d
    # derived metrics (match battery.render): median convergence target + mean rank
    best = sorted(min(v for _, v in m["traj"]) for m in a.values())
    t = best[len(best) // 2]
    for m in a.values():
        m["conv"] = next((s for s, v in m["traj"] if v <= t), m["traj"][-1][0])
        m["ttq"] = m["ms"] * m["conv"] / 1000.0

    def ranked(key):
        order = sorted(a, key=lambda n: a[n][key])
        return {n: i + 1 for i, n in enumerate(order)}

    rk = {k: ranked(k) for k in ("ms", "lms", "conv", "te", "gap", "bpp", "cgap")}
    for n in a:
        a[n]["mean_rank"] = sum(rk[k][n] for k in rk) / len(rk)
    return a, sig, t


def hbar(ax, a, key, title, *, fmt="{:.3f}", reverse=False):
    names = sorted(a, key=lambda n: a[n][key], reverse=reverse)
    vals = [a[n][key] for n in names]
    cols = [GRP_COLOR[group_of(n, a[n])] for n in names]
    y = range(len(names))
    ax.barh(list(y), vals, color=cols)
    ax.set_yticks(list(y)); ax.set_yticklabels([disp(n) for n in names], fontsize=8)
    ax.invert_yaxis()  # best (smallest) on top
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8)
    vmax = max(vals) if vals else 1
    for yi, v in zip(y, vals, strict=True):
        ax.text(v + vmax * 0.01, yi, fmt.format(v), va="center", fontsize=7)
    ax.set_xlim(0, vmax * 1.18)


def main():
    a, sig, t = load_active()
    os.makedirs(f"{HERE}/plots", exist_ok=True)

    # ---------- dashboard: 6 panels ----------
    fig, axes = plt.subplots(3, 2, figsize=(15, 17))
    hbar(axes[0][0], a, "mean_rank", "Overall (mean rank — lower = better all-rounder)", fmt="{:.1f}")
    hbar(axes[0][1], a, "gap", "Generalization — train-val GAP (the headline)", fmt="{:+.4f}")
    hbar(axes[1][0], a, "ttq", f"Time to target (s)  [target loss <= {t:.4f}, median]", fmt="{:.1f}")
    hbar(axes[1][1], a, "bpp", "Memory — optimizer state (B/param)", fmt="{:.2f}")
    hbar(axes[2][0], a, "lms", "LoRA speed — ms/step (512 tiny tensors)", fmt="{:.1f}")
    hbar(axes[2][1], a, "cgap", "Continuity — train-val gap at CONSTANT LR", fmt="{:+.4f}")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GRP_COLOR.values()]
    fig.legend(handles, [GRP_LABEL[f] for f in GRP_COLOR], loc="upper center",
               ncol=4, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(f"kaon control battery — {len(a)} optimizers @ {sig}  (each at best config; lower=better)",
                 fontsize=13, y=0.975)
    fig.text(0.5, 0.005, "* = wrapper over Adakaon (ScheduleFree / Lookahead / SAM)",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=(0, 0.012, 1, 0.955))
    fig.savefig(f"{HERE}/plots/dashboard.png", dpi=150)
    plt.close(fig)

    # ---------- headline scatter: loss vs gap ----------
    fig, ax = plt.subplots(figsize=(11, 8))
    for n, m in a.items():
        c = GRP_COLOR[group_of(n, m)]
        marker = "*" if n in WRAPPERS else "o"        # wrappers (Adakaon base) get a star
        size = 180 if n in WRAPPERS else 70
        ax.scatter(m["te"], m["gap"], s=size, color=c, marker=marker,
                   edgecolors="k", linewidths=0.5, zorder=3)
        ax.annotate(disp(n), (m["te"], m["gap"]), fontsize=8, xytext=(5, 3),
                    textcoords="offset points")
    ax.set_xlabel("held-out loss  (lower = better fit ->)", fontsize=11)
    ax.set_ylabel("train-val GAP  (lower = generalizes better, down)", fontsize=11)
    ax.set_title("Loss x Generalization — the Pareto view (bottom-left is best)\n"
                 "ADOPT/AdaPNM = lowest gap; Adakaon-bf16/Lookahead = lowest loss; Grams/Adai = worst",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, zorder=0)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, mec="k", label=GRP_LABEL[f])
               for f, c in GRP_COLOR.items() if f != "discarded" or any(group_of(n, a[n]) == "discarded" for n in a)]
    handles.append(plt.Line2D([], [], marker="*", ls="", color="#e6843c", mec="k",
                              markersize=12, label="* wrapper over Adakaon"))
    ax.legend(handles=handles, fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{HERE}/plots/loss_vs_gap.png", dpi=150)
    plt.close(fig)
    print(f"wrote dashboard.png + loss_vs_gap.png ({len(a)} optimizers, target<={t:.4f})")


if __name__ == "__main__":
    main()
