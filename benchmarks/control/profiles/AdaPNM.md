# Profile — AdaPNM

> What this optimizer *likes*, found by [`profiler.py`](profiler.py) on the reproducible proxy (`C=128`, `N=1400`, 2 seed(s)). Greedy: each section fixes the previous winner; ranked by **held-out loss** (which penalizes under- *and* over-fitting), with the **train–val gap** shown as the overfit diagnostic. Proxy LRs are ~100x real — relative knobs, not recommendations.

**Identity:** positive-negative momentum (best generalization / constant-LR) · registry LR `2.4e-03`.

> ⚠️ **Scale & objective.** This is a *short* run (mild overfitting), so the held-out-loss optimum favors **less regularization** (lower LR, lighter knobs, single-res). On real small-data LoRA overfitting bites far harder — there the **gap** column matters more, and the **registry's shipped config is gap-tuned** (usually *more* regularizing than the loss-optimum below). Read this as the optimizer's response *surface* — which way each knob pushes loss vs gap — not a ship-it config. The gap-ranked field view is [`RANKINGS.md`](RANKINGS.md).
>
> Concretely here: by held-out loss the greedy path lands at `lr 6e-4 / rex / warmup 8% / single-res / beta0=0`, but the **gap** column moves the opposite way — higher LR, cosine, the progressive curriculum, and `beta0=1` each *lower the gap* (at some loss cost). The shipped `lr 2.4e-3 / beta0=0.5 / progressive` sits where that gap is bought cheaply for longer, harder-overfitting runs.

## 1. Ideal LR (REX + progressive curriculum)

| lr | loss | gap |  |
|---|---|---|---|
| 6.0e-04 | 0.0847 | +0.0114 | ⬅ likes |
| 1.2e-03 | 0.0894 | +0.0080 |  |
| 2.4e-03 | 0.0906 | +0.0052 |  |
| 4.8e-03 | 0.1039 | +0.0048 |  |
| 9.6e-03 | 0.1231 | +0.0029 |  |

**Likes:** lr ≈ `6.0e-04` (best held-out loss); if you want max regularization, `9.6e-03` gives the lowest gap (higher loss). (LR optimum is ~resolution-invariant on this proxy.)


## 2. Schedule (at the ideal LR)

| schedule | loss | gap |  |
|---|---|---|---|
| const | 0.0915 | +0.0103 |  |
| rex | 0.0847 | +0.0114 | ⬅ likes |
| cosine | 0.0927 | +0.0082 |  |
| linear | 0.0894 | +0.0087 |  |

**Likes:** `rex`. The lowest *gap* is `cosine` (more regularizing).


## 3. Warmup (linear, at the ideal LR + schedule)

| warmup | loss | gap |  |
|---|---|---|---|
| 0% | 0.0847 | +0.0114 |  |
| 3% | 0.0823 | +0.0140 |  |
| 8% | 0.0821 | +0.0137 | ⬅ likes |

**Likes warmup:** yes, ~8%.


## 4. Resolution-curriculum dependence

| curriculum | loss | gap |  |
|---|---|---|---|
| single-res | 0.0804 | +0.0169 | ⬅ likes |
| progressive | 0.0847 | +0.0113 |  |

**Curriculum Δgap:** `-0.0056` (more negative = leans harder on the data-noise for its regularization; the curriculum helps every optimizer, so use it regardless).


## 5. Optimizer-specific knobs

| variant | loss | gap |  |
|---|---|---|---|
| cautious=on | 0.0847 | +0.0114 |  |
| cautious=off | 0.0853 | +0.0106 |  |
| beta0=0 (PNM off) | 0.0826 | +0.0114 | ⬅ likes |
| beta0=1 (full PNM) | 0.0846 | +0.0096 |  |

**Best knob (by held-out loss):** `beta0=0 (PNM off)`.


---
### TL;DR — what it likes

- **LR** ≈ `6.0e-04` (proxy scale) · **schedule** `rex` · **warmup** 8% · **curriculum** single-res (Δgap `-0.0056`) · **knob** `beta0=0 (PNM off)`.

