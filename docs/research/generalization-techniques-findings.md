# Deep-research findings — optimizer for diffusion fine-tuning (generalization-first)

Synthesized from a fan-out web research run (21 sources, 93 claims, 25 adversarially
verified → 19 confirmed / 6 refuted). Companion to `OPTIMIZER_RESEARCH_CHARTER.md`.

## Bottom line

Our thesis is **confirmed with high confidence and theory**: the denoising-score-matching
loss minimum *is* memorization, memorization is provably worst in small data, so ranking by
the **train–val gap, not loss, is correct**, and aggressive low-loss optimization (more steps,
more width, more momentum) worsens it. Two optimizer-level families have **diffusion-MEASURED**
generalization evidence — **flat-minima/SAM** and **weight-averaging (EMA/SWA/post-hoc-EMA/
LCSC)** — but **ALL of it is from-scratch, low-res, unconditional pixel-space** (CIFAR/FFHQ/
LSUN/ImageNet) or pretraining/distillation. **None** is bf16 LoRA fine-tuning of SDXL/Flux/
Cosmos. So the mechanism is sound; the transfer to our target is **extrapolated** and must be
settled by our own `val/gap` + FID/KID test.

## Confirmed (diffusion-measured or theory)

1. **Loss minimum = memorization; gap is the right signal (HIGH).** DSM optimum reproduces
   training data; generalization comes from *not* fully minimizing. Effective memorization
   rises monotonically as data shrinks (>90% at 1k imgs, ~0 at 20k); longer training + larger
   width raise it. `2310.02664`, `2502.03435` (large-LR is anti-memorization), `2505.17638`.
   - Caveat: depth is non-monotonic (inverted-U), and loss↔memorization is directional, not
     1:1 — use the explicit held-out eps-MSE gap, not loss. `2509.25705`.
2. **Flat minima improve diffusion quality + robustness; SAM is the direct lever (HIGH).**
   ICCV 2025 (Lee et al.): ADM+SAM cuts CIFAR FID 34.5→9.0 @20 steps; lowest flatness; reduces
   exposure bias. **SAM+SWA = single best config.** `2503.11078`.
3. **Weight-averaging is a first-class, regime-dependent quality knob (HIGH).** EDM2 **post-hoc
   EMA** = set EMA length *after* training from snapshots (`2312.02696`). **LCSC** (searched
   linear combo of checkpoints) beats EMA on FID/KID/PickScore and generalizes to held-out
   (searched CC12M → better MS-COCO FID) `2404.02241`. EMA is a *weak* instance of averaging.

## Memory-budget shortlist (cheap flat-minima + averaging on the no-momentum baseline)

Ranked for the ~1–2 B/param, ≤16GB target. **Evidence strength flagged honestly:**

| direction | memory | evidence | note |
|---|---|---|---|
| **EMA / post-hoc-EMA** | +1 weight copy (or snapshots) | diffusion-measured (as wrapper) | standard practitioner regularizer; **but post-hoc EMA favors LONG training, helps little on small data** |
| **Late-phase-only SAM** | ~0 (only final epochs do 2× pass) | classification-only (`2410.10373` ICLR spotlight) | a few late epochs ≈ full SAM; amortizes the 2× cost; mechanism-strong, diffusion-extrapolated |
| **Momentum-SAM (MSAM)** | **ZERO extra** (perturb along momentum) | classification-only (`2401.12033`) | gains provably from gap-reduction; **CONFLICT: needs a momentum buffer — clashes with the β1=0 baseline** |
| **SAF (sharpness-aware "free")** | **has a memory cost** (stores past outputs; OOM-prone → MESA EMA variant +15%) | classification-only (`2022 NeurIPS`) | near-base-optimizer compute; "free" hides memory |
| **LCSC checkpoint-merge** | snapshots + search budget | diffusion-measured FID/KID | pretrain/distill context, not small LoRA; gradient-free → optimizes non-diff metrics |

Orthogonal (architectural, not optimizer — complementary): **T-LoRA** `2507.05964` shrinks
LoRA rank at high (noisy) timesteps to fight memorization.

## Refuted (do NOT carry these claims forward)

- ❌ "Weight-averaging MOVES the Pareto frontier / reaches basins SGD can't" — refuted (1-2 /
  0-3). Treat averaging as **better-tuned descent along the same tradeoff**, not frontier-moving.
- ❌ "SAM > EMA/SWA for flatness" — refuted (0-3). They're **comparable**; EMA/SWA often match
  SAM on FID (SAM's distinctive win is *robustness*/exposure-bias, not raw FID).
- ❌ "EMA reduces memorization" and "EMA drove the ImageNet-512 FID 1.81 SOTA" — both refuted (0-3).
- ❌ "High-timestep is where LoRA overfits" — contested (1-2); apply T-LoRA's rank schedule as a
  heuristic, not a law.
- ⚠️ **Bi-LoRA** (`2508.19564`): naive SAM-on-LoRA confines the perturbation to a restricted
  subspace and is **ineffective without architectural changes** — read before implementing SAM on LoRA.

## The charter questions the literature did NOT answer (our differentiated research)

These are open — *nobody published them* — so they're where koptim can contribute, settled by
our own gap+FID test (not by more lit search):

1. **Why does β1=0 / no-momentum generalize better in diffusion fine-tuning?** No primary
   source. And it **conflicts with MSAM** (which reuses momentum) — a real implementation tension.
2. **Is any Muon/AdaMuon/Shampoo/SOAP variant gap-good (not just loss-good)?** No published
   gap-specific evidence (matches our in-house finding: Muon-family good-on-loss, poor-on-gap).
3. **Schedule-free / Prodigy / Mechanic re-aimed at generalization** — not found (our Autofusion
   freeze-to-free is already in this unexplored space).

## Recommended next experiment (falsification)

On a real small-LoRA set, rank by **frozen-noise held-out eps-MSE gap + FID/KID** (the live
`val/gap` metric + a KID probe), NOT loss:

> baseline = no-momentum factored-Adam + REX + progressive-res
> vs  + **model-EMA**
> vs  + **late-phase SAM** (only final ~10–20%)
> vs  + **MSAM** (resolve the β1=0 conflict: run with a light momentum just for the perturbation)

Whichever lowers the gap / improves KID at equal-or-less memory wins. Implement EMA first
(cheapest, standard); late-SAM second (amortized cheap, mechanism-strong); MSAM third (zero-mem
but momentum-tension). Heed Bi-LoRA before SAM-on-LoRA.

## Sources (primary unless noted)
2310.02664 · 2502.03435 · 2505.17638 · 2509.25705 · 2503.11078 (ICCV25 flatness) ·
2312.02696 (EDM2 post-hoc EMA) · 2404.02241 (LCSC) · 2401.12033 (MSAM) · 2410.10373 (late-SAM) ·
NeurIPS22 SAF · 2507.05964 (T-LoRA) · 2508.19564 (Bi-LoRA) · 2303.09556 (min-SNR) ·
practitioner (secondary): OneTrainer wiki, ai-toolkit (EMA), SimpleTuner OPTIONS.
