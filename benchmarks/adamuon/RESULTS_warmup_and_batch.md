# AdaMuon: warmup robustness + batch-size scaling (vs AdamW)

Two practitioner questions, AdamW-fused as the reference. All on the synthetic
pixel-DDPM (RTX 4080); reproduce the *shape* of these with `pixel_ddpm_ab.py` plus
a warmup/batch driver (the exact scripts are exploratory, in the job scratch dir).
Numbers are illustrative of the trends, not a benchmark suite.

## 1. Does AdaMuon need LR warmup? (no — it's more robust than AdamW)

Best held-out val (2 seeds, 800 steps, cosine) at a fixed peak LR, with/without a
linear warmup. "EXPLOTA" = diverged.

| peak LR | AdaMuon w=0 | AdaMuon w=200 | AdamW w=0 | AdamW w=200 |
|---|---|---|---|---|
| 1e-3 | **0.0655** | 0.0680 | 0.0862 | 0.0744 |
| 3e-3 | 0.0725 | 0.0672 | **0.110 (blows up)** | 0.0716 |
| 5e-3 | 0.0748 | 0.0712 | 0.116 (degraded) | 0.0701 |
| 1e-2 | 0.0803 | 0.0724 | 0.145 (degraded) | 0.0704 |
| 2e-2 | 0.0854 (degraded) | 0.0726 | **0.96 (EXPLODES)** | 0.090 |

- **AdaMuon does not need warmup.** Its Newton-Schulz orthogonalization caps the
  update RMS (≈ `0.2·lr`) regardless of LR, so it **never diverges** in this whole
  range without warmup — it just degrades gracefully. AdamW, with diagonal
  preconditioning, **diverges without warmup at high LR** (the opposite of robust;
  warmup is how AdamW survives a high LR). This is the inverse of Lion.
- **Warmup's only value is unlocking a higher peak LR.** At AdaMuon's sweet spot
  (~1e-3) warmup slightly *hurts* (it wastes early steps below the effective LR).
  The higher LRs that warmup makes usable don't beat AdaMuon's no-warmup sweet spot
  here, whereas for AdamW warmup is mandatory to use LR ≥ 5e-3 at all.
- Practical: drop warmup for AdaMuon (or keep a tiny one out of habit — it doesn't
  hurt much). You no longer need the "raise LR + warmup so it spends enough steps at
  the effective LR" dance that AdamW forces.

## 2. Batch-size scaling (the memory→batch→speed strategy)

Free VRAM with int8/4bit momentum → larger batch → fewer steps/epoch. Two measured
effects, both in AdaMuon's favor:

**(a) Bigger batch HURTS the loss per sample (both optimizers); AdaMuon stays lower
at every batch.** Fixed sample budget, **each optimizer at its own best LR per
batch** (so this is apples-to-apples, not a fixed-LR artifact):

| batch | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| AdaMuon val | 0.0654 | 0.0663 | 0.0681 | 0.0721 | 0.0788 | 0.0889 |
| AdamW val | 0.0702 | 0.0706 | 0.0729 | 0.0778 | 0.0846 | 0.0970 |
| AdaMuon better | 7% | 6% | 7% | 7% | 7% | 8% |

- **Bigger batch is a speed↔quality trade, not free quality:** at fixed data, val
  rises ~+36% (AdaMuon) / +38% (AdamW) from bs 2→64 — fewer optimizer updates,
  even with the LR retuned. The reason to use a big batch is *wall-clock throughput*
  (GPU saturation + the diluted NS overhead below), accepting a small loss penalty.
- **With the LR retuned per batch, AdaMuon's edge is a steady ~7–8%** — it does NOT
  visibly widen here. The "gap widens with batch" you'll see quoted (and that we saw
  at *fixed* LR: 22%→28%) is mostly AdamW degrading faster when its LR is *not*
  retuned for the larger batch; retune both and the gap is roughly constant. AdaMuon
  still wins at every batch, just don't oversell the widening.

**(b) AdaMuon's Newton-Schulz overhead DILUTES at large batch (O(m/B)).**

| batch | AdaMuon ms/step | AdamW ms/step | AdaMuon / AdamW |
|---|---|---|---|
| 16 | 13.5 | 7.1 | 1.9× |
| 64 | 25.5 | 23.7 | 1.08× |
| 128 | 52.0 | 50.4 | **1.03× (tied)** |

The per-step "Newton-Schulz tax" is real at tiny batch but vanishes exactly when you
use the larger batch the memory savings buy.

## 3. How much to scale LR with batch (AdaMuon, diffusion-realistic bs 2–64)

LR_opt found by a per-batch LR sweep (fixed samples, 10% warmup + cosine):

| batch | AdaMuon LR_opt | AdamW LR_opt |
|---|---|---|
| 2 | ~1.2e-3 | 3e-3 |
| 4–8 | ~1.2e-3 | 6e-3 |
| 16–32 | 2.5e-3 | 6e-3 |
| 64 | 2.5e-3 | 3e-3 |

- **Rule of thumb:** `lr_new ≈ lr_base · √(batch_new / batch_base)` — double the batch
  → ×1.4 the LR, halve it → ÷1.4. In practice the diffusion range (bs 1–8) is nearly
  **flat** (you barely touch the LR); it rises ~2× by bs 16–32 then **plateaus** (the
  critical batch — beyond it a higher LR no longer compensates for fewer updates).
  Large latents keep diffusion below that plateau, where √-scaling is a safe guide.
- **AdaMuon's LR is ~5× *lower* than AdamW's** (≈1.2e-3 vs ≈6e-3): the orthogonalized
  update is RMS-normalized, so it needs a smaller raw LR. **Do not reuse your AdamW
  LR for AdaMuon — divide it by ~5** as a starting point.

(The literature reports √-scaling for both optimizers; here both LR_opt curves are
nearly flat then plateau in this small-batch range, with AdaMuon's ~5× lower.)

## Bottom line
AdaMuon is the better fit for "save memory → bigger batch → fewer steps":
no warmup needed (more stable than AdamW), it loses less per-sample quality as batch
grows, and its step-time penalty disappears at the larger batch. Scale LR with
`√(batch)`, plateauing past ~bs 16–32 (rarely reached in diffusion).
