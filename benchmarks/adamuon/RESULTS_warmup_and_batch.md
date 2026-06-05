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

**(a) AdaMuon keeps more quality-per-sample at large batch — the gap WIDENS.**
Fixed sample budget (same data, fewer steps as batch grows):

| | bs 16 | bs 64 | bs 128 |
|---|---|---|---|
| AdaMuon val | 0.0567 | 0.0673 | 0.0796 |
| AdamW-fused val | 0.0732 | 0.0936 | 0.1104 |
| AdaMuon advantage | 22% | **28%** | **28%** |

This is the documented Muon property ("scaling batch widens the gap vs AdamW"),
confirmed on this AdaMuon.

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

| batch | LR_opt | (×, vs bs 8) |
|---|---|---|
| 2 | 6e-4 | 0.5× |
| 4 | 1.2e-3 | 1× |
| 8 | 1.2e-3 | 1× |
| 16 | 2.5e-3 | 2× |
| 32 | 2.5e-3 | 2× (plateau) |
| 64 | 2.5e-3 | 2× (plateau) |

**Rule of thumb:** `lr_new ≈ lr_base · √(batch_new / batch_base)` — i.e. **double the
batch → ×1.4 the LR** (and halve the batch → ÷1.4). It holds cleanly across the
small-batch range and **plateaus around bs 16–32** (the critical batch for this
task — beyond it, a higher LR no longer compensates for the fewer updates).
Diffusion batches (images/latents are large, so bs is usually 1–8) sit safely *below*
that plateau, in the regime where √-scaling works.

(AdamW's optimum sat above the tested grid here, so its scaling isn't pinned down by
this run — the literature reports √-scaling for both, but AdamW loses more
quality-per-sample at large batch, per §2a.)

## Bottom line
AdaMuon is the better fit for "save memory → bigger batch → fewer steps":
no warmup needed (more stable than AdamW), it loses less per-sample quality as batch
grows, and its step-time penalty disappears at the larger batch. Scale LR with
`√(batch)`, plateauing past ~bs 16–32 (rarely reached in diffusion).
