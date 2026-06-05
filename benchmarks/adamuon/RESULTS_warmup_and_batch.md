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

## 4. Gradient accumulation = a real big batch? Does it help the noise?

Accumulate N micro-batches of size B (one optimizer step per N×B samples) — the way
to simulate a big batch under a VRAM limit.

**(a) It IS a real big batch (equivalent).** At effective batch 32 (AdaMuon, fixed
samples), three ways land on the same loss — the gradient is linear, so accumulating
micro-batches = the big-batch gradient:

| effective-32 via… | final val |
|---|---|
| real bs 32 | 0.0788 |
| accum 8×4 | 0.0783 |
| accum 4×8 | 0.0785 |
| (real bs 4, 8× more updates) | **0.0663** |

The three effective-32 runs overlap; the tiny spread is float summation order. So
accumulation faithfully reproduces the big batch — **including its downside** (fewer
optimizer updates), which is why real bs 4 (8× more steps) still wins per sample.

**(b) It does reduce gradient noise — but de-noising didn't help here.** Cosine
similarity between two independent gradient estimates over E samples (higher = less
noise):

| effective batch E | 4 | 8 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| grad cosine | 0.13 | 0.39 | 0.58 | 0.77 | 0.82 |

Noise drops monotonically with E (accum gives the same de-noising as a real big
batch). But the noisier small batch converged to a *lower* loss (§4a) — on this task
the small-batch noise was helping (more updates + regularization), not hurting.

**Takeaway:** grad-accum is exactly a big batch (use it when you genuinely need the
larger effective batch, e.g. instability at tiny batch). It is not "small-batch
quality at big-batch memory" — it carries the big batch's fewer-updates penalty.
AdaMuon's orthogonalization already normalizes the update, so small-batch noise
rarely destabilizes it → you usually don't need accumulation for stability. The
memory→bigger-batch play is for wall-clock throughput, not for the loss.

## 5. Batch-size schedules over training (vary batch as training progresses)

Idea: instead of a constant batch, schedule it over training. Two shapes, both with
cosine LR decay (AdaMuon, C=64, 16384 samples, 2 seeds, val MSE / wall-clock):
- **incr 1→64** — small batch early (many noisy updates while far from the optimum),
  big batch late (clean gradients, fast, as LR→0). Cosine *ramps the batch up*.
- **decr 64→1** — the intuitive "general/fast early, fine detail late". Cosine ramps
  the batch *down*.

**The LR coupling is decisive.** Two ways to set the LR:
- **`lr = lr_peak·cos(p)·√(batch(p)/16)`** — couples LR to the current batch (the √
  rule from §3, so each batch gets an appropriate LR), or
- **`lr = lr_peak·cos(p)`** — scheduler only, same LR for every batch.

| schedule | with √batch coupling | clean (cosine only) | wall |
|---|---|---|---|
| **incr 1→64** | **0.0589** (= best) | 0.066 | 22 s |
| decr 64→1 | 0.0738 (worst) | 0.067 | 23 s |
| const 16 | 0.0616 | 0.062 | 11 s |
| const 4 | 0.0588 | 0.059 | 42 s |

Findings:
- **The standout — `incr 1→64` WITH the √batch coupling — matched the best
  (constant-small-batch) quality at ~half the wall-clock** (0.0589 in 22 s vs const-4's
  0.0588 in 42 s). Small batch early gets its appropriately-low LR (many cheap updates
  where progress is steepest); big batch late settles cleanly as LR→0.
- **The √batch coupling is what makes it work.** Remove it and `incr` drops to 0.066
  (small-batch-early now over-LR'd); the headline win disappears.
- **`decr 64→1` (the intuitive one) is the worst WITH the coupling** — the √batch term
  inflates its big-batch-early LR (to ~5e-3) and wastes the high-LR phase on few
  updates. Removing the coupling recovers it to ~0.067 (≈ `incr`-clean), but it still
  doesn't beat a good constant batch.
- **Sobering caveat:** clean of the LR coupling, neither schedule beats a well-tuned
  constant batch here — `const 16` (0.062, 11 s) and `const 4` (0.059, 42 s) form the
  Pareto front; the schedules sit inside it. The schedule's value is real only in the
  `incr + √batch` corner (best quality at half the time).

Mechanism note: "fine detail late" argues for a **clean** gradient late (big batch /
low LR), not a noisy small batch — small-batch noise late just jitters around the
minimum and blurs detail. So the intuitive decreasing schedule has it backwards; the
detail/curriculum effect people actually exploit comes from **progressive resolution**
(low-res→high-res), where bigger latents force smaller batches late and the *data*
(not the batch) supplies the new detail.

## 6. Resolution curriculum (the *right* lever for "general → detail")

§5 showed a batch-size schedule doesn't deliver the "coarse early, fine late" idea.
The lever that *does* is **resolution**. Pure test — FIXED batch (16), CONSTANT LR
(no scheduler, no batch changes), so only the resolution varies. High-freq synthetic
images at 64² (the detail target) + a downsampled 16² version; the conv U-Net is
resolution-agnostic; **evaluate at 64²** (where the fine detail lives). 600 steps, 2
seeds.

| condition | **val @64² (detail)** | val @16² | wall |
|---|---|---|---|
| large only (ceiling) | **0.0597** | 0.339 | 6.8 s |
| **small → large** (coarse→fine) | **0.0637** | 0.296 | 6.4 s |
| mixed (interleaved) | 0.0771 | 0.151 | 6.3 s |
| large → small (reverse) | 0.1692 | 0.149 | 6.3 s |
| small only (floor) | 0.2170 | 0.140 | 6.5 s |

- **`small → large` nearly matches the `large only` detail ceiling (0.0637 vs 0.0597)
  using only HALF the high-res steps** (300 vs 600). The cheap low-res phase warms up
  the coarse structure so the high-res phase learns the detail efficiently.
- **Order is the key: you must END on the high-res (detail) data.** The reverse
  (`large → small`) collapses to 0.169 — late low-res training *erases* the
  high-frequency capability (catastrophic-forgetting-like). So "fine detail late" is
  literally correct, expressed as resolution.
- **`small only` can't reproduce detail at all** (0.217 — it never saw the high
  frequencies); `mixed` works but the ordered curriculum beats it.
- **At small scale all conditions cost the same** (resolutions are cheap), so the win
  here is "ceiling detail at no extra cost, with half the expensive steps." At
  production scale (high-res is expensive) `small→large` *also* saves compute — a
  Pareto win. This is the established progressive-/multi-resolution training practice.

Caveat: tiny synthetic task, conv U-Net, 2 seeds — the *direction* (small→large good;
ending on detail essential; small-only can't do detail) is clear and matches practice;
treat the magnitudes as illustrative. (Contrast with §5: scheduling *batch size* for
the same "general→detail" goal did not work — resolution is the right knob.)

## Bottom line (all of the above)

- **Warmup (§1):** AdaMuon doesn't need it — the orthogonalization caps the update
  RMS, so it never diverges without warmup (AdamW does at high LR). Drop it, or keep a
  tiny one out of habit.
- **Batch size (§2–3):** bigger batch *hurts* loss per sample (a speed↔quality trade,
  not free quality) for both optimizers; AdaMuon stays ~7–8% better at every batch and
  its Newton-Schulz step-tax dilutes to ~tied at large batch. Scale LR ≈ `√(batch)`
  (nearly flat in the diffusion range, plateau past ~bs 16–32). AdaMuon's LR is **~5×
  lower** than AdamW's — don't reuse the AdamW LR.
- **Gradient accumulation (§4):** mathematically *is* a real big batch — it de-noises
  the gradient but carries the fewer-updates penalty; not "small-batch quality at
  big-batch memory." Use only when you truly need the larger effective batch.
- **Batch-size schedules (§5):** scheduling batch over training does **not** robustly
  beat a well-tuned constant batch; the one bright corner (`incr 1→64` + √batch LR
  coupling) only ties constant-small-batch quality at half the wall-clock.
- **Resolution curriculum (§6):** *this* is the right "general → fine detail" lever.
  `small→large` (ending on high-res) nearly matches the high-res ceiling with half the
  expensive steps; ending on low-res erases detail. Do general→detail via resolution,
  not batch.
