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

## 7. Combining all three: resolution curriculum + memory-coupled batch + per-tier LR

The natural next question: **stack §5 + §6** — a resolution curriculum (§6) where each
tier also rides the memory→batch trade (§2) under one schedule (§5). Coarse/cheap tiers
afford a big batch; the expensive high-res tier is forced to a small batch (large
latents), exactly the diffusion reality. Same high-freq 64² data, eval @64², 2 seeds.
Tier plan `(res, micro_bs, accum, n_steps)`; `eff = micro_bs·accum`; LR per tier
`= lr_peak·cos(p_tier)·√(eff/16)`.

| config | tier plan | **val @64²** | wall | peak px/step |
|---|---|---|---|---|
| `fullres_big_ceil` (ceiling) | bs16 @64², 600 st | **0.0610** | 7.4 s | 65536 |
| **`combined_accum`** | 16²·bs64 → 32²·bs16 → 64²·bs4×**accum4** | **0.0630** | 16.1 s | **16384** |
| `fullres_small_ptc` | bs4 @64², 600 st | 0.0735 | 7.4 s | 16384 |
| `combined_pertier` | 16²·bs64 → 32²·bs16 → 64²·bs4 (no accum) | 0.0780 | 7.3 s | 16384 |

**The LR schedule is again decisive — and the first attempt failed for exactly that
reason.** A naive *global* cosine over the whole run scored **0.0894 (worst of all)**:
the LR decayed to ≈0 right as the expensive high-res detail tier began, starving the
phase that actually learns the fine detail. The fix is a **per-tier cosine (warm
restart)** so every resolution tier — especially the final high-res one — gets the full
LR. This single change moved the combined design from worst (0.089) to near-ceiling
(0.063).

What the corrected numbers say about *which* lever matters:

- **The combined design DOES reach the ceiling — but its payoff is peak memory, not
  speed.** `combined_accum` lands at 0.0630 vs the 0.0610 big-batch ceiling (within 3%)
  while peaking at **1/4 the activation memory** (16384 vs 65536 px/step). That is the
  real win: *big-batch quality at small-batch VRAM.* The cost is wall-clock — 16.1 s vs
  7.4 s (~2.2×), because accumulation runs `accum`× the fwd/bwd per optimizer step (same
  total compute as the real big batch, with no kernel-batching speedup). It is a
  **memory↔time trade**, not a free lunch.
- **The effective batch at the final high-res tier is the lever — not the curriculum
  itself.** `combined_pertier` (same curriculum, same per-tier cosine, but bs4 / no
  accum at high-res) scores 0.0780 — *worse* than simply training bs4 at full
  resolution throughout (`fullres_small_ptc`, 0.0735). On this proxy the cheap low-res
  warm-up tiers don't pay for themselves; what closes the gap to the ceiling is raising
  the **effective batch in the high-res tier** (accum4 → eff16), which `combined_accum`
  does and `combined_pertier` does not.

**So, "the best way to combine them" (answer to the design question):**
1. **LR — per-tier cosine / warm restart, scaled by `√(eff_batch)`.** A single global
   schedule kills the final detail phase; restart the cosine each tier so high-res gets
   full LR (this was the difference between 0.089 and 0.063).
2. **Resolution — small→large** (§6), ending on the detail target. Cheap and correct in
   direction, but on this small proxy it added little on its own.
3. **Batch/memory — the operative knob is a *large effective batch at the final
   high-res tier*.** When VRAM forbids a real big batch there (large latents),
   **gradient accumulation recovers ~the big-batch ceiling at small-batch peak memory**
   — paid for in wall-clock.

Net: the trio works and is genuinely useful **when you are VRAM-bound at high
resolution** (you buy ceiling-quality detail at a fraction of the activation memory).
If VRAM is *not* the constraint, just use the big batch at the target resolution — it
hits the same quality faster. The curriculum and the per-tier schedule are enablers of
the memory play, not independent quality wins on this task.

Caveat: synthetic 64² task, conv U-Net, 2 seeds, AdaMuon-only — directional, not a
magnitude benchmark. The robust, reusable findings are *(a)* never let a global
schedule starve the final high-res tier (per-tier restart), and *(b)* accumulation is
how you reach the high-res effective-batch ceiling under a VRAM cap, trading wall-clock
for memory.

## 8. LR-schedule *shape*: stable-then-cosine (the lever §5/§7 under-weighted)

§5 and §7 chased the **batch** as the knob and treated the LR schedule as a passive
cosine. The reframing here: maybe the dynamic-batch "win" was mostly the √batch LR
coupling (§5's own caveat), and the real lever is just **fix the batch to the largest
that fits at the target resolution** (= the *smallest* batch, bs4@64²) and **shape the
LR well** — specifically a *stable LR held high, then a cosine decay over the back half*
(the trapezoid / WSD schedule). Controlled test, eval @64²: every arm swept over
LR ∈ {6e-4, 1.2e-3, 2.4e-3}, **best-per-arm** (so the √batch confound is controlled by
the sweep, not by a coupling term). 2 seeds, ~6.5 s wall for *all* arms (equal compute).

| arm | schedule | batch / res | **val @64²** | best LR |
|---|---|---|---|---|
| `big_ceil` (ceiling) | cosine | bs16 @64² | **0.0601** | 2.4e-3 |
| **`lowres_big_stable_then_cos`** | stable→cosine | curriculum→bs4@64² | **0.0689** | 2.4e-3 |
| `hi_stable_then_cos` | stable→cosine | bs4 @64² | 0.0700 | 1.2e-3 |
| `lowres_big_cosine` | per-tier cosine | curriculum→bs4@64² | 0.0710 | 2.4e-3 |
| `hi_cosine` | cosine | bs4 @64² | 0.0715 | 1.2e-3 |
| `lowres_big_const` | constant | curriculum→bs4@64² | 0.0723 | 1.2e-3 |
| `hi_const` | constant | bs4 @64² | 0.0738 | 6e-4 |

- **Schedule shape is a real ~5% lever at fixed batch.** `constant` (0.0738) < `cosine`
  (0.0715) < **`stable→cosine`** (0.0700). Holding the LR high does the bulk of the
  work; a short final decay lands it without burning steps ramping up from zero. This is
  the WSD/trapezoid result, reproduced for AdaMuon on diffusion.
- **It stacks with the resolution curriculum, and a *global* trapezoid beats §7's
  per-tier restarts.** `curriculum + stable→cosine` (0.0689) is the best at bs4 memory —
  better than `curriculum + per-tier cosine` (0.0710, §7's fix). Why: the high-res tier
  occupies the *back half* of training, so a single stable→cosine holds peak LR through
  every coarse/low-res tier and decays **exactly across the detail phase**. This
  *generalizes* §7's "don't let the schedule starve the high-res tier": you don't need
  per-tier warm restarts — you need the decay to land **on** the detail phase and the
  stable-high LR to cover everything before it. (§8 supersedes §7's per-tier-cosine
  recommendation: prefer one stable→cosine timed so its decay covers the final high-res
  tier.)
- **But shaping the LR does NOT replace a real big batch.** Every bs4 arm tops out at
  ~0.069; the bs16 ceiling (0.0601) is ~13 % better at *the same wall-clock*. So
  stable→cosine is the best you can do *at that memory*, not a substitute for the bigger
  batch when VRAM allows it.
- **Reconciles §5.** With the LR swept honestly (no √batch coupling), a fixed small
  batch + stable→cosine **beats the dynamic-batch schedules** — confirming §5's caveat
  that the `incr 1→64` headline was largely an LR-coupling artifact, not the batch
  schedule itself.

**The three operating points (pick by your constraint):**

| constraint | config | val @64² | wall | peak mem |
|---|---|---|---|---|
| memory-bound **and** time-bound | bs4 + curriculum + **stable→cosine** | 0.0689 | ~6.5 s | bs4 |
| memory-bound, time to spare | + grad-accum → eff-16 (§7) | 0.0630 | ~16 s | bs4 |
| memory to spare | real **bs16** @64² + cosine | 0.0601 | ~6.5 s | bs16 |

The user's stable→cosine moved the cheap-and-fast point from 0.0735 (plain cosine) to
**0.0689** — the best return per unit of memory *and* wall-clock on this proxy.

Caveat: synthetic 64² task, conv U-Net, 2 seeds, AdaMuon — directional. The robust,
reusable takeaway: **hold the LR high and decay it (cosine) over the final/detail phase**
beats both a constant LR and a from-zero cosine, and a single well-timed decay beats
per-tier restarts.

## 9. LR-schedule *shape*, explored end-to-end (waypoint scheduler) → just use REX

§8 settled that the schedule shape matters; this section maps the shape space properly.
A general **waypoint scheduler** (piecewise smoothstep/cosine interpolation through
control points `(t, lr_frac∈[0,1])`, scaled by the swept peak LR) realizes arbitrary
shapes, so we can test the hand-drawn family directly:

- **#1 trapezoid** — stable plateau then an S-decay (= §8's `stable→cosine`).
- **#2 prodigy_bell** — a left-skewed warmup *bell* (quick ramp to an early peak ~p=0.22,
  long decay tail), imitating the effective-LR curve Prodigy produces by d-adaptation.
- **#3 waypoint_2hump** — dip early → peak at the resolution switch → tail.
- plus **REX** (`lr_max·(1−p)/((1−d)+d(1−p))`, d=0.9 — high plateau, *sharper-than-cosine*
  final drop), plain **cosine**, and **constant**.

AdaMuon, bs8 fixed, 2 seeds, LR swept per arm, eval @64². Two regimes:
**(a)** resolution curriculum 32²→64² at the midpoint; **(b)** single fixed 64² resolution.

| shape | (a) curriculum 32²→64² | (b) single-res 64² |
|---|---|---|
| **rex** | 0.0630 | **0.0619** |
| waypoint_2hump | 0.0634 | 0.0624 |
| hump_rextail (#3 front + REX tail) | **0.0629** | 0.0627 |
| trapezoid (#1) | 0.0644 | 0.0631 |
| prodigy_bell (#2) | 0.0673 | 0.0644 |
| cosine | 0.0690 | 0.0651 |
| constant | 0.0665 | 0.0655 |

**The one principle that orders the whole table:** *hold the LR high for most of training,
then let it fall to ~0 at the very end.* REX is the cleanest realization and **wins in
both regimes** — and it's the simplest (no curriculum-aware tuning, no waypoints).
**Practical recommendation: just use REX.**

- **REX is regime-robust.** It tops both the curriculum (0.0630) and the single-resolution
  (0.0619) runs, and refines §8 (REX > `stable→cosine`/trapezoid in both). It needs no
  switch to align to, unlike the 2-hump shapes whose mid-training peak only meant
  something next to the resolution switch.
- **The only ordering that flips between regimes is `const` vs `cosine`.** In the
  curriculum, `const` (0.0665) **beats** `cosine` (0.0690) because a from-step-0 cosine has
  decayed to ~half LR by the time the high-res/detail phase begins, starving it (the §7
  effect). At single resolution that reverses — `cosine` (0.0651) beats `const` (0.0655)
  because there's no detail phase to starve, so decaying-to-0 helps the final settle. REX
  satisfies the curriculum rule automatically (it holds high until ~85%, so its decay lands
  *after* the switch).

**Negative results worth recording (each kills a tempting idea):**

- **Imposing Prodigy's bell as a *fixed* schedule does NOT reproduce Prodigy** (#2 =
  0.0673, near the bottom in the curriculum). The early peak wastes the high LR on the
  coarse phase. Prodigy's benefit is *adaptation* (it auto-tunes the LR magnitude online),
  not the static shape — an argument for **KProdigy** (in-repo), not a hand-drawn bell. But
  note (§9.1): that adaptation would *not* push the LR **up** at the high-res switch — the
  optimal LR is resolution-invariant here — so KProdigy's value is removing the LR sweep,
  not finding a higher LR for detail.
- **A non-zero LR floor at the end hurts.** `rex_plateau` (hold 0.15·peak for the last 15 %
  instead of decaying to 0) scored 0.0644 vs REX's 0.0630 — the tail must reach ~0 so the
  fine detail *settles* (echoes §4: late non-zero LR jitters the minimum and blurs detail).
- **Warm restarts / double-hump are *neutral*, not helpful.** Two REX humps (warm restart)
  with monotonic resolution = 0.0631 (to-zero restart) / 0.0632 (high-valley restart) ≈
  single REX 0.0630. The extra hump neither helps nor hurts on this task.
- **…unless you couple the restart with resolution *cycling*, which hurts.** Two REX cycles
  over res 32²→64²→32²→64² scored 0.066 (vs 0.063 monotonic) — returning to low-res in the
  second cycle erases the detail learned in the first (the §6 large→small effect). Keep
  resolution **monotonic small→large**; restart the LR over it if you like, but never cycle
  the resolution back down.
- **The hand-drawn #3 + REX tail (`hump_rextail`) is the marginal best in the curriculum
  (0.0629)** but ties REX within seed noise (±0.0002) and loses the simplicity — not worth
  the waypoints over plain REX.

Caveat: synthetic 64² task, conv U-Net, 2 seeds, AdaMuon — directional. Reusable takeaway:
**REX (or any hold-high-then-decay-to-zero shape) over a monotonic small→large curriculum;
avoid from-step-0 cosine if you run a curriculum, avoid a non-zero LR floor, and don't cycle
resolution.**

### 9.1 Does higher resolution want a higher LR? (no — the optimum is ~flat)

A natural follow-up (and a correction of a hypothesis floated above): if entering the
high-res tier shifted the optimal LR *up*, that would argue for re-warming the LR at the
switch. So we swept the LR at each single fixed resolution (REX, AdaMuon bs8, 600 steps,
2 seeds, eval at the training resolution):

| train resolution | best LR | LR sweep (val, `*`=best) |
|---|---|---|
| 16² | 2.4e-3 | 3e-4:.157 6e-4:.146 1.2e-3:.142 **2.4e-3:.141** 4.8e-3:.146 |
| 32² | 1.2e-3 | 3e-4:.146 6e-4:.134 **1.2e-3:.127** 2.4e-3:.128 4.8e-3:.135 |
| 64² | 1.2e-3 | 3e-4:.069 6e-4:.065 **1.2e-3:.062** 2.4e-3:.063 4.8e-3:.068 |

- **No.** Higher resolution does *not* prefer a higher LR — if anything the optimum drifts
  slightly *down* (16²→2.4e-3, 32²/64²→1.2e-3) then stabilizes. **1.2e-3 is near-optimal at
  all three resolutions**; the optimum is essentially resolution-invariant.
- **Why:** AdaMuon's update is **RMS-normalized** (orthogonalization + factored second
  moment → applied RMS ≈ 0.2·lr regardless of gradient scale). That normalization decouples
  the optimal LR from gradient magnitude/noise — exactly what resolution changes — so the
  "more pixels = bigger effective batch = higher LR" (§3) argument is cancelled. (A
  non-normalized optimizer like AdamW might behave differently; not tested.)
- **Consequence for the curriculum:** the rule is *hold the LR **high** through the switch*,
  not *raise* it — consistent with `rex` (holds high) tying `waypoint_2hump` (peaks at the
  switch). And KProdigy over a curriculum would auto-tune the single, resolution-stable LR
  (removing the sweep), **not** discover a higher LR for the detail phase.

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
- **All three combined (§7):** stacking curriculum + per-tier `√batch` LR + memory-coupled
  batch reaches **within 3% of the big-batch ceiling at 1/4 the peak activation memory** —
  but the payoff is *memory, not speed* (accumulation pays it back in ~2.2× wall-clock).
  The operative knob is the **effective batch at the high-res tier** (via accumulation
  under a VRAM cap) — the curriculum alone added little. Worth it precisely when you're
  VRAM-bound at high resolution.
- **LR-schedule shape (§8):** the lever §5/§7 under-weighted. At fixed batch,
  `constant` < `cosine` < **`stable→cosine`** (trapezoid/WSD) — hold the LR high, decay it
  over the back half. It stacks with the curriculum (`curriculum + stable→cosine` = 0.0689,
  the best at small-batch memory) and a *single* well-timed decay **beats §7's per-tier
  restarts** (just time the decay to cover the final high-res/detail phase). This is the
  best return per unit of memory *and* wall-clock — but it still doesn't reach a real big
  batch (0.060) if you can afford the VRAM. Confirms §5's caveat: the dynamic-batch
  "win" was largely the √batch LR coupling, not the batch schedule.
- **LR shape, fully mapped (§9): just use REX.** Across a waypoint-scheduler sweep of the
  whole shape family (trapezoid, a Prodigy-style warmup bell, custom 2-hump, REX, cosine,
  constant), **REX wins in both the curriculum *and* single-resolution regimes** (0.0630 /
  0.0619) and is the simplest. One principle orders everything: *hold LR high for most of
  training, fall to ~0 at the very end.* Don't impose Prodigy's bell as a fixed shape (its
  magic is adaptation → use KProdigy), don't leave a non-zero LR floor (the detail must
  settle), and warm restarts are neutral *unless* you cycle resolution back down (which
  erases detail — keep it monotonic small→large). **And the optimal LR is
  resolution-invariant (§9.1)** — AdaMuon's RMS-normalized update decouples it from gradient
  scale, so higher resolution does *not* want a higher LR; hold it high through the switch,
  don't raise it.
