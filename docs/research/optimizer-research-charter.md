# Research charter — the best optimizer for diffusion fine-tuning

Purpose: direct a research workflow to find/design the next optimizer for `kaon`, now that
we know what to actually optimize for. This corrects the premise AdaMuon was built on.

## What we learned (the premise that was wrong)

**AdaMuon was built on the theory that reaching a low loss fast is the goal.** Our campaign
showed that is **false for diffusion fine-tuning**:

- On a small-data LoRA, **train loss is a misleading objective** — the config that minimizes
  loss does so by *overfitting/memorizing*, and produces **worse samples**.
- Empirically, a real visual A/B was won by **Adafusion with no momentum** (its "weakest"
  config) over AdaMuon (its best), despite AdaMuon's lower loss. The proxy reproduced the
  ranking only via the **train–val gap**, not absolute loss.
- AdaMuon's strength (fast convergence, low loss via orthogonalized momentum) is a **liability**
  when the risk is memorization. It has no good fit on small-data LoRA; it would only shine in
  the underfitting / abundant-data regime.
- Every "make loss lower" knob (momentum, cautious, high `rex_d`, staged curriculum) **raises
  the overfitting gap**; every regularizing knob (no-momentum, mixed/progressive resolution,
  cosine/low `rex_d`) lowers the gap at a small loss cost. **The knobs are substitutes** — once
  the optimizer regularizes, piling on more gives diminishing returns.
- Adafusion's momentum is *dispensable* (β1=0 regularizes and stays strong); AdaMuon's is
  load-bearing (it IS the orthogonalization). That structural difference is why Adafusion fits.

**Corrected objective:** not "lowest loss fastest" but **best generalization / perceptual
sample quality per unit of compute & memory**, with overfitting/memorization as the primary
failure mode for the dominant use case (small-data LoRA/fine-tune on consumer GPUs).

## The metric the research must optimize (avoid the loss trap)

Rank candidates by **generalization, not train loss**. Concretely, the evaluation protocol:

1. **Primary (cheap, on a real model):** deterministic held-out **validation loss with frozen
   noise+timestep**, and the **train–val gap**, weighted toward **mid-SNR timesteps** (the gap
   concentrates there). Track the **val-loss valley** (the early-stopping sweet spot). This is
   already implemented live in renga-flow (`val/loss`, `val/gap`).
2. **Confirmation (perceptual, real model):** **KID** (not raw FID — biased at small sample
   counts) on a few-hundred-sample set; **CLIP score** for prompt adherence; a systematic
   **visual A/B**. These are the ground truth; the gap is the cheap proxy that should *predict*
   them.
3. **Memorization / collapse:** nearest-neighbor-to-train distance, diversity (Vendi) — **only
   on a real model** (they were unreliable on synthetic pixels; do NOT trust them on a toy proxy).
4. **Report Pareto fronts** of (sample quality, memorization, memory, wall-clock) — never a
   single-loss leaderboard.

Use BOTH regimes: (a) **small-data LoRA** (overfit risk — the dominant case) and (b) a
**full/large fine-tune** (underfit/throughput case). A good optimizer should not be a disaster
in either.

## Desiderata for the optimizer (ranked)

1. **Generalization-first / memorization-resistant.** Built-in implicit regularization
   (the property that let Adafusion-nomom win). Should reach good *samples*, not just low loss.
2. **Pareto-efficient loss↔gap.** For a target sample quality, the lowest memorization; ideally
   moves the whole (loss, gap) frontier, not just trades along it.
3. **Memory-efficient.** Adafactor-class state (~1–2 B/param), bf16-correct updates (stochastic
   rounding, no Kahan/CPU offload), int8/4bit momentum dial. Must enable full-FT where AdamW OOMs.
4. **Robust / low-babysitting.** No warmup needed, never diverges across a wide LR range, LR
   ~resolution-invariant, parameter-free or near-parameter-free LR a plus.
5. **Regime-aware.** Either auto-adapts its regularization to data size, or exposes one clear
   knob (e.g. momentum on/off) to move between the overfit-risk and underfit regimes.
6. **Per-step speed** acceptable, but **secondary** to convergence-to-*quality* and to memory.
   (Real diffusion training is model-fwd/bwd-bound anyway.)

## Candidate research directions to investigate (with rationale)

For each: does it improve *generalization/samples* (not just loss)? memory cost? evidence on
diffusion specifically vs only LLMs? interaction with bf16 + LoRA?

- **Implicit-regularization-first optimizers** — why does removing momentum (Adafusion β1=0)
  help generalization? Generalize this: weak/decoupled momentum, gradient noise injection,
  label/target smoothing on the eps/v target.
- **Sharpness-aware minimization (SAM and efficient variants: ASAM, ESAM, LookSAM, SAF).**
  Directly targets *flat minima* = generalization. Cost is ~2× fwd/bwd — is the sample-quality
  gain worth it for diffusion LoRA? Any efficient single-pass approximations?
- **Weight averaging / EMA** — model-weight EMA is a known strong regularizer for diffusion;
  SWA / LAWA / latest-weight-averaging. Is the optimizer the right place, or a wrapper? Combine
  with the above.
- **Schedule-free & parameter-free LR (Defazio schedule-free, Prodigy, Mechanic/Autofusion).**
  Kill the LR/scheduler babysitting — but re-tune their target toward generalization, not just
  fast convergence (our `Autofusion` already does freeze-to-free on Adafusion).
- **Diffusion-specific loss/timestep handling baked into the optimizer or training:** min-SNR /
  sigmoid / soft-min-SNR weighting, mid-SNR emphasis (where the gap lives), immiscible/optimal-
  transport noise pairing, REPA-style representation alignment — do these reduce the gap?
- **Second-moment / preconditioning that doesn't overfit:** Adafactor-style factoring, Shampoo/
  SOAP/Muon-family — but evaluated on *generalization* (Muon-family looked strong on loss and
  poor on our gap; verify whether any variant is gap-good).
- **Explicit anti-memorization for small data:** the progressive-resolution / mixed-bucket
  curriculum we found, dropout/stochastic-depth on adapters, caption/conditioning perturbation.

## Concrete questions the research should answer

1. Among published optimizers, which have **measured generalization/sample-quality** evidence on
   *diffusion fine-tuning* (not just LLM loss)? Cite and rank.
2. Is there an optimizer that **moves the (loss, gap) Pareto frontier** vs Adafusion-nomom, or do
   all just slide along the same tradeoff?
3. Does **SAM / flat-minima** seeking beat implicit regularization (no-momentum) on diffusion
   LoRA samples, and at what compute cost?
4. Can **EMA/weight-averaging** be folded into the optimizer to get most of the regularization
   "for free" (memory permitting)?
5. What's the best **parameter-free** option whose *discovered* behavior favors generalization
   (not just fastest descent)?
6. For full-FT (the memory-bound, underfit regime), what changes — is a *different* optimizer
   warranted there, or one optimizer with a regime knob?

## Constraints (kaon context)

- Consumer GPUs (≤16 GB), bf16 weights, LoRA + full-FT; Adafactor-class memory is the budget.
- Must compose with the kaon toolkit (foreach batching, momentum codec int8/4bit, stochastic
  rounding, dtype-safe checkpointing).
- Deliverables should be falsifiable on the live `val/gap` metric (renga-flow) + a real visual
  A/B, not on a synthetic loss leaderboard.

## Baseline to beat

`Adafusion(betas=(0.0,0.999), cautious=False, bf16 stochastic-rounding)` + REX `rex_d=0.9` +
progressive-floor resolution curriculum (`512+1024 → 768+1024 → 1024`, 40/40/20, final 20%
large-only) — the current best generalization-at-good-loss recipe (see
`benchmarks/adamuon/RESULTS_generalization_and_schedule.md`). A new optimizer must beat this on
**sample quality / gap at equal-or-less memory**, not on train loss.
