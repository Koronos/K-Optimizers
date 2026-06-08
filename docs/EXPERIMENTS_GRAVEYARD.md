# Experiments graveyard — not-merged optimizer attempts (kaon / K-Optimizers)

> Why this file exists: so an agent (or human) does **not** re-implement an optimizer/technique that
> was already built and **measured**, nor merge a branch by accident. Each entry says **what**, its
> **measured verdict**, **where the branch/results are**, and its **status**. The authoritative numbers
> live in [`benchmarks/control/RANKINGS.md`](../benchmarks/control/RANKINGS.md) and
> [`benchmarks/control/RESULTS_candidates_v2.md`](../benchmarks/control/RESULTS_candidates_v2.md)
> (on branch `integration/candidates-v2`).
>
> **Read the verdicts as specialists, not a leaderboard:** the battery ranks objective
> overfitting/convergence, not perceptual quality. The axes that matter for small-data LoRA are the
> **train–val gap** (memorization-resistance) and **time→quality**, NOT lowest loss.
>
> Training-tool / speed experiments (fp8, 4-bit base, compile) live in **renga-flow**'s graveyard, not here.

Legend: ⛔ REJECTED (measured, no win) · ↩ SUPERSEDED (renamed/absorbed)

> The candidates-v2 **winners were promoted to `main`** in commit `1879645` ("Promote 6 verified
> candidate optimizers + wrapper mixins"): **AdaBelief, ScheduleFree, ADOPT, AdamP, Lookahead, SAM**.
> Those are NOT in this graveyard — see "Already on main" at the bottom. Only the candidates that
> did **not** make it are recorded here.

---

## ⛔ REJECTED — built + measured, no win on the axes that matter (NOT on main)
On `integration/candidates-v2` (per-optimizer working branches `worktree-agent-*`). All implemented
with bit-exact foreach↔per-param parity + reference tests, tuned, and benchmarked — each lands
mid-pack and heavier; **none beats the in-house Lion / Adakaon / AdaPNM** (or the 6 promoted
candidates) on any axis that matters, so none was promoted. Confirmed absent from `src/kaon/` on main.

- **MARS** — variance-reduction corrected gradient feeding AdamW. Mid-pack (mean rank ~8.9), heavier.
- **AdEMAMix** — two-EMA momentum (fast + slow). Mid-pack (~12.0). *(This is the shipped-form of the early in-house "Gemini".)*
- **Adan** — adaptive Nesterov momentum (3 buffers). Mid-pack (~11.7), among the slowest per-step.
- **Grams** — Adam magnitude × sign(current grad). Mid-pack (~12.0).
- **Adai** — adaptive per-coord inertia (flat-minima); heavy (fp32 beta1_prod), SGD-scale LR. Mid-pack (~12.0).
- **SMMF factored sign-magnitude momentum** — `feat/smmf-momentum`. Negative: **worse fidelity AND slower** than the shipped int8 momentum codec.
- **fp8 (E4M3) momentum** — `feat/4bit-momentum` (commit `c948e86`, 2026-06-04). 1 B/param, no scale/packing (just a bf16↔e4m3 cast). **DOMINATED by int8**, which is *also* 1 B/param but near-lossless and robust — fp8 occupies no useful point on the memory/fidelity frontier (int8 owns 1 B/p, 4bit owns 0.5 B/p). And it's **regime-fragile**: with small steady increments pure RTN *stalls* (EMA freezes on the coarse grid — the commit's finding, fixed with SR); but with noisier grads the per-element SR *destroys* the direction (measured weight-delta cosine vs fp32 = **0.495** SR vs int8's 0.99999, on a 15-tensor×150-step CPU replay). One mode or the other breaks depending on gradient stats — int8 at the same byte budget has neither failure. **The `4bit` momentum from this same branch DID ship to main** (`momentum_dtype="4bit"`, ~0.5 B/p, cosine ~0.975–0.99); only fp8 is rejected.

## ↩ SUPERSEDED — early names that became the shipped versions
- **Janus** — `feat/janus` → became **AdaPNM** (on main; constant-LR generalization champion).
- **Liofusion** — `feat/liofusion` → became **Lion** (on main).
- **Orphan** — `feat/orphan` (early in-house ADOPT on the factored/quantized backend) → reimplemented + promoted as **ADOPT** (on main).
- **Gemini** — `feat/gemini` (early in-house AdEMAMix) → reimplemented as **AdEMAMix** in candidates-v2 — but AdEMAMix was then **rejected** (see above), so this line did not ship.
- **integration/candidates** (v1) → superseded by **integration/candidates-v2**.

---

## Already on `main` (NOT graveyard — for contrast, so nobody re-litigates)
- **In-house:** Lion, Adakaon (factored/quantized; nomom + bf16/int8/**4bit** momentum), AdaPNM (+ RMS-clip divergence fix), AdaMuon, KProdigy, Autokaon, Gradient Centralization, the momentum codec, the control battery / RANKINGS infra.
- **Promoted candidates (`1879645`):** **AdaBelief, ScheduleFree, ADOPT, AdamP, Lookahead, SAM** (+ the WrapsOptimizer/SlowWeights schedule-free mixin and the `opt_state_bytes_per_param` wrapper-traversal fix — the "backend dividend").
  - Open validation note (not a graveyard status): **ADOPT** reaches its rank-1 gap partly by *underfitting*, and **SAM** moves the gap frontier at ~2×/step — both still want a real FID/KID LoRA test to confirm the perceptual payoff.

---

_Maintained by: update whenever an optimizer/technique is rejected, parked, or superseded. One entry with the measured verdict + branch + pointer. Training-tool/speed attempts → `renga-flow/docs/EXPERIMENTS_GRAVEYARD.md`._
