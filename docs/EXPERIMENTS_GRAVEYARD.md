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

Legend: ⛔ REJECTED (measured, no win) · ↩ SUPERSEDED (renamed/absorbed) · ⏸ PARKED (built or measured, pending decision/merge)

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
- **PAdam partial adaptivity** (`adaptivity` p<0.5 on Adakaon, 2026-06-10, MSAM campaign round 5) — the research-catalog "generalize like SGD, converge like Adam" bet (`1806.06763`). Measured on the control battery at `b1=0.9 + wd0.1`, `lr=1.2e-3`: **p=0.25 underfits badly** (te 0.0951 vs 0.0741 base), p=0.375 costs loss with NO gap win (te 0.0779, cgap +0.0152 ≈ base), composed with MSAM− it collapses further (te 0.1056; the tight cgap +0.0040 there is the consistent-underfit mirage, same trap as AdaPNM-const). The Adafactor RMS-clip does NOT make p LR-invariant in practice — convergence slows hard at fixed budget. *Caveat (per the graveyard-is-not-a-verdict rule): only tested at lr=1.2e-3, one beta1, fixed 2000-step budget; a p-specific LR sweep was not done.* The `adaptivity` knob was reverted from Adakaon at ship-time cleanup (measured-null-knob purge precedent: `decay_rate`).
- **AdamP beta1 dial** (`betas[0]` 0.5/0.2/0.0 on AdamP+wd0.05, 2026-06-10) — does cutting momentum tighten AdamP's gap while keeping the projection? **No win at any point**: b1=0.5 diverged on the scheduled run (te 0.4119), b1=0.2/0.0 trade loss away (te 0.0781/0.0790 vs 0.0742 default) for ~no scheduled-gap gain; only b1=0.0 tightens const-gap (+0.0077) but from a *worse* const loss (0.4801 — broken). Default `beta1=0.9` stays.
- **STORM variance-reduction enricher** on Adakaon (`variance_reduction`/`vr_coeff` flag POC, 2026-06-15) — momentum-based VR (Cutkosky-Orabona 2019) as a backend enricher, the hoped-for structural partner to the negative-momentum lookahead (attack the const-LR noise ball instead of sharpness). **REJECTED: variance reduction is a frontier-*slider*, not a *mover*, in this regime — it trades loss for gap (= regularization), at any implementation/strength.** Measured on the C=40 multi-res proxy (N=2000, eval@64) vs Adakaon b1=0.9 baseline (te 0.0776 / gap 0.0149): (a) cheap version (stored `g_{t-1}` on the preconditioned update): gap↓ to 0.0089 but te↑ to 0.082; (b) cheap on raw grad: BOTH worse (noise amplification, gap 0.0165); (c) **the CORRECT 2-gradient STORM** `d_t = g(x_t,z_t)+(1-a)(d_{t-1}-g(x_{t-1},z_t))` (same batch, two weight points — needs a 2nd backward, NOT free) done in the proxy loop: same slider pattern (a=0.9/0.5/0.3 → te 0.081/0.080/0.085, gap 0.0117/0.0110/**0.0040**), and **a=0.1 diverged** (te 0.93 — the tight gap there is the underfit mirage). **Why it can't move the frontier here:** on a small overfit-prone set the SGD noise *helps* fit (implicit augmentation); VR removes that useful noise → less overfit (gap↓) but worse fidelity (loss↑). No free-lunch like SAM→lookahead: STORM's correction is a finite-difference/Hessian term with no good cheap proxy (dropping it → plain momentum; stored-grad approx → fails, esp. under multi-res). The whole **cheap-VR lane was already rejected** here: see **MARS** and **Adan** above. The lookahead worked because it's **within-step** (no cross-step gradient-correlation dependence) — so the next try is the within-step **sharpness/GSAM** lineage, not more variance reduction. POC code reverted from `adakaon.py` (no residue); scripts in the session tmp.
- **GSAM surrogate-gap term on Nekaon** (`gsam_alpha` flag on MSAM, 2026-06-15) — Surrogate Gap Minimization (Zhuang et al., ICLR 2022, `2203.08065`): after SAM's perturbed gradient `g_adv`, *ascend* the component of the clean gradient orthogonal to `g_adv` (`d = g_adv − α·(g − proj_{g_adv} g)`) to minimize the sharpness/surrogate gap. The motivation was a **zero-cost** version riding Nekaon's lookahead: `g_adv` is already the perturbed-point gradient and the stored momentum `m` is the clean-gradient proxy, so the term costs no extra gradient (vs SAM/GSAM's 2×). **REJECTED on Nekaon: it SLIDES the frontier (gap↓, loss↑), does not move it — the lookahead already sits at the achievable sharpness frontier here.** Proxy (C=40 multi-res, N=2000, fp32 momentum, gsam_alpha 0/0.05/0.1/0.2): te 0.0745→0.0767→0.0771→0.0794, gap 0.0105→0.0095→0.0089→0.0086. Plain Nekaon (α=0) is the best loss point. Confirmed NOT a momentum-proxy artifact: the **true 2-gradient GSAM** (real clean grad, on Adakaon b1=0.9) was also marginal — α=0.05/0.1/0.2/0.3 only tightened gap 0.0092→~0.0085 at a loss cost, no win over plain SAM (te 0.0768). **Caveat / why it's logged not closed:** GSAM is a real frontier-mover in its papers — it just adds nothing *on top of Nekaon's negative-momentum lookahead*, which already extracts the within-step sharpness win. **It may well help on an optimizer that lacks a SAM-like mechanism** (a plain Adam/Adakaon base without the lookahead) — worth re-trying there, not on Nekaon. Together with STORM above, the 2026-06-15 search concluded: neither variance-reduction (cross-step) nor surrogate-gap (within-step++) moves the loss/gap frontier past the lookahead on this setup; the corner `cte<0.0700 AND cgap<0.0070` stays unreached (closest single points: Nekaon-b0.9-wd0.3 loss-side, Nekaon-wd0.3 gap-side). POC code reverted from `msam.py`/`nekaon.py` (no residue, `gsam_alpha=0` was a verified no-op); scripts in session tmp.

## 📉 Nekaon campaign side-results (2026-06-10) — measured, not shipped
The campaign that produced **Nekaon** (negative momentum-lookahead; ON main with the MSAM wrapper)
also measured these. Numbers in `battery_round*.log` (job tmp) and the campaign probes (pruned from
`results.json` at ship; reproducible from `registry.py` history).

- **Uphill (SAM-sign) momentum climb** — MSAM as published (`rho>0`): DOMINATED by the negative
  direction on both loss and gap at the same |rho| (cte 0.0883/+0.0079 vs 0.0875/+0.0066 at b1=0.2).
  The wrapper keeps `rho>0` available for ablation, but the kaon default direction is negative.
- **Iterate averaging at constant LR** — Lookahead(k=5,a=0.5) and ScheduleFree do NOT cut the
  const-LR noise floor on this proxy (cte 0.0825 / 0.0814 vs base 0.0805); the const-LR loss lever
  is momentum (b1→0.9), not averaging. (ScheduleFree IS now honestly measured — the battery's
  `evald()` fix evaluates it at its averaged `x`.)
- **Per-tensor perturbation radius at the global-radius rho** — `norm="tensor"` with rho=0.3 is
  ~12x the total perturbation (no √T split) → destroys training (te 0.17). Not an apples
  comparison; re-probe at rho/√T if single-pass fusion ever needs it. The shipped step-scaled
  `norm="none"` needs no cross-param sync anyway.
- **`lr_const` micro-tuning** — slides the (cte, cgap) trade at every beta1 (tiny genuine move at
  b1=0.5–0.7, ~noise scale); exactly the proxy-budget-specific knob the robust design avoids.
  Nekaon ships with `lr_const = lr` semantics (no special const-LR value).
- **beta1=0.95** — no gain over 0.9 anywhere on the dial.
- **MSAM over AdaMuon** (PoC, post-ship) — "loss specialist + gap mechanism = balance"? **No: strictly
  dominated by Nekaon on both axes** at every rho (best point rho=0.6: cte 0.0867/cgap +0.0147 vs
  Nekaon's 0.0801/+0.0067; rho=0.3 worsens BOTH vs bare AdaMuon). Mechanistic read: AdaMuon's real
  update is the Newton-Schulz *orthogonalization* of its momentum, so a perturbation along the raw
  momentum is decorrelated from where the update actually goes — the lookahead loses its
  anticipation property and degrades to badly-aimed noise. The lookahead needs a base whose update
  follows its momentum (the Adakaon family).

## ↩ SUPERSEDED — early names that became the shipped versions
- **Janus** — `feat/janus` → became **AdaPNM** (on main; constant-LR generalization champion).
- **Liofusion** — `feat/liofusion` → became **Lion** (on main).
- **Orphan** — `feat/orphan` (early in-house ADOPT on the factored/quantized backend) → reimplemented + promoted as **ADOPT** (on main).
- **Gemini** — `feat/gemini` (early in-house AdEMAMix) → reimplemented as **AdEMAMix** in candidates-v2 — but AdEMAMix was then **rejected** (see above), so this line did not ship.
- **integration/candidates** (v1) → superseded by **integration/candidates-v2**.

---

## ✅ BUILT + VERIFIED — "send more to Triton" campaign (fused kernels), 2026-06-09
All four candidates BUILT, parity-verified, and unified on `feat/fused-extra-kernels` (pending merge
to `main`). Measured with `benchmarks/fused/bench_fused.py` (new) + the battery's fused twins + a
renga-flow Anima LoKr real-config A/B. Verdicts (RTX 4080), **native-foreach → fused-total** bf16:

- **#1 batched chunked kernel** — many-same-shape >tile_cap factors (Cosmos LoKr 236× 512×512) step
  through one batched pointer-array chunked kernel; **#4 fused reductions** then removes the 248 MB
  fp32 grad stack + does GC in-kernel via dual row/col reduction kernels (subsumes #5 GC). Combined:
  **big Adakaon 4.82×, AdaPNM 5.15×** (≈13–14 ms → 2.7 ms). **#2 fused 1-D** (biases/norms): 12–34×.
  **#3 conv** `[out,in,kh,kw]` matrixized to `(out, in·kh·kw)`: 4.4–4.6×. Parity 522/522 + vs native
  (fp32 ~5e-7, bf16 <2e-2, conv ~1e-7). Toggles `_fused_big_batched` / `_fused_reductions` (default on).
- **HONEST real-workload caveat (the decisive measurement):** on the user's **Anima DiT LoKr at
  512/768/1024 px**, real `iter_sec` is **unchanged within ~1–2 % noise** vs both the prior fused and
  non-fused — because `optimizer.step` is **<1 % of the DiT iteration** at these resolutions (the
  matmul/attention fwd/bwd dominates; see [[cosmos-lokr-step-profile]]). Loss parity across all
  resolutions Δ~1e-4. So the **5× is a real optimizer-step win that only moves wall-clock in
  optimizer-bound regimes** (tiny models, huge LoRA/LoKr bags, very low res, conv/1-D-heavy full
  fine-tunes) — it is "free" (parity-correct, never slower) but NOT a throughput lever for high-res
  DiT LoKr. Full design: [`docs/FUSED_REDUCTIONS_DESIGN.md`](FUSED_REDUCTIONS_DESIGN.md).
- Side fix: the control battery now times **steady-state** ms/step (excludes one-time Triton JIT) —
  that artifact had shown a *false* "fused UNet regression".

---

## Already on `main` (NOT graveyard — for contrast, so nobody re-litigates)
- **In-house:** **Nekaon** (Adakaon + k-step negative momentum-lookahead — zero-cost flat-minima; the 2026-06-10 campaign flagship, with the **MSAM** wrapper it generalizes), Lion, Adakaon (factored/quantized; nomom + bf16/int8/**4bit** momentum), AdaPNM (+ RMS-clip divergence fix), AdaMuon, KProdigy, Autokaon, Gradient Centralization, the momentum codec, the control battery / RANKINGS infra.
- **Promoted candidates (`1879645`):** **AdaBelief, ScheduleFree, ADOPT, AdamP, Lookahead, SAM** (+ the WrapsOptimizer/SlowWeights schedule-free mixin and the `opt_state_bytes_per_param` wrapper-traversal fix — the "backend dividend").
  - Open validation note (not a graveyard status): **ADOPT** reaches its rank-1 gap partly by *underfitting*, and **SAM** moves the gap frontier at ~2×/step — both still want a real FID/KID LoRA test to confirm the perceptual payoff.

---

_Maintained by: update whenever an optimizer/technique is rejected, parked, or superseded. One entry with the measured verdict + branch + pointer. Training-tool/speed attempts → `renga-flow/docs/EXPERIMENTS_GRAVEYARD.md`._
