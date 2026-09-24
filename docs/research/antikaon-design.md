# Antikaon (provisional name) — zero-state flat-minima regularization for momentum-free Adakaon

Design note, 2026-09-24. Author: numerics/math design agent. Nothing in the repo was modified.
Companion files in this folder: `sim_drift.py` / `sim_drift.json` / `sim_drift.log` (bf16 drift
simulation), `check_regularizer.py` (Monte-Carlo check of the implicit regularizer), `antipgd.txt`,
`rwp.txt` (extracted paper text).

Legend for every claim: **[paper]** verified against the paper text, **[code]** verified in the kaon
source at main (worktree `bugfix-adakaon-unscaled-momentum`, 0814c70), **[sim]** measured by a script in
this folder, **[inference]** derivation or judgement of mine, not measured.

## 0. Verdict in three lines

The hypothesis is **sound and worth building**, with four corrections. (1) The "one fused write" is
exactly RWP/Anti-PGD on the clean iterate *provided the write uses stochastic rounding*; plain RTN is
**not** safe (it is only unbiased when the noise dithers the write, which Rademacher noise and sub-ulp
regimes do not guarantee — measured, §3). (2) Carrying ξ in the weights costs nothing extra in the
regime where the step is ≳ ulp (measured: identical walk at 1-ulp steps, +5–13 % std at 0.4-ulp
steps), but in the sub-ulp-step regime (full fine-tunes at lr≈1e-5) it raises the SR rounding walk of
the clean iterate toward the saturated ulp²/6 per step (measured 2.4× std, ≈5× variance at σ = 0.66 ulp)
— quantified in §3; it is the same regime where the noise is inert anyway, and Kahan (+2 B/param) is the
escape hatch. (3) Decoupled
weight decay must act on `z = w − ξ`, not on the live `w` (free inside the kernel). (4) The ARWP
variance formula is not scale-free as written; the scale-free version is a **quarter-power of the
normalized factored v̂**, which is exactly "Σ ∝ P" in Adam's metric and regularizes `Tr(P·H)`.

## 1. Update rule and its equivalence

### 1.1 What the papers say [paper]

* Anti-PGD (Orvieto et al. 2022, arXiv 2202.02831), eqs. (1)–(3): PGD is
  `w_{n+1} = w_n − η∇L(w_n) + ξ_{n+1}`; Anti-PGD is `w_{n+1} = w_n − η∇L(w_n) + (ξ_{n+1} − ξ_n)`
  with `ξ_n` i.i.d., mean 0, covariance `σ²I`. With `z_n := w_n − ξ_n` it is exactly
  `z_{n+1} = z_n − η∇L(z_n + ξ_n)` (their eq. 3). Eqs. (5)–(6):
  `E[z_{n+1}|z_n] = z_n − η∇L̃(z_n) + O(η E‖ξ‖³)` with `L̃(z) = L(z) + (σ²/2)·Tr(∇²L(z))`.
  Theorem 2.1 is stated for **Rademacher** noise (`±σ` with prob. ½), needs `η = Θ(ε/σ²)` small, and
  bounds the average `‖∇L̃(z_n)‖²` by `O(ε) + O(σ³)`. PGD has *no* such bias (their eq. 21:
  `E[w_{n+1}|w_n] = w_n − η∇L(w_n)`), the noise just accumulates as a random walk.
  Experimental σ: 0.05 at η=0.01 (quadratic model), 0.1 at η=0.001 (matrix sensing), "robust to
  hyper-parameters" (App. D). For ResNet-18/CIFAR-10 they *stop* injecting noise before evaluating.
* RWP (Li et al., TMLR 2024, arXiv 2404.00357): RWP is `w_{t+1} = w_t − γ_t ∇L_B(w_t + ε_r)` (their
  eq. 10) with `ε_r` i.i.d. per step — **identical to Anti-PGD's z-form**. Filter-wise generation
  `ε_r ~ N(0, σ²·diag({‖w^(j)‖²}))` (Bisla et al. 2022 convention, motivated by scale invariance under
  BN/ReLU). m-RWP: `λ∇L(w) + (1−λ)∇L(w+ε)` with `λ = 0.5`, two *different* batches, σ=0.015 CIFAR /
  0.005 ImageNet (m-ARWP); plain RWP/ARWP σ = 0.01 CIFAR, 0.003 ImageNet, cosine-increasing σ schedule.
  ARWP eq. (14): `ε_{r,t} ~ N(0, σ²I / sqrt(1 + η_a Σ_{i<t} β^{t−i−1} g_i²))`, `g_i` = the
  *historical perturbed* gradients, defaults η_a = 0.1, β = 0.99 (their sensitivity section; the
  hyper-parameter list once says η = 1, the experiments use 0.1). Their Fig. 1: RWP needs a
  perturbation radius **~two orders of magnitude larger** than AWP (SAM) to reach the same expected
  perturbed loss — "random" noise is far less efficient per unit radius than a directed climb.

**Key identity [paper+inference]:** Anti-PGD's clean iterate `z` and single-sample RWP are the *same
algorithm*. "Keeping ξ in the live weights" is therefore not a new optimizer; it is the snapshot-free
*implementation* of RWP. Everything the RWP literature knows (radius trade-off, ARWP shaping, m-RWP)
transfers verbatim.

### 1.2 The proposed rule on momentum-free Adakaon [code+inference]

Adakaon's per-parameter no-momentum step (`adakaon.py::_step_one_param`, lines 1651–1715 [code]) is:
`row,col ← EMA(g²+eps1)`; `P_n = r ⊗ c = diag(v̂_n)^{-1/2}` (rank-1, `_factored.py:63`);
`u_n = P_n g_n / max(1, rms(P_n g_n)/clip)`; `δ_n = u_n + wd·w_n` (cautious is a no-op without
momentum [code docstring]); `w ← w − η·δ_n` through `subtract_one_` (SR for bf16).

Antikaon step `n` (live bf16 weights hold `w_n = z_n + ξ_n`):

```
g_n      = ∇L_{B_n}(w_n)                       # gradient AT the perturbed point (that is the mechanism)
ξ_n      = σ_n · S_n ⊙ ε(seed, pid, n)         # regenerated: S_n from row/col BEFORE this step's EMA update
row,col ← EMA(g_n² + eps1)                     # v sees the perturbed gradient (correct, see 1.3)
u_n      = clip_rms( (r_n ⊗ c_n) · g_n )
δ_n      = u_n + wd · (w_n − ξ_n)              # decay acts on z, NOT on the live w  (free: ξ_n is in hand)
ξ_{n+1}  = σ_{n+1} · S_{n+1} ⊙ ε(seed, pid, n+1)   # S_{n+1} from the row/col just updated
w_{n+1}  = SR( w_n − η δ_n + (ξ_{n+1} − ξ_n) )     # ONE write, fp32 accumulate, stochastic rounding
```

so that `z_{n+1} = z_n − η δ_n(z_n + ξ_n) + r_n` where `r_n` is the SR rounding error of that single
write (E[r_n | everything] = 0). Nothing persistent is stored: `ξ` is a pure function of
`(seed, param index, step, group lr, row/col)`. The two scalars it depends on that can change between the
write and the removal — the group `lr` and the noise scale — are frozen per step in Python floats
(`_xi_lr[gid]`), the same trick MSAM uses for `_estep_scale`/`_eclamp` [code msam.py:258–296].

Implementation identity that keeps the existing writer and the foreach/per-param bit-parity: since
the write is `p −= lr·delta`, pass `delta' = delta − (ξ_{n+1} − ξ_n)/lr` (one fp32 FMA) into the
unchanged `subtract_one_` / `subtract_batched_` [code _backend.py:232,272].

### 1.3 Does v see the right gradient? [inference, consistent with paper]

Yes, and it should. `E[g_i(z+ξ)²] = g_i(z)² + Σ_j H_ij² Σ_jj + …`, so `v` is inflated by the
perturbation-induced gradient component `Hξ`. Two reasons this is right: (i) ARWP itself builds its
shaping from the *historical perturbed* gradients [paper]; (ii) the update direction is `P·g(z+ξ)`,
so the normalizer must be the second moment of *that* quantity or the RMS-clip and the Adam-metric
stability argument break. Consequence to be aware of: in the sharpest directions the term
`|Hξ|_i/√v_i ≈ k_σ·η·λ_P` is comparable to the true gradient once `k_σ ≳ 1` (η·λ_max(P^{1/2}HP^{1/2}) ≈ 2
near Adam's edge of stability), so the noise both drives the flatness bias *and* damps the effective LR
there through `v` — a self-stabilizing pair, which is why RWP tolerates much larger σ under adaptive
optimizers than under SGD. The RMS-clip bounds the total step regardless.

### 1.4 The implicit regularizer with the preconditioner [inference, verified by MC]

Second-order Taylor expansion with a *fixed* covariance Σ and a *fixed* symmetric preconditioner P:

`E_ξ[ P ∇L(z+ξ) ] = P ∇L(z) + ½ P Σ_{jk} Σ_jk ∇(∂_jk L)(z) + O(E‖ξ‖³) = P ∇( L + ½ Tr(Σ·∇²L) )(z)`.

**P does not enter the regularizer; it only sets the metric of the descent.** Verified by Monte Carlo
on a random cubic+quartic in 6-D with diagonal anisotropic Σ and diagonal P: the MC mean of
`P·∇L(z+ξ)` matches `P·∇(L + ½Tr(ΣH))` within 1.06 standard errors (SE 1e-5) and rejects the bare
`P·∇L` at 374 SE [sim: `check_regularizer.py`]. Hence:

| noise covariance Σ | implicit objective | remark |
|---|---|---|
| `σ²I` (plain SGD or Adakaon, no shaping) | `L + ½σ² Tr(H)` | Anti-PGD/RWP as published |
| `σ²·diag(‖w_j‖²)` filter-wise | `L + ½σ² Σ_j ‖w_j‖² Tr(H_jj)` | scale-invariant sharpness (Bisla) |
| `σ² v̄^{1/2}·P`, `P = diag(v̂)^{-1/2}` (ARWP-shaped, §2.3) | `L + ½σ² v̄^{1/2} Tr(P·H)` | **preconditioned sharpness**, the quantity that governs Adam's edge of stability |

Caveats: (a) when Σ depends on z (weights or v̂) the expectation is `½Σ_jk Σ_jk(z)∂_i∂_jk L`, which is
the gradient of `½Tr(ΣH)` *only with Σ frozen* — a quasi-gradient; Σ from a slow EMA is effectively
frozen per step (same status as Adam's `v`). (b) The RMS-clip is a per-tensor scalar and the cautious
mask is a no-op without momentum, so neither changes the expectation argument beyond a scalar. (c) The
three terms in `δ_n` see different points only if wd acts on `w` instead of `z`: that would add a
zero-mean `−η·wd·ξ_n` jitter to `z` (variance `η²wd²σ²` per step, tiny but pointless) — hence the
`(w_n − ξ_n)` in the rule.

## 2. Noise shape and scale

### 2.1 Units: measure σ in optimizer steps [inference, precedent in code]

Adakaon's update is RMS-normalized (`rms(u) ≤ clip = 1`, typically ≈1 without momentum), so a
coordinate moves ≈ `η` per step regardless of its weight scale. Nekaon measures its lookahead in
**steps** (`e = k·η·m`, `norm="none"`; nekaon.py docstring: "the step-unit formulation is the
invariant", validated under LR ×0.5/×2 [code]). Follow that precedent:

`σ_n = k_σ · η_n · clip_threshold` (per coordinate RMS of ξ), with `k_σ` dimensionless.

* LR-scale invariance: `k_σ` is a lookahead-like radius in steps; halving the LR halves σ, exactly as
  Nekaon's `k` behaves (the *regularization strength* `½σ²Tr(ΣH)` then scales as η², which is the
  behaviour Nekaon's calibration transferred with; the alternative anchor below is the ablation).
* Tensor-scale invariance: a step-unit σ is already scale-free in Adam's metric (all coordinates move
  ~η/step). It is *not* invariant to a scale-invariant reparametrization of a normalized layer, which is
  what the filter-wise `‖w_j‖` anchor gives. Offer it as `sigma_ref="weight"`:
  `σ_ij = k_w · max(RMS_row_i(w), k_σ·η)` — the floor is what makes it work for LoRA `B = 0`.
* LoRA floor: with `sigma_ref="step"` no floor is needed: a fresh `B` row is displaced by ~`k_σ` steps
  of noise, the same scale its first updates have. With the weight anchor the floor above applies.
* Bounded noise: use **Rademacher** `ε ∈ {−1,+1}` by default (Anti-PGD's Theorem 2.1 is stated for it
  [paper]; same covariance → same regularizer to second order; `max|ξ| = σ`, no tails to blow a
  bf16 coordinate; one `tl.rand` per element). `noise="gaussian"` as an option (`tl.randn`).

### 2.2 Equal-perturbation comparison with Nekaon [inference]

Nekaon `k=1.5`: `RMS(e) = 1.5·η·RMS(m)`, and `RMS(m) ≈ sqrt((1−β1)/(1+β1))·RMS(u) ≈ 0.58` at β1=0.5
for uncorrelated update directions (up to 1 for coherent ones) → **RMS(e) ≈ 0.9–1.5 η**, i.e.
`k_σ ≈ 1–1.5` is "equal RMS perturbation". But equal RMS is not equal *effect*: a directed climb along
the momentum acts on the top curvature (`‖He‖ ~ λ_max·ρ`) while random noise spreads over all
directions (`E‖Hξ‖² = Tr(HΣ)`); Li et al. measure that RWP needs ~100× the L2 radius of SAM for the same
expected perturbed loss [paper]. Their per-coordinate σ = 0.01 on ResNet conv weights (RMS ≈ 0.03–0.06)
is a 15–30 % relative displacement; on our proxy (η = 1.2e-3, RMS(w) ≈ 0.05) that is `k_σ ≈ 6–12`.

**Recommendation:** gate sweep `k_σ ∈ {1.5, 5, 15}` (equal-RMS-to-Nekaon, mid, RWP-literature scale),
extend to 40 if the gap is still improving monotonically at 15. Default to be *measured*; my prior is
`k_σ = 5`. Report each run's realized `RMS(ξ)/RMS(w)` so the numbers can be compared with the papers.

### 2.3 ARWP shaping from the factored v̂, without materializing anything [inference from paper eq. 14 + code]

ARWP's std is `σ·(1 + η_a S_t)^{-1/4}` with `S_t ≈ v_t/(1−β)`, i.e. **variance ∝ v^{-1/2} ∝ P** for
`η_a S ≫ 1`: ARWP is "Σ ∝ Adam's preconditioner" with a floor of σ. Written literally it is not
scale-free (`η_a` fixes an absolute gradient scale). Scale-free version:

```
S_ij = clamp( (v̄ / v̂_ij)^{1/4}, 1/s_cap, s_cap ),   v̄ = mean_ij v̂_ij
```

With Adakaon's factors [code _factored.py:110–115]: `v̂_ij = (row_i / mean(row)) · col_j`, so
`v̄ = mean(col)` and

`S_ij = mean(col)^{1/4} · sqrt(r_factor_i) · sqrt(c_factor_j)` — a rank-1 product of a length-R and a
length-C vector, both obtained from the `r_factor`/`c_factor` the step already computes (one `sqrt` each
and one scalar per tensor). For 1-D params `S_i = (mean(v)/v_i)^{1/4}`. The cap (`s_cap = 4` proposed:
no coordinate gets more than 4× the mean noise, i.e. dead coordinates with `v̂ → 0` do not get infinite
noise) plays the role of ARWP's `1 + …` floor. `E[S²] ≈ 1` by construction so `k_σ` keeps its meaning.
`shape="none"` (S ≡ 1) is the ablation arm. The shaped variant regularizes `Tr(P·H)` (table in §1.4);
which of the two correlates with the gap in this regime is precisely the experimental question.

### 2.4 The convergence/regularization trade-off knob (m-RWP without a second gradient) [inference]

m-RWP's `λ` mixes a clean gradient in, which costs a second backward. Two free analogues on the same
one-write mechanism: **duty cycle** (`ξ_n = 0` on steps with `n mod d ≠ 0`; regularizer strength ×1/d)
and **antithetic pairs** (§5), which cancel the first-order noise injected into `z` without touching the
regularizer. Antithetic pairs are the one I would ship as an option; duty cycle only if k_σ=15 shows the
"convergence issue" RWP reports.

## 3. bf16 numerics

### 3.1 Analysis [inference]

Write `x_n = w_n − ηδ_n + Δξ_n` (fp32), `w_{n+1} = R(x_n)`, `f = frac(x_n/ulp)`.

* **SR**: `E[r|x] = 0`, `Var[r|x] = f(1−f)·ulp²`. With `|Δξ| ≫ ulp` the fractional part is uniform →
  `Var = ulp²/6` per step, **whatever the update size**. Plain SR training with sub-ulp updates has
  `Var ≈ |ηu|·ulp` per step (`f ≈ |ηu|/ulp ≪ 1`). So carrying ξ raises the SR walk of `z` only when
  `|ηu| < ulp/6`, by a factor up to `ulp/(6|ηu|)`; above that the baseline walk is already saturated and
  the design costs **zero** extra rounding noise. `E[z_rec − z_ref] = 0` always (SR is conditionally
  unbiased regardless of what is added). Over N steps: `RMS drift ≤ ulp·sqrt(N/6)` (41 ulp at 10k) in the
  worst case of *no feedback*; real training corrects rounding errors in curved directions with time
  constant ~1/(ηλ) steps, flat directions keep them (and do not matter to first order).
* **RTN**: `r(x) = RTN(x) − x` is a deterministic sawtooth of period ulp. If the *added* noise has a
  smooth density of width ≥ ulp, `r` becomes uniform on `±ulp/2`, zero-mean and independent of `z`
  (**subtractive dither**: we know ξ exactly and subtract it) with `Var = ulp²/12` — half of SR, no RNG
  for rounding. But two conditions: the dither must be *spread* (Gaussian/uniform, **not Rademacher**,
  whose two values `z ± σ` leave the fractional part non-uniform), and `σ ≥ ulp(w)` per coordinate
  (bias of the smoothed sawtooth ∝ `exp(−2π²σ²/ulp²)`: negligible at σ = ulp, ~0.7 % of an ulp per step
  at σ = ulp/2, which accumulates coherently). Below that RTN is biased.
* **Inert condition**: the perturbation is realized only if `σ_i ≥ ½ ulp(w_i)`, i.e.
  `k_σ·η ≥ |w_i|/256` (bf16 has 7 explicit mantissa bits). With SR a sub-ulp ξ is realized
  stochastically as `{0, ±ulp}` with the right mean — unbiased but with a *different* (larger) covariance
  than designed; with RTN it vanishes. Emit MSAM's inert warning [code msam.py:313–392] at the same
  threshold, in the same sampled way.
* **Two writes per step (MSAM style)** is strictly worse: a `−ξ_n` then `+ξ_{n+1}` pair does not cancel
  under SR (two independent draws: measured 19 % L2 drift over 4000 cycles in the MSAM campaign
  [code docstring]) and under RTN needs `|ξ| < ½ulp` to be exact, which is the inert regime. The single
  combined write is what makes the design work at all.
* **Kahan** (`bf16_method="kahan"`, +2 B/param, already supported by `subtract_one_` [code]) makes the
  combined write exact to ~fp32; it is the escape hatch for the sub-ulp regime, not the default.
* **Eval/train round trip**: `w_eval = RTN(w − ξ_n)` has error ≤ ½ ulp(z) — the best bf16 representation
  of `z_n`, which is never exactly representable, so this *is* "exact removal" in the only sense
  available. `w_train = RTN(w_eval + ξ_n)` returns to the original `w_n` except where the subtraction
  crossed a binade upward (rounding error of the first RTN up to `ulp(w)` there) — a one-time, zero-mean,
  ≤1 ulp drift per eval cycle on a small fraction of coordinates.

### 3.2 Simulation [sim: `sim_drift.py`, results in `sim_drift.json`]

Setup: 2^20 bf16 coordinates, `w0 ~ 0.05·N(0,1)`, 10 000 steps, prescribed fp32 update stream
`δ = η·(0.1·c + 0.99·ν)` (`c = ±1` re-drawn every 50 steps, `ν ~ N(0,1)`; RMS ≈ 1 like Adakaon's `u`,
net movement ≈ the weight scale at η = 1e-3), Rademacher `ξ = k·η·s` regenerated from `(seed, step)`,
kaon's own `add_stochastic_` for SR. `z_rec = w.float() − ξ_N` vs an fp32 reference on the same stream.
Errors in units of `ulp_ref = ulp(RMS(z))`. No feedback (worst case).

Full table: run `summarize.py sim_drift.json` (46 runs). Condensed (bias = mean signed error of
`z_rec − z_ref` in `ulp_ref` ± its standard error; std = its spread; `err/moved` = ‖error‖ / ‖total
movement of z‖; RMS(w) = 0.05, `ulp_ref = 2.44e-4`):

| regime (step/ulp) | scheme | k_σ (σ/ulp) | bias ± se (ulp) | std (ulp) | rel L2 vs ‖z‖ | err/moved |
|---|---|---|---|---|---|---|
| lr 1e-3 (1.02) | sr (baseline, no ξ) | 0 | −0.02 ± 0.02 | 24.0 | 0.178 | 0.19 |
| | **sr+ξ** | 0.5 / 1 / 4 / 16 (0.5–16) | all within ±1σ of 0 | **24.0 / 24.0 / 24.0 / 24.4** | 0.178–0.181 | 0.19 |
| | rtn+ξ | 0.5 / 1 / 4 / 16 | within ±2σ of 0 | 16.9 / 16.9 / 16.9 / 17.1 | 0.125 | — |
| | 2w-rtn (MSAM-style) | 1 / 4 | ok | 26.5 / **47.5** | 0.197 / 0.353 | — |
| | 2w-sr | 1 / 4 | +0.08 ± 0.03 | 33.2 / 35.9 | 0.247 / 0.267 | — |
| | kahan+ξ | 4 | 0.00 | **1.43** | 0.011 | — |
| lr 1e-4 (0.41) | sr (baseline) | 0 | +0.06 ± 0.04 | 41.4 | 0.196 | 0.92 |
| | **sr+ξ** | 0.5 / 1 / 4 / 16 (0.2–6.6) | within ±1.5σ | **43.4 / 45.9 / 45.6 / 46.7** | 0.206–0.221 | 0.91–0.96 |
| | rtn+ξ | 0.5 / 1 / 4 / 16 | within ±2.2σ | 28.8 / 30.9 / 30.6 / 31.9 | 0.136–0.151 | 0.61–0.68 |
| | 2w-sr | 1 / 4 | +0.09 ± 0.08 | **79.3 / 82.5** | 0.376 / 0.391 | 1.6 |
| | kahan+ξ | 4 | 0.00 | **0.67** | 0.003 | 0.013 |
| | sr+ξ / rtn+ξ, Gaussian | 4 | −0.05 ± 0.05 / −0.03 ± 0.03 | 47.6 / 33.8 | 0.226 / 0.160 | — |
| lr 1e-5 (0.041) | sr (baseline) | 0 | +0.01 ± 0.02 | 16.7 | 0.082 | 3.4 |
| | **sr+ξ** | 0.5 / 1 / 4 / 16 (0.02–0.66) | within ±2σ | **18.4 / 21.7 / 33.6 / 39.3** | 0.090–0.192 | 3.7–7.9 |
| | rtn+ξ | 0.5 … 16 | +0.006 ± 0.005 (but corr. with drift +0.03) | 4.8–5.7 | 0.023–0.028 | **0.96–1.14 (stalled)** |
| | 2w-rtn | 1 / 4 | ok | 16.7 / 16.8 (= baseline: the pair is **inert**, σ < ½ulp) | 0.082 | 3.4 |
| | 2w-sr | 1 / 4 | ok | 31.3 / **51.8** | 0.153 / 0.253 | 6.3 / 10.4 |
| | kahan+ξ | 4 | 0.00 | **0.49** | 0.002 | 0.10 |

Eval/train round trips (100 cycles `RTN(w−ξ)` then `RTN(w_eval+ξ)`): drift std ≤ 0.1 ulp_ref in every
run (0.005–0.096), zero mean, fraction of coordinates touched 0.03 %–11 % (grows with σ/ulp, the
binade-crossing mechanism of §3.1); the eval view itself is within 0.49 local ulp of the fp32 `z` (max).

Findings:
1. **SR on the combined write is unbiased everywhere** — 46 runs, every bias within ~2 standard errors
   of zero, no sign pattern, no correlation with the drift direction. **[sim]**
2. **Cost of carrying ξ under SR**: none at ≥1-ulp steps (24.0 vs 24.0 at every k); +5–13 % std at
   0.4-ulp steps; in the sub-ulp regime (0.04-ulp steps, full fine-tune at 1e-5) the walk grows from
   16.7 to 18–39 ulp with σ (2.4× std, ~5× variance at σ = 0.66 ulp), saturating toward the predicted
   `sqrt(N/6) = 40.8`. Exactly the §3.1 prediction. Note that in that regime plain SR's walk is already
   3.4× the total movement (no-feedback model): bf16+SR at 1e-5 is a noisy regime with or without ξ.
3. **RTN on the combined write** is unbiased with √2 lower variance when either the step or σ is
   ≳ 0.4 ulp (the continuous update noise dithers it, so even Rademacher passes there), but in the
   sub-ulp regime it **stalls** — error ≈ 100 % of the movement and correlated with the drift — the
   classic RTN failure SR exists to prevent. Not a safe default.
4. **Two writes per step** are strictly worse: SR pairs double the walk (79–82 vs 41 at 1e-4;
   MSAM's finding reproduced), RTN pairs are inert below ½ ulp and blow up above it (47.5 at k=4, 1e-3).
5. **Kahan** makes the combined write exact to 0.5–1.4 ulp over 10 k steps (rel. L2 0.2–1 %), at +2 B/param.
6. Gaussian vs Rademacher: no difference for SR (47.6 vs 45.6, within run noise).

### 3.3 Scheme decision

**SR on the single combined write** (unbiased for every noise law and every σ, costs no extra rounding
noise once `η·rms(u) ≳ ulp/6`, and is what every bf16 kaon optimizer already does). RTN only as an
opt-in for Gaussian noise with `σ ≥ ulp` (halves the walk variance). Kahan for the sub-ulp regime. Never
two writes.

## 4. Eval / checkpoint / resume semantics

* `eval()`: `w ← RTN(w.float() − ξ_n)` for every param that has state; `train()`: `w ← RTN(w.float() + ξ_n)`.
  Idempotent via a `_train_mode` flag (reuse `TrainEvalWeights`' plumbing pattern [code _wrappers.py:214]).
  Error bounds in §3.1. `step()` outside train mode raises, as MSAM does [code msam.py:677].
* `state_dict()` additions (a few scalars, no tensors): `noise_seed` (int64), `step` (already `_t`),
  `noise_backend` ("triton" | "torch"), `train_mode` (must be `False` — a train-mode checkpoint carries
  ξ and a fresh optimizer cannot know to remove it; raise on load exactly like `_msam_meta.train_mode`
  [code msam.py:746]), per-group `xi_lr` (the lr frozen into the live ξ; unneeded when saved in eval
  mode, kept for the invariant). Since ξ_n is regenerated from `(seed, pid, n)` and `S_n` from the
  restored `row/col`, resume is exact: the checkpoint holds `RTN(z_n)`, `train()` re-installs
  `RTN(RTN(z_n) + ξ_n)`.
* Param identity `pid` = index in the flattened `param_groups` order (the same key
  `WrapsInnerOptimizer.state_dict` uses [code _wrappers.py:325]); add a param group → new indices only at
  the end. Document that re-ordering params between runs changes the noise (same class of caveat as SR
  stream ids [code _stochastic_rounding.py:225–233]).
* Noise backend: the Triton kernels use Philox via `tl.rand(seed, offs)` [code _fused_triton.py:434];
  the torch path would use a `torch.Generator`. The two streams differ, so a run's ξ identity is its
  backend: record it and refuse to `train()` a checkpoint under the other backend (or, cleaner, generate
  ξ for the torch path on CUDA with a tiny Triton `noise` kernel so CUDA has one stream and only CPU
  differs).
* Gradient accumulation: ξ_n is fixed between two `step()` calls (no write happens), so every
  micro-batch sees the same perturbed point — the correct semantics (one RWP sample per optimizer step).
  Multiple optimizer steps on one batch: each step draws a new ξ; fine.
* Trainer-side EMA of weights, `torch.compile`, sharding: an EMA of the live `w` averages a zero-mean
  σ-jitter in (harmless, `σ·sqrt((1−β)/(1+β))`); a trainer that saves `model.state_dict()` without
  `optimizer.eval()` bakes ξ in — the same footgun MSAM/Nekaon already document.

## 5. Correlated vs i.i.d. noise

* **i.i.d. ξ per step (RWP) and the Anti-PGD increment are the same thing** on the clean iterate
  (§1.1); "PGD" (accumulating noise, `w_{n+1} = w_n − ηg + ξ`) is the one with no regularizer
  [paper eq. 21]. There is no choice to make here.
* **AR(1)** `ξ_{n+1} = ρξ_n + sqrt(1−ρ²)ε_{n+1}`: the one-step conditional mean still gives
  `½Tr(ΣH)` with the *marginal* Σ (the correlation only enters at `O(η²ρσ²)` through the previous step's
  displacement). What ρ > 0 changes is the **variance** injected into `z`: the per-step noise `−ηPHξ_n`
  becomes correlated and its accumulation grows by `(1+ρ)/(1−ρ)` — a bigger noise ball for the same
  regularizer, i.e. strictly worse (Anti-PGD's App. C, Remark C.6, says the same for correlated
  injection: "as ρ→1 the total accumulated variance explodes" [paper]). Regeneration from the seed in
  closed form needs `ξ_n = Σ_k ρ^{n−k} sqrt(1−ρ²) ε_k`, i.e. `O(K)` Philox draws per element per step
  (K ≈ 60 for ρ = 0.9 at 1e-3 truncation) or 4 B/param of state. **Discard.**
* **Antithetic pairs** (the one correlated option that adds something, cost 0, closed form):
  `ξ_{2m} = +ε_m`, `ξ_{2m+1} = −ε_m` (`seed = m = n//2`, sign = `(−1)^n`). Marginal Σ unchanged, so the
  regularizer is unchanged; the Hutchinson-type term `ξ⊗ξ` is identical for both signs, so its variance
  halves per pair; and the first-order noise `−ηPH ξ` injected into `z` cancels across the pair up to
  `O(η²σ‖∇³L‖‖u‖)` (z moves between the two steps). This attacks exactly the "convergence issue" RWP
  reports at large σ without a second gradient, and it removes the `ulp`-scale SR dithering of nothing
  (the pair's `Δξ` is `−2ε_m` then `ε_{m+1}+ε_m` — still one write). Ship as `antithetic=True` option;
  ablation arm in §6.

## 6. Minimal falsifiable plan (control battery)

Protocol: constant LR, held-out loss and train–val gap measured through `evald()` (which calls
`opt.eval()` [code benchmarks/control/battery.py:77–84] — mandatory here: the *train-mode* loss is at the
perturbed point and is higher by `≈½Tr(ΣH)`). Gate C=40, N=600 → 2000, seeds 43/44; then C=128 N=2000;
then Anima LoRA (subject-split, KID probe as the research note prescribes).

Arms (all bf16 weights, `bf16_method="stochastic_rounding"`):

| arm | config | why |
|---|---|---|
| A0 Adakaon-nomom | `betas=(0,.999)`, `cautious=False`, lr 1.2e-3 | the base (0.032 B/param) |
| A1 Nekaon | defaults (`k=1.5`, β1=.5, 4bit, wd .1) | the reference to beat (0.56 B/param) |
| B1–B3 Antikaon | `k_σ ∈ {1.5, 5, 15}`, `shape="v"`, Rademacher, i.i.d. | main sweep |
| C1 ablation | `k_σ=5`, `shape="none"` | is the `Tr(PH)` shaping doing anything? |
| C2 ablation | `k_σ=5`, `antithetic=True` | does cancelling the injected noise buy loss at equal gap? |
| C3 (optional) | `k_σ=5`, `sigma_ref="weight"` | anchor question; only if B shows any signal |

Success criterion (frontier mover, not slider): at the same budget and seeds, a B/C arm must
(i) beat A0 on **both** axes by more than the two-seed spread, and (ii) be non-dominated by A1
(loss ≤ A1's *or* gap ≤ A1's) — at 0.032 B/param that is a new frontier point by itself; the corner
`te < 0.0700 ∧ gap < 0.0070` recorded as unreached in the graveyard is the stretch target. If every
B arm only trades loss for gap along the A0→A1 line (the STORM/GSAM pattern), the hypothesis is refuted
for this regime — log it in the graveyard with the numbers. Also run B2 at lr ×0.5 and ×2 once to check
the step-unit anchor transfers (Nekaon's validation).

Cheap diagnostics, all at the end of training on fixed batches, in eval mode:
* **Expected sharpness at the training radius**: `E_ξ[L(z+ξ)] − L(z)` with 8 fresh ξ (same S, same
  σ as trained), train and val batch. Forward passes only. This *is* the regularizer `½Tr(ΣH)`;
  Anti-PGD's PAC-Bayes reading (their Thm 2.2/eq. 8) makes it the quantity that should predict the gap.
  Report it for A0/A1 at the B2 radius too (equal-perturbation comparison across optimizers).
* **Hutchinson `Tr(H)` and `Tr(PH)`**: 8 probes `εᵀHε` (Rademacher) and `εᵀHε` with `ε ~ S·Rademacher`
  through `torch.autograd.functional.hvp` — feasible on the synthetic U-Net (small, double-backward
  through GroupNorm/conv is fine); log the same for the LoRA run if the DiT's double backward fits in
  8 GB, otherwise skip it there.
* Free running trace: `loss(train-mode) − loss(eval-mode)` on the same batch every N/10 steps = the
  perturbation cost during training (RWP's convergence trade-off made visible).
* Realized `RMS(ξ)/RMS(w)` per tensor class (conv, 1-D, LoRA A/B) so σ is comparable to the papers.

## 7. Implementation plan

* **Where**: a new `src/kaon/antikaon.py`, `class Antikaon(Adakaon)` (the Rakaon/Nekaon pattern: a
  preset/subclass, one code path). It forces `betas[0]=0`, `cautious=False` unless overridden, adds
  `k_sigma`, `shape ∈ {"v","none"}`, `noise ∈ {"rademacher","gaussian"}`, `antithetic`, `sigma_ref`,
  `s_cap`, `noise_seed`, plus `eval()/train()`, `_train_mode`, and the `state_dict` meta of §4. Not a
  wrapper: the noise has to ride *inside* the weight write to be one write.
* **Per-param path**: in `_step_one_param` after `delta` is formed: compute `S_n` from the pre-update
  `row/col` (before `update_factored_state`; it is `sqrt` of factors already needed), `S_{n+1}` after,
  draw `ε_n`, `ε_{n+1}` (`torch.Generator.manual_seed(mix(seed,pid,n))`, `randint`/`randn`),
  `delta −= (ξ_{n+1} − ξ_n)/lr`, and subtract `ξ_n` from the weight-decay term. Two full-size fp32
  temporaries per param, transient (same class as the SR torch path's temporaries).
* **Foreach path**: identical on the stacked `[N,R,C]` delta; noise generated per param slice with
  per-param seeds (so bucket composition never changes a param's noise). Bit-parity with the per-param
  path is preserved if both draw the same per-param sequence — pin it in
  `test_foreach_matches_per_param` like today.
* **Fused Triton**: yes, fusable and cheap. The apply kernels (`_chunked_apply_batched_g`,
  `_adakaon_tile_kernel`, `_adam_1d_kernel` [code _fused_triton.py:1159,…]) already load `p`, `r`, `c`,
  a `seed` and call `sr_round(res, seed, offs)`; add a `NOISE: tl.constexpr` branch that computes
  `s = mean_col_q * sqrt(r_i) * sqrt(c_j)` (one extra scalar per tensor from the reductions), draws two
  `tl.rand` (Rademacher: `sign(u − .5)`) with step-keyed seeds, and forms
  `res = p − lr·delta + σ_{n+1}·s_{n+1}·ε_{n+1} − σ_n·s_n·ε_n`. Needs the *pre-update* factors for `s_n`:
  the reduction kernel can emit them (R+C floats per tensor, transient) or the apply kernel recomputes
  them from the pre-update `row/col` if the EMA is moved into the apply kernel — implementation choice.
  Cost: ALU only (two Philox draws + ~6 FMAs per element); memory traffic unchanged, so in the
  bandwidth-bound big-tensor regime it is ≈free, and in the launch-bound LoRA regime the fused kernel
  count is unchanged. `eval()/train()` are one extra fused axpy pass each (reuse the `_axpy` kernel with
  the noise generator instead of a momentum pointer).
* **Extra state**: 0 B/param persistent; `O(#groups)` Python floats; `R+C` transient floats per tensor.
* **Risks** (ordered): (1) the sub-ulp regime raises the SR walk (§3) — warn, offer Kahan; (2) the
  perturbation is confined to the LoRA subspace on adapters (Bi-LoRA's caveat, shared with Nekaon/SAM);
  (3) the noise backend identity on resume (§4); (4) the `evald()`-must-use-eval-mode contract for every
  consumer that logs a loss (renga's val loop); (5) the RWP convergence trade-off at large `k_σ`
  (mitigations: antithetic, duty cycle, lower `k_σ`); (6) `wd` on `z` vs `w` — trivial but must be in the
  kernel; (7) `deterministic_reductions` interplay: the noise itself is deterministic given the seed, so
  it does not add run-to-run nondeterminism.

## 8. What is verified vs inferred (summary)

* Verified against the papers: the Anti-PGD/PGD equations and the `L + σ²/2 Tr(H)` regularizer (eqs. 1–6,
  Thm 2.1 with Rademacher noise); RWP eq. 10, filter-wise covariance, m-RWP `λ = 0.5` on two batches,
  ARWP eq. 14 with η_a = 0.1, β = 0.99, the σ values, and the "two orders of magnitude" radius statement.
* Verified in code: Adakaon's nomom step and writer, factored inverse factors, SR primitive and the
  Triton `sr_round`/apply kernels, MSAM's RTN-vs-SR round-trip lesson and freezing of per-cycle
  scalars, `evald()` calling `opt.eval()`.
* Verified by simulation: the MC regularizer identity with a preconditioner; the bf16 drift numbers of §3.2.
* Inference: the `Tr(PH)` reading of ARWP, the step-unit σ anchor and the sweep range, the antithetic
  variance argument, the LoRA behaviour, every performance statement (no timing was measured, per the
  brief), and of course whether any of this moves the frontier — that is what §6 is for.
