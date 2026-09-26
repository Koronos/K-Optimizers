# Compact Kahan — `bf16_method="kahan8"` (1 B/param) and `"kahan16"` (2 B/param), every path

Design, simulation and implementation notes for the compact fixed-point Kahan weight write.
Scripts and raw results live in `docs/research/compact-kahan/` (`sim_compact_kahan.py`,
`sim_compact_kahan.json`, `sim_table.md`, `measure_memory.py`, `memory_table.md`,
`bench_write.py`, `bench_table.md`). Tags: **[derivation]** analytic, **[sim]** measured by
the simulation, **[code]** measured on the implementation, **[inference]** reasoned, not measured.

## 0. Verdict

* **`kahan8`**: bf16 weight + one `uint8` residual per parameter, in units of the weight's own
  bf16 ulp, with **stochastic rounding of the residual**. It is a float with a 16-bit
  significand stored as `(bf16, byte)`. +1 B/param (vs +2 for the legacy `kahan`), supported on
  the per-param, foreach and **fused Triton** paths of Adakaon (all eight apply kernels), and
  on the native paths of every optimizer that shares the backend writer.
* Accuracy **[sim]**: 0.09–0.19 ulp drift of the tracked value after 10 000 steps in every
  regime (1, 0.4, 0.04 and 0.004 ulp/step, with and without an Antikaon-style perturbation),
  bias 0 within 2 SE, **no stall** below the grid (`lost` ≈ 0). Plain stochastic rounding
  walks by 17–45 ulp in the same runs; the legacy bf16 `kahan` buffer is as accurate on
  dithered updates but **stalls (31 % of the movement lost)** on pure sub-grid drift.
* `kahan4` (0.5 B/param) is 16× worse (1.5–3 ulp) and not implemented.
* **`kahan16`** (2 B/param, implemented — §8): the same codec at 16 bits, where the pair IS an
  fp32 master weight split in two (bit for bit), with the same coverage as `kahan8`. Given
  the same bf16 gradients a `kahan16` run is the fp32-weight run of the same optimizer.
* The legacy `kahan` stays untouched (its checkpoints hold `shift`; per-param only; the only
  Kahan that takes fp16). `kahan8` is the recommended Kahan; `kahan16` the exact one.

## 1. Representation

A bf16 weight `w` (8 significant bits) and a residual byte `lo`. Let `w16` be the bf16 bit
pattern and `B = 8` the residual width. The pair encodes the fp32 value

```
bits32(z) = (trunc16 << 16) | (lo << (16 - B)),   trunc16 = w16 - (lo >> (B - 1))
```

i.e. **an fp32 whose low `16 - B` mantissa bits are zero** — a 16-bit-significand float.
`trunc16` is `z` truncated toward zero to bf16; the stored `w16` is `trunc16` **plus a carry**
when the residual's top bit is set, i.e. `w = round-half-away(z)`: the forward pass sees the
nearest bf16 to the compensated value, never a truncation biased toward zero (a truncated
store would shrink every weight by ~2^-9 on average).

Properties **[derivation, verified by `test_codec_*` on CPU and CUDA]**:

* **Implicit scale.** The residual of a nearest rounding is `< ulp(w)/2`, so `B` bits at
  `ulp/2^B` cover it; bf16's exponent field already says which ulp. No per-block scale, no
  extra memory, no cross-lane reduction in the kernels (a per-block absmax scale would have
  cost memory and a reduction, for a residual whose range is *known* a priori).
* **Binade crossings are free.** The `+1` carry on the 16-bit pattern *is* the crossing
  (`0x3FFF + 1 = 0x4000`, 1.9921875 → 2.0). The pattern is monotone in magnitude through
  subnormals and zero, so `0`, `±0.0`, `2^-133` (smallest bf16 subnormal) and `2^-126` all
  round-trip; a residual below half an ulp of zero stays a residual of `w = 0`.
* **Integer-only decode.** No float arithmetic ever touches the residual, so a subnormal
  residual cannot be flushed to zero by a kernel's FTZ mode.
* **Non-finite propagate.** NaN/inf are stored as the plain cast would store them, with
  `lo = 0` — the `sr_round` policy: a diverged run surfaces.
* **Encode(decode(·)) is the identity**, `|z_stored − z| ≤ unit/2` with `unit = ulp(w)/256`.

## 2. Rounding of the residual and error bounds **[derivation]**

Each write computes `z' = z + αδ` exactly-ish in fp32 (24 significant bits) and must drop
`16 − B = 8` low bits. Two policies were derived and simulated:

* **Round half away (RN).** Deterministic. Per-step error `|e| ≤ unit/2`. When the update
  dithers the residual (its fractional part in grid units is ~uniform, true whenever the
  update noise ≫ unit) the errors are independent, `Var = unit²/12`, and after N steps the
  drift is `unit·√(N/12)` = **0.11 ulp at 10k** for B=8. When it does not — a coherent
  update with `|αδ| < unit/2` — RN **stalls**, exactly as plain RTN stalls below ulp/2, only
  256× further down. With `|w| ~ 1` and lr 1e-6 that regime is real (`unit = 2^-15`, step
  ~1e-6·u ≈ 1 unit for Adakaon's normalised `u`).
* **Stochastic rounding of the residual (SR, the `kahan8` policy).** Uniform noise in
  `[0, 256)` added to the fp32 bits before the mask (the SR bit trick, at the residual's
  grain). `E[z_stored | z'] = z'` **exactly**, so the error is a martingale: bias 0 for every
  step size, `Var ≤ unit²/4` per step, `unit²/6` when the fractional part is uniform.
  Drift after N steps: `unit·√(N/6)` = **0.16 ulp at 10k, 0.5 at 100k, 1.6 at 1M** for B=8;
  16× that for B=4 (2.6 ulp at 10k). No stall at any step size.

The sim (§3) matches both: `kahan8-rn` 0.076–0.16 ulp with dithered updates and a 36 % stall
on pure drift; `kahan8-sr` 0.09–0.19 ulp everywhere and `lost ≈ 0`. SR of the residual is the
hybrid the task brief asked to evaluate ("pocos bits + redondeo estocástico del residuo") and
it is what makes a *short* residual safe; it is the policy shipped.

For comparison, plain **bf16 SR** (no state): per-step `Var = f(1−f)·ulp²` with `f` the
fractional part of the step in ulps — `≈ |step|·ulp` for sub-ulp steps, saturating at
`ulp²/6` — so the walk is `√(N·|step|/ulp)` ulp: 20 ulp at 10k steps of 0.04 ulp, 41 ulp
saturated. The legacy **bf16 `kahan` buffer** rounds the residual to bf16 (relative 2^-9 of
the residual itself, ≤ 2^-10 ulp per step) *and* rounds the step to bf16 before accumulating
(`shift -= delta.to(bf16)`): comparable per-step error to `kahan8`, but deterministic — hence
its stall on pure sub-grid drift (§3).

**Antikaon's combined write** `w − ηδ + ξ_{n+1} − ξ_n` **[derivation + sim]**: the write is a
single `compensated_add_` of the combined fp32 increment. Because `|Δξ|` is typically ≫ unit
the fractional part is uniform and the walk sits at its maximum `unit·√(N/6)`; that is the
same 0.16 ulp — the perturbation carries through the compensated value exactly (sim: 0.093 vs
0.093 ulp at lr 1e-3 with/without ξ; 0.187 vs 0.187 at lr 1e-5). The value the eval sees
(`w − ξ`) additionally carries the ≤ ½-ulp residual, which is what the Antikaon sim's
"kahan+ξ 0.49 ulp" figure was mostly measuring (it read `w.float()`, not `w + shift`).

## 3. Simulation **[sim]**

`sim_compact_kahan.py`: 2^20 bf16 coordinates, `w0 ~ 0.05·N(0,1)`, 10 000 steps, update
stream `δ = lr·(0.1c + 0.99ν)` (`c = ±1` redrawn every 50 steps, `ν ~ N(0,1)`; RMS ≈ 1 like
Adakaon's normalised update), optional Rademacher `ξ = 4·lr·s` carried in the weights, fp32
reference on the same stream, no feedback (worst case). Errors in `ulp_ref = ulp(RMS z)`;
`err` is on the value the scheme *tracks* (bf16 + compensation, decoded), `fwd std` on the bare
bf16 the forward sees, `lost` the fraction of the net movement not realised (stall). Full
table: `docs/research/compact-kahan/sim_table.md` (56 runs). Condensed:

| regime (step/ulp) | scheme | B/param | bias ± se | std (ulp) | max | fwd std | lost |
|---|---|---|---|---|---|---|---|
| 1.02 | sr | 0 | −0.004 ± 0.023 | 23.9 | 343 | 23.9 | 0.000 |
| 1.02 | kahan (bf16 shift) | 2 | 0.000 ± 0.000 | 0.234 | 1.20 | 0.324 | 0.000 |
| 1.02 | **kahan8-sr** | 1 | 0.000 ± 0.000 | **0.093** | 1.11 | 0.242 | 0.000 |
| 1.02 | kahan8-rn | 1 | 0.000 ± 0.000 | 0.076 | 0.80 | 0.237 | 0.000 |
| 1.02 | kahan4-sr | 0.5 | 0.001 ± 0.002 | 1.50 | 18.6 | 1.51 | 0.000 |
| 1.02 + ξ(k=4) | sr | 0 | −0.033 ± 0.024 | 24.0 | 359 | 24.0 | 0.000 |
| 1.02 + ξ(k=4) | kahan | 2 | −0.002 ± 0.001 | 1.41 | 6.5 | 1.43 | 0.000 |
| 1.02 + ξ(k=4) | **kahan8-sr** | 1 | 0.000 ± 0.000 | **0.093** | 1.09 | 0.242 | 0.000 |
| 0.41 | sr | 0 | −0.057 ± 0.041 | 41.4 | 397 | 41.4 | 0.000 |
| 0.41 | kahan | 2 | 0.000 ± 0.000 | 0.114 | 0.81 | 0.367 | 0.000 |
| 0.41 | **kahan8-sr** | 1 | 0.000 ± 0.000 | **0.191** | 2.54 | 0.398 | 0.000 |
| 0.41 | kahan4-sr | 0.5 | 0.001 ± 0.003 | 3.04 | 38.1 | 3.06 | 0.000 |
| 0.41 + ξ(k=4) | sr | 0 | −0.026 ± 0.045 | 45.5 | 621 | 45.5 | −0.002 |
| 0.41 + ξ(k=4) | kahan | 2 | 0.000 ± 0.000 | 0.460 | 2.59 | 0.578 | 0.000 |
| 0.41 + ξ(k=4) | **kahan8-sr** | 1 | 0.000 ± 0.000 | **0.190** | 2.42 | 0.398 | 0.000 |
| 0.041 | sr | 0 | −0.024 ± 0.016 | 16.7 | 145 | 16.7 | 0.003 |
| 0.041 | kahan | 2 | 0.000 ± 0.000 | 0.054 | 0.73 | 0.343 | 0.000 |
| 0.041 | **kahan8-sr** | 1 | 0.000 ± 0.000 | **0.187** | 3.00 | 0.387 | 0.000 |
| 0.041 | kahan8-rn | 1 | 0.000 ± 0.000 | 0.160 | 2.01 | 0.375 | 0.000 |
| 0.041 | kahan4-sr | 0.5 | 0.000 ± 0.003 | 2.84 | 33.7 | 2.86 | 0.000 |
| 0.041 | kahan4-rn | 0.5 | −0.001 ± 0.002 | 1.97 | 19.0 | 2.00 | **0.060** |
| 0.041 + ξ(k=4) | sr | 0 | −0.026 ± 0.033 | 33.6 | 303 | 33.6 | 0.007 |
| 0.041 + ξ(k=4) | kahan | 2 | 0.000 ± 0.000 | 0.081 | 0.84 | 0.348 | 0.000 |
| 0.041 + ξ(k=4) | **kahan8-sr** | 1 | 0.000 ± 0.000 | **0.187** | 2.78 | 0.387 | 0.000 |
| 0.004 pure drift | sr | 0 | −0.002 ± 0.002 | 1.93 | 20.2 | 1.93 | 0.004 |
| 0.004 pure drift | kahan | 2 | 0.000 ± 0.000 | 0.244 | 1.35 | 0.271 | **0.311** |
| 0.004 pure drift | **kahan8-sr** | 1 | 0.000 ± 0.000 | **0.114** | 1.09 | 0.220 | **0.000** |
| 0.004 pure drift | kahan8-rn | 1 | 0.000 ± 0.000 | 0.271 | 1.39 | 0.271 | **0.362** |
| 0.004 pure drift | kahan4-sr | 0.5 | 0.000 ± 0.001 | 0.481 | 4.21 | 0.549 | −0.004 |
| 0.004 pure drift | kahan4-rn | 0.5 | 0.000 ± 0.000 | 0.288 | 1.39 | 0.288 | **0.417** |
| 0.004 pure drift + ξ | kahan | 2 | 0.000 ± 0.000 | 0.144 | 1.16 | 0.252 | **0.166** |
| 0.004 pure drift + ξ | **kahan8-sr** | 1 | 0.000 ± 0.000 | 0.153 | 1.53 | 0.231 | 0.000 |
| any | kahan16 (fp32 split) | 2 | 0 | 0.000 | 0.00 | 0.21–0.35 | 0.000 |

Findings:

1. Every compensated scheme is unbiased (all within 2 SE of 0). `kahan8-sr`'s drift is
   0.09–0.19 ulp in every regime, matching `unit·√(N/6)` = 0.16, and is *independent* of
   the step size and of ξ — the martingale bound. **[sim, derivation]**
2. The bf16 `kahan` buffer is as accurate as `kahan8` on dithered updates (0.05–0.46 ulp) but
   **loses 31 % of the movement on pure 0.004-ulp drift** (17 % with ξ): its residual is
   rounded to bf16 each step, and a step below that rounding's grain is dropped
   deterministically. `kahan8-rn` stalls the same way (36 %); `kahan8-sr` does not (0.0 %).
   The hybrid is the design, not a refinement. **[sim]**
3. `kahan4` is 16× worse than `kahan8` (1.5–3 ulp; 0.5 ulp on pure drift) and its RN variant
   stalls 6–42 %. The 0.5 B/param it saves is not worth 16× the drift; not implemented.
4. `fwd std` — what the forward pass sees — is `√(std² + residual²)`: 0.22–0.40 ulp for every
   compensated scheme, dominated by the ≤ ½-ulp residual by construction. That is the
   Antikaon sim's "0.49 ulp": residual, not drift.
5. SR-only bf16 in the sub-ulp regime is 3–7× the total movement (`err/moved`), with or without
   ξ: the Antikaon design's motivation for Kahan stands.

## 4. Implementation **[code]**

* `src/kaon/_compact_kahan.py` — the torch reference codec: `decode` (integer-only, three
  int32 temporaries), `encode_` (in place; consumes `z` and the noise as scratch),
  `compensated_add_` (decode → exact fp32 add → SR encode), `init_residual`, the method table
  `COMPACT_KAHAN_BITS = {"kahan8": 8}` and the state key `RESIDUAL_KEY = "kahan_lo"`.
* `src/kaon/_backend.py` — `BF16_METHODS`, `validate_bf16_method`, `init_bf16_state` (one call
  per optimizer `_init_state`: allocates `shift` for `kahan`, `kahan_lo` for `kahan8`),
  `per_param_only_bf16_method` (`kahan` only), `_ck_write_` (Triton one-launch axpy on CUDA,
  torch reference elsewhere — the `_sr_write_` twin, same `SR_TRITON` switch), and the two
  writers: `subtract_one_` (per-param) and `subtract_batched_(…, comp=)` (foreach; the bucket's
  `state["kahan_lo"]` views come from `ForeachChunk.cviews`; a bf16 `kahan8` bucket without
  them is refused, never silently written uncompensated).
* `src/kaon/_fused_triton.py` — device helpers `ck_decode` / `ck_store` (the identical bit
  manipulation; `ck_store` reuses the `tl.rand(seed, offs)` draw `sr_round` uses, scaled to the
  8 dropped bits), the one-launch `_ck_axpy_kernel` / `ck_add_` / `ck_add_supported`, a `c_addr`
  residual pointer array in `PointerArrayCache`, `OneDimPointerCache` and `BigPointerCache`
  (aliases `p_addr` when absent — never dereferenced, like `mscale_addr` for float momenta),
  and a `CK: tl.constexpr` (0 or the residual width) on all eight Adakaon apply kernels:
  `_adakaon_tile_kernel`, `_adam_1d_kernel`, `_chunked_apply`, `_chunked_apply_batched`,
  `_chunked_apply_batched_g`, `_chunked_nomom_apply_batched_g`, `_chunked_4bit_apply_batched_g`,
  `_chunked_int8_apply_batched_g`. Under `CK` the kernel loads the bf16 bits + byte, decodes the
  exact `z`, computes `z − lr·delta` and stores `(bf16, byte)`: **+1 B/elem read, +1 B/elem
  written, no extra launch, no temporaries**. The delta itself (weight decay term, cautious
  mask) still uses the bf16 `w`, exactly as the native path does, so the two stay comparable.
* `src/kaon/adakaon.py` — `kahan8` is fused-eligible (`bf_ok`), every launch passes `CK` /
  `SR` consistently (`_ck_bits`), fp16 params are refused at `add_param_group` like SR.
* `src/kaon/_foreach_plan.py` — `kahan_lo` is a watched state key (pointer tables bake it);
  `ForeachChunk.cviews`.
* Other optimizers (AdaBelief, AdamP, AdaMuon, ADOPT, KProdigy, Lion, AdaPNM): shared
  validation, `init_bf16_state`, the foreach predicate, `comp=` at their writer call sites —
  `kahan8` rides their foreach and per-param paths. **AdaPNM's fused route declines it** and
  reports `bf16_method=kahan8` through its existing fused-decline reason (native/foreach
  fallback; extending its five kernels is mechanical but was not done). **ScheduleFree** keeps
  its own `y`-write and rejects `kahan8` with its ValueError. **Lookahead** routes the
  `theta ← phi` sync per-param for `kahan` and through the batched writer with the inner's
  residual for `kahan8` (the sync writes the inner's `p`, so the inner's `kahan_lo` is the
  compensation, as the 0.7.13 fix established for `shift`; the residual is looked up only
  for bf16 chunks — an fp32 param under `kahan8` has none). **MSAM / Nekaon** climbs are
  **kahan8-aware** (§4b): the perturbation is applied to and removed from the DECODED value
  and re-encoded with the residual's stochastic rounding, in the torch path (one stacked
  `_ck_write_` per bucket, noise from the wrapper's own checkpointed stream) and in the
  fused `_axpy_momentum_batched` kernel (`CK` constexpr, `c_addr` in the plan, the plan
  witness sees a rebound `kahan_lo`). The first cut perturbed the bare bf16 weight and left
  `kahan_lo` alone, on the assumption that this was a bounded ≤ 1-ulp inconsistency — it is
  not: see §4b. **SAM** perturbs `w` through `add_stochastic_` and restores it by an exact
  `copy_` of its snapshot, so `(w, lo)` is untouched by the pair. **[code]**
* **Externally written weights.** A weight written from outside the codec (pruning, a
  re-initialisation to zero, loading a model while keeping the optimizer state) keeps its
  old residual byte, which re-attaches to the new pattern's ulp: finite and within one ulp
  everywhere — including `±0`, where the decoder now skips the carry (256 of the 2²⁴ states
  used to wrap to a NaN pattern; `test_decode_is_finite_for_every_state_with_a_finite_weight`
  enumerates all 2²⁴ on the torch and the Triton decoder). **[code]**
* **A group switched to `kahan8` mid-run** (a scheduler/user writing `group["bf16_method"]`)
  gets zero residuals allocated on the spot, with one warning, on every route
  (`kaon._backend.ensure_residuals`; per-param, `ForeachPlanMixin._foreach_chunks`, and the
  fused host, which then rebuilds its pointer cache). A fused launch with `CK` set and no
  residual array is refused (`Adakaon._c_addr_arg`); the first cut substituted `p_addr`, so
  the kernel wrote residue bytes over the weights (values up to ~1e36, silently). **[code]**
* **Checkpoint / resume.** `kahan_lo` is an ordinary per-param state tensor: saved by
  `state_dict`, restored `uint8`-exact by `load_state_dict_preserving_dtypes`, and the
  residual's SR noise draws from the optimizer's checkpointed `SRStream` (generator on the
  torch path, launch seed on the Triton axpy, `self._t` in the fused kernels), so a resume is
  **bit-identical** to the uninterrupted run (`test_resume_is_bit_exact_and_keeps_uint8`, CPU
  and fused CUDA).
* **Parity.** Per-param == foreach **bit-exact** on the torch path with the residual noise
  pinned (same deterministic codec; with live noise the two consume the generator in bucket
  order and differ by ≤ 1 grid unit per step). Fused == native within a few grid units (0.05
  ulp over 6 steps on the one-block/1-D/0-D/conv routes; 0.34 ulp on big buckets where a
  cautious-mask flip between reduction orders is a whole step on one coordinate — the same
  class of difference the SR tests tolerate; ≤ 0.1 ulp with `cautious=False` except 4-bit
  momentum's known code-flip amplification).

### 4b. The MSAM / Nekaon climb **[code, `climb_e2e.py`]**

The review's end-to-end check (one 64×64 bf16 weight, lr 1e-5 ≈ 0.04 ulp/step, 300 steps,
the same bf16 gradients to an fp32 reference of the same optimizer; error = max |z − z_fp32|
in ulp at the RMS weight, on the clean weights after `eval()`) showed the first cut's climb
**destroying the advantage**: the perturbation `e` was added to and removed from the bare
bf16 `w` with RTN while `kahan_lo` stayed put. Wherever `|e| ≥ ulp_local/2` (with
`N(0, 0.05)` weights and `e` = 1.5e-5 that is every coordinate below |w| ≈ 4e-3, ~8 % of
them) the climb moves `w` by `round_ulp(e)`, the base step then
re-encodes the residual against the *perturbed* pattern, and the removal takes only
`round_ulp(e)` off again — the sub-ulp part of `e` is folded into the clean value **every
step, with a constant sign**. That is a coherent drift, not a bounded inconsistency: the
"≤ 1 ulp per climb" claim of the first write-up was wrong.

Fix: perturb and restore the DECODED value and re-encode through the codec (stochastic
rounding of the residual, so the pair is unbiased at the 1/256-ulp grain; MSAM's reason for
RTN — SR pairs not cancelling at the bf16 grain, 19 % L2 drift — does not apply 65536× down
in variance). Torch path: one stacked `_ck_write_` per bucket; fused path: `CK` on
`_axpy_momentum_batched`. Measured (`climb_e2e_before.md` / `climb_e2e_after.md`):

| optimizer | route | kahan8 before | **kahan8 after** | SR | fp32 reference moved |
|---|---|---|---|---|---|
| Adakaon (no climb) | foreach | 0.298 | 0.298 | 21.8 | 37.9 ulp |
| Adakaon (no climb) | fused | 0.281 | 0.281 | 18.7 | |
| MSAM ρ=0.05 | foreach | 12.07 | **0.322** | 23.5 | |
| MSAM ρ=0.05 | fused | 12.07 | **0.416** | 18.7 | |
| Nekaon k=1.5 | foreach | 25.17 (worse than SR) | **0.400** | 24.6 | |
| Nekaon k=1.5 | fused | 25.17 | **0.463** | 18.7 | |

After the fix the climbing optimizers sit within 0.1–0.2 ulp of the climb-free Adakaon run
(the two extra residual roundings per step) and 40–60× below stochastic rounding.
`test_msam_nekaon_climb_keeps_the_kahan8_advantage` pins it (100 steps, < 0.6 ulp and
> 5× better than SR, both routes).

### Memory **[code, `measure_memory.py`, 8.47 M bf16 params, bf16 momentum, RTX 3000 Ada 8 GB]**

| bf16_method | fused | state B/param | compensation B/param | first-step peak B/param |
|---|---|---|---|---|
| stochastic_rounding | False | 2.039 | 0.000 | 20.4 |
| stochastic_rounding | True | 2.039 | 0.000 | 2.05 |
| kahan | False | 4.039 | 2.000 | 6.2 |
| kahan8 | False | 3.039 | **1.000** | 21.3 |
| kahan8 | True | 3.039 | **1.000** | 3.05 |
| none | – | 2.039 | 0.000 | 21.1 |

The compensation is exactly 1.000 B/param; the fused step adds no transient, and the native
(foreach) step's first-step peak is the same as SR's (21.3 vs 20.4 B/param — the stacked
chunk and its fp32 delta). A first cut of the native path measured ~48 B/param there, from
the torch codec's int32 scratch over the stacked chunk — that is why the native CUDA writers
were routed to the one-launch Triton axpy (`_ck_write_`), whose only transient is the stacked
bucket itself (2 + 1 B/elem, asserted ≤ 3.5 B/elem in
`test_native_cuda_writer_takes_the_kernel_without_int32_scratch`). The torch reference now
uses in-place integer ops (~12 B/elem transient; CPU / non-contiguous views only).

## 5. Speed (ORIENTATIVE) **[code, `bench_write.py`]**

See `docs/research/compact-kahan/bench_table.md` and the "Speed" note in the report for the
run's electrical state. Numbers were taken on a laptop while another agent was using the GPU,
so they are indicative only; the definitive serial comparison is pending (owner: the user).

Electrical state: laptop on AC (`Win32_Battery.BatteryStatus = 2`), RTX 3000 Ada Laptop GPU,
`power.limit N/A`, `clocks.max.sm 3105 MHz`; during the run `nvidia-smi` reported 97 % GPU
utilisation from the other agent's test run, SM clock 1500 MHz, 21 W draw, 62 °C. CUDA-event
medians (q1, q3) of 50 reps for the writers and 30 for the steps:

| what | SR (Triton) | kahan8 (Triton) | kahan (bf16 shift, torch) | notes |
|---|---|---|---|---|
| standalone weight write, 2^22 elems | 0.070 ms | **0.074 ms** | 0.386 ms | SR torch path 1.17 ms; kahan8 torch reference 5.8 ms (CPU/reference only) |
| fused step, LoRA bag 200×(256,256)+100×(512,) | 2.90 ms | 3.63 ms (q1 2.98) | – | wide quartiles under contention |
| fused step, big 2×(1024,1200) | 0.295 ms | **0.318 ms** | – | +8 %: +1 B/elem read + written |
| fused step, UNet-ish 8×(1024,1024)+16×(4096,) | 2.61 ms | 2.03 ms | – | noise (q1 2.20 vs 1.97) |
| native (foreach) step, LoRA bag | 16.7 ms | 17.7 ms | – | |
| native step, big 2×(1024,1200) | 2.89 ms | 4.09 ms | – | the big weights loop per-param through `_ck_write_` |
| native step, UNet-ish | 13.4 ms | 14.3 ms | – | |

Reading: the fused kernels pay the extra byte in and out and nothing else (+5–8 % on the
bandwidth-bound big write, within noise on launch-bound bags); the one-launch Triton axpy
makes the native CUDA write as fast as the SR one (0.074 vs 0.070 ms) and 5× faster than the
legacy bf16 `kahan` write. The torch reference codec is slow (5.8 ms / 2^22) and is only
reached on CPU or on a non-contiguous view. All of this is to be re-measured serially.

## 6. Decisions

* **`kahan8`, SR residual, implicit exponent scale, round-half-away stored weight.** Rationale
  in §§1–3. Name: `kahan8` (the residual width in bits; `kahan16` is the same table's 16).
* **Legacy `kahan` kept as is**, not aliased: its checkpoints carry `shift` (bf16), its
  numerics differ (it stalls on sub-grain drift, `kahan8` does not), and it is the only Kahan
  that accepts fp16 parameters. It remains per-param only. Docs mark `kahan8` as the
  recommended Kahan.
* `kahan4` evaluated and rejected (16× the drift for 0.5 B/param; nibble packing would also
  have forced the even-C constraint on the tile kernel).
* The per-block-scale variant was not simulated: the residual's range is known a priori
  (`< ulp/2`), so a block scale can only lose resolution where the block's ulps differ, and
  it costs memory plus a cross-lane reduction in every kernel. **[inference]**

## 7. Pending / risks

* AdaPNM's fused kernels (5) do not take `CK`; the group falls back to native/foreach with the
  existing decline reason. Mechanical to extend.
* ScheduleFree rejects `kahan8` (its `y`/`z` write is its own; `test_schedulefree_rejects_kahan8`).
* The residual's stochastic rounding consumes one Philox draw per element per step in the fused
  kernels — the same draw the SR write already consumed, so no added cost; on the torch
  reference it is one `randint` (int32) per element like SR's. The MSAM/Nekaon climb adds two
  such draws per step (climb + removal) on `kahan8`, where it used to draw none (RTN).
* The `kahan8` climb round trip is unbiased but not exact (≤ 1 grid unit = ulp/256 per
  climb/removal, random sign); an exact scheme would need the clean pattern stored
  somewhere for the removal, i.e. memory. Not worth it at 0.4 vs 0.3 ulp (§4b).
* Kernels touched by the review fixes, hence to be re-timed: `sr_round` (a `tl.minimum` on
  the noise, every SR launch), `ck_decode` (one `tl.where` for the ±0 guard, every `CK`
  launch), `ck_store` (split into `ck_store_noise` + a wrapper, same arithmetic plus the
  clamp) and `_axpy_momentum_batched` (new `c_addr` argument and `CK` path — the
  MSAM/Nekaon climb). The Adakaon apply kernels' host signatures are unchanged.

## 8. `kahan16` — the fp32 master split in two **[code, `tests/test_kahan16.py`]**

### 8.1 Representation and the stored-bf16 decision **[derivation]**

The §1 codec with `B = 16`: `bits32(z) = (trunc16 << 16) | lo`, `trunc16 = w16 − (lo >> 15)`.
There are no dropped bits, so **every fp32 is representable and the pair is bit for bit the
fp32**: `lo` is the fp32's low half, `w` its high half — carried by one when `lo`'s top bit
is set. A write is `z' = z − lr·δ` in plain fp32 arithmetic (round to nearest) followed by an
exact split; nothing else rounds, so **no residual SR is needed** (the only rounding is fp32's
own, the same one an fp32 master weight takes).

Decision: **the stored bf16 is the NEAREST one (ties away from zero), not the truncated high
half.** Two alternatives were considered:

* *truncated* (`w = high16(z)`, `lo = low16(z)`): the simplest split, but the forward then
  sees `z` rounded toward zero — every weight shrunk by ~½ ulp on average (bias ~2⁻⁹
  relative), which is exactly what `kahan8` was designed to avoid;
* *nearest with a signed residual* (`lo = z − RN(z)` as a signed int16): also exact, but a
  second convention next to `kahan8`'s, and a signed residual has no room for the half-ulp
  tie in both directions without special-casing.

The `kahan8` convention (the top residual bit IS the carry) gives the nearest bf16 at the cost
of the same one add in encode and one subtract in decode that `kahan8` already pays, and the
residual stays the fp32's literal low half. Ties (`lo == 0x8000`) round away from zero where
`.to(torch.bfloat16)` rounds to even: the stored bf16 differs from the RNE cast of the master
in exactly those 1-in-65536 patterns whose truncation is even (asserted exhaustively). RNE
itself is not decodable: after a tie `w` is always even, and `trunc16` could be `w` or `w − 1`.

### 8.2 Storage, state key, conversion

* **Dtype `int16`**, holding the uint16 pattern. torch 2.12's `uint16` has no
  `_foreach_copy_` on CUDA (the foreach writers' write-back); `int16` has every op and a
  1:1 Triton pointer type. Every reader masks `& 0xFFFF` after the sign-extending load; the
  torch encoder sign-extends explicitly before narrowing (the int32→int16 cast does wrap on
  CPU and CUDA, but that is implementation-defined in C++).
* **Same state key `kahan_lo`**: the residual's dtype identifies the width that wrote it
  (`residual_bits_of`). `WATCHED_STATE_KEYS` already watches it, pointer arrays store
  `data_ptr`s (width-agnostic), `ForeachChunk.cviews` and the MSAM plan key on it. A second
  key would have doubled every one of those sites for no information the dtype does not carry.
* **Mid-run switch** (`group["bf16_method"]` written after the state exists), handled by
  `kaon._backend.ensure_residuals(params, states, bits)` on every route (per-param writer,
  foreach plan hook, AdaPNM's bucket views, Adakaon's fused host, Lookahead's sync, Antikaon's
  clean write): no residual → a zero one (one warning, as for `kahan8`); a residual of the
  other width → **converted** (`convert_residual`: decode with the stored width, re-encode at
  the new one; one warning). Widening `kahan8 → kahan16` is exact (the 8-bit grid is a subset;
  the bf16 does not move); narrowing is one round-half-away at ulp/512. The new tensor is a
  watched rebinding, so plans and pointer tables rebuild (the foreach hook rebuilds the
  current step's views on the spot; `ForeachChunk.cviews` is `None` for a chunk of mixed
  widths so the hook never trusts a half-converted chunk). Converting instead of raising was
  chosen because the switch is value-preserving and cheap, and raising would make a scheduler
  that changes precision mid-run impossible. Readers that only DECODE (the MSAM/Nekaon climb
  removal, Antikaon's clean read) use the stored width, not the group's: the removal at the
  top of the step right after a switch must decode the residual the climb encoded, before
  the inner step converts it. `_ck_write_` refuses a residual of the wrong width outright,
  and `subtract_batched_` checks EVERY residual view of the bucket (`torch.stack` would
  promote a uint8 + int16 mix to int16 and pass the stack's check). The MSAM/Nekaon CLIMB
  normalizes its bucket (`_ck_ready`: allocate / convert to the group's width) — the only
  place a bucket can stay mixed is a param the inner never steps again (no grad) after a
  switch, which would otherwise have kept the whole bucket on the bare-bf16 climb forever.

### 8.3 Paths

Everything `kahan8` has, with `bits` threaded instead of the literal 8: `subtract_one_` /
`subtract_batched_(comp=)`; the Triton `ck_decode` / `ck_store_noise` / `ck_store` (a
`BITS == 16` constexpr branch: mask after the int16 load, int16 store, **no `tl.rand` draw**),
a `ck_ptr` helper that types a pointer-array entry as `uint8` or `int16`, `ck_add_` /
`ck_add_supported(…, bits)`; the eight Adakaon apply kernels and `_axpy_momentum_batched`
take `CK=16` (the `kahan8` specializations compile to the same code as before); the foreach
budget counts 0/1/2 B of residual stack. **Traffic: +2 B/elem read and +2 B/elem written
under `CK=16`** (vs +1/+1 for `kahan8`), no extra launch, no temporaries.

**The stacked torch add runs row by row** (`compensated_add_(rows=True)` from
`subtract_batched_`, `kahan16` only). The CPU `add_(alpha=)` kernel is not layout-invariant:
its vector body contracts `z + alpha·d` into an FMA, its scalar tail rounds the product
first. One add over a stacked `[N, *shape]` bucket puts the tails where the STACK ends, the
fp32 foreach writer (`_foreach_sub_` over the per-param views) and the per-param writer where
EACH TENSOR ends — the first cut was 1 fp32 ulp off on those coordinates (numel-15 and
numel-7 tensors: 5 coordinates after 30 steps), which the review caught and `_bag`'s shapes
had hidden. Row by row reproduces the per-tensor tails exactly. `kahan8` ignores the flag —
its ulp/256 grid absorbs a 1-fp32-ulp difference and its reviewed numerics are unchanged
(checked bit for bit against the previous commit on MSAM + Adakaon, both routes). The Triton
axpy is elementwise and layout-invariant already.

### 8.4 Verified **[code]**

* Codec: all 2³² fp32 patterns on CUDA (torch encode/decode 21 s, the Triton helpers against
  the torch codec 9 s), 2²⁰ on CPU: decode(encode(x)) == x bit for bit on the finite domain
  (`±0`, subnormals, binade crossings, FLT_MAX carrying the stored bf16 into inf while the
  master stays exact); `lo` == low half; `w` == `(bits + 0x8000) >> 16`; differs from the RNE
  cast only on ties; non-finite propagate with `lo = 0`; every `(w, lo)` state with a finite
  `w` decodes finite.
* Trajectory, same bf16 gradients to a bf16-`kahan16` run and an fp32-weight run OF THE
  SAME ROUTE, Gradient Centralization off: **bit-exact** for Adakaon (bf16/int8/4-bit
  momentum, nomom, cautious) per-param and foreach on CPU — including buckets whose tensors
  end in SIMD tails (`test_misaligned_tail_bucket_is_still_the_fp32_writer`) — and
  per-param, foreach and fused (every route incl. the big bucket with
  `deterministic_reductions=True`) on CUDA; Lion, AdaBelief, ADOPT, KProdigy, AdaMuon, AdaPNM
  per-param and foreach. **foreach vs per-param** is NOT bit-exact in general, for fp32
  weights either: the stacked update math (second moment, normalisation) has its own SIMD
  tails, so an fp32 Adakaon differs between its two routes by an fp32 ulp on a few
  coordinates of such a bucket; `kahan16` reproduces that difference exactly (same
  coordinates, same bits) and adds none of its own. The stored bf16 is the half-away nearest of the
  master, asserted per coordinate. Fused vs native `kahan16` differ only as the fp32 paths
  do (< 0.01 ulp; 4-bit's code-flip amplification < 0.5).
* MSAM / Nekaon climb (torch and fused `CK=16`): the run IS the fp32 run, bit for bit, in
  the perturbed train state and after `eval()`.
* Resume bit-identical with `int16` kept (per-param, foreach, fused); +2.000 B/param.

### 8.5 Where `kahan16` is NOT the fp32 run (by design, documented)

**0.7.16 closed the three Adakaon/Nekaon gaps this section used to list** (the reason
`kahan16` Nekaon drifted 2.3 / 7.4 ulp from fp32 at lr 1e-4 / 3e-4 in `benchmarks/lowlr_bf16`,
where Nekaon runs wd=0.1 and GC):

* **Weight decay** now reads the decoded value `z` (`kaon._backend.weight_value` per-param,
  `ForeachChunk.value_stack` foreach; the fused kernels use `zc` in the `WD` term of all eight
  apply kernels AND in the six cautious KEEP passes — `wd_value` — so the survivor count is
  taken on the same `delta` the apply writes with). One compiler detail had to be pinned: in
  the direct 4-bit / int8 apply kernels under `cautious_wd="full"`, `delta·scale + wd·p` has
  two legal FMA contractions and the CK variant compiled the other one; an explicit
  `tl.fma(wd, p, delta)` pins the one the fp32/SR variants already had (SR/fp32 unchanged,
  checked bit for bit against 0.7.15 on 360 configs).
* **Gradient Centralization** of a low-precision gradient in a compact-Kahan group runs in
  fp32 on the copy the update reads (the per-param `p.grad.float()`, the foreach
  `grad_stack()`), not in place on the bf16 `p.grad` — which is left uncentralized. The fused
  kernels always did GC in fp32 registers. SR / none / legacy kahan keep the in-place bf16 GC.
* **Lookahead's sync** reads the decoded `theta` (and snapshots `phi` from it): a `kahan16`
  Lookahead is now its fp32 twin bit for bit.

With the three, Adakaon and Nekaon under `kahan16` reproduce the fp32 run of the same route
bit for bit **with the shipped defaults** (wd, GC, cautious, both `cautious_wd`) on every
route incl. every fused sub-route (`tests/test_nekaon_kahan.py`). `kahan8`'s numerics change
intentionally (its decay/GC/theta now read its ~ulp/256 value instead of the bf16).

What is still NOT the fp32 run:

* **The other optimizers' weight decay** (Lion, AdaBelief, ADOPT, KProdigy, AdaMuon, AdaPNM,
  AdamP) still reads the stored bf16 `p` — `wd·lr·(z − w)` per step, ≤ `wd·lr·½ulp`. The
  helper is there (`weight_value` / `value_stack`); wiring each optimizer (and AdaPNM's fused
  kernels) is a follow-up. **AdamP's projection** reads the bf16 weight too. Their native GC
  also still runs in the grad's dtype.
* The fp32 reference itself is not invariant to bucket composition on CUDA (a foreach bucket
  of N same-shape tensors reduces differently from N-1), so a mixed fp32/bf16 group is
  bit-comparable only against a reference with the same bucketing.

* (Fixed in the review round) MSAM's inert-climb warning was not method-aware: it compared
  the displacement with half a bf16 ulp even when the climb goes through the residual. It now
  compares with half the residual grid (`ulp/2^bits`: ulp/256 under `kahan8`, ulp/65536
  under `kahan16`) for bf16 weights of a compact-Kahan group.

### 8.6 When to use which **[inference]**

* **SR** (0 B): steps ≳ 1 ulp (LoRA/adapter LRs, pre-training) or memory-bound runs.
* **`kahan8`** (+1 B): sub-ulp LRs; ~0.2 ulp at 10k steps, 0.5 at 100k, 1.6 at 1M (the
  √N walk of §2) — the default Kahan.
* **`kahan16`** (+2 B): fp32-master numerics exactly — very long sub-ulp runs where
  `kahan8`'s √N walk matters, the reference arm of an experiment, or whenever +2 B/param is
  affordable. Same memory as the legacy `kahan`, strictly better (exact, every path).
* **legacy `kahan`** (+2 B): fp16 parameters only, or checkpoints that carry `shift`.
