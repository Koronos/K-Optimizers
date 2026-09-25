# Compact Kahan — `bf16_method="kahan8"` (1 B/param, every path)

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
* `kahan4` (0.5 B/param) is 16× worse (1.5–3 ulp) and not implemented; `kahan16` (2 B/param)
  is an exact fp32 master weight split in two and is only in the simulation as a reference.
* The legacy `kahan` stays untouched (its checkpoints hold `shift`; per-param only). `kahan8`
  is the recommended Kahan.

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
  compensation, as the 0.7.13 fix established for `shift`). **SAM / MSAM / Nekaon** climbs
  write `p` without touching `kahan_lo`: the residual then re-attaches as a fraction of the
  new `ulp(w)` — a bounded ≤ 1-ulp inconsistency per climb, the same the legacy `shift`
  buffer has, and their climb writes were SR/cast-rounded before anyway. **[inference]**
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
  in §§1–3. Name: `kahan8` (the residual width in bits; a future `kahan4` would fit the table).
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
* MSAM / SAM / Nekaon climb writes bypass the residual (bounded, see §4); a climb-aware
  writer would carry it exactly.
* ScheduleFree rejects `kahan8` (its `y`/`z` write is its own).
* `foreach_budget`'s `bytes_per_elem` estimate does not include the residual's 1 B; the
  transient is 3 B/elem on the Triton axpy, so the 10 %-of-free-VRAM chunk rule still holds
  with margin, but a `kahan8`-aware estimate would be tidier.
* The `hybrid` SR of the residual consumes one Philox draw per element per step in the fused
  kernels — the same draw the SR write already consumed, so no added cost; on the torch
  reference it is one `randint` (int32) per element like SR's.
