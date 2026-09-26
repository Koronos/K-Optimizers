# Low-discrepancy rounding for bf16 weight writes

CPU-only simulation of two low-discrepancy (LD) rounding candidates against the shipped
schemes, a long-horizon (100k steps) follow-up, a round of remedies against the aliasing it
exposed, and the variant that shipped as the EXPERIMENTAL `bf16_method="kahan8ld"`.

The simulator is bit-equivalent to kaon's writers (`check_equivalence.py`): a kahanB pair
`(w, lo)` is simulated as one fp32 whose low `16-B` mantissa bits are zero, so a write is
`z = z + upd` (fp32 RN), `bits += noise`, `bits &= -2**(16-B)`. Errors are in `ulp_ref` (the
bf16 ulp at the RMS of the fp32 reference); `std` is the spread of the tracked-value error,
`dir-bias` its mean projected on each element's movement sign (negative = the scheme lags
the reference), `lost` the fraction of the net movement not realised. Streams, seeds and
regimes are described in `sim_lowdisc.py`'s header.

## Files

| file | what |
|---|---|
| `sim_lowdisc.py` | the simulator (schemes, streams, metrics) |
| `check_equivalence.py` | bit-exactness of the sim vs `kaon._compact_kahan` / `_stochastic_rounding` writers, and of the `kahan8-ld-kaon` scheme vs kaon's `kahan8ld` writer |
| `run_all.sh` → `results/` | main grid: 1M coordinates, 10k steps, every scheme |
| `run_long.sh` → `results_long/` | 2**16 coordinates, 100k steps, the candidates |
| `run_remedies.sh` → `results_remedies/` | remedies against the aliasing, 100k steps |
| `run_remedies2.sh` → `results_remedies2/` | second remedy round (`b256k`) |
| `run_kaon_check.sh` → `results_kaon/` | the shipped noise (`kaon._compact_kahan.ld_noise`) in the same harness |
| `sim_ema_bf16.py` → `results/ema_bf16*.json` | side audit: EMA state stored in bf16 RN vs SR vs fp32 |
| `make_tables.py` → `tables.md` | every table below (and the per-regime detail) from the JSONs |

Reproduce: `bash run_all.sh`, `bash run_long.sh`, `bash run_remedies.sh`, `bash run_remedies2.sh`,
`bash run_kaon_check.sh` (CPU only, `PYTHONPATH=../../../../src`), then `python make_tables.py`.

## The candidates

* **A. LD stochastic rounding of the whole bf16 write (cost 0 B/param).** Plain bf16 SR with
  the 16 noise bits `u16 = (h16(i) + n·K) mod 2**16` (`h16` a fixed per-element hash, `n` the
  write counter, `K` an odd Weyl increment: φ, the plastic number R2, √2−1, and a deliberately
  bad 1/3).
* **B. `kahan8` with the residual rounded by an LD sequence.** Same idea at the residual's 8
  dropped bits: `u8 = (h8(i) + n·159) mod 256` instead of the residual's SR (`kahan8-ld`;
  `kahan8-ld-r2` uses 193; `kahan4-ld` is a 12-dropped-bit control).

## 10k steps: B wins everywhere, A does not

Std of the tracked-value error at 10k steps (ulp_ref); `(x %)` = movement lost when ≥ 1 %.
Full per-regime detail in `tables.md`.

| regime | sr | ld-phi | ld-r2 | kahan8-sr | kahan8-rn | kahan8-ld |
|---|---|---|---|---|---|---|
| adam-lr1e-06 | 2.735 | 2.450 | 2.288 | 0.144 | 0.770 (40 %) | 0.076 |
| alias-P2-lr1e-06 | 6.104 | 2.322 (1 %) | 1.393 | 0.204 | 14.930 (-98 %) | 0.105 |
| alias-P3-lr1e-06 | 4.872 | 2.146 (2 %) | 1.313 (1 %) | 0.198 | 10.987 (-145 %) | 0.050 |
| alias-P2-lr1e-04 | 32.892 | 135.3 (60 %) | 98.120 (37 %) | 0.155 | 3.622 (-1 %) | 0.012 |
| coh-lr1e-06 | 6.135 | 2.300 | 1.309 | 0.200 | 15.856 (-12 %) | 0.003 |
| orig-drift-lr1e-06 | 1.931 | 1.505 | 1.682 | 0.114 | 0.271 (86 %) | 0.021 |
| orig-lr1e-03 | 23.936 | 16.890 | 16.879 | 0.093 | 0.076 | 0.066 |

## 100k steps: B's picture is mixed

Std @100k (ulp_ref), `100k/30k` growth, directional bias @100k.

| regime | sr | kahan8-sr | kahan8-ld | kahan8-ld-r2 | kahan8-ld growth 100k/30k | kahan8-ld dir-bias |
|---|---|---|---|---|---|---|
| coh-lr1e-06 | 12.096 | 0.505 | **0.003** | 0.004 | 0.81 | +0.0002 ± 0.0000 |
| coh-lr1e-07 | 6.170 | 0.364 | **0.002** | 0.008 | 0.72 | +0.0001 ± 0.0000 |
| orig-drift-lr1e-06 | 6.119 | 0.361 | **0.065** | 0.143 | 1.82 | +0.0000 ± 0.0003 |
| adam-lr1e-06 | 8.584 | 0.455 | **0.239** | 0.257 | 1.82 | +0.0010 ± 0.0009 |
| noisy-mu0.0005-s0.005 | 19.150 | 0.597 | **0.423** | 0.425 | 1.85 | −0.0031 ± 0.0017 |
| orig-lr1e-05 | 52.592 | 0.594 | **0.418** | 0.418 | 1.83 | −0.0011 ± 0.0016 |
| alias-P7-lr1e-06 | 15.390 | 0.602 | **0.011** | 0.010 | 1.07 | +0.0002 ± 0.0000 |
| alias-P2-lr1e-06 | 19.426 | 0.656 | 1.864 | 0.762 | **5.01** | **−0.1091 ± 0.0073** |
| alias-P3-lr1e-06 | 15.423 | 0.629 | 1.562 | 1.610 | **6.41** | **−0.0970 ± 0.0061** |

* **Coherent and drift streams stay bounded** (0.003 against 0.36–0.5 for `kahan8-sr`): the
  sigma-delta behaviour the Weyl dither was chosen for.
* **Noise-dominated and Adam-like streams keep a constant-factor gain** (×0.53 Adam-like, ×0.7
  noise-dominated) but grow as `sqrt(N)` like SR (growth 1.82–1.85 per ×3.3 steps): once the
  update itself is noisy, the dither's ordering cannot beat the update's own walk; it only
  removes the rounding's share of it.
* **Aliasing with period 2 and 3 at lr 1e-6 grows SUPERLINEARLY** (×5–6.4 per ×3.3 steps) to
  1.86 / 1.56 ulp — worse than `kahan8-sr`'s 0.66 / 0.63 — with a **significant directional
  bias** (−0.109 ± 0.007: the weights lag). Working hypothesis (not isolated): the fixed Weyl
  dither stays phase-locked with a short-period update for thousands of writes, so the
  rounding errors correlate with the update's phase instead of averaging out. The R2
  increment (193) fixes P2 (0.76) but not P3 (1.61), consistent with any single fixed
  increment resonating with SOME short period — and the block-phase remedy below, which only
  breaks the phase lock, removes most of it.

### Verdict on A: rejected

The LD SR of the whole weight keeps SR's `sqrt(N)` growth in the noisy regimes, aliases badly
(P2/P3 at lr 1e-4: 60 % / 37 % of the movement lost for φ / R2 at 10k steps) and carries
large directional biases at 100k (−4 to −6 ulp on P2/P3). Not implemented.

## Remedies against B's aliasing (100k steps)

Three cheap families, each keeping `h8` and the 159 increment:

* **(a) block phase**: `u8 = (h8(i) + n·159 + r(i, n // K)) mod 256`, `r` a random phase per
  (element, block of K writes), `K ∈ {64, 256, 1024}` (`-b64`, `-b256`, `-b1024`); `-ld16-b256`
  combines it with (b);
* **(b) higher-rank counter**: 16 bits of the φ sequence, top byte:
  `u8 = ((h16(i) + n·40503) mod 2**16) >> 8` (`-ld16`);
* **(c) mixed dither**: `u8 = (h8(i) + n·159 + j) mod 256`, `j` iid uniform in `[0, J)`
  (`-j32`; `-ld16-j64` on top of (b)).

D = 2**16, 100 000 steps, same streams and seeds as `results_long`. Cells: std of the tracked-value error (ulp_ref) at 100k; **bold** = directional bias > 3 se; `(x)` = ratio to kahan8-sr.

| regime | kahan8-sr | kahan8-ld | kahan8-ld-r2 | kahan8-ld-b64 | kahan8-ld-b256 | kahan8-ld-b1024 | kahan8-ld16 | kahan8-ld16-b256 | kahan8-ld-j32 | kahan8-ld16-j64 |
|---|---|---|---|---|---|---|---|---|---|---|
| adam-lr1e-06 | 0.455 | 0.239 (0.53) | 0.257 (0.56) | 0.252 (0.55) | 0.241 (0.53) | 0.240 (0.53) | 0.239 (0.53) | 0.243 (0.54) | 0.312 (0.69) | 0.365 (0.80) |
| alias-P2-lr1e-06 | 0.656 | **1.864** (2.84) | 0.762 (1.16) | **0.135** (0.21) | **0.151** (0.23) | **0.499** (0.76) | **1.694** (2.58) | **0.172** (0.26) | **0.323** (0.49) | 0.410 (0.62) |
| alias-P3-lr1e-06 | 0.629 | **1.562** (2.48) | **1.610** (2.56) | 0.138 (0.22) | **0.115** (0.18) | **0.202** (0.32) | **1.562** (2.48) | **0.135** (0.21) | 0.297 (0.47) | 0.416 (0.66) |
| alias-P7-lr1e-06 | 0.602 | **0.011** (0.02) | 0.010 (0.02) | 0.256 (0.43) | 0.166 (0.27) | 0.129 (0.21) | 0.016 (0.03) | 0.179 (0.30) | 0.291 (0.48) | 0.398 (0.66) |
| coh-lr1e-06 | 0.505 | **0.003** (0.01) | **0.004** (0.01) | 0.093 (0.18) | 0.005 (0.01) | **0.005** (0.01) | **0.005** (0.01) | 0.045 (0.09) | 0.225 (0.45) | 0.317 (0.63) |
| coh-lr1e-07 | 0.364 | **0.002** (0.01) | **0.008** (0.02) | 0.094 (0.26) | **0.003** (0.01) | **0.003** (0.01) | **0.006** (0.02) | 0.064 (0.18) | 0.268 (0.74) | 0.317 (0.87) |
| noisy-mu0.0005-s0.005 | 0.597 | 0.423 (0.71) | 0.425 (0.71) | 0.434 (0.73) | 0.427 (0.71) | 0.426 (0.71) | 0.422 (0.71) | 0.427 (0.71) | 0.468 (0.78) | 0.508 (0.85) |
| orig-drift-lr1e-06 | 0.361 | 0.065 (0.18) | 0.143 (0.39) | 0.111 (0.31) | 0.068 (0.19) | 0.066 (0.18) | 0.068 (0.19) | 0.089 (0.24) | 0.266 (0.74) | 0.314 (0.87) |
| orig-lr1e-05 | 0.594 | 0.418 (0.70) | 0.418 (0.70) | 0.429 (0.72) | 0.422 (0.71) | 0.421 (0.71) | 0.421 (0.71) | 0.421 (0.71) | 0.467 (0.79) | 0.507 (0.85) |
| **worst ratio** | 1.00 (adam-lr1e-06) | 2.84 (alias-P2-lr1e-06) | 2.56 (alias-P3-lr1e-06) | 0.73 (noisy-mu0.0005-s0.005) | 0.71 (noisy-mu0.0005-s0.005) | 0.76 (alias-P2-lr1e-06) | 2.58 (alias-P2-lr1e-06) | 0.71 (noisy-mu0.0005-s0.005) | 0.79 (orig-lr1e-05) | 0.87 (coh-lr1e-07) |

**Second round.** The residual P2/P3 directional bias of `b256` (−0.008 / −0.004 ulp, 13 / 10
se) suggested also re-drawing the Weyl increment per (element, block) among four good odd
ones (159, 97, 193, 105): `b256k`.

D = 2**16, 100 000 steps, same streams and seeds as `results_long`. Cells: std of the tracked-value error (ulp_ref) at 100k; **bold** = directional bias > 3 se; `(x)` = ratio to kahan8-sr.

| regime | kahan8-sr | kahan8-ld-b256 | kahan8-ld-b256k |
|---|---|---|---|
| adam-lr1e-06 | 0.455 | 0.241 (0.53) | 0.245 (0.54) |
| alias-P2-lr1e-06 | 0.656 | **0.151** (0.23) | 0.042 (0.06) |
| alias-P3-lr1e-06 | 0.629 | **0.115** (0.18) | **0.126** (0.20) |
| alias-P7-lr1e-06 | 0.602 | 0.166 (0.27) | 0.138 (0.23) |
| coh-lr1e-06 | 0.505 | 0.005 (0.01) | **0.005** (0.01) |
| orig-lr1e-05 | 0.594 | 0.422 (0.71) | 0.420 (0.71) |
| **worst ratio** | 1.00 (adam-lr1e-06) | 0.71 (orig-lr1e-05) | 0.71 (orig-lr1e-05) |

### Verdict on B: ship `kahan8-ld-b256` as `bf16_method="kahan8ld"` (experimental)

* **(b) alone does nothing** against the aliasing (`ld16` 1.69 / 1.56 on P2/P3): a finer
  quantisation of φ does not change which short periods the sequence resonates with.
* **(c) trades the whole advantage away** (`j32`: coherent 0.23 instead of 0.005; the jitter
  re-introduces a walk) — it is "SR with a little LD", not the other way round.
* **(a) is the remedy.** Re-randomising the phase caps the phase-locked stretch at K writes,
  so the aliased error becomes a sum of independent per-block contributions (a `sqrt(N)` walk
  with a tiny step) instead of a coherent drift. K trades the two failure modes: K = 64 is
  the best on aliasing but re-randomises too often for the coherent regime (0.093); K = 1024
  keeps the coherent 0.005 but leaves P2 at 0.50. **K = 256 has the best worst case: 0.71× of
  `kahan8-sr`'s error over all nine regimes** (the worst being the noise-dominated streams),
  keeps the coherent (0.005) and Adam-like (0.241) advantage and cuts P2 / P3 from 1.86 / 1.56
  to 0.15 / 0.12 ulp.
* **Remaining risk**: a small but statistically significant directional bias on P2/P3
  (−0.008 ± 0.0006 and −0.004 ± 0.0004 ulp at 100k, growing roughly linearly: −0.0006,
  −0.0020, −0.0082 at 10k/30k/100k on P2), i.e. 0.02 % / 0.01 % of the movement lost where
  `kahan8-sr` loses nothing measurable. It is 13× smaller than the pure LD's and far below
  `kahan8-sr`'s error spread, but it IS a bias: over 1M steps it would extrapolate to
  ~0.1 ulp of lag on an exactly periodic update. `b256k` removes it on P2 (0.042 ulp, bias
  gone) but not on P3 (−0.0064 ± 0.0005), for the cost of a per-element increment lookup —
  not a clear enough win to take instead.
* A real update is neither exactly periodic nor sign-coherent for 100k steps; whether any of
  this shows up in training is what `benchmarks/lowlr_bf16` (`ada-k8ld`, `nek-k8ld`) measures.

## The shipped noise in the same harness

`kaon._compact_kahan.ld_noise` uses 32-bit hashes (lowbias32) instead of the sim's
splitmix64, and one fixed key per parameter: `h = mix32(i ^ key)`,
`r = mix32(h ^ mix32((n // 256 + 1)·golden ^ salt))`, `u8 = (h>>24 + r>>24 + 159·n) mod 256`.
Statistically the same scheme; `check_equivalence.py` pins the sim's `kahan8-ld-kaon` to
kaon's writer bit for bit, and `run_kaon_check.sh` reruns the key regimes with it — the
shipped noise reproduces `kahan8-ld-b256` within the seed noise everywhere (P2/P3 directional
bias −0.0080 ± 0.0006 / −0.0032 ± 0.0004; the coherent one, +0.0001, is 1e-4 ulp):

D = 2**16, 100 000 steps, same streams and seeds as `results_long`. Cells: std of the tracked-value error (ulp_ref) at 100k; **bold** = directional bias > 3 se; `(x)` = ratio to kahan8-sr.

| regime | kahan8-sr | kahan8-ld-b256 | kahan8-ld-kaon |
|---|---|---|---|
| adam-lr1e-06 | 0.455 | 0.241 (0.53) | 0.240 (0.53) |
| alias-P2-lr1e-06 | 0.656 | **0.151** (0.23) | **0.156** (0.24) |
| alias-P3-lr1e-06 | 0.629 | **0.115** (0.18) | **0.114** (0.18) |
| alias-P7-lr1e-06 | 0.602 | 0.166 (0.27) | 0.165 (0.27) |
| coh-lr1e-06 | 0.505 | 0.005 (0.01) | **0.005** (0.01) |
| noisy-mu0.0005-s0.005 | 0.597 | 0.427 (0.71) | 0.427 (0.71) |
| orig-lr1e-05 | 0.594 | 0.422 (0.71) | 0.420 (0.71) |
| **worst ratio** | 1.00 (adam-lr1e-06) | 0.71 (noisy-mu0.0005-s0.005) | 0.71 (noisy-mu0.0005-s0.005) |

## Implementation notes (`bf16_method="kahan8ld"`)

* Same state as `kahan8` (one `uint8` `kahan_lo`); `kahan8 ↔ kahan8ld` switches reuse it.
* The counter `n` is Adakaon's checkpointed step `_t` (the native path advances it too since
  this change): `n = t` for a bare Adakaon; under MSAM/Nekaon each write of a step gets its
  own value — removal `3t`, base step `3t+1`, climb `3t+2`, i.e. one global counter that
  advances by one per write, like the write counter `nw` of the simulator. (The sim's
  `multi-*` three-writes-per-step regimes were not run at the long horizon, so this choice is
  not validated by the tables above.) An `eval()`/`train()` pair re-uses the counters of the
  surrounding removal/climb.
* A first end-to-end check (review measurement, **one seed only**: 128×128 weights, lr 1e-5,
  2000 steps, error of the decoded value against the fp32 run, ulp): Adakaon kahan8 0.088 vs
  kahan8ld 0.051; Nekaon 0.144 vs 0.096; Nekaon with `low_vram_above` 0.084 vs 0.062.
* The key is a hash of the parameter's ordinal in the optimizer, never of its slot in a stack,
  so per-param, foreach and fused writes dither each element identically whatever it is
  batched with. The noise is deterministic: per-param == foreach bit for bit; the Triton
  helpers (`ld_noise_dev`, the keyed `ck_ptr` handle with `CK=9`, `_ck_axpy_ld_kernel`) are
  bit twins of the torch reference; fused vs native differ only by the fp32 paths' own
  arithmetic differences.
* Only Adakaon (and MSAM/Nekaon over it) accept it; every other writer refuses a `kahan8ld`
  write without its noise rather than falling back to SR.
* Known limitation: with MSAM/Nekaon's stride 3, parameters that never climb (Nekaon's
  `low_vram_above` group without momentum) still advance 3 counters per step, i.e. their
  dither increment per write is `3·159 ≡ 221 (mod 256)`, not the simulated 159.
* Known limitation: the dither pattern is FIXED across runs (the key salt is a constant), so
  two seeds of an experiment share the same per-element pattern; only the data and the init
  differ between them.
* Cost note: each per-param / stacked native CUDA write copies the row keys host→device
  (a tiny `torch.tensor(keys).to(device)` per write); the fused Adakaon buckets build their
  keyed residual array once per pointer cache instead.
