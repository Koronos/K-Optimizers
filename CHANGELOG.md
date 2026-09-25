# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **`bf16_method="kahan8"` — compact Kahan at +1 B/param, on every path.** The legacy
  `"kahan"` keeps a bf16 compensation buffer (+2 B/param, the model again) and only exists
  on the per-parameter writer. `kahan8` stores the residual as ONE `uint8` per parameter in
  units of the weight's own bf16 ulp: the pair `(bf16, byte)` is an fp32 whose low 8
  mantissa bits are zero — a 16-bit-significand float whose stored bf16 half is the nearest
  bf16 (round half away), so the forward pass never sees a truncation. The exponent of the
  weight is the scale (no per-block scale, no reduction), binade crossings are the `+1`
  carry on the 16-bit pattern, zero/subnormals sit on the same monotone pattern, the decode
  is integer-only, non-finites propagate as `sr_round` does.

  The residual is **stochastically rounded** at its own grid (the SR bit trick on the 8
  dropped bits): unbiased for every step size, so the tracked value is a martingale with a
  `ulp/256 · sqrt(N/6)` walk — 0.16 ulp at 10k steps. Simulated (2^20 coords, 10k steps,
  fp32 reference, `docs/research/compact-kahan.md`): 0.09–0.19 ulp drift in every regime
  from 1 to 0.004 ulp/step, with or without an Antikaon-style perturbation carried in the
  weights, bias 0 within 2 SE, **no stall**; plain stochastic rounding walks 17–45 ulp, and
  the legacy bf16 `kahan` buffer — as accurate on dithered updates — **loses 31 % of the
  movement on pure sub-grid drift** because it rounds deterministically. A 4-bit residual
  was evaluated (16× the drift for 0.5 B/param) and not shipped.

  Coverage: `subtract_one_` (per-param) and `subtract_batched_(…, comp=)` (foreach, the
  bucket's residual views via `ForeachChunk.cviews`) in `kaon._backend`; a one-launch Triton
  axpy (`kaon._fused_triton.ck_add_`) that the native CUDA writers take, as fast as the SR
  write (0.074 vs 0.070 ms per 2^22 elements, orientative) and with no int32 scratch; and
  **Adakaon's fused path** — all eight apply kernels (`_adakaon_tile_kernel`,
  `_adam_1d_kernel`, the lone and batched chunked kernels incl. the in-Triton int8/4-bit
  and no-momentum routes) take a `CK` constexpr and read/write the residual byte in the same
  launch (+1 B/elem in, +1 B/elem out, no extra launch). AdaBelief, AdamP, AdaMuon, ADOPT,
  KProdigy, Lion and AdaPNM accept it on their foreach and per-param paths (AdaPNM's fused
  route declines it with its existing reason string); Lookahead's slow-weight sync carries
  the inner's residual (bf16 chunks only — a mixed fp32/bf16 group is fine); ScheduleFree
  rejects it. **MSAM / Nekaon climbs are kahan8-aware**: the perturbation is applied to and
  removed from the decoded compensated value and re-encoded (torch path and the fused
  `_axpy_momentum_batched` kernel, `CK` constexpr). Perturbing the bare bf16 weight while
  the residual stays put loses the sub-ulp part of the climb coherently every step —
  measured 12 ulp (MSAM) and 25 ulp (Nekaon, worse than stochastic rounding) after 300
  steps at lr 1e-5 against 0.3 for climb-free Adakaon; with the fix 0.32–0.46 ulp
  (`docs/research/compact-kahan.md` §4b). Robustness: a finite weight never decodes to a
  non-finite value (all 2^24 `(bf16, byte)` states enumerated; a `±0` weight written from
  outside with a set top residual bit used to wrap to NaN), a group switched to `kahan8`
  mid-run gets zero residuals allocated with one warning on every route (a fused launch
  never substitutes another buffer for a missing residual array — it refuses), the
  residual's stochastic rounding is exactly unbiased over the enumerated 256 noise values
  (torch and Triton), and the foreach chunk budget counts the residual stack. `kahan_lo` is ordinary per-param state
  (`uint8`, `p.shape`): saved by `state_dict`, restored bit-exactly by
  `load_state_dict_preserving_dtypes`, watched by the pointer-cache generation, and its
  noise draws from the optimizer's checkpointed stream, so a resume is bit-identical to the
  uninterrupted run. fp16 parameters are refused (use `"kahan"`). Measured state: 3.04 B/param
  vs 4.04 (`kahan`) and 2.04 (SR) with bf16 momentum — exactly +1.000 B/param, no first-step
  transient on the fused path. `tests/test_compact_kahan.py`.

  `"kahan"` is unchanged (checkpoints with `shift` load as before) and documented as legacy;
  `BF16_METHODS` / `validate_bf16_method` / `init_bf16_state` /
  `per_param_only_bf16_method` in `kaon._backend` replace the per-optimizer copies of the
  method check, the `shift` allocation and the foreach predicate.
- Experimental `Rakaon`: momentum-free variance shrinkage, tensorwise or contiguous
  block variance, RMS clipping, and checkpointable BF16 stochastic rounding.
  Proxy studies have not established superiority over existing optimizers.
- Reproducible diffusion studies measuring time to evaluated loss/gap and allocated
  memory, plus an Anima/Pets pilot with seeded adapter fingerprints and previews.

## [0.7.13]

### Fixed
- **`Lookahead` + `bf16_method="kahan"` on bf16/fp16 parameters no longer dies with
  `KeyError: 'shift'` at the first slow-weight sync.** The sync's `theta <- phi` reset goes
  through `subtract_one_`, whose Kahan branch reads the parameter's compensation buffer
  `state["shift"]`. `Lookahead._sync_one` handed it the **wrapper's** per-parameter state,
  which only ever holds `phi`/`phi_scale`/`phi_numel`/`phi_block`/`backup` — so the very
  first sync raised, on `foreach=True` and `foreach=False` alike (`_sync` routes Kahan to
  the per-parameter path either way, because `subtract_batched_` has no Kahan branch at
  all). Any `Lookahead(..., bf16_method="kahan")` over low-precision parameters was
  unusable; every `slow_dtype` was affected, and the failure was immediate rather than
  silent.

  The fix hands the sync the **inner optimizer's** state instead. Kahan compensation is a
  property of the *weight*, not of whoever writes it: the inner step and the sync both
  write the same `p`, so the bits both of them drop have to accumulate in one buffer. A
  buffer of the wrapper's own would have split the residue in half and, worse, carried
  compensation for a `theta` that the `theta <- phi` reset had already overwritten into the
  next inner step. The buffer the inner allocates is already there under the same predicate
  (`is_low_precision(p) and bf16_method == "kahan"`), on the right device and dtype, so
  nothing new is allocated and no checkpoint key is added. `SAM`/`MSAM`/`Nekaon` have no
  analogous call site — SAM's climb writes through `add_stochastic_` and is undone by an
  exact `copy_` of its snapshot, MSAM's is a plain cast — and `_sync_foreach` needs nothing
  because Kahan never reaches it.

  **Cost: zero.** One different dict reference at the sync's single call site — no extra
  memory, no extra kernel, nothing on the step path. Verified bit-identical to the pre-fix
  tree over 576 non-Kahan configurations (`Lookahead` over Adakaon × CPU/CUDA ×
  fp32/bf16 parameters × `bf16_method` none/stochastic_rounding × fp32/bf16/int8/4-bit
  `slow_dtype` × fp32/bf16/int8 momentum × foreach on/off × fused/native × momentum
  on/off), digesting the live weights, the stored `phi` and every inner state tensor.
  Numerically the repaired sync earns its keep: against an fp32 reference fed the same
  bf16-rounded gradients, 24 steps at `k=2` leave the Kahan run's slow weights ~2.8× closer
  in max-abs and ~2.9× closer in L2 than `bf16_method="none"`.

- **A wrapper's own state now resumes on the parameter's device, so
  `torch.load(path, map_location="cpu")` works.** `WrapsInnerOptimizer._load_wrapped`
  installed the checkpoint's per-parameter dicts verbatim, so a state dict loaded
  anywhere other than the parameters' device left the *wrapper's* buffers there. With
  the near-universal consumer idiom — `torch.load(ckpt, map_location="cpu")` into a CUDA
  model — `Lookahead`'s slow weights `phi` (and its int8/4-bit scales) stayed on the CPU
  and the next slow-weight sync died with `RuntimeError: Expected all tensors to be on
  the same device, cuda:0 and cpu`. The inner optimizer was never affected:
  `torch.optim.Optimizer.load_state_dict` (reached through
  `load_state_dict_preserving_dtypes`) already puts inner state on the param's device,
  which is why only the wrapper half broke.

  The fix applies exactly that policy to the wrapper's own state, in the one place every
  wrapper loads through: **device follows the parameter, storage dtype is preserved
  bit-for-bit** — `phi` is never widened from bf16 to fp32 or narrowed back, and the
  int8/4-bit code payloads and their fp32 scales keep their dtypes. Both directions are
  covered (a CUDA checkpoint restored under CPU parameters too), and an explicit
  `map_location=torch.device("cuda:0")` keeps working.

  Affected in practice: **`Lookahead`** (persistent `phi`/`phi_scale`, plus the `backup`
  a checkpoint taken in eval mode carries) and **`SAM`** (a checkpoint taken between
  `first_step` and `second_step` carries `old_p`; SAM's restore is a cross-device `copy_`,
  so it did not raise — it just silently synchronized). **`MSAM`/`Nekaon`** keep no
  persistent per-parameter state of their own (the perturbation is recomputed from the
  inner momentum, which torch migrates), so they were correct already and are now covered
  by the same contract. `Schedule-Free` is not a wrapper of this kind and goes through
  `load_state_dict_preserving_dtypes` directly.

  **Cost: zero per step, and zero on a same-device load.** The migration lives only on
  the checkpoint path, and a tensor already on the parameter's device is installed as
  itself — no `.to()`, no allocation, no copy. Verified bit-identical to the pre-fix tree
  across 168 configurations (`Lookahead`/`SAM`/`MSAM`/`Nekaon` over Adakaon x CPU/CUDA x
  fp32/bf16 parameters x fp32/bf16/int8/4-bit momentum x fused/native, `Lookahead` also x
  fp32/bf16/int8/4-bit `slow_dtype`), 3264 state and parameter buffers digested; and a
  `map_location="cpu"` resume is bit-identical to the uninterrupted run over
  `slow_dtype` x `momentum_dtype` (16 arms).

- **Gradient Centralization silently froze every parameter whose fan-in was 1.** GC
  subtracts, per output row, the gradient's mean over the fan-in dims; over a row of ONE
  element that mean *is* the element, so `g - mean(g)` was identically zero and the
  parameter never moved — on the native, foreach, one-block-fused, lone-big-fused and
  batched-big-fused routes alike, at every momentum dtype. The shapes that hit it are
  ordinary: rank-1 LoRA up-projections `(out, 1)`, one-input 1x1 convs
  `(out, 1, 1, 1)`, some projections. It affected all eight optimizers that default GC on
  (`AdaMuon` only when it is switched on), and it is what escalated to NaN in ADOPT
  (fixed separately above). `fan_in == 0` (an empty weight) fell on the same cliff, where
  GC manufactured a NaN out of `mean()` of nothing; it is skipped too.

  **GC is now skipped where it is undefined**, which required one predicate rather than a
  patch per site: GC lives at nine host sites and sixteen Triton kernels, and a
  native-only skip makes fused and native disagree. `kaon._backend.gc_applies(shape)`
  (`ndim >= 2 and fan_in > 1`) is the single definition; the torch sites call it, and the
  Triton kernels — which cannot call Python — receive it per shape/tile bucket as their
  existing `GC` `tl.constexpr`, via `kaon._fused_triton.bucket_gc_ok`. That is sound
  because the predicate is uniform inside every bucket the fused paths build (the big
  routes bucket by exact shape; the one-block routes bucket by the padded tile, and
  `BC == next_pow2(C) == 1` exactly when `C == 1`), and `bucket_gc_ok` **raises** rather
  than guessing if a future bucket key ever mixes the two. The AdaPNM/Adakaon batched-big
  routes take particular care: their mom/apply kernels re-apply GC from the `rowmean` the
  reduction kernel wrote, so the one resolved flag is threaded to both instead of being
  re-derived, which is what keeps `rowmean` coherent with the reduction that produced it.

  **Cost: zero per step.** The predicate is evaluated where plans and pointer caches are
  built, not per step — `centralize_grads_` applies it once per distinct shape bucket
  (~4 calls on a 428-adapter bag instead of 428), `PointerArrayCache`/`AdaPnmCache` store
  it per tile bucket, and `BigPointerCache`/`BigPnmCache` store it per shape bucket, so a
  launch site only ANDs two booleans. Measured with `torch.profiler` on bags with no
  fan-in-1 parameter: identical kernel set, identical launch counts, and A/B-paired
  wall time within run-to-run noise on LoRA-428 fused and foreach.

  **Verified bit-identical for fan-in >= 2** against the pre-fix tree across 736
  configurations — 7 optimizers x fp32/bf16 parameters x fp32/bf16/int8/4bit momentum x
  fused/native x foreach/per-param, over four bags (428 LoRA factors, 448 0-D scalars, a
  mixed bag with 3x3 and 1x1 convs plus 1-D/0-D, and a UNet/DiT-shaped bag of distinct
  big shapes plus a same-shape batched bucket), hashing final weights and every state
  tensor after N steps. **732 of 736 hash-identical, zero differences.** The remaining 4
  are the UNet bag on AdaPNM's batched-big fused route with fp32 parameters (all four
  momentum dtypes), which is not reproducible against *itself* on either tree — its
  column/rms reductions accumulate with fp32 `tl.atomic_add`, so the summation order
  varies per run. Compared by envelope instead, 4 interleaved runs of each tree: the
  base-vs-branch spread is 2.384e-07, exactly equal to base-vs-base and
  branch-vs-branch (1 ULP at that magnitude).

- **No Adakaon/AdaPNM route can step a RETIRED state buffer any more** (and no wrapper on
  top of one: MSAM, Nekaon, Lookahead, SAM). Optimizers that do not install the watch —
  Lion, AdamP, AdaBelief, ADOPT, AdaMuon, ScheduleFree, KProdigy — are unchanged, and so
  is a wrapper over one of those. Every cross-step cache in kaon
  was validated against a *parameter* witness (ids, `data_ptr`s, contiguity) and nothing
  observed `self.state`. The fused pointer tables bake the `data_ptr()` of
  `m`/`m_scale`/`row`/`col`/`v` at build; the native foreach plan caches *views* of the
  same buffers (`ForeachChunk.state_views`, plus the codec's `m`/`m_scale` lists). So a
  state buffer swapped out while every parameter stood still left them addressing a dead
  tensor and the next step **wrote** it. Confirmed on all four fused routes (one-block,
  big, 1-D, 0-D) *and* on the native plan, for three ordinary operations:

  - **`del opt.state[p]`** — the recovery `check_state_geometry` itself documents. It was
    safe only *after* a refused rebind (the refusing cache had never finished building)
    and silently unsafe from a steady state: on the big route the retired momentum was
    written (`max|Δp|` vs the per-parameter reference 5.24e-4), and the fresh state never
    went through `_init_state` at all.
  - **`opt.state[p].clear()`** — same, and the same dict is then refilled with fresh
    buffers by the next `_init_state`, so both generations are live at once.
  - **`opt.state[p]["m"] = …` / `["row"] = …`** — an external EMA, a partial
    `load_state_dict` that bypasses the optimizer's own loader, a checkpoint tool that
    casts one buffer, a requant that does not follow the in-place codec contract.

  `Adakaon.load_state_dict`, `AdaPNM.load_state_dict` and the AutoLR base-state reset were
  always safe (they call `_invalidate_fused_caches` explicitly) and still are; the fix is
  for the paths that never go through the optimizer. The per-parameter path never had the
  blind spot — it re-reads `self.state[p]` every step — and is the reference the tests
  compare against.

  **Watching a `dict` means covering every slot it mutates through**, which is more than
  the methods one thinks of. `|=` reaches `nb_inplace_or`, a different C slot from both
  `__setitem__` and `update`, and it went through the first round of this work unhooked:
  `opt.state[p] |= {"m": fresh}` retired the buffer while the counter stood still (measured
  4/4 on every fused route, on the foreach plan, and through MSAM / Nekaon / Lookahead /
  SAM), and `opt.state |= {p: {...}}` additionally left the incoming plain dict unadopted —
  a mapping that looked watched with an entry that was not. Reassignment was the other hole:
  `opt.state = defaultdict(dict)` invalidated every cache exactly once (the generation went
  N -> 0, so the next step looked fine) and then left a counter-less mapping behind, 4/4
  written on the step after. Both are closed — `__ior__` on both classes, and
  `WatchedStateMixin.__setattr__` re-wrapping any mapping assigned to `.state` (interception
  on the WRITE, not a `state` property: a property getter would cost a Python call on every
  `self.state` READ, and an AdaPNM fused step makes 1687-1884 of those on the 428-parameter
  bag (Adakaon fused: 631; Adakaon native: 1). At 21-42 ns more per access than a plain
  attribute that is +40…+71 µs/step — one to two orders of magnitude above the 0.13-0.80 µs
  the guard itself costs. The same step writes the attribute 1-4 times. Two machines.)
  `tests/test_state_identity_witness.py` now sweeps `dict`'s entire mutation API from a
  table against both classes, so a new mutator has to be added rather than remembered.

  **The fix is a counter, not a witness field**, because no field is both cheap enough and
  complete. `Adakaon.state` / `AdaPNM.state` are now a `kaon._foreach_plan.WatchedState`:
  the mapping counts every rebinding of a key some cache bakes, and `_fused_partition`,
  `_foreach_chunks`, `_WitnessedCache.{built_from,stale,revalidate}` and
  `MSAM._momentum_params` all carry the count they were built at. Priced against the
  three-field parameter witness on the 428-parameter LoRA bag
  (`benchmarks/fused/bench_state_witness.py`, 25 pairs, 95% CI, **two machines** — the
  ordering was identical on both, the absolutes differ by ~2x):

  | candidate | added host / step | % of the fused step | `del` | `state[p]["m"] = …` | `state[p].clear()` |
  |---|---|---|---|---|---|
  | state `m` `data_ptr` field | +110 … +144 µs | 7.0 … 9.2% | yes | yes | **no** |
  | state dict `id` field | +45 … +95 µs | 2.9 … 6.0% | yes | **no** | **no** |
  | **generation counter (shipped)** | **0.13 … 0.80 µs** | **0.01 … 0.09%** | yes | yes | yes |

  The counter's row is an ACCOUNTING figure — 133 ns/call measured directly over 10k
  calls, times the calls per step (1 native, 2-6 fused, counted by wrapping the
  function) — because the paired end-to-end test cannot resolve it: the same harness that
  put the two witness fields well outside their CIs returns −1.8 µs [−6.9, +3.2] and
  +12.2 µs [+4.2, +20.2] for the counter on the two machines, i.e. zero either side of
  its own noise floor. Populating a key
  that was *absent* does not move it, so `_init_state` stays free and a parameter's first
  step costs no rebuild on its second. `torch.profiler` counts are identical to the
  pre-fix tree on every bag — no new CUDA kernel, launch, `aten::` call or
  synchronisation — and the Python bytecode executed per step moves by +0.05…+0.74%
  (Adakaon) and +0.34…+1.45% (AdaPNM).

  **The one cost that is not free**, reported rather than hidden: with the watch installed
  every `state[key] = value` goes through a Python-level `__setitem__`. Adakaon makes no
  per-step state write and pays nothing. AdaPNM makes one per parameter
  (`state["step"] += 1`), measured **+56 … +107 µs (1.1 … 1.7% of that bag's AdaPNM fused
  step)** through the hook across the two machines, so that single write now goes through
  `dict.__setitem__` instead: **+18.5 … +48 µs (0.35 … 0.9%)**, of which the whole
  remainder is the unbound-method call form and not the dict being a subclass. On the
  slower machine that is still above the 0.5% budget for AdaPNM, and it is the price of
  covering the `state[p]["m"] = …` and `state[p].clear()` rows above; `"step"` is not a
  baked key, and `tests/test_state_identity_witness.py` sweeps `dict`'s ENTIRE mutation
  API to pin both the invariant that keeps the bypass legal and the fact that no other
  route out of it exists.

  Verified **bit-identical** to the pre-fix tree across 240 configurations (Adakaon /
  AdaPNM / MSAM / Nekaon / Lookahead × fused/native × fp32/bf16 parameters ×
  fp32/bf16/int8/4bit momentum × LoRA-428 / 448×0-D / mixed / UNet-chunked-big bags),
  plus 40 more with `deterministic_reductions=True`. The only 56 configurations that
  differ are exactly the 56 the pre-fix tree does not reproduce **against itself** (the
  chunked-big route's fp32 atomics — the reason `deterministic_reductions` exists); with
  that switch on they are bit-identical, and AdaPNM's big route, which has no such switch,
  shows the same worst `|Δ|` as the pre-fix tree's own run-to-run spread (3.9e-3 bf16 /
  2.4e-7 fp32, i.e. 1 ULP of the parameter dtype).

  `MSAM._momentum_params` gained the same field: its `(id(state), len(state), group
  sizes)` key saw `del opt.state[p]` but not `opt.state[p].clear()`, which left its cached
  plan holding the emptied dicts and raised a bare `KeyError: 'm'` from inside the wrapper.
  A rebound `st["m"]` was already caught by `_plan_addrs_valid`. **This only helps over a
  WATCHED base** (Adakaon / AdaPNM, including through Nekaon): with
  `base_optimizer=Lion`/`AdamP`/etc. the generation is a constant, so
  `MSAM(base_optimizer=Lion)` + `inner.state[p].clear()` still raises `KeyError: 'm'`
  exactly as before (measured).

  Also fixed, found while wiring the above: `benchmarks/fused/bench_wd_mblock.py` and
  `AdaPNM._probe_routing` (the `KAON_PROBE_LOG` diagnostic) unpacked `_fused_part`'s memo
  at a fixed arity and would have raised `ValueError: too many values to unpack` the moment
  a field was added. Both now slice the four route lists off the END (`entry[-4:]`), which
  is what every other consumer already did.
- **Fused path: a `p.data` rebind that changes the SHAPE can no longer step a plan that
  disagrees with the optimizer state.** Every fused pointer cache
  (`PointerArrayCache`, `BigPointerCache`, `AdaPnmCache`, `OneDimPointerCache`,
  `OneDimPnmCache`, `BigPnmCache`) now validates the state geometry against each
  parameter when it is built — `row`/`col` against the effective 2-D shape on the
  factored routes, `v` against `numel` on the 1-D/0-D route — and raises a
  `RuntimeError` naming the parameter, the mismatch and the recovery (reshape before
  constructing the optimizer, or `del opt.state[p]`). Affects Adakaon and AdaPNM.

  Two real out-of-bounds paths, not hypothetical ones, both confirmed silent on 0.7.12:

  - **Rebind of both storage and shape** (`p.data = p.data.view(64, 16).clone()`): the
    witness moved, the plan was rebuilt with the *new* `R`/`C`, and the kernel indexed 64
    entries of a 16-element `row`. The overrun is a **write**, not just a read — it lands
    on the neighbouring allocation, measured as `state[p]["col"]` being modified 512 B
    past `row` (Δ 2.36e-3) — and the step raised nothing and left finite weights.
  - **`load_state_dict` from a checkpoint with different shapes.** A checkpoint saved from
    `(512,16)` weights loads onto `(16,512)` ones without complaint (same param count,
    same numel; `load_state_dict` does not compare shapes), after which the factored state
    carries `row=512`/`col=16` against an effective shape of `(16,512)`: 2048 B written
    into a 64-byte `col`, leaving `p` non-finite with no exception. No parameter was
    rebound here, so no witness field is involved — what catches it is the cache rebuild
    `load_state_dict` already forces, plus the new check.

  **Cost: zero per step.** The check runs only when a cache is built, so a steady-state
  step does not execute it. Verified bit-identical to 0.7.12 across 32 configurations
  (Adakaon/AdaPNM x fused/native x fp32/bf16 parameters x fp32/bf16/int8/4bit momentum,
  on a bag covering the one-block, chunked-big, 1-D, 0-D and conv routes), 1528 state
  and parameter buffers digested.

- The native foreach plan already refused such a rebind (it re-stacks by effective shape
  every step, so `torch.stack` raises); its witness is deliberately unchanged, and
  `Adakaon._fused_partition` now keys on `ft.param_witness` so the fused key is exactly
  as strong as the pointer caches' own — which is what the caches' O(1) `built_from`
  revalidation assumes.

- **The shared bf16 stochastic-rounding kernel took its seed from a process-global counter
  that no checkpoint saved**, so a resume in a fresh process applied a *different* noise
  sequence than the run it continued: with every state tensor restored bit-exactly, a
  resumed run's bf16 weights diverged from the uninterrupted run's by up to **4.7e-2**
  absolute (CUDA, native/foreach path, 4+4 steps) — for **every** optimizer with bf16
  params, and for `Lookahead`'s `phi` sync (whose bf16-correct write goes through the same
  kernel) even over an Adakaon whose own fused counter was already checkpointed. Measured
  before: Adakaon fused `0.0` (already fixed by `_adakaon_meta`), Adakaon native `4.7e-2`,
  Lookahead `1.6e-2`, SAM/MSAM/Nekaon native `4.7e-2`. MSAM had fixed only its own climb
  (`_msam_meta.axpy_seed`).

  The counter is now **per owner** (`kaon._stochastic_rounding.SRStream`): the optimizer
  passes it to the writers (`sr=self.sr_stream`), saves it in its `state_dict` and restores
  it on load. A process-global counter could not be fixed any other way — restoring it from
  optimizer A's checkpoint would rewind optimizer B's noise, and leaving it alone breaks
  A's own resume; per-owner streams make both resumes exact at once, and a mixed stream
  identity (checkpointed, so it survives a load in a different order) keeps their noise
  independent. One stream covers both SR paths: the Triton kernel's launch counter and the
  torch reference path's private per-device generator, whose state is checkpointed too, so
  CPU params and `SR_TRITON = False` resume exactly as well. The `_reseed_hooks` registry is
  gone — `reseed_stochastic_rounding()` now bumps a module epoch that every stream checks on
  use, which also covers streams created after the call.

  Zero measurable step cost: no added kernel and no added sync, and the host-side seed
  derivation came out at **0.71–0.94x** of 0.7.12's (paired in-process microbenchmark,
  −0.05 to −0.4 µs/draw — the per-device cache is keyed by the `torch.device` object and
  holds a bound seed reader, avoiding a `device.type` read that costs ~0.7 µs on its own),
  and the stream is reached through a lazily-installed instance attribute rather than a
  property (~2x cheaper per read: 130–159 ns → 67–82 ns, i.e. −0.02 to −0.03 ms/step at one
  read per weight on a 428-adapter bag). Per-step wall clock on the LoRA-428 bf16 bag is
  inside the run-to-run spread of the shared card (±25% on either tree). New
  `tests/test_sr_seed_checkpoint.py` sweeps the cross-process resume over every optimizer ×
  momentum dtype × foreach/per-param/fused × CPU/CUDA × `SR_TRITON`, plus two optimizers in
  one process not clobbering each other, `torch.load(map_location=…)` round trips, and a
  two-real-process resume of a nested `SAM(base_optimizer=Lookahead)`.
- **A `map_location`'d checkpoint crashed the first step after a resume** (introduced with
  the fix above, before release). The torch reference path's position is a
  `torch.Generator` state — a **CPU** byte tensor even for a CUDA generator — and the
  standard `torch.load(ckpt, map_location=device)` moves it to CUDA, where
  `Generator.set_state` refuses it: `TypeError: RNG state must be a torch.ByteTensor`.
  Because the restored state is staged and applied on the first draw, it blew up on the
  first `step()` after a load that looked clean. Reached by default on bf16 CUDA `SAM` (its
  climb calls `add_stochastic_` directly), under `SR_TRITON = False`, on a `channels_last`
  weight (the kernel cannot index a strided target), and on any CPU-trained checkpoint
  resumed on GPU. The state is now normalised to CPU and **validated** at load, so a corrupt
  `gen` blob is a checkpoint `ValueError` at `load_state_dict` instead of a `TypeError` from
  inside the optimizer.
- **Nested wrappers shared one stochastic-rounding checkpoint key.**
  `SAM(base_optimizer=Lookahead, …)` is public API and makes three noise owners; the single
  class-level `_sr_wrap_meta` meant the outer wrapper's `state_dict` overwrote the
  intermediate one's position and the resume silently continued from the wrong draw
  (measured 2.34e-2 on bf16 weights). The key is now derived per wrapper from the same
  `state_key` that already namespaces its per-param state (`_sr_wrap_meta_sam`,
  `_sr_wrap_meta_lookahead`), with the bare name kept as a read fallback.
- **`Lookahead.load_state_dict` bypassed the inner Adakaon's own loader.** It handed
  `load_state_dict_preserving_dtypes` straight to the wrapper mixin, so
  `Adakaon.load_state_dict` never ran for the inner optimizer and a resume through
  Lookahead was *not* equivalent to a resume on a bare Adakaon. Three consequences,
  all fixed by delegating to `inner.load_state_dict` (what `SAM`/`MSAM`/`Nekaon`
  already did):
  - **The fused path's stochastic-rounding seed counter was lost.** `_adakaon_meta`
    (`fused_step`) was silently dropped by torch's loader, so `Adakaon._t` restarted at
    `0` on every resume and the bf16 weight-write noise stream desynchronised: measured
    on CUDA (64×64 bf16 params, `fused=True`, bf16 momentum, 4+4 steps) a resumed run
    diverged from an uninterrupted one by up to **4.7e-2** absolute. Now bit-identical.
  - **Adakaon's host-side caches survived the load.** `_invalidate_fused_caches` never
    ran, so the foreach plan and the fused pointer tables stayed cached under an
    `id(group)` the loader had already replaced — one orphaned entry leaked per load,
    aliasing state tensors that no longer belonged to the optimizer, and a reused
    CPython `id(group)` could have handed a live group a stale plan.
  - **The inner's group-key back-fill never ran**, so a checkpoint predating an Adakaon
    group key (e.g. `cautious_wd`) resumed through Lookahead without it and the next
    `step()` died with `KeyError` — the very failure the 0.7.12 back-fill batch fixed
    everywhere else. A pre-0.7.11 checkpoint's lr-scaled momentum was also left
    unmigrated (it is now rescaled to direction units, as on a direct load).

  The dtype-exact resume the old call was there for is unchanged:
  `Adakaon.load_state_dict` delegates to the same
  `load_state_dict_preserving_dtypes`. `_load_wrapped` now documents the contract, and
  new structural tests pin it for `Lookahead`, `SAM` and `MSAM`. Step path untouched:
  86 CPU + 102 CUDA A/B cases (every optimizer × momentum dtype × param dtype ×
  foreach/fused, plus the direct non-wrapper load) hash bit-identically to 0.7.12.

- **ADOPT turned a finite gradient into NaN through the factored second moment.**
  The Adafactor reconstruction divides the row stats by their own mean. That mean is
  zero only when the whole row vector is zero, which needs `eps1 == 0` — and with the
  shipped defaults ADOPT is the only optimizer here that passes `eps1 == 0` (to match
  the official implementation, which adds no eps inside `g ** 2`). An all-zero
  gradient therefore left `row == 0` and the reconstruction computed `0 / 0`; the NaN
  survived the `clamp(max=1/eps)` cap and poisoned the momentum and the weights,
  permanently. Reported for 2-D params whose fan-in is 1 (`(5, 1)`, `(129, 1)`,
  `(1, 1)`, `(R, 1, 1, 1)`), where Gradient Centralization zeroes the gradient by
  construction, but reachable on **any** shape from a genuinely zero gradient (a dead
  branch, a frozen slice, a masked loss term). Both routes were affected (`foreach`
  and per-param).

  `factored_inv_sqrt_factors` grew a `floor` keyword that bounds the row-mean divisor
  from below, and **only ADOPT passes it** (`_MIN_NORMAL`, the smallest normal fp32);
  its inlined copy in `ADOPT._factored_bucket` does the same. The degenerate case then
  saturates at the same `1 / eps` cap the non-factored path already produced for
  `v == 0` (`denom = max(sqrt(0), eps)`) — verified to give the bit-identical step to
  the 1-D path. NaNs arriving *from* the gradient still propagate.

  The floor is deliberately **not** unconditional. `+inf` is only useful to a caller
  that caps the reconstruction afterwards; ADOPT and KProdigy do, while Adakaon,
  AdaPNM, AdaBelief, AdamP, AdaMuon and ScheduleFree do not, and for them a subnormal
  row mean (reachable with a large `beta2` and tiny gradients) is a legitimate finite
  update that a floor would move rather than rescue. With the `floor=0.0` default no
  clamp executes at all, so those six run byte-for-byte the same code as 0.7.12.

  Bit-identity vs 0.7.12: 240 dumped cases per configuration across all 10 optimizers
  (4 momentum dtypes x 3 weight-dtype/bf16 combos x 2 stack budgets, 12 params, 6
  steps, checkpoint round-trip) — **0 differences** on CPU, on CUDA, and on the fused
  big route. In an adversarial subnormal-row-mean dump (`beta2 = 1-1e-8`/`1-1e-10`,
  gradients down to 1e-38 and 0, plus explicit `eps=0.0` where the constructor allows
  it) the only differences are 36 ADOPT tensors, every one of them `base NaN ->
  finite`. Cost on the hot `foreach` path: one extra elementwise op per *shape bucket*
  per step, i.e. +10 aten ops on a 428-tensor LoRA bag; paired A/B 1.006x
  [0.986, 1.027], not significant.

- **`Adakaon.load_state_dict` now back-fills *every* missing param-group key** from
  `self.defaults`, like the other nine optimizers have since 0.7.12. It named
  `cautious_wd` alone — the one key that had bitten — so a checkpoint written before any
  other group key existed (`momentum_4bit_block`, `bf16_method`, `clip_threshold`,
  `gradient_centralization`, `cautious`, `eps`, `momentum_dtype`, `betas`,
  `weight_decay`) came back without it (torch restores `param_groups` from the
  *checkpoint's* dicts) and died with `KeyError` on the first resumed step. Values the
  checkpoint does carry still win (`setdefault`), including per-group overrides that
  differ from the resuming instance's constructor arguments; a resume from a complete
  checkpoint is bit-identical to an uninterrupted run.

- **Lion silently dropped the update of a non-contiguous 2-D weight.** Its pre-plan
  bucket flattened the write-back target with `p.data.reshape(length)`; for a
  non-contiguous tensor `reshape` returns a **copy**, so `_foreach_sub_` wrote the copy
  and the weight never moved — while its momentum kept advancing, so nothing looked
  wrong from the state. Only `ndim > 2` was guarded by a contiguity check, so this
  landed on any weight someone rebound to a transposed view (`p.data = p.data.t()` on a
  square matrix — an in-place transpose keeps `id`, `data_ptr` **and** `shape`). The
  shared plan keeps the identity view for a 2-D bucket and `param_witness` rebuilds the
  plan when contiguity moves, which fixes it. This is the only behaviour difference in
  the Lion migration below.
- **A bucket mixing conv kernels with the same matrixized shape raised.** A factored
  bucket is keyed on the *effective* 2-D shape, so `(16,8,3,3)` and `(16,24,3,1)` (both
  `(16,72)`) share it — but `ForeachChunk.grad_stack`'s fast path stacked the **raw**
  gradients (one `view` of the stack instead of N per-param views), and `torch.stack`
  refuses to mix sizes. Every shared-plan optimizer — Adakaon, AdaMuon, AdaBelief,
  AdamP, ADOPT, ScheduleFree — raised `RuntimeError: stack expects each tensor to be
  equal size` on such a group. The fast path now requires a common raw shape and falls
  back to the per-param views otherwise.
- `KProdigy(second_moment="factored", factor_conv_as_matrix=False)` raised
  `ValueError: too many values to unpack` on any `ndim > 2` weight — in **both** passes,
  so that configuration (with `foreach=True`, the default) could not take a single step.
  It keeps an N-D Adafactor pair (`row` is `[O, I, kh]` for a 4-D kernel), and both the
  pass-2 bucket and the pass-1 second-moment EMA unpacked `R, C = eff` on a rank-4
  tuple. Pass 2 now routes those weights to the per-parameter path, which has always
  handled them; pass 1's reductions already worked on the last two axes for any rank, so
  it only had to stop unpacking. Bit-identical for `factor_conv_as_matrix=True`
  (`reshape(eff)` is `reshape(R, C)` there).

### Changed
- **The three pointer-witness "scan stays in C" locks no longer measure wall time.**
  `test_param_witness_scans_in_c` (both witness copies) and
  `test_plan_witness_scan_stays_c_level` now count the Python bytecode one witness call
  executes and require that count to be flat in the number of parameters — a `map` over
  unbound C methods is flat by construction, while a generator expression, a list
  comprehension *or* an inlined `for` loop each execute ~15 opcodes per parameter. Same
  mutant killed, deterministically: the old wall-clock comparison against a genexpr
  reference failed in 2-3 of 12 runs on a contended machine with no code change. The
  fused case is now also exercised with `ft.SHAPE_WITNESS` **on**, which pins the
  fourth (`strides`) field's scan as well. The timed checks survive as opt-in smoke tests
  (`KAON_PERF_TESTS=1`), out of the default suite.

- **Lion and KProdigy moved onto the shared foreach plan** (`kaon._foreach_plan`),
  which now covers every batched optimizer in kaon. Both reconstructed their buckets and
  every derived view list from scratch on each step and passed `views=None` to the
  momentum codec; they now consume `ForeachChunk` (`pviews`, `state_views`,
  `param_stack()`, `grad_stack()`) and
  `codec.*_stacked(..., views=chunk.momentum_views(codec))`. Lion additionally loses its
  private `_dequant_stacked` / `_store_stacked` / `_dequant_one` / `_store_one` and its
  hand-rolled `m` / `m_scale` / `m_numel` / `m_block` allocation — the codec owns that
  layout and there is now one copy of it.
- `ForeachSpec` gains four additive, default-off bucketing flags — `matrixize`,
  `raw_shape_key`, `scalar_bucket`, `insertion_order` — for the same reason `key_major`
  exists: the *partition* and the *order* of the buckets decide the order the
  stochastic-rounding draws are consumed in, so they are wire-visible in bf16 weights,
  and an optimizer that bucketed differently before the migration has to keep doing so.
  Lion pins the last three (one dict keyed by exact shape, first-appearance order);
  KProdigy declares two specs and picks between them per group via the new
  `_foreach_spec(group)` hook, because its bucketing depends on `second_moment`.
  `ForeachChunk` now accepts an `eff` of arbitrary rank when `matrixize` is off (which
  is how `second_moment="full"` keeps a conv — and a channels_last conv — in the
  weight's own layout), and `_build_foreach_plan` builds one insertion-ordered dict of
  which the default family order is a **stable** sort, so the six already-migrated
  optimizers do not move a bit.
- **Bit-identical to `bugfix/follow-ups-0-7-13` @ f88face** apart from the two Lion/
  KProdigy bugs above: 512 configurations × 2 devices (CPU and CUDA, 5 steps each, every
  weight and every state buffer compared exactly, plus KProdigy's `d` / `d_max` /
  `d_numerator` / `k` / `d_hat` step by step) — fp32/bf16 params × momentum
  `float32`/`bfloat16`/`int8`/`4bit` × `bf16_method` `none`/`stochastic_rounding` ×
  mixed (0-D/(1,)/1-D/2-D/two same-`eff` convs/square) / 448 × 0-D / 428-tensor LoRA /
  conv bags, KProdigy over both `second_moment` modes and a `beta1`/`slice_p`/`decouple`/
  `use_bias_correction`/`d_update_freq`/`cautious` sweep, plus intermittent `p.grad=None`,
  a `p.data` rebind to fresh storage, an in-place transpose of a square weight,
  `load_state_dict` into the SAME optimizer with plans already cached,
  `add_param_group`, a whole-group per-parameter fallback, a stack-budget re-chunk, and
  MSAM / SAM wrapping both. 504/512 identical on each device; the 8 that differ are all
  the `transpose` scenario, i.e. the Lion fix. The harness's own floor was established
  by running the reference tree against itself twice: 512/512.
- **Adakaon's foreach bucketing/view plan is now the shared one** (`kaon._foreach_plan`).
  `ForeachPlanMixin` plus a three-line `ForeachSpec` (`factored_state=("row", "col")`,
  `flat_state=("v",)`) replace
  `kaon.adakaon._ForeachPlan` / `_ForeachChunk` / `_identity` / `_foreach_buckets` /
  `_foreach_plan` / `_param_witness` — **−175 lines** of duplicated machinery. Adakaon is
  the optimizer this design was extracted *from* and the last one still carrying its own
  copy; all six batched optimizers now share one implementation, one staleness witness and
  one test file. **Bit-identical to 0.7.12**: 650 configurations × 3 device/kernel modes
  (1950 runs, 7 steps each, every weight and every state buffer hashed) — fp32/bf16 params
  × momentum `float32`/`bfloat16`/`int8`/`4bit` × plain / `cautious_wd="full"` /
  `betas[0]=0` / `cautious=False` × nine interference scenarios (intermittent
  `p.grad=None`, param-set growth, `p.data` rebind to fresh storage, in-place transpose of
  a square weight, `load_state_dict`, `add_param_group`, a whole-group per-parameter
  fallback step, a stack-budget re-chunk) on CPU, CUDA-native and CUDA-fused. All 1950
  hashes match. The sweep stops short of the chunked-big fused route (weights above the
  lone-big cutoff); that route is bit-identical too, but only under
  `deterministic_reductions=True`, because its atomics are not reproducible run to run
  on either tree (16 extra configurations checked that way).
- `Adakaon._fused_partition` keys on `kaon._foreach_plan.param_witness` instead of its own
  duplicate. The witness now has **two** copies instead of three (the shared one, which
  guards the native path in a build without Triton, and the one inside `_fused_triton`);
  the shape-rebind limit and the witness cost measurements moved onto `_fused_partition`,
  where the fused-specific consequence (pointer arrays carrying the new R/C against the
  old `row`/`col` buffers) actually lives.
- `Adakaon.add_param_group` now drops the cached foreach plans, which is the mixin's
  behaviour and the safer one: `id(group)` is the cache key and CPython reuses the ids of
  dead objects, so a group added after one was dropped could previously land on a stale
  plan. Numerically a no-op (covered by the `add_param_group` bit-identity scenario above).

- AdaBelief and AdamP dropped their private `_dequant_stacked` / `_store_stacked`
  reimplementations of the codec's stacked read/write and call the shared codec
  directly (same values; one fewer copy of the int8 row-scale and 4-bit block
  layouts to keep in step).

### Performance
- **Adakaon's foreach path hands the momentum codec its cached view lists too.** It was
  the last user of the codec's stacked entry points still rebuilding them per step: both
  bucket bodies now pass `views=chunk.momentum_views(codec)` to `ema_stacked`. Measured
  per step (RTX 3000 Ada Laptop, bf16 params + SR, `torch.profiler`, medians of 3
  interleaved repeats):

- **Lion's batched step stops rebuilding its bucketing and every view list on each step.**
  Per step (RTX 3000 Ada Laptop, bf16 params + `stochastic_rounding`, `torch.profiler`),
  base → now:

  | bag | codec | `aten::view` | `aten::reshape` | `aten::copy_` |
  |---|---|---|---|---|
  | 448 × 0-D | int8 | 3147 → **9** | 2693 → **2** | 453 → **5** |
  | 448 × 0-D | bf16 | 1799 → **6** | 1347 → **1** | 4 → 4 |
  | 448 × 0-D | 4-bit | 1364 → **18** | 1355 → **9** | 454 → **6** |
  | 428-tensor LoRA | int8 | 3031 → **29** | 2583 → **6** | 446 → **18** |
  | 428-tensor LoRA | bf16 | 1737 → **20** | 1293 → **3** | 12 → 12 |
  | 300 convs + 128 × 0-D | int8 | 3031 → **31** | 2583 → **6** | 443 → **15** |
  | 24 × (256,256) | int8 | 180 → **10** | 149 → **2** | 29 → **5** |

  `aten::select` (the `stack`/`unbind` residue) does not move and the CUDA launch count
  moves by `+0 … +3` — the one `_foreach_copy_` that replaces N per-parameter `m_scale`
  writes. One counter goes the *other* way and is worth stating plainly: on the quantized
  codecs `aten::as_strided` **rises** by about one per parameter (448 × 0-D int8
  1345 → **1793**, 4-bit 1351 → **1799**; 428-tensor LoRA int8 1571 → **1715**), the
  price of routing the momentum through the codec's cached scale views. It buys
  `view`+`reshape` **5840 → 11** on that same bag, so the net dispatch count is down by
  ~5.4 k ops per step; on the float codecs `as_strided` does not move at all.

  Unprofiled host wall time per step: **−29 … −78 %** (median of 80). That figure is
  supporting evidence only: the same harness with the reference tree as *both* arms
  reported −23 … +22 % on this shared GPU, so the clock resolves ~±25 % here and the
  counters are what rules out a regression. `max_memory_allocated` **never rises** and in
  several configurations falls — measured over 48 (Lion + KProdigy × 4 bags × 4 momentum
  dtypes): 0 higher, 14 lower, 34 identical, the largest being −11 % on Lion's
  24 × (256,256) bag and −4 % on the 428-tensor LoRA one. Resident state is byte-for-byte
  identical either way; what moves is the transient peak, and only downward.
- **KProdigy's share is much smaller** and only its *pass 2* is on the plan; pass 1 (the
  global D reduction and the two `d`-scaled EMAs) keeps its own bucketing, which spans
  param groups and covers parameters pass 2 sends to the per-parameter loop. Pass 1 is
  where most of the view traffic lives, so: 448 × 0-D int8 `aten::view` 3148 → **2706** /
  `reshape` 2245 → **1797**; 300 convs + 128 × 0-D factored int8 3936 → **2614** /
  1551 → **1123**, bf16 2384 → **1489**; 428-tensor LoRA int8 2474 → **2052** /
  1299 → **871**. On bags whose buckets are already in their effective layout (all-2-D
  LoRA, float codecs) it is **+4 … +5** views per step — one `aten::view` per bucket, the
  single reshape of the gradient *stack* that replaces N per-param gradient views. That
  cost is per bucket, not per parameter, so it does not scale; pass 1 is the largest
  remaining opportunity there and is untouched. See `docs/foreach-batching.md`
  ("Lion and KProdigy's migration").
- No measurable change on any path, which is the intended result: the migration moves
  host-side bookkeeping between modules and changes no kernel and no dispatch. Verified
  contention-immune first — the CUDA launch count and the
  `aten::view`/`reshape`/`select`/`as_strided`/`unbind`/`flatten` count for one step are
  **identical** between 0.7.12 and this branch on every bag and both kernel modes (e.g.
  428 LoRA-shaped adapters 194 launches / 4328 view-ops, 448 0-D scalars 77 / 3150,
  300 matrixized convs + 128 scalars 167 / 3935, 128×(512,512)+64×(1024,) 486 / 1790).
- The host-side call the two implementations actually disagree about — one cached-plan
  retrieval — costs **+0.15 … +0.8%** of a 57–96 µs call (paired, order-alternating,
  n=1500 pairs on CPU), i.e. **≤ 0.6 µs per step** on a 4.5–35 ms step. The same harness
  run against the reference tree twice (a null A/B) reports −0.5 … +0.4%, its own bias
  floor.
- Paired GPU wall clock (both trees in one process over *shared* parameter bags, 5
  interleaved repeats × 100 pairs per bag per mode, RTX 3000 Ada Laptop, bf16 params /
  bf16 momentum): every bag and both kernel modes inside ±1.8%, none significant except a
  1.0% *win* on 128×(512,512)+64×(1024,) foreach. Caveat, and the reason the control
  matters: the same harness with the reference tree as BOTH arms reported a "significant"
  +3.1% on one bag, so on a shared laptop GPU this design resolves ~3% and not 2% — the
  identical counters above, not the clock, are what rules out a regression.
  See `docs/foreach-batching.md` ("Adakaon's migration onto the shared module").

  | bag (foreach) | codec | `aten::view` | `aten::copy_` |
  |---|---|---|---|
  | 448 × 0-D | int8 | 905 → **9** | 453 → **5** |
  | 428-tensor LoRA | int8 | 897 → **41** | 448 → **20** |
  | 128×(512,512)+64×(1024,) | int8 | 449 → **65** | 222 → **30** |
  | 24×(320,320,3,3) | int8 | 85 → **37** | 39 → **15** |
  | 448 × 0-D | 4-bit | 6 → 6 | 454 → **6** |
  | 428-tensor LoRA | 4-bit | 24 → 24 | 452 → **24** |
  | 128×(512,512)+64×(1024,) | 4-bit | 36 → 36 | 228 → **36** |
  | 24×(320,320,3,3) | 4-bit | 18 → 18 | 42 → **18** |

  The removed per-parameter `m_scale` write-back becomes 1–6 more `_foreach_copy_` calls,
  and for 4-bit that collapse is the *whole* gain: its `m` is a nibble-packed byte string
  with no effective layout, so it never had per-parameter `view`s to cache (that column
  does not move) but it did write every parameter's scale with its own `copy_`. Only the
  **float codecs (`float32` and `bfloat16`) are count-for-count unchanged** — their lists
  were already served by the plan's identity-keyed `mat` lookup (see below) and the cached
  views replace it at the same cost. Peak allocated memory unchanged (0.00 MiB on every
  bag and both kernel modes) — the lists are views of `state["m"]` / `state["m_scale"]`.
- `ForeachSpec.momentum_cache` and `ForeachChunk.mat` are **gone**. The flag prebuilt an
  identity-keyed `{state["m"]: view(state["m"])}` lookup for the codec's `mat` callback;
  now that every optimizer on the shared plan passes `views=`, the codec's stacked path
  never calls `mat` at all, so the dict was a strictly worse cache (`Tensor.__hash__` is
  a Python-level call in torch) of the very same views. Adakaon was its last user.
  Call sites pass `chunk.view` — the same callback — which the codec keeps only as the
  fallback for a layout `stacked_views` declined (a non-contiguous buffer).
  `ForeachChunk.__init__` and `ForeachPlan.rechunk` lost their now-unused `group`
  argument.
- **The fused big-tensor route stops rebuilding its bucket lists.** `Adakaon._fused_big`
  re-derived the same-shape/dtype/device split of the "big" subset on every step, and a
  fresh list is what made `BigPointerCache` unable to recognize itself (`built_from` is
  list identity), so every step fell back to `stale()` — a full `param_witness` tuple over
  the bucket, once per big shape bucket, *on top of* the sweep `_fused_partition` had
  already run for the whole group. `Adakaon._big_shape_buckets` memoizes the split per
  group and revalidates it by the identity of the route list, which the partition and the
  non-contiguous-grad demotion already guarantee — the same contract `_fused_one_block` and
  `_fused_one_dim` have used since 0.7.12. Witness sweeps **per step**, measured with
  `torch.profiler` and a call counter:

  | bag (fused) | partition | big pointer caches | total |
  |---|---|---|---|
  | 80 big tensors over 4 shapes | 1 | **4 → 0** | **5 → 1** |
  | 200×(256,256) + 100×(512,) + 128×0-D | 1 | **1 → 0** | 2 → 1 |
  | 24×(320,320,3,3) | 1 | **1 → 0** | 2 → 1 |
  | 128×(512,512) + 64×(1024,) | 1 | **1 → 0** | 2 → 1 |

  A sweep costs (measured directly, CPU-only, medians of 200 calls) 4.1 µs over 20
  params, 15.8 µs over 80, 35.6 µs over 200 and 71.8 µs over 428 — so the 80-tensor bag
  above drops ~16 µs of pure host work per step and the LoRA bag's 200-tensor big bucket
  ~36 µs. Peak allocated memory unchanged (0.00 MiB on every bag and mode). **The wall
  clock does not resolve this on a shared laptop GPU** and no claim is made from it. On
  the 80-tensor bag, 200 timed steps × 6 interleaved repeats of base / arm / base: the arm
  lands at −14.7 % (bf16 momentum) and −8.7 % (int8) of the base's host time while the
  base-vs-base *control* lands at +6.4 % and −0.5 % — and the per-repeat spread of that
  control against its paired base repeat is −42 … +83 %. So the direction is plausibly a
  small win and certainly not a regression, but the sweep counts are the evidence.
- `AdaPNM._fused_big` had the identical defect (it *called* `built_from` but always missed,
  because it rebuilt the lists too, and then paid the `stale` compare); it gets the same
  memo, keyed per `(group, lag)` and pruned by `_prune_lag_caches` with the pointer caches.
  Witness sweeps on 80 big tensors over 4 shapes: **5 → 1** per step.
- **`_WitnessedCache.revalidate`** (new, additive) replaces AdaPNM's
  `not cache.built_from(plist) and cache.stale(plist)` on all three of its fused routes.
  That combination had a hole that opened once and never closed: one step with a
  non-contiguous gradient anywhere in the group makes `_fused_demote` rebuild all four
  route lists, so the caches are handed a *fresh list over unchanged parameters* —
  `built_from` misses, `stale` correctly says nothing moved, the branch is skipped, and
  the cache is left holding `src` from the previous generation. From then on `built_from`
  misses on **every** step and the per-bucket `param_witness` sweep is back for good:
  measured **1.0 → 3.17** sweeps per step after a single strided-grad step, permanently.
  `revalidate` is the same two checks plus the rebind they were missing — it adopts the
  fresh list when the witness agrees, so the next step is back on the O(1) identity path,
  and still rebuilds the tables when the witness actually moved. At most one sweep, only
  when identity misses. Adakaon was never affected (its routes use a bare
  `not built_from`, so a fresh list simply rebuilds).
- Bit-identity: 272 configurations (fp32/bf16 params × `float32`/`bfloat16`/`int8`/`4bit`
  momentum × foreach/fused × 5 bags — 428-tensor LoRA, 448 × 0-D, convs+1-D+0-D mixed,
  24×(320,320,3,3), 128×(512,512)+64×(1024,) — × 7 interference scenarios ×
  Adakaon/Nekaon/MSAM/Lookahead), 6 steps each, every weight and every state tensor of
  both the wrapper and the inner optimizer hashed byte-for-byte. **272/272 match** the
  reference tree, which also matched itself run-to-run (the control) with the chunked-big
  fused route under `deterministic_reductions=True`. The same matrix was re-run after the
  `_fused_big` memo: 272/272 again. For AdaPNM (which has no `deterministic_reductions`
  knob, so its fused big reductions are not reproducible run to run on *either* tree) 512
  configurations were hashed: all 352 reproducible ones match bit-for-bit, and the 160 that
  do not are exactly the set the base tree already fails to reproduce against itself —
  agreeing to rel ≤ 1.8e-7 against a base-vs-base spread of 1.3e-7.

### Added

- `kaon._fused_triton.SHAPE_WITNESS` (default `False`) adds per-parameter strides to the
  fused staleness witness, which makes the remaining case detectable: a rebind that
  changes the shape and **nothing else** moves no witness field today, so the plan is
  never rebuilt, never revalidated, and the weight goes on being optimized as the shape
  it used to have. For a plain `view` that variant stays in bounds (the storage is the same
  size), so it degrades quality rather than corrupting memory — which is why it is opt-in
  while the fix above is not.

  It also has a **blind spot**: strides are a proxy for the shape, and a truncation along
  dim 0 of a contiguous tensor does not move them (`(16,64) -> p.data[:8]` keeps `(64,1)`;
  a 1-D `[:256]` keeps `(1,)`). Only `numel` moves there, and `numel` is the *complementary*
  field — cheaper than strides but blind to the `view` case, which preserves numel; only
  `strides + numel` is complete, at the sum of both costs. That blind spot is worse than a
  stale factorization and is pinned by a test: the param keeps being stepped at its
  pre-narrowing extent, so the optimizer writes past the param's current `numel` *inside the
  original storage* (measured: 324 of the 512 elements beyond a narrowed `(16,64)` modified,
  max Δ 1.5e-3), which silently rewrites any sibling view of that storage — a split QKV,
  anything out of `chunk()`/`split()`. Neither the flag nor the build-time check closes it
  (nothing triggers a rebuild, so the check never runs); what does in practice is the
  narrowing also moving the storage, which is the ordinary case and is caught by default.

  It is opt-in because it **cannot be made cheap**: the witness is a host sweep over every
  parameter in the group, run once per group per step plus once per big shape bucket, and a
  shape field is one more sweep. Measured with `benchmarks/fused/bench_shape_witness.py`
  (paired and interleaved, against each bag's minimum step) on **two machines**, both laptop
  GPUs shared with other jobs — the ordering of the candidate fields was stable in every run,
  the absolutes were not, so the spread is given rather than a point estimate:

  | bag | witness calls/step | added host | % of step |
  |---|---|---|---|
  | 428 LoRA-ish (200x(256,256) + 100x(512,) + 128x 0-D) | 2 (428p + 200p) | +61..65 µs | 3.0..3.6% |
  | 448x 0-D (launch-bound) | 1 (448p) | +35..55 µs | 8..18% |
  | UNet/DiT-ish, 80 params | 5 (80p + 4 buckets) | +14..17 µs | 1.1..1.7% |

  `Tensor.stride` is the cheapest field measured that can see the reported bug (+42..65 µs
  on the 428-parameter bag, against +65..104 for `Tensor.size`); `numel` is +22..32 µs.
  `Tensor.dim` is the floor for any per-parameter field at +13..25 µs and cannot see it at
  all — already ~1% of that step and >4% of the 0-D one, where the three existing fields are
  themselves ~20-25% of the step. So no per-step shape witness fits a 1%-of-step budget, and
  the launch-bound 0-D regime is where it hurts most. Turning it on is numerically inert
  (bit-identical to 0.7.12 across the same 32 configurations); flipping it mid-run is safe
  and rebuilds every plan once.

### Known
- **`ScheduleFree` is stochastic by default even over an fp32 model, so two runs in one
  process are not bit-identical without `kaon.reseed_stochastic_rounding()`.** Reported as
  "`kaon.ScheduleFree` is not reproducible against itself": identical seed, fp32
  parameters, fp32 gradients, GC off, same gradients, and the weights part ways at the
  **second** step (measured `5.9e-3` on CPU, `1.2e-2` on CUDA on a `(6,3)`/`(8,4)` bag) —
  on both the foreach and per-parameter paths, and unaffected by
  `torch.use_deterministic_algorithms(True)`, while every other kaon optimizer has a noise
  floor of exactly `0.0` in the same harness. **Not a bug in the step**: bisected to
  `momentum_dtype="bfloat16"` (the default), which stores `z` in bf16 and must therefore
  write it back stochastically or freeze it outright (`schedulefree.py:414` `_store_z`).
  That makes ScheduleFree the only optimizer here that draws SR noise with no
  low-precision *weight* in the model, which in turn exposes it to the per-owner stream's
  first-draw identity allocator (`_stochastic_rounding.py:99` `_next_stream`, the
  "Known limitation / follow-up" above): run 1 takes stream 0, run 2 takes stream 1, and
  the differently-rounded `z` reaches the weights one step later through the
  `ckp1*(y - z)` averaging term. Proven by construction — with `momentum_dtype` in
  `float32`/`int8`/`4bit` there is no draw at all (no stream id, no `_sr_meta`) and the
  runs are bit-identical with **no** reseed; handing both runs the same pinned
  `SRStream(0)` is also bit-identical with no reseed; and under the documented protocol
  (`torch.manual_seed` + `kaon.reseed_stochastic_rounding()`) ScheduleFree is bit-identical
  against itself across all 32 combinations of device x foreach/per-param x fp32/bf16
  parameters x fp32/bf16/int8/4bit momentum. Not fixed here: the two candidate fixes are
  flipping the `momentum_dtype` default to `"float32"` (`schedulefree.py:236` — wire-visible,
  and it doubles `z`'s footprint) or deriving an `SRStream` identity from something stable
  about the owner (already the recorded 0.7.13 follow-up, and it moves the noise of every
  process holding two rounding owners). The behaviour and both escape hatches are now
  documented in `ScheduleFree`'s module docstring ("Reproducibility") and on the
  `momentum_dtype` argument, and pinned by `tests/test_schedulefree_determinism.py` —
  including a strict xfail (`::test_default_config_is_reproducible_under_manual_seed_alone`)
  that flips to XPASS the day either fix lands.
- **A zero `eps` still NaNs the non-capping factored optimizers.** `AdaBelief`,
  `AdamP`, `AdaPNM` and `ScheduleFree` accept `eps=0.0`, and `Adakaon` accepts
  `eps=(0.0, ...)`; that makes their `eps1` zero, so a zero gradient reaches the same
  `0 / 0` in the reconstruction. Only `ADOPT` and `KProdigy` validate `eps > 0`. This
  is unchanged from 0.7.12 (both trees NaN identically) and is not addressed here: the
  `floor` fix above is unsound for these callers because they do not cap the
  reconstruction, so the right fix is either input validation or adding a cap.
- **`ScheduleFree` is not reproducible against itself at its DEFAULT `bfloat16`
  momentum.** Two identical runs (same seed, same grads, GC off, fp32 params) give
  different weights — measured diverging at the second step on a `(6, 1)` bag, 7.8e-3
  apart, and the same on a `(6, 3)` bag, on this branch and on ea46330 alike. It is the
  stochastically-rounded `z` write drawing from a global noise stream that is not reseeded
  between runs; at `momentum_dtype` `float32`, `int8` or `4bit` the repeat noise floor is
  exactly 0. Unrelated to the GC fix (found while building its tests); not diagnosed or
  addressed here.
- **`AdaMuon` leaves some `(1, 1)` parameters untouched**, with GC on and off
  identically — its orthogonalized update on a 1x1 matrix has constant module (`0.2 * lr`)
  and the cautious mask then cancels it exactly over the step sequence, so the weight
  returns to its starting value rather than never moving. Not a defect, and independent of
  GC; noted because `test_fanin_1_trains_with_gc_native` therefore compares which params
  moved against the GC-off control instead of asserting movement outright (an
  `init != final` check cannot tell a frozen param from an exactly-cancelling one).

## [0.7.12]

This release is a full correctness and performance audit of the 0.7.11 fused/Triton
path: memory-safety and compilation fixes across the Adakaon kernels, a uniform
"propagate" policy for non-finite values, a unified in-place momentum codec, and
exact resume for bf16/fp16 parameters, plus measured performance gains across the
fused big-tensor route, stochastic rounding, and several optimizers' foreach paths.

**Behaviour changes** (numeric trajectories or defaults that move as a result of
this audit):
- Default bf16 momentum EMA now accumulates in fp32 (shared codec and KProdigy's
  own EMA).
- ScheduleFree's bf16 `z` buffer is now stochastically rounded on every write.
- Stochastic rounding on CUDA defaults to a Triton kernel (a different, still
  unbiased noise stream than the old torch path; reset with
  `kaon.reseed_stochastic_rounding()`).
- `lr` is applied inside the weight-write kernel instead of a separate multiply
  (sub-ULP reordering).
- AdaBelief, AdamP, AdaPNM and ADOPT bias-correct each parameter on its own step
  count, so late gradients are no longer over-corrected.
- int8 `m_scale` for 1-D parameters is shape `()` instead of `(1,)`.

### Fixed
- **Hot `gradient_centralization` flip corrupted the fused big path.**
  `BigPointerCache` aliases `rowmean` onto `rowsum` when GC is off (it saves `N*R`
  floats and nothing reads it), but a param group is a mutable dict and a scheduler
  can flip the flag mid-run. The cache was keyed only on the parameter witness, so
  nothing moved and it was not rebuilt: `_reduce_rowcol` then wrote the per-row
  means over the row sums and the factored EMA was built from means — measured
  **1.1e-3** relative divergence from the native path, silently. `gc` is now part
  of the cache's validity.
- **Race in the chunked 4-bit requant** (`_chunked_4bit_apply_batched_g`). It stored
  the per-block absmax scales and then `tl.load`ed them back to quantize — the
  storing lane and the reading lanes are different lanes of the same program, with
  no barrier between, so a lane could quantize against the **previous step's** scale.
  Surfaced by the determinism work above: 4-bit momentum stayed nondeterministic
  (5.1e-4 relative spread over identical runs) even with the reductions made
  two-pass. The scale is now kept in registers, which drops 4-bit's default-path
  spread to **5.1e-8** and removes the per-block loop at the same time.
- **Fused Triton path (Adakaon): pointer caches validate the WEIGHT storage.** Every
  fused cache (`PointerArrayCache`, `BigPointerCache`, `OneDimPointerCache`,
  `AdaPnmCache`, `OneDimPnmCache`) and the routing partition keyed on `id(p)` only, so a
  `p.data` rebind (external EMA, `.to()`, block-swap offloaders, FSDP reshard) left the
  kernel writing the retired storage: use-after-free of neighbouring tensors and a
  parameter that silently stopped training. Caches now witness `(id, data_ptr,
  is_contiguous)` per step (measured +41 us/step on a 428-tensor bag; the `shape` field
  was rejected at 2x that cost). A rebind that changes the *shape* of `p.data` is not
  supported (factored state is shape-bound) and is pinned by an `xfail` test.
- **`reduction_tile` returned a non-power-of-2 `BR`** and `tl.arange` refused to compile:
  any bag with >= 2 same-shape tensors of `R in [5, 127]` not a power of 2 (LoRA ranks
  12/24/48/96 over 640-1280 channels, `(9,640)`, `(96,96)`, or one tensor with
  `beta1=0`) crashed on the first step. Rounded up; all consumers already mask rows.
- **`momentum_4bit_block != 128` wrote past `m_scale`** in the one-block kernel (`BLK`
  hard-coded to 128, unmasked scale store). The one-block route now only takes 4-bit
  state whose block is 128 (others go native, bit-identical to the per-param path), and
  `requant_4bit`/`dequant_4bit` mask both the store and the load with the real
  `m_scale` capacity.
- **Non-contiguous gradients were read through `data_ptr` ignoring strides** on all
  three fused routes (the check only existed for `ndim > 2`, and was cached). Grad
  contiguity is now re-checked every step; offending tensors take the native path for
  that step without rebuilding the caches (memoised demotion set, 3.8x cheaper than a
  rebuild-per-step).
- **Triton specialised an integer argument equal to 1** (`C.to(tl.float32)` on a
  `(20000, 1)` weight) into a Python `int` and failed to compile; also reached AdaPNM
  through the shared `_reduce_rowcol`.
- **`reseed_stochastic_rounding()` now resets every SR noise stream.** With the
  Triton bf16 write enabled (the new default) it reset only the torch generators,
  while the kernel's own seed counter kept running — so `torch.manual_seed(s)` +
  `kaon.reseed_stochastic_rounding()`, the documented recipe for re-seeding to the
  *same* value inside one process, stopped reproducing a bf16 run. Measured
  `reproducible: False` on a 5-step Adakaon bf16+SR run with the default, `True`
  with `SR_TRITON=False`. `kaon._stochastic_rounding` now keeps a `_reseed_hooks`
  registry that `kaon._fused_triton` appends to on import (no module-level import
  the other way, so a Triton-less build is unaffected); the kernel reset stays
  internal, and `kaon.reseed_stochastic_rounding()` remains the single public call.
  Affected all ten optimizers' weight writes.
- **Stochastic rounding no longer consumes the global RNG.** Noise comes from a
  per-device `torch.Generator` owned by the module, seeded from the global initial seed
  and re-seeded whenever `torch.manual_seed` changes it, so dataloader / dropout streams
  no longer depend on how many parameters were rounded. Re-seeding to the *same* value
  inside one process is not observable; call `kaon.reseed_stochastic_rounding()` then.
- **Stochastic rounding preserves NaN on CUDA.** The int32 bit-trick added noise to the
  canonical NaN pattern `0x7FFFFFFF`, overflowed the sign bit and wrote `-0.0` instead:
  a diverging run looked healthy while weights were silently zeroed (CPU kept the NaN,
  so no CPU test could see it). NaN now propagates like every other PyTorch op; `+-inf`
  and finite overflow to `inf` behave exactly as before (bit-identical to 0.7.11 with
  the same noise).
- **fp16 weights + `bf16_method="stochastic_rounding"` silently fell back to
  round-to-nearest**; the constructor and `add_param_group` now raise
  `NotImplementedError` (validated before the group is added).
- **Non-finite policy is now uniform: propagate.** `sr_round` no longer turns NaN into
  `-0.0` (int32 overflow) or a low-payload NaN into `+inf`; the seven cautious sites
  multiply by the mask instead of `tl.where`, so an `inf`/NaN gradient produces the same
  non-finite tensor on the fused and native paths (the fused path used to freeze the
  tensor forever in silence).
- **Lion / AdaBelief / AdamP / KProdigy requantize momentum in place.** Their
  `_store_one` / `_store_stacked` (and KProdigy's int8 EMA) reassigned
  `state["m"]` / `state["m_scale"]` on every step. MSAM/Nekaon cache `data_ptr`
  tables into those buffers for the fused climb, so a reassignment left the plan
  reading freed memory (measured climb error ≈ 76% of the bound with
  `MSAM(Lion, momentum_dtype="int8")`). All four now delegate to
  `_MomentumCodec.store_one` / `store_stacked`, which `copy_` into the existing
  tensors — the same contract Adakaon's codecs already followed since 0.7.8.
  Numeric output is bit-identical (same quantizers); only storage identity changes.
  Note: same int8 `m_scale` shape fix as AdaPNM's above (`()`, not `(1,)`,
  where the old foreach path had rewritten it via reassignment); old
  checkpoints with either layout still load.
- **Resume under bf16/fp16 params is byte-identical again.**
  `load_state_dict_preserving_dtypes` used to cast state back to the saved dtype
  *after* torch had already rounded floating buffers through the param dtype
  (~0.3% relative drift per resume on `m_scale`/`row`/`col`/`v`). It now
  re-applies the checkpoint tensors (values + dtype), `copy_` when identity can
  be kept for MSAM's cached pointers, and accepts int/str state keys (JSON drift).
- **`centralize_grads_` groups by `(shape, device, dtype)`.** A param group mixing CPU
  and CUDA tensors crashed in `torch.stack` on the first step (default config).
- **Mixed-device param groups** (CPU + CUDA) crashed in `torch.stack` inside the foreach
  and fused bucketing; `device` is part of every bucket key and kernels launch under the
  bucket's device.
- **Every optimizer's `load_state_dict` now backfills missing `param_groups` keys**
  (the AdaMuon fix above, applied everywhere else). `torch.optim.Optimizer.load_state_dict`
  replaces each `param_groups` dict with the checkpoint's (only `params` carries over),
  so any hyperparameter added since a checkpoint was written vanished from the resumed
  group and the first `step()` died with `KeyError`. Fixed in ADOPT, AdaBelief, AdamP,
  KProdigy, AdaPNM, Lion, ScheduleFree (own `defaults`), and SAM / Lookahead (which did
  not previously carry a `self.defaults` at all — added one for their own per-group keys:
  SAM's `rho`/`adaptive`, Lookahead's `k`/`alpha`/`slow_dtype`/`slow_4bit_block`/
  `la_step`/`train_mode`). Values the checkpoint *does* carry are never clobbered. MSAM
  and Nekaon needed no change: both keep their own hyperparameters (`rho`, `norm`) as
  instance attributes, not per-group keys, and fully delegate to the inner optimizer's
  `load_state_dict`. Adakaon's own backfill is out of scope here (concurrent audit batch).
- **AdaBelief / AdamP bias-correct on a per-parameter step.** Both advanced
  `group["step"]` on every `step()`, including groups with no gradients, so a
  param whose first gradient arrives at step 100 was corrected with `bc1 ≈ 1`
  instead of `1 - beta1` and took a first update ~10x too short. Each param now
  carries `state["step"]` for `bc1`/`bc2` (foreach buckets by it alongside
  shape/dtype, 0-D and shape-(1,) params still riding as batched views) while
  `group["step"]` stays the global clock for schedulers. Trajectories change only
  for params with late gradients; when every param has a gradient from step 1 the
  output is bit-identical to before. Checkpoints without `state["step"]` infer the
  group clock on load.
- **AdamP projects bf16 weights in fp32 on the per-param path.** `_project_one`
  normalized the weight in its storage dtype, so a bf16 weight was projected
  through bf16 norms and dot products while the foreach path (which stacks in
  fp32) was not: the two diverged by 6e-3 relative on the same inputs. Both paths
  now share one fp32 working dtype and write back to the parameter dtype only in
  the final subtract.
- **AdaMuon `compile=True` no longer recompiles every step.** It wrapped the whole
  step body, so Dynamo installed a guard per parameter on *whether that parameter has
  a gradient* and one on the *literal value* of `group["lr"]`. Any LR schedule, or a
  grad set that varies (MoE routing, CFG dropout, partial gradient accumulation),
  therefore burned through `recompile_limit` (8) and fell back to eager **silently**;
  `add_param_group` cost two more recompiles. Measured on a 6-weight bag: 8 compiled
  graphs and multi-second recompiles in both scenarios, `add_param_group` +8.2 s.
  The compiled unit is now the pure-tensor bucket math (`_factored_math`,
  `_nonfactored_pre_math`, `_post_math` and their per-parameter twins), with the grad
  filter, bucketing, momentum codec and state write-back left in eager Python — so
  the graphs are guarded on shapes and dtypes only. Same scenarios now compile **1**
  graph, and `add_param_group` costs 4 ms instead of 8.2 s. Eager output is
  bit-identical to 0.7.11 (verified over 20 configs x foreach on/off, CPU and CUDA).
  Trade-off: at *constant* lr with a fixed grad set the old whole-step graph
  specialized better than the new per-bucket kernels on multi-shape models (12
  distinct small weights 0.27x -> 0.76x eager-relative; U-Net-like 0.41x -> 0.84x).
  **With an LR schedule attached that peak does not exist**: the same two sets measured
  1.03x and 1.00x before the fix (compile did nothing) versus 0.29x and 0.79x after —
  i.e. in the regime real runs are in, `compile=True` goes from a no-op to a 1.3-3.4x
  step speedup. Single-slice buckets now stack with a zero-copy `unsqueeze` instead of
  copying.
- **AdaMuon resumes from a pre-0.7.12 checkpoint.** `torch.optim.Optimizer.load_state_dict`
  *replaces* each `param_groups` dict with the checkpoint's (only `params` is carried
  over), so a checkpoint written before `bias_correction` existed left the live group
  without that key and the next `step()` died with `KeyError: 'bias_correction'`.
  `load_state_dict` now backfills any key the checkpoint predates from `self.defaults`
  (values the checkpoint *does* carry still win, so a resumed run keeps its tuning), and
  both read sites use `group.get(...)`. Verified bit-identical resume against real 0.7.11
  checkpoints across bf16/int8/4bit/fp32 momentum with and without weight decay. Any
  future hyperparameter is covered by the same backfill.
- **AdaMuon `step()` no longer raises on a gradient-less param group.** A group where
  nothing has a gradient — MoE routing leaving an expert unrouted for a step, CFG
  dropout, partial gradient accumulation, or a bare `step()` with nothing backwarded —
  reached the `foreach` path's `params[0].device` probe with an empty list and raised
  `IndexError`. Such groups are now skipped and their parameters left untouched.
- **AdaMuon `clip_threshold` documentation was wrong about when it fires.** The
  docstring called it "a near no-op in steady state". The factored second moment has
  no bias correction, so on a real proxy-U-Net run (β₂=0.999) the mean `rms(u)`
  *before* the clip measures 31.9 at step 1, 11.0 at step 10, 3.56 at 100, 1.20 at
  1000 and 0.98 at 2999 — i.e. almost exactly `1/√(1-β₂ᵗ)`. The clip is active on 100 %
  of weight buckets for the first `~1/(1-β₂)` iterations and 70-80 % after, and is what
  sets the early effective step size; it is in effect the second moment's bias
  correction. Documented as a first-order hyperparameter, not a safety net.
- **AdaMuon `ns_steps=2` rationale corrected.** The docstring credited the default to
  "5 over-orthogonalizes". Measured singular-value spectra say otherwise: the quintic
  settles into a band ≈[0.67, 1.20] and never leaves it, `ns=2` is already inside it
  on skinny matrices (mean sv 0.89-1.06 at 4:1-16:1 — LoRA shapes and matrixized
  convs) but heavily *under*-orthogonalized on square ones (mean sv 0.56 at 256²,
  0.31 at 1024², with near-zero directions). The default stands as an empirical sweep
  result on a skinny-weight model; `ns_steps` should be re-swept on models with large
  square weights. Table in docs/adamuon.md.
- **AdaPNM handles empty groups and late gradients consistently.** Parameter-local
  steps retain the correct bias correction and global PNM parity, while the shared
  momentum codec preserves state-buffer identities required by cached pointers.
- **AdaPNM int8 1-D scales now use scalar shape `()` instead of `(1,)`.** The value
  is unchanged; checkpoints have been verified across foreach and per-parameter
  paths in both directions.
- **Fused Triton path (AdaPNM): the same five defects, now closed on AdaPNM's own routes.**
  The batch-A fixes above landed on Adakaon and on the shared caches, but AdaPNM's launchers
  had not been carried over: `_fused_partition` and the `AdaPnmCache` / `OneDimPnmCache`
  callers still keyed on `id(p)` alone, so a `p.data` rebind kept the one-block and 1-D
  kernels writing the *retired* storage (measured: every parameter in the bag scribbled,
  ~1e-2 divergence from native). The routing key is now
  `_fused_triton.param_witness` (`id`, `data_ptr`, `is_contiguous`) and each pointer cache
  is revalidated per step (`built_from` when the lag bucketing hands back the partition's own
  list, `stale` otherwise). Also fixed on AdaPNM: `_adapnm_tile_kernel` passed `NS = NB` to
  `requant_4bit`/`dequant_4bit` with a hard-coded 128-element block, so any other
  `momentum_4bit_block` wrote past *both* momenta's scale buffers (measured with a canary:
  32 fp32 past a 32-entry `m_pos_scale`/`m_neg_scale` on a `(64,128)` weight at block 256) —
  the one-block route now requires block 128 and the kernel receives each tensor's real scale
  capacity; grad contiguity is re-checked every step (it was only checked for `ndim > 2`, and
  cached, so transposed/strided grads reached all three routes at ~1e-2 from native); the five
  AdaPNM cautious sites multiply by the survivor mask instead of `tl.where`, so a non-finite
  update propagates as it does natively instead of freezing the tensor; and `device` joined the
  native foreach bucket keys and the big-route shape buckets (a CPU + CUDA group took the whole
  native step down in `torch.stack`).
  Cost on a 428-parameter bag: witness key 62 us/step (vs 19 us for the old ids-only key),
  grad-contiguity sweep 68 us/step, cache revalidation 0.3 us/step via `built_from` (a naive
  per-bucket `stale` would have cost 70 us). Verified bit-identical to the pre-fix build across
  {fp32, bf16} params x {fp32, bf16, int8, 4bit} momentum x cautious on/off on every
  atomic-free route (0/976 state tensors differ); the big route's `tl.atomic_add` on `colsum`
  is not run-to-run reproducible in either build (49/512 tensors differ base-vs-base too).
- **ADOPT** — per-parameter ``state["step"]`` governs the step-0 ``v`` init and the
  ``step**0.25`` clip so a param whose first gradient arrives late (or a ``.step()``
  with no grads) no longer leaves factored ``row``/``col`` at zero and diverges to
  NaN; ``momentum_dtype`` is resolved per param group (not silently from the
  constructor kwarg). Checkpoints without per-param ``step`` seed the counter from
  ``max(1, group["step"] - 1)`` when ``v``/``row`` already exist.
- **KProdigy's bf16 momentum EMA runs in fp32.** The EMA was computed in bf16
  (three roundings: gradient cast, product, sum) and was not invariant to
  batching — CPU bf16 elementwise kernels round the vectorized body and the
  scalar tail differently, so a `numel % 16 != 0` tensor picked up an ulp of
  momentum error when it was folded into a stack (1.07e-4 relative on D, 3.4e-2
  on the weights, with shapes (3,7)/(5,11)/(127,)). Per-param and foreach now
  both widen to fp32 and round once on write, aligned with this release's change
  to the shared bf16 codec. bf16 momentum runs differ from previous releases
  here; the change is a fidelity improvement and the two paths are again
  bit-identical.
- **Lion `foreach` no longer silently skips `channels_last` convs.** The batched
  write flattened via `reshape`, which copies a non-contiguous weight and drops
  the update (max|Δw| = 0 over many steps). `ndim > 2` params now require
  contiguity for the foreach path (same gate as Adakaon) and fall back to
  per-param otherwise.
- **ScheduleFree stochastically rounds bf16 `z` on every write.** This is independent
  of `bf16_method` and consumes RNG even when model weights are fp32, so sub-ULP
  `z` updates remain unbiased without a Kahan/shift buffer.
- **Lookahead** — ``_sync_foreach`` chunks stacked slow-weight syncs with
  ``foreach_budget`` instead of materializing an unbounded ``[N, *shape]`` transient.
- **MSAM fused-plan witnesses cover `m` / `m_scale`.** `_plan_addrs_valid` now
  re-reads `data_ptr` from the live state dicts (and every weight in the bucket),
  so a base that still reassigns momentum invalidates the plan instead of climbing
  on dangling pointers. `MSAM(AdaPNM, …)` with `rho != 0` raises `TypeError`
  (dual `m_pos`/`m_neg`, no single `m`); `rho=0` remains a passthrough. The
  inert-lookahead threshold is mode-aware (`none`: per-coordinate `|rho|*lr*clip`;
  `global`/`tensor`: `|rho|` vs weight L2 norms) so `norm="global", rho=0.3,
  lr=1e-6` no longer spuriously suggests raising lr.
- **SAM** — ``second_step`` restores every ``old_p`` snapshot even when the second
  backward left ``p.grad`` as ``None``; global grad norm and the climb are batched
  (``torch._foreach_norm``, stacked chunks bounded by ``foreach_budget``). The
  batched norm accumulates in fp32 so ``scale`` is slightly more accurate on bf16
  weights than the old per-param bf16 reduction.
- **ktune** — OOM timings format as ``OOM`` instead of raising ``TypeError``; when
  CUDA+Triton are available, sweeps ``fused=True`` over ``fused_tile_cap`` and
  reports the best tile.

### Changed
- **Default bf16 momentum EMA now runs in fp32** (then `copy_` into the bf16
  buffer), matching the fused Triton kernels. Previously `_FloatCodec` did
  `m.lerp_(update.to(bf16), …)`, rounding the update *before* the EMA — the only
  `momentum_dtype` where native and fused diverged (~3.5e-4 rel vs ~1e-7
  elsewhere). **Numeric trajectory of the default `momentum_dtype="bfloat16"`
  changes** (fidelity improvement; fp32/int8/4bit unchanged). Foreach vs
  per-param parity for bf16 is now **1 fp32 ULP** (`rtol=1e-6`, `atol=1e-9`)
  rather than bit-exact: a 1-ULP difference in the stacked vs per-param update
  (distinct reduction order) used to be hidden by rounding that update to bf16
  *before* the EMA. fp32 momentum remains bit-exact.
- **Lion warns on `momentum_dtype="4bit"`** (still accepted for checkpoint
  compat): measured ~12–13% sign flips; loss ~32× / ~3916× worse vs bf16.
  Prefer `int8`. Docs/docstring updated (old "+0.005 loss" was wrong).
- **`warn_if_4bit_high_beta1`** in `_momentum_codec` (wired from Lion in this
  lot): 4bit + `beta1 >= 0.99` amplifies quant error ~`1/sqrt(1-beta1^2)`
  (measured block-128 table in `docs/momentum.md`).
- **`warn_if_4bit_high_beta1` wired into every other β1-EMA momentum consumer**:
  AdaBelief, AdamP, ADOPT, AdaPNM (checked against `betas[0]`, not the unrelated
  `beta0` negative-momentum mix), KProdigy, AdaMuon and **Adakaon** (the last one
  landed with this batch, once its file was free). Deliberately **not** wired into ScheduleFree:
  its quantized `z` buffer is a plain accumulator (`z -= lr_t * d`), not decayed
  by `beta1`, so the warning's AR(1)-amplification argument does not apply to it
  (see `docs/momentum.md`).

### Performance
- **`momentum_4bit_block` is a runtime kernel scalar in Adakaon's one-block tile
  kernel** (`_adakaon_tile_kernel`), and `PointerArrayCache` buckets by
  `state["m_block"]` alongside tile/dtype/device. The kernel used to hardcode
  `BLK = min(R*C, 128)`, so a routing guard diverted **every other block size to the
  native path** — correct, but it gave up the fused route for a legitimate
  configuration. Measured cost of that diversion (RTX 3000 Ada, paired geometric
  mean, 95% CI, 300 one-block tensors): **16.2–20.3x** slower, **427/436 kernel
  launches → 2**, and **39.5 MiB → 0** of per-step transient, across
  `block ∈ {64, 256, 0}` × {fp32, bf16}. The new route's own worst case — a padded
  tile (no `EXACT` single-reduction, so `requant_4bit`'s general loop is O(numel·NB))
  at `block=8`, 900 blocks over a 64×128 tile — is still **1.7–2.5x** faster than going
  native, so no residual block-size guard is warranted. `EXACT`/`FBLK` now take the
  bucket's real block, which **must divide the tile** — a precondition that used to be
  automatic and is now enforced by `PointerArrayCache.exact4` (without it a `(64,128)`
  weight at `momentum_4bit_block=96` raises a Triton `CompilationError` mid-run); the
  `NS` capacity masks stay as the memory-safety backstop. **No added JIT cost at the
  defaults**: the whole-surface baseline (every fused route × every momentum storage ×
  fp32/bf16 params, clean `TRITON_CACHE_DIR`) is **45 variants across 11 kernels before
  and after**, cold-cache first-step compile 38.6/37.9 s (before) vs 37.2/37.8 s (after).
  Per-block specialization is confined to the `EXACT` path's `FBLK` constexpr (+1 variant
  per distinct block on an unpadded tile; padded tiles share one variant for all blocks,
  which is the runtime argument doing its job). New A/B + variant census:
  `benchmarks/fused/bench_wd_mblock.py` (`--case mblock`, `mblock_worst`, `wd`, `jit`).
- **A lone big 2-D weight takes the BATCHED chunked kernel** (`_fused_big`, `N == 1`).
  The per-tensor `_chunked_step` it replaces blocked the CPU twice per tensor per
  step (`float(rms)`, `keep.item()`) and materialized fp32 `g` + `g*g`. Measured
  (RTX 3000 Ada, paired geometric mean, 95% CI): a bag of 12 DISTINCT big shapes —
  the UNet/DiT case, one tensor per shape bucket — **3.38x [3.01,3.80] fp32 /
  3.26x [3.04,3.50] bf16**, 337→108 launches, and the step no longer synchronizes
  *for its own reductions* (see the caveat below). Single tensors 1.21–4.75x; peak
  transient on `(4096,4096)` **156→0 MiB** (272→0 with 4-bit). **No fidelity cost
  for any momentum dtype**: over six repeats, both the per-tensor and the batched
  arm sit at 4.64e-8 relative to native on a (1024,512) 4-bit weight, the same
  fp32-reduction-order noise the float momenta carry. (A first measurement put
  4-bit at ~4e-4 here; that was the scale store/reload race fixed under *Fixed*
  below, not a property of the batched kernel. The `test_chunked_4bit_*` bounds are
  **1e-6**, tightened from the pre-existing 5e-4.)
- **Direct in-kernel int8 momentum for the batched big path**
  (`_chunked_int8_{keep,apply}_batched_g`), the mirror of the 4-bit pair. Replaces
  `dequant_stacked` → generic kernels → `_quant_int8_stacked`: **120→6 launches**,
  peak transient **58.5→0 MiB** (100x(256,256)), **2.77–5.81x**, and it drops the
  `ptr_array` host→device copy the codec path made every step. Conditional on
  `C <= 1024 and 1024 % C == 0` so a chunk owns whole rows (int8 scales per row);
  other shapes keep the codec fallback.
- *Caveat on the "sync-free" claims above.* What the two changes remove is the
  optimizer's OWN synchronizations — `float(rms)` and `keep.item()` per tensor per
  step on the lone-big path, and the codec's staging copy on the int8 path. One
  synchronizing operation remains and is **not** addressed here: the grad
  pointer-array refresh (`refresh_grads`) does a pageable host→device copy whenever
  a gradient is reallocated, which with the default `zero_grad(set_to_none=True)`
  is every step. Verified: with grads reused across steps a big-bucket step probes
  clean under `set_sync_debug_mode("error")`; with grads reallocated it does not.
  A pinned staging buffer would make that copy async and was measured at
  **4.4–4.8 µs per bucket per refresh, flat in N** — not taken, because it needs a
  CUDA event (or a buffer ring) to stop the next step overwriting a copy still in
  flight, and 4.4 µs/bucket is ~30x smaller than what the launch packing below
  removes.
- **Nine→six launches per big bucket per step.** `colsum`/`rms`/`keep` are adjacent
  slices of one buffer (one `zero_()` instead of three) and the `grid=1`
  `_finish_rms` is folded into every consumer (`inv_rms_clip`). Isolated cost of
  the four removed launches, paired over 500 pairs: **45.6 ± 13.2 µs** at one
  bucket, **1.76 ms ± 0.04** at ten and **5.01 ms ± 0.29** at forty — a FIXED
  per-bucket cost, which is what a many-bucket step pays over and over. Reduction
  scratch also drops: `rfac`/`cfac` are written in
  place over `rowsum`/`colsum` and `rowmean` is allocated only under GC — 0.98→0.59
  MiB for 100x(512,512), 9.78→5.87 MiB for 1000x(512,512).
- **4-bit per-block absmax is one axis reduction** on unpadded tiles
  (`requant_4bit(EXACT=...)`, bucketed by `PointerArrayCache`). The general path
  loops `NB` times over the whole tile — O(numel·NB) — which measured 4-bit
  momentum 1.78x/1.88x/3.15x slower than bf16 on `(64,128)`/`(128,64)`/`(16,512)`
  (NB=64) but only 1.07x at NB=16. **1.18–1.65x** on the requant (the ratio grows
  with NB; n.s. at NB=16), taking 4-bit vs bf16 to **0.98x/0.96x/1.16x**.
  Bit-identical — verified on weights, packed codes AND scales (`max` is
  order-independent).
- **Triton bf16 stochastic-rounding write** (`sr_add_`, used by `subtract_one_` /
  `subtract_batched_` on CUDA). Replaces `add_stochastic_`'s 7–8-kernel chain and
  its two parameter-sized temporaries with one kernel and none: **3.43x** (1 M),
  **4.76x** (4 M), **6.39x** (13 M), 9→1 launches, transient 13/52/162→**0 MiB**.
  `subtract_batched_` on 200x(256,256) bf16: **4.64x**, 12→4 launches, peak
  188.5→**26.0 MiB**. End to end on a mixed bag (200x(256,256) + 100x(512,) + 128
  scalars, native path, bf16): **1.195x [1.143,1.250]**. Different (also unbiased)
  noise stream than the torch path, and it applies to **every** optimizer's bf16
  weight write, not only Adakaon's; `kaon.reseed_stochastic_rounding()` resets it
  along with the torch generators, and `kaon._backend.SR_TRITON = False` pins the
  reference implementation.
- **`lr` rides the weight write** (`subtract_one_`/`subtract_batched_` take `alpha`)
  instead of a separate `delta.mul_(lr)` pass over the stacked bucket. Applied to
  the per-parameter path AND both foreach buckets, because `Tensor.sub_(d, alpha=lr)`
  and `torch._foreach_sub_` with `alpha` are bit-identical — folding it in only one
  would break `foreach == per-param`. Quiet-GPU paired A/B: **+4.4% [+0.7, +8.1]**
  on 200x(256,256), **+3.7% [+0.9, +6.6]** on 50x(512,512). Sub-ulp vs the old
  order (`mul_` then subtract can round differently from a contracted `a - lr*b`);
  bit-identical on bf16 params.
- **Shared foreach bucketing + cached view plan (`kaon._foreach_plan`).** AdaMuon,
  AdaBelief, AdamP, ADOPT and ScheduleFree each carried their own copy of the same
  bucketing (effective shape / dtype / matrixize / device, plus a per-optimizer key
  such as the per-parameter step) + budget chunking, and every `*_bucket` body
  rebuilt its derived view lists (`[mat(p.data) …]`, `[flat_view(state[k]) …]`, the
  codec's `mat` callback) **on every step**. All five now share one
  `ForeachPlanMixin`, which caches the bucketing and those views per param group.
  Bit-identical in every path (per-param and foreach, fp32/bf16 params, all four
  `momentum_dtype`s, 0-D/(1,)/1-D/2-D/conv, checkpoint round trip), verified against
  the pre-refactor tree.
  Measured on an RTX 3000 Ada (bf16 params, 3 interleaved rounds × 40 reps/arm,
  `min` of the per-step wall time; `aten::view`+`reshape` counted with
  `torch.profiler`):

  | bag | Δ view/reshape per step | Δ CPU self | Δ ms/step (min) | Δ ms/step (median) |
  |---|---|---|---|---|
  | 448 × 0-D scalars | −27 … −50 % | −13 … −46 % | **−32 … −50 %** | −36 … −55 % |
  | 300 × conv (16,8,3,3) + 128 × 0-D | −25 … −49 % | −7 … −33 % | **−21 … −38 %** | −27 … −46 % |
  | 200 × (256,256) + 100 × (512,) + 128 × 0-D | −10 … −23 % | −1 … −19 % | −0 … −9 % | −0 … −7 % |

  The win scales with how many bucket entries need a *real* view: it is large on
  launch-bound bags (0-D scalars, matrixized convs) and inside noise on a bag whose
  params are mostly already in their effective layout (a `(256,256)` weight's `mat`
  is the identity, so there was never a view to cache) — that bag is GPU-bound.
  Caching pins no memory (every cached tensor is a view of a param or a state buffer
  the optimizer already owns) and gradients are deliberately **never** cached: a
  retained `p.grad` view would keep the previous step's gradient set alive.
  `optimizer._foreach_cache_enabled = False` is the A/B switch (numerically a no-op).
  A stale plan is impossible: it is rebuilt on a param-set change, a `p.data` rebind
  (fresh storage *or* a transpose that only moves strides — `(ids, data_ptrs,
  contiguity)` witness), a per-parameter-clock partition change, and a stack-budget
  re-chunk, and dropped outright by `load_state_dict`, `add_param_group`, an AutoLR
  base-state reset and any group that falls back to the per-parameter loop.
  Adakaon keeps its own equivalent plan for now (it is coupled to the fused Triton
  caches); merging it into the shared one is follow-up work.
- **int8 `ema_one`**: `.float().mul_(scale)` (one temp), drop `delta.clone()`,
  write codes with `(m/scale).round_().clamp_` into `state["m"]` — measured
  13→8 B/elem transient, 14→12 kernels, ~1.29×, bit-identical. Same clone
  removal on int8/4bit `ema_stacked` / 4bit `ema_one` (the strided 4bit dequant
  slice is kept as-is: materialising it changed the `lerp_` kernel and broke the
  per-param/stacked bit-exactness).
- `add_stochastic_` writes back with `copy_(fp32)` (no bf16 temporary) and draws int32
  noise directly (4 B/elem transient instead of 10 B/elem).
- `subtract_batched_` casts the stacked delta once per bucket instead of once per
  parameter when the stack dtype differs from the params (`aten::_to_copy` no longer
  scales with bucket size; 5-15x on the cast-heavy bags).
- **`eps1` on the row/col means instead of the `[N,R,C]` square** in the native
  factored foreach bucket (`mean(x+eps) == mean(x)+eps`, so a full read-modify-write
  pass over the bucket disappears). **Bit-identical** in every tested config,
  including large `eps1` and 1e-10 gradients. Quiet-GPU paired A/B (geometric mean
  of 150–300 per-rep ratios, 95% CI): +1.4% [-1.1, +3.9] on 200x(256,256), **+2.6%
  [+0.5, +4.8]** on 50x(512,512), ~0 on 400x(64,64).
- **AdamP's batched projection removes the radial temporaries.** The stacked path
  built six `[N, R, C]` tensors (normalized weights, radial components and two
  `torch.where` results) even though at most one branch fires per slice. It now
  reduces to one broadcast coefficient per row/tensor applied with `addcmul_`:
  31.2 -> 20.4 ms isolated (-35%) and 496 -> 369 MB of step peak (-26%). The fp32
  reassociation makes it differ from the normalized-vector form by ~5e-8
  relative. The per-param path also dropped its two host synchronizations per
  param per step (`if cos.max() < ...`) for an on-device mask.
- **AdaPNM's big (chunked, batched) route caches its pointer arrays** in a new
  `BigPnmCache`, the AdaPNM counterpart of `BigPointerCache`: grad / weight / both momenta
  address arrays plus the `rowmean`/`rowsum`/`colsum`/`keep`/`rms` scratch were rebuilt from
  fresh `torch.tensor([...])` allocations on *every* step, while the one-block and 1-D routes
  have cached theirs since 0.7.9. Measured on an 8x `(512,512)` bucket, steady state:
  caching-allocator allocations 22 -> 13-14 per step and the pointer arrays' host-to-device
  copies 4 -> 0-1 per step (two independent measurements). CUDA **kernel** launches are
  unchanged — the win is host-side allocation and H2D traffic, not launch count. Same math,
  same state.
- **AdaPNM fused caches are keyed by `(group, lag)`.** Stable late-gradient buckets
  reuse their pointer caches instead of rebuilding them every step, avoiding the
  measured 3.9 -> 29 ms/step regression and reducing 400 reconstructions to a
  stable cache set; inactive lags are pruned to keep memory bounded.
- **KProdigy folds the D statistics on the host in one transfer.** Pass 1 summed
  2N zero-dim GPU tensors in a Python loop and then called `.item()` twice; the
  partials are now stacked, copied once, and folded sequentially in
  `numpy.float32` in the same order (2253 -> 128 kernel launches and 2 -> 0
  synchronizations per step on a 428-tensor bag; the fold itself 20.4 -> 0.32 ms).
  Steps that do not update D skip the transfer entirely. The bf16 momentum EMA is
  batched per bucket (20.44 -> 5.75 ms) and the dead `_flat_full_bucket` wrapper
  is gone. The per-param/foreach *D trajectory* is no longer bit-identical — the
  batched `[B, L]` reduction is a different tree, ~3.3e-7 relative at
  `slice_p=11` on CPU — and the tests now pin it to 1e-6 relative.
- **ScheduleFree's bf16 stochastic-rounding writes go through the shared
  `_backend._sr_write_`.** Three of them did not: the `z` write-back (per-param and
  batched) and the per-parameter `y` write-back called
  `_stochastic_rounding.add_stochastic_` directly, so while every other optimizer's bf16
  write took the one-launch Triton kernel, ScheduleFree stayed on the torch reference —
  ~7 extra kernels and two parameter-sized temporaries (an fp32 upcast and an int32 noise
  draw) per call. The `y` case bit hardest on big models: any weight above
  `foreach_batch_cutoff` falls to the per-parameter path *even under `foreach=True`*, so a
  UNet/DiT bag paid it on exactly its largest tensors, at every `momentum_dtype`. Two
  smaller wastes went with it — the increment `z_new - z_stored` is now one promoting
  kernel instead of `.float().neg_().add_()` (same single temporary, three passes into
  one, bit-identical), and a one-parameter bucket rounds straight into its own `z`
  instead of `torch.stack`-ing a copy and copying it back. Paired geometric mean, 80
  interleaved reps, 95% CI, RTX 3000 Ada, against the pre-fix tree — UNet/DiT bag (8
  distinct big shapes): `z=bf16` **1.512x [1.488, 1.537]** foreach / **1.584x
  [1.574, 1.594]** per-param, `z=float32` **1.226x [1.205, 1.247]**, `z=int8` **1.197x
  [1.170, 1.224]**; 448 → 328 launches and step peak **520.4 → 337.6 MiB** (`z=bf16`),
  464.1 → 309.4 MiB (`z=float32`). LoRA-shaped bag (200x(256,256) + 100x(512,) + 128
  scalars), `z=bf16`: **1.261x [1.240, 1.283]**, 297.9 → 198.7 MiB. All six
  configurations now step **1.05–1.22x faster than 0.7.11**, which did not stochastically
  round `z` at all. Semantics unchanged: the rounding is still unbiased and still
  governed by `kaon.reseed_stochastic_rounding()`; on CPU, fp16 or a strided target
  `_sr_write_` falls back to the same `add_stochastic_` as before.
- **MSAM's fused-plan pointer witness scans in C.** `_plan_addrs_valid` re-reads
  `data_ptr` for every weight and every `m` / `m_scale` in the plan, twice per step (the
  removal and the climb each launch a kernel off the cached device pointer tables), so on
  a small-tensor bag it is pure Python call overhead in the hot path: 300x(16,8,3,3) +
  128 scalars measured **114 µs per validation** at `momentum_dtype="4bit"`. The three
  generator expressions — one of them with a dict subscript per parameter — are now
  `tuple(map(...))` over module-level accessors (`Tensor.data_ptr`,
  `operator.itemgetter`), so the per-parameter loop runs inside `map`/`tuple`:
  **114.4 → 95.2 µs** (4bit) and **89.0 → 61.5 µs** (bf16) per validation on that bag,
  **134.7 → 113.8** / **113.4 → 71.1 µs** on 200x(256,256) + 100x(512,) + 128 scalars.
  Step-level, paired, against the pre-fix tree: Nekaon **1.071x [1.029, 1.114]** (bf16)
  and 1.025x [0.966, 1.088] (4bit) on the conv bag, 1.022x [0.987, 1.058] on the LoRA
  bag. The witness is deliberately **not** thinned further: it still costs 0.12–0.23 ms
  per step on a 428-parameter bag, which is why that bag runs at 0.69–0.88x of 0.7.11,
  where the witness was one pointer per bucket (1.7–2.1 µs) and missed exactly the
  reassigned-momentum and rebound-weight cases the *Fixed* entry above closes. Dropping
  either of the two validations per step would leave a window in which the very next
  fused launch reads or writes freed CUDA memory.
- **ScheduleFree's one-parameter foreach buckets alias instead of stacking**
  (`ForeachSpec(single_alias=True)`, the contract AdaMuon has had since 0.7.11). Every
  bucket of a big-unique-shape model is `N == 1`, and `grad_stack()` / `param_stack()`
  were calling `torch.stack` on a single tensor — a full same-dtype copy — before the
  widening `.float()`. They now `unsqueeze` and widen straight from the parameter's
  storage. Safe because both stacks are read-only in `_factored_bucket` /
  `_nonfactored_bucket` (the y-update builds its own mutable stack) and because
  `_param_foreach_eligible` already rejects an `ndim > 2` param whose data or grad is
  non-contiguous, which is the only case where the matrixizing `view` on the unsqueezed
  tensor would not be expressible. **Bit-identical**, verified over 162 configurations
  (5 shape mixes x {fp32, bf16} params x 4 `momentum_dtype` x `inner_momentum`
  {0, 0.9} x `bf16_method` {none, stochastic_rounding}, plus strided 2-D grads).
  Deterministically **8 fewer kernel launches per step** on the UNet/DiT bag (316 → 308
  at `z=float32`); the wall-clock effect is small and mostly inside noise — paired
  in-process A/B, 80 interleaved reps, 95% CI: 1.025x [1.001, 1.049] at `z=int8`,
  1.008x [0.997, 1.019] at `z=bf16`, 1.004x [0.990, 1.019] at `z=float32`, and step peak
  unchanged (the elided copies are transient and the allocator was reusing the blocks).

### Added
- **`Adakaon(cautious_wd="masked" | "full")`** — where decoupled `weight_decay`
  sits relative to the cautious mask. Adakaon has always folded it into the delta
  *before* the mask, on every path, so the decay went through the mask too:
  measured as the fraction of the requested `lr*wd*p` each coordinate actually
  receives, **1.49x on survivors / 0.005x on rejected at keep=0.64** (1.985x /
  0.0013x at keep=0.50) — the aggregate shrinkage is preserved, its per-coordinate
  distribution is not. `"full"` is the Cautious Optimizers paper's own placement:
  the mask applies to the momentum/update term only and `lr*wd*p` reaches **every**
  coordinate. Implemented on all three native paths and all ~13 Triton kernels
  (per-param == foreach exactly; fused within the path's own noise, in both modes).
  **`"masked"` stays the default** — the A/B did not promote it: on the diffusion
  proxy (C=128 U-Net, 2000 steps, REX + progressive curriculum, `wd` ∈ {0.01, 0.05}
  × lr ×{0.5,1,2} × 3 seeds, arms interleaved), `full − masked` over the 18 paired
  runs is `+0.00036 [-0.00009,+0.00080]` on held-out loss and
  `+0.00032 [-0.00013,+0.00078]` on the train–val gap — not resolvable — but at the
  **tuned lr (×2.0 at both `wd`)** `full` loses 2/3 seeds on both metrics. Full table
  in `docs/adakaon.md`. Note ×2.0 is the **top of the swept lr grid**, so the optimum is
  not bracketed from above; a wider sweep could move the tuned point. `"masked"` is
  **bit-identical to pre-0.7.12 Adakaon** (verified over 24 configurations: 4
  `momentum_dtype` × {fp32, bf16} params × {per-param, foreach, fused}) and the new
  `WDFULL` constexpr adds **no** compiled Triton variant at the default (whole-surface
  baseline 45 before and after; `"full"` adds 5). In the fused path `"full"` is ~9%
  *faster* per step (the keep-count kernels stop reading the weights). Guarded by a
  parity check against the **per-parameter** loop at ~10x each dtype's measured floor
  **and** by a semantic probe that measures the decay each coordinate actually receives
  — the parity check alone left a kernel ignoring `WDFULL` (i.e. `"full"` silently
  degrading to `"masked"`) undetected; all 8 such single-kernel mutants are now killed.
- **`Adakaon(deterministic_reductions=True)`** — the fused big-tensor path becomes
  bit-reproducible run to run. Its batched reductions accumulate `colsum`/`rms` with
  fp32 atomics, whose completion order the scheduler picks; measured spread over 4
  identical runs (max|Δp| / weight scale) was 5.1e-8 (fp32 momentum), 3.9e-6 (bf16),
  8.1e-6 (int8). The flag switches those two to the two-pass (partials → fixed-order
  reduce) form the design doc held in reserve. Costs 0–9% and an `N*ceil(R/BR)*C`
  fp32 buffer per bucket (1.4→8.8 MiB on 236x(512,512)); default off. `keep` needs
  nothing — it is an int32 atomic, and integer addition is exact in any order.
- **AdaMuon `bias_correction`** (default `False`) — divides the factored second moment
  by `1 - β₂ᵗ`, per parameter. The correction cancels out of the row factor (a ratio of
  row statistics) and survives only in the column factor, so it reduces exactly to one
  `√(1-β₂ᵗ)` multiply on the normalized update *before* the clip: no extra state beyond
  a per-parameter step counter (`state["step"]`, now always maintained and checkpointed;
  checkpoints written without it resume at `t=1`). `t` is per parameter, not global, so
  a weight that only sometimes receives a gradient (MoE routing, CFG dropout, partial
  accumulation) is corrected by its own update count.
  **Default `False` because `clip_threshold=1.0` already does the same job, harder.**
  The uncorrected `rms(u)` measures almost exactly `1/√(1-β₂ᵗ)`, which is the factor the
  correction removes, so while the clip binds both settings emit the *same* update:
  measured applied RMS (units of `0.2·lr`) 1.0000 vs 1.0000 at step 1, 1.0000 vs 0.9953
  at step 1000, 1.0000 vs 0.9905 at step 3000 — a ≤1 % per-tensor rescale, never more.
  The paired pixel-DDPM A/B (3 seeds at the tuned lr + an upward lr sweep) shows no
  gain: the correction wins on both loss and gap on 1 of 3 seeds and loses on 2, a
  spread consistent with trajectory noise around a no-op. The configuration where it
  *does* matter is the clip disabled, where it recovers almost all of the clip's value
  (2-seed mean val 0.0701 for `clip=1.0` alone, 0.0870 for clip-off with no
  correction, **0.0714** for clip-off with the correction — it recovers 92 % of the
  clip's benefit) — i.e. the two are alternative implementations of the same
  normalization and the clip is marginally the better one. Turn it on when you
  raise or disable `clip_threshold`, or when composing with a layer that assumes an
  unbiased `1/√v`. Numbers and reasoning in docs/adamuon.md.

## [0.7.11]

### Fixed
- **Adakaon no longer bakes the LR into the momentum EMA.** Every path (per-param,
  foreach factored/non-factored, and all fused Triton kernels: one-block, chunked,
  batched, direct-4bit, no-momentum, 1-D) applied `lr` to the update BEFORE the first
  moment's EMA, so the stored momentum was an average of *lr-scaled* updates: under any
  LR schedule (warmup, cosine, restarts) the current step was driven by a mix of
  historical LRs instead of the current one, and the update direction itself changed
  when earlier LRs differed. The momentum now stores the **LR-independent direction**
  (preconditioned, RMS-clipped update) and `lr` scales the complete
  `momentum + weight_decay * p` delta at the end — algebraically identical at constant
  LR (pinned by `test_unscaled_momentum_preserves_constant_lr_legacy_update`), and a
  step at LR `x` is now the same step regardless of what LRs preceded it (pinned by
  `test_momentum_direction_is_independent_of_lr_history`). Weight decay moves with it
  (`alpha=wd`, scaled by `lr` at the end) so the decoupled decay still tracks the
  current LR exactly as before.
- **MSAM/Nekaon convert the direction back to step units.** The `norm="none"` climb
  (Nekaon's lookahead) is now `e = rho * lr * m_direction`, frozen per climb cycle like
  the existing per-element bound, so `rho`/`k` keeps meaning "lookahead in optimizer
  steps" — now measured at the CURRENT lr, which is the semantics the docs always
  promised (`test_dynamic_lr_lookahead_equals_k_current_optimizer_steps` pins
  `lookahead == k * (the step that was just taken)` across LR changes). The conversion
  keys off the owner's `_momentum_is_unscaled` marker: wrapping an optimizer with
  lr-scaled momentum (AdaBelief, AdamP, ...) keeps the historical `e = rho * m`.
  `norm="global"`/`"tensor"` normalize the momentum and are invariant to the unit change.
- **Old checkpoints migrate on load.** `state_dict()` stamps
  `_adakaon_meta["momentum_units"] = 2`; loading a checkpoint without it rescales the
  momentum `m -> m / lr` per group through the codec's exact `scale_` (quantized codecs
  scale `m_scale`, zero requant error), so a pre-0.7.11 resume continues bit-compatibly
  at the checkpoint's LR.
- Quantized (int8/4bit) momentum foreach-vs-per-param parity is now "equal to one fp32
  ULP" instead of bit-exact: the per-slice and stacked scale arithmetic go through
  different kernels now that lr is applied after the requant round-trip. fp32/bf16
  momentum parity remains bit-exact, and the full fused suite (116 tests) passes
  against the fixed native path on CUDA.

## [0.7.10]

### Performance
- **0-D scalars are batched** in all nine foreach optimizers (Adakaon, AdaBelief, AdamP,
  AdaMuon, AdaPNM, ADOPT, KProdigy, Lion, ScheduleFree). A bag of 0-D parameters —
  LyCORIS `use_scalar` gates and friends — was excluded from the batched path outright
  and stepped one at a time, ~22 CUDA launches per scalar per step of pure CPU dispatch.
  They now ride the non-factored bucket as length-1 **views** (`_backend.flat_view`),
  sharing the `L == 1` bucket with real shape-`(1,)` params; for Adakaon they also reach
  the Triton 1-D kernel, which is shape-free (base pointer + element count) and needs
  nothing but `numel() == 1`. Measured on 448 scalars, bf16 momentum: native
  330.0 -> 6.4 ms/step (51x), fused 305.9 -> 0.45 ms/step (684x). The persisted state
  keeps its 0-D shape, so checkpoints stay interchangeable with the per-parameter path
  in both directions.
- Removed the per-parameter cast anti-pattern from every batched stacking site (33
  sites). `torch.stack([x.float() for x in xs])` launches one widening kernel per
  parameter; `torch.stack(xs).float()` launches one per bucket. Numerically identical
  (bf16 -> fp32 is exact widening), and it was 901 of the 949 CUDA launches in a
  non-factored Adakaon bucket step.
- **`TILE_CAP` 131072 -> 8192**, with the 1-D ceiling split out into a new
  `TILE_CAP_1D` (131072, i.e. unchanged behaviour for 1-D). `TILE_CAP` is the 2-D
  one-block/chunked crossover, and 0.7.7's chunked-kernel rewrite moved it down by 16x —
  the old value dates from when the alternative was the native foreach path. Re-measured
  on the current kernels (min of 60 A/B-interleaved reps, fp32 and bf16 params, N=50 and
  N=200 bags, all four configs agreeing): one_block wins to 8192 (1.1-2.5x), chunked
  wins from 16384 (1.0-1.9x) and 32768 (1.5-1.8x). End to end, 200x (256,256) steps
  10.18 -> 2.26 ms in fp32 (4.5x) and 10.87 -> 1.55 ms in bf16 (7.0x); a mixed 2-D bag
  8.46 -> 1.93 ms (4.4x); bags already below the cap are unchanged within noise. The
  crossover is an occupancy effect (tensors-per-CTA against SM count), so it is GPU
  dependent and stays overridable per optimizer with `fused_tile_cap=`.
- **The 2-D and 1-D caps are now separate constants**, because they answer different
  questions. Over the 2-D cap a tensor goes to the **chunked** kernel, which is faster —
  that cap is an occupancy crossover and wants to be low. Over the 1-D cap there is no
  chunked route and the tensor falls to the **native** path, which is slower — that cap
  is a one-program capability bound. Sharing one constant would have turned the 2-D win
  into a 2.0-2.2x regression for 1-D tensors of length 16384, a reachable shape (a
  14336-wide FFN's norm weight pads to 16384 lanes). `fused_tile_cap=` keeps its name
  and now means the 2-D crossover only; it no longer moves the 1-D ceiling.
- Adakaon caches the native foreach path's bucketing and derived views per param group
  (`_ForeachPlan` / `_ForeachChunk`). The cached objects are views of tensors the
  optimizer already owns, so nothing extra is pinned; gradient views are deliberately
  **not** cached, since a retained `p.grad` view would hold the previous step's gradient
  storage alive and add a whole gradient set to peak memory. Invalidated by param-set
  change, `p.data` rebind, `load_state_dict`, `_autolr_reset_base_state`,
  `add_param_group`, and a budget-driven re-chunk. `_foreach_cache_enabled = False`
  restores the uncached behaviour for A/B measurement; it is numerically a no-op.
- Combined effect on a realistic full-fine-tune-with-LyCORIS bag (200x (256,256) + 100x
  (512,) + 128 scalars, bf16): native 122.1 -> 14.7 ms/step, fused 118.9 -> 2.26 ms/step.

### Fixed
- int8 `m_scale` layout in the batched requant of AdaBelief, AdamP, AdaPNM and Lion.
  `_store_stacked` hardcoded the scale to `(row, 1)` for `ndim >= 2` and `(1,)`
  otherwise, which is not what the per-parameter `_quant_int8` produces for anything
  that is not exactly 2-D: a conv's scale is `(R, 1, 1, 1)` and a 0-D param's is a
  scalar. A parameter stepped once by the foreach path could then never be stepped
  per-parameter again — the stored scale mis-broadcast against the momentum and raised,
  a hard error rather than a silent skew. Both paths now go through the shared
  `int8_scale_shape`.
- KProdigy's int8 pass-1 momentum update raised on any bag mixing 0-D params with int8
  momentum: the stack had no axis to reduce and the `[B, 1]` scale broadcast to
  `[B, B]`. The stack now goes through `flat_view`, which is the identity above 0-D.

### Validation
- ~200 new tests: batched-vs-per-param parity and cross-path checkpoint round trips for
  0-D params in each of the nine optimizers, fused routing and parity for 0-D under
  every momentum dtype (including int8/4bit and `beta1=0`, which the 1-D kernel has
  carried since 0.7.7), the six foreach-cache invalidation paths, the `TILE_CAP` route
  flip, and `tests/test_stacked_cast.py`, which counts `aten::_to_copy` kernels raised
  at `torch.stack` sites through a dispatch mode and fails if they scale with bucket
  size. Suite: 858 passed, 1 skipped, against 655 passed on 0.7.9.
- The cap split is pinned by two dedicated tests, because the failure it prevents is a
  silent perf regression that no correctness assertion would catch: one on the
  predicates (at the same 16384 padded lanes, a 1-D tensor is eligible for the fused 1-D
  kernel while a 2-D tensor is not) and one end to end (a bag of 16384-long 1-D params
  lands in `one_dim`, not `native`).

## [0.7.9]

### Changed
- **Breaking.** `MSAM.load_state_dict` now raises when the checkpoint was saved in
  train mode. Such a checkpoint stores weights that already carry the lookahead
  perturbation, and a fresh optimizer cannot know to remove it, so resuming baked one
  perturbation into the weights per resume (measured `k * lr * clip_threshold`, e.g.
  1.5e-4 at `k=1.5, lr=1e-4`) with nothing in the state dict to detect it. `state_dict`
  now records `train_mode`; checkpoints written before this release load unchanged.
  Call `optimizer.eval()` before saving, as the docs already required.
- The MSAM/Nekaon climb round trip writes low-precision weights with
  **round-to-nearest instead of stochastic rounding**, on both the torch and Triton
  paths. Stochastic rounding exists so an *accumulating* update below the weight's ulp
  is not lost; the climb accumulates nothing (applied at the end of a step, removed at
  the start of the next), so two independent SR draws do not cancel and the weights
  random-walk. Measured on bf16: 19% relative L2 drift after 4000 climb/removal cycles,
  growing as sqrt(N), against exactly zero on fp32. Round-to-nearest makes the pair land
  back on the same stored value: measured drift exactly zero, while the climb is still
  applied wherever bf16 can represent it (0.95x of the fp32 perturbation at `lr=1e-4`).
  Affected MSAM in all three `norm` modes and therefore Nekaon; Lookahead and SAM were
  never affected because they restore an exact weight snapshot instead of recomputing
  the perturbation.

### Added
- Inert-lookahead warning. The climb can be too small to do anything in two distinct
  ways: below half a low-precision ulp (unrepresentable), or representable but so small
  that the gradient at the perturbed point is indistinguishable from the true one.
  Measured on a real MLP with fp32 weights, `|dw|/|w| = 2.3e-5` moved the gradient by
  0.018% while `3.7e-3` moved it by 2.1%, so the mechanism is reported inert below
  ~1e-4 relative displacement. Fires once, after the condition has held for 50
  consecutive climbs so an LR warmup does not trip it. At `lr <= 1e-6` with the default
  `k=1.5` this means Nekaon has been equivalent to Adakaon plus rounding noise, which
  was previously silent.

### Fixed
- The fused perturbation plan now validates its cached **weight** pointers. 0.7.8 made
  every momentum writer requantize in place so `m`/`m_scale` pointers cannot dangle, but
  nothing pinned a weight's storage: an external EMA, a `.to()` or an FSDP reshard
  rebinds `p.data` and left `p_addr` addressing freed memory, with the plan's
  invalidation key unable to observe it.

### Validation
- `tests/test_msam_climb_precision.py`: 20 tests over the round-trip contract (every
  momentum codec, every `norm` mode, torch and Triton paths), a guard that the fix does
  not simply stop perturbing, the warning's fire/quiet conditions including the warmup
  window, checkpoint rejection with backward compatibility, and pointer invalidation.
  15 of them fail without this release.

## [0.7.7]

### Performance
- Expanded Adakaon's Triton backend across the full fine-tuning shape mix:
  quantized 1-D state, convolutions, large matrices, and the momentum-free
  `beta1=0` path now remain fused instead of falling back to Python/foreach.
- Replaced large-tensor 4-bit dequantize/EMA/requantize temporaries with aligned
  in-kernel block processing. Standard 64/128-element blocks allocate no
  momentum-sized fp32 temporary; non-aligned custom blocks retain the compatible
  fallback.
- Cached large-tensor reduction buffers, pointer arrays and Nekaon/MSAM momentum
  dispatch plans, while preserving invalidation on reset, load and late gradients.
- Generalized Nekaon/MSAM's fused perturbation to fp32, bf16, int8 and 4-bit
  momentum, avoiding per-parameter stochastic-rounding and dequantization calls.

### Fixed
- Kept int8/4-bit momentum buffer identities stable across requantization so
  fused pointer caches cannot retain stale addresses.
- Separated Triton buckets by parameter dtype; mixed fp32/bf16 parameter groups
  no longer share an incompatible compile-time pointer interpretation.
- Made odd-length 4-bit storage a tested framework contract: `ceil(n/2)` bytes,
  correct final low nibble, and a canonical zero unused high nibble across native,
  stacked and Triton implementations.

### Validation
- Added native/fused parity for Adakaon and Nekaon across bf16/int8/4-bit,
  momentum-free steps, late gradients, convs and large/odd shapes.
- Added a dedicated odd-length 4-bit contract battery spanning lengths 1–1025,
  block sizes 1/7/64/128, individual/stacked codecs and 1-D/2-D/3-D EMA state.

## [0.7.6]

### Safety
- Quarantined `auto_lr=True` after real-training and proxy failures showed that
  gradient/trajectory-only controllers can silently overshoot a workload's safe
  learning rate. Enabling it now raises before the first optimizer step.
- Retained the constructor arguments temporarily for a clear migration error.
  `auto_lr=False` has no extra state or step overhead, and legacy checkpoints
  load their base optimizer state while discarding the retired AutoLR blob.
- Moved the DoWG, range-test, Mechanic, LR-servo, MoMo/AdamG and NGN-MDv1
  investigations to `docs/EXPERIMENTS_GRAVEYARD.md`; none is advertised as a
  production learning-rate solution.

## [0.7.5]

### Changed
- Replaced the bounded DoWG probe behind `auto_lr=True` with continuous Mechanic:
  six discounted bettors, an explicitly stored anchored trajectory, no loss callback, geometric
  ramp, contact detector, fuse, freeze, or step horizon.
- `auto_lr_d0` is deprecated and ignored; low and high legacy values now produce
  identical safe-start trajectories. `auto_lr_fuse_rel` remains accepted for
  source compatibility but no longer caps the continuous controller.
- Mechanic keeps one native-dtype parameter anchor and one persistent fp32
  normalized trajectory (~6 B/trainable parameter for bf16 weights). Bounded
  shape batches accelerate small tensors, while oversized tensors use an unstacked
  path to avoid multiplying full-tensor temporaries. The explicit trajectory prevents
  a safe sub-ULP seed from stalling on bf16 weights.
- Nekaon/MSAM now expose an internal true/virtual/live protocol so their lookahead
  is removed before Mechanic measurement and reapplied with the exact selected
  scale. Native and fused/Triton paths share the same lifecycle.

### Fixed
- Scalar overflow and non-finite reconstruction fail closed without returning
  corrupted parameters. Non-finite gradients skip the base optimizer and warn once.
- AutoLR checkpoints serialize all bettor state, anchors, fp32 trajectories, and
  the fused MSAM stochastic-rounding seed through the public mixin contract.
  Nekaon reconstructs its live view before the first resumed forward/backward;
  incompatible 0.7.4 DoWG checkpoints fail explicitly.

### Validation
- Added paired multi-seed fixed-LR comparisons for Adakaon, Nekaon, and Lion,
  plus quantized checkpoint tests, bf16 stochastic-rounding/Kahan coverage, and
  native-Windows CUDA/Triton true/live tests.
- Added 5,000-step, three-seed stability runs for each AutoLR optimizer and an
  explicit fixed-LR update-equivalence gate with cautious updates and weight decay.

## [0.7.4]

### Changed
- **AutoLR is now fully autonomous.** `auto_lr=True` starts the low-VRAM DoWG
  controller immediately and no longer uses a loss-driven range test, closure, or
  trainer decision. `optimizer.report_loss(loss)` is a deprecated compatibility
  no-op in this release (one warning per optimizer) and will be removed in 0.8.0.
- AutoLR snapshots every trainable parameter from the beginning, including
  late-gradient parameters. It uses instant spike detection plus a fixed-reference
  log-gradient level guard: the first eight finite norms form an immutable baseline
  and later eight-sample windows cannot move that baseline.
- Discovery grows only before its first contact. A contact rolls parameters back,
  clears the base state and fused caches, and restarts DoWG accumulation; a
  comparable second contact freezes the LR. Discovery also freezes conservatively
  at its fuse or fixed 192-step budget, recording `edge_confirmed`, `fuse_bound`,
  or `budget_bound` in checkpoint state.
- An explicit `auto_lr_d0` can no longer enlarge its own fuse without limit. It
  retains at most 4× bounded compatibility headroom over the data-relative fuse;
  higher seeds are clamped with a diagnostic, making recovery from an accidentally
  high starting value autonomous and bounded.
- Repeated non-finite gradients are skipped with a diagnostic after one
  rollback/backoff, preventing unbounded LR reduction.

### Fixed
- Fused Adakaon and AdaPNM reset their state, counters, partitions, and pointer
  caches before rebuilding buffers after an AutoLR rollback or state replacement.
  Fused pointer arrays are never reused across a reset/load that replaces state
  tensors.
- AutoLR checkpoint state now serializes its fixed baseline, rolling window,
  contacts, counter, and freeze reason. Checkpoints carrying the retired 0.7.3
  loss-probe state load compatibly and resume with the autonomous controller.

### Documentation
- Replaced the retired `Autokaon`/Mechanic documentation with
  [`Adakaon(auto_lr=True)`](docs/autolr.md): AutoLR is described as a conservative
  autonomous dynamic step-size controller, not a universal optimal-LR detector.

### Added
- **`gradient_centralization`** — a composable, **zero-state** gradient preprocessor
  (Gradient Centralization, Yong et al. 2020, arXiv:2004.01461): subtract the per-output-row
  gradient mean over the fan-in dims for every ≥2-D weight, at the top of the step, before the
  optimizer reads `p.grad`. One implementation in `kaon._backend.centralize_grads_`, wired into
  every optimizer. **On by default** for `Adakaon`, `Lion`, `AdaPNM`, `KProdigy`; **off** for
  `AdaMuon` (its Newton-Schulz orthogonalization already handles the directional structure — GC
  hurt it). Measured on the proxy (gap lens, 3 seed-pairs each): a free held-out-loss win of
  **~-0.003..-0.006** for the factored-Adam / sign optimizers, at no memory cost and negligible
  speed cost. Disable with `gradient_centralization=False`. Note: it is a *modifier* — the
  reference-equivalence properties (e.g. KProdigy ≡ reference Prodigy) hold with it off, and it
  can hurt very small / non-conv problems, so it is a per-optimizer opt-out.
- **`AdaPNM`** — **Adam + Positive-Negative Momentum** (Xie et al. 2021,
  *Manipulating Stochastic Gradient Noise to Improve Generalization*, arXiv:2103.17182)
  on the kaon backend (factored quantized second moment, int8/4bit momentum codec,
  stochastic-rounding bf16, cautious, foreach). PNM's negative-momentum term injects
  anti-correlated noise — a built-in *implicit regularizer* (flat-minima seeking)
  **without** SAM's extra forward/backward. Tuned defaults **`betas=(0.8, 0.999)`,
  `beta0=0.5`** (`beta1` is the loss↔gap dial; the proxy sweep bottoms at `0.8`). On the
  synthetic gap proxy it reaches ~Lion/AdamW loss at **36–44% lower train–val gap**, and
  it is the **most gap-robust optimizer at constant LR** (no schedule needed — resumable;
  35–43% lower gap than the field). Developed under the code name *Janus*.
  See [docs/adapnm.md](docs/adapnm.md).
- **`Lion`** — **Lion's sign-momentum** update (`sign(β1·m+(1-β1)·g)`, single momentum
  buffer, **no second moment**) on Adakaon's backend: the shared int8/4bit momentum codec,
  stochastic-rounding bf16 weight update, cautious masking, and **foreach batching** (bit-exact
  vs the per-param path). Lightest state in the family — **~1 B (int8) / 0.5 B (4bit) per
  param** — targeting Lion's implicit regularization for small-data diffusion fine-tuning.
  `lr` is Lion-scale (~AdamW/5); `betas` are a measured loss↔generalization dial
  (`(0.95,0.98)` for loss, higher β2 for a lower train–val gap). No `eps`/`clip_threshold`
  (the sign update is unit-magnitude — nothing to clip). See [docs/lion.md](docs/lion.md).
- **`AdaMuon`** — Muon's Newton-Schulz orthogonalized momentum + an Adafactor-style
  **factored, quantized second moment of the orthogonalized update**. Targets
  beating AdamW on convergence/precision at **near-Adafactor memory** (~1–2 B/param;
  reuses Adakaon's int8/4bit momentum codec, foreach batching, stochastic rounding,
  dtype-safe checkpointing). See [docs/adamuon.md](docs/adamuon.md).
  - Tuned defaults `ns_steps=2`, `cautious=True`, `betas=(0.95, 0.999)`; a single
    `lr` governs 2-D and 1-D params (all RMS-normalized to applied RMS ≈ `0.2·lr`).
    `lr` is Muon-scale — start ~`1e-3` for diffusion (the API default `2e-2` is
    LLM-scale).
  - `clip_threshold=1.0` validated as the optimum and **load-bearing** (an RMS ceiling
    on the *normalized update*, Adafactor-style — not gradient clipping; off ≈ +24%).
  - Optional `compile=True` — whole-step `torch.compile` (AdaMuon-only by design);
    workload-dependent, benchmark it.
  - Reproducible harnesses + evaluation under `benchmarks/adamuon/`.
- **`Autokaon`** — a parameter-free learning rate on Adakaon's update via a
  [Mechanic](https://arxiv.org/abs/2306.00144) scalar tuner (an update-agnostic
  online LR tuner — **Mechanic, *not* Prodigy**), with a **freeze-to-free**
  handoff. This historical implementation was retired in 0.7.4 in favour of
  autonomous `auto_lr=True`.
  - Train at `lr=1.0`; the tuner discovers the effective LR (read via `get_d()`),
    keeping Adakaon's exact normalize-then-momentum update verbatim.
  - `lr_freeze` (default `"auto"`; also `int N` / `None`) ends adaptation: it folds
    the discovered LR `S` into the inner Adakaon's `lr`, **frees the Mechanic
    `ref` buffer**, and routes every later `step()` straight to the base — so after
    freeze it is **byte-for-byte and speed-for-speed plain Adakaon at `lr=S`**.
    With the default `adakaon_betas=(0.0, 0.999)` (beta1=0) the handoff is
    bit-exact (Adakaon's update is then linear in `lr`); `"auto"` freezes on an
    LR plateau.
  - **Minimal, parameter-free API:** the common case is
    `Autokaon(params, **adakaon_kwargs)`. The empirical scaffolding that
    accumulated across iterations (`store_delta`, `s_init_rel`, `scale_floor_frac`,
    the auto-freeze `tol`/`patience`/`max_frac`) was collapsed to internal
    constants once iteration-3 validated on a real SDXL LoRA that the data-relative
    cap generalizes (val flat across `scale_cap_rel` 3–12). The only LR-equivalent
    knob, `scale_cap_rel` (default `6`), is kept but marked advanced / rarely
    needed.
  - Only per-param state while adapting is the irreducible `ref` (one extra copy of
    the weights); `Delta` is reconstructed on the fly as `(p-ref)/sum(s)`.
  - `adakaon_betas` passthrough sets the inner momentum betas (the tuner `betas`
    kwarg shadows them); all other Adakaon knobs forward through `**kwargs`.
  - **Naming:** the optimizer is `Autokaon`. (It went through the working names
    `AdakaonProdigy` — a misnomer, it is Mechanic, not Prodigy — and
    `AdaptiveAdakaon` during development; neither shipped, both are removed.)
- **KProdigy now reuses Adakaon's full update engine.** KProdigy's pass-2 weight
  update (previously a per-parameter Python loop) is now backed by Adakaon's
  foreach batching, momentum codec (`float32`/`bfloat16`/`int8`/`4bit`), cautious
  masking, conv-aware matrixized factoring, and stochastic-rounding bf16 weights —
  with Prodigy's effective learning rate (`lr × D`) folded into the update.
  KProdigy's **D-estimation (pass 1) is unchanged**: the two-pass global reduction,
  `slice_p`, `independent_d`, `d_coef` etc. produce a bit-identical D trajectory and
  final weights vs the previous release (verified on CPU fp32 across every
  dtype/second-moment combo).
  - New KProdigy args mirroring Adakaon: `momentum_dtype="4bit"`, `cautious`,
    `foreach` (default `True`), `foreach_batch_cutoff`, `foreach_stack_budget`,
    `momentum_4bit_block`.
  - The momentum codecs + quant helpers were extracted from `adakaon.py` into a
    shared `kaon._momentum_codec` module (re-exported from `kaon.adakaon` for
    backwards compatibility) — no duplicated implementations.
  - foreach == per-param: **bit-exact on fp32 weights (CPU and CUDA)** across
    `momentum_dtype ∈ {float32, bfloat16, int8, 4bit}`, cautious on/off, on 2-D +
    conv + 1-D params. New foreach-parity/4bit/cautious tests.
  - Update-backend speedup (foreach vs the old per-param loop): **~1.8× on a
    LoRA-like distribution**, **~1.5× on the SDXL full-FT distribution**. (Smaller
    than Adakaon's because KProdigy's pass-1 D-reduction is per-parameter in both
    arms — only the pass-2 update is batched.)
  - Memory on the SDXL UNet shape distribution (bytes/param): factored/4bit +
    `slice_p=11` = **1.27 B/param** (vs Adakaon 4bit 0.54, AdamW-class 8–14).
    The Prodigy D-state (`s`+`p0`) dominates at `slice_p=1`; `slice_p` is the lever.
- `Adakaon(foreach=True)` (now the default) — multi-tensor batching of the
  step. Params are bucketed by shape, each bucket stacked into one tensor, and
  the entire update (EMA + reconstruction + RMS clip + momentum + weight decay +
  cautious + stochastic rounding) runs as a handful of batched kernels per bucket
  instead of a per-parameter Python loop. Two branches: `ndim >= 2` factored
  `[N, R, C]`, `ndim == 1` (biases/norms) non-factored `[N, L]`.
  - **~19× faster** optimizer step on adapter training (the hot case): measured
    on a real SDXL UNet + PEFT LoRA r=8 (1434 tiny trainable tensors), the step
    drops from **318 ms → 16 ms** (1.45× of fused AdamW; was 28×).
  - **Full fine-tune: ~1.3×** (SDXL UNet, 1680 params: 339 ms → 256 ms; Cosmos
    685 params: 239 ms → 231 ms). The smaller win is expected — a full fine-tune's
    optimizer time is dominated by real bandwidth work on large weights, where
    per-tensor launch overhead is noise; batching only removes that overhead, which
    dominates in the many-small-tensors (adapter) regime.
  - Element-for-element equal to the per-parameter path (bit-exact on CPU; ~1e-8
    on CUDA from reduction order). Stochastic-rounding draws legitimately differ
    (unbiased either way). 11 new parity/coverage tests.
  - **Two decoupled knobs** (see `docs/foreach-batching.md`):
    - `foreach_batch_cutoff` (default `2_000_000` elements) — the **performance**
      threshold: weights larger than this loop instead of stacking (batching only
      helps while launch overhead dominates; large weights are bandwidth-bound).
      It is an absolute element count, *not* a fraction of VRAM — a budget sweep on
      SDXL and Cosmos full fine-tunes showed the same model-independent crossover.
    - `foreach_stack_budget` (default `None`) — the **memory-safety** chunk cap:
      `min(free_VRAM × 0.10 / 48, 4 × cutoff)`. The VRAM term keeps batching from
      OOM-ing a full fine-tune; the `4 × cutoff` cap stops over-stacking medium
      weights (measured slower past ~8 M). An int pins a fixed cap. Decoupling the
      two means raising the budget never pulls large weights into stacking, and a
      roomy/huge card stays in the measured optimum instead of degrading.
    - The transient divisor (48 B/element) is itself model-independent — measured
      byte-for-byte identical on SDXL and Cosmos.
  - Falls back to the per-parameter path for what it doesn't batch: 0-D scalars,
    large weights, `bf16_method="kahan"`, fp16+SR, non-contiguous matrixized
    convs, and single-param (gradient-release) optimizers.
- `momentum_dtype="int8"` is now **foreach-batched** (previously excluded from the
  fast path and always looped per-parameter). The per-row absmax quantization is
  done on the stacked layout — dequant → fp32 EMA → requant per bucket — which is
  element-for-element equal to the per-param int8 path: the per-row absmax of the
  stacked `[N, R, C]`/`[N, L]` momentum (reduce only the trailing axis) reproduces
  each tensor's per-param scale exactly. This makes "cheap momentum that fits"
  (1 B/param, ~2.6 GB state on a 2.57 B SDXL UNet) also fast on adapter training.
  - **~8× faster** on a LoRA-like distribution (320 tiny tensors, 2.7 M params):
    98 ms → 12 ms. Full fine-tune ~1.17× (most weights are large and loop by the
    cutoff). Bit-exact on CPU; CUDA max abs diff ~7e-9 (float reduction order).
  - 3 new parity/coverage tests (int8 in `test_foreach_matches_per_param`,
    int8 + weight decay, and `test_foreach_int8_chunking_is_exact`).
- `ktune` console script (`uv run ktune --model <ckpt>.safetensors --gpu N`) to
  check the foreach cutoff on your own GPU/model.
- `momentum_dtype="4bit"` — signed linear 4-bit momentum (round-to-nearest) with a
  per-block absmax scale, two nibbles packed per byte for a real ~0.5 B/param store
  (+ a small per-block fp32 scale, ~0.03 B/param at the default block 128). New
  `momentum_4bit_block` knob (default `128`; `<=0` = whole-tensor). Foreach-batched
  and bit-exact vs the per-parameter path on CPU.

### Changed
- **Renamed `Autofusion` → `Autokaon`** (follows the `Adafusion → Adakaon` rename —
  the in-house optimizers built on the kaon backend now share the `*kaon` family
  name). Class, module (`kaon.autofusion` → `kaon.autokaon`), tests, and docs
  (`docs/autofusion.md` → the then-current Autokaon documentation) renamed. No
  behaviour change; the historical documentation was removed in 0.7.4.
- **Renamed `Adafusion` → `Adakaon`.** The flagship optimizer is the one that most
  fully exercises the shared **kaon** backend (factored second moment + quantized
  momentum codec + stochastic rounding + foreach + cautious) — every other optimizer
  reuses its machinery — so it now carries the framework's name. Class, module
  (`kaon.adafusion` → `kaon.adakaon`), tests, and docs (`docs/adafusion.md` →
  `docs/adakaon.md`) renamed; `Autokaon`'s `adafusion_*` kwargs are now
  `adakaon_*`. No behaviour change.
- `cautious` now defaults to **`True`**. Measured on a mini pixel-DDPM (8 paired
  seeds, per-arm best LR): with momentum it lowers held-out val loss ~1.4% (paired
  t=−4.07, p<0.05); it is a literal no-op without momentum (the mask is all-ones —
  mask-active fraction ≈ 0). Set `cautious=False` for no-momentum configs to skip
  the then-useless masking op.
- Refactored Adakaon's momentum handling into a unified per-dtype **momentum
  codec** (`init_state` / `ema_one` / `ema_stacked`). The dequant → fp32 EMA →
  requant logic for each `momentum_dtype` now lives in exactly one place instead of
  being copy-pasted across the per-parameter step and the two foreach buckets; the
  step functions call `codec.ema_*`. Pure restructuring — float32/bfloat16/int8
  remain bit-for-bit identical (existing parity tests pass unchanged).

### Removed
- **Removed the plain `Muon` optimizer.** It keeps a full-precision momentum buffer with
  an AdamW fallback for 1-D params (~2 B/param, no quantization) — it does not fit a
  memory-efficient library, and `AdaMuon` (orthogonalized momentum + a factored, quantized
  second moment) supersedes it. Its Newton-Schulz kernel (`zeropower_via_newtonschulz5`,
  the only piece `AdaMuon` used) moved into `adamuon.py`; the `Muon` class, `docs/muon.md`,
  and the tests are gone.
- Removed `decay_rate`, `factor_conv_as_matrix`, and `compile` (alpha cleanup —
  no reliable benefit / superseded).
  - `decay_rate` (HF Adafactor adaptive `beta2_t = 1 - step**decay_rate`): a
    paired-seed convergence experiment (8 seeds, with/without momentum) found no
    reliable benefit on diffusion (all comparisons ns or not surviving multiple-
    comparison correction). beta2 is now always the fixed `betas[1]`; the adaptive
    branch, the `_one_minus_beta2_vec` helper, and the per-state `step` counter
    (its only user) are gone.
  - `factor_conv_as_matrix`: conv-aware factoring is **always on** — a 4-D conv
    kernel `[out, in, kh, kw]` is always reshaped to 2-D `[out, in·kh·kw]` before
    the second moment is factored. The legacy `False` path was already removed; the
    kwarg was a dead no-op.
  - `compile`: `torch.compile` of the per-tensor factored core measured neutral-to-
    negative across model sizes and is superseded by `foreach` batching; the kwarg
    was a dead no-op (the compiled path was already removed).

## [0.2.0] - 2026-06

### Added
- `KProdigy` — memory-efficient Prodigy (parameter-free D-adaptation),
  reimplemented natively rather than vendored from the research repo:
  - Exact D-estimation math: the full second moment + fp32 momentum path
    reproduces reference `prodigyopt.Prodigy` to ~1e-4 on the D estimate.
  - kaon memory toolkit: `momentum_dtype` (`float32`/`bfloat16`/`int8`),
    `second_moment="factored"` (Adafactor row+col; experimental — inflates D),
    `slice_p` (sliced D statistics), and stochastic-rounding / Kahan bf16 weight
    updates (`bf16_method`).
  - **Sane defaults** that fix the original repo's footguns: `d_update_freq=1`
    (not 5) and `use_bias_correction=False` (not True), both of which starved
    the D-bootstrap so the effective LR failed to rise.
  - `independent_d` (auto-on for >1 param group): per-group D so SDXL UNet and
    Text Encoder adapt independently.
  - `benchmarks/bench_kprodigy_d.py` characterizing the D (effective-LR)
    trajectory across defaults, memory variants, and dataset scale/conditioning.
  - 26 tests (parity, memory variants, bf16 stochastic rounding, independent-D).

## [0.1.0] - 2026-06

Initial release of `kaon` (K-Optimizers).

### Added
- `Adakaon` — conv-aware factored optimizer:
  - Factored second moment with the **conv-aware fix** (reshape 4-D conv kernels
    to 2-D before factoring → near-zero state vs ~0.4 B/param for the last-dims
    variant on a diffusion UNet).
  - Optional first-moment momentum in `float32` / `bfloat16` / `int8`
    (`momentum_dtype`). bf16 momentum matches fp32 quality at half the state.
  - bf16-correct weight updates via stochastic rounding (no buffer) or Kahan.
  - Optional `cautious` masking, `decay_rate` (HF Adafactor schedule),
    `clip_threshold`, decoupled `weight_decay`.
  - `compile=True` routes the factored core through `torch.compile`
    (~+30% on large 2-D weights).
- `Muon` — orthogonalized-momentum (Newton-Schulz) with an AdamW fallback for
  1-D params, auto-routed by rank; `momentum_dtype` for bf16 momentum.
- Tests for both optimizers.
