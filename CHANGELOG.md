# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

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
  `beta0` negative-momentum mix), KProdigy, AdaMuon. Adakaon is pending (file
  under a concurrent audit batch). Deliberately **not** wired into ScheduleFree:
  its quantized `z` buffer is a plain accumulator (`z -= lr_t * d`), not decayed
  by `beta1`, so the warning's AR(1)-amplification argument does not apply to it
  (see `docs/momentum.md`).

### Performance
- **int8 `ema_one`**: `.float().mul_(scale)` (one temp), drop `delta.clone()`,
  write codes with `(m/scale).round_().clamp_` into `state["m"]` — measured
  13→8 B/elem transient, 14→12 kernels, ~1.29×, bit-identical. Same clone
  removal on int8/4bit `ema_stacked` / 4bit `ema_one` (the strided 4bit dequant
  slice is kept as-is: materialising it changed the `lerp_` kernel and broke the
  per-param/stacked bit-exactness).

### Fixed
- **Resume under bf16/fp16 params is byte-identical again.**
  `load_state_dict_preserving_dtypes` used to cast state back to the saved dtype
  *after* torch had already rounded floating buffers through the param dtype
  (~0.3% relative drift per resume on `m_scale`/`row`/`col`/`v`). It now
  re-applies the checkpoint tensors (values + dtype), `copy_` when identity can
  be kept for MSAM's cached pointers, and accepts int/str state keys (JSON drift).
- **ScheduleFree stochastically rounds bf16 `z` on every write.** This is independent
  of `bf16_method` and consumes RNG even when model weights are fp32, so sub-ULP
  `z` updates remain unbiased without a Kahan/shift buffer.
- **AdaPNM handles empty groups and late gradients consistently.** Parameter-local
  steps retain the correct bias correction and global PNM parity, while the shared
  momentum codec preserves state-buffer identities required by cached pointers.
- **AdaPNM int8 1-D scales now use scalar shape `()` instead of `(1,)`.** The value
  is unchanged; checkpoints have been verified across foreach and per-parameter
  paths in both directions.
- **Lion / AdaBelief / AdamP / KProdigy requantize momentum in place.** Their
  `_store_one` / `_store_stacked` (and KProdigy's int8 EMA) reassigned
  `state["m"]` / `state["m_scale"]` on every step. MSAM/Nekaon cache `data_ptr`
  tables into those buffers for the fused climb, so a reassignment left the plan
  reading freed memory (measured climb error ≈ 76% of the bound with
  `MSAM(Lion, momentum_dtype="int8")`). All four now delegate to
  `_MomentumCodec.store_one` / `store_stacked`, which `copy_` into the existing
  tensors — the same contract Adakaon's codecs already followed since 0.7.8.
  Numeric output is bit-identical (same quantizers); only storage identity changes.
  Note: int8 `m_scale` for 1-D params stays shape `()` (as allocated); the old
  foreach path had rewritten it to `(1,)` via reassignment. Values match; old
  checkpoints with either layout still load.
- **MSAM fused-plan witnesses cover `m` / `m_scale`.** `_plan_addrs_valid` now
  re-reads `data_ptr` from the live state dicts (and every weight in the bucket),
  so a base that still reassigns momentum invalidates the plan instead of climbing
  on dangling pointers. `MSAM(AdaPNM, …)` with `rho != 0` raises `TypeError`
  (dual `m_pos`/`m_neg`, no single `m`); `rho=0` remains a passthrough. The
  inert-lookahead threshold is mode-aware (`none`: per-coordinate `|rho|*lr*clip`;
  `global`/`tensor`: `|rho|` vs weight L2 norms) so `norm="global", rho=0.3,
  lr=1e-6` no longer spuriously suggests raising lr.
- **Lion `foreach` no longer silently skips `channels_last` convs.** The batched
  write flattened via `reshape`, which copies a non-contiguous weight and drops
  the update (max|Δw| = 0 over many steps). `ndim > 2` params now require
  contiguity for the foreach path (same gate as Adakaon) and fall back to
  per-param otherwise.
- **ADOPT** — per-parameter ``state["step"]`` governs the step-0 ``v`` init and the
  ``step**0.25`` clip so a param whose first gradient arrives late (or a ``.step()``
  with no grads) no longer leaves factored ``row``/``col`` at zero and diverges to
  NaN; ``momentum_dtype`` is resolved per param group (not silently from the
  constructor kwarg). Checkpoints without per-param ``step`` seed the counter from
  ``max(1, group["step"] - 1)`` when ``v``/``row`` already exist.
- **SAM** — ``second_step`` restores every ``old_p`` snapshot even when the second
  backward left ``p.grad`` as ``None``; global grad norm and the climb are batched
  (``torch._foreach_norm``, stacked chunks bounded by ``foreach_budget``). The
  batched norm accumulates in fp32 so ``scale`` is slightly more accurate on bf16
  weights than the old per-param bf16 reduction.
- **Lookahead** — ``_sync_foreach`` chunks stacked slow-weight syncs with
  ``foreach_budget`` instead of materializing an unbounded ``[N, *shape]`` transient.
- **ktune** — OOM timings format as ``OOM`` instead of raising ``TypeError``; when
  CUDA+Triton are available, sweeps ``fused=True`` over ``fused_tile_cap`` and
  reports the best tile.
- **Stochastic rounding preserves NaN on CUDA.** The int32 bit-trick added noise to the
  canonical NaN pattern `0x7FFFFFFF`, overflowed the sign bit and wrote `-0.0` instead:
  a diverging run looked healthy while weights were silently zeroed (CPU kept the NaN,
  so no CPU test could see it). NaN now propagates like every other PyTorch op; `+-inf`
  and finite overflow to `inf` behave exactly as before (bit-identical to 0.7.11 with
  the same noise).
- **Stochastic rounding no longer consumes the global RNG.** Noise comes from a
  per-device `torch.Generator` owned by the module, seeded from the global initial seed
  and re-seeded whenever `torch.manual_seed` changes it, so dataloader / dropout streams
  no longer depend on how many parameters were rounded. Re-seeding to the *same* value
  inside one process is not observable; call `kaon.reseed_stochastic_rounding()` then.
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
- **`centralize_grads_` groups by `(shape, device, dtype)`.** A param group mixing CPU
  and CUDA tensors crashed in `torch.stack` on the first step (default config).
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

### Added
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

### Performance
- `add_stochastic_` writes back with `copy_(fp32)` (no bf16 temporary) and draws int32
  noise directly (4 B/elem transient instead of 10 B/elem).
- `subtract_batched_` casts the stacked delta once per bucket instead of once per
  parameter when the stack dtype differs from the params (`aten::_to_copy` no longer
  scales with bucket size; 5-15x on the cast-heavy bags).
- **AdaPNM's big (chunked, batched) route caches its pointer arrays** in a new
  `BigPnmCache`, the AdaPNM counterpart of `BigPointerCache`: grad / weight / both momenta
  address arrays plus the `rowmean`/`rowsum`/`colsum`/`keep`/`rms` scratch were rebuilt from
  fresh `torch.tensor([...])` allocations on *every* step, while the one-block and 1-D routes
  have cached theirs since 0.7.9. Measured on an 8x `(512,512)` bucket, steady state:
  caching-allocator allocations 22 -> 13-14 per step and the pointer arrays' host-to-device
  copies 4 -> 0-1 per step (two independent measurements). CUDA **kernel** launches are
  unchanged — the win is host-side allocation and H2D traffic, not launch count. Same math,
  same state.
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
- **fp16 weights + `bf16_method="stochastic_rounding"` silently fell back to
  round-to-nearest**; the constructor and `add_param_group` now raise
  `NotImplementedError` (validated before the group is added).
- **Non-finite policy is now uniform: propagate.** `sr_round` no longer turns NaN into
  `-0.0` (int32 overflow) or a low-payload NaN into `+inf`; the seven cautious sites
  multiply by the mask instead of `tl.where`, so an `inf`/NaN gradient produces the same
  non-finite tensor on the fused and native paths (the fused path used to freeze the
  tensor forever in silence).
- **Mixed-device param groups** (CPU + CUDA) crashed in `torch.stack` inside the foreach
  and fused bucketing; `device` is part of every bucket key and kernels launch under the
  bucket's device.
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
- **AdamP's batched projection removes the radial temporaries.** The stacked path
  built six `[N, R, C]` tensors (normalized weights, radial components and two
  `torch.where` results) even though at most one branch fires per slice. It now
  reduces to one broadcast coefficient per row/tensor applied with `addcmul_`:
  31.2 -> 20.4 ms isolated (-35%) and 496 -> 369 MB of step peak (-26%). The fp32
  reassociation makes it differ from the normalized-vector form by ~5e-8
  relative. The per-param path also dropped its two host synchronizations per
  param per step (`if cos.max() < ...`) for an on-device mask.

### Performance
- **AdaPNM fused caches are keyed by `(group, lag)`.** Stable late-gradient buckets
  reuse their pointer caches instead of rebuilding them every step, avoiding the
  measured 3.9 -> 29 ms/step regression and reducing 400 reconstructions to a
  stable cache set; inactive lags are pruned to keep memory bounded.

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
