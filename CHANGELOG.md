# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **`Liofusion`** — **Lion's sign-momentum** update (`sign(β1·m+(1-β1)·g)`, single momentum
  buffer, **no second moment**) on Adafusion's backend: the shared int8/4bit momentum codec,
  stochastic-rounding bf16 weight update, cautious masking, and **foreach batching** (bit-exact
  vs the per-param path). Lightest state in the family — **~1 B (int8) / 0.5 B (4bit) per
  param** — targeting Lion's implicit regularization for small-data diffusion fine-tuning.
  `lr` is Lion-scale (~AdamW/5); `betas` are a measured loss↔generalization dial
  (`(0.95,0.98)` for loss, higher β2 for a lower train–val gap). No `eps`/`clip_threshold`
  (the sign update is unit-magnitude — nothing to clip). See [docs/liofusion.md](docs/liofusion.md).
- **`AdaMuon`** — Muon's Newton-Schulz orthogonalized momentum + an Adafactor-style
  **factored, quantized second moment of the orthogonalized update**. Targets
  beating AdamW on convergence/precision at **near-Adafactor memory** (~1–2 B/param;
  reuses Adafusion's int8/4bit momentum codec, foreach batching, stochastic rounding,
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
- **`Autofusion`** — a parameter-free learning rate on Adafusion's update via a
  [Mechanic](https://arxiv.org/abs/2306.00144) scalar tuner (an update-agnostic
  online LR tuner — **Mechanic, *not* Prodigy**), with a **freeze-to-free**
  handoff. See [docs/autofusion.md](docs/autofusion.md) for the full design,
  the minimal API, and the validated campaign results.
  - Train at `lr=1.0`; the tuner discovers the effective LR (read via `get_d()`),
    keeping Adafusion's exact normalize-then-momentum update verbatim.
  - `lr_freeze` (default `"auto"`; also `int N` / `None`) ends adaptation: it folds
    the discovered LR `S` into the inner Adafusion's `lr`, **frees the Mechanic
    `ref` buffer**, and routes every later `step()` straight to the base — so after
    freeze it is **byte-for-byte and speed-for-speed plain Adafusion at `lr=S`**.
    With the default `adafusion_betas=(0.0, 0.999)` (beta1=0) the handoff is
    bit-exact (Adafusion's update is then linear in `lr`); `"auto"` freezes on an
    LR plateau.
  - **Minimal, parameter-free API:** the common case is
    `Autofusion(params, **adafusion_kwargs)`. The empirical scaffolding that
    accumulated across iterations (`store_delta`, `s_init_rel`, `scale_floor_frac`,
    the auto-freeze `tol`/`patience`/`max_frac`) was collapsed to internal
    constants once iteration-3 validated on a real SDXL LoRA that the data-relative
    cap generalizes (val flat across `scale_cap_rel` 3–12). The only LR-equivalent
    knob, `scale_cap_rel` (default `6`), is kept but marked advanced / rarely
    needed.
  - Only per-param state while adapting is the irreducible `ref` (one extra copy of
    the weights); `Delta` is reconstructed on the fly as `(p-ref)/sum(s)`.
  - `adafusion_betas` passthrough sets the inner momentum betas (the tuner `betas`
    kwarg shadows them); all other Adafusion knobs forward through `**kwargs`.
  - **Naming:** the optimizer is `Autofusion`. (It went through the working names
    `AdafusionProdigy` — a misnomer, it is Mechanic, not Prodigy — and
    `AdaptiveAdafusion` during development; neither shipped, both are removed.)
- **KProdigy now reuses Adafusion's full update engine.** KProdigy's pass-2 weight
  update (previously a per-parameter Python loop) is now backed by Adafusion's
  foreach batching, momentum codec (`float32`/`bfloat16`/`int8`/`4bit`), cautious
  masking, conv-aware matrixized factoring, and stochastic-rounding bf16 weights —
  with Prodigy's effective learning rate (`lr × D`) folded into the update.
  KProdigy's **D-estimation (pass 1) is unchanged**: the two-pass global reduction,
  `slice_p`, `independent_d`, `d_coef` etc. produce a bit-identical D trajectory and
  final weights vs the previous release (verified on CPU fp32 across every
  dtype/second-moment combo).
  - New KProdigy args mirroring Adafusion: `momentum_dtype="4bit"`, `cautious`,
    `foreach` (default `True`), `foreach_batch_cutoff`, `foreach_stack_budget`,
    `momentum_4bit_block`.
  - The momentum codecs + quant helpers were extracted from `adafusion.py` into a
    shared `koptim._momentum_codec` module (re-exported from `koptim.adafusion` for
    backwards compatibility) — no duplicated implementations.
  - foreach == per-param: **bit-exact on fp32 weights (CPU and CUDA)** across
    `momentum_dtype ∈ {float32, bfloat16, int8, 4bit}`, cautious on/off, on 2-D +
    conv + 1-D params. New foreach-parity/4bit/cautious tests.
  - Update-backend speedup (foreach vs the old per-param loop): **~1.8× on a
    LoRA-like distribution**, **~1.5× on the SDXL full-FT distribution**. (Smaller
    than Adafusion's because KProdigy's pass-1 D-reduction is per-parameter in both
    arms — only the pass-2 update is batched.)
  - Memory on the SDXL UNet shape distribution (bytes/param): factored/4bit +
    `slice_p=11` = **1.27 B/param** (vs Adafusion 4bit 0.54, AdamW-class 8–14).
    The Prodigy D-state (`s`+`p0`) dominates at `slice_p=1`; `slice_p` is the lever.
- `Adafusion(foreach=True)` (now the default) — multi-tensor batching of the
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
- `cautious` now defaults to **`True`**. Measured on a mini pixel-DDPM (8 paired
  seeds, per-arm best LR): with momentum it lowers held-out val loss ~1.4% (paired
  t=−4.07, p<0.05); it is a literal no-op without momentum (the mask is all-ones —
  mask-active fraction ≈ 0). Set `cautious=False` for no-momentum configs to skip
  the then-useless masking op.
- Refactored Adafusion's momentum handling into a unified per-dtype **momentum
  codec** (`init_state` / `ema_one` / `ema_stacked`). The dequant → fp32 EMA →
  requant logic for each `momentum_dtype` now lives in exactly one place instead of
  being copy-pasted across the per-parameter step and the two foreach buckets; the
  step functions call `codec.ema_*`. Pure restructuring — float32/bfloat16/int8
  remain bit-for-bit identical (existing parity tests pass unchanged).

### Removed
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
  - koptim memory toolkit: `momentum_dtype` (`float32`/`bfloat16`/`int8`),
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

Initial release of `koptim` (K-Optimizers).

### Added
- `Adafusion` — conv-aware factored optimizer:
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
