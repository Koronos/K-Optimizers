# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
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

### Deprecated
- `Adafusion(compile=...)` is now a no-op. `torch.compile` of the per-tensor
  factored core measured neutral-to-negative across model sizes and is superseded
  by `foreach` batching; the argument is still accepted so existing configs don't
  break, but does nothing (the compiled path and `koptim/_compiled.py` were removed).

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
