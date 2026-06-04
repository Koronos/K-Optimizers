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
  - **Full fine-tune: ~1.3×** (SDXL UNet, 1680 params: 353 ms → 279 ms). The
    smaller win is expected — a full fine-tune's optimizer time is dominated by
    real bandwidth work on large weights, where per-tensor launch overhead is
    noise; batching only removes that overhead, which dominates in the
    many-small-tensors (adapter) regime.
  - Element-for-element equal to the per-parameter path (bit-exact on CPU; ~1e-8
    on CUDA from reduction order). Stochastic-rounding draws legitimately differ
    (unbiased either way). 6 new parity/coverage tests.
  - **Bucket chunking** (`_MAX_STACK_ELEMS`) bounds the transient stack memory so
    batching never OOMs a full fine-tune — preserving Adafusion's low-memory story.
    Large weights (which can't share a stack and are bandwidth-bound) route to the
    per-param loop to skip stack/copy overhead.
  - Falls back to the per-parameter path for what it doesn't batch: 0-D scalars,
    large weights, `momentum_dtype="int8"`, `bf16_method="kahan"`, fp16+SR,
    non-contiguous matrixized convs, and single-param (gradient-release)
    optimizers. For eligible params it supersedes `compile` (no per-tensor graph).

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
