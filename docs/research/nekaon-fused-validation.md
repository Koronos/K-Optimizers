# Nekaon diagnostic optimization and fused validation

## Implemented change

MSAM/Nekaon's sampled inactivity warning now runs every ten climbs by default,
instead of every climb, within the existing 200-climb observation window. This
reduces diagnostic weight reads from at most 200 to at most 20 per visited group.
`inert_check_interval=1` restores the previous cadence. The warning requires
sampled evidence spanning at least 50 climbs; intervening transient changes can
be missed. It remains a heuristic and its message no longer claims every weight
is immovable from a sampled mean. No update, perturbation, momentum or checkpoint
tensor changes are introduced by this diagnostic setting.

Seven targeted diagnostic tests pass, including exact FP32 parameter trajectory
equality with independent nonconstant gradients, cadence/read limits and input
validation. The related Nekaon/MSAM/climb suite passed before the extra cadence
test (58 passed, one opt-in timing test skipped, Python 3.10).

Six selected fused tests passed in the training runtime (Python 3.13), covering
LoRA FP32 parity, chunked 4-bit parity, batched quantized states, BF16 stochastic
rounding, and cache invalidation on state load. Fused/native stochastic BF16 paths
are not promised to yield bitwise-identical training trajectories.

## Using the existing optimized Adakaon route

Explicitly set `fused=True` in either `Adakaon` or `Nekaon` to enable Adakaon's
existing fused implementation. The public library default remains unchanged:
performance depends on backend, shapes and full-model memory residency. The
current work optimizes Nekaon's diagnostic and validates this existing Adakaon
configuration; it does not introduce a new Adakaon update kernel.

The reproducible real-model diagnostic accepts `--fused`:

```text
python -m benchmarks.anima.run_lookahead_timing SOURCE.toml NEW_OUTPUT_DIR --fused
```

This runs k=0, k=1.5, k=1.5, k=0 with four steps each, at the source resolution,
without evaluation. k=0 exercises Nekaon's inner Adakaon without perturbations.
Optimizer timing has explicit CUDA synchronization. This is a compatibility and
performance diagnostic, not a convergence comparison. The first step is excluded
from medians but retained in the raw record.

All four fused runs at 1024px completed and saved checkpoints, each with 5.91 GiB
peak allocator memory. All initial adapter SHA256 fingerprints matched.

| Order | k | Median optimizer ms | Median full step seconds |
|---|---:|---:|---:|
| 1 | 0 | 24.143 | 29.838 |
| 2 | 1.5 | 26.356 | 28.048 |
| 3 | 1.5 | 27.099 | 29.733 |
| 4 | 0 | 25.672 | 35.231 |

The earlier native ABBA runs measured 1.6–1.8 seconds inside the optimizer and
25–27 seconds per full step. Fused reduces the measured optimizer component in
these experiments, but full-step speed did not improve. These separate sequential
experiments cannot establish the cause of the longer non-optimizer component.
Do not claim a global speedup, convergence benefit, or a quality ranking.

The four-step runs do not reach the first scheduled diagnostic at step 10; the
cadence and eventual warning are covered by the unit tests. The optimized isolated
benchmark retains all measurements in `optimizer_hotpath_optimized_results.json`;
short wall-clock timings fluctuate and should not override the structural result
of 90% fewer diagnostic reads. Its profiled step 26 is unsampled by design.

Raw training timings: `benchmarks/anima/lookahead_fused_results.json`. Original
configs and logs remain in `tmp/anima-lookahead-fused/`.
