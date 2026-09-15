# Lookahead timing control at 1024px

Four fresh Anima/Pets runs completed, including checkpoint writes. The order was
k=0, k=1.5, k=1.5, k=0. Each used the original 1024px Pets training configuration,
seed 45, constant LR 1e-5, BF16 rank-16 LoRA, four steps and no evaluation.
CUDA synchronization surrounded optimizer timing. The first step is excluded
from each median below. This is a short diagnostic, not a stable throughput estimate.

| Order | k | Median total step seconds | Median optimizer seconds |
|---|---:|---:|---:|
| 1 | 0 | 25.721 | 1.634 |
| 2 | 1.5 | 25.272 | 1.609 |
| 3 | 1.5 | 25.498 | 1.716 |
| 4 | 0 | 26.579 | 1.750 |

The nearly fourfold speed advantage previously observed for k=0 did not
reproduce. Do not attribute that earlier wall-clock difference to lookahead.
The cause of the earlier fast run remains unknown: these measurements do not
identify thermal, scheduling, memory-residency or other system effects. Telemetry
was collected before each run, not continuously during training. The diagnostic
also differs from the quality run by explicit timing synchronization and omission
of evaluation. No statistically established speed winner follows from four steps.

## Quality result remains negative for the proposed fix

The completed 100-step, single-seed BF16 host-restored stochastic lookahead screen
matched all initial adapter fingerprints and initial evaluation losses. Final val
loss was 0.10987700 versus ordinary Nekaon's 0.10987150; absolute gap was
0.01265973 versus 0.01263244. The fix used the same observed 5.899 GiB CUDA peak,
plus 66.063 MiB host snapshot storage. These small quality differences do not
establish significance or a benefit. None met val <0.07 and abs gap <0.007;
the initial absolute gap was already 0.01255923.

The implementation solves numerical perturbation/restoration in BF16, but this
screen does not justify promoting it to a default optimizer. No perceptual-detail
or long-run benefit has been established. Historical 256px results remain separate
because resolution, LR, seed, steps and evaluation subset changed.

Artifacts: `benchmarks/anima/lookahead_timing_results.json`,
`benchmarks/anima/sr1024_results.json`, `benchmarks/anima/sr1024_history.md`.
Reproduction: `benchmarks/anima/run_lookahead_timing.py`; original diagnostic
configs/logs are retained in `tmp/anima-lookahead-abba/`.
