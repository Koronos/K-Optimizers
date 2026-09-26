# Low-LR bf16: the weight-write methods head to head

Status: **run 2026-09-25** at four LRs (results below; raw in `results_lr*.json`, logs in `run_lr*.log`).

`benchmarks/antikaon_lowlr` measured Antikaon at a fine-tune LR; this runner keeps its
U-Net, dataset, loss, eval and fine-tune protocol (fp32 pre-training at the battery's
`lr_const`, rounded once to bf16; constant-LR fine-tune from that same start, same data order)
and drops Antikaon: it compares every `bf16_method` on two rules.

| arm | rule | weights | write | extra state |
|---|---|---|---|---|
| `ada-sr` / `nek-sr` | Adakaon-nomom / Nekaon | bf16 | stochastic rounding | 0 B/param |
| `ada-kahan` / `nek-kahan` | | bf16 | legacy `kahan` (bf16 `shift`; **per-param path only**) | 2 B/param |
| `ada-k8` / `nek-k8` | | bf16 | `kahan8` (uint8 residual, SR at ulp/256) | 1 B/param |
| `ada-k16` / `nek-k16` | | bf16 | `kahan16` (int16 residual: an fp32 master split in two) | 2 B/param |
| `ada-fp32` / `nek-fp32` | | fp32 | — (the reference of the rule) | 2 B/param more |

Rules: Adakaon-nomom = betas (0, 0.999), cautious off, bf16 momentum; Nekaon = k 1.5, betas
(0.5, 0.999), wd 0.1, 4-bit momentum (the `benchmarks/control/registry.py` configs).

Metrics per arm and seed: `test`, `train`, `gap = test - train` at the clean weights
(Nekaon without its climb), `test_start`, `test_full` (the clean iterate at its full value —
decoded residual / `p + shift` — in an fp32 copy of the net), `dist_ulp` and `dist_rel` of
that full value to the rule's fp32 arm, and **ms/step** (forward + backward + step) and
**ms/opt** (the optimizer step alone), CUDA events, first 20 steps excluded. The timings are
**orientative**: the arms run serially in one process with no control over power state or
clocks (laptop GPU: 60 W on AC, 35 W on battery — record which). Note `*-kahan` takes the
per-param path, so its ms/opt is not comparable with the foreach/fused arms as a write cost.

Expectation to check (not a result): `*-k16` should sit at `dist_ulp ≈ 0` for Adakaon-nomom
(bit-exact to fp32 given the same bf16 gradients, except Gradient Centralization, which runs
on the bf16 grad in bf16, and weight decay, which reads the bf16 weight — Nekaon has
wd 0.1), so its `test` differs from `*-fp32` only through the forward seeing the nearest bf16.

## Run

From the repo root (the worktree), GPU plugged in:

```
PYTHONPATH=src python benchmarks/lowlr_bf16/run_lowlr.py --smoke          # plumbing (6 steps, all arms)
PYTHONPATH=src python benchmarks/lowlr_bf16/run_lowlr.py                  # 10 arms x seeds 0,1, 8000 steps, lr 1e-5
PYTHONPATH=src python benchmarks/lowlr_bf16/run_lowlr.py --lr 1e-5 --seeds 0,1 --steps 8000 \
    --arms ada-sr,ada-k8,ada-k16,ada-fp32
```

Flags: `--lr`, `--seeds`, `--steps`, `--arms` (comma-separated keys), `--pretrain`, `--C`,
`--out`. Output: `results.json` (`smoke.json` under `--smoke`) + a summary table. The fp32
pre-training is cached per `(C, bs, n, seed)` in `cache/` (git-ignored).


## Results (2026-09-25, AC power; ms columns orientative — CPU was shared with simulations)

8000 constant-LR fine-tune steps from the same bf16 start. `dist_ulp` = distance of the full
value to the rule's fp32 arm (lower = closer to the fp32 trajectory). Median step/ulp from
`--probe` at 1e-5: p50 0.06 (so ≈0.02 at 3e-6, ≈0.6 at 1e-4, ≈1.8 at 3e-4).

| lr (seeds) | arm | SR test / dist | kahan (legacy) | kahan8 | kahan16 | fp32 test |
|---|---|---|---|---|---|---|
| 3e-6 (0) | Adakaon | 0.09236 / 5.76 | 0.09230 / 0.061 | 0.09229 / 0.112 | 0.09228 / 0.043 | 0.09226 |
| 3e-6 (0) | Nekaon | 0.09215 / 4.78 | 0.09216 / 0.149 | 0.09215 / 0.179 | 0.09215 / 0.067 | 0.09210 |
| 1e-5 (0,1) | Adakaon | 0.08846 / 10.47 | 0.08836 / 0.092 | 0.08836 / 0.134 | 0.08836 / 0.083 | 0.08832 |
| 1e-5 (0,1) | Nekaon | 0.08791 / 8.65 | 0.08791 / 1.200 | 0.08790 / 0.256 | 0.08788 / 0.192 | 0.08785 |
| 1e-4 (0,1) | Adakaon | 0.08167 / 24.0 | — | 0.08169 / 0.44 | 0.08154 / 0.42 | 0.08163 |
| 1e-4 (0,1) | Nekaon | 0.08083 / 22.0 | — | 0.08039 / 2.28 | 0.08034 / 2.26 | 0.08035 |
| 3e-4 (0,1) | Adakaon | 0.08297 / 29.0 | — | 0.08278 / 1.35 | 0.08280 / 1.30 | 0.08267 |
| 3e-4 (0,1) | Nekaon | 0.07755 / 28.3 | — | 0.07775 / 7.49 | 0.07782 / 7.36 | 0.07785 |

Reading:
- The compensated methods track the fp32 trajectory 20–100× closer than SR at every LR.
- In **test loss** the differences are at or below the seed spread (sd 0.0003–0.003). At low LR
  SR is consistently a hair worse (≈+0.0001 at 1e-5/3e-6); at 1e-4 Nekaon-SR is +0.0005 worse
  (sd 0.0016); at 3e-4 Nekaon-SR is 0.0003 *better* than fp32 (rounding noise acting as a mild
  regularizer; sd 0.0003). On this proxy the write method does not change quality measurably.
- kahan8 ≈ kahan16 everywhere; kahan16 is slightly closer to fp32. Legacy `kahan` adds nothing
  (1.2 ulp with Nekaon at 1e-5 because its climb bypasses the shift; 1.6–2× slower).
- dist grows with LR for Nekaon even with kahan16 (2.3 / 7.4 ulp): chaotic amplification of the
  remaining non-fp32 bits (decay and GC read the bf16 weight), not write error.
