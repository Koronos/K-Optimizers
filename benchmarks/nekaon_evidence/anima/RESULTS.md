# Nekaon vs Adakaon vs AdamW on Anima / Pets

Rank-16 LoRA on Cosmos Predict2 (Anima), bfloat16, Oxford-IIIT Pets subset at 256 px, constant LR, 200 steps, evaluation before the first step and every 100 steps on eight fixed images per split at nine noise quantiles. Previews off. Runs are paired within a seed: the adapter initialization is seeded before the trainer starts and the initial SHA-256 is required to match across arms.

kaon 0.7.14 at commit 0814c7025d6a4a5707940d17ca14878f79235674.

## Read this before the numbers

- **Confound.** Every arm runs its own house configuration and only the learning rate is tuned: the kaon arms use betas (0.5, 0.999), weight decay 0.1, cautious updates, gradient centralization, 4-bit momentum and stochastic rounding, while `adamw_fused` uses betas (0.9, 0.999), weight decay 0.01 and none of those. A difference between AdamW and a kaon arm is therefore NOT attributable to the algorithm alone; the table of `[optimizer]` blocks in this report is the exact confound.
- **Boundary.** No arm selected a learning rate at an end of its screened grid, so every selected LR is bracketed on both sides.
- **Telemetry.** The `nekaon_fused` arm (k = 1.5) pays MSAM's inert-lookahead telemetry that `adakaon_fused` and `adamw_fused` do not: `_warn_if_inert` stays armed for its first 200 climbs and samples every 10th, so a 200-step run here is instrumented end to end and performs roughly twenty device-to-host synchronizations. That cost is real for a user running this configuration today, but it is telemetry rather than optimizer arithmetic, and it counts against Nekaon in active seconds and ms/step.

LR selection rule (fixed before the data was seen): per arm, the phase-A learning rate with the lowest final val/loss on the phase-A seed; exact ties go to the smaller learning rate.

Selected learning rates: `adakaon_fused` 2.5e-05, `adamw_fused` 2.5e-05, `nekaon_fused` 5e-05.

## Arm configurations as run

Every arm runs its own house configuration and only the learning rate is tuned: the kaon arms use betas (0.5, 0.999), weight decay 0.1, cautious updates, gradient centralization, 4-bit momentum and stochastic rounding, while `adamw_fused` uses betas (0.9, 0.999), weight decay 0.01 and none of those. A difference between AdamW and a kaon arm is therefore NOT attributable to the algorithm alone; the table of `[optimizer]` blocks in this report is the exact confound.

Read verbatim from each run's own TOML.

| `[optimizer]` key | nekaon_fused | adakaon_fused | adamw_fused |
| --- | --- | --- | --- |
| type | `kaon.Nekaon` | `kaon.Adakaon` | `torch.optim.AdamW` |
| lr | 5e-05 | 2.5e-05 | 2.5e-05 |
| k | 1.5 | — | — |
| betas | [0.5, 0.999] | [0.5, 0.999] | [0.9, 0.999] |
| weight_decay | 0.1 | 0.1 | 0.01 |
| momentum_dtype | `4bit` | `4bit` | — |
| bf16_method | `stochastic_rounding` | `stochastic_rounding` | — |
| cautious | true | true | — |
| gradient_centralization | true | true | — |
| auto_lr | false | false | — |
| fused | true | true | true |
| clip_threshold | — | 1 | — |
| eps | — | — | 1e-08 |

## Phase A — learning-rate screen (seed 43)

| Arm | LR | Selected | Final val | Raw gap | Active s | ms/step | Peak GiB |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| adakaon_fused | 1.25e-05 |  | 0.128187 | -0.011521 | 166.7 | 833.6 | 4.665 |
| adakaon_fused | 2.5e-05 | yes | 0.128082 | -0.011350 | 155.5 | 777.3 | 4.665 |
| adakaon_fused | 5e-05 |  | 0.128692 | -0.010680 | 170.6 | 853.1 | 4.665 |
| adakaon_fused | 0.0001 |  | 0.128461 | -0.010135 | 180.9 | 904.6 | 4.665 |
| adakaon_fused | 0.0002 |  | 0.130237 | -0.009871 | 164.9 | 824.7 | 4.665 |
| adamw_fused | 1.25e-05 |  | 0.128723 | -0.011600 | 157.3 | 786.6 | 4.763 |
| adamw_fused | 2.5e-05 | yes | 0.128309 | -0.011223 | 154.9 | 774.5 | 4.763 |
| adamw_fused | 5e-05 |  | 0.128554 | -0.011322 | 156.4 | 781.8 | 4.763 |
| adamw_fused | 0.0001 |  | 0.129885 | -0.010727 | 168.6 | 843.1 | 4.763 |
| adamw_fused | 0.0002 |  | 0.133015 | -0.009621 | 156.6 | 783.1 | 4.763 |
| nekaon_fused | 1.25e-05 |  | 0.128261 | -0.011525 | 172.4 | 862.0 | 4.665 |
| nekaon_fused | 2.5e-05 |  | 0.128160 | -0.011397 | 159.6 | 798.1 | 4.665 |
| nekaon_fused | 5e-05 | yes | 0.128102 | -0.011359 | 161.0 | 805.2 | 4.665 |
| nekaon_fused | 0.0001 |  | 0.128386 | -0.010653 | 180.9 | 904.7 | 4.665 |
| nekaon_fused | 0.0002 |  | 0.129132 | -0.010235 | 165.5 | 827.6 | 4.665 |

## Per-arm means

Mean with a 95% percentile bootstrap interval (10000 resamples, seed 20260918).

Seed 43 chose the learning rates, so it is shown second: the held-out seeds carry no selection bias, while the full paired set is larger but optimistic by construction.

### Held-out seeds, no selection bias (44, 45, 46, 47)

| Arm | LR | Final val eps-MSE | Raw gap (val - train_eval) | Active train seconds | ms / step | Peak allocator GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nekaon_fused | 5e-05 | 0.128383 [0.128197, 0.128569] | -0.010687 [-0.010951, -0.010423] | 168.4 [166.1, 170.6] | 841.8 [830.5, 853.0] | 4.665 [4.665, 4.665] |
| adakaon_fused | 2.5e-05 | 0.128381 [0.128254, 0.128495] | -0.011150 [-0.011321, -0.010968] | 161.9 [159.8, 164.4] | 809.8 [799.1, 822.2] | 4.665 [4.665, 4.665] |
| adamw_fused | 2.5e-05 | 0.128657 [0.128620, 0.128701] | -0.011182 [-0.011336, -0.011082] | 161.7 [156.1, 169.8] | 808.6 [780.4, 848.8] | 4.763 [4.763, 4.763] |

### All 5 paired seeds, includes the LR-selection seed (43, 44, 45, 46, 47)

| Arm | LR | Final val eps-MSE | Raw gap (val - train_eval) | Active train seconds | ms / step | Peak allocator GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nekaon_fused | 5e-05 | 0.128327 [0.128139, 0.128516] | -0.010821 [-0.011135, -0.010508] | 166.9 [163.6, 170.1] | 834.4 [818.0, 850.4] | 4.665 [4.665, 4.665] |
| adakaon_fused | 2.5e-05 | 0.128321 [0.128174, 0.128460] | -0.011190 [-0.011334, -0.011030] | 160.7 [157.7, 163.4] | 803.3 [788.3, 816.9] | 4.665 [4.665, 4.665] |
| adamw_fused | 2.5e-05 | 0.128588 [0.128446, 0.128686] | -0.011190 [-0.011303, -0.011100] | 160.4 [155.8, 166.8] | 801.8 [779.0, 834.1] | 4.763 [4.763, 4.763] |

## Paired differences by seed

Held-out first, then the full paired set, over the same metrics.

### Held-out seeds, no selection bias (44, 45, 46, 47)

#### nekaon_fused − adamw_fused

A negative difference favours nekaon_fused on every metric here (all are lower-is-better).

| Metric | Mean difference [95% CI] | Seeds won by nekaon_fused | Interval crosses zero |
| --- | ---: | ---: | :---: |
| Final val eps-MSE | -0.000274 [-0.000470, -0.000094] | 4/4 | no |
| Raw gap (val - train_eval) | +0.000495 [+0.000313, +0.000678] | 0/4 | no |
| Active train seconds | 6.6 [1.5, 10.9] | 1/4 | no |
| ms / step | 33.2 [7.3, 54.6] | 1/4 | no |
| Peak allocator GiB | -0.098 [-0.098, -0.098] | 4/4 | no |

#### nekaon_fused − adakaon_fused

A negative difference favours nekaon_fused on every metric here (all are lower-is-better).

| Metric | Mean difference [95% CI] | Seeds won by nekaon_fused | Interval crosses zero |
| --- | ---: | ---: | :---: |
| Final val eps-MSE | 0.000002 [-0.000227, 0.000154] | 1/4 | yes |
| Raw gap (val - train_eval) | +0.000463 [+0.000340, +0.000684] | 0/4 | no |
| Active train seconds | 6.4 [1.9, 9.8] | 0/4 | no |
| ms / step | 32.0 [9.5, 48.8] | 0/4 | no |
| Peak allocator GiB | 0.000 [0.000, 0.000] | 0/4 | yes |

### All 5 paired seeds, includes the LR-selection seed (43, 44, 45, 46, 47)

#### nekaon_fused − adamw_fused

A negative difference favours nekaon_fused on every metric here (all are lower-is-better).

| Metric | Mean difference [95% CI] | Seeds won by nekaon_fused | Interval crosses zero |
| --- | ---: | ---: | :---: |
| Final val eps-MSE | -0.000260 [-0.000424, -0.000111] | 5/5 | no |
| Raw gap (val - train_eval) | +0.000369 [+0.000097, +0.000618] | 1/5 | no |
| Active train seconds | 6.5 [2.4, 10.3] | 1/5 | no |
| ms / step | 32.7 [12.0, 51.4] | 1/5 | no |
| Peak allocator GiB | -0.098 [-0.098, -0.098] | 5/5 | no |

#### nekaon_fused − adakaon_fused

A negative difference favours nekaon_fused on every metric here (all are lower-is-better).

| Metric | Mean difference [95% CI] | Seeds won by nekaon_fused | Interval crosses zero |
| --- | ---: | ---: | :---: |
| Final val eps-MSE | 0.000006 [-0.000171, 0.000133] | 1/5 | yes |
| Raw gap (val - train_eval) | +0.000369 [+0.000139, +0.000617] | 1/5 | no |
| Active train seconds | 6.2 [3.0, 8.9] | 0/5 | no |
| ms / step | 31.2 [15.2, 44.6] | 0/5 | no |
| Peak allocator GiB | 0.000 [0.000, 0.000] | 0/5 | yes |

## Pairing validation

All seeds pass: **yes**. Within each seed the arms must share the initial adapter SHA-256 and agree on the step-zero losses to 1e-06.

| Seed | Arms | Adapter SHA matches | Initial train_eval | Initial val | OK |
| ---: | --- | :---: | ---: | ---: | :---: |
| 43 | adakaon_fused, adamw_fused, nekaon_fused | yes | 0.150594 | 0.137754 | yes |
| 44 | adakaon_fused, adamw_fused, nekaon_fused | yes | 0.150594 | 0.137754 | yes |
| 45 | adakaon_fused, adamw_fused, nekaon_fused | yes | 0.150594 | 0.137754 | yes |
| 46 | adakaon_fused, adamw_fused, nekaon_fused | yes | 0.150594 | 0.137754 | yes |
| 47 | adakaon_fused, adamw_fused, nekaon_fused | yes | 0.150594 | 0.137754 | yes |

## Timings and electrical state

The GPU is power limited to 60 W on AC and 35 W on battery. Runs under different signatures are NOT comparable on active seconds or ms/step; quality metrics are unaffected.

The `nekaon_fused` arm (k = 1.5) pays MSAM's inert-lookahead telemetry that `adakaon_fused` and `adamw_fused` do not: `_warn_if_inert` stays armed for its first 200 climbs and samples every 10th, so a 200-step run here is instrumented end to end and performs roughly twenty device-to-host synchronizations. That cost is real for a user running this configuration today, but it is telemetry rather than optimizer arithmetic, and it counts against Nekaon in active seconds and ms/step.

Timings comparable across all runs: **yes**.

| Power signature | Runs |
| --- | ---: |
| `ac/max_limit=60.00 W` | 35 |

## What this does not show

- **The arms are not matched.** Every arm runs its own house configuration and only the learning rate is tuned: the kaon arms use betas (0.5, 0.999), weight decay 0.1, cautious updates, gradient centralization, 4-bit momentum and stochastic rounding, while `adamw_fused` uses betas (0.9, 0.999), weight decay 0.01 and none of those. A difference between AdamW and a kaon arm is therefore NOT attributable to the algorithm alone; the table of `[optimizer]` blocks in this report is the exact confound.
- 5 paired seeds is a small sample. The bootstrap intervals are wide by construction and an interval that crosses zero means the campaign did not separate the arms on that metric, not that they are equal.
- Validation loss here is epsilon-MSE on eight held-out images at nine fixed noise quantiles. It is not FID, not a perceptual score, and not a measure of image quality.
- The raw gap is descriptive. Eight images per split cannot estimate population generalization, and the split does not establish exclusion from Anima's pretraining data.
- Timings come from a single power-limited laptop GPU, one run per configuration, with thermal drift only partly mitigated by interleaving the arms. Peak allocator memory includes allocations from earlier in the same process. The Nekaon arm also carries the inert-lookahead telemetry described above.
- Optimizer state bytes are not measured directly: the trainer reports the allocator peak and the optimizer parameter count, not a per-optimizer state accounting.
- 200 steps of a rank-16 LoRA on 96 images is a screening protocol. Nothing here transfers automatically to full fine-tuning, other models, or other datasets.
