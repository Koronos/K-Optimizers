# Anima / Pets pilot

Seed 42, constant LR 0.0001, rank-16 LoRA, 200 steps at 256px.
Eight fixed images per evaluation split, nine noise quantiles.
Initialization fingerprints match. This is a pilot, not a tuned ranking.

| Optimizer | Train eval | Val | Raw gap | Change in gap | Active train s | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| nekaon | 0.139010 | 0.129074 | -0.009936 | +0.002904 | 272.1 | 5.794 |
| adakaon | 0.139430 | 0.129599 | -0.009831 | +0.003010 | 281.8 | 5.794 |
| rakaon_isotropic | 0.139460 | 0.129149 | -0.010311 | +0.002529 | 255.0 | 5.767 |
| rakaon_block64 | 0.139420 | 0.129573 | -0.009847 | +0.002994 | 261.8 | 5.768 |

Raw gap = val − train eval. Change in gap subtracts the step-zero gap;
it is descriptive, not a generalization bound. Active time excludes evaluation
and previews; allocator peaks include prior preview allocations. Run order and
laptop thermals can affect timing. No FID, perceptual ranking, multi-seed
confirmation, or full fine-tuning claim follows from this pilot.
