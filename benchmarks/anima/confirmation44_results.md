# Anima / Pets pilot

Seed 44, constant LR 0.0001, rank-16 LoRA, 200 steps at 256px.
32 fixed images per evaluation split, nine noise quantiles.
Initialization fingerprints match. This is a pilot, not a tuned ranking.

| Optimizer | Train eval | Val | Raw gap | Change in gap | Active train s | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| rakaon_isotropic | 0.142925 | 0.133323 | -0.009603 | +0.001353 | 175.9 | 4.624 |
| nekaon | 0.142332 | 0.132850 | -0.009482 | +0.001473 | 211.3 | 4.652 |

Raw gap = val − train eval. Change in gap subtracts the step-zero gap;
it is descriptive, not a generalization bound. Active time excludes evaluation
and previews; allocator peaks include prior preview allocations. Run order and
laptop thermals can affect timing. No FID, perceptual ranking, multi-seed
confirmation within this single-seed report, or full fine-tuning claim follows from this pilot.
