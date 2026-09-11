# Anima / Pets pilot

Seed 43, constant LR 0.01, rank-16 LoRA, 200 steps at 256px.
32 fixed images per evaluation split, nine noise quantiles.
Initialization fingerprints match. This is a pilot, not a tuned ranking.

| Optimizer | Train eval | Val | Raw gap | Change in gap | Active train s | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| gram_d001 | 0.143532 | 0.133006 | -0.010526 | +0.000430 | 469.0 | 4.624 |
| gram_d01 | 0.143310 | 0.132821 | -0.010490 | +0.000466 | 346.1 | 4.624 |

Raw gap = val − train eval. Change in gap subtracts the step-zero gap;
it is descriptive, not a generalization bound. Active time excludes evaluation
and previews; allocator peaks include prior preview allocations. Run order and
laptop thermals can affect timing. No FID, perceptual ranking, multi-seed
confirmation within this single-seed report, or full fine-tuning claim follows from this pilot.
