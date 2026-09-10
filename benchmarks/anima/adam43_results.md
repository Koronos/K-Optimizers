# Anima / Pets pilot

Seed 43, constant LR 0.0001, rank-16 LoRA, 200 steps at 256px.
32 fixed images per evaluation split, nine noise quantiles.
Initialization fingerprints match. This is a pilot, not a tuned ranking.

| Optimizer | Train eval | Val | Raw gap | Change in gap | Active train s | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| adamw_fused | 0.142867 | 0.133572 | -0.009295 | +0.001660 | 160.5 | 4.763 |

Raw gap = val − train eval. Change in gap subtracts the step-zero gap;
it is descriptive, not a generalization bound. Active time excludes evaluation
and previews; allocator peaks include prior preview allocations. Run order and
laptop thermals can affect timing. No FID, perceptual ranking, multi-seed
confirmation within this single-seed report, or full fine-tuning claim follows from this pilot.
