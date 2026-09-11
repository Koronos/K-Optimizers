# Anima / Pets pilot

Seed 45, constant LR 1e-05, rank-16 LoRA, 100 steps at 1024px.
8 fixed images per evaluation split, nine noise quantiles.
Initialization fingerprints match. This is a pilot, not a tuned ranking.

| Optimizer | Train eval | Val | Raw gap | Change in gap | Active train s | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| nekaon_k0 | 0.097235 | 0.109881 | +0.012645 | +0.000086 | 664.9 | 5.890 |
| nekaon | 0.097239 | 0.109871 | +0.012632 | +0.000073 | 2540.0 | 5.899 |
| nekaon_sr_host | 0.097217 | 0.109877 | +0.012660 | +0.000101 | 2601.2 | 5.899 |

Raw gap = val − train eval. Change in gap subtracts the step-zero gap;
it is descriptive, not a generalization bound. Active time excludes evaluation
and previews; allocator peaks include prior preview allocations. Run order and
laptop thermals can affect timing. No FID, perceptual ranking, multi-seed
confirmation within this single-seed report, or full fine-tuning claim follows from this pilot.
