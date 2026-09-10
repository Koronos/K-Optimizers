# Anima paired confirmation

Descriptive same-seed comparison; this is not a universal optimizer ranking.

Inputs: confirmation43_results.json, confirmation44_results.json, adam43_results.json, adam44_results.json

| Seed | Arm | Source | Final step | Val | Raw gap | Active seconds | Peak GiB |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 43 | adamw_fused | adam43_results.json | 200 | 0.133572 | -0.00929528 | 160.458 | 4.763 |
| 43 | nekaon | confirmation43_results.json | 200 | 0.132597 | -0.00901842 | 195.96 | 4.652 |
| 43 | rakaon_isotropic | confirmation43_results.json | 200 | 0.133067 | -0.00933479 | 181.433 | 4.624 |
| 44 | adamw_fused | adam44_results.json | 200 | 0.134044 | -0.00960796 | 153.107 | 4.763 |
| 44 | nekaon | confirmation44_results.json | 200 | 0.13285 | -0.00948192 | 211.349 | 4.652 |
| 44 | rakaon_isotropic | confirmation44_results.json | 200 | 0.133323 | -0.00960261 | 175.924 | 4.624 |

Validation: model, dataset, protocol, initial adapter SHA, and initial losses were checked within each seed.
