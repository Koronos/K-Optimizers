# SR lookahead: controlled comparison and historical context

The current report validates matching initialization, initial metrics and complete training/evaluation records.
Differences below are candidate minus control; negative is better for each listed metric.

| Control | Delta val | Delta abs gap | Delta active seconds | Delta peak GiB |
|---|---:|---:|---:|---:|
| nekaon_k0 | -0.00000364 | +0.00001458 | +1936.26 | +0.009 |
| nekaon | +0.00000550 | +0.00002729 | +61.21 | +0.000 |

## Historical results — different protocols, not a paired ranking

| Report | Arm | Seed | Pixels | LR | Steps | Eval images/split | Val | Abs gap | Active s | Peak GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current | nekaon_k0 | 45 | 1024 | 1e-05 | 100 | 8 | 0.10988063 | 0.01264515 | 664.92 | 5.890 |
| current | nekaon | 45 | 1024 | 1e-05 | 100 | 8 | 0.10987150 | 0.01263244 | 2539.96 | 5.899 |
| current | nekaon_sr_host | 45 | 1024 | 1e-05 | 100 | 8 | 0.10987700 | 0.01265973 | 2601.18 | 5.899 |
| confirmation43 | nekaon | 43 | 256 | 0.0001 | 200 | 32 | 0.13259706 | 0.00901842 | 195.96 | 4.652 |
| confirmation43 | rakaon_isotropic | 43 | 256 | 0.0001 | 200 | 32 | 0.13306728 | 0.00933479 | 181.43 | 4.624 |
| confirmation44 | rakaon_isotropic | 44 | 256 | 0.0001 | 200 | 32 | 0.13332285 | 0.00960261 | 175.92 | 4.624 |
| confirmation44 | nekaon | 44 | 256 | 0.0001 | 200 | 32 | 0.13285008 | 0.00948192 | 211.35 | 4.652 |
| adam43 | adamw_fused | 43 | 256 | 0.0001 | 200 | 32 | 0.13357155 | 0.00929528 | 160.46 | 4.763 |
| adam44 | adamw_fused | 44 | 256 | 0.0001 | 200 | 32 | 0.13404396 | 0.00960796 | 153.11 | 4.763 |
| fast43 | rakaon_isotropic | 43 | 256 | 0.0002 | 100 | 32 | 0.13574749 | 0.00961237 | 85.43 | 4.624 |
| momentum43 | rakaon_m05 | 43 | 256 | 0.0001 | 200 | 32 | 0.13296953 | 0.00915758 | 204.13 | 4.763 |
| gram43_lr0 | gram_d001 | 43 | 256 | 0.001 | 200 | 32 | 0.13367306 | 0.01083502 | 301.04 | 4.624 |
| gram43_lr0 | gram_d01 | 43 | 256 | 0.001 | 200 | 32 | 0.13474713 | 0.01081391 | 407.46 | 4.624 |
| gram43_lr1 | gram_d001 | 43 | 256 | 0.01 | 200 | 32 | 0.13300622 | 0.01052557 | 468.99 | 4.624 |
| gram43_lr1 | gram_d01 | 43 | 256 | 0.01 | 200 | 32 | 0.13282059 | 0.01048958 | 346.07 | 4.624 |

Resolution, LR, seed, number of steps and evaluation subsets differ across reports. Do not infer a quality gain or slowdown from historical absolute loss/time differences.
This single-seed screen provides no uncertainty estimate or perceptual-detail measurement. Active time excludes evaluation; laptop thermals and run order can affect timing. An absolute gap can shrink because training deteriorates, so inspect validation and train loss together.
