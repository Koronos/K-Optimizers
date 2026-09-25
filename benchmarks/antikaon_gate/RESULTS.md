# Antikaon quality gate — control battery, constant LR

Only quality (held-out loss, train-val gap) is scored; ms/step is NOT measured or reported (GPU shared with another job during this run).

## Manifest

- commit: `bc22a1e63a87d34de7ecb78e8497ec107887e9fa` (branch `feature/antikaon`)
- dataset fingerprint (sha256, proxy/dataset.py): `b0bad0f5fb622dd7…` (identical for stage 1 and stage 2 — fixed synthetic dataset, seed-independent)
- proxy weight dtype: `torch.float32` (fp32 — the harness always trains at fp32; Antikaon's `bf16_method` stochastic-rounding path is NOT exercised by this gate, only its noise/shaping math)
- stage 1: C=40, N=600, seeds=[0], bs=8, schedule=const
- stage 2: C=40, N=2000, seeds=[0, 1], bs=8, schedule=const
- LR (constant, per arm): A0/A1/B*/C* = 1.2e-3 (registry `lr_const`); B2 lr x0.5 = 6e-4; B2 lr x2 = 2.4e-3
- gate reference: test <= 0.07 and gap <= 0.007

## Stage 1 (C=40, N=600, seed=0 — divergence/underfit screen)

| arm | held-out loss (mean +- std) | train-val gap (mean +- std) | B/param | per-seed |
|---|---:|---:|---:|---|
| A0 Adakaon-nomom | 0.09747 (+-0.00000) | +0.00356 (+-0.00000) | 0.103 | s0: te=0.09747 gap=+0.00356 |
| A1 Nekaon (defaults) | 0.09219 (+-0.00000) | +0.00572 (+-0.00000) | 0.634 | s0: te=0.09219 gap=+0.00572 |
| B1 k_sigma=1.5 | 0.09131 (+-0.00000) | +0.00435 (+-0.00000) | 0.103 | s0: te=0.09131 gap=+0.00435 |
| B2 k_sigma=5 | 0.09544 (+-0.00000) | +0.00353 (+-0.00000) | 0.103 | s0: te=0.09544 gap=+0.00353 |
| B3 k_sigma=15 | 0.10598 (+-0.00000) | +0.00120 (+-0.00000) | 0.103 | s0: te=0.10598 gap=+0.00120 |
| C1 shape=none | 0.09619 (+-0.00000) | +0.00350 (+-0.00000) | 0.103 | s0: te=0.09619 gap=+0.00350 |
| C2 antithetic | 0.09309 (+-0.00000) | +0.00296 (+-0.00000) | 0.103 | s0: te=0.09309 gap=+0.00296 |
| C3 sigma_ref=weight | 0.09537 (+-0.00000) | +0.00360 (+-0.00000) | 0.115 | s0: te=0.09537 gap=+0.00360 |
| B2 lr x0.5 | 0.09469 (+-0.00000) | +0.00381 (+-0.00000) | 0.103 | s0: te=0.09469 gap=+0.00381 |
| B2 lr x2 | 0.09453 (+-0.00000) | +0.00226 (+-0.00000) | 0.103 | s0: te=0.09453 gap=+0.00226 |

## Stage 2 (C=40, N=2000, seeds=0,1 — late-overfit exposure)

| arm | held-out loss (mean +- std) | train-val gap (mean +- std) | B/param | per-seed |
|---|---:|---:|---:|---|
| A0 Adakaon-nomom | 0.08001 (+-0.00499) | +0.01127 (+-0.00023) | 0.103 | s0: te=0.08353 gap=+0.01111; s1: te=0.07648 gap=+0.01143 |
| A1 Nekaon (defaults) | 0.07496 (+-0.00024) | +0.01168 (+-0.00033) | 0.634 | s0: te=0.07513 gap=+0.01191; s1: te=0.07478 gap=+0.01144 |
| B0 k_sigma=0.5 | 0.07988 (+-0.00458) | +0.01230 (+-0.00011) | 0.103 | s0: te=0.08313 gap=+0.01238; s1: te=0.07664 gap=+0.01222 |
| B1 k_sigma=1.5 | 0.07843 (+-0.00223) | +0.01106 (+-0.00071) | 0.103 | s0: te=0.08000 gap=+0.01056; s1: te=0.07685 gap=+0.01156 |
| B2 k_sigma=5 | 0.07967 (+-0.00315) | +0.01009 (+-0.00141) | 0.103 | s0: te=0.08190 gap=+0.00909; s1: te=0.07745 gap=+0.01109 |
| B3 k_sigma=15 | 0.08838 (+-0.00030) | +0.00653 (+-0.00203) | 0.103 | s0: te=0.08859 gap=+0.00510; s1: te=0.08816 gap=+0.00796 |
| C1 shape=none | 0.07921 (+-0.00019) | +0.01028 (+-0.00357) | 0.103 | s0: te=0.07908 gap=+0.00776; s1: te=0.07934 gap=+0.01280 |
| C2 antithetic | 0.07898 (+-0.00162) | +0.00965 (+-0.00012) | 0.103 | s0: te=0.08013 gap=+0.00973; s1: te=0.07783 gap=+0.00956 |
| C3 sigma_ref=weight | 0.08012 (+-0.00245) | +0.01023 (+-0.00184) | 0.115 | s0: te=0.08185 gap=+0.00892; s1: te=0.07839 gap=+0.01153 |
| B2 lr x0.5 | 0.07983 (+-0.00206) | +0.01147 (+-0.00154) | 0.103 | s0: te=0.08129 gap=+0.01038; s1: te=0.07837 gap=+0.01256 |
| B2 lr x2 | 0.08209 (+-0.00101) | +0.00936 (+-0.00222) | 0.103 | s0: te=0.08280 gap=+0.00779; s1: te=0.08137 gap=+0.01093 |

## Gate verdict (stage 2, the long two-seed gate)

Reference corner: test <= 0.07, gap <= 0.007. Frontier-mover check (docs/research/antikaon-design.md §6): a B/C arm must beat A0 on BOTH axes by more than the two-seed spread, AND be non-dominated by A1 (loss<=A1 OR gap<=A1).

| arm | reaches test<=.0700 & gap<=.0070 | beats A0 both axes beyond spread | non-dominated by A1 | verdict |
|---|---|---|---|---|
| B0 k_sigma=0.5 | no | no | no | trades loss<->gap (dominated line) |
| B1 k_sigma=1.5 | no | no | yes | trades loss<->gap (dominated line) |
| B2 k_sigma=5 | no | no | yes | trades loss<->gap (dominated line) |
| B3 k_sigma=15 | no | no | yes | trades loss<->gap (dominated line) |
| C1 shape=none | no | no | yes | trades loss<->gap (dominated line) |
| C2 antithetic | no | no | yes | trades loss<->gap (dominated line) |
| C3 sigma_ref=weight | no | no | yes | trades loss<->gap (dominated line) |
| B2 lr x0.5 | no | no | yes | trades loss<->gap (dominated line) |
| B2 lr x2 | no | no | yes | trades loss<->gap (dominated line) |

