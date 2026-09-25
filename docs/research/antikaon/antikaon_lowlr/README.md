# Antikaon low-LR bf16 quality experiment

> **Status: descartado 2026-09-25.** Antikaon fue evaluado y no promovido (ver
> `docs/EXPERIMENTS_GRAVEYARD.md`). `run_lowlr.py` se eliminó del árbol; el código vive en el
> tag `graveyard/antikaon` (`git show graveyard/antikaon:benchmarks/antikaon_lowlr/run_lowlr.py`).
> Este README y los resultados (`results.json`, `smoke.json`, `run.log`) se conservan como
> evidencia histórica — solo se ejecutó `--smoke`, la corrida completa nunca se lanzó.

The control battery and the (now archived) `antikaon_gate` runner train the proxy U-Net in **fp32 at
lr 1.2e-3**, ~100× a real fine-tune LR: every update is several bf16 ulps and the
`bf16_method` never matters. This experiment keeps the same U-Net, dataset, loss and eval
(`benchmarks/control/battery.py` + `benchmarks/proxy`, loaded by file path as `run_gate.py`
does) but trains with **bf16 weights at lr 1e-5**, where a typical update is a fraction of a
bf16 ulp — the regime `bf16_method="kahan8"` exists for. Protocol, arms and metrics are
documented in the header of `run_lowlr.py`.

## LR choice (`--probe`, measured on the 300-step pre-trained start)

Per-coordinate `lr·|u| / ulp_bf16(w)` (update directions of 10 fp32 Adakaon-nomom steps),
quantiles over a 200 k-coordinate sample of all parameters; |w| p10/p50/p90 = 0.004 / 0.023 / 0.053:

| lr | p10 | p50 | p90 |
|---|---:|---:|---:|
| 5e-6 | 0.010 | 0.030 | 0.16 |
| **1e-5** | **0.021** | **0.061** | **0.32** |
| 1.5e-5 | 0.031 | 0.091 | 0.48 |
| 2e-5 | 0.042 | 0.121 | 0.64 |
| 3e-5 | 0.063 | 0.181 | 0.97 |

`1e-5` puts the 10–90 % band in the 0.01–0.3 ulp target (1-D gains/biases: p10 0.001,
p50 0.039, p90 0.33).

## Run

From the repo root (the worktree), with the GPU plugged in (laptop: 60 W on AC):

```
PYTHONPATH=src python benchmarks/antikaon_lowlr/run_lowlr.py            # 9 arms x seeds 0,1
PYTHONPATH=src python benchmarks/antikaon_lowlr/run_lowlr.py --twins    # + fp32 twins of anti*/nek
```

Defaults: C=40, bs=8, pre-train 300 steps (fp32, lr 1.2e-3, cached in `cache/`), fine-tune
8000 constant-LR steps at 1e-5, seeds 0 and 1. Output: `results.json` + a summary table.

**Estimated duration** (not measured here; extrapolated from the gate's stage-2 manifest,
40 k fp32 steps in 1567 s ≈ 39 ms/step on a shared GPU): 9 arms × 2 seeds × 8000 steps =
144 k steps ≈ 1.5 h; the Antikaon noise loop, Nekaon's climbs and the evals push it to
~1.5–2.5 h. `--twins` adds 3 arms (≈ +35 %: ~2–3.3 h). `--seeds 0` halves it.

## Smoke (6 fine-tune steps after 8 pre-train steps, seed 0) — plumbing only

All 12 arms run, every metric is produced. One effect is already visible and is worth
watching in the full run: with `kahan8` the clean iterate stays on the fp32 trajectory
(`dist_rel` 0.03 vs 1.7–3.7 for SR; its fp32 view scores like the fp32 run), but the
**stored bf16 model** — what `test` evaluates — is the *nearest* bf16 to that iterate, which
does not move until a coordinate crosses a half-ulp boundary; SR's stored model is an
unbiased random rounding that realizes sub-ulp movement in expectation. After 6 sub-ulp
steps the kahan8 bf16 model has therefore realized only ~2/3 of the loss decrease
(`test` 0.2686 vs 0.2658 for fp32/SR, `test_full` 0.2657). The lag is bounded by half an ulp
of accumulated movement, so it should shrink relative to the progress as the fine-tune moves
the weights by several ulps; the full run's `test` vs `test_full` columns measure it.
