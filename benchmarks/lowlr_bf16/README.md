# Low-LR bf16: the weight-write methods head to head

Status: **prepared, not run** (only `--smoke` has been executed, to check the plumbing).

`benchmarks/antikaon_lowlr` measured Antikaon at a fine-tune LR; this runner keeps its
U-Net, dataset, loss, eval and fine-tune protocol (fp32 pre-training at the battery's
`lr_const`, rounded once to bf16; constant-LR fine-tune from that same start, same data order)
and drops Antikaon: it compares every `bf16_method` on two rules.

| arm | rule | weights | write | extra state |
|---|---|---|---|---|
| `ada-sr` / `nek-sr` | Adakaon-nomom / Nekaon | bf16 | stochastic rounding | 0 B/param |
| `ada-kahan` / `nek-kahan` | | bf16 | legacy `kahan` (bf16 `shift`; **per-param path only**) | 2 B/param |
| `ada-k8` / `nek-k8` | | bf16 | `kahan8` (uint8 residual, SR at ulp/256) | 1 B/param |
| `ada-k16` / `nek-k16` | | bf16 | `kahan16` (int16 residual: an fp32 master split in two) | 2 B/param |
| `ada-fp32` / `nek-fp32` | | fp32 | — (the reference of the rule) | 2 B/param more |

Rules: Adakaon-nomom = betas (0, 0.999), cautious off, bf16 momentum; Nekaon = k 1.5, betas
(0.5, 0.999), wd 0.1, 4-bit momentum (the `benchmarks/control/registry.py` configs).

Metrics per arm and seed: `test`, `train`, `gap = test - train` at the clean weights
(Nekaon without its climb), `test_start`, `test_full` (the clean iterate at its full value —
decoded residual / `p + shift` — in an fp32 copy of the net), `dist_ulp` and `dist_rel` of
that full value to the rule's fp32 arm, and **ms/step** (forward + backward + step) and
**ms/opt** (the optimizer step alone), CUDA events, first 20 steps excluded. The timings are
**orientative**: the arms run serially in one process with no control over power state or
clocks (laptop GPU: 60 W on AC, 35 W on battery — record which). Note `*-kahan` takes the
per-param path, so its ms/opt is not comparable with the foreach/fused arms as a write cost.

Expectation to check (not a result): `*-k16` should sit at `dist_ulp ≈ 0` for Adakaon-nomom
(bit-exact to fp32 given the same bf16 gradients, except Gradient Centralization, which runs
on the bf16 grad in bf16, and weight decay, which reads the bf16 weight — Nekaon has
wd 0.1), so its `test` differs from `*-fp32` only through the forward seeing the nearest bf16.

## Run

From the repo root (the worktree), GPU plugged in:

```
PYTHONPATH=src python benchmarks/lowlr_bf16/run_lowlr.py --smoke          # plumbing (6 steps, all arms)
PYTHONPATH=src python benchmarks/lowlr_bf16/run_lowlr.py                  # 10 arms x seeds 0,1, 8000 steps, lr 1e-5
PYTHONPATH=src python benchmarks/lowlr_bf16/run_lowlr.py --lr 1e-5 --seeds 0,1 --steps 8000 \
    --arms ada-sr,ada-k8,ada-k16,ada-fp32
```

Flags: `--lr`, `--seeds`, `--steps`, `--arms` (comma-separated keys), `--pretrain`, `--C`,
`--out`. Output: `results.json` (`smoke.json` under `--smoke`) + a summary table. The fp32
pre-training is cached per `(C, bs, n, seed)` in `cache/` (git-ignored).
