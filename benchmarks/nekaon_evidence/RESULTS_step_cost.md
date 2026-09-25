# Optimizer step cost at diffusion scale

Cost only — wall clock, VRAM and state bytes of the optimizer step. Nothing on this page says anything about the quality of the resulting model.

* kaon `0.7.14` @ `0814c7025d6a` (dirty worktree)
* torch 2.12.0+cu130 / CUDA 13.0 / triton 3.7.1 / python 3.10.18
* NVIDIA RTX 3000 Ada Generation Laptop GPU — `[N/A], 60.00 W, 35.00 W, 3105 MHz, 56` (power.limit, power.max_limit, power.default_limit, clocks.max.sm, temp)
* power: on AC (BatteryStatus=2) — 2026-09-18T18:21:38-0600
* bag: R=5, 780 tensors, 663.1 M params; paired reps=150 at R=3; kaon path=Triton-fused

## Per step, per arm (one arm resident)

| arm | params | ms/step | state B/p | resident B/p | peak alloc GiB | peak reserved GiB | step transient GiB | launches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adamw_bf16` | 663.1 M | 48.67 | 4.00 | 8.03 | 4.96 | 5.26 | 0.00 | 9 |
| `adamw_fp32` | 663.1 M | 842.47 | 8.00 | 16.00 | 9.88 | 10.63 | 0.00 | 9 |
| `adakaon_bf16` | 663.1 M | 85.37 | 2.01 | 6.04 | 3.73 | 3.94 | 0.00 | 38 |
| `adakaon_4bit` | 663.1 M | 77.97 | 0.54 | 4.57 | 2.82 | 2.94 | 0.00 | 38 |
| `nekaon_bf16` | 663.1 M | 115.84 | 2.01 | 6.04 | 3.73 | 3.94 | 0.00 | 54 |
| `nekaon_4bit` | 663.1 M | 99.75 | 0.54 | 4.57 | 2.82 | 2.94 | 0.00 | 54 |
| `nekaon_k0_4bit` | 663.1 M | 74.89 | 0.54 | 4.57 | 2.82 | 2.94 | 0.00 | 38 |

`state B/p` counts only the optimizer's own tensors; `resident B/p` is peak allocated over the parameter count — weights + grads + state + the step's transient, which is what actually decides whether a run fits.

The solo ms/step column is unpaired and therefore the weakest number here — read the paired ratios below instead.

## Paired against `adamw_bf16` (ABAB, geometric mean, 95% CI)

| arm | ms/step | baseline ms/step | ratio | 95% CI | verdict |
| --- | ---: | ---: | ---: | :---: | --- |
| `adamw_fp32` | 165.48 | 411.32 | 0.402x | [0.401, 0.403] | faster |
| `adakaon_bf16` | 45.66 | 29.35 | 1.550x | [1.545, 1.554] | slower |
| `adakaon_4bit` | 40.34 | 29.70 | 1.380x | [1.362, 1.399] | slower |
| `nekaon_bf16` | 68.27 | 29.33 | 2.319x | [2.312, 2.326] | slower |
| `nekaon_4bit` | 57.14 | 29.48 | 1.925x | [1.917, 1.933] | slower |
| `nekaon_k0_4bit` | 39.94 | 29.32 | 1.359x | [1.353, 1.364] | slower |

`n.s.` = the CI straddles 1.0, i.e. the difference is not resolvable at this rep count — not a claim of equality. The interval is Student-t (n-1 d.o.f.), as in `benchmarks/control/battery.py`. Each pair is preceded by 30 warm-up pairs and the first 5 measured pairs are dropped from both arms, so Triton's re-autotune at the paired scale is outside the samples. MSAM's inert-lookahead telemetry is disarmed on every arm: it syncs to the host only when `rho != 0`, so leaving it on would charge the two lookahead arms for a bounded first-~2000-steps cost. The Anima campaign (200 steps) does pay it, and reports it.

## Share of a real training step (proxy UNet C=256, batch 8, 64px)

The full-step column should agree across arms to within the optimizer difference — where it does not, the run was contended and the share is only indicative.

| arm | trainable params | full step ms | optimizer ms | optimizer share |
| --- | ---: | ---: | ---: | ---: |
| `adamw_bf16` | 11.16 M | 57.66 | 1.30 | 2.3% |
| `adamw_fp32` | 11.16 M | 96.98 | 2.04 | 2.1% |
| `adakaon_bf16` | 11.16 M | 64.93 | 4.17 | 6.4% |
| `adakaon_4bit` | 11.16 M | 70.12 | 6.85 | 9.8% |
| `nekaon_bf16` | 11.16 M | 66.65 | 4.66 | 7.0% |
| `nekaon_4bit` | 11.16 M | 69.89 | 6.84 | 9.8% |
| `nekaon_k0_4bit` | 11.16 M | 69.26 | 5.83 | 8.4% |

## Capacity on this GPU (largest bag that fits before OOM)

| arm | max R | trainable params | OOM at R | OOM at params |
| --- | ---: | ---: | ---: | ---: |
| `adamw_bf16` | 12 | 1591.4 M | - | - |
| `adamw_fp32` | 12 | 1591.4 M | - | - |
| `adakaon_bf16` | 12 | 1591.4 M | - | - |
| `adakaon_4bit` | 12 | 1591.4 M | - | - |
| `nekaon_bf16` | 12 | 1591.4 M | - | - |
| `nekaon_4bit` | 12 | 1591.4 M | - | - |
| `nekaon_k0_4bit` | 12 | 1591.4 M | - | - |

Capacity is a ceiling for THIS bag on THIS card with nothing else resident; a real trainer also holds activations.

## What each arm is

* **`adamw_bf16`** (bfloat16 params) — torch.optim.AdamW(fused) on bf16 params: state is bf16 too, 4 B/p. No stochastic rounding — the cheap thing people actually do. Baseline of the paired ratios.
* **`adamw_fp32`** (float32 params) — Plain fp32 training: fp32 weights (4 B/p) + fp32 grad (4 B/p) + fp32 AdamW state (8 B/p) = 16 B/p resident. There is no bf16 copy and hence no master weight — the 8 B/p is the STATE alone, which is what the 'AdamW costs 8 bytes per parameter' folklore means. Only fits at small R.
* **`adakaon_bf16`** (bfloat16 params) — Adakaon: bf16 momentum + factored second moment, stochastic-rounded bf16 writes.
* **`adakaon_4bit`** (bfloat16 params) — Adakaon with 4-bit quantized momentum.
* **`nekaon_bf16`** (bfloat16 params) — Nekaon k=1.5 over the bf16-momentum Adakaon: the negative-momentum lookahead on top of the same inner step.
* **`nekaon_4bit`** (bfloat16 params) — Nekaon k=1.5 with 4-bit momentum — the Anima configuration.
* **`nekaon_k0_4bit`** (bfloat16 params) — Nekaon with k=0: the lookahead is inert, so the gap to nekaon_4bit isolates what the lookahead itself costs and the gap to adakaon_4bit is the wrapper's own tax.
