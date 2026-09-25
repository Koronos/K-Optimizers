# Optimizer step cost at diffusion scale

Cost only — wall clock, VRAM and state bytes of the optimizer step. Nothing on this page says anything about the quality of the resulting model.

* kaon `0.7.14` @ `0814c7025d6a` (dirty worktree)
* torch 2.12.0+cu130 / CUDA 13.0 / triton 3.7.1 / python 3.10.18
* NVIDIA RTX 3000 Ada Generation Laptop GPU — `[N/A], 60.00 W, 35.00 W, 3105 MHz, 58` (power.limit, power.max_limit, power.default_limit, clocks.max.sm, temp)
* power: on AC (BatteryStatus=2) — 2026-09-18T18:29:14-0600
* bag: R=3, 468 tensors, 397.9 M params; paired reps=150 at R=3; kaon path=Triton-fused

## Per step, per arm (one arm resident)

| arm | params | ms/step | state B/p | resident B/p | peak alloc GiB | peak reserved GiB | step transient GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adakaon_4bit` | 397.9 M | 54.13 | 0.54 | 4.58 | 1.70 | 1.75 | 0.00 |
| `nekaon_4bit` | 397.9 M | 64.94 | 0.54 | 4.58 | 1.70 | 1.75 | 0.00 |
| `nekaon_k0_4bit` | 397.9 M | 52.10 | 0.54 | 4.58 | 1.70 | 1.75 | 0.00 |
| `nekaon_bf16` | 397.9 M | 75.90 | 2.01 | 6.04 | 2.24 | 2.37 | 0.00 |
| `adakaon_bf16` | 397.9 M | 54.54 | 2.01 | 6.04 | 2.24 | 2.37 | 0.00 |

`state B/p` counts only the optimizer's own tensors; `resident B/p` is peak allocated over the parameter count — weights + grads + state + the step's transient, which is what actually decides whether a run fits.

The solo ms/step column is unpaired and therefore the weakest number here — read the paired ratios below instead.

## Paired against `adakaon_4bit` (ABAB, geometric mean, 95% CI)

| arm | ms/step | baseline ms/step | ratio | 95% CI | verdict |
| --- | ---: | ---: | ---: | :---: | --- |
| `nekaon_4bit` | 57.95 | 40.04 | 1.450x | [1.445, 1.454] | slower |
| `nekaon_k0_4bit` | 40.67 | 40.64 | 1.001x | [0.999, 1.003] | n.s. |
| `nekaon_bf16` | 69.74 | 39.58 | 1.761x | [1.759, 1.764] | slower |
| `adakaon_bf16` | 47.37 | 40.01 | 1.185x | [1.183, 1.186] | slower |

`n.s.` = the CI straddles 1.0, i.e. the difference is not resolvable at this rep count — not a claim of equality. The interval is Student-t (n-1 d.o.f.), as in `benchmarks/control/battery.py`. Each pair is preceded by 30 warm-up pairs and the first 5 measured pairs are dropped from both arms, so Triton's re-autotune at the paired scale is outside the samples. MSAM's inert-lookahead telemetry is disarmed on every arm: it syncs to the host only when `rho != 0`, so leaving it on would charge the two lookahead arms for a bounded first-~2000-steps cost. The Anima campaign (200 steps) does pay it, and reports it.

## What each arm is

* **`adakaon_4bit`** (bfloat16 params) — Adakaon with 4-bit quantized momentum.
* **`nekaon_4bit`** (bfloat16 params) — Nekaon k=1.5 with 4-bit momentum — the Anima configuration.
* **`nekaon_k0_4bit`** (bfloat16 params) — Nekaon with k=0: the lookahead is inert, so the gap to nekaon_4bit isolates what the lookahead itself costs and the gap to adakaon_4bit is the wrapper's own tax.
* **`nekaon_bf16`** (bfloat16 params) — Nekaon k=1.5 over the bf16-momentum Adakaon: the negative-momentum lookahead on top of the same inner step.
* **`adakaon_bf16`** (bfloat16 params) — Adakaon: bf16 momentum + factored second moment, stochastic-rounded bf16 writes.
