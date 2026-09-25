# Optimizer step cost at diffusion scale

Cost only — wall clock, VRAM and state bytes of the optimizer step. Nothing on this page says anything about the quality of the resulting model.

* kaon `0.7.14` @ `0814c7025d6a` (dirty worktree)
* torch 2.12.0+cu130 / CUDA 13.0 / triton 3.7.1 / python 3.10.18
* NVIDIA RTX 3000 Ada Generation Laptop GPU — `[N/A], 60.00 W, 35.00 W, 3105 MHz, 65` (power.limit, power.max_limit, power.default_limit, clocks.max.sm, temp)
* power: on AC (BatteryStatus=2) — 2026-09-18T18:30:45-0600
* bag: R=5, 780 tensors, 663.1 M params; paired reps=100 at R=5; kaon path=native/foreach

## Per step, per arm (one arm resident)

| arm | params | ms/step | state B/p | resident B/p | peak alloc GiB | peak reserved GiB | step transient GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `adamw_bf16` | 663.1 M | 48.76 | 4.00 | 8.03 | 4.96 | 5.26 | 0.00 |
| `adamw_fp32` | 663.1 M | 828.74 | 8.00 | 16.00 | 9.88 | 10.63 | 0.00 |
| `adakaon_bf16` | 663.1 M | 394.28 | 2.01 | 7.25 | 4.48 | 5.01 | 0.76 |
| `adakaon_4bit` | 663.1 M | 564.28 | 0.54 | 5.79 | 3.57 | 3.83 | 0.76 |
| `nekaon_bf16` | 663.1 M | 459.60 | 2.01 | 7.25 | 4.48 | 5.01 | 0.76 |
| `nekaon_4bit` | 663.1 M | 585.35 | 0.54 | 5.79 | 3.57 | 3.83 | 0.76 |
| `nekaon_k0_4bit` | 663.1 M | 553.78 | 0.54 | 5.79 | 3.57 | 3.83 | 0.76 |

`state B/p` counts only the optimizer's own tensors; `resident B/p` is peak allocated over the parameter count — weights + grads + state + the step's transient, which is what actually decides whether a run fits.

The solo ms/step column is unpaired and therefore the weakest number here — read the paired ratios below instead.

## What each arm is

* **`adamw_bf16`** (bfloat16 params) — torch.optim.AdamW(fused) on bf16 params: state is bf16 too, 4 B/p. No stochastic rounding — the cheap thing people actually do. Baseline of the paired ratios.
* **`adamw_fp32`** (float32 params) — Plain fp32 training: fp32 weights (4 B/p) + fp32 grad (4 B/p) + fp32 AdamW state (8 B/p) = 16 B/p resident. There is no bf16 copy and hence no master weight — the 8 B/p is the STATE alone, which is what the 'AdamW costs 8 bytes per parameter' folklore means. Only fits at small R.
* **`adakaon_bf16`** (bfloat16 params) — Adakaon: bf16 momentum + factored second moment, stochastic-rounded bf16 writes.
* **`adakaon_4bit`** (bfloat16 params) — Adakaon with 4-bit quantized momentum.
* **`nekaon_bf16`** (bfloat16 params) — Nekaon k=1.5 over the bf16-momentum Adakaon: the negative-momentum lookahead on top of the same inner step.
* **`nekaon_4bit`** (bfloat16 params) — Nekaon k=1.5 with 4-bit momentum — the Anima configuration.
* **`nekaon_k0_4bit`** (bfloat16 params) — Nekaon with k=0: the lookahead is inert, so the gap to nekaon_4bit isolates what the lookahead itself costs and the gap to adakaon_4bit is the wrapper's own tax.
