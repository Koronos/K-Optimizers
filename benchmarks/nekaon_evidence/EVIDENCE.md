# Nekaon: what is measured, and what it does and does not show

Does Nekaon beat Adakaon and AdamW on speed, quality and memory for diffusion fine-tuning?

This page exists so the claim can be checked rather than believed. Where Nekaon wins, where it loses and where the measurement cannot tell are printed in the same tables, in the same font, next to the intervals that produced them.

## Sources

| Source | Present | kaon | Commit | GPU | Electrical state | Detail |
| --- | --- | --- | --- | --- | --- | --- |
| `anima` | yes | 0.7.14 | `0814c7025d6a` | — | all runs share one signature (ac/max_limit=60.00 W) | 5 paired seeds |
| `step_cost` | yes | 0.7.14 | `0814c7025d6a` | NVIDIA RTX 3000 Ada Generation Laptop GPU | on AC (BatteryStatus=2) | R=5, 663.1 M params |
| `step_cost_vs_adakaon` | yes | 0.7.14 | `0814c7025d6a` | NVIDIA RTX 3000 Ada Generation Laptop GPU | on AC (BatteryStatus=2) | R=3, 397.9 M params, baseline `adakaon_4bit` |
| `step_cost_native` | yes | 0.7.14 | `0814c7025d6a` | NVIDIA RTX 3000 Ada Generation Laptop GPU | on AC (BatteryStatus=2) | R=5, 663.1 M params, native (non-Triton), unpaired |
| `battery` | yes | 0.7.14 | `0814c70` | NVIDIA RTX 3000 Ada Generation Laptop GPU | AC | 8 arms |

## Verdict

| Axis | Nekaon vs AdamW | Nekaon vs Adakaon |
| --- | --- | --- |
| Quality | **mixed** (2W / 1L / 2 n.s. / 0 tie) | **mixed** (2W / 2L / 1 n.s. / 0 tie) |
| Speed | **loses** (0W / 5L / 6 n.s. / 0 tie) | **loses** (0W / 4L / 4 n.s. / 0 tie / 1 excluded) |
| Memory | **wins** (4W / 0L / 0 n.s. / 0 tie) | **n.s.** (0W / 0L / 1 n.s. / 3 tie) |

A roll-up is never the finding. `mixed` means the rows below disagree, and the rows are the answer; `n.s.` means every row's interval covered zero. Read the level tables.

### How each verdict is made

Every verdict on this page is one of seven words, each produced by a rule, never by reading a
number and choosing an adjective:

* **wins** -- the paired 95% interval for the difference (or the ratio) lies entirely on the
  side that favours the challenger.
* **loses** -- the interval lies entirely on the other side.
* **n.s.** -- the interval covers zero (or 1.0 for a ratio). *Not a win.* With five seeds the
  interval is wide, so `n.s.` is the expected outcome of a small real effect, and it is
  reported exactly as loudly as a win.
* **tie by construction** -- the two arms run the same code path with the same state, so the
  metric is identical by definition and no measurement is needed (see the memory axis).
* **no evidence** -- the source that would answer this was not supplied, or could not answer.
* **invalid (oversubscribed)** -- the arm or its baseline held more bytes than this GPU has
  while the row was being measured. Under Windows WDDM the CUDA allocator serves the surplus
  from host RAM instead of raising OOM, so the number is a PCIe measurement wearing an
  optimizer's name. The row is printed, and it enters no verdict.
* **mixed** -- the rows rolled up into this axis disagree; the rows, not the roll-up, are the
  result.

Intervals: the Anima campaign uses the percentile bootstrap of the paired mean that
`anima/aggregate.py` computed; the control battery and the SDXL step cost use Student-t with
n-1 d.o.f. (the same `_T95` table `battery.py::ci95` and `step_cost.py::gmean_ci` use, so the
three sets of intervals are built the same way). N is printed next to every verdict, because
a verdict from five seeds is not the same object as a verdict from sixty reps.


## Quality

Final validation loss and the train/val gap. Nothing here is perceptual: there is no FID and no KID in this report, so a quality win is a win on an objective loss, on these tasks, at these learning rates.

### Nekaon vs AdamW — quality

| Level | Metric | Value / 95% CI | N | Verdict | Source |
| --- | --- | ---: | ---: | --- | --- |
| control battery (proxy diffusion), paired by seed | Test loss (REX) (Nekaon-fused - torch.AdamW (fused)) | -0.00146425 [-0.00509317, 0.00216468] | 5 | **n.s.** | `battery` |
| control battery (proxy diffusion), paired by seed | Train-test gap (REX) (Nekaon-fused - torch.AdamW (fused)) | -0.00214563 [-0.00342052, -0.00087074] | 5 | **wins** | `battery` |
| control battery (proxy diffusion), paired by seed | Train-test gap (const LR) (Nekaon-fused - torch.AdamW (fused)) | -0.00226993 [-0.00682037, 0.00228051] | 5 | **n.s.** | `battery` |
| real model (Anima/Pets LoRA), paired by seed | Final val eps-MSE | -0.000260481 [-0.000423908, -0.000110507] | 5 | **wins** | `anima` |
| real model (Anima/Pets LoRA), paired by seed | Raw gap (val - train_eval) | 0.000368962 [9.7096e-05, 0.000617963] | 5 | **loses** | `anima` |

### Nekaon vs Adakaon — quality

| Level | Metric | Value / 95% CI | N | Verdict | Source |
| --- | --- | ---: | ---: | --- | --- |
| control battery (proxy diffusion), paired by seed | Test loss (REX) (Nekaon-fused - Adakaon-4bit-fused) | 0.00352054 [0.000725981, 0.00631511] | 5 | **loses** | `battery` |
| control battery (proxy diffusion), paired by seed | Train-test gap (REX) (Nekaon-fused - Adakaon-4bit-fused) | -0.0109251 [-0.0136863, -0.00816377] | 5 | **wins** | `battery` |
| control battery (proxy diffusion), paired by seed | Train-test gap (const LR) (Nekaon-fused - Adakaon-4bit-fused) | -0.00945633 [-0.0123042, -0.00660848] | 5 | **wins** | `battery` |
| real model (Anima/Pets LoRA), paired by seed | Final val eps-MSE | 5.71907e-06 [-0.000170803, 0.000133339] | 5 | **n.s.** | `anima` |
| real model (Anima/Pets LoRA), paired by seed | Raw gap (val - train_eval) | 0.000368631 [0.000138971, 0.000617069] | 5 | **loses** | `anima` |

## Speed

Three levels that are never mixed into one sentence, because they answer three different questions: **(a)** the optimizer step alone at SDXL parameter scale, **(b)** the full training step on a real model, and **(c)** how much of that step the optimizer even is -- the ceiling on what any optimizer speedup can buy end to end.

### Nekaon vs AdamW — speed

| Level | Metric | Value / 95% CI | N | Verdict | Source |
| --- | --- | ---: | ---: | --- | --- |
| (a) optimizer step alone, SDXL parameter scale | nekaon_4bit / adamw_bf16 time ratio | 1.92496 [1.91686, 1.93309] | 145 | **loses** | `step_cost` |
| (b) full training step on the proxy model | ms / iteration (Nekaon-fused - torch.AdamW (fused)) | 11.8332 [9.98303, 13.6833] | 5 | **loses** | `battery` |
| (b) full training step on the proxy model | ms / optimizer step (Nekaon-fused - torch.AdamW (fused)) | 3.46179 [2.12281, 4.80077] | 5 | **loses** | `battery` |
| (b) full training step on the real model | Active train seconds | 6.53618 [2.39626, 10.2779] | 5 | **loses** | `anima` |
| (b) full training step on the real model | ms / step | 32.6809 [11.9813, 51.3893] | 5 | **loses** | `anima` |
| (c) optimizer share of a full fwd+bwd+step | adakaon_bf16 optimizer fraction<br>*a ceiling, not a contest: making this optimizer infinitely fast would cut the whole step by at most 6.4%* | 0.0642338 (deterministic) | — | **n.s.** | `step_cost` |
| (c) optimizer share of a full fwd+bwd+step | adamw_bf16 optimizer fraction<br>*a ceiling, not a contest: making this optimizer infinitely fast would cut the whole step by at most 2.3%* | 0.0225381 (deterministic) | — | **n.s.** | `step_cost` |
| (c) optimizer share of a full fwd+bwd+step | adamw_fp32 optimizer fraction<br>*a ceiling, not a contest: making this optimizer infinitely fast would cut the whole step by at most 2.1%* | 0.0210735 (deterministic) | — | **n.s.** | `step_cost` |
| (c) optimizer share of a full fwd+bwd+step | nekaon_4bit optimizer fraction<br>*a ceiling, not a contest: making this optimizer infinitely fast would cut the whole step by at most 9.8%* | 0.0978134 (deterministic) | — | **n.s.** | `step_cost` |
| (c) optimizer share of a full fwd+bwd+step | nekaon_bf16 optimizer fraction<br>*a ceiling, not a contest: making this optimizer infinitely fast would cut the whole step by at most 7.0%* | 0.0698778 (deterministic) | — | **n.s.** | `step_cost` |
| (c) optimizer share of a full fwd+bwd+step | nekaon_k0_4bit optimizer fraction<br>*a ceiling, not a contest: making this optimizer infinitely fast would cut the whole step by at most 8.4%* | 0.0841183 (deterministic) | — | **n.s.** | `step_cost` |

### Nekaon vs Adakaon — speed

| Level | Metric | Value / 95% CI | N | Verdict | Source |
| --- | --- | ---: | ---: | --- | --- |
| (a) optimizer step alone, SDXL parameter scale | adakaon_bf16 / adakaon_4bit time ratio<br>*context row: this pair is Adakaon against Adakaon -- the 4-bit codec knob at SDXL scale, not Nekaon against Adakaon -- so it is printed with its interval and rolled up into nothing* | 1.1845 [1.18256, 1.18645] | 145 | **loses** | `step_cost_vs_adakaon` |
| (a) optimizer step alone, SDXL parameter scale | nekaon_4bit / adakaon_4bit time ratio | 1.44978 [1.44522, 1.45435] | 145 | **loses** | `step_cost_vs_adakaon` |
| (a) optimizer step alone, SDXL parameter scale | nekaon_bf16 / adakaon_4bit time ratio | 1.7612 [1.75861, 1.7638] | 145 | **loses** | `step_cost_vs_adakaon` |
| (a) optimizer step alone, SDXL parameter scale | nekaon_k0_4bit / adakaon_4bit time ratio | 1.0006 [0.998607, 1.00259] | 145 | **n.s.** | `step_cost_vs_adakaon` |
| (b) full training step on the proxy model | ms / iteration (Nekaon-fused - Adakaon-4bit-fused) | 0.664145 [-1.35451, 2.6828] | 5 | **n.s.** | `battery` |
| (b) full training step on the proxy model | ms / optimizer step (Nekaon-fused - Adakaon-4bit-fused) | 1.14722 [-0.469687, 2.76413] | 5 | **n.s.** | `battery` |
| (b) full training step on the real model | Active train seconds | 6.2364 [3.03208, 8.9199] | 5 | **loses** | `anima` |
| (b) full training step on the real model | ms / step | 31.182 [15.1604, 44.5995] | 5 | **loses** | `anima` |
| (c) optimizer share of a full fwd+bwd+step | adakaon_4bit optimizer fraction<br>*a ceiling, not a contest: making this optimizer infinitely fast would cut the whole step by at most 9.8%* | 0.0977499 (deterministic) | — | **n.s.** | `step_cost` |

## Memory

Also three levels: **(a)** optimizer state bytes per parameter, which is arithmetic and carries no interval; **(b)** peak VRAM during real training; **(c)** the largest parameter count that still fits on this GPU.

> Nekaon vs Adakaon at the same `momentum_dtype` hold the **same memory, by construction**: Nekaon is a negative-momentum lookahead layer over Adakaon that climbs and descends the existing momentum buffer in place and allocates no state of its own. Any B/p difference measured between them is instrumentation noise, not a property of the optimizers -- read that row as a tie unless the two are running different `momentum_dtype`s. The memory advantage over AdamW is a different claim entirely: it comes from Adakaon's factored second moment and its quantized momentum codec, and the lookahead contributes exactly nothing to it.

### Nekaon vs AdamW — memory

| Level | Metric | Value / 95% CI | N | Verdict | Source |
| --- | --- | ---: | ---: | --- | --- |
| (a) optimizer state bytes per parameter | Bytes per param (state) (Nekaon-fused - torch.AdamW (fused))<br>*state bytes per param is deterministic: it is compared directly, not through an interval, and any spread across seeds would be measurement noise rather than seed variance* | -7.43672 [-7.43672, -7.43672] | 5 | **wins** | `battery` |
| (a) optimizer state bytes per parameter | nekaon_4bit 0.540 B/p vs adamw_bf16 4.000 B/p | -3.45968 (deterministic) | — | **wins** | `step_cost` |
| (b) peak VRAM during real training | Peak allocator GiB | -0.098 [-0.098, -0.098] | 5 | **wins** | `anima` |
| (b) peak allocator bytes at SDXL parameter scale | nekaon_4bit 2.82 GiB vs adamw_bf16 4.96 GiB | -2.13166 (deterministic) | — | **wins** | `step_cost` |
| (c) largest parameter count that fits on this GPU | nekaon_4bit 1591.4 M vs adamw_bf16 1591.4 M<br>*no OOM reached on any arm: the CUDA allocator oversubscribed into host memory (Windows WDDM), so the fit-limit is not measurable on this machine; both arms simply reached the top of the sweep (R=12 and R=12)* | 0 (deterministic) | — | **no evidence** | `step_cost` |

### Nekaon vs Adakaon — memory

| Level | Metric | Value / 95% CI | N | Verdict | Source |
| --- | --- | ---: | ---: | --- | --- |
| (a) optimizer state bytes per parameter | Bytes per param (state) (Nekaon-fused - Adakaon-4bit-fused)<br>*state bytes per param is deterministic: it is compared directly, not through an interval, and any spread across seeds would be measurement noise rather than seed variance* | 0 [0, 0] | 5 | **tie by construction** | `battery` |
| (a) optimizer state bytes per parameter | nekaon_4bit 0.540 B/p vs adakaon_4bit 0.540 B/p<br>*identical by construction: Nekaon is a lookahead layer over Adakaon and allocates no state of its own, so at the same `momentum_dtype` the two hold the same buffers* | 0 (deterministic) | — | **tie by construction** | `step_cost` |
| (b) peak VRAM during real training | Peak allocator GiB | 0 [0, 0] | 5 | **n.s.** | `anima` |
| (b) peak allocator bytes at SDXL parameter scale | nekaon_4bit 2.82 GiB vs adakaon_4bit 2.82 GiB | 2.28882e-05 (deterministic) | — | **tie by construction** | `step_cost` |
| (c) largest parameter count that fits on this GPU | nekaon_4bit 1591.4 M vs adakaon_4bit 1591.4 M<br>*no OOM reached on any arm: the CUDA allocator oversubscribed into host memory (Windows WDDM), so the fit-limit is not measurable on this machine; both arms simply reached the top of the sweep (R=12 and R=12)* | 0 (deterministic) | — | **no evidence** | `step_cost` |

## Attribution: which knob did it

Each pair below differs in exactly one setting, so its difference is attributable to that setting and to nothing else. This is the only place in the report where a mechanism — rather than a whole optimizer — is what is being measured.

| Knob | Metric | Paired difference [95% CI] | N | Verdict |
| --- | --- | ---: | ---: | --- |
| lookahead (k=1.5 vs k=0) | Test loss (REX) | 0.00363797 [-0.0022723, 0.00954824] | 5 | **n.s.** |
| lookahead (k=1.5 vs k=0) | Train-test gap (REX) | -0.00363308 [-0.00506272, -0.00220344] | 5 | **wins** |
| lookahead (k=1.5 vs k=0) | Train-test gap (const LR) | -0.00232738 [-0.00380185, -0.00085291] | 5 | **wins** |
| lookahead (k=1.5 vs k=0) | ms / iteration | 1.53288 [1.17679, 1.88897] | 5 | **loses** |
| lookahead (k=1.5 vs k=0) | ms / optimizer step | 1.04995 [-1.10073, 3.20063] | 5 | **n.s.** |
| lookahead (k=1.5 vs k=0) | Bytes per param (state) | 0 [0, 0] | 5 | **tie by construction** |
| 4-bit codec on Nekaon | Test loss (REX) | 0.000695289 [-0.00231849, 0.00370907] | 5 | **n.s.** |
| 4-bit codec on Nekaon | Train-test gap (REX) | 0.000478031 [-0.000409163, 0.00136522] | 5 | **n.s.** |
| 4-bit codec on Nekaon | Train-test gap (const LR) | 0.000155844 [-0.00028309, 0.000594779] | 5 | **n.s.** |
| 4-bit codec on Nekaon | ms / iteration | -0.333122 [-1.82246, 1.15621] | 5 | **n.s.** |
| 4-bit codec on Nekaon | ms / optimizer step | 2.3571 [1.24919, 3.46501] | 5 | **loses** |
| 4-bit codec on Nekaon | Bytes per param (state) | -1.46875 [-1.46875, -1.46875] | 5 | **wins** |
| 4-bit codec on Adakaon | Test loss (REX) | 0.00154775 [-0.000884287, 0.00397978] | 5 | **n.s.** |
| 4-bit codec on Adakaon | Train-test gap (REX) | 0.000530803 [-0.000720537, 0.00178214] | 5 | **n.s.** |
| 4-bit codec on Adakaon | Train-test gap (const LR) | 0.000883609 [-0.00060589, 0.00237311] | 5 | **n.s.** |
| 4-bit codec on Adakaon | ms / iteration | 0.31344 [-1.89639, 2.52327] | 5 | **n.s.** |
| 4-bit codec on Adakaon | ms / optimizer step | 1.21236 [-1.35528, 3.77999] | 5 | **n.s.** |
| 4-bit codec on Adakaon | Bytes per param (state) | -1.46875 [-1.46875, -1.46875] | 5 | **wins** |

A negative difference favours the left arm of the pair. The lookahead row is the only measurement in this report of what Nekaon's own mechanism does, isolated from the codec and from Adakaon.

> There is deliberately **no `nekaon_4bit / nekaon_k0_4bit` row at SDXL scale here.** Both arms are paired against `adakaon_4bit`, not against each other, and the stored file keeps only each pair's geometric mean and interval — not the per-rep times. Dividing the two ratios would give a point estimate whose interval cannot be reconstructed (the two pairs share a baseline, so their errors are correlated by an unknown amount, and treating them as independent would understate it). A correct interval needs a `step_cost.py` run whose `--baseline` is `nekaon_k0_4bit`; until then the lookahead's isolated cost is the battery's single-knob pair below, and the SDXL rows above state only what was measured: each arm against `adakaon_4bit`.

## Native (non-Triton) path, unpaired

`step_cost.py --native --skip-paired` at R=5 (663.1 M params, 100 reps per arm). **Every number below is a solo timing with no interval and no pairing**: the arms ran one after another, not interleaved, so nothing here is comparable with the paired ratios above and nothing here carries a verdict. It is printed to show what the Triton kernels are standing in for, and its only safe reading is the ordering of the arms.

| Arm | ms / step (solo, no interval) | State B/p | Peak allocated |
| --- | ---: | ---: | --- |
| `adamw_bf16` | 48.76 | 4.000 | 4.96 GiB |
| `adakaon_bf16` | 394.3 | 2.009 | 4.48 GiB |
| `nekaon_bf16` | 459.6 | 2.009 | 4.48 GiB |
| `nekaon_k0_4bit` | 553.8 | 0.540 | 3.57 GiB |
| `adakaon_4bit` | 564.3 | 0.540 | 3.57 GiB |
| `nekaon_4bit` | 585.3 | 0.540 | 3.57 GiB |
| `adamw_fp32` | 828.7 | 8.000 | 9.88 GiB — **exceeds VRAM: spilled to host** |

## Measurement integrity: VRAM oversubscription

This GPU has 8.00 GiB of VRAM.

Windows drives it through WDDM, where an allocation larger than VRAM does not raise `CUDA out of memory`: the driver pages the surplus into host RAM and the step keeps running over PCIe. So an arm whose peak exceeds the line above was not measured on this card alone, and neither its own timing nor the baseline it was interleaved with is a measurement of an optimizer. The rule is applied below to every arm of every step-cost file, and the arms it did not touch are listed too. The peak is the one the solo phase recorded; an arm that spilled there is treated as spilled in its paired phase too, even though the pair runs at a smaller R — which the baseline medians in the last column corroborate, since an interleaved baseline slows down by an order of magnitude next to a spilled arm.

> **Capacity: no OOM reached on any arm: the CUDA allocator oversubscribed into host memory (Windows WDDM), so the fit-limit is not measurable on this machine.** Memory level (c) is `no evidence` on this machine — the sweep measured where the loop stopped, not where the GPU did.

| File | Arm | Peak allocated | Solo timing | Paired timing |
| --- | --- | --- | --- | --- |
| `step_cost` (R=5) | `adamw_bf16` | 4.96 GiB | 48.67 ms | not paired in this file |
| `step_cost` (R=5) | `adamw_fp32` | 9.88 GiB — **exceeds VRAM: spilled to host** | 842.5 ms — **invalid (oversubscribed)** | ratio 0.4021 vs `adamw_bf16` at R=3 (baseline median 411.3 ms) — **invalid (oversubscribed)** |
| `step_cost` (R=5) | `adakaon_bf16` | 3.73 GiB | 85.37 ms | ratio 1.55 vs `adamw_bf16` at R=3 (baseline median 29.35 ms) |
| `step_cost` (R=5) | `adakaon_4bit` | 2.82 GiB | 77.97 ms | ratio 1.38 vs `adamw_bf16` at R=3 (baseline median 29.7 ms) |
| `step_cost` (R=5) | `nekaon_bf16` | 3.73 GiB | 115.8 ms | ratio 2.319 vs `adamw_bf16` at R=3 (baseline median 29.33 ms) |
| `step_cost` (R=5) | `nekaon_4bit` | 2.82 GiB | 99.75 ms | ratio 1.925 vs `adamw_bf16` at R=3 (baseline median 29.48 ms) |
| `step_cost` (R=5) | `nekaon_k0_4bit` | 2.82 GiB | 74.89 ms | ratio 1.359 vs `adamw_bf16` at R=3 (baseline median 29.32 ms) |
| `step_cost_vs_adakaon` (R=3) | `adakaon_4bit` | 1.70 GiB | 54.13 ms | not paired in this file |
| `step_cost_vs_adakaon` (R=3) | `nekaon_4bit` | 1.70 GiB | 64.94 ms | ratio 1.45 vs `adakaon_4bit` at R=3 (baseline median 40.04 ms) |
| `step_cost_vs_adakaon` (R=3) | `nekaon_k0_4bit` | 1.70 GiB | 52.1 ms | ratio 1.001 vs `adakaon_4bit` at R=3 (baseline median 40.64 ms) |
| `step_cost_vs_adakaon` (R=3) | `nekaon_bf16` | 2.24 GiB | 75.9 ms | ratio 1.761 vs `adakaon_4bit` at R=3 (baseline median 39.58 ms) |
| `step_cost_vs_adakaon` (R=3) | `adakaon_bf16` | 2.24 GiB | 54.54 ms | ratio 1.185 vs `adakaon_4bit` at R=3 (baseline median 40.01 ms) |
| `step_cost_native` (R=5) | `adamw_bf16` | 4.96 GiB | 48.76 ms | not paired in this file |
| `step_cost_native` (R=5) | `adamw_fp32` | 9.88 GiB — **exceeds VRAM: spilled to host** | 828.7 ms — **invalid (oversubscribed)** | not paired in this file |
| `step_cost_native` (R=5) | `adakaon_bf16` | 4.48 GiB | 394.3 ms | not paired in this file |
| `step_cost_native` (R=5) | `adakaon_4bit` | 3.57 GiB | 564.3 ms | not paired in this file |
| `step_cost_native` (R=5) | `nekaon_bf16` | 4.48 GiB | 459.6 ms | not paired in this file |
| `step_cost_native` (R=5) | `nekaon_4bit` | 3.57 GiB | 585.3 ms | not paired in this file |
| `step_cost_native` (R=5) | `nekaon_k0_4bit` | 3.57 GiB | 553.8 ms | not paired in this file |

## Caveats

None of these are hedges; each one bounds what the tables above may be used to claim.

* **Hyperparameter confound, AdamW vs the kaon arms (Anima).** Every arm runs its own house configuration and only the learning rate is tuned: the kaon arms use betas (0.5, 0.999), weight decay 0.1, cautious updates, gradient centralization, 4-bit momentum and stochastic rounding, while `adamw_fused` uses betas (0.9, 0.999), weight decay 0.01 and none of those. A difference between AdamW and a kaon arm is therefore NOT attributable to the algorithm alone; the table of `[optimizer]` blocks in this report is the exact confound.
* **Inert-lookahead telemetry counts against Nekaon on time.** The `nekaon_fused` arm (k = 1.5) pays MSAM's inert-lookahead telemetry that `adakaon_fused` and `adamw_fused` do not: `_warn_if_inert` stays armed for its first 200 climbs and samples every 10th, so a 200-step run here is instrumented end to end and performs roughly twenty device-to-host synchronizations. That cost is real for a user running this configuration today, but it is telemetry rather than optimizer arithmetic, and it counts against Nekaon in active seconds and ms/step.
* **n is small** -- 5 paired seed(s) on the real model. A bootstrap interval over that many points is wide; `n.s.` there means the campaign is too small to resolve the effect, not that the effect is zero.
* **This GPU is driven through Windows WDDM, whose CUDA allocator oversubscribes into host RAM instead of raising OOM.** The capacity sweep reached the top of its range on every arm without a single OOM, so no fit limit was found; and `adamw_fp32` peaked at 9.88 GiB on a card holding 8.00 GiB. Memory level (c) is therefore `no evidence` on this machine rather than a result, and every timing taken while an arm was spilled is marked `invalid (oversubscribed)` and enters no verdict. Both need a Linux/TCC driver, or a card the workload fits on, to become measurable.
* **The SDXL scale is a parameter bag, not SDXL.** `step_cost.py` allocates a tensor-shape census of an SDXL-like UNet and steps the optimizer on it. That reproduces the optimizer's cost at that parameter count and dtype; it runs no diffusion and says nothing about the resulting model.
* **The control battery is a synthetic proxy, not perception.** It ranks objective loss, overfitting gap and convergence on a small diffusion proxy at LRs roughly 100x a real fine-tune. There is no FID, no KID and no human judgement anywhere in this report: nothing here licenses a claim about image quality. Confirm on a real LoRA with FID/KID before believing the quality axis.
* **The battery runs 5 seeds.** Student-t on 5 points gives a wide interval; most small real differences will land on `n.s.`.
* **All timings come from a power-limited laptop GPU** (60 W on AC, 35 W on battery). Ratios measured adjacently inside one process survive that; absolute milliseconds do not transfer to a desktop or a datacentre card.
* **`_warn_if_inert` is armed on Nekaon and on no other arm.** Its device-to-host synchronisations are real cost for a user running this configuration today, but they are telemetry, not optimizer arithmetic -- a Nekaon speed loss of that magnitude is a property of the build, not of the algorithm.
