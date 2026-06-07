# Adafusion `foreach` batching — design & tuning

`Adafusion` steps parameters with multi-tensor (stacked) ops instead of a
per-parameter Python loop (`foreach=True`, the default). This note explains how it
works, the two knobs that control it, and the measurements behind their defaults.

## Why

The per-parameter loop launches a separate set of CUDA kernels for every weight.
That cost (~0.22 ms/tensor of dispatch on an RTX 4080) is invisible for a handful
of large weights but dominates when many tensors are stepped at once — exactly the
adapter case. On a real SDXL UNet + PEFT LoRA r=8 (1434 tiny trainable tensors) the
optimizer step was **318 ms** vs **11 ms** for fused AdamW.

`foreach` buckets parameters by shape, stacks each bucket into one tensor, and runs
the whole update (EMA + reconstruction + RMS clip + momentum + weight decay +
cautious + stochastic rounding) as a handful of batched kernels:

- `ndim >= 2` → factored bucket `[N, R, C]`
- `ndim == 1` (biases/norms, the bulk of a full fine-tune) → non-factored `[N, L]`

It is **element-for-element equal** to the per-parameter path (bit-exact on CPU,
~1e-8 on CUDA from reduction order; stochastic-rounding draws legitimately differ,
unbiased either way). Anything it doesn't cover — 0-D scalars, `momentum_dtype=
"int8"`, `bf16_method="kahan"`, fp16+SR, non-contiguous matrixized convs,
single-param (gradient-release) optimizers — transparently falls back to the loop.

### Measured

| workload | per-param loop | foreach | vs fused AdamW |
|---|---|---|---|
| SDXL + LoRA r=8 (1434 tensors) | 318 ms | **15 ms** | 1.45× |
| SDXL full fine-tune (1680 tensors) | 339 ms | **256 ms** | — |
| Cosmos full fine-tune (685 tensors) | 239 ms | **231 ms** | — |

foreach is a large win for adapters (many tiny tensors, launch-bound) and a modest
one for full fine-tunes (dominated by large bandwidth-bound weights — there is less
launch overhead to remove there).

## The two knobs (deliberately decoupled)

A single "budget" used to control both *which* weights get batched and *how many*
are stacked at once. Those are different concerns and pulling them apart matters:

### 1. `foreach_batch_cutoff` — performance (default `2_000_000` elements)

Weights with more elements than the cutoff are stepped by the loop instead of being
stacked. Batching only pays off while per-tensor launch overhead dominates; a large
weight's update is compute/bandwidth-bound, so stacking it just adds copy traffic
and is *slower*.

This is an **absolute element count, not a fraction of VRAM** — the crossover is a
hardware property, not a memory-budget one. A budget sweep on SDXL and Cosmos full
fine-tunes (perf cutoff varied directly):

```
cutoff      131k   500k    1M     2M     4M     8M    16M
SDXL ms     272    272    266    277    272    398    547
Cosmos ms   231    232    236    235    234    326    394
```

Both models show a broad flat optimum up to ~4 M and a sharp slowdown beyond — the
*same* crossover, confirming it does not depend on the model. `2_000_000` sits in
the middle of the plateau. Raise it only if profiling your GPU shows a higher
crossover.

> **Why not auto-tune it online?** The optimum is a broad, flat plateau (a 30×
> range of cutoffs is within noise), it is static per (model, GPU), and online
> step-time is noisy and buried under the forward/backward. A hill-climber would
> chase noise on a flat surface to rediscover a constant. A fixed default wins.

### 2. `foreach_stack_budget` — memory safety (default `None` = adaptive)

The max elements in a single stacked chunk. Stacking allocates a few transient fp32
copies of the chunk, so an unbounded bucket of large weights can OOM a full
fine-tune. The budget bounds that.

- `None` → `min(adaptive_to_free_VRAM, 4 × foreach_batch_cutoff)`.
  - The VRAM term, `free_bytes × 0.10 / 48`, shrinks the chunk when a big model
    already fills the card and is the OOM-safety floor. The `48` is the measured
    peak transient bytes per stacked element (see below).
  - The `4 × cutoff` cap stops *over-stacking*: beyond a few cutoff-sized tensors,
    stacking medium weights just adds copy bandwidth. Measured on SDXL full FT
    (cutoff fixed at 2 M):

    ```
    chunk budget   4M    8M   16M   32M   64M   100M
    ms            281   261   318   350   354    355
    ```

    8 M (= 4 × 2 M) is the sweet spot; bigger is slower. Tying the cap to the
    cutoff keeps a single performance knob and means a roomy card never over-stacks.
- `int` → a fixed cap, returned verbatim (reproducibility, or a hard ceiling on a
  shared GPU). Not subject to the `4× cutoff` cap — you asked for an exact value.

Because the two are decoupled, **raising the stack budget never pulls large weights
into stacking** — it only allows bigger chunks of the already-eligible small ones.

#### The transient factor (`48 bytes/element`) is model-independent

The VRAM term divides by the peak transient bytes per stacked element. This is a
property of the optimizer's intermediate tensors, not the model — measured
byte-for-byte identical on SDXL and Cosmos shapes, and independent of tensor size,
aspect ratio, and conv-vs-linear. It depends only on path and config:

| path | common (`beta1=0`+SR) | worst (momentum+wd+cautious) |
|---|---|---|
| 2-D factored | 24 B | 38 B |
| 1-D non-factored | 28 B | 42 B |

`48` = worst observed (42.1) + margin, so the budget is a true ceiling: a chunk's
transient stays at or below the requested VRAM fraction.

## Check on your own GPU + model (`ktune`)

The defaults were tuned on an RTX 4080. The performance cutoff tracks a hardware
crossover, so a very different GPU *might* prefer another value. `ktune` measures it
on your machine — it reads the parameter shapes from a checkpoint's `.safetensors`
header (no full load, no real weights needed), builds matching tensors on your GPU,
sweeps the cutoff, and tells you whether to keep the default or change it:

```bash
# full fine-tune of a UNet / DiT transformer
uv run ktune --model /path/to/unet.safetensors --gpu 0

# a full SDXL checkpoint: select the UNet keys
uv run ktune --model /path/to/sdxl.safetensors --filter model.diffusion_model.

# the LoRA-adapter distribution instead of full weights
uv run ktune --model /path/to/unet.safetensors --lora-rank 8

# match your training config so the timing is representative
uv run ktune --model /path/to/transformer.safetensors --momentum bf16 --wd 0.01
```

It prints the per-param-loop baseline, the foreach speedup, a cutoff sweep, and a
verdict like `==> Keep the default` or `==> Consider foreach_batch_cutoff=…`.
Equivalent without the console script: `python -m kaon.tune --model …`.

## Tuning cheat-sheet

- **Default (`None`, `2_000_000`)**: near-optimal for SDXL/Cosmos LoRA *and* full
  fine-tune on any card. Start here.
- **Shared GPU / want a hard memory ceiling**: pin `foreach_stack_budget=<int>`
  (e.g. `2_000_000`).
- **Exotic GPU where profiling shows large weights still benefit from stacking**:
  raise `foreach_batch_cutoff` (the stack cap follows at 4×).
- **Disable batching entirely**: `foreach=False` (per-parameter loop; e.g.
  gradient-release setups already step one param per optimizer).
