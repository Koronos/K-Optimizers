# K-Optimizers

> Memory-efficient PyTorch optimizers for **bf16 diffusion fine-tuning**.

[![status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#status)
[![license: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

`koptim` is a small collection of optimizers aimed at training diffusion models
on commodity GPUs, where optimizer state is precious and weights are bf16.

- **`Adafusion`** — a conv-aware factored optimizer. Reaches **AdamW-level
  quality at a fraction of AdamW's optimizer memory**, with bf16-correct weight
  updates (stochastic rounding — *no* Kahan buffer, *no* CPU offload).
- **`Muon`** — an orthogonalized-momentum optimizer (Newton-Schulz) with an
  AdamW fallback for 1-D/embedding params. Highest convergence quality, at half
  of AdamW's state.
- **`KProdigy`** — a memory-efficient **Prodigy** (parameter-free
  D-adaptation): train at `lr=1.0` and the optimizer finds the effective LR
  itself. Matches reference Prodigy bit-for-bit at its defaults, then adds the
  koptim memory toolkit (bf16/int8 momentum, factored second moment,
  stochastic-rounding bf16 updates, per-group independent D for SDXL UNet+TE).

`Adafusion` and `Muon` are standard `torch.optim.Optimizer`s that work
one-parameter-at-a-time, so they drop into per-parameter / gradient-release
training loops unchanged. `KProdigy` needs a global reduction over all
parameters each step (the D estimate), so it is a normal two-pass `step()`
optimizer (no gradient-release).

## Install

```bash
uv pip install git+https://github.com/Koronos/K-Optimizers
# or, from a clone:
uv pip install -e .
```

## Quickstart

```python
from koptim import Adafusion

# Minimum-VRAM recipe (replaces "Adafactor beta1=0 + Kahan offloaded to CPU"):
opt = Adafusion(
    model.parameters(),
    lr=1e-4,
    betas=(0.0, 0.999),                  # beta1=0 -> no momentum (near-zero state)
    bf16_method="stochastic_rounding",   # no Kahan buffer, no CPU<->GPU offload
)

for batch in loader:
    loss = model(batch); loss.backward()
    opt.step(); opt.zero_grad()
```

```python
from koptim import Muon

opt = Muon(model.parameters(), lr=2e-2, momentum_dtype="bfloat16")
```

```python
from koptim import KProdigy

# Parameter-free: lr stays 1.0; D adapts. For SDXL UNet+TE, pass two param
# groups and KProdigy gives each its own D automatically.
opt = KProdigy(model.parameters(), lr=1.0, momentum_dtype="bfloat16")
```

## KProdigy — why

[Prodigy](https://arxiv.org/abs/2306.06101) estimates the distance `D` to the
solution on the fly and uses it as the effective learning rate — no LR to tune,
no schedule. The catch in practice is memory (reference Prodigy keeps *four*
fp32 buffers — double AdamW) and fragile defaults. `KProdigy` keeps the exact
D-estimation math but:

- stores the first moment in **bf16 / int8** and (optionally) the second moment
  **factored** (Adafactor row+col), with **stochastic-rounding** bf16 weight
  updates — the same toolkit as `Adafusion`, so D-adaptation no longer costs
  more memory than AdamW;
- ships **sane defaults** (`d_update_freq=1`, `use_bias_correction=False`). The
  original research repo defaulted these the other way and it *starved the
  D-bootstrap* — the effective LR failed to rise. See
  `benchmarks/bench_kprodigy_d.py` for the measured trajectories;
- gives each param group its **own D** (`independent_d`, auto-on for >1 group)
  so on SDXL the UNet and Text Encoder don't burn each other's learning rate.

> Status: the full-precision path (bf16 momentum + full fp32 second moment)
> reproduces reference Prodigy to ~1e-4 on D and is the recommended default.
> `second_moment="factored"` is experimental (it inflates D somewhat — measure
> on your model first). With bf16 weights, keep `bf16_method="stochastic_rounding"`:
> at `d0=1e-6` the early updates are tiny and naive bf16 rounding truncates them
> to zero, stalling the D-bootstrap.

## Adafusion — why

To keep AdamW's per-coordinate adaptivity you normally pay two full state
buffers (8 B/param). Adafusion factors the second moment **conv-aware** (reshape
`[out,in,kh,kw] -> [out, in·kh·kw]` before factoring → near-zero state on convs
*and* attention) and keeps an optional momentum buffer in **bf16 or int8**,
recovering AdamW-quality convergence at 1–2 B/param. Stochastic rounding does the
bf16-correct update with **no extra state**, so unlike Adafactor+Kahan you never
allocate (or CPU-offload) a compensation buffer.

### Results (validated)

Mini pixel-DDPM on real CC0 images, held-out validation, 4 seeds:

| optimizer | val loss (↓) | optimizer state |
|---|---|---|
| AdamW | 0.0400 ± 0.0025 | 8 B/param |
| AdamW-8bit | 0.0364 | 2 B/param |
| **Adafusion** (bf16 momentum) | **0.0318 ± 0.0006** | **2 B/param** |

Beats AdamW by ~20% on held-out diffusion loss (non-overlapping across seeds) at
1/4 the optimizer memory. On a real 2.1 B-param DiT transformer, the no-momentum
config uses **0.01 GB** of optimizer state (vs AdamW's 8.4 GB), and `foreach`
batching (default) keeps its per-step cost competitive with fused AdamW
([docs/foreach-batching.md](docs/foreach-batching.md)).

> Honest caveat: small-scale benchmarks. At *zero* optimizer state
> (Adafactor-class), AdamW-quality is not achievable — momentum (~1–2 B/param)
> is the floor for the quality. Adafusion gives you the dial.

## Recipes

| Goal | Configuration |
|---|---|
| **Minimum VRAM** (large model) | `Adafusion(..., betas=(0.0,0.999), bf16_method="stochastic_rounding")` |
| **LoRA / LoKr adapters** (many small weights) | `Adafusion(..., betas=(0.0,0.999), bf16_method="stochastic_rounding")` — `foreach=True` (default) batches the hundreds of adapter tensors |
| **AdamW-quality, low memory** | `Adafusion(..., betas=(0.9,0.999), momentum_dtype="bfloat16")` |
| **Lion8bit-class memory + momentum** | `Adafusion(..., betas=(0.9,0.999), momentum_dtype="int8")` |
| **Best convergence (memory available)** | `Muon(..., lr=2e-2, momentum_dtype="bfloat16")` |
| **No LR to tune (SDXL UNet+TE)** | `KProdigy([{"params": unet, "lr": 1.0}, {"params": te, "lr": 1.0}])` |
| **Parameter-free + minimum VRAM** | `KProdigy(..., second_moment="factored", momentum_dtype="bfloat16", slice_p=11)` |

> Note: in HF Adafactor, `beta1=0.0` (≠ `None`) still allocates a momentum
> buffer. `Adafusion(betas=(0.0, ...))` is true no-momentum.

> Adafusion batches the step across parameters by default (`foreach=True`) — a
> ~19× faster optimizer step on adapter training. Defaults are tuned for both
> LoRA and full fine-tune; see [docs/foreach-batching.md](docs/foreach-batching.md)
> for the design and the `foreach_batch_cutoff` / `foreach_stack_budget` knobs.

## API

```python
Adafusion(
    params, lr=1e-3, betas=(0.9, 0.999), eps=(1e-30, 1e-3), weight_decay=0.0, *,
    clip_threshold=1.0,
    momentum_dtype="bfloat16",          # "float32" | "bfloat16" | "int8" | "4bit"
    momentum_4bit_block=128,            # block size for 4bit momentum
    cautious=True,                      # cautious masking; helps w/ momentum, no-op without (set False if beta1=0)
    bf16_method="stochastic_rounding",  # "stochastic_rounding" | "kahan" | "none"
    foreach=True,                       # multi-tensor batching (docs/foreach-batching.md)
    foreach_batch_cutoff=2_000_000,     # weights bigger than this loop instead of stacking
    foreach_stack_budget=None,          # chunk memory cap (None = adaptive to free VRAM)
)

Muon(
    params, lr=2e-2, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0, *,
    adamw_lr=3e-4, adamw_betas=(0.9, 0.999), adamw_eps=1e-8, adamw_weight_decay=0.0,
    bf16_method="stochastic_rounding", momentum_dtype="float32",
)

KProdigy(
    params, lr=1.0, betas=(0.9, 0.999), beta3=None, eps=1e-8, weight_decay=0.0, *,
    decouple=True,
    use_bias_correction=False,          # keep off (the repo's True default hurt D)
    safeguard_warmup=False, d0=1e-6, d_coef=1.0, growth_rate=float("inf"),
    d_update_freq=1,                    # keep 1 (>1 starves the D-bootstrap)
    slice_p=1,                          # 11 -> ~11x less D-state, ~0.3% D error
    independent_d=None,                 # None -> auto: on when >1 param group
    momentum_dtype="bfloat16",          # "float32" | "bfloat16" | "int8"
    second_moment="full",               # "full" | "factored" (experimental)
    eps_factored=1e-30,
    bf16_method="stochastic_rounding",  # "stochastic_rounding" | "kahan" | "none"
    factor_conv_as_matrix=True,
)
```

## Testing

```bash
uv run pytest
uv run ruff check src tests
```

## Status

v0.2 alpha. API may change.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
