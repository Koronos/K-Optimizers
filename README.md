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

Both are standard `torch.optim.Optimizer`s and work one-parameter-at-a-time, so
they drop into per-parameter / gradient-release training loops unchanged.

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
    compile=True,                        # ~+30% on large 2-D weights (DiT/transformer)
)

for batch in loader:
    loss = model(batch); loss.backward()
    opt.step(); opt.zero_grad()
```

```python
from koptim import Muon

opt = Muon(model.parameters(), lr=2e-2, momentum_dtype="bfloat16")
```

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
config uses **0.01 GB** of optimizer state (vs AdamW's 8.4 GB); `compile=True`
brings its per-step cost in line with AdamW on large 2-D weights.

> Honest caveat: small-scale benchmarks. At *zero* optimizer state
> (Adafactor-class), AdamW-quality is not achievable — momentum (~1–2 B/param)
> is the floor for the quality. Adafusion gives you the dial.

## Recipes

| Goal | Configuration |
|---|---|
| **Minimum VRAM** (large model) | `Adafusion(..., betas=(0.0,0.999), bf16_method="stochastic_rounding", compile=True)` |
| **AdamW-quality, low memory** | `Adafusion(..., betas=(0.9,0.999), momentum_dtype="bfloat16")` |
| **Lion8bit-class memory + momentum** | `Adafusion(..., betas=(0.9,0.999), momentum_dtype="int8")` |
| **Best convergence (memory available)** | `Muon(..., lr=2e-2, momentum_dtype="bfloat16")` |
| **HF-Adafactor drop-in** | `Adafusion(..., betas=(0.0,0.999), decay_rate=-0.8)` |

> Note: in HF Adafactor, `beta1=0.0` (≠ `None`) still allocates a momentum
> buffer. `Adafusion(betas=(0.0, ...))` is true no-momentum.

## API

```python
Adafusion(
    params, lr=1e-3, betas=(0.9, 0.999), eps=(1e-30, 1e-3), weight_decay=0.0, *,
    clip_threshold=1.0,
    decay_rate=None,                    # HF Adafactor adaptive beta2 (e.g. -0.8)
    momentum_dtype="bfloat16",          # "float32" | "bfloat16" | "int8"
    cautious=False,                     # cautious masking (opt-in regularizer)
    bf16_method="stochastic_rounding",  # "stochastic_rounding" | "kahan" | "none"
    factor_conv_as_matrix=True,         # the conv-aware factoring fix
    compile=False,                      # torch.compile the factored core
)

Muon(
    params, lr=2e-2, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0, *,
    adamw_lr=3e-4, adamw_betas=(0.9, 0.999), adamw_eps=1e-8, adamw_weight_decay=0.0,
    bf16_method="stochastic_rounding", momentum_dtype="float32",
)
```

## Testing

```bash
uv run pytest
uv run ruff check src tests
```

## Status

v0.1 alpha. API may change.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
