# K-Optimizers

> Memory-efficient PyTorch optimizers for **bf16 diffusion fine-tuning**.

[![status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#status)
[![license: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

`kaon` is a small collection of optimizers aimed at training diffusion models on
commodity GPUs, where optimizer state is precious and weights are bf16.

- **`Adafusion`** — a conv-aware factored optimizer. Reaches **AdamW-level quality
  at a fraction of AdamW's optimizer memory**, with bf16-correct weight updates
  (stochastic rounding — *no* Kahan buffer, *no* CPU offload).
  → [docs/adafusion.md](docs/adafusion.md)
- **`Muon`** — an orthogonalized-momentum optimizer (Newton-Schulz) with an AdamW
  fallback for 1-D/embedding params. Highest convergence quality, at half of
  AdamW's state. → [docs/muon.md](docs/muon.md)
- **`AdaMuon`** — **Muon's orthogonalized momentum + an Adafactor-style factored,
  quantized second moment** of the orthogonalized update. Aims to beat AdamW on
  convergence/precision at **near-Adafactor memory** (~1–2 B/param, int8/4bit dial).
  Tuned defaults `ns_steps=2`, `cautious=True`; optional `compile=True` (whole-step
  `torch.compile`, AdaMuon-only). → [docs/adamuon.md](docs/adamuon.md)
- **`Lion`** — **Lion's sign-momentum** (one buffer, no second moment) on
  Adafusion's backend (codec, stochastic-rounding bf16, cautious, foreach). Lightest
  state in the family — **~1 B (int8) / 0.5 B (4bit) per param** — with Lion's implicit
  regularization. `betas` are a loss↔generalization dial. → [docs/lion.md](docs/lion.md)
- **`AdaPNM`** — **Adam + Positive-Negative Momentum** (Xie et al. 2021) on the kaon
  backend. PNM's negative-momentum term injects anti-correlated noise — a built-in
  *implicit regularizer* (flat-minima seeking) **without** SAM's extra forward/backward.
  Tuned defaults `betas=(0.8, 0.999)`, `beta0=0.5` (`beta1` is the loss↔gap dial). The
  most **gap-robust at constant LR** (no schedule needed — resumable); experimental.
- **`KProdigy`** — a memory-efficient **Prodigy** (parameter-free D-adaptation):
  train at `lr=1.0` and the optimizer finds the effective LR itself. Matches
  reference Prodigy bit-for-bit at its defaults, then adds the kaon memory
  toolkit. → [docs/kprodigy.md](docs/kprodigy.md)
- **`Autofusion`** — a parameter-free LR on **Adafusion's** update via a
  [Mechanic](https://arxiv.org/abs/2306.00144) scalar tuner (Mechanic, *not*
  Prodigy): train at `lr=1.0` and it auto-discovers the LR, keeping Adafusion's
  exact update. Its headline is **freeze-to-free** (`lr_freeze`): after warmup it
  folds the discovered LR into the base, frees the tuner's `ref` buffer, and
  becomes **byte-for-byte and speed-for-speed plain Adafusion**.
  → [docs/autofusion.md](docs/autofusion.md)

`Adafusion`, `Muon`, `AdaMuon`, and `Lion` are standard `torch.optim.Optimizer`s that work
one-parameter-at-a-time, so they drop into per-parameter / gradient-release
training loops unchanged. `KProdigy` needs a global reduction over all parameters
each step (the D estimate), so it is a normal two-pass `step()` optimizer (no
gradient-release).

## Install

```bash
uv pip install git+https://github.com/Koronos/K-Optimizers
# or, from a clone:
uv pip install -e .
```

## Quickstart

```python
from kaon import Adafusion

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
from kaon import Muon

opt = Muon(model.parameters(), lr=2e-2, momentum_dtype="bfloat16")
```

```python
from kaon import AdaMuon

# Orthogonalized momentum + factored quantized 2nd moment; cautious + ns_steps=2
# are the tuned defaults. lr is Muon-scale: start ~1e-3 for diffusion (the API
# default 2e-2 is LLM/Muon-scale — see docs/adamuon.md). int8/4bit dial the memory.
opt = AdaMuon(model.parameters(), lr=1e-3, momentum_dtype="int8")
```

```python
from kaon import Lion

# Lion sign-momentum, lightest state (no second moment). lr is Lion-scale (~AdamW/5).
# betas are a loss<->generalization dial: (0.95,0.98) for loss, higher beta2 for less overfit.
opt = Lion(model.parameters(), lr=2e-4, betas=(0.95, 0.98), momentum_dtype="4bit")
```

```python
from kaon import KProdigy

# Parameter-free: lr stays 1.0; D adapts. For SDXL UNet+TE, pass two param
# groups and KProdigy gives each its own D automatically.
opt = KProdigy(model.parameters(), lr=1.0, momentum_dtype="bfloat16")
```

```python
from kaon import Autofusion

# Parameter-free Adafusion: lr stays 1.0; a Mechanic tuner finds the LR. By
# default (lr_freeze="auto") it freezes on an LR plateau, frees the tuner state,
# and runs as pure Adafusion at the discovered LR (free thereafter). The common
# case is just Autofusion(params, **adafusion_kwargs).
opt = Autofusion(
    model.parameters(),
    bf16_method="stochastic_rounding",   # adafusion_betas=(0.0, 0.999) by default
)                                        # => bit-exact freeze
```

## Recipes

| Goal | Configuration |
|---|---|
| **Minimum VRAM** (large model) | `Adafusion(..., betas=(0.0,0.999), bf16_method="stochastic_rounding")` |
| **LoRA / LoKr adapters** (many small weights) | `Adafusion(..., betas=(0.0,0.999), bf16_method="stochastic_rounding")` — `foreach=True` (default) batches the hundreds of adapter tensors |
| **AdamW-quality, low memory** | `Adafusion(..., betas=(0.9,0.999), momentum_dtype="bfloat16")` |
| **Lion8bit-class memory + momentum** | `Adafusion(..., betas=(0.9,0.999), momentum_dtype="int8")` |
| **Best convergence (memory available)** | `Muon(..., lr=2e-2, momentum_dtype="bfloat16")` |
| **Beat-AdamW precision at Adafactor memory** | `AdaMuon(..., lr=1e-3, momentum_dtype="int8")` — orthogonalized momentum + factored 2nd moment; `ns_steps=2`/`cautious=True` defaults, optional `compile=True` |
| **Absolute minimum state (no 2nd moment)** | `Lion(..., lr=2e-4, betas=(0.95,0.98), momentum_dtype="4bit")` — Lion sign-momentum at 0.5 B/param; implicit regularization for small-data fine-tuning |
| **No LR to tune (SDXL UNet+TE)** | `KProdigy([{"params": unet, "lr": 1.0}, {"params": te, "lr": 1.0}])` |
| **Parameter-free + minimum VRAM** | `KProdigy(..., second_moment="factored", momentum_dtype="bfloat16", slice_p=11)` |
| **No LR to tune + ~free after warmup** | `Autofusion(..., bf16_method="stochastic_rounding")` — auto-discovers the LR, then freezes to plain Adafusion |

## Docs

- [docs/adafusion.md](docs/adafusion.md) — Adafusion design, validated results,
  full API.
- [docs/autofusion.md](docs/autofusion.md) — the Mechanic LR tuner, freeze-to-free,
  the minimal API + the one advanced knob, and the campaign results.
- [docs/kprodigy.md](docs/kprodigy.md) — memory-efficient Prodigy design + API.
- [docs/muon.md](docs/muon.md) — Muon design + API.
- [docs/lion.md](docs/lion.md) — Lion (sign-momentum) design, the
  betas loss↔generalization dial, memory, and the proxy evaluation.
- [docs/adamuon.md](docs/adamuon.md) — AdaMuon design, the `clip_threshold`/`lr`
  validation, `compile=True`, and the pixel-DDPM evaluation.
- [docs/foreach-batching.md](docs/foreach-batching.md) — multi-tensor batching of
  the step (`foreach_batch_cutoff` / `foreach_stack_budget`).
- [docs/momentum.md](docs/momentum.md) — the cheap-momentum dial (int8 / bf16 /
  4bit) and the ideas that were rejected.

## Testing

```bash
uv run pytest
uv run ruff check src tests
```

## Status

v0.2 alpha. API may change.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
