# K-Optimizers

> Memory-efficient PyTorch optimizers for **bf16 diffusion fine-tuning**.

[![status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#status)
[![license: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

`kaon` is a small collection of optimizers aimed at training diffusion models on
commodity GPUs, where optimizer state is precious and weights are bf16.

- **`Nekaon`** — **Adakaon + k-step negative momentum-lookahead**: every gradient is
  evaluated *k optimizer-steps ahead* along the smoothed update direction, a
  flat-minima / anti-memorization bias at **zero extra passes and zero extra state**
  (it solves SAM's 2×-forward/backward problem) — and at **0.56 B/param by default**
  (4-bit momentum carries the mechanism with no measured loss; int8/bf16 dial up).
  `beta1` is the regime knob — `0.2` cuts the constant-LR train–val gap **−45%** vs
  its own no-lookahead twin (field record), `0.9` is the fidelity mode; the lookahead
  `k` is measured in *steps*, so it self-scales across LRs/schedules/models (validated
  under LR ×0.5/×2 at fixed `k`). → [docs/nekaon.md](docs/nekaon.md)
- **`Adakaon`** — a conv-aware factored optimizer. Reaches **AdamW-level quality
  at a fraction of AdamW's optimizer memory**, with bf16-correct weight updates
  (stochastic rounding — *no* Kahan buffer, *no* CPU offload).
  → [docs/adakaon.md](docs/adakaon.md)
- **`AdaMuon`** — **Muon's orthogonalized momentum + an Adafactor-style factored,
  quantized second moment** of the orthogonalized update. Aims to beat AdamW on
  convergence/precision at **near-Adafactor memory** (~1–2 B/param, int8/4bit dial).
  Tuned defaults `ns_steps=2`, `cautious=True`; optional `compile=True` (whole-step
  `torch.compile`, AdaMuon-only). → [docs/adamuon.md](docs/adamuon.md)
- **`Lion`** — **Lion's sign-momentum** (one buffer, no second moment) on
  Adakaon's backend (codec, stochastic-rounding bf16, cautious, foreach). Lightest
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
- **`Autokaon`** — a parameter-free LR on **Adakaon's** update via a
  [Mechanic](https://arxiv.org/abs/2306.00144) scalar tuner (Mechanic, *not*
  Prodigy): train at `lr=1.0` and it auto-discovers the LR, keeping Adakaon's
  exact update. Its headline is **freeze-to-free** (`lr_freeze`): after warmup it
  folds the discovered LR into the base, frees the tuner's `ref` buffer, and
  becomes **byte-for-byte and speed-for-speed plain Adakaon**.
  → [docs/autokaon.md](docs/autokaon.md)

`Adakaon`, `AdaMuon`, and `Lion` are standard `torch.optim.Optimizer`s that work
one-parameter-at-a-time, so they drop into per-parameter / gradient-release
training loops unchanged. `KProdigy` needs a global reduction over all parameters
each step (the D estimate), so it is a normal two-pass `step()` optimizer (no
gradient-release).

📊 **Head-to-head rankings** across speed / convergence / generalization / memory / constant-LR
robustness live in the reproducible **[control battery](benchmarks/control/RANKINGS.md)** — and
adding a new optimizer to the tables is a one-liner (see [benchmarks/README.md](benchmarks/README.md)).

## Install

```bash
uv pip install git+https://github.com/Koronos/K-Optimizers
# or, from a clone:
uv pip install -e .
```

## Quickstart

```python
from kaon import Adakaon

# Minimum-VRAM recipe (replaces "Adafactor beta1=0 + Kahan offloaded to CPU"):
opt = Adakaon(
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
from kaon import Autokaon

# Parameter-free Adakaon: lr stays 1.0; a Mechanic tuner finds the LR. By
# default (lr_freeze="auto") it freezes on an LR plateau, frees the tuner state,
# and runs as pure Adakaon at the discovered LR (free thereafter). The common
# case is just Autokaon(params, **adakaon_kwargs).
opt = Autokaon(
    model.parameters(),
    bf16_method="stochastic_rounding",   # adakaon_betas=(0.0, 0.999) by default
)                                        # => bit-exact freeze
```

## Recipes

Each line below is a **complete call** — paste it, swap in your `model.parameters()`, and
re-tune `lr` on your data. The **benchmark behind each recipe** (real numbers + links) lives in
**[docs/recipes.md](docs/recipes.md)**.

**Small-data fine-tuning (LoRA / DreamBooth)** — rank by the train–val *gap*, not the loss:

```python
from kaon import AdaPNM, Lion, Adakaon

# Best generalization; the only one happy on a constant (resumable) LR:
AdaPNM(model.parameters(), lr=2e-3, betas=(0.8, 0.999), beta0=0.5)

# Lightest state (no 2nd moment), regularizing sign-momentum — ~0.5 B/param:
Lion(model.parameters(), lr=2e-4, betas=(0.95, 0.98), momentum_dtype="4bit")

# Minimum VRAM, regularizing (no momentum) — near-zero optimizer state:
Adakaon(model.parameters(), lr=1e-4, betas=(0.0, 0.999), bf16_method="stochastic_rounding")
```

**Maximum quality / fastest convergence-to-quality:**

```python
from kaon import AdaMuon, Adakaon

# Beat AdamW on convergence at Adafactor memory (~1 B/param int8; ns_steps=2/cautious defaults):
AdaMuon(model.parameters(), lr=1e-3, momentum_dtype="int8")

# AdamW-quality at 1/4–1/8 the memory (bf16 = 2 B/param; int8 = ~1 B/param, near-lossless):
Adakaon(model.parameters(), lr=1e-4, betas=(0.9, 0.999), momentum_dtype="bfloat16")
Adakaon(model.parameters(), lr=1e-4, betas=(0.9, 0.999), momentum_dtype="int8")
```

**Parameter-free (no LR to tune):**

```python
from kaon import KProdigy, Autokaon

# Prodigy D-adaptation; one LR for SDXL UNet + text-encoder (each gets its own D):
KProdigy([{"params": unet.parameters(), "lr": 1.0},
          {"params": text_encoder.parameters(), "lr": 1.0}])

# Parameter-free AND minimum VRAM (~1.3 B/param on the SDXL UNet):
KProdigy(model.parameters(), lr=1.0, second_moment="factored", momentum_dtype="int8", slice_p=11)

# Mechanic tuner that freezes to byte-for-byte plain Adakaon after warmup:
Autokaon(model.parameters(), bf16_method="stochastic_rounding")
```

`foreach=True` (the default) batches many-small-tensor (LoRA/LoKr) steps — **318 ms → 15 ms** on
a 1434-adapter SDXL UNet. See [docs/foreach-batching.md](docs/foreach-batching.md).

## Docs

- [docs/recipes.md](docs/recipes.md) — every recipe as a paste-ready call **plus the
  benchmark that justifies it** (numbers + links + how to reproduce).
- [docs/adakaon.md](docs/adakaon.md) — Adakaon design, validated results,
  full API.
- [docs/autokaon.md](docs/autokaon.md) — the Mechanic LR tuner, freeze-to-free,
  the minimal API + the one advanced knob, and the campaign results.
- [docs/kprodigy.md](docs/kprodigy.md) — memory-efficient Prodigy design + API.
- [docs/lion.md](docs/lion.md) — Lion (sign-momentum) design, the
  betas loss↔generalization dial, memory, and the proxy evaluation.
- [docs/adapnm.md](docs/adapnm.md) — AdaPNM (Positive-Negative Momentum) design, the
  beta1 loss↔gap frontier, the constant-LR robustness result, memory, and API.
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
