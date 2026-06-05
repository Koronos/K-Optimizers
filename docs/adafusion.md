# Adafusion — design & API

> A conv-aware factored optimizer: **AdamW-level quality at a fraction of AdamW's
> optimizer memory**, with bf16-correct weight updates (stochastic rounding — *no*
> Kahan buffer, *no* CPU offload).

## Why

To keep AdamW's per-coordinate adaptivity you normally pay two full state buffers
(8 B/param). Adafusion factors the second moment **conv-aware** (reshape
`[out,in,kh,kw] → [out, in·kh·kw]` before factoring → near-zero state on convs
*and* attention) and keeps an optional momentum buffer in **bf16 or int8**,
recovering AdamW-quality convergence at 1–2 B/param. Stochastic rounding does the
bf16-correct update with **no extra state**, so unlike Adafactor+Kahan you never
allocate (or CPU-offload) a compensation buffer.

## Results (validated)

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
([foreach-batching.md](foreach-batching.md)).

> Honest caveat: small-scale benchmarks. At *zero* optimizer state
> (Adafactor-class), AdamW-quality is not achievable — momentum (~1–2 B/param) is
> the floor for the quality. Adafusion gives you the dial; see
> [momentum.md](momentum.md) for the int8/bf16/4bit momentum trade-offs.

> Note: in HF Adafactor, `beta1=0.0` (≠ `None`) still allocates a momentum buffer.
> `Adafusion(betas=(0.0, ...))` is true no-momentum.

## API

```python
Adafusion(
    params, lr=1e-3, betas=(0.9, 0.999), eps=(1e-30, 1e-3), weight_decay=0.0, *,
    clip_threshold=1.0,
    momentum_dtype="bfloat16",          # "float32" | "bfloat16" | "int8" | "4bit"
    momentum_4bit_block=128,            # block size for 4bit momentum
    cautious=True,                      # cautious masking; helps w/ momentum, no-op without (set False if beta1=0)
    bf16_method="stochastic_rounding",  # "stochastic_rounding" | "kahan" | "none"
    foreach=True,                       # multi-tensor batching (foreach-batching.md)
    foreach_batch_cutoff=2_000_000,     # weights bigger than this loop instead of stacking
    foreach_stack_budget=None,          # chunk memory cap (None = adaptive to free VRAM)
)
```

`Adafusion` is a standard `torch.optim.Optimizer` that works one parameter at a
time, so it drops into per-parameter / gradient-release training loops unchanged.

## See also

- [foreach-batching.md](foreach-batching.md) — the multi-tensor batching design
  and the `foreach_batch_cutoff` / `foreach_stack_budget` knobs.
- [momentum.md](momentum.md) — why int8 is the recommended cheap momentum and what
  cheaper ideas were rejected.
- [autofusion.md](autofusion.md) — a parameter-free LR on top of this update.
