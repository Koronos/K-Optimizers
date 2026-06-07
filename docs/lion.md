# Lion — sign-momentum on Adakaon's memory/precision backend

<!-- Developed under the provisional code name "Liofusion". -->

> **Lion** (Chen et al., *Symbolic Discovery of Optimization Algorithms*, arXiv:2302.06675)
> — a sign-of-momentum update with a single momentum buffer (no second moment) — running on
> **Adakaon's** quantized-momentum codec, stochastic-rounding bf16 weights, cautious masking,
> and foreach batching. The result: Lion's minimal state at **bf16/int8/4bit** momentum, with
> Adakaon's precision and the kaon memory toolkit.

## Why

Lion drops the second moment entirely (just one momentum buffer → half of Adam's state), and
its sign update has a known *implicit-regularization* flavour. For small-data diffusion
fine-tuning — where the enemy is memorization, not raw loss (see
[RESULTS_generalization_and_schedule.md](../benchmarks/adamuon/RESULTS_generalization_and_schedule.md))
— that regularization is exactly what you want. Lion is the vehicle to test "Lion's memory
+ implicit regularization" inside the kaon framework, A/B-comparable to Adakaon.

## The update (per parameter, decoupled weight decay)

```
c       = sign(beta1 * m + (1 - beta1) * g)   # direction from the OLD momentum + current grad
update  = c                                   # +1 / 0 / -1 per coordinate
p      -= lr * (update + weight_decay * p)    # decoupled WD, folded into the step
m       = beta2 * m + (1 - beta2) * g         # momentum EMA on the RAW grad, AFTER computing c
```

This is exactly Lion: the **direction** uses the `beta1`-interpolated momentum; the stored
momentum is then advanced with the (usually larger) `beta2`. The EMA is on the **raw gradient**
(unlike Adakaon, which takes momentum of the already-normalized update). No `rsqrt`, no
second-moment `eps`, no `clip_threshold` — the sign update is unit-magnitude per coordinate, so
there is no preconditioner blow-up to clip; each step is bounded by construction.

## Memory & the no-second-moment win

One momentum buffer, no second moment: **~2 B/param (bf16) / ~1 B (int8) / ~0.5 B (4bit)**.
The 4bit path (0.5 B/param) is Lion's signature — lighter than Adakaon (which still
keeps a small factored second moment) and far under AdamW (8 B). Memory is its strongest axis.

## Reused from Adakaon (shared backend — no duplication)

- **Momentum codec** (`bfloat16`/`int8`/`4bit`, `momentum_4bit_block`) — int8 per-row absmax,
  4-bit per-block nibble-packed.
- **Stochastic-rounding bf16 weight update** (`bf16_method="stochastic_rounding"`; also
  `kahan` / `none`) — bf16-correct, no Kahan buffer / CPU offload by default.
- **Cautious masking** (`cautious=True`, default): zero the coordinates where the update sign
  disagrees with the gradient (`update·g <= 0`), rescale the survivors. For Lion's pure-sign
  update this is a per-coordinate agreement filter.
- **Foreach batching** (`foreach`, `foreach_batch_cutoff`, `foreach_stack_budget`): stacked
  multi-tensor ops bucketed by shape — **bit-exact vs the per-parameter path** (verified). This
  is the decisive win for **LoRA/LoKr** (hundreds of tiny adapter tensors → one stacked op
  instead of a kernel launch per tensor).
- **dtype-safe checkpointing** (`load_state_dict_preserving_dtypes`) — quantized momentum is not
  upcast to fp32 on resume.

## API

```python
Lion(
    params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, *,
    cautious=True, momentum_dtype="bfloat16", momentum_4bit_block=128,
    bf16_method="stochastic_rounding", foreach=True,
    foreach_batch_cutoff=2_000_000, foreach_stack_budget=None,
)
```

- **`lr` is Lion-scale** — the sign update has unit per-coordinate magnitude, so use an lr
  **~3–10× smaller** than your AdamW/Adakaon lr.
- **`betas=(β1, β2)` are a loss↔generalization dial** (measured on the synthetic proxy): `β2`
  (the momentum decay) trades loss for the train–val gap.
  - **`(0.95, 0.98)`** → lowest loss (the classic Lion betas; recommended starting point for
    convergence; on the proxy it beat the no-momentum Adakaon baseline on loss).
  - `(0.9, 0.99)` (the constructor default) → balanced.
  - `(0.9, 0.999)` / `(0.99, 0.999)` → higher β2 = smoother momentum = lower gap (more
    regularization), at a higher loss.
- **`weight_decay`** was ~neutral on the proxy; Lion's literature uses larger decoupled WD —
  re-tune on a real run.
- **`cautious`** lowers loss (raises the gap slightly) — the usual loss↔gap lever; on by default.

## Evaluation (synthetic pixel-DDPM proxy — directional)

On the registered proxy ([proxy_dataset.py](../benchmarks/adamuon/proxy_dataset.py),
train=32 / test=96, REX + progressive-resolution recipe), tuned Lion `(0.95, 0.98)` is
**Pareto-competitive**: at the larger C=128 / 2500-step setting it reached **AdaMuon's loss at
roughly half AdaMuon's train–val gap**, and beat the no-momentum Adakaon baseline on both
axes — at the lightest memory. The sign-momentum's implicit regularization shows *more* at the
larger/longer scale.

Caveat: this is the synthetic MSE-gap proxy, not real sample quality. Validate on a real LoRA
with the live `val/gap` metric + FID/KID before claiming a perceptual win — the proxy ranks
objective-overfitting, not perceptual fidelity.

## See also

- [adakaon.md](adakaon.md) — the backend Lion reuses.
- [momentum.md](momentum.md) — the int8/4bit momentum codec.
- [foreach-batching.md](foreach-batching.md) — the multi-tensor batching.
- [RESULTS_generalization_and_schedule.md](../benchmarks/adamuon/RESULTS_generalization_and_schedule.md)
  — why the train–val gap (not loss) is the objective for small-data fine-tuning.
