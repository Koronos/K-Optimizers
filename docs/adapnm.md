# AdaPNM — Adam + Positive-Negative Momentum on Adakaon's memory/precision backend

<!-- Developed under the provisional code name "Janus". -->

> **AdaPNM** — the adaptive variant of **Positive-Negative Momentum** (Xie et al. 2021,
> *Positive-Negative Momentum: Manipulating Stochastic Gradient Noise to Improve
> Generalization*, ICML 2021, arXiv:2103.17182) — running on **Adakaon's** factored
> quantized second moment, int8/4bit momentum codec, stochastic-rounding bf16 weights,
> cautious masking, and foreach batching. It is the **generalization-bucket** optimizer: a
> built-in *implicit regularizer* (flat-minima seeking) that lowers the **train–val gap**
> without SAM's extra forward/backward.

## Why

For small-data diffusion fine-tuning the enemy is **memorization**, not raw loss — the
lowest-loss config overfits and produces *worse* samples (see
[RESULTS_generalization_and_schedule.md](../benchmarks/adamuon/RESULTS_generalization_and_schedule.md)).
The right ranking signal is the **train–val gap**, not the training loss. PNM is the one
optimizer in this library whose mechanism *directly targets the gap*: its negative-momentum
term injects controlled **anti-correlated gradient noise**, which biases SGD toward flatter
minima. On the synthetic gap proxy AdaPNM reaches ~Lion/AdamW loss at **36–44 % lower gap**,
and — the headline below — it is the **most gap-robust optimizer at constant LR**.

## The idea (PNM / AdaPNM)

Vanilla momentum *averages away* the stochastic gradient noise. PNM keeps **two** momentum
buffers, feeds the gradient to only **one** of them each step (alternating which), and forms
the update direction as a positive-negative mix:

```
pn         = ((1 + beta0) * m_pos  -  beta0 * m_neg) / noise_norm
noise_norm = sqrt((1 + beta0)^2 + beta0^2)
```

The `(1+beta0)` / `−beta0` coefficients **amplify the momentum signal and enlarge the variance
of the injected noise** in a controlled way; the larger, more isotropic noise is what seeks
flat minima. `noise_norm` renormalizes so the effective step magnitude is preserved. AdaPNM
then divides `pn` by the Adam second-moment denominator `sqrt(v_hat) + eps`, so it slots into
the factored-`v` backend directly.

## The exact update (per group, step `t`, decoupled weight decay)

```
t odd : (m_pos, m_neg) = (exp_avg,     neg_exp_avg)      # alternate which buffer is positive
t even: (m_pos, m_neg) = (neg_exp_avg, exp_avg)

m_pos  = beta1**2 * m_pos + (1 - beta1**2) * grad        # NOTE: first-moment decay is beta1^2
v      = beta2 * v + (1 - beta2) * grad**2               # factored Adam second moment
bc1    = 1 - beta1**t                                    # bias correction uses beta1 (not ^2)
denom  = sqrt(v / (1 - beta2**t)) + eps
pn     = ((1 + beta0) * m_pos - beta0 * m_neg) / sqrt((1+beta0)**2 + beta0**2)
p     -= (lr / bc1) * pn / denom
```

Two subtleties match kozistr's `pytorch_optimizer` AdaPNM verbatim: (1) the first-moment decay
is **`beta1**2`** while the **bias correction uses `beta1`**; (2) only the *positive* buffer is
EMA-updated each step — the negative buffer is the stale (one-step-old) momentum that gets
subtracted. `beta0` is kozistr's `beta3`.

## The dials — `betas` and `beta0`

**`beta1` is the loss↔gap dial** (the analogue of Lion's betas). On the proxy the loss is
**U-shaped in `beta1` and bottoms at 0.8**; the train–val gap stays in a tight low band across
`0.95 → 0.7`, then climbs below `~0.7` with no loss gain. The frontier
(`beta2=0.999`, `beta0=0.5`, `cautious=True`, C=128 / 2500 steps, 2 seeds):

| `beta1` | test loss | gap | |
|---|---|---|---|
| 0.95 | 0.0866 | +0.0065 | over-regularized |
| 0.90 (old default) | 0.0822 | +0.0079 | |
| 0.85 | 0.0817 | +0.0084 | |
| **0.80** | **0.0796** | **+0.0080** | **← elbow: shipped default** |
| 0.75 | 0.0813 | +0.0077 | |
| 0.70 | 0.0826 | +0.0080 | |
| 0.60 | 0.0817 | +0.0103 | gap tax begins |
| 0.50 | 0.0827 | +0.0116 | gap tax, no loss gain |

- **`betas=(0.8, 0.999)`** is the shipped default. Raise `beta1` toward `0.95` for more
  regularization (lower gap, higher loss); `beta2=0.999` is the sweet spot.
- **`beta0` ∈ [0, 1]`** is the positive-negative coefficient. `beta0=0` turns PNM **off**
  (plain debiased Adam-momentum) and is **measurably worse on the proxy → PNM is load-bearing**.
  `beta0=1` is canonical PNM `(2·m_pos − m_neg)/√5`. **`beta0=0.5`** (default) is the measured
  sweet spot.
- The scheduler is only a *minor* loss↔gap dial — across REX/cosine/linear/plateau the gap
  stays in `0.0059–0.0079`. **The gap is schedule-insensitive: PNM owns it, not the schedule.**

## The headline — robustness at **constant LR**

A decaying schedule lowers the gap *artificially* (the late-LR anneal is borrowed
regularization) and breaks resumable training (it needs the total step count `N` and dies at
`lr → 0`). The fine-tuning-realistic test is **constant LR**: no `N`, no anneal crutch, trivially
resumable. There, AdaPNM's self-injected noise carries the regularization that the schedule
used to provide. Best-gap per optimizer at constant LR (no schedule, C=128 / 2500, 2 seeds):

| optimizer (constant LR) | test loss | gap |
|---|---|---|
| **AdaPNM** | **0.0847** | **+0.0065** |
| AdamW | 0.0852 | +0.0101 |
| Lion | 0.0843 | +0.0103 |
| Adakaon (β1=0) | 0.0839 | +0.0114 |

AdaPNM has **35–43 % lower gap than the entire field at constant LR**, at equal loss — and its
gap even *improves* vs the scheduled run (`+0.0073 → +0.0065`). Its sweet spot is a **high**
constant LR (more noise → more regularization). This is the property that makes it a strong
default for open-ended / resumable fine-tuning runs.

## Memory

Two momentum buffers (the PNM pos/neg pair) + a factored second moment. The momentum floor is
therefore **~2× a single-buffer optimizer** — but both buffers go through the int8/4bit codec,
so: **~4 B/param (bf16) / ~2 B (int8) / ~1 B (4bit)** for momentum, plus the small factored `v`
(row+column EMAs, ~0 for 2-D weights). Heavier than Lion (one buffer) but far under AdamW (8 B);
the cost buys the gap behaviour.

## Reused from Adakaon (shared backend — no duplication)

- **Factored second moment** (`kaon._factored`) — row+column EMAs for `ndim≥2`, full `v` for
  `ndim==1`; conv kernels matrixized to `[out, in·kh·kw]`; Adafactor-style RMS-clipped
  `1/sqrt(v_hat)` reconstruction with `eps` folded in via `eps1`.
- **Momentum codec** (`bfloat16`/`int8`/`4bit`) — int8 per-row absmax, 4-bit per-block
  nibble-packed; here driven by AdaPNM's own raw-gradient EMA on the positive buffer.
- **Stochastic-rounding bf16 weight update** — bf16-correct, no Kahan buffer / CPU offload.
- **Cautious masking** (`cautious=True`) — see the note below.
- **Foreach batching** — bit-exact vs the per-parameter path; the decisive win for LoRA/LoKr.
- **dtype-safe checkpointing** (`load_state_dict_preserving_dtypes`) — quantized momentum is not
  upcast to fp32 on resume.

## API

```python
AdaPNM(
    params, lr=1e-3, betas=(0.8, 0.999), beta0=0.5, eps=1e-8, weight_decay=0.0, *,
    cautious=True, ams_bound=False, momentum_dtype="bfloat16", momentum_4bit_block=128,
    bf16_method="stochastic_rounding", foreach=True,
    foreach_batch_cutoff=2_000_000, foreach_stack_budget=None,
)
```

- **`betas=(beta1, beta2)`** — `beta1` is the loss↔gap dial (default `0.8`, the proxy elbow);
  `beta2=0.999` is the second-moment decay. (Internally the first-moment EMA decays at
  `beta1**2`, the bias correction at `beta1` — matching kozistr.)
- **`beta0`** — the pos-neg coefficient (default `0.5`); `0` = PNM off, `1` = canonical PNM.
- **`cautious`** — zeroes update coordinates that disagree with the current gradient. **Note the
  tension:** PNM's `−beta0·m_neg` term is *designed* to oppose the instantaneous gradient on
  noisy coordinates (that is the noise-manipulation mechanism), and cautious removes exactly
  those — so it partially damps the implicit regularizer. On by default for consistency; **this
  is the knob to ablate first** (`cautious=False`) when chasing the generalization benefit.
- **`ams_bound`** — defaults `False`. A factored `v` has no materialized matrix to take a
  running max over, so AMSBound applies only on the non-factored (1-D) path; on factored 2-D
  weights it is a no-op. (This is the one deliberate deviation from kozistr's AdaPNM default.)
- **`lr`** — Adam-scale, but AdaPNM's regularization is strongest at a *higher* constant LR than
  you would use with a decaying schedule; tune it on a real run.

## Evaluation (synthetic pixel-DDPM proxy — directional)

All numbers above are from the registered proxy
([dataset.py](../benchmarks/proxy/dataset.py), train=32 / test=96, ranked by the
deterministic train–val eps-MSE **gap**, not loss). AdaPNM is the proxy's **generalization
champion**: best gap among the usable-loss optimizers, and the clear winner at constant LR.

**Caveat:** this is the synthetic MSE-gap proxy, not perceptual sample quality. The proxy once
ranked a Muon-style optimizer above the actual visual-A/B winner. Validate on a real LoRA with
the live `val/gap` metric + FID/KID before claiming a perceptual win.

## See also

- [adakaon.md](adakaon.md) — the backend AdaPNM reuses.
- [lion.md](lion.md) — the sibling "lightest state" sign-momentum optimizer.
- [momentum.md](momentum.md) — the int8/4bit momentum codec.
- [RESULTS_generalization_and_schedule.md](../benchmarks/adamuon/RESULTS_generalization_and_schedule.md)
  — why the train–val gap (not loss) is the objective for small-data fine-tuning.
