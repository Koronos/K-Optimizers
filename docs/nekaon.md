# Nekaon — Adakaon + k-step negative momentum-lookahead

> **Nekaon** is **Adakaon** plus one structural mechanism: between steps the live weights
> are displaced **k optimizer-steps ahead** along the smoothed (preconditioned, clipped,
> lr-scaled) update direction, so every gradient is evaluated at the *anticipated* point
> (extragradient / Nesterov-style) while the update lands on the true iterate. It is the
> answer to **SAM's main problem**: SAM buys its flat-minima bias with a second
> forward/backward per step (~2× the GEMM phase); Nekaon's perturbation costs **zero extra
> passes and zero extra memory** — the momentum buffer already exists, and the perturbation
> is recomputed from it on removal. Related published mechanism: **Momentum-SAM** (Becker
> et al. 2024, arXiv:2401.12033), shipped faithfully as the
> [`MSAM`](../src/kaon/msam.py) wrapper; Nekaon is the in-house variant that **measured
> better**: negative (downhill) direction instead of MSAM's uphill climb, and a
> **step-scaled** lookahead instead of a fixed weight-space radius.

## The update

```
# end of step t (inside opt.step(), after the Adakaon update):
w_live <- w + k * m_t        # m = Adakaon's momentum = EMA of the lr-scaled,
                             #     1/sqrt(v)-preconditioned, RMS-clipped update
# training loop: forward/backward  -> grad is evaluated AT the lookahead point
# start of step t+1:
w_live <- w_live - k * m_t   # exact removal (m unchanged in between)
adakaon_step(grad_at_lookahead)   # update lands on the TRUE weights
```

Everything else — the factored second moment, the int8/4bit momentum codec,
stochastic-rounding bf16 writes, cautious masking, gradient centralization, foreach
batching, dtype-preserving resume — is Adakaon, unchanged. `k=0` *is* Adakaon.

## Why step-units (the structural choice)

A fixed SAM/MSAM radius `rho` lives in weight-space units: the right value depends on the
model's weight scale, the LR, and the schedule — exactly the kind of knob that needs
re-tuning per model. Nekaon's `e = k * m` is measured in **optimizer steps**:

* it self-scales with the LR and any schedule (LR decays → the lookahead decays with it);
* the `1/sqrt(v)` preconditioning absorbs per-coordinate scale differences;
* the model's weight scale never enters.

Calibration on the control proxy: the best fixed radius (`rho=0.3`) translated to the
**same `k ≈ 1.7` at `beta1=0.2` and `beta1=0.9`** — the step-unit number is the invariant.
The mechanism's effect holds with `k` **fixed** while the LR varies ×0.5 / ×2 (the
robustness gate it had to pass to ship), and needs **no cross-parameter norm**, so it
batches into a few stacked ops per shape bucket.

## What it buys (control battery, 2026-06-10; lower is better)

The mechanism is a **generalization regularizer**, and `beta1` is the regime knob:

| config (wd=0.1, k=1.5) | held-out loss (const-LR) | train–val gap (const-LR) |
|---|---|---|
| **`beta1=0.5` (the default — canonical battery entry)** | **0.0806** | **+0.0056** (#1 of the field) |
| `beta1=0.2`, k=0 (baseline) | 0.0804 | +0.0084 |
| `beta1=0.2`, k=1.5 (gap mode) | 0.0879 | **+0.0046** (−45% vs its k=0 twin) |
| `beta1=0.7`, k=1.5 (frontier midpoint) | 0.0779 | +0.0091 |
| `beta1=0.9`, k=0 (baseline) | 0.0724 | +0.0148 |
| `beta1=0.9`, k=1.5 (fidelity mode) | 0.0711 | +0.0141 (~neutral, harmless) |

The default `beta1=0.5` was chosen over the equidistant-to-target `beta1=0.7`
deliberately: 0.5 *meets* the gap objective outright with noise margin (the
anti-memorization axis — the ranking signal for small-data fine-tuning), while 0.7
misses both axes slightly. Slide toward 0.7–0.9 when underfitting, toward 0.2 when
memorizing.

The default **dominates the previous best loss+gap combo** (Adakaon `beta1=0.2`:
0.0805 / +0.0090) — same constant-LR loss, **−38% gap** — and takes the continuity
table's #1 with ~0.015 *better* loss than the prior const-LR gap champion (AdaPNM:
+0.0061 at a collapsed 0.0954).

* **`beta1=0.2` — anti-memorization mode** (small-data LoRA, the dominant kaon use case):
  the lookahead cuts the train–val gap by ~45% vs its own k=0 twin, far below every other
  optimizer in the battery (prev. const-LR gap champion AdaPNM: +0.0068 at collapsed loss).
* **`beta1=0.9` — fidelity mode** (abundant data / underfit risk): near-best constant-LR
  loss of the field; the lookahead is ~neutral there, so leaving it on is harmless.
* `weight_decay=0.1` default: measured frontier-mover (improves loss AND gap together) on
  both bases.

Both rows are **constant-LR** numbers — Nekaon is built for resumable, schedule-free
training (the continuity scenario), where it keeps (rather than loses) its generalization.

## Cost

* **Memory:** identical to Adakaon at the same `momentum_dtype` (the perturbation is
  recomputed from the stored momentum — no persistent state of its own). The **default
  is `momentum_dtype="4bit"`: 0.56 B/param measured** (~14× less than torch fused
  AdamW's 8) — the quantized momentum carries the lookahead with no measurable loss
  (the whole dial is flat within the proxy's noise):

  | momentum_dtype | const-LR loss | const-LR gap | B/param |
  |---|---|---|---|
  | **4bit (default)** | 0.0802 | +0.0066 | **0.56** |
  | int8 | 0.0806 | +0.0064 | 1.04 |
  | bfloat16 | 0.0806 | +0.0056 | 2.03 |
* **Step time:** two extra perturbation passes per step; no extra forward/backward, no
  global sync — and at the default 4-bit they run through a dedicated Triton kernel
  (`_axpy_4bit_batched`: dequant + bf16-SR axpy in ONE launch per bucket; parity vs the
  torch path 6e-8, measured perturbation cost 4.4 → 0.6 ms on the C=128 proxy). **Pass
  `fused=True` on GPU** to also run the inner Adakaon through its Triton kernels (same
  math + state — parity ≤2.4e-7):

  | regime (battery) | Nekaon (4bit) | Nekaon-fused (4bit) |
  |---|---|---|
  | 512-tiny-tensor LoRA bag | 7.73 ms | **1.48 ms** |
  | C=128 UNet step | 18.0 ms | **15.8 ms** |

  (For scale: Adakaon-bf16 measures ~14.3/3.9 ms in the same tables at 3.6× the memory.)
  The remaining 4-bit residue is the inner native conv-path codec (quant/requant) —
  unexercised further because on a real DiT step (GEMM-bound) the optimizer is <1% of
  the iteration — vs SAM's +100%.

## Stability — the per-element climb bound

A real Cosmos LoKr run NaN'd at step ~406 (triggered by an extreme-aspect resolution
bucket). Same failure channel that once NaN'd AdaPNM: a near-zero factored col-EMA makes
the denominator explode on one channel, and the Adafactor RMS clip bounds the update's
*RMS*, **not its per-element max** — so a runaway channel concentrates ~`sqrt(n)*lr`
spikes on a few coordinates. The lookahead then *amplifies* what plain Adakaon survives:
the momentum accumulates the spike, the weights live displaced `k`-fold along it between
steps (feedback through the gradient), and the 4-bit codec smears a spiked block's absmax
over its 128 neighbours.

The guard (always on, no new knob): every coordinate's climb is capped at

```
|e_i| <= |k| * clip_threshold * lr      # "no further than k maximum-allowed update steps"
```

frozen at climb time per group (an LR-scheduler change between steps must not corrupt the
exact removal). Inactive in the normal regime (typical `|m_i| ~ lr`); it bites exactly on
the runaway channel. The same cap runs inside the Triton 4-bit kernel. This is the moral
twin of the `clip_threshold` that fixed AdaPNM's real-training divergence.

## train() / eval() contract

Between steps the live weights deliberately sit at the lookahead point. **Sampling,
validation and checkpointing must bracket with `opt.eval()` / `opt.train()`** (same
contract as Lookahead / Schedule-Free / MSAM); always checkpoint in eval mode — a
train-mode checkpoint stores perturbed weights and a fresh optimizer cannot know to remove
the displacement on resume.

```python
from kaon import Nekaon

opt = Nekaon(model.parameters(), lr=1e-4, k=1.5, betas=(0.2, 0.999))  # LoRA: gap mode
...
opt.eval()    # true weights
sample_or_checkpoint(model)
opt.train()   # back to the lookahead point
```

## Knobs

* `k` (default `1.5`) — lookahead distance in steps; the loss↔gap dial (`0` = Adakaon,
  larger = stronger regularization). `1.5` was the calibrated invariant on the proxy.
* `betas[0]` — the regime knob: `0.2` anti-memorization (small-data LoRA), `0.9` fidelity
  (full fine-tunes / abundant data). Must be > 0 (the lookahead rides the momentum).
* `weight_decay` (default `0.1`) — measured frontier-mover; lower it for adapters whose
  scale you don't want shrunk (LoKr factors at high adapter weights).
* Everything else (`momentum_dtype` int8/4bit, `cautious`, `gradient_centralization`,
  `foreach`, `bf16_method`) is Adakaon's, defaults unchanged.

## Negative results recorded on the way (see the [graveyard](EXPERIMENTS_GRAVEYARD.md))

Uphill (SAM-sign) climb — dominated by the negative direction on both axes. Lookahead/
Schedule-Free averaging — does not cut the constant-LR noise floor here. PAdam partial
adaptivity — underfits at fixed budget. `lr_const` micro-tuning — slides the frontier
without moving it (and is exactly the proxy-tuned knob the design avoids).
