# Nekaon — Adakaon + k-step negative momentum-lookahead

> **Nekaon** is **Adakaon** plus one structural mechanism: between steps the live weights
> are displaced **k optimizer-steps ahead** along the smoothed (preconditioned, clipped)
> update direction scaled by the current lr, so every gradient is evaluated at the *anticipated* point
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
w_live <- w + k * lr * m_t   # m = Adakaon's momentum = EMA of the
                             #     1/sqrt(v)-preconditioned, RMS-clipped update
                             #     DIRECTION (LR-independent since 0.7.11);
                             #     * lr converts it to steps at the CURRENT lr
# training loop: forward/backward  -> grad is evaluated AT the lookahead point
# start of step t+1:
w_live <- w_live - k * lr * m_t   # exact removal (m and the frozen lr unchanged)
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
  global sync — and at the default 4-bit they run through the shared Triton momentum
  kernel (`_axpy_momentum_batched`: dequant + bf16-SR axpy in one launch per bucket).
  **Pass
  `fused=True` on GPU** to also run the inner Adakaon through its Triton kernels (same
  math + state). Large/conv 4-bit state now updates directly in-kernel with no fp32
  momentum-sized temporary:

  | regime (RTX 3000 Ada, cautious + GC) | Nekaon native (4bit) | Nekaon fused (4bit) |
  |---|---|---|
  | 512-tiny-tensor LoRA bag | 52.77 ms | **0.63 ms** |
  | large 512×512 bag | 42.74 ms | **5.39 ms** |
  | large convolution bag | 27.38 ms | **3.83 ms** |

  Standard 64/128-element 4-bit blocks take the direct route. Non-aligned custom
  blocks retain the compatible fallback rather than silently changing codec semantics.

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

## Low LR / Kahan (`bf16_method="kahan8"` / `"kahan16"`)

With bf16 weights every write rounds to the bf16 grid (ulp = 2⁻⁸ relative). When the
per-step update is a fraction of an ulp — full fine-tunes at lr ≲ 1e-5, long low-LR tails —
stochastic rounding (the default) stays unbiased but random-walks away from the fp32
trajectory. The compact-Kahan methods keep the missing bits in a per-parameter residual
(`state["kahan_lo"]`, see [`docs/research/compact-kahan.md`](research/compact-kahan.md)):

```python
opt = Nekaon(model.parameters(), lr=1e-5, bf16_method="kahan8")   # no other configuration
```

| method | extra state | total optimizer state (4-bit momentum) | what it tracks |
|---|---|---|---|
| `"stochastic_rounding"` (default) | 0 | **0.56 B/param** | the bf16 weight, unbiased |
| `"kahan8"` | +1 B/param (uint8) | **1.56 B/param** | the value to ulp/256 (SR at the residual's grid) |
| `"kahan16"` | +2 B/param (int16) | **2.56 B/param** | an fp32 master, exactly |

When to use which:

* **SR** — steps of about an ulp or more (LoRA/adapter LRs, pre-training), or when every
  byte counts. On the low-LR proxy (`benchmarks/lowlr_bf16`, 8000 steps) its test loss was
  within seed noise of fp32 even where it drifted 8–28 ulp from the fp32 trajectory.
* **`kahan8`** — sub-ulp LRs; the default choice for low-LR bf16 fine-tunes. Measured with
  Nekaon at lr 1e-5: 0.26 ulp from the fp32 run vs 8.6 for SR (0.7.15, before the decoded
  decay below). Its residual walk grows as √steps (~0.2 ulp at 10k, 0.5 at 100k).
* **`kahan16`** — fp32-master numerics: very long sub-ulp runs, reference arms, or whenever
  +2 B/param is affordable. Given the same gradients, a `kahan16` Nekaon **is** the
  fp32-weight Nekaon of the same route, bit for bit — with the defaults (wd 0.1, Gradient
  Centralization, cautious), on per-param, foreach and every fused route (since 0.7.16: the
  decay reads the decoded value and GC runs in fp32 — see below).

What Nekaon does with the residual:

* The lookahead climb and its removal go through the DECODED value (`bf16 + residual`), so
  the climb/removal pair leaves the clean value intact (to ~ulp/256 for `kahan8`, exactly for
  `kahan16`). The forward sees the nearest bf16, which moves when the decoded value crosses a
  rounding boundary — an unbiased dither of the sub-ulp climb. The inert-lookahead warning
  knows this (it compares with the residual grid, not the bf16 ulp): at lr 1e-5 it fires for
  SR and not for `kahan8`/`kahan16`.
* Weight decay (`delta += wd·p`, both `cautious_wd` placements, native and fused, keep pass
  included) reads the decoded value. Gradient Centralization of a bf16 gradient runs in fp32
  on the copy the update reads; `p.grad` itself is left uncentralized under these methods.
* Every route and option works unchanged: fused / foreach / per-param, `low_vram_above` (the
  momentum-free group is compensated too), mixed fp32/bf16 groups (only bf16 params get a
  residual), `add_param_group`, parameters without gradients, every `momentum_dtype`,
  `cautious`, GC, eval/train, and a mid-run method switch (SR → kahan: zero residual, one
  warning; kahan8 ↔ kahan16: converted, one warning; kahan → SR: the stale residual is ignored).

Checkpoints and export — **the residual only means something next to the bf16 weight it was
written for**:

* **Save in eval mode** (as always with Nekaon): `opt.eval()`, save the model and the
  optimizer, `opt.train()`. The model's bf16 tensor is then the *nearest* bf16 to the full
  value — the right thing for bf16 inference; the full value is bf16 + residual.
* **Resume: load the optimizer AFTER the model.** Load the model weights first, then build
  the optimizer (or keep it) and `opt.load_state_dict(...)`. Loading model weights after the
  optimizer state (or re-initialising, pruning, EMA-swapping them) leaves residuals attached
  to bf16 values they were not written for. A checkpoint saved with another `bf16_method`
  brings its method with it (torch restores every group hyperparameter) — the load warns;
  set `opt.param_groups[i]["bf16_method"]` afterwards to switch back.
* **fp32 export** — to save the full-precision weights (an fp32 checkpoint, serving in fp32,
  continuing elsewhere without the optimizer), decode them; works for any kaon optimizer
  using `kahan8`/`kahan16`:

  ```python
  opt.eval()
  torch.save(kaon.full_precision_state_dict(model, opt), "model_fp32.pt")  # names + buffers
  values = kaon.decode_weights(opt)          # or {param: fp32 tensor}
  opt.train()
  ```

  Both refuse to run while Nekaon's live weights carry the climb (call `eval()` first).

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
