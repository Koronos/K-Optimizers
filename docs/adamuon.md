# AdaMuon — design & API

> Muon's Newton-Schulz orthogonalized momentum + an Adafactor-style **factored,
> quantized second moment of the orthogonalized update**. Aims to beat AdamW on
> precision at near-Adafactor memory. Separate from `Muon` (the simpler
> heavy-ball hybrid) and from `Adakaon` (whose backend it reuses).

## Why

A deep-research sweep (adversarially fact-checked) found the best-evidenced
direction for *beating* AdamW on convergence/precision without blowing up memory
is **orthogonalized momentum (Muon) + variance adaptation (the "Ada" part)** —
AdaMuon improves on plain Muon by adding an Adam-style second moment on the
orthogonalized update, and that variance adaptation is the credited factor. All
published evidence is on LLMs, not diffusion: the fine-detail-fidelity advantage
for SDXL/Flux is the hypothesis this optimizer is built to test, not an
established fact.

AdaMuon clones ~90% of `Adakaon`'s backend (factored second moment, int8/4bit
momentum codec, foreach batching, stochastic rounding, dtype-safe checkpointing)
and inserts the orthogonalization.

## Pipeline (and how it differs from Adakaon)

`Adakaon`: factor the second moment of the **gradient** → normalize → take
momentum of the **normalized update**.

`AdaMuon` reverses the order for ≥2-D weights:

1. **First moment of the RAW gradient** — `m = β1·m + (1-β1)·g`, kept quantized
   (bf16/int8/4bit) in the shared codec.
2. **Orthogonalize** `m` with a Newton-Schulz iteration (`ns_steps`, default **2**)
   → `O ≈ U·Vᵀ`.
3. **Factored second moment OF `O`** (row+col EMA) → `u = O · inv_sqrt(v)` — with no
   bias correction unless `bias_correction=True`.
4. **RMS scale** to a shape-independent target, then apply at `lr`.

1-D params (biases, norm scales) are **not** orthogonalized — they use
Adakaon's non-factored Adam-style path (full per-coordinate second moment, same
quantized momentum), RMS-normalized to the same target so one `lr` governs the
whole model.

### Update-norm: why `0.2`, not `0.2·√max(R,C)`

Plain Muon scales `O` (RMS `≈1/√max(R,C)`) by `0.2·√max(R,C)` for a
shape-independent applied RMS of `0.2`. In AdaMuon the factored `inv_sqrt(v)`
already brings `u` to RMS `≈1` (its `c_factor ≈ √max(R,C)`), so reapplying
`√max(R,C)` would double-count the shape and make the update grow with layer
size. AdaMuon therefore scales by the **constant** `0.2` only.

### `clip_threshold` is a first-order hyperparameter, not a safety net

`clip_threshold` (default `1.0`) is an RMS ceiling in the RMS≈1 domain, and it is
**load-bearing**: turning it *off* costs **+24 %** final val even at the optimal lr,
and at a 16×-too-high lr it nearly diverges (0.158 vs 0.064).

The reason is that the factored second moment carries **no bias correction** by
default, so `v` is cold by a factor `1-β₂ᵗ` and `1/√v` is correspondingly too large.
Measured `rms(u)` *before* the clip on a real proxy-U-Net run (C=64, lr=1.2e-3,
β₂=0.999; mean and max over the 2-D weight buckets, and the share of buckets holding
at least one weight above the ceiling):

| step | mean `rms(u)` pre-clip | max | buckets above `clip=1.0` |
|---|---|---|---|
| 1 | 31.9 | 32.8 | 100 % |
| 10 | 11.0 | 13.0 | 100 % |
| 100 | 3.56 | 4.09 | 100 % |
| 1000 | 1.20 | 1.34 | 100 % |
| 2000 | 1.07 | 1.24 | 80 % |
| 2999 | 0.98 | 1.11 | 70 % |

Those are almost exactly `1/√(1-β₂ᵗ)`. So the clip is active on essentially **every**
step for the first `~1/(1-β₂)` iterations and on 70-80 % of buckets after that: it is
what sets the effective step size early in training — **not** "a near no-op in steady
state", as this document and the docstring used to claim. Put another way, the clip is
itself acting as the second moment's bias correction (see `bias_correction` below).

That also explains why `clip` and `lr` are **coupled**: clip caps the applied RMS at
`≈ clip·0.2·lr`, so `clip<1` at a good lr just lowers the effective lr (measured:
`clip 0.5 ≡ lr/2`, `clip 0.25 ≡ lr/4`) and a tight clip *rescues* a too-high lr.
**Tune lr; leave `clip=1.0`** (it sits exactly at the RMS≈1 knee — beats both 0.5,
which throttles, and 2.0, which lets early spikes through). See the evaluation note
for the numbers.

### Momentum semantics ≠ `Muon`

`Muon` uses heavy-ball (`m = momentum·m + g`) + Nesterov; **AdaMuon uses an
Adam-style EMA lerp** (`m = β1·m + (1-β1)·g`) — the canonical AdaMuon form, and
what the shared codec implements (so int8/4bit momentum and bit-exact checkpoint
resume come for free). A learning rate tuned for `Muon` will not transfer
directly.

## Memory

Factored second moment (row+col, ~0) + one quantized first moment: ~2 B/param
(bf16) / ~1 B (int8) / ~0.5 B (4bit). Adafactor-class, well under AdamW. Newton-
Schulz runs in bf16 internally regardless of `momentum_dtype`.

## API

```python
AdaMuon(
    params, lr=2e-2, betas=(0.95, 0.999), eps=(1e-30, 1e-3), weight_decay=0.0, *,
    ns_steps=2, clip_threshold=1.0, bias_correction=False, momentum_dtype="bfloat16",
    momentum_4bit_block=128, cautious=True, bf16_method="stochastic_rounding",
    foreach=True, foreach_batch_cutoff=2_000_000, foreach_stack_budget=None,
)
```

- `lr` is Muon-scale (larger than Adam); a single `lr` covers 2-D and 1-D (all
  normalized to applied RMS `≈0.2·lr`).
- `betas=(β1, β2)`: `β1` first-moment EMA (`β1=0` → no momentum buffer); `β2`
  factored second-moment decay.
- `cautious` is **on by default** (validated): a paired pixel-DDPM sweep showed it
  flips AdaMuon from a loss to a win vs Adakaon (~2% on all seeds). `ns_steps` is
  **2** by default, not the LLM-standard 5 — chosen because it was both faster and
  lower-val on a paired pixel-DDPM sweep. See the spectrum note below for what
  `ns=2` actually does (it is *not* "5 over-orthogonalizes"). Both re-tune per task.
- `foreach=True` batches the step (bucketed by shape, with a batched `bmm`
  Newton-Schulz) — the decisive win for LoRA/LoKr (hundreds of tiny 2-D weights).
  The batched 2-D path matches the per-parameter path within bf16 NS tolerance
  (both unbiased); 1-D buckets and all fp32 ops are bit-exact.

### What `ns_steps=2` actually does (aspect-ratio dependent)

Singular values of the Newton-Schulz factor of an iid-Gaussian matrix (5 draws per
cell, mean / min / 10th percentile of the whole spectrum):

| shape | aspect | ns=1 | ns=2 | ns=3 | ns=5 |
|---|---|---|---|---|---|
| (256, 256) | 1:1 | 0.18 / 0.00 / 0.03 | **0.56 / 0.00 / 0.12** | 0.85 / 0.00 / 0.40 | 0.89 / 0.00 / 0.70 |
| (1024, 1024) | 1:1 | 0.09 / 0.00 / 0.02 | **0.31 / 0.00 / 0.06** | 0.80 / 0.00 / 0.20 | 0.89 / 0.01 / 0.70 |
| (256, 2304) | 9:1 | 0.21 / 0.14 / 0.16 | **0.68 / 0.48 / 0.54** | 1.11 / 0.86 / 0.95 | 0.97 / 0.73 / 0.76 |
| (128, 1152) | 9:1 | 0.30 / 0.20 / 0.23 | **0.89 / 0.65 / 0.74** | 0.85 / 0.67 / 0.69 | 0.87 / 0.67 / 0.69 |
| (64, 256) | 4:1 | 0.41 / 0.21 / 0.27 | **1.06 / 0.68 / 0.85** | 0.83 / 0.67 / 0.69 | 0.85 / 0.68 / 0.69 |
| (16, 256) | 16:1 | 0.78 / 0.61 / 0.67 | **1.00 / 0.78 / 0.82** | 0.79 / 0.67 / 0.68 | 0.80 / 0.67 / 0.68 |

Reads:

- **The quintic does not converge to 1; it settles into a band ≈[0.67, 1.20].** More
  steps do not push the spectrum past that band, so "5 over-orthogonalizes" is not
  what the numbers show — at ns≥3 every aspect ratio sits inside the band.
- **On skinny matrices (LoRA/adapter shapes, and conv weights matrixized to
  `(out, in·kh·kw)`) `ns=2` is already essentially orthogonal** (mean 0.89–1.06,
  min ≥0.48). There is little left for steps 3–5 to do, so paying for them buys
  nothing — which is where the "faster AND not worse" result comes from.
- **On square matrices `ns=2` is heavily *under*-orthogonalized** — mean 0.56 at
  256², 0.31 at 1024², with a long tail of near-zero singular values. There the
  update still points largely along the momentum's dominant directions; whether
  that helps or hurts is a per-task empirical question, and the proxy U-Net (whose
  2-D weights are 8:1–9:1 after matrixization) never exercises the square case.
- The default is therefore an **empirical sweep result on skinny-weight models**,
  not a statement about orthogonality. On a model with large square weights
  (transformer QKV/MLP blocks, 1:1) re-sweep `ns_steps` — 3 is where the square
  spectrum first fills out.
- Caveat: measured on iid-Gaussian matrices. A real momentum matrix has a decaying
  spectrum, so the absolute numbers shift; the aspect-ratio *ordering* is what the
  table is for.


### `bias_correction`: why the factored `v` does not need it (default off)

`bias_correction=True` divides the factored second moment by `1 - β₂ᵗ`, per parameter.
It costs one multiply and no extra state, because of an identity worth writing down:

```
r_factor = rsqrt(row / mean(row))          <- a RATIO of row stats: 1/(1-β₂ᵗ) cancels
c_factor = rsqrt(col / (1-β₂ᵗ)) = √(1-β₂ᵗ) · rsqrt(col)
```

so correcting `v` reduces **exactly** to scaling the normalized update by `√(1-β₂ᵗ)`,
applied before the clip. `t` is tracked **per parameter** (`state["step"]`,
checkpointed), not as a global step count: a parameter that only sometimes receives a
gradient (MoE routing, CFG dropout, partial gradient accumulation) has a colder second
moment than the run's step count suggests, and a global `t` would under-correct it by
orders of magnitude.

**The default is `False`, and the reason is that `clip_threshold=1.0` already does the
same job — harder.** The uncorrected `rms(u)` is almost exactly `1/√(1-β₂ᵗ)` (31.9 at
step 1, 3.56 at 100, 1.20 at 1000 — see the clip table above), which is precisely the
factor the correction removes. So while the clip binds, *both* configurations emit an
update whose RMS is exactly `clip`, and they are the same update. Measured applied RMS
(in units of `0.2·lr`, single 64² weight, β₂=0.999, `clip=1.0`):

| step | `√(1-β₂ᵗ)` | applied RMS, `bias_correction=False` | ...`=True` | ratio |
|---|---|---|---|---|
| 1 | 0.032 | 1.0000 | 1.0000 | 1.0000 |
| 10 | 0.100 | 1.0000 | 0.9991 | 0.9991 |
| 100 | 0.309 | 1.0000 | 1.0000 | 1.0000 |
| 1000 | 0.795 | 1.0000 | 0.9953 | 0.9953 |
| 3000 | 0.975 | 1.0000 | 0.9905 | 0.9905 |

At `clip=1.0` the correction is therefore a **≤1 % per-tensor rescale**, never more.
It is not a fix for a bug the clip was hiding: the clip *is* a bias correction — a
cruder, per-tensor-normalizing one that is strictly stronger early on, which is why
turning the clip off costs +24 % val while the correction is worth ~1 %.

The paired-seed A/B on the pixel-DDPM proxy (same dataset/model/schedule as the control
battery, C=128, N=2000, REX, `int8` momentum, `ns_steps=2`, `cautious=True`, identical
init/data/noise per seed; lr swept upward because the correction can only *shrink*
the step) reads:

| lr | seed | val off | val **on** | gap off | gap **on** | on better on both? |
|---|---|---|---|---|---|---|
| 1.2e-03 | 0 | 0.06931 | 0.07084 | 0.02151 | 0.02204 | **no** |
| 1.2e-03 | 1 | 0.07120 | 0.07178 | 0.02187 | 0.02280 | **no** |
| 1.2e-03 | 2 | 0.07213 | 0.07182 | 0.02224 | 0.02164 | yes |
| 1.2e-03 | **mean** | **0.07088** | **0.07148** | **0.02187** | **0.02216** | |
| 2.4e-03 | 0 | 0.07732 | 0.07914 | 0.01997 | 0.02247 | **no** |
| 4.8e-03 | 0 | 0.07762 | 0.07645 | 0.01729 | 0.01404 | yes |

Across the 5 paired runs `bias_correction=True` matched or beat `False` on held-out loss
in 2/5, on the train-val gap in 2/5, and on **both** in 2/5. The rule set before running
was "flip the default only if it wins on loss AND gap on every seed", so the default
stays `False`.

At the tuned lr the spread is a couple of percent of val and **not consistent across
seeds** (2 of 3 go against the correction, 1 for it) — about what a ≤1 % per-tensor
rescale should look like once a 2000-step trajectory has had time to diverge from it.
The lr sweep is the tell: at 4× the tuned lr the correction *does* win on both metrics,
which is exactly how a small **effective-lr reduction** behaves — the same shape as the
`clip 0.5 ≡ lr/2` result above. It is a step-size knob, not a new mechanism, and at the
lr you would actually tune to there is no upside worth a step counter and a multiply.
So the default is unchanged.

The configuration where the correction *is* the normalizer is the one that
shows what it is for — same proxy, lr 1.2e-3, seeds averaged:

| arm | train | val | gap |
|---|---|---|---|
| `clip=1.0`, no correction (**default**) | 0.04850 | 0.07009 | 0.02158 |
| clip off, no correction | 0.07593 | 0.08699 | 0.01105 |
| clip off, **`bias_correction=True`** | 0.04928 | 0.07139 | 0.02211 |

(`n = 2` seed(s) per arm. Read the gap next to the train loss: clip-off-without-correction only looks
gap-friendly because it underfits.)

That is the useful reading of the whole exercise: **the clip and the correction are
two implementations of the same normalization, and either one works — the clip is
just slightly better here.** Disabling the clip without the correction is what breaks
(the classic +24 % result); disabling it *with* the correction gets you back to within
~2 % of the default — it recovers 92 % of the clip's benefit.

**When to turn it on.** When you raise or disable `clip_threshold` and still want the
update at RMS≈1 — the correction is the principled way to do it, and it turns the clip
back into a true safety ceiling rather than the thing setting your step size. Likewise
when composing AdaMuon with a trust-region / auto-lr layer that assumes an unbiased
`1/√v`, or with a short `β₂` horizon where you want the early steps to be genuinely
small rather than clipped to the ceiling. With the default `clip_threshold=1.0` there
is nothing to gain.

## Performance: `torch.compile` (`compile=True`)

`AdaMuon(..., compile=True)` compiles the step's **pure-tensor math** — the factored
2-D bucket kernel, the non-factored `ndim<=1` kernel, and their per-parameter twins —
fusing each one's elementwise chain (Newton-Schulz + factored second moment + clip +
scale + cautious). The parameter bookkeeping (the `p.grad is not None` filter, the
shape bucketing, the momentum codec, the state write-back) stays in eager Python.

### Why not compile the whole step

It used to. Compiling the step body made Dynamo guard on things that change during
normal training:

- **one guard per parameter on whether that parameter has a gradient** — because the
  body evaluates `p.grad is not None` for every parameter inside the traced frame;
- **one guard on the literal value of `group["lr"]`**.

So a MoE / CFG-dropout / partial-gradient-accumulation step (the grad set moves) or
*any LR schedule* (lr moves every step) recompiled the whole step until
`recompile_limit` (8) was reached, after which `torch.compile` **silently falls back
to eager for the rest of the run**. Measured on a 6-weight bag: 8 compiled graphs in
both scenarios; `add_param_group` cost 2 further recompiles (~8.2 s). Now: **1 graph**
in all of them, and `add_param_group` costs ~4 ms.

Guards are on shapes and dtypes only, which automatic-dynamic generalizes after the
second distinct shape — so a model with many weight shapes settles into a handful of
dynamic graphs instead of one per shape, and stops recompiling. Step-varying scalars
(`lr`, and the `bias_correction` factor) are handed to the kernels as 0-D tensors under
compile precisely so Dynamo cannot specialize on their value; the multiply is
bit-identical to the Python-float multiply eager uses.

### Measured speedup

`opt.step()` microbench, RTX 3000 Ada, eager vs compiled ratio (<1 = compiled is
faster), constant lr, one scenario per process (Dynamo caches per code object, so
measuring several optimizers in one process is not representative):

| param set | ratio | |
|---|---|---|
| 12 small **distinct**-shaped 2-D weights (defeat `foreach`) | **0.76×** | helps |
| U-Net-like mix (convs + 2-D + 1-D, 11 tensors) | **0.84×** | helps |
| 128 identical LoRA-shaped weights (`foreach`-batched) | **0.71×** | helps |
| single 2048² weight (per-parameter path) | **0.71×** | helps |
| two 512² weights | **0.58×** | helps |
| 4×1024² + 4×1024 full-fine-tune-like | ~1.00× | neutral |

### What the guard fix cost and bought

With a *constant* lr and a *fixed* grad set, the old whole-step graph could specialize
every bucket shape into a single artifact and beat the new per-bucket kernels on
multi-shape models. **Attach an LR schedule — as every real run has — and that peak
disappears**, because the old graph recompiled itself out of existence. Both regimes,
eager-vs-compiled ratio, one process each:

| param set | lr | old whole-step | per-bucket (now) |
|---|---|---|---|
| 12 distinct small 2-D weights | constant | 0.27× | 0.76× |
| 12 distinct small 2-D weights | **scheduled** | **1.03×** (compile did nothing) | **0.29×** |
| U-Net-like mix | constant | 0.41× | 0.84× |
| U-Net-like mix | **scheduled** | **1.00×** (compile did nothing) | **0.79×** |

So the fix loses peak throughput only in a regime you cannot train in, and in the
regime you do train in it turns `compile=True` from a no-op into a 1.3–3.4× step
speedup. The `foreach`-batched LoRA bag and the single-huge-weight cases got faster
even at constant lr (0.94 → 0.71 and 0.97 → 0.71), because the compiled kernel is no
longer wrapped in a whole-step graph.

The **eager** path (`compile=False`, the default) is unaffected by the restructure:
interleaved before/after medians over the same six scenarios come out at 0.965-1.009×,
i.e. unchanged to within noise, with the four multi-tensor cases slightly ahead thanks
to the zero-copy single-slice stacking.

(Absolute ms are not comparable across rows — the GPU was shared — which is why every
row is a within-process ratio.) Benchmark your own workload, and remember the
optimizer is usually a small fraction of a real diffusion step (SDXL is UNet-bound).

A realistic multi-shape check: the U-Net-like set (6 distinct bucket shapes) with an
LR schedule *and* a rotating grad set compiles **10** graphs across the 6 kernels over
30 steps and never trips `recompile_limit` — it reaches a steady compiled state
instead of degrading.

One-time warmup (~4 s for the first graph). Numerically equivalent to eager: the
0-D-tensor scalars are bit-identical, and the one place the compiled kernel is
*written* differently — weight decay as `p * (lr·wd)` instead of `add_(alpha=lr·wd)`,
because an `alpha=` float would specialize the graph — is fused away by Inductor
(measured: the non-orthogonalized 1-D/0-D buckets come out **bit-identical** compiled
vs eager, with `weight_decay` on). The residual difference is Inductor reassociating
the **bf16 Newton-Schulz**, so it shows up only on `ndim>=2` weights and at ~1e-5
relative — the same order as the existing foreach-vs-per-param bf16 gap, and no larger
than before this change. Stochastic rounding stays unbiased — no host syncs in the
step. Not recommended on CPU (inconsistent).

Note: compiling *only* the Newton-Schulz does **not** help on LoRA-rank matrices
(too small — wrapper overhead exceeds the fusion gain); the win is fusing the whole
bucket. This flag is **AdaMuon-only by design**: [`Adakaon`](adakaon.md) has little
fusable elementwise math (no orthogonalization), so a compile was ~neutral there and
not worth the API surface — Adakaon stays lean.

## Checkpointing

`load_state_dict` is overridden (shared `load_state_dict_preserving_dtypes`
helper) so a quantized first moment is not silently upcast to fp32 on resume —
preserving both the memory and bit-exact resume.

State also carries a per-parameter `state["step"]` (a plain `int`, always maintained
even with `bias_correction=False`, so the flag can be switched on mid-run or across a
resume without the correction restarting from a cold `t`). Checkpoints written before
the counter existed load fine and restart that parameter's `t` at 1.

`load_state_dict` also **backfills** hyperparameters the checkpoint predates. torch
replaces each `param_groups` dict wholesale with the checkpoint's (carrying over only
`params`), so a key added after the checkpoint was written would otherwise disappear
from the live group and raise `KeyError` on the next step. Keys the checkpoint carries
win, so a resumed run keeps its own tuning; only genuinely missing ones fall back to the
constructor defaults.

## Evaluation (v1, self-contained pixel-DDPM proxy)

Paired-seed A/B vs `Adakaon` (and `AdamW8bit` / `Lion8bit` / fp32 AdamW) on a
small pixel-space DDPM (conv UNet, C=128, 3 seeds, identical init/data/noise per
seed, LR swept per arm, held-out val MSE). Not real SDXL/Flux — a first signal.

- **Defaults matter.** With the *original* defaults (`ns_steps=5`, `cautious=False`)
  AdaMuon **lost** to Adakaon (0.0710 vs 0.0697). Two changes flipped it:
  `cautious=True` (helps ~2%, all seeds) and `ns_steps=2` (faster *and* better here
  — see the spectrum note above for why, which is not "5 over-orthogonalizes"). Both
  are now the defaults.
- **Tuned (`ns_steps=2`, `cautious=True`, lr 1e-3) AdaMuon wins on everything that
  matters:**
  - convergence/step — reaches each val target in ~35 % fewer steps;
  - convergence/wall-clock — reaches `val≤0.070` in ~7.9 s vs Adakaon ~9.6 s,
    *despite* ~12 vs ~9.4 ms/step (faster convergence beats slower steps);
  - final quality — floors at ~0.065 vs Adakaon's ~0.069 (which it never beats);
  - memory — tie (2.03 B/param).
- Comfortably beats AdamW8bit (0.0762) / Lion8bit (0.0767) / fp32 AdamW (~0.088).
- **`clip_threshold` / `lr` re-validation** (single-res 64², REX, bs8, 2 seeds, eval@64²):

  | lr @ clip=1.0 | val | | clip @ lr=1.2e-3 | val | clip @ lr=1e-2 (stress) | val |
  |---|---|---|---|---|---|---|
  | 6e-4 | 0.0649 | | 0.25 | 0.0685 | 0.25 | 0.0635 |
  | **1.2e-3** | **0.0619** | | 0.50 | 0.0649 | 0.50 | 0.0681 |
  | 2.4e-3 | 0.0628 | | **1.00** | **0.0619** | 1.00 | 0.0715 |
  | 1e-2 | 0.0715 | | 2.00 | 0.0629 | 2.00 | 0.0773 |
  | 2e-2 (API default) | 0.0776 | | off | 0.0770 | off | 0.1577 |

  Reads: proxy `lr*=1.2e-3` (default 2e-2 is ~16× high, +25%); `clip=1.0` is the optimum
  *and* load-bearing (off = +24% at the good lr, near-divergence at high lr); `clip<1`
  acts as an effective-lr cap (0.5≡lr/2, 0.25≡lr/4); tune lr, keep `clip=1.0`.
- Open work: claw back the per-step gap (lower the non-NS overhead, bf16 the
  post-NS factored region) and re-find the sweet spot at scale.

## Known caveats / to validate

- All "beats AdamW" evidence is **LLM, not diffusion** — validate fine-detail
  fidelity empirically on SDXL/Flux LoRA before claiming the result.
- The factored second moment is computed on `O` (near-orthonormal, small
  magnitude); its mean scale differs from gradient-based Adakaon. **Re-validated on
  the synthetic proxy** (single-res 64², REX, 2 seeds): `clip_threshold=1.0` is the
  optimum (load-bearing, see above) and the proxy `lr*≈1.2e-3` — note the **API default
  `lr=2e-2` is Muon/LLM-scale, ~16× high for this diffusion proxy** (val 0.078 vs 0.062
  at 1.2e-3; it degrades gracefully, never diverges — the §1 robustness). Lower the lr
  by ~10–16× from the Muon default for diffusion-scale work; still re-confirm on a real
  SDXL/Flux run (proxy ≠ real model).
- Newton-Schulz on tiny LoRA matrices is launch-bound; the batched `bmm` path
  mitigates it but the crossover vs `foreach=False` should be profiled per GPU.

## Follow-ups (not in v1)

- Revisit automatic step-size control only after a model-agnostic safety signal
  is validated on real fine-tuning workloads; the current AutoLR is quarantined.

## See also

- [muon.md](muon.md) — the simpler heavy-ball Muon hybrid this builds on.
- [adakaon.md](adakaon.md), [kprodigy.md](kprodigy.md),
  [foreach-batching.md](foreach-batching.md),
  [momentum.md](momentum.md).
