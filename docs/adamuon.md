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
2. **Orthogonalize** `m` with a 5-step Newton-Schulz iteration → `O ≈ U·Vᵀ`.
3. **Factored second moment OF `O`** (row+col EMA) → `u = O · inv_sqrt(v̂)`.
4. **RMS scale** to a shape-independent target, then apply at `lr`.

1-D params (biases, norm scales) are **not** orthogonalized — they use
Adakaon's non-factored Adam-style path (full per-coordinate second moment, same
quantized momentum), RMS-normalized to the same target so one `lr` governs the
whole model.

### Update-norm: why `0.2`, not `0.2·√max(R,C)`

Plain Muon scales `O` (RMS `≈1/√max(R,C)`) by `0.2·√max(R,C)` for a
shape-independent applied RMS of `0.2`. In AdaMuon the factored `inv_sqrt(v̂)`
already brings `u` to RMS `≈1` (its `c_factor ≈ √max(R,C)`), so reapplying
`√max(R,C)` would double-count the shape and make the update grow with layer
size. AdaMuon therefore scales by the **constant** `0.2` only. `clip_threshold`
(default `1.0`) is an RMS ceiling in the RMS≈1 domain. It is **load-bearing, not
cosmetic**: a proxy sweep (REX, no warmup) showed that turning it *off* costs **+24 %**
final val even at the optimal lr — the early steps (cold factored `v̂`, full lr) spike
the update RMS and clip catches exactly those; at a 16×-too-high lr, off nearly diverges
(0.158 vs 0.064). It is near a no-op only in true steady state. `clip` and `lr` are
**coupled** — clip caps the applied RMS at `≈ clip·0.2·lr`, so `clip<1` at a good lr just
lowers the effective lr (measured: `clip 0.5 ≡ lr/2`, `clip 0.25 ≡ lr/4`) and a tight
clip *rescues* a too-high lr. **Tune lr; leave `clip=1.0`** (it sits exactly at the
RMS≈1 knee — beats both 0.5, which throttles, and 2.0, which lets early spikes through).
See the evaluation note for the numbers.

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
    ns_steps=2, clip_threshold=1.0, momentum_dtype="bfloat16",
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
  **2** by default, not the LLM-standard 5 — 5 over-orthogonalizes here (slower AND
  worse); 2 was the sweet spot (faster + better). Both re-tune per task; see the
  evaluation note below.
- `foreach=True` batches the step (bucketed by shape, with a batched `bmm`
  Newton-Schulz) — the decisive win for LoRA/LoKr (hundreds of tiny 2-D weights).
  The batched 2-D path matches the per-parameter path within bf16 NS tolerance
  (both unbiased); 1-D buckets and all fp32 ops are bit-exact.

## Performance: `torch.compile` (`compile=True`)

`AdaMuon(..., compile=True)` wraps the whole step body in `torch.compile`
(`fullgraph=False`), fusing the step's elementwise chain. **The speedup is
workload-dependent — benchmark it for yours.** Because AdaMuon's step is heavy on
fusable elementwise math (Newton-Schulz + factored + cautious + scale), it helps
broadly. An adversarial `opt.step()` microbench (RTX 4080, eager vs compiled
ratio, <1 = faster):

| param set | ratio | |
|---|---|---|
| many small **distinct**-shaped tensors (defeat `foreach`) | **0.34×** | huge win |
| single huge / single tiny / two params | 0.56–0.74× | helps |
| compute-bound full fine-tune (large weights) | ~0.98× | ~neutral |
| already-`foreach`-batched pure-LoRA | ~1.03× | ~neutral/slight loss |

So it is a no-op (or tiny loss) where eager is already efficient (foreach-batched
LoRA, compute-bound full-FT) or where the model fwd/bwd dominates (real SDXL is
UNet-bound). One-time warmup; numerically equivalent to eager (bit-exact per step;
stochastic rounding unbiased; no crashes across dtypes/shapes — verified). Not
recommended on CPU (inconsistent).

Note: compiling *only* the Newton-Schulz does **not** help on LoRA-rank matrices
(too small — wrapper overhead exceeds the fusion gain); the win is the whole-step
fusion. This flag is **AdaMuon-only by design**: [`Adakaon`](adakaon.md) has
little fusable elementwise math (no orthogonalization), so a whole-step compile was
~neutral there and not worth the API surface — Adakaon stays lean.

## Checkpointing

`load_state_dict` is overridden (shared `load_state_dict_preserving_dtypes`
helper) so a quantized first moment is not silently upcast to fp32 on resume —
preserving both the memory and bit-exact resume.

## Evaluation (v1, self-contained pixel-DDPM proxy)

Paired-seed A/B vs `Adakaon` (and `AdamW8bit` / `Lion8bit` / fp32 AdamW) on a
small pixel-space DDPM (conv UNet, C=128, 3 seeds, identical init/data/noise per
seed, LR swept per arm, held-out val MSE). Not real SDXL/Flux — a first signal.

- **Defaults matter.** With the *original* defaults (`ns_steps=5`, `cautious=False`)
  AdaMuon **lost** to Adakaon (0.0710 vs 0.0697). Two changes flipped it:
  `cautious=True` (helps ~2%, all seeds) and `ns_steps=2` (5 over-orthogonalizes —
  slower *and* worse). Both are now the defaults.
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

- Schedule-free / Prodigy parameter-free LR (kill scheduler dependence for fine
  detail) — analogous to the `Adakaon → Autofusion` relationship.

## See also

- [muon.md](muon.md) — the simpler heavy-ball Muon hybrid this builds on.
- [adakaon.md](adakaon.md), [kprodigy.md](kprodigy.md),
  [autofusion.md](autofusion.md), [foreach-batching.md](foreach-batching.md),
  [momentum.md](momentum.md).
```
