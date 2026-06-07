# Recipes — and the benchmarks behind them

Every recipe below is a **complete, paste-ready call** (swap `model.parameters()` for your
params / param-groups). The **Evidence** line is the measurement that put it here, with a link
to the source. `lr` values are starting points at the scale each optimizer expects — always
re-tune `lr` on your data (the numbers below are from small proxies and a couple of real runs,
not your dataset).

> The north-star for **small-data fine-tuning** is the **train–val gap**, not the training
> loss: on small data the lowest-loss config memorizes and samples *worse*. Rank by held-out
> gap / sample quality. See
> [RESULTS_generalization_and_schedule.md](../benchmarks/adamuon/RESULTS_generalization_and_schedule.md).

---

## Small-data fine-tuning (LoRA / DreamBooth) — optimize for generalization

### Best generalization, and the only one happy on a constant LR (resumable)

```python
from kaon import AdaPNM

opt = AdaPNM(model.parameters(), lr=2e-3, betas=(0.8, 0.999), beta0=0.5)
```

PNM's negative-momentum term injects anti-correlated noise → a built-in implicit regularizer
(flat-minima seeking). `beta1` is the loss↔gap dial (`0.8` is the measured elbow).

**Evidence:** on the synthetic gap proxy, **~Lion/AdamW loss at 36–44 % lower train–val gap**;
and at **constant LR** (no schedule, resumable) it is the most gap-robust of the field — gap
`+0.0065` vs AdamW `+0.0101` / Lion `+0.0103` / Adakaon-nomom `+0.0114`, and its gap *improves*
without the schedule. Its sweet spot is a **high** constant LR. → [docs/adapnm.md](adapnm.md).

### Absolute minimum state (no second moment), regularizing

```python
from kaon import Lion

opt = Lion(model.parameters(), lr=2e-4, betas=(0.95, 0.98), momentum_dtype="4bit")
```

One momentum buffer, no second moment; the sign update's implicit regularization. `lr` is
Lion-scale (~AdamW/5); `betas` are a loss↔gap dial (`(0.95,0.98)` for loss, higher β2 for a
lower gap).

**Evidence:** lightest state in the family — **~0.5 B/param at 4bit**. On the proxy at the
larger C=128/2500 setting, tuned Lion reached **AdaMuon's loss at ~half its train–val gap** and
beat the no-momentum Adakaon baseline on both axes. → [docs/lion.md](lion.md).

### Minimum VRAM, regularizing (no momentum)

```python
from kaon import Adakaon

opt = Adakaon(model.parameters(), lr=1e-4, betas=(0.0, 0.999),
              bf16_method="stochastic_rounding")
```

`beta1=0` → no momentum buffer (near-zero optimizer state); factored second moment; bf16-correct
weights via stochastic rounding (no Kahan buffer, no CPU offload). This is the
no-momentum factored-Adam baseline a regularizing LoRA wants.

**Evidence:** on a real **2.1 B-param DiT** the no-momentum config uses **0.01 GB** of optimizer
state vs AdamW's **8.4 GB**; the real-run Cosmos/LoKr A/B was won by Adakaon-nomom on perceptual
quality despite a higher loss. → [docs/adakaon.md](adakaon.md),
[RESULTS_generalization_and_schedule.md](../benchmarks/adamuon/RESULTS_generalization_and_schedule.md).

---

## Maximum quality / fastest convergence-to-quality

### Beat AdamW on convergence at Adafactor memory

```python
from kaon import AdaMuon

opt = AdaMuon(model.parameters(), lr=1e-3, momentum_dtype="int8")
```

Newton-Schulz orthogonalized momentum + a factored quantized second moment. `ns_steps=2`,
`cautious=True`, `clip_threshold=1.0` are the tuned, **load-bearing** defaults. `lr` is
Muon-scale — start ~`1e-3` for diffusion (the API default `2e-2` is LLM-scale).

**Evidence:** full training from scratch (pixel-DDPM) — **0.0652 vs AdamW-fused 0.0854** at
**1.04 vs 8.0 B/param**, reaching AdamW's quality **~2× faster in wall-clock** (4× fewer steps).
Real SDXL LoRA — lower floor than AdamW (**0.0900 vs 0.0928**) at **1/3 the optimizer memory**.
→ [docs/adamuon.md](adamuon.md),
[RESULTS_vs_adamw_adafactor.md](../benchmarks/adamuon/RESULTS_vs_adamw_adafactor.md).

### Best convergence quality (when memory is available)

```python
from kaon import Muon

opt = Muon(model.parameters(), lr=2e-2, momentum_dtype="bfloat16")
```

Orthogonalized momentum with an AdamW fallback for 1-D/embedding params; half of AdamW's state.

**Evidence:** the convergence-quality ceiling of the family; AdaMuon is its near-Adafactor-memory
descendant (above). → [docs/muon.md](muon.md).

### AdamW-quality at a fraction of AdamW's memory

```python
from kaon import Adakaon

# bf16 momentum — 2 B/param
opt = Adakaon(model.parameters(), lr=1e-4, betas=(0.9, 0.999), momentum_dtype="bfloat16")

# int8 momentum — ~1 B/param, near-lossless
opt = Adakaon(model.parameters(), lr=1e-4, betas=(0.9, 0.999), momentum_dtype="int8")
```

**Evidence:** mini pixel-DDPM (held-out, 4 seeds) — Adakaon bf16 **0.0318 ± 0.0006 at 2 B/param**
beats **AdamW 0.0400 at 8 B/param** (~20 %, non-overlapping) at 1/4 the memory. int8 momentum is
**near-lossless** (cos 0.9999 vs fp32) at **1 B/param**. → [docs/adakaon.md](adakaon.md),
[docs/momentum.md](momentum.md).

---

## Parameter-free (no LR to tune)

### One LR for SDXL UNet + text-encoder, auto-discovered (Prodigy)

```python
from kaon import KProdigy

opt = KProdigy([
    {"params": unet.parameters(), "lr": 1.0},
    {"params": text_encoder.parameters(), "lr": 1.0},
])
```

Train at `lr=1.0`; D-adaptation finds each group's effective LR. Bit-for-bit identical to
reference Prodigy at its defaults, then adds the kaon memory toolkit.

**Evidence:** D trajectory + final weights **bit-identical** to reference Prodigy (verified on
CPU fp32 across every dtype/second-moment combo). → [docs/kprodigy.md](kprodigy.md).

### Parameter-free **and** minimum VRAM

```python
from kaon import KProdigy

opt = KProdigy(model.parameters(), lr=1.0, second_moment="factored",
               momentum_dtype="int8", slice_p=11)
```

`slice_p=11` shrinks the Prodigy D-state ~11× (~0.3 % D error); factored + quantized momentum
shrink the rest.

**Evidence:** **≈1.3 B/param** on the SDXL UNet shape distribution (vs reference Prodigy's four
fp32 buffers); D trajectory unchanged. → [docs/kprodigy.md](kprodigy.md).

### Parameter-free, then ~free after warmup (Mechanic, *not* Prodigy)

```python
from kaon import Autokaon

opt = Autokaon(model.parameters(), bf16_method="stochastic_rounding")
```

A Mechanic scalar tuner auto-discovers the LR on top of Adakaon's update; the default
`lr_freeze="auto"` then folds the discovered LR into the base, frees the tuner's `ref` buffer,
and runs as **byte-for-byte plain Adakaon** thereafter.

**Evidence:** validated on a real SDXL LoRA (the data-relative scale cap generalizes — val flat
across `scale_cap_rel` 3–12); with the default `adakaon_betas=(0.0,0.999)` the freeze handoff is
**bit-exact**. → [docs/autokaon.md](autokaon.md).

---

## Speed — many small tensors (LoRA / LoKr adapters)

`foreach=True` is the default and the decisive win when you step **many tiny tensors** at once
(it stacks them into batched kernels instead of one Python-loop launch per tensor). It is on for
all the recipes above; nothing to configure.

**Evidence:** real SDXL UNet + PEFT LoRA r=8 (1434 trainable tensors) — the optimizer step drops
from **318 ms → 15 ms** (1.45× of fused AdamW; was 28×). Full fine-tunes see a smaller ~1.3×
(bandwidth-bound on large weights). Bit-exact vs the per-parameter path.
→ [docs/foreach-batching.md](foreach-batching.md).

> Note: `momentum_dtype="int8"` and `bf16_method="kahan"` are *not* foreach-covered and fall
> back to the per-parameter loop. For the many-tiny-tensor adapter case, prefer no-momentum or
> `bfloat16`/`4bit` momentum to keep the foreach speedup.

---

## Reproduce

**The cross-optimizer control battery** ranks every optimizer across all these dimensions and
regenerates its tables incrementally — see
[`benchmarks/control/RANKINGS.md`](../benchmarks/control/RANKINGS.md) (and
[`battery.py`](../benchmarks/control/battery.py) to run it / add a contender).

The shared, optimizer-agnostic proxy lives under
[`benchmarks/proxy/`](../benchmarks/proxy/):

- `dataset.py` — the registered deterministic synthetic dataset (train=32 / test=96) that all the
  train–val-gap numbers use.
- `harness.py` — the pixel-DDPM U-Net + loss model/loss core (plus the LoRA-like adapter-bag
  speed proxy).

Deeper, optimizer-specific deep-dives stay under [`benchmarks/adamuon/`](../benchmarks/adamuon/):

- `pixel_ddpm_ab.py` — synthetic full-training A/B harness (pixel-DDPM U-Net).
- `sdxl_lora_ab.py` — real SDXL LoRA A/B harness (`adamw_fused` / `adafactor` arms included).
- `RESULTS_vs_adamw_adafactor.md`, `RESULTS_generalization_and_schedule.md` — the full tables.

KProdigy's D-trajectory parity / memory harness is
[`benchmarks/bench_kprodigy_d.py`](../benchmarks/bench_kprodigy_d.py).

**All numbers here are from small synthetic proxies plus a couple of real-model runs** (one SDXL
LoRA). They rank objective overfitting and convergence, not perceptual fidelity — confirm on a
real LoRA with FID/KID + a live `val/gap` metric before treating any ranking as settled.
