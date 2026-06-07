# Cheap momentum in Adakaon

> Design notes + the experiments behind why **int8 is the recommended cheap
> momentum**, and why every cheaper/simpler idea we tried was rejected. Read this
> before re-implementing sub-int8 momentum — we very likely already tested it.

## TL;DR

Adakaon factors the **second** moment for ~free (Adafactor row+col). The
**first** moment (momentum) is the only real state cost. The dial, cheapest → safest:

| `momentum_dtype` | state | quality | use it when |
|---|---|---|---|
| *(none)* `betas=(0.0, ...)` | 0 B/param | trains, but **~10–12% worse val loss than having momentum** | absolute minimum VRAM |
| **`"int8"`** ✅ | **1 B/param** | **near-lossless** (cos 0.9999 vs fp32, matches fp32 convergence) | **default cheap momentum — recommended** |
| `"4bit"` (supported, lighter) | 0.53 B/param | ties int8 in convergence (validated up to 27M params / 64² / 10k steps) | you must go below 1 B/param (huge model / 1024² / batch>1) |
| `"bfloat16"` | 2 B/param | full-precision momentum | memory is plentiful |
| `"float32"` | 4 B/param | reference | — |

**Recommendation:** `momentum_dtype="int8"`. Near-lossless, foreach-batched
(fast), and it lets a full fine-tune of an SDXL-class UNet (2.57 B params) fit
*with* momentum in ~13 GB on a 16 GB card — where `"bfloat16"` momentum (15.4 GB)
OOMs. Below int8 you trade away nothing measurable on quality *if* you use 4-bit
(see below), but you should know exactly what was tried and why.

## Why momentum is the expensive part

An optimizer does two separable jobs: **(1)** normalize the per-coordinate step
scale, and **(2)** smooth the descent direction over time. Adakaon gets (1) from
the factored second moment (non-negative + low-rank → compresses to ~0 state) and
(2) from the first-moment EMA. The second moment compresses to almost nothing
because it has exploitable structure; the **first moment does not** — it is signed
and full-rank, so it carries genuine ~1-byte-per-coordinate information that no
clever factorization removes. That asymmetry is the whole story below.

## What we shipped: int8 momentum

`momentum_dtype="int8"` quantizes the first moment to int8 with a **per-row
absmax scale** (`scale = absmax/127`), foreach-batched in both the factored
`[N,R,C]` and non-factored `[N,L]` paths. It is *not* the same as a "bitsandbytes
8-bit Adam": 8-bit Adam quantizes **both** moments (2 B/param), while Adakaon
factors the second moment (~0) and only quantizes the first (1 B/param total) —
half the state at similar quality.

Measured (real SDXL UNet gradients, error-accumulating state): momentum cosine vs
fp32 = **0.9999**, rel-L2 0.018, and a convergence run matched fp32 final loss
exactly. Foreach batching: LoRA step 98 → 12 ms (≈8×), SDXL full-FT 534 → 458 ms.

## What we tried and REJECTED (so you don't retread)

All three sub-int8 ideas were implemented, tested on **real gradients**, and
rejected with data. Branches are kept unmerged for reference.

### SMMF — rank-1 factored sign-magnitude momentum (~0.14 B/param)
Adafactor-style rank-1 factorization of `|M|` + a 1-bit sign matrix. **Rejected.**
Real momentum `|M|` is **not** low-rank (top-SVD energy ≈ 0.66 on a 640² attention
weight), so one rank-1 factor can't represent it: cos(delta) vs fp32 = **0.75**,
convergence ~5× worse, and 1.5–3.8× slower than int8. The published iterative-NMF
warm-start would cap at ~0.81 — still far below int8.

### SR-4bit — stochastically-rounded 4-bit momentum (0.53 B/param)
Hypothesis: round-to-nearest 4-bit has a systematic bias that accumulates over a
long horizon; stochastic rounding (unbiased) would fix it. **Rejected as a clean
negative.** SR's unbiasedness is real (per-element bias 3.5e-2 → 3.2e-4) but buys
**zero** convergence benefit — SR-4bit ties plain RTN-4bit at normal scale
(paired t = −0.64) *and* under a long-horizon high-LR stress regime (t = −0.39).
RTN 4-bit does not degrade at this scale, and SR costs +21% on full-FT (a
`rand_like` over every momentum element each step). RTN-4bit is enough.

### FP8 (E4M3) — native low-bit float, "conversion = one cast" (1.0 B/param)
The most elegant idea: a native float needs **no scale tensor and no packing**, so
the per-step "conversion" collapses to a single cast. **Rejected — and the reason
is the most instructive result here.** Real Adakaon momentum is *tiny*
(absmax ≈ 1e-4, mean|m| ≈ 1.6e-5), and E4M3's smallest representable value is
≈ 2.4e-4. So **100% of momentum coordinates fall below the fp8 floor**: no-scale
rounding freezes the whole buffer to 0 (convergence *worse* than no momentum),
and stochastic-cast injects oversized floor-level noise (diverges). Adding a
per-row absmax scale rescues it (cos 0.998) — but that is exactly int8's
machinery, now with worse precision (3 mantissa bits). FP8 is genuinely simpler
and ~20% faster, but non-functional without the scale.

## The deep lesson: you cannot go simpler *or* smaller for free

The FP8 failure makes the principle concrete: **int8's per-row scale is not
incidental overhead — it is essential.** Momentum magnitudes live ~100× below the
floor of any fixed 1-byte float, so you *must* rescale per row to represent them.
You cannot skip the "conversion"; the conversion is what makes a tiny, full-rank
signal representable in a byte.

And on a GPU it doesn't even cost much: the optimizer step is **bandwidth-bound**,
the dequant/requant happens **on-chip in registers** (it never touches global
memory), and it measures as only ~5–13% over fp32-momentum traffic. The real lever
is *bytes moved*, not the arithmetic — which is why compression helps and why
"avoid the conversion" optimizes a cost that isn't the bottleneck.

So: where the math had structure (the second moment), Adakaon already takes the
order-of-magnitude win (factoring). The first moment has no such structure — it
needs ~8 bits + a scale, full stop.

## Methodology (reuse this to evaluate your own variant)

1. **Judge by convergence, not per-step fidelity.** Per-step cosine and a
   fixed-LR toy MLP both *wrongly* flagged 4-bit as lossy; a proper
   per-arm-best-LR, paired-seed convergence run on real data showed it ties int8.
   Momentum changes the effective step size, so a shared LR confounds everything —
   sweep LR per arm.
2. **Use real gradients for fidelity.** Low-bit / low-rank fidelity depends on the
   gradient's structure; iid-Gaussian gradients unfairly punish factoring.
3. **Maintain each scheme's own error-accumulating compressed state** and compare
   the **applied weight-delta** (not just the stored momentum) to an fp32 reference.

## Pareto summary (real SDXL grads + mini-DDPM convergence)

| scheme | B/param | cos(delta) | convergence | speed vs int8 | verdict |
|---|---|---|---|---|---|
| bf16 momentum | 2.0 | ~1.0 | ref | — | OOMs SDXL full FT @16 GB |
| **int8** | 1.0 | 0.9999 | = fp32 | 1.0× | **recommended** |
| 4bit (RTN) | 0.53 | 0.97 | ties int8 (validated at scale) | ~1.1× slower | shipped; lighter option |
| SR-4bit | 0.53 | 0.97 | ties RTN-4bit | 1.2× slower | rejected (no gain) |
| FP8 E4M3 | 1.0 | broken | worse than no-mom | 0.8× | rejected (below float floor) |
| SMMF factored | 0.14 | 0.75 | ~5× worse | 1.5–3.8× slower | rejected (not rank-1) |

## Validation at scale

All the convergence numbers above were first measured on a *mini* pixel-space DDPM
(1.5 M-param UNet, 32×32, ≤6 k steps). The recurring worry was scale: does the
`int8 ≈ 4bit ≈ bf16` tie survive a bigger model and a longer horizon, or does
low-bit momentum quietly accumulate error?

We re-ran the head-to-head **18× larger (27 M-param UNet), at 4× resolution
(64×64), over a 6.6× longer horizon (10 k steps)**, 3 paired seeds, per-arm best LR
(which shifted to 1e-3 for every arm at this size). Held-out val loss:

| arm | state | val loss (mean ± std, 3 seeds) | paired vs bf16 |
|---|---|---|---|
| no-momentum | 0 | 0.018506 ± 0.00130 | +0.00193 (t = +21) — far worse |
| **4bit (RTN)** | 0.53 B/p | 0.016628 ± 0.00118 | +0.000049 (t = +0.86) — **tie** |
| **int8** | 1.0 B/p | 0.016631 ± 0.00128 | +0.000051 (t = +0.64) — **tie** |
| bf16 (ref) | 2.0 B/p | 0.016579 ± 0.00115 | — |

**The tie holds — if anything tighter than at small scale.** int8 and 4bit are both
statistically indistinguishable from full-precision (bf16) momentum
(`|t| < 1`), 4bit vs int8 is `t = −0.03` (essentially zero), and all three beat
no-momentum by ~10 % with decisive paired t-stats (−17 … −44). No sign of
accumulated quantization bias over the longer horizon or sensitivity to the larger
model.

> Honest caveat: this is still a pixel-space DDPM on a handful of CC0 photos, not a
> production latent SDXL/DiT (~860 M params, 128² latents, 100 k+ steps). The result
> strengthens confidence substantially but does not fully close the gap to that
> scale. Nothing observed across 32×32→64×64, 1.5 M→27 M params, and 1.5 k→10 k
> steps hints at low-bit momentum breaking down.
