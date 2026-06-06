# Optimizer catalog (2nd deep-research) — long-tail candidates to port

Mined from kozistr **pytorch_optimizer** (106→140+ optimizers, the richest registry, enumerable
via `get_supported_optimizers()`), **timm.optim**, and **SimpleTuner SDNQ**; cross-checked vs
primary arXiv/NeurIPS/ICLR. Companion to `OPTIMIZER_RESEARCH_CHARTER.md`.

## TWO caveats that override every entry
1. **Speedup hype is mostly weak-baseline artifacts.** Stanford "Fantastic Pretraining
   Optimizers" (arXiv:2509.02046): fair tuning shrinks claimed 2× → **≤1.4× (≈1.1× at 1.2B)**;
   ranking is **scale/data-ratio dependent** (Muon wins low-ratio, SOAP/Kron high-ratio). →
   Falsify every candidate vs a **well-tuned AdamW**, ranked by **gap + FID**, never loss.
2. **ZERO diffusion evidence in the entire catalog** — all LLM/vision. Our prior (low-loss-fast
   = overfit on small-data diffusion) means LLM token-efficiency wins may NOT transfer to FID,
   and the **anti-forgetting** ones could *entrench memorization*. These are hypotheses to TEST.

## Shortlist — single-extra-buffer, quantizable, foreach-able, cautious-compatible (port these)

| candidate | claim | mechanism | extra state | flags |
|---|---|---|---|---|
| **ADOPT** (2411.02853, NeurIPS'24) | convergence | drop-in Adam fix, optimal O(1/√T) for ANY β2 | **none** (uses m,v) | cheapest port; cautious=True in kozistr |
| **AdEMAMix** (2409.03137, ICLR'25) | convergence+speed | two first-moment EMAs (fast β1 + slow β3≈0.9999) | **+1 buffer** (m2) | +95% token-eff; ⚠️ "slows forgetting" may ENTRENCH memorization |
| **MARS** (2411.10438, ICML'25) | convergence | variance reduction (STORM) on preconditioned update | +1 prev-grad buffer | port the **AdamW/Lion** instance, NOT the Shampoo one |
| **MADGRAD** (2101.11075, JMLR'22) | conv+generalization | momentumized dual-averaged grad; "WD often zero" | small | timm-shipped; needs LR sweep |
| **CAME / StableAdamW** | speed/stability | confidence-guided Adafactor / stable AdamW | factored / Adam-class | **already shipped QUANTIZED in SimpleTuner SDNQ** (proof of portability) |

## Generalization bucket (our gap+FID lens — explicit flat-minima / anti-overfit)

| candidate | mechanism | cost |
|---|---|---|
| **PNM** (2103.17182, ICML'21) | positive-negative momentum (two momenta) as implicit regularization | cheap, quantizable |
| **PAdam** (1901.09517, ICLR'19) | partial adaptivity (p∈[0,0.5]) → generalize like SGD, keep Adam speed | ~no extra state |
| **DualAdam/InvAdam** (2603.07122) | explore flat minima then switch to Adam; **uses diffusion theory to escape sharp minima** | two momenta |
| **SWATS** (1712.07628) | auto-switch Adam→SGD for generalization | ~no state |
| **SAM family** (SAM/ASAM/GSAM/WSAM/FriendlySAM) | sharpness-aware flat minima | **2× fwd/bwd** (real cost); all vision/LLM |

## Long-tail extras (catalog, lighter evidence)
Adan (2208.06677), **Conda** (2509.24218, claims 2-2.5× AdamW — very recent), **SNSM**
(subset-norm + subspace-momentum, 2411.07120), LaProp, AdaBelief, AdaHessian (2nd-order, extra
Hutchinson passes), NvNovoGrad, Tiger, Kate, SPAM/StableSPAM, Grokfast, Aida, AdamP, AdaGO.

## Lower priority — resist the 1-2-buffer quantization codec
**Kron/PSGD** (Kronecker preconditioner — "non-quantizable" concern was *refuted/uncertain*),
**SOAP**, Shampoo-instance of MARS, full preconditioners. Low-rank ones (GaLore/QGaLore/Fira/
APOLLO) are memory-savers — we already have that, not the goal.

## Recommended next round (port + test with our framework + gap+FID harness)
1. **ADOPT** — zero extra state, cleanest port, provable convergence; a clean "does convergence-
   speed help or just overfit faster?" test.
2. **AdEMAMix** — the highest-upside + highest-risk (anti-forgetting vs our overfitting concern);
   one buffer, cautious-compatible. The most interesting to settle on the gap.
3. **PNM** or **PAdam** — generalization-bucket, cheap, directly in our lens (flat minima without
   SAM's 2× cost).
4. (optional) **MADGRAD** — the "zero weight decay" dual-claim is distinctive.

All are ≤1-extra-buffer → drop into Liofusion/Adafusion's quantized + foreach + cautious backend.
Falsify each on the registered proxy (train-val gap) + a real LoRA (FID/KID), vs tuned AdamW —
NOT loss.

### Key sources
pytorch_optimizer (kozistr) · timm.optim · SimpleTuner SDNQ (Disty0/sdnq) · ADOPT 2411.02853 ·
AdEMAMix 2409.03137 · MARS 2411.10438 · MADGRAD 2101.11075 · PNM 2103.17182 · PAdam 1901.09517 ·
SWATS 1712.07628 · Cautious 2411.16085 · Stanford benchmark 2509.02046
