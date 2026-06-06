# Research notes — finding the next optimizer for diffusion fine-tuning

Forward-looking research that directs koptim's optimizer roadmap. Distinct from the *results*
docs (what we measured, in `benchmarks/`): these are *what to build next and why*, synthesized
from fan-out, adversarially-verified web research.

## The through-line

1. **Corrected objective** ([optimizer-research-charter.md](optimizer-research-charter.md)) —
   AdaMuon was built on "reach a low loss fast"; we proved that is the **wrong objective** for
   small-data diffusion (the denoising-loss minimum *is* memorization). Rank by the **train–val
   gap + FID**, never by training loss. Memory is *not* a selection criterion — the int8/4bit
   momentum codec makes any 1–2-buffer optimizer Adafactor-class.

2. **Techniques vs base optimizer**
   ([generalization-techniques-findings.md](generalization-techniques-findings.md)) — 1st deep
   research. The diffusion-MEASURED generalization wins are optimizer-level *techniques*
   (flat-minima / SAM, weight-averaging / EMA), but all that evidence is from-scratch low-res
   pixel models (FID), not bf16 LoRA — so it's extrapolated. Our own proxy then showed the
   **base optimizer also matters** for the gap (Liofusion), which the techniques alone don't
   capture on synthetic data.

3. **Candidate catalog**
   ([optimizer-candidates-catalog.md](optimizer-candidates-catalog.md)) — 2nd deep research.
   A long-tail catalog mined from `pytorch_optimizer` (kozistr, 106→140+), timm, and SimpleTuner
   SDNQ, grouped by claim (speed / generalization / convergence-speed) with a portability lens
   (quantizable + foreach + cautious). Includes the generalization bucket (PNM, PAdam, DualAdam,
   SWATS, SAM-family) and the shortlist to port-and-test (ADOPT, AdEMAMix, MARS, MADGRAD, CAME).

## The two caveats that govern all of it

- **Speedup hype is mostly weak-baseline artifacts** (Stanford 2509.02046): fair tuning shrinks
  claimed 2× to ≤1.4×, and ranking flips with data-to-model ratio. Falsify against a *well-tuned*
  AdamW.
- **Zero diffusion-measured evidence** in either catalog — everything is LLM/vision. These are
  hypotheses to test on the registered proxy (train–val gap) + a real LoRA (FID/KID), not winners.

## Status of what's already built from this

- **Liofusion** (Lion sign-momentum on the koptim backend) — shipped (`docs/liofusion.md`); the
  proof that a regularizing base optimizer wins on the gap.
- **AdafusionEx** (Adafusion + EMA/MSAM techniques) — parked on `feat/adafusionex`; the techniques
  are not provable on the synthetic proxy (need a real run / FID).
- **Next round**: port ADOPT / AdEMAMix / a generalization-bucket method (PNM/PAdam), test by
  gap + FID. See the catalog's "recommended next round".
