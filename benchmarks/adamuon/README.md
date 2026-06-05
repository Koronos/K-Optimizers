# AdaMuon vs Adafusion — reproducible benchmarks

These scripts back the claim that **AdaMuon** (orthogonalized momentum + factored
quantized variance) is competitive with / better than **Adafusion** on diffusion
training, at equal-or-less optimizer memory. They are intentionally small and
self-documenting so others can **reproduce them and point out flaws** in the
method or conclusions.

Two harnesses:

| script | data | what it measures | needs |
|---|---|---|---|
| `pixel_ddpm_ab.py` | **synthetic** (generated in-process) | convergence (held-out val MSE) of a from-scratch conv U-Net | `torch`, `koptim` |
| `sdxl_lora_ab.py` | your images + an SDXL checkpoint | LoRA-finetune convergence (deterministic objective probe) | `torch`, `diffusers`, `peft`, `koptim` |

No paths are hard-coded; the SDXL harness reads everything from env vars.

## Method (and why)

* **Best-config vs best-config.** Each optimizer is swept over LR and reported at
  its own best. Don't compare at a shared LR — these are normalized-update
  optimizers on different scales.
* **Paired seeds.** For a given seed, model/LoRA init, data, and per-step
  noise/timestep draws are identical across optimizers, so the delta is the
  optimizer alone.
* **Metric depends on the regime:**
  * *pixel_ddpm* trains **from scratch**, so held-out val MSE is a clean
    convergence signal.
  * *sdxl_lora* finetunes a **pretrained** UNet, where (a) held-out eps-MSE
    measures *overfitting*, not progress, and (b) the per-step random-timestep
    train loss is too noisy. So it reports a **deterministic objective probe**:
    eps-MSE on the train set at a *fixed* set of (timestep, noise) draws — a
    low-variance estimate of the objective being minimized.
* **Speed.** On SDXL the step is **UNet-bound**; the optimizer is <1% of wall
  time (check the `ms/step` column — AdaMuon and Adafusion match). So
  convergence-per-step == convergence-per-second there. On the tiny pixel U-Net
  the optimizer is a larger fraction, so AdaMuon's Newton-Schulz shows a per-step
  cost — but it still wins time-to-quality via faster convergence.
* **Memory.** Optimizer-state bytes/param is measured directly (`state ... B/p`).

## Run

```bash
pip install -e .            # installs koptim

# Synthetic, self-contained (no downloads). Reproduces the headline table:
python benchmarks/adamuon/pixel_ddpm_ab.py --preset headline
python benchmarks/adamuon/pixel_ddpm_ab.py --preset memory_ladder
python benchmarks/adamuon/pixel_ddpm_ab.py --optims "adamuon:1e-3:cos,adafusion:1e-3:cos"

# Real SDXL LoRA (needs a single-file SDXL checkpoint + a folder of images):
export ADAMUON_SDXL_CKPT=/path/to/sdxl.safetensors
export ADAMUON_IMG_DIR=/path/to/images
export ADAMUON_CACHE=./adamuon_sdxl_cache
python benchmarks/adamuon/sdxl_lora_ab.py precompute --res 512
python benchmarks/adamuon/sdxl_lora_ab.py ab --cosine --steps 500 --seeds 2 --rank 16
```

If single-file loading errors on transformers v5 (`CLIPTextModel has no attribute
text_model`), point `ADAMUON_SF_PATCH=module.path:function` at a patch that fixes
`diffusers.loaders.single_file_utils`; the VAE/UNet-only path here usually does
not need it.

## Results we obtained

Hardware: single RTX 4080. AdaMuon config: `ns_steps=2, cautious=True,
betas=(0.95,0.999), clip_threshold=1.0`. Adafusion: `betas=(0.9,0.999),
cautious=True`. Both at their swept-best LR (1e-3 here).

**Synthetic pixel-DDPM** (U-Net C=128, 3 seeds, 800 steps, val MSE):

| optimizer | val | state B/param |
|---|---|---|
| AdaMuon + cosine | **0.0651** | 2.03 |
| AdaMuon (constant) | 0.0677 | 2.03 |
| AdaMuon int8 + cosine | ~0.065 | 1.04 |
| AdaMuon 4bit | 0.0694 | 0.56 |
| Adafusion + cosine | 0.0701 | 2.03 |
| Adafusion (constant) | 0.0717 | 2.03 |
| AdamW8bit / Lion8bit / AdamW | 0.076 / 0.077 / 0.088 | 2.06 / 1.03 / 8.0 |

* `ns_steps=2` beats the LLM-standard 5 (5 over-orthogonalizes — slower *and*
  worse). `cautious=True` and a cosine schedule each help. int8 momentum ties
  bf16; even 4bit (0.56 B/param) beats Adafusion-bf16. No-momentum (beta1=0,
  ~0.03 B/param) is a near-stateless extreme but clearly worse — momentum helps.

**Real SDXL LoRA** (Illustrious-XL, rank 16, +cosine, 2 seeds, 500 steps,
deterministic objective):

| optimizer | best objective | ms/step | state B/param |
|---|---|---|---|
| AdaMuon-bf16 + cos | **0.09003** | 466.6 | 2.25 |
| AdaMuon-int8 + cos | 0.09015 | 474.1 | **1.37** |
| AdaMuon-4bit + cos | 0.09098 | 473.7 | **0.78** |
| Adafusion + cos | 0.09057 | 466.7 | 2.25 |

AdaMuon reaches a lower objective (and `train<=0.0918` in 350 vs 400 steps);
ms/step is identical (UNet-bound). **AdaMuon-int8 matches bf16 quality at 61% of
Adafusion's optimizer memory** → Pareto-dominant (≥ convergence, tied speed, less
memory).

## Known limitations (please scrutinize)

* **SDXL margin is small** (~0.6% final objective vs ~7% on the synthetic task) —
  a pretrained model is saturated, so there is little headroom. The win is
  consistent across seeds and clear on the memory/convergence frontier, but it is
  not a blow-out.
* **Fixed text conditioning** in the SDXL harness (a constant embedding, to avoid
  the version-fragile single-file CLIP loader). It does not test
  text-conditioned learning; a production run should encode real prompts.
* **Small studies**: few images, low rank, 2–3 seeds, short horizons, one GPU.
  Treat as a directional signal, not a benchmark suite.
* **Synthetic data** in pixel_ddpm is not natural-image statistics.
* All "orthogonalized momentum beats AdamW" prior evidence is from LLMs; diffusion
  fine-detail superiority remains a hypothesis these scripts only begin to probe.
