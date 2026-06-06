# Generalization vs loss: choosing the optimizer + schedule for small-data diffusion LoRA

When fine-tuning on a small dataset (a LoRA), the **training loss is misleading**: a
stronger optimizer can drive train loss down while *overfitting/memorizing* and producing
**worse samples**. This doc records a campaign that (1) confirms that disconnect, (2) finds
which knobs trade loss for generalization, and (3) lands on a recommended optimizer +
schedule recipe. All on the self-contained synthetic pixel-DDPM proxy (conv U-Net C=64),
with a **fixed cached dataset** (`build_dataset.py`: 128 imgs, **train=32**, test=96 held
out, resolutions 64/48/32 = 1024/768/512-analog), eval @64², 2 seeds.

> **Hard caveat up front.** A synthetic MSE proxy can rank *objective overfitting* (the
> train–val gap) but **cannot measure perceptual sample quality**. Sample-based proxies
> (Vendi diversity, NN-to-train) were *unreliable* on synthetic images (§7). The gap is the
> usable proxy signal; **the real model is the arbiter** — use the live `val/gap` metric
> (shipped in renga-flow) on a real run to find the operating point.

## TL;DR — recommended config (proxy scale; user maps LR to real scale)

**Optimizer — pick by which problem dominates:**

| pick | optimizer | when |
|---|---|---|
| **A — max generality** | `Adafusion(betas=(0.0, 0.999), cautious=False, momentum_dtype="bfloat16")` — **no momentum** | overfitting is the enemy; small data; perceptual quality matters (won a real visual A/B even from its "worst" config) |
| **B — more detail** | `AdaMuon(betas=(0.95, 0.999), cautious=True, ns_steps=2, momentum_dtype="bfloat16")` — **keep momentum** | you need sharper detail and can spend a little generality |

**Schedule (both):** **REX `rex_d=0.9`**, no warmup, `lr_min=0` (decay to ~0). Cosine is
the more-regularizing alternative (flatter gap, slightly higher loss).

**Resolution (both):** **phased `~80% mix → ~20% big`** — a long mixed-resolution phase
(shuffled small+large, regularizes) then a short large-only phase (consolidates detail; the
REX decay lands here). REX optimum ~20% big; cosine ~30%.

**LR at this proxy scale:** **≈1.2e-3** for both (Adafusion-nomom tolerated 6e-4–1.2e-3;
AdaMuon liked 1.2e-3–2.4e-3). **Do not copy this number to a real run** — it is proxy-scale.
The transferable relation: **AdaMuon's optimum ≈ AdamW's ÷5** (measured), Adafusion similar
scale; the optimal LR is otherwise ~resolution-invariant (RMS-normalized update).

---

## 1. The core finding: train loss misleads; the train–val gap predicts quality

Real-run evidence (user's Cosmos/LoKr): **Adafusion in its *worst* config (no momentum) +
disordered buckets still beat AdaMuon (good config) in the *visual* test**, despite AdaMuon
winning on loss. On the proxy, the **train–val gap** (not absolute loss) ranks these the way
the eyes did:

| config | test (MSE) | **gap** | |
|---|---|---|---|
| AdaMuon mom staged | 0.0800 (lowest loss) | **+0.0198** (most overfit) | "wins loss, loses visual" |
| Adafusion NOmom mixed | 0.0921 (highest loss) | **+0.0110** (least overfit) | "loses loss, wins visual" |

The config that minimizes loss does so by **overfitting hardest**. Web research confirms the
train–val gap is the recognized overfitting signal for diffusion fine-tuning, that it grows
with longer training / smaller data / bigger models, and that it concentrates at
**intermediate (mid-SNR) timesteps** — so weight the gap toward mid-`t`.

## 2. The regularization axis — knobs that trade loss for generalization

Every knob that lowers loss raises the overfitting gap, and vice-versa. Measured directions
(↓gap = more regularization):

| knob | toward low loss | toward low gap (regularize) |
|---|---|---|
| momentum (`beta1`) | on (0.9 / 0.95) | **0.0** *(Adafusion only — see §3)* |
| LR schedule | `rex_d=0.9` (holds LR high) | **cosine / low `rex_d`** (decays earlier) |
| resolution order | staged curriculum | **mixed / disordered** |
| `cautious` (AdaMuon w/ momentum) | `True` | `False` *(no-op without momentum)* |

- **`cautious`** only matters *with* momentum (it masks update–gradient sign disagreements;
  with `beta1=0` the update *is* the gradient → nothing to mask, exact no-op). With momentum
  it lowers loss but raises the gap.
- These knobs are **substitutes, not all additive** (§4): once the optimizer is regularized,
  piling on more regularization gives diminishing returns and costs loss.

## 3. AdaMuon: momentum is load-bearing — do NOT disable it

A tempting idea ("AdaMuon without momentum = its speed + no-momentum's regularization")
**does not work**:

| | best test | best gap |
|---|---|---|
| AdaMuon **mom=on** | **0.0724** | +0.0073 (mixed) |
| AdaMuon mom=off | 0.0884 (+22% worse loss) | +0.0077 |
| Adafusion mom=off | 0.0829 | **+0.0029** (regularization champion) |

AdaMuon **is** orthogonalized *momentum*; with `beta1=0` it orthogonalizes the raw noisy
per-batch gradient → much worse loss, and its gap is *no better* than AdaMuon-mom + mixed
(0.0077 vs 0.0073). **To regularize AdaMuon, use mixed resolution — never `beta1=0`.**
Adafusion's momentum, by contrast, *is* dispensable (it's factored-Adam; `beta1=0`
regularizes and stays strong — the gap champion).

## 4. Resolution curriculum, refined

- **medium→large beats small→large** (0.0630 vs 0.0666): the coarse tier should sit ~½ the
  target (2× gap), not ¼ (4×) — too-far transfers poorly. More tiers don't help: 3-tier ties
  or loses to 2-tier `med→large` at equal high-res budget. *(Closer-still 768→1024 ≈1.33×
  warmup gives even less curriculum benefit — see the phased result §6 for the better lever.)*
- **staged beats mixed on loss** (ordered, ending on detail = 0.069 vs mixed 0.081 at 3-res),
  but **mixed regularizes** (lower gap). Reverse (large→small) is catastrophic (erases detail).
- **large-only vs curriculum depends on the optimizer (substitutes!):** for the already-
  regularized Adafusion-nomom, **large-only gives the *best* loss** with the gap only
  marginally higher — the curriculum barely earns its keep. For the overfitter AdaMuon-mom,
  large-only is *worst* (biggest gap); the curriculum is what tames it.

## 5. Master sweep — Pareto front (loss vs gap)

108 configs (optimizer × momentum × cautious × scheduler{cosine,rex0.9,rex0.5} ×
order{staged,mixed} × lr), 2 seeds, on the cached dataset. The Pareto front of
(test-loss, gap) is a clean tradeoff curve:

| test | gap | config |
|---|---|---|
| 0.0724 | +0.0123 | AdaMuon mom caut rex0.9 **staged** lr2.4e-3 — loss-optimal (overfits) |
| 0.0770 | +0.0081 | AdaMuon mom caut rex0.9 **mixed** lr2.4e-3 — mixed cuts AdaMuon's gap ~34% |
| 0.0832 | +0.0067 | Adafusion mom rex0.5 mixed lr2.4e-3 |
| 0.0892 | +0.0059 | Adafusion **NOmom** rex0.9 mixed lr1.2e-3 |
| 0.1012 | **+0.0029** | Adafusion NOmom cosine mixed lr2.4e-3 — generalization-optimal |

Loss-optimal corner = AdaMuon+momentum+cautious+staged; generalization corner =
Adafusion+no-momentum+mixed+cosine. **`mixed` is the key regularizer for AdaMuon.**

## 6. Phased `mix → big`: recover detail at ~flat gap (the operating point)

The best single lever for "better loss while keeping generality": a **long mixed phase then
a short large-only phase**. Sweeping the big-phase fraction (REX/cosine, lr1.2e-3):

| big-phase % | Adafusion-nomom REX | AdaMuon-mom REX |
|---|---|---|
| 0% (pure mix) | 0.0985/+0.0046 | 0.0814/+0.0083 |
| **15%** | 0.0897/+0.0050 | 0.0762/+0.0082 |
| **20%** | 0.0889/+0.0050 | 0.0763/+0.0090 |
| 25% | 0.0883/+0.0053 | 0.0751/+0.0089 |
| 50% | 0.0845/+0.0064 | 0.0731/+0.0117 |
| 100% (pure big) | 0.0839/+0.0084 | 0.0777/**+0.0177** |

- **The first ~15–20% big phase recovers most of the detail/loss for almost no gap cost** —
  for AdaMuon the 15% point is nearly *free* (gap unchanged vs pure-mix, loss −0.005). Past
  ~25% the loss barely improves while the gap climbs. **Knee ≈ 20% (REX) / 30% (cosine).**
- **Cosine corroborates REX**: same monotone-loss / flat-gap shape, just more gradual
  (cosine's gap stays flatter longer → can push big% to ~30–35%; higher absolute loss).
- For AdaMuon, **pure-big (100%) is the worst** (biggest gap) — the mix phase is essential.

This solves *both* of the user's problems at once: the long mix phase fights overfitting, the
short final big phase (with the REX decay landing on it) restores fine detail.

## 7. Fast generalization metric (what to actually measure)

Per the research (cited in the team notes): the recognized, cheap signal is a **deterministic
held-out validation loss** — freeze the noise `eps` and timestep `t` per val item once, then
each eval recompute eps-MSE (forward-only, no sampling) → a smooth, comparable curve — and
track the **train–val gap**, ideally weighted toward mid-SNR timesteps. This is now live in
renga-flow (`val/loss`, `val/gap` in TensorBoard + UI).

**Negative result:** sample-based proxies (**Vendi** diversity, **NN-to-train** distance) were
*unreliable on the synthetic proxy* — they didn't track the overfitting axis and even
contradicted it (the loss-optimal AdaMuon had the *highest* NN-to-train = least memorization).
Synthetic sinusoid "images" lack perceptual structure, so pixel-space diversity/memorization
are not valid quality proxies; those need a real model (FID/KID/CLIP). On the proxy, **only
the mid-SNR gap is trustworthy.**

## Caveats

Synthetic pixel-DDPM, conv U-Net, 2 seeds, train=32 — directional. The proxy ranks
objective-overfitting (gap), not perceptual beauty. Confirm on a real run via the live
`val/gap`: the operating point is the checkpoint at the **val-loss valley** (minimum before the
gap turns up). The recommended recipe (§TL;DR) is the proxy's best (loss, gap) Pareto region;
the real run sets the exact LR and the parteaguas gap value.
