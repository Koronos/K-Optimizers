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

## TL;DR — recommended recipe (proxy scale; user maps LR to real scale)

**Optimizer:** for small-data LoRA where overfitting is the enemy, **`Adakaon` with no
momentum** is the fit — it consistently held the lowest train–val gap *and* won a real
visual A/B (even from its "worst" config). `AdaMuon` is **not** a good fit *here*: its
strength (fast convergence / low loss) is a liability when the risk is memorization, and it
never out-generalized Adakaon-nomom (§3, §5). AdaMuon shines in the *opposite* regime —
underfitting / abundant data / when you want fast convergence — not on a small LoRA.

```python
# Adakaon, no momentum — the small-LoRA recommendation
Adakaon(params, lr=<real-scale>, betas=(0.0, 0.999),
          cautious=False, momentum_dtype="bfloat16")
```
(If you do use AdaMuon — only when underfitting — keep its momentum: `betas=(0.95,0.999),
cautious=True, ns_steps=2`. Its momentum is load-bearing; never `beta1=0` (§3). Regularize
it via the mixed/progressive resolution schedule, not by disabling momentum.)

**Schedule:** **REX `rex_d=0.9`**, no warmup, `lr_min=0` (decay to ~0). Cosine is the
more-regularizing alternative (flatter gap, slightly higher loss).

**Resolution — a progressive-floor curriculum (raise the minimum resolution over training,
ending on the detail target), final 20% at large-only.** Two recipes by priority (§6):

| priority | resolution schedule (frac of training) | Adakaon-nomom test/gap |
|---|---|---|
| **loss** | `512+1024 [40%] → 768+1024 [40%] → 1024 [20%]` | **0.0878 / +0.0040** |
| **gap** | `512+768+1024 [40%] → 768+1024 [40%] → 1024 [20%]` | 0.0908 / **+0.0038** |

Both **Pareto-beat a flat mix**. The loss recipe (start at the 2× floor 512+1024) is the
all-round pick — for Adakaon it is *also* near gap-optimal. The gap recipe (start with all
three resolutions = max scale diversity) shaves the gap further at a small loss cost; for
**AdaMuon** the gap pick is `512+768+1024 [60%] → 768+1024 [20%] → 1024 [20%]` (gap +0.0075).
Map 512/768/1024 to your real bucket sizes; the **80% earlier vs 20% final-detail split** and
the **progressive narrowing** are what matter.

**LR at this proxy scale:** **≈1.2e-3** (Adakaon-nomom tolerated 6e-4–1.2e-3; AdaMuon
1.2e-3–2.4e-3). **Do not copy this number to a real run** — it is proxy-scale. Transferable
relation: **AdaMuon's optimum ≈ AdamW's ÷5** (measured), Adakaon ~AdamW scale; the optimal
LR is otherwise ~resolution-invariant (RMS-normalized update).

---

## 1. The core finding: train loss misleads; the train–val gap predicts quality

Real-run evidence (user's Cosmos/LoKr): **Adakaon in its *worst* config (no momentum) +
disordered buckets still beat AdaMuon (good config) in the *visual* test**, despite AdaMuon
winning on loss. On the proxy, the **train–val gap** (not absolute loss) ranks these the way
the eyes did:

| config | test (MSE) | **gap** | |
|---|---|---|---|
| AdaMuon mom staged | 0.0800 (lowest loss) | **+0.0198** (most overfit) | "wins loss, loses visual" |
| Adakaon NOmom mixed | 0.0921 (highest loss) | **+0.0110** (least overfit) | "loses loss, wins visual" |

The config that minimizes loss does so by **overfitting hardest**. Web research confirms the
train–val gap is the recognized overfitting signal for diffusion fine-tuning, that it grows
with longer training / smaller data / bigger models, and that it concentrates at
**intermediate (mid-SNR) timesteps** — so weight the gap toward mid-`t`.

## 2. The regularization axis — knobs that trade loss for generalization

Every knob that lowers loss raises the overfitting gap, and vice-versa. Measured directions
(↓gap = more regularization):

| knob | toward low loss | toward low gap (regularize) |
|---|---|---|
| momentum (`beta1`) | on (0.9 / 0.95) | **0.0** *(Adakaon only — see §3)* |
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
| Adakaon mom=off | 0.0829 | **+0.0029** (regularization champion) |

AdaMuon **is** orthogonalized *momentum*; with `beta1=0` it orthogonalizes the raw noisy
per-batch gradient → much worse loss, and its gap is *no better* than AdaMuon-mom + mixed
(0.0077 vs 0.0073). **To regularize AdaMuon, use mixed resolution — never `beta1=0`.**
Adakaon's momentum, by contrast, *is* dispensable (it's factored-Adam; `beta1=0`
regularizes and stays strong — the gap champion).

## 4. Resolution curriculum, refined

- **medium→large beats small→large** (0.0630 vs 0.0666): the coarse tier should sit ~½ the
  target (2× gap), not ¼ (4×) — too-far transfers poorly. More tiers don't help: 3-tier ties
  or loses to 2-tier `med→large` at equal high-res budget. *(Closer-still 768→1024 ≈1.33×
  warmup gives even less curriculum benefit — see the phased result §6 for the better lever.)*
- **staged beats mixed on loss** (ordered, ending on detail = 0.069 vs mixed 0.081 at 3-res),
  but **mixed regularizes** (lower gap). Reverse (large→small) is catastrophic (erases detail).
- **large-only vs curriculum depends on the optimizer (substitutes!):** for the already-
  regularized Adakaon-nomom, **large-only gives the *best* loss** with the gap only
  marginally higher — the curriculum barely earns its keep. For the overfitter AdaMuon-mom,
  large-only is *worst* (biggest gap); the curriculum is what tames it.

### 4.1 Who *depends* on the curriculum noise — AdaPNM vs Adakaon vs Lion

The resolution curriculum is **data-side noise**; an optimizer with its own **gradient-side**
regularization (AdaPNM's positive-negative momentum) should need it less. Single-resolution
(all-1024) vs the `512+1024 → 768+1024 → 1024` (40/40/20) curriculum, everything else fixed
(REX d=0.9, C=128 / 2500, eval@64, 2 seeds):

| optimizer | single (test/gap) | prog (test/gap) | Δgap | Δloss |
|---|---|---|---|---|
| **AdaPNM** (β1=0.8, β0=0.5) | 0.0839 / **+0.0146** | 0.0802 / **+0.0070** | −0.0076 | −0.0037 |
| Adakaon-nomom | 0.1012 / +0.0501 | 0.0836 / +0.0196 | −0.0305 | −0.0176 |
| Lion (0.95,0.98) | 0.1053 / +0.0549 | 0.0769 / +0.0125 | −0.0424 | −0.0284 |

- **Everyone benefits** (all Δ negative — the curriculum is free regularization, never harmful;
  use it regardless of optimizer).
- **Dependence (Δgap magnitude):** Lion (−0.0424) > Adakaon-nomom (−0.0305) ≫ **AdaPNM (−0.0076)**.
  Adakaon/Lion *overfit catastrophically* single-resolution (gap ~0.05) — the curriculum is their
  only regularizer (load-bearing). AdaPNM barely needs it: its single-res gap (0.0146) is already
  ~3.5× lower than the others' best.
- **Absolute gap (the objective):** AdaPNM wins in **both** regimes (single 0.0146 vs 0.05+;
  prog 0.0070 vs 0.0125 / 0.0196). The data-noise (curriculum) and gradient-noise (PNM)
  regularizers **stack** — different axes, additive.
- **Takeaway:** Adakaon/Lion are the kings of *depending on* the curriculum noise; **AdaPNM is the
  king of generalizing with or without it**, and stacking the curriculum still helps it for free.

## 5. Master sweep — Pareto front (loss vs gap)

108 configs (optimizer × momentum × cautious × scheduler{cosine,rex0.9,rex0.5} ×
order{staged,mixed} × lr), 2 seeds, on the cached dataset. The Pareto front of
(test-loss, gap) is a clean tradeoff curve:

| test | gap | config |
|---|---|---|
| 0.0724 | +0.0123 | AdaMuon mom caut rex0.9 **staged** lr2.4e-3 — loss-optimal (overfits) |
| 0.0770 | +0.0081 | AdaMuon mom caut rex0.9 **mixed** lr2.4e-3 — mixed cuts AdaMuon's gap ~34% |
| 0.0832 | +0.0067 | Adakaon mom rex0.5 mixed lr2.4e-3 |
| 0.0892 | +0.0059 | Adakaon **NOmom** rex0.9 mixed lr1.2e-3 |
| 0.1012 | **+0.0029** | Adakaon NOmom cosine mixed lr2.4e-3 — generalization-optimal |

Loss-optimal corner = AdaMuon+momentum+cautious+staged; generalization corner =
Adakaon+no-momentum+mixed+cosine. **`mixed` is the key regularizer for AdaMuon.**

## 6. Phased `mix → big`: recover detail at ~flat gap (the operating point)

The best single lever for "better loss while keeping generality": a **long mixed phase then
a short large-only phase**. Sweeping the big-phase fraction (REX/cosine, lr1.2e-3):

| big-phase % | Adakaon-nomom REX | AdaMuon-mom REX |
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

This solves *both* problems at once: the long mix phase fights overfitting, the short final
big phase (with the REX decay landing on it) restores fine detail.

### 6.1 Progressive floor beats a flat mix (the final schedule)

Going further: instead of a *fixed* mix composition, **raise the minimum resolution over
training** (widest scale diversity early, narrow toward the target, end large-only). All
schedules end with 20% large-only@1024; REX `rex_d=0.9`, lr1.2e-3, 2 seeds. (512=32², 768=48²,
1024=64².)

| schedule | Adakaon-nomom test/gap | AdaMuon-mom test/gap |
|---|---|---|
| flat `512+1024 → 1024` (80/20) | 0.0885 / +0.0050 | 0.0767 / +0.0088 |
| flat `512+768+1024 → 1024` (80/20) | 0.0905 / +0.0044 | 0.0775 / +0.0077 |
| **prog `512+1024 → 768+1024 → 1024` (40/40/20)** | **0.0878 / +0.0040** | **0.0753 / +0.0085** |
| **prog `512+768+1024 → 768+1024 → 1024` (40/40/20)** | 0.0908 / **+0.0038** | 0.0749 / +0.0089 |
| prog `512+768+1024 → 768+1024 → 1024` (60/20/20) | 0.0901 / +0.0041 | 0.0764 / **+0.0075** |

- **The progressive floor Pareto-beats every flat mix** for both optimizers — it is a smooth
  general→detail curriculum (regularize when plastic, sharpen late).
- **Loss-priority:** start at the 2× floor `512+1024 → 768+1024 → 1024 (40/40/20)` — best loss
  (0.0878), and for Adakaon *also* near gap-optimal. The all-round pick.
- **Gap-priority:** start with **all three** resolutions (max scale diversity = more
  augmentation = lower gap). Adakaon-nomom `3→768→1024 (40/40/20)` → gap +0.0038 (lowest in
  the whole campaign); AdaMuon `3→768→1024 (60/20/20)` → gap +0.0075.
- **The valuable coarse tier is the 2× one (512), not the 1.33× one (768):** dropping 768
  (`512+1024`) keeps the best loss; dropping 512 is strictly worse. 768 only earns its place
  as an intermediate *step* in the progressive narrowing.

(Differences are small, ~2-seed noise; the robust directions are: progressive ≥ flat; "start
with all 3" lowers the gap; "start at 512+1024" lowers the loss; always end on a short
large-only phase.)

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
