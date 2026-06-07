# Autokaon — parameter-free LR on Adakaon (Mechanic, *not* Prodigy)

<!-- Formerly named "Autofusion". -->

> A scalar learning-rate tuner wrapped around Adakaon's update, with a
> **freeze-to-free** handoff: after warmup it becomes byte-for-byte plain
> Adakaon at the discovered LR. Train at `lr=1.0`, never tune an LR, pay
> ~nothing once it freezes.

## How it works

`Autokaon` does **not** estimate a distance-to-solution like Prodigy. It uses
**Mechanic** (Cutkosky, Defazio & Mehta, NeurIPS 2023,
[arXiv:2306.00144](https://arxiv.org/abs/2306.00144)) — an online learning-rate
*tuner* that wraps an arbitrary base optimizer and learns a single **scalar**
multiplier `s` on the base update by coin-betting / reward maximisation. It is
update-agnostic *by construction*: it only ever sees the gradient and the base
optimizer's update vector `Delta`, never how that update was formed.

The base optimizer here is an internal `Adakaon` run at `lr=1`. Each step
(while adapting):

1. snapshot `p`, run `base.step()` to get the base update `u_t = p_after − p_before`;
2. reconstruct the accumulated displacement `Delta_t = (p − ref) / sum(s)` on the
   fly (no stored `Delta` buffer);
3. form the Mechanic gradient `h_t = <Delta_t, g_t + decay_t>` summed over params;
4. run the per-beta scalar tuner to update `s`;
5. set `p = ref + sum(s) · Delta_t`.

The discovered effective LR is `sum(s)`, read with `get_d()`.

### Why Mechanic and not Prodigy

A matched-effective-LR ablation found `KProdigy` (Prodigy's Adam-form
D-adaptation) converges ~2× worse than `Adakaon` on a mini pixel-DDPM, and the
gap isolates entirely to **first-moment placement relative to the √v
normalization**:

- KProdigy / Adam / Prodigy: `delta = ema(d·g) / √v` — momentum of the *raw*
  gradient, then normalize.
- Adakaon: `delta = ema(clip(g / √v))` — normalize + RMS-clip first, *then*
  momentum.

Adakaon's ordering is the better one, and Prodigy's D-estimator is derived for
the Adam form, so it doesn't transplant cleanly. Mechanic sidesteps the problem:
because it only sees `Delta` and `g`, it keeps **Adakaon's update verbatim** and
just auto-discovers its scale.

## Freeze-to-free (the headline)

Mechanic's scale converges to a stable operating LR. Once it has, the per-step
wrapper overhead (snapshot + grad clone + Delta passes) and the `ref` buffer are
pure waste. `lr_freeze` ends adaptation:

- `"auto"` (**default**) — freeze when `sum(s)` plateaus (relative change stays
  small and near the running max for a number of consecutive steps).
- `int N` — freeze after `N` steps.
- `None` — never freeze (plain Mechanic-tuned Adakaon).

On freeze, `Autokaon` records `S = sum(s)`, **sets the inner Adakaon's `lr` to
`S`**, frees `ref` and the tuner scalars, and routes every later `step()` straight
to `base.step()`. After freeze it **is** plain Adakaon at `lr=S` — same memory
(`ref` gone), same speed (no wrapper passes), same update — *by construction*.

With the default `adakaon_betas=(0.0, 0.999)` (beta1=0, the minimum-VRAM
config), Adakaon's update is linear in `lr`, so the handoff is **bit-exact**:
`base.step(lr=1)` then `p = ref + S·Δ` equals `base.step(lr=S)`.

With `beta1 > 0` (momentum) the first-moment EMA is also linear in `lr` but
*carries history*: during warmup it accumulated `lr=1`-scaled updates while the
applied step was `S·Δ`. Freeze therefore folds `S` into the **stored momentum**
as well as the lr — otherwise the first frozen step throws the full `lr=1`-scaled
momentum at the `lr=S` regime, a one-time blow-up (measured ~500× the surrounding
steps — the little "celebration" jump when the LR locks in). With the fold the
momentum handoff is exact too: bit-exact for `float32`/`int8`/`4bit` (the
quantized codecs just rescale their per-row/block `m_scale`) and rounding-exact
for `bfloat16`. So freeze is seamless at any `beta1`.

### LR schedules after freeze (the Prodigy + Cosine pattern)

`Autokaon` is a proper `torch.optim.Optimizer`, so a standard LR scheduler
attaches to it directly. The intended pattern mirrors how parameter-free
optimizers are usually run — *discover* the LR, then *decay* it:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

opt = Autokaon(model.parameters(), lr_freeze="auto")
sched = CosineAnnealingLR(opt, T_max=total_steps)

for step in range(total_steps):
    ...                       # loss.backward()
    opt.step()
    sched.step()              # ignored during warmup; shapes the LR after freeze
    opt.zero_grad()
```

The scheduler writes `param_groups[...]["lr"]`. **During warmup that lr is
ignored** — the Mechanic scale (`sum(s)`) drives the update — so the scheduler is
effectively a no-op until freeze. **After freeze** the base runs alone off
`param_groups["lr"]`, so the scheduler now shapes the discovered LR `S` (Cosine
decays from `S` toward 0). Net effect: Autokaon finds the peak LR for you, then
Cosine anneals it — no manual peak-LR tuning. (A scheduler that reads
`get_last_lr()` before freeze sees the base lr, not the live Mechanic scale; use
`get_d()` for the effective LR during warmup.)

### Checkpointing

`state_dict()` / `load_state_dict()` capture the **full** optimizer — the base
Adakaon *and* the Mechanic tuner (`s`/`v`/`reward`/`max_product`/`ref`/seeds)
*and* the freeze bookkeeping — so a checkpoint taken **mid-warmup** resumes the LR
adaptation exactly (no cold-start re-bootstrap), and a **frozen** checkpoint
resumes frozen (no disruptive un-freeze + re-warmup). The snapshot is independent:
`state_dict()` deep-copies the base state, so you may keep training after taking
one without the checkpoint aliasing the live state.

## API

```python
Autokaon(
    params, lr=1.0, *,
    s_init="auto",                  # data-relative LARS seed (or a fixed float, e.g. 1e-8)
    lr_freeze="auto",               # "auto" (plateau) | int N | None (never)  — headline feature
    scale_cap="auto",               # ceiling on the discovered LR (or a float, or None)
    scale_cap_rel=6.0,              # advanced / rarely needed (see below)
    betas=(0.9, ..., 0.999999),     # the 6 Mechanic tuner recency horizons
    s_decay=0.01, eps=1e-8,
    adakaon_betas=(0.0, 0.999),   # inner Adakaon momentum betas (beta1=0 => bit-exact freeze)
    foreach_warmup=True,            # batch the warmup passes (LoRA: many small tensors)
    **adakaon_kwargs,             # clip_threshold, cautious, momentum_dtype, bf16_method, foreach, ...
)
```

**The common case is just `Autokaon(params, **adakaon_kwargs)`** — minimal and
parameter-free. The empirical scaffolding that accumulated across the tuning
campaign (`store_delta`, `s_init_rel`, `scale_floor_frac`, the auto-freeze
`tol`/`patience`/`max_frac`) is now **internal constants** at their validated
defaults, because iteration-3 showed those defaults generalize.

### The one advanced knob: `scale_cap_rel`

`scale_cap="auto"` lands a hard ceiling on the discovered LR at
`scale_cap_rel × (data-relative seed)`, which is the **load-bearing stability
fix**: on short horizons the Mechanic scale is prone to a large transient spike,
and the cap converts that into a robust ceiling near the operating LR. The default
`scale_cap_rel=6` **generalizes** (validated below — val flat across 3–12 on a
real SDXL LoRA), so almost no one needs to touch it. It is exposed only for a
power user training a very LR-sensitive model who wants a tighter/looser ceiling.

## Validated results (the campaign)

- **Real SDXL LoRA.** The auto-discovered LR is **1.66e-3, stable across seeds**,
  and lands the model **within ~1.8% of tuned Adakaon's val loss** — no LR
  search. The data-relative cap **generalizes**: validation is flat across
  `scale_cap_rel` 3–12.
- **Freeze == Adakaon.** After freeze the optimizer is plain Adakaon: measured
  **1.04× the speed** of (i.e. essentially identical to) plain Adakaon, at
  **0.5 B/param** extra state during the short warmup only, on the real LoRA.
- **Honest caveats:**
  - The warmup step is ~**8× the cost of one optimizer step** (it runs the base
    step plus the foreach-batched Mechanic passes). It is short and then frozen, so
    in practice it is ~2× a full step **during warmup only**, then free.
  - It **matches but does not beat** tuned Adakaon — the value prop is "no LR to
    tune," not better quality.
  - The real-LoRA task wasn't very LR-sensitive, which is part of why the cap
    generalized so flatly; a more LR-sensitive model is exactly where
    `scale_cap_rel` (the one advanced knob) might matter.

### Future work

For **full fine-tunes at the VRAM edge**, the warmup `ref` buffer adds ~2 B/param
(one extra weight copy + grad clone) — negligible for LoRA but gigabytes for a
full FT. CPU-offloading / block-swapping that `ref` buffer during warmup (it is
only touched on the wrapper passes, and is freed entirely at freeze) would let
full-FT users get freeze-to-free without the warmup-time VRAM bump.

## See also

- [adakaon.md](adakaon.md) — the base optimizer's design and API.
- [foreach-batching.md](foreach-batching.md) — the multi-tensor batching reused for
  the warmup passes.
- [momentum.md](momentum.md) — the cheap-momentum dial (`adakaon_betas` /
  `momentum_dtype`).
