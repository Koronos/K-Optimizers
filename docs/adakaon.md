# Adakaon — design & API

<!-- Formerly named "Adafusion". -->

> A conv-aware factored optimizer: **AdamW-level quality at a fraction of AdamW's
> optimizer memory**, with bf16-correct weight updates (stochastic rounding — *no*
> Kahan buffer, *no* CPU offload).

## Why

To keep AdamW's per-coordinate adaptivity you normally pay two full state buffers
(8 B/param). Adakaon factors the second moment **conv-aware** (reshape
`[out,in,kh,kw] → [out, in·kh·kw]` before factoring → near-zero state on convs
*and* attention) and keeps an optional momentum buffer in **bf16 or int8**,
recovering AdamW-quality convergence at 1–2 B/param. Stochastic rounding does the
bf16-correct update with **no extra state**, so unlike Adafactor+Kahan you never
allocate (or CPU-offload) a compensation buffer.

## Results (validated)

Mini pixel-DDPM on real CC0 images, held-out validation, 4 seeds:

| optimizer | val loss (↓) | optimizer state |
|---|---|---|
| AdamW | 0.0400 ± 0.0025 | 8 B/param |
| AdamW-8bit | 0.0364 | 2 B/param |
| **Adakaon** (bf16 momentum) | **0.0318 ± 0.0006** | **2 B/param** |

Beats AdamW by ~20% on held-out diffusion loss (non-overlapping across seeds) at
1/4 the optimizer memory. On a real 2.1 B-param DiT transformer, the no-momentum
config uses **0.01 GB** of optimizer state (vs AdamW's 8.4 GB), and `foreach`
batching (default) keeps its per-step cost competitive with fused AdamW
([foreach-batching.md](foreach-batching.md)).

> Honest caveat: small-scale benchmarks. At *zero* optimizer state
> (Adafactor-class), AdamW-quality is not achievable — momentum (~1–2 B/param) is
> the floor for the quality. Adakaon gives you the dial; see
> [momentum.md](momentum.md) for the int8/bf16/4bit momentum trade-offs.

> Note: in HF Adafactor, `beta1=0.0` (≠ `None`) still allocates a momentum buffer.
> `Adakaon(betas=(0.0, ...))` is true no-momentum.

## API

```python
Adakaon(
    params, lr=1e-3, betas=(0.9, 0.999), eps=(1e-30, 1e-3), weight_decay=0.0, *,
    clip_threshold=1.0,
    momentum_dtype="bfloat16",          # "float32" | "bfloat16" | "int8" | "4bit"
    momentum_4bit_block=128,            # block size for 4bit momentum
    cautious=True,                      # cautious masking; helps w/ momentum, no-op without (set False if beta1=0)
    cautious_wd="masked",               # weight decay inside ("masked") or outside ("full") the mask
    bf16_method="stochastic_rounding",  # "stochastic_rounding" | "kahan" | "none"
    foreach=True,                       # multi-tensor batching (foreach-batching.md)
    foreach_batch_cutoff=2_000_000,     # weights bigger than this loop instead of stacking
    foreach_stack_budget=None,          # chunk memory cap (None = adaptive to free VRAM)
    fused=False,                        # Triton GPU backend (recommended for CUDA fine-tuning)
)
```

With `fused=True`, fp32/bf16 parameters remain fused across LoRA matrices,
biases/norms, convolutions and large full-finetune tensors. 4-bit momentum
dequantizes, updates and requantizes inside the Triton kernels without allocating
a momentum-sized fp32 temporary — **at any `momentum_4bit_block`** since 0.7.12:
the one-block tile kernel takes the absmax block as a runtime scalar and buckets
its pointer arrays by it, instead of the old hardcoded 128 that diverted every
other block size to the native path (measured cost of that diversion on a
300-tensor bag: **16–20× slower**, 427 kernel launches instead of 2 and 39.5 MiB
of transient per step). The big/chunked route keeps its codec fallback only for
`m_block` values that do not tile a 1024-element chunk. Odd element counts are
supported by the shared nibble-storage contract; an odd **column** count still
leaves the one-block route (its nibble packing pairs adjacent columns).

`Adakaon` is a standard `torch.optim.Optimizer` that works one parameter at a
time, so it drops into per-parameter / gradient-release training loops unchanged.

## On `torch.compile`

`Adakaon` intentionally exposes **no** `compile` flag. A whole-step
`torch.compile` was benchmarked (adversarial `opt.step()` microbench, RTX 4080) and
came out ~neutral on most shapes and a slight loss on trivial steps — Adakaon's
step has little fusable elementwise math (no orthogonalization), so it is not worth
the API surface. The flag lives on [`AdaMuon`](adamuon.md), whose heavy
Newton-Schulz math it does speed up. (Model-level `torch.compile` on your *network*
is orthogonal and a separate, larger win — see your trainer's docs.)

## Weight decay and the cautious mask (`cautious_wd`)

Cautious masking (Liang et al. 2024) zeroes the update coordinates whose sign
disagrees with the gradient and rescales the survivors by `1/keep` so the mean
step magnitude is preserved. Adakaon has always folded decoupled `weight_decay`
into the delta **before** that mask, on every path (per-parameter, foreach and all
~13 Triton kernels). The consequence is easy to miss: the decay goes through the
mask too.

Measured as *the fraction of the requested `lr*wd*p` each coordinate actually
receives*:

| placement | keep | on survivors | on rejected |
|---|---|---|---|
| `"masked"` (default) | 0.64 | **1.49x** (≈`1/keep`) | **0.005x** |
| `"masked"` (default) | 0.50 | **1.985x** | **0.0013x** |
| `"full"` | any | 1.000x | 1.000x |

The aggregate shrinkage is preserved (that is what `1/keep` buys), but its
per-coordinate distribution is not: a coordinate the mask rejects is not decayed
at all that step, and a survivor is over-decayed. In the Cautious Optimizers paper
the decay is independent of the mask.

`cautious_wd="full"` is that independent placement: the mask applies to the
momentum/update term only, and `lr*wd*p` is subtracted from **every** coordinate.
`"masked"` stays the default (see the A/B below). With `cautious=False` the mask
is the identity and the two modes are the same arithmetic; with
`weight_decay=0` the knob does nothing. Every path implements both — per-parameter,
foreach and fused are element-for-element equal in either mode, and `"masked"` is
**bit-identical** to pre-0.7.12 Adakaon (verified over 24 configurations:
4 `momentum_dtype` x {fp32, bf16} params x {per-param, foreach, fused}).

Two test guards back that, because one is not enough. A *parity* check against the
per-parameter loop, at ~10x each dtype's measured fused-vs-per-param floor, is necessary
but blind on its own: a kernel that ignores the placement flag moves the weights by only
~1e-4 relative, so `"full"` would degrade silently to `"masked"`. The second guard is
*semantic* — under `"full"` the mask does not depend on `weight_decay`, so running the
same step with and without decay must differ by exactly `lr*wd*p` on **every**
coordinate. Measured deviation from that identity: **6e-5 for `"full"`, 0.8–1.0 for
`"masked"`** on every route and momentum dtype. Eight single-kernel mutants (one per
kernel that folds the decay) are all caught by it.

### A/B: which placement actually trains better?

Measured on the repo's diffusion proxy (`benchmarks/proxy/harness.py` driven by
`benchmarks/control/battery.py`'s `train()`): C=128 U-Net, 2000 steps, REX schedule +
progressive-resolution curriculum, Adakaon `betas=(0.9, 0.999)`, bf16 momentum,
`cautious=True`. Grid = `weight_decay` ∈ {0.01, 0.05} × lr ∈ {0.5, 1, 2} × 1.2e-3 ×
3 seeds, arms **interleaved inside each cell** (same seed back to back). Lower is
better in both columns; `w` counts the seeds where `"full"` was better.

| wd | lr | held-out loss `masked` | `full` | Δ | w | train–val gap `masked` | `full` | Δ | w |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | ×0.5 | 0.07849 | **0.07833** | −0.00017 | 3/3 | +0.02303 | **+0.02287** | −0.00017 | 2/3 |
| 0.01 | ×1.0 | **0.07446** | 0.07465 | +0.00019 | 1/3 | **+0.02116** | +0.02121 | +0.00006 | 1/3 |
| 0.01 | **×2.0** | **0.07158** | 0.07280 | +0.00122 | 1/3 | **+0.01563** | +0.01676 | +0.00113 | 1/3 |
| 0.05 | ×0.5 | **0.07795** | 0.07811 | +0.00016 | 1/3 | +0.02280 | **+0.02278** | −0.00003 | 1/3 |
| 0.05 | ×1.0 | **0.07341** | 0.07375 | +0.00034 | 1/3 | **+0.02029** | +0.02087 | +0.00058 | 1/3 |
| 0.05 | **×2.0** | **0.07045** | 0.07085 | +0.00040 | 1/3 | **+0.01565** | +0.01602 | +0.00037 | 1/3 |

Over all 18 paired runs, `full − masked` is **+0.00036 [−0.00009, +0.00080]** on loss
and **+0.00032 [−0.00013, +0.00078]** on the gap (95% CI) — *not resolvable*: the
intervals straddle zero and `full` wins 8/18 and 7/18. But the promotion rule for this
knob was "equal or better on loss **and** gap, in all 3 seeds, at the tuned lr", and
**×2.0 is the tuned lr at both weight decays** (best held-out loss in each row block).
There `full` loses 2/3 seeds on both metrics, at both `wd`. So:

> **`cautious_wd` stays `"masked"` by default.** The theoretically cleaner placement is
> available and fully supported, but it did not earn the default on this proxy.

Caveat on "the tuned lr": **×2.0 is the top of the swept grid**, so the optimum is bounded
from below but not from above — held-out loss was still improving at the edge. A wider
sweep could move the tuned point and, with it, the verdict; the grid was fixed in advance
at ×0.5/×1/×2 and is reported as run rather than extended after seeing the result.

The knob is worth trying if you are tuning weight decay for generalization — the whole
point is that `"masked"` makes the *effective* decay depend on the mask's keep rate, so
a `wd` tuned under one placement is not the same `wd` under the other. In the fused
path `"full"` is also ~9% cheaper per step (paired, `bench_wd_mblock.py --case wd`:
masked/full = 1.100x [1.086, 1.115] fp32, 1.087x [1.075, 1.099] bf16), because the
keep-count kernels no longer have to read the weights.

## Checkpointing

The normal `torch.save(opt.state_dict())` → `opt.load_state_dict(torch.load(...))`
workflow resumes **bit-exactly** and **preserves the configured `momentum_dtype`**.
This needs care: torch's default `Optimizer.load_state_dict` upcasts every state
tensor to the param's dtype (fp32), which would silently inflate a quantized first
moment back to fp32 on resume — e.g. `int8` → `fp32` is 4× the momentum bytes,
defeating the whole point of choosing it. `Adakaon` overrides `load_state_dict`
to restore each tensor to how it was checkpointed (bf16→fp32→bf16 and the
int8/4bit *codes* round-trip through fp32 losslessly).

> Note: `state_dict()` returns references to the live state (standard torch). To
> snapshot in-process and keep training the *same* optimizer before loading the
> dict elsewhere, `torch.save` it first — serialization freezes the snapshot. We
> deliberately do **not** deep-copy inside `state_dict()` so checkpointing never
> doubles peak VRAM (the case `Adakaon` is built for).

`Adakaon.load_state_dict` does three more things beyond that dtype-exact restore, and
all three matter for a resume to reproduce an uninterrupted run: it restores the fused
path's stochastic-rounding seed counter (and migrates a pre-0.7.11 lr-scaled momentum to
direction units) from the `_adakaon_meta` blob, back-fills a group key the checkpoint
predates, and drops every host-side cache that aliases the state tensors the load just
replaced (the fused pointer tables and the foreach plans). Wrapping optimizers
(`Lookahead`, `SAM`, `MSAM`, `Nekaon`) therefore restore their inner Adakaon by calling
*its* `load_state_dict`, so a checkpoint resumed through a wrapper restores the inner
Adakaon exactly as a bare one would.

Since 0.7.13 that also covers the **bf16 stochastic-rounding noise position**, which used
to be the one piece of a bf16 run's state no checkpoint carried. The bf16 weight write is
seeded from a counter, so a resume that restarted it at 0 rounded differently than the run
it continued — measured up to `4.7e-2` on bf16 weights four steps after a resume, with
every state tensor restored bit-exactly. A resume in a **fresh process, on the same
device**, is now bit-identical to the uninterrupted run on the native bf16 path and through
the wrappers, not just within one process. (Moving devices is a different question: a
CPU-trained checkpoint resumed on CUDA switches which SR implementation rounds the weights —
the Triton kernel instead of the torch reference — so it resumes *correctly* but not
bit-identically to a continuous CPU run.)

Where that position lives depends on which write does the rounding, and the two are
separate blobs:

| write | seeded from | checkpoint key |
|---|---|---|
| Adakaon's **fused** path (`fused=True`) | `Adakaon._t`, the fused step counter | `_adakaon_meta.fused_step` |
| the **native** path (per-param and foreach), every optimizer | the optimizer's own `SRStream` | `_sr_meta` |
| a **wrapper's own** write (`Lookahead`'s slow-weight sync, `SAM`'s climb) | that wrapper's own `SRStream` | `_sr_wrap_meta_<wrapper>` |

So a purely fused Adakaon run emits **no** `_sr_meta` at all (its noise never goes through
the shared kernel), and a nested wrapper stack writes one key per level —
`SAM(base_optimizer=Lookahead)` carries `_sr_wrap_meta_sam`, `_sr_wrap_meta_lookahead` and
the inner Adakaon's `_sr_meta`. Three more things to know:

- **Reseed before loading, not after.** `kaon.reseed_stochastic_rounding()` restarts every
  noise stream; calling it *after* `load_state_dict` throws away the position the
  checkpoint just restored.
- **A pre-0.7.13 checkpoint has no position to restore**, so it back-fills a fresh stream
  (draw 0) and loads without complaint; only the resume-exactness is missing. A new
  checkpoint still loads on 0.7.12 — the extra top-level keys are ignored there.
- **An optimizer that never rounds carries no position.** fp32 parameters,
  `bf16_method="kahan"` and the fused path never draw, so they neither take a noise-stream
  identity nor add a key — which is also what keeps a mixed fp32 + bf16 run's bf16 noise on
  the 0.7.12 sequence.

## See also

- [foreach-batching.md](foreach-batching.md) — the multi-tensor batching design
  and the `foreach_batch_cutoff` / `foreach_stack_budget` knobs.
- [momentum.md](momentum.md) — why int8 is the recommended cheap momentum and what
  cheaper ideas were rejected.
- [autolr.md](autolr.md) — why the experimental AutoLR path is quarantined.
