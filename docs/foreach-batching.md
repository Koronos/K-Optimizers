# `foreach` batching — design & tuning

`Adakaon` — and AdaMuon, AdaBelief, AdamP, ADOPT, ScheduleFree, AdaPNM, Lion,
KProdigy — step parameters with multi-tensor (stacked) ops instead of a
per-parameter Python loop (`foreach=True`, the default). This note explains how it
works, the two knobs that control it, the measurements behind their defaults, and
the cached bucketing/view plan the batched path runs on. Measurements below are
Adakaon's unless stated otherwise; the mechanism and the knobs are the same
everywhere.

## Why

The per-parameter loop launches a separate set of CUDA kernels for every weight.
That cost (~0.22 ms/tensor of dispatch on an RTX 4080) is invisible for a handful
of large weights but dominates when many tensors are stepped at once — exactly the
adapter case. On a real SDXL UNet + PEFT LoRA r=8 (1434 tiny trainable tensors) the
optimizer step was **318 ms** vs **11 ms** for fused AdamW.

`foreach` buckets parameters by shape, stacks each bucket into one tensor, and runs
the whole update (EMA + reconstruction + RMS clip + momentum + weight decay +
cautious + stochastic rounding) as a handful of batched kernels:

- `ndim >= 2` → factored bucket `[N, R, C]`
- `ndim <= 1` (biases/norms, the bulk of a full fine-tune, plus 0-D scalars such
  as LyCORIS `use_scalar` gates, which ride the bucket as length-1 rows) →
  non-factored `[N, L]`

It is **element-for-element equal** to the per-parameter path (bit-exact on CPU,
~1e-8 on CUDA from reduction order; stochastic-rounding draws legitimately differ,
unbiased either way); `momentum_dtype="int8"` is also batched (per-row absmax
dequant/EMA/requant on the stacked layout), as is `"4bit"` (per-block absmax,
packed nibbles). Anything it doesn't cover — `bf16_method="kahan"`, fp16+SR,
non-contiguous matrixized convs, single-param (gradient-release) optimizers —
transparently falls back to the loop.

### Measured

| workload | per-param loop | foreach | vs fused AdamW |
|---|---|---|---|
| SDXL + LoRA r=8 (1434 tensors) | 318 ms | **15 ms** | 1.45× |
| SDXL full fine-tune (1680 tensors) | 339 ms | **256 ms** | — |
| Cosmos full fine-tune (685 tensors) | 239 ms | **231 ms** | — |

foreach is a large win for adapters (many tiny tensors, launch-bound) and a modest
one for full fine-tunes (dominated by large bandwidth-bound weights — there is less
launch overhead to remove there).

0-D scalars are the extreme of the launch-bound regime, and until 0.7.10 they were
excluded from batching entirely. Adakaon on 448 bare scalars with bf16 momentum
(`benchmarks/fused/bench_scalar0d.py`, median of 50 reps, RTX 3000 Ada Laptop):

| regime | native before | native after | fused before | fused after |
|---|---|---|---|---|
| 448 × `()` scalars | 330.0 ms | **6.4 ms** | 305.9 ms | **0.45 ms** |
| 448 × `(1,)` (control) | 20.1 ms | 4.9 ms | 0.51 ms | 0.27 ms |
| mixed 2-D + 1-D + 0-D | 122.1 ms | 14.7 ms | 118.9 ms | 2.26 ms |

The `(1,)` control is the point: a shape-`(1,)` param and a 0-D param are the same
amount of work, and the gap between them was pure per-tensor dispatch. They now share
the `L == 1` bucket, so the two rows track each other. The `after` column also carries
the two other changes in that release (one widening cast per bucket instead of one per
param, and the cached bucketing/view plan), which is why the `(1,)` control improved
too.

## The two knobs (deliberately decoupled)

A single "budget" used to control both *which* weights get batched and *how many*
are stacked at once. Those are different concerns and pulling them apart matters:

### 1. `foreach_batch_cutoff` — performance (default `2_000_000` elements)

Weights with more elements than the cutoff are stepped by the loop instead of being
stacked. Batching only pays off while per-tensor launch overhead dominates; a large
weight's update is compute/bandwidth-bound, so stacking it just adds copy traffic
and is *slower*.

This is an **absolute element count, not a fraction of VRAM** — the crossover is a
hardware property, not a memory-budget one. A budget sweep on SDXL and Cosmos full
fine-tunes (perf cutoff varied directly):

```
cutoff      131k   500k    1M     2M     4M     8M    16M
SDXL ms     272    272    266    277    272    398    547
Cosmos ms   231    232    236    235    234    326    394
```

Both models show a broad flat optimum up to ~4 M and a sharp slowdown beyond — the
*same* crossover, confirming it does not depend on the model. `2_000_000` sits in
the middle of the plateau. Raise it only if profiling your GPU shows a higher
crossover.

> **Why not auto-tune it online?** The optimum is a broad, flat plateau (a 30×
> range of cutoffs is within noise), it is static per (model, GPU), and online
> step-time is noisy and buried under the forward/backward. A hill-climber would
> chase noise on a flat surface to rediscover a constant. A fixed default wins.

### 2. `foreach_stack_budget` — memory safety (default `None` = adaptive)

The max elements in a single stacked chunk. Stacking allocates a few transient fp32
copies of the chunk, so an unbounded bucket of large weights can OOM a full
fine-tune. The budget bounds that.

- `None` → `min(adaptive_to_free_VRAM, 4 × foreach_batch_cutoff)`.
  - The VRAM term, `free_bytes × 0.10 / 48`, shrinks the chunk when a big model
    already fills the card and is the OOM-safety floor. The `48` is the measured
    peak transient bytes per stacked element (see below).
  - The `4 × cutoff` cap stops *over-stacking*: beyond a few cutoff-sized tensors,
    stacking medium weights just adds copy bandwidth. Measured on SDXL full FT
    (cutoff fixed at 2 M):

    ```
    chunk budget   4M    8M   16M   32M   64M   100M
    ms            281   261   318   350   354    355
    ```

    8 M (= 4 × 2 M) is the sweet spot; bigger is slower. Tying the cap to the
    cutoff keeps a single performance knob and means a roomy card never over-stacks.
- `int` → a fixed cap, returned verbatim (reproducibility, or a hard ceiling on a
  shared GPU). Not subject to the `4× cutoff` cap — you asked for an exact value.

Because the two are decoupled, **raising the stack budget never pulls large weights
into stacking** — it only allows bigger chunks of the already-eligible small ones.

#### The transient factor (`48 bytes/element`) is model-independent

The VRAM term divides by the peak transient bytes per stacked element. This is a
property of the optimizer's intermediate tensors, not the model — measured
byte-for-byte identical on SDXL and Cosmos shapes, and independent of tensor size,
aspect ratio, and conv-vs-linear. It depends only on path and config:

| path | common (`beta1=0`+SR) | worst (momentum+wd+cautious) |
|---|---|---|
| 2-D factored | 24 B | 38 B |
| 1-D non-factored | 28 B | 42 B |

`48` = worst observed (42.1) + margin, so the budget is a true ceiling: a chunk's
transient stays at or below the requested VRAM fraction.

## Check on your own GPU + model (`ktune`)

The defaults were tuned on an RTX 4080. The performance cutoff tracks a hardware
crossover, so a very different GPU *might* prefer another value. `ktune` measures it
on your machine — it reads the parameter shapes from a checkpoint's `.safetensors`
header (no full load, no real weights needed), builds matching tensors on your GPU,
sweeps the cutoff, and tells you whether to keep the default or change it:

```bash
# full fine-tune of a UNet / DiT transformer
uv run ktune --model /path/to/unet.safetensors --gpu 0

# a full SDXL checkpoint: select the UNet keys
uv run ktune --model /path/to/sdxl.safetensors --filter model.diffusion_model.

# the LoRA-adapter distribution instead of full weights
uv run ktune --model /path/to/unet.safetensors --lora-rank 8

# match your training config so the timing is representative
uv run ktune --model /path/to/transformer.safetensors --momentum bf16 --wd 0.01
```

It prints the per-param-loop baseline, the foreach speedup, a cutoff sweep, and a
verdict like `==> Keep the default` or `==> Consider foreach_batch_cutoff=…`.
Equivalent without the console script: `python -m kaon.tune --model …`.

## Tuning cheat-sheet

- **Default (`None`, `2_000_000`)**: near-optimal for SDXL/Cosmos LoRA *and* full
  fine-tune on any card. Start here.
- **Shared GPU / want a hard memory ceiling**: pin `foreach_stack_budget=<int>`
  (e.g. `2_000_000`).
- **Exotic GPU where profiling shows large weights still benefit from stacking**:
  raise `foreach_batch_cutoff` (the stack cap follows at 4×).
- **Disable batching entirely**: `foreach=False` (per-parameter loop; e.g.
  gradient-release setups already step one param per optimizer).

## The cached bucketing/view plan

Bucketing is not free. Every step the optimizer has to decide which parameters stack
together, split each bucket by the current chunk budget, and then — inside each bucket
body — build the lists of *views* the batched kernels write through: the (matrixized)
parameter views, the `row`/`col` or `v`/`s` state buffers, the momentum codec's `mat`
callback. On a bag of many tiny tensors that bookkeeping is a real fraction of the
step: 0-D scalars alone cost ~5 `flat_view` calls per parameter per step, and each one
materializes a fresh `aten::view`.

None of it changes between steps. The bucketing is a function of the parameters'
shapes, dtypes and devices; the views are functions of tensors the optimizer **already
owns** (`p.data` and its own `self.state` buffers). `kaon._foreach_plan` therefore
computes them once and caches them per param group:

- `ForeachSpec` — declared once per optimizer class: which state keys the bucket
  bodies walk, the optimizer-specific extra bucket key (the per-parameter step for
  AdaBelief / AdamP / ADOPT, `t` for AdaMuon with `bias_correction`), whether a
  one-element bucket may alias instead of stacking (AdaMuon), and the bucketing an
  optimizer is *pinned* to (see "Per-optimizer bucketing is part of the contract"
  below). KProdigy is the one optimizer whose bucketing depends on the **group**
  rather than the class — its `second_moment` decides whether an `ndim >= 2` weight
  carries a factored `row`/`col` pair or a full per-coordinate `v` — so it declares two
  specs and routes between them through the `_foreach_spec(group)` hook.
- `ForeachPlan` — one group's bucket list plus the chunk split; re-chunked only when
  `budget // bucket_size` actually moves (the VRAM-adaptive budget wobbles every step,
  the chunk length almost never does).
- `ForeachChunk` — one stacked chunk's cached views.

### The momentum codec's own view lists

The first-moment codec (`kaon._momentum_codec`) is the other half of the same problem.
Its stacked entry points — `ema_stacked` / `store_stacked` / `dequant_stacked` — walk
per-parameter lists of exactly the same kind (`[mat(state["m"]) …]`, the per-row
`[state["m_scale"].view(rowshape) …]`, the requant's write-back targets) and rebuilt
them on every step, once per *use site*: an int8 bucket cost four such lists per step.

Each codec therefore exposes `stacked_views(states, view, eff) -> _StackedViews | None`,
built **once per chunk** by `ForeachChunk.momentum_views(codec)` and handed to every
stacked call as `views=`. **Every optimizer that calls those stacked entry points
passes them** — AdaBelief, AdamP, ADOPT, AdaMuon, Adakaon, Lion, KProdigy and AdaPNM.
(ScheduleFree is on the shared plan too, but it reaches the codec through the
`CodecBuffer` helpers, which take `states` rather than a `mat` callback and so have no
view lists to prebuild.) `views=None` still runs the original code and is
bit-identical. **AdaPNM** was the last optimizer doing its own bucketing; since 0.7.18 it
is on the shared plan too (the param's lag behind the group step is its `extra_key`), and
its two momenta reach the codec through per-chunk alias dicts (`{"m": state["m_pos"], ...}`)
whose `stacked_views` are built once per chunk. The table below was measured on the first four to adopt it plus Adakaon's
own follow-up rows; Lion's and KProdigy's are in their own section further down.

Because nothing calls the codec's `mat` argument on the cached path any more, the
plan's old `ForeachSpec(momentum_cache=…)` flag — which prebuilt an identity-keyed
`{state["m"]: view(state["m"])}` dict for exactly that callback — is gone, together
with `ForeachChunk.mat`. Callers pass `chunk.view`, which is the same callback and is
only reached when `stacked_views` declined the layout.

`stacked_views` returns `None` for a layout it cannot alias — a
non-contiguous `m` or `m_scale`, where a `reshape` would hand back a detached copy —
and the codec's existing per-parameter fallbacks take over. Passing a views object
built for another bucket is *ignored*, not misread: every consumer checks its `eff`.
That check is defense in depth — every caller passes the `eff` it built the views
with, and a re-chunk hands out fresh chunks — but the failure it prevents (a read in
another bucket's shape, a write into another bucket's buffers) would be silent. It is
also what makes the guard worth its one tuple compare: a *per-codec* rather than
per-chunk cache was tried as a deliberate mutant and every bucket after the first
either stepped the wrong slice or raised a shape error.

Two things follow from the lists being views:

- **No memory is pinned** (the same argument as the rest of the plan). This is why
  the codec caches *view lists* and not a stacked `cat(out=)` scratch buffer: a
  persistent fp32 buffer per bucket would pin real memory for the process's lifetime
  and feed back into the free-VRAM-adaptive chunk budget, for an optimizer family
  whose pitch is memory.
- **The storage-identity contract is preserved.** Writes through the cached lists land
  in the very buffers MSAM/Nekaon cached `data_ptr` of, and an in-place requant
  (4-bit's included) leaves the cache valid. They go stale only when those buffers are
  *replaced* — `load_state_dict` — which is the plan-level invalidation the chunk
  already performs.

Measured per step (RTX 3000 Ada Laptop, bf16 params + `stochastic_rounding`, counted
with `torch.profiler`; `aten::select` is the stack/unbind residue below and does not
move):

| bag | codec | `aten::view` | `aten::reshape` | `aten::copy_` |
|---|---|---|---|---|
| AdaBelief, 448 × 0-D | int8 | 1802 → **10** | 1348 → **3** | 452 → **4** |
| AdaBelief, 448 × 0-D | bf16 | 903 → **7** | 450 → **2** | 3 → 3 |
| AdaBelief, 428-tensor LoRA | bf16 | 874 → **18** | 432 → **4** | 6 → 6 |
| ADOPT, 428-tensor LoRA | int8 | 874 → **18** | 2 → 2 | 436 → **8** |
| AdaMuon, 448 × 0-D | 4-bit | 20 → 20 | 8 → 8 | 454 → **6** |
| **Adakaon**, 448 × 0-D | int8 | 905 → **9** | — | 453 → **5** |
| **Adakaon**, 428-tensor LoRA | int8 | 897 → **41** | — | 448 → **20** |
| **Adakaon**, 128×(512,512)+64×(1024,) | int8 | 449 → **65** | — | 222 → **30** |
| **Adakaon**, 24×(320,320,3,3) | int8 | 85 → **37** | — | 39 → **15** |
| **Adakaon**, 448 × 0-D | 4-bit | 6 → 6 | — | 454 → **6** |
| **Adakaon**, 428-tensor LoRA | 4-bit | 24 → 24 | — | 452 → **24** |
| **Adakaon**, 128×(512,512)+64×(1024,) | 4-bit | 36 → 36 | — | 228 → **36** |
| **Adakaon**, 24×(320,320,3,3) | 4-bit | 18 → 18 | — | 42 → **18** |

The Adakaon rows are the follow-up (against the pre-change tree; the counts are
deterministic and identical across repeats). Only its **float codecs (`float32` and
`bfloat16`) are count-for-count unchanged** — those lists were already served by the
`momentum_cache` dict that this change deletes, and the cached views replace them at the
same cost. int8 wins on both columns (the scale views *and* the write-back); 4-bit wins
on `copy_` alone, for the reason spelled out just below.

The `copy_` column is the second half of the change: the quantized codecs wrote each
parameter's new `m_scale` with its own `copy_` (the scale shapes differ per parameter
layout), which the cached scale views collapse into one `_foreach_copy_`. 4-bit gains
only that — its `m` is a packed byte string with no effective layout, so it never had
per-parameter views to cache.

### Per-optimizer bucketing is part of the contract

Bucket order is numerically inert on its own: buckets touch disjoint parameters and
disjoint state, so reordering them cannot change a single fp32 value. It is **not** inert
for bf16 weights under stochastic rounding. Each bucket draws its rounding noise in one
shot, sized by the bucket, from a stream shared by the whole step (a module-owned
`torch.Generator`, or the Triton kernel's seed counter). Merge two buckets, split one, or
step them in a different order and every bf16 weight's last bit moves — unbiased either
way, but a run stops reproducing against its own history.

So an optimizer that bucketed differently before it moved onto this module has to keep
doing so, and `ForeachSpec` says how, declaratively:

| flag | what it pins | who sets it |
|---|---|---|
| `key_major` | all of key 0's buckets, then all of key 1's, instead of factored-then-flat | ADOPT |
| `insertion_order` | first-appearance order across both families, not factored-then-flat | Lion, KProdigy (`second_moment="full"`) |
| `scalar_bucket` | 0-D params get their own bucket instead of joining the `L == 1` one | Lion, KProdigy |
| `raw_shape_key` | two convs that matrixize to the same `[R, C]` stay apart | Lion |
| `matrixize=False` | `eff` is the weight's own shape, any rank; `view` is the identity | KProdigy (`second_moment="full"`) |

None of those splits is *wanted*: merging is strictly less work. They are the price of a
bf16+SR run not changing when the plumbing under it does, and each is anchored by a frozen
bf16 bit-pattern vector in `tests/test_foreach_plan.py` (`_FROZEN_SR_BITS`,
`_PINNED_SR_BITS`) captured from the tree before the migration. Regenerate one only when a
change to the *math* is intended.

`matrixize=False` is the one that is not purely historical. `second_moment="full"` keeps a
per-coordinate `v` shaped like the weight, so there is nothing to matrixize *for*, and not
reshaping is also what lets a channels_last conv keep being batched at all — a
`view(R, C)` of it does not exist, which is why every *factored* path has to reject a
non-contiguous conv and fall back to the per-parameter loop.

### Lion and KProdigy's migration

Both were the last two optimizers rebuilding their buckets and every derived view list on
every step, `views=None` into the codec included. Lion also carried its own copies of the
codec's stacked read/write and of the `m` / `m_scale` / `m_numel` / `m_block` layout.
Measured per step on a shared RTX 3000 Ada Laptop, bf16 params + `stochastic_rounding`,
counted with `torch.profiler` (Lion; `aten::select` is the stack/unbind residue and does
not move, and the CUDA launch count moves by `+0…+3` — the `_foreach_copy_` that replaces
N per-parameter scale writes):

| bag | codec | `aten::view` | `aten::reshape` | `aten::copy_` |
|---|---|---|---|---|
| 448 × 0-D | int8 | 3147 → **9** | 2693 → **2** | 453 → **5** |
| 448 × 0-D | bf16 | 1799 → **6** | 1347 → **1** | 4 → 4 |
| 448 × 0-D | 4-bit | 1364 → **18** | 1355 → **9** | 454 → **6** |
| 428-tensor LoRA | int8 | 3031 → **29** | 2583 → **6** | 446 → **18** |
| 428-tensor LoRA | bf16 | 1737 → **20** | 1293 → **3** | 12 → 12 |
| 300 convs + 128 × 0-D | int8 | 3031 → **31** | 2583 → **6** | 443 → **15** |
| 24 × (256,256) | int8 | 180 → **10** | 149 → **2** | 29 → **5** |

One counter moves the *other* way: on the quantized codecs `aten::as_strided` rises by
about one per parameter (448 × 0-D int8 1345 → 1793, 4-bit 1351 → 1799; 428-tensor LoRA
int8 1571 → 1715) — the price of routing the momentum through the codec's cached scale
views. Against `view`+`reshape` going 5840 → 11 on that same bag it is a net ~5.4 k fewer
dispatched ops per step, and on the float codecs `as_strided` does not move at all.

Unprofiled host wall time per step follows: **−29 … −78 %** (median of 80, Lion). Treat
that number as supporting evidence only — the same harness with the *reference* tree as
both arms reported −23 … +22 % on this (shared) GPU, so the clock here resolves ~±25 %
and the counters above are what actually rules out a regression. Peak allocated memory
**never rises**: over 48 configurations (both optimizers × 4 bags × 4 momentum dtypes) 34
are identical and 14 fall, by up to 11 % on Lion's 24 × (256,256) bag. Resident state is
byte-identical either way — the caches hold only views — so what moves is the transient
peak, and only downward.

**KProdigy's share is much smaller, and it is worth being precise about why.** Only its
*pass 2* (the weight update) runs on this plan; *pass 1* — the global D reduction plus the
`d`-scaled momentum and second-moment EMAs — keeps its own three bucketings, because they
span param groups and include the parameters pass 2 sends to the per-parameter loop, which
is not what a per-group plan describes. Pass 1 is where most of the per-step view traffic
lives:

| bag | codec | `aten::view` | `aten::reshape` | of which pass 1 |
|---|---|---|---|---|
| 448 × 0-D, `full` | int8 | 3148 → **2706** | 2245 → **1797** | ~2706 / ~1797 |
| 448 × 0-D, `full` | 4-bit | 923 → 927 | 908 → 908 | all of it |
| 300 convs + 128 × 0-D, `factored` | int8 | 3936 → **2614** | 1551 → **1123** | ~2300 / ~1100 |
| 300 convs + 128 × 0-D, `factored` | bf16 | 2384 → **1489** | 560 → 560 | ~1489 |
| 428-tensor LoRA, `full` | int8 | 2474 → **2052** | 1299 → **871** | ~2050 / ~870 |
| 428-tensor LoRA, `full` | bf16 | 889 → 894 | 292 → 292 | all of it |

The `+4 … +5` rows are the honest other side of the trade and they do not scale: a bucket
whose parameters are already in their effective layout (a 2-D weight, or a 0-D scalar the
pre-plan `_full_bucket` stacked raw) never had per-param views to cache, and the chunk
adds exactly **one** `aten::view` per bucket per step — the single reshape of the gradient
*stack* that replaces N per-param gradient views. Where there are real per-param views to
remove (quantized momentum, matrixized convs) it removes several hundred to 1300 of them.
Pass 1 is the largest remaining opportunity in KProdigy and is untouched here.

### Staleness

A cached view of memory the optimizer no longer owns would silently step a detached
buffer, so the plan is rebuilt or dropped on all of:

| event | detected by |
|---|---|
| the param set changes (including the foreach/per-param split moving a param) | `param_witness` — `id` |
| `p.data = <fresh storage>` (external EMA, `.to(dtype/device)`, an offloader's block swap) | `param_witness` — `data_ptr` |
| `p.data = p.data.t()` on a square weight (id, pointer and shape all unchanged) | `param_witness` — `is_contiguous` |
| parameters fall out of lockstep (per-parameter clocks split into several buckets) | the plan's key *partition* signature |
| the stack budget moves | `ForeachPlan.rechunk` |
| **a state buffer is retired while every parameter stands still** — `del opt.state[p]`, `opt.state[p].clear()`, `opt.state[p]["m"] = ...`, `["row"] = ...` | `state_generation` — the state mapping's own counter (`WatchedState`) |
| `load_state_dict` (it **replaces** the state tensors), `add_param_group`, an AutoLR base-state reset, a group falling back to the per-parameter loop | dropped explicitly |

The state-identity row is 0.7.14 and it closes a hole no parameter field can see. Every
row above it observes the *parameters*; the plan caches views of the *state*
(`ForeachChunk.state_views` aliases `row`/`col`/`v`, and the codec's own lists alias
`m`/`m_scale`). `del opt.state[p]` moves no parameter at all, so the plan was reused, the
chunk kept stepping the retired buffers **and** the fresh state never went through
`_init_state` — confirmed writing every retired buffer on the native plan and on all four
fused routes, byte for byte identically with the caches on and off.

It is a counter and not a witness field because a witness field cannot be made cheap
enough *or* complete. `self.state` is a `WatchedState` whose per-param dicts count every
rebinding of a key some cache bakes (`WATCHED_STATE_KEYS`), so a cache compares ONE
integer and there is no per-step sweep at all. Priced against the three-field parameter
witness on the reference bags
(`benchmarks/fused/bench_state_witness.py --case field`, 25 pairs, 95% CI, ranges spanning
TWO machines — the ordering and the conclusion were identical on both, the absolutes differ
by ~2x):

| candidate | 428-param LoRA bag | sees `del` | sees `state[p]["m"] = ...` | sees `state[p].clear()` |
|---|---|---|---|---|
| (a) `+ tuple(map(Tensor.data_ptr, m buffers))` | +110 … +144 µs, 7.0-9.2% of the fused step | yes | yes | **no** (dict, params and pointers can all come back identical) |
| (b) `+ tuple(map(id, state dicts))` | +45 … +95 µs, 2.9-6.0% | yes | **no** | **no** |
| (c) `state_generation()` — **shipped** | 133 ns/call × 1-6 calls/step = 0.13-0.80 µs, **0.01-0.09%** | yes | yes | yes |

Row (c) is accounting, not a paired A/B: the same harness that puts (a) and (b) well
outside their CIs returns −1.8 µs [−6.9, +3.2] and +12.2 µs [+4.2, +20.2] for the counter
on the two machines — zero either side of its own noise floor. The load-independent
figures are the ones to hold it to: Python bytecode per step moves by +0.05…+0.74%
(Adakaon) and +0.34…+1.45% (AdaPNM), and every `torch.profiler` CUDA column is identical.

The counter is read once per group per step in `_foreach_chunks` and once per group plus
once per fused route/bucket in `Adakaon._fused_partition` and the pointer caches —
1 call/step on a native step, 2-6 on a fused one (`--case calls`). It moves only when a
baked key is *rebound*: populating a key that was absent does not count (nothing could
have cached it), which is what keeps `_init_state` free and stops a parameter's first step
from costing a rebuild on its second. Optimizers that do not install the watch read a
constant, so nothing outside Adakaon/AdaPNM changes at all.

The one place the guard is not free is the WRITE side: every `state[key] = value` now goes
through a Python-level `__setitem__`. Adakaon makes no per-step state write, so it pays
nothing. AdaPNM writes one per parameter (`state["step"] += 1`), which is why that write
alone goes through `dict.__setitem__` — see `AdaPNM._prepare_param_steps` for the
+56…+107 µs vs +18.5…+48 µs measurement (two machines, 1.1-1.7% vs 0.35-0.9% of that
bag's AdaPNM fused step), and `tests/test_state_identity_witness.py` for the invariant
that keeps the bypass legal.

Watching a `dict` means covering **every** slot it mutates through, which is more than the
methods one thinks of: `|=` reaches `nb_inplace_or`, a different slot from both
`__setitem__` and `update`, and it slipped through the first round —
`opt.state[p] |= {"m": fresh}` retired the buffer while the counter stood still, and
`opt.state |= {p: {...}}` additionally left the incoming plain dict unadopted. The test
file now sweeps the whole API from a table (`__setitem__`, `operator.setitem`, `__ior__`,
`__delitem__`, `clear`, `pop`, `popitem`, `update` × mapping/kwargs/pairs, `setdefault`,
`__missing__`) against both classes, with the two deliberate `dict.__setitem__` /
`dict.update` bypasses listed as such, so a new mutator has to be added to the table
rather than remembered. Assignment is covered too: `opt.state = defaultdict(dict)` is
re-wrapped by `WatchedStateMixin.__setattr__` — interception on the WRITE, which a step
does 1-4 times, rather than a `state` property whose getter would cost a Python call on
every `self.state` read. Counted with a counting property: an AdaPNM fused step makes
1687-1884 of those on the 428-parameter bag (Adakaon fused 631, Adakaon native 1), and a
property access costs 21-42 ns more than a plain attribute — +40…+71 µs/step, one to two
orders of magnitude above the 0.13-0.80 µs the guard costs. Two machines.

The `load_state_dict` row is why a **wrapper** (`Lookahead`, `SAM`, `MSAM`, `Nekaon`)
must restore its inner optimizer through the inner's *own* `load_state_dict` and never
through a lower-level loader: that drop is the inner's, so bypassing it leaves a plan
(and, on Adakaon, the fused pointer tables) cached under a dead `id(group)`, aliasing
state tensors the load replaced.

A rebind that changes the *shape* is deliberately not supported (the factored second
moment is bound to the effective 2-D shape and there is no meaningful migration of an
EMA onto a different factorization). On the **native** plan the stale bucketing raises a
size mismatch on the next step, which is the intended outcome — that is why the native
witness carries no shape field. The **fused** path cannot rely on that (it freezes
`Rs`/`Cs` and the row/col pointer arrays into the plan), so since 0.7.13 every fused
pointer cache validates the state geometry against the parameter whenever it is built:

- shape changed **and** something else moved too (fresh storage, dtype, device,
  contiguity) → the witness moves, the plan rebuilds, and
  `check_state_geometry` raises a message naming the parameter and what its `row`/`col`
  no longer fit. Free: build-time only. Before 0.7.13 this rebuilt the plan with the
  *new* `R`/`C` against the *old* buffers and **wrote** through them — measured as the
  neighbouring allocation (`state[p]["col"]`, 512 B past `row`) being modified, with no
  exception raised.
- **`load_state_dict` from a checkpoint whose shapes differ** → the same mismatch with no
  rebind at all. A checkpoint saved from `(512,16)` weights loads onto `(16,512)` ones
  without complaint (same param count, same numel, and shapes are not compared), after
  which `row=512`/`col=16` face an effective shape of `(16,512)`: 2048 B into a 64-byte
  `col`, and `p` came out non-finite on 0.7.12. `load_state_dict` already drops the
  caches, so the rebuild runs the check.
- shape changed and **nothing else** → no field moves, so the plan is never rebuilt and
  the weight keeps being stepped as its pre-rebind geometry. For a plain `view` that stays
  in bounds (the storage is the same size), so it costs quality, not memory safety.
  Detecting it needs a per-param host sweep per step — measured at 3.0-3.6% of a
  428-parameter LoRA step and 8-18% of a launch-bound 0-D step (two machines, both shared)
  — so it is opt-in: `kaon._fused_triton.SHAPE_WITNESS = True` adds per-param strides to
  the fused witness. See `benchmarks/fused/bench_shape_witness.py` for the full table and
  why strides beat `torch.Size`.

The opt-in field has a blind spot worth knowing about: strides are a *proxy* for the
shape, and truncating dim 0 of a contiguous tensor does not move them — `(16,64) ->
p.data[:8]` keeps `(64,1)`, a 1-D `[:256]` keeps `(1,)`. Only `numel` moves, and `numel`
is the complementary field (cheaper, but blind to the `view` case that preserves numel;
only both together are complete). Such a parameter keeps being stepped at its
pre-narrowing extent, which means the optimizer writes past the parameter's current
`numel` *inside the original storage* — so a sibling view of that storage (a split QKV,
anything from `chunk()`/`split()`) is silently rewritten. Neither the flag nor the
build-time check closes that; what does in practice is the narrowing also moving the
storage, which is the ordinary case and is caught by default.

The recovery is always the same: reshape *before* constructing the optimizer, or
`del opt.state[p]` to restart that parameter's second moment at its new shape. That
recovery is only *sound* because of the state-identity row above: before 0.7.14 it
happened to work after a refused rebind (the refusing cache had never finished building)
and silently did not work from a steady state, where the tables were already there and
went on writing the buffers the user had just dropped. Note the
refusing step may be **partially applied** — the fused subsets dispatch in order (native,
one-block, big, 1-D) and the ones ahead of the failing one already launched — so carry on
from the next step rather than retrying it.

The codec's cached view lists ride the same table: they live on the chunk, so every row
that rebuilds or drops the plan rebuilds them too. On top of that they are keyed on the
codec *instance*, so a group whose `momentum_dtype` changes cannot read another codec's
storage layout.

Note that the per-parameter clock's *value* changes every step while the *partition* it
induces does not — so the plan survives it. Only a parameter that actually skips a step
splits a bucket, and once split the two halves advance in lockstep again and the plan is
reused as before.

### The fused side: one witness sweep, not one per bucket

Adakaon's Triton routing keeps its own caches next to this plan, guarded by the same
three fields (`Adakaon._fused_partition` calls the very same `param_witness`). Those
caches are validated in one of two ways, and which one applies is decided by whether the
caller can hand back **the same list object** it was built from:

- `_WitnessedCache.built_from(plist)` — O(1) list identity. Legitimate only because
  `_fused_partition` re-derives the witness across the whole group *every step* and
  returns the very same route lists while nothing moved, and `_fused_demote` rebuilds
  them into fresh objects the moment the non-contiguous-grad set changes. "This is the
  list I was built from" is therefore exactly as strong as recomparing the tuples.
- `_WitnessedCache.stale(plist)` — the full witness tuple, for a caller that cannot
  offer that guarantee.
- `_WitnessedCache.revalidate(plist)` — the same tuple, but it **adopts** a fresh list
  object that describes the same parameters (AdaPNM's routes, where a mixed-lag group
  re-splits every step).

All three also compare the state-identity generation, and they compare it **first** — one
integer against `len(plist)` element compares. That is not redundant with
`_fused_partition`'s own check: the partition's check makes the routing hand back fresh
lists, which is a *route* from a state change to a cache, and `revalidate` is precisely
the method that severs it — it takes the fresh list and says "same parameters, keep the
tables". Measured before the caches carried the generation themselves: 4/4 retired
`m_pos` buffers written on AdaPNM's one-block route and 3/3 on its 1-D route, with the
partition rebuilding correctly on every one of those steps. The generation therefore lives
where the dangling pointers live.

The one-block and 1-D routes always took the first path. The **big** route could not:
`_fused_big` re-derived its same-shape/dtype/device buckets on every step, so the lists
were new objects and `BigPointerCache` had to re-sweep. `Adakaon._big_shape_buckets`
memoizes the split per group, so the big route validates by identity too — and when the
memo does hand out a fresh list (something moved), a rebuilt pointer cache is the correct
and conservative outcome, so Adakaon needs no `stale` fallback at all.
`AdaPNM._big_shape_buckets` is the same memo keyed per `(group, lag)`, but AdaPNM keeps
its `stale` fallback: a genuinely mixed-lag group gets fresh sub-lists from
`_local_step_buckets` every step, and there only the full compare can tell a `p.data`
rebind from a harmless re-bucketing. Measured witness sweeps per step:

| bag (fused) | partition | per-bucket | total |
|---|---|---|---|
| 80 big tensors over four shapes | 1 | 4 → **0** | 5 → **1** |
| 200×(256,256) + 100×(512,) + 128×0-D | 1 | 1 → **0** | 2 → **1** |
| 24×(320,320,3,3) | 1 | 1 → **0** | 2 → **1** |
| 128×(512,512) + 64×(1024,) | 1 | 1 → **0** | 2 → **1** |

The memo is dropped by `_invalidate_fused_caches` with the pointer caches, and holds a
reference to the route list it split — so that list's `id` cannot be recycled underneath
it while the entry lives. What it does **not** watch is the one thing nothing here
watches: a shape-changing `p.data = p.data.view(...)` rebind (see the note above).

What a sweep costs, measured directly (CPU-only, medians of 200 calls on this machine):
4.1 µs over 20 params, 15.8 µs over 80, 35.6 µs over 200, 71.8 µs over 428. So the
80-tensor / four-shape bag above sheds ~16 µs of host work per step, and the LoRA bag's
200-tensor big bucket ~36 µs.

The GPU wall clock does not resolve any of this on a shared laptop card, and nothing is
claimed from it. Best attempt, on the 80-tensor / four-shape bag with 200 timed steps per
measurement and six interleaved base / arm / base repeats: the arm sits at −14.7 %
(bf16 momentum) and −8.7 % (int8) of the base's host time — but the base-vs-base
*control* sits at +6.4 % and −0.5 %, and its per-repeat deviation from the paired base
repeat spans −42 … +83 %. The sweep counts and the microbenchmark above are the
measurement; the clock only rules out a large regression.

### Measured

`optimizer._foreach_cache_enabled = False` drops the cross-step cache (numerically a
no-op) — the A/B arm these were measured against. RTX 3000 Ada Laptop, bf16 params,
`momentum_dtype="bfloat16"`, 3 interleaved rounds × 40 reps per arm, `min` per-step wall
time, `aten::view`+`aten::reshape` counted with `torch.profiler`. Ranges span the five
optimizers that adopted the shared module at extraction time (AdaMuon / AdaBelief /
AdamP / ADOPT / ScheduleFree):

| bag | Δ view/reshape per step | Δ CPU self | Δ ms/step (min) | Δ ms/step (median) |
|---|---|---|---|---|
| 448 × 0-D scalars | −27 … −50 % | −13 … −46 % | **−32 … −50 %** | −36 … −55 % |
| 300 × conv (16,8,3,3) + 128 × 0-D | −25 … −49 % | −7 … −33 % | **−21 … −38 %** | −27 … −46 % |
| 200 × (256,256) + 100 × (512,) + 128 × 0-D | −10 … −23 % | −1 … −19 % | −0 … −9 % | −0 … −7 % |

The win scales with how many bucket entries need a *real* view. A `(256,256)` weight is
already in its effective layout, so its `mat` is the identity and there was never a view
to cache; a bag dominated by such weights is GPU-bound and the plan only trims host
work. 0-D scalars and matrixized convs are the opposite extreme — every list entry is a
real view, and they are exactly the launch-bound bags `foreach` exists for.

### Adakaon's migration onto the shared module

Adakaon reached this design first and carried its own copy of it (`_ForeachPlan`,
`_ForeachChunk`, `_param_witness`) until the copy was retired. The numbers above are
therefore the shared implementation's, measured on the other five. Folding Adakaon's copy
into `kaon._foreach_plan` had to be **neutral**, which is the only acceptable outcome for
a plumbing move, and neutrality was measured three ways — hardest evidence first:

1. **Contention-immune counters.** The CUDA launch count and the
   `aten::view`/`reshape`/`select`/`as_strided`/`unbind`/`flatten` count for one step are
   *identical* between the two implementations, on every bag and in both kernel modes:
   428 LoRA adapters 194 / 4328, 448 0-D scalars 77 / 3150, 300 matrixized convs + 128
   scalars 167 / 3935, 200×(256,256)+100×(512,)+128×0-D 296 / 3570, 24×(320,320,3,3)
   259 / 311, 128×(512,512)+64×(1024,) 486 / 1790. Same kernels, same dispatch, same
   order.
2. **The host-side call itself.** One cached-plan retrieval on CPU
   (`_foreach_chunks` vs the old `_foreach_plan`), paired and order-alternating,
   n=1500 pairs: **+0.15 … +0.8 %** of a 57–96 µs call, i.e. **≤ 0.6 µs per step** on a
   4.5–35 ms step. The same harness run against the reference tree *twice* (a null A/B)
   produced −0.5 … +0.4 %, so that is its own bias floor.
3. **Paired GPU wall clock**, both trees loaded into one process over *shared* parameter
   bags (so allocation order cannot favour an arm), 5 interleaved repeats × 100 pairs per
   bag per mode: every bag/mode inside ±1.8 %, none significant except a 1.0 % *win* on
   the 128×(512,512) bag's foreach path. This one comes with a caveat that is the point of
   running a control: the same harness, with the reference tree as *both* arms, reported a
   "significant" +3.1 % on one bag. On a shared laptop GPU this design resolves ~3 %, not
   2 %, so the counters above — not the clock — are what actually rules out a regression.

### Known ceiling: the `stack` / `unbind` residue

What the view caches do **not** remove is `aten::select`: a 448-scalar bag still costs
~1344 of them, three `torch.stack` + three `unbind` per step (the state stack and its
write-back, the codec's, and `subtract_batched_`'s delta slices). Those operate on
tensors that are *freshly allocated every step* — the stacked update, the delta — so
there are no cross-step views to cache. Removing them needs a persistent stacked
buffer, which is the trade rejected above.
