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
  AdaBelief / AdamP / ADOPT, `t` for AdaMuon with `bias_correction`), whether to
  prebuild the codec's `mat` lookup, and the bucketing an optimizer is *pinned* to
  (see "Per-optimizer bucketing is part of the contract" below). KProdigy is the one
  optimizer whose bucketing depends on the **group** rather than the class — its
  `second_moment` decides whether an `ndim >= 2` weight carries a factored `row`/`col`
  pair or a full per-coordinate `v` — so it declares two specs and routes between them
  through the `_foreach_spec(group)` hook.
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
stacked call as `views=`. **Every optimizer on the shared plan passes them** — AdaBelief,
AdamP, ADOPT, AdaMuon, ScheduleFree, Adakaon, Lion and KProdigy. `views=None` still runs
the original code and is bit-identical, which is what **AdaPNM** (the last optimizer doing
its own bucketing) keeps using. The gains in the table below were measured on the first
four to adopt it; Lion's and KProdigy's are in their own section further down.

`stacked_views` returns `None` for a layout it cannot alias — a
non-contiguous `m` or `m_scale`, where a `reshape` would hand back a detached copy —
and the codec's existing per-parameter fallbacks take over. Passing a views object
built for another bucket is *ignored*, not misread: every consumer checks its `eff`.
That check is defense in depth — every caller passes the `eff` it built the views with,
and a re-chunk hands out fresh chunks — but the failure it prevents (a read in another
bucket's shape, a write into another bucket's buffers) would be silent.

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
| `load_state_dict` (it **replaces** the state tensors), `add_param_group`, an AutoLR base-state reset, a group falling back to the per-parameter loop | dropped explicitly |

A rebind that changes the *shape* is deliberately not supported (the factored second
moment is bound to the effective 2-D shape and there is no meaningful migration of an
EMA onto a different factorization); the stale bucketing raises a size mismatch on the
next step, which is the intended outcome.

The codec's cached view lists ride the same table: they live on the chunk, so every row
that rebuilds or drops the plan rebuilds them too. On top of that they are keyed on the
codec *instance*, so a group whose `momentum_dtype` changes cannot read another codec's
storage layout.

Note that the per-parameter clock's *value* changes every step while the *partition* it
induces does not — so the plan survives it. Only a parameter that actually skips a step
splits a bucket, and once split the two halves advance in lockstep again and the plan is
reused as before.

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
