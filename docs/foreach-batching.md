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
  prebuild the codec's `mat` lookup.
- `ForeachPlan` — one group's bucket list plus the chunk split; re-chunked only when
  `budget // bucket_size` actually moves (the VRAM-adaptive budget wobbles every step,
  the chunk length almost never does).
- `ForeachChunk` — one stacked chunk's cached views.

Caching them **pins no memory**: every cached tensor is a view of something the
optimizer holds anyway. Gradients are the one deliberate exception — a retained view of
`p.grad` would keep the previous step's gradient storage alive (`set_to_none=True`
allocates a fresh grad every backward), adding a whole gradient set to peak memory, for
an optimizer family whose entire pitch is memory. Instead `ForeachChunk.grad_stack()`
stacks the **raw** gradients and reshapes the *stack* once: `torch.stack` always writes
a contiguous output, so that is element-for-element the same buffer as stacking N
per-parameter reshapes, at one `view` per bucket instead of N.

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
EMA onto a different factorization). On the **native** plan the stale bucketing raises a
size mismatch on the next step, which is the intended outcome — that is why the native
witness carries no shape field. The **fused** path cannot rely on that (it freezes
`Rs`/`Cs` and the row/col pointer arrays into the plan), so since 0.7.13 every fused
pointer cache validates the state geometry against the parameter whenever it is built:

- shape changed **and** something else moved too (fresh storage, dtype, device,
  contiguity) → the witness moves, the plan rebuilds, and
  `check_state_geometry` raises a message naming the parameter and what its `row`/`col`
  no longer fit. Free: build-time only. Before 0.7.13 this rebuilt the plan with the
  *new* `R`/`C` against the *old* buffers and stepped straight through them.
- shape changed and **nothing else** → no field moves, so the plan is never rebuilt and
  the weight keeps being stepped as its pre-rebind geometry. It stays in bounds (a view
  shares the whole storage), so this costs quality, not memory safety. Detecting it needs
  a per-param host sweep per step — measured at 3.2-3.6% of a 428-parameter LoRA step and
  10-18% of a launch-bound 0-D step — so it is opt-in:
  `kaon._fused_triton.SHAPE_WITNESS = True` adds per-param strides to the fused witness.
  See `benchmarks/fused/bench_shape_witness.py` for the full table and why strides beat
  `torch.Size`.

Either way the recovery is the same: reshape *before* constructing the optimizer, or
`del opt.state[p]` to restart that parameter's second moment at its new shape.

Note that the per-parameter clock's *value* changes every step while the *partition* it
induces does not — so the plan survives it. Only a parameter that actually skips a step
splits a bucket, and once split the two halves advance in lockstep again and the plan is
reused as before.

### Measured

`optimizer._foreach_cache_enabled = False` drops the cross-step cache (numerically a
no-op) — the A/B arm these were measured against. RTX 3000 Ada Laptop, bf16 params,
`momentum_dtype="bfloat16"`, 3 interleaved rounds × 40 reps per arm, `min` per-step wall
time, `aten::view`+`aten::reshape` counted with `torch.profiler`. Ranges span the five
optimizers (AdaMuon / AdaBelief / AdamP / ADOPT / ScheduleFree):

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
