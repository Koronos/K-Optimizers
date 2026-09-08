# Candidate #4 (+#5): fuse the factored reductions into Triton — BUILT + VERIFIED

**Status:** BUILT and merged into `feat/fused-extra-kernels` (toggle `_fused_reductions`, default ON).
Measured ANOTHER 2.76–2.79× on top of #1 (big regime → ~5× over native-foreach); parity vs native
fp32 ~5e-7 / bf16 <2e-2; full repo 522/522. Subsumes candidate #5 (GC is done in the reduction
kernel). This doc is kept as the implementation record. **Real-workload caveat:** on Anima DiT LoKr
at 512–1024 px the 5× optimizer win is invisible in `iter_sec` (optimizer is <1 % of the DiT step);
it's "free" + correct, and a real lever only in optimizer-bound regimes. The design below is what
shipped.

## The finding (measured, RTX 4080)

In the **batched big path** (`_chunked_step_batched`, the dominant Cosmos LoKr regime — 236× 512×512
factors), the torch **reductions are 73–80 % of the step**, the Triton mom/apply kernels only ~20–27 %:

```
Adakaon batched 236x512x512: reductions=5.41ms  full=7.41ms  (73%)
AdaPNM   batched 236x512x512: reductions=5.81ms  full=7.27ms  (80%)
```

Breakdown of `_chunked_reductions_batched` (≈5.4 ms):

```
torch.stack([grad.float().reshape(R,C)])  = 2.24ms   <- 248MB [N,R,C] fp32 gather+cast
  + Gradient Centralization (g.sub_(mean)) = 1.20ms
gsq = g*g                                  = 0.83ms
row/col EMA lerp + foreach_copy            = 0.86ms
rms via bmm                                = 0.41ms
```

So **stack + GC + gsq ≈ 4.3 ms (80 % of reductions)** is the prize: it's the 248 MB fp32 stack
materialization + two extra full passes. A pointer-array reduction kernel (read grad once, no stack,
GC + row/col sums in-register) could remove most of it — potentially ~2× the real big step again,
on top of candidate #1's 1.7–2.1×.

## Why it's a real refactor (not a quick win)

The torch stack serves **double duty**: the reductions AND feeding the mom/apply kernels (#1), which
read the GC'd `g` from the stacked `[N,n]` buffer. To remove the stack fully, the mom/apply kernels
must also (a) read grad via a pointer array and (b) apply GC in-kernel — i.e. the change reaches the
already-verified #1 kernels. A "bounded" version that keeps writing the stack from a kernel saves
only GC+gsq (~1.2–2 ms, ~1.2×) and still pays the 248 MB write — not worth the atomic-kernel
complexity. The full no-stack design is the one worth doing.

## Full design (no stack anywhere; GC in-kernel — subsumes #5)

Per-tensor outputs needed by the existing mom/apply kernels: `r_factor[N,R]`, `c_factor[N,C]`,
`inv_rms_lr[N]`, and the GC'd grad (read in-kernel by mom/apply instead of from a stack).

1. **`_factored_reduce` kernel** — grid `N*ceil(R/BR)`, one program per (tensor, row-block of BR
   rows), contiguous `[BR, C]` load via the grad pointer array (bf16/fp32, LOWP constexpr):
   - per-row mean over C → GC: `g' = g - rowmean` (mask padded cols to 0 so the mean is exact).
   - `rowsum_gsq[t, r0:r0+BR] = sum_c g'^2`  (stored directly — the program owns these rows, no atomic).
   - `colsum_gsq[t, :] += sum_rows g'^2`  (atomic_add into `col[N,C]`; few blocks per tensor keep
     contention bounded — tune BR for the atomic/occupancy trade-off).
2. **torch** (cheap, [N,R]/[N,C], no stack): `row.lerp_(rowsum/C + eps1)`, `col.lerp_(colsum/R + eps1)`,
   copy back; `r_factor = (row/row.mean).rsqrt`, `c_factor = col.rsqrt`.
3. **`_factored_rms` kernel** — grid `N*ceil(R/BR)`: re-read grad, GC, accumulate
   `r_factor[r]^2 * sum_c (g'^2 * c_factor[c]^2)` → atomic_add to `rms[N]`. torch: `inv_rms_lr =
   lr / max(sqrt(rms/n)/clip, 1)`.
4. **mom/apply kernels (#1) change**: drop the stacked `g_ptr`; read grad via the grad pointer array
   (`g_addr[t]`, LOWP) and apply GC in-register (reuse the per-row-mean trick, or pass the precomputed
   `rowmean[N,R]` from step 1). Everything else (momentum EMA, cautious, WD, subtract, SR) unchanged.

AdaPNM is the same (its reductions have no rms — the clip is already in-kernel from candidate #1; just
needs the row/col-sum kernel + GC-in-mom/apply). Conv (#3) rides this unchanged (matrixized R,C).

### Risks to validate
- **Atomic contention** on `col[N,C]` / `rms[N]` — the main perf risk; if atomics dominate, switch the
  col reduction to a two-pass (partials → reduce) or a column-tiled layout. MUST beat the torch
  baseline (5.4 ms) to ship — measure with `benchmarks/fused/bench_fused.py --regime big`.
- Parity: GC-in-kernel must match `centralize_grads_` (mean over the matrixized fan-in) bit-for-bit in
  fp32; reuse the existing `_run_parity` net (exact fp32, bounded bf16).
- Padded-lane masking in the mean (padded cols must not bias the per-row mean).

### GC is a PER-BUCKET constexpr, and the predicate has one definition (0.7.13)

`GC` reaches every kernel above as a `tl.constexpr`. A Triton program cannot call Python, and one
launch serves one bucket, so **whether GC applies must be decided on the host, per bucket, and must
agree with what `centralize_grads_` does on the native route** — otherwise the same weight takes a
different step depending on which route it landed on. The contract:

1. **The predicate is `kaon._backend.gc_applies(shape)` — `len(shape) >= 2 and
   math.prod(shape[1:]) > 1` — and nothing else.** It is written as a product over the fan-in dims,
   not as `numel // shape[0]`, so that an empty output dim (`shape[0] == 0`) answers instead of
   raising `ZeroDivisionError`. GC is undefined for a fan-in of 1: the mean of a one-element row *is* the
   element, so `g - mean(g)` is identically zero, which froze `(out, 1)` / `(out, 1, 1, 1)` weights
   silently until 0.7.13. Every host site resolves the flag through that function; no site open-codes
   `C > 1`. `tests/test_gc_fanin_1.py::test_gc_applies_is_the_only_definition` walks the sources and
   fails if one does.
2. **It is uniform inside every bucket, and that is checked, not assumed.**
   `kaon._fused_triton.bucket_gc_ok(plist)` evaluates `gc_applies` over the bucket and **raises** if
   the members disagree. It holds today for two different reasons: the big routes bucket by exact
   shape, and the one-block routes bucket by the padded tile where `BC = next_pow2(C) == 1` exactly
   when `C == 1`, so a fan-in-1 tensor can never share a tile with a tensor GC applies to. A future
   change to a bucket key that broke that gets an error, not a divergence.
3. **The reduction and the mom/apply kernels must be given the SAME resolved flag.** On the
   batched-big routes (`_chunked_mom_batched_g` / `_chunked_apply_batched_g` and their AdaPNM,
   4-bit, int8 and nomom variants) GC is re-applied downstream from the `rowmean` that
   `_reduce_rowcol` wrote. Two independently-derived flags would centralize the update with a mean
   the reductions never subtracted — so `Adakaon._chunked_step_batched` and
   `AdaPNM._chunked_step_batched` compute it once and thread it through
   `_chunked_reductions_fused` / `_chunked_reductions_batched` / `_chunked_step_batched_nomom`
   instead of re-reading the group dict.
4. **Zero per-step cost.** The predicate is evaluated where the plan or pointer cache is built:
   `centralize_grads_` per distinct shape bucket, `PointerArrayCache`/`AdaPnmCache` per tile bucket
   (`bucket["gc_ok"]`), `BigPointerCache`/`BigPnmCache` per shape bucket (`cache.gc_ok`). A launch
   site only ANDs two booleans. `BigPointerCache.gc` stays the *effective* flag, because its
   `rowmean`-onto-`rowsum` alias is only valid while GC is off — the caller compares against
   `group["gradient_centralization"] and cache.gc_ok` and rebuilds when that moves.

## How to resume
Implement the three kernel changes above on this branch behind a `self._fused_reductions` toggle
(default False until it beats 5.4 ms with parity, then True), add parity tests mirroring the #1
batched tests, and run `bench_fused.py --regime big` + the battery before/after. If it lands a
measured win with parity, merge into `feat/fused-extra-kernels`.

---

## 0.7.12 follow-up: the reserved two-pass, used for DETERMINISM (not contention)

The "Risks to validate" section above kept a two-pass (partials → reduce) column reduction in
reserve **in case atomics dominated the runtime**. They did not — the atomic version shipped and
won. But the other consequence of `tl.atomic_add` on fp32 turned out to matter: float addition is
not associative and the scheduler picks the order row-blocks reach the atomic, so **the same
inputs give different bits on every run**.

Measured on this branch (RTX 3000 Ada, 4 runs of 6 steps, 3×(512,512), fused big path,
max|Δp| / weight scale across runs):

| momentum | atomic reductions | two-pass (`deterministic_reductions=True`) |
|---|---|---|
| float32  | 5.1e-8 | **0** (bit-reproducible) |
| bfloat16 | 3.9e-6 | **0** |
| int8     | 8.1e-6 | **0** |
| 4bit     | 5.1e-8 *(after the requant fix below; 7.0e-4 before)* | **0** |

`_reduce_rowcol_det` / `_reduce_rms_det` store one partial per (tensor, row-block) — an address
only that program writes — and `_reduce_colpart` / `_reduce_rmspart` sum them in a fixed
sequential order. `keep` is left alone: it is an **int32** atomic, and integer addition is
associative and exact whatever the order.

Cost, measured: **0–8% slower** (6→8 launches per bucket) plus an `N * ceil(R/BR) * C` fp32
partial buffer — 1.38→8.77 MiB for a 236×(512,512) bucket, 0.09→32.10 MiB for 2×(4096,4096).
**Default OFF**: for every momentum kind the run-to-run spread is at or below the dtype's own
noise, and the memory is not free. It is exposed as `Adakaon(deterministic_reductions=True)` for
reproducibility work and for debugging a divergence.

### The bug the determinism work surfaced

Two-pass reductions made fp32/bf16/int8 bit-reproducible but **4-bit stayed at 5.1e-4**, which
meant a second nondeterminism source. It was a **race in `_chunked_4bit_apply_batched_g`**: the
requant stored the per-block absmax scales to global memory and then `tl.load`ed them back to
quantize. The storing lane and the reading lanes are different lanes of the same program with no
barrier between, so a lane could divide by the *previous step's* scale. Keeping the scale in
registers (one axis reduction over a `(BLOCK//FBLOCK, FBLOCK)` reshape, then a broadcast) removes
the race and the `BLOCK//FBLOCK`-iteration loop at once, and drops 4-bit's *default-path* spread
from 7.0e-4 to 5.1e-8. Guarded by
`test_chunked_4bit_requant_no_longer_reloads_its_own_scales`.

The same reshape-and-reduce shape is now used by `requant_4bit`'s `EXACT` path (unpadded tiles,
1.19–1.63× on the one-block requant) and by the new `_chunked_int8_apply_batched_g`, which was
written this way from the start and never had the round-trip.

## `momentum_4bit_block` as a runtime scalar (0.7.12)

`_adakaon_tile_kernel` used to hardcode its 4-bit absmax block at `BLK = min(R*C, 128)`, for both
the dequant and the requant. Under any other `momentum_4bit_block` that was not merely "reads the
wrong scales": the requant writes `ceil(numel/128)` floats into an `m_scale` sized for the **real**
block count, i.e. past the end of the buffer (a `(64,128)` weight at `block=0` wrote 63 floats out
of bounds). The fix at the time was a **routing guard** — send those tensors to the native path —
which is safe but expensive: it gives up the fused route entirely for a legitimate configuration.

The block is now a **runtime kernel argument**, and `PointerArrayCache` buckets by
`state["m_block"]` (alongside tile / dtype / device) so every tensor in one launch shares it. Three
consequences:

* **Runtime, not `constexpr`, on purpose.** Triton specializes an integer argument only on `== 1`
  and `% 16 == 0`, so the kernel *body* does not specialize per block. Measured on a **padded**
  tile (`EXACT` off, `FBLK == 0`), blocks 128/64/256/0/32 compile to **one shared variant**. What
  does specialize is `FBLK`, the `EXACT` fast path's `constexpr` block, which must be constant for
  the `(nb, FBLK)` reshape: on an **unpadded** power-of-two tile each distinct block size costs
  **+1** variant of `_adakaon_tile_kernel` (measured +1 for each of 64/256/0/32 on top of 128). A
  training run configures one block size, so in practice that is +0.

  The number to compare across builds is the whole-surface baseline printed by
  `benchmarks/fused/bench_wd_mblock.py --case jit` — every fused route × every momentum storage ×
  fp32/bf16 params. With a clean `TRITON_CACHE_DIR` it is **45 variants across 11 kernels before
  and after**, at a first-step compile cost of 38.6/37.9 s (before) vs 37.2/37.8 s (after): no
  added JIT cost at the defaults. (`cautious_wd="full"` adds 5 to that baseline; `"masked"`, the
  default, adds 0.)
* **`EXACT` generalizes — and its precondition becomes load-bearing.** It now needs the block to
  divide the tile as well as the tile to be unpadded (`(BR*BC) % blk == 0`); `BR*BC` is a power of
  two, so a divisor of it is one too and the reshape stays legal. `FBLK` is the bucket's real block
  instead of a hardcoded 128, so the divisibility that used to be automatic is now something the
  **host** has to enforce: drop that term from `PointerArrayCache.exact4` and a `(64,128)` weight at
  `momentum_4bit_block=96` raises a Triton `CompilationError` mid-run. Guarded by
  `test_exact_requant_rejects_a_block_that_does_not_divide_the_tile` (the probe, showing *why*) and
  `test_4bit_block_that_does_not_divide_the_tile_still_steps` (the host guard, end to end).
* **The `NS` capacity masks stay.** They are no longer the routing decision's backstop (the bucket
  key *is* the block) but they still turn a future routing mistake into dropped stores rather than
  memory corruption. `fourbit_kernel_blocks(numel, block)` computes the required capacity;
  `block <= 0` keeps the legacy `min(numel, 128)` for `AdaPnmCache`, whose tile kernel still
  hardcodes the constant.

**Measured value of the recovered route** (RTX 3000 Ada, paired A/B, 300 one-block tensors,
`benchmarks/fused/bench_wd_mblock.py --case mblock`): 16.2–20.3× faster than the native
degradation, 427/436 kernel launches → **2**, 39.5 MiB of per-step transient → **0**. The worst
case for the new route — a *padded* tile (no `EXACT`, so `requant_4bit`'s general `for b in
range(NB)` loop is O(numel·NB)) at `block=8`, i.e. 900 blocks over a 64×128 tile — is still
**1.7–2.5× faster** than going native (`--case mblock_worst`; 1.73 / 1.77 / 2.00× over three
paired runs here and 2.53× on a second rig — the spread is GPU contention, the sign is not), so no
residual block-size guard is warranted.
