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

## How to resume
Implement the three kernel changes above on this branch behind a `self._fused_reductions` toggle
(default False until it beats 5.4 ms with parity, then True), add parity tests mirroring the #1
batched tests, and run `bench_fused.py --regime big` + the battery before/after. If it lands a
measured win with parity, merge into `feat/fused-extra-kernels`.
