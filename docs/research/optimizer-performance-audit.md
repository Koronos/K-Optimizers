# Nekaon / Adakaon performance audit

Scope: current experimental worktree, CUDA BF16, 896 real Anima adapter shapes
(34,635,776 parameters), momentum 4bit, beta1=.5, beta2=.999, LR=1e-5, decay=.1,
cautious and gradient centralization enabled. Synthetic fixed gradients, one
parameter group; no forward/backward. Five warmup steps, twenty timed steps,
CUDA synchronization at timing boundaries. CPU profiler observes one later step.
No production optimizer implementation was changed by this audit.

| Variant | Native ms/step | Fused repeat ms/step |
|---|---:|---:|
| Adakaon | 31.974 | 3.607 |
| Nekaon k=0 | 29.823 | 2.862 |
| Nekaon | 32.455 | 5.974 |
| Nekaon, warning disabled for measurement | 30.659 | 4.326 |

These short sequential measurements are diagnostic, not a statistically controlled
ranking. The first fused Adakaon measurement was anomalous (168.659 ms); repeating
the process with compilation caches available yielded 3.607 ms. Its cause was not
isolated, so retain both results rather than claiming a proven compilation cost.
The k=0 differences from identical inner Adakaon also caution against overinterpreting
small absolute timing differences. Real backward reallocates gradient pointers;
this fixed-gradient benchmark does not measure that pointer-table rebuild cost.

## Findings

1. **Default native route was used in previous training.** Adakaon defaults to
   `fused=False`; the Anima configs did not override it. Native foreach recorded
   96 stack operations per step. Gradients, weights and codec/state chunks require
   real temporary copies. These are intentional memory-bounded batches, not an
   identified stale-cache bug. Fused repeat recorded no stacks in Adakaon, and
   only the diagnostic stack in Nekaon. Test fused in full training before changing
   defaults or making time-to-quality claims.
2. **Nekaon's normal warning path synchronizes.** `_warn_if_inert` converts a CUDA
   reduction to Python float, once per visited group in norm='none', for up to 200
   checks (or until it warns). One `_local_scalar_dense` was observed in our single
   group, absent when only the warning was disabled. About 1.6–1.8 ms difference
   was observed; not enough to explain the earlier fourfold total-step anomaly.
   This is enabled diagnostic overhead, not opt-in debugging. A future fix should
   make the check asynchronous or explicitly configurable without changing weight
   updates. Its sample-based wording also overstates complete inactivity.
3. **Native adaptive budgeting queries free CUDA memory each step.**
   `foreach_budget` calls `torch.cuda.mem_get_info` when no explicit budget is set.
   That is intentional OOM protection; its overhead was not isolated here. Do not
   remove it or freeze a large budget without checking memory under full training.
4. **Fused fallbacks need shape-specific care.** Irregular dimensions can take
   the general 4-bit route with dense FP32 scratch. The old scalar-reading
   chunked route is behind an internal fallback toggle, not the default fused
   lone-big route. No new stale-pointer bug was established in supported state
   load/reset paths. Pointer validation must remain in place for safe rebinding.

## Regression checks

The targeted suite (Nekaon, MSAM, climb precision, GC fan-in=1, degenerate fan-in)
on Python 3.13 / torch 2.12: **641 passed, 2 failed, 1 skipped**. Both failures are
the structural opcode-count check `test_plan_witness_scan_stays_c_level`: the
small call records zero opcodes, the large one 110/86. The exact two tests pass on
Python 3.10 / torch 2.13. This suggests a runtime-sensitive tracing test, not proof
of a replaced C-level scan; the environments also differ in torch/pytest versions.
The test helper was not changed or weakened. The skipped wall-clock test is opt-in.

Artifacts: `benchmarks/anima/audit_optimizer_hotpath.py` and
`optimizer_hotpath_results.json`, `optimizer_hotpath_fused_results.json`,
`optimizer_hotpath_fused_repeat_results.json` in that directory.

Conclusion: there are measurable avoidable costs, especially not enabling fused
and the warning's host synchronization. No audit can establish the absence of all
performance traps, and this work does not attribute the old 4x anomaly to Nekaon.
