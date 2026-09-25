# Post-Nekaon optimizer search (2026-07-29)

## Question and gate

Search for a fast, memory-efficient optimizer that can move the Kaon constant-LR
loss/generalization frontier toward the control-battery corner:

- held-out loss `<= 0.0700`
- train/held-out gap `<= 0.0070`

The search deliberately used a clean worktree at Kaon `0.7.7` (`856e505`), not the
dirty AutoLR/MoMo prototype checkout.  All runs were sequential on the native-Windows
CUDA environment (RTX 3000 Ada, 8 GB), with constant LR `1.2e-3` and the control
battery's progressive `32/48/64` resolution sequence.

The staged gate was:

1. `C=40, N=600, seed=0` to reject divergence/underfitting cheaply.
2. Repeat promising points with seed 1.
3. `C=40, N=2000, seeds=0,1` to expose the late overfitting that invalidated earlier
   short-horizon ideas.
4. Run `C=128, N=2000` only if a candidate moved the long two-seed frontier.

No candidate passed gate 3, so no C=128 result is claimed.

## Implemented hypotheses

The isolated research harness is [`../new_optimizer_search.py`](../new_optimizer_search.py).
Candidates are intentionally not exported from `kaon` and use a slow fp32/per-parameter
reference path: quality hypotheses were tested before spending work on codecs, foreach,
or Triton.

### 1. TangentDual and controls

- fixed inverse-adaptive blend (`g * sqrt(v)`), as a schedule-free DualAdam control;
- inverse-adaptive direction projected orthogonal to the current gradient;
- instantaneous-vs-momentum residual projected orthogonal to the gradient.

At `C=40/N=600`, none dominated Adakaon or Nekaon.  Increasing tangent strength opened
the gap and/or raised held-out loss.  The first proposed TangentDual mechanism is therefore
falsified on this proxy.

### 2. LookKaon

LookSAM-style periodic *true* SAM probes every `k` steps, caching the orthogonal SAM
component between probes.  This was tested because LookSAM reports SAM-like quality with
periodic second gradients ([CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Towards_Efficient_and_Scalable_Sharpness-Aware_Minimization_CVPR_2022_paper.html)).

The published-scale `alpha=0.5` severely underfit (`test=0.0955`, `gap=0.0026`) even with
`rho=5e-4`.  Reducing alpha to `0.01-0.10` restored loss but produced no point better than
Adakaon.  It also needs one extra backward every `k` steps and a cached gradient-sized
buffer, so it was rejected before long runs.

### 3. Coherence-conditioned momentum

The adaptive direction `u_t` and stored momentum `m_t` define a per-tensor coherence:

```text
c_t = clamp(cos(m_t, u_t), 0, 1)
beta_t = beta_floor + (beta_ceiling - beta_floor) * c_t
m_t <- beta_t * m_t + (1 - beta_t) * u_t
```

This needs no parameter-sized state beyond Adakaon's existing momentum/second moment.
The instantaneous version produced the only long single-seed frontier movement:

| C=40, N=2000, seed 0 | test | gap |
|---|---:|---:|
| Nekaon beta=.7 | 0.074744 | 0.011133 |
| coherence floor=.7 | **0.074556** | **0.008972** |

It did not reproduce as strict dominance in seed 1 (`0.07081/0.01019` versus Nekaon's
`0.07023/0.00964`).  It remains non-dominated on the two-seed mean, but still slides the
frontier rather than crossing the target corner.

### 4. Coherence EMA and latch

To preserve high-momentum fidelity early without knowing the run horizon, coherence was
smoothed with one scalar per tensor, initialized to `1`:

```text
cbar_t = 0.99 * cbar_(t-1) + 0.01 * c_t
beta_t = beta_floor + (0.9 - beta_floor) * cbar_t
```

This improved the instantaneous controller.  A hard irreversible switch at coherence
thresholds `0.10/0.05/0.02` was also tested.  The latch compressed gap further only by
raising loss, so the smooth controller was superior.

### 5. Coherence-guided decay and coherent lookahead

- lower weight decay for coherent tensors and higher decay for incoherent tensors;
- Nekaon-style downhill lookahead driven by coherence-conditioned momentum.

Both looked useful at 600 steps but failed at 2000: selective decay regularized by
underfitting, while the lookahead duplicated the coherence controller's regularization
and was dominated.

## Long two-seed result

Means below combine the exact seed-0/seed-1 JSON results from this directory.

| optimizer / candidate | mean test | mean gap | interpretation |
|---|---:|---:|---|
| Nekaon beta=.9 | **0.070879** | 0.012470 | loss extreme |
| Nekaon beta=.7 | 0.072487 | 0.010385 | existing middle point |
| instantaneous coherence, floor=.7 | 0.072683 | 0.009581 | cheaper gap, slight loss cost |
| coherence EMA, floor=.5, decay=.99 | 0.073224 | 0.009052 | best autonomous coherence point |
| coherence EMA, floor=.6, decay=.99 | 0.073266 | 0.009165 | dominated by floor=.5 |
| selective decay, floor=.6 | 0.073907 | **0.008620** | underfitting regularizer |

None reaches `test<=.0700` and `gap<=.0070`; none dominates Nekaon beta=.9 on loss while
also dominating Nekaon beta=.7 on gap.  The apparent short-run wins were not sufficient
evidence and were correctly rejected by the long gate.

## Cost and production feasibility

The reference candidate reads fp32 momentum and performs Python/per-tensor reductions,
so its measured `~90-98 ms/step` must **not** be compared with Nekaon's optimized
`~64-69 ms/step` in these research runs.

If the coherence mechanism had passed quality gates, it could have reused 4-bit momentum
and factored `v`; persistent overhead would be one scalar per tensor, not per parameter.
The dot product and norms could share the existing fused reduction pass.  Because quality
failed first, no speculative foreach/Triton implementation was made and no speed or
`0.56 B/param` production claim is recorded.

## Decision

- Do **not** add or export a new optimizer from this campaign.
- Archive TangentDual, LookKaon, hard coherence switching, selective decay and coherent
  lookahead as rejected on this proxy.
- Keep the smooth coherence-momentum reference only as a research lead.  It is the most
  efficient loss-for-gap exchange found, but it is not the requested solution.
- Do not run or publish C=128/Rengu results until a mechanism first passes the long
  two-seed proxy gate.

