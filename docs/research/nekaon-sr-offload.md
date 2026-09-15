# BF16 stochastic lookahead with host restoration

Experimental implementation: `benchmarks/nekaon_sr_offload.py`. Not a released
optimizer or a demonstrated quality improvement.

The ordinary BF16 lookahead can round small perturbations to zero. This experiment
uses stochastic rounding for the lookahead write and saves the exact original
BF16 weights in pinned host RAM. Restoration copies those original bits back;
it does not subtract a second randomly rounded perturbation. Eval/train cycles
replay the same private random stream so validation does not change the training
view. There is no persistent FP32 GPU weight copy. Temporary FP32 arithmetic
still exists, as in the base optimizer.

For the Anima rank-16 adapter's 34,635,776 optimizer parameters, the host snapshot
uses 69,271,552 bytes (66.063 MiB). Full fine-tuning scales this cost with the number
of optimized parameters and has not been validated with this implementation.

## Integration and verification

The first three-step 1024px smoke trained successfully but failed when saving:
Rengu's resume checkpoint writer did not enter the optimizer's true-weight view.
The seeded experiment launcher now wraps the entire checkpoint save in that
view, restoring the live view in `finally`. This hook applies only to the
experimental class and does not modify the Rengu checkout.

Seven unit tests passed for CPU/CUDA sub-ULP activity, repeated view changes,
checkpoint continuation, and checkpoint view restoration on success/error.
Two subsequent real 1024px diagnostic runs both completed and saved step 2:

| Variant | Step 1 / 2 training seconds | Step 1 / 2 optimizer seconds | Peak CUDA GiB |
| --- | --- | --- | --- |
| Nekaon RNE | 25.574 / 25.603 | 1.382 / 1.723 | 5.90 |
| SR + host snapshot | 25.321 / 26.663 | 1.713 / 2.793 | 5.90 |

These are two-step diagnostic observations, not a throughput benchmark. Explicit
CUDA synchronization surrounded optimizer timing. Runs were sequential and used
the same seed 45, BF16 Anima rank 16, fixture images at 1024px, and LR 1e-5.
Logs: `tmp/anima-sr1024-rne-diagnostic.log` and
`tmp/anima-sr1024-host-diagnostic.log`. The earlier failed smoke was much slower
(82–90 seconds per step); its cause was not established, so it is not used to
estimate overhead.

## Quality experiment launched

`tmp/anima-sr1024/manifest.json` specifies three sequential arms: Nekaon k=0,
ordinary Nekaon, and stochastic lookahead with exact host restoration. All use
seed 45, constant LR 1e-5, 100 steps, 1024px Oxford Pets images, and identical
initialization. Fixed train/validation evaluations use eight images per split
and nine noise quantiles at steps 0, 50, and 100. The test split is untouched.
The optimizer timing synchronization hook is disabled for these quality runs.

The quality experiment has now completed; see `nekaon-lookahead-timing.md` and
`benchmarks/anima/sr1024_results.json`. No useful quality improvement was observed.
Stochastic activity alone is not evidence of
useful lookahead: rare BF16-sized jumps may add noise. Compare validation loss,
absolute train/validation gap, and active training time before selecting this
approach. This small, single-seed screen cannot establish perceptual detail
retention, long-run convergence, or superiority across diffusion models.
