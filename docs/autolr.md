# AutoLR — continuous autonomous step size

`auto_lr=True` enables Kaon's built-in Mechanic controller on optimizers that use
the shared AutoLR mixin, including Adakaon, Nekaon, Lion, and AdaPNM.

```python
from kaon import Nekaon

optimizer = Nekaon(model.parameters(), auto_lr=True)
```

The trainer does not participate: there is no loss callback, closure requirement,
LR sweep, scheduler, candidate list, or step budget. The optimizer starts from a
fixed absolute low seed and continuously adapts one global step size from gradients and its
anchored parameter trajectory using six discounted Mechanic bettors.

Unlike the retired 0.7.4 DoWG probe, the controller has no geometric ramp, contact
detector, rollback phase, fuse, or arbitrary 192-step horizon. It remains adaptive;
on the tested nonlinear proxies its scale settles near the useful fixed-LR region
and can still respond if the training regime changes.

## Memory and optimizer composition

The persistent tensor overhead is one native-dtype anchor plus one fp32 normalized
trajectory for every trainable parameter (about 6 B/parameter for bf16 weights, or
8 B/parameter for fp32 weights). Steps use bounded working buffers for shape-compatible
small tensors. A tensor larger than the internal batching budget takes an unstacked path,
so batching never creates several full-size stacked copies of a large weight. The explicit
fp32 trajectory is required for correctness:
the safe `1e-6` seed is below one bf16 ULP for many weights, so reading displacement
back from the materialized model would silently pin the controller at its seed.

This cost applies only to parameters owned by the optimizer (for example, adapter
weights in LoRA/LoKr), but it makes AutoLR inappropriate when its extra trajectory
does not fit the training VRAM budget.

Nekaon and MSAM keep a live lookahead view between steps. Their internal protocol
removes the previous lookahead before Mechanic measures the true iterate, removes
the virtual unit-scale lookahead after the base step, reconstructs the scaled true
iterate, and reapplies a lookahead with exactly that same scale. This protocol is
covered on native and fused/Triton paths.

## Compatibility controls

- `auto_lr=True` enables continuous Mechanic.
- `auto_lr_scale` is an optional explicit multiplier; leave it at `1.0` for the
  autonomous path.
- `auto_lr_d0` is deprecated and ignored. High and low values produce the same
  trajectory, so an old configuration cannot accidentally start hot.
- `auto_lr_fuse_rel` is accepted for source compatibility but no longer caps or
  freezes the controller.

`optimizer.get_d()` returns the effective step size. `optimizer.is_frozen()` is
always false because Mechanic is continuous.

`optimizer.report_loss(loss)` remains a deprecated warning-once no-op and will be
removed in 0.8.0.

The controller assumes one replicated optimizer trajectory. Ordinary data-parallel
training is compatible when gradients and parameters are synchronized before `step()`.
Sharded optimizer/parameter schemes that give each rank only part of the global
gradient-trajectory inner product need an explicit collective reduction and are not
currently supported by AutoLR.

## Checkpoints

The public `AutoLRMixin.state_dict()` contract stores the anchor, fp32 trajectory, and all
six bettor accumulators alongside the host optimizer state; custom hosts that override
serialization must delegate through the mixin.
A 0.7.5 Mechanic checkpoint resumes exactly, including Nekaon's first live
forward/backward after returning from the required eval-view checkpoint. Checkpoints
made by the retired 0.7.4 DoWG controller
fail closed instead of guessing a migration between different algorithms; resume
from an unadapted model checkpoint when switching controller generations.

AutoLR is still an optimizer, not an oracle. No gradient-only method can identify a
finite optimum on an unbounded linear objective. Release validation therefore uses
paired nonlinear multi-seed batteries and real low-resolution training smokes,
rather than a fixed number of discovery steps.
