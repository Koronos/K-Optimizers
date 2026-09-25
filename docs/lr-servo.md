# LR Servo (experimental)

`lr_servo=True` is a bounded local learning-rate corrector for Kaon optimizers
that already keep momentum. It is available on Adakaon (with `beta1 > 0`),
Nekaon, and Lion:

```python
optimizer = Nekaon(model.parameters(), lr=1e-4, lr_servo=True)
```

The supplied LR remains the prior. Every four steps the controller measures the
cosine between the current gradient and the preceding momentum direction. It
smooths that signal and adjusts LR multiplicatively in log space. The controller
keeps a positive alignment reserve instead of deliberately reaching an
oscillatory stability edge.

The trust region is fixed at `0.25x` to `4x` of each parameter group's initial
LR. This is intentional: the servo can correct a nearby LR, but cannot safely
discover one from an arbitrary seed or undo a catastrophic first step. It does
not consume loss, closures, trainer feedback, a step horizon, or a model
snapshot. `auto_lr=True` and `lr_servo=True` are mutually exclusive.

Persistent memory is a handful of scalars per parameter group and no tensor per
parameter. The sampled reduction uses bounded transient fp32 chunks. In the
current prototype, the many-small-tensor LoRA microbenchmark adds roughly
27--31% to the optimizer-only step; a proxy including forward/backward measured
about 3--13% median overhead. A future fused reduction is required before this
should become the default.

`optimizer.get_lr_servo_scale()` returns the current multiplier for the first
parameter group. Checkpoints preserve the controller exactly. When enabling the
servo while loading an older checkpoint, its stored LR becomes the new 1.0x
prior.
