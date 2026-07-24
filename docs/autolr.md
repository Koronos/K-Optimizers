# AutoLR — autonomous dynamic step size

`auto_lr=True` enables Kaon's built-in, low-VRAM DoWG controller on an optimizer
such as `Adakaon` or `AdaPNM`.

```python
from kaon import Adakaon

optimizer = Adakaon(model.parameters(), auto_lr=True)
```

It is deliberately independent of the trainer: no `report_loss(loss)`, closure,
LR-finder callback, or trainer-side candidate selection is part of the algorithm.
The optimizer observes its own updates and finite gradient norms, starts from a
conservative data-relative scale, and grows geometrically only until it reaches a
stability contact.

At a contact, AutoLR restores the initial trainable parameters, clears the base
optimizer state, invalidates fused buffers where applicable, and restarts its DoWG
accumulators. A comparable second contact freezes the selected LR. If no contact
appears, discovery still ends at the conservative fuse or its fixed step budget.
Non-finite gradients cause one rollback/backoff; repeated non-finite gradients are
diagnosed and skipped rather than ratcheting the LR down indefinitely.

## What AutoLR promises

AutoLR supplies an autonomous, conservative dynamic step size and a bounded
discovery phase. It is not a universal detector of the globally optimal LR: flat or
otherwise uninformative gradient regimes can reach the fuse or budget without
revealing a sharp edge. Validate important workloads as usual and use a fixed LR
when you already have a well-established training recipe.

The controller keeps at most one parameter snapshot during discovery. It snapshots
all trainable parameters from the start, including parameters that receive gradients
only later in training.

## Controls

- `auto_lr=True` enables the controller.
- `auto_lr_d0` optionally requests a positive initial scale. Leave it as `None`
  for the data-relative conservative seed. A request may add bounded compatibility
  headroom (at most 4× the data-relative fuse), but anything higher is clamped
  with a warning; `d0` cannot enlarge or bypass its safety bound without limit.
- `auto_lr_scale` multiplies the discovered scale when a deliberate domain prior is
  available.
- `auto_lr_fuse_rel` sets the conservative relative safety ceiling.

`optimizer.get_d()` returns the currently selected effective LR and
`optimizer.is_frozen()` reports whether discovery has finished. The freeze reason
is retained in the optimizer checkpoint state for diagnostics.

## Integration and checkpointing

Use the ordinary PyTorch loop; the trainer does not participate in AutoLR:

```python
for batch in loader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

`optimizer.report_loss(loss)` remains temporarily as a deprecated compatibility
no-op in 0.7.4. It emits one warning and does not change the AutoLR trajectory; it
will be removed in 0.8.0.

`state_dict()` includes the controller's baseline, rolling window, contacts,
counter, and freeze reason. Checkpoints from 0.7.3 that contain the retired
loss-driven probe state load compatibly; that state is ignored so resumed AutoLR
uses the autonomous controller.
