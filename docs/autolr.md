# AutoLR — quarantined

`auto_lr=True` is disabled and raises `RuntimeError` during optimizer construction,
before parameters or optimizer state can be modified.

Kaon's DoWG, loss-range-test and continuous Mechanic implementations all passed
useful synthetic checks, but none established a model-agnostic safety boundary.
In real fine-tuning, the gradient-only controller could increase the LR gradually
past the useful region and damage the model around step 100. Mechanic removed the
fixed discovery horizon, but added roughly 6 bytes per trainable bf16 parameter
and still did not reliably outperform a good fixed LR. A controller with a rare
silent destructive failure is not an acceptable production feature.

Use an explicit LR:

```python
from kaon import Adakaon

optimizer = Adakaon(model.parameters(), lr=1e-4)
```

The old arguments remain temporarily in optimizer signatures so existing
configuration files receive an actionable error instead of silently changing
behavior. `auto_lr=False` is the normal zero-overhead path.

Legacy checkpoints containing `_autolr` still load: Kaon discards only that
retired controller blob and restores the base optimizer state. Resume with an
explicit LR appropriate for the workload.

The implementations and benchmark evidence remain recoverable from Git history
and the research branches listed in
[EXPERIMENTS_GRAVEYARD.md](EXPERIMENTS_GRAVEYARD.md). AutoLR should not return to
production until a new design survives multi-seed proxy tests and real low-resolution
fine-tuning, including deliberately low and high seeds, without a load-bearing
step horizon, hidden LR range, or irreversible parameter damage.
