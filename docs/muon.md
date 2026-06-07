# Muon — design & API

> An orthogonalized-momentum optimizer (Newton-Schulz) with an AdamW fallback for
> 1-D / embedding params. Highest convergence quality here, at half of AdamW's
> state.

## Why

Muon orthogonalizes the momentum of each ≥2-D weight via a few Newton-Schulz
iterations before the step, which empirically improves convergence on matrix-shaped
parameters. 1-D parameters (biases, norms) and embeddings don't benefit from
orthogonalization, so Muon **auto-routes** them to a built-in AdamW fallback. The
momentum buffer can be stored in bf16 (`momentum_dtype`), and weights are updated
bf16-correctly via stochastic rounding (no compensation buffer).

`Muon` is a standard `torch.optim.Optimizer` that works one parameter at a time, so
it drops into per-parameter / gradient-release training loops unchanged.

## API

```python
Muon(
    params, lr=2e-2, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0, *,
    adamw_lr=3e-4, adamw_betas=(0.9, 0.999), adamw_eps=1e-8, adamw_weight_decay=0.0,
    bf16_method="stochastic_rounding", momentum_dtype="float32",
)
```

## See also

- [adafusion.md](adafusion.md), [kprodigy.md](kprodigy.md),
  [autofusion.md](autofusion.md) — the other optimizers in `kaon`.
