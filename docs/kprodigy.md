# KProdigy — design & API

> A memory-efficient **Prodigy** (parameter-free D-adaptation): train at `lr=1.0`
> and the optimizer finds the effective LR itself. Matches reference Prodigy
> bit-for-bit at its defaults, then adds the kaon memory toolkit.

## Why

[Prodigy](https://arxiv.org/abs/2306.06101) estimates the distance `D` to the
solution on the fly and uses it as the effective learning rate — no LR to tune, no
schedule. The catch in practice is memory (reference Prodigy keeps *four* fp32
buffers — double AdamW) and fragile defaults. `KProdigy` keeps the exact
D-estimation math but:

- stores the first moment in **bf16 / int8 / 4bit** and (optionally) the second
  moment **factored** (Adafactor row+col), with **stochastic-rounding** bf16
  weight updates — the same toolkit as `Adakaon`, so D-adaptation no longer costs
  more memory than AdamW. Its pass-2 update runs on the shared, cached
  bucketing/view plan ([foreach-batching.md](foreach-batching.md)); pass 1 (the
  global D reduction and the two `d`-scaled EMAs) keeps its own bucketing, because
  that reduction spans param groups. Note that `second_moment="factored"` with
  `factor_conv_as_matrix=False` keeps an **N-D** Adafactor pair, which the batched
  bucket cannot work in, so those conv kernels take the per-parameter path;
- ships **sane defaults** (`d_update_freq=1`, `use_bias_correction=False`). The
  original research repo defaulted these the other way and it *starved the
  D-bootstrap* — the effective LR failed to rise. See
  `benchmarks/bench_kprodigy_d.py` for the measured trajectories;
- gives each param group its **own D** (`independent_d`, auto-on for >1 group) so
  on SDXL the UNet and Text Encoder don't burn each other's learning rate.

`KProdigy` needs a global reduction over all parameters each step (the D estimate),
so it is a normal two-pass `step()` optimizer (no gradient-release).

> Status: the full-precision path (bf16 momentum + full fp32 second moment)
> reproduces reference Prodigy to ~1e-4 on D and is the recommended default.
> `second_moment="factored"` is experimental (it inflates D somewhat — measure on
> your model first). With bf16 weights, keep `bf16_method="stochastic_rounding"`:
> at `d0=1e-6` the early updates are tiny and naive bf16 rounding truncates them to
> zero, stalling the D-bootstrap.

## API

```python
KProdigy(
    params, lr=1.0, betas=(0.9, 0.999), beta3=None, eps=1e-8, weight_decay=0.0, *,
    decouple=True,
    use_bias_correction=False,          # keep off (the repo's True default hurt D)
    safeguard_warmup=False, d0=1e-6, d_coef=1.0, growth_rate=float("inf"),
    d_update_freq=1,                    # keep 1 (>1 starves the D-bootstrap)
    slice_p=1,                          # 11 -> ~11x less D-state, ~0.3% D error
    independent_d=None,                 # None -> auto: on when >1 param group
    momentum_dtype="bfloat16",          # "float32" | "bfloat16" | "int8"
    second_moment="full",               # "full" | "factored" (experimental)
    eps_factored=1e-30,
    bf16_method="stochastic_rounding",  # "stochastic_rounding" | "kahan" | "none"
    factor_conv_as_matrix=True,
)
```

## Checkpointing

`torch.save(opt.state_dict())` → `load_state_dict` resumes **bit-exactly**: the
**D estimate** (`d` / `d_numerator` / `d_max` / `d_hat`) rides along in
`param_groups`, and the first moment keeps its configured `momentum_dtype`.
KProdigy overrides `load_state_dict` for the same reason as Adakaon — torch's
default upcasts the quantized momentum to fp32 on load (e.g. `int8` → `fp32`, 4×
the bytes), so it restores the stored dtype instead. (Same `state_dict()`
live-reference caveat as Adakaon — `torch.save` to freeze a snapshot.)

## See also

- [adakaon.md](adakaon.md) — the update engine KProdigy's pass-2 reuses.
- [momentum.md](momentum.md) — the cheap-momentum dial shared with Adakaon.
- [autolr.md](autolr.md) — safety status of the retired shared AutoLR add-on.
