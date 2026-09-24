# Antikaon — zero-state flat-minima noise for momentum-free Adakaon (experimental)

> **Antikaon** is momentum-free **Adakaon** plus a seeded random perturbation that lives in
> the weights between steps: every gradient is taken at `w = z + xi`, the update lands on the
> clean iterate `z`. On `z` this is exactly Anti-PGD (Orvieto et al. 2022) / single-sample RWP
> (Li et al. 2024), whose implicit objective is `L(z) + ½ Tr(Σ·H(z))`. It costs **no extra
> forward/backward and no per-element state** (`xi` is regenerated from a seed). Whether it
> moves the loss/gap frontier is **not measured yet** — see the plan in
> [`research/antikaon-design.md`](research/antikaon-design.md) §6.

## The update

```
# live weights hold w_n = z_n + xi_n
g_n     = grad L(w_n)                                   # at the perturbed point
row,col <- EMA(g_n²)                                    # Adakaon's factored v
delta_n = clip_rms(P_n g_n) + wd·(w_n − xi_n)           # decay on z, not on w
w_{n+1} = write(w_n − lr·delta_n + xi_{n+1} − xi_n)     # ONE combined weight write
xi      = sigma · S · eps(noise_seed, param index, per-param step)
```

* `sigma = k_sigma · lr · clip_threshold` — the radius is in **optimizer steps** (Nekaon's
  invariant), so it follows the LR and any schedule. The radius used for a given `xi` is frozen
  when it is installed, so a scheduler change between steps cannot desynchronize the removal.
* `S` (`shape="v"`) = `clamp((mean(v̂)/v̂)^¼, 1/s_cap, s_cap)`, a rank-1 product of the factored
  second moment: noise covariance ∝ Adam's preconditioner, regularizer = `Tr(P·H)`.
  `shape="none"` is isotropic (`Tr(H)`).
* `eps` is Rademacher (default) or Gaussian (`noise="gaussian"`). `antithetic=True` pairs
  consecutive draws with opposite signs.
* `sigma_ref="weight"` (ablation): `sigma_i = max(k_weight·RMS_row_i(w), k_sigma·lr·clip)`.

## Usage

```python
from kaon import Antikaon

opt = Antikaon(model.parameters(), lr=1e-4, k_sigma=5.0)   # betas=(0, 0.999) by default

for batch in loader:
    model(batch).backward()
    opt.step(); opt.zero_grad()

opt.eval()                 # live weights -> clean iterate z (validation, sampling, saving)
torch.save({"model": model.state_dict(), "opt": opt.state_dict()}, path)
opt.train()                # re-install the same xi and keep training
```

* **Always validate and checkpoint in eval mode.** The train-mode loss is at the perturbed
  point (higher by ≈ `½Tr(ΣH)`); a train-mode checkpoint is refused on load. `step()` in eval
  mode raises.
* **Resume**: load the model weights first, then `opt.load_state_dict(...)` — it restores the
  noise seed, re-installs `xi` and leaves the optimizer in train mode. Resuming is bit-exact
  with a run that did `eval()`/`train()` at the same step.
* The noise of a parameter is keyed by its index in the flattened `param_groups`; re-ordering
  parameters between runs changes the noise (never the correctness of an eval-mode checkpoint).
* `opt.live_noise(p)` returns the installed `xi` (diagnostics, e.g. `RMS(xi)/RMS(w)`).

## bf16

The combined write goes through Adakaon's own writers, so every `bf16_method` works:

* `"stochastic_rounding"` (default): the combined write is unbiased for every noise law and
  radius (measured, design §3.2). Once `lr·rms(update) ≳ ulp/6` carrying `xi` costs no extra
  rounding noise; in the sub-ulp regime (full fine-tunes at very low LR) it raises the SR walk
  of `z` toward `ulp²/6` per step.
* `"kahan"` (+2 B/param, per-parameter path): keeps the clean iterate exact to ~fp32 — the
  choice for very low LR, where SR would otherwise realize a sub-ulp `xi` only stochastically.
* A radius below half a bf16 ulp at the sampled weight scale triggers a one-time warning (the
  MSAM inert-lookahead heuristic, same cadence).

`eval()`/`train()` use round-to-nearest (like MSAM's climb): error ≤ ½ ulp of `z`, and a ≤ 1 ulp
return only where the subtraction crossed a binade.

## Paths and limits

* Per-parameter and native foreach paths; bit-exact with each other on fp32 CPU params (on
  CUDA they agree to Adakaon's own ~1-ulp foreach/per-param reduction parity).
* **No Triton fused path yet**: `fused=True` warns and runs the foreach path (same math and
  state). The noise is defined in one place (`Antikaon._noise`) so a fused `NOISE` branch can
  reproduce it; it will need its own noise-backend id (recorded in the checkpoint).
* The noise draw is one `torch.Generator` call per parameter per step (two: `xi_n` and
  `xi_{n+1}`), plus up to three transient full-size fp32 tensors per parameter/chunk.
  Performance has not been measured.

Control-battery arms (registry entry `Antikaon`): `k_sigma ∈ {1.5, 5, 15}`, `shape="none"`,
`antithetic=True`, `sigma_ref="weight"` — the design's B1–B3 / C1–C3.
