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
* **Resume — load the optimizer AFTER the model.** `opt.load_state_dict(...)` restores the
  noise seed and ends by re-installing `xi` on top of whatever the weights are at that moment,
  leaving the optimizer in train mode. Loading the model afterwards would overwrite the
  re-installed noise (the next step would then remove an `xi` that is not there). Resuming is
  bit-exact with a run that did `eval()`/`train()` at the same step.
* The checkpoint records the noise backend including the device type (`"torch-cuda"`,
  `"torch-cpu"`): a torch.Generator stream differs per device, so resuming on another device
  warns (training stays correct — the checkpoint holds `z` — but the noise sequence differs).
* The noise law (`k_sigma`, `shape`, `noise`, `antithetic`, `sigma_ref`, `k_weight`, `s_cap`)
  is **read-only** after construction: `xi` is never stored, its removal regenerates it, so it
  must be removed with the law that installed it. The radius of each installed `xi` is frozen
  per parameter, so LR schedules, `grad=None` steps and new param groups are safe.
* The noise of a parameter is keyed by its index in the flattened `param_groups`; re-ordering
  parameters between runs changes the noise (never the correctness of an eval-mode checkpoint).
* `opt.live_noise(p)` returns the installed `xi` (diagnostics, e.g. `RMS(xi)/RMS(w)`).

## bf16

The combined write goes through Adakaon's own writers, so every `bf16_method` works:

* `"stochastic_rounding"` (default): the combined write is unbiased for every noise law and
  radius (measured, design §3.2). Once `lr·rms(update) ≳ ulp/6` carrying `xi` costs no extra
  rounding noise; in the sub-ulp regime (full fine-tunes at very low LR) it raises the SR walk
  of `z` toward `ulp²/6` per step.
* `"kahan8"` (+1 B/param, every path; the recommended choice for low LR): the compact Kahan of
  `docs/research/compact-kahan.md` — the bf16 weight plus one `uint8` residual in units of its
  own ulp, stochastically rounded at `ulp/256`. The clean iterate `z = decode(p, lo) − xi`
  follows an fp32 run of the same rule (same `xi`, same gradients) to the residual grid:
  measured at lr 1e-5 on ~0.05 weights (step ≈ 0.04 ulp, `xi` ≈ 0.06–0.2 ulp), 300 steps,
  weight decay 0.1 — RMS distance **0.020 ulp with `kahan8` against 2.6–3.9 ulp with SR**
  (130–200×), per-parameter and foreach alike, while the fp32 run moved up to 21 ulp
  (`tests/test_antikaon.py::test_kahan8_low_lr_clean_iterate_tracks_fp32_far_better_than_sr`).
* `"kahan"` (legacy, +2 B/param, per-parameter path only): a bf16 compensation buffer; exact on
  dithered updates but it rounds deterministically, so it loses part of pure sub-grid drift
  (compact-kahan note). Kept for compatibility.
* **Weight decay and `sigma_ref="weight"` read the compensated value.** The decay term is
  `wd · (w − xi_n)` with `w = p + shift` (`kahan`) or `decode(p, lo)` (`kahan8`), not the bare
  bf16 weight (reading the bare weight misses `lr·wd·residual` every step: 12–26 residual units
  after 20 steps of strong decay, against 1–3 with the fix). Same for the row RMS of
  `sigma_ref="weight"`.
* **Inert-noise warning.** A radius below half the *resolution the stored weight keeps* at the
  sampled weight scale triggers a one-time warning (the MSAM inert-lookahead heuristic, same
  cadence). The resolution is the bf16 ulp for SR / `none`, and `ulp/256` for the compensated
  methods: with `kahan8` (or `kahan`) a sub-ulp `xi` is **not** inert — it is kept in the
  residual and the clean iterate moves exactly as designed. So `lr=1e-5, k_sigma=5` on ~0.05
  weights warns under SR and not under `kahan8`; the compensated methods warn only once the
  radius falls under `ulp/512`. What the compensation cannot change is that the forward pass
  sees the nearest bf16 to `z + xi`: a sub-ulp `xi` reaches the *gradient* only through the
  coordinates it tips across a rounding boundary. That is a property of evaluating a bf16 model;
  whether the resulting perturbation still helps quality at real fine-tune LRs is what
  `benchmarks/antikaon_lowlr/` measures.

`eval()`/`train()` use round-to-nearest (like MSAM's climb): error ≤ ½ ulp of `z`, and a ≤ 1 ulp
return only where the subtraction crossed a binade. Under `bf16_method="kahan"` they act on the
compensated value `p + shift` and keep the rounding residual in `shift`, so the eval view is
`z` to ~fp32 and repeated eval/train cycles do not move the clean weight (measured with
`xi ≫ ulp ≫ step`: eval-view error 0.8 % of a step, 20-cycle drift 0.08 % of a step, against
490 % / 11 % for a plain RTN round trip). Under `"kahan8"` they decode, add/subtract `xi` in
fp32 and re-encode with the residual's stochastic rounding (the same codec as the writer, noise
from the optimizer's checkpointed stream, so resumes stay bit-exact): the eval view is `z` to the
residual grid, the forward pass sees the nearest bf16, and 20 cycles leave only the unbiased
`ulp/256` walk (same regime as above: eval-view error ≤ 24 % of a step at the worst coordinate,
20-cycle drift RMS 3 % of a step — coarser than legacy `kahan`'s ~fp32 buffer on this dithered
round trip, and far below the step; a plain RTN round trip gives 490 % / 11 %). The read/write of the clean value lives in
`Antikaon._read_clean` / `_write_clean` (and the stacked `_clean_stack`).

## Paths and limits

* Per-parameter and native foreach paths; bit-exact with each other on fp32 CPU params (on
  CUDA they agree to Adakaon's own ~1-ulp foreach/per-param reduction parity).
* **No Triton fused path yet**: `fused=True` warns and runs the foreach path (same math and
  state; with `kahan8` the foreach bucket write carries the residual views). The noise is defined in one place (`Antikaon._noise`) so a fused `NOISE` branch can
  reproduce it; it will need its own noise-backend id (recorded in the checkpoint).
* `sigma_ref="weight"` reads the row RMS of the clean iterate `z` when a noise is installed.
* **Pending (performance):** the noise draw is a Python loop of `torch.Generator` calls, two
  per parameter per step (`xi_n` and `xi_{n+1}`), even on the foreach path, plus up to three
  transient full-size fp32 tensors per parameter/chunk. On bags of hundreds of small tensors
  (LoRA) this loop may dominate the step. Not measured; the fused `NOISE` branch is the fix.

Control-battery arms (registry entry `Antikaon`): `k_sigma ∈ {1.5, 5, 15}`, `shape="none"`,
`antithetic=True`, `sigma_ref="weight"` — the design's B1–B3 / C1–C3.
