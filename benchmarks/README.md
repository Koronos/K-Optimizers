# Benchmarks

```
benchmarks/
├── proxy/                  # shared, optimizer-AGNOSTIC proxy (used by everything)
│   ├── dataset.py          #   the registered deterministic synthetic dataset (train=32/test=96)
│   └── harness.py          #   pixel-DDPM U-Net + DDPM loss + the LoRA-like adapter-bag speed probe
├── control/                # the cross-optimizer CONTROL BATTERY  ← start here
│   ├── registry.py         #   every optimizer + its best config (add a contender here)
│   ├── battery.py          #   runs the battery, caches results, regenerates the rankings
│   ├── results.json        #   the per-optimizer cache (git-tracked so rankings are reproducible)
│   └── RANKINGS.md         #   the generated ranked tables  ← read here
└── adamuon/                # optimizer-specific deep-dives (historical campaign)
    ├── pixel_ddpm_ab.py     #   AdaMuon-vs-Adakaon convergence A/B (CLI, presets)
    ├── sdxl_lora_ab.py      #   real SDXL LoRA A/B (adamw_fused / adafactor arms)
    └── RESULTS_*.md         #   the AdaMuon campaign write-ups
```

## The control battery

One reproducible suite that scores **every** optimizer at its best config across the dimensions
we actually care about, and emits easy-to-read **rankings** ([`control/RANKINGS.md`](control/RANKINGS.md)):

| dimension | what it measures |
|---|---|
| ⚡ per-iteration speed | `ms/step` on the C=128 U-Net (full-FT-like) **and** on a 512-tiny-tensor adapter bag (LoRA-like, launch-bound — where `foreach` pays off) |
| ⏱️ convergence speed | steps to reach a common held-out quality target |
| ⏱️ time × quality | wall-clock to that target = ms/step × steps |
| 🎯 loss × generalization | final held-out loss **and the train–val gap** (the real objective for small-data fine-tuning) |
| 💾 memory | measured optimizer-state bytes/param |
| 🔁 continuity | the train–val gap at **constant LR** (no schedule → resumable) and its change vs the scheduled gap |

Everything runs on the reproducible multi-resolution proxy (`512/768/1024` ≙ `32²/48²/64²`) with
the REX d=0.9 + progressive-resolution recipe — the same setup that stands in for a full
fine-tune (wide U-Net) and a LoRA (the adapter bag).

### Run it

```bash
python benchmarks/control/battery.py            # measure every registry optimizer, refresh rankings
python benchmarks/control/battery.py --new      # measure only optimizers missing from the cache
python benchmarks/control/battery.py --render-only   # rebuild RANKINGS.md from the cache (no training)
python benchmarks/control/battery.py --quick    # smaller/faster settings (smoke; separate cache)
```

### Add a contender (the whole point)

1. Add one entry to [`control/registry.py`](control/registry.py) — its best config + the LR it
   wants at the proxy scale.
2. Run **just it**:
   ```bash
   python benchmarks/control/battery.py --only YourOptimizer
   ```
   Its data is measured, merged into `results.json`, and **every ranking is regenerated against
   the whole field** — you never re-measure the others.

Only entries measured at the *same* settings signature (`C`/`N`/seeds/dataset-fingerprint) are
ranked together; the battery flags any stale entries to re-run.

## Caveats (read before trusting a ranking)

- **Proxy LRs are ~100× real-training LRs.** They are relative knobs for the synthetic benchmark,
  not recommendations for your data — always re-tune `lr` on a real run.
- The metrics rank **objective overfitting / convergence**, not **perceptual fidelity**. The proxy
  once ranked a Muon-style optimizer above the real visual-A/B winner. Treat the rankings as a
  strong signal and a regression guard, **not** a settled verdict — confirm on a real LoRA with
  FID/KID + a live `val/gap` metric.
- GPU non-determinism (cudnn) is ~0.001 on these losses; differences below that are noise.
