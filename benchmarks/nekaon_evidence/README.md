# Nekaon evidence package

Three measurements and one consolidator. The question is *"does Nekaon beat Adakaon and
AdamW on speed, quality and memory for diffusion fine-tuning?"*, and no single script can
answer it: quality needs a model that trains, speed needs both the optimizer in isolation
and the step it lives inside, and memory needs bytes-per-param, peak VRAM and the
capacity ceiling. Each script answers one part; `report.py` puts them next to each other
and applies the verdict rules.

| Script | Answers | Runs on |
| --- | --- | --- |
| [`anima/campaign.py`](anima/campaign.py) + [`anima/aggregate.py`](anima/aggregate.py) | Quality (final val eps-MSE, raw train/val gap), plus real-model ms/step, active seconds and peak VRAM | GPU, **from WSL** (the Rengu-Flow trainer lives there) |
| [`step_cost.py`](step_cost.py) | The optimizer step alone at SDXL parameter scale: paired time ratios, state B/p, peak/reserved VRAM, capacity under this GPU, and the optimizer's share of a full fwd+bwd+step | GPU, Windows |
| [`../control/battery.py`](../control/battery.py) | The 5-seed control battery on the proxy diffusion task, with `per_seed` runs retained so every comparison is paired | GPU, Windows |
| [`report.py`](report.py) | Consolidates all three into `EVIDENCE.md` + `evidence.json` with a verdict per axis and per comparison | CPU, anywhere |

`step_cost.py` is run more than once, and the runs are not interchangeable: the paired
ratios in one file are all taken against that file's `--baseline`, so measuring Nekaon
against Adakaon needs its own run with `--baseline adakaon_4bit`. The finished runs live in
[`results/`](results/) and are read by name — see phase 4.

## Order — strictly serial on the GPU

There is one GPU and it is power limited (60 W on AC, 35 W on battery). Two measurements
running at once contaminate each other's timings, and a run started on battery is not
comparable with one started on AC — `step_cost.py` refuses outright, and `aggregate.py`
reports mixed signatures as *not comparable*. So:

1. **Plug in the laptop.** Everything below refuses or self-flags otherwise.
2. `step_cost.py` (Windows) — the longest single run; nothing else may touch the GPU.
3. `battery.py --out evidence` (Windows) — again alone.
4. The Anima campaign (WSL) — again alone, and it is the longest of the three overall.
5. `report.py` — CPU only, so it may run while anything else is running.

Steps 2–4 can be reordered, but never overlapped. Step 5 needs whichever of them have
finished; it renders a "not measured" section for the rest, so a partial report is a
legitimate intermediate artifact.

## Phase 1 — optimizer step cost at SDXL scale (Windows)

Allocates a tensor-shape census of an SDXL-like UNet (`R=5` ≈ 663 M params) and steps
each optimizer on it. Nothing here trains anything; it is cost only.

```powershell
$env:PYTHONPATH = "src"
python benchmarks/nekaon_evidence/step_cost.py `
    --out benchmarks/nekaon_evidence/step_cost.json `
    --markdown benchmarks/nekaon_evidence/STEP_COST.md `
    --reps 60 --paired-R 3 --fraction --capacity
```

* `--fraction` adds level (c): the optimizer's share of a real fwd+bwd+step on the proxy
  UNet. Without it the report cannot state the ceiling on any end-to-end speed claim.
* `--capacity` adds the max-parameter sweep. It is a coarse sweep in `R` and it OOMs on
  purpose, repeatedly; expect it to be slow. On Windows/WDDM it may never OOM at all — see
  *What this machine cannot measure* below.
* `--render-only <json>` rebuilds the markdown from a finished run without measuring.

Two further runs feed the consolidator, and both are cheap next to the one above:

```powershell
# Nekaon against Adakaon directly — the only file with that pairing.
python benchmarks/nekaon_evidence/step_cost.py `
    --out benchmarks/nekaon_evidence/results/step_cost_R3_vs_adakaon.json `
    --R 3 --reps 150 --baseline adakaon_4bit `
    --arms adakaon_4bit nekaon_4bit nekaon_k0_4bit nekaon_bf16 adakaon_bf16

# The reference (non-Triton) kernels, solo timings only.
python benchmarks/nekaon_evidence/step_cost.py `
    --out benchmarks/nekaon_evidence/results/step_cost_R5_native.json `
    --R 5 --reps 100 --native --skip-paired
```

## Phase 2 — control battery, 5 seeds (Windows)

```powershell
$env:PYTHONPATH = "src"
python benchmarks/control/battery.py --out evidence --seeds 5
```

Writes `benchmarks/control/results_evidence.json` and `RANKINGS_evidence.md`, entirely on
the side: `--out evidence` never touches the historical `results.json`. The `--out evidence`
tag is what `report.py` expects, and the 5 seeds are what make the paired intervals
possible — a 2-seed run yields a Student-t interval too wide to say anything.

## Phase 3 — the Anima campaign (WSL)

The trainer is Rengu-Flow and it runs under WSL. From a WSL shell, with the Pets subset
already staged:

```bash
cd /mnt/c/Users/Koronos/Documents/Repos/K-Optimizers
python benchmarks/nekaon_evidence/anima/campaign.py \
    --root tmp/nekaon-evidence-anima \
    --pets-root /tmp/pets/subset \
    --steps 200
```

21 runs: 9 in phase A (the LR screen, one seed, three LRs, three arms) and 12 in phase B
(four held-out seeds x three arms at the selected LRs). The driver is resumable — rerun the
same command after an interruption and it skips what is already on disk — and it refuses
to resolve phase B's learning rates until phase A is complete.

Then aggregate, still from WSL or from Windows (it only reads JSON):

```bash
python benchmarks/nekaon_evidence/anima/aggregate.py --root tmp/nekaon-evidence-anima
```

which writes `benchmarks/nekaon_evidence/anima/results.json` and `RESULTS.md` beside
itself (`--json-out` / `--md-out` move them).

## Phase 4 — consolidate (CPU, anywhere)

This is the command that produced the committed `EVIDENCE.md` / `evidence.json`:

```powershell
$env:PYTHONPATH = "src"; $env:CUDA_VISIBLE_DEVICES = "-1"
python benchmarks/nekaon_evidence/report.py `
    --anima                 benchmarks/nekaon_evidence/anima/results.json `
    --step-cost             benchmarks/nekaon_evidence/results/step_cost_R5.json `
    --step-cost-vs-adakaon  benchmarks/nekaon_evidence/results/step_cost_R3_vs_adakaon.json `
    --step-cost-native      benchmarks/nekaon_evidence/results/step_cost_R5_native.json `
    --battery               benchmarks/control/results_evidence.json `
    --out                   benchmarks/nekaon_evidence/EVIDENCE.md `
    --json-out              benchmarks/nekaon_evidence/evidence.json
```

Every source flag is optional. A source that is absent produces a `no evidence` axis and
a header row saying why — it is never silently skipped, and never rendered as neutral.

* `--step-cost-vs-adakaon` takes a second `step_cost.py` run whose `--baseline` is
  `adakaon_4bit`. It is the only file that times Nekaon **directly** against Adakaon at
  SDXL scale, in one process, so with it the "Nekaon vs Adakaon, optimizer step alone" cell
  becomes a real paired ratio with an interval. Without it the cell falls back to a
  ratio-of-ratios through the shared `adamw_bf16` baseline, which has no interval and
  earns no verdict.
* `--step-cost-native` takes a `--native --skip-paired` run. It is printed as its own
  unpaired table, labelled as such, and contributes no findings and no verdicts.

### What this machine cannot measure

The GPU here is driven through **Windows WDDM**, whose CUDA allocator oversubscribes into
host RAM rather than raising `CUDA out of memory`. Two consequences are applied by rule,
not by hand:

* a `--capacity` sweep in which **no arm OOMs** has found the top of the sweep, not a fit
  limit, so memory level (c) becomes `no evidence` with that note attached;
* any timing row whose arm or baseline peaked above `meta.gpu_total_bytes` is marked
  **`invalid (oversubscribed)`** and enters no verdict — it timed the PCIe bus. Peak-VRAM
  rows above that line are still printed, labelled `exceeds VRAM: spilled to host`.

The "Measurement integrity" section of `EVIDENCE.md` lists every arm of every step-cost
file against that rule, including the ones it did not touch.

### What the report will and will not say

It emits one of seven words per cell, each from a stated rule: **wins**, **loses**,
**n.s.**, **tie by construction**, **no evidence**, **invalid (oversubscribed)**,
**mixed**. An interval that covers zero is `n.s.`
and is printed as prominently as a win. The speed axis is split into three levels that are
never merged into one sentence — the optimizer alone, the full step, and the optimizer's
share of that step — and the memory axis likewise into B/p, peak VRAM and capacity.

It will not say anything about perceptual quality. There is no FID and no KID anywhere in
this package; the quality axis is objective loss on a proxy and on one small LoRA.

## Tests

All CPU, no GPU, no measurement needed:

```powershell
$env:PYTHONPATH = "src"; $env:CUDA_VISIBLE_DEVICES = "-1"
python -m pytest tests/test_nekaon_evidence_report.py `
                 tests/test_nekaon_evidence_anima.py `
                 tests/test_nekaon_evidence_step_cost.py -q
```
