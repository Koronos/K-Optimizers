# Oxford-IIIT Pet diffusion subset

`prepare_pets.py` consumes the upstream `images.tar.gz` and writes a fixed
subset of four classes (`Bengal`, `Siamese`, `beagle`, `shiba_inu`): 24 train,
8 validation, and 8 test images per class (96/32/32 total). Selection is seeded
and sorted by hash, rejects corrupt images and unsafe tar members, and filters
exact duplicates plus near-duplicates using dHash before splitting. This is a
practical deterministic filter, not a complete guarantee against all visual
duplicates. Each image gets a class-based caption, a sidecar `.txt`, and
`manifest.json` with hashes and provenance.

The archive is roughly 800 MB and must be downloaded in full; the script only
extracts selected regular files. The Oxford VGG page identifies the dataset as
[CC BY-SA 4.0](https://robots.ox.ac.uk/~vgg/data/pets/) and notes that image
copyright remains with the original owners. Captions here are generated class
labels, not human captions, so this benchmark can probe class/subject
adherence and memorization, but does not promise broad image quality or
language-grounding performance.

Example (after obtaining the archive separately):

```powershell
python benchmarks/anima/prepare_pets.py path/to/images.tar.gz --output data/pets
```

## Anima pilot in the existing WSL environment

Run from this worktree in WSL. The model paths in `smoke.toml` and the generator
refer to the existing Rengu-Flow installation; adjust them for another machine.
Do not enable `expandable_segments` in this WSL setup.

```bash
export PYTHONPATH="$PWD/src:$PWD"
uv run --no-sync --project /home/koronos/Rengu-Flow deepspeed --num_gpus=1 \
  --module benchmarks.anima.run_seeded --config benchmarks/anima/smoke.toml
uv run --no-sync --project /home/koronos/Rengu-Flow python \
  benchmarks/anima/generate_comparison.py --steps 200 --lr 1e-4 --gap-threshold .007
ANIMA_INIT_SEED=42 uv run --no-sync --project /home/koronos/Rengu-Flow deepspeed \
  --num_gpus=1 --module benchmarks.anima.run_seeded \
  --config tmp/anima-comparison/rakaon_block64.toml
```

Run the other generated TOMLs sequentially on the same GPU. The launcher seeds
at adapter creation, after cache construction, and logs an initial weight SHA-256.
Require identical fingerprints for paired comparisons. Rengu's `train_seed` acts
later and alone cannot ensure identical adapter initialization. The launcher only
instruments the experiment process; it does not edit Rengu-Flow.

The pilot evaluates eight fixed training images and eight validation images at
nine noise quantiles, before training and every 100 steps. `train_eval/loss` is
the training side of the gap; `train/loss` is a noisy minibatch metric. Report raw
`val - train_eval` and its change from step zero, because sample difficulty can
produce a nonzero gap before training. Neither quantity with eight images gives
a precise estimate of population generalization. Test images are untouched;
the split does not establish exclusion from Anima's original pretraining data.

The initial single-LR, single-seed comparison is a screening experiment, not a
tuned optimizer ranking. Nekaon/Adakaon use their configured momentum and decay;
Rakaon is momentum-free. Previews use the same seed at every checkpoint and are
qualitative examples, not FID or a blinded perceptual study.

`summarize_runs.py RUN_DIRECTORY -o results.json` reads observed TensorBoard tags
and `bench_steps.csv`. Pass one actual run directory per argument. Batch times
exclude evaluation/previews; CUDA peaks can include earlier preview allocations.
Compare complete runs, not a merged parent containing several restarts.

The first block64 Pets run (without an adapter-initialization fingerprint) is
retained as a calibration run and must not be treated as a paired comparison.

`run_comparison.py tmp/anima-comparison/manifest.json` runs all generated arms
sequentially and stops on trainer failure. After completion,
`report_comparison.py tmp/anima-comparison/manifest.json` checks initialization
fingerprints, initial evaluation agreement, final evaluation steps and complete
training CSVs before producing the JSON/Markdown report.

## Full-transformer compatibility smoke

`smoke_full.toml` has no adapter and requests `gradient_release=true` plus 24
swapped blocks, batch/accumulation both one. In Rengu this creates one optimizer
per parameter and updates in post-accumulate backward hooks while each block is
on CUDA, then evicts blocks to CPU. This sacrifices cross-parameter batching and
uses CPU RAM/transfer bandwidth to lower VRAM. Text encoder and LLM adapter remain
frozen. The seeded launcher logs the parameter count actually given to optimizers.
Use the same DeepSpeed launch command with this config; it writes a full model
checkpoint under `tmp/anima-full-smoke`, approximately 4 GB. Three steps only test
integration and memory; they cannot establish full fine-tuning convergence.
