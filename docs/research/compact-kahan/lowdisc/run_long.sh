#!/usr/bin/env bash
# Long-horizon follow-up (100k steps, D = 2**16) for the candidates, CPU only.
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=-1
export PYTHONPATH=../../../../src
PY=${PY:-/c/Users/Koronos/Documents/Repos/Rengu-Flow/.venv/Scripts/python.exe}
mkdir -p results_long logs
"$PY" sim_lowdisc.py --d 65536 --n 100000 --threads 2 --out results_long \
  --schemes sr,ld-phi,ld-r2,kahan8-sr,kahan8-rn,kahan8-ld,kahan8-ld-r2,kahan4-ld \
  --checkpoints 1000,3000,10000,30000,100000 \
  --regimes 'coh-lr1e-06,coh-lr1e-07,adam-lr1e-06,alias-P2-lr1e-06,alias-P3-lr1e-06,alias-P7-lr1e-06,orig-drift-lr1e-06,orig-lr1e-05,noisy-mu0.0005-s0.005'
