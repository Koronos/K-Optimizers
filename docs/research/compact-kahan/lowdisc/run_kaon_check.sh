#!/usr/bin/env bash
# The shipped kahan8ld noise (kaon._compact_kahan.ld_noise) in the long-horizon harness.
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=-1
export PYTHONPATH=../../../../src
PY=${PY:-/c/Users/Koronos/Documents/Repos/Rengu-Flow/.venv/Scripts/python.exe}
mkdir -p results_kaon logs
for reg in alias-P2-lr1e-06 alias-P3-lr1e-06 alias-P7-lr1e-06 coh-lr1e-06 adam-lr1e-06 orig-lr1e-05 \
           noisy-mu0.0005-s0.005; do
  "$PY" sim_lowdisc.py --d 65536 --n 100000 --threads 2 --out results_kaon \
    --schemes kahan8-sr,kahan8-ld-b256,kahan8-ld-kaon \
    --checkpoints 1000,3000,10000,30000,100000 --regimes "$reg" > "logs/kaon-$reg.log" 2>&1 &
done
wait
echo done
