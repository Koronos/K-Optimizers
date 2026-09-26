#!/usr/bin/env bash
# Remedies against kahan8-ld's long-horizon aliasing: 100k steps, D = 2**16, CPU only,
# one process per regime (2 threads each).
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=-1
export PYTHONPATH=../../../../src
PY=${PY:-/c/Users/Koronos/Documents/Repos/Rengu-Flow/.venv/Scripts/python.exe}
mkdir -p results_remedies logs
SCHEMES=kahan8-sr,kahan8-ld,kahan8-ld-r2,kahan8-ld-b64,kahan8-ld-b256,kahan8-ld-b1024,kahan8-ld16,kahan8-ld16-b256,kahan8-ld-j32,kahan8-ld16-j64
for reg in alias-P2-lr1e-06 alias-P3-lr1e-06 alias-P7-lr1e-06 coh-lr1e-06 coh-lr1e-07 adam-lr1e-06 \
           orig-lr1e-05 orig-drift-lr1e-06 noisy-mu0.0005-s0.005; do
  "$PY" sim_lowdisc.py --d 65536 --n 100000 --threads 2 --out results_remedies --schemes "$SCHEMES" \
    --checkpoints 1000,3000,10000,30000,100000 --regimes "$reg" > "logs/rem-$reg.log" 2>&1 &
done
wait
echo done
