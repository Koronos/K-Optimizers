#!/usr/bin/env bash
# Second remedy round: kahan8-ld-b256 with the Weyl increment also re-drawn per block.
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=-1
export PYTHONPATH=../../../../src
PY=${PY:-/c/Users/Koronos/Documents/Repos/Rengu-Flow/.venv/Scripts/python.exe}
mkdir -p results_remedies2 logs
for reg in alias-P2-lr1e-06 alias-P3-lr1e-06 alias-P7-lr1e-06 coh-lr1e-06 adam-lr1e-06 orig-lr1e-05; do
  "$PY" sim_lowdisc.py --d 65536 --n 100000 --threads 2 --out results_remedies2 \
    --schemes kahan8-sr,kahan8-ld-b256,kahan8-ld-b256k \
    --checkpoints 1000,3000,10000,30000,100000 --regimes "$reg" > "logs/rem2-$reg.log" 2>&1 &
done
wait
echo done
