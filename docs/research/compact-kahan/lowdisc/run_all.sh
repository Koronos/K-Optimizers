#!/usr/bin/env bash
# Launch the full grid on the CPU as 5 parallel processes (4 threads each). GPU hidden.
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=-1
export PYTHONPATH=../../../../src
PY=${PY:-/c/Users/Koronos/Documents/Repos/Rengu-Flow/.venv/Scripts/python.exe}
mkdir -p results logs
G1='orig-lr*,orig-drift*'
G2='multi-*'
G3='coh-*,noisy-*,adam-lr1e-04,adam-lr1e-05'
G4='alias-P2-*,alias-P3-*'
G5='alias-P7-*,adam-lr1e-06'
i=0
for g in "$G1" "$G2" "$G3" "$G4" "$G5"; do
  i=$((i+1))
  "$PY" sim_lowdisc.py --regimes "$g" --threads 4 --out results > "logs/g$i.log" 2>&1 &
done
wait
echo done
