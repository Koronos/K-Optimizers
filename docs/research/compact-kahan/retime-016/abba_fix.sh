#!/bin/bash
# usage: abba.sh <shapes> <modes> <rounds> <outprefix>
S=/c/Users/Koronos/AppData/Local/Temp/claude/C--Users-Koronos-Documents-Repos-K-Optimizers/78593ade-87c4-49ac-91af-2d6bea9fb51a/scratchpad/nk
PY=/c/Users/Koronos/Documents/Repos/Rengu-Flow/.venv/Scripts/python.exe
WT=/c/Users/Koronos/Documents/Repos/K-Optimizers/.claude/worktrees/nekaon-kahan
: > $S/$4.txt
for r in $(seq 1 $3); do
  for side in A B B A; do
    if [ $side = A ]; then (cd $S/base && PYTHONPATH=src $PY ../selfcuda.py A $1 $2 2>/dev/null | grep SELF >> $S/$4.txt)
    else (cd $WT && PYTHONPATH=src $PY $S/selfcuda.py B $1 $2 2>/dev/null | grep SELF >> $S/$4.txt); fi
  done
done
$PY - $S/$4.txt <<'PY'
import sys, statistics, collections
d = collections.defaultdict(lambda: {"A": [], "B": []})
for line in open(sys.argv[1]):
    _, side, label, v = line.split(); d[label][side].append(float(v))
for k, v in d.items():
    a, b = statistics.median(v["A"]), statistics.median(v["B"])
    print(f"{k:42s} A {a:7.3f}  B {b:7.3f}  B/A {b/a:6.3f}  (n={len(v['A'])})")
PY
