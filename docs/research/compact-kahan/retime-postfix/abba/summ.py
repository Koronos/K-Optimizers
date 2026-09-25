import json, sys, statistics as st
from collections import defaultdict
rows = [json.loads(l) for l in open(sys.argv[1])]
order = []
agg = defaultdict(lambda: defaultdict(list))
for row in rows:
    r = row["r"]
    if r["tag"] not in order: order.append(r["tag"])
    for k, v in r.items():
        if k.startswith(("iso_", "il_", "pre_")): agg[r["tag"]][k].append(v[0])
keys = sorted({k for t in agg.values() for k in t})
def q(v):
    v = sorted(v); n=len(v)
    import numpy as np
    return np.median(v), np.percentile(v,25), np.percentile(v,75)
print("| commit | " + " | ".join(keys) + " |")
print("|---|" + "---|"*len(keys))
for t in order:
    cells=[]
    for k in keys:
        v = agg[t].get(k)
        if not v: cells.append("-"); continue
        m,a,b = q(v); cells.append(f"{m:.3f} [{a:.3f}-{b:.3f}]")
    print(f"| {t} | " + " | ".join(cells) + " |")
