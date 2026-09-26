"""Bit-exactness check: sim_lowdisc's one-fp32 representation == kaon's real writers.

* kahanB write (B = 8, 4; RN and injected noise) vs ``kaon._compact_kahan`` decode -> fp32 add
  -> ``encode_(..., noise)``: the decoded value and the forward bf16 must match bit for bit.
* plain SR write vs ``kaon._stochastic_rounding._add_stochastic_bf16_`` with the same noise
  injected (``torch.randint`` is patched for the duration of the call).
* ``kahan8-ld-kaon`` (the shipped ``bf16_method="kahan8ld"`` dither) vs kaon's
  ``compensated_add_`` handed the optimizers' ``LDNoise``.

CPU only; imports src/ read-only.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import torch

import kaon._stochastic_rounding as srm
from kaon._compact_kahan import decode, encode_

from sim_lowdisc import SCHEMES, Scheme

assert not torch.cuda.is_available()
D, STEPS = 1 << 16, 200
g = torch.Generator().manual_seed(0)
w0 = (torch.randn(D, generator=g) * 0.05).to(torch.bfloat16)
bad = 0
for name in ("kahan8-rn", "kahan8-sr", "kahan8-ld", "kahan4-sr", "kahan4-ld", "sr", "ld-phi"):
    dropped = SCHEMES[name][0]
    s = Scheme(name, w0.float(), D, 1234)
    p = w0.clone()
    lo = torch.zeros(D, dtype=torch.uint8)
    for n in range(STEPS):
        upd = torch.randn(D, generator=g) * 1e-5 * (1 + 10 * (n % 3 == 0))
        r16 = torch.randint(0, 1 << 16, (D,), generator=g, dtype=torch.int32)
        s.write(upd, n, r16)
        # the exact noise the scheme used this write
        if s.kind == "iid":
            noise = r16 >> (16 - dropped)
        elif s.kind == "rn":
            noise = None
        else:
            noise = (s.h + ((n * s.k) & ((1 << dropped) - 1))) & ((1 << dropped) - 1)
        if dropped == 16:
            orig = torch.randint
            srm.torch.randint = lambda *a, **k: noise.clone()
            try:
                srm._add_stochastic_bf16_(p, upd, 1.0)
            finally:
                srm.torch.randint = orig
            ref_z, ref_fwd = p.float(), p.float()
        else:
            bits = 16 - dropped
            z = decode(p, lo, bits)
            z.add_(upd)
            encode_(z, p, lo, bits, None if noise is None else noise.clone())
            ref_z, ref_fwd = decode(p, lo, bits), p.float()
    dz = int((s.z.view(torch.int32) != ref_z.view(torch.int32)).sum())
    df = int((s.fwd().view(torch.int32) != ref_fwd.view(torch.int32)).sum())
    bad += dz + df
    print(f"{name:10s} after {STEPS} writes: tracked mismatches {dz}/{D}, forward-bf16 mismatches {df}/{D}")
# The shipped bf16_method="kahan8ld": the sim's "kahan8-ld-kaon" scheme vs kaon's own writer
# (compensated_add_ with the LDNoise the optimizers hand it), same updates, same counter.
from kaon._compact_kahan import LDNoise, compensated_add_, ld_key  # noqa: E402

s = Scheme("kahan8-ld-kaon", w0.float(), D, 1234)
p = w0.clone()
lo = torch.zeros(D, dtype=torch.uint8)
for n in range(STEPS):
    upd = torch.randn(D, generator=g) * 1e-5 * (1 + 10 * (n % 3 == 0))
    s.write(upd, n, None)
    compensated_add_(p, lo, upd, 1.0, 8, LDNoise([ld_key(0)], n))
ref_z = decode(p, lo, 8)
dz = int((s.z.view(torch.int32) != ref_z.view(torch.int32)).sum())
df = int((s.fwd().view(torch.int32) != p.float().view(torch.int32)).sum())
bad += dz + df
print(f"{'kahan8-ld-kaon':10s} after {STEPS} writes: tracked mismatches {dz}/{D}, forward-bf16 mismatches {df}/{D}")
print("OK: bit-exact" if bad == 0 else f"FAIL: {bad} mismatches")
raise SystemExit(0 if bad == 0 else 1)
