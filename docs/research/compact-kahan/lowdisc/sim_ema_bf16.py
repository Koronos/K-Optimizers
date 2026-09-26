"""EMA state stored in low precision: bf16 RN (what kaon does) vs bf16 SR vs fp32. CPU only.

Audit follow-up (see README): kaon computes every EMA in fp32 and ROUNDS ON WRITE-BACK.
``v`` (full or factored row/col) is always fp32; the first moment ``m`` is stored in
``momentum_dtype`` — ``"bfloat16"`` by default — and written back with round-to-nearest
(``copy_`` / ``tl.store(.., m.to(tl.bfloat16))``), int8 (per-row absmax) and 4bit (block-128
absmax) with ``round_()``. An EMA step ``x <- x + (1-beta)(target - x)`` whose increment is
below half an ulp of ``x`` is dropped by RN: the stored EMA stalls.

Streams (per element i, ``D = 512 x 512``, T = 10 000 steps, fp32 reference fed the same
stream):
  first moment  g_t = mu_i + s * nu_t,  mu_i ~ N(0,1), for noise/signal s in {0.1, 1, 10}
  first moment  sign flip: as s = 0.1, but mu -> -mu at T/2 (tracking a change)
  second moment target g_t**2 with g_t = mu_i + nu_t, whose scale drops 10x at T/2 (v should
                fall ~100x; the Collage example: beta2 = 0.999 -> v*beta2 rounds to v)

Methods: fp32 (reference), bf16-rn (kaon's _FloatCodec / fused store), bf16-sr (same with
stochastic rounding on the write), int8-rn (kaon's per-row absmax codec), 4bit-rn (kaon's
block-128 absmax codec). Metrics at T (and just before the change for the change streams):
relative bias ``mean(x - x_ref) / rms(x_ref)``, the magnitude ratio
``mean(x sign(x_ref)) / mean(|x_ref|)`` (< 1 = EMA shrunk toward 0 by stalls), relative RMS error ``rms(x - x_ref) /
rms(x_ref)``, and the stall fraction over the last 500 steps: the share of (element, step)
where the fp32 EMA increment was non-zero but the stored value did not change.
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import torch

assert not torch.cuda.is_available()
R = C = 512
T = 10_000


def sr_bf16_(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    b = x.view(torch.int32)
    b.add_(torch.randint(0, 1 << 16, x.shape, generator=gen, dtype=torch.int32)).bitwise_and_(-0x10000)
    return x


def q_int8(m: torch.Tensor) -> torch.Tensor:
    """kaon._momentum_codec._quant_int8 followed by dequant (per-row absmax, RN)."""
    scale = m.abs().amax(dim=1, keepdim=True).clamp_(min=1e-12) / 127.0
    return (m / scale).round_().clamp_(-127, 127).mul_(scale)


def q_4bit(m: torch.Tensor) -> torch.Tensor:
    """kaon's block-128 absmax 4-bit codec (codes -7..7, RN), quant+dequant."""
    blk = m.reshape(-1, 128)
    scale = blk.abs().amax(dim=1, keepdim=True).clamp_(min=1e-12) / 7.0
    return (blk / scale).round_().clamp_(-7, 7).mul_(scale).reshape(m.shape)


def store(method: str, x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    if method == "fp32":
        return x
    if method == "bf16-rn":
        return x.to(torch.bfloat16).float()
    if method == "bf16-sr":
        return sr_bf16_(x, gen)
    if method == "int8-rn":
        return q_int8(x)
    if method == "4bit-rn":
        return q_4bit(x)
    raise ValueError(method)


def run(stream: str, beta: float, methods: list[str], seed: int = 0) -> list[dict]:
    g = torch.Generator().manual_seed(seed)
    sgen = torch.Generator().manual_seed(seed + 1)
    mu = torch.randn(R, C, generator=g)
    if stream.startswith("m-s"):
        s = float(stream[3:])
    else:
        s = 0.1 if stream == "m-flip" else 1.0
    st = {m: torch.zeros(R, C) for m in methods}
    stall = {m: 0 for m in methods}
    moving = 0
    out = []
    t0 = time.time()
    for t in range(T):
        if stream == "m-flip" and t == T // 2:
            mu = -mu
        scale = 0.1 if (stream == "v-drop" and t >= T // 2) else 1.0
        x = mu + s * torch.randn(R, C, generator=g)
        target = (x * scale) ** 2 if stream == "v-drop" else x
        ref_old = st["fp32"]
        ref_new = ref_old + (1 - beta) * (target - ref_old)
        if t >= T - 500:
            moving += int(((ref_new - ref_old) != 0).sum())
        for m in methods:
            if m == "fp32":
                continue
            old = st[m]
            new = store(m, old + (1 - beta) * (target - old), sgen)
            if t >= T - 500:
                stall[m] += int(((new == old) & (ref_new != ref_old)).sum())
            st[m] = new
        st["fp32"] = ref_new
        if (t + 1) in (T // 2, T):
            ref = st["fp32"]
            rr = float(ref.pow(2).mean().sqrt())
            for m in methods:
                e = st[m] - ref
                out.append(dict(stream=stream, beta=beta, method=m, t=t + 1, size=f"{R}x{C}",
                                rel_bias=float(e.mean()) / rr,
                                rel_bias_se=float(e.std()) / rr / (R * C) ** 0.5,
                                rel_rms=float(e.pow(2).mean().sqrt()) / rr,
                                # magnitude ratio: mean(x * sign(ref)) / mean(|ref|). < 1 means the
                                # stored EMA is SHRUNK toward 0 (a stall short of the target); the
                                # signed rel_bias cancels over +-mu and cannot see it.
                                ratio_mean=float((st[m] * ref.sign()).mean() / ref.abs().mean()),
                                stall=(stall[m] / moving if (t + 1 == T and moving) else None)))
    print(f"{stream} beta={beta} {time.time() - t0:.0f}s", flush=True)
    return out


def main() -> None:
    """``sim_ema_bf16.py OUT.json [SIZE] [--skip DONE.json]``: SIZE sets R = C (default 512);
    cases already present in DONE.json are skipped (used to finish the grid at a smaller size
    when the CPU was shared — the size of every case is recorded in its rows)."""
    global R, C
    args = [a for a in sys.argv[1:]]
    skip: set[tuple[str, float]] = set()
    if "--skip" in args:
        i = args.index("--skip")
        for r in json.load(open(args[i + 1])):
            skip.add((r["stream"], r["beta"]))
        del args[i:i + 2]
    path = args[0] if args else "results/ema_bf16.json"
    if len(args) > 1:
        R = C = int(args[1])
    torch.set_num_threads(4)
    grid = []
    for beta in (0.9, 0.99, 0.999):
        for stream in ("m-s0.1", "m-s1", "m-s10", "m-flip"):
            grid.append((stream, beta, ["fp32", "bf16-rn", "bf16-sr", "int8-rn", "4bit-rn"]))
    grid.append(("m-s1", 0.5, ["fp32", "bf16-rn", "bf16-sr", "int8-rn", "4bit-rn"]))  # Nekaon: 4bit, beta1 0.5
    for beta in (0.9, 0.999):
        grid.append(("v-drop", beta, ["fp32", "bf16-rn", "bf16-sr"]))
    res = []
    for stream, beta, methods in grid:
        if (stream, beta) in skip:
            continue
        res += run(stream, beta, methods)
        with open(path, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
