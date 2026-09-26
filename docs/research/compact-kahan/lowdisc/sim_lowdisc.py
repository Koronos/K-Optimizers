"""Low-discrepancy rounding for bf16 weight writes — CPU-only simulation.

Two candidates from the literature pass on bf16 weight writes, against the shipped schemes:

* **A. Low-discrepancy SR (cost 0).** Plain bf16 stochastic rounding, but the 16 noise bits
  are ``u16 = (h16(i) + n*K) mod 2**16`` instead of fresh random bits: ``h16`` is a FIXED
  per-element hash (splitmix64), ``n`` the write counter, ``K`` an odd Weyl increment
  (``40503 = round(0.618034 * 2**16)``; also the plastic-number R2 constant, sqrt(2)-1 and
  a deliberately bad 1/3). Over the random hash each threshold is uniform -> each write is
  unbiased; along ``n`` the thresholds are a Weyl sequence -> a constant sub-ulp step is
  realised like a sigma-delta modulator (bounded error instead of a sqrt(N) walk).
* **B. kahan8 with the residual rounded by a low-discrepancy sequence.** Same idea at the
  residual's 8 dropped bits: ``u8 = (h8(i) + n*159) mod 256`` replaces the residual's SR.
  ``kahan4-ld`` (12 dropped bits, ``K = 2531``) as a control.

Remedies against B's long-horizon aliasing (``run_remedies*.sh``, README): (a) a random phase
per (element, block of K writes) added to the Weyl sequence (``kahan8-ld-b{64,256,1024}``;
``-b256k`` also re-draws the increment per block), (b) a 16-bit phi counter, top byte
(``kahan8-ld16``), (c) a small iid jitter (``-j32``, ``-ld16-j64``). ``kahan8-ld-kaon`` is the
shipped ``bf16_method="kahan8ld"`` noise, taken from ``kaon._compact_kahan.ld_noise``.

Representation (bit-exact equivalent of ``kaon._compact_kahan``, verified by
``check_equivalence.py``): a kahanB pair ``(w, lo)`` IS an fp32 whose low ``16-B`` mantissa
bits are zero, so each scheme's tracked value is held as ONE fp32 tensor ``z`` and a write is
``z = z + upd`` (fp32, RN) followed by ``bits += noise; bits &= -2**(16-B)``. Plain SR is
the ``B = 0`` case (``bits += u16; bits &= -2**16``), which is exactly
``kaon._stochastic_rounding._add_stochastic_bf16_``. The bf16 weight the forward sees is
``z`` rounded half-away (the codec's carry). No src/ code is modified; no GPU is used.

Harness conventions follow ``../sim_compact_kahan.py``: ``w0 ~ 0.05 N(0,1)``, 2**20
coordinates, 10 000 steps, fp32 reference fed the same stream, no feedback, errors in
``ulp_ref = ulp_bf16(RMS z_ref)``; ``lost`` = fraction of the net movement not realised;
``dir_bias`` = mean of the error times sign(movement), with its SE (the plain mean is blind to a
sign-symmetric under-move).
Metrics are taken at N = 1k, 3k and 10k to see sqrt(N) growth vs a bounded error.

Regimes (``--regimes`` filters by name, ``foo*`` by prefix):
  orig-*   the existing sim's streams (validation against its table): delta = lr(0.1c +
           0.99 nu), c = +-1 redrawn every 50 steps; lr 1e-3..1e-5; pure drift lr 1e-6; with
           and without xi (k=4, carried in the weights, ONE combined write per step).
  coh-*    coherent: delta = lr * c_i, c_i = +-1 fixed per element for the whole run.
  noisy-*  noise-dominated: delta = u0 (mu c_i + sigma nu), sigma >> mu (u0 = 2**-12, the
           bf16 ulp at |w| = 0.05).
  adam-*   realistic: g = 0.1 c + nu (c redrawn every 1000 steps), m = EMA beta1=0.9,
           v = EMA beta2=0.999, delta = lr * mhat / (sqrt(vhat) + 1e-8).
  multi-*  MSAM/Nekaon-style: THREE writes per step (+xi, -xi, -delta), each with its own n.
  alias-*  periodic: delta_n = lr * c_i * (cos(2 pi n / P) + 0.1), P = 2, 3, 7.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import torch

assert not torch.cuda.is_available(), "this simulation must run on the CPU only"
dev = torch.device("cpu")

U0 = 2.0 ** -12  # bf16 ulp at |w| in [2**-5, 2**-4) — the RMS of w0 = 0.05
CHECKPOINTS = (1_000, 3_000, 10_000)

# name -> (dropped bits, noise kind, Weyl increment or None)
#   dropped = 16 for plain bf16 SR, 16 - B for kahanB
SCHEMES: dict[str, tuple[int, str, int | None]] = {
    "sr":            (16, "iid", None),
    "ld-phi":        (16, "ld", 40503),      # round(0.618034 * 2**16), odd
    "ld-r2":         (16, "ld", 49471),      # round(1/rho * 2**16), rho = plastic (R2's alpha1), odd
    "ld-sqrt2":      (16, "ld", 27145),      # (sqrt2 - 1) * 2**16 -> nearest odd
    "ld-third":      (16, "ld", 21845),      # ~1/3: rational-like control, period-3 resonance
    "ld-phi-nohash": (16, "ld0", 40503),     # same sequence for EVERY element (no h16): control
    "kahan8-sr":     (8, "iid", None),
    "kahan8-rn":     (8, "rn", None),
    "kahan8-ld":     (8, "ld", 159),         # round(0.618 * 256) -> odd
    "kahan8-ld-r2":  (8, "ld", 193),         # round(0.75488 * 256), odd
    "kahan4-sr":     (12, "iid", None),
    "kahan4-ld":     (12, "ld", 2531),       # round(0.618034 * 4096), odd
    # --- remedies against kahan8-ld's long-horizon aliasing (P = 2, 3 at lr 1e-6) ---
    # (a) phase re-randomised per time block: u8 = h8(i) + n*159 + r(i, n // K)
    "kahan8-ld-b64":   (8, "ldblk", (159, 64)),
    "kahan8-ld-b256":  (8, "ldblk", (159, 256)),
    "kahan8-ld-b1024": (8, "ldblk", (159, 1024)),
    # (a') same, and the Weyl increment also re-drawn per (element, block) from four good ones
    "kahan8-ld-b256k": (8, "ldblkk", (159, 256)),
    # the shipped bf16_method="kahan8ld": (a) at K = 256 with kaon's own 32-bit hashes
    # (kaon._compact_kahan.ld_noise, the bits every kaon writer uses)
    "kahan8-ld-kaon":  (8, "kaon", None),
    # (b) 16-bit phi sequence, top byte: u8 = ((h16(i) + n*40503) mod 2**16) >> 8
    "kahan8-ld16":     (8, "ld16", 40503),
    "kahan8-ld16-b256": (8, "ld16blk", (40503, 256)),
    # (c) mixed dither: u8 = h8(i) + n*159 + j, j iid uniform in [0, J)
    "kahan8-ld-j32":   (8, "ldj", (159, 32)),
    "kahan8-ld-j64":   (8, "ldj", (159, 64)),
    "kahan8-ld16-j64": (8, "ld16j", (40503, 64)),
}


def splitmix64(x: np.ndarray) -> np.ndarray:
    z = x + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def element_hash(d: int, seed: int, dropped: int) -> torch.Tensor:
    """Fixed per-element offset in [0, 2**dropped): top bits of splitmix64(i, seed)."""
    with np.errstate(over="ignore"):
        i = np.arange(d, dtype=np.uint64) + np.uint64(seed) * np.uint64(0x100000001B3)
        h = splitmix64(i) >> np.uint64(64 - dropped)
    return torch.from_numpy(h.astype(np.int32))


def block_hash(d: int, seed: int, block: int, bits: int) -> torch.Tensor:
    """Per-(element, time block) random phase in [0, 2**bits) — remedy (a)."""
    with np.errstate(over="ignore"):
        i = (np.arange(d, dtype=np.uint64) + np.uint64(seed) * np.uint64(0x100000001B3)
             + np.uint64(block + 1) * np.uint64(0xD1B54A32D192ED03))
        h = splitmix64(i) >> np.uint64(64 - bits)
    return torch.from_numpy(h.astype(np.int32))


def ulp_bf16(x: float) -> float:
    return 2.0 ** (math.floor(math.log2(max(abs(x), 1e-30))) - 7)


class Scheme:
    def __init__(self, name: str, z0: torch.Tensor, d: int, seed: int) -> None:
        self.name = name
        self.dropped, self.kind, self.k = SCHEMES[name]
        self.z = z0.clone()
        self.mask = -(1 << self.dropped)
        self.h = None
        self.d, self.seed = d, seed
        self.blk, self.r = -1, None
        if self.kind in ("ld", "ldblk", "ldj", "ldblkk"):
            self.h = element_hash(d, seed + 7919 * self.dropped, self.dropped)
        elif self.kind in ("ld16", "ld16blk", "ld16j"):
            self.h = element_hash(d, seed + 7919 * self.dropped, 16)
        elif self.kind == "ld0":
            self.h = torch.zeros(d, dtype=torch.int32)
        self.buf = torch.empty(d, dtype=torch.int32)

    def _block_phase(self, n: int, bits: int) -> torch.Tensor:
        k = self.k[1]
        if n // k != self.blk:
            self.blk = n // k
            self.r = block_hash(self.d, self.seed + 31, self.blk, bits)
        return self.r

    def write(self, upd: torch.Tensor, n: int, r16: torch.Tensor) -> None:
        z = self.z
        z.add_(upd)
        b = z.view(torch.int32)
        if self.kind == "rn":
            b.add_(1 << (self.dropped - 1))
        elif self.kind == "iid":
            # one shared 16-bit draw per write; each scheme uses its top ``dropped`` bits
            torch.bitwise_right_shift(r16, 16 - self.dropped, out=self.buf)
            b.add_(self.buf)
        elif self.kind == "kaon":
            from kaon._compact_kahan import ld_key, ld_noise
            b.add_(ld_noise([ld_key(0)], self.d, n, "cpu").view(-1))
        elif self.kind == "ldblkk":
            m = (1 << self.dropped) - 1
            r = self._block_phase(n, 16)                  # 16 random bits per (element, block)
            ks = torch.tensor([159, 97, 193, 105], dtype=torch.int32)[r & 3]
            # the counter restarts at each block so the in-block sequence is h + r + j*k
            torch.mul(ks, n % self.k[1], out=self.buf)
            self.buf.add_(self.h).add_(r >> 8)
            self.buf.bitwise_and_(m)
            b.add_(self.buf)
        elif self.kind in ("ldblk", "ldj"):
            m = (1 << self.dropped) - 1
            torch.add(self.h, (n * self.k[0]) & m, out=self.buf)
            if self.kind == "ldblk":
                self.buf.add_(self._block_phase(n, self.dropped))
            else:  # jitter: top log2(J) bits of the shared iid draw
                self.buf.add_(r16 >> (16 - (self.k[1].bit_length() - 1)))
            self.buf.bitwise_and_(m)
            b.add_(self.buf)
        elif self.kind in ("ld16", "ld16blk", "ld16j"):
            k16 = self.k if self.kind == "ld16" else self.k[0]
            torch.add(self.h, (n * k16) & 0xFFFF, out=self.buf)
            if self.kind == "ld16blk":
                self.buf.add_(self._block_phase(n, 16))
            elif self.kind == "ld16j":   # jitter J/256 of the byte grid = J*256 on the 16-bit grid
                self.buf.add_((r16 >> (16 - (self.k[1].bit_length() - 1))) << 8)
            self.buf.bitwise_and_(0xFFFF).bitwise_right_shift_(16 - self.dropped)
            b.add_(self.buf)
        else:  # ld / ld0: (h + n K) mod 2**dropped
            off = (n * self.k) & ((1 << self.dropped) - 1)
            torch.add(self.h, off, out=self.buf)
            self.buf.bitwise_and_((1 << self.dropped) - 1)
            b.add_(self.buf)
        b.bitwise_and_(self.mask)

    def fwd(self) -> torch.Tensor:
        """The bf16 weight the forward pass sees (kahan: z rounded half-away; sr: z itself)."""
        if self.dropped == 16:
            return self.z.clone()
        b = self.z.view(torch.int32).clone()
        b.add_(0x8000).bitwise_and_(-0x10000)
        return b.view(torch.float32)


# --------------------------------------------------------------------------- streams
class Stream:
    """Yields, per step, the list of weight writes and the net delta the reference takes."""

    k_xi = 0.0

    def __init__(self, d: int, gen: torch.Generator) -> None:
        self.d, self.gen = d, gen
        self.xi = torch.zeros(d)

    def rademacher(self) -> torch.Tensor:
        return torch.randint(0, 2, (self.d,), generator=self.gen, dtype=torch.float32).mul_(2).sub_(1)

    def step(self, n: int) -> tuple[list[torch.Tensor], torch.Tensor]:
        raise NotImplementedError


class Orig(Stream):
    def __init__(self, d, gen, lr, k, noise_scale):
        super().__init__(d, gen)
        self.lr, self.k_xi, self.ns = lr, k, noise_scale
        self.c = self.rademacher()
        if k > 0:
            self.xi = k * lr * self.rademacher()

    def step(self, n):
        if n % 50 == 0:
            self.c = self.rademacher()
        delta = 0.1 * self.lr * self.c
        if self.ns > 0:
            delta.add_(torch.randn(self.d, generator=self.gen), alpha=self.lr * self.ns * 0.99)
        if self.k_xi > 0:
            xi_next = self.k_xi * self.lr * self.rademacher()
            upd = (xi_next - self.xi) - delta
            self.xi = xi_next
        else:
            upd = -delta
        return [upd], delta


class Coherent(Stream):
    def __init__(self, d, gen, lr):
        super().__init__(d, gen)
        self.delta = lr * self.rademacher()
        self.neg = -self.delta

    def step(self, n):
        return [self.neg], self.delta


class Noisy(Stream):
    def __init__(self, d, gen, mu, sigma):
        super().__init__(d, gen)
        self.mu_c = U0 * mu * self.rademacher()
        self.s = U0 * sigma

    def step(self, n):
        delta = torch.randn(self.d, generator=self.gen).mul_(self.s).add_(self.mu_c)
        return [-delta], delta


class AdamLike(Stream):
    def __init__(self, d, gen, lr, b1=0.9, b2=0.999):
        super().__init__(d, gen)
        self.lr, self.b1, self.b2 = lr, b1, b2
        self.m = torch.zeros(d)
        self.v = torch.zeros(d)

    def step(self, n):
        if n % 1000 == 0:
            self.c = self.rademacher()
        g = torch.randn(self.d, generator=self.gen).add_(self.c, alpha=0.1)
        self.m.lerp_(g, 1 - self.b1)
        self.v.lerp_(g * g, 1 - self.b2)
        t = n + 1
        mh = self.m / (1 - self.b1 ** t)
        vh = self.v / (1 - self.b2 ** t)
        delta = mh.div_(vh.sqrt_().add_(1e-8)).mul_(self.lr)
        return [-delta], delta


class MultiWrite(Stream):
    """+xi (climb), -xi (restore), -delta (base step): three writes per step."""

    def __init__(self, d, gen, lr, k, noise_scale):
        super().__init__(d, gen)
        self.base = Orig(d, gen, lr, 0.0, noise_scale)
        self.kl = k * lr

    def step(self, n):
        (upd,), delta = self.base.step(n)
        xi = self.kl * self.rademacher()
        return [xi, -xi, upd], delta


class Alias(Stream):
    def __init__(self, d, gen, lr, period):
        super().__init__(d, gen)
        c = self.rademacher()
        self.pattern = [lr * c * (math.cos(2 * math.pi * j / period) + 0.1) for j in range(period)]
        self.period = period

    def step(self, n):
        delta = self.pattern[n % self.period]
        return [-delta], delta


def regimes() -> dict[str, tuple]:
    r: dict[str, tuple] = {}
    for lr in (1e-3, 1e-4, 1e-5):
        r[f"orig-lr{lr:.0e}"] = (Orig, lr, 0.0, 1.0)
        r[f"orig-lr{lr:.0e}-xi"] = (Orig, lr, 4.0, 1.0)
    r["orig-drift-lr1e-06"] = (Orig, 1e-6, 0.0, 0.0)
    r["orig-drift-lr1e-06-xi"] = (Orig, 1e-6, 4.0, 0.0)
    for lr in (1e-5, 1e-6, 1e-7):
        r[f"coh-lr{lr:.0e}"] = (Coherent, lr)
    r["noisy-mu0.05-s0.5"] = (Noisy, 0.05, 0.5)
    r["noisy-mu0.0005-s0.005"] = (Noisy, 0.0005, 0.005)
    for lr in (1e-4, 1e-5, 1e-6):
        r[f"adam-lr{lr:.0e}"] = (AdamLike, lr)
    r["multi-lr1e-04"] = (MultiWrite, 1e-4, 4.0, 1.0)
    r["multi-lr1e-05"] = (MultiWrite, 1e-5, 4.0, 1.0)
    r["multi-drift-lr1e-06"] = (MultiWrite, 1e-6, 4.0, 0.0)
    for lr in (1e-4, 1e-5, 1e-6):
        for p in (2, 3, 7):
            r[f"alias-P{p}-lr{lr:.0e}"] = (Alias, lr, p)
    return r


def metrics(name: str, s: Scheme, xi: torch.Tensor, z_ref: torch.Tensor, z_start: torch.Tensor,
            n: int, step_rms: float) -> dict:
    d = z_ref.numel()
    u = ulp_bf16(float(z_ref.pow(2).mean().sqrt()))
    err = (s.z - xi) - z_ref
    e = err / u
    move = z_ref - z_start
    mm = float((move * move).sum())
    fe = ((s.fwd() - xi) - z_ref) / u
    std = float(e.std())
    # Directional bias: the error projected on each element's movement sign. The plain mean
    # is blind to a scheme that systematically under-moves (the +-c signs cancel); this one
    # is not. Negative = the scheme lags the reference (lost movement).
    ed = e * torch.sign(move)
    return dict(
        scheme=name, n=n, ulp_ref=u, step_over_ulp=step_rms / u,
        bias_ulp=float(e.mean()), bias_se_ulp=std / math.sqrt(d), std_ulp=std,
        dir_bias_ulp=float(ed.mean()), dir_bias_se_ulp=float(ed.std()) / math.sqrt(d),
        rms_ulp=float(e.pow(2).mean().sqrt()), max_ulp=float(e.abs().max()),
        fwd_std_ulp=float(fe.std()),
        lost=float(-(err * move).sum() / mm) if mm > 0 else float("nan"),
        err_over_moved=float(err.norm() / math.sqrt(mm)) if mm > 0 else float("nan"),
    )


def run(reg: str, spec: tuple, d: int, nsteps: int, schemes: list[str], seed: int = 1234) -> list[dict]:
    gen = torch.Generator().manual_seed(seed)
    w0 = torch.randn(d, generator=gen) * 0.05
    stream = spec[0](d, gen, *spec[1:])
    w = (w0 + stream.xi).to(torch.bfloat16).float()   # everyone starts from the same bf16 weights
    z_ref = w - stream.xi
    z_start = z_ref.clone()
    sch = {name: Scheme(name, w, d, seed) for name in schemes}
    rgen = torch.Generator().manual_seed(seed + 1)
    r16 = torch.empty(d, dtype=torch.int32)
    out, nw, sq_step, t0 = [], 0, 0.0, time.time()
    for n in range(nsteps):
        writes, delta = stream.step(n)
        z_ref.sub_(delta)
        sq_step += float(delta.pow(2).mean())
        for upd in writes:
            torch.randint(0, 1 << 16, (d,), generator=rgen, dtype=torch.int32, out=r16)
            for s in sch.values():
                s.write(upd, nw, r16)
            nw += 1
        if n + 1 in CHECKPOINTS:
            rms = math.sqrt(sq_step / (n + 1))
            for name, s in sch.items():
                m = metrics(name, s, stream.xi, z_ref, z_start, n + 1, rms)
                m["regime"] = reg
                out.append(m)
            print(f"[{reg}] n={n + 1} {time.time() - t0:.0f}s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default="", help="comma-separated names; a trailing * matches a prefix (default: all)")
    ap.add_argument("--d", type=int, default=1 << 20)
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--schemes", default="", help="comma-separated subset of SCHEMES (default: all)")
    ap.add_argument("--checkpoints", default="", help="comma-separated steps (default: 1000,3000,10000)")
    args = ap.parse_args()
    global CHECKPOINTS
    if args.checkpoints:
        CHECKPOINTS = tuple(int(c) for c in args.checkpoints.split(","))
    schemes = [s for s in args.schemes.split(",") if s] or list(SCHEMES)
    if args.threads:
        torch.set_num_threads(args.threads)
    prefixes = [p for p in args.regimes.split(",") if p]
    def pick(k: str) -> bool:
        return not prefixes or any(k.startswith(p[:-1]) if p.endswith("*") else k == p for p in prefixes)
    regs = {k: v for k, v in regimes().items() if pick(k)}
    os.makedirs(args.out, exist_ok=True)
    print(f"device={dev} D={args.d} N={args.n} threads={torch.get_num_threads()} regimes={list(regs)}", flush=True)
    for reg, spec in regs.items():
        path = os.path.join(args.out, f"{reg}.json")
        if os.path.exists(path):
            print(f"skip {reg} (exists)", flush=True)
            continue
        res = run(reg, spec, args.d, args.n, schemes)
        with open(path, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
