#!/usr/bin/env python
"""Self-contained convergence A/B for AdaMuon vs Adakaon (and AdamW) on a small
pixel-space DDPM with PROCEDURALLY GENERATED data — no external models or data.

Why this exists
---------------
A controlled, reproducible optimizer comparison. It trains a small conv U-Net
from scratch on synthetic high-frequency images with the standard DDPM
epsilon-prediction loss, and reports held-out validation MSE. Because training is
from scratch, held-out val MSE is a clean convergence signal here (unlike LoRA
finetuning of a *pretrained* model — see sdxl_lora_ab.py for that regime and its
deterministic-probe metric).

Protocol (fair, paired)
-----------------------
* Per-arm LR is swept; report each optimizer at ITS best LR (best-vs-best).
* Paired seeds: for a given seed, model init + data + per-step noise/timestep
  draws are identical across optimizers, so differences are the optimizer alone.
* Memory: optimizer-state bytes/param is measured directly.

Reproduce the headline:
    python pixel_ddpm_ab.py --preset headline
Single arm / custom sweep:
    python pixel_ddpm_ab.py --optims "adamuon:1e-3:cos,adakaon:1e-3:cos" --steps 800 --seeds 3

Requires only: torch, kaon (pip install -e . at the repo root).
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from kaon import Adakaon, AdaMuon

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float32


# ----------------------------- synthetic data -----------------------------
def gen_data(n: int, H: int = 32, seed: int = 0) -> torch.Tensor:
    """n grayscale images = sums of random 2-D sinusoids (high-freq detail) + a blob."""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, H), indexing="ij")
    xs = []
    for _ in range(n):
        img = torch.zeros(H, H)
        for _ in range(5):
            fx = torch.randint(1, 9, (1,), generator=g).item()
            fy = torch.randint(1, 9, (1,), generator=g).item()
            ph = torch.rand(1, generator=g).item() * 6.283
            amp = torch.rand(1, generator=g).item()
            img = img + amp * torch.sin(6.283 * (fx * xx + fy * yy) + ph)
        cx, cy = torch.rand(2, generator=g).tolist()
        img = img + 1.5 * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 0.05)
        xs.append(img)
    x = torch.stack(xs)[:, None]
    return (x - x.mean()) / (x.std() + 1e-6)


# ----------------------------- model -----------------------------
def temb(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * freqs[None]
    return torch.cat([a.sin(), a.cos()], -1)


class Block(nn.Module):
    def __init__(self, ci, co, td=64):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ci); self.c1 = nn.Conv2d(ci, co, 3, 1, 1)
        self.temb = nn.Linear(td, co)
        self.n2 = nn.GroupNorm(8, co); self.c2 = nn.Conv2d(co, co, 3, 1, 1)
        self.skip = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x, te):
        h = self.c1(F.silu(self.n1(x))) + self.temb(te)[..., None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, C=128, td=64):
        super().__init__()
        self.td = td
        self.tmlp = nn.Sequential(nn.Linear(td, td), nn.SiLU(), nn.Linear(td, td))
        self.inp = nn.Conv2d(1, C, 3, 1, 1)
        self.d1 = Block(C, C, td); self.down = nn.Conv2d(C, 2 * C, 3, 2, 1)
        self.mid = Block(2 * C, 2 * C, td)
        self.up = nn.ConvTranspose2d(2 * C, C, 4, 2, 1)
        self.u1 = Block(2 * C, C, td)
        self.outn = nn.GroupNorm(8, C); self.out = nn.Conv2d(C, 1, 3, 1, 1)

    def forward(self, x, t):
        te = self.tmlp(temb(t, self.td))
        h1 = self.d1(self.inp(x), te)
        h = self.mid(self.down(h1), te)
        h = self.u1(torch.cat([self.up(h), h1], 1), te)
        return self.out(F.silu(self.outn(h)))


def make_alphas(T=1000):
    betas = torch.linspace(1e-4, 2e-2, T, device=DEV)
    return torch.cumprod(1 - betas, 0)


def batch_loss(net, x, idx, ac, gen):
    lat = x[idx]
    t = torch.randint(0, 1000, (lat.shape[0],), device=DEV, generator=gen)
    noise = torch.randn(lat.shape, device=DEV, generator=gen)
    a = ac[t].view(-1, 1, 1, 1)
    noisy = a.sqrt() * lat + (1 - a).sqrt() * noise
    return F.mse_loss(net(noisy, t), noise)


@torch.no_grad()
def val_loss(net, x, vidx, ac, reps=6):
    net.eval(); g = torch.Generator(device=DEV); accs = []
    for r in range(reps):
        g.manual_seed(1000 + r); accs.append(float(batch_loss(net, x, vidx, ac, g)))
    net.train(); return sum(accs) / len(accs)


# ----------------------------- optimizer factories -----------------------------
def make_optimizer(name, params, lr):
    """name in {adamuon, adamuon_int8, adamuon_4bit, adamuon_nomom, adakaon, adakaon_int8, adamw}."""
    if name == "adamuon":
        return AdaMuon(params, lr=lr, betas=(0.95, 0.999), clip_threshold=1.0, ns_steps=2, cautious=True)
    if name == "adamuon_int8":
        return AdaMuon(params, lr=lr, betas=(0.95, 0.999), ns_steps=2, cautious=True, momentum_dtype="int8")
    if name == "adamuon_4bit":
        return AdaMuon(params, lr=lr, betas=(0.95, 0.999), ns_steps=2, cautious=True, momentum_dtype="4bit")
    if name == "adamuon_nomom":
        return AdaMuon(params, lr=lr, betas=(0.0, 0.999), ns_steps=2, cautious=True)
    if name == "adakaon":
        return Adakaon(params, lr=lr, betas=(0.9, 0.999), cautious=True, momentum_dtype="bfloat16")
    if name == "adakaon_int8":
        return Adakaon(params, lr=lr, betas=(0.9, 0.999), cautious=True, momentum_dtype="int8")
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999))
    if name == "adamw_fused":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), fused=True)
    if name == "adafactor":  # the memory-constrained full-FT baseline
        return torch.optim.Adafactor(params, lr=lr)
    raise ValueError(f"unknown optimizer {name!r}")


def opt_state_bytes(opt):
    b = 0
    for st in opt.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                b += v.numel() * v.element_size()
    return b


def train(name, lr, seed, steps, x, tr, va, ac, channels, bs, cosine):
    torch.manual_seed(seed)
    if DEV == "cuda":
        torch.cuda.manual_seed_all(seed)
    net = UNet(C=channels).to(DEV).to(DT)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = make_optimizer(name, params, lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps) if cosine else None
    g = torch.Generator(device=DEV); g.manual_seed(seed + 12345)
    best = 1e9; t0 = time.time(); n = 0
    for it in range(steps):
        bidx = torch.tensor([tr[(it * bs + j) % len(tr)] for j in range(bs)], device=DEV)
        loss = batch_loss(net, x, bidx, ac, g)
        opt.zero_grad(); loss.backward(); opt.step()
        if sched:
            sched.step()
        n += 1
        if (it + 1) % max(1, steps // 8) == 0:
            best = min(best, val_loss(net, x, va, ac))
    if DEV == "cuda":
        torch.cuda.synchronize()
    nparam = sum(p.numel() for p in params)
    return best, (time.time() - t0) / n * 1000, opt_state_bytes(opt) / nparam


PRESETS = {
    # name: list of "optim:lr:[cos]"
    "headline": [
        "adamuon:1e-3:cos", "adamuon_int8:1e-3:cos", "adamuon_4bit:1e-3:cos",
        "adakaon:1e-3:cos", "adamuon:1e-3", "adakaon:1e-3", "adamw:1e-3",
    ],
    "lr_sweep_adamuon": [f"adamuon:{lr}" for lr in ("3e-4", "1e-3", "3e-3")],
    "lr_sweep_adakaon": [f"adakaon:{lr}" for lr in ("3e-4", "1e-3", "3e-3")],
    "memory_ladder": ["adamuon:1e-3:cos", "adamuon_int8:1e-3:cos", "adamuon_4bit:1e-3:cos", "adamuon_nomom:1e-3:cos"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--n-images", type=int, default=80)
    ap.add_argument("--preset", choices=list(PRESETS), default=None)
    ap.add_argument("--optims", type=str, default=None,
                    help='comma list of "optim:lr[:cos]", e.g. "adamuon:1e-3:cos,adakaon:1e-3:cos"')
    A = ap.parse_args()

    arms_spec = PRESETS[A.preset] if A.preset else (A.optims or "adamuon:1e-3:cos,adakaon:1e-3:cos").split(",")
    x = gen_data(A.n_images, seed=7).to(DEV).to(DT)
    ntr = int(A.n_images * 0.75); tr = list(range(ntr)); va = list(range(ntr, A.n_images))
    ac = make_alphas()
    print(f"device={DEV} channels={A.channels} steps={A.steps} seeds={A.seeds} bs={A.bs} "
          f"train={len(tr)} val={len(va)}", flush=True)

    results = {}
    for spec in arms_spec:
        parts = spec.split(":"); name = parts[0]; lr = float(parts[1])
        cosine = len(parts) > 2 and parts[2] == "cos"
        vals = []; ms = bp = 0.0
        for sd in range(A.seeds):
            v, ms, bp = train(name, lr, sd, A.steps, x, tr, va, ac, A.channels, A.bs, cosine)
            vals.append(v)
        mean = sum(vals) / len(vals)
        results[spec] = (mean, ms, bp)
        tag = name + ("+cos" if cosine else "")
        print(f"  {tag:16s} lr={lr:.1e}  val={mean:.5f}  {ms:.1f}ms/step  {bp:.2f}B/param  "
              f"seeds={[f'{v:.4f}' for v in vals]}", flush=True)

    best_spec = min(results, key=lambda s: results[s][0])
    print(f"\nBEST: {best_spec}  val={results[best_spec][0]:.5f}")


if __name__ == "__main__":
    main()
