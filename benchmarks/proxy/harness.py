"""Shared synthetic pixel-DDPM harness for the optimizer benchmarks.

A small conv U-Net + DDPM epsilon-prediction loss on the procedurally generated,
reproducible proxy dataset ([dataset.py](dataset.py)). This is the *optimizer-agnostic*
model/loss core — every benchmark (the AdaMuon A/Bs, the control battery, ad-hoc sweeps)
imports it instead of re-defining the model.

Two scales stand in for two real regimes:
  * ``C=128`` (wide U-Net) — a *full fine-tune*-like regime (large bandwidth-bound weights).
  * ``C=64``  (narrow)      — a lighter regime.
For the *LoRA*-like many-tiny-tensor regime (launch-bound optimizer steps) see
``lora_bag`` below — a bag of small adapter-shaped tensors, no model needed.

Importable by file path (``importlib``) so scripts need no package install.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float32


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


# ----------------------------- DDPM loss -----------------------------
def make_alphas(T=1000):
    betas = torch.linspace(1e-4, 2e-2, T, device=DEV)
    return torch.cumprod(1 - betas, 0)


def batch_loss(net, x, idx, ac, gen):
    """Deterministic given ``gen``: noise + timestep are drawn from the passed generator."""
    lat = x[idx]
    t = torch.randint(0, 1000, (lat.shape[0],), device=DEV, generator=gen)
    noise = torch.randn(lat.shape, device=DEV, generator=gen)
    a = ac[t].view(-1, 1, 1, 1)
    noisy = a.sqrt() * lat + (1 - a).sqrt() * noise
    return F.mse_loss(net(noisy, t), noise)


@torch.no_grad()
def eval_loss(net, x, idx, ac, reps=8, seed0=5000):
    """Held-out (or train) eps-MSE averaged over ``reps`` frozen noise+timestep draws."""
    net.eval(); g = torch.Generator(device=DEV); acc = []
    for r in range(reps):
        g.manual_seed(seed0 + r)
        acc.append(float(batch_loss(net, x, torch.tensor(idx, device=DEV), ac, g)))
    net.train(); return sum(acc) / len(acc)


# ----------------------------- memory -----------------------------
def opt_state_bytes_per_param(opt, params):
    # Sum the optimizer's own state AND any wrapped inner/base optimizer's state, so
    # wrappers (Lookahead -> .inner, SAM -> .base_optimizer) report their TRUE footprint
    # (the wrapped optimizer keeps its factored/momentum state in a separate .state dict).
    seen = set()
    b = 0
    opts = []
    stack = [opt]
    while stack:
        o = stack.pop()
        if o is None or id(o) in seen or not hasattr(o, "state"):
            continue
        seen.add(id(o))
        opts.append(o)
        stack.append(getattr(o, "inner", None))
        stack.append(getattr(o, "base_optimizer", None))
    for o in opts:
        for st in o.state.values():
            for v in st.values():
                if torch.is_tensor(v):
                    b += v.numel() * v.element_size()
    nparam = sum(p.numel() for p in params)
    return b / max(1, nparam)


# ----------------------------- LoRA-like speed proxy -----------------------------
def lora_bag(n_tensors=512, shape=(8, 16), seed=0):
    """A bag of small adapter-shaped trainable tensors (no model) — the launch-bound
    many-tiny-tensor regime where ``foreach`` batching matters. Returns a list of leaf
    params with random grads attached, ready to be stepped."""
    g = torch.Generator().manual_seed(seed)
    params = []
    for _ in range(n_tensors):
        p = torch.randn(*shape, generator=g).to(DEV).to(DT).requires_grad_(True)
        p.grad = torch.randn(*shape, generator=g).to(DEV).to(DT)
        params.append(p)
    return params
