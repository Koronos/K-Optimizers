#!/usr/bin/env python
"""Real SDXL LoRA optimizer A/B: AdaMuon vs Adakaon on a pretrained SDXL UNet.

Two subcommands:
  precompute  Encode images to VAE latents (cached) + a FIXED text embedding.
  ab          Train ONLY a PEFT LoRA on the cached latents, paired-seed A/B.

Paths come from ENV VARS (no hard-coded paths):
  ADAMUON_SDXL_CKPT   single-file SDXL .safetensors checkpoint   (required)
  ADAMUON_IMG_DIR     folder of training images (.jpg/.png)      (required for precompute)
  ADAMUON_CACHE       latent cache dir          (default: ./adamuon_sdxl_cache)
  ADAMUON_SF_PATCH    optional "module.path:func" applied before from_single_file
                      (some transformers v5 setups need a single-file CLIP patch;
                       we only load the VAE/UNet so it is usually NOT required)

Method notes (so others can reproduce AND critique)
---------------------------------------------------
* The UNet is PRETRAINED, so held-out eps-MSE measures overfitting, not
  optimization progress, and the per-step random-timestep train loss is too noisy
  to rank optimizers. We therefore report a DETERMINISTIC OBJECTIVE PROBE: eps-MSE
  on the train set with a FIXED set of (timestep, noise) draws — a low-variance
  estimate of the very objective the optimizer minimizes. Lower-faster = better.
* Text conditioning is a FIXED deterministic embedding (the same for every image),
  so the cross-attention input is constant. This keeps the harness independent of
  the (version-fragile) single-file CLIP loader; for an OPTIMIZER A/B the
  conditioning only needs to be consistent. Limitation: it does NOT exercise
  text-conditioned learning — a real LoRA run would encode real prompts.
* Per-step wall-clock on SDXL is UNet-bound; the optimizer is <1% of the step, so
  AdaMuon's Newton-Schulz overhead is negligible here (verify via the ms/step
  column). Convergence-per-step therefore equals convergence-per-second.
* Small study by design (few images, low rank, few seeds). A signal, not a proof.

Requires: torch, diffusers, peft, kaon. Run with an env that has working
diffusers single-file loading for your checkpoint.
"""
from __future__ import annotations

import argparse
import glob
import importlib
import os
import random
import time

import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16


def _env(name, default=None, required=False):
    v = os.environ.get(name, default)
    if required and not v:
        raise SystemExit(f"set ${name} (see module docstring)")
    return v


def _maybe_single_file_patch():
    spec = os.environ.get("ADAMUON_SF_PATCH")
    if not spec:
        return
    mod, fn = spec.split(":")
    getattr(importlib.import_module(mod), fn)()


def make_alphas(T=1000):
    betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, T, device=DEV) ** 2
    return torch.cumprod(1.0 - betas, 0)


# ----------------------------- precompute -----------------------------
def cmd_precompute(A):
    import numpy as np
    from diffusers import AutoencoderKL
    from PIL import Image

    ckpt = _env("ADAMUON_SDXL_CKPT", required=True)
    img_dir = _env("ADAMUON_IMG_DIR", required=True)
    out = _env("ADAMUON_CACHE", "./adamuon_sdxl_cache")
    os.makedirs(out, exist_ok=True)
    _maybe_single_file_patch()

    vae = AutoencoderKL.from_single_file(ckpt, torch_dtype=torch.float32).to(DEV).eval()
    g = torch.Generator().manual_seed(2024)
    prompt_embeds = torch.randn(1, 77, 2048, generator=g) * 0.5   # fixed conditioning
    pooled = torch.randn(1, 1280, generator=g) * 0.5

    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg"))) + sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not imgs:
        raise SystemExit(f"no images in {img_dir}")
    print(f"{len(imgs)} images, RES={A.res}", flush=True)
    for i, ip in enumerate(imgs):
        img = Image.open(ip).convert("RGB").resize((A.res, A.res), Image.BICUBIC)
        arr = (np.asarray(img).astype(np.float32) / 127.5) - 1.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEV, torch.float32)
        with torch.no_grad():
            lat = vae.encode(t).latent_dist.sample() * vae.config.scaling_factor
        torch.save({"latent": lat.float().cpu(), "prompt_embeds": prompt_embeds.clone(), "pooled": pooled.clone()},
                   os.path.join(out, f"rec_{i:02d}.pt"))
    torch.save({"add_time_ids": torch.tensor([[A.res, A.res, 0, 0, A.res, A.res]], dtype=torch.float32),
                "n": len(imgs)}, os.path.join(out, "meta.pt"))
    print(f"cached {len(imgs)} latents -> {out}", flush=True)


# ----------------------------- model + train -----------------------------
def build_unet(rank):
    from diffusers import UNet2DConditionModel
    from peft import LoraConfig
    ckpt = _env("ADAMUON_SDXL_CKPT", required=True)
    _maybe_single_file_patch()
    unet = UNet2DConditionModel.from_single_file(ckpt, torch_dtype=DT)
    unet.requires_grad_(False); unet.to(DEV)
    cfg = LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian",
                     target_modules=["to_q", "to_k", "to_v", "to_out.0"])
    unet.add_adapter(cfg)
    return unet


def reinit_lora(unet, seed, rank):
    """Paired manual re-init: lora_A gaussian, lora_B zero — identical per seed."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    with torch.no_grad():
        for nm, p in unet.named_parameters():
            if not p.requires_grad:
                continue
            if "lora_A" in nm:
                p.copy_((torch.randn(p.shape, generator=g, device=DEV) * (1.0 / rank)).to(p.dtype))
            elif "lora_B" in nm:
                p.zero_()
    return [p for p in unet.parameters() if p.requires_grad]


def loss_on_batch(unet, recs, idxs, ati, ac, gen):
    tot = 0.0
    for i in idxs:
        r = recs[i]
        lat = r["latent"].to(DEV, DT); pe = r["prompt_embeds"].to(DEV, DT); pooled = r["pooled"].to(DEV, DT)
        t = torch.randint(0, 1000, (1,), device=DEV, generator=gen).long()
        noise = torch.randn(lat.shape, device=DEV, dtype=DT, generator=gen)
        a = ac[t].view(-1, 1, 1, 1).to(DT)
        noisy = a.sqrt() * lat + (1 - a).sqrt() * noise
        pred = unet(noisy, t, encoder_hidden_states=pe,
                    added_cond_kwargs={"text_embeds": pooled, "time_ids": ati.to(DEV, DT)}).sample
        tot = tot + F.mse_loss(pred.float(), noise.float())
    return tot / len(idxs)


@torch.no_grad()
def probe(unet, recs, idxs, ati, ac, reps, seed):
    """Deterministic objective estimate (fixed t/noise) — low variance."""
    unet.eval(); g = torch.Generator(device=DEV); accs = []
    for r in range(reps):
        g.manual_seed(seed + r); accs.append(float(loss_on_batch(unet, recs, idxs, ati, ac, g)))
    unet.train(); return sum(accs) / len(accs)


def make_optimizer(name, params, lr):
    from kaon import Adakaon, AdaMuon
    md = {"adamuon": "bfloat16", "adamuon_int8": "int8", "adamuon_4bit": "4bit"}
    if name.startswith("adamuon"):
        return AdaMuon(params, lr=lr, betas=(0.95, 0.999), clip_threshold=1.0, ns_steps=2,
                       cautious=True, momentum_dtype=md[name])
    if name == "adakaon":
        return Adakaon(params, lr=lr, betas=(0.9, 0.999), cautious=True, momentum_dtype="bfloat16")
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999))
    if name == "adamw_fused":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), fused=True)
    if name == "adafactor":
        return torch.optim.Adafactor(params, lr=lr)
    raise ValueError(name)


def opt_state_bytes(opt):
    b = 0
    for st in opt.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                b += v.numel() * v.element_size()
    return b


def train(unet, name, lr, seed, steps, recs, tr, va, ati, ac, every, bs, rank, cosine):
    params = reinit_lora(unet, seed, rank)
    opt = make_optimizer(name, params, lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps) if cosine else None
    g = torch.Generator(device=DEV); g.manual_seed(seed + 12345)
    traj = []; cum = 0.0; nparam = sum(p.numel() for p in params)
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    traj.append((0, probe(unet, recs, tr, ati, ac, 6, 700), probe(unet, recs, va, ati, ac, 4, 999), 0.0))
    it = 0
    while it < steps:
        if DEV == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(every):
            bidx = [tr[(it * bs + j) % len(tr)] for j in range(bs)]
            loss = loss_on_batch(unet, recs, bidx, ati, ac, g)
            opt.zero_grad(); loss.backward(); opt.step()
            if sched:
                sched.step()
            it += 1
        if DEV == "cuda":
            torch.cuda.synchronize()
        cum += (time.time() - t0) * 1000
        traj.append((it, probe(unet, recs, tr, ati, ac, 6, 700), probe(unet, recs, va, ati, ac, 4, 999), cum))
    peak = (torch.cuda.max_memory_allocated() / 1e6) if DEV == "cuda" else 0.0
    return {"traj": traj, "ms_step": cum / steps, "peak_mb": peak,
            "state_Bperparam": opt_state_bytes(opt) / nparam}


def cmd_ab(A):
    cache = _env("ADAMUON_CACHE", "./adamuon_sdxl_cache")
    recs = [torch.load(f, map_location="cpu", weights_only=False)
            for f in sorted(glob.glob(os.path.join(cache, "rec_*.pt")))]
    meta = torch.load(os.path.join(cache, "meta.pt"), map_location="cpu", weights_only=False)
    if not recs:
        raise SystemExit(f"no cache in {cache}; run `precompute` first")
    n = meta["n"]; ati = meta["add_time_ids"]
    idx = list(range(n)); random.Random(1234).shuffle(idx)
    nval = max(2, n // 4); va = sorted(idx[:nval]); tr = sorted(idx[nval:])
    ac = make_alphas()
    arms = A.optims.split(",")
    print(f"SDXL LoRA A/B | n={n} train={len(tr)} val={len(va)} rank={A.rank} steps={A.steps} "
          f"seeds={A.seeds} bs={A.bs} cosine={A.cosine}", flush=True)
    unet = build_unet(A.rank)

    out = {}
    for spec in arms:
        name, lr = spec.split(":"); lr = float(lr)
        seeds = [train(unet, name, lr, sd, A.steps, recs, tr, va, ati, ac, A.every, A.bs, A.rank, A.cosine)
                 for sd in range(A.seeds)]
        npts = len(seeds[0]["traj"]); ns = len(seeds)
        avg = [(seeds[0]["traj"][k][0],
                sum(s["traj"][k][1] for s in seeds) / ns,
                sum(s["traj"][k][2] for s in seeds) / ns,
                sum(s["traj"][k][3] for s in seeds) / ns) for k in range(npts)]
        out[spec] = avg
        bt = min(p[1] for p in avg)
        print(f"\n=== {name} lr={lr:.1e} best_train={bt:.5f} ms/step={seeds[0]['ms_step']:.1f} "
              f"peak={seeds[0]['peak_mb']:.0f}MB state={seeds[0]['state_Bperparam']:.2f}B/p ===", flush=True)
        for step, trn, v, t in avg:
            print(f"  step {step:4d} train={trn:.5f} val={v:.5f} t={t / 1000:.1f}s", flush=True)

    print("\n=== TIME-TO-TARGET (deterministic train objective) ===")
    allmin = min(min(p[1] for p in avg) for avg in out.values())
    for tgt in (allmin * 1.05, allmin * 1.02, allmin * 1.005):
        line = f"  train<= {tgt:.5f}: "
        for spec, avg in out.items():
            hit = next((p for p in avg if p[1] <= tgt), None)
            reached = f"{hit[0]}st/{hit[3] / 1000:.0f}s" if hit else "never"
            line += f"{spec.split(':')[0]} {reached:>15s} | "
        print(line, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("precompute"); p.add_argument("--res", type=int, default=512); p.set_defaults(fn=cmd_precompute)
    a = sub.add_parser("ab")
    a.add_argument("--optims", type=str,
                   default="adamuon:1e-3,adamuon_int8:1e-3,adamuon_4bit:1e-3,adakaon:1e-3")
    a.add_argument("--steps", type=int, default=500); a.add_argument("--seeds", type=int, default=2)
    a.add_argument("--every", type=int, default=50); a.add_argument("--bs", type=int, default=2)
    a.add_argument("--rank", type=int, default=16); a.add_argument("--cosine", action="store_true")
    a.set_defaults(fn=cmd_ab)
    A = ap.parse_args()
    A.fn(A)


if __name__ == "__main__":
    main()
