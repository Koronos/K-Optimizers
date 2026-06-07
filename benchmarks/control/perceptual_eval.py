"""Perceptual-proxy evaluation — sample quality + memorization for every registry optimizer.

A FID/KID-STYLE evaluation on the synthetic pixel-DDPM proxy. For each optimizer it:
  1. trains the proxy U-Net at its best config (REX + progressive curriculum, like the battery),
  2. **generates** images by running the reverse DDPM sampler from the trained model,
  3. scores the generated distribution against the HELD-OUT real images:
       - KID  : unbiased polynomial-kernel MMD^2 (stable at small sample counts) — primary,
       - FD   : a low-dim Fréchet distance (FID-analog) in the same feature space,
     both in a compact, data-appropriate feature space (radial 2-D FFT power profile + pixel
     stats — the proxy images are sums of sinusoids, so their frequency content IS the signal),
  4. measures **memorization**: mean nearest-neighbour pixel distance from each generated image
     to the 32 TRAIN images (lower = closer to copying train), with the real test set's
     train-distance as the "natural" reference,
  5. saves an 8x8 grid of generated samples (the visual A/B).

This is the faithful *idea* of FID on what we can actually run: it samples and measures the
generated distribution + memorization, a real step beyond the eps-MSE gap. It is NOT the real
SDXL/Flux LoRA FID (that needs the base model + a real dataset + per-optimizer LoRA training and
a much larger compute budget). The metrics here rank distributional sample quality on the proxy;
the real perceptual judge is still FID/KID + CLIP + a human A/B on a real LoRA.

    python perceptual_eval.py                 # all registry optimizers
    python perceptual_eval.py --only ADOPT,SAM,Lookahead
    python perceptual_eval.py --quick         # tiny smoke (2 optims, few samples/steps)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import time

import numpy as np
import torch

REPO = "/media/koronos/arca/repos/K-Optimizers"
HERE = f"{REPO}/benchmarks/control"


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


H = _load("harness", f"{REPO}/benchmarks/proxy/harness.py")
D = _load("dataset", f"{REPO}/benchmarks/proxy/dataset.py")
REG = _load("registry", f"{HERE}/registry.py")
B = _load("battery", f"{HERE}/battery.py")
OPTIMIZERS = REG.OPTIMIZERS
DEV = H.DEV


# ----------------------------- train one model (returns the net) -----------------------------
def train_model(make, lr, *, schedule, seq, seed, data, tr, ac, channels, bs, n):
    """Train the proxy U-Net (Schedule-Free / SAM aware) and return it ready to sample from."""
    torch.manual_seed(seed)
    if DEV == "cuda":
        torch.cuda.manual_seed_all(seed)
    net = H.UNet(C=channels).to(DEV).to(H.DT)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = make(params, lr)
    sf = callable(getattr(opt, "train", None)) and callable(getattr(opt, "eval", None))
    sam = callable(getattr(opt, "first_step", None)) and callable(getattr(opt, "second_step", None))
    if sf:
        opt.train()
    g = torch.Generator(device=DEV); g.manual_seed(seed + 12345)
    pos = 0
    for it, Rr in enumerate(seq):
        mult = B.rex(it / n) if schedule == "rex" else 1.0
        for pg in opt.param_groups:
            pg["lr"] = lr * mult
        idx = torch.tensor([tr[(pos + j) % len(tr)] for j in range(bs)], device=DEV); pos += bs
        if sam:
            gstate = g.get_state()
            opt.zero_grad(); H.batch_loss(net, data[Rr], idx, ac, g).backward()
            opt.first_step(zero_grad=True)
            g.set_state(gstate)
            H.batch_loss(net, data[Rr], idx, ac, g).backward()
            opt.second_step(zero_grad=True)
        else:
            opt.zero_grad(); H.batch_loss(net, data[Rr], idx, ac, g).backward(); opt.step()
    if sf:
        opt.eval()  # sample at the averaged / slow weights (Schedule-Free x, Lookahead phi)
    net.eval()
    return net


# ----------------------------- DDPM ancestral sampler (respaced, cheap) -----------------------------
@torch.no_grad()
def sample_ddpm(net, m, *, num_steps=150, T=1000, H_=64, seed=0):
    """Respaced ancestral DDPM sampling (iDDPM-style): evaluate the model at ``num_steps`` of the
    original ``T`` timesteps, with the diffusion math on the respaced alpha-bars. Stochastic (keeps
    sample diversity, so mode-collapse/memorisation shows up), ~T/num_steps cheaper than full DDPM.
    Returns [m,1,H,H] on DEV."""
    full_ac = torch.cumprod(1.0 - torch.linspace(1e-4, 2e-2, T, device=DEV), 0)
    steps = torch.linspace(0, T - 1, num_steps, device=DEV).round().long().unique()  # ascending
    ac_s = full_ac[steps]
    ac_s_prev = torch.cat([torch.ones(1, device=DEV), ac_s[:-1]])
    betas_s = (1.0 - ac_s / ac_s_prev).clamp(0, 0.999)
    alphas_s = 1.0 - betas_s
    g = torch.Generator(device=DEV); g.manual_seed(1234 + seed)
    x = torch.randn(m, 1, H_, H_, device=DEV, generator=g)
    for i in range(len(steps) - 1, -1, -1):
        tt = torch.full((m,), int(steps[i]), device=DEV, dtype=torch.long)  # model sees ORIGINAL t
        eps = net(x, tt)
        a, a_bar, a_bar_prev = alphas_s[i], ac_s[i], ac_s_prev[i]
        mean = (x - (1 - a) / (1 - a_bar).sqrt() * eps) / a.sqrt()
        if i > 0:
            var = betas_s[i] * (1 - a_bar_prev) / (1 - a_bar)
            x = mean + var.sqrt() * torch.randn(m, 1, H_, H_, device=DEV, generator=g)
        else:
            x = mean
    return x


# ----------------------------- features (radial FFT profile + pixel stats) -----------------------------
def _radial_bins(H_, nbins):
    yy, xx = torch.meshgrid(torch.arange(H_) - H_ // 2, torch.arange(H_) - H_ // 2, indexing="ij")
    r = torch.sqrt(xx.float() ** 2 + yy.float() ** 2)
    r = r / r.max() * (nbins - 1)
    return r.round().long().clamp(0, nbins - 1)


@torch.no_grad()
def features(imgs, nbins=24):
    """Compact, data-appropriate features: radial 2-D FFT power profile + per-image pixel stats."""
    imgs = imgs.to(DEV).float()
    m, _, H_, _ = imgs.shape
    spec = torch.fft.fftshift(torch.fft.fft2(imgs[:, 0]), dim=(-2, -1)).abs()  # [m,H,H]
    spec = torch.log1p(spec)
    bins = _radial_bins(H_, nbins).to(DEV).reshape(-1)
    flat = spec.reshape(m, -1)
    prof = torch.zeros(m, nbins, device=DEV)
    prof.index_add_(1, bins, flat)
    counts = torch.bincount(bins, minlength=nbins).clamp(min=1).float()
    prof = prof / counts  # [m, nbins] mean log-power per radial bin
    px = imgs.reshape(m, -1)
    stats = torch.stack([px.mean(1), px.std(1), px.amin(1), px.amax(1)], dim=1)  # [m,4]
    out = torch.cat([prof, stats], dim=1).cpu().numpy()  # [m, nbins+4]
    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)


def pixel_std(imgs):
    """Mean per-image pixel std (a scale/contrast diagnostic; the proxy reals sit near ~1)."""
    return float(imgs.reshape(len(imgs), -1).float().std(1).mean())


# ----------------------------- metrics -----------------------------
def _sq_dists(X, Y):
    return (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2 * X @ Y.T


def real_gamma(real_feat):
    """RBF bandwidth fixed from the REAL set (median heuristic) — used for ALL models so a
    mis-scaled model gets a BOUNDED 'very different' MMD (~1), not an outlier-driven blow-up."""
    d2 = _sq_dists(real_feat, real_feat)
    med = np.median(d2[d2 > 0])
    return 1.0 / (med + 1e-12)


def mmd(X, Y, gamma):
    """Unbiased Gaussian-kernel MMD^2 with a FIXED bandwidth. Bounded + outlier-robust (far points
    contribute ~0, not ^3). Lower = closer distributions; ~0 when X,Y match."""
    m, n = len(X), len(Y)
    Kxx = np.exp(-gamma * _sq_dists(X, X))
    Kyy = np.exp(-gamma * _sq_dists(Y, Y))
    Kxy = np.exp(-gamma * _sq_dists(X, Y))
    sxx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    syy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    return float(sxx + syy - 2 * Kxy.mean())


def frechet(X, Y):
    """Low-dim Fréchet distance (FID-analog) via eigenvalues of C1@C2 (no scipy)."""
    mu1, mu2 = X.mean(0), Y.mean(0)
    C1, C2 = np.cov(X, rowvar=False), np.cov(Y, rowvar=False)
    diff = mu1 - mu2
    eig = np.linalg.eigvals(C1 @ C2)
    tr_sqrt = np.sqrt(np.abs(eig)).sum()
    return float(diff @ diff + np.trace(C1) + np.trace(C2) - 2 * tr_sqrt)


def nn_pixel_dist(a_imgs, b_imgs):
    """Mean over a of the min L2 pixel distance to b (memorization probe). Lower = closer to b."""
    a = a_imgs.reshape(len(a_imgs), -1).float()
    b = b_imgs.reshape(len(b_imgs), -1).float().to(a.device)
    d = torch.cdist(a, b)  # [|a|,|b|]
    return float(d.min(dim=1).values.mean())


# ----------------------------- sample grid -----------------------------
def save_grid(imgs, path, nrow=8):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return False
    n = min(nrow * nrow, len(imgs))
    x = imgs[:n, 0].cpu().numpy()
    x = (x - x.min()) / (np.ptp(x) + 1e-8)
    fig, axes = plt.subplots(nrow, nrow, figsize=(nrow, nrow))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            ax.imshow(x[i], cmap="gray", vmin=0, vmax=1)
    fig.subplots_adjust(0, 0, 1, 1, 0.02, 0.02)
    fig.savefig(path, dpi=64)
    plt.close(fig)
    return True


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--m", type=int, default=192)
    ap.add_argument("--steps", type=int, default=150)
    a = ap.parse_args()

    # cheap-but-faithful defaults: C=96 (half the FLOPs of the C=128 battery), N=1500, 192 samples,
    # 150 respaced sampling steps. All <=~3GB VRAM with batch 8 train / 192 sample.
    C = 64 if a.quick else 96
    N = 400 if a.quick else a.n
    M = 64 if a.quick else a.m
    STEPS = 60 if a.quick else a.steps

    ds = D.build_proxy_dataset()
    data = {k: v.to(DEV).to(H.DT) for k, v in ds["DATA"].items()}
    tr, te = ds["TR"], ds["TE"]
    ac = H.make_alphas()
    seq = B.seq_prog(N)

    real = data[64][te]                       # 96 held-out reals (the target distribution)
    train_imgs = data[64][tr]                 # 32 train images (memorization reference)
    real_raw = features(real)
    # z-normalise features by the REAL set's per-dim mean/std so the polynomial kernel is
    # well-scaled (otherwise the raw log-FFT magnitudes dominate and KID is uninterpretable).
    mu, sd = real_raw.mean(0), real_raw.std(0) + 1e-6
    def norm(f):
        return (f - mu) / sd
    real_feat = norm(real_raw)
    gamma = real_gamma(real_feat)                 # fixed RBF bandwidth from the real set
    real_std = pixel_std(real)
    # natural references: a real-test split vs itself (MMD floor) and test->train distance
    half = len(real) // 2
    mmd_floor = mmd(real_feat[:half], real_feat[half:], gamma)
    test_to_train = nn_pixel_dist(real, train_imgs)

    names = [s.strip() for s in a.only.split(",")] if a.only else list(OPTIMIZERS)
    if a.quick and not a.only:
        names = ["Adakaon-bf16", "ADOPT"]

    os.makedirs(f"{HERE}/samples", exist_ok=True)
    print(f"PERCEPTUAL eval | C={C} N={N} M={M} steps={STEPS} | MMD floor (real vs real)="
          f"{mmd_floor:.4f} | real pixel-std={real_std:.3f} | test->train NN={test_to_train:.3f}\n",
          flush=True)

    rows = {}
    for name in names:
        if name not in OPTIMIZERS:
            print(f"  {name}: not in registry — skipped", flush=True); continue
        spec = OPTIMIZERS[name]
        try:
            t0 = time.time()
            random.seed(0)
            net = train_model(spec["make"], spec["lr"], schedule="rex", seq=seq, seed=0,
                              data=data, tr=tr, ac=ac, channels=C, bs=8, n=N)
            gen = sample_ddpm(net, M, num_steps=STEPS)
            gf = norm(features(gen))
            row = dict(
                mmd=mmd(gf, real_feat, gamma),
                fd=frechet(gf, real_feat),
                mem=nn_pixel_dist(gen, train_imgs),
                std_ratio=pixel_std(gen) / real_std,
                sec=time.time() - t0,
            )
            rows[name] = row
            save_grid(gen, f"{HERE}/samples/{name.replace(' ', '_').replace('(', '').replace(')', '')}.png")
            print(f"  {name:16s} MMD={row['mmd']:.4f}  FD={row['fd']:.3f}  "
                  f"mem={row['mem']:.2f}  std×real={row['std_ratio']:.2f}  ({row['sec']:.0f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {name:16s} FAILED: {type(e).__name__}: {e}", flush=True)

    out = dict(mmd_floor=mmd_floor, test_to_train=test_to_train, real_std=real_std,
               cfg=dict(C=C, N=N, M=M, steps=STEPS), rows=rows)
    with open(f"{HERE}/perceptual_results.json", "w") as f:
        json.dump(out, f, indent=1)

    # ranked table
    if rows:
        order = sorted(rows, key=lambda n: rows[n]["mmd"])
        print("\n== RANKED by MMD (sample-quality / generalization; lower=better) ==", flush=True)
        print(f"{'#':>2}  {'optimizer':18s} {'MMD':>8} {'FD':>9} {'mem':>7} {'std×real':>9}", flush=True)
        for i, n in enumerate(order):
            r = rows[n]
            print(f"{i+1:>2}  {n:18s} {r['mmd']:>8.4f} {r['fd']:>9.3f} {r['mem']:>7.2f} "
                  f"{r['std_ratio']:>9.2f}", flush=True)
        print(f"\nrefs: MMD floor (real vs real) = {mmd_floor:.4f} (~0=identical)  |  "
              f"natural test->train NN = {test_to_train:.2f}  (mem < this ⇒ nearer train than real ⇒ "
              f"memorizing)  |  std×real ~1 = correctly scaled samples", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
