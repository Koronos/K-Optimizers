"""Canonical, DETERMINISTIC synthetic dataset for the generalization/optimizer A/B.

This is the fixed dataset behind the generalization campaign
(`RESULTS_generalization_and_schedule.md`) and the optimizer comparisons. Registering it
here (instead of a scratch script) makes every optimizer comparison reproducible and
directly comparable across runs.

Determinism: all randomness flows through a local `torch.Generator(seed)` — it never reads
the global RNG, so `build_proxy_dataset()` returns byte-identical tensors on every call
regardless of surrounding code. (CPU float ops are deterministic; verified by
`tests/test_proxy_dataset.py` and the `--check` self-test below.)

High-frequency synthetic images at 64² (the detail target) + area-downsampled 48²/32²
(= 1024/768/512 analogs) for the resolution curriculum. Small train (overfit-prone, like a
LoRA) + large held-out test (stable generalization estimate).

Usage:
    from proxy_dataset import build_proxy_dataset
    d = build_proxy_dataset()                 # {"DATA": {64,48,32}, "TR": [...], "TE": [...]}
    # CLI: python proxy_dataset.py --save dataset.pt   (also prints a determinism fingerprint)
"""
from __future__ import annotations

import argparse
import hashlib

import torch
import torch.nn.functional as F

# Canonical parameters — DO NOT change without bumping a dataset version (it would silently
# invalidate cross-run comparisons against recorded results).
SEED = 7
N_IMAGES = 128
N_TRAIN = 32          # small train -> overfitting is possible (LoRA-like)
RESOLUTIONS = (64, 48, 32)   # 1024 / 768 / 512 analogs; 64 is the detail/eval target


def gen_hi(n: int, H: int = 64, seed: int = 0) -> torch.Tensor:
    """n grayscale 64² images = sums of 6 high-freq 2-D sinusoids + a gaussian blob.

    All randomness via the local generator `g` -> fully deterministic, global-RNG-independent.
    """
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, H), indexing="ij")
    xs = []
    for _ in range(n):
        img = torch.zeros(H, H)
        for _ in range(6):
            fx = torch.randint(2, 18, (1,), generator=g).item()
            fy = torch.randint(2, 18, (1,), generator=g).item()
            ph = torch.rand(1, generator=g).item() * 6.283
            amp = torch.rand(1, generator=g).item()
            img = img + amp * torch.sin(6.283 * (fx * xx + fy * yy) + ph)
        cx, cy = torch.rand(2, generator=g).tolist()
        img = img + 1.2 * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 0.02)
        xs.append(img)
    return torch.stack(xs)[:, None]


def _norm(t: torch.Tensor) -> torch.Tensor:
    return (t - t.mean()) / (t.std() + 1e-6)


def build_proxy_dataset(seed: int = SEED, n_images: int = N_IMAGES, n_train: int = N_TRAIN):
    """Return the canonical proxy dataset: {"DATA": {res: tensor}, "TR": idx, "TE": idx}.

    Deterministic for fixed (seed, n_images, n_train). DATA tensors are on CPU float32;
    move to device at use. 64²=detail target, 48²/32² are area-downsampled curriculum tiers.
    """
    hi = gen_hi(n_images, 64, seed)
    data = {64: _norm(hi)}
    for r in (res for res in RESOLUTIONS if res != 64):
        data[r] = _norm(F.interpolate(hi, size=r, mode="area"))
    tr = list(range(n_train))
    te = list(range(n_train, n_images))
    return {"DATA": data, "TR": tr, "TE": te, "seed": seed, "n_images": n_images}


def fingerprint(ds) -> str:
    """sha256 over the float32 bytes of every resolution (sorted) — a determinism fingerprint."""
    h = hashlib.sha256()
    for r in sorted(ds["DATA"]):
        h.update(str(r).encode())
        h.update(ds["DATA"][r].contiguous().to(torch.float32).numpy().tobytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save", type=str, default=None, help="path to save the dataset (.pt)")
    ap.add_argument("--check", action="store_true", help="regenerate twice and assert identical")
    a = ap.parse_args()
    ds = build_proxy_dataset()
    fp = fingerprint(ds)
    print(f"proxy dataset: resolutions={sorted(ds['DATA'])} train={len(ds['TR'])} "
          f"test={len(ds['TE'])} | fingerprint={fp[:16]}…")
    if a.check:
        fp2 = fingerprint(build_proxy_dataset())
        assert fp == fp2, "NON-DETERMINISTIC: fingerprints differ across calls!"
        print("determinism OK (two builds byte-identical)")
    if a.save:
        torch.save(ds, a.save)
        print(f"saved -> {a.save}")


if __name__ == "__main__":
    main()
