"""The optimizer registry for the control battery.

**To add an optimizer to every ranking: add ONE entry here.** Give it its best-known
config and the LR it wants at the proxy scale (the proxy LRs are ~100x real-training LRs —
they are *relative* knobs for the synthetic benchmark, not recommendations for your data).
Then run ``battery.py`` and it appears in all the tables.

Each entry:
  make      : (params, lr) -> a torch Optimizer at the optimizer's best config
  lr        : scheduled-LR proxy value (used with the REX + progressive-resolution recipe)
  lr_const  : constant-LR value (the 'continuity' scenario; often = lr or a notch lower)
  family    : 'reference' | 'in-house' | 'published'  (for grouping in the tables)
  blurb     : one-line identity for the tables
"""
from __future__ import annotations

import torch

from kaon import (
    ADOPT,
    SAM,
    AdaBelief,
    Adakaon,
    AdamP,
    AdaMuon,
    AdaPNM,
    Lion,
    Lookahead,
    ScheduleFree,
)

OPTIMIZERS = {
    # --- reference baseline ---
    "torch.AdamW (fused)": dict(
        make=lambda p, lr: torch.optim.AdamW(p, lr=lr, betas=(0.9, 0.999), fused=True),
        lr=1.2e-3, lr_const=1.2e-3, family="reference",
        blurb="torch.optim.AdamW, fused kernel — the EXTERNAL reference (not a kaon optimizer)",
        # frozen: external torch optimizer, unaffected by kaon changes -> a full run keeps its
        # cached numbers (re-measured only if missing or the settings signature changes).
        frozen=True,
    ),
    # --- in-house (kaon family) ---
    "Adakaon-nomom": dict(
        make=lambda p, lr: Adakaon(p, lr=lr, betas=(0.0, 0.999), cautious=False, momentum_dtype="bfloat16"),
        lr=6e-4, lr_const=1.2e-3, family="in-house",
        blurb="factored Adam, no momentum (minimum VRAM, regularizing)",
    ),
    "Adakaon-bf16": dict(
        make=lambda p, lr: Adakaon(p, lr=lr, betas=(0.9, 0.999), cautious=True, momentum_dtype="bfloat16"),
        lr=1.2e-3, lr_const=1.2e-3, family="in-house",
        blurb="factored Adam, bf16 momentum (AdamW-quality, low memory)",
    ),
    "Adakaon-bf16 (fused)": dict(
        make=lambda p, lr: Adakaon(p, lr=lr, betas=(0.9, 0.999), cautious=True, momentum_dtype="bfloat16", fused=True),
        lr=1.2e-3, lr_const=1.2e-3, family="in-house",
        blurb="Adakaon-bf16 with the Triton-fused step — same quality/memory, fused LoRA speed",
    ),
    # --- published algorithms on the kaon backend ---
    "Lion": dict(
        make=lambda p, lr: Lion(p, lr=lr, betas=(0.95, 0.98), cautious=True, momentum_dtype="bfloat16"),
        lr=3e-4, lr_const=6e-4, family="published",
        blurb="sign-momentum, no 2nd moment (lightest state)",
    ),
    "AdaPNM": dict(
        make=lambda p, lr: AdaPNM(p, lr=lr, betas=(0.8, 0.999), beta0=0.5, cautious=True, momentum_dtype="bfloat16"),
        lr=2.4e-3, lr_const=2.4e-3, family="published",
        blurb="positive-negative momentum (best generalization / constant-LR)",
        # optional: constructor knobs the profiler A/Bs ("what does it like?")
        variants={
            "cautious=on": lambda p, lr: AdaPNM(p, lr=lr, betas=(0.8, 0.999), beta0=0.5, cautious=True, momentum_dtype="bfloat16"),
            "cautious=off": lambda p, lr: AdaPNM(p, lr=lr, betas=(0.8, 0.999), beta0=0.5, cautious=False, momentum_dtype="bfloat16"),
            "beta0=0 (PNM off)": lambda p, lr: AdaPNM(p, lr=lr, betas=(0.8, 0.999), beta0=0.0, cautious=True, momentum_dtype="bfloat16"),
            "beta0=1 (full PNM)": lambda p, lr: AdaPNM(p, lr=lr, betas=(0.8, 0.999), beta0=1.0, cautious=True, momentum_dtype="bfloat16"),
        },
    ),
    "AdaPNM (fused)": dict(
        make=lambda p, lr: AdaPNM(p, lr=lr, betas=(0.8, 0.999), beta0=0.5, cautious=True, momentum_dtype="bfloat16", fused=True),
        lr=2.4e-3, lr_const=2.4e-3, family="published",
        blurb="AdaPNM with the Triton-fused step — gap champion, now fast (no speed Achilles heel)",
    ),
    "AdaMuon": dict(
        make=lambda p, lr: AdaMuon(p, lr=lr, betas=(0.95, 0.999), ns_steps=2, cautious=True, momentum_dtype="int8"),
        lr=2.4e-3, lr_const=2.4e-3, family="published",
        blurb="orthogonalized momentum + factored 2nd moment (convergence)",
    ),
    # --- candidates-v2 (under evaluation; tuned on the C96/N800 proxy, ranked by held-out loss) ---
    "AdaBelief": dict(
        # best config: beta2=0.999 (the loss-tuned 0.95 was gap-worst at scale; 0.999 is the better
        # all-rounder -- similar loss, ~35% lower gap). This IS AdaBelief's best overall config.
        make=lambda p, lr: AdaBelief(p, lr=lr, betas=(0.9, 0.999), cautious=True, momentum_dtype="bfloat16"),
        lr=1e-3, lr_const=1e-3, family="published",
        blurb="Adam on the variance of (g-m) — belief in the gradient (light, generalizing)",
    ),
    "ADOPT": dict(
        make=lambda p, lr: ADOPT(p, lr=lr, betas=(0.9, 0.9999), cautious=True, momentum_dtype="bfloat16"),
        lr=4e-3, lr_const=4e-3, family="published",
        blurb="modified Adam, converges with any beta2 (v-lag + normalize-then-momentum)",
    ),
    "AdamP": dict(
        make=lambda p, lr: AdamP(p, lr=lr, weight_decay=0.05, cautious=True, momentum_dtype="bfloat16"),
        lr=1e-3, lr_const=1e-3, family="published",
        blurb="AdamW minus the radial update on scale-invariant weights (gap-oriented)",
    ),
    # --- wrappers over Adakaon (gap-frontier techniques; ~2x cost for SAM) ---
    "Lookahead": dict(
        make=lambda p, lr: Lookahead(p, lr=lr, k=5, alpha=0.5, betas=(0.9, 0.999),
                                     cautious=True, momentum_dtype="bfloat16"),
        lr=2.4e-3, lr_const=2.4e-3, family="published",
        blurb="k-step slow-weight averaging over Adakaon (weight-averaging regularizer)",
    ),
    "SAM": dict(
        make=lambda p, lr: SAM(p, lr=lr, rho=0.05, betas=(0.9, 0.999),
                               cautious=True, momentum_dtype="bfloat16"),
        lr=1.2e-3, lr_const=1.2e-3, family="published",
        blurb="sharpness-aware (flat minima) over Adakaon; 2x cost, lowers the gap",
    ),
    "ScheduleFree": dict(
        make=lambda p, lr: ScheduleFree(p, lr=lr, warmup_steps=100, cautious=True, momentum_dtype="bfloat16"),
        lr=5e-3, lr_const=5e-3, family="published",
        blurb="iterate-averaging AdamW, no LR schedule (constant-LR / resumable)",
    ),
}
