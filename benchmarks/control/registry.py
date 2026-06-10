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

from kaon import Adakaon, AdaMuon, AdaPNM, AdamP, Lion, Nekaon, ScheduleFree

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
    "Nekaon": dict(
        make=lambda p, lr: Nekaon(p, lr=lr, k=1.5, betas=(0.5, 0.999), weight_decay=0.1, momentum_dtype="4bit"),
        lr=1.2e-3, lr_const=1.2e-3, family="in-house",
        blurb="Adakaon + k-step negative momentum-lookahead (zero-cost flat-minima; beta1 = regime knob)",
        # The regime dial (the profiler's A/B): beta1 picks the operating point on the
        # loss<->gap frontier at the same structural k=1.5.
        variants={
            "balanced (b1=0.5)": lambda p, lr: Nekaon(p, lr=lr, k=1.5, betas=(0.5, 0.999), weight_decay=0.1, momentum_dtype="bfloat16"),
            "gap mode (b1=0.2)": lambda p, lr: Nekaon(p, lr=lr, k=1.5, betas=(0.2, 0.999), weight_decay=0.1, momentum_dtype="bfloat16"),
            "frontier (b1=0.7)": lambda p, lr: Nekaon(p, lr=lr, k=1.5, betas=(0.7, 0.999), weight_decay=0.1, momentum_dtype="bfloat16"),
            "fidelity (b1=0.9)": lambda p, lr: Nekaon(p, lr=lr, k=1.5, betas=(0.9, 0.999), weight_decay=0.1, momentum_dtype="bfloat16"),
            "k=0 (plain Adakaon+wd)": lambda p, lr: Nekaon(p, lr=lr, k=0.0, betas=(0.5, 0.999), weight_decay=0.1, momentum_dtype="4bit"),
            "int8 momentum (1.04 B/p)": lambda p, lr: Nekaon(p, lr=lr, k=1.5, betas=(0.5, 0.999), weight_decay=0.1, momentum_dtype="int8"),
            "bf16 momentum (2.03 B/p)": lambda p, lr: Nekaon(p, lr=lr, k=1.5, betas=(0.5, 0.999), weight_decay=0.1, momentum_dtype="bfloat16"),
        },
    ),
    "Nekaon-fused": dict(
        make=lambda p, lr: Nekaon(p, lr=lr, k=1.5, betas=(0.5, 0.999), weight_decay=0.1, momentum_dtype="4bit", fused=True),
        lr=1.2e-3, lr_const=1.2e-3, family="in-house",
        blurb="Nekaon, Triton-fused inner step (same math; speed twin of Nekaon)",
    ),
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
    # Triton-fused twins of the bf16 configs above — SAME math/state, so loss/gap/memory must match
    # their non-fused twin (the parity check end-to-end); they exist to compare the fused SPEED
    # (ms/step, lora ms) before vs after. GPU-only (skipped/raise without Triton).
    "Adakaon-bf16-fused": dict(
        make=lambda p, lr: Adakaon(p, lr=lr, betas=(0.9, 0.999), cautious=True, momentum_dtype="bfloat16", fused=True),
        lr=1.2e-3, lr_const=1.2e-3, family="in-house",
        blurb="Adakaon-bf16, Triton-fused step (same math; speed twin of Adakaon-bf16)",
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
    "AdaPNM-fused": dict(
        make=lambda p, lr: AdaPNM(p, lr=lr, betas=(0.8, 0.999), beta0=0.5, cautious=True, momentum_dtype="bfloat16", fused=True),
        lr=2.4e-3, lr_const=2.4e-3, family="published",
        blurb="AdaPNM, Triton-fused step (same math; speed twin of AdaPNM)",
    ),
    "AdaMuon": dict(
        make=lambda p, lr: AdaMuon(p, lr=lr, betas=(0.95, 0.999), ns_steps=2, cautious=True, momentum_dtype="int8"),
        lr=2.4e-3, lr_const=2.4e-3, family="published",
        blurb="orthogonalized momentum + factored 2nd moment (convergence)",
    ),
    "AdamP": dict(
        make=lambda p, lr: AdamP(p, lr=lr, weight_decay=0.05, cautious=True, momentum_dtype="bfloat16"),
        lr=1e-3, lr_const=1e-3, family="published",
        blurb="AdamW minus the radial update on scale-invariant weights (gap-oriented; no fused twin)",
    ),
    "ScheduleFree": dict(
        make=lambda p, lr: ScheduleFree(p, lr=lr, betas=(0.9, 0.999), momentum_dtype="bfloat16"),
        lr=2.5e-3, lr_const=2.5e-3, family="published",
        blurb="Schedule-Free AdamW (iterate averaging, no schedule) — evaluated at its averaged x",
    ),
}
