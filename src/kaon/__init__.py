"""K-Optimizers (``kaon``) — memory-efficient PyTorch optimizers for diffusion.

Optimizers:
    Adakaon: conv-aware factored optimizer (AdamW-quality at Adafactor memory).
    AdaMuon: orthogonalized momentum + factored quantized variance (Adafactor memory).
    KProdigy: memory-efficient parameter-free Prodigy (D-adaptation).
    Autokaon: parameter-free LR on Adakaon via a Mechanic tuner (freeze-to-free).
    Lion: sign-momentum (EvoLved Sign Momentum) on Adakaon's quantized-momentum backend (experimental).
    AdaPNM: Adam + positive-negative momentum on the factored/quantized backend (experimental).
    AdaBelief: Adam on the variance of the gradient residual (g - m) on the factored backend (candidate).
    MARS: variance-reduction corrected gradient (STORM-style) feeding AdamW on the factored backend (candidate).
    AdEMAMix: two-EMA momentum mixture (fast + slow long-horizon) on the factored backend (candidate).
    Adan: adaptive Nesterov momentum (grad + grad-difference EMAs) on the factored backend (candidate).
    ScheduleFree: Schedule-Free AdamW (iterate averaging, no LR schedule) on the factored backend (candidate).
    ADOPT: modified Adam that converges with any beta2 (v-lag + normalize-then-momentum) (candidate).
    Grams: Adam's magnitude with the update direction set by sign(current gradient) (candidate).
    AdamP: AdamW minus the radial update component on scale-invariant weights (candidate).
    Adai: adaptive per-coordinate inertia (momentum) from the global-normalized variance (candidate).
    Lookahead: k-step slow-weight averaging wrapper over Adakaon (candidate wrapper).
    SAM: Sharpness-Aware Minimization (two-pass flat-minima) wrapper over Adakaon (candidate wrapper).

Quickstart::

    from kaon import Adakaon

    optimizer = Adakaon(
        model.parameters(),
        lr=1e-4,
        betas=(0.0, 0.999),                 # beta1=0 -> no momentum (minimum VRAM)
        bf16_method="stochastic_rounding",  # no Kahan buffer, no CPU offload
    )
"""

from kaon._version import __version__
from kaon.adabelief import AdaBelief
from kaon.adai import Adai
from kaon.adakaon import Adakaon
from kaon.adamp import AdamP
from kaon.adamuon import AdaMuon
from kaon.adan import Adan
from kaon.adapnm import AdaPNM
from kaon.ademamix import AdEMAMix
from kaon.adopt import ADOPT
from kaon.autokaon import Autokaon
from kaon.grams import Grams
from kaon.kprodigy import KProdigy
from kaon.lion import Lion
from kaon.lookahead import Lookahead
from kaon.mars import MARS
from kaon.sam import SAM
from kaon.schedulefree import ScheduleFree

__all__ = [
    "ADOPT",
    "AdEMAMix",
    "AdaBelief",
    "Adai",
    "AdaMuon",
    "AdaPNM",
    "Adakaon",
    "Adan",
    "AdamP",
    "Autokaon",
    "Grams",
    "KProdigy",
    "Lion",
    "Lookahead",
    "MARS",
    "SAM",
    "ScheduleFree",
    "__version__",
]
