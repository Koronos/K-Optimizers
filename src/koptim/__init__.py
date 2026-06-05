"""K-Optimizers (``koptim``) — memory-efficient PyTorch optimizers for diffusion.

Optimizers:
    Adafusion: conv-aware factored optimizer (AdamW-quality at Adafactor memory).
    Muon: orthogonalized-momentum optimizer with an AdamW fallback (hybrid).
    AdaMuon: orthogonalized momentum + factored quantized variance (Adafactor memory).
    KProdigy: memory-efficient parameter-free Prodigy (D-adaptation).
    Autofusion: parameter-free LR on Adafusion via a Mechanic tuner (freeze-to-free).

Quickstart::

    from koptim import Adafusion, Muon

    optimizer = Adafusion(
        model.parameters(),
        lr=1e-4,
        betas=(0.0, 0.999),                 # beta1=0 -> no momentum (minimum VRAM)
        bf16_method="stochastic_rounding",  # no Kahan buffer, no CPU offload
    )
"""

from koptim._version import __version__
from koptim.adafusion import Adafusion
from koptim.adamuon import AdaMuon
from koptim.autofusion import AdafusionProdigy, AdaptiveAdafusion, Autofusion
from koptim.kprodigy import KProdigy
from koptim.muon import Muon

__all__ = [
    "AdaMuon",
    "Adafusion",
    "AdafusionProdigy",
    "AdaptiveAdafusion",
    "Autofusion",
    "KProdigy",
    "Muon",
    "__version__",
]
