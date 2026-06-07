"""K-Optimizers (``kaon``) — memory-efficient PyTorch optimizers for diffusion.

Optimizers:
    Adakaon: conv-aware factored optimizer (AdamW-quality at Adafactor memory).
    Muon: orthogonalized-momentum optimizer with an AdamW fallback (hybrid).
    AdaMuon: orthogonalized momentum + factored quantized variance (Adafactor memory).
    KProdigy: memory-efficient parameter-free Prodigy (D-adaptation).
    Autofusion: parameter-free LR on Adakaon via a Mechanic tuner (freeze-to-free).
    Lion: sign-momentum (EvoLved Sign Momentum) on Adakaon's quantized-momentum backend (experimental).
    AdaPNM: Adam + positive-negative momentum on the factored/quantized backend (experimental).

Quickstart::

    from kaon import Adakaon, Muon

    optimizer = Adakaon(
        model.parameters(),
        lr=1e-4,
        betas=(0.0, 0.999),                 # beta1=0 -> no momentum (minimum VRAM)
        bf16_method="stochastic_rounding",  # no Kahan buffer, no CPU offload
    )
"""

from kaon._version import __version__
from kaon.adakaon import Adakaon
from kaon.adamuon import AdaMuon
from kaon.adapnm import AdaPNM
from kaon.autofusion import Autofusion
from kaon.kprodigy import KProdigy
from kaon.lion import Lion
from kaon.muon import Muon

__all__ = [
    "AdaMuon",
    "AdaPNM",
    "Adakaon",
    "Autofusion",
    "KProdigy",
    "Lion",
    "Muon",
    "__version__",
]
