"""``warn_if_4bit_high_beta1`` wiring — kaon 0.7.11 audit, mechanical batch.

``kaon._momentum_codec.warn_if_4bit_high_beta1`` already existed and was already wired
into Lion (``lion.py``, see ``test_lion_warns_on_4bit_high_beta1`` in ``test_lion.py``).
This batch wires the same one-line call into every other optimizer whose momentum is an
actual beta1-decayed EMA through the 4bit codec: AdaBelief, AdamP, ADOPT, AdaPNM (using
``betas[0]``), KProdigy and AdaMuon.

Adakaon is EXCLUDED (file locked for another concurrent audit batch — see the dispatch);
it gets the same wiring when that batch lands.

ScheduleFree is deliberately EXCLUDED too, despite accepting ``momentum_dtype="4bit"``:
its quantized ``z`` buffer is a plain accumulator (``z -= lr_t * d``), not a
beta1-decayed EMA — ``beta1`` there only weights the ``y = beta1*x + (1-beta1)*z``
interpolation point, so the warning's ``1/sqrt(1-beta1**2)`` AR(1)-filter argument does
not describe ``z``'s error dynamics. Wiring it in would tie a false-positive threshold to
an unrelated hyperparameter. See ``docs/momentum.md`` for the full note.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from kaon import ADOPT, AdaBelief, AdamP, AdaMuon, AdaPNM, KProdigy

# (name, optimizer class, kwargs beyond params/momentum_dtype/betas)
_SPECS = [
    pytest.param("AdaBelief", AdaBelief, {}, id="AdaBelief"),
    pytest.param("AdamP", AdamP, {}, id="AdamP"),
    pytest.param("ADOPT", ADOPT, {}, id="ADOPT"),
    pytest.param("KProdigy", KProdigy, dict(lr=1.0), id="KProdigy"),
    pytest.param("AdaMuon", AdaMuon, {}, id="AdaMuon"),
]


def _warns_amplification(cls, kwargs, betas, momentum_dtype):
    p = [torch.nn.Parameter(torch.randn(4, 4))]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cls(p, momentum_dtype=momentum_dtype, betas=betas, **kwargs)
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    return any("amplif" in m.lower() or "1/sqrt" in m for m in msgs)


@pytest.mark.parametrize("name, cls, kwargs", _SPECS)
def test_warns_on_4bit_high_beta1(name, cls, kwargs):
    assert _warns_amplification(cls, kwargs, (0.995, 0.999), "4bit")


@pytest.mark.parametrize("name, cls, kwargs", _SPECS)
def test_no_warning_on_int8_high_beta1(name, cls, kwargs):
    assert not _warns_amplification(cls, kwargs, (0.995, 0.999), "int8")


@pytest.mark.parametrize("name, cls, kwargs", _SPECS)
def test_no_warning_on_4bit_low_beta1(name, cls, kwargs):
    assert not _warns_amplification(cls, kwargs, (0.9, 0.999), "4bit")


def test_adapnm_uses_betas0_not_beta0():
    """AdaPNM's own ``beta0`` (negative-momentum mix, unrelated) must not be confused
    with ``betas[0]`` (the EMA decay the warning is actually about)."""
    p = [torch.nn.Parameter(torch.randn(4, 4))]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # High betas[0], default (low) beta0 -> must still warn on betas[0].
        AdaPNM(p, momentum_dtype="4bit", betas=(0.995, 0.999), beta0=0.5)
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("amplif" in m.lower() or "1/sqrt" in m for m in msgs), msgs

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Low betas[0], high beta0 -> must NOT warn (beta0 is not the decay the warning covers).
        AdaPNM(p, momentum_dtype="4bit", betas=(0.8, 0.999), beta0=0.99)
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert not any("amplif" in m.lower() or "1/sqrt" in m for m in msgs), msgs
