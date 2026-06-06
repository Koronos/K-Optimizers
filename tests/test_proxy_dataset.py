"""The canonical proxy dataset must be deterministic (byte-identical across builds) so
optimizer comparisons are reproducible and comparable to recorded results."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "proxy_dataset",
    Path(__file__).resolve().parents[1] / "benchmarks" / "adamuon" / "proxy_dataset.py",
)
proxy_dataset = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(proxy_dataset)


def test_build_is_deterministic() -> None:
    """Two independent builds are byte-identical (local-generator, global-RNG-independent)."""
    import torch

    # Perturb the global RNG between builds to prove independence from it.
    a = proxy_dataset.build_proxy_dataset()
    torch.rand(1000)
    b = proxy_dataset.build_proxy_dataset()
    assert proxy_dataset.fingerprint(a) == proxy_dataset.fingerprint(b)
    for r in a["DATA"]:
        assert torch.equal(a["DATA"][r], b["DATA"][r])


def test_shapes_and_split() -> None:
    d = proxy_dataset.build_proxy_dataset()
    assert sorted(d["DATA"]) == [32, 48, 64]
    assert d["DATA"][64].shape == (128, 1, 64, 64)
    assert d["DATA"][48].shape == (128, 1, 48, 48)
    assert d["DATA"][32].shape == (128, 1, 32, 32)
    assert len(d["TR"]) == 32 and len(d["TE"]) == 96
    assert set(d["TR"]).isdisjoint(d["TE"])  # held-out test
