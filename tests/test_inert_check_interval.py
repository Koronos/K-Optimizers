"""Diagnostic sampling must not change optimizer updates."""
import pytest
import torch

from kaon import Nekaon


def test_unsampled_check_does_not_read_weights(monkeypatch):
    p = torch.nn.Parameter(torch.ones(8, 8))
    opt = Nekaon([p], inert_check_interval=10)
    def forbidden(*args, **kwargs):
        raise AssertionError("diagnostic read on an unsampled step")
    monkeypatch.setattr(torch, "stack", forbidden)
    for _ in range(9):
        opt._warn_if_inert()
    with pytest.raises(AssertionError, match="unsampled"):
        opt._warn_if_inert()


def test_interval_preserves_updates():
    p = torch.nn.Parameter(torch.ones(8, 8))
    q = torch.nn.Parameter(p.detach().clone())
    a = Nekaon([p], inert_check_interval=1, momentum_dtype="float32")
    b = Nekaon([q], inert_check_interval=10, momentum_dtype="float32")
    for _ in range(12):
        p.grad = torch.arange(64, dtype=p.dtype).reshape_as(p) / 100
        q.grad = p.grad.clone()
        a.step()
        b.step()
        assert torch.equal(p, q)


def test_default_diagnostic_has_twenty_reads_at_most(monkeypatch):
    opt = Nekaon([torch.nn.Parameter(torch.ones(8, 8))], lr=.1)
    original = torch.stack
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, "stack", counted)
    for _ in range(250):
        opt._warn_if_inert()
    assert len(calls) == 20


@pytest.mark.parametrize("interval", [0, -1, True, 1.5])
def test_invalid_interval(interval):
    with pytest.raises(ValueError, match="positive integer"):
        Nekaon([torch.nn.Parameter(torch.ones(2))], inert_check_interval=interval)
