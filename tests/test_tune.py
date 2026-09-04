"""Unit tests for kaon.tune helpers (OOM formatting and best-selection)."""

from __future__ import annotations

from kaon.tune import _best_timing, _fmt_ms


def test_fmt_ms_none_is_oom():
    assert _fmt_ms(None) == "OOM"


def test_fmt_ms_formats_float():
    assert _fmt_ms(12.345) == "    12.3 ms/step"


def test_best_timing_excludes_none():
    results = [(500_000, None), (2_000_000, 15.0), (4_000_000, 12.5)]
    assert _best_timing(results) == (4_000_000, 12.5)


def test_best_timing_all_none_returns_none():
    assert _best_timing([(1, None), (2, None)]) is None
