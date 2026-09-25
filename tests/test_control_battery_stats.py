"""The control battery's statistics/plumbing layer — no GPU, no training.

Covers the guarantees the battery makes about its *evidence* rather than about any optimizer:
confidence intervals over the per-seed runs, backward compatibility with cache entries measured
before per-seed retention, the ``--out TAG`` side-cache, the AC-power gate, and the fact that the
seed count is part of the settings signature.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "control_battery",
    Path(__file__).resolve().parents[1] / "benchmarks" / "control" / "battery.py",
)
battery = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(battery)

FP = "f1775030db99" + "0" * 20


def entry(*, gap, te, cgap=0.005, cte=0.09, per_seed=None, sig, ms=10.0, opt_ms=None, blurb="b"):
    """A cache entry shaped like ``measure()`` output; ``per_seed=None`` = a legacy (pre-CI) row."""
    m = dict(te=te, gap=gap, tr=te - gap, ms=ms, bpp=2.0, cgap=cgap, cte=cte, lms=3.0,
             traj=[[100, te + 0.05], [200, te]], family="in-house", blurb=blurb, sig=sig)
    if per_seed is not None:
        m["per_seed"] = per_seed
    if opt_ms is not None:
        m["opt_ms"] = opt_ms
    return m


def seeded(values):
    """per_seed block for the metrics the tables read, from {metric: [per-seed values]}."""
    return {k: list(v) for k, v in values.items()}


# ----------------------------- confidence intervals -----------------------------
def test_ci95_is_the_student_t_interval() -> None:
    # xs = 1..5: mean 3, sd 1.5811, sem 0.70711, t(4, .975) = 2.776 -> 1.9629
    assert battery.ci95([1, 2, 3, 4, 5]) == pytest.approx(2.776 * 1.5811388 / 5 ** 0.5, rel=1e-4)
    # n=2 uses the (very wide) t(1) = 12.706 — the reason 2 seeds rarely separate anything
    assert battery.ci95([0.010, 0.012]) == pytest.approx(12.706 * 0.001414214 / 2 ** 0.5, rel=1e-4)
    # a constant sample has zero spread, not a missing interval
    assert battery.ci95([0.5, 0.5, 0.5]) == 0.0


def test_ci95_is_none_without_replication() -> None:
    assert battery.ci95([0.1]) is None       # one seed: a point estimate, no interval
    assert battery.ci95([]) is None
    assert battery.ci95(None) is None        # legacy entry: no per_seed at all


def test_entry_ci_and_overlap_semantics() -> None:
    a = entry(gap=0.011, te=0.10, sig="s", per_seed=seeded({"gap": [0.010, 0.012]}))
    b = entry(gap=0.014, te=0.10, sig="s", per_seed=seeded({"gap": [0.013, 0.015]}))
    far = entry(gap=0.040, te=0.10, sig="s", per_seed=seeded({"gap": [0.0395, 0.0405]}))
    legacy = entry(gap=0.012, te=0.10, sig="s")

    assert battery.entry_ci(a, "gap") == pytest.approx(battery.ci95([0.010, 0.012]))
    assert battery.entry_ci(legacy, "gap") is None
    assert battery.overlaps(a, b, "gap")          # within noise -> a tie, not a win
    assert not battery.overlaps(a, far, "gap")    # separated -> a real difference
    # a row with no per-seed data makes no statistical claim and is never declared a tie
    assert not battery.overlaps(a, legacy, "gap")
    assert not battery.overlaps(legacy, a, "gap")


# ----------------------------- rendering -----------------------------
def _store(sig):
    """Four entries: three with per-seed data (two of them tied on gap) and one legacy row."""
    return {
        "New-leader": entry(
            gap=0.011, te=0.100, cgap=0.004, cte=0.090, ms=10.0, opt_ms=6.0, sig=sig,
            per_seed=seeded({"gap": [0.010, 0.012], "te": [0.099, 0.101],
                             "cgap": [0.0035, 0.0045], "cte": [0.089, 0.091]})),
        "New-tied": entry(
            gap=0.014, te=0.102, cgap=0.0045, cte=0.092, ms=11.0, opt_ms=7.0, sig=sig,
            per_seed=seeded({"gap": [0.013, 0.015], "te": [0.101, 0.103],
                             "cgap": [0.0040, 0.0050], "cte": [0.091, 0.093]})),
        "New-worse": entry(
            gap=0.040, te=0.120, cgap=0.030, cte=0.130, ms=12.0, opt_ms=8.0, sig=sig,
            per_seed=seeded({"gap": [0.0395, 0.0405], "te": [0.119, 0.121],
                             "cgap": [0.0295, 0.0305], "cte": [0.129, 0.131]})),
        "Legacy": entry(gap=0.020, te=0.110, cgap=0.010, cte=0.100, ms=13.0, sig=sig),
    }


def rows_of(md, heading):
    """The `{optimizer: table row}` map of ONE section (row names repeat across tables)."""
    section = md.split(heading, 1)[1].split("\n## ", 1)[0]
    return {ln.split("|")[2].strip(): ln for ln in section.splitlines() if ln.startswith("|")}


def test_render_mixes_legacy_and_per_seed_entries(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(battery, "HERE", str(tmp_path))
    cfg = battery.build_cfg(False, FP)
    battery.render(_store(cfg["sig"]), cfg, False)

    md = (tmp_path / "RANKINGS.md").read_text(encoding="utf-8")
    quality = rows_of(md, "## 🎯 Loss × generalization")

    # new entries carry `mean ± CI95`; the legacy row keeps the old bare-mean format
    assert "+0.0110 ± 0.0127" in quality["New-leader"]
    assert "0.1000 ± 0.0127" in quality["New-leader"]
    assert "±" not in quality["Legacy"]
    assert "+0.0200" in quality["Legacy"]
    # the tie marker lands on the row overlapping the leader, and only there
    assert quality["New-tied"].split("|")[1].strip() == "2 ≈"
    assert quality["Legacy"].split("|")[1].strip() == "3"
    assert quality["New-worse"].split("|")[1].strip() == "4"
    # the interval legend appears once there is any interval to read
    assert "tied within the CI" in md
    # the continuity table gets the same treatment (ranked by const-LR gap)
    continuity = rows_of(md, "## 🔁 Continuity")
    assert "+0.0040 ± 0.0064" in continuity["New-leader"]
    assert "±" not in continuity["Legacy"]
    # opt-only timing is reported next to ms/step with its share
    speed = rows_of(md, "## ⚡ Per-iteration speed")
    assert "opt-only ms/step" in md
    assert "6.00 (60%)" in speed["New-leader"]
    assert speed["Legacy"].rstrip().endswith("— |")  # legacy row has no opt-only number


def test_render_of_a_legacy_only_cache_is_unchanged(tmp_path, monkeypatch) -> None:
    """A cache with no per-seed data must render exactly as it always did: no ±, no ≈, no legend,
    no opt-only column (this is what keeps the historical RANKINGS.md reproducible)."""
    monkeypatch.setattr(battery, "HERE", str(tmp_path))
    cfg = battery.build_cfg(False, FP)
    store = {n: m for n, m in _store(cfg["sig"]).items()}
    for m in store.values():
        m.pop("per_seed", None)
        m.pop("opt_ms", None)
    battery.render(store, cfg, False)

    md = (tmp_path / "RANKINGS.md").read_text(encoding="utf-8")
    assert "±" not in md
    assert "≈" not in md
    assert "tied within the CI" not in md
    assert "opt-only ms/step" not in md


def test_render_stamps_the_electrical_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(battery, "HERE", str(tmp_path))
    cfg = battery.build_cfg(False, FP)
    store = _store(cfg["sig"])
    for m in store.values():
        if "per_seed" in m:
            m["meta"] = dict(timestamp="2026-09-18T10:00:00-06:00", kaon_version="0.7.14",
                             commit="abc1234", gpu="NVIDIA RTX 3000 Ada Generation Laptop GPU",
                             power="[N/A], 60.00 W, 35.00 W, 3105 MHz", battery_status=2, on_ac=True)
    battery.render(store, cfg, False)

    md = (tmp_path / "RANKINGS.md").read_text(encoding="utf-8")
    assert "Measured on **AC**" in md
    assert "60.00 W, 35.00 W" in md
    assert "0.7.14" in md and "abc1234" in md


# ----------------------------- --out TAG side-cache -----------------------------
def test_out_tag_never_touches_the_default_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(battery, "HERE", str(tmp_path))
    assert battery.store_path(False) == f"{tmp_path}/results.json"
    assert battery.store_path(True) == f"{tmp_path}/results_quick.json"
    assert battery.store_path(False, "evidence") == f"{tmp_path}/results_evidence.json"
    assert battery.store_path(True, "evidence") == f"{tmp_path}/results_evidence.json"
    assert battery.rankings_path() == f"{tmp_path}/RANKINGS.md"
    assert battery.rankings_path("evidence") == f"{tmp_path}/RANKINGS_evidence.md"

    # pre-existing default files that a tagged run must leave byte-identical
    (tmp_path / "results.json").write_text("{}", encoding="utf-8")
    (tmp_path / "RANKINGS.md").write_text("HISTORICAL", encoding="utf-8")

    cfg = battery.build_cfg(False, FP)
    store = _store(cfg["sig"])
    battery.save_store(store, False, "evidence")
    battery.render(store, cfg, False, "evidence")

    assert (tmp_path / "results.json").read_text(encoding="utf-8") == "{}"
    assert (tmp_path / "RANKINGS.md").read_text(encoding="utf-8") == "HISTORICAL"
    assert (tmp_path / "results_evidence.json").exists()
    tagged = (tmp_path / "RANKINGS_evidence.md").read_text(encoding="utf-8")
    assert "results_evidence.json" in tagged  # the header points at its own cache
    assert battery.load_store(False, "evidence").keys() == store.keys()


# ----------------------------- the AC-power gate -----------------------------
def test_measuring_aborts_when_not_on_ac(monkeypatch) -> None:
    monkeypatch.delenv("KAON_BATTERY_ALLOW_DC", raising=False)
    monkeypatch.setattr(battery, "battery_status", lambda: 1)  # 1 = discharging
    with pytest.raises(SystemExit) as e:
        battery.require_ac()
    assert "AC power" in str(e.value)

    monkeypatch.setattr(battery, "battery_status", lambda: None)  # unreadable -> still refuse
    with pytest.raises(SystemExit):
        battery.require_ac()


def test_ac_gate_passes_on_ac_and_can_be_overridden(monkeypatch) -> None:
    monkeypatch.delenv("KAON_BATTERY_ALLOW_DC", raising=False)
    monkeypatch.setattr(battery, "battery_status", lambda: 2)  # 2 = on AC
    ps = battery.require_ac()
    assert ps["on_ac"] is True and ps["battery_status"] == 2

    monkeypatch.setattr(battery, "battery_status", lambda: 1)
    monkeypatch.setenv("KAON_BATTERY_ALLOW_DC", "1")
    assert battery.require_ac()["on_ac"] is False  # explicit opt-in, no abort


def test_run_metadata_records_version_power_and_time(monkeypatch) -> None:
    monkeypatch.setattr(battery, "battery_status", lambda: 2)
    meta = battery.run_metadata()
    assert meta["timestamp"] and "T" in meta["timestamp"]
    assert meta["on_ac"] is True
    for k in ("kaon_version", "commit", "gpu", "power", "device"):
        assert k in meta


# ----------------------------- settings signature -----------------------------
def test_seed_count_is_part_of_the_signature() -> None:
    assert "_s2_" in battery.build_cfg(False, FP)["sig"]
    assert "_s5_" in battery.build_cfg(False, FP, 5)["sig"]
    assert battery.build_cfg(False, FP, 5)["seeds"] == 5
    # a 5-seed measurement is a different (stronger) claim: it must not be ranked with 2-seed rows
    assert battery.build_cfg(False, FP, 5)["sig"] != battery.build_cfg(False, FP, 2)["sig"]
    # --quick keeps its own defaults but still honours an explicit override
    assert battery.build_cfg(True, FP)["seeds"] == 1
    assert battery.build_cfg(True, FP, 3)["seeds"] == 3


# ----------------------------- the controlled-comparison arms -----------------------------
def _built(name):
    """``(class name, k, param_group)`` for one registry entry, on 4x4 CPU tensors."""
    import torch

    spec = battery.OPTIMIZERS[name]
    opt = spec["make"]([torch.zeros(4, 4, requires_grad=True)], spec["lr"])
    return type(opt).__name__, getattr(opt, "k", None), getattr(opt, "inner", opt).param_groups[0]


def _knobs(name):
    cls, k, g = _built(name)
    return (cls, k, g["betas"], g["weight_decay"], g["momentum_dtype"], g["cautious"])


@pytest.mark.parametrize(
    ("arm", "base", "knob"),
    [("Nekaon-bf16-fused", "Nekaon-fused", 4),        # momentum_dtype
     ("Nekaon-k0-4bit-fused", "Nekaon-fused", 1),     # k
     ("Adakaon-4bit-fused", "Adakaon-bf16-fused", 4)],  # momentum_dtype
)
def test_each_control_arm_differs_from_its_base_in_exactly_one_knob(arm, base, knob) -> None:
    """These arms exist to attribute a delta. An arm that moved two knobs (or none) would
    make its pairing unreadable, which is exactly what it is here to prevent."""
    a, b = _knobs(arm), _knobs(base)
    differing = [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]
    assert differing == [knob], f"{arm} vs {base}: {a} vs {b}"


def test_the_two_four_bit_controls_are_not_the_same_optimizer_twice() -> None:
    """Nekaon-k0-4bit-fused IS Adakaon at Nekaon's betas/wd with an inert lookahead. If
    Adakaon-4bit-fused also took Nekaon's betas/wd the pair would be one optimizer measured
    twice and neither the codec nor the lookahead would be isolated by anything."""
    _, _, nek = _built("Nekaon-k0-4bit-fused")
    _, _, ada = _built("Adakaon-4bit-fused")
    assert nek["momentum_dtype"] == ada["momentum_dtype"] == "4bit"
    assert (nek["betas"], nek["weight_decay"]) != (ada["betas"], ada["weight_decay"])
    _, _, ada_bf16 = _built("Adakaon-bf16-fused")
    assert (ada["betas"], ada["weight_decay"]) == (ada_bf16["betas"], ada_bf16["weight_decay"])


def test_control_arms_inherit_the_learning_rate_of_the_base_they_are_paired_with() -> None:
    """A re-tuned LR would stop the arm from being a controlled comparison."""
    for arm, base in (("Nekaon-bf16-fused", "Nekaon-fused"),
                      ("Nekaon-k0-4bit-fused", "Nekaon-fused"),
                      ("Adakaon-4bit-fused", "Adakaon-bf16-fused")):
        for key in ("lr", "lr_const"):
            assert battery.OPTIMIZERS[arm][key] == battery.OPTIMIZERS[base][key], (arm, key)
