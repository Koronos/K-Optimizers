"""Offline tests for the Nekaon evidence campaign driver and aggregator.

Nothing here touches a GPU or the Rengu-Flow trainer: the queue, the TOML generation,
the resume logic, the LR selection rule and the statistics are all pure functions over
files and dictionaries.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

try:  # Python 3.11+; `tomli` is the identical backport for 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - interpreter dependent
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

requires_toml = pytest.mark.skipif(tomllib is None, reason="neither tomllib nor tomli is installed")

ROOT = Path(__file__).parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Registering first keeps `dataclasses` able to resolve the defining module on 3.10.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


campaign = _load("nekaon_evidence_campaign", ROOT / "benchmarks" / "nekaon_evidence" / "anima" / "campaign.py")
aggregate = _load("nekaon_evidence_aggregate", ROOT / "benchmarks" / "nekaon_evidence" / "anima" / "aggregate.py")

PETS = "/tmp/pets/subset"


# ------------------------------------------------------------------------ the queue


def test_queue_is_21_runs_interleaved_by_arm() -> None:
    selected = {arm: 1.0e-4 for arm in campaign.ARMS}
    queue = campaign.build_queue(selected)

    assert len(queue) == 21
    assert sum(1 for spec in queue if spec.phase == "a") == 9
    assert sum(1 for spec in queue if spec.phase == "b") == 12
    # Every consecutive block of three cycles through the arms in the same order.
    for start in range(0, len(queue), len(campaign.ARMS)):
        block = queue[start : start + len(campaign.ARMS)]
        assert [spec.arm for spec in block] == list(campaign.ARMS)
        assert len({spec.seed for spec in block}) == 1
        assert len({spec.lr for spec in block}) == 1
    assert [spec.seed for spec in queue if spec.phase == "b"][::3] == list(campaign.PHASE_B_SEEDS)


def test_queue_is_deterministic_and_phase_b_unresolved_without_phase_a() -> None:
    first = [spec.as_dict() for spec in campaign.build_queue(None)]
    second = [spec.as_dict() for spec in campaign.build_queue(None)]
    assert first == second

    queue = campaign.build_queue(None)
    assert all(spec.resolved for spec in queue if spec.phase == "a")
    assert not any(spec.resolved for spec in queue if spec.phase == "b")
    assert campaign.pending(queue, ROOT / "does-not-exist") == [
        spec for spec in queue if spec.phase == "a"
    ]


def test_run_ids_are_unique_and_stable() -> None:
    queue = campaign.build_queue({arm: 2.0e-4 for arm in campaign.ARMS})
    ids = [spec.run_id for spec in queue]
    assert len(set(ids)) == len(ids)
    assert ids[0] == f"a_seed43_{campaign.lr_token(5.0e-5)}_nekaon_fused"
    assert campaign.lr_token(5.0e-5) != campaign.lr_token(1.0e-4)


# --------------------------------------------------------------- generated TOML


def _materialize(spec, tmp_path: Path) -> dict[str, Any]:
    config = campaign.materialize(spec, tmp_path, pets_root=PETS, steps=200, eval_every=100, eval_images=8)
    return tomllib.loads(config.read_text(encoding="utf-8"))


@requires_toml
def test_kaon_arms_request_fused_and_keep_the_house_configuration(tmp_path: Path) -> None:
    for arm, expected_type in (("nekaon_fused", "kaon.Nekaon"), ("adakaon_fused", "kaon.Adakaon")):
        spec = campaign.RunSpec(phase="a", arm=arm, seed=43, lr=1.0e-4)
        optimizer = _materialize(spec, tmp_path)["optimizer"]
        assert optimizer["type"] == expected_type
        assert optimizer["fused"] is True
        assert optimizer["lr"] == pytest.approx(1.0e-4)
        assert optimizer["momentum_dtype"] == "4bit"
        assert optimizer["bf16_method"] == "stochastic_rounding"
        assert optimizer["cautious"] is True
        assert optimizer["gradient_centralization"] is True
        assert optimizer["auto_lr"] is False
        assert optimizer["betas"] == [0.5, 0.999]


@requires_toml
def test_adamw_arm_keeps_its_baseline_configuration(tmp_path: Path) -> None:
    spec = campaign.RunSpec(phase="b", arm="adamw_fused", seed=45, lr=2.0e-4)
    optimizer = _materialize(spec, tmp_path)["optimizer"]
    assert optimizer["type"] == "torch.optim.AdamW"
    assert optimizer["betas"] == [0.9, 0.999]
    assert optimizer["weight_decay"] == pytest.approx(0.01)
    assert optimizer["fused"] is True
    assert "momentum_dtype" not in optimizer


@requires_toml
def test_generated_toml_matches_the_protocol_and_is_byte_identical_on_regeneration(tmp_path: Path) -> None:
    spec = campaign.RunSpec(phase="a", arm="nekaon_fused", seed=43, lr=1.0e-4)
    config_path = campaign.materialize(spec, tmp_path, pets_root=PETS)
    first = config_path.read_bytes()
    config = tomllib.loads(first.decode("utf-8"))

    assert config["max_steps"] == 200
    assert config["eval_every_n_steps"] == 100
    assert config["eval_before_first_step"] is True
    assert config["train_seed"] == 43
    assert config["lr_scheduler"] == "constant"
    assert config["adapter"]["rank"] == 16
    assert config["model"]["dtype"] == "bfloat16"
    assert config["preview"]["enabled"] is False
    assert [entry["name"] for entry in config["eval_datasets"]] == ["train_eval", "val"]
    dataset = tomllib.loads(Path(config["dataset"]).read_text(encoding="utf-8"))
    assert dataset["resolutions"] == [256]

    assert campaign.materialize(spec, tmp_path, pets_root=PETS).read_bytes() == first


def test_materialize_refuses_an_unresolved_run(tmp_path: Path) -> None:
    spec = campaign.RunSpec(phase="b", arm="nekaon_fused", seed=44, lr=None)
    with pytest.raises(ValueError, match="unresolved"):
        campaign.materialize(spec, tmp_path, pets_root=PETS)


# ----------------------------------------------------------------- resumability


def _write_record(root: Path, spec, **overrides: Any) -> Path:
    record: dict[str, Any] = {
        "schema": "nekaon-evidence-anima-run-v1",
        "run_id": spec.run_id,
        "phase": spec.phase,
        "arm": spec.arm,
        "seed": spec.seed,
        "lr": spec.lr,
        "steps": 200,
        "adapter_initial_sha256": "a" * 64,
        "power_signature": "ac/max_limit=60.00 W",
        "provenance": {"kaon_version": "0.7.14", "commit": "deadbeef"},
        "run": {
            "train_eval": {"points": [{"step": 0, "value": 0.30}, {"step": 200, "value": 0.20}]},
            "val": {"points": [{"step": 0, "value": 0.31}, {"step": 200, "value": 0.25}]},
            "gap": {"points": [{"step": 0, "value": 0.01}, {"step": 200, "value": 0.05}]},
            "bench_csv": [{"points": [{"step": 200, "active_train_seconds": 400.0, "cuda_peak_gb": 5.5}]}],
        },
    }
    record.update(overrides)
    path = campaign.results_path(root, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_completed_runs_are_skipped_on_resume(tmp_path: Path) -> None:
    selected = {arm: 1.0e-4 for arm in campaign.ARMS}
    queue = campaign.build_queue(selected)
    assert len(campaign.pending(queue, tmp_path)) == 21

    done = [queue[0], queue[1], queue[5]]
    for spec in done:
        _write_record(tmp_path, spec)

    remaining = campaign.pending(queue, tmp_path)
    assert len(remaining) == 18
    assert not any(spec.run_id in {other.run_id for other in done} for spec in remaining)
    # Order of what is left is unchanged: resuming never reshuffles the campaign.
    assert [spec.run_id for spec in remaining] == [
        spec.run_id for spec in queue if spec.run_id not in {other.run_id for other in done}
    ]


def test_manifest_records_state_version_and_rule(tmp_path: Path) -> None:
    selected = {arm: 1.0e-4 for arm in campaign.ARMS}
    queue = campaign.build_queue(selected)
    _write_record(tmp_path, queue[0])
    manifest = campaign.write_manifest(
        tmp_path, queue, selected, steps=200, eval_every=100, eval_images=8,
        resolution=256, pets_root=PETS,
    )
    assert manifest["provenance"]["kaon_version"] == campaign.kaon_version()
    assert manifest["protocol"]["lr_selection_rule"] == campaign.LR_SELECTION_RULE
    states = [entry["state"] for entry in manifest["queue"]]
    assert states[0] == "done"
    assert states.count("pending") == 20
    assert "power" in manifest
    assert json.loads((tmp_path / "campaign.json").read_text(encoding="utf-8")) == manifest


def test_unresolved_phase_b_is_reported_as_unresolved(tmp_path: Path) -> None:
    queue = campaign.build_queue(None)
    manifest = campaign.write_manifest(
        tmp_path, queue, {}, steps=200, eval_every=100, eval_images=8,
        resolution=256, pets_root=PETS,
    )
    assert [entry["state"] for entry in manifest["queue"]].count("unresolved") == 12


# ------------------------------------------------------------- LR selection rule


def _phase_a_record(arm: str, lr: float, final_val: float) -> dict[str, Any]:
    return {
        "phase": "a", "arm": arm, "lr": lr, "seed": 43,
        "run_id": f"a_seed43_{campaign.lr_token(lr)}_{arm}",
        "run": {"val": {"points": [{"step": 0, "value": 0.3}, {"step": 200, "value": final_val}]}},
    }


def test_lr_rule_picks_the_lowest_final_val() -> None:
    records = [
        _phase_a_record("nekaon_fused", 5.0e-5, 0.260),
        _phase_a_record("nekaon_fused", 1.0e-4, 0.240),
        _phase_a_record("nekaon_fused", 2.0e-4, 0.255),
    ]
    assert campaign.select_lrs(records, arms=("nekaon_fused",)) == {"nekaon_fused": 1.0e-4}


def test_lr_rule_breaks_exact_ties_toward_the_smaller_lr() -> None:
    records = [
        _phase_a_record("adamw_fused", 2.0e-4, 0.250),
        _phase_a_record("adamw_fused", 5.0e-5, 0.250),
        _phase_a_record("adamw_fused", 1.0e-4, 0.260),
    ]
    assert campaign.select_lrs(records, arms=("adamw_fused",)) == {"adamw_fused": 5.0e-5}


def test_resolve_lrs_waits_for_a_complete_phase_a(tmp_path: Path) -> None:
    queue = campaign.build_queue(None)
    phase_a = [spec for spec in queue if spec.phase == "a"]
    for spec in phase_a[:-1]:
        _write_record(tmp_path, spec)
    assert campaign.resolve_lrs(tmp_path) == {}
    _write_record(tmp_path, phase_a[-1])
    assert set(campaign.resolve_lrs(tmp_path)) == set(campaign.ARMS)


# ---------------------------------------------------------------- the aggregator


def _record(arm: str, seed: int, lr: float, *, phase: str, val: float, gap: float,
            seconds: float, peak: float, signature: str = "ac/max_limit=60.00 W",
            config: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "run_id": f"{phase}_seed{seed}_{campaign.lr_token(lr)}_{arm}",
        "phase": phase, "arm": arm, "seed": seed, "lr": lr, "steps": 200,
        "adapter_initial_sha256": f"{seed:064d}",
        "power_signature": signature,
        "provenance": {"kaon_version": "0.7.14", "commit": "deadbeef"},
        "run": {
            "train_eval": {"points": [{"step": 0, "value": 0.30}, {"step": 200, "value": val - gap}]},
            "val": {"points": [{"step": 0, "value": 0.31}, {"step": 200, "value": val}]},
            "gap": {"points": [{"step": 0, "value": 0.01}, {"step": 200, "value": gap}]},
            "bench_csv": [{"points": [{"step": 200, "active_train_seconds": seconds, "cuda_peak_gb": peak}]}],
        },
    }
    if config is not None:
        record["config"] = config
    return record


def _synthetic_campaign(nekaon_val, adamw_val, *, selected: dict[str, float] | None = None,
                        configs: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Five paired seeds; seed 43 comes from the phase-A screen at the selected LR."""
    records: list[dict[str, Any]] = []
    selected = selected or {"nekaon_fused": 1.0e-4, "adakaon_fused": 1.0e-4, "adamw_fused": 1.0e-4}
    configs = configs or {}
    for arm in campaign.ARMS:
        for lr in campaign.PHASE_A_LRS:
            # The selected LR is the best one by construction.
            penalty = 0.0 if lr == selected[arm] else 0.02
            base = {"nekaon_fused": nekaon_val[0], "adakaon_fused": 0.250, "adamw_fused": adamw_val[0]}[arm]
            records.append(_record(arm, 43, lr, phase="a", val=base + penalty, gap=0.05,
                                   seconds=400.0, peak=5.5, config=configs.get(arm)))
    for index, seed in enumerate(campaign.PHASE_B_SEEDS, start=1):
        for arm in campaign.ARMS:
            value = {"nekaon_fused": nekaon_val[index], "adakaon_fused": 0.250,
                     "adamw_fused": adamw_val[index]}[arm]
            records.append(_record(arm, seed, selected[arm], phase="b", val=value, gap=0.05,
                                   seconds=400.0 + index, peak=5.5, config=configs.get(arm)))
    return records


def test_aggregate_reports_a_clear_win_when_every_seed_agrees() -> None:
    nekaon = [0.240, 0.241, 0.239, 0.240, 0.242]
    adamw = [0.260, 0.261, 0.259, 0.262, 0.260]
    result = aggregate.summarize(_synthetic_campaign(nekaon, adamw), resamples=2000)

    assert result["selected_lrs"] == {arm: 1.0e-4 for arm in campaign.ARMS}
    assert result["seeds"]["paired"] == [43, 44, 45, 46, 47]
    assert result["seeds"]["held_out"] == [44, 45, 46, 47]
    assert result["validation"]["ok"] is True
    assert result["power"]["timings_comparable"] is True

    arm = result["arms"]["nekaon_fused"]["metrics"]["final_val"]["bootstrap"]
    assert arm["n"] == 5
    assert arm["mean"] == pytest.approx(sum(nekaon) / 5)
    assert arm["ci_low"] <= arm["mean"] <= arm["ci_high"]

    paired = result["paired"]["nekaon_fused_minus_adamw_fused"]["metrics"]["final_val"]
    assert paired["left_wins"] == 5
    assert paired["bootstrap"]["crosses_zero"] is False
    assert paired["bootstrap"]["mean"] < 0

    markdown = aggregate.to_markdown(result)
    assert "nekaon_fused − adamw_fused" in markdown
    assert "5/5" in markdown


def test_aggregate_reports_an_interval_that_crosses_zero() -> None:
    nekaon = [0.240, 0.262, 0.238, 0.265, 0.244]
    adamw = [0.260, 0.239, 0.259, 0.241, 0.258]
    result = aggregate.summarize(_synthetic_campaign(nekaon, adamw), resamples=2000)

    paired = result["paired"]["nekaon_fused_minus_adamw_fused"]["metrics"]["final_val"]
    assert paired["left_wins"] == 3
    assert paired["bootstrap"]["crosses_zero"] is True
    assert paired["bootstrap"]["ci_low"] < 0 < paired["bootstrap"]["ci_high"]

    markdown = aggregate.to_markdown(result)
    assert "| yes |" in markdown
    assert "crosses zero" in markdown


def test_bootstrap_is_deterministic_and_centred() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    first = aggregate.bootstrap_ci(values, resamples=1000)
    second = aggregate.bootstrap_ci(values, resamples=1000)
    assert first == second
    assert first["mean"] == pytest.approx(3.0)
    assert first["ci_low"] < 3.0 < first["ci_high"]
    assert first["resamples"] == 1000
    assert aggregate.bootstrap_ci([2.0])["ci_low"] is None
    assert aggregate.bootstrap_ci([]) is None


def test_aggregate_flags_a_broken_pairing() -> None:
    records = _synthetic_campaign([0.24] * 5, [0.26] * 5)
    for record in records:
        if record["seed"] == 45 and record["arm"] == "adamw_fused":
            record["adapter_initial_sha256"] = "f" * 64
            record["run"]["val"]["points"][0]["value"] = 0.9
    result = aggregate.summarize(records, resamples=500)

    assert result["validation"]["ok"] is False
    seed45 = result["validation"]["seeds"]["45"]
    assert seed45["adapter_sha_matches"] is False
    assert seed45["initial_losses"]["val"]["ok"] is False
    assert "**no**" in aggregate.to_markdown(result)


def test_aggregate_flags_runs_measured_in_a_different_electrical_state() -> None:
    records = _synthetic_campaign([0.24] * 5, [0.26] * 5)
    for record in records:
        if record["seed"] == 46:
            record["power_signature"] = "battery/max_limit=60.00 W"
    result = aggregate.summarize(records, resamples=500)

    assert result["power"]["timings_comparable"] is False
    assert set(result["power"]["signatures"]) == {"ac/max_limit=60.00 W", "battery/max_limit=60.00 W"}
    assert "Timings comparable across all runs: **no**" in aggregate.to_markdown(result)


def test_ms_per_step_follows_active_seconds() -> None:
    record = _record("nekaon_fused", 44, 1.0e-4, phase="b", val=0.24, gap=0.05, seconds=500.0, peak=5.5)
    assert aggregate.METRICS["ms_per_step"]["get"](record) == pytest.approx(2500.0)
    assert aggregate.METRICS["peak_gib"]["get"](record) == pytest.approx(5.5)


# ------------------------------------------------------ declared confounds in the report

_NEKAON_TOML = """
run_name = "anima_pets_nekaon_fused"

[optimizer]
type = "kaon.Nekaon"
lr = 1e-4
betas = [0.5, 0.999]
weight_decay = 0.1
cautious = true
gradient_centralization = true
momentum_dtype = "4bit"
fused = true
"""

_ADAMW_TOML = """
run_name = "anima_pets_adamw_fused"

[optimizer]
type = "torch.optim.AdamW"
lr = 1e-4
betas = [0.9, 0.999]
weight_decay = 0.01
fused = true
"""


def _configured_campaign(**kwargs: Any) -> list[dict[str, Any]]:
    return _synthetic_campaign(
        [0.240] * 5, [0.260] * 5,
        configs={"nekaon_fused": _NEKAON_TOML, "adamw_fused": _ADAMW_TOML},
        **kwargs,
    )


@requires_toml
def test_arm_configs_report_the_optimizer_block_each_arm_actually_ran() -> None:
    result = aggregate.summarize(_configured_campaign(), resamples=200)
    configs = result["arm_optimizer_configs"]

    assert configs["nekaon_fused"]["optimizer"]["weight_decay"] == pytest.approx(0.1)
    assert configs["nekaon_fused"]["optimizer"]["cautious"] is True
    assert configs["nekaon_fused"]["optimizer"]["betas"] == [0.5, 0.999]
    assert configs["adamw_fused"]["optimizer"]["weight_decay"] == pytest.approx(0.01)
    assert configs["adamw_fused"]["optimizer"]["betas"] == [0.9, 0.999]
    # The arm with no TOML in its records says why, instead of silently vanishing.
    assert configs["adakaon_fused"]["optimizer"] is None
    assert configs["adakaon_fused"]["reason"]

    markdown = aggregate.to_markdown(result)
    assert "## Arm configurations as run" in markdown
    assert "| `[optimizer]` key | nekaon_fused | adakaon_fused | adamw_fused |" in markdown
    # One row per key, one column per arm; the arm without a TOML shows an em dash.
    assert "| weight_decay | 0.1 | — | 0.01 |" in markdown
    assert "| betas | [0.5, 0.999] | — | [0.9, 0.999] |" in markdown
    assert "| cautious | true | — | — |" in markdown
    assert "| type | `kaon.Nekaon` | — | `torch.optim.AdamW` |" in markdown


def test_the_hyperparameter_confound_is_declared_even_without_configs() -> None:
    markdown = aggregate.to_markdown(aggregate.summarize(
        _synthetic_campaign([0.24] * 5, [0.26] * 5), resamples=200))

    assert "NOT attributable to the algorithm alone" in markdown
    # Declared up front and repeated among the limitations.
    assert markdown.count("house configuration") >= 2
    assert "## What this does not show" in markdown


# ------------------------------------------------------------ learning-rate boundary


def test_boundary_is_false_when_the_selected_lr_sits_inside_the_grid() -> None:
    result = aggregate.summarize(_synthetic_campaign([0.24] * 5, [0.26] * 5), resamples=200)

    assert all(entry["boundary"] is False for entry in result["lr_boundary"].values())
    assert result["arms"]["nekaon_fused"]["boundary"] is False
    assert result["lr_boundary"]["nekaon_fused"]["grid"] == list(campaign.PHASE_A_LRS)

    markdown = aggregate.to_markdown(result)
    assert "boundary of the grid" not in markdown
    assert "No arm selected a learning rate at an end of its screened grid" in markdown


def test_boundary_is_flagged_when_an_arm_picks_the_smallest_lr() -> None:
    selected = {"nekaon_fused": 5.0e-5, "adakaon_fused": 1.0e-4, "adamw_fused": 1.0e-4}
    result = aggregate.summarize(
        _synthetic_campaign([0.24] * 5, [0.26] * 5, selected=selected), resamples=200)

    assert result["selected_lrs"]["nekaon_fused"] == pytest.approx(5.0e-5)
    assert result["lr_boundary"]["nekaon_fused"]["boundary"] is True
    assert result["lr_boundary"]["adamw_fused"]["boundary"] is False
    assert result["arms"]["nekaon_fused"]["boundary"] is True

    markdown = aggregate.to_markdown(result)
    assert "LR selected at the boundary of the grid for `nekaon_fused`" in markdown
    assert "the optimum is not bracketed" in markdown
    assert "lower bound" in markdown
    # The flag rides along with the arm's LR wherever the LR is printed.
    assert "5e-05 (boundary)" in markdown


# --------------------------------------------------------- provisional LR selection


def test_incomplete_phase_a_makes_the_selection_provisional() -> None:
    records = [
        record for record in _synthetic_campaign([0.24] * 5, [0.26] * 5)
        if not (record["phase"] == "a" and record["arm"] == "nekaon_fused"
                and record["lr"] == pytest.approx(2.0e-4))
    ]
    result = aggregate.summarize(records, resamples=200)

    assert result["phase_a"]["complete"] is False
    assert result["phase_a"]["runs"] == 8
    assert result["phase_a"]["expected"] == 9
    assert result["phase_a"]["per_arm"]["nekaon_fused"] == 2

    markdown = aggregate.to_markdown(result)
    assert "Provisional selection (phase A incomplete: 8/9 runs)" in markdown
    assert "not the definitive comparison" in markdown


def test_a_complete_phase_a_is_not_called_provisional() -> None:
    result = aggregate.summarize(_synthetic_campaign([0.24] * 5, [0.26] * 5), resamples=200)

    assert result["phase_a"]["complete"] is True
    assert result["phase_a"]["runs"] == 9
    assert "Provisional selection" not in aggregate.to_markdown(result)


# ------------------------------------------------------------- held-out comes first


def test_held_out_tables_precede_the_full_paired_set() -> None:
    markdown = aggregate.to_markdown(aggregate.summarize(
        _synthetic_campaign([0.24] * 5, [0.26] * 5), resamples=200))

    held = "### Held-out seeds, no selection bias (44, 45, 46, 47)"
    full = "### All 5 paired seeds, includes the LR-selection seed (43, 44, 45, 46, 47)"
    # Once under the per-arm means and once under the paired differences, held-out first.
    assert markdown.count(held) == 2
    assert markdown.count(full) == 2
    positions = [index for index, line in enumerate(markdown.splitlines())
                 if line in (held, full)]
    labels = [markdown.splitlines()[index] for index in positions]
    assert labels == [held, full, held, full]
    # Both ranges carry the same paired-difference detail, not a reduced appendix.
    assert markdown.count("Seeds won by nekaon_fused") == 4
    assert "4/4" in markdown and "5/5" in markdown


def test_timing_section_declares_the_nekaon_telemetry_cost() -> None:
    markdown = aggregate.to_markdown(aggregate.summarize(
        _synthetic_campaign([0.24] * 5, [0.26] * 5), resamples=200))

    assert "## Timings and electrical state" in markdown
    assert "_warn_if_inert" in markdown
    assert "counts against Nekaon in active seconds and ms/step" in markdown
    # The caveat lives inside the timings section, not only in the headline.
    section = markdown[markdown.index("## Timings and electrical state"):
                       markdown.index("## What this does not show")]
    assert "_warn_if_inert" in section
    assert "adakaon_fused` and `adamw_fused` do not" in section


# ------------------------------------------------------------------ power gating


def test_require_ac_power_refuses_battery_and_unknown() -> None:
    with pytest.raises(SystemExit, match="on battery"):
        campaign.require_ac_power({"ac": {"on_ac": False}})
    with pytest.raises(SystemExit, match="could not determine"):
        campaign.require_ac_power({"ac": {"on_ac": None}})
    campaign.require_ac_power({"ac": {"on_ac": True}})


def test_power_signature_distinguishes_ac_from_battery() -> None:
    ac = {"ac": {"on_ac": True}, "gpu": {"fields": {"power.max_limit": "60.00 W"}}}
    battery = {"ac": {"on_ac": False}, "gpu": {"fields": {"power.max_limit": "60.00 W"}}}
    assert campaign.power_signature(ac) != campaign.power_signature(battery)
    assert campaign.power_signature(ac).startswith("ac/")
