"""CPU-side tests for the Nekaon evidence step-cost benchmark.

Everything that does not need a GPU: the parametric SDXL bag, the arm factories, the
paired statistic, the markdown renderer (from a synthetic payload, so the report can be
reviewed without owning a measurement) and the refuse-on-battery guard.
"""

from __future__ import annotations

import importlib.util
import json
import math
import weakref
from pathlib import Path

import pytest
import torch

STEP_COST = Path(__file__).parents[1] / "benchmarks" / "nekaon_evidence" / "step_cost.py"
spec = importlib.util.spec_from_file_location("step_cost", STEP_COST)
assert spec and spec.loader
step_cost = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step_cost)


# ----------------------------------------------------------------- the bag
def test_bag_counts_match_the_sdxl_reference_scale() -> None:
    """R=3 is the shape census the measurement scale was chosen from: 468 tensors,
    397.9 M params — a regression here silently moves every reported number."""
    assert len(step_cost.bag_shapes(3)) == 468
    assert step_cost.bag_numel(3) == 397_862_400
    assert len(step_cost.bag_shapes(1)) == 156
    assert step_cost.bag_numel(5) == pytest.approx(663_104_000, rel=1e-9)


def test_bag_scales_linearly_in_R() -> None:
    for R in (1, 2, 4, 7):
        assert len(step_cost.bag_shapes(R)) == R * 156
        assert step_cost.bag_numel(R) == R * step_cost.bag_numel(1)


def test_bag_shape_mix_is_the_documented_one() -> None:
    shapes = step_cost.bag_shapes(1)
    assert shapes.count((1280, 1280)) == 12
    assert shapes.count((640, 5120)) == 4
    assert shapes.count((5120, 640)) == 4
    assert shapes.count((1280, 1280, 3, 3)) == 4
    assert sum(1 for s in shapes if len(s) == 1) == 120


def test_bag_rejects_non_positive_scale() -> None:
    with pytest.raises(ValueError, match="R must be >= 1"):
        step_cost.bag_shapes(0)


# ----------------------------------------------------------------- arms
def test_arm_table_covers_both_adamw_baselines_with_the_right_dtypes() -> None:
    assert step_cost.BASELINE == "adamw_bf16"
    assert step_cost.BASELINE in step_cost.ARMS
    assert step_cost.ARMS["adamw_bf16"][0] is torch.bfloat16
    # The 16 B/p arm is the ONLY one that trains in fp32; everything else is bf16 params.
    assert step_cost.ARMS["adamw_fp32"][0] is torch.float32
    for name, (dtype, _, note) in step_cost.ARMS.items():
        if name != "adamw_fp32":
            assert dtype is torch.bfloat16, name
        assert note.strip(), name


@pytest.mark.parametrize(
    ("arm", "cls_name", "momentum", "k"),
    [
        ("adakaon_bf16", "Adakaon", "bfloat16", None),
        ("adakaon_4bit", "Adakaon", "4bit", None),
        ("nekaon_bf16", "Nekaon", "bfloat16", 1.5),
        ("nekaon_4bit", "Nekaon", "4bit", 1.5),
        ("nekaon_k0_4bit", "Nekaon", "4bit", 0.0),
    ],
)
def test_kaon_arms_carry_the_anima_configuration(arm, cls_name, momentum, k) -> None:
    params = [torch.zeros(4, 4, dtype=torch.bfloat16, requires_grad=True)]
    opt = step_cost.ARMS[arm][1](params, True)  # native=True: no Triton on a CPU tensor
    assert type(opt).__name__ == cls_name
    group = getattr(opt, "inner", opt).param_groups[0]
    assert group["betas"] == (0.5, 0.999)
    assert group["weight_decay"] == 0.1
    assert group["momentum_dtype"] == momentum
    assert group["cautious"] is True
    if k is not None:
        assert opt.k == k


def test_native_flag_selects_the_torch_path_and_leaves_adamw_alone() -> None:
    params = [torch.zeros(4, 4, dtype=torch.bfloat16, requires_grad=True)]
    native = step_cost.ARMS["adakaon_4bit"][1](params, True)
    fused = step_cost.ARMS["adakaon_4bit"][1](params, False)
    assert getattr(native, "inner", native)._fused is False
    assert getattr(fused, "inner", fused)._fused is True


def test_adamw_arms_use_a_plain_torch_adamw() -> None:
    params = [torch.zeros(4, 4, requires_grad=True)]
    opt = step_cost._adamw(params, fused=False)
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["betas"] == (0.9, 0.999)


# ----------------------------------------------------------------- statistics
def test_gmean_ci_recovers_a_constant_ratio_with_a_degenerate_interval() -> None:
    logs = [math.log(1.25)] * 20
    ratio, lo, hi = step_cost.gmean_ci(logs)
    assert ratio == pytest.approx(1.25)
    assert lo == pytest.approx(1.25)
    assert hi == pytest.approx(1.25)


def test_gmean_ci_widens_with_spread_and_brackets_the_mean() -> None:
    logs = [math.log(r) for r in (0.8, 1.0, 1.2, 1.5, 0.9, 1.1)]
    ratio, lo, hi = step_cost.gmean_ci(logs)
    assert lo < ratio < hi


def test_single_sample_has_no_interval() -> None:
    ratio, lo, hi = step_cost.gmean_ci([0.0])
    assert ratio == pytest.approx(1.0)
    assert math.isnan(lo) and math.isnan(hi)


@pytest.mark.parametrize(
    ("lo", "hi", "expected"),
    [(1.02, 1.08, "slower"), (0.90, 0.98, "faster"), (0.95, 1.05, "n.s."),
     (float("nan"), float("nan"), "n.s.")],
)
def test_verdict_only_calls_a_difference_when_the_ci_clears_one(lo, hi, expected) -> None:
    assert step_cost.verdict_of(lo, hi) == expected


def test_state_bytes_per_param_walks_the_inner_chain() -> None:
    class Fake:
        def __init__(self, state, inner=None):
            self.state = state
            self.inner = inner

    param = torch.zeros(100)
    inner = Fake({0: {"exp_avg": torch.zeros(100, dtype=torch.bfloat16)}})  # 200 B
    outer = Fake({0: {"phi": torch.zeros(100, dtype=torch.bfloat16)}}, inner)  # +200 B
    assert step_cost.state_bytes_per_param(outer, [param]) == pytest.approx(4.0)
    assert step_cost.state_bytes_per_param(inner, [param]) == pytest.approx(2.0)


def test_state_bytes_ignores_non_tensor_state() -> None:
    class Fake:
        state = {0: {"step": 7, "exp_avg": torch.zeros(10, dtype=torch.float32)}}
        inner = None

    assert step_cost.state_bytes_per_param(Fake(), [torch.zeros(10)]) == pytest.approx(4.0)


# ----------------------------------------------------------------- power guard
@pytest.mark.parametrize(
    ("status", "ok"),
    [("2", True), ("2\n", True), ("", True), ("1", False), ("1\n2", False)],
)
def test_on_ac_accepts_only_status_two_or_no_battery(status, ok) -> None:
    assert step_cost.on_ac(status)[0] is ok


def test_main_refuses_to_measure_on_battery(monkeypatch, tmp_path, capsys) -> None:
    """The clock is meaningless at the 35 W battery power limit, so the run must abort
    before it allocates anything — not produce numbers with a caveat."""
    monkeypatch.setattr(step_cost.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(step_cost, "battery_status", lambda: "1")
    monkeypatch.setattr(step_cost, "collect_meta", lambda: {
        "on_ac": False, "power_note": "on battery (BatteryStatus='1')"})

    def explode(*a, **kw):  # pragma: no cover - must never be reached
        raise AssertionError("measurement started on battery")

    monkeypatch.setattr(step_cost, "measure_solo", explode)
    rc = step_cost.main(["--R", "1", "--out", str(tmp_path / "x.json")])
    assert rc == 3
    assert "REFUSING to measure" in capsys.readouterr().err
    assert not (tmp_path / "x.json").exists()


def test_main_exits_cleanly_without_cuda(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(step_cost.torch.cuda, "is_available", lambda: False)
    assert step_cost.main(["--R", "1", "--out", str(tmp_path / "x.json")]) == 2


def test_baseline_flag_defaults_to_the_module_baseline() -> None:
    assert step_cost.parse_args([]).baseline == step_cost.BASELINE == "adamw_bf16"
    assert step_cost.parse_args(["--baseline", "adakaon_4bit"]).baseline == "adakaon_4bit"


def test_main_adds_a_missing_baseline_to_arms_and_says_so(monkeypatch, capsys) -> None:
    """``--arms`` without the chosen ``--baseline`` must still pair correctly: the
    baseline is added so it is resident, and the user is told it happened."""
    monkeypatch.setattr(step_cost.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(step_cost, "collect_meta", lambda: {"on_ac": True, "power_note": "ac",
                                                            "kaon_version": "0.7.14",
                                                            "commit": "abc"})
    seen: dict[str, list[str]] = {}

    def solo(names, R, native, reps, want_launches):
        seen["solo"] = list(names)
        return {}

    def paired(names, R, native, reps, baseline=None):
        seen["paired"] = list(names)
        seen["paired_baseline"] = baseline
        return {}

    monkeypatch.setattr(step_cost, "measure_solo", solo)
    monkeypatch.setattr(step_cost, "measure_paired", paired)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        rc = step_cost.main(["--arms", "nekaon_4bit", "adakaon_4bit",
                             "--baseline", "adakaon_4bit", "--reps", "2",
                             "--out", f"{tmp}/o.json", "--markdown", f"{tmp}/o.md"])
    assert rc == 0
    assert set(seen["solo"]) == {"nekaon_4bit", "adakaon_4bit"}  # already present: no-op
    assert seen["paired_baseline"] == "adakaon_4bit"

    seen.clear()
    with tempfile.TemporaryDirectory() as tmp:
        rc = step_cost.main(["--arms", "nekaon_4bit",
                             "--baseline", "adakaon_4bit", "--reps", "2",
                             "--out", f"{tmp}/o.json", "--markdown", f"{tmp}/o.md"])
    assert rc == 0
    assert set(seen["solo"]) == {"nekaon_4bit", "adakaon_4bit"}  # auto-added
    out = capsys.readouterr().out
    assert "adakaon_4bit" in out and "not in --arms" in out


# ----------------------------------------------------------------- rendering
SYNTHETIC = {
    "schema": "nekaon_evidence.step_cost/1",
    "meta": {"kaon_version": "0.7.14", "commit": "0" * 40, "dirty": False,
             "torch": "2.12.0+cu130", "torch_cuda": "13.0", "triton": "3.7.1",
             "python": "3.12.0", "gpu": "NVIDIA RTX 3000 Ada Generation Laptop GPU",
             "nvidia_smi": "[N/A], 60.00 W, 35.00 W, 3105 MHz, 45",
             "power_note": "on AC (BatteryStatus=2)", "timestamp": "2026-09-18T10:00:00"},
    "config": {"R": 5, "reps": 60, "native": False, "baseline": "adamw_bf16"},
    "bag": {"R": 5, "tensors": 780, "params": 663_104_000},
    "arms": {
        "adamw_bf16": {"param_dtype": "bfloat16", "params": 663_104_000, "tensors": 780,
                       "state_bytes_per_param": 4.0, "peak_allocated_bytes": 4 * 2**30,
                       "peak_reserved_bytes": 5 * 2**30,
                       "step_transient_bytes": 64 * 2**20, "ms_step_solo": 10.0,
                       "note": "baseline", "cuda_launches": 12},
        "nekaon_4bit": {"param_dtype": "bfloat16", "params": 663_104_000, "tensors": 780,
                        "state_bytes_per_param": 1.6, "peak_allocated_bytes": 3 * 2**30,
                        "peak_reserved_bytes": 4 * 2**30,
                        "step_transient_bytes": 32 * 2**20, "ms_step_solo": 13.0,
                        "note": "anima config", "cuda_launches": 40},
    },
    "paired": {
        "nekaon_4bit": {"baseline": "adamw_bf16", "reps": 60, "ms_arm_median": 13.0,
                        "ms_baseline_median": 10.0, "ratio": 1.3, "ci_lo": 1.27,
                        "ci_hi": 1.33, "verdict": "slower"},
    },
    "fraction": {"C": 256, "batch": 8, "px": 64, "steps": 12, "arms": {
        "nekaon_4bit": {"ms_step_total": 100.0, "ms_optimizer": 3.0, "params": 20_000_000,
                        "optimizer_fraction": 0.03, "measured_steps": 8}}},
    "capacity": {"nekaon_4bit": {"max_R": 9, "params": 1_193_587_200, "oom_at_R": 10,
                                 "oom_at_params": 1_326_208_000, "swept": [1, 16, 1]}},
}


def test_render_covers_every_section_and_quotes_the_provenance() -> None:
    md = step_cost.render_markdown(SYNTHETIC)
    for needle in ("Per step, per arm", "Paired against", "Share of a real training step",
                   "Capacity on this GPU", "What each arm is", "0.7.14",
                   "RTX 3000 Ada", "1.300x", "[1.270, 1.330]", "slower", "1.60",
                   "3.0%", "663.1 M"):
        assert needle in md, needle


def test_render_shows_the_baseline_stored_in_the_json_not_the_module_default() -> None:
    """``--render-only`` must respect whatever baseline the run was paired against, even
    though ``step_cost.BASELINE`` (the CLI default) is a different arm."""
    payload = json.loads(json.dumps(SYNTHETIC))
    payload["config"]["baseline"] = "adakaon_4bit"
    payload["paired"] = {"nekaon_4bit": {"baseline": "adakaon_4bit", "reps": 60,
                                         "ms_arm_median": 13.0, "ms_baseline_median": 12.0,
                                         "ratio": 1.08, "ci_lo": 1.05, "ci_hi": 1.11,
                                         "verdict": "slower"}}
    md = step_cost.render_markdown(payload)
    assert "Paired against `adakaon_4bit`" in md
    assert step_cost.BASELINE == "adamw_bf16"  # the module default is untouched


def test_render_omits_sections_that_were_not_measured() -> None:
    payload = {k: v for k, v in SYNTHETIC.items() if k not in ("capacity", "fraction")}
    md = step_cost.render_markdown(payload)
    assert "Capacity on this GPU" not in md
    assert "Share of a real training step" not in md
    assert "Paired against" in md


def test_render_drops_the_launch_column_when_not_collected() -> None:
    payload = json.loads(json.dumps(SYNTHETIC))
    for arm in payload["arms"].values():
        arm.pop("cuda_launches")
    assert "launches" not in step_cost.render_markdown(payload)


def test_render_only_rebuilds_the_report_from_a_json(tmp_path, capsys) -> None:
    src = tmp_path / "step_cost_R5.json"
    src.write_text(json.dumps(SYNTHETIC), encoding="utf-8")
    md = tmp_path / "RESULTS_step_cost.md"
    assert step_cost.main(["--render-only", str(src), "--markdown", str(md)]) == 0
    assert md.read_text(encoding="utf-8") == step_cost.render_markdown(SYNTHETIC)
    assert "rendered" in capsys.readouterr().out


# ----------------------------------------------------------------- GPU smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_smallest_bag_steps_and_reports_state_bytes() -> None:
    params, opt = step_cost.build_arm("adakaon_4bit", 1, native=True)
    try:
        opt.step()
        bpp = step_cost.state_bytes_per_param(opt, params)
        assert 0.0 < bpp < 4.0  # 4-bit momentum + factored v is well under bf16 AdamW
    finally:
        params = opt = None
        step_cost.reclaim()


def test_render_marks_a_pair_that_did_not_fit() -> None:
    """A pair that OOMs is reported as such, not silently dropped from the table."""
    payload = json.loads(json.dumps(SYNTHETIC))
    payload["paired"]["adamw_fp32"] = {"baseline": "adamw_bf16", "reps": 60,
                                             "R": 5, "oom": True}
    md = step_cost.render_markdown(payload)
    assert "did not fit beside the baseline at R=5" in md
    assert "1.300x" in md  # the arms that did fit are still reported


def test_render_marks_an_arm_that_did_not_fit_at_all() -> None:
    payload = json.loads(json.dumps(SYNTHETIC))
    payload["arms"]["adamw_fp32"] = {"param_dtype": "float32", "R": 5, "oom": True,
                                           "note": "16 B/p"}
    md = step_cost.render_markdown(payload)
    assert "OOM at R=5" in md
    assert "`adamw_fp32`" in md


def test_solo_phase_records_an_oom_arm_instead_of_dying(monkeypatch) -> None:
    """A run that loses one arm to OOM must still report the others — at R=5 the fp32
    pure-fp32 arm needs ~10.6 GB and is expected to drop out on an 8 GB card."""
    def oom(*a, **kw):
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(step_cost, "build_arm", oom)
    monkeypatch.setattr(step_cost, "reclaim", lambda: None)
    out = step_cost.measure_solo(["adamw_fp32"], 5, native=False, reps=1,
                                 want_launches=False)
    assert out["adamw_fp32"]["oom"] is True
    assert out["adamw_fp32"]["R"] == 5


def test_paired_phase_records_a_pair_that_does_not_fit(monkeypatch) -> None:
    built = {"n": 0}

    def build(name, R, native, seed=0):
        built["n"] += 1
        if built["n"] > 1:  # the baseline builds; the partner does not fit beside it
            raise torch.OutOfMemoryError("CUDA out of memory")
        return [torch.zeros(1)], object()

    monkeypatch.setattr(step_cost, "build_arm", build)
    monkeypatch.setattr(step_cost, "reclaim", lambda: None)
    out = step_cost.measure_paired(["adamw_bf16", "nekaon_4bit"], 5, native=False, reps=2)
    assert out["nekaon_4bit"] == {"baseline": "adamw_bf16", "reps": 2, "R": 5, "oom": True}


def test_measure_paired_accepts_a_custom_baseline_and_excludes_it(monkeypatch) -> None:
    """``--baseline adakaon_4bit`` must hold adakaon_4bit fixed as the reference and drop
    it from the list of contenders — pairing it against itself would be content-free."""
    built: list[str] = []

    def build(name, R, native, seed=0):
        built.append(name)
        return [torch.zeros(1)], _FakeArm()

    monkeypatch.setattr(step_cost, "build_arm", build)
    monkeypatch.setattr(step_cost, "paired_samples",
                        lambda a, b, reps, **kw: ([1.1, 1.2], [1.0, 1.0]))
    monkeypatch.setattr(step_cost, "reclaim", lambda: None)
    out = step_cost.measure_paired(["adamw_bf16", "adakaon_4bit", "nekaon_4bit"], 5,
                                   native=False, reps=2, baseline="adakaon_4bit")
    assert "adakaon_4bit" not in out  # the baseline never pairs against itself
    assert set(out) == {"adamw_bf16", "nekaon_4bit"}
    assert out["adamw_bf16"]["baseline"] == "adakaon_4bit"
    assert out["nekaon_4bit"]["baseline"] == "adakaon_4bit"
    assert built[0] == "adakaon_4bit"  # the baseline is the one built once, up front


def test_paired_scale_defaults_below_the_solo_scale(monkeypatch) -> None:
    """The paired phase holds two arms, so it must not silently inherit an --R that only
    fits one."""
    seen = {}
    monkeypatch.setattr(step_cost.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(step_cost, "collect_meta", lambda: {"on_ac": True,
                                                            "power_note": "ac",
                                                            "kaon_version": "0.7.14",
                                                            "commit": "abc"})
    monkeypatch.setattr(step_cost, "measure_solo", lambda *a, **kw: {})

    def record(names, R, native, reps, baseline=None):
        seen["R"] = R
        return {}

    monkeypatch.setattr(step_cost, "measure_paired", record)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        step_cost.main(["--R", "5", "--reps", "2", "--out", f"{tmp}/o.json",
                        "--markdown", f"{tmp}/o.md"])
    assert seen["R"] == step_cost.DEFAULT_PAIRED_R == 3
    seen.clear()
    with tempfile.TemporaryDirectory() as tmp:
        step_cost.main(["--R", "5", "--paired-R", "5", "--reps", "2",
                        "--out", f"{tmp}/o.json", "--markdown", f"{tmp}/o.md"])
    assert seen["R"] == 5


# ----------------------------------------------------------------- reclaiming
def _stub_cuda_counters(monkeypatch) -> None:
    """Enough of ``torch.cuda`` to run a measurement phase against fake arms on CPU."""
    for name, value in (("synchronize", None), ("reset_peak_memory_stats", None),
                        ("empty_cache", None), ("memory_allocated", 0),
                        ("max_memory_allocated", 1), ("max_memory_reserved", 1)):
        monkeypatch.setattr(torch.cuda, name, lambda *a, _v=value, **kw: _v)


class _FakeArm:
    """Stands in for an optimizer: has ``.state`` (for the byte accounting) and ``.step``."""

    state: dict = {}

    def step(self) -> None:
        pass


def test_reclaim_takes_no_objects_to_free() -> None:
    """The signature IS the fix: a helper cannot free its caller's references, so it must
    not pretend to -- ``del o`` inside ``free_all`` unbound only the helper's local name."""
    import inspect

    assert list(inspect.signature(step_cost.reclaim).parameters) == []


def test_solo_phase_releases_the_arm_before_reclaiming(monkeypatch) -> None:
    """empty_cache() must run with the arm already dead. While it was alive its blocks
    stayed reserved and inflated the NEXT arm's peak_reserved and OOM point -- and ARMS
    puts the nekaon arms last, so the bias fell on exactly the arms under test."""
    _stub_cuda_counters(monkeypatch)
    refs: list = []
    alive_at_reclaim: list[bool] = []

    def build(name, R, native, seed=0):
        arm = _FakeArm()
        refs.append(weakref.ref(arm))
        return [torch.zeros(4)], arm

    monkeypatch.setattr(step_cost, "build_arm", build)
    monkeypatch.setattr(step_cost, "reclaim",
                        lambda: alive_at_reclaim.append(refs[-1]() is not None))
    out = step_cost.measure_solo(["adamw_bf16"], 1, native=True, reps=1,
                                 want_launches=False)
    assert "adamw_bf16" in out
    assert alive_at_reclaim == [False]  # the bound opt.step no longer pins the arm
    assert refs[0]() is None


def test_paired_phase_releases_both_arms_before_reclaiming(monkeypatch) -> None:
    _stub_cuda_counters(monkeypatch)
    refs: dict = {}
    alive_at_reclaim: list[bool] = []

    def build(name, R, native, seed=0):
        arm = _FakeArm()
        refs[name] = weakref.ref(arm)
        return [torch.zeros(4)], arm

    monkeypatch.setattr(step_cost, "build_arm", build)
    monkeypatch.setattr(step_cost, "paired_samples",
                        lambda a, b, reps, **kw: ([1.0, 1.1], [1.0, 1.0]))
    monkeypatch.setattr(step_cost, "reclaim",
                        lambda: alive_at_reclaim.append(
                            {n for n, r in refs.items() if r() is not None}))
    step_cost.measure_paired(["adamw_bf16", "nekaon_4bit"], 1, native=True, reps=2)
    # Inner reclaim: the partner is gone, the baseline is still needed for the next pair.
    # Outer reclaim: the baseline is gone too, so the next PHASE starts on a clean card.
    assert alive_at_reclaim == [{"adamw_bf16"}, set()]


def test_capacity_sweep_releases_the_arm_before_reclaiming(monkeypatch) -> None:
    """The bias this removes is largest here: a sweep that reclaims with the previous arm
    still resident reports a smaller max_R for whatever ran after it."""
    _stub_cuda_counters(monkeypatch)
    refs: list = []
    alive_at_reclaim: list[bool] = []

    def build(name, R, native, seed=0):
        arm = _FakeArm()
        refs.append(weakref.ref(arm))
        return [torch.zeros(4)], arm

    monkeypatch.setattr(step_cost, "build_arm", build)
    monkeypatch.setattr(step_cost, "reclaim",
                        lambda: alive_at_reclaim.append(refs[-1]() is not None))
    out = step_cost.measure_capacity(["nekaon_4bit"], native=True, start=1, stop=2,
                                     step_r=1)
    assert out["nekaon_4bit"]["max_R"] == 2
    assert alive_at_reclaim == [False, False]


# ----------------------------------------------------------------- inert telemetry
def _cpu_nekaon(k: float = 1.5):
    params = [torch.zeros(8, 8, dtype=torch.float32, requires_grad=True)]
    for p in params:
        p.grad = torch.ones_like(p)
    return params, step_cost._nekaon(params, "bfloat16", fused=False, k=k)


def test_disarm_inert_telemetry_stops_msam_from_syncing_to_the_host() -> None:
    """MSAM samples weight scales every inert_check_interval climbs and converts device
    reductions to Python floats. Only rho != 0 arms pay it, so leaving it armed would tax
    nekaon_bf16/nekaon_4bit and not adamw_*/adakaon_*/nekaon_k0_4bit -- a cost of the
    first ~2000 steps landing in a measurement of the update in regime."""
    _, armed = _cpu_nekaon()
    for _ in range(25):
        armed.step()
    assert armed._inert_checks > 0  # the heuristic is live by default

    _, quiet = _cpu_nekaon()
    step_cost.disarm_inert_telemetry(quiet)
    for _ in range(25):
        quiet.step()
    assert quiet._inert_checks == 0


def test_disarm_is_a_noop_on_arms_without_the_heuristic() -> None:
    """Uniformity is the point: it is called for every arm, and the arms that never had
    the heuristic must be untouched rather than raising."""
    params = [torch.zeros(4, 4, requires_grad=True)]
    opt = step_cost._adamw(params, fused=False)
    assert step_cost.disarm_inert_telemetry(opt) is opt
    opt.step()


def test_build_arm_disarms_every_arm_it_builds(monkeypatch) -> None:
    monkeypatch.setattr(step_cost, "sdxl_bag",
                        lambda R, dtype, seed=0: [torch.zeros(8, 8, dtype=dtype,
                                                              requires_grad=True)])
    for name in ("nekaon_bf16", "nekaon_k0_4bit", "adakaon_bf16", "adamw_bf16"):
        _, opt = step_cost.build_arm(name, 1, native=True)
        for o in (opt, getattr(opt, "inner", None)):
            if o is not None and hasattr(o, "_inert_warned"):
                assert o._inert_warned is True, name


# ----------------------------------------------------------------- paired sampling
def test_paired_samples_drops_the_ramp_up_pairs_from_both_arms(monkeypatch) -> None:
    """The paired phase runs at a different R than the solo phase, so Triton re-autotunes;
    30 warm pairs plus 5 discarded measured pairs keep that out of the samples."""
    _stub_cuda_counters(monkeypatch)  # the loop brackets every rep with a synchronize
    calls = {"a": 0, "b": 0}

    def a():
        calls["a"] += 1

    def b():
        calls["b"] += 1

    ta, tb = step_cost.paired_samples(a, b, reps=20)
    assert len(ta) == len(tb) == 20 - step_cost.DEFAULT_PAIRED_DISCARD
    assert calls["a"] == calls["b"] == step_cost.DEFAULT_PAIRED_WARM + 20


def test_paired_samples_keeps_everything_on_a_run_too_short_to_trim(monkeypatch) -> None:
    _stub_cuda_counters(monkeypatch)
    ta, tb = step_cost.paired_samples(lambda: None, lambda: None, reps=4, warm=0)
    assert len(ta) == len(tb) == 4


def test_paired_defaults_are_the_documented_warmup_and_discard() -> None:
    assert step_cost.DEFAULT_PAIRED_WARM == 30
    assert step_cost.DEFAULT_PAIRED_DISCARD == 5


def test_gmean_ci_uses_student_t_below_thirty_samples() -> None:
    """battery.py::ci95 uses Student-t; a short --reps run here would otherwise quote a
    z=1.96 interval that is too narrow, and the two reports would not be comparable."""
    import statistics as stats

    logs = [math.log(r) for r in (1.10, 1.20, 1.30, 1.40)]
    ratio, lo, hi = step_cost.gmean_ci(logs)
    half_t = step_cost._T95[3] * stats.stdev(logs) / math.sqrt(4)
    assert lo == pytest.approx(math.exp(math.log(ratio) - half_t))
    assert hi == pytest.approx(math.exp(math.log(ratio) + half_t))
    assert step_cost._T95[3] > 1.96  # strictly wider than the normal approximation


def test_gmean_ci_falls_back_to_the_normal_approximation_past_the_table() -> None:
    import statistics as stats

    logs = [math.log(1.0 + 0.001 * i) for i in range(40)]
    ratio, lo, hi = step_cost.gmean_ci(logs)
    half_z = 1.96 * stats.stdev(logs) / math.sqrt(40)
    assert lo == pytest.approx(math.exp(math.log(ratio) - half_z))
