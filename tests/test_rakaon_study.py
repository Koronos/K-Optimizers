from pathlib import Path


def test_time_to_quality_does_not_reward_transient_or_missed_targets(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from rakaon_study import target_hits

    trajectory = [
        dict(step=200, training_seconds=10, wall_seconds=12, te=.079, gap=.009),
        dict(step=400, training_seconds=20, wall_seconds=24, te=.079, gap=.006),
        dict(step=600, training_seconds=30, wall_seconds=36, te=.082, gap=.006),
        dict(step=800, training_seconds=40, wall_seconds=48, te=.078, gap=.005),
        dict(step=1000, training_seconds=50, wall_seconds=60, te=.0775, gap=.005),
    ]
    result = target_hits(trajectory)
    assert result["val<0.08"]["first"]["step"] == 200
    joint = result["val<0.08_abs_gap<0.007"]
    assert joint["first"]["step"] == 400
    assert joint["first_of_two_consecutive"]["step"] == 800
    assert joint["first_of_two_consecutive"]["training_seconds"] == 40
    assert joint["first_of_two_consecutive"]["wall_seconds"] == 48
    assert result["val<0.07_abs_gap<0.007"] == dict(first=None, first_of_two_consecutive=None)


def test_negative_gap_is_not_automatically_a_success(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from rakaon_study import target_hits

    result = target_hits([dict(step=200, training_seconds=1, wall_seconds=2, te=.06, gap=-.02)])
    assert result["val<0.07"]["first"] is not None
    assert result["val<0.07_abs_gap<0.007"]["first"] is None
