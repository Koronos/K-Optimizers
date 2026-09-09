"""Keep minibatch losses out of the held-out generalization comparison."""
import pytest

pytest.importorskip("tensorboard")
from benchmarks.anima.summarize_runs import summarize_run  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402


def test_gap_uses_evaluated_train_not_minibatches(tmp_path):
    with SummaryWriter(str(tmp_path)) as writer:
        writer.add_scalar("train/loss", .01, 100)
        writer.add_scalar("train_eval/loss", .08, 100)
        writer.add_scalar("val/loss", .09, 100)
        writer.add_scalar("val/loss_quantile_0.1", .5, 100)
    result = summarize_run(tmp_path)
    assert result["train_eval"]["tag"] == "train_eval/loss"
    assert result["gap"]["points"][0]["value"] == pytest.approx(.01)


def test_missing_train_eval_does_not_fall_back(tmp_path):
    with SummaryWriter(str(tmp_path)) as writer:
        writer.add_scalar("train/loss", .01, 100)
        writer.add_scalar("val/loss", .09, 100)
    result = summarize_run(tmp_path)
    assert result["train_eval"]["points"] is None
    assert result["gap"]["points"] is None


def test_nonfinite_final_loss_is_not_silently_dropped(tmp_path):
    with SummaryWriter(str(tmp_path)) as writer:
        writer.add_scalar("val/loss", .09, 100)
        writer.add_scalar("val/loss", float("nan"), 200)
    with pytest.raises(ValueError, match="Nonfinite"):
        summarize_run(tmp_path)


def test_multiple_runs_are_not_merged(tmp_path):
    for name in ("first", "second"):
        with SummaryWriter(str(tmp_path / name)) as writer:
            writer.add_scalar("val/loss", .09, 100)
    with pytest.raises(ValueError, match="Multiple event"):
        summarize_run(tmp_path)
