"""Schema-level tests for the stdlib Anima comparison config generator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")

GENERATOR = Path(__file__).parents[1] / "benchmarks" / "anima" / "generate_comparison.py"
spec = importlib.util.spec_from_file_location("generate_comparison", GENERATOR)
assert spec and spec.loader
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def test_custom_eval_protocol_and_arm_subset(tmp_path: Path) -> None:
    manifest = generator.generate(
        output_dir=tmp_path / "comparison",
        pets_root="/tmp/pets/subset",
        steps=200,
        lr=1.0e-4,
        seed=43,
        previews=False,
        gap_threshold=None,
        eval_images=32,
        eval_every=200,
        arms=("nekaon", "rakaon_isotropic"),
    )

    assert [item["arm"] for item in manifest["arms"]] == ["nekaon", "rakaon_isotropic"]
    assert manifest["dataset"]["train_eval"]["max_images"] == 32
    assert manifest["dataset"]["val"]["max_images"] == 32
    assert manifest["protocol"]["eval_every_n_steps"] == 200
    assert manifest["protocol"]["train_seed"] == 43
    assert manifest["protocol"]["adapter_init_seed"] == 43

    for arm in ("nekaon", "rakaon_isotropic"):
        config = tomllib.loads((tmp_path / "comparison" / f"{arm}.toml").read_text())
        assert config["max_steps"] == 200
        assert config["eval_every_n_steps"] == 200
        assert config["train_seed"] == 43
        assert [entry["name"] for entry in config["eval_datasets"]] == ["train_eval", "val"]
        for entry in config["eval_datasets"]:
            dataset = tomllib.loads(Path(entry["config"]).read_text())
            assert dataset["max_images"] == 32

    saved_manifest = json.loads((tmp_path / "comparison" / "manifest.json").read_text())
    assert "ANIMA_INIT_SEED=43" in saved_manifest["command_template"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"eval_images": 0}, "eval_images"),
        ({"eval_images": 33}, "eval_images"),
        ({"eval_every": 0}, "eval_every"),
        ({"steps": 201}, "multiple"),
        ({"lr": float("nan")}, "finite"),
    ],
)
def test_generator_rejects_unsafe_protocol_values(tmp_path: Path, kwargs: dict[str, object], message: str) -> None:
    defaults: dict[str, object] = {
        "output_dir": tmp_path / "comparison",
        "pets_root": "/tmp/pets/subset",
        "steps": 200,
        "lr": 1.0e-4,
        "seed": 42,
        "previews": False,
        "gap_threshold": None,
    }
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=message):
        generator.generate(**defaults)


def test_default_protocol_keeps_all_arms_and_eval_defaults(tmp_path: Path) -> None:
    manifest = generator.generate(
        output_dir=tmp_path / "comparison",
        pets_root="/tmp/pets/subset",
        steps=200,
        lr=1.0e-4,
        seed=42,
        previews=False,
        gap_threshold=None,
    )
    assert [item["arm"] for item in manifest["arms"]] == list(generator.ARMS)
    assert manifest["protocol"]["eval_every_n_steps"] == 100
    assert manifest["dataset"]["val"]["max_images"] == 8


@pytest.mark.parametrize(("arm", "beta1"), [("rakaon_m05", .5), ("rakaon_m09", .9)])
def test_momentum_recipe_changes_only_beta1(arm, beta1):
    base = tomllib.loads("\n".join(generator._optimizer_config("rakaon_isotropic", .0001)))
    candidate = tomllib.loads("\n".join(generator._optimizer_config(arm, .0001)))
    assert candidate.pop("beta1") == beta1
    assert candidate == base
