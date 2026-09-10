import json
from pathlib import Path

import pytest

pytest.importorskip("tomllib")

from benchmarks.anima.generate_visual_review import generate


def _source_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    config = source / "nekaon.toml"
    config.write_text('[model]\ntype = "cosmos_predict2"\n\n[adapter]\ntype = "lora"\n', encoding="utf-8")
    run_dir = source / "run"
    (run_dir / "step199").mkdir(parents=True)
    (run_dir / "step200").mkdir(parents=True)
    (run_dir / "step199" / "old.safetensors").write_bytes(b"old")
    (run_dir / "step200" / "final.safetensors").write_bytes(b"final")
    log = source / "nekaon.log"
    log.write_text(f"Run dir: {run_dir}\nTraining complete.\n", encoding="utf-8")
    manifest = source / "manifest.json"
    manifest.write_text(json.dumps({"protocol": {"steps": 200, "adapter_init_seed": 42},
                                    "arms": [{"arm": "nekaon", "config": str(config)}]}, indent=2), encoding="utf-8")
    return manifest, config, log


def test_generate_uses_exact_protocol_checkpoint_and_preserves_sources(tmp_path):
    manifest, config, log = _source_tree(tmp_path)
    original_manifest = manifest.read_bytes()
    original_config = config.read_bytes()
    original_log = log.read_bytes()
    result = generate(manifest, tmp_path / "review", ("nekaon",))

    assert result["arms"][0]["source_run_dir"].endswith("run/step200")
    generated = (tmp_path / "review" / "nekaon_visual.toml").read_text(encoding="utf-8")
    assert "step200" in generated
    assert "step199" not in generated
    assert manifest.read_bytes() == original_manifest
    assert config.read_bytes() == original_config
    assert log.read_bytes() == original_log


def test_generate_rejects_output_equal_to_source_directory(tmp_path):
    manifest, _, _ = _source_tree(tmp_path)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        generate(manifest, manifest.parent, ("nekaon",))
