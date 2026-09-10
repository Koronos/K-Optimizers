"""Generate one-step visual-review configs from completed Anima comparison runs.

The generated jobs load a saved adapter and render previews before their first
(and only) training step.  The images are qualitative step-0 views of the loaded
checkpoint; they are not training metrics.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import tomllib

DEFAULT_ARMS = ("nekaon", "rakaon_isotropic")
PROMPTS = (
    ("Bengal cat", "a Bengal cat in a sunlit garden"),
    ("Siamese cat", "a Siamese cat sitting beside a window"),
    ("beagle", "a beagle in a grassy park"),
    ("shiba inu", "a shiba inu on a quiet forest path"),
)


def _q(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return _q(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _dump_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []

    def emit_table(table: dict[str, Any], prefix: str = "") -> None:
        scalars = [(key, value) for key, value in table.items()
                   if not isinstance(value, dict) and not (isinstance(value, list) and value and isinstance(value[0], dict))]
        if prefix:
            lines.append(f"[{prefix}]")
        lines.extend(f"{key} = {_toml_value(value)}" for key, value in scalars)
        if scalars:
            lines.append("")
        for key, value in table.items():
            if isinstance(value, dict):
                emit_table(value, f"{prefix}.{key}" if prefix else key)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                heading = f"{prefix}.{key}" if prefix else key
                for item in value:
                    lines.append(f"[[{heading}]]")
                    lines.extend(f"{item_key} = {_toml_value(item_value)}"
                                 for item_key, item_value in item.items()
                                 if not isinstance(item_value, (dict, list)))
                    lines.append("")

    emit_table(data)
    return "\n".join(lines).rstrip() + "\n"


def _run_dir_from_log(log_path: Path) -> list[Path]:
    if not log_path.is_file():
        return []
    found: list[Path] = []
    pattern = re.compile(r"^Run dir: (.+)$")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            found.append(Path(match.group(1).strip('"\'')))
    return found


def _checkpoint(run_roots: list[Path], step: int) -> Path:
    roots = list(dict.fromkeys(run_roots))
    if len(roots) != 1:
        raise ValueError("Expected exactly one Run dir in the source log")
    directory = roots[0] / f"step{step}"
    safetensors = list(directory.glob("*.safetensors"))
    if len(safetensors) != 1:
        raise ValueError(f"expected exactly one safetensors in {directory}, found {len(safetensors)}")
    return directory


def _config_for_review(source: dict[str, Any], checkpoint: Path, output: Path) -> dict[str, Any]:
    config = json.loads(json.dumps(source))
    config.pop("eval_datasets", None)
    config["max_steps"] = 1
    config["output_dir"] = output.as_posix()
    config["save_every_n_steps"] = 999999
    for key in tuple(config):
        if "export" in key.lower() and "step" in key.lower():
            config[key] = 999999
    config["eval_before_first_step"] = False
    config["eval_every_n_steps"] = 999999
    adapter = config.setdefault("adapter", {})
    adapter["init_from_existing"] = checkpoint.as_posix()
    preview = config.setdefault("preview", {})
    preview.update({
        "enabled": True,
        "preview_before_first_step": True,
        "preview_every_n_steps": 999999,
        "width": 512,
        "height": 512,
        "num_inference_steps": 20,
        "guidance_scale": 4.0,
        "seed": 20260909,
        "seed_stride": 0,
    })
    preview["prompts"] = [{"name": name, "prompt": prompt} for name, prompt in PROMPTS]
    return config


def generate(manifest_path: Path, output_dir: Path, arms: tuple[str, ...]) -> dict[str, Any]:
    if output_dir.resolve() == manifest_path.parent.resolve():
        raise ValueError("refusing to overwrite the source manifest directory")
    if not arms or len(set(arms)) != len(arms):
        raise ValueError("arms must be nonempty and unique")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "protocol" not in manifest or "arms" not in manifest:
        raise ValueError("manifest must contain protocol and arms")
    output_dir.mkdir(parents=True, exist_ok=True)
    by_name = {arm["arm"]: arm for arm in manifest["arms"]}
    step = int(manifest["protocol"].get("steps", 200))
    generated = []
    for name in arms:
        if name not in by_name:
            raise ValueError(f"arm {name!r} is absent from manifest")
        source_path = Path(by_name[name]["config"])
        source = tomllib.loads(source_path.read_text(encoding="utf-8"))
        log_path = manifest_path.parent / f"{name}.log"
        if "Training complete." not in log_path.read_text(encoding="utf-8"):
            raise ValueError(f"Source training is incomplete: {log_path}")
        roots = _run_dir_from_log(log_path)
        checkpoint = _checkpoint(roots, step)
        config_path = output_dir / f"{name}_visual.toml"
        run_output = output_dir / "runs" / name
        config_path.write_text(_dump_toml(_config_for_review(source, checkpoint, run_output)), encoding="utf-8")
        tomllib.loads(config_path.read_text(encoding="utf-8"))
        generated.append({"arm": name, "config": config_path.as_posix(), "output_dir": run_output.as_posix(),
                          "source_run_dir": checkpoint.as_posix()})
    review_manifest = {
        "schema": "anima-visual-review-v1",
        "generated_by": "benchmarks/anima/generate_visual_review.py",
        "note": "Previews are step-0 images from the loaded checkpoint, not training metrics.",
        "protocol": {"adapter_init_seed": manifest["protocol"].get("adapter_init_seed", manifest["protocol"].get("train_seed"))},
        "arms": generated,
    }
    (output_dir / "manifest.json").write_text(json.dumps(review_manifest, indent=2) + "\n", encoding="utf-8")
    return review_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    args = parser.parse_args()
    output = args.output_dir or manifest_default_output(args.manifest)
    if output.resolve() == args.manifest.parent.resolve():
        raise ValueError("refusing to generate visual review files in the source manifest directory")
    result = generate(args.manifest, output, tuple(args.arms))
    print(f"Generated {len(result['arms'])} visual-review configs in {output}")


def manifest_default_output(manifest: Path) -> Path:
    return manifest.parent / "visual-review"


if __name__ == "__main__":
    main()
