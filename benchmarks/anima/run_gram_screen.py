"""Run four bounded Gram trials sequentially; preserve failed runs for diagnosis."""
import argparse
import json
import subprocess
from pathlib import Path

from benchmarks.anima.generate_comparison import DEFAULT_PETS_ROOT, generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefix = ["uv", "run", "--no-sync", "--project", "/home/koronos/Rengu-Flow", "python"]
    plan = dict(seed=43, steps=200, eval_every=100, eval_images=32,
                lrs=[.001, .01], dampings=[.001, .01], scheduler="constant",
                purpose="Exploration only; equal two-LR budget per damping. No held-out confirmation.")
    if (output / "plan.json").exists():
        raise FileExistsError("Use a new output directory; existing experiment is preserved")
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for index, lr in enumerate(plan["lrs"]):
        directory = output / f"lr{index}"
        generate(output_dir=directory, pets_root=DEFAULT_PETS_ROOT, steps=200,
                 lr=lr, seed=43, previews=False, gap_threshold=None,
                 eval_images=32, eval_every=100, arms=("gram_d001", "gram_d01"))
        manifest = directory / "manifest.json"
        print(f"Starting Gram LR={lr}, two damping values; logs: {directory}", flush=True)
        with (directory / "driver.log").open("w") as stream:
            subprocess.run(prefix + ["benchmarks/anima/run_comparison.py", str(manifest)],
                           cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)
        subprocess.run(prefix + ["benchmarks/anima/report_comparison.py", str(manifest),
                                "--output", str(directory / "results.json")], cwd=root, check=True)
        print(f"Completed LR={lr}: {directory / 'results.md'}", flush=True)


if __name__ == "__main__":
    main()
