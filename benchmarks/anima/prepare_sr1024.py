"""Prepare the BF16-only low-LR comparison plus a three-step 1024px smoke."""
import argparse
from pathlib import Path

from benchmarks.anima.generate_comparison import DEFAULT_PETS_ROOT, _optimizer_config, generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    if directory.exists():
        raise FileExistsError("Use a fresh output directory")
    generate(output_dir=directory, pets_root=DEFAULT_PETS_ROOT, steps=100, lr=1e-5,
             seed=45, previews=False, gap_threshold=None, eval_images=8, eval_every=50,
             resolution=1024, arms=("nekaon_k0", "nekaon", "nekaon_sr_host"))
    source = Path(__file__).with_name("smoke.toml").read_text()
    dataset = Path(__file__).with_name("smoke_dataset.toml").read_text().replace("[256]", "[1024]")
    (directory / "smoke_dataset.toml").write_text(dataset)
    start, tail = source.split("[optimizer]")
    tail = tail[tail.index("[preview]"):]
    old_dataset = next(line for line in start.splitlines() if line.startswith("dataset ="))
    old_output = next(line for line in start.splitlines() if line.startswith("output_dir ="))
    start = start.replace(old_dataset, f'dataset = "{directory / "smoke_dataset.toml"}"')
    start = start.replace(old_output, f'output_dir = "{directory / "smoke_runs"}"')
    smoke = start + "[optimizer]\n" + "\n".join(_optimizer_config("nekaon_sr_host", 1e-5)) + "\n\n" + tail
    (directory / "smoke.toml").write_text(smoke)
    print(directory / "manifest.json")


if __name__ == "__main__":
    main()
