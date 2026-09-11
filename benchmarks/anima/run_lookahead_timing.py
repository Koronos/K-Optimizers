"""Bounded ABBA timing control at 1024px; no convergence claims."""
import argparse
import json
import os
import re
import statistics
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fused", action="store_true", help="Enable fused inner Adakaon in every arm")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = args.source.read_text()
    results = []
    env = {**os.environ, "PYTHONPATH": f"{root / 'src'}:{root}",
           "ANIMA_INIT_SEED": "45", "ANIMA_OPT_TIMING": "1"}
    for index, k in enumerate((0.0, 1.5, 1.5, 0.0)):
        name = f"run{index}_k{k:g}"
        config = source
        if args.fused:
            config = config.replace("[optimizer]", "[optimizer]\nfused = true")
        replacements = {"run_name": f'"{name}"', "output_dir": f'"{output / name}"',
                        "max_steps": "4", "eval_before_first_step": "false",
                        "eval_every_n_steps": "999999", "k": str(k)}
        for key, value in replacements.items():
            config, count = re.subn(rf"^{key} = .*$", f"{key} = {value}", config, flags=re.MULTILINE)
            if count != 1:
                raise ValueError(f"Expected exactly one {key}, found {count}")
        config_path = output / f"{name}.toml"
        config_path.write_text(config)
        telemetry = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used",
             "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()
        with (output / f"{name}.log").open("w") as log:
            subprocess.run(["uv", "run", "--no-sync", "--project", "/home/koronos/Rengu-Flow",
                            "deepspeed", "--num_gpus=1", "--module", "benchmarks.anima.run_seeded",
                            "--config", str(config_path)], cwd=root, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        log = (output / f"{name}.log").read_text()
        if "Training complete." not in log:
            raise RuntimeError(f"Incomplete {name}")
        step_times = [float(v) for v in re.findall(r"\[bench\].*?iter_sec=([0-9.]+)", log)]
        opt_times = [float(v) for v in re.findall(r"\[optimizer diagnostic\] seconds=([0-9.]+)", log)]
        if len(step_times) != 4 or len(opt_times) != 4:
            raise RuntimeError(f"Incomplete timing {name}")
        results.append(dict(name=name, k=k, step_seconds=step_times, optimizer_seconds=opt_times,
                            median_step_after_first=statistics.median(step_times[1:]),
                            median_optimizer_after_first=statistics.median(opt_times[1:]),
                            telemetry_before=telemetry))
        (output / "results.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(results[-1]), flush=True)


if __name__ == "__main__":
    main()
