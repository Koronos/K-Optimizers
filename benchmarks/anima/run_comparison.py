"""Run generated Anima arms sequentially, stopping on the first failed trainer."""
import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--arms", nargs="+")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    worktree = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["ANIMA_INIT_SEED"] = str(manifest["protocol"]["adapter_init_seed"])
    env["PYTHONPATH"] = os.pathsep.join([str(worktree / "src"), str(worktree)])
    arms = {arm["arm"]: arm for arm in manifest["arms"]}
    for name in args.arms or arms:
        arm = arms[name]
        config = Path(arm["config"])
        if not config.is_file():
            raise FileNotFoundError(config)
        command = ["uv", "run", "--no-sync", "--project", "/home/koronos/Rengu-Flow",
                   "deepspeed", "--num_gpus=1", "--module", "benchmarks.anima.run_seeded",
                   "--config", str(config)]
        with (
            (args.manifest.parent / f"{name}.log").open("w", encoding="utf-8") as log,
            subprocess.Popen(command, cwd=worktree, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1) as process,
        ):
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            result = process.wait()
        if result:
            raise SystemExit(f"{name} failed with exit code {result}; see its log")


if __name__ == "__main__":
    main()
