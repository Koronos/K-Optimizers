"""Report completed, fingerprint-matched Anima pilot runs without ranking claims."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from summarize_runs import summarize_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/anima/comparison_results.json"))
    parser.add_argument("--arms", nargs="+", help="Explicit subset of completed arms")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    runs = []
    by_name = {arm["arm"]: arm for arm in manifest["arms"]}
    for name in args.arms or by_name:
        arm = by_name[name]
        log_path = args.manifest.parent / f"{arm['arm']}.log"
        log = log_path.read_text()
        if "Training complete." not in log:
            raise RuntimeError(f"Incomplete run: {log_path}")
        fingerprint = re.search(r"adapter_initial_sha256=([0-9a-f]{64})", log)
        directory = re.search(r"^Run dir: (.+)$", log, re.MULTILINE)
        if not fingerprint or not directory:
            raise RuntimeError(f"Missing run provenance: {log_path}")
        run = summarize_run(Path(directory[1].strip()))
        expected = manifest["protocol"]["steps"]
        for metric in ("train_eval", "val"):
            points = run[metric]["points"]
            if not points or points[0]["step"] != 0 or points[-1]["step"] != expected:
                raise RuntimeError(f"Missing initial/final {metric} evaluation: {log_path}")
        if len(run["bench_csv"]) != 1:
            raise RuntimeError(f"Expected one bench CSV: {log_path}")
        steps = [point["step"] for point in run["bench_csv"][0]["points"]]
        if steps != list(range(1, expected + 1)):
            raise RuntimeError(f"Missing or repeated training steps: {log_path}")
        run.update(arm=arm["arm"], adapter_initial_sha256=fingerprint[1],
                   config=Path(arm["config"]).read_text(),
                   log_sha256=hashlib.sha256(log_path.read_bytes()).hexdigest())
        runs.append(run)
    if len({run["adapter_initial_sha256"] for run in runs}) != 1:
        raise RuntimeError("Initial adapter weights differ; comparison is not paired")
    for metric in ("train_eval", "val"):
        initial = [run[metric]["points"][0]["value"] for run in runs]
        if not all(math.isclose(value, initial[0], abs_tol=1e-6) for value in initial):
            raise RuntimeError(f"Initial {metric} loss differs despite matching adapter weights")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"protocol": manifest, "reported_arms": [run["arm"] for run in runs],
                                      "runs": runs}, indent=2) + "\n")
    lines = ["# Anima / Pets pilot", "",
             f"Seed {manifest['protocol']['train_seed']}, constant LR {manifest['protocol']['lr']:g}, "
             f"rank-16 LoRA, {manifest['protocol']['steps']} steps at {manifest['dataset']['resolution']}px.",
             f"{manifest['dataset']['val']['max_images']} fixed images per evaluation split, nine noise quantiles.",
             "Initialization fingerprints match. This is a pilot, not a tuned ranking.", "",
             "| Optimizer | Train eval | Val | Raw gap | Change in gap | Active train s | Peak GiB |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for run in runs:
        train = run["train_eval"]["points"][-1]["value"]
        val = run["val"]["points"][-1]["value"]
        gap = run["gap"]["points"][-1]["value"]
        excess = run["excess_gap"]["points"][-1]["value"]
        bench = run["bench_csv"][0]["points"]
        seconds = bench[-1]["active_train_seconds"]
        peak = max(point["cuda_peak_gb"] for point in bench)
        lines.append(f"| {run['arm']} | {train:.6f} | {val:.6f} | {gap:+.6f} | {excess:+.6f} | {seconds:.1f} | {peak:.3f} |")
    lines += ["", "Raw gap = val − train eval. Change in gap subtracts the step-zero gap;",
              "it is descriptive, not a generalization bound. Active time excludes evaluation",
              "and previews; allocator peaks include prior preview allocations. Run order and",
              "laptop thermals can affect timing. No FID, perceptual ranking, multi-seed",
              "confirmation within this single-seed report, or full fine-tuning claim follows from this pilot.", ""]
    args.output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
