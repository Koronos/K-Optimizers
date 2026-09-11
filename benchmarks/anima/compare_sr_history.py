"""Compare completed SR arms directly and tabulate historical protocols separately."""
import argparse
import json
from pathlib import Path


def metrics(run):
    train = run["train_eval"]["points"][-1]["value"]
    val = run["val"]["points"][-1]["value"]
    bench = run["bench_csv"][0]["points"]
    return val, abs(val - train), bench[-1]["active_train_seconds"], max(p["cuda_peak_gb"] for p in bench)


def render(current, historical):
    runs = {r["arm"]: r for r in current["runs"]}
    candidate = metrics(runs["nekaon_sr_host"])
    lines = ["# SR lookahead: controlled comparison and historical context", "",
             "The current report validates matching initialization, initial metrics and complete training/evaluation records.",
             "Differences below are candidate minus control; negative is better for each listed metric.", "",
             "| Control | Delta val | Delta abs gap | Delta active seconds | Delta peak GiB |",
             "|---|---:|---:|---:|---:|"]
    for name in ("nekaon_k0", "nekaon"):
        baseline = metrics(runs[name])
        delta = [a - b for a, b in zip(candidate, baseline, strict=True)]
        lines.append(f"| {name} | {delta[0]:+.8f} | {delta[1]:+.8f} | {delta[2]:+.2f} | {delta[3]:+.3f} |")
    lines += ["", "## Historical results — different protocols, not a paired ranking", "",
              "| Report | Arm | Seed | Pixels | LR | Steps | Eval images/split | Val | Abs gap | Active s | Peak GiB |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, report in [("current", current), *historical]:
        manifest = report["protocol"]
        protocol, dataset = manifest["protocol"], manifest["dataset"]
        for run in report["runs"]:
            val, gap, seconds, peak = metrics(run)
            lines.append(f"| {name} | {run['arm']} | {protocol['train_seed']} | {dataset['resolution']} | "
                         f"{protocol['lr']:g} | {protocol['steps']} | {dataset['val']['max_images']} | "
                         f"{val:.8f} | {gap:.8f} | {seconds:.2f} | {peak:.3f} |")
    lines += ["", "Resolution, LR, seed, number of steps and evaluation subsets differ across reports. "
              "Do not infer a quality gain or slowdown from historical absolute loss/time differences.",
              "This single-seed screen provides no uncertainty estimate or perceptual-detail measurement. "
              "Active time excludes evaluation; laptop thermals and run order can affect timing. "
              "An absolute gap can shrink because training deteriorates, so inspect validation and train loss together.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("current", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent
    names = ("confirmation43", "confirmation44", "adam43", "adam44", "fast43",
             "momentum43", "gram43_lr0", "gram43_lr1")
    history = [(name, json.loads((root / f"{name}_results.json").read_text())) for name in names]
    args.output.write_text(render(json.loads(args.current.read_text()), history), encoding="utf-8")


if __name__ == "__main__":
    main()
