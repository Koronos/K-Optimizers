"""Serial, resumable Nekaon vs Adakaon vs AdamW campaign on Anima/Pets.

The campaign answers one question with paired seeds: on a rank-16 LoRA fine-tune of
Anima (Cosmos Predict2) over the Oxford-IIIT Pets subset at 256 px, how do
``kaon.Nekaon``, ``kaon.Adakaon`` (both with ``fused = true``) and
``torch.optim.AdamW(fused=True)`` compare on held-out epsilon-MSE, raw gap, active
training seconds, milliseconds per step and peak allocator memory.

Two phases:

* **Phase A** — learning-rate screen on a single seed (43): every arm at every
  candidate LR (9 runs).
* **Phase B** — confirmation seeds (44..47) at each arm's LR selected from phase A
  (12 runs).

The selection rule is fixed before the data is seen and recorded in every artifact:
``LR_SELECTION_RULE``. Phase A's selected-LR run doubles as the seed-43 member of the
confirmation set, so the paired comparison spans five seeds; the aggregator also
reports the four held-out seeds separately because seed 43 chose the LR.

The dataset is not re-prepared here: unless ``--pets-root`` overrides it, the runs read
``generate_comparison.DEFAULT_PETS_ROOT``, which points at the Pets subset already
prepared under the sibling ``.worktrees/diffusion-optimizer`` worktree
(``/mnt/c/.../K-Optimizers/.worktrees/diffusion-optimizer/tmp/pets/subset`` as seen from
WSL). That path is recorded in the manifest and in every generated TOML.

Runs execute strictly one at a time (single 8 GB GPU) and interleaved by arm
(ABC ABC ...) so thermal drift is spread across arms rather than concentrated in one.
The driver is resumable: a run whose ``*_results.json`` already exists is skipped.

Launch from the worktree in WSL::

    export PYTHONPATH="$PWD/src:$PWD"
    uv run --no-sync --project /home/koronos/Rengu-Flow python \\
        benchmarks/nekaon_evidence/anima/campaign.py

``--dry-run`` materializes and prints the queue without training.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

WORKTREE = Path(__file__).resolve().parents[3]
RFL_ROOT = "/home/koronos/Rengu-Flow"

ARMS: tuple[str, ...] = ("nekaon_fused", "adakaon_fused", "adamw_fused")
PHASE_A_SEED = 43
PHASE_A_LRS: tuple[float, ...] = (5.0e-5, 1.0e-4, 2.0e-4)
PHASE_B_SEEDS: tuple[int, ...] = (44, 45, 46, 47)
STEPS = 200
EVAL_EVERY = 100
EVAL_IMAGES = 8
RESOLUTION = 256

LR_SELECTION_RULE = (
    "per arm, the phase-A learning rate with the lowest final val/loss on the "
    "phase-A seed; exact ties go to the smaller learning rate"
)

_GENERATOR_PATH = WORKTREE / "benchmarks" / "anima" / "generate_comparison.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_comparison", _GENERATOR_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - packaging accident
        raise RuntimeError(f"cannot load {_GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lr_token(lr: float) -> str:
    """Stable, collision-free directory token for a learning rate."""
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    return f"lr{lr:.3e}"


@dataclasses.dataclass(frozen=True)
class RunSpec:
    """One training run in the queue. ``lr`` is None while phase A is unresolved."""

    phase: str
    arm: str
    seed: int
    lr: float | None

    @property
    def run_id(self) -> str:
        if self.lr is None:
            raise ValueError("run_id requires a resolved learning rate")
        return f"{self.phase}_seed{self.seed}_{lr_token(self.lr)}_{self.arm}"

    @property
    def resolved(self) -> bool:
        return self.lr is not None

    def directory(self, root: Path) -> Path:
        return root / self.run_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "arm": self.arm,
            "seed": self.seed,
            "lr": self.lr,
            "run_id": self.run_id if self.resolved else None,
        }


def build_queue(
    selected_lrs: dict[str, float] | None = None,
    *,
    arms: tuple[str, ...] = ARMS,
    phase_a_seed: int = PHASE_A_SEED,
    phase_a_lrs: tuple[float, ...] = PHASE_A_LRS,
    phase_b_seeds: tuple[int, ...] = PHASE_B_SEEDS,
) -> list[RunSpec]:
    """Return the full campaign queue, interleaved by arm within each block."""
    selected_lrs = selected_lrs or {}
    queue: list[RunSpec] = []
    for lr in phase_a_lrs:
        for arm in arms:
            queue.append(RunSpec(phase="a", arm=arm, seed=phase_a_seed, lr=lr))
    for seed in phase_b_seeds:
        for arm in arms:
            queue.append(RunSpec(phase="b", arm=arm, seed=seed, lr=selected_lrs.get(arm)))
    return queue


def results_dir(root: Path) -> Path:
    return root / "results"


def results_path(root: Path, spec: RunSpec) -> Path:
    return results_dir(root) / f"{spec.run_id}_results.json"


def is_complete(root: Path, spec: RunSpec) -> bool:
    return spec.resolved and results_path(root, spec).is_file()


def pending(queue: list[RunSpec], root: Path) -> list[RunSpec]:
    """Runs that still need to execute: resolved LR and no final results file."""
    return [spec for spec in queue if spec.resolved and not is_complete(root, spec)]


# --------------------------------------------------------------------------- LR rule


def _final_value(record: dict[str, Any], metric: str) -> float | None:
    points = ((record.get("run") or {}).get(metric) or {}).get("points")
    if not points:
        return None
    return float(max(points, key=lambda point: point["step"])["value"])


def select_lrs(records: list[dict[str, Any]], *, arms: tuple[str, ...] = ARMS) -> dict[str, float]:
    """Apply ``LR_SELECTION_RULE`` to completed phase-A records."""
    best: dict[str, float] = {}
    for arm in arms:
        candidates: list[tuple[float, float]] = []
        for record in records:
            if record.get("phase") != "a" or record.get("arm") != arm:
                continue
            value = _final_value(record, "val")
            if value is None or not math.isfinite(value):
                raise ValueError(f"{record.get('run_id')}: no finite final val/loss")
            candidates.append((value, float(record["lr"])))
        if not candidates:
            continue
        best[arm] = min(candidates, key=lambda item: (item[0], item[1]))[1]
    return best


def load_records(root: Path) -> list[dict[str, Any]]:
    directory = results_dir(root)
    if not directory.is_dir():
        return []
    records = []
    for path in sorted(directory.glob("*_results.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def resolve_lrs(root: Path, *, arms: tuple[str, ...] = ARMS, phase_a_lrs: tuple[float, ...] = PHASE_A_LRS) -> dict[str, float]:
    """Selected LRs, but only once phase A is complete for every arm."""
    records = load_records(root)
    counts = {arm: sum(1 for r in records if r.get("phase") == "a" and r.get("arm") == arm) for arm in arms}
    if any(counts[arm] < len(phase_a_lrs) for arm in arms):
        return {}
    return select_lrs(records, arms=arms)


# ----------------------------------------------------------------------- provenance


def kaon_version() -> str:
    """Read the version without importing torch (this driver plans, it does not train)."""
    text = (WORKTREE / "src" / "kaon" / "_version.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not match:  # pragma: no cover - defensive
        raise RuntimeError("cannot parse kaon version")
    return match[1]


def _run_git(arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments, "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _wsl_git_dir() -> str | None:
    """Translate this worktree's Windows ``gitdir`` pointer into a WSL path.

    A worktree created from Windows stores an absolute ``C:/...`` gitdir, which git
    inside WSL cannot follow; the commit would silently be recorded as unknown.
    """
    pointer = WORKTREE / ".git"
    if os.name == "nt" or not pointer.is_file():
        return None
    try:
        text = pointer.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text.startswith("gitdir:"):
        return None
    path = text.split(":", 1)[1].strip()
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", path)
    if not match:
        return None
    return f"/mnt/{match[1].lower()}/" + match[2].replace("\\", "/")


def git_commit() -> str | None:
    commit = _run_git(["-C", str(WORKTREE)])
    if commit:
        return commit
    git_dir = _wsl_git_dir()
    return _run_git([f"--git-dir={git_dir}"]) if git_dir else None


def provenance() -> dict[str, Any]:
    return {"kaon_version": kaon_version(), "commit": git_commit(), "worktree": WORKTREE.as_posix()}


# ---------------------------------------------------------------------- power state

GPU_POWER_FIELDS = (
    "power.limit", "power.max_limit", "power.default_limit", "clocks.max.sm", "temperature.gpu",
)


def gpu_power_state() -> dict[str, Any]:
    """``nvidia-smi`` power/clock/temperature snapshot, or an error record."""
    command = ["nvidia-smi", f"--query-gpu={','.join(GPU_POWER_FIELDS)}", "--format=csv,noheader"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)}
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return {"available": False, "error": "nvidia-smi returned no rows"}
    values = [value.strip() for value in lines[0].split(",")]
    return {
        "available": True,
        "raw": lines[0],
        "fields": dict(zip(GPU_POWER_FIELDS, values, strict=False)),
        "gpu_count": len(lines),
    }


def _ac_from_sysfs() -> bool | None:
    supplies = sorted(Path("/sys/class/power_supply").glob("*")) if os.name != "nt" else []
    for supply in supplies:
        try:
            if (supply / "type").read_text(encoding="utf-8").strip() != "Mains":
                continue
            return (supply / "online").read_text(encoding="utf-8").strip() == "1"
        except OSError:
            continue
    return None


def _ac_from_powershell() -> bool | None:
    executable = "powershell" if os.name == "nt" else "powershell.exe"
    command = [executable, "-NoProfile", "-Command", "(Get-CimInstance Win32_Battery).BatteryStatus"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    codes = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not codes:
        return None
    # Win32_Battery.BatteryStatus: 1 = discharging, 2 = on AC line power.
    return all(code == "2" for code in codes)


def ac_power_state() -> dict[str, Any]:
    """Whether the laptop is on wall power. ``on_ac`` is None when undetermined.

    This matters: the GPU runs at a 60 W limit on AC and 35 W on battery, so timings
    taken in different electrical states are not comparable.
    """
    sysfs = _ac_from_sysfs()
    shell = _ac_from_powershell()
    on_ac = sysfs if sysfs is not None else shell
    return {
        "on_ac": on_ac,
        "sysfs_mains_online": sysfs,
        "win32_battery_on_ac": shell,
        "source": "sysfs" if sysfs is not None else ("win32_battery" if shell is not None else "unknown"),
    }


def power_state() -> dict[str, Any]:
    return {"ac": ac_power_state(), "gpu": gpu_power_state()}


def power_signature(state: dict[str, Any]) -> str:
    """Compact comparability key: runs with different signatures are not comparable."""
    ac = state.get("ac") or {}
    fields = (state.get("gpu") or {}).get("fields") or {}
    on_ac = ac.get("on_ac")
    label = "ac" if on_ac is True else "battery" if on_ac is False else "unknown"
    limit = fields.get("power.max_limit", "unknown")
    return f"{label}/max_limit={limit}"


def require_ac_power(state: dict[str, Any]) -> None:
    """Refuse to measure on battery (or with an undetermined electrical state)."""
    ac = state.get("ac") or {}
    if ac.get("on_ac") is True:
        return
    if ac.get("on_ac") is False:
        raise SystemExit(
            "Refusing to start: the laptop is on battery. The GPU power limit drops from "
            "60 W to 35 W, so timings would not be comparable with the rest of the campaign. "
            "Plug in the charger and rerun; completed runs are kept and skipped."
        )
    raise SystemExit(
        "Refusing to start: could not determine whether the laptop is on AC power "
        f"(probes: {ac}). Timing measurements require a known 60 W AC state."
    )


# ------------------------------------------------------------------------ materialize


def materialize(
    spec: RunSpec,
    root: Path,
    *,
    pets_root: str | None = None,
    steps: int = STEPS,
    eval_every: int = EVAL_EVERY,
    eval_images: int = EVAL_IMAGES,
    resolution: int = RESOLUTION,
) -> Path:
    """Write the arm's TOML (and its dataset TOMLs) and return the config path."""
    if not spec.resolved:
        raise ValueError(f"cannot materialize {spec.arm}/seed {spec.seed}: learning rate unresolved")
    generator = _load_generator()
    directory = spec.directory(root)
    generator.generate(
        output_dir=directory,
        pets_root=(pets_root or generator.DEFAULT_PETS_ROOT).rstrip("/"),
        steps=steps,
        lr=spec.lr,
        seed=spec.seed,
        previews=False,
        gap_threshold=None,
        eval_images=eval_images,
        eval_every=eval_every,
        arms=(spec.arm,),
        resolution=resolution,
    )
    return directory / f"{spec.arm}.toml"


def write_manifest(
    root: Path,
    queue: list[RunSpec],
    selected_lrs: dict[str, float],
    *,
    steps: int,
    eval_every: int,
    eval_images: int,
    resolution: int,
    pets_root: str,
) -> dict[str, Any]:
    manifest = {
        "schema": "nekaon-evidence-anima-campaign-v1",
        "generated_by": "benchmarks/nekaon_evidence/anima/campaign.py",
        "provenance": provenance(),
        "protocol": {
            "arms": list(ARMS),
            "phase_a_seed": PHASE_A_SEED,
            "phase_a_lrs": list(PHASE_A_LRS),
            "phase_b_seeds": list(PHASE_B_SEEDS),
            "steps": steps,
            "eval_every_n_steps": eval_every,
            "eval_images": eval_images,
            "resolution": resolution,
            "previews": False,
            "pets_root": pets_root,
            "lr_selection_rule": LR_SELECTION_RULE,
            "execution": "serial, single GPU, interleaved by arm (ABC ABC ...)",
        },
        "power": power_state(),
        "power_note": (
            "This is a laptop: the GPU runs at a 60 W limit on AC and 35 W on battery. "
            "The driver refuses to start a run off AC, and every run records its own state."
        ),
        "selected_lrs": selected_lrs,
        "queue": [
            dict(spec.as_dict(), state=("done" if is_complete(root, spec) else "pending" if spec.resolved else "unresolved"))
            for spec in queue
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "campaign.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------- execute


def summarize_completed_run(log_text: str, expected_steps: int) -> tuple[str, dict[str, Any]]:
    """Validate a finished trainer log and summarize its TensorBoard/bench output.

    Imports ``summarize_runs`` lazily: it needs TensorBoard, which only the training
    environment has.
    """
    if "Training complete." not in log_text:
        raise RuntimeError("trainer log has no 'Training complete.' marker")
    fingerprint = re.search(r"adapter_initial_sha256=([0-9a-f]{64})", log_text)
    directory = re.search(r"^Run dir: (.+)$", log_text, re.MULTILINE)
    if not fingerprint or not directory:
        raise RuntimeError("trainer log is missing the adapter fingerprint or run directory")
    sys.path.insert(0, str(WORKTREE / "benchmarks" / "anima"))
    try:
        from summarize_runs import summarize_run  # noqa: PLC0415 - training-only dependency
    finally:
        sys.path.pop(0)
    run = summarize_run(Path(directory[1].strip()))
    for metric in ("train_eval", "val"):
        points = (run.get(metric) or {}).get("points")
        if not points or points[0]["step"] != 0 or points[-1]["step"] != expected_steps:
            raise RuntimeError(f"missing initial/final {metric} evaluation")
    if len(run["bench_csv"]) != 1:
        raise RuntimeError("expected exactly one bench_steps.csv")
    steps = [point["step"] for point in run["bench_csv"][0]["points"]]
    if steps != list(range(1, expected_steps + 1)):
        raise RuntimeError("missing or repeated training steps in bench_steps.csv")
    return fingerprint[1], run


def execute(
    spec: RunSpec,
    root: Path,
    config: Path,
    *,
    steps: int,
    eval_every: int,
    eval_images: int,
    resolution: int,
    echo: bool = True,
) -> Path:
    """Run one arm to completion and write its ``*_results.json``.

    Refuses to start off AC power: the 35 W battery limit would make the timing
    numbers incomparable with the AC runs.
    """
    power_before = power_state()
    require_ac_power(power_before)
    directory = spec.directory(root)
    log_path = directory / "run.log"
    env = os.environ.copy()
    env["ANIMA_INIT_SEED"] = str(spec.seed)
    env["PYTHONPATH"] = os.pathsep.join([str(WORKTREE / "src"), str(WORKTREE)])
    command = [
        "uv", "run", "--no-sync", "--project", RFL_ROOT,
        "deepspeed", "--num_gpus=1", "--module", "benchmarks.anima.run_seeded",
        "--config", str(config),
    ]
    with (
        log_path.open("w", encoding="utf-8") as log,
        subprocess.Popen(command, cwd=WORKTREE, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1) as process,
    ):
        for line in process.stdout:  # type: ignore[union-attr]
            log.write(line)
            log.flush()
            if echo:
                print(line, end="", flush=True)
        code = process.wait()
    if code:
        raise SystemExit(f"{spec.run_id} failed with exit code {code}; see {log_path}")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    sha, run = summarize_completed_run(log_text, steps)
    parameters = re.search(r"optimizer_parameter_count=(\d+)", log_text)
    record = {
        "schema": "nekaon-evidence-anima-run-v1",
        "run_id": spec.run_id,
        "phase": spec.phase,
        "arm": spec.arm,
        "seed": spec.seed,
        "lr": spec.lr,
        "steps": steps,
        "eval_every_n_steps": eval_every,
        "eval_images": eval_images,
        "resolution": resolution,
        "provenance": provenance(),
        "power_before": power_before,
        "power_after": power_state(),
        "power_signature": power_signature(power_before),
        "lr_selection_rule": LR_SELECTION_RULE,
        "config_path": config.as_posix(),
        "config": config.read_text(encoding="utf-8"),
        "adapter_initial_sha256": sha,
        "optimizer_parameter_count": int(parameters[1]) if parameters else None,
        "log_path": log_path.as_posix(),
        "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
        "run": run,
    }
    destination = results_path(root, spec)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return destination


# ------------------------------------------------------------------------------ CLI


def _print_queue(queue: list[RunSpec], root: Path, selected_lrs: dict[str, float]) -> None:
    print(f"campaign root: {root}")
    state = power_state()
    print(f"power: {power_signature(state)}  gpu={(state['gpu'].get('raw') or state['gpu'].get('error'))}")
    print(f"LR selection rule: {LR_SELECTION_RULE}")
    print(f"selected LRs: {selected_lrs or '(pending phase A)'}")
    print(f"{len(queue)} runs total")
    print(f"{'#':>3}  {'phase':<5}  {'seed':>4}  {'lr':>10}  {'arm':<14}  {'state':<10}  config")
    for index, spec in enumerate(queue, start=1):
        if not spec.resolved:
            print(f"{index:>3}  {spec.phase:<5}  {spec.seed:>4}  {'pending':>10}  {spec.arm:<14}  "
                  f"{'unresolved':<10}  (awaits phase A)")
            continue
        state = "done" if is_complete(root, spec) else "pending"
        config = spec.directory(root) / f"{spec.arm}.toml"
        print(f"{index:>3}  {spec.phase:<5}  {spec.seed:>4}  {spec.lr:>10.3e}  {spec.arm:<14}  "
              f"{state:<10}  {config}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=WORKTREE / "tmp" / "nekaon-evidence-anima")
    parser.add_argument("--pets-root", default=None)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--phase", choices=("a", "b", "all"), default="all")
    parser.add_argument("--lrs", nargs="+", type=float, default=list(PHASE_A_LRS))
    parser.add_argument("--phase-a-seed", type=int, default=PHASE_A_SEED)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PHASE_B_SEEDS))
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--eval-images", type=int, default=EVAL_IMAGES)
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    parser.add_argument("--limit", type=int, default=None, help="run at most this many pending runs")
    parser.add_argument("--dry-run", action="store_true", help="materialize and print the queue, train nothing")
    parser.add_argument(
        "--planning-lr", type=float, default=None,
        help="dry-run only: pretend phase A selected this LR for every arm so the whole queue materializes",
    )
    args = parser.parse_args(argv)

    if args.planning_lr is not None and not args.dry_run:
        parser.error("--planning-lr is a dry-run planning aid only")

    generator = _load_generator()
    pets_root = (args.pets_root or generator.DEFAULT_PETS_ROOT).rstrip("/")
    arms = tuple(args.arms)
    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)

    selected = resolve_lrs(root, arms=arms, phase_a_lrs=tuple(args.lrs))
    if not selected and args.planning_lr is not None:
        selected = {arm: args.planning_lr for arm in arms}
    queue = build_queue(
        selected,
        arms=arms,
        phase_a_seed=args.phase_a_seed,
        phase_a_lrs=tuple(args.lrs),
        phase_b_seeds=tuple(args.seeds),
    )
    if args.phase != "all":
        queue = [spec for spec in queue if spec.phase == args.phase]

    materialize_kwargs = dict(
        pets_root=pets_root, steps=args.steps, eval_every=args.eval_every,
        eval_images=args.eval_images, resolution=args.resolution,
    )
    for spec in queue:
        if spec.resolved:
            materialize(spec, root, **materialize_kwargs)
    write_manifest(root, queue, selected, steps=args.steps, eval_every=args.eval_every,
                   eval_images=args.eval_images, resolution=args.resolution, pets_root=pets_root)
    _print_queue(queue, root, selected)
    if args.dry_run:
        remaining = len(pending(queue, root))
        print(f"\ndry run: {remaining} run(s) would execute; nothing was trained")
        return 0
    if os.name == "nt":
        parser.error("training must be launched from WSL; the Rengu-Flow environment is Linux-only")

    executed = 0
    while True:
        selected = resolve_lrs(root, arms=arms, phase_a_lrs=tuple(args.lrs)) or selected
        queue = build_queue(selected, arms=arms, phase_a_seed=args.phase_a_seed,
                            phase_a_lrs=tuple(args.lrs), phase_b_seeds=tuple(args.seeds))
        if args.phase != "all":
            queue = [spec for spec in queue if spec.phase == args.phase]
        todo = pending(queue, root)
        if not todo or (args.limit is not None and executed >= args.limit):
            break
        spec = todo[0]
        print(f"\n=== [{executed + 1}] {spec.run_id} ===", flush=True)
        config = materialize(spec, root, **materialize_kwargs)
        execute(spec, root, config, steps=args.steps, eval_every=args.eval_every,
                eval_images=args.eval_images, resolution=args.resolution)
        executed += 1
        write_manifest(root, queue, selected, steps=args.steps, eval_every=args.eval_every,
                       eval_images=args.eval_images, resolution=args.resolution, pets_root=pets_root)

    unresolved = [spec for spec in queue if not spec.resolved]
    print(f"\ncompleted {executed} run(s) this invocation; "
          f"{len(pending(queue, root))} pending, {len(unresolved)} awaiting phase A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
