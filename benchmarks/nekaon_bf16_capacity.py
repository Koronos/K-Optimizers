"""Upper bound on representable Nekaon lookahead in a saved BF16 LoRA."""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def measure(path):
    totals = {lr: {kind: [0, 0] for kind in ("A", "B")} for lr in (1e-4, 1e-5, 1e-6)}
    with safe_open(str(path), framework="pt", device="cpu") as archive:
        for key in archive.keys():  # noqa: SIM118 — safe_open exposes keys(), not dict iteration.
            if "llm_adapter" in key:
                continue  # This group had LR zero in the recorded Anima experiment.
            kind = "A" if "lora_A" in key else "B" if "lora_B" in key else None
            if kind is None:
                raise ValueError(f"unrecognized trainable factor: {key}")
            weight = archive.get_tensor(key)
            if weight.dtype != torch.bfloat16:
                raise ValueError("expected the actual BF16 adapter checkpoint")
            for lr, groups in totals.items():
                bound = 1.5 * lr  # k=1.5 and clip_threshold=1, actual per-coordinate cap.
                move = ((weight.float() + bound).bfloat16() != weight) | ((weight.float() - bound).bfloat16() != weight)
                groups[kind][0] += int(move.sum())
                groups[kind][1] += weight.numel()
    return {str(lr): {kind: dict(parameters=n, maximum_movable_fraction=m / n)
                     for kind, (m, n) in groups.items()} for lr, groups in totals.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = dict(checkpoint=str(args.checkpoint), k=1.5, clip_threshold=1,
                  caveat="Upper bound using maximum allowed displacement, NOT actual momentum or gradient change. Frozen llm_adapter excluded.",
                  results=measure(args.checkpoint))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
