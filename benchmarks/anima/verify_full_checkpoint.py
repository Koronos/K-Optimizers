"""Check that selected full-model matrices changed after the compatibility smoke."""
import argparse
import json

import torch
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original")
    parser.add_argument("trained")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = []
    with safe_open(args.original, framework="pt", device="cpu") as original, \
            safe_open(args.trained, framework="pt", device="cpu") as trained:
        keys = sorted(set(original.keys()) & set(trained.keys()))
        for key in keys:
            shape = original.get_slice(key).get_shape()
            if len(shape) != 2 or min(shape) < 128 or "blocks.0." not in key:
                continue
            before, after = original.get_tensor(key), trained.get_tensor(key)
            if (before.shape != after.shape or before.dtype != after.dtype
                    or not torch.isfinite(after).all()):
                raise RuntimeError(f"Invalid saved tensor: {key}")
            changed = torch.count_nonzero(before != after).item()
            records.append(dict(key=key, shape=shape, changed_elements=changed,
                                elements=before.numel(), dtype=str(before.dtype),
                                maximum_absolute_change=(before.float() - after.float()).abs().max().item()))
            if len(records) == 8:
                break
    if not records or not all(record["changed_elements"] > 0 for record in records):
        raise RuntimeError("Selected full-model matrices did not all change")
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(dict(original=args.original, trained=args.trained, checked=records,
                       scope="Selected matrices only; this is not a quality evaluation"), handle, indent=2)


if __name__ == "__main__":
    main()
