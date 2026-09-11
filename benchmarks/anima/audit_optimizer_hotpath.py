"""Optimizer-only CUDA audit on real adapter shapes; no training-quality claim."""
import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

from kaon import Adakaon, Nekaon


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fused", action="store_true")
    args = parser.parse_args()
    with safe_open(args.adapter, framework="pt", device="cpu") as source:
        shapes = [tuple(source.get_slice(key).get_shape()) for key in source.keys()  # noqa: SIM118 -- safe_open is not a dict
                  if "lora_" in key and "llm_adapter" not in key]
    if not shapes:
        raise ValueError("No adapter tensors")
    results = []
    for name in ("adakaon", "k0", "nekaon", "nekaon_warning_disabled"):
        torch.manual_seed(45)
        params = [torch.nn.Parameter(torch.randn(shape, device="cuda", dtype=torch.bfloat16) * .01)
                  for shape in shapes]
        for p in params:
            p.grad = torch.randn_like(p) * .01
        kwargs = dict(lr=1e-5, betas=(.5, .999), weight_decay=.1, momentum_dtype="4bit",
                      cautious=True, gradient_centralization=True, fused=args.fused)
        opt = Adakaon(params, **kwargs) if name == "adakaon" else Nekaon(params, k=0 if name == "k0" else 1.5, **kwargs)
        if name == "nekaon_warning_disabled":
            opt._inert_checks = opt._INERT_MAX_CHECKS
        for _ in range(5):
            opt.step()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            opt.step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 1000 / 20
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            opt.step()
            torch.cuda.synchronize()
        events = {e.key: e.count for e in prof.key_averages()
                  if any(s in e.key for s in ("_local_scalar_dense", "aten::stack", "aten::clone"))}
        result = dict(variant=name, fused=args.fused, tensors=len(params), parameters=sum(p.numel() for p in params),
                      mean_step_ms=ms, cpu_operator_counts=events)
        results.append(result)
        print(json.dumps(result), flush=True)
        del opt, params, p
        torch.cuda.empty_cache()
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
