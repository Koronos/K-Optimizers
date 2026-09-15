"""Isolated optimizer-step timing and allocated memory, no fwd/bwd.

Run separately from training jobs. Peak allocation excludes model activations;
the adapter bag is a launch-cost proxy, not a LoRA quality experiment.
"""
import json
import statistics
import time
from pathlib import Path

import torch
from control import battery as B

from kaon import Adakaon, Nekaon, Rakaon


def measure(factory, params):
    for p in params:
        p.grad = torch.randn_like(p)
    torch.cuda.synchronize()
    before_optimizer = torch.cuda.memory_allocated()
    opt = factory(params)
    for _ in range(10):
        opt.step()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(40):
        start = time.perf_counter()
        opt.step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return dict(median_step_ms=statistics.median(times),
                state_bpp=B.H.opt_state_bytes_per_param(opt, params),
                persistent_allocated_bytes=baseline - before_optimizer,
                step_extra_peak_bytes=torch.cuda.max_memory_allocated() - baseline)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This performance protocol requires CUDA")
    factories = {
        "AdamW-fused": lambda p: torch.optim.AdamW(p, lr=.0012, fused=True),
        "Adakaon": lambda p: Adakaon(p, lr=.0012, betas=(0., .999), cautious=False),
        "Nekaon": lambda p: Nekaon(p, lr=.0012, betas=(.5, .999), weight_decay=.3, momentum_dtype="4bit"),
        "Rakaon-0.1": lambda p: Rakaon(p, lr=.0012, shrinkage=.1),
        "Rakaon-1": lambda p: Rakaon(p, lr=.0012, shrinkage=1),
    }
    results = dict(device=torch.cuda.get_device_name(), torch=torch.__version__, runs=[])
    for dtype in (torch.float32, torch.bfloat16):
        for regime in ("unet128", "adapter_bag"):
            for name, factory in factories.items():
                torch.manual_seed(42)
                if regime == "unet128":
                    params = list(B.H.UNet(C=128).to("cuda", dtype=dtype).parameters())
                else:
                    params = [torch.nn.Parameter(p.detach().to(dtype)) for p in B.H.lora_bag()]
                row = dict(optimizer=name, regime=regime, dtype=str(dtype), **measure(factory, params))
                results["runs"].append(row)
                print(json.dumps(row), flush=True)
                del params
    Path("benchmarks/rakaon_performance.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
