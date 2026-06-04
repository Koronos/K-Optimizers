"""``ktune`` — check whether Adafusion's ``foreach`` knobs need tuning on *your*
GPU and *your* model.

The defaults (``foreach_batch_cutoff=2_000_000``, adaptive ``foreach_stack_budget``)
were tuned on an RTX 4080 against SDXL and Cosmos. The performance cutoff is an
absolute element count tied to a hardware crossover, so a very different GPU *might*
prefer a different value. This tool measures it directly: it reads the parameter
shapes from a model checkpoint (``.safetensors`` header only — no full load), builds
matching tensors on your GPU, sweeps the cutoff, and tells you whether to keep the
default or change it.

Usage (from the repo):

    uv run ktune --model /path/to/unet.safetensors --gpu 0
    uv run ktune --model /path/to/sdxl.safetensors --filter model.diffusion_model.
    uv run ktune --model /path/to/unet.safetensors --lora-rank 8   # LoRA distribution
    uv run ktune --model /path/to/transformer.safetensors --momentum bf16 --wd 0.01

Only the shapes matter for optimizer timing, so random weights are fine and no real
checkpoint values are read. Match the flags (``--betas``, ``--momentum``, ``--wd``,
``--cautious``) to your training config for a representative measurement.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import struct
import sys
from collections.abc import Sequence

import torch

from koptim import Adafusion
from koptim.adafusion import _FOREACH_BATCH_CUTOFF


def _read_safetensors_shapes(path: str, prefix: str | None) -> list[tuple[int, ...]]:
    """Read tensor shapes from a safetensors header (no tensor data is loaded)."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    shapes = []
    for name, meta in header.items():
        if prefix and not name.startswith(prefix):
            continue
        shape = tuple(meta.get("shape", ()))
        if shape:
            shapes.append(shape)
    return shapes


def _lora_shapes(shapes: Sequence[tuple[int, ...]], rank: int) -> list[tuple[int, ...]]:
    """Synthesize the trainable shapes of a LoRA adapter over the 2-D/conv weights."""
    out = []
    for s in shapes:
        if len(s) < 2:
            continue
        fan_out = s[0]
        fan_in = math.prod(s[1:])
        out.append((rank, fan_in))   # lora_A
        out.append((fan_out, rank))  # lora_B
    return out


def _build_params(shapes: Sequence[tuple[int, ...]], device: torch.device) -> list[torch.nn.Parameter]:
    params = []
    for s in shapes:
        p = torch.nn.Parameter(torch.randn(*s, device=device, dtype=torch.bfloat16) * 0.02)
        p.grad = torch.randn_like(p) * 0.01
        params.append(p)
    return params


def _opt_kwargs(args: argparse.Namespace) -> dict:
    b1, b2 = (float(x) for x in args.betas.split(","))
    kw = dict(
        lr=1e-4,
        betas=(b1, b2),
        bf16_method="stochastic_rounding",
        weight_decay=args.wd,
        cautious=args.cautious,
    )
    if args.momentum != "off":
        kw["momentum_dtype"] = args.momentum
        if b1 == 0.0:  # momentum requested but beta1=0 disables it -> use a sane beta1
            kw["betas"] = (0.9, b2)
    return kw


def _time_step(params, opt_kwargs, *, foreach, cutoff=None, iters=40, warmup=12) -> float | None:
    gc.collect()
    if params[0].is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    kw = dict(opt_kwargs, foreach=foreach)
    if foreach and cutoff is not None:
        kw["foreach_batch_cutoff"] = cutoff
    opt = Adafusion(params, **kw)
    try:
        for _ in range(warmup):
            opt.step()
        if params[0].is_cuda:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            ts = []
            for _ in range(iters):
                start.record()
                opt.step()
                end.record()
                torch.cuda.synchronize()
                ts.append(start.elapsed_time(end))
        else:  # CPU fallback (not representative of GPU kernel behaviour)
            import time
            ts = []
            for _ in range(iters):
                t0 = time.perf_counter()
                opt.step()
                ts.append((time.perf_counter() - t0) * 1e3)
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        del opt
        gc.collect()
    ts.sort()
    return ts[len(ts) // 2]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ktune",
        description="Check whether Adafusion's foreach cutoff needs tuning on your GPU/model.",
    )
    ap.add_argument("--model", required=True, help="path to a .safetensors checkpoint (header is read for shapes)")
    ap.add_argument("--gpu", type=int, default=0, help="CUDA device index (default 0)")
    ap.add_argument("--filter", default=None, help="only keys with this prefix (e.g. model.diffusion_model.)")
    ap.add_argument("--lora-rank", type=int, default=None, help="benchmark a LoRA adapter of this rank instead of full weights")
    ap.add_argument("--betas", default="0.0,0.999", help="comma betas, e.g. 0.9,0.999 (default 0.0,0.999)")
    ap.add_argument("--momentum", choices=("off", "bfloat16", "float32"), default="off", help="momentum dtype")
    ap.add_argument("--wd", type=float, default=0.0, help="weight decay")
    ap.add_argument("--cautious", action="store_true", help="cautious masking")
    ap.add_argument("--cutoffs", default="500000,1000000,2000000,4000000,8000000",
                    help="comma list of cutoffs to sweep (elements)")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device — CPU timings are NOT representative of the GPU crossover.", file=sys.stderr)
        device = torch.device("cpu")
    else:
        if args.gpu >= torch.cuda.device_count():
            ap.error(f"--gpu {args.gpu} but only {torch.cuda.device_count()} CUDA device(s) visible")
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)

    shapes = _read_safetensors_shapes(args.model, args.filter)
    if not shapes:
        ap.error(f"no tensors found in {args.model}" + (f" with prefix {args.filter!r}" if args.filter else ""))
    kind = "full fine-tune"
    if args.lora_rank is not None:
        shapes = _lora_shapes(shapes, args.lora_rank)
        kind = f"LoRA rank {args.lora_rank}"

    n_params = sum(math.prod(s) for s in shapes)
    bytes_needed = n_params * 2 * 2  # bf16 weights + bf16 grads
    if device.type == "cuda":
        free = torch.cuda.mem_get_info(device)[0]
        name = torch.cuda.get_device_name(device)
        if bytes_needed > free * 0.9:
            print(f"WARNING: needs ~{bytes_needed/1e9:.1f} GB (weights+grads) but only "
                  f"{free/1e9:.1f} GB free on {name}. Try --lora-rank, or free the GPU.", file=sys.stderr)
    else:
        name = "cpu"

    print("=== ktune: Adafusion foreach cutoff check ===")
    print(f"model   : {args.model}")
    print(f"shapes  : {len(shapes)} tensors, {n_params/1e9:.2f} B params  [{kind}]")
    print(f"device  : {device} ({name})")
    print(f"config  : betas={args.betas} momentum={args.momentum} wd={args.wd} "
          f"cautious={args.cautious} bf16 stochastic-rounding")
    print()

    params = _build_params(shapes, device)
    opt_kwargs = _opt_kwargs(args)

    loop_ms = _time_step(params, opt_kwargs, foreach=False)
    default_ms = _time_step(params, opt_kwargs, foreach=True, cutoff=_FOREACH_BATCH_CUTOFF)
    print(f"per-param loop            : {loop_ms:8.1f} ms/step")
    if default_ms:
        speed = loop_ms / default_ms
        print(f"foreach @ default ({_FOREACH_BATCH_CUTOFF//1000}k)  : {default_ms:8.1f} ms/step  ({speed:.2f}x vs loop)")
    print()

    cutoffs = [int(x) for x in args.cutoffs.split(",")]
    print("cutoff sweep:")
    results: list[tuple[int, float | None]] = []
    for c in cutoffs:
        ms = _time_step(params, opt_kwargs, foreach=True, cutoff=c)
        results.append((c, ms))
        tag = "OOM" if ms is None else f"{ms:8.1f} ms"
        star = "  <- default" if c == _FOREACH_BATCH_CUTOFF else ""
        print(f"  cutoff {c//1000:>6}k : {tag}{star}")

    ok = [(c, ms) for c, ms in results if ms is not None]
    if not ok:
        print("\nAll cutoffs OOM'd — lower the model size (try --lora-rank) or free VRAM.")
        return 1
    best_c, best_ms = min(ok, key=lambda r: r[1])
    print()
    if default_ms is not None and best_ms >= default_ms * 0.95:
        print(f"==> Keep the default: foreach_batch_cutoff={_FOREACH_BATCH_CUTOFF} "
              f"(within ~5% of the best, {best_c//1000}k).")
    else:
        print(f"==> Consider foreach_batch_cutoff={best_c} on this GPU "
              f"({default_ms:.0f} ms -> {best_ms:.0f} ms, "
              f"{(1 - best_ms/default_ms)*100:.0f}% faster). The stack budget cap follows at 4x.")
    if args.lora_rank is not None:
        print("    (LoRA tensors are tiny, so the cutoff rarely binds — defaults are almost always fine.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
