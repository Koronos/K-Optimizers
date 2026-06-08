"""PoC: use AdamW's FUSED CUDA kernel from outside torch.optim.AdamW, inside a kaon optimizer.

`torch._fused_adamw_` (the ATen op behind `torch.optim.AdamW(fused=True)`) is callable directly
on our own tensor lists — one CUDA launch does decoupled-WD + both EMAs + bias-corrected update.
This builds a kaon-native `FusedAdamW` around it and benchmarks the OPTIMIZER STEP ONLY (no
fwd/bwd) at growing scale, to quantify what the fused kernel buys vs kaon's factored/quantized
optimizers — and what it costs in memory.

Hard constraint (the catch): the fused kernel needs DENSE exp_avg + exp_avg_sq; it cannot consume
kaon's factored/quantized state. So this is "AdamW speed at AdamW-class memory", not a drop-in
accelerator for the memory-efficient optimizers.

Findings (measured, RTX 4080, torch 2.10):
- kaon.FusedAdamW (this PoC, calling torch._fused_adamw_ directly) MATCHES / slightly BEATS
  torch.optim.AdamW(fused=True) — a leaner per-step Python path. So the fused kernel IS usable
  externally inside a kaon optimizer.
- The optimizer step is NOT negligible at scale: on a 4096-adapter "big LoRA", kaon's factored
  optimizers take 25-36 ms/step vs ~9 ms fused — 16-27 ms of pure overhead per step, i.e. minutes
  over a long run. Most of that gap is launch/Python overhead (bucketing + many small kernels +
  codec dequant/requant), not fundamental compute.
- Memory: fp32 params -> 8 B/param fused state. bf16/fp16 params REQUIRE state of the SAME dtype
  (fp32 state is rejected) -> 4 B/param fused state (half of fp32 AdamW; still > kaon's ~2-2.75
  factored, and bf16 second-moment is lower-precision than kaon's factored v).

    python benchmarks/control/fused_poc.py
"""
from __future__ import annotations

import importlib.util
import time
from typing import Any

import torch
from torch.optim import Optimizer

REPO = "/media/koronos/arca/repos/K-Optimizers"


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


H = _load("harness", f"{REPO}/benchmarks/proxy/harness.py")
from kaon import Adakaon, ScheduleFree  # noqa: E402

DEV = "cuda"


class FusedAdamW(Optimizer):
    """AdamW via the fused CUDA kernel (`torch._fused_adamw_`), kaon API. Dense fp32 state."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):  # noqa: ANN001
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            # bucket by (device, dtype) — the fused op wants a uniform TensorList
            buckets: dict[tuple[Any, ...], list] = {}
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "step" not in st:
                    st["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    st["m"] = torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format)
                    st["v"] = torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format)
                key = (p.device, p.dtype)
                buckets.setdefault(key, ([], [], [], [], []))
                ps, gs, ms, vs, ss = buckets[key]
                ps.append(p); gs.append(p.grad); ms.append(st["m"]); vs.append(st["v"]); ss.append(st["step"])
            for (ps, gs, ms, vs, ss) in buckets.values():
                torch._foreach_add_(ss, 1)
                torch._fused_adamw_(
                    ps, gs, ms, vs, [], ss,
                    lr=group["lr"], beta1=b1, beta2=b2, weight_decay=group["weight_decay"],
                    eps=group["eps"], amsgrad=False, maximize=False, grad_scale=None, found_inf=None,
                )
        return loss


# ----------------------------- timing -----------------------------
def bytes_per_param(opt, params):
    return H.opt_state_bytes_per_param(opt, params)


def time_step(make, params, reps=100, warmup=20):
    opt = make(params)
    for _ in range(warmup):
        opt.step()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.time()
        opt.step()
        torch.cuda.synchronize(); ts.append((time.time() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2], bytes_per_param(opt, params)


def lora_bag(n_tensors, shape):
    g = torch.Generator().manual_seed(0)
    ps = []
    for _ in range(n_tensors):
        p = torch.randn(*shape, generator=g).to(DEV).requires_grad_(True)
        p.grad = torch.randn(*shape, generator=g).to(DEV)
        ps.append(p)
    return ps


def dense_params(shapes):
    g = torch.Generator().manual_seed(0)
    ps = []
    for s in shapes:
        p = torch.randn(*s, generator=g).to(DEV).requires_grad_(True)
        p.grad = torch.randn(*s, generator=g).to(DEV)
        ps.append(p)
    return ps


OPTS = {
    "torch.AdamW(fused)": lambda p: torch.optim.AdamW(p, lr=1e-3, fused=True),
    "kaon.FusedAdamW (PoC)": lambda p: FusedAdamW(p, lr=1e-3),
    "Adakaon-bf16": lambda p: Adakaon(p, lr=1e-3, betas=(0.9, 0.999), cautious=True, momentum_dtype="bfloat16"),
    "ScheduleFree": lambda p: ScheduleFree(p, lr=1e-3, momentum_dtype="bfloat16"),
}


def fresh(params):
    """A private copy of the params WITH their grads re-attached (so each optimizer steps a
    real, independent set — cloning otherwise drops .grad and the step becomes a no-op)."""
    out = []
    for p in params:
        q = p.detach().clone().requires_grad_(True)
        q.grad = p.grad.clone()
        out.append(q)
    return out


def run(title, params):
    nparam = sum(p.numel() for p in params)
    print(f"\n== {title} ==  ({len(params)} tensors, {nparam/1e6:.2f}M params)")
    print(f"  {'optimizer':24s} {'ms/step':>9} {'B/param':>9} {'state MB':>9}")
    base = None
    for name, make in OPTS.items():
        try:
            ms, bpp = time_step(make, fresh(params))
            if base is None:
                base = ms
            print(f"  {name:24s} {ms:9.3f} {bpp:9.2f} {bpp*nparam/1e6:9.1f}   ({ms/base:.1f}x AdamW-fused)")
        except Exception as e:  # noqa: BLE001
            print(f"  {name:24s} FAILED: {type(e).__name__}: {e}")


def main():
    torch.manual_seed(0)
    # 1) LoRA-like launch-bound regime, growing tensor count
    run("LoRA bag  256 tiny tensors", lora_bag(256, (8, 16)))
    run("LoRA bag  1024 tiny tensors", lora_bag(1024, (8, 16)))
    run("LoRA bag  4096 tiny tensors (big LoRA)", lora_bag(4096, (16, 32)))
    # 2) dense bandwidth-bound regime, growing model size (few large 2-D weights)
    run("dense  ~4M params  (8 x [512,1024])", dense_params([(512, 1024)] * 8))
    run("dense  ~67M params (8 x [2048,4096])", dense_params([(2048, 4096)] * 8))
    print("\n(ms = optimizer.step() ONLY, median; state = persistent optimizer memory.)")


if __name__ == "__main__":
    main()
