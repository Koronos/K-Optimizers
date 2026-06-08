"""PoC #2: can we ride the AdamW fused kernel WITH kaon's quantization (on-the-fly conversion)?

The fused kernel (`torch._fused_adamw_`) is templated on a floating `scalar_type` and requires
param/grad/exp_avg/exp_avg_sq to share dtype+shape+layout (verified: it rejects int8 and factored
shapes). So to use it with kaon's int8 momentum we must DEQUANT the momentum to the param dtype
before the call and REQUANT after — and the factored second moment cannot be used at all (the
kernel does a *dense* v update; reconstruct-then-refactor is lossy + costly), so v must go dense.

This benchmarks (bf16 params — kaon's real setting), OPTIMIZER STEP ONLY:
  - torch.AdamW(fused)        : bf16 dense state (4 B/param)        -- the speed/quality baseline
  - FusedAdamW (lean)         : same, our leaner wrapper
  - FusedAdamW-int8mom        : int8 momentum (kaon codec) dequant->fused->requant + dense bf16 v
                                (3 B/param) -- the "ride the fused kernel WITH our quantization" idea
  - Adakaon-bf16 (native)     : factored v + codec momentum + cautious + GC (2 B/param)

The question: does the on-the-fly dequant/requant keep us near fused speed, or does the conversion
overhead (extra launches) eat the win?

VERDICT (measured, RTX 4080, bf16 params, batched requant): the conversion route does NOT work.
  LoRA 4096 adapters : AdamW-fused 9 ms | int8mom-onthefly 138 ms (15x!) | Adakaon native 45 ms
  dense 67M params   : AdamW-fused 4 ms | int8mom-onthefly 11 ms (2.8x)  | Adakaon native 10 ms
- LoRA (launch-bound): the dequant+requant reintroduce and AMPLIFY exactly the launch overhead we
  wanted to remove -> 15x slower than fused, 3x slower than even kaon's own native path. Dead end.
- dense (few large tensors): on par with native Adakaon (no win) and heavier (3 vs 2 B/param), with
  vanilla-AdamW math (no factoring / cautious / GC).
Plus the factored second moment can't ride the kernel at all (dense v required). So: the AdamW fused
kernel CANNOT be leveraged with kaon's quantization. The only fused-compatible memory reduction is
bf16 DENSE state (4 B/param, no quant/factor -> that's just bf16 AdamW, natively supported). True
fused-speed-at-kaon-memory needs a CUSTOM (Triton) kernel doing quant+factored+cautious+GC in one
launch -- the conversion approach is proven not to.

    python benchmarks/control/fused_quant_poc.py
"""
from __future__ import annotations

import importlib.util
import time
from collections import defaultdict
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
from kaon import Adakaon  # noqa: E402
from kaon._momentum_codec import _quant_int8_stacked  # noqa: E402

DEV = "cuda"


def _buckets(group, state, keys):
    out: dict[tuple[Any, ...], list] = {}
    for p in group["params"]:
        if p.grad is None:
            continue
        st = state(p)
        lists = out.setdefault((p.device, p.dtype), tuple([] for _ in range(len(keys) + 2)))
        lists[0].append(p); lists[1].append(p.grad)
        for i, k in enumerate(keys):
            lists[i + 2].append(st[k])
    return out


class FusedAdamW(Optimizer):
    """AdamW via the fused kernel; dense state at the param dtype (bf16 params -> 4 B/param)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):  # noqa: ANN001
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "step" not in st:
                    st["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    st["m"] = torch.zeros_like(p); st["v"] = torch.zeros_like(p)
            for (ps, gs, ms, vs, ss) in _buckets(group, lambda p: self.state[p], ["m", "v", "step"]).values():  # noqa: B023
                torch._foreach_add_(ss, 1)
                torch._fused_adamw_(ps, gs, ms, vs, [], ss, lr=group["lr"], beta1=b1, beta2=b2,
                                    weight_decay=group["weight_decay"], eps=group["eps"],
                                    amsgrad=False, maximize=False, grad_scale=None, found_inf=None)
        return None


class FusedAdamWInt8Mom(Optimizer):
    """Ride the fused kernel WITH int8 momentum: dequant m -> fused -> requant m; dense bf16 v.

    int8 momentum (1 B/param, per-tensor absmax) + dense bf16 v (2 B/param) = 3 B/param. The v
    MUST be dense (the kernel can't take a factored v). Vanilla AdamW math (no cautious/GC)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):  # noqa: ANN001
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "step" not in st:
                    st["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    st["m_codes"] = torch.zeros_like(p, dtype=torch.int8)
                    st["m_scale"] = torch.ones((), dtype=torch.float32, device=p.device)
                    st["v"] = torch.zeros_like(p)  # bf16 dense
            for (device, dtype), parts in _buckets(
                group, lambda p: self.state[p], ["m_codes", "m_scale", "v", "step"]  # noqa: B023
            ).items():
                ps, gs, codes, scales, vs, ss = parts
                # --- dequant int8 momentum -> dense (param dtype), batched ---
                ms = [(c.to(torch.float32) * s).to(dtype) for c, s in zip(codes, scales, strict=True)]
                torch._foreach_add_(ss, 1)
                torch._fused_adamw_(ps, gs, ms, vs, [], ss, lr=group["lr"], beta1=b1, beta2=b2,
                                    weight_decay=group["weight_decay"], eps=group["eps"],
                                    amsgrad=False, maximize=False, grad_scale=None, found_inf=None)
                # --- requant the updated dense momentum -> int8, BATCHED per shape (stacked absmax) ---
                byshape: dict[Any, list] = defaultdict(lambda: ([], [], []))
                for m, c, s in zip(ms, codes, scales, strict=True):
                    g3 = byshape[tuple(m.shape)]
                    g3[0].append(m); g3[1].append(c); g3[2].append(s)
                for mlist, clist, slist in byshape.values():
                    stk = torch.stack([m.float().reshape(1, -1) for m in mlist])  # [N,1,numel]
                    q, sc = _quant_int8_stacked(stk)                              # [N,1,numel],[N,1,1]
                    torch._foreach_copy_([c.reshape(1, -1) for c in clist], list(q.unbind(0)))
                    for sdst, scv in zip(slist, sc.unbind(0), strict=True):
                        sdst.copy_(scv.reshape(()))
        return None


def bf16_lora_bag(n, shape):
    g = torch.Generator().manual_seed(0)
    ps = []
    for _ in range(n):
        p = torch.randn(*shape, generator=g).to(DEV).to(torch.bfloat16).requires_grad_(True)
        p.grad = torch.randn(*shape, generator=g).to(DEV).to(torch.bfloat16)
        ps.append(p)
    return ps


def bf16_dense(shapes):
    g = torch.Generator().manual_seed(0)
    ps = []
    for s in shapes:
        p = torch.randn(*s, generator=g).to(DEV).to(torch.bfloat16).requires_grad_(True)
        p.grad = torch.randn(*s, generator=g).to(DEV).to(torch.bfloat16)
        ps.append(p)
    return ps


def fresh(params):
    out = []
    for p in params:
        q = p.detach().clone().requires_grad_(True); q.grad = p.grad.clone(); out.append(q)
    return out


def time_step(make, params, reps=80, warmup=20):
    opt = make(params)
    for _ in range(warmup):
        opt.step()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.time(); opt.step()
        torch.cuda.synchronize(); ts.append((time.time() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2], H.opt_state_bytes_per_param(opt, params)


OPTS = {
    "torch.AdamW(fused)": lambda p: torch.optim.AdamW(p, lr=1e-3, fused=True),
    "FusedAdamW (lean)": lambda p: FusedAdamW(p, lr=1e-3),
    "FusedAdamW-int8mom": lambda p: FusedAdamWInt8Mom(p, lr=1e-3),
    "Adakaon-bf16 (native)": lambda p: Adakaon(p, lr=1e-3, betas=(0.9, 0.999), cautious=True, momentum_dtype="bfloat16"),
}


def run(title, params):
    nparam = sum(p.numel() for p in params)
    print(f"\n== {title} ==  ({len(params)} tensors, {nparam/1e6:.2f}M params, bf16)")
    print(f"  {'optimizer':24s} {'ms/step':>9} {'B/param':>9}")
    base = None
    for name, make in OPTS.items():
        try:
            ms, bpp = time_step(make, fresh(params))
            base = base or ms
            print(f"  {name:24s} {ms:9.3f} {bpp:9.2f}   ({ms/base:.1f}x fused)")
        except Exception as e:  # noqa: BLE001
            print(f"  {name:24s} FAILED: {type(e).__name__}: {e}")


def main():
    torch.manual_seed(0)
    run("LoRA bag 4096 tiny tensors", bf16_lora_bag(4096, (16, 32)))
    run("dense ~67M params (8 x [2048,4096])", bf16_dense([(2048, 4096)] * 8))
    print("\n(ms = optimizer.step() ONLY, median. int8mom = int8 momentum + dense bf16 v, on-the-fly.)")


if __name__ == "__main__":
    main()
