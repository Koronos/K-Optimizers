"""FusedAdakaon — a battery-rankable optimizer wrapping the fused Triton Adakaon kernel (PoC).

Routes eligible >=2-D weights (matrixized to [R,C], bucketed by shape, stacked) through the fused
Triton kernel from triton_full_poc (GC + factored v + momentum + cautious + bf16 SR, in 2 launches),
and 1-D params (biases / norm scales) through a small torch AdamW fallback.

⚠️ BATTERY VERDICT (the honest result -- this is why we tested it): FAITHFUL but NOT FASTER.
  vs Adakaon-bf16 @ C128/N2000:  te 0.0722 vs 0.0741, gap +0.0203 vs +0.0213, MEMORY 2.04 vs 2.03
  B/param (SAME), but ms/step 15.4 vs 14.4 and LoRA 4.34 vs 3.68 ms -- 0.8-1.0x across all tensor
  sizes. The PoC's 20-60x was measured on PRE-STACKED data; a real optimizer must STACK the separate
  params (grad/p/m/row/col) and SCATTER the results back every step. Native Adakaon writes p in-place
  (no p-copy) and already stacks/foreaches internally, so the extra stack+scatter of p AND m (~2x the
  weight bandwidth) eats the kernel's launch savings. The fused kernel only wins on data that is
  ALREADY stacked (a single [N,R,C] LoRA-adapter parameter), OR via a POINTER-ARRAY multi-tensor
  kernel (torch's MultiTensorApply pattern) that runs in-place on the separate tensors with no
  stacking -- the genuinely hard version, and the real path to a faster Adakaon(fused=True).
"""
from __future__ import annotations

import importlib.util

import torch
from torch.optim import Optimizer

_REPO = "/media/koronos/arca/repos/K-Optimizers"


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


# the fused kernel + the stacked step function live in triton_full_poc
_FP = _load("triton_full_poc", f"{_REPO}/benchmarks/control/triton_full_poc.py")
fused_adakaon_step = _FP.fused_adakaon_step


class FusedAdakaon(Optimizer):
    """Adakaon-bf16 via the fused Triton kernel for >=2-D weights; torch AdamW for 1-D."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps1=1e-30, clip=1.0,
                 cautious=True, gradient_centralization=True, weight_decay=0.0):
        beta1, beta2 = betas
        defaults = dict(lr=lr, beta1=float(beta1), beta2=float(beta2), eps1=eps1, clip=clip,
                        cautious=cautious, gc=gradient_centralization, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._t = 0

    @torch.no_grad()
    def step(self, closure=None):  # noqa: ANN001
        loss = closure() if closure is not None else None
        self._t += 1
        for group in self.param_groups:
            elig: dict[tuple[int, int], list] = {}     # (R,C) -> params (matrixized 2-D)
            one_d = []
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim >= 2 and p.is_cuda:
                    R = p.shape[0]                      # noqa: N806
                    C = p.numel() // R                  # noqa: N806
                    elig.setdefault((R, C), []).append(p)
                else:
                    one_d.append(p)
            for (R, C), plist in elig.items():          # noqa: N806
                self._fused_bucket(plist, R, C, group)
            for p in one_d:
                self._adam_1d(p, group)
        return loss

    @torch.no_grad()
    def _fused_bucket(self, plist, R, C, group):  # noqa: N803
        states = [self.state[p] for p in plist]
        for p, st in zip(plist, states, strict=True):
            if "m" not in st:
                # momentum ALWAYS bf16 (2 B/param) regardless of param dtype -- matches
                # Adakaon-bf16's footprint; the kernel reads/writes m at bf16, p at its own dtype.
                st["m"] = torch.zeros(R, C, dtype=torch.bfloat16, device=p.device)
                st["row"] = torch.zeros(R, dtype=torch.float32, device=p.device)
                st["col"] = torch.zeros(C, dtype=torch.float32, device=p.device)
        # stack the bucket: grad/p/m -> [N,R,C], row/col -> [N,R]/[N,C]
        grad = torch.stack([p.grad.reshape(R, C) for p in plist])
        p_st = torch.stack([p.data.reshape(R, C) for p in plist])
        m_st = torch.stack([st["m"] for st in states])
        row = torch.stack([st["row"] for st in states])
        col = torch.stack([st["col"] for st in states])
        fused_adakaon_step(
            p_st, grad, row, col, m_st,
            lr=group["lr"], beta1=group["beta1"], beta2=group["beta2"], eps1=group["eps1"],
            clip=group["clip"], gc=group["gc"], cautious=group["cautious"], seed=self._t,
        )
        # scatter the updated state/weights back to the per-param tensors
        torch._foreach_copy_([st["m"] for st in states], list(m_st.unbind(0)))
        torch._foreach_copy_([st["row"] for st in states], list(row.unbind(0)))
        torch._foreach_copy_([st["col"] for st in states], list(col.unbind(0)))
        torch._foreach_copy_([p.data.reshape(R, C) for p in plist], list(p_st.unbind(0)))

    @torch.no_grad()
    def _adam_1d(self, p, group):
        """Plain AdamW for 1-D params (biases / norm scales) -- few params, not the hot path."""
        st = self.state[p]
        if "exp_avg" not in st:
            st["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
            st["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
            st["t1d"] = 0
        st["t1d"] += 1
        t = st["t1d"]
        g = p.grad.float()
        m, v = st["exp_avg"], st["exp_avg_sq"]
        b1, b2 = group["beta1"], group["beta2"]
        m.mul_(b1).add_(g, alpha=1 - b1)
        v.mul_(b2).addcmul_(g, g, value=1 - b2)
        denom = (v / (1 - b2**t)).sqrt_().add_(1e-8)
        step = group["lr"] / (1 - b1**t)
        upd = (m / denom).mul_(step)
        p.data.add_((-upd).to(p.dtype))   # 1-D params are few -> plain cast (no SR needed)


__all__ = ["FusedAdakaon"]
