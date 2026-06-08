"""Unit tests for AdaPNM(fused=True) — the Triton-fused positive-negative-momentum step.

The fused path reuses the shared kaon Triton núcleo (gradient_centralize, factored_rc, the int8/4bit
dequant/requant primitives, sr_round) and adds only AdaPNM's two-momentum machinery: the pos/neg
buffers (roles alternate by step parity), the raw-grad EMA on the positive buffer, the pos-neg mix /
noise_norm renorm, decoupled WD applied BEFORE the step, and no RMS-clip.

Correctness criterion: AdaPNM(fused=True) must match AdaPNM(fused=False) — same math, same state.
Skips cleanly when CUDA or Triton is unavailable (the kernels are GPU-only).
"""
from __future__ import annotations

import pytest
import torch

from kaon import AdaPNM
from kaon._fused_triton import HAS_TRITON

pytestmark = pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()),
    reason="AdaPNM fused step requires CUDA + Triton",
)

DEV = "cuda"


def _bag(shapes, dtype=torch.float32, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return [torch.randn(*s, generator=g, device=DEV, dtype=dtype).requires_grad_(True) for s in shapes]


def _parts(opt):
    ob, big, nat = [], [], []
    for (_ids, o, b, n) in opt._fused_part.values():
        ob += o
        big += b
        nat += n
    return ob, big, nat


def _run_parity(shapes, dtype, mdtype, *, cautious=True, gc=True, wd=0.0, steps=6, seed=1):
    """Step AdaPNM(fused=True) and native AdaPNM on identical params+grads; return max|Δp| and scale."""
    cfg = dict(lr=2e-3, betas=(0.8, 0.999), beta0=0.5, eps=1e-30, weight_decay=wd,
               cautious=cautious, gradient_centralization=gc, momentum_dtype=mdtype)
    pv = _bag(shapes, dtype, seed)
    pn = [p.detach().clone().requires_grad_(True) for p in pv]
    ov, on = AdaPNM(pv, fused=True, **cfg), AdaPNM(pn, **cfg)
    gen = torch.Generator(device=DEV).manual_seed(7)
    for _ in range(steps):
        gs = [torch.randn(*p.shape, generator=gen, device=DEV, dtype=dtype) for p in pv]
        for p, g in zip(pv, gs):
            p.grad = g.clone()
        for p, g in zip(pn, gs):
            p.grad = g.clone()
        ov.step()
        on.step()
    torch.cuda.synchronize()
    d = max((a.detach().float() - b.detach().float()).abs().max().item() for a, b in zip(pv, pn))
    scale = max(b.detach().float().abs().max().item() for b in pn)
    return d, scale, ov


# ----------------------------------------------------------------- one-block parity
@pytest.mark.parametrize("cautious", [True, False])
@pytest.mark.parametrize("gc", [True, False])
def test_fp32_parity_exact(cautious, gc):
    # fp32 momentum -> exact vs native (the alternation + pos-neg mix reproduce native bit-closely)
    d, _, _ = _run_parity([(8, 16)] * 4, torch.float32, "float32", cautious=cautious, gc=gc)
    assert d < 1e-5, f"max|Δp|={d:.2e}"


def test_fp32_parity_weight_decay():
    d, _, _ = _run_parity([(8, 16), (16, 8)], torch.float32, "float32", wd=0.05)
    assert d < 1e-5, f"max|Δp|={d:.2e}"


def test_mixed_shapes_bucketing():
    d, _, ov = _run_parity([(8, 16), (16, 8), (12, 20), (16, 320)], torch.float32, "float32")
    assert d < 1e-5, f"max|Δp|={d:.2e}"
    assert len(_parts(ov)[0]) == 4 and len(_parts(ov)[1]) == 0


@pytest.mark.parametrize("mdtype", ["bfloat16", "int8", "4bit"])
def test_quant_momentum_parity(mdtype):
    # libdevice.rint requant -> int8/4bit track native closely; bf16 within ULP
    d, scale, _ = _run_parity([(8, 16)] * 4, torch.float32, mdtype)
    assert d / scale < 5e-3, f"{mdtype} rel={d/scale:.2e}"


def test_bf16_params_sr():
    d, scale, _ = _run_parity([(8, 16), (16, 8)], torch.bfloat16, "bfloat16")
    assert d / scale < 5e-2, f"rel={d/scale:.2e}"  # independent SR -> expectation match


# ----------------------------------------------------------------- chunked (big tensor) parity
def test_chunked_fp32_parity():
    d, _, ov = _run_parity([(1024, 512)], torch.float32, "float32")
    assert d < 1e-5, f"max|Δp|={d:.2e}"
    assert len(_parts(ov)[1]) == 1 and len(_parts(ov)[0]) == 0


@pytest.mark.parametrize("mdtype", ["int8", "4bit"])
def test_chunked_quant_parity(mdtype):
    d, scale, ov = _run_parity([(1024, 512)], torch.float32, mdtype)
    assert d / scale < 5e-4, f"{mdtype} rel={d/scale:.2e}"
    assert len(_parts(ov)[1]) == 1


def test_mixed_one_block_chunked_native():
    # small (one-block) + big (chunked) + 1-D (native), all parity at once
    d, _, ov = _run_parity([(8, 16), (16, 320), (1024, 512), (64,)], torch.float32, "float32")
    assert d < 1e-5, f"max|Δp|={d:.2e}"
    ob, big, nat = _parts(ov)
    assert len(ob) == 2 and len(big) == 1 and len(nat) == 1


# ----------------------------------------------------------------- memory + convergence
def test_two_momenta_memory():
    # AdaPNM carries TWO momenta (m_pos + m_neg); int8 -> ~2 B/param of momentum
    ps = _bag([(32, 48)] * 4, torch.float32)
    for p in ps:
        p.grad = torch.randn_like(p)
    opt = AdaPNM(ps, fused=True, momentum_dtype="int8")
    opt.step()
    torch.cuda.synchronize()
    st = opt.state[ps[0]]
    assert st["m_pos"].dtype == torch.int8 and st["m_neg"].dtype == torch.int8
    assert st["m_pos"].numel() == ps[0].numel() and st["m_neg"].numel() == ps[0].numel()


def test_fused_converges():
    torch.manual_seed(0)
    w = torch.randn(16, 24, device=DEV).requires_grad_(True)
    target = torch.randn(16, 24, device=DEV)
    opt = AdaPNM([w], fused=True, lr=5e-2, momentum_dtype="4bit")
    losses = []
    for _ in range(80):
        opt.zero_grad()
        loss = (w - target).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    torch.cuda.synchronize()
    assert torch.isfinite(w).all()
    assert losses[-1] < losses[0] * 0.3, f"{losses[0]:.3f} -> {losses[-1]:.3f}"


def test_alternation_runs_many_steps():
    # the pos/neg roles alternate by step parity -> run enough steps to exercise both phases repeatedly
    d, _, _ = _run_parity([(8, 16)] * 3, torch.float32, "float32", steps=11)
    assert d < 1e-5, f"max|Δp|={d:.2e}"
