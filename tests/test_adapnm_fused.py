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


def _clone(ps):
    return [p.detach().clone().requires_grad_(True) for p in ps]


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


# ----------------------------------------------------------------- RMS-clip (divergence guard)
def _clip_trigger_step_rms(fused, clip, shape=(4, 4), warm=60):
    """rms(|Δp|) on a trigger step: a hot cell in a column kept cold during warmup. The cold col's
    EMA is tiny so c_factor=rsqrt(col) is huge -> the (v_hat-normalized) update RMS exceeds 1, which
    the clip must cap. Deterministic: fp32 momentum, cautious/GC off."""
    import math

    torch.manual_seed(0)
    rr, cc = shape
    p = (0.01 * torch.randn(rr, cc, device=DEV)).requires_grad_(True)
    opt = AdaPNM([p], fused=fused, clip_threshold=clip, lr=1e-2, betas=(0.8, 0.999), beta0=0.5,
                 eps=1e-8, cautious=False, gradient_centralization=False,
                 momentum_dtype="float32", bf16_method="none")
    g = torch.Generator(device=DEV).manual_seed(0)
    for _ in range(warm):
        gr = 0.05 * torch.randn(rr, cc, generator=g, device=DEV)
        gr[:, 0] = 0.0
        p.grad = gr
        opt.step()
    before = p.detach().clone()
    gr = 0.05 * torch.randn(rr, cc, generator=g, device=DEV)
    gr[:, 0] = 0.0
    gr[0, 0] = 30.0
    p.grad = gr
    opt.step()
    torch.cuda.synchronize()
    dp = p.detach() - before
    return float(dp.norm() / math.sqrt(dp.numel()))


@pytest.mark.parametrize("fused", [False, True])
def test_clip_bounds_factored_update(fused):
    # The guard that stopped the real Cosmos LoKr NaN: clip=1 caps rms(step) ~<= clip*step_size,
    # while clip=0 (unclamped PNM) leaves it several x larger. Same bound on native and fused.
    step_size = 1e-2 / (1.0 - 0.8 ** 61)
    r_unclipped = _clip_trigger_step_rms(fused, 0.0)
    r_clipped = _clip_trigger_step_rms(fused, 1.0)
    assert r_clipped <= 1.3 * step_size, f"clip should bound step rms ~<= step_size, got {r_clipped:.2e}"
    assert r_unclipped > 2.0 * r_clipped, f"clip=0 should be much larger: {r_unclipped:.2e} vs {r_clipped:.2e}"


def test_clip_default_on_and_disable():
    # default clip_threshold is 1.0 (the stability guard, matching Adakaon); 0 disables it
    p = _bag([(8, 8)], torch.float32)[0]
    assert AdaPNM([p]).param_groups[0]["clip_threshold"] == 1.0
    assert AdaPNM([p], clip_threshold=0.0).param_groups[0]["clip_threshold"] == 0.0


def test_clip_chunked_parity_with_native():
    # the big-tensor (chunked) path also clips: fused(clip=1) must still match native(clip=1).
    # A LONE big tensor keeps the per-tensor fused-chunked kernel (no batch to amortize).
    d, _, ov = _run_parity([(1024, 512)], torch.float32, "float32")
    assert d < 1e-5, f"max|Δp|={d:.2e}"
    assert len(_parts(ov)[1]) == 1  # confirms it took the (single-big) chunked path


def test_big_batched_routes_and_parity():
    # >=2 same-shape big (>cap) factors take the batched chunked kernel (~2 launches for the bucket);
    # result must still match a pure-native AdaPNM exactly (fp32).
    d, _, ov = _run_parity([(1024, 512)] * 3, torch.float32, "float32")
    assert d < 1e-5, f"max|Δp|={d:.2e}"
    ob, big, nat = _parts(ov)
    assert len(big) == 3 and len(ob) == 0  # all classified big; dispatched batched-chunked


@pytest.mark.parametrize("cautious", [True, False])
@pytest.mark.parametrize("gc", [True, False])
def test_big_batched_features(cautious, gc):
    d, _, _ = _run_parity([(512, 512)] * 3, torch.float32, "float32", cautious=cautious, gc=gc, wd=0.05)
    assert d < 1e-5, f"cautious={cautious} gc={gc} max|Δp|={d:.2e}"


@pytest.mark.parametrize("mdtype", ["bfloat16", "int8", "4bit"])
def test_big_batched_quant_parity(mdtype):
    d, scale, _ = _run_parity([(512, 512)] * 3, torch.float32, mdtype, wd=0.05)
    bound = 5e-3 if mdtype == "bfloat16" else 5e-4
    assert d / scale < bound, f"{mdtype} rel={d/scale:.2e}"


def test_big_batched_bf16_params_sr():
    d, scale, _ = _run_parity([(512, 512)] * 3, torch.bfloat16, "bfloat16")
    assert d / scale < 5e-2, f"rel={d/scale:.2e}"


def test_big_batched_alternation_many_steps():
    # cross the pos/neg parity boundary several times in the batched path (both swap orderings).
    d, _, _ = _run_parity([(512, 512)] * 3, torch.float32, "float32", steps=11)
    assert d < 1e-5, f"max|Δp|={d:.2e}"


def test_big_batched_matches_native_foreach_toggle():
    # batched chunked path (default) must equal the in-fused native-foreach fallback (toggle off).
    # fp32 momentum so the two paths match near-exactly (bf16 momentum rounds differently per path).
    cfg = dict(lr=2e-3, weight_decay=0.05, cautious=True, gradient_centralization=True,
               momentum_dtype="float32")
    pv = _bag([(512, 512)] * 3, torch.float32, seed=2)
    pn = _clone(pv)
    ov = AdaPNM(pv, fused=True, **cfg)
    on = AdaPNM(pn, fused=True, **cfg)
    on._fused_big_batched = False
    gen = torch.Generator(device=DEV).manual_seed(9)
    for _ in range(6):
        gs = [torch.randn(*p.shape, generator=gen, device=DEV) for p in pv]
        for p, g in zip(pv, gs):
            p.grad = g.clone()
        for p, g in zip(pn, gs):
            p.grad = g.clone()
        ov.step(); on.step()
    torch.cuda.synchronize()
    d = max((a - b).abs().max().item() for a, b in zip(pv, pn))
    assert d < 1e-5, f"batched vs native-foreach max|Δp|={d:.2e}"
