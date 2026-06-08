"""Unit tests for the experimental Triton fused kernel (``kaon._fused_triton``).

Covers the CURRENT functionality (bf16/fp32 momentum) before int8/4bit is added:
  * fp32 parity with native Adakaon (the kernel is exact there),
  * bf16 momentum / bf16-param (SR) fidelity bounds,
  * tile bucketing for mixed shapes (the regression that made a naive global tile 9x slower),
  * native-fallback routing (1-D / conv / high-rank) with end-to-end parity,
  * the reusable host predicate ``fused_eligible`` and device primitive ``sr_round``,
  * pointer-cache correctness across grad reallocation, memory footprint, grad=None.

Skips cleanly when CUDA or Triton is unavailable (the kernel is GPU-only).
"""
from __future__ import annotations

import pytest
import torch

from kaon import Adakaon
from kaon._fused_triton import HAS_TRITON, FusedAdakaon, fused_eligible, next_pow2_tile, warps_for

pytestmark = pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()),
    reason="Triton fused kernel requires CUDA + Triton",
)

DEV = "cuda"


# ----------------------------------------------------------------- helpers
def _bag(shapes, dtype=torch.float32, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return [torch.randn(*s, generator=g, device=DEV, dtype=dtype).requires_grad_(True) for s in shapes]


def _clone(ps):
    return [p.detach().clone().requires_grad_(True) for p in ps]


def _run_parity(shapes, dtype, mdtype, *, cautious=True, gc=True, steps=6, seed=1):
    """Step FusedAdakaon and native Adakaon on identical params+grads; return max|Δp| and scale."""
    cfg = dict(lr=2e-3, betas=(0.9, 0.999), cautious=cautious,
               gradient_centralization=gc, momentum_dtype=mdtype)
    pv = _bag(shapes, dtype, seed)
    pn = _clone(pv)
    ov = FusedAdakaon(pv, **cfg)
    on = Adakaon(pn, **cfg)
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


# ----------------------------------------------------------------- host helpers
def test_eligibility_predicate():
    big = (1 << 16)  # tile cap
    assert fused_eligible(torch.zeros(8, 16, device=DEV))            # tiny 2-D
    assert fused_eligible(torch.zeros(16, 320, device=DEV))          # low-rank LoRA
    assert fused_eligible(torch.zeros(16, 320, device=DEV, dtype=torch.bfloat16))
    assert not fused_eligible(torch.zeros(128, device=DEV))          # 1-D
    assert not fused_eligible(torch.zeros(8, 8, 3, 3, device=DEV))   # conv ndim>2
    assert not fused_eligible(torch.zeros(8, 16))                    # cpu
    assert not fused_eligible(torch.zeros(8, 16, device=DEV, dtype=torch.float16))  # fp16
    assert not fused_eligible(torch.zeros(256, 1024, device=DEV))    # tile > cap
    # non-contiguous
    t = torch.zeros(16, 32, device=DEV).t()
    assert not fused_eligible(t)
    assert big == (1 << 16)


def test_warps_and_tile_helpers():
    assert next_pow2_tile(8, 16) == (8, 16)
    assert next_pow2_tile(17, 33) == (32, 64)
    assert warps_for(100) == 1
    assert warps_for(40000) == 16
    assert warps_for(8192) == 4


# ----------------------------------------------------------------- fp32 parity (exact)
@pytest.mark.parametrize("cautious", [True, False])
@pytest.mark.parametrize("gc", [True, False])
def test_fp32_parity_exact(cautious, gc):
    d, scale, _ = _run_parity([(8, 16)] * 6, torch.float32, "float32", cautious=cautious, gc=gc)
    assert d < 1e-5, f"max|Δp|={d:.2e}"


def test_fp32_parity_mixed_shapes_multiple_buckets():
    shapes = [(8, 16), (16, 8), (12, 20), (32, 24), (16, 64)]
    d, scale, ov = _run_parity(shapes, torch.float32, "float32")
    assert d < 1e-5, f"max|Δp|={d:.2e}"
    # mixed shapes must bucket into >1 tile (the fix for the over-padding regression)
    assert len(ov._cache.buckets) >= 4


def test_fp32_parity_lora_shapes():
    d, _, _ = _run_parity([(16, 320), (320, 16), (8, 1280)], torch.float32, "float32")
    assert d < 1e-5, f"max|Δp|={d:.2e}"


# ----------------------------------------------------------------- bf16 momentum / params
def test_bf16_momentum_parity_bounded():
    # FusedAdakaon runs the EMA in fp32 then rounds to bf16; native bf16 lerps in bf16. Equivalent
    # (measured null) but not bit-identical -> bound the divergence, don't demand exactness.
    d, scale, _ = _run_parity([(8, 16)] * 6, torch.float32, "bfloat16")
    assert d / scale < 5e-3, f"rel={d/scale:.2e}"


def test_bf16_params_sr_finite_and_close():
    d, scale, _ = _run_parity([(8, 16), (16, 8), (12, 20)], torch.bfloat16, "bfloat16")
    # independent SR draws -> only matches in expectation; bound the per-step divergence
    assert d / scale < 5e-2, f"rel={d/scale:.2e}"


# ----------------------------------------------------------------- int8 momentum (in-kernel)
def test_int8_parity_with_native():
    # in-kernel per-row dequant/requant; libdevice.rint matches torch.round (half-to-even), so the
    # quantized trajectory tracks native Adakaon(int8) tightly (not just in expectation).
    d, scale, _ = _run_parity([(8, 16)] * 6, torch.float32, "int8")
    assert d / scale < 5e-4, f"rel={d/scale:.2e}"


def test_int8_parity_mixed_shapes():
    shapes = [(8, 16), (16, 8), (12, 20), (16, 320)]
    d, scale, ov = _run_parity(shapes, torch.float32, "int8")
    assert d / scale < 5e-4, f"rel={d/scale:.2e}"
    assert len(ov._cache.buckets) >= 3                       # mixed tiles still bucket


def test_int8_state_layout_and_memory():
    ps = _bag([(32, 48)] * 6, torch.float32)
    for p in ps:
        p.grad = torch.randn_like(p)
    ov = FusedAdakaon(ps, momentum_dtype="int8")
    on = Adakaon([p.detach().clone().requires_grad_(True) for p in ps], momentum_dtype="int8")
    for p in on.param_groups[0]["params"]:
        p.grad = torch.randn_like(p)
    ov.step()
    on.step()
    torch.cuda.synchronize()
    st = ov.state[ps[0]]
    assert st["m"].dtype == torch.int8 and st["m"].element_size() == 1      # 1 byte/param
    assert st["m"].numel() == ps[0].numel()
    assert st["m_scale"].shape == (32,)                                     # per-row scale

    def bpp(opt, params):
        b = sum(v.numel() * v.element_size() for p in params for v in opt.state[p].values()
                if torch.is_tensor(v))
        return b / sum(p.numel() for p in params)

    fused_bpp = bpp(ov, ps)
    native_bpp = bpp(on, on.param_groups[0]["params"])
    assert abs(fused_bpp - native_bpp) < 1e-6, f"{fused_bpp} vs {native_bpp}"
    assert fused_bpp < 1.5                                                  # ~1 B/param + factored


def test_int8_converges():
    torch.manual_seed(0)
    w = torch.randn(16, 24, device=DEV).requires_grad_(True)
    target = torch.randn(16, 24, device=DEV)
    opt = FusedAdakaon([w], lr=5e-2, momentum_dtype="int8")
    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss = (w - target).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    torch.cuda.synchronize()
    assert torch.isfinite(w).all()
    assert losses[-1] < losses[0] * 0.2, f"{losses[0]:.3f} -> {losses[-1]:.3f}"


def test_int8_with_bf16_params():
    d, scale, _ = _run_parity([(16, 32), (32, 16)], torch.bfloat16, "int8")
    assert d / scale < 5e-2, f"rel={d/scale:.2e}"  # bf16-param SR -> expectation match


# ----------------------------------------------------------------- native fallback routing
def test_fallback_routing_and_parity():
    # tiny fused + big-2D (tile>cap) + conv + 1-D -> the last three go to the inner native Adakaon
    shapes = [(8, 16)] * 4 + [(256, 1024)] + [(64,)]
    pv = _bag(shapes, torch.float32, 1)
    # add a conv param (ndim>2)
    convv = torch.randn(8, 8, 3, 3, device=DEV).requires_grad_(True)
    pv.append(convv)
    pn = _clone(pv)
    cfg = dict(lr=2e-3, betas=(0.9, 0.999), cautious=True, gradient_centralization=True,
               momentum_dtype="float32")
    ov, on = FusedAdakaon(pv, **cfg), Adakaon(pn, **cfg)
    gen = torch.Generator(device=DEV).manual_seed(7)
    for _ in range(6):
        gs = [torch.randn(*p.shape, generator=gen, device=DEV) for p in pv]
        for p, g in zip(pv, gs):
            p.grad = g.clone()
        for p, g in zip(pn, gs):
            p.grad = g.clone()
        ov.step()
        on.step()
    torch.cuda.synchronize()
    assert len(ov._fused) == 4                                   # only the tiny 2-D weights fuse
    assert ov.inner is not None and len(ov.inner.param_groups[0]["params"]) == 3
    d = max((a.detach() - b.detach()).abs().max().item() for a, b in zip(pv, pn))
    assert d < 1e-5, f"max|Δp|={d:.2e}"


# ----------------------------------------------------------------- memory footprint
def test_bf16_momentum_two_bytes_per_param():
    ps = _bag([(32, 48)] * 8, torch.float32)
    for p in ps:
        p.grad = torch.randn_like(p)
    opt = FusedAdakaon(ps, momentum_dtype="bfloat16")
    opt.step()
    torch.cuda.synchronize()
    # state per fused param: m (bf16, 2B) + row (fp32) + col (fp32). Momentum is the dominant term.
    for p in ps:
        st = opt.state[p]
        assert st["m"].dtype == torch.bfloat16
        assert st["m"].numel() == p.numel()
        assert st["m"].element_size() == 2


# ----------------------------------------------------------------- pointer cache across realloc
def test_grad_realloc_still_correct():
    shapes = [(8, 16), (16, 8)]
    pv = _bag(shapes, torch.float32, 1)
    pn = _clone(pv)
    cfg = dict(lr=2e-3, momentum_dtype="float32")
    ov, on = FusedAdakaon(pv, **cfg), Adakaon(pn, **cfg)
    gen = torch.Generator(device=DEV).manual_seed(7)
    for _ in range(5):
        # fresh grad tensors each step (new data_ptr) -> exercises refresh_grads()
        for p in pv:
            p.grad = None
        for p in pn:
            p.grad = None
        gs = [torch.randn(*p.shape, generator=gen, device=DEV) for p in pv]
        for p, g in zip(pv, gs):
            p.grad = g.clone()
        for p, g in zip(pn, gs):
            p.grad = g.clone()
        ov.step()
        on.step()
    torch.cuda.synchronize()
    d = max((a.detach() - b.detach()).abs().max().item() for a, b in zip(pv, pn))
    assert d < 1e-5, f"max|Δp|={d:.2e}"


def test_grad_none_param_is_skipped():
    pv = _bag([(8, 16), (8, 16)], torch.float32, 1)
    opt = FusedAdakaon(pv, momentum_dtype="float32")
    pv[0].grad = torch.randn_like(pv[0])
    # pv[1].grad stays None
    before = pv[1].detach().clone()
    opt.step()
    torch.cuda.synchronize()
    assert torch.equal(pv[1].detach(), before)        # untouched
    assert not torch.equal(pv[0].detach(), _bag([(8, 16)], torch.float32, 1)[0].detach())


# ----------------------------------------------------------------- reusable device primitive
def test_sr_round_unbiased():
    """sr_round (the reusable bf16 SR primitive) is unbiased: averaging many draws -> the value."""
    import triton
    import triton.language as tl

    from kaon._fused_triton import sr_round

    @triton.jit
    def _probe(out_ptr, val, seed, N, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        res = tl.full((BLOCK,), val, tl.float32) * 1.0
        r = sr_round(res, seed, offs)
        tl.store(out_ptr + offs, r, mask=offs < N)

    N = 4096
    # a value strictly between two bf16 representables
    lo = torch.tensor([1.0], dtype=torch.bfloat16).float().item()
    val = lo + (torch.tensor([1.0], dtype=torch.bfloat16).float().item()) * 0  # 1.0 is representable
    val = 1.0 + 0.003  # between 1.0 and the next bf16 step (~0.0078)
    acc = torch.zeros(N, device=DEV)
    K = 300
    for k in range(K):
        out = torch.empty(N, device=DEV)
        _probe[(1,)](out, val, k + 1, N, BLOCK=N)
        acc += out
    torch.cuda.synchronize()
    mean = (acc / K).mean().item()
    assert abs(mean - val) < 5e-4, f"SR mean {mean} vs {val}"
    # every draw must be one of the two bf16 neighbours of val
    out = torch.empty(N, device=DEV)
    _probe[(1,)](out, val, 12345, N, BLOCK=N)
    torch.cuda.synchronize()
    uniq = torch.unique(out.bfloat16()).numel()
    assert uniq <= 2
