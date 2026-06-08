"""Unit tests for the experimental Triton fused kernel (``kaon._fused_triton``).

Covers:
  * fp32 parity with native Adakaon (exact), bf16 / bf16-param (SR) fidelity bounds,
  * int8 (1 B/param) and 4bit (0.5 B/param) momentum: parity vs native, memory, convergence,
  * decoupled weight_decay (parity + shrink), tile bucketing for mixed shapes,
  * native-fallback routing (1-D / conv / high-rank / odd-C-4bit) with end-to-end parity,
  * the reusable device primitives in isolation — ``sr_round``, ``requant_int8``, ``requant_4bit``,
    ``gradient_centralize``, ``factored_rc`` — each checked against its native/codec reference,
  * host helpers (``fused_eligible`` / tile sizing), pointer-cache across grad realloc, grad=None.

Skips cleanly when CUDA or Triton is unavailable (the kernel is GPU-only).
"""
from __future__ import annotations

import pytest
import torch

from kaon import Adakaon
from kaon._fused_triton import (
    HAS_TRITON,
    TILE_CAP,
    FusedAdakaon,
    fused_eligible,
    next_pow2_tile,
    warps_for,
)

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


def _run_parity(shapes, dtype, mdtype, *, cautious=True, gc=True, wd=0.0, steps=6, seed=1):
    """Step FusedAdakaon and native Adakaon on identical params+grads; return max|Δp| and scale."""
    cfg = dict(lr=2e-3, betas=(0.9, 0.999), weight_decay=wd, cautious=cautious,
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
    assert TILE_CAP == 1 << 17                                       # measured crossover (131072)
    assert fused_eligible(torch.zeros(8, 16, device=DEV))            # tiny 2-D
    assert fused_eligible(torch.zeros(16, 320, device=DEV))          # low-rank LoRA
    assert fused_eligible(torch.zeros(16, 320, device=DEV, dtype=torch.bfloat16))
    assert fused_eligible(torch.zeros(128, 1024, device=DEV))        # 131072 lanes == cap (medium)
    assert not fused_eligible(torch.zeros(128, device=DEV))          # 1-D
    assert not fused_eligible(torch.zeros(8, 8, 3, 3, device=DEV))   # conv ndim>2
    assert not fused_eligible(torch.zeros(8, 16))                    # cpu
    assert not fused_eligible(torch.zeros(8, 16, device=DEV, dtype=torch.float16))  # fp16
    assert not fused_eligible(torch.zeros(512, 1024, device=DEV))    # 524288 lanes > cap -> native
    # non-contiguous
    t = torch.zeros(16, 32, device=DEV).t()
    assert not fused_eligible(t)


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


def test_fp32_parity_medium_tiles_near_cap():
    # tiles up to the raised cap (131072 lanes) fuse and stay exact vs native (measured 1.2-1.4x faster)
    d, _, ov = _run_parity([(128, 1024), (256, 512)], torch.float32, "float32")
    assert d < 1e-5, f"max|Δp|={d:.2e}"
    assert len(ov._fused) == 2                                 # both fused at the new cap


# ----------------------------------------------------------------- chunked (big-tensor) path
def test_chunked_parity_fp32():
    # 1024x512 = 524288 lanes > cap -> chunked path; exact vs native (~2.5x faster on bigger ones)
    d, _, ov = _run_parity([(1024, 512)], torch.float32, "float32")
    assert d < 1e-5, f"max|Δp|={d:.2e}"
    assert len(ov._big) == 1 and len(ov._fused) == 0


@pytest.mark.parametrize("cautious", [True, False])
@pytest.mark.parametrize("gc", [True, False])
def test_chunked_parity_features(cautious, gc):
    d, _, _ = _run_parity([(1024, 512)], torch.float32, "float32", cautious=cautious, gc=gc, wd=0.05)
    assert d < 1e-5, f"max|Δp|={d:.2e}"


def test_chunked_bf16_momentum():
    d, scale, _ = _run_parity([(1024, 512)], torch.float32, "bfloat16")
    assert d / scale < 5e-3, f"rel={d/scale:.2e}"


def test_chunked_bf16_params_sr():
    d, scale, _ = _run_parity([(1024, 512)], torch.bfloat16, "bfloat16")
    assert d / scale < 5e-2, f"rel={d/scale:.2e}"


def test_chunked_mixed_with_small_and_1d():
    # a model-ish mix: small (one-block) + big (chunked) + 1-D (native), all parity vs native at once
    shapes = [(8, 16), (16, 320), (1024, 512)]
    d, _, ov = _run_parity(shapes, torch.float32, "float32")
    assert d < 1e-5, f"max|Δp|={d:.2e}"
    assert len(ov._fused) == 2 and len(ov._big) == 1


def test_chunked_int8_big_routes_to_native():
    # chunked quant is future work -> a big int8 tensor goes to the native inner optimizer
    ps = _bag([(1024, 512)], torch.float32)
    for p in ps:
        p.grad = torch.randn_like(p)
    ov = FusedAdakaon(ps, momentum_dtype="int8")
    ov.step()
    torch.cuda.synchronize()
    assert len(ov._big) == 0 and len(ov._fused) == 0 and ov.inner is not None
    assert ov.inner.state[ps[0]]["m"].dtype == torch.int8
    assert torch.isfinite(ps[0]).all()


# ----------------------------------------------------------------- decoupled weight decay
def test_weight_decay_parity_fp32():
    # decoupled wd folded into delta before cautious -- must match native exactly (fp32)
    d, scale, _ = _run_parity([(8, 16), (16, 8), (12, 20)], torch.float32, "float32", wd=0.05)
    assert d < 1e-5, f"max|Δp|={d:.2e}"


@pytest.mark.parametrize("mdtype", ["int8", "4bit"])
def test_weight_decay_parity_quant(mdtype):
    d, scale, _ = _run_parity([(8, 16)] * 4, torch.float32, mdtype, wd=0.05)
    assert d / scale < 5e-4, f"{mdtype} rel={d/scale:.2e}"


def test_weight_decay_shrinks_weights():
    # with no gradient signal pulling them, wd>0 should shrink weights vs wd=0
    torch.manual_seed(0)
    shapes = [(16, 32)] * 3
    p0 = _bag(shapes, torch.float32, 5)
    p_wd = [p.detach().clone().requires_grad_(True) for p in p0]
    p_no = [p.detach().clone().requires_grad_(True) for p in p0]
    o_wd = FusedAdakaon(p_wd, lr=1e-2, weight_decay=0.2, momentum_dtype="float32")
    o_no = FusedAdakaon(p_no, lr=1e-2, weight_decay=0.0, momentum_dtype="float32")
    gen = torch.Generator(device=DEV).manual_seed(9)
    for _ in range(10):
        gs = [torch.randn(*p.shape, generator=gen, device=DEV) * 0.01 for p in p0]
        for p, g in zip(p_wd, gs):
            p.grad = g.clone()
        for p, g in zip(p_no, gs):
            p.grad = g.clone()
        o_wd.step()
        o_no.step()
    torch.cuda.synchronize()
    n_wd = sum(p.norm().item() for p in p_wd)
    n_no = sum(p.norm().item() for p in p_no)
    assert n_wd < n_no, f"wd norm {n_wd:.3f} !< no-wd {n_no:.3f}"


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


def test_int8_quant_primitives_match_codec():
    """The REUSABLE device primitive requant_int8 == native _quant_int8 (bit-for-bit)."""
    import triton
    import triton.language as tl

    from kaon._fused_triton import requant_int8
    from kaon._momentum_codec import _quant_int8

    R, C, BR, BC = 8, 24, 8, 32

    @triton.jit
    def _probe(m_ptr, code_ptr, scale_ptr, R, C, BR: tl.constexpr, BC: tl.constexpr):
        ri = tl.arange(0, BR)[:, None]
        ci = tl.arange(0, BC)[None, :]
        m2 = (ri < R) & (ci < C)
        idx = ri * C + ci
        rr = tl.arange(0, BR)
        m = tl.load(m_ptr + idx, mask=m2, other=0.0)
        requant_int8(m, m2, code_ptr, idx, scale_ptr, rr, R)

    m = torch.randn(R, C, device=DEV)
    code = torch.zeros(R, C, dtype=torch.int8, device=DEV)
    scale = torch.zeros(R, device=DEV)
    _probe[(1,)](m, code, scale, R, C, BR=BR, BC=BC)
    torch.cuda.synchronize()
    q_ref, scale_ref = _quant_int8(m)
    assert torch.equal(code, q_ref)                              # codes bit-identical
    assert torch.allclose(scale, scale_ref.view(-1), atol=0, rtol=0)


# ----------------------------------------------------------------- 4bit momentum (in-kernel)
def test_4bit_parity_with_native():
    # even-C 4bit fuses in-kernel; per-128-block dequant/requant matches native Adakaon(4bit) closely
    d, scale, _ = _run_parity([(8, 16)] * 6, torch.float32, "4bit")
    assert d / scale < 5e-4, f"rel={d/scale:.2e}"


def test_4bit_parity_mixed_even_shapes():
    shapes = [(8, 16), (16, 8), (12, 20), (16, 320)]           # all even C
    d, scale, ov = _run_parity(shapes, torch.float32, "4bit")
    assert d / scale < 5e-4, f"rel={d/scale:.2e}"
    assert len(ov._fused) == len(shapes)                       # all fused (even C)


def test_4bit_half_byte_per_param():
    ps = _bag([(16, 64)] * 4, torch.float32)
    for p in ps:
        p.grad = torch.randn_like(p)
    ov = FusedAdakaon(ps, momentum_dtype="4bit")
    ov.step()
    torch.cuda.synchronize()
    st = ov.state[ps[0]]
    assert st["m"].dtype == torch.uint8
    assert st["m"].numel() == ps[0].numel() // 2               # packed two codes/byte -> 0.5 B/param


def test_4bit_odd_C_routes_to_native():
    # odd column count can't pack cleanly into bytes -> native fallback (still correct)
    ps = _bag([(8, 15)], torch.float32)
    for p in ps:
        p.grad = torch.randn_like(p)
    ov = FusedAdakaon(ps, momentum_dtype="4bit")
    ov.step()
    torch.cuda.synchronize()
    assert len(ov._fused) == 0 and ov.inner is not None
    assert ov.inner.state[ps[0]]["m"].dtype == torch.uint8     # 4bit on the native optimizer
    assert torch.isfinite(ps[0]).all()


def test_4bit_converges():
    torch.manual_seed(0)
    w = torch.randn(16, 24, device=DEV).requires_grad_(True)
    target = torch.randn(16, 24, device=DEV)
    opt = FusedAdakaon([w], lr=5e-2, momentum_dtype="4bit")
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


def test_4bit_quant_primitive_matches_codec():
    """The REUSABLE device primitive requant_4bit == native _quant_4bit (packed bytes + scale)."""
    import triton
    import triton.language as tl

    from kaon._fused_triton import requant_4bit
    from kaon._momentum_codec import _quant_4bit

    R, C, BR, BC = 8, 16, 8, 16
    numel = R * C
    Chalf, BLK = C // 2, min(128, numel)
    NB = (numel + BLK - 1) // BLK

    @triton.jit
    def _probe(m_ptr, packed_ptr, scale_ptr, R, C, Chalf, NB, BLK, BR: tl.constexpr, BC: tl.constexpr):
        ri = tl.arange(0, BR)[:, None]
        ci = tl.arange(0, BC)[None, :]
        m2 = (ri < R) & (ci < C)
        idx = ri * C + ci
        m = tl.load(m_ptr + idx, mask=m2, other=0.0)
        requant_4bit(m, m2, idx, R, C, Chalf, packed_ptr, scale_ptr, NB, BLK, BR, BC)

    m = torch.randn(R, C, device=DEV)
    packed = torch.zeros(R * Chalf, dtype=torch.uint8, device=DEV)
    scale = torch.zeros(NB, device=DEV)
    _probe[(1,)](m, packed, scale, R, C, Chalf, NB, BLK, BR=BR, BC=BC)
    torch.cuda.synchronize()
    p_ref, s_ref, _ = _quant_4bit(m, 128)
    assert torch.equal(packed, p_ref)                          # packed bytes bit-identical
    assert torch.allclose(scale, s_ref, atol=0, rtol=0)


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
    # all three paths exercised at once: tiny 2-D -> one-block, 256x1024 -> chunked, conv + 1-D -> native
    assert len(ov._fused) == 4
    assert len(ov._big) == 1
    assert ov.inner is not None and len(ov.inner.param_groups[0]["params"]) == 2
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


# ----------------------------------------------------------------- reusable factored primitives
def test_gradient_centralize_primitive():
    """gradient_centralize device helper == torch GC (subtract per-row fan-in mean)."""
    import triton
    import triton.language as tl

    from kaon._fused_triton import gradient_centralize

    R, C, BR, BC = 6, 10, 8, 16

    @triton.jit
    def _probe(g_ptr, o_ptr, R, C, BR: tl.constexpr, BC: tl.constexpr):
        ri = tl.arange(0, BR)[:, None]
        ci = tl.arange(0, BC)[None, :]
        m2 = (ri < R) & (ci < C)
        idx = ri * C + ci
        g = tl.load(g_ptr + idx, mask=m2, other=0.0)
        tl.store(o_ptr + idx, gradient_centralize(g, m2, C.to(tl.float32)), mask=m2)

    g = torch.randn(R, C, device=DEV)
    o = torch.zeros(R, C, device=DEV)
    _probe[(1,)](g, o, R, C, BR=BR, BC=BC)
    torch.cuda.synchronize()
    assert torch.allclose(o, g - g.mean(dim=1, keepdim=True), atol=1e-5)


def test_factored_rc_primitive():
    """factored_rc device helper == native update_factored_state + factored_inv_sqrt_factors,
    and it updates the row/col EMA state in place."""
    import triton
    import triton.language as tl

    from kaon._factored import factored_inv_sqrt_factors, update_factored_state
    from kaon._fused_triton import factored_rc

    R, C, BR, BC = 6, 10, 8, 16

    @triton.jit
    def _probe(g_ptr, rowp, colp, rfac, cfac, R, C, beta2, eps1, BR: tl.constexpr, BC: tl.constexpr):
        ri = tl.arange(0, BR)[:, None]
        ci = tl.arange(0, BC)[None, :]
        rr = tl.arange(0, BR)
        cc = tl.arange(0, BC)
        m2 = (ri < R) & (ci < C)
        idx = ri * C + ci
        g = tl.load(g_ptr + idx, mask=m2, other=0.0)
        rf, cf = factored_rc(g, rowp, colp, rr, cc, R, C, R.to(tl.float32), C.to(tl.float32), beta2, eps1)
        tl.store(rfac + rr, rf, mask=rr < R)
        tl.store(cfac + cc, cf, mask=cc < C)

    g = torch.randn(R, C, device=DEV)
    row = torch.zeros(R, device=DEV)
    col = torch.zeros(C, device=DEV)
    rfac = torch.zeros(R, device=DEV)
    cfac = torch.zeros(C, device=DEV)
    _probe[(1,)](g, row, col, rfac, cfac, R, C, 0.999, 1e-30, BR=BR, BC=BC)
    torch.cuda.synchronize()
    row_r = torch.zeros(R, device=DEV)
    col_r = torch.zeros(C, device=DEV)
    update_factored_state(g, row_r, col_r, 0.999, 1e-30)
    rf_r, cf_r = factored_inv_sqrt_factors(row_r, col_r)
    assert torch.allclose(rfac, rf_r.view(-1), atol=1e-4)
    assert torch.allclose(cfac, cf_r.view(-1), atol=1e-4)
    assert torch.allclose(row, row_r, atol=1e-5)            # row EMA updated in place
    assert torch.allclose(col, col_r, atol=1e-5)
