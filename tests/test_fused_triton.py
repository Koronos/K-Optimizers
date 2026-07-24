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
    fused_1d_eligible,
    fused_eligible,
    next_pow2_tile,
    warps_for,
)

pytestmark = pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()),
    reason="Triton fused kernel requires CUDA + Triton",
)

DEV = "cuda"

if HAS_TRITON:
    import triton
    import triton.language as tl

    from kaon._fused_triton import (
        factored_rc,
        gradient_centralize,
        requant_4bit,
        requant_int8,
        sr_round,
    )

    @triton.jit
    def _int8_quant_probe(m_ptr, code_ptr, scale_ptr, R, C, BR: tl.constexpr, BC: tl.constexpr):
        ri = tl.arange(0, BR)[:, None]
        ci = tl.arange(0, BC)[None, :]
        mask = (ri < R) & (ci < C)
        idx = ri * C + ci
        rows = tl.arange(0, BR)
        momentum = tl.load(m_ptr + idx, mask=mask, other=0.0)
        requant_int8(momentum, mask, code_ptr, idx, scale_ptr, rows, R)

    @triton.jit
    def _fourbit_quant_probe(
        m_ptr, packed_ptr, scale_ptr, R, C, Chalf, NB, BLK,
        BR: tl.constexpr, BC: tl.constexpr,
    ):
        ri = tl.arange(0, BR)[:, None]
        ci = tl.arange(0, BC)[None, :]
        mask = (ri < R) & (ci < C)
        idx = ri * C + ci
        momentum = tl.load(m_ptr + idx, mask=mask, other=0.0)
        requant_4bit(
            momentum, mask, idx, R, C, Chalf, packed_ptr, scale_ptr,
            NB, BLK, BR, BC,
        )

    @triton.jit
    def _sr_round_probe(out_ptr, val, seed, N, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        values = tl.full((BLOCK,), val, tl.float32) * 1.0
        rounded = sr_round(values, seed, offsets)
        tl.store(out_ptr + offsets, rounded, mask=offsets < N)

    @triton.jit
    def _gradient_centralize_probe(g_ptr, out_ptr, R, C, BR: tl.constexpr, BC: tl.constexpr):
        ri = tl.arange(0, BR)[:, None]
        ci = tl.arange(0, BC)[None, :]
        mask = (ri < R) & (ci < C)
        idx = ri * C + ci
        grad = tl.load(g_ptr + idx, mask=mask, other=0.0)
        tl.store(
            out_ptr + idx,
            gradient_centralize(grad, mask, C.to(tl.float32)),
            mask=mask,
        )

    @triton.jit
    def _factored_rc_probe(
        g_ptr, row_ptr, col_ptr, row_factor_ptr, col_factor_ptr,
        R, C, beta2, eps1, BR: tl.constexpr, BC: tl.constexpr,
    ):
        ri = tl.arange(0, BR)[:, None]
        ci = tl.arange(0, BC)[None, :]
        rows = tl.arange(0, BR)
        cols = tl.arange(0, BC)
        mask = (ri < R) & (ci < C)
        idx = ri * C + ci
        grad = tl.load(g_ptr + idx, mask=mask, other=0.0)
        row_factor, col_factor = factored_rc(
            grad, row_ptr, col_ptr, rows, cols, R, C,
            R.to(tl.float32), C.to(tl.float32), beta2, eps1,
        )
        tl.store(row_factor_ptr + rows, row_factor, mask=rows < R)
        tl.store(col_factor_ptr + cols, col_factor, mask=cols < C)


# ----------------------------------------------------------------- helpers
def _fused(params, **kw):
    """The fused optimizer under test is just ``Adakaon(fused=True)`` (no separate class)."""
    return Adakaon(params, fused=True, **kw)


def _parts(opt):
    """(one_block, big, one_dim, native) param lists from the cached fused partition (after a step)."""
    ob, big, od, nat = [], [], [], []
    for (_ids, o, b, d, n) in opt._fused_part.values():
        ob += o
        big += b
        od += d
        nat += n
    return ob, big, od, nat


def _buckets(opt):
    """All one-block tile buckets across the cached pointer-array caches (after a step)."""
    return [bk for cache in opt._fused_ob_caches.values() for bk in cache.buckets]


def _bag(shapes, dtype=torch.float32, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return [torch.randn(*s, generator=g, device=DEV, dtype=dtype).requires_grad_(True) for s in shapes]


def _clone(ps):
    return [p.detach().clone().requires_grad_(True) for p in ps]


def _run_parity(shapes, dtype, mdtype, *, cautious=True, gc=True, wd=0.0, steps=6, seed=1):
    """Step Adakaon(fused=True) and native Adakaon on identical params+grads; return max|Δp| and scale."""
    cfg = dict(lr=2e-3, betas=(0.9, 0.999), weight_decay=wd, cautious=cautious,
               gradient_centralization=gc, momentum_dtype=mdtype)
    pv = _bag(shapes, dtype, seed)
    pn = _clone(pv)
    ov = _fused(pv, **cfg)
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


def test_autolr_resets_rebuild_caches_and_preserve_native_parity():
    """Two AutoLR contacts must not leave pointer arrays targeting discarded state."""
    pv = _bag([(8, 16), (32,)], torch.float32, seed=31)
    pn = _clone(pv)
    cfg = dict(lr=2e-3, momentum_dtype="float32", cautious=False,
               gradient_centralization=False)
    fused = Adakaon(pv, fused=True, **cfg)
    native = Adakaon(pn, **cfg)
    gen = torch.Generator(device=DEV).manual_seed(32)
    retired_caches = []
    retired_state_tensors = []

    for contact in range(3):
        gs = [torch.randn(p.shape, generator=gen, device=DEV) for p in pv]
        for p, g in zip(pv, gs, strict=True):
            p.grad = g.clone()
        for p, g in zip(pn, gs, strict=True):
            p.grad = g.clone()
        fused.step()
        native.step()

        current_caches = tuple(fused._fused_ob_caches.values()) + tuple(fused._fused_od_caches.values())
        assert fused._fused_ob_caches and fused._fused_od_caches
        assert all(new is not old for new in current_caches for old in retired_caches)
        if contact == 2:
            break

        retired_caches.extend(current_caches)
        retired_state_tensors.extend(
            value
            for state in fused.state.values()
            for value in state.values()
            if torch.is_tensor(value)
        )
        fused._autolr_reset_base_state()
        native._autolr_reset_base_state()
        assert fused._t == 0
        assert not fused.state
        assert not fused._fused_part
        assert not fused._fused_ob_caches
        assert not fused._fused_od_caches

    assert all(
        new is not old
        for state in fused.state.values()
        for new in state.values()
        if torch.is_tensor(new)
        for old in retired_state_tensors
    )
    for a, b in zip(pv, pn, strict=True):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_load_state_dict_invalidates_pointer_caches():
    ps = _bag([(8, 16), (32,)], seed=33)
    opt = Adakaon(ps, fused=True, momentum_dtype="float32")
    for p in ps:
        p.grad = torch.ones_like(p)
    opt.step()
    old_caches = tuple(opt._fused_ob_caches.values()) + tuple(opt._fused_od_caches.values())
    assert old_caches

    opt.load_state_dict(opt.state_dict())
    assert not opt._fused_part
    assert not opt._fused_ob_caches
    assert not opt._fused_od_caches

    for p in ps:
        p.grad = torch.ones_like(p)
    opt.step()
    new_caches = tuple(opt._fused_ob_caches.values()) + tuple(opt._fused_od_caches.values())
    assert new_caches
    assert all(new is not old for new in new_caches for old in old_caches)


# ----------------------------------------------------------------- host helpers
def test_eligibility_predicate():
    assert TILE_CAP == 1 << 17                                       # measured crossover (131072)
    assert fused_eligible(torch.zeros(8, 16, device=DEV))            # tiny 2-D
    assert fused_eligible(torch.zeros(16, 320, device=DEV))          # low-rank LoRA
    assert fused_eligible(torch.zeros(16, 320, device=DEV, dtype=torch.bfloat16))
    assert fused_eligible(torch.zeros(128, 1024, device=DEV))        # 131072 lanes == cap (medium)
    assert not fused_eligible(torch.zeros(128, device=DEV))          # 1-D
    assert fused_eligible(torch.zeros(8, 8, 3, 3, device=DEV))       # conv ndim>2 -> matrixized (8,72)
    assert not fused_eligible(torch.zeros(8, 8, 256, 256, device=DEV))  # conv too big for one block
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
    assert len(_buckets(ov)) >= 4


def test_fp32_parity_lora_shapes():
    d, _, _ = _run_parity([(16, 320), (320, 16), (8, 1280)], torch.float32, "float32")
    assert d < 1e-5, f"max|Δp|={d:.2e}"


def test_fp32_parity_medium_tiles_near_cap():
    # tiles up to the raised cap (131072 lanes) fuse and stay exact vs native (measured 1.2-1.4x faster)
    d, _, ov = _run_parity([(128, 1024), (256, 512)], torch.float32, "float32")
    assert d < 1e-5, f"max|Δp|={d:.2e}"
    assert len(_parts(ov)[0]) == 2                                 # both fused at the new cap


# ----------------------------------------------------------------- chunked (big-tensor) path
def test_chunked_parity_fp32():
    # 1024x512 = 524288 lanes > cap -> chunked path; exact vs native (~2.5x faster on bigger ones)
    d, _, ov = _run_parity([(1024, 512)], torch.float32, "float32")
    assert d < 1e-5, f"max|Δp|={d:.2e}"
    assert len(_parts(ov)[1]) == 1 and len(_parts(ov)[0]) == 0


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
    assert len(_parts(ov)[0]) == 2 and len(_parts(ov)[1]) == 1


def test_chunked_int8_parity():
    # big int8 momentum via the codec (dequant -> fp32 temp -> kernels -> requant); 1 B/param
    d, scale, ov = _run_parity([(1024, 512)], torch.float32, "int8")
    assert d / scale < 5e-4, f"rel={d/scale:.2e}"
    assert len(_parts(ov)[1]) == 1
    st = ov.state[_parts(ov)[1][0]]
    assert st["m"].dtype == torch.int8 and st["m"].numel() == _parts(ov)[1][0].numel()  # 1 B/param


def test_chunked_4bit_parity():
    d, scale, ov = _run_parity([(1024, 512)], torch.float32, "4bit")
    assert d / scale < 5e-4, f"rel={d/scale:.2e}"
    assert len(_parts(ov)[1]) == 1
    st = ov.state[_parts(ov)[1][0]]
    assert st["m"].dtype == torch.uint8 and st["m"].numel() == _parts(ov)[1][0].numel() // 2  # 0.5 B/param


def test_chunked_4bit_odd_C():
    # the chunked codec packs the flat tensor, so 4bit handles odd C (unlike the one-block path)
    d, scale, ov = _run_parity([(1024, 513)], torch.float32, "4bit")
    assert d / scale < 5e-4, f"rel={d/scale:.2e}"
    assert len(_parts(ov)[1]) == 1


@pytest.mark.parametrize("mdtype", ["int8", "4bit"])
def test_chunked_quant_features(mdtype):
    d, scale, _ = _run_parity([(1024, 512)], torch.float32, mdtype, wd=0.05, cautious=True, gc=False)
    assert d / scale < 5e-4, f"{mdtype} rel={d/scale:.2e}"


# ------------------------------------------- batched chunked (>=2 same-shape big tensors) parity
# The Cosmos LoKr regime: many same-shape factors > tile_cap. >=2 same-shape big tensors take the
# batched chunked kernel (~2 launches for the whole bucket); a lone big tensor keeps the per-tensor
# chunked kernel. Both must match native exactly (fp32) / within the dtype bound. 512x512 > cap.
def test_big_batched_routes_and_parity_fp32():
    d, _, ov = _run_parity([(512, 512)] * 3, torch.float32, "float32")
    ob, big, od, nat = _parts(ov)
    assert len(big) == 3 and len(ob) == 0 and len(nat) == 0   # all big, dispatched batched-chunked
    assert d < 1e-5, f"max|Δp|={d:.2e}"


@pytest.mark.parametrize("cautious", [True, False])
@pytest.mark.parametrize("gc", [True, False])
def test_big_batched_features(cautious, gc):
    d, _, _ = _run_parity([(512, 512)] * 3, torch.float32, "float32", cautious=cautious, gc=gc, wd=0.05)
    assert d < 1e-5, f"cautious={cautious} gc={gc} max|Δp|={d:.2e}"


def test_big_batched_bf16_momentum():
    d, scale, _ = _run_parity([(512, 512)] * 3, torch.float32, "bfloat16")
    assert d / scale < 5e-3, f"rel={d/scale:.2e}"


def test_big_batched_bf16_params_sr():
    d, scale, _ = _run_parity([(512, 512)] * 3, torch.bfloat16, "bfloat16")
    assert d / scale < 5e-2, f"rel={d/scale:.2e}"


@pytest.mark.parametrize("mdtype", ["int8", "4bit"])
def test_big_batched_quant_parity(mdtype):
    d, scale, _ = _run_parity([(512, 512)] * 3, torch.float32, mdtype, wd=0.05)
    assert d / scale < 5e-4, f"{mdtype} rel={d/scale:.2e}"


def test_big_batched_4bit_odd_C():
    # chunked codec packs the flat tensor -> 4bit handles odd C in the batched path too
    d, scale, _ = _run_parity([(512, 511)] * 2, torch.float32, "4bit")
    assert d / scale < 5e-4, f"rel={d/scale:.2e}"


def test_big_batched_matches_native_foreach_toggle():
    # the batched chunked path (default) must equal the in-fused native-foreach fallback (toggle off).
    # fp32 momentum so the two paths match near-exactly (bf16 momentum rounds differently per path).
    cfg = dict(lr=2e-3, weight_decay=0.05, cautious=True, gradient_centralization=True,
               momentum_dtype="float32")
    pv = _bag([(512, 512)] * 3, torch.float32, seed=2)
    pn = _clone(pv)
    ov = Adakaon(pv, fused=True, **cfg)              # batched chunked
    on = Adakaon(pn, fused=True, **cfg)
    on._fused_big_batched = False                    # native-foreach fallback
    gen = torch.Generator(device=DEV).manual_seed(9)
    for _ in range(6):
        gs = [torch.randn(*p.shape, generator=gen, device=DEV) for p in pv]
        for p, g in zip(pv, gs):
            p.grad = g.clone()
        for p, g in zip(pn, gs):
            p.grad = g.clone()
        ov.step()
        on.step()
    torch.cuda.synchronize()
    d = max((a - b).abs().max().item() for a, b in zip(pv, pn))
    assert d < 1e-5, f"batched vs native-foreach max|Δp|={d:.2e}"


# ------------------------------------------------- one-block non-factored 1-D (biases / norm scales)
# Many tiny 1-D tensors are the launch-bound regime (like the 2-D LoRA bag); the fused 1-D kernel owns
# one per program. fp32/bf16 momentum only — int8/4bit 1-D routes to native.
def test_one_dim_eligibility_predicate():
    assert fused_1d_eligible(torch.zeros(1024, device=DEV))
    assert fused_1d_eligible(torch.zeros(2048, device=DEV, dtype=torch.bfloat16))
    assert not fused_1d_eligible(torch.zeros(8, 16, device=DEV))             # 2-D -> not the 1-D path
    assert not fused_1d_eligible(torch.zeros(TILE_CAP * 2, device=DEV))      # too big for one block


def test_one_dim_routes_and_parity_fp32():
    d, _, ov = _run_parity([(1024,)] * 4, torch.float32, "float32")
    ob, big, od, nat = _parts(ov)
    assert len(od) == 4 and len(ob) == 0 and len(big) == 0 and len(nat) == 0
    assert d < 1e-5, f"max|Δp|={d:.2e}"


@pytest.mark.parametrize("cautious", [True, False])
def test_one_dim_features(cautious):
    d, _, _ = _run_parity([(1024,), (512,)], torch.float32, "float32", cautious=cautious, wd=0.05)
    assert d < 1e-5, f"cautious={cautious} max|Δp|={d:.2e}"


def test_one_dim_bf16_momentum():
    d, scale, _ = _run_parity([(1024,)] * 3, torch.float32, "bfloat16")
    assert d / scale < 5e-3, f"rel={d/scale:.2e}"


def test_one_dim_bf16_params_sr():
    d, scale, _ = _run_parity([(1024,)] * 3, torch.bfloat16, "bfloat16")
    assert d / scale < 5e-2, f"rel={d/scale:.2e}"


def test_one_dim_mixed_lengths_bucketing():
    # different lengths share a block bucket by next_pow2(L), masked by the true length
    d, _, ov = _run_parity([(1000,), (1024,), (700,), (512,)], torch.float32, "float32")
    assert len(_parts(ov)[2]) == 4
    assert d < 1e-5, f"max|Δp|={d:.2e}"


@pytest.mark.parametrize("mdtype", ["int8", "4bit"])
def test_one_dim_quant_routes_to_native(mdtype):
    # quant momentum has no 1-D codec analogue here -> native (still correct)
    d, scale, ov = _run_parity([(1024,)] * 3, torch.float32, mdtype)
    ob, big, od, nat = _parts(ov)
    assert len(od) == 0 and len(nat) == 3      # all on the native path
    assert d / scale < 5e-4, f"{mdtype} rel={d/scale:.2e}"


def test_one_dim_mixed_with_2d():
    # a realistic mix: 2-D one-block + 1-D fused at once, all parity vs native
    d, _, ov = _run_parity([(8, 16), (16, 16), (1024,), (256,)], torch.float32, "float32")
    ob, big, od, nat = _parts(ov)
    assert len(ob) == 2 and len(od) == 2 and len(nat) == 0
    assert d < 1e-5, f"max|Δp|={d:.2e}"


# ------------------------------------------------- conv (ndim>2) matrixized to (out, in*kh*kw)
# A contiguous conv's row-major storage IS its (out, in*kh*kw) view, so it rides the 2-D fused paths
# (one-block when small, chunked when big) with no copy. fp32/bf16 momentum only (quant -> native).
def test_conv_one_block_routes_and_parity():
    d, _, ov = _run_parity([(16, 8, 3, 3)] * 4, torch.float32, "float32")   # eff (16,72) -> one-block
    ob, big, od, nat = _parts(ov)
    assert len(ob) == 4 and len(big) == 0 and len(nat) == 0
    assert d < 1e-5, f"max|Δp|={d:.2e}"


@pytest.mark.parametrize("cautious", [True, False])
def test_conv_one_block_features(cautious):
    d, _, _ = _run_parity([(16, 8, 3, 3), (24, 8, 3, 3)], torch.float32, "float32", cautious=cautious, wd=0.05)
    assert d < 1e-5, f"cautious={cautious} max|Δp|={d:.2e}"


def test_conv_big_batched_routes_and_parity():
    d, _, ov = _run_parity([(256, 128, 3, 3)] * 3, torch.float32, "float32")  # eff (256,1152) -> chunked big
    ob, big, od, nat = _parts(ov)
    assert len(big) == 3 and len(ob) == 0 and len(nat) == 0
    assert d < 1e-5, f"max|Δp|={d:.2e}"


def test_conv_bf16_momentum():
    d, scale, _ = _run_parity([(16, 8, 3, 3)] * 4, torch.float32, "bfloat16")
    assert d / scale < 5e-3, f"rel={d/scale:.2e}"


@pytest.mark.parametrize("mdtype", ["int8", "4bit"])
def test_conv_quant_routes_to_native(mdtype):
    # quant momentum's per-row requant would reshape the conv state -> conv routes to native
    d, scale, ov = _run_parity([(16, 8, 3, 3)] * 4, torch.float32, mdtype)
    ob, big, od, nat = _parts(ov)
    assert len(nat) == 4 and len(ob) == 0 and len(big) == 0
    assert d / scale < 5e-4, f"{mdtype} rel={d/scale:.2e}"


# ------------------------------------------------- candidate #4: fused reductions (no [N,R,C] stack)
def _run_toggle(shapes, dtype, mdtype, *, fused_red, cautious=True, gc=True, wd=0.05, steps=6, seed=3):
    cfg = dict(lr=2e-3, betas=(0.9, 0.999), weight_decay=wd, cautious=cautious,
               gradient_centralization=gc, momentum_dtype=mdtype)
    ps = _bag(shapes, dtype, seed)
    opt = Adakaon(ps, fused=True, **cfg)
    opt._fused_reductions = fused_red
    gen = torch.Generator(device=DEV).manual_seed(11)
    for _ in range(steps):
        for p in ps:
            p.grad = torch.randn(*p.shape, generator=gen, device=DEV, dtype=dtype)
        opt.step()
    torch.cuda.synchronize()
    return ps


@pytest.mark.parametrize("gc", [True, False])
@pytest.mark.parametrize("shapes", [[(512, 512)] * 3, [(256, 128, 3, 3)] * 3])
def test_fused_reductions_matches_torch_reductions_fp32(shapes, gc):
    a = _run_toggle(shapes, torch.float32, "float32", fused_red=False, gc=gc)
    b = _run_toggle(shapes, torch.float32, "float32", fused_red=True, gc=gc)
    d = max((x.detach() - y.detach()).abs().max().item() for x, y in zip(a, b))
    assert d < 1e-4, f"gc={gc} max|Δp|={d:.2e}"   # atomic reduction order -> ~1e-7, well under


def test_fused_reductions_matches_native_bf16():
    # default path (fused reductions ON) must still match native within the bf16 bound
    d, scale, _ = _run_parity([(512, 512)] * 3, torch.bfloat16, "bfloat16")
    assert d / scale < 5e-2, f"rel={d/scale:.2e}"


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
    o_wd = _fused(p_wd, lr=1e-2, weight_decay=0.2, momentum_dtype="float32")
    o_no = _fused(p_no, lr=1e-2, weight_decay=0.0, momentum_dtype="float32")
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
    # Adakaon(fused=True) runs the EMA in fp32 then rounds to bf16; native bf16 lerps in bf16. Equivalent
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
    assert len(_buckets(ov)) >= 3                       # mixed tiles still bucket


def test_int8_state_layout_and_memory():
    ps = _bag([(32, 48)] * 6, torch.float32)
    for p in ps:
        p.grad = torch.randn_like(p)
    ov = _fused(ps, momentum_dtype="int8")
    on = Adakaon([p.detach().clone().requires_grad_(True) for p in ps], momentum_dtype="int8")
    for p in on.param_groups[0]["params"]:
        p.grad = torch.randn_like(p)
    ov.step()
    on.step()
    torch.cuda.synchronize()
    st = ov.state[ps[0]]
    assert st["m"].dtype == torch.int8 and st["m"].element_size() == 1      # 1 byte/param
    assert st["m"].numel() == ps[0].numel()
    assert st["m_scale"].shape == (32, 1)                                   # per-row codec scale

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
    opt = _fused([w], lr=5e-2, momentum_dtype="int8")
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
    from kaon._momentum_codec import _quant_int8

    R, C, BR, BC = 8, 24, 8, 32

    m = torch.randn(R, C, device=DEV)
    code = torch.zeros(R, C, dtype=torch.int8, device=DEV)
    scale = torch.zeros(R, device=DEV)
    _int8_quant_probe[(1,)](m, code, scale, R, C, BR=BR, BC=BC)
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
    assert len(_parts(ov)[0]) == len(shapes)                       # all fused (even C)


def test_4bit_half_byte_per_param():
    ps = _bag([(16, 64)] * 4, torch.float32)
    for p in ps:
        p.grad = torch.randn_like(p)
    ov = _fused(ps, momentum_dtype="4bit")
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
    ov = _fused(ps, momentum_dtype="4bit")
    ov.step()
    torch.cuda.synchronize()
    ob, big, od, nat = _parts(ov)
    assert len(ob) == 0 and len(big) == 0 and len(nat) == 1    # small odd-C 4bit -> native subset
    assert ov.state[ps[0]]["m"].dtype == torch.uint8           # 4bit momentum on the native path
    assert torch.isfinite(ps[0]).all()


def test_4bit_converges():
    torch.manual_seed(0)
    w = torch.randn(16, 24, device=DEV).requires_grad_(True)
    target = torch.randn(16, 24, device=DEV)
    opt = _fused([w], lr=5e-2, momentum_dtype="4bit")
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
    from kaon._momentum_codec import _quant_4bit

    R, C, BR, BC = 8, 16, 8, 16
    numel = R * C
    Chalf, BLK = C // 2, min(128, numel)
    NB = (numel + BLK - 1) // BLK

    m = torch.randn(R, C, device=DEV)
    packed = torch.zeros(R * Chalf, dtype=torch.uint8, device=DEV)
    scale = torch.zeros(NB, device=DEV)
    _fourbit_quant_probe[(1,)](m, packed, scale, R, C, Chalf, NB, BLK, BR=BR, BC=BC)
    torch.cuda.synchronize()
    p_ref, s_ref, _ = _quant_4bit(m, 128)
    assert torch.equal(packed, p_ref)                          # packed bytes bit-identical
    assert torch.allclose(scale, s_ref, atol=0, rtol=0)


# ----------------------------------------------------------------- routing across all paths
def test_fallback_routing_and_parity():
    # all paths at once: tiny 2-D -> one-block, small conv -> one-block (matrixized), big 2-D ->
    # chunked, 1-D -> one-dim, and a non-contiguous 2-D -> native fallback. All parity vs native.
    shapes = [(8, 16)] * 4 + [(256, 1024)] + [(64,)]
    pv = _bag(shapes, torch.float32, 1)
    convv = torch.randn(8, 8, 3, 3, device=DEV).requires_grad_(True)   # ndim>2 -> one-block (eff 8x72)
    pv.append(convv)
    noncontig = torch.randn(32, 16, device=DEV).t().requires_grad_(True)  # non-contiguous -> native
    pv.append(noncontig)
    pn = _clone(pv)
    cfg = dict(lr=2e-3, betas=(0.9, 0.999), cautious=True, gradient_centralization=True,
               momentum_dtype="float32")
    ov, on = _fused(pv, **cfg), Adakaon(pn, **cfg)
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
    ob, big, od, nat = _parts(ov)
    assert len(ob) == 5                                        # 4 tiny 2-D + the small conv
    assert len(big) == 1                                       # 256x1024 -> chunked
    assert len(od) == 1                                        # the (64,) 1-D weight -> fused 1-D path
    assert len(nat) == 1                                       # the non-contiguous 2-D stays native
    d = max((a.detach() - b.detach()).abs().max().item() for a, b in zip(pv, pn))
    assert d < 1e-5, f"max|Δp|={d:.2e}"


# ----------------------------------------------------------------- memory footprint
def test_bf16_momentum_two_bytes_per_param():
    ps = _bag([(32, 48)] * 8, torch.float32)
    for p in ps:
        p.grad = torch.randn_like(p)
    opt = _fused(ps, momentum_dtype="bfloat16")
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
    ov, on = _fused(pv, **cfg), Adakaon(pn, **cfg)
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
    opt = _fused(pv, momentum_dtype="float32")
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
    N = 4096
    # a value strictly between two bf16 representables
    lo = torch.tensor([1.0], dtype=torch.bfloat16).float().item()
    val = lo + (torch.tensor([1.0], dtype=torch.bfloat16).float().item()) * 0  # 1.0 is representable
    val = 1.0 + 0.003  # between 1.0 and the next bf16 step (~0.0078)
    acc = torch.zeros(N, device=DEV)
    K = 300
    for k in range(K):
        out = torch.empty(N, device=DEV)
        _sr_round_probe[(1,)](out, val, k + 1, N, BLOCK=N)
        acc += out
    torch.cuda.synchronize()
    mean = (acc / K).mean().item()
    assert abs(mean - val) < 5e-4, f"SR mean {mean} vs {val}"
    # every draw must be one of the two bf16 neighbours of val
    out = torch.empty(N, device=DEV)
    _sr_round_probe[(1,)](out, val, 12345, N, BLOCK=N)
    torch.cuda.synchronize()
    uniq = torch.unique(out.bfloat16()).numel()
    assert uniq <= 2


# ----------------------------------------------------------------- reusable factored primitives
def test_gradient_centralize_primitive():
    """gradient_centralize device helper == torch GC (subtract per-row fan-in mean)."""
    R, C, BR, BC = 6, 10, 8, 16

    g = torch.randn(R, C, device=DEV)
    o = torch.zeros(R, C, device=DEV)
    _gradient_centralize_probe[(1,)](g, o, R, C, BR=BR, BC=BC)
    torch.cuda.synchronize()
    assert torch.allclose(o, g - g.mean(dim=1, keepdim=True), atol=1e-5)


def test_factored_rc_primitive():
    """factored_rc device helper == native update_factored_state + factored_inv_sqrt_factors,
    and it updates the row/col EMA state in place."""
    from kaon._factored import factored_inv_sqrt_factors, update_factored_state

    R, C, BR, BC = 6, 10, 8, 16

    g = torch.randn(R, C, device=DEV)
    row = torch.zeros(R, device=DEV)
    col = torch.zeros(C, device=DEV)
    rfac = torch.zeros(R, device=DEV)
    cfac = torch.zeros(C, device=DEV)
    _factored_rc_probe[(1,)](g, row, col, rfac, cfac, R, C, 0.999, 1e-30, BR=BR, BC=BC)
    torch.cuda.synchronize()
    row_r = torch.zeros(R, device=DEV)
    col_r = torch.zeros(C, device=DEV)
    update_factored_state(g, row_r, col_r, 0.999, 1e-30)
    rf_r, cf_r = factored_inv_sqrt_factors(row_r, col_r)
    assert torch.allclose(rfac, rf_r.view(-1), atol=1e-4)
    assert torch.allclose(cfac, cf_r.view(-1), atol=1e-4)
    assert torch.allclose(row, row_r, atol=1e-5)            # row EMA updated in place
    assert torch.allclose(col, col_r, atol=1e-5)


# ----------------------------------------------------------------- fused/native unification
def test_fused_and_native_share_state_format():
    """The fused 2-D paths (one-block + chunked) keep BYTE-COMPATIBLE state with native (same keys /
    dtypes / shapes), so a run can be checkpoint-resumed across ``fused``. (1-D params go native in
    both; native foreach-batches them and the int8 codec's per-param vs stacked scale shape differs
    cosmetically there — orthogonal to fusion, so this checks the fused tensors.)"""
    shapes = [(8, 16), (1024, 512)]             # one-block + chunked
    pv = _bag(shapes, torch.float32, 3)
    pn = _clone(pv)
    of = Adakaon(pv, fused=True, momentum_dtype="int8")
    on = Adakaon(pn, momentum_dtype="int8")
    gen = torch.Generator(device=DEV).manual_seed(11)
    for _ in range(3):
        gs = [torch.randn(*p.shape, generator=gen, device=DEV) for p in pv]
        for p, g in zip(pv, gs):
            p.grad = g.clone()
        for p, g in zip(pn, gs):
            p.grad = g.clone()
        of.step()
        on.step()
    torch.cuda.synchronize()
    for a, b in zip(pv, pn):
        sa, sb = of.state[a], on.state[b]
        assert set(sa) == set(sb), f"state keys differ: {set(sa)} vs {set(sb)}"
        for k in sa:
            if torch.is_tensor(sa[k]):
                assert sa[k].dtype == sb[k].dtype and sa[k].shape == sb[k].shape, f"{k}: {sa[k].shape}"


# --------------------------------------------------- stale grad-pointer regression (real NaN)
def test_refresh_grads_revalidates_every_pointer():
    """Root cause of the 2026-06-10 real-training Nekaon NaN: ``refresh_grads`` used only
    the FIRST tensor's grad address as the staleness sentinel, so when the caching
    allocator reused tensor #0's address while moving the others (the pattern a new
    latent shape's backward produces), the kernels read freed memory as gradients.
    This reproduces the allocator pattern — param #0's grad keeps its address (same
    tensor refilled), the rest are fresh tensors and their OLD buffers are poisoned —
    and demands bit-parity with the native path."""

    def run(fused):
        torch.manual_seed(0)
        ps = [(torch.randn(64, 48, device="cuda") * 0.01).requires_grad_(True) for _ in range(4)]
        opt = Adakaon(ps, lr=1e-3, betas=(0.6, 0.999), momentum_dtype="int8", fused=fused)
        g = torch.Generator(device="cuda").manual_seed(1)
        old = [torch.randn(p.shape, generator=g, device="cuda") for p in ps]
        for p, gr in zip(ps, old, strict=True):
            p.grad = gr
        opt.step()
        new = [torch.randn(p.shape, generator=g, device="cuda") for p in ps]
        ps[0].grad.copy_(new[0])                 # SAME address, new values
        for i in (1, 2, 3):
            ps[i].grad = new[i].clone()          # NEW addresses
            old[i].fill_(float("nan"))           # poison the freed-and-reused-memory stand-in
        opt.step()
        return [p.detach().clone() for p in ps], opt

    w_fused, of = run(True)
    w_native, _ = run(False)
    for a, b in zip(w_fused, w_native, strict=True):
        assert torch.allclose(a, b, atol=1e-5), "fused path read stale grad pointers"
    for p in of.param_groups[0]["params"]:
        st = of.state[p]
        assert torch.isfinite(st["row"]).all() and torch.isfinite(st["col"]).all()
