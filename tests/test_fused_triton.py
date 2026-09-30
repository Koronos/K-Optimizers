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

import math

import pytest
import torch

import kaon
from kaon import Adakaon, Nekaon
from kaon._fused_triton import (
    HAS_TRITON,
    TILE_CAP,
    TILE_CAP_1D,
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
        # NB blocks written, NS = NB of capacity: the probe's scale buffer is sized for the
        # kernel's own block count, so the bounded store never drops one.
        requant_4bit(
            momentum, mask, idx, R, C, Chalf, packed_ptr, scale_ptr,
            NB, NB, BLK, BR, BC,
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
    for entry in opt._fused_part.values():
        o, b, d, n = entry[-4:]              # leading field is the staleness witness
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
    # tuple-form randn (identical draws to the *s varargs form) so 0-D shapes `()` work too
    return [torch.randn(s, generator=g, device=DEV, dtype=dtype).requires_grad_(True) for s in shapes]


def _clone(ps):
    return [p.detach().clone().requires_grad_(True) for p in ps]


def _run_parity(
    shapes, dtype, mdtype, *, cautious=True, gc=True, wd=0.0, steps=6, seed=1, beta1=0.9,
    momentum_4bit_block=128, cautious_wd="masked", native_foreach=True,
):
    """Step Adakaon(fused=True) and native Adakaon on identical params+grads; return max|Δp| and scale.

    ``native_foreach=False`` makes the reference the PER-PARAMETER loop (the semantics of
    record), instead of the foreach path that merely agrees with it."""
    cfg = dict(lr=2e-3, betas=(beta1, 0.999), weight_decay=wd, cautious=cautious,
               gradient_centralization=gc, momentum_dtype=mdtype,
               momentum_4bit_block=momentum_4bit_block, cautious_wd=cautious_wd)
    pv = _bag(shapes, dtype, seed)
    pn = _clone(pv)
    ov = _fused(pv, **cfg)
    on = Adakaon(pn, foreach=native_foreach, **cfg)
    gen = torch.Generator(device=DEV).manual_seed(7)
    for _ in range(steps):
        gs = [torch.randn(tuple(p.shape), generator=gen, device=DEV, dtype=dtype) for p in pv]
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


@pytest.mark.parametrize(
    "shapes",
    [
        [(8, 16)] * 8,
        [(1024,)] * 8,
        [(512, 512)] * 2,
        [(320, 320, 3, 3)] * 2,
    ],
)
def test_fused_no_momentum_parity_and_zero_state(shapes):
    """beta1=0 uses fused 2-D/1-D/chunked kernels without allocating momentum."""
    diff, scale, opt = _run_parity(
        shapes,
        torch.float32,
        "bfloat16",
        beta1=0.0,
        cautious=True,
        gc=True,
        wd=0.01,
        steps=4,
    )
    assert diff <= 2e-5 * max(scale, 1.0)
    assert all("m" not in state for state in opt.state.values())
    one_block, big, one_dim, native = _parts(opt)
    assert not native
    assert one_block or big or one_dim


def test_internal_resets_rebuild_caches_and_preserve_native_parity():
    """Internal state resets must not retain pointer arrays to discarded state."""
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
    assert TILE_CAP == 1 << 13                        # measured one_block/chunked crossover (8192)
    assert fused_eligible(torch.zeros(8, 16, device=DEV))            # tiny 2-D
    assert fused_eligible(torch.zeros(16, 320, device=DEV))          # low-rank LoRA
    assert fused_eligible(torch.zeros(16, 320, device=DEV, dtype=torch.bfloat16))
    assert fused_eligible(torch.zeros(64, 128, device=DEV))          # 8192 lanes == cap (medium)
    assert not fused_eligible(torch.zeros(128, 128, device=DEV))     # 16384 lanes -> chunked wins
    assert not fused_eligible(torch.zeros(256, 256, device=DEV))     # 65536 lanes -> chunked wins 4.5x
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
    # tiles right at the cap (8192 lanes) still take the one-block kernel (measured 1.1-1.3x over
    # chunked there) and stay exact vs native
    d, _, ov = _run_parity([(64, 128), (128, 64)], torch.float32, "float32")
    assert d < 1e-5, f"max|Δp|={d:.2e}"
    assert len(_parts(ov)[0]) == 2                                 # both one-block at the cap


def test_fp32_parity_just_above_cap_routes_chunked():
    # one lane-doubling past the cap (16384) flips to the chunked path; same math
    d, _, ov = _run_parity([(128, 128), (128, 128)], torch.float32, "float32")
    assert d < 1e-5, f"max|Δp|={d:.2e}"
    assert len(_parts(ov)[0]) == 0 and len(_parts(ov)[1]) == 2


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


# 4-BIT BOUND on the single-tensor chunked path: 1e-4. Six isolated repeats of each case
# measure 4.64e-8, but the row/col reductions still use fp32 atomics whose summation order
# changes with GPU scheduling, and in 4-bit a 5e-8 wobble can push one element across a
# quantisation bin (2.9e-5 relative observed once under a shared GPU in the full suite). The
# bound therefore tolerates a single bin flip and stays ~4x below the 3.4e-4..6.2e-4 spread the
# scale store/reload race used to produce (see
# ``test_chunked_4bit_requant_no_longer_reloads_its_own_scales``): it is what would catch that
# race coming back without flaking on atomics. ``deterministic_reductions=True`` removes the
# wobble entirely (pinned separately below).
def test_chunked_4bit_parity():
    d, scale, ov = _run_parity([(1024, 512)], torch.float32, "4bit")
    assert d / scale < 1e-4, f"rel={d/scale:.2e}"
    assert len(_parts(ov)[1]) == 1
    st = ov.state[_parts(ov)[1][0]]
    assert st["m"].dtype == torch.uint8 and st["m"].numel() == _parts(ov)[1][0].numel() // 2  # 0.5 B/param


def test_chunked_4bit_odd_C():
    # the chunked codec packs the flat tensor, so 4bit handles odd C (unlike the one-block path)
    d, scale, ov = _run_parity([(1024, 513)], torch.float32, "4bit")
    assert d / scale < 1e-4, f"rel={d/scale:.2e}"
    assert len(_parts(ov)[1]) == 1


@pytest.mark.parametrize("mdtype", ["int8", "4bit"])
def test_chunked_quant_features(mdtype):
    d, scale, _ = _run_parity([(1024, 512)], torch.float32, mdtype, wd=0.05, cautious=True, gc=False)
    limit = 1e-4 if mdtype == "4bit" else 5e-4      # see the 4-bit bound note above
    assert d / scale < limit, f"{mdtype} rel={d/scale:.2e}"


# ------------------------------------------- batched chunked (>=2 same-shape big tensors) parity
# The Cosmos LoKr regime: many same-shape factors > tile_cap. Every big shape bucket takes the
# batched chunked kernel (~2 launches for the whole bucket), N == 1 included since 0.7.12 — see
# ``_fused_big_lone_batched`` and the lone-big tests below. Both must match native exactly (fp32) /
# within the dtype bound. 512x512 > cap.
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
    # Triton's segmented max reduction differs from torch.amax by a few fp32
    # ulps; near half-grid values can therefore choose an adjacent 4-bit code
    # and diverge slightly over repeated EMA steps. The bound remains below 0.1%.
    limit = 8e-4 if mdtype == "4bit" else 5e-4
    assert d / scale < limit, f"{mdtype} rel={d/scale:.2e}"


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
# one per program. All momentum codecs are updated inside the same kernel.
def test_one_dim_eligibility_predicate():
    assert TILE_CAP_1D == 1 << 17                     # 1-D one-program bound (131072), not the 2-D cap
    assert fused_1d_eligible(torch.zeros(1024, device=DEV))
    assert fused_1d_eligible(torch.zeros(2048, device=DEV, dtype=torch.bfloat16))
    assert not fused_1d_eligible(torch.zeros(8, 16, device=DEV))             # 2-D -> not the 1-D path
    assert fused_1d_eligible(torch.zeros(TILE_CAP_1D, device=DEV))           # exactly at the 1-D cap
    assert not fused_1d_eligible(torch.zeros(TILE_CAP_1D * 2, device=DEV))   # too big for one block


def test_one_dim_cap_is_independent_of_the_two_dim_crossover():
    """The 2-D cap is an occupancy crossover (over it, the CHUNKED kernel is faster); the 1-D cap
    is a one-program capability bound (over it there is only NATIVE, which is slower). Coupling
    them cost 2x on 1-D tensors of 16384 lanes, so the asymmetry is pinned here: at the SAME
    16384 padded lanes a 1-D tensor must stay fused while a 2-D tensor must go chunked."""
    assert TILE_CAP < TILE_CAP_1D
    lanes = 1 << 14
    assert fused_1d_eligible(torch.zeros(lanes, device=DEV))                 # 1-D 16384 -> fused
    assert not fused_eligible(torch.zeros(128, 128, device=DEV))             # 2-D 16384 -> chunked
    # and the 2-D knob must not drag the 1-D ceiling down with it
    assert fused_1d_eligible(torch.zeros(lanes, device=DEV), TILE_CAP_1D)


def test_one_dim_above_two_dim_cap_still_routes_fused():
    """End-to-end counterpart of the predicate asymmetry: a bag of 16384-long 1-D params is
    2x faster fused than native, so it must land in ``one_dim`` even though the same padded
    lane count sends a 2-D weight to the chunked path."""
    d, _, ov = _run_parity([(1 << 14,)] * 3, torch.float32, "float32")
    ob, big, od, nat = _parts(ov)
    assert len(od) == 3 and len(nat) == 0 and len(big) == 0
    assert d < 1e-5, f"max|Δp|={d:.2e}"


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
def test_one_dim_quant_routes_to_fused(mdtype):
    d, scale, ov = _run_parity([(1024,)] * 3, torch.float32, mdtype)
    ob, big, od, nat = _parts(ov)
    assert len(od) == 3 and len(nat) == 0
    assert d / scale < 5e-4, f"{mdtype} rel={d/scale:.2e}"


@pytest.mark.parametrize("mdtype", ["int8", "4bit"])
def test_one_dim_quant_small_odd_lengths(mdtype):
    # Exercises scalar packing and different codec block sizes, including two
    # lengths that share the same padded Triton block.
    d, scale, ov = _run_parity([(1,), (7,), (65,), (100,)], torch.float32, mdtype)
    assert len(_parts(ov)[2]) == 4
    assert d / scale < 5e-4, f"{mdtype} rel={d/scale:.2e}"
    if mdtype == "4bit":
        for p in _parts(ov)[2]:
            if p.numel() % 2:
                assert int(ov.state[p]["m"][-1] >> 4) == 0


def test_one_dim_mixed_with_2d():
    # a realistic mix: 2-D one-block + 1-D fused at once, all parity vs native
    d, _, ov = _run_parity([(8, 16), (16, 16), (1024,), (256,)], torch.float32, "float32")
    ob, big, od, nat = _parts(ov)
    assert len(ob) == 2 and len(od) == 2 and len(nat) == 0
    assert d < 1e-5, f"max|Δp|={d:.2e}"


def test_pointer_buckets_separate_parameter_dtypes():
    # Mixed precision in one param group is common when norms/embeddings stay
    # fp32. LOWP is a constexpr, so sharing a bucket would reinterpret pointers.
    fp = _bag([(8, 16), (100,)], torch.float32, seed=35)
    bf = _bag([(8, 16), (100,)], torch.bfloat16, seed=36)
    for p in fp + bf:
        p.grad = torch.randn_like(p)
    opt = _fused(fp + bf, lr=1e-3, momentum_dtype="bfloat16")
    opt.step()
    ob = [bk for cache in opt._fused_ob_caches.values() for bk in cache.buckets]
    od = [bk for cache in opt._fused_od_caches.values() for bk in cache.buckets]
    assert {bk["lowp"] for bk in ob} == {False, True}
    assert {bk["lowp"] for bk in od} == {False, True}


@pytest.mark.parametrize("mdtype", ["bfloat16", "int8", "4bit"])
def test_nekaon_fused_mixed_routes_match_native(mdtype):
    shapes = [(8, 16)] * 4 + [(100,)] * 3 + [(16, 8, 3, 3)] * 2
    pv = _bag(shapes, torch.float32, seed=41)
    pn = _clone(pv)
    cfg = dict(
        lr=1e-3, k=1.5, betas=(0.5, 0.999), momentum_dtype=mdtype,
        cautious=True, gradient_centralization=True, weight_decay=0.03,
    )
    ov, on = Nekaon(pv, fused=True, **cfg), Nekaon(pn, fused=False, **cfg)
    gen = torch.Generator(device=DEV).manual_seed(42)
    for _ in range(5):
        grads = [torch.randn(p.shape, generator=gen, device=DEV) for p in pv]
        for a, b, g in zip(pv, pn, grads, strict=True):
            a.grad = g.clone()
            b.grad = g.clone()
        ov.step()
        on.step()
    torch.cuda.synchronize()
    d = max((a - b).abs().max().item() for a, b in zip(pv, pn, strict=True))
    scale = max(p.abs().max().item() for p in pn)
    assert d / scale < (6e-3 if mdtype == "bfloat16" else 8e-4), f"{mdtype} rel={d/scale:.2e}"


def test_nekaon_fused_cache_admits_late_gradient():
    pv = _bag([(8, 16), (100,)], torch.float32, seed=51)
    pn = _clone(pv)
    cfg = dict(
        lr=1e-3, k=1.5, betas=(0.5, 0.999), momentum_dtype="4bit",
        cautious=False, gradient_centralization=False,
    )
    ov, on = Nekaon(pv, fused=True, **cfg), Nekaon(pn, fused=False, **cfg)
    for ps in (pv, pn):
        ps[0].grad = torch.randn_like(ps[0])
        ps[1].grad = None
    on_grad = pv[0].grad.clone()
    pn[0].grad = on_grad
    ov.step()
    on.step()
    # State and all pointer/bucket caches were already built without p[1]. Its
    # first gradient must invalidate and rebuild the complete MSAM dispatch plan.
    for ps in (pv, pn):
        ps[0].grad = torch.randn_like(ps[0])
        ps[1].grad = torch.randn_like(ps[1])
    pn[0].grad.copy_(pv[0].grad)
    pn[1].grad.copy_(pv[1].grad)
    ov.step()
    on.step()
    assert len(ov._momentum_params()) == 2
    assert sum(bk["N"] for bk in ov._axpy_cache["buckets"]) == 2
    d = max((a - b).abs().max().item() for a, b in zip(pv, pn, strict=True))
    assert d < 2e-3, f"late-gradient fused/native max|Δp|={d:.2e}"


def test_one_dim_mixed_dtype_buckets_split():
    """fp32 and bf16 1-D params of the SAME length must land in SEPARATE kernel
    buckets: `lowp` (the kernel's pointer type) is per-bucket, so a shared bucket
    would read half the bag through the wrong pointer type."""
    cfg = dict(lr=2e-3, betas=(0.9, 0.999), momentum_dtype="float32")
    pv = _bag([(512,)] * 2, torch.float32, seed=3) + _bag([(512,)] * 2, torch.bfloat16, seed=4)
    pn = _clone(pv)
    ov = _fused(pv, **cfg)
    on = Adakaon(pn, **cfg)
    gen = torch.Generator(device=DEV).manual_seed(9)
    for _ in range(6):
        for p, q in zip(pv, pn):
            g = torch.randn(tuple(p.shape), generator=gen, device=DEV, dtype=p.dtype)
            p.grad, q.grad = g.clone(), g.clone()
        ov.step()
        on.step()
    torch.cuda.synchronize()
    buckets = [bk for cache in ov._fused_od_caches.values() for bk in cache.buckets]
    assert len(buckets) == 2, f"expected fp32/bf16 split, got {len(buckets)} bucket(s)"
    for a, b in zip(pv, pn):
        rel = (a.detach().float() - b.detach().float()).abs().max().item()
        scale = max(b.detach().float().abs().max().item(), 1e-3)
        assert rel / scale < 5e-2, f"dtype={a.dtype} rel={rel/scale:.2e}"


# ------------------------------------------------- 0-D scalars (LyCORIS use_scalar gates)
# A 0-D param has a valid base pointer and numel()==1, which is all the shape-free 1-D kernel
# needs — it rides the one_dim path as a length-1 tensor. Per-param dispatch for a bag of
# scalars is ~22 CUDA launches per scalar per step (measured 592x slower than shape-(1,)).
def test_zero_dim_routes_to_one_dim_and_parity_fp32():
    d, _, ov = _run_parity([()] * 6, torch.float32, "float32")
    ob, big, od, nat = _parts(ov)
    assert len(od) == 6 and len(ob) == 0 and len(big) == 0 and len(nat) == 0
    assert d < 1e-5, f"max|Δp|={d:.2e}"


def test_zero_dim_mixed_with_1d_and_2d():
    # the real LoKr layout: adapter matrices + per-module 0-D scalar gates
    d, _, ov = _run_parity([(8, 16), (16, 16), (), (), (1,), (256,)], torch.float32, "float32")
    ob, big, od, nat = _parts(ov)
    assert len(ob) == 2 and len(od) == 4 and len(nat) == 0
    assert d < 1e-5, f"max|Δp|={d:.2e}"


@pytest.mark.parametrize("cautious", [True, False])
def test_zero_dim_features(cautious):
    d, _, _ = _run_parity([()] * 4, torch.float32, "float32", cautious=cautious, wd=0.05)
    assert d < 1e-5, f"cautious={cautious} max|Δp|={d:.2e}"


def test_zero_dim_bf16_momentum():
    d, scale, _ = _run_parity([()] * 4, torch.float32, "bfloat16")
    assert d / scale < 5e-3, f"rel={d/scale:.2e}"


def test_zero_dim_bf16_params_sr():
    d, scale, _ = _run_parity([()] * 4, torch.bfloat16, "bfloat16")
    assert d / scale < 5e-2, f"rel={d/scale:.2e}"


@pytest.mark.parametrize("mdtype", ["int8", "4bit"])
def test_zero_dim_quant_routes_to_one_dim(mdtype):
    # The 1-D kernel carries int8/4bit momentum (scalar scale / one 2-lane nibble block),
    # so a 0-D scalar rides it under quant momentum exactly like a shape-(1,) param does.
    d, scale, ov = _run_parity([()] * 4, torch.float32, mdtype)
    ob, big, od, nat = _parts(ov)
    assert len(od) == 4 and len(nat) == 0
    assert d / scale < 5e-4, f"{mdtype} rel={d/scale:.2e}"


@pytest.mark.parametrize("mdtype", ["int8", "4bit"])
def test_zero_dim_quant_buckets_apart_from_one_dim(mdtype):
    # A 0-D scalar and a shape-(1,) param share BL == next_pow2(numel); the mixed bag must
    # still match native, and the 2-D weight must stay on its own (one-block) route.
    d, scale, ov = _run_parity([(), (), (1,), (256,), (8, 16)], torch.float32, mdtype)
    ob, big, od, nat = _parts(ov)
    assert len(ob) == 1 and len(od) == 4 and len(nat) == 0
    assert d / scale < 5e-4, f"{mdtype} rel={d/scale:.2e}"


def test_zero_dim_no_momentum_routes_to_one_dim():
    # beta1==0 (no ``m``) is a supported 1-D kernel mode — it reuses ``v_addr`` as a
    # harmless valid pointer — so the scalars stay on the fused route and stay bit-exact.
    cfg = dict(lr=2e-3, betas=(0.0, 0.999), cautious=False, momentum_dtype="float32",
               weight_decay=0.0, gradient_centralization=True)
    pv = _bag([()] * 4, torch.float32, seed=2)
    pn = _clone(pv)
    ov = _fused(pv, **cfg)
    on = Adakaon(pn, foreach=False, **cfg)
    gen = torch.Generator(device=DEV).manual_seed(7)
    for _ in range(6):
        for p, q in zip(pv, pn):
            g = torch.randn((), generator=gen, device=DEV)
            p.grad, q.grad = g.clone(), g.clone()
        ov.step()
        on.step()
    torch.cuda.synchronize()
    ob_, big, od, nat = _parts(ov)
    assert len(od) == 4 and len(nat) == 0
    for a, b in zip(pv, pn):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


# ------------------------------------------------- conv (ndim>2) matrixized to (out, in*kh*kw)
# A contiguous conv's row-major storage IS its (out, in*kh*kw) view, so it rides the 2-D fused paths
# (one-block when small, chunked when big) with no copy for every momentum codec.
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
def test_conv_quant_routes_to_fused(mdtype):
    d, scale, ov = _run_parity([(16, 8, 3, 3)] * 4, torch.float32, mdtype)
    ob, big, od, nat = _parts(ov)
    assert len(ob) == 4 and len(nat) == 0 and len(big) == 0
    assert d / scale < 5e-4, f"{mdtype} rel={d/scale:.2e}"


@pytest.mark.parametrize("mdtype", ["int8", "4bit"])
def test_conv_big_quant_routes_to_fused(mdtype):
    d, scale, ov = _run_parity(
        [(256, 128, 3, 3)] * 2, torch.float32, mdtype, steps=3
    )
    ob, big, od, nat = _parts(ov)
    assert len(big) == 2 and len(ob) == 0 and len(nat) == 0
    assert d / scale < 8e-4, f"{mdtype} rel={d/scale:.2e}"


@pytest.mark.parametrize("block", [64, 128, 96])
def test_big_4bit_partial_chunk_and_block_parity(block):
    # n=257*513 is odd and ends in a partial 1024-element chunk. 64/128 use
    # direct in-kernel requantization; 96 deliberately exercises the compatible
    # fp32-temp fallback because its codec blocks cross chunk boundaries.
    d, scale, ov = _run_parity(
        [(257, 513)] * 2, torch.float32, "4bit", steps=3,
        momentum_4bit_block=block,
    )
    assert len(_parts(ov)[1]) == 2
    assert d / scale < 8e-4, f"block={block} rel={d/scale:.2e}"
    for p in _parts(ov)[1]:
        assert p.numel() % 2 == 1
        assert int(ov.state[p]["m"][-1] >> 4) == 0


# ------------------------------------------------- lone big tensor -> the batched (N=1) kernel
# 0.7.12 routes a big shape bucket of ONE tensor through ``_chunked_step_batched`` too. The
# per-tensor ``_chunked_step`` it replaces is correct but blocks the CPU twice per tensor per step
# (``float(rms)``, ``keep.item()``) and materializes fp32 ``g`` + ``g*g``; on a bag of DISTINCT big
# shapes (UNet/DiT, one tensor per bucket) that measured 3.2-3.6x slower and +12 MiB of transient.
def test_lone_big_batched_is_sync_free_and_per_tensor_is_not():
    """The routing change's WHY: the per-tensor arm synchronizes, the batched arm does not.

    SCOPE: this holds for the step's OWN synchronizations — ``float(rms)`` and ``keep.item()``
    in ``_chunked_step``. The grads below are attached once and reused, so ``refresh_grads``
    finds the same pointers and does not rebuild. It is NOT a claim that a training step is
    sync-free: with the default ``zero_grad(set_to_none=True)`` every backward allocates fresh
    gradients, ``refresh_grads`` then rebuilds the pointer array with a pageable host->device
    copy, and that copy synchronizes on both arms. Measured at 4.4-4.8 us per bucket; a pinned
    staging buffer would remove it but needs a CUDA event to stop the next step overwriting a
    copy still in flight, so it is left as follow-up (see the CHANGELOG caveat).
    """
    def one_step(lone):
        ps = _bag([(1024, 1024)], torch.float32, seed=71)
        for p in ps:
            p.grad = torch.randn_like(p)
        opt = _fused(ps, lr=1e-3)
        opt._fused_big_lone_batched = lone
        opt.step()                      # warm: JIT + state alloc + pointer caches
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            opt.step()
            torch.cuda.synchronize()
            return None
        except RuntimeError as exc:
            return str(exc)
        finally:
            torch.cuda.set_sync_debug_mode("default")

    assert one_step(True) is None, "batched lone-big step must be fully device-resident"
    assert one_step(False) is not None, "per-tensor chunked step is expected to sync (the baseline)"


def _lone_big_arms(mdtype, dtype=torch.float32, steps=6):
    """max|Δp|/scale of (batched N=1, per-tensor chunked) vs NATIVE, plus the two arms' own gap."""
    cfg = dict(lr=2e-3, weight_decay=0.05, cautious=True, gradient_centralization=True,
               momentum_dtype=mdtype)
    pb = _bag([(1024, 512)], dtype, seed=72)
    pt, pn = _clone(pb), _clone(pb)
    ob = _fused(pb, **cfg)
    ot = _fused(pt, **cfg)
    ot._fused_big_lone_batched = False
    on = Adakaon(pn, **cfg)
    gen = torch.Generator(device=DEV).manual_seed(13)
    for _ in range(steps):
        gs = [torch.randn(tuple(p.shape), generator=gen, device=DEV, dtype=dtype) for p in pb]
        for ps in (pb, pt, pn):
            for p, g in zip(ps, gs):
                p.grad = g.clone()
        ob.step()
        ot.step()
        on.step()
    torch.cuda.synchronize()

    def gap(xs, ys):
        return max((a.detach().float() - b.detach().float()).abs().max().item()
                   for a, b in zip(xs, ys))

    scale = max(p.detach().float().abs().max().item() for p in pn)
    return gap(pb, pn) / scale, gap(pt, pn) / scale, gap(pb, pt) / scale


@pytest.mark.parametrize("mdtype", ["float32", "bfloat16"])
def test_lone_big_batched_no_worse_than_per_tensor_vs_native(mdtype):
    """Float momenta: routing N=1 to the batched kernel does not degrade fidelity vs native.

    NOT bit-identical to the per-tensor kernel, and deliberately not asserted to be: the
    per-tensor arm computes its reductions in TORCH (``_chunked_reductions``: ``gsq.mean`` +
    ``bmm`` matvec) while the batched arm uses the Triton reduction kernels (atomic fp32
    accumulation over row blocks). Different summation orders, so the two differ by fp32 ulps —
    the same pre-existing deviation the ``_fused_reductions`` toggle carries for N>=2. What must
    hold is that the batched arm is no FURTHER from native than the per-tensor arm was, plus a
    ulp allowance. Measured (1024,512) fp32: both 4.6e-8 (float32 momentum) / 3.3e-6 (bf16),
    with the two arms 4.6e-8 apart.
    """
    batched, per_tensor, between = _lone_big_arms(mdtype)
    assert between < 1e-5, f"arms {between:.2e} apart — more than a reduction-order ulp"
    assert batched <= max(per_tensor * 1.5, 1e-5), \
        f"batched rel={batched:.2e} vs per-tensor rel={per_tensor:.2e}"


def test_lone_big_4bit_keeps_the_per_tensor_fidelity():
    """4-bit costs NOTHING in fidelity for routing a lone big tensor to the batched kernel.

    It was expected to: the per-tensor path requantizes with the torch codec (``_quant_4bit``)
    while the batched path requantizes in kernel, and the first measurement put the batched arm
    at ~4e-4 relative to native against the per-tensor arm's 4.6e-8. That gap was the scale
    store/reload race, not the kernel's arithmetic (see
    ``test_chunked_4bit_requant_no_longer_reloads_its_own_scales``); with it fixed both arms sit
    at 4.6e-8 — the fp32 reduction-order noise every momentum dtype carries — over six repeats.
    Asserted for BOTH arms so a regression in either is caught here.
    """
    batched, per_tensor, between = _lone_big_arms("4bit")
    # 1e-4: one 4-bit bin flip from atomic ordering is tolerated, the old race is not.
    assert per_tensor < 1e-4, f"per-tensor arm rel={per_tensor:.2e}"
    assert batched < 1e-4, f"batched arm rel={batched:.2e}"
    assert between < 1e-4, f"arms {between:.2e} apart"


def test_big_4bit_direct_path_never_dequantizes_to_stacked_temp(monkeypatch):
    ps = _bag([(512, 512)] * 2, torch.float32, seed=61)
    for p in ps:
        p.grad = torch.randn_like(p)
    opt = _fused(ps, lr=1e-3, momentum_dtype="4bit", momentum_4bit_block=128)
    codec = opt._codec(opt.param_groups[0])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("direct chunked 4-bit path allocated the legacy fp32 stack")

    monkeypatch.setattr(codec, "dequant_stacked", forbidden)
    opt.step()


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


# ``cautious_wd="full"`` moves the decay OUTSIDE the cautious mask. It has to be wired into
# every kernel that folds wd into the delta — the tile kernel, the 1-D kernel, the per-tensor
# chunked pair, the batched chunked pair (both the stacked-grad and the pointer-array ``_g``
# variants) and the three direct-momentum chunked pairs (no-momentum, 4-bit, int8). Two
# complementary guards, because ONE of them is not enough:
#
# * a PARITY check against the per-parameter loop, at a threshold ~10x each dtype's measured
#   fused-vs-per-param floor (fp32 6.0e-8, bf16 2.6e-6, int8 5.3e-6, 4bit 2.7e-5 relative);
# * a SEMANTIC check, below, that measures the decay each coordinate actually receives.
#
# The parity check alone was demonstrably not enough: with a blanket 5e-4 bound, a kernel that
# ignores ``WDFULL`` (i.e. collapses ``if WD and not WDFULL`` back to ``if WD``) only moves the
# weights by ~1e-4 relative, so "full" would degrade silently to "masked" in production with the
# whole suite green. The semantic check separates the two placements by four orders of magnitude.
_WD_ROUTES = [
    pytest.param([(8, 16)] * 4, 0.9, "one_block", id="one_block"),
    pytest.param([(1024,)] * 4, 0.9, "one_dim", id="one_dim"),
    pytest.param([()] * 4, 0.9, "one_dim", id="zero_dim"),
    pytest.param([(512, 512)] * 3, 0.9, "big", id="big_batched"),
    pytest.param([(512, 512)], 0.9, "big", id="big_lone"),
    pytest.param([(512, 512)] * 2, 0.0, "big", id="big_nomom"),
]

# ~10x the worst measured fused-vs-per-param relative floor for that momentum storage, across
# every route in ``_WD_ROUTES`` and both placements. NOT a blanket number: a bound loose enough
# for 4-bit would be blind to a wiring bug on the fp32 routes.
_WD_PARITY_TOL = {"float32": 1e-6, "bfloat16": 3e-5, "int8": 6e-5, "4bit": 3e-4}


@pytest.mark.parametrize("shapes,beta1,route", _WD_ROUTES)
@pytest.mark.parametrize("mdtype", ["float32", "bfloat16", "int8", "4bit"])
@pytest.mark.parametrize("arm", ["masked", "full"])
def test_cautious_wd_parity_with_the_per_param_loop(shapes, beta1, route, mdtype, arm):
    """Both placements must match the PER-PARAMETER step on every fused route."""
    d, scale, opt = _run_parity(shapes, torch.float32, mdtype, wd=0.05, beta1=beta1,
                                cautious=True, cautious_wd=arm, steps=4, native_foreach=False)
    one_block, big, one_dim, native = _parts(opt)
    got = {"one_block": one_block, "big": big, "one_dim": one_dim}
    assert got[route] and not native, f"expected the {route} route, got native={len(native)}"
    tol = _WD_PARITY_TOL[mdtype]
    assert d / max(scale, 1.0) < tol, f"{mdtype}/{route}/{arm} rel={d/scale:.2e} (tol {tol:.0e})"


# ----------------------------------------------------------------- the semantic guard
# WHAT IT MEASURES. Under ``cautious_wd="full"`` the mask is computed on the bare momentum, so
# it does not depend on ``weight_decay`` at all: running the SAME step twice, once with wd and
# once with wd=0, must differ by EXACTLY ``lr*wd*p`` on every coordinate — including the ones
# the mask rejects. That identity IS the definition of the mode, and it is what a kernel
# ignoring ``WDFULL`` breaks: a rejected coordinate then receives no decay, so the deviation
# from the identity jumps from ~6e-5 to ~0.8-1.0 (a rejected coordinate's whole ``lr*wd*p``).
# ``"masked"`` violates the identity BY CONSTRUCTION and is asserted to do so — that contrast
# is what proves the probe measures the placement and not something else.
#
# ``deterministic_reductions=True`` so the two runs share a bit-identical prefix (the big
# route's fp32-atomic reductions are otherwise run-to-run nondeterministic); the helper
# asserts that rather than assuming it.
_WD_SEM_LR, _WD_SEM_WD, _WD_SEM_STEPS = 1e-2, 0.1, 3


def _wd_identity_deviation(shapes, beta1, mdtype, arm, toggles=None):
    """max |(p_no_wd - p_wd) - lr*wd*p_before| / (lr*wd*max|p_before|) for one fused config."""
    runs = []
    for last_wd in (_WD_SEM_WD, 0.0):
        ps = _bag(shapes, torch.float32, seed=101)
        opt = _fused(ps, lr=_WD_SEM_LR, betas=(beta1, 0.999), weight_decay=_WD_SEM_WD,
                     cautious=True, momentum_dtype=mdtype, cautious_wd=arm,
                     deterministic_reductions=True)
        for k, v in (toggles or {}).items():
            setattr(opt, k, v)
        gen = torch.Generator(device=DEV).manual_seed(103)
        grads = [[torch.randn(tuple(p.shape), generator=gen, device=DEV) for p in ps]
                 for _ in range(_WD_SEM_STEPS)]
        for gs in grads[:-1]:
            for p, g in zip(ps, gs, strict=True):
                p.grad = g.clone()
            opt.step()
        torch.cuda.synchronize()
        before = [p.detach().clone() for p in ps]
        opt.param_groups[0]["weight_decay"] = last_wd
        for p, g in zip(ps, grads[-1], strict=True):
            p.grad = g.clone()
        opt.step()
        torch.cuda.synchronize()
        runs.append((before, [p.detach().clone() for p in ps]))
    (before, with_wd), (before2, no_wd) = runs
    assert all(torch.equal(a, b) for a, b in zip(before, before2, strict=True)), (
        "the two runs' shared prefix diverged — the probe would compare different states"
    )
    step = _WD_SEM_LR * _WD_SEM_WD
    dev = max(((n - w) - step * b).abs().max().item()
              for w, n, b in zip(with_wd, no_wd, before, strict=True))
    return dev / (step * max(b.abs().max().item() for b in before))


@pytest.mark.parametrize("shapes,beta1,route", _WD_ROUTES)
@pytest.mark.parametrize("mdtype", ["float32", "bfloat16", "int8", "4bit"])
def test_cautious_wd_full_decays_rejected_coordinates_on_every_route(shapes, beta1, route, mdtype):
    """"full" gives EVERY coordinate its lr*wd*p; "masked" demonstrably does not."""
    full = _wd_identity_deviation(shapes, beta1, mdtype, "full")
    assert full < 5e-3, (
        f"{mdtype}/{route}: cautious_wd='full' withheld the decay from some coordinate "
        f"(deviation from the identity = {full:.2e}); a kernel is ignoring WDFULL"
    )
    masked = _wd_identity_deviation(shapes, beta1, mdtype, "masked")
    assert masked > 0.5, (
        f"{mdtype}/{route}: the probe is not sensitive here — 'masked' should violate the "
        f"identity by ~1.0, measured {masked:.2e}"
    )


# The batched big route has kernel pairs behind internal A/B toggles that the default config
# never reaches (``_chunked_{mom,apply}_batched`` without the fused reductions, the per-tensor
# ``_chunked_{mom,apply}``, and the int8 codec fallback). ``cautious_wd`` had to be threaded
# through all of them too, so each gets BOTH guards.
_WD_TOGGLES = [
    pytest.param({"_fused_reductions": False}, "float32", [(512, 512)] * 2, id="chunked_batched"),
    pytest.param({"_fused_big_lone_batched": False}, "float32", [(512, 512)], id="per_tensor_fp32"),
    pytest.param({"_fused_big_lone_batched": False}, "4bit", [(512, 512)], id="per_tensor_4bit"),
    pytest.param({"_direct_int8": False}, "int8", [(512, 512)] * 2, id="int8_codec_fallback"),
]


@pytest.mark.parametrize("toggles,mdtype,shapes", _WD_TOGGLES)
@pytest.mark.parametrize("arm", ["masked", "full"])
def test_cautious_wd_parity_on_the_toggled_big_kernels(toggles, mdtype, shapes, arm):
    cfg = dict(lr=2e-3, betas=(0.9, 0.999), weight_decay=0.05, cautious=True,
               gradient_centralization=True, momentum_dtype=mdtype, cautious_wd=arm)
    pv = _bag(shapes, torch.float32, seed=17)
    pn = _clone(pv)
    ov, on = _fused(pv, **cfg), Adakaon(pn, foreach=False, **cfg)
    for k, v in toggles.items():
        setattr(ov, k, v)
    gen = torch.Generator(device=DEV).manual_seed(19)
    for _ in range(4):
        gs = [torch.randn(tuple(p.shape), generator=gen, device=DEV) for p in pv]
        for p, g in zip(pv, gs, strict=True):
            p.grad = g.clone()
        for p, g in zip(pn, gs, strict=True):
            p.grad = g.clone()
        ov.step()
        on.step()
    torch.cuda.synchronize()
    assert _parts(ov)[1], "these shapes are meant to route to the big (chunked) path"
    d = max((a.detach() - b.detach()).abs().max().item() for a, b in zip(pv, pn, strict=True))
    scale = max(b.detach().abs().max().item() for b in pn)
    tol = _WD_PARITY_TOL[mdtype]
    assert d / scale < tol, f"{toggles}/{mdtype}/{arm} rel={d/scale:.2e} (tol {tol:.0e})"


@pytest.mark.parametrize("toggles,mdtype,shapes", _WD_TOGGLES)
def test_cautious_wd_full_decays_rejected_coordinates_on_the_toggled_kernels(
    toggles, mdtype, shapes
):
    full = _wd_identity_deviation(shapes, 0.9, mdtype, "full", toggles)
    assert full < 5e-3, (
        f"{toggles}/{mdtype}: cautious_wd='full' withheld the decay from some coordinate "
        f"(deviation = {full:.2e}); a kernel is ignoring WDFULL"
    )
    masked = _wd_identity_deviation(shapes, 0.9, mdtype, "masked", toggles)
    assert masked > 0.5, f"{toggles}/{mdtype}: probe not sensitive, masked deviation {masked:.2e}"


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


# ================================================= 0.7.12 batched-big performance batch
# Direct in-kernel int8 momentum, the packed reduction scratch, and the Triton bf16
# stochastic-rounding write. Each has a mechanism assertion (the thing that makes it fast)
# next to a parity assertion (the thing that must not change).

# ------------------------------------------------------------------- direct int8 momentum
def test_big_int8_direct_path_never_dequantizes_to_stacked_temp(monkeypatch):
    """C=512 divides BLOCK=1024, so the bucket requantizes in kernel - no codec, no fp32 temp."""
    ps = _bag([(512, 512)] * 2, torch.float32, seed=81)
    for p in ps:
        p.grad = torch.randn_like(p)
    opt = _fused(ps, lr=1e-3, momentum_dtype="int8")
    codec = opt._codec(opt.param_groups[0])

    def forbidden(*_a, **_kw):
        raise AssertionError("direct chunked int8 path allocated the legacy fp32 stack")

    monkeypatch.setattr(codec, "dequant_stacked", forbidden)
    opt.step()


def _spy_codec_fallback(monkeypatch, opt):
    """Record every ``dequant_stacked`` call — i.e. every big bucket that took the codec
    fallback (dequant -> fp32 temp -> requant) instead of an in-kernel int8 route."""
    codec = opt._codec(opt.param_groups[0])
    seen = []
    real = codec.dequant_stacked

    def spy(*a, **k):
        seen.append(1)
        return real(*a, **k)

    monkeypatch.setattr(codec, "dequant_stacked", spy)
    return seen


def test_big_int8_direct_path_declines_when_rows_are_too_narrow(monkeypatch):
    """C=100 neither divides the 1024-element chunk nor reaches ``INT8_ROWS_MIN_C``: a chunk
    would touch ~12 rows, one segmented max each. The routing must keep the codec fallback;
    this asserts the guard is REACHED."""
    import kaon._fused_triton as ft
    assert ft.int8_route(100) == "codec"
    ps = _bag([(256, 100)] * 2, torch.float32, seed=82)
    for p in ps:
        p.grad = torch.randn_like(p)
    opt = _fused(ps, lr=1e-3, momentum_dtype="int8")
    seen = _spy_codec_fallback(monkeypatch, opt)
    opt.step()
    assert len(_parts(opt)[1]) == 2
    assert seen, "a bucket whose rows are too narrow for the row route must keep the codec"


_INT8_ROWS = [
    [(256, 128, 3, 3)] * 2,          # matrixized 3x3 conv, C = 1152 (a row spans two chunks)
    [(64, 4608)] * 2,                # DiT fc2, C = 4608
    [(96, 1280), (96, 1280)],        # C = 1280, R not a power of two
    [(40, 1536)],                    # a LONE big tensor (N == 1)
    [(300, 320)] * 2,                # C = 320 < BLOCK, 1024 % 320 != 0: rows straddle chunks
]


@pytest.mark.parametrize("shapes", _INT8_ROWS, ids=lambda s: "x".join(map(str, s[0])))
def test_big_int8_rows_route_skips_the_codec_fallback(monkeypatch, shapes):
    """Rows that SPAN chunks (C > 1024, or 1024 % C != 0) take the two-pass in-kernel route:
    no ``dequant_stacked`` (no fp32 [N,R,C] temp) on any step."""
    import kaon._fused_triton as ft
    R, C = shapes[0][0], math.prod(shapes[0][1:])  # noqa: N806
    assert ft.int8_route(C) == "rows"
    ps = _bag(shapes, torch.float32, seed=86)
    opt = _fused(ps, lr=1e-3, momentum_dtype="int8")
    seen = _spy_codec_fallback(monkeypatch, opt)
    for _ in range(2):
        for p in ps:
            p.grad = torch.randn_like(p)
        opt.step()
    assert len(_parts(opt)[1]) == len(shapes)
    assert not seen, "a row-spanning int8 bucket fell back to the codec"
    cache = next(iter(opt._fused_big_caches.values()))
    assert cache.rowmax is not None and cache.rowmax.numel() == len(shapes) * R
    zs = cache._zeros.untyped_storage().data_ptr()
    assert cache.rowmax.untyped_storage().data_ptr() == zs, "rowmax must share the one zero_()"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("cautious", [True, False])
@pytest.mark.parametrize("shapes", _INT8_ROWS, ids=lambda s: "x".join(map(str, s[0])))
def test_big_int8_rows_route_equals_codec_fallback(shapes, cautious, dtype):
    """The row route and the codec fallback are two implementations of one codec. Measured
    (deterministic reductions on both, so only the codec differs): the FIRST step is
    bit-identical in the weights, and over 30 steps the drift stays at fp32-ulp level —
    max 6.7e-5 relative in fp32, and in bf16 0 except where an ulp-level momentum difference
    flips one SR draw (7.9e-4, one bf16 ulp of a small weight); row scales identical, codes
    within one rounding tie."""
    cfg = dict(lr=2e-3, weight_decay=0.05, cautious=cautious, gradient_centralization=True,
               momentum_dtype="int8", deterministic_reductions=True)
    pd = _bag(shapes, dtype, seed=87)
    pc = _clone(pd)
    od = _fused(pd, **cfg)
    oc = _fused(pc, **cfg)
    oc._direct_int8 = False
    gen = torch.Generator(device=DEV).manual_seed(19)
    for step in range(30):
        gs = [torch.randn(tuple(p.shape), generator=gen, device=DEV, dtype=dtype) for p in pd]
        for ps in (pd, pc):
            for p, g in zip(ps, gs, strict=True):
                p.grad = g.clone()
        od.step()
        oc.step()
        if step == 0:
            assert all(torch.equal(a, b) for a, b in zip(pd, pc, strict=True)), \
                "first step must be bit-identical to the codec fallback"
    torch.cuda.synchronize()
    for a, b in zip(pd, pc, strict=True):
        sa, sb = od.state[a], oc.state[b]
        assert sa["m"].dtype == torch.int8 and sa["m"].shape == sb["m"].shape
        assert sa["m_scale"].shape == sb["m_scale"].shape
        rs = ((sa["m_scale"] - sb["m_scale"]).abs() / sb["m_scale"].abs()).max().item()
        assert rs < 1e-6, f"row scales rel {rs:.2e}"
        dq = (sa["m"].int() - sb["m"].int()).abs()
        assert dq.max().item() <= 1 and dq.float().mean().item() < 1e-3
    d = max((a.float() - b.float()).abs().max().item() for a, b in zip(pd, pc, strict=True))
    scale = max(b.float().abs().max().item() for b in pc)
    tol = 2e-4 if dtype == torch.float32 else 2e-3
    assert d / scale < tol, f"rows route vs codec rel={d / scale:.2e} after 30 steps"


@pytest.mark.parametrize("bad", ["nan", "inf"])
@pytest.mark.parametrize("shape", [(96, 1152), (128, 512)], ids=["rows", "aligned"])
def test_big_int8_in_kernel_routes_propagate_non_finite_like_the_codec(shape, bad):
    """NaN policy (0.7.12): a non-finite grad poisons the step identically on every path. The
    codec's ``amax`` of a NaN row is NaN (scale NaN) and ``NaN.to(int8)`` is 0; ``tl.max`` /
    ``tl.atomic_max`` drop NaN and Triton's clamp picks a bound, so both in-kernel int8 routes
    (row-spanning and aligned) used to leave finite scales and different codes."""
    import kaon._fused_triton as ft
    assert ft.int8_route(shape[1]) == ("rows" if shape[1] == 1152 else "aligned")
    out = []
    for direct in (True, False):
        ps = _bag([shape] * 2, torch.float32, seed=99)
        opt = _fused(ps, lr=1e-3, momentum_dtype="int8", deterministic_reductions=True)
        opt._direct_int8 = direct
        gen = torch.Generator(device=DEV).manual_seed(5)
        for step in range(2):
            for p in ps:
                g = torch.randn(tuple(p.shape), generator=gen, device=DEV)
                if step == 1:
                    g[3, 5] = float(bad)
                p.grad = g
            opt.step()
        torch.cuda.synchronize()
        out.append([(opt.state[p]["m_scale"].clone(), opt.state[p]["m"].clone(), p.detach().clone())
                    for p in ps])
    for (sd, md, pd), (sc, mc, pc) in zip(*out, strict=True):
        assert sc.isnan().any(), "the codec reference no longer makes NaN scales"
        assert torch.equal(sd.isnan(), sc.isnan()), "NaN scale rows differ from the codec"
        assert torch.equal(sd[~sd.isnan()], sc[~sc.isnan()])
        assert torch.equal(md, mc), "codes differ from the codec under a non-finite grad"
        assert torch.equal(pd.isnan(), pc.isnan())


def test_big_int8_rows_route_matches_native():
    """End to end against the native per-row int8 codec, on a DiT-width bucket."""
    d, scale, ov = _run_parity([(128, 1152)] * 2, torch.float32, "int8", wd=0.05)
    assert len(_parts(ov)[1]) == 2
    assert d / scale < 5e-4, f"rel={d / scale:.2e}"


@pytest.mark.parametrize("shape", [(512, 512), (1024, 512), (2048, 64), (300, 1024)])
def test_big_int8_direct_matches_native(shape):
    """Every row-aligned shape: the in-kernel int8 codec tracks native within the int8 bound."""
    d, scale, ov = _run_parity([shape] * 2, torch.float32, "int8", wd=0.05)
    assert len(_parts(ov)[1]) == 2
    assert d / scale < 5e-4, f"{shape} rel={d / scale:.2e}"


def test_big_int8_direct_equals_codec_fallback():
    """The two implementations of the same codec agree to fp32-reduction ulps."""
    cfg = dict(lr=2e-3, weight_decay=0.05, cautious=True, gradient_centralization=True,
               momentum_dtype="int8")
    pd = _bag([(512, 512)] * 2, torch.float32, seed=83)
    pc = _clone(pd)
    od = _fused(pd, **cfg)
    oc = _fused(pc, **cfg)
    oc._direct_int8 = False
    gen = torch.Generator(device=DEV).manual_seed(17)
    for _ in range(6):
        gs = [torch.randn(*p.shape, generator=gen, device=DEV) for p in pd]
        for ps in (pd, pc):
            for p, g in zip(ps, gs):
                p.grad = g.clone()
        od.step()
        oc.step()
    torch.cuda.synchronize()
    d = max((a - b).abs().max().item() for a, b in zip(pd, pc))
    scale = max(b.abs().max().item() for b in pc)
    assert d / scale < 5e-4, f"direct vs codec rel={d / scale:.2e}"


# ------------------------------------------------------------------- packed reduction scratch
def test_big_reduction_scratch_is_one_zeroed_block_with_aliased_factors():
    """The mechanism behind 9 -> 6 launches per bucket, asserted structurally.

    Timing it in a unit test would be flaky; what the optimization actually IS is the
    aliasing, so that is what is checked. A regression that re-split the buffers would
    silently restore the three extra launches with every other test still green.
    """
    ps = _bag([(512, 512)] * 3, torch.float32, seed=84)
    for p in ps:
        p.grad = torch.randn_like(p)
    opt = _fused(ps, lr=1e-3, gradient_centralization=True)
    opt.step()
    cache = next(iter(opt._fused_big_caches.values()))
    zs = cache._zeros.untyped_storage().data_ptr()
    for name in ("colsum", "rms", "keep"):
        buf = getattr(cache, name)
        assert buf.untyped_storage().data_ptr() == zs, f"{name} is not part of the packed block"
    assert cache.keep.dtype == torch.int32
    assert cache.rfac is cache.rowsum and cache.cfac is cache.colsum
    assert cache.rowmean is not cache.rowsum          # GC on -> its own buffer
    assert not hasattr(cache, "inv_rms")              # folded into the consumers


def test_big_reduction_scratch_drops_rowmean_without_gc():
    ps = _bag([(512, 512)] * 3, torch.float32, seed=85)
    for p in ps:
        p.grad = torch.randn_like(p)
    opt = _fused(ps, lr=1e-3, gradient_centralization=False)
    opt.step()
    cache = next(iter(opt._fused_big_caches.values()))
    assert cache.rowmean is cache.rowsum, "GC off must not pin a rowmean buffer"


# ------------------------------------------------------------------- Triton bf16 SR write
def test_sr_add_is_unbiased_and_matches_the_torch_path_in_expectation():
    """``sr_add_`` must round stochastically with the SAME expectation as the torch path.

    A single draw cannot be compared (different noise streams by construction), so this
    compares the MEAN over many independent writes of a delta deliberately chosen to sit
    between two bf16 grid points - where round-to-nearest would lose it entirely.
    """
    import kaon._fused_triton as ft
    from kaon._stochastic_rounding import add_stochastic_
    n = 1 << 16
    # bf16 keeps 7 explicit mantissa bits (8 of precision counting the implicit leading 1),
    # so the gap between 1.0 and the next representable value is 2^-7. Derived from torch
    # rather than hardcoded, and cross-checked, so the constant cannot drift out of the
    # comment: a wrong ulp here would silently make the tolerance meaningless.
    ulp = (torch.tensor(1.0, dtype=torch.bfloat16).nextafter(
        torch.tensor(2.0, dtype=torch.bfloat16)).float().item() - 1.0)
    assert ulp == 2.0 ** -7, ulp
    tri = torch.ones(n, device=DEV, dtype=torch.bfloat16)
    tor = torch.ones(n, device=DEV, dtype=torch.bfloat16)
    delta = torch.full((n,), 0.25 * ulp, device=DEV)     # a quarter ulp: RNE would drop it
    for _ in range(40):
        ft.sr_add_(tri, delta, 1.0)
        add_stochastic_(tor, delta, alpha=1.0)
    got, ref = tri.float().mean().item(), tor.float().mean().item()
    exact = 1.0 + 40 * 0.25 * ulp
    assert abs(got - exact) < 0.05 * ulp, f"triton mean {got} vs exact {exact}"
    assert abs(got - ref) < 0.05 * ulp, f"triton mean {got} vs torch mean {ref}"


def test_sr_add_propagates_non_finite_weights():
    """NaN/inf must survive the write - the point of ``sr_round``'s finiteness guard."""
    import kaon._fused_triton as ft
    p = torch.tensor([float("nan"), float("inf"), float("-inf"), 1.0],
                     device=DEV, dtype=torch.bfloat16)
    ft.sr_add_(p, torch.zeros(4, device=DEV), 1.0)
    assert torch.isnan(p[0]) and p[1] == float("inf") and p[2] == float("-inf")
    assert torch.isfinite(p[3])


def test_sr_add_supported_rejects_what_the_kernel_cannot_index():
    import kaon._fused_triton as ft
    good = torch.zeros(64, device=DEV, dtype=torch.bfloat16)
    assert ft.sr_add_supported(good, torch.zeros(64, device=DEV))
    assert not ft.sr_add_supported(good.cpu(), torch.zeros(64))            # CPU target
    assert not ft.sr_add_supported(torch.zeros(64, device=DEV),
                                   torch.zeros(64, device=DEV))            # fp32 target
    assert not ft.sr_add_supported(good, torch.zeros(64, device=DEV,
                                                    dtype=torch.bfloat16))  # bf16 source
    strided = torch.zeros(128, device=DEV, dtype=torch.bfloat16)[::2]
    assert not ft.sr_add_supported(strided, torch.zeros(64, device=DEV))   # strided target


def test_sr_write_falls_back_to_torch_when_triton_is_off(monkeypatch):
    """``triton=False`` must reach the torch implementation; ``True`` must not."""
    from kaon import _backend as bk
    calls = []
    real = bk.add_stochastic_

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(bk, "add_stochastic_", spy)
    p = torch.zeros(64, device=DEV, dtype=torch.bfloat16)
    bk._sr_write_(p, torch.ones(64, device=DEV), -1.0, triton=False)
    assert calls, "triton=False must use the torch path"
    calls.clear()
    bk._sr_write_(p, torch.ones(64, device=DEV), -1.0, triton=True)
    assert not calls, "triton=True must use the kernel on a supported pair"


def test_sr_write_reaches_both_weight_writers(monkeypatch):
    """Both public writers route bf16+SR through a Triton SR write: the per-param one through
    ``_sr_write_``, the batched one IN PLACE over its views (``sr_add_views_``, 0.7.18) — and
    through ``_sr_write_`` on the stack when the views cannot take it (a strided view)."""
    import kaon._fused_triton as ft
    from kaon import _backend as bk
    seen = []
    real, real_views = bk._sr_write_, ft.sr_add_views_

    def spy(*a, **k):
        seen.append("stack")
        return real(*a, **k)

    def spy_views(*a, **k):
        seen.append("views")
        return real_views(*a, **k)

    monkeypatch.setattr(bk, "_sr_write_", spy)
    monkeypatch.setattr(ft, "sr_add_views_", spy_views)
    p = torch.zeros(8, 8, device=DEV, dtype=torch.bfloat16)
    bk.subtract_one_(p, torch.ones(8, 8, device=DEV), {}, "stochastic_rounding", alpha=1e-3)
    bk.subtract_batched_([p, p.clone()], torch.ones(2, 8, 8, device=DEV),
                         "stochastic_rounding", alpha=1e-3)
    q = torch.zeros(8, 8, device=DEV, dtype=torch.bfloat16)
    bk.subtract_batched_([p.t(), q], torch.ones(2, 8, 8, device=DEV),
                         "stochastic_rounding", alpha=1e-3)
    assert seen == ["stack", "views", "stack"]


# ---------------------------------------------- 4-bit requant: single-axis reduction (item 9a)
def _fourbit_state(shape, n, exact, steps=6, seed=91):
    """Step a 4-bit bag with the one-block tile kernel, optionally forcing the general loop."""
    ps = _bag([shape] * n, torch.float32, seed)
    opt = _fused(ps, lr=2e-3, weight_decay=0.05, momentum_dtype="4bit")
    gen = torch.Generator(device=DEV).manual_seed(7)
    for i in range(steps):
        for p in ps:
            p.grad = torch.randn(*p.shape, generator=gen, device=DEV)
        opt.step()
        if i == 0 and not exact:            # the pointer caches exist after the first step
            for cache in opt._fused_ob_caches.values():
                for bk in cache.buckets:
                    bk["exact4"] = False
                    bk["fblk"] = 0
    torch.cuda.synchronize()
    return ps, opt


@pytest.mark.parametrize("shape", [(64, 128), (128, 64), (16, 512), (64, 64)])
def test_fourbit_single_axis_reduction_is_bit_identical_to_the_loop(shape):
    """The rewritten per-block absmax must be BIT-identical, not merely close.

    ``max`` is exact whatever the reduction order, so reshaping the tile into
    ``(NB, BLK)`` and reducing one axis computes the same value as the NB-iteration loop
    over the whole tile — weights, packed codes AND scales. Measured 1.19-1.63x faster
    (the ratio grows with NB), which took 4-bit momentum from 1.8-3.2x slower than bf16
    on these shapes to 0.82-1.23x.
    """
    pa, oa = _fourbit_state(shape, 8, exact=True)
    pb, ob = _fourbit_state(shape, 8, exact=False)
    assert all(torch.equal(a, b) for a, b in zip(pa, pb)), "weights differ"
    assert all(torch.equal(oa.state[a]["m"], ob.state[b]["m"]) for a, b in zip(pa, pb))
    assert all(torch.equal(oa.state[a]["m_scale"], ob.state[b]["m_scale"])
               for a, b in zip(pa, pb))


def test_fourbit_exact_tile_flag_tracks_the_bucket_shape():
    """The fast path is claimed only where the tile is unpadded — that is its precondition."""
    ps = _bag([(64, 128)] * 4, torch.float32, seed=92)          # powers of two -> exact
    for p in ps:
        p.grad = torch.randn_like(p)
    opt = _fused(ps, lr=1e-3, momentum_dtype="4bit")
    opt.step()
    bks = _buckets(opt)
    assert bks and all(b["exact4"] and b["fblk"] == 128 for b in bks)

    ps = _bag([(60, 100)] * 4, torch.float32, seed=93)          # padded tile -> general loop
    for p in ps:
        p.grad = torch.randn_like(p)
    opt = _fused(ps, lr=1e-3, momentum_dtype="4bit")
    opt.step()
    bks = _buckets(opt)
    assert bks and not any(b["exact4"] for b in bks)


# ------------------------------------------- deterministic reductions + the requant race (9b)
def _big_run(mdtype, det, shape=(512, 512), n=3, steps=6):
    ps = _bag([shape] * n, torch.float32, seed=94)
    opt = Adakaon(ps, lr=2e-3, weight_decay=0.05, fused=True, momentum_dtype=mdtype,
                  deterministic_reductions=det)
    gen = torch.Generator(device=DEV).manual_seed(7)
    for _ in range(steps):
        for p in ps:
            p.grad = torch.randn(*p.shape, generator=gen, device=DEV)
        opt.step()
    torch.cuda.synchronize()
    return ps


@pytest.mark.parametrize("mdtype", ["float32", "bfloat16", "int8", "4bit"])
def test_deterministic_reductions_make_the_big_path_bit_reproducible(mdtype):
    """``deterministic_reductions=True`` must give the SAME bits on a repeated run.

    The default path accumulates ``colsum``/``rms`` with fp32 atomics whose completion order
    the scheduler picks, so identical inputs drift run to run: measured max|Δp| / scale over
    4 runs = 5.1e-8 (fp32), 3.9e-6 (bf16), 8.1e-6 (int8). Two-pass partials remove it.
    """
    runs = [_big_run(mdtype, True) for _ in range(3)]
    assert all(all(torch.equal(a, b) for a, b in zip(runs[0], r)) for r in runs[1:])


@pytest.mark.parametrize("shape", [(257, 513), (300, 1024), (1024, 512)])
def test_deterministic_reductions_cover_partial_chunks(shape):
    runs = [_big_run("4bit", True, shape=shape, n=2) for _ in range(3)]
    assert all(all(torch.equal(a, b) for a, b in zip(runs[0], r)) for r in runs[1:])


def test_chunked_4bit_requant_no_longer_reloads_its_own_scales():
    """Regression guard for the in-kernel 4-bit requant RACE.

    ``_chunked_4bit_apply_batched_g`` used to store the per-block scales and then
    ``tl.load`` them back to quantize — a cross-lane store/load inside one program with no
    barrier, so a lane could divide by the PREVIOUS step's scale. It made 4-bit momentum
    nondeterministic (5.1e-4 relative spread over identical runs) *even with* the reductions
    made two-pass. With the scale kept in registers the spread is 5.1e-8, i.e. the atomics'
    own noise, so this asserts the default (atomic) path is already ~1e-7 for 4-bit.
    """
    runs = [_big_run("4bit", False) for _ in range(4)]
    scale = max(p.abs().max().item() for p in runs[0])
    spread = max(max((a - b).abs().max().item() for a, b in zip(runs[0], r)) for r in runs[1:])
    # Atomic ordering can flip one 4-bit bin (~3e-5); the race produced 3.4e-4..6.2e-4.
    assert spread / scale < 1e-4, f"4-bit run-to-run spread {spread / scale:.2e} — race back?"


def test_deterministic_reductions_agree_with_the_atomic_path():
    """Same arithmetic, different order: the two must agree to reduction-order ulps."""
    a = _big_run("float32", False)
    b = _big_run("float32", True)
    scale = max(p.abs().max().item() for p in a)
    d = max((x - y).abs().max().item() for x, y in zip(a, b))
    assert d / scale < 1e-6, f"det vs atomic rel={d / scale:.2e}"


def test_deterministic_reductions_allocate_partials_only_when_on():
    ps = _bag([(512, 512)] * 2, torch.float32, seed=95)
    for p in ps:
        p.grad = torch.randn_like(p)
    off = _fused(ps, lr=1e-3)
    off.step()
    assert all(getattr(c, "_partials", None) is None for c in off._fused_big_caches.values())
    on = Adakaon(ps, lr=1e-3, fused=True, deterministic_reductions=True)
    on.step()
    assert all(getattr(c, "_partials", None) is not None
               for c in on._fused_big_caches.values())


# ------------------------------------------- reseeding must reach the Triton SR noise stream
# The Triton bf16 write has its OWN seed counter, so ``kaon.reseed_stochastic_rounding()`` has
# to reset it too. When it did not, re-seeding to the SAME value inside one process — the case
# ``kaon._stochastic_rounding``'s docstring exists to cover — stopped reproducing any bf16 run
# on a Triton build, for every optimizer, silently. These run a real Adakaon step so both
# writers are exercised: the foreach bucket goes through ``subtract_batched_`` and the lone
# 0-D/odd params through ``subtract_one_``.
def _sr_repro_run(sr_triton, seed=1234, steps=5):
    from kaon import _backend as bk
    saved = bk.SR_TRITON
    bk.SR_TRITON = sr_triton
    try:
        torch.manual_seed(seed)                     # the SAME seed every call
        kaon.reseed_stochastic_rounding()           # the only public reset
        ps = [torch.randn(64, 64, device=DEV, dtype=torch.bfloat16).requires_grad_(True)
              for _ in range(4)]
        ps += [torch.randn((), device=DEV, dtype=torch.bfloat16).requires_grad_(True)]
        opt = Adakaon(ps, lr=1e-2, weight_decay=0.01, bf16_method="stochastic_rounding")
        gen = torch.Generator(device=DEV).manual_seed(7)
        for _ in range(steps):
            for p in ps:
                p.grad = torch.randn(tuple(p.shape), generator=gen, device=DEV,
                                     dtype=torch.bfloat16)
            opt.step()
        torch.cuda.synchronize()
        return [p.detach().clone() for p in ps], torch.rand(4, device=DEV)
    finally:
        bk.SR_TRITON = saved


@pytest.mark.parametrize("sr_triton", [True, False])
def test_reseed_reproduces_a_bf16_sr_run_on_both_write_paths(sr_triton):
    """``torch.manual_seed(s)`` + ``reseed_stochastic_rounding()`` must reproduce the weights."""
    a, _ = _sr_repro_run(sr_triton)
    b, _ = _sr_repro_run(sr_triton)
    assert all(torch.equal(x, y) for x, y in zip(a, b)), (
        f"SR_TRITON={sr_triton}: re-seeding to the same value did not reproduce the run"
    )


@pytest.mark.parametrize("sr_triton", [True, False])
def test_sr_noise_stays_isolated_from_the_user_rng(sr_triton):
    """Stochastic rounding must not consume the global stream: same seed -> same ``torch.rand``."""
    _, ra = _sr_repro_run(sr_triton)
    _, rb = _sr_repro_run(sr_triton)
    assert torch.equal(ra, rb)


def test_reseed_reaches_the_kernels_fallback_stream_and_stays_internal():
    """One public reseed entry point, and the kernel's fallback counter hangs off it.

    Since 0.7.13 the kernel counter is an :class:`~kaon._stochastic_rounding.SRStream` like
    every other, so the reset arrives through the module reseed epoch the streams check on
    use — no registry, and no second public entry point to reset it with.
    """
    import kaon._fused_triton as ft
    assert not hasattr(ft, "reseed_sr_kernel"), "the kernel reset must not be a second public API"
    stream = ft._PROCESS_SR_STREAM
    assert stream.stream_id == 0, "the fallback must keep reproducing stream 0's sequence"
    stream.next_seed(torch.device(DEV))
    stream.next_seed(torch.device(DEV))
    assert stream.draws == 2
    kaon.reseed_stochastic_rounding()
    assert stream.draws == 0 or stream.snapshot()["draws"] == 0, (
        "reseed_stochastic_rounding must restart the kernel counter"
    )
    assert stream.stream_id == 0, "the fallback's id is pinned, so a reseed must keep it"


# ------------------------------------- gradient_centralization flipped on a live param group
def _gc_flip_run(fused, start_gc, flip_to, flip_at=3, steps=6):
    ps = _bag([(512, 512)] * 3, torch.float32, seed=96)
    opt = Adakaon(ps, lr=2e-3, weight_decay=0.05, cautious=True, fused=fused,
                  momentum_dtype="float32", gradient_centralization=start_gc)
    gen = torch.Generator(device=DEV).manual_seed(7)
    for i in range(steps):
        if i == flip_at:                       # schedulers do reach into the group dict
            for group in opt.param_groups:
                group["gradient_centralization"] = flip_to
        for p in ps:
            p.grad = torch.randn(*p.shape, generator=gen, device=DEV)
        opt.step()
    torch.cuda.synchronize()
    return ps


@pytest.mark.parametrize(("start_gc", "flip_to"), [(False, True), (True, False)])
def test_gc_flipped_mid_run_still_matches_native(start_gc, flip_to):
    """``BigPointerCache`` aliases ``rowmean`` onto ``rowsum`` when GC is off.

    That is only sound while GC STAYS off. A param group is a mutable dict, so a scheduler can
    flip the flag between steps with no parameter moving — which the witness cannot see. With
    the alias live under ``GC=True`` the reduction kernel wrote the per-row means over the row
    sums and the factored EMA came out of means: measured 1.1e-3 relative divergence from
    native, silently. ``gc`` is part of the cache's validity now.
    """
    fused = _gc_flip_run(True, start_gc, flip_to)
    native = _gc_flip_run(False, start_gc, flip_to)
    scale = max(p.abs().max().item() for p in native)
    d = max((a - b).abs().max().item() for a, b in zip(fused, native))
    assert d / scale < 1e-6, f"gc {start_gc}->{flip_to}: fused vs native rel={d / scale:.2e}"


def test_big_cache_records_gc_and_rebuilds_on_a_flip():
    ps = _bag([(512, 512)] * 2, torch.float32, seed=97)
    for p in ps:
        p.grad = torch.randn_like(p)
    opt = _fused(ps, lr=1e-3, gradient_centralization=False)
    opt.step()
    first = next(iter(opt._fused_big_caches.values()))
    assert first.gc is False and first.rowmean is first.rowsum
    for group in opt.param_groups:
        group["gradient_centralization"] = True
    opt.step()
    second = next(iter(opt._fused_big_caches.values()))
    assert second is not first, "the cache must be rebuilt when gc flips"
    assert second.gc is True and second.rowmean is not second.rowsum


# ------------------------------------------------- int8 direct-path guard: BOTH halves matter
@pytest.mark.parametrize(("shape", "route"), [
    ((512, 512), "aligned"),   # C=512 divides 1024
    ((300, 1024), "aligned"),  # C=1024 divides 1024 (one row per chunk)
    ((256, 513), "rows"),      # C=513 <= 1024 but does NOT divide it -> rows span chunks
    ((64, 4096), "rows"),      # C=4096 > 1024 -> a row spans four chunks
    ((256, 100), "codec"),     # rows span chunks AND are too narrow for the row route
])
def test_big_int8_guard_routes_on_row_alignment(shape, route, monkeypatch):
    """``C <= 1024 and 1024 % C == 0`` — the second half is load-bearing on its own.

    ``(256, 513)`` is under the block size and still splits rows, so the ALIGNED kernel's
    per-row absmax would have several writing programs and store a scale computed from part of
    a row. It must take the cross-program row route (0.7.18; the codec fallback before), which
    is the one whose cache carries the ``rowmax`` accumulator; too-narrow rows keep the codec.
    """
    import kaon._fused_triton as ft
    assert ft.int8_route(shape[1]) == route
    direct = route != "codec"
    ps = _bag([shape] * 2, torch.float32, seed=98)
    for p in ps:
        p.grad = torch.randn_like(p)
    opt = _fused(ps, lr=1e-3, momentum_dtype="int8")
    codec = opt._codec(opt.param_groups[0])
    seen = []
    real = codec.dequant_stacked

    def spy(*a, **k):
        seen.append(1)
        return real(*a, **k)

    monkeypatch.setattr(codec, "dequant_stacked", spy)
    opt.step()
    assert bool(seen) is (not direct), (
        f"{shape}: expected {'the in-kernel path' if direct else 'the codec fallback'}"
    )
    cache = next(iter(opt._fused_big_caches.values()))
    assert (cache.rowmax is not None) is (route == "rows")


@pytest.mark.parametrize("shape", [(256, 513), (64, 4096)])
def test_big_int8_row_split_shapes_still_match_native(shape):
    """And the fallback they take must be correct, not merely taken."""
    d, scale, ov = _run_parity([shape] * 2, torch.float32, "int8", wd=0.05)
    assert len(_parts(ov)[1]) == 2
    assert d / scale < 5e-4, f"{shape} rel={d / scale:.2e}"


# ------------------------------------------------------------------- in-place foreach bf16 write
def _views_bucket(n_views, shape, seed, bits=0):
    """``n_views`` bf16 params (views of separate storages, like a foreach plan's pviews), their
    residuals under compact Kahan, and a stacked fp32 delta."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    views = [torch.randn(shape, generator=g, device=DEV).bfloat16() for _ in range(n_views)]
    lows = []
    if bits:
        dt = torch.int16 if bits == 16 else torch.uint8
        hi = 2 ** 15 if bits == 16 else 256
        lo = -hi if bits == 16 else 0
        lows = [torch.randint(lo, hi, shape, generator=g, device=DEV).to(dt) for _ in views]
    delta = torch.randn((n_views, *shape), generator=g, device=DEV) * 1e-2
    return views, lows, delta


@pytest.mark.parametrize("bits", [0, 8, 16], ids=["sr", "kahan8", "kahan16"])
@pytest.mark.parametrize("shape", [(64, 48), (3000,), (7, 1030)])
def test_in_place_views_write_is_bit_identical_to_the_stacked_write(bits, shape):
    """``sr_add_views_`` / ``ck_add_views_`` write each view in place and must reproduce
    ``stack -> sr_add_/ck_add_ -> _foreach_copy_`` bit for bit: same seed from the stream,
    same per-element noise counter (the stacked index), weights AND residuals."""
    import kaon._fused_triton as ft
    from kaon._stochastic_rounding import SRStream
    views, lows, delta = _views_bucket(5, shape, seed=90, bits=bits)
    ref_v = [v.clone() for v in views]
    ref_l = [c.clone() for c in lows]
    sa, sb = SRStream(11), SRStream(11)
    stacked = torch.stack(ref_v)
    if bits:
        slo = torch.stack(ref_l)
        ft.ck_add_(stacked, slo, delta, -0.7, bits, sa)
        torch._foreach_copy_(ref_l, list(slo.unbind(0)))
        assert ft.ck_add_views_supported(views, lows, delta, bits)
        ft.ck_add_views_(views, lows, delta, -0.7, bits, sb)
    else:
        ft.sr_add_(stacked, delta, -0.7, sa)
        assert ft.sr_add_views_supported(views, delta)
        ft.sr_add_views_(views, delta, -0.7, sb)
    torch._foreach_copy_(ref_v, list(stacked.unbind(0)))
    torch.cuda.synchronize()
    assert sa.draws == sb.draws == 1
    for a, b in zip(views, ref_v, strict=True):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))
    for a, b in zip(lows, ref_l, strict=True):
        assert torch.equal(a, b)


def test_in_place_views_write_refuses_what_it_cannot_index():
    import kaon._fused_triton as ft
    views, lows, delta = _views_bucket(3, (8, 8), seed=91, bits=8)
    assert ft.sr_add_views_supported(views, delta)
    assert not ft.sr_add_views_supported([views[0].t()] + views[1:], delta)      # strided
    assert not ft.sr_add_views_supported(views[:2], delta)                        # count
    assert not ft.sr_add_views_supported([v.float() for v in views], delta)       # dtype
    assert not ft.sr_add_views_supported(views, delta.bfloat16())
    assert not ft.ck_add_views_supported(views, [c.to(torch.int16) for c in lows], delta, 8)
    assert not ft.ck_add_views_supported(views, lows[:2], delta, 8)


def test_in_place_views_pointer_array_follows_a_rebind():
    """The pointer array is content-addressed: a view list whose storage moved gets a new
    array, never the old one."""
    import kaon._fused_triton as ft
    views, _, delta = _views_bucket(2, (16,), seed=92)
    ft.sr_add_views_(views, delta, 1.0)
    moved = [v.clone() for v in views]
    before = [v.clone() for v in views]
    ft.sr_add_views_(moved, delta, 1.0)
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(views, before, strict=True))
    assert not all(torch.equal(a, b) for a, b in zip(moved, before, strict=True))


@pytest.mark.parametrize("kernel", ["sr", "ck8", "ck16", "decode", "sr_views", "ck_views"])
def test_int64_indexing_is_bit_identical_below_the_threshold(kernel):
    """The ``I64`` variant (taken for >= 2**31-element index spaces, whose int32 offsets used to
    wrap negative and slip past the ``offs < n`` mask) must compute exactly what the int32
    variant computes wherever int32 did not wrap — same Philox noise included (an int64
    counter below 2**32 draws the int32 stream). Allocating 2**31 elements is not possible on
    a test GPU, so the variant is forced on a small buffer."""
    import kaon._fused_triton as ft
    n = 5000
    g = torch.Generator(device=DEV).manual_seed(93)
    p0 = torch.randn(n, generator=g, device=DEV).bfloat16()
    d = torch.randn(n, generator=g, device=DEV) * 1e-2
    bits = 16 if kernel == "ck16" else 8
    lo0 = torch.randint(0, 256, (n,), generator=g, device=DEV).to(torch.uint8)
    if bits == 16:
        lo0 = torch.randint(-2**15, 2**15, (n,), generator=g, device=DEV).to(torch.int16)
    outs = []
    for i64 in (False, True):
        p, lo = p0.clone(), lo0.clone()
        grid = ((n + 1023) // 1024,)
        if kernel == "sr":
            ft._sr_axpy_kernel[grid](p, d, -0.5, n, 123, BLOCK=1024, I64=i64)
            res = (p,)
        elif kernel in ("ck8", "ck16"):
            ft._ck_axpy_kernel[grid](p, lo, d, -0.5, n, 123, BITS=bits, BLOCK=1024, I64=i64)
            res = (p, lo)
        elif kernel == "decode":
            out = torch.empty(n, device=DEV)
            ft._ck_decode_kernel[grid](p, lo, out, n, BITS=8, BLOCK=1024, I64=i64)
            res = (out,)
        else:
            views = list(p.view(5, 1000).unbind(0))
            lows = list(lo.view(5, 1000).unbind(0))
            K = 1  # noqa: N806
            parr = ft._view_ptr_array(views, p.device)
            if kernel == "sr_views":
                ft._sr_axpy_views_kernel[(5,)](parr, d, -0.5, 1000, K, 123, BLOCK=1024, I64=i64)
            else:
                ft._ck_axpy_views_kernel[(5,)](parr, ft._view_ptr_array(lows, p.device), d, -0.5,
                                               1000, K, 123, BITS=8, BLOCK=1024, I64=i64)
            res = (p, lo)
        torch.cuda.synchronize()
        outs.append(res)
    for a, b in zip(*outs, strict=True):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    assert not ft.needs_i64(2**31 - 4096) and ft.needs_i64(2**31)


def test_grad_pointer_refresh_tracks_every_step():
    """Grads reallocated on EVERY step (the varying-sequence-length pattern), old buffers
    poisoned right after the step that read them: every route must read this step's grads —
    parity with native over 8 steps."""
    shapes = [(64, 48)] * 3 + [(512, 512)] * 2 + [(96,)] * 3

    def run(fused):
        torch.manual_seed(0)
        ps = [(torch.randn(s, device=DEV) * 0.01).requires_grad_(True) for s in shapes]
        opt = Adakaon(ps, lr=1e-3, fused=fused, deterministic_reductions=True)
        g = torch.Generator(device=DEV).manual_seed(3)
        keep = []
        for _ in range(8):
            gs = [torch.randn(s, generator=g, device=DEV) for s in shapes]
            for p, gr in zip(ps, gs, strict=True):
                p.grad = gr
            opt.step()
            for old in keep:
                old.fill_(float("nan"))          # a freed buffer the allocator handed back
            keep = gs
        torch.cuda.synchronize()
        return [p.detach().clone() for p in ps]

    wf, wn = run(True), run(False)
    for a, b in zip(wf, wn, strict=True):
        assert torch.isfinite(a).all()
        # 1.05e-5 is the big route's ordinary fused-vs-native drift over 8 steps (the same
        # with the pre-0.7.18 refresh); a stale pointer reads NaN/garbage (poisoned buffers).
        assert torch.allclose(a, b, atol=5e-5), "a fused route read a stale grad pointer"


def test_in_place_views_write_refuses_duplicate_or_overlapping_views():
    """The same param twice in a bucket (or overlapping views of one storage) would make the
    in-place write a race between programs; those buckets keep the stacked path (where the
    last copy-back wins, as before). Disjoint views of ONE storage are fine."""
    import kaon._fused_triton as ft
    from kaon import _backend as bk
    x = torch.randn(128, device=DEV).bfloat16()
    d2 = torch.randn(2, 64, device=DEV)
    assert ft.sr_add_views_supported([x[:64], x[64:]], d2)
    assert not ft.sr_add_views_supported([x[:64], x[:64]], d2)
    assert not ft.sr_add_views_supported([x[:64], x[32:96]], d2)
    lo = torch.zeros(128, device=DEV, dtype=torch.uint8)
    assert not ft.ck_add_views_supported([x[:64], x[64:]], [lo[:64], lo[:64]], d2, 8)
    # end to end: a duplicated view goes through the stack, so every element of the param
    # is one of the two stacked rows (the copy-back of two rows into one param has no
    # defined order — the pre-0.7.18 behaviour, kept), never a torn in-place mixture
    from kaon._stochastic_rounding import SRStream
    p = torch.randn(64, device=DEV).bfloat16()
    stack = torch.stack([p, p])
    ft.sr_add_(stack, d2, -1e-2, SRStream(3))
    bk.subtract_batched_([p, p], d2, "stochastic_rounding", alpha=1e-2, sr=SRStream(3))
    torch.cuda.synchronize()
    assert ((p == stack[0]) | (p == stack[1])).all()
