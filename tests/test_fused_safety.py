"""Crash / memory-safety regressions for the Triton fused path (audit batch A).

Every test here has a confirmed repro against the pre-fix code; none of them is a
performance or a numerics-quality assertion — they all guard a way the fused step could
corrupt memory, refuse to compile, or silently disagree with the native step:

  * ``p.data`` REBIND mid-training. The fused pointer caches were keyed on ``id(p)`` only,
    so an external EMA / ``.to()`` / block-swap offloader that rebinds the SAME parameter to
    fresh storage left the kernels writing the step into the RETIRED buffer (use-after-free).
    Covered on all four routes: one-block, 1-D, 0-D and big-batched.
  * ``reduction_tile`` handing a non-power-of-2 ``BR`` to ``tl.arange`` -> ``CompilationError``
    for perfectly ordinary shapes ((96, 96), (9, 640), (12, 1024), (65, 65)).
  * ``momentum_4bit_block != 128`` on a one-block shape: the tile kernel hardcoded 128-element
    blocks, so it wrote past the (shorter) ``m_scale`` buffer. The block is a runtime kernel
    scalar since 0.7.12 (and ``PointerArrayCache`` buckets by it), so the route is kept for
    every block size — the memory-safety property, not the routing, is what is asserted.
  * Non-contiguous GRADS: the kernels index from ``data_ptr()`` with row-major arithmetic and
    ignored strides, so a transposed / strided grad silently stepped the wrong numbers.
  * fp16 params + ``bf16_method="stochastic_rounding"``: unsupported, and silently degraded
    to round-to-nearest instead of saying so.
  * NON-FINITE PROPAGATION. Policy is propagate (PyTorch-standard), identically in the native
    and the fused step: cautious must not freeze a NaN tensor, and ``sr_round`` must not turn
    a NaN payload into ``-0.0`` via int32 overflow.

Skips cleanly when CUDA or Triton is unavailable (the kernels are GPU-only).
"""
from __future__ import annotations

import pytest
import torch

from kaon import Adakaon
from kaon._fused_triton import HAS_TRITON, reduction_tile

pytestmark = pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()),
    reason="Triton fused kernel requires CUDA + Triton",
)

DEV = "cuda"

if HAS_TRITON:
    import triton
    import triton.language as tl

    from kaon._fused_triton import requant_4bit, sr_round

    @triton.jit
    def _sr_round_passthrough(in_ptr, out_ptr, seed, N, BLOCK: tl.constexpr):
        """Round every input lane through ``sr_round`` — the probe for NaN/inf preservation."""
        offs = tl.arange(0, BLOCK)
        mask = offs < N
        val = tl.load(in_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, sr_round(val, seed, offs), mask=mask)

    @triton.jit
    def _requant_4bit_exact_probe(m_ptr, packed_ptr, scale_ptr, R, C, Chalf, NB, NS, BLK,
                                  BR: tl.constexpr, BC: tl.constexpr, FBLK: tl.constexpr):
        """Drive ``requant_4bit``'s ``EXACT`` single-reduction path with an arbitrary ``FBLK``.

        The path reshapes the tile to ``(BR*BC // FBLK, FBLK)``, so ``FBLK`` must DIVIDE
        ``BR*BC``. Since 0.7.12 ``FBLK`` is the bucket's real ``momentum_4bit_block`` instead
        of a hardcoded 128, and it is ``PointerArrayCache`` that keeps the invariant."""
        ri = tl.arange(0, BR)[:, None]
        ci = tl.arange(0, BC)[None, :]
        mask = (ri < R) & (ci < C)
        idx = ri * C + ci
        m = tl.load(m_ptr + idx, mask=mask, other=0.0)
        requant_4bit(m, mask, idx, R, C, Chalf, packed_ptr, scale_ptr, NB, NS, BLK,
                     BR, BC, True, FBLK)

    @triton.jit
    def _requant_4bit_probe(m_ptr, packed_ptr, scale_ptr, R, C, Chalf, NB, NS, BLK,
                            BR: tl.constexpr, BC: tl.constexpr):
        """Drive ``requant_4bit`` with an m_scale CAPACITY (``NS``) smaller than the block count
        the kernel writes (``NB``) — the layout the routing guard is there to prevent."""
        ri = tl.arange(0, BR)[:, None]
        ci = tl.arange(0, BC)[None, :]
        mask = (ri < R) & (ci < C)
        idx = ri * C + ci
        m = tl.load(m_ptr + idx, mask=mask, other=0.0)
        requant_4bit(m, mask, idx, R, C, Chalf, packed_ptr, scale_ptr, NB, NS, BLK, BR, BC)


# ----------------------------------------------------------------- helpers
def _bag(shapes, dtype=torch.float32, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return [torch.randn(s, generator=g, device=DEV, dtype=dtype).requires_grad_(True)
            for s in shapes]


def _clone(ps):
    return [p.detach().clone().requires_grad_(True) for p in ps]


def _parts(opt):
    """(one_block, big, one_dim, native) from the cached fused partition (after a step)."""
    ob, big, od, nat = [], [], [], []
    for entry in opt._fused_part.values():
        o, b, d, n = entry[-4:]          # the leading witness fields are not of interest here
        ob += o
        big += b
        od += d
        nat += n
    return ob, big, od, nat


def _plain_grads(gen, plist):
    return [torch.randn(tuple(p.shape), generator=gen, device=DEV, dtype=p.dtype) for p in plist]


def _drive(pairs, steps, gen, grads_for=_plain_grads, mutate=None):
    """Step every (params, optimizer) pair on IDENTICAL gradients for ``steps`` steps.

    The grads are re-DRAWN per optimizer from the same seed rather than cloned, so a fixture
    that returns a non-contiguous grad keeps that layout for every pair (``clone()`` would
    quietly compact it and defeat the strided-grad tests).
    """
    for step in range(steps):
        if mutate is not None:
            mutate(step, pairs)
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=gen, device=DEV).item())
        for plist, opt in pairs:
            draw = torch.Generator(device=DEV).manual_seed(seed)
            for p, g in zip(plist, grads_for(draw, plist), strict=True):
                p.grad = g
            opt.step()
    torch.cuda.synchronize()


def _maxdiff(pa, pb):
    return max((a.detach().float() - b.detach().float()).abs().max().item()
               for a, b in zip(pa, pb, strict=True))


def _pair(shapes, cfg, dtype=torch.float32, seed=3):
    pv = _bag(shapes, dtype, seed)
    pn = _clone(pv)
    return pv, pn, Adakaon(pv, fused=True, **cfg), Adakaon(pn, **cfg)


_FP32_CFG = dict(lr=2e-3, betas=(0.9, 0.999), momentum_dtype="float32",
                 weight_decay=0.02, cautious=True, gradient_centralization=True)

# One representative bag per fused route (the routing is asserted, not assumed).
_ROUTES = {
    "one_block": [(8, 16)] * 4,
    "one_dim": [(32,)] * 3,
    "zero_dim": [()] * 3,
    "big": [(512, 512)] * 2,
}


def _assert_route(opt, route):
    ob, big, od, nat = _parts(opt)
    assert not nat, f"{route}: {len(nat)} params fell to the native path"
    if route == "one_block":
        assert ob and not big and not od
    elif route == "big":
        assert big and not ob and not od
    else:
        assert od and not ob and not big


# ----------------------------------------------------------------- 1. p.data rebind
@pytest.mark.parametrize("route", list(_ROUTES))
def test_data_rebind_stops_writing_the_retired_storage(route):
    """``p.data = p.data.clone()`` mid-training must move the kernel to the NEW buffer.

    The retired tensor is kept alive here on purpose: on the pre-fix code the cached
    pointer array still addressed it, so the step landed in memory the optimizer no longer
    owns — in real training (an EMA, or Rengu-Flow's block-swap offloader) that buffer is
    freed and reused, which is a silent corruption or an illegal memory access.
    """
    pv, pn, ov, on = _pair(_ROUTES[route], _FP32_CFG)
    retired: list[tuple[torch.Tensor, torch.Tensor]] = []

    def mutate(step, _pairs):
        if step != 2:
            return
        for plist in (pv, pn):
            for p in plist:
                old = p.data
                p.data = old.clone()
                if plist is pv:
                    retired.append((old, old.clone()))

    _drive([(pv, ov), (pn, on)], 5, torch.Generator(device=DEV).manual_seed(11), mutate=mutate)
    _assert_route(ov, route)
    assert retired, "the mutation hook never ran"
    for old, snapshot in retired:
        assert torch.equal(old, snapshot), f"{route}: the fused step wrote the RETIRED storage"
    d = _maxdiff(pv, pn)
    assert d < 1e-5, f"{route}: max|Δp| vs native = {d:.2e} after a rebind"


@pytest.mark.parametrize("route", list(_ROUTES))
def test_data_rebind_to_bf16_stays_finite(route):
    """A rebind that also changes dtype re-buckets the cache (``lowp``/SR flip with it)."""
    pv = _bag(_ROUTES[route], torch.float32, seed=5)
    ov = Adakaon(pv, fused=True, **_FP32_CFG)
    retired: list[tuple[torch.Tensor, torch.Tensor]] = []

    def mutate(step, _pairs):
        if step != 2:
            return
        for p in pv:
            old = p.data
            p.data = old.to(torch.bfloat16)
            retired.append((old, old.clone()))

    _drive([(pv, ov)], 5, torch.Generator(device=DEV).manual_seed(13), mutate=mutate)
    assert all(p.dtype == torch.bfloat16 for p in pv)
    for old, snapshot in retired:
        assert torch.equal(old, snapshot), f"{route}: the fused step wrote the RETIRED storage"
    for p in pv:
        assert torch.isfinite(p.detach().float()).all(), f"{route}: non-finite weights after rebind"


# ----------------------------------------------------------------- 2. reduction_tile BR
@pytest.mark.parametrize(
    "R,C", [(96, 96), (9, 640), (12, 1024), (65, 65), (1, 1), (3, 7), (512, 512), (1024, 4)]
)
def test_reduction_tile_row_block_is_a_power_of_two(R, C):  # noqa: N803
    """``BR`` reaches ``tl.arange(0, BR)`` — Triton rejects a non-power-of-2 range."""
    BR, BC, RB = reduction_tile(R, C)  # noqa: N806
    assert BR >= 1 and BR & (BR - 1) == 0, f"BR={BR} is not a power of two"
    assert triton.next_power_of_2(C) == BC
    assert RB * BR >= R and (RB - 1) * BR < R, f"RB={RB} does not tile R={R} in BR={BR} blocks"


@pytest.mark.parametrize(
    "shapes,beta1",
    [
        ([(96, 96)] * 2, 0.9),
        ([(9, 640)] * 2, 0.9),
        ([(12, 1024)] * 2, 0.9),
        ([(65, 65)] * 3, 0.9),
        ([(96, 96)], 0.0),          # a lone big tensor with beta1=0 still takes the batched path
    ],
)
def test_awkward_row_counts_compile_and_match_native(shapes, beta1):
    """Shapes whose row count is not a power of two used to raise ``CompilationError``."""
    cfg = dict(_FP32_CFG, betas=(beta1, 0.999))
    pv, pn, ov, on = _pair(shapes, cfg, seed=17)
    _drive([(pv, ov), (pn, on)], 4, torch.Generator(device=DEV).manual_seed(19))
    assert _parts(ov)[1], "these shapes are meant to route to the big (chunked) path"
    d = _maxdiff(pv, pn)
    assert d < 1e-5, f"max|Δp|={d:.2e}"


# ----------------------------------------------------------------- 3. momentum_4bit_block
# The tile kernel took its 4-bit absmax block from a hardcoded ``BLK = min(R*C, 128)``, so any
# other ``momentum_4bit_block`` wrote ceil(numel/128) scales into an ``m_scale`` sized for the
# REAL block count — past the end of the buffer (a (64,128) weight at block=0 wrote 63 floats
# out of bounds). That was fixed by a ROUTING GUARD that sent those tensors to the native path.
# Since 0.7.12 the block is a RUNTIME kernel scalar and ``PointerArrayCache`` buckets by it, so
# every block size keeps the fused route; the memory-safety property is unchanged and is what
# these tests assert (plus the ``NS`` canary below, which is the second line of defence).
@pytest.mark.parametrize("block", [64, 256, 32, 0])
def test_4bit_block_other_than_128_keeps_the_one_block_route(block):
    """Every ``momentum_4bit_block`` now takes the one-block kernel and matches native.

    ``momentum_4bit_block=0`` means "one block over the whole tensor", 64 means 128 blocks for
    a (64, 128) weight. The stored layout (``m_block`` / ``m_scale`` length) must be exactly
    what the codec would produce — the kernel adapts to it, never the other way round.
    """
    cfg = dict(_FP32_CFG, momentum_dtype="4bit", momentum_4bit_block=block)
    pv, pn, ov, on = _pair([(64, 128)] * 2, cfg, seed=23)   # (64,128) == TILE_CAP -> one-block
    _drive([(pv, ov), (pn, on)], 4, torch.Generator(device=DEV).manual_seed(29))
    ob, _big, _od, nat = _parts(ov)
    assert len(ob) == 2 and not nat, "every 4-bit block size must take the one-block kernel"
    per = 64 * 128
    expect_block = per if block == 0 else block
    for p in pv:
        st = ov.state[p]
        assert st["m_block"] == expect_block
        assert st["m_scale"].numel() == (per + expect_block - 1) // expect_block
    # Same strict bound the pre-0.7.12 (native-routed) version of this test used: the runtime
    # block reproduces the codec exactly, so the measured gap is 2.4e-7 absolute for every
    # block size here. A looser bound would hide a genuine block-indexing bug.
    d = _maxdiff(pv, pn)
    assert d < 1e-5, f"max|Δp|={d:.2e}"


def test_4bit_block_128_still_takes_the_one_block_route():
    """The DEFAULT 4-bit configuration keeps its fused route (and its numbers)."""
    pv, pn, ov, on = _pair([(64, 128)] * 2, dict(_FP32_CFG, momentum_dtype="4bit"), seed=23)
    _drive([(pv, ov), (pn, on)], 4, torch.Generator(device=DEV).manual_seed(29))
    ob, _big, _od, nat = _parts(ov)
    assert len(ob) == 2 and not nat
    scale = max(p.detach().abs().max().item() for p in pn)
    d = _maxdiff(pv, pn)
    assert d / scale < 8e-4, f"rel={d / scale:.2e}"


def test_4bit_blocks_of_different_sizes_get_their_own_launch():
    """One launch carries ONE runtime block, so a group mixing layouts must bucket by it.

    Two same-tile tensors whose ``m_block`` differs (here: a checkpoint-style layout planted
    into one tensor's state) would otherwise share a launch and one of them would be
    dequantized against the other's block size.

    The native reference runs ``foreach=False`` on purpose: the codec's stacked path
    (``_FourBitCodec.ema_stacked``) stacks ``m_scale`` across the bucket and cannot represent a
    group whose members carry different block layouts at all. That is a pre-existing native
    limitation, orthogonal to the routing property under test here.
    """
    cfg = dict(_FP32_CFG, momentum_dtype="4bit", momentum_4bit_block=64)
    pv = _bag([(64, 128)] * 2, torch.float32, 41)
    pn = _clone(pv)
    ov, on = Adakaon(pv, fused=True, **cfg), Adakaon(pn, foreach=False, **cfg)
    _drive([(pv, ov), (pn, on)], 1, torch.Generator(device=DEV).manual_seed(43))
    # Re-init one tensor's momentum with a DIFFERENT block layout, as a resumed checkpoint would.
    for opt, plist in ((ov, pv), (on, pn)):
        st = opt.state[plist[0]]
        st["m_block"] = 128
        st["m_scale"] = torch.ones(64 * 128 // 128, dtype=torch.float32, device=DEV)
        st["m"].fill_(0x88)
    ov._invalidate_fused_caches()
    _drive([(pv, ov), (pn, on)], 3, torch.Generator(device=DEV).manual_seed(47))
    buckets = ov._fused_ob_caches[id(ov.param_groups[0])].buckets
    assert len(buckets) == 2, f"mixed m_block must give 2 buckets, got {len(buckets)}"
    assert sorted(b["blk"] for b in buckets) == [64, 128]
    # 1.2e-7 measured; a tensor dequantized against the OTHER bucket's block would be off by
    # the difference between two absmax scales, orders of magnitude above this.
    d = _maxdiff(pv, pn)
    assert d < 1e-6, f"max|Δp|={d:.2e}"


def test_exact_requant_rejects_a_block_that_does_not_divide_the_tile():
    """Why ``PointerArrayCache.exact4`` has to test divisibility: Triton refuses to compile.

    ``requant_4bit``'s ``EXACT`` path reshapes the ``[BR, BC]`` tile to ``(BR*BC // FBLK, FBLK)``.
    That is only a reshape when ``FBLK`` divides ``BR*BC``; otherwise it is a ``CompilationError``
    at launch, i.e. a hard crash in the middle of a training run. Before 0.7.12 the invariant was
    free (``FBLK`` was a hardcoded ``min(BR*BC, 128)`` against a power-of-two tile); now the block
    comes from ``state["m_block"]`` and the host has to enforce it.
    """
    R = C = BR = BC = 16  # noqa: N806
    m = torch.randn(R * C, device=DEV)
    packed = torch.zeros(R * C // 2, dtype=torch.uint8, device=DEV)

    def drive(fblk):
        nb = (R * C + fblk - 1) // fblk
        scale = torch.ones(max(nb, 1), dtype=torch.float32, device=DEV)
        _requant_4bit_exact_probe[(1,)](m, packed, scale, R, C, C // 2, nb, scale.numel(),
                                        fblk, BR=BR, BC=BC, FBLK=fblk)
        torch.cuda.synchronize()

    drive(64)                       # divides 256 -> compiles and runs
    for fblk in (96, 48, 24):       # do NOT divide 256
        with pytest.raises(triton.compiler.errors.CompilationError):
            drive(fblk)


def test_4bit_block_that_does_not_divide_the_tile_still_steps():
    """The host guard in action, end to end.

    A (64,128) weight at ``momentum_4bit_block=96`` has ``8192 % 96 != 0``, so ``exact4`` must
    stay False and the general segmented-absmax loop must be used. Drop the divisibility term
    from ``PointerArrayCache``'s ``exact4`` predicate and this raises ``CompilationError``.
    """
    cfg = dict(_FP32_CFG, momentum_dtype="4bit", momentum_4bit_block=96)
    pv, pn, ov, on = _pair([(64, 128)] * 2, cfg, seed=61)
    _drive([(pv, ov), (pn, on)], 3, torch.Generator(device=DEV).manual_seed(67))
    ob, _big, _od, nat = _parts(ov)
    assert len(ob) == 2 and not nat, "a non-dividing block still belongs on the one-block route"
    bucket = ov._fused_ob_caches[id(ov.param_groups[0])].buckets[0]
    assert bucket["blk"] == 96
    assert not bucket["exact4"], "96 does not divide the 64x128 tile — EXACT must be off"
    assert bucket["fblk"] == 0
    d = _maxdiff(pv, pn)
    assert d < 1e-5, f"max|Δp|={d:.2e}"


def test_4bit_odd_column_count_still_leaves_the_one_block_route():
    """The nibble packing assumes an even column count; odd C keeps going native."""
    cfg = dict(_FP32_CFG, momentum_dtype="4bit", momentum_4bit_block=64)
    pv, pn, ov, on = _pair([(8, 15)] * 2, cfg, seed=53)
    _drive([(pv, ov), (pn, on)], 3, torch.Generator(device=DEV).manual_seed(59))
    ob, _big, _od, nat = _parts(ov)
    assert not ob and len(nat) == 2, "odd C must not take the one-block 4-bit kernel"
    scale = max(p.detach().abs().max().item() for p in pn)
    d = _maxdiff(pv, pn)
    assert d / scale < 8e-4, f"rel={d / scale:.2e}"


# ----------------------------------------------------------------- 4. non-contiguous grads
def _transposed_grads(gen, plist):
    return [torch.randn((p.shape[1], p.shape[0]), generator=gen, device=DEV, dtype=p.dtype).t()
            for p in plist]


def _strided_grads(gen, plist):
    return [torch.randn((p.shape[0] * 2,), generator=gen, device=DEV, dtype=p.dtype)[::2]
            for p in plist]


@pytest.mark.parametrize(
    "shapes,grads,route",
    [
        ([(8, 16)] * 3, _transposed_grads, "one_block"),
        ([(512, 512)] * 2, _transposed_grads, "big"),
        ([(64,)] * 3, _strided_grads, "one_dim"),
    ],
)
def test_non_contiguous_grads_match_native(shapes, grads, route):
    """The kernels read the grad from ``data_ptr()`` row-major — strides are invisible to them,
    so a non-contiguous grad must not reach them (it steps the transposed numbers instead)."""
    pv, pn, ov, on = _pair(shapes, _FP32_CFG, seed=31)
    gen = torch.Generator(device=DEV).manual_seed(37)
    assert not any(g.is_contiguous() for g in grads(gen, pv)), "the fixture grads are contiguous"
    _drive([(pv, ov), (pn, on)], 5, gen, grads_for=grads)
    d = _maxdiff(pv, pn)
    assert d < 1e-6, f"{route}: max|Δp|={d:.2e} with non-contiguous grads"


def test_non_contiguous_grad_does_not_poison_the_contiguous_neighbours():
    """One strided grad demotes ONLY its own tensor for that step."""
    pv, pn, ov, on = _pair([(8, 16)] * 3, _FP32_CFG, seed=41)

    def grads(gen, plist):
        gs = _plain_grads(gen, plist)
        gs[1] = torch.randn((16, 8), generator=gen, device=DEV, dtype=torch.float32).t()
        return gs

    _drive([(pv, ov), (pn, on)], 5, torch.Generator(device=DEV).manual_seed(43), grads_for=grads)
    d = _maxdiff(pv, pn)
    assert d < 1e-6, f"max|Δp|={d:.2e}"


# ----------------------------------------------------------------- 5. fp16 + SR
def test_fp16_params_with_stochastic_rounding_raise():
    """``kaon.add_stochastic_`` has no fp16 implementation and ``sr_round`` is bf16-only:
    the step silently fell back to round-to-nearest instead of saying so."""
    p = torch.zeros(4, 4, dtype=torch.float16, device=DEV, requires_grad=True)
    with pytest.raises(NotImplementedError, match="float16"):
        Adakaon([p], bf16_method="stochastic_rounding")
    with pytest.raises(NotImplementedError, match="float16"):
        Adakaon([torch.zeros(4, 4, device=DEV, requires_grad=True)]).add_param_group(
            {"params": [p]}
        )
    Adakaon([p], bf16_method="kahan")            # the supported fp16 routes still construct
    Adakaon([p], bf16_method="none")


# ----------------------------------------------------------------- 6. non-finite propagation
@pytest.mark.parametrize("route", ["one_block", "one_dim", "big"])
def test_non_finite_grad_propagates_like_native(route):
    """Finiteness policy is PROPAGATE, and it has to be the same policy on both paths.

    ``tl.where(keep, delta, 0)`` made the fused cautious mask swallow a NaN delta and FREEZE
    the tensor, while native's ``delta.mul_(mask)`` (0 * NaN == NaN) propagates it. A frozen
    tensor is a silently dead weight; a NaN one is a training run that stops and gets fixed.
    """
    pv, pn, ov, on = _pair(_ROUTES[route], _FP32_CFG, seed=47)

    def grads(gen, plist):
        gs = _plain_grads(gen, plist)
        gs[0].reshape(-1)[0] = float("inf")
        return gs

    _drive([(pv, ov), (pn, on)], 2, torch.Generator(device=DEV).manual_seed(53), grads_for=grads)
    fused_finite = torch.isfinite(pv[0].detach().float()).all().item()
    native_finite = torch.isfinite(pn[0].detach().float()).all().item()
    assert native_finite is False, "the native reference is expected to propagate"
    assert fused_finite == native_finite, "fused froze the tensor where native propagated"


@pytest.mark.parametrize(
    "bits,kind",
    [
        (0x7FC00000, "nan"),        # canonical quiet NaN
        (0xFFC00000, "nan"),        # negative quiet NaN
        (0x7FFFFFFF, "nan"),        # NaN with a full payload: +noise overflows int32 -> -0.0
        (0x7F800000, "inf"),
        (0xFF800000, "inf"),
    ],
)
def test_sr_round_preserves_non_finite(bits, kind):
    """``sr_round``'s int32 bit-trick must not manufacture a finite value out of NaN/inf."""
    n = 256
    signed = bits - (1 << 32) if bits >= (1 << 31) else bits
    src = torch.full((n,), signed, dtype=torch.int32).view(torch.float32).to(DEV)
    out = torch.zeros(n, dtype=torch.float32, device=DEV)
    _sr_round_passthrough[(1,)](src, out, 1234, n, BLOCK=n)
    torch.cuda.synchronize()
    if kind == "nan":
        finite = out[~torch.isnan(out)]
        assert finite.numel() == 0, f"{finite[:4].tolist()} came back non-NaN"
    else:
        assert torch.equal(out, src)


def test_sr_round_still_rounds_finite_values():
    """The non-finite guard must not disturb the ordinary stochastic-rounding draw."""
    n = 4096
    val = 1.0 + 3.0 / 512.0                     # strictly between two bf16 grid points
    src = torch.full((n,), val, dtype=torch.float32, device=DEV)
    out = torch.zeros(n, dtype=torch.float32, device=DEV)
    _sr_round_passthrough[(1,)](src, out, 99, n, BLOCK=n)
    torch.cuda.synchronize()
    grid = torch.unique(out)
    assert grid.numel() == 2, f"expected the two bracketing bf16 values, got {grid.tolist()}"
    assert torch.equal(out.to(torch.bfloat16).float(), out)
    assert abs(out.mean().item() - val) < 3e-4, "the draw is no longer unbiased"


# ----------------------------------------------------------------- 7. mixed-device group
@pytest.mark.parametrize("fused", [False, True])
def test_multi_device_group_foreach_step(fused):
    """One param group holding CPU and CUDA weights of the same shape must step both.

    The native foreach path stacks a bucket with ``torch.stack``, and the bucket key was
    ``(shape, dtype)`` with no device in it: a CPU and a CUDA weight of the same shape landed
    in one bucket and the stack raised "Expected all tensors to be on the same device", taking
    the whole step down instead of stepping each on its own device. The fused pointer caches
    bucket by device for the same reason (their index arrays live on the bucket's device).

    ``gradient_centralization`` is off here: the GC pre-pass has the same shape-only bucketing
    in ``kaon._backend.centralize_grads_``, and that fix is a separate change to a file this
    one does not touch.
    """
    params = [
        torch.randn(8, 8).requires_grad_(True),
        torch.randn(8, 8, device=DEV).requires_grad_(True),
        torch.randn(8).requires_grad_(True),
        torch.randn(8, device=DEV).requires_grad_(True),
    ]
    opt = Adakaon(params, lr=1e-2, fused=fused, gradient_centralization=False)
    before = [p.detach().clone() for p in params]
    gen = torch.Generator().manual_seed(61)
    for _ in range(2):
        for p in params:
            p.grad = torch.randn(tuple(p.shape), generator=gen).to(p.device)
        opt.step()
    torch.cuda.synchronize()
    for p, b in zip(params, before, strict=True):
        assert p.device == b.device, "a param changed device"
        assert torch.isfinite(p.detach()).all()
        assert not torch.equal(p.detach(), b), f"{tuple(p.shape)} on {p.device} never moved"


# ================================================================= rework (external review)
# ----------------------------------------------------------------- 8. NS bounds BOTH ends
def test_requant_4bit_capacity_bounds_the_scale_load_too():
    """``NS`` has to bound the scale LOAD, not just the store.

    With ``NB > NS`` the store is masked (nothing is written past ``m_scale``) but the
    per-lane reload ``scale_ptr + blk`` still addressed the missing blocks: an out-of-range
    READ whose value then divides the momentum. Here the word just past the buffer holds a
    sentinel, so an unmasked load quantizes block 1 against 1e30 and collapses every code to
    zero; the masked load falls back to the neutral 1.0.
    """
    R, C, BLK = 2, 128, 128                                            # noqa: N806
    NB, NS = 2, 1                    # noqa: N806 — the kernel writes 2 blocks, the buffer holds 1
    sentinel = 1e30
    m = torch.empty(R * C, dtype=torch.float32, device=DEV)
    m[:BLK] = 7.0                                # block 0: absmax 7 -> scale 1.0 -> code 7
    m[BLK:] = 3.0                                # block 1: must quantize against the neutral 1.0
    scale = torch.tensor([0.0, sentinel], dtype=torch.float32, device=DEV)
    untouched = scale[1].clone()                 # the fp32 round-trip of the sentinel
    packed = torch.zeros(R * C // 2, dtype=torch.uint8, device=DEV)
    _requant_4bit_probe[(1,)](m, packed, scale, R, C, C // 2, NB, NS, BLK, BR=2, BC=128)
    torch.cuda.synchronize()
    assert torch.equal(scale[1], untouched), "the scale STORE wrote past the buffer"
    assert abs(scale[0].item() - 1.0) < 1e-6
    lo, hi = packed[:BLK // 2], packed[BLK // 2:]
    assert torch.all(lo == 0xFF), f"block 0 (in range) mis-quantized: {lo[:4].tolist()}"
    # code 3 -> nibble 3+8 = 11 -> 0xBB. An unmasked load would read 1e30 and give 0 -> 0x88.
    assert torch.all(hi == 0xBB), f"block 1 (out of range) read past m_scale: {hi[:4].tolist()}"


# ----------------------------------------------------------------- 9. big buckets by device
def test_big_shape_buckets_split_by_device():
    """``_fused_big`` groups by EXACT shape before handing a list to ``BigPointerCache``, whose
    index arrays live on ``plist[0].device``. Without the device in that key, two GPUs of the
    same shape would be launched against pointer arrays built on the first one.

    Exercised on the grouping helper directly: reaching it end-to-end needs two CUDA devices
    (the big route requires ``p.is_cuda``), and this machine has one.
    """
    from kaon.adakaon import _same_shape_device_buckets

    a = torch.zeros(64, 64, device=DEV)
    b = torch.zeros(64, 64, device=DEV)
    c = torch.zeros(64, 64)                       # same shape + dtype, different device
    d = torch.zeros(64, 64, device=DEV, dtype=torch.bfloat16)
    buckets = _same_shape_device_buckets([a, b, c, d])
    assert len(buckets) == 3, f"expected shape x dtype x device buckets, got {len(buckets)}"
    assert sorted(len(v) for v in buckets.values()) == [1, 1, 2]
    for plist in buckets.values():
        assert len({p.device for p in plist}) == 1


# ----------------------------------------------------------------- 10. witness sees layout
def test_transpose_rebind_keeps_the_same_pointer_but_must_reroute():
    """``p.data = p.data.t()`` on a SQUARE weight keeps ``id`` AND ``data_ptr`` AND ``shape``.

    Only the strides change — and the fused kernels index row-major from ``data_ptr``, so the
    param must leave the fused routes the moment it stops being contiguous. An (id, data_ptr)
    witness cannot see it: the cached partition kept dispatching it to the one-block kernel.
    """
    pv, pn, ov, on = _pair([(32, 32)] * 2, _FP32_CFG, seed=67)

    def mutate(step, _pairs):
        if step != 2:
            return
        for plist in (pv, pn):
            plist[0].data = plist[0].data.t()

    _drive([(pv, ov), (pn, on)], 5, torch.Generator(device=DEV).manual_seed(71), mutate=mutate)
    assert not pv[0].is_contiguous()
    ob, _big, _od, nat = _parts(ov)
    assert any(q is pv[0] for q in nat), "the transposed param stayed on a fused route"
    assert any(q is pv[1] for q in ob), "its contiguous neighbour lost the fused route"
    d = _maxdiff(pv, pn)
    assert d < 1e-6, f"max|Delta p|={d:.2e} after a transpose rebind"


def _shape_rebind_step(fused):
    """Two normal steps, then ``p.data = p.data.view(...)`` and one more. Returns the exception
    the last step raised, or None."""
    ps = _bag([(16, 64)] * 2, torch.float32, seed=73)
    opt = Adakaon(ps, fused=fused, **_FP32_CFG)
    _drive([(ps, opt)], 2, torch.Generator(device=DEV).manual_seed(79))
    ptr = ps[0].data_ptr()
    ps[0].data = ps[0].data.view(64, 16)
    assert ps[0].data_ptr() == ptr and ps[0].is_contiguous(), "the rebind moved more than shape"
    for p in ps:
        p.grad = torch.randn(tuple(p.shape), device=DEV)
    try:
        opt.step()
        torch.cuda.synchronize()
    except RuntimeError as exc:
        return opt, ps, exc
    return opt, ps, None


def test_shape_changing_rebind_is_caught_on_the_native_path():
    """DOCUMENTED LIMIT: ``p.data = p.data.view(...)`` mid-training is NOT supported.

    Such a rebind keeps the id, the ``data_ptr`` AND contiguity, so no witness field short of
    collecting ``torch.Size`` per param every step sees it — and that field was measured at 123 µs
    of a 254 µs witness on a 1633 µs step (see ``kaon._foreach_plan.param_witness`` and
    ``Adakaon._fused_partition``), which is not a price worth
    paying for an unsupported operation. Watching it would not even fix the case: the factored
    ``row``/``col`` state is bound to the OLD effective 2-D shape and cannot be migrated, so
    rebuilding the plan just points the new R/C at the old buffers and writes past them.

    What IS guaranteed is that the native path refuses rather than steps: its bucketing stacks by
    the effective shape and the stack raises. See the companion xfail for the fused path.
    """
    _opt, _ps, exc = _shape_rebind_step(fused=False)
    assert exc is not None, "the native path silently stepped a reshaped weight"
    assert "size" in str(exc).lower()


@pytest.mark.xfail(
    reason="documented limit: a shape-changing p.data rebind is unsupported and the fused path "
           "does not detect it — the cached partition keeps stepping the PRE-rebind geometry "
           "(R=16, C=64), which stays inside the row/col buffers so nothing faults, but the "
           "weight is no longer being optimized as the shape the caller now sees. Follow-up: "
           "validate row/col lengths against p in the fused plan, or migrate the state.",
    strict=True,
)
def test_shape_changing_rebind_is_caught_on_the_fused_path():
    """The fused mirror of the test above — expected to fail until the follow-up lands."""
    opt, ps, exc = _shape_rebind_step(fused=True)
    st = opt.state[ps[0]]
    assert st["row"].numel() == 16 and st["col"].numel() == 64   # state kept the old geometry
    assert exc is not None, "the fused path silently stepped a reshaped weight"


# ----------------------------------------------------------------- 11. sr_round bit fidelity
@pytest.mark.parametrize(
    "bits",
    [0x7FC00000, 0xFFC00000, 0x7FFFFFFF, 0x7F800001, 0xFF800001, 0x7F800000, 0xFF800000],
)
def test_sr_round_returns_non_finite_bits_untouched(bits):
    """Zeroing the noise is not enough: the mantissa mask itself turns a low-payload NaN
    (``0x7F800001``) into +inf. A non-finite input has to come back bit-for-bit."""
    n = 256
    signed = bits - (1 << 32) if bits >= (1 << 31) else bits
    src = torch.full((n,), signed, dtype=torch.int32).view(torch.float32).to(DEV)
    out = torch.zeros(n, dtype=torch.float32, device=DEV)
    _sr_round_passthrough[(1,)](src, out, 4321, n, BLOCK=n)
    torch.cuda.synchronize()
    got = out.view(torch.int32).cpu()
    assert torch.all(got == signed), f"{hex(bits)} -> {hex(got[0].item() & 0xFFFFFFFF)}"


# ----------------------------------------------------------------- 12. add_param_group atomic
def test_rejected_param_group_is_not_added():
    """The fp16 check must run BEFORE the group is appended: catching the error must not leave
    the optimizer holding a group it cannot step."""
    good = torch.zeros(4, 4, device=DEV, requires_grad=True)
    bad = torch.zeros(4, 4, dtype=torch.float16, device=DEV, requires_grad=True)
    opt = Adakaon([good], lr=1e-3, bf16_method="stochastic_rounding")
    with pytest.raises(NotImplementedError, match="float16"):
        opt.add_param_group({"params": [bad]})
    assert len(opt.param_groups) == 1
    assert not any(q is bad for g in opt.param_groups for q in g["params"])
    good.grad = torch.randn_like(good)
    opt.step()                                     # still usable


def test_non_dict_param_group_keeps_torch_error():
    """The fp16 pre-check indexes ``param_group["params"]``; a non-dict has to reach torch so the
    caller still gets ``TypeError: param group must be a dict``, not a KeyError/TypeError from us."""
    opt = Adakaon([torch.zeros(4, 4, device=DEV, requires_grad=True)], lr=1e-3)
    with pytest.raises(TypeError, match="must be a dict"):
        opt.add_param_group([torch.zeros(4, 4, device=DEV, requires_grad=True)])


def test_param_group_params_may_be_a_generator():
    """``params`` is materialized by the pre-check; torch must still see the full list."""
    ws = [torch.zeros(4, 4, device=DEV, requires_grad=True) for _ in range(3)]
    opt = Adakaon([torch.zeros(4, 4, device=DEV, requires_grad=True)], lr=1e-3)
    opt.add_param_group({"params": (w for w in ws)})
    assert len(opt.param_groups[1]["params"]) == 3


# ----------------------------------------------------------------- 13. demotion is cached
def test_persistent_strided_grad_does_not_rebuild_the_cache_every_step():
    """A tensor demoted for a non-contiguous grad rebuilds the route lists; while the demoted
    SET is unchanged those lists must be reused, or every step throws away the pointer arrays
    of the whole bucket (and rebuilds a handful of index tensors) for nothing."""
    pv = _bag([(8, 16)] * 4, torch.float32, seed=83)
    ov = Adakaon(pv, fused=True, **_FP32_CFG)

    def grads(gen, plist):
        gs = _plain_grads(gen, plist)
        gs[1] = torch.randn((16, 8), generator=gen, device=DEV, dtype=torch.float32).t()
        return gs

    gen = torch.Generator(device=DEV).manual_seed(89)
    _drive([(pv, ov)], 2, gen, grads_for=grads)
    seen = [next(iter(ov._fused_ob_caches.values()))]
    for _ in range(3):
        _drive([(pv, ov)], 1, gen, grads_for=grads)
        seen.append(next(iter(ov._fused_ob_caches.values())))
    assert all(c is seen[0] for c in seen), "the one-block cache was rebuilt on a stable set"


def test_alternating_contiguous_and_strided_grads_match_native():
    """contiguous -> strided -> contiguous on consecutive steps: the demotion is per step, and
    its cache must not keep a tensor on the native path once its grad is contiguous again."""
    pv, pn, ov, on = _pair([(8, 16)] * 3, _FP32_CFG, seed=97)
    state = {"step": 0}

    def grads(gen, plist):
        gs = _plain_grads(gen, plist)
        if state["step"] % 2 == 1:
            gs[0] = torch.randn((16, 8), generator=gen, device=DEV, dtype=torch.float32).t()
        return gs

    gen = torch.Generator(device=DEV).manual_seed(101)
    for _ in range(6):
        _drive([(pv, ov), (pn, on)], 1, gen, grads_for=grads)
        state["step"] += 1
    d = _maxdiff(pv, pn)
    assert d < 1e-6, f"max|Delta p|={d:.2e} across alternating grad layouts"


# ----------------------------------------------------------------- 14. shape corners
@pytest.mark.parametrize("shapes", [[(10, 20)] * 2, [(6, 34)] * 2])
def test_4bit_partial_last_block_parity(shapes):
    """numel not a multiple of 128: the last absmax block is partial on both paths."""
    per = shapes[0][0] * shapes[0][1]
    assert per % 128 != 0
    pv, pn, ov, on = _pair(shapes, dict(_FP32_CFG, momentum_dtype="4bit"), seed=103)
    _drive([(pv, ov), (pn, on)], 4, torch.Generator(device=DEV).manual_seed(107))
    ob, _big, _od, nat = _parts(ov)
    assert len(ob) == 2 and not nat
    for p in pv:
        assert ov.state[p]["m_scale"].numel() == (per + 127) // 128
    scale = max(p.detach().abs().max().item() for p in pn)
    d = _maxdiff(pv, pn)
    assert d / scale < 8e-4, f"rel={d / scale:.2e}"


@pytest.mark.parametrize("shapes", [[(1, 64)] * 3, [(64, 1)] * 3, [(1, 1)] * 3,
                                    [(1, 20000)] * 2, [(20000, 1)] * 2])
def test_degenerate_2d_shapes_match_native(shapes):
    """R == 1 and C == 1 exercise the row/col reductions at their limits (``row_mean_all`` over
    a single row, ``reduction_tile`` with one row per program) on both the one-block and the
    big route."""
    pv, pn, ov, on = _pair(shapes, _FP32_CFG, seed=109)
    _drive([(pv, ov), (pn, on)], 4, torch.Generator(device=DEV).manual_seed(113))
    ob, big, od, nat = _parts(ov)
    assert (ob or big) and not nat, "these are 2-D and must stay on a fused route"
    d = _maxdiff(pv, pn)
    assert d < 1e-5, f"{shapes[0]}: max|Delta p|={d:.2e}"
