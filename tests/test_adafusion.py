"""Tests for the Adafusion optimizer."""

from __future__ import annotations

import io
import math

import pytest
import torch

from koptim import Adafusion

from .conftest import train_steps


def test_conv_factoring_reduces_state():
    """Conv-aware factoring stores ~0 state for a conv kernel.

    A 4-D kernel [out, in, kh, kw] is reshaped to [out, in*kh*kw] and factored to
    row+col EMAs (out + in*kh*kw floats), far below the full out*in*kh*kw numel.
    """
    p = torch.nn.Parameter(torch.randn(64, 32, 3, 3))
    opt = Adafusion([p], lr=1e-3, betas=(0.0, 0.999))
    p.grad = torch.randn_like(p)
    opt.step()
    state_floats = sum(v.numel() for v in opt.state[p].values() if torch.is_tensor(v))
    # row (64) + col (288); well under 1/50 of the full 18432 numel.
    assert state_floats < p.numel() / 50, f"conv state should be tiny: {state_floats} vs {p.numel()}"


def test_bf16_momentum_is_half_state():
    """bf16 momentum buffer is half the bytes of fp32 momentum."""
    def mom_bytes(dtype: str) -> int:
        p = torch.nn.Parameter(torch.randn(128, 128))
        opt = Adafusion([p], lr=1e-3, betas=(0.9, 0.999), momentum_dtype=dtype)
        p.grad = torch.randn_like(p)
        opt.step()
        return opt.state[p]["m"].numel() * opt.state[p]["m"].element_size()

    assert mom_bytes("bfloat16") * 2 == mom_bytes("float32")


def test_overfits_regression():
    torch.manual_seed(0xC0DE)
    model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 8))
    opt = Adafusion(model.parameters(), lr=3e-3, betas=(0.9, 0.999))
    x = torch.randn(64, 32)
    y = torch.randn(64, 8)
    initial = (model(x) - y).pow(2).mean().item()
    train_steps(model, opt, [(x, y)] * 80)
    final = (model(x) - y).pow(2).mean().item()
    assert final < 0.5 * initial, f"loss did not drop: {initial:.4f} -> {final:.4f}"


def test_conv_net_trains_no_nan():
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(4, 16, 3, padding=1), torch.nn.GELU(),
        torch.nn.Conv2d(16, 4, 3, padding=1),
    )
    opt = Adafusion(net.parameters(), lr=3e-3, betas=(0.9, 0.999))
    x = torch.randn(8, 4, 16, 16)
    y = torch.randn(8, 4, 16, 16)
    for _ in range(30):
        opt.zero_grad()
        loss = (net(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
    assert math.isfinite(loss.item())


def test_bf16_weights_train_no_nan():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 8)).to(torch.bfloat16)
    opt = Adafusion(model.parameters(), lr=3e-3, betas=(0.9, 0.999), bf16_method="stochastic_rounding")
    x = torch.randn(64, 32, dtype=torch.bfloat16)
    y = torch.randn(64, 8, dtype=torch.bfloat16)
    for _ in range(30):
        opt.zero_grad()
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)


def test_cautious_runs():
    torch.manual_seed(0)
    model = torch.nn.Linear(16, 16)
    opt = Adafusion(model.parameters(), lr=1e-3, betas=(0.9, 0.999), cautious=True)
    x = torch.randn(8, 16)
    (model(x)).pow(2).mean().backward()
    opt.step()  # must not raise


def _parity_params():
    """A mix that exercises every fast-path branch (factored, conv, and 1-D).

    Includes repeated 2-D shapes and repeated 1-D lengths (so buckets have N>1),
    distinct lengths, and a conv (matrixize).
    """
    g = torch.Generator().manual_seed(0)
    shapes = [
        (64, 128), (128, 64), (64, 128),      # 2-D, one shape repeated -> bucket N=2
        (32, 8, 3, 3),                        # conv (matrixize)
        (8, 96), (96, 8),                     # LoRA-like 2-D
        (40,), (40,), (128,), (320,),         # 1-D: repeated length + distinct lengths
    ]
    return [torch.nn.Parameter(torch.randn(*s, generator=g) * 0.05) for s in shapes]


@pytest.mark.parametrize(
    "cfg",
    [
        dict(lr=1e-3, betas=(0.0, 0.999)),                                  # no momentum
        dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="float32"),        # fp32 momentum
        dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="bfloat16"),       # bf16 momentum
        dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="int8"),           # int8 momentum
        dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="int8", weight_decay=0.02),  # int8 + wd
        dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="4bit"),           # 4-bit momentum
        dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="4bit", weight_decay=0.02),  # 4bit + wd
        dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="4bit", momentum_4bit_block=64),
        dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="4bit", momentum_4bit_block=0),  # whole-tensor
        dict(lr=1e-3, betas=(0.9, 0.999), weight_decay=0.02),               # weight decay
        dict(lr=1e-3, betas=(0.9, 0.999), cautious=True),                   # cautious mask
    ],
)
def test_foreach_matches_per_param(cfg):
    """foreach=True is element-for-element equal to the per-parameter path.

    fp32 params keep stochastic rounding a no-op, so the only difference between
    the two code paths would be a real bug. Bit-exact on CPU.
    """
    pa = _parity_params()
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    oa = Adafusion(pa, foreach=True, **cfg)
    ob = Adafusion(pb, foreach=False, **cfg)
    gg = torch.Generator().manual_seed(7)
    for _ in range(10):
        for a, b in zip(pa, pb, strict=False):
            grad = torch.randn(*a.shape, generator=gg) * 0.02
            a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb, strict=False):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


def test_foreach_chunking_is_exact():
    """A tiny stack budget forces buckets to split and large weights to route to
    the loop — the result must still equal the per-parameter path exactly."""
    pa = _parity_params()
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    # budget=200 elems: every 2-D shape splits into many chunks and the larger
    # tensors fall to the per-param branch — a stress test of both code paths.
    oa = Adafusion(pa, lr=1e-3, betas=(0.9, 0.999), foreach=True, foreach_stack_budget=200)
    ob = Adafusion(pb, lr=1e-3, betas=(0.9, 0.999), foreach=False)
    gg = torch.Generator().manual_seed(7)
    for _ in range(8):
        for a, b in zip(pa, pb, strict=False):
            grad = torch.randn(*a.shape, generator=gg) * 0.02
            a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb, strict=False):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


def test_foreach_int8_chunking_is_exact():
    """int8 momentum: a tiny stack budget splits buckets and routes large tensors
    to the per-param loop — the batched int8 requant must still match per-param."""
    pa = _parity_params()
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    oa = Adafusion(pa, lr=1e-3, betas=(0.9, 0.999), momentum_dtype="int8",
                   foreach=True, foreach_stack_budget=200)
    ob = Adafusion(pb, lr=1e-3, betas=(0.9, 0.999), momentum_dtype="int8", foreach=False)
    gg = torch.Generator().manual_seed(7)
    for _ in range(8):
        for a, b in zip(pa, pb, strict=False):
            grad = torch.randn(*a.shape, generator=gg) * 0.02
            a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb, strict=False):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


def test_foreach_4bit_chunking_is_exact():
    """4-bit momentum: a tiny stack budget splits buckets and routes large tensors
    to the per-param loop — the batched 4-bit pack/dequant/EMA/requant must still
    match the per-param path bit-for-bit."""
    pa = _parity_params()
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    oa = Adafusion(pa, lr=1e-3, betas=(0.9, 0.999), momentum_dtype="4bit",
                   foreach=True, foreach_stack_budget=200)
    ob = Adafusion(pb, lr=1e-3, betas=(0.9, 0.999), momentum_dtype="4bit", foreach=False)
    gg = torch.Generator().manual_seed(7)
    for _ in range(8):
        for a, b in zip(pa, pb, strict=False):
            grad = torch.randn(*a.shape, generator=gg) * 0.02
            a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb, strict=False):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


def test_4bit_pack_roundtrip():
    """Nibble pack/unpack round-trips for even and odd element counts, and
    dequant(quant(m)) stays within the ~1/7 absmax 4-bit grid error."""
    from koptim.adafusion import (
        _dequant_4bit,
        _pack_nibbles,
        _quant_4bit,
        _unpack_nibbles,
    )

    g = torch.Generator().manual_seed(3)
    for k in (1, 2, 3, 7, 8, 9, 127, 128, 129):
        nib = torch.randint(0, 16, (k,), generator=g, dtype=torch.uint8)
        packed = _pack_nibbles(nib)
        assert packed.numel() == (k + 1) // 2
        assert torch.equal(_unpack_nibbles(packed, k), nib)
    for shape in [(64, 128), (7,), (32, 8, 3, 3), (129,)]:
        m = torch.randn(*shape, generator=g)
        packed, scale, numel = _quant_4bit(m, 128)
        rec = _dequant_4bit(packed, scale, numel, 128).view(shape)
        # per-block grid step is absmax/7; reconstruction error must be bounded by it.
        assert (rec - m).abs().max() <= m.abs().max() / 7.0 / 2.0 + 1e-6


def test_4bit_memory_is_half_byte_per_param():
    """The 4-bit store is a real 0.5 B/param packed buffer plus small block scales."""
    p = torch.nn.Parameter(torch.randn(512, 512))
    opt = Adafusion([p], betas=(0.9, 0.999), momentum_dtype="4bit", momentum_4bit_block=128)
    p.grad = torch.randn_like(p)
    opt.step()
    st = opt.state[p]
    assert st["m"].dtype == torch.uint8
    assert st["m"].numel() == (p.numel() + 1) // 2          # exactly 0.5 B/param packed
    packed_bpp = st["m"].numel() / p.numel()
    scale_bpp = st["m_scale"].numel() * st["m_scale"].element_size() / p.numel()
    assert packed_bpp == 0.5
    assert packed_bpp + scale_bpp < 0.55                    # total well under int8's 1.0


def test_4bit_trains_no_nan():
    """A tiny regression converges with 4-bit momentum, no NaN."""
    torch.manual_seed(0)
    x = torch.randn(256, 16)
    y = x @ torch.randn(16, 1)
    model = torch.nn.Linear(16, 1)
    opt = Adafusion(model.parameters(), lr=1e-2, betas=(0.9, 0.999), momentum_dtype="4bit")
    first = last = None
    for _ in range(200):
        opt.zero_grad()
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
    assert math.isfinite(last) and last < first


def test_4bit_invalid_momentum_dtype_rejected():
    p = [torch.nn.Parameter(torch.randn(4, 4))]
    with pytest.raises(ValueError):
        Adafusion(p, momentum_dtype="2bit")


def test_foreach_batch_cutoff_validation():
    p = [torch.nn.Parameter(torch.randn(4, 4))]
    Adafusion(p, foreach_batch_cutoff=1)
    Adafusion(p, foreach_batch_cutoff=5_000_000)
    with pytest.raises(ValueError):
        Adafusion(p, foreach_batch_cutoff=0)


def test_foreach_budget_capped_at_4x_cutoff():
    """The adaptive chunk budget never exceeds 4x the cutoff (over-stacking guard)
    and scales with the cutoff. Deterministic on CPU (no VRAM read)."""
    cpu = torch.device("cpu")
    p = [torch.nn.Parameter(torch.randn(4, 4))]
    assert Adafusion(p, foreach_batch_cutoff=2_000_000)._foreach_budget(cpu) == 8_000_000
    assert Adafusion(p, foreach_batch_cutoff=5_000_000)._foreach_budget(cpu) == 20_000_000
    # an explicit budget is respected verbatim (not capped)
    assert Adafusion(p, foreach_stack_budget=99_000_000)._foreach_budget(cpu) == 99_000_000


def test_foreach_batch_cutoff_routes_large_to_loop_exactly():
    """A weight above the cutoff loops; smaller ones stack — result is identical.

    The cutoff is decoupled from the (here ample) stack budget, so the large
    tensor loops on its size alone, not on memory pressure.
    """
    torch.manual_seed(0)
    pa = [
        torch.nn.Parameter(torch.randn(1500, 1500) * 0.02),  # 2.25 M > cutoff -> loop
        torch.nn.Parameter(torch.randn(200, 200) * 0.02),    # 40 k <= cutoff -> batch
        torch.nn.Parameter(torch.randn(200, 200) * 0.02),    # bucket-mate
    ]
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    oa = Adafusion(pa, lr=1e-3, betas=(0.9, 0.999), foreach=True,
                   foreach_batch_cutoff=1_000_000, foreach_stack_budget=10**9)
    ob = Adafusion(pb, lr=1e-3, betas=(0.9, 0.999), foreach=False)
    gg = torch.Generator().manual_seed(1)
    for _ in range(6):
        for a, b in zip(pa, pb, strict=False):
            grad = torch.randn(*a.shape, generator=gg) * 0.02
            a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb, strict=False):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


def test_foreach_single_param_uses_fallback():
    """A lone eligible param (e.g. gradient-release) still steps correctly."""
    p = torch.nn.Parameter(torch.randn(16, 16))
    opt = Adafusion([p], lr=1e-3, betas=(0.9, 0.999), foreach=True)
    p.grad = torch.randn_like(p)
    before = p.detach().clone()
    opt.step()
    assert torch.isfinite(p).all() and not torch.equal(before, p.detach())


def test_foreach_bf16_weights_train_no_nan():
    """Batched stochastic-rounding update stays finite over many steps."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 32), torch.nn.GELU(),
        torch.nn.Linear(32, 8),
    ).to(torch.bfloat16)
    opt = Adafusion(model.parameters(), lr=3e-3, betas=(0.0, 0.999),
                    bf16_method="stochastic_rounding", foreach=True)
    x = torch.randn(64, 32, dtype=torch.bfloat16)
    y = torch.randn(64, 8, dtype=torch.bfloat16)
    for _ in range(40):
        opt.zero_grad()
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
def test_checkpoint_roundtrip_preserves_momentum_dtype(momentum_dtype):
    """A torch.save/load checkpoint resumes BIT-EXACTLY and keeps the configured
    momentum dtype.

    torch's default ``Optimizer.load_state_dict`` upcasts every state tensor to
    the param's dtype (fp32), which would silently inflate a quantized first
    moment back to fp32 on resume (int8 -> fp32 is 4x the momentum bytes —
    defeating ``momentum_dtype``) and break exact resume. ``Adafusion`` overrides
    ``load_state_dict`` to restore the stored dtype.
    """
    torch.manual_seed(0)
    p_ref = torch.randn(16, 8)
    grads = [torch.randn(16, 8) for _ in range(10)]

    a = torch.nn.Parameter(p_ref.clone())
    opt_a = Adafusion([a], lr=1e-3, betas=(0.9, 0.999), momentum_dtype=momentum_dtype)
    for g in grads[:5]:
        a.grad = g.clone()
        opt_a.step()

    # Serialize the way real training does (the snapshot is frozen by save).
    buf = io.BytesIO()
    torch.save(opt_a.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, weights_only=False)

    b = torch.nn.Parameter(a.detach().clone())
    opt_b = Adafusion([b], lr=1e-3, betas=(0.9, 0.999), momentum_dtype=momentum_dtype)
    opt_b.load_state_dict(sd)

    # Momentum kept its configured storage dtype (not silently upcast to fp32).
    assert opt_b.state[b]["m"].dtype == opt_a.state[a]["m"].dtype

    for g in grads[5:]:
        a.grad = g.clone()
        opt_a.step()
        b.grad = g.clone()
        opt_b.step()
    assert torch.equal(a, b), "resumed run must continue bit-exactly"
