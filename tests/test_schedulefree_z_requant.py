"""ScheduleFree's quantized ``z`` (``momentum_dtype="int8"`` / ``"4bit"``) must keep moving.

``z`` advances by ``lr_t * d`` per step, which at any realistic LR is far below half a quant
step of its codec (``absmax/254`` int8, ``absmax/14`` 4-bit). The codec's round-to-nearest
requant therefore wrote the OLD codes straight back and ``z`` froze — the same stall
``ScheduleFree._store_z`` already fixed for a bf16 ``z`` with stochastic rounding. Since
0.7.18 the quantized ``z`` is requantized STOCHASTICALLY (``kaon._momentum_codec._round_``,
``floor(x + u)``, unbiased), from the optimizer's own checkpointed SR stream.

Measured before the fix (CPU, ``|w| ~ 0.02``, unit grads, 50 steps; "capture" is the
projection of the quantized ``z``'s drift onto the fp32 ``z``'s, 1.0 = the whole drift):
int8 0.017 at lr=1e-5 and 0.41 at lr=1e-4; 4-bit ~0.0 even at lr=1e-3.
"""

from __future__ import annotations

import io

import pytest
import torch

from kaon import ScheduleFree, reseed_stochastic_rounding
from kaon._momentum_codec import (
    _dequant_4bit,
    _quant_4bit,
    _quant_4bit_stacked,
    _quant_int8,
    _quant_int8_stacked,
)
from kaon._wrappers import CodecBuffer

QUANTIZED = ["int8", "4bit"]
SHAPES = [(64, 96), (64, 96), (128,)]


def _bag(device="cpu"):
    return [torch.nn.Parameter((torch.randn(s) * 0.02).to(device)) for s in SHAPES]


def _z_drift(md, *, foreach, lr, steps=50):
    torch.manual_seed(0)
    reseed_stochastic_rounding()
    params = _bag()
    z0 = torch.cat([p.detach().flatten() for p in params])
    opt = ScheduleFree(params, lr=lr, momentum_dtype=md, foreach=foreach, warmup_steps=0)
    g = torch.Generator().manual_seed(5)
    for _ in range(steps):
        for p in params:
            p.grad = torch.randn(p.shape, generator=g)
        opt.step()
    z = torch.cat([CodecBuffer.read(opt.state[p], "z", md, p).flatten() for p in params])
    return z - z0


@pytest.mark.parametrize("foreach", [True, False], ids=["foreach", "per_param"])
@pytest.mark.parametrize("md", QUANTIZED)
def test_quantized_z_follows_the_fp32_z_at_a_small_lr(md, foreach):
    ref = _z_drift("float32", foreach=foreach, lr=1e-4)
    got = _z_drift(md, foreach=foreach, lr=1e-4)
    capture = (got @ ref / (ref @ ref)).item()
    assert 0.8 < capture < 1.2, (
        f"{md} z captured {capture:.3f} of the fp32 z's drift (1.0 = all of it; the "
        "round-to-nearest requant froze it: int8 0.41, 4bit ~0)"
    )


# ----------------------------------------------------------------- the primitive
def test_round_to_nearest_is_still_the_default():
    """``gen=None`` must stay bit-identical to the historical codecs (momenta, phi)."""
    m = torch.randn(33, 50)
    q, sc = _quant_int8(m)
    absmax = m.abs().amax(dim=1, keepdim=True).clamp_(min=1e-12)
    ref = (m / (absmax / 127.0)).round_().clamp_(-127, 127).to(torch.int8)
    assert torch.equal(q, ref) and torch.equal(sc, absmax / 127.0)


@pytest.mark.parametrize("md", QUANTIZED)
def test_stochastic_requant_is_unbiased_and_in_range(md):
    torch.manual_seed(3)
    x = torch.randn(4, 256) * 0.01
    x[:, ::128] = 1.0                   # an outlier per row / 4-bit block: codes sit near 0
    gen = torch.Generator().manual_seed(9)
    acc = torch.zeros_like(x)
    trials = 400
    for _ in range(trials):
        if md == "int8":
            q, sc = _quant_int8(x, gen)
            assert int(q.abs().max()) <= 127
            acc += q.float() * sc
        else:
            packed, sc, n = _quant_4bit(x, 128, gen)
            acc += _dequant_4bit(packed, sc, n, 128).view_as(x)
    mean = acc / trials
    step = 1.0 / (127.0 if md == "int8" else 7.0)
    # Standard error of the mean of a draw bounded by one quant step: <= step / sqrt(trials).
    assert (mean - x).abs().max() < 5 * step / trials ** 0.5
    # ... whereas round-to-nearest flattens the sub-half-step coordinates to exactly 0.
    if md == "int8":
        q0, sc0 = _quant_int8(x)
        rne = q0.float() * sc0
    else:
        p0, s0, n0 = _quant_4bit(x, 128)
        rne = _dequant_4bit(p0, s0, n0, 128).view_as(x)
    small = x.abs() < step / 2
    assert small.any() and bool((rne[small] == 0).all())


def test_stacked_and_per_param_stochastic_requant_draw_the_same_noise():
    """Param-major noise layout: a stacked draw == N per-param draws on one generator."""
    x = torch.randn(3, 5, 7) * 0.01
    g1, g2 = torch.Generator().manual_seed(4), torch.Generator().manual_seed(4)
    qs, ss = _quant_int8_stacked(x, g1)
    for i in range(3):
        q, s = _quant_int8(x[i], g2)
        assert torch.equal(qs[i], q) and torch.equal(ss[i], s)
    flat = torch.randn(3, 300) * 0.01                 # 300 is not a block multiple
    g1, g2 = torch.Generator().manual_seed(4), torch.Generator().manual_seed(4)
    ps, ss = _quant_4bit_stacked(flat, 128, g1)
    for i in range(3):
        p, s, _ = _quant_4bit(flat[i], 128, g2)
        assert torch.equal(ps[i], p) and torch.equal(ss[i], s)


# --------------------------------------------------- identity + checkpoint witnesses
@pytest.mark.parametrize("md", QUANTIZED)
def test_z_requant_keeps_the_storage_identity(md):
    params = _bag()
    opt = ScheduleFree(params, lr=1e-3, momentum_dtype=md)
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()
    ptrs = {(id(p), k): opt.state[p][k].data_ptr() for p in params for k in ("z", "z_scale")}
    for _ in range(3):
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
    assert ptrs == {(id(p), k): opt.state[p][k].data_ptr()
                    for p in params for k in ("z", "z_scale")}


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("foreach", [True, False], ids=["foreach", "per_param"])
@pytest.mark.parametrize("md", QUANTIZED)
def test_resume_reproduces_the_z_requant_noise(md, foreach, device):
    """The requant noise is optimizer state: a resume continues it bit for bit."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    def grads(step):
        g = torch.Generator().manual_seed(100 + step)
        return [torch.randn(s, generator=g).to(device) for s in SHAPES]

    def fresh():
        torch.manual_seed(0)
        params = _bag(device)
        return params, ScheduleFree(params, lr=1e-4, momentum_dtype=md, foreach=foreach)

    reseed_stochastic_rounding()
    pa, oa = fresh()
    for step in range(6):
        for p, g in zip(pa, grads(step), strict=True):
            p.grad = g
        oa.step()

    reseed_stochastic_rounding()
    pb, ob = fresh()
    for step in range(3):
        for p, g in zip(pb, grads(step), strict=True):
            p.grad = g
        ob.step()
    buf = io.BytesIO()
    torch.save({"opt": ob.state_dict(), "p": [p.detach().clone() for p in pb]}, buf)
    buf.seek(0)
    ck = torch.load(buf, weights_only=False, map_location=device)
    reseed_stochastic_rounding()                      # a new process: allocator back at 0
    pc, oc = fresh()
    with torch.no_grad():
        for p, saved in zip(pc, ck["p"], strict=True):
            p.copy_(saved)
    oc.load_state_dict(ck["opt"])
    for step in range(3, 6):
        for p, g in zip(pc, grads(step), strict=True):
            p.grad = g
        oc.step()
    for a, c in zip(pa, pc, strict=True):
        assert torch.equal(a, c)
        for k in ("z", "z_scale"):
            assert torch.equal(oa.state[a][k], oc.state[c][k])
