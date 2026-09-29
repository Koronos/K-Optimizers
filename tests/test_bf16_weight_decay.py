"""Decoupled weight decay on low-precision weights: ADOPT, AdaBelief and AdamP.

Up to 0.7.17 these three applied the decay as ``p *= (1 - lr*wd)`` straight on the weight.
On a bf16 weight that multiply is round-to-nearest in bf16, with neither stochastic
rounding nor the compact-Kahan residual: at a diffusion-typical ``lr*wd ~ 1e-5`` the factor
rounds every coordinate back to itself, so the decay was a silent no-op (200 steps of
``lr=1e-4, wd=0.1`` on a bf16 ones matrix moved it by exactly 0.0). The decay now rides the
fp32 delta (``delta += lr*wd*p``, the decoded value under kahan8/kahan16) and goes through
the same SR / Kahan write as the update, like Adakaon's.

Also here: the factored reconstruction with ``eps == 0`` and an all-zero gradient, which
used to be ``0 / 0`` -> NaN on AdaBelief and AdamP.
"""
from __future__ import annotations

import pytest
import torch

from kaon import ADOPT, AdaBelief, AdamP
from kaon._compact_kahan import RESIDUAL_KEY, decode, residual_bits_of

CUDA = torch.cuda.is_available()
CLASSES = [ADOPT, AdaBelief, AdamP]
ROUTES = ["per_param", "foreach"] + (["cuda"] if CUDA else [])


def _dev(route: str) -> str:
    return "cuda" if route == "cuda" else "cpu"


def _decay_run(cls, dtype, route, bf16_method="stochastic_rounding", cautious=False, steps=200):
    """Two same-shape weights (so foreach really batches) and a 1-D one, zero grads."""
    dev = _dev(route)
    ps = [torch.nn.Parameter(torch.ones(s, device=dev, dtype=dtype)) for s in [(32, 32), (32, 32), (1024,)]]
    # AdamP: a zero gradient is "scale-invariant", so the projection fires and damps the
    # decay by wd_ratio; 1.0 keeps the expected decay the same for all three.
    extra = {"wd_ratio": 1.0} if cls is AdamP else {}
    opt = cls(ps, lr=1e-4, weight_decay=0.1, cautious=cautious, bf16_method=bf16_method,
              foreach=route != "per_param", **extra)
    for _ in range(steps):
        for p in ps:
            p.grad = torch.zeros_like(p)
        opt.step()
    return ps, opt


def _value(opt, p):
    st = opt.state[p]
    if p.dtype == torch.bfloat16 and RESIDUAL_KEY in st:
        lo = st[RESIDUAL_KEY]
        return decode(p.data, lo, residual_bits_of(lo))
    return p.data.float()


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("cautious", [False, True])
@pytest.mark.parametrize("method", ["stochastic_rounding", "kahan8", "kahan16"])
def test_bf16_weight_decay_is_applied(cls, route, cautious, method):
    """The bf16 run decays like the fp32 run (it used to stay at exactly 1.0)."""
    p32, _ = _decay_run(cls, torch.float32, route, cautious=cautious)
    p16, o16 = _decay_run(cls, torch.bfloat16, route, bf16_method=method, cautious=cautious)
    for a, b in zip(p16, p32, strict=True):
        ref = float(b.detach().mean())
        assert ref < 1.0 - 1e-3                                   # fp32 does decay
        got = float(_value(o16, a).mean())
        # SR is unbiased but noisy (~1e-4 std of a 1024-coord mean after 200 steps); kahan8 /
        # kahan16 track the fp32 value. The total decay being checked is ~2e-3.
        tol = 4e-4 if method == "stochastic_rounding" else 1e-5
        assert abs(got - ref) < tol, (tuple(a.shape), got, ref)


def _bag(dtype, dev):
    torch.manual_seed(0)
    shapes = [(16, 12), (16, 12), (8, 6), (12,), (12,), (), (4, 3, 3, 3)]
    # bf16-representable starts, so the fp32 twin begins at exactly the same values
    return [torch.nn.Parameter((torch.randn(s, device=dev) * 0.05).to(torch.bfloat16).to(dtype))
            for s in shapes]


def _drive(opts, bags, steps=7):
    gg = torch.Generator().manual_seed(7)
    for _ in range(steps):
        grads = [(torch.randn(p.shape, generator=gg) * 0.02).to(torch.bfloat16) for p in bags[0]]
        for bag in bags:
            for p, g in zip(bag, grads, strict=True):
                p.grad = g.to(device=p.device, dtype=p.dtype).clone()
        for o in opts:
            o.step()


_KAHAN16_CASES = [(ADOPT, {}), (AdaBelief, {}), (AdamP, {}),
                  (AdamP, {"delta": 10.0})]  # delta=10: the projection always fires (wd_ratio)


@pytest.mark.parametrize(("cls", "extra"), _KAHAN16_CASES)
@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("md", ["bfloat16", "int8"])
@pytest.mark.parametrize("cautious", [False, True])
def test_kahan16_with_decay_is_the_fp32_run(cls, extra, route, md, cautious):
    """wd=0.1: kahan16 bf16 weights reproduce the fp32-weight run of the same route bit for
    bit (GC off: the shared GC centralizes a bf16 gradient in bf16, which no weight format
    can undo)."""
    dev = _dev(route)
    p32, p16 = _bag(torch.float32, dev), _bag(torch.bfloat16, dev)
    kw = dict(lr=1e-3, weight_decay=0.1, momentum_dtype=md, cautious=cautious,
              gradient_centralization=False, foreach=route != "per_param", **extra)
    o32 = cls(p32, **kw)
    o16 = cls(p16, bf16_method="kahan16", **kw)
    _drive([o32, o16], [p32, p16])
    for a, b in zip(p16, p32, strict=True):
        z = _value(o16, a)
        assert torch.equal(z.view(torch.int32), b.data.contiguous().view(torch.int32)), (
            tuple(a.shape), float((z - b.data).abs().max()))


@pytest.mark.parametrize("cls", [AdaBelief, AdamP, ADOPT])
@pytest.mark.parametrize("route", ROUTES)
def test_eps_zero_all_zero_grad_stays_finite(cls, route):
    """eps=0 (no eps1 inside the factored square) + an all-zero gradient: the row mean is 0
    and the reconstruction used to be 0/0 -> NaN weights. A zero second moment with a zero
    first moment is a zero update."""
    dev = _dev(route)
    torch.manual_seed(0)
    ps = [torch.nn.Parameter(torch.randn(s, device=dev)) for s in [(8, 6), (8, 6), (4, 3, 3)]]
    p0 = [p.detach().clone() for p in ps]
    eps = 1e-6 if cls is ADOPT else 0.0  # ADOPT's eps is a cap (> 0); its eps1 is always 0
    opt = cls(ps, lr=1e-3, eps=eps, foreach=route != "per_param")
    for _ in range(3):
        for p in ps:
            p.grad = torch.zeros_like(p)
        opt.step()
    for p, q in zip(ps, p0, strict=True):
        assert torch.isfinite(p).all()
        assert torch.equal(p.detach(), q)


@pytest.mark.parametrize("cls", [AdaBelief, AdamP])
@pytest.mark.parametrize("route", ROUTES)
def test_eps_zero_dead_row_stays_finite(cls, route):
    """eps=0 and ONE output row whose gradient is always zero (a dead unit): its row
    statistic is exactly 0, so ``rsqrt`` of it is inf and ``0 * inf`` NaN'd the row."""
    dev = _dev(route)
    torch.manual_seed(0)
    ps = [torch.nn.Parameter(torch.randn(8, 6, device=dev)) for _ in range(2)]
    opt = cls(ps, lr=1e-3, eps=0.0, gradient_centralization=False, foreach=route != "per_param")
    for _ in range(3):
        for p in ps:
            g = torch.randn_like(p)
            g[2] = 0.0
            p.grad = g
        opt.step()
    for p in ps:
        assert torch.isfinite(p).all()
