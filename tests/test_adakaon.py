"""Tests for the Adakaon optimizer."""

from __future__ import annotations

import copy
import io
import math

import pytest
import torch

from kaon import Adakaon

from .conftest import train_steps


def test_conv_factoring_reduces_state():
    """Conv-aware factoring stores ~0 state for a conv kernel.

    A 4-D kernel [out, in, kh, kw] is reshaped to [out, in*kh*kw] and factored to
    row+col EMAs (out + in*kh*kw floats), far below the full out*in*kh*kw numel.
    """
    p = torch.nn.Parameter(torch.randn(64, 32, 3, 3))
    opt = Adakaon([p], lr=1e-3, betas=(0.0, 0.999))
    p.grad = torch.randn_like(p)
    opt.step()
    state_floats = sum(v.numel() for v in opt.state[p].values() if torch.is_tensor(v))
    # row (64) + col (288); well under 1/50 of the full 18432 numel.
    assert state_floats < p.numel() / 50, f"conv state should be tiny: {state_floats} vs {p.numel()}"


def test_bf16_momentum_is_half_state():
    """bf16 momentum buffer is half the bytes of fp32 momentum."""
    def mom_bytes(dtype: str) -> int:
        p = torch.nn.Parameter(torch.randn(128, 128))
        opt = Adakaon([p], lr=1e-3, betas=(0.9, 0.999), momentum_dtype=dtype)
        p.grad = torch.randn_like(p)
        opt.step()
        return opt.state[p]["m"].numel() * opt.state[p]["m"].element_size()

    assert mom_bytes("bfloat16") * 2 == mom_bytes("float32")


def test_overfits_regression():
    torch.manual_seed(0xC0DE)
    model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 8))
    opt = Adakaon(model.parameters(), lr=3e-3, betas=(0.9, 0.999))
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
    opt = Adakaon(net.parameters(), lr=3e-3, betas=(0.9, 0.999))
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
    opt = Adakaon(model.parameters(), lr=3e-3, betas=(0.9, 0.999), bf16_method="stochastic_rounding")
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
    opt = Adakaon(model.parameters(), lr=1e-3, betas=(0.9, 0.999), cautious=True)
    x = torch.randn(8, 16)
    (model(x)).pow(2).mean().backward()
    opt.step()  # must not raise


@pytest.mark.parametrize("weight_decay,cautious", [(0.0, False), (0.07, False), (0.07, True)])
def test_unscaled_momentum_preserves_constant_lr_legacy_update(weight_decay, cautious):
    """Moving lr outside the momentum EMA is algebraically the old update at constant lr."""
    lr = 3e-3
    beta1, beta2 = 0.8, 0.95
    initial = torch.tensor([0.4, -0.3, 0.2, -0.1], dtype=torch.float32)
    param = torch.nn.Parameter(initial.clone())
    reference = initial.clone()
    opt = Adakaon(
        [param],
        lr=lr,
        betas=(beta1, beta2),
        eps=(1e-12, 1e-3),
        weight_decay=weight_decay,
        clip_threshold=0.7,
        momentum_dtype="float32",
        cautious=cautious,
        gradient_centralization=False,
        foreach=False,
    )
    legacy_v = torch.zeros_like(reference)
    legacy_m = torch.zeros_like(reference)  # old state: stored in step (lr-scaled) units
    gradients = [
        torch.tensor([0.3, -0.2, -0.1, 0.4]),
        torch.tensor([-0.1, -0.4, 0.2, 0.3]),
        torch.tensor([0.5, 0.1, -0.3, -0.2]),
    ]

    for grad in gradients:
        param.grad = grad.clone()
        opt.step()

        # Replay the pre-0.7.11 1-D fp32 path in its original operation order.
        grad_work = grad.clone()
        grad_sq = grad_work * grad_work
        grad_sq.add_(1e-12)
        legacy_v.lerp_(grad_sq, 1.0 - beta2)
        update = grad_work.mul(legacy_v.rsqrt())
        update.div_((update.norm() / math.sqrt(update.numel()) / 0.7).clamp_(min=1.0))
        update.mul_(lr)
        legacy_m.mul_(beta1).add_(update, alpha=1.0 - beta1)
        delta = legacy_m.clone()
        if weight_decay:
            delta.add_(reference, alpha=lr * weight_decay)
        if cautious:
            mask = (delta * grad_work > 0).to(delta.dtype)
            delta.mul_(mask).div_(mask.mean().clamp_(min=1e-8))
        reference.sub_(delta)

    torch.testing.assert_close(param.detach(), reference, rtol=2e-6, atol=2e-8)
    torch.testing.assert_close(opt.state[param]["m"], legacy_m / lr, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("foreach", [False, True])
def test_momentum_direction_is_independent_of_lr_history(foreach):
    """Different earlier LRs must not contaminate a later step at the same LR."""
    shapes = [(4, 3), (4, 3), (7,), (7,)]
    generator = torch.Generator().manual_seed(29)
    params_a = [torch.nn.Parameter(torch.randn(shape, generator=generator)) for shape in shapes]
    params_b = [torch.nn.Parameter(param.detach().clone()) for param in params_a]
    common = dict(
        betas=(0.85, 0.97),
        momentum_dtype="float32",
        cautious=False,
        gradient_centralization=False,
        foreach=foreach,
    )
    opt_a = Adakaon(params_a, lr=1e-5, **common)
    opt_b = Adakaon(params_b, lr=2e-1, **common)

    grads_1 = [torch.randn(shape, generator=generator) for shape in shapes]
    for pa, pb, grad in zip(params_a, params_b, grads_1, strict=True):
        pa.grad = grad.clone()
        pb.grad = grad.clone()
    opt_a.step()
    opt_b.step()
    for pa, pb in zip(params_a, params_b, strict=True):
        torch.testing.assert_close(opt_a.state[pa]["m"], opt_b.state[pb]["m"], rtol=0, atol=0)

    # Re-anchor parameters and use the same current LR. Equal direction state must
    # now produce exactly the same update despite the different first-step LRs.
    for pa, pb in zip(params_a, params_b, strict=True):
        pb.data.copy_(pa.data)
    opt_a.param_groups[0]["lr"] = 2e-2
    opt_b.param_groups[0]["lr"] = 2e-2
    grads_2 = [torch.randn(shape, generator=generator) for shape in shapes]
    for pa, pb, grad in zip(params_a, params_b, grads_2, strict=True):
        pa.grad = grad.clone()
        pb.grad = grad.clone()
    opt_a.step()
    opt_b.step()

    for pa, pb in zip(params_a, params_b, strict=True):
        torch.testing.assert_close(pa, pb, rtol=0, atol=0)


def test_legacy_checkpoint_momentum_is_migrated_on_load():
    """A pre-0.7.11 checkpoint (lr-scaled momentum, no momentum_units meta) resumes
    identically: load rescales m -> m/lr into direction units, so the next step at
    the checkpoint lr reproduces the legacy trajectory."""
    lr = 4e-3
    param = torch.nn.Parameter(torch.tensor([0.4, -0.3, 0.2, -0.1]))
    opt = Adakaon([param], lr=lr, betas=(0.8, 0.95), momentum_dtype="float32", foreach=False)
    param.grad = torch.tensor([0.3, -0.2, -0.1, 0.4])
    opt.step()

    # Forge the legacy layout: momentum in lr-scaled units, meta without momentum_units.
    legacy = copy.deepcopy(opt.state_dict())
    legacy["state"][0]["m"].mul_(lr)
    legacy["_adakaon_meta"] = {"fused_step": legacy["_adakaon_meta"]["fused_step"]}

    restored_param = torch.nn.Parameter(param.detach().clone())
    restored = Adakaon(
        [restored_param], lr=lr, betas=(0.8, 0.95), momentum_dtype="float32", foreach=False
    )
    restored.load_state_dict(legacy)
    torch.testing.assert_close(restored.state[restored_param]["m"], opt.state[param]["m"])

    # And a current-format checkpoint round-trips untouched.
    roundtrip = Adakaon(
        [torch.nn.Parameter(param.detach().clone())], lr=lr, betas=(0.8, 0.95),
        momentum_dtype="float32", foreach=False,
    )
    roundtrip.load_state_dict(copy.deepcopy(opt.state_dict()))
    key = next(iter(roundtrip.state))
    torch.testing.assert_close(roundtrip.state[key]["m"], opt.state[param]["m"], rtol=0, atol=0)


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
    """foreach=True is numerically equal to the per-parameter path.

    fp32 params keep stochastic rounding a no-op, so any difference between the
    two code paths would be a real bug. fp32 momentum stays bit-exact on CPU.
    bf16 momentum (the default) can differ by one fp32 ULP: the codec now runs
    the EMA in fp32 (matching fused Triton) then ``copy_``s into the bf16
    buffer, so a 1-ULP difference in the per-param vs stacked *update* (distinct
    reduction order) is no longer hidden by rounding that update to bf16
    *before* the EMA. Quantized momentum can likewise differ by one fp32 ULP
    because the per-slice and stacked scale arithmetic go through different
    kernels now that lr is applied after the (unscaled) requant round-trip.
    """
    torch.manual_seed(0)
    pa = _parity_params()
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    oa = Adakaon(pa, foreach=True, **cfg)
    ob = Adakaon(pb, foreach=False, **cfg)
    gg = torch.Generator().manual_seed(7)
    for _ in range(10):
        for a, b in zip(pa, pb, strict=False):
            grad = torch.randn(*a.shape, generator=gg) * 0.02
            a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    md = cfg.get("momentum_dtype", "bfloat16")
    # Default / omitted dtype is bf16. beta1=0 allocates no momentum — bit-exact.
    no_mom = cfg.get("betas", (0.9, 0.999))[0] == 0.0
    if no_mom or md == "float32":
        rtol, atol = 0, 0
    elif md in ("int8", "4bit"):
        rtol, atol = 2e-7, 5e-9
    else:  # bfloat16: 1 fp32 ULP (visible now that EMA is in fp32)
        rtol, atol = 1e-6, 1e-9
    for a, b in zip(pa, pb, strict=False):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=rtol, atol=atol)


def test_foreach_chunking_is_exact():
    """A tiny stack budget forces buckets to split and large weights to route to
    the loop — the result must still equal the per-parameter path exactly."""
    pa = _parity_params()
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    # budget=200 elems: every 2-D shape splits into many chunks and the larger
    # tensors fall to the per-param branch — a stress test of both code paths.
    oa = Adakaon(pa, lr=1e-3, betas=(0.9, 0.999), foreach=True, foreach_stack_budget=200)
    ob = Adakaon(pb, lr=1e-3, betas=(0.9, 0.999), foreach=False)
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
    oa = Adakaon(pa, lr=1e-3, betas=(0.9, 0.999), momentum_dtype="int8",
                   foreach=True, foreach_stack_budget=200)
    ob = Adakaon(pb, lr=1e-3, betas=(0.9, 0.999), momentum_dtype="int8", foreach=False)
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
    match the per-param path (to one fp32 ULP; see test_foreach_matches_per_param
    on why quantized momentum is no longer bit-for-bit)."""
    pa = _parity_params()
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    oa = Adakaon(pa, lr=1e-3, betas=(0.9, 0.999), momentum_dtype="4bit",
                   foreach=True, foreach_stack_budget=200)
    ob = Adakaon(pb, lr=1e-3, betas=(0.9, 0.999), momentum_dtype="4bit", foreach=False)
    gg = torch.Generator().manual_seed(7)
    for _ in range(8):
        for a, b in zip(pa, pb, strict=False):
            grad = torch.randn(*a.shape, generator=gg) * 0.02
            a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb, strict=False):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=2e-7, atol=5e-9)


def _scalar_bag(seed: int = 11):
    """0-D scalars (LyCORIS ``use_scalar``-style gates) plus shape-(1,) bucket-mates.

    All numel==1, so the whole bag lands in the non-factored ``L == 1`` bucket —
    0-D as length-1 views, (1,) as-is — exercising the mixed 0-D/(1,) stacking.
    """
    g = torch.Generator().manual_seed(seed)
    shapes = [(), (), (), (1,), (1,)]
    return [torch.nn.Parameter(torch.randn(s, generator=g) * 0.05) for s in shapes]


_SCALAR_CFGS = [
    dict(lr=1e-3, betas=(0.0, 0.999)),                                  # no momentum
    dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="float32"),        # fp32 momentum
    dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="bfloat16"),       # bf16 momentum
    dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="int8"),           # int8 momentum
    dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="int8", weight_decay=0.02),
    dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="4bit"),           # 4-bit momentum
    dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="4bit", weight_decay=0.02),
    dict(lr=1e-3, betas=(0.9, 0.999), weight_decay=0.02),               # weight decay
    dict(lr=1e-3, betas=(0.9, 0.999), cautious=True),                   # cautious mask
    dict(lr=1e-3, betas=(0.9, 0.999), cautious=False),
]


@pytest.mark.parametrize("cfg", _SCALAR_CFGS)
def test_foreach_scalar_0d_matches_per_param(cfg):
    """0-D scalars through the batched non-factored bucket are element-for-element
    equal to the per-parameter path (fp32 params keep SR a no-op — bit-exact).

    For a scalar the batched RMS clip ``norm(dim=1)/sqrt(1)`` must equal the
    per-param ``rms()`` (= |x|), and the cautious per-slice mask mean must equal
    the scalar mask — this test is the proof, not an assumption.
    """
    pa = _scalar_bag()
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    oa = Adakaon(pa, foreach=True, **cfg)
    ob = Adakaon(pb, foreach=False, **cfg)
    gg = torch.Generator().manual_seed(7)
    for _ in range(10):
        for a, b in zip(pa, pb, strict=False):
            grad = torch.randn(a.shape, generator=gg) * 0.02
            a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb, strict=False):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)
    # State keeps the per-param layout: 0-D params -> 0-D `v` (checkpoint compat).
    for a in pa:
        if a.ndim == 0:
            assert oa.state[a]["v"].ndim == 0


def test_foreach_scalar_0d_takes_batched_path(monkeypatch):
    """0-D scalars must actually ride the batched bucket, not silently fall back
    to the ~22-launches-per-scalar per-param loop (the LyCORIS use_scalar
    pathology this guards against)."""
    looped = []
    orig = Adakaon._step_one_param
    monkeypatch.setattr(
        Adakaon, "_step_one_param",
        lambda self, p, group: (looped.append(p), orig(self, p, group))[1],
    )
    params = _scalar_bag()
    opt = Adakaon(params, lr=1e-3, betas=(0.9, 0.999), foreach=True)
    for p in params:
        p.grad = torch.randn(p.shape) * 0.02
    opt.step()
    assert not looped, f"{len(looped)} params fell back to the per-param loop"


def test_foreach_scalar_0d_mixed_with_other_shapes():
    """0-D scalars mixed with 1-D/2-D/conv params: scalars stay bit-exact vs the
    per-param path and the other buckets keep their existing contract (the repo's
    own parity tests own the exact claim for them; here they get a 1e-7 bound so
    this test isn't hostage to value-dependent last-ulp drift)."""
    g = torch.Generator().manual_seed(3)
    shapes = [(), (), (1,), (40,), (40,), (8, 16), (8, 16), (4, 4, 3, 3)]
    pa = [torch.nn.Parameter(torch.randn(s, generator=g) * 0.05) for s in shapes]
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    oa = Adakaon(pa, lr=1e-3, betas=(0.9, 0.999), foreach=True)
    ob = Adakaon(pb, lr=1e-3, betas=(0.9, 0.999), foreach=False)
    gg = torch.Generator().manual_seed(5)
    for _ in range(8):
        for a, b in zip(pa, pb, strict=False):
            grad = torch.randn(a.shape, generator=gg) * 0.02
            a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb, strict=False):
        if a.numel() == 1:
            torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)
        else:
            torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=1e-7)


def test_foreach_scalar_0d_bf16_none_exact():
    """bf16 0-D params with bf16_method='none': batched equals per-param exactly
    (same cast-and-subtract, no stochastic draws involved)."""
    g = torch.Generator().manual_seed(13)
    pa = [torch.nn.Parameter((torch.randn((), generator=g) * 0.05).to(torch.bfloat16)) for _ in range(6)]
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    oa = Adakaon(pa, lr=1e-3, betas=(0.9, 0.999), bf16_method="none", foreach=True)
    ob = Adakaon(pb, lr=1e-3, betas=(0.9, 0.999), bf16_method="none", foreach=False)
    gg = torch.Generator().manual_seed(17)
    for _ in range(10):
        for a, b in zip(pa, pb, strict=False):
            grad = (torch.randn((), generator=gg) * 0.02).to(torch.bfloat16)
            a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb, strict=False):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


def test_foreach_scalar_0d_bf16_sr_trains_finite():
    """bf16 0-D params with stochastic rounding: the batched write stays finite
    and the scalars actually move (SR draws differ from per-param by design)."""
    g = torch.Generator().manual_seed(23)
    params = [torch.nn.Parameter((torch.randn(()) * 0.05).to(torch.bfloat16)) for _ in range(8)]
    before = [p.detach().clone() for p in params]
    opt = Adakaon(params, lr=1e-2, betas=(0.9, 0.999), bf16_method="stochastic_rounding", foreach=True)
    for _ in range(30):
        for p in params:
            p.grad = (torch.randn((), generator=g) * 0.5 + 0.5).to(torch.bfloat16)
        opt.step()
    assert all(torch.isfinite(p).all() for p in params)
    assert any(not torch.equal(a, p.detach()) for a, p in zip(before, params, strict=False))


def test_scalar_0d_checkpoint_roundtrip_across_paths():
    """A checkpoint saved mid-training by the batched path loads into a per-param
    optimizer (and vice versa) and continues bit-exactly — the 0-D state layout
    is identical on both paths (0-D `v`/`m`, scalar `m_scale`)."""
    for save_foreach, load_foreach in [(True, False), (False, True)]:
        pa = _scalar_bag(seed=29)
        oa = Adakaon(pa, lr=1e-3, betas=(0.9, 0.999), momentum_dtype="int8",
                     weight_decay=0.02, foreach=save_foreach)
        gg = torch.Generator().manual_seed(31)
        grads = [[torch.randn(p.shape, generator=gg) * 0.02 for p in pa] for _ in range(8)]
        for step in range(4):
            for p, gr in zip(pa, grads[step], strict=False):
                p.grad = gr.clone()
            oa.step()
        buf = io.BytesIO()
        torch.save({"opt": oa.state_dict(), "params": [p.detach().clone() for p in pa]}, buf)
        buf.seek(0)
        ckpt = torch.load(buf, weights_only=False)
        pb = [torch.nn.Parameter(t.clone()) for t in ckpt["params"]]
        ob = Adakaon(pb, lr=1e-3, betas=(0.9, 0.999), momentum_dtype="int8",
                     weight_decay=0.02, foreach=load_foreach)
        ob.load_state_dict(ckpt["opt"])
        for step in range(4, 8):
            for plist, opt in ((pa, oa), (pb, ob)):
                for p, gr in zip(plist, grads[step], strict=False):
                    p.grad = gr.clone()
                opt.step()
        for a, b in zip(pa, pb, strict=False):
            torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)
        for b in pb:
            if b.ndim == 0:
                assert ob.state[b]["v"].ndim == 0


_SCALAR_CKPT_PATH_CFGS = [
    dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="float32"),
    dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="bfloat16"),
    dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="int8"),
    dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="4bit"),
    dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="int8", weight_decay=0.02, cautious=True),
    dict(lr=1e-3, betas=(0.0, 0.999)),
]
_SCALAR_CKPT_PATH_IDS = ["fp32", "bf16", "int8", "4bit", "int8_wd_cautious", "beta1_0"]


@pytest.mark.parametrize("cfg", _SCALAR_CKPT_PATH_CFGS, ids=_SCALAR_CKPT_PATH_IDS)
def test_scalar_0d_checkpoint_per_param_saved_foreach_loaded(cfg):
    """A checkpoint saved by the per-parameter path (the pre-existing, unbatched
    path this diff never touches) loads into the batched foreach path and
    continues bit-exactly, across the momentum dtypes and knob combos the audit
    validated externally. Complements
    ``test_scalar_0d_checkpoint_roundtrip_across_paths``, which only
    cross-checks foreach True<->False for one (int8+weight_decay) config —
    this pins the specific per-param -> foreach direction across the dtypes a
    future refactor of ``_nonfactored_bucket``'s ``flat()`` view could break."""
    pa = _scalar_bag(seed=53)
    oa = Adakaon(pa, foreach=False, **cfg)
    gg = torch.Generator().manual_seed(59)
    grads = [[torch.randn(p.shape, generator=gg) * 0.02 for p in pa] for _ in range(8)]
    for step in range(4):
        for p, gr in zip(pa, grads[step], strict=False):
            p.grad = gr.clone()
        oa.step()
    buf = io.BytesIO()
    torch.save({"opt": oa.state_dict(), "params": [p.detach().clone() for p in pa]}, buf)
    buf.seek(0)
    ckpt = torch.load(buf, weights_only=False)
    pb = [torch.nn.Parameter(t.clone()) for t in ckpt["params"]]
    ob = Adakaon(pb, foreach=True, **cfg)
    ob.load_state_dict(ckpt["opt"])
    for step in range(4, 8):
        for plist, opt in ((pa, oa), (pb, ob)):
            for p, gr in zip(plist, grads[step], strict=False):
                p.grad = gr.clone()
            opt.step()
    for a, b in zip(pa, pb, strict=False):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


@pytest.mark.parametrize("momentum_dtype,expected", [
    ("float32", {"m": ((), torch.float32)}),
    ("bfloat16", {"m": ((), torch.bfloat16)}),
    ("int8", {"m": ((), torch.int8), "m_scale": ((), torch.float32)}),
    ("4bit", {"m": ((1,), torch.uint8), "m_scale": ((1,), torch.float32)}),
])
def test_scalar_0d_state_schema_by_momentum_dtype(momentum_dtype, expected):
    """Pins the checkpoint schema of a 0-D scalar's persisted state, per
    ``momentum_dtype``, as produced by the *batched* path (``_scalar_bag`` has
    5 numel==1 params, so ``_step_foreach`` actually runs the ``L == 1`` bucket
    instead of the always-safe per-param fallback that needs >=2 same-numel
    params to trigger).

    ``v`` is always a 0-D fp32 second-moment buffer regardless of
    ``momentum_dtype``; ``m``/``m_scale`` keep the exact per-param shapes (0-D
    stays 0-D — never silently promoted to shape ``(1,)`` by the ``flat()``
    view). A refactor that changes any of these breaks loading old checkpoints
    even though the training math stays correct."""
    pa = _scalar_bag(seed=61)  # 3x 0-D + 2x (1,) sharing the L==1 bucket
    opt = Adakaon(pa, lr=1e-3, betas=(0.9, 0.999), momentum_dtype=momentum_dtype, foreach=True)
    for p in pa:
        p.grad = torch.randn(p.shape) * 0.02
    opt.step()
    for p in pa:
        if p.ndim != 0:
            continue
        st = opt.state[p]
        assert st["v"].shape == () and st["v"].dtype == torch.float32
        for key, (shape, dtype) in expected.items():
            assert st[key].shape == shape, f"{momentum_dtype}.{key} shape = {tuple(st[key].shape)}"
            assert st[key].dtype == dtype, f"{momentum_dtype}.{key} dtype = {st[key].dtype}"
        if momentum_dtype == "4bit":
            assert st["m_numel"] == 1
            assert st["m_block"] == 1


def test_scalar_0d_state_schema_no_momentum():
    """``beta1=0`` skips the momentum codec entirely: a 0-D scalar's persisted
    state must be just ``v`` (0-D fp32) — no ``m``/``m_scale`` keys leaking in.
    Pinned separately from the momentum dtypes above since it's a different
    code path (no codec touched at all)."""
    pa = _scalar_bag(seed=67)
    opt = Adakaon(pa, lr=1e-3, betas=(0.0, 0.999), foreach=True)
    for p in pa:
        p.grad = torch.randn(p.shape) * 0.02
    opt.step()
    for p in pa:
        if p.ndim == 0:
            assert set(opt.state[p].keys()) == {"v"}
            assert opt.state[p]["v"].shape == ()
            assert opt.state[p]["v"].dtype == torch.float32


def test_4bit_pack_roundtrip():
    """Nibble pack/unpack round-trips for even and odd element counts, and
    dequant(quant(m)) stays within the ~1/7 absmax 4-bit grid error."""
    from kaon.adakaon import (
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
    opt = Adakaon([p], betas=(0.9, 0.999), momentum_dtype="4bit", momentum_4bit_block=128)
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
    opt = Adakaon(model.parameters(), lr=1e-2, betas=(0.9, 0.999), momentum_dtype="4bit")
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
        Adakaon(p, momentum_dtype="2bit")


def test_foreach_batch_cutoff_validation():
    p = [torch.nn.Parameter(torch.randn(4, 4))]
    Adakaon(p, foreach_batch_cutoff=1)
    Adakaon(p, foreach_batch_cutoff=5_000_000)
    with pytest.raises(ValueError):
        Adakaon(p, foreach_batch_cutoff=0)


def test_foreach_budget_capped_at_4x_cutoff():
    """The adaptive chunk budget never exceeds 4x the cutoff (over-stacking guard)
    and scales with the cutoff. Deterministic on CPU (no VRAM read)."""
    from kaon._backend import foreach_budget

    cpu = torch.device("cpu")
    assert foreach_budget(None, 2_000_000, 48, cpu) == 8_000_000
    assert foreach_budget(None, 5_000_000, 48, cpu) == 20_000_000
    # an explicit budget is respected verbatim (not capped)
    assert foreach_budget(99_000_000, 2_000_000, 48, cpu) == 99_000_000


def test_foreach_batch_cutoff_routes_large_to_loop_exactly():
    """A weight above the cutoff loops; smaller ones stack — result matches.

    The cutoff is decoupled from the (here ample) stack budget, so the large
    tensor loops on its size alone, not on memory pressure. Default momentum
    is bf16, whose EMA now runs in fp32, so foreach vs per-param is equal to
    one fp32 ULP rather than bit-exact (see ``test_foreach_matches_per_param``).
    """
    torch.manual_seed(0)
    pa = [
        torch.nn.Parameter(torch.randn(1500, 1500) * 0.02),  # 2.25 M > cutoff -> loop
        torch.nn.Parameter(torch.randn(200, 200) * 0.02),    # 40 k <= cutoff -> batch
        torch.nn.Parameter(torch.randn(200, 200) * 0.02),    # bucket-mate
    ]
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    oa = Adakaon(pa, lr=1e-3, betas=(0.9, 0.999), foreach=True,
                   foreach_batch_cutoff=1_000_000, foreach_stack_budget=10**9)
    ob = Adakaon(pb, lr=1e-3, betas=(0.9, 0.999), foreach=False)
    gg = torch.Generator().manual_seed(1)
    for _ in range(6):
        for a, b in zip(pa, pb, strict=False):
            grad = torch.randn(*a.shape, generator=gg) * 0.02
            a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb, strict=False):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=1e-6, atol=1e-9)


def test_foreach_single_param_uses_fallback():
    """A lone eligible param (e.g. gradient-release) still steps correctly."""
    p = torch.nn.Parameter(torch.randn(16, 16))
    opt = Adakaon([p], lr=1e-3, betas=(0.9, 0.999), foreach=True)
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
    opt = Adakaon(model.parameters(), lr=3e-3, betas=(0.0, 0.999),
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
    defeating ``momentum_dtype``) and break exact resume. ``Adakaon`` overrides
    ``load_state_dict`` to restore the stored dtype.
    """
    torch.manual_seed(0)
    p_ref = torch.randn(16, 8)
    grads = [torch.randn(16, 8) for _ in range(10)]

    a = torch.nn.Parameter(p_ref.clone())
    opt_a = Adakaon([a], lr=1e-3, betas=(0.9, 0.999), momentum_dtype=momentum_dtype)
    for g in grads[:5]:
        a.grad = g.clone()
        opt_a.step()

    # Serialize the way real training does (the snapshot is frozen by save).
    buf = io.BytesIO()
    torch.save(opt_a.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, weights_only=False)

    b = torch.nn.Parameter(a.detach().clone())
    opt_b = Adakaon([b], lr=1e-3, betas=(0.9, 0.999), momentum_dtype=momentum_dtype)
    opt_b.load_state_dict(sd)

    # Momentum kept its configured storage dtype (not silently upcast to fp32).
    assert opt_b.state[b]["m"].dtype == opt_a.state[a]["m"].dtype

    for g in grads[5:]:
        a.grad = g.clone()
        opt_a.step()
        b.grad = g.clone()
        opt_b.step()
    assert torch.equal(a, b), "resumed run must continue bit-exactly"


# ----------------------------------------------------------- native foreach view cache
#
# ``Adakaon._foreach_plans`` caches, per param group, the bucketing and every view the
# stacked step derives from it (the ``v`` views, the codec's ``mat`` lookup, the ``p.data``
# views). The cache is a pure speedup, so the contract is twofold: it must be numerically
# invisible, and it must never outlive the tensors it points at. These tests define
# "correct" as the *uncached* path (``_foreach_cache_enabled = False``, which rebuilds every
# list per param per step exactly as the pre-cache code did) and demand bit-exact equality
# through every event that can rot a cached view.


def _cache_bag(seed: int = 17):
    """Every native-foreach bucket kind at once: a mixed 0-D/``(1,)`` ``L == 1`` bucket,
    a repeated 1-D length, a repeated 2-D shape and a repeated conv (matrixize)."""
    g = torch.Generator().manual_seed(seed)
    shapes = [(), (), (1,), (7,), (7,), (16, 32), (16, 32), (8, 4, 3, 3), (8, 4, 3, 3)]
    return [torch.nn.Parameter(torch.randn(s, generator=g) * 0.05) for s in shapes]


_CACHE_CFGS = [
    dict(lr=1e-3, betas=(0.9, 0.999)),                                             # bf16 momentum
    dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="float32", weight_decay=0.02),
    dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="int8", weight_decay=0.02),
    dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="4bit"),
    dict(lr=1e-3, betas=(0.0, 0.999), weight_decay=0.02),                          # no momentum
]
_CACHE_IDS = ["bf16mom", "fp32mom_wd", "int8_wd", "4bit", "beta1_0"]


def _cached_pair(cfg, **kw):
    """Two identical Adakaons over identical params — the first cached, the second not."""
    pa, pb = _cache_bag(), _cache_bag()
    oa, ob = Adakaon(pa, **cfg, **kw), Adakaon(pb, **cfg, **kw)
    ob._foreach_cache_enabled = False
    return pa, pb, oa, ob


def _drive(pairs, steps, gen, mutate=None):
    """Step every ``(params, opt)`` pair on the SAME gradients for ``steps`` steps.

    ``mutate(step, params, opt)`` runs before each optimizer's step, so every arm sees
    identical interference.
    """
    for step in range(steps):
        grads = [torch.randn(p.shape, generator=gen) * 0.02 for p in pairs[0][0]]
        for plist, opt in pairs:
            if mutate is not None:
                mutate(step, plist, opt)
            for p, gr in zip(plist, grads, strict=True):
                p.grad = gr.clone().to(p.dtype)   # .to() is a no-op unless a test recasts p
            opt.step()


def _assert_same(pa, pb, oa, ob):
    """Weights AND optimizer state must be bit-identical between the two arms."""
    for a, b in zip(pa, pb, strict=True):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)
        sa, sb = oa.state[a], ob.state[b]
        assert set(sa) == set(sb), f"state keys diverged: {set(sa)} vs {set(sb)}"
        for k, va in sa.items():
            if torch.is_tensor(va):
                assert torch.equal(va, sb[k]), f"state[{k}] diverged"
            else:
                assert va == sb[k], f"state[{k}] diverged: {va} vs {sb[k]}"


@pytest.mark.parametrize("cfg", _CACHE_CFGS, ids=_CACHE_IDS)
def test_foreach_cache_is_numerically_invisible(cfg):
    """Cached and uncached runs agree bit-for-bit on weights and on every state buffer."""
    pa, pb, oa, ob = _cached_pair(cfg)
    _drive([(pa, oa), (pb, ob)], 8, torch.Generator().manual_seed(23))
    assert oa._foreach_plans, "the cached arm never built a plan — the test is vacuous"
    assert not ob._foreach_plans, "the uncached arm must not store a plan"
    _assert_same(pa, pb, oa, ob)


def test_foreach_cache_plan_is_reused_then_rebuilt_on_data_rebind():
    """The plan survives an ordinary step (that is the whole point) and is dropped the
    moment a ``p.data`` pointer moves — the guard that makes the reuse safe."""
    ps = _cache_bag()
    opt = Adakaon(ps, lr=1e-3, betas=(0.9, 0.999))
    gen = torch.Generator().manual_seed(29)

    def one_step():
        for p in ps:
            p.grad = torch.randn(p.shape, generator=gen) * 0.02
        opt.step()

    one_step()
    plan = next(iter(opt._foreach_plans.values()))
    one_step()
    assert next(iter(opt._foreach_plans.values())) is plan, "an unchanged step rebuilt the plan"
    ps[0].data = ps[0].data.clone()          # same shape/dtype, different storage
    one_step()
    assert next(iter(opt._foreach_plans.values())) is not plan, "a p.data rebind was not caught"


@pytest.mark.parametrize("rebind", ["clone", "bfloat16"])
def test_foreach_cache_renewed_when_param_data_is_rebound(rebind):
    """``p.data = p.data.to(...)`` mid-training points the param at fresh storage. Cached
    ``p.data`` views of the OLD storage would silently stop updating the live weights (and
    a dtype change also re-keys the bucket), so the plan has to be rebuilt."""
    cfg = dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="float32", weight_decay=0.02,
               bf16_method="none")  # "none" keeps bf16 writes free of stochastic-rounding draws

    def mutate(step, plist, _opt):
        if step == 3:
            for p in plist:
                p.data = p.data.clone() if rebind == "clone" else p.data.to(torch.bfloat16)

    pa, pb, oa, ob = _cached_pair(cfg)
    _drive([(pa, oa), (pb, ob)], 6, torch.Generator().manual_seed(37), mutate)
    assert pa[0].dtype == (torch.float32 if rebind == "clone" else torch.bfloat16)
    _assert_same(pa, pb, oa, ob)


@pytest.mark.parametrize("cfg", _CACHE_CFGS, ids=_CACHE_IDS)
def test_foreach_cache_invalidated_by_load_state_dict(cfg):
    """``load_state_dict`` REPLACES every state tensor (torch rebuilds ``self.state``, and
    the dtype-preserving loader casts on top of that). A plan still holding views of the
    old ``v``/``row``/``col``/``m`` would keep training a detached copy of the state while
    the checkpointed buffers sat untouched — silent, and invisible to the param-identity
    guard because the params never changed."""
    pa, pb, oa, ob = _cached_pair(cfg)
    pairs = [(pa, oa), (pb, ob)]
    gen = torch.Generator().manual_seed(41)
    _drive(pairs, 3, gen)
    ckpts = [copy.deepcopy(o.state_dict()) for _, o in pairs]
    saved = [[p.detach().clone() for p in pl] for pl, _ in pairs]
    _drive(pairs, 3, gen)
    for (plist, opt), sd, ws in zip(pairs, ckpts, saved, strict=True):
        opt.load_state_dict(sd)
        for p, w in zip(plist, ws, strict=True):
            p.data.copy_(w)                  # in place: isolates state replacement alone
    _drive(pairs, 3, gen)
    _assert_same(pa, pb, oa, ob)


@pytest.mark.parametrize("cfg", _CACHE_CFGS, ids=_CACHE_IDS)
def test_foreach_cache_invalidated_by_autolr_reset(cfg):
    """An auto_lr rollback calls ``_autolr_reset_base_state``, which clears ``self.state``.
    The next step lazily re-allocates it, so a surviving plan would step orphaned buffers
    and never touch the fresh ones."""
    def mutate(step, _plist, opt):
        if step == 3:
            opt._autolr_reset_base_state()

    pa, pb, oa, ob = _cached_pair(cfg)
    _drive([(pa, oa), (pb, ob)], 7, torch.Generator().manual_seed(43), mutate)
    _assert_same(pa, pb, oa, ob)


def test_foreach_cache_invalidated_by_param_set_change():
    """Mutating ``group["params"]`` in place keeps the group object (and its cache key)
    alive while changing what the buckets contain — caught by the per-param identity guard,
    not by the group key."""
    cfg = dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="int8", weight_decay=0.02)

    def mutate(step, plist, opt):
        params = opt.param_groups[0]["params"]
        if step == 2:
            params.pop(3)                    # drop a (7,) param out of the L==7 bucket
        elif step == 4:
            params.insert(3, plist[3])       # and put it back, state and all

    pa, pb, oa, ob = _cached_pair(cfg)
    _drive([(pa, oa), (pb, ob)], 7, torch.Generator().manual_seed(47), mutate)
    _assert_same(pa, pb, oa, ob)


def test_foreach_cache_invalidated_by_param_swapped_for_a_storage_alias():
    """The nastiest param-set change: a param replaced by a *different* Parameter object
    that aliases the same storage. Every ``data_ptr`` is unchanged, so only the per-param
    identity guard can see it — and it must, because the replacement has its own (empty)
    entry in ``self.state`` and has to start from freshly initialized buffers."""
    cfg = dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="int8", weight_decay=0.02)

    def mutate(step, plist, opt):
        if step == 3:
            alias = torch.nn.Parameter(plist[4].detach())   # shares storage -> same data_ptr
            plist[4] = opt.param_groups[0]["params"][4] = alias

    pa, pb, oa, ob = _cached_pair(cfg)
    _drive([(pa, oa), (pb, ob)], 7, torch.Generator().manual_seed(67), mutate)
    assert pa[4] in oa.state, "the alias never got its own state entry"
    _assert_same(pa, pb, oa, ob)


def test_foreach_cache_rechunks_when_the_stack_budget_shrinks():
    """Chunk boundaries are numerically invisible (every reduction is per-slice), but they
    ARE the VRAM-safety ceiling: the adaptive budget moves with free VRAM on every step, so
    a cached plan that never re-split would keep stacking chunks the card no longer fits.
    Pinned structurally, since no numerical assertion can see it."""
    ps = [torch.nn.Parameter(torch.randn(4) * 0.05) for _ in range(8)]
    opt = Adakaon(ps, lr=1e-3, betas=(0.9, 0.999), foreach_stack_budget=100)

    def one_step():
        for p in ps:
            p.grad = torch.randn(4) * 0.02
        opt.step()
        return [len(c.plist) for c in next(iter(opt._foreach_plans.values())).chunks]

    assert one_step() == [8], "budget 100 // 4 elems = 25 per chunk -> one chunk of 8"
    opt._foreach_stack_budget = 12
    assert one_step() == [3, 3, 2], "budget 12 // 4 elems = 3 per chunk"
    opt._foreach_stack_budget = 100
    assert one_step() == [8], "the plan must widen again when the budget recovers"


def test_foreach_cache_invalidated_by_add_param_group():
    """A param group added mid-training gets its own cache entry and does not disturb the
    existing one."""
    cfg = dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="int8", weight_decay=0.02)
    extra = [torch.nn.Parameter(torch.zeros(7) + 0.1) for _ in range(2)]

    def mutate(step, _plist, opt):
        if step == 3:
            opt.add_param_group({"params": [torch.nn.Parameter(e.detach().clone())
                                            for e in extra], "lr": 5e-3})

    pa, pb, oa, ob = _cached_pair(cfg)
    _drive([(pa, oa), (pb, ob)], 6, torch.Generator().manual_seed(53), mutate)
    _assert_same(pa, pb, oa, ob)


def test_foreach_cache_invalidated_by_per_param_fallback():
    """A step where the whole group falls back to the per-parameter loop re-allocates the
    quantized first moment (the int8/4bit codecs requant into *fresh* tensors rather than
    copying in place). The param set is unchanged, so no guard can see it — the fallback
    itself has to drop the plan, or the next batched step stacks a dead ``m``."""
    from kaon._backend import FOREACH_BATCH_CUTOFF

    cfg = dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="int8", weight_decay=0.02)

    def mutate(step, _plist, opt):
        # cutoff 0 -> every param is "too large to stack" -> the group loops per param.
        opt._foreach_batch_cutoff = 0 if step == 3 else FOREACH_BATCH_CUTOFF

    pa, pb, oa, ob = _cached_pair(cfg)
    _drive([(pa, oa), (pb, ob)], 7, torch.Generator().manual_seed(59), mutate)
    assert not any(torch.equal(oa.state[p]["m"], torch.zeros_like(oa.state[p]["m"]))
                   for p in pa), "momentum never moved — the fallback step did nothing"
    _assert_same(pa, pb, oa, ob)


def test_two_optimizers_over_same_params_do_not_share_cache():
    """Two Adakaons stepping the SAME params keep separate plans and separate state; the
    interleaved result must equal the same interleaving with the cache off."""
    cfg = dict(lr=1e-3, betas=(0.9, 0.999), momentum_dtype="int8", weight_decay=0.02)
    pa, pb = _cache_bag(), _cache_bag()
    a1, a2 = Adakaon(pa, **cfg), Adakaon(pa, **cfg)
    b1, b2 = Adakaon(pb, **cfg), Adakaon(pb, **cfg)
    b1._foreach_cache_enabled = b2._foreach_cache_enabled = False
    gen = torch.Generator().manual_seed(61)
    for _ in range(6):
        grads = [torch.randn(p.shape, generator=gen) * 0.02 for p in pa]
        for plist, first, second in ((pa, a1, a2), (pb, b1, b2)):
            for opt in (first, second):
                for p, gr in zip(plist, grads, strict=True):
                    p.grad = gr.clone()
                opt.step()
    assert a1._foreach_plans and a2._foreach_plans
    assert set(map(id, a1._foreach_plans.values())).isdisjoint(map(id, a2._foreach_plans.values()))
    _assert_same(pa, pb, a1, b1)
    _assert_same(pa, pb, a2, b2)


# --------------------------------------------------------------------- cautious_wd
# Decoupled weight decay used to be folded into the delta BEFORE the cautious mask on every
# path, so a masked-out coordinate got essentially NO decay and a survivor got it multiplied
# by the survivor rescale 1/keep (measured, as a fraction of the REQUESTED lr*wd*p: 1.49x on
# survivors / 0.005x on rejected at keep=0.64, and 1.985x / 0.0013x at keep=0.50 — the
# aggregate shrinkage is preserved, its per-coordinate distribution is not).
# ``cautious_wd="full"`` applies the same lr*wd to every coordinate and masks only the update,
# which is the Cautious Optimizers paper's own placement.
_WD_CFG = dict(lr=5e-3, betas=(0.9, 0.999), weight_decay=0.1, cautious=True)


def _wd_bag(seed: int = 71):
    g = torch.Generator().manual_seed(seed)
    shapes = [(), (1,), (7,), (16, 32), (8, 4, 3, 3)]
    return [torch.nn.Parameter(torch.randn(s, generator=g) * 0.05) for s in shapes]


def test_cautious_wd_rejects_unknown_placement():
    p = [torch.nn.Parameter(torch.randn(4, 4))]
    with pytest.raises(ValueError, match="cautious_wd"):
        Adakaon(p, lr=1e-3, cautious_wd="outside")


def test_cautious_wd_defaults_to_masked():
    p = [torch.nn.Parameter(torch.randn(4, 4))]
    assert Adakaon(p, lr=1e-3).param_groups[0]["cautious_wd"] == "masked"


@pytest.mark.parametrize("foreach", [False, True])
def test_cautious_wd_full_decays_every_coordinate(foreach):
    """The mode's whole point: under "full" the decay reaches the masked-out coordinates too.

    One step from a known state, with the momentum forced to a value whose sign disagrees with
    the gradient on exactly half the coordinates. Under "masked" those coordinates move by ~0;
    under "full" they move by exactly ``lr * wd * p``.
    """
    lr, wd = 5e-3, 0.1
    out = {}
    for arm in ("masked", "full"):
        p = torch.nn.Parameter(torch.full((2, 8), 0.5))
        opt = Adakaon([p], lr=lr, betas=(0.9, 0.999), weight_decay=wd, cautious=True,
                      cautious_wd=arm, momentum_dtype="float32", foreach=foreach,
                      gradient_centralization=False)
        # Grad alternates sign; a positive momentum then disagrees on every odd column.
        g = torch.ones(2, 8)
        g[:, 1::2] = -1.0
        p.grad = g.clone()
        opt.step()                                     # step 1 builds the state
        before = p.detach().clone()
        opt.state[p]["m"].fill_(1.0)                   # momentum positive everywhere
        p.grad = g.clone()
        opt.step()
        out[arm] = (before - p.detach())               # the applied -lr*delta, sign flipped
    rejected = slice(1, None, 2)                       # grad<0, momentum>0 -> masked out
    assert out["masked"][:, rejected].abs().max() < 1e-6, "masked: rejected coords should not move"
    expected = lr * wd * 0.5
    torch.testing.assert_close(out["full"][:, rejected],
                               torch.full((2, 4), expected), rtol=2e-3, atol=1e-9)


@pytest.mark.parametrize("md", ["bfloat16", "float32", "int8", "4bit"])
def test_cautious_wd_per_param_matches_foreach(md):
    """Both placements must be element-for-element identical across the two native paths."""
    for arm in ("masked", "full"):
        cfg = dict(_WD_CFG, momentum_dtype=md, cautious_wd=arm)
        pa, pb = _wd_bag(), _wd_bag()
        oa, ob = Adakaon(pa, **cfg, foreach=True), Adakaon(pb, **cfg, foreach=False)
        _drive([(pa, oa), (pb, ob)], 6, torch.Generator().manual_seed(73))
        _assert_same(pa, pb, oa, ob)


@pytest.mark.parametrize("foreach", [False, True])
def test_cautious_wd_is_a_no_op_without_cautious_masking(foreach):
    """With ``cautious=False`` the mask is the identity, so the two orders are the same add."""
    cfg = dict(_WD_CFG, cautious=False, momentum_dtype="float32", foreach=foreach)
    pa, pb = _wd_bag(), _wd_bag()
    oa = Adakaon(pa, **cfg, cautious_wd="masked")
    ob = Adakaon(pb, **cfg, cautious_wd="full")
    _drive([(pa, oa), (pb, ob)], 6, torch.Generator().manual_seed(79))
    _assert_same(pa, pb, oa, ob)


@pytest.mark.parametrize("foreach", [False, True])
def test_cautious_wd_is_a_no_op_without_weight_decay(foreach):
    cfg = dict(_WD_CFG, weight_decay=0.0, momentum_dtype="float32", foreach=foreach)
    pa, pb = _wd_bag(), _wd_bag()
    oa = Adakaon(pa, **cfg, cautious_wd="masked")
    ob = Adakaon(pb, **cfg, cautious_wd="full")
    _drive([(pa, oa), (pb, ob)], 6, torch.Generator().manual_seed(83))
    _assert_same(pa, pb, oa, ob)


def test_cautious_wd_survives_a_checkpoint_without_the_key():
    """A checkpoint written before the key existed must not KeyError on the next step."""
    pa = _wd_bag()
    oa = Adakaon(pa, **_WD_CFG)
    _drive([(pa, oa)], 2, torch.Generator().manual_seed(89))
    sd = copy.deepcopy(oa.state_dict())
    for g in sd["param_groups"]:
        g.pop("cautious_wd")
    pb = _wd_bag()
    ob = Adakaon(pb, **_WD_CFG)
    ob.load_state_dict(sd)
    assert ob.param_groups[0]["cautious_wd"] == "masked"
    _drive([(pb, ob)], 1, torch.Generator().manual_seed(97))


# --------------------------------------------------------------------- 4bit + high beta1
# ``kaon._momentum_codec.warn_if_4bit_high_beta1`` (see tests/test_4bit_high_beta1_warning.py
# for the rest of the family; Adakaon lives here because that file was written while this one
# was locked by a concurrent audit batch).
def _warns_amplification(betas, momentum_dtype):
    import warnings
    p = [torch.nn.Parameter(torch.randn(4, 4))]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Adakaon(p, momentum_dtype=momentum_dtype, betas=betas)
    return any("amplif" in str(w.message).lower() or "1/sqrt" in str(w.message)
               for w in caught if issubclass(w.category, UserWarning))


def test_adakaon_warns_on_4bit_high_beta1():
    assert _warns_amplification((0.995, 0.999), "4bit")


def test_adakaon_no_warning_on_int8_high_beta1():
    assert not _warns_amplification((0.995, 0.999), "int8")


def test_adakaon_no_warning_on_4bit_low_beta1():
    assert not _warns_amplification((0.9, 0.999), "4bit")


# ------------------------------------------------------- witness cost (perf regression lock)
# Every cached fused/foreach plan revalidates itself with a per-parameter
# ``(id, data_ptr, is_contiguous)`` witness, once or twice per step. On a 428-parameter bag
# that is ~47 µs of pure Python call overhead — 4-11% of the step — and the ONLY thing
# keeping it there is that all three copies of the witness scan with ``map`` (the loop runs
# in C) rather than with generator expressions, which measured 75 µs for the same work.
# The three copies are deliberate duplicates (see their docstrings); this locks all of them.

def _witness_impls():
    from kaon._foreach_plan import param_witness as foreach_plan_witness
    from kaon.adakaon import _param_witness as adakaon_witness

    impls = [("kaon.adakaon._param_witness", adakaon_witness),
             ("kaon._foreach_plan.param_witness", foreach_plan_witness)]
    try:
        from kaon._fused_triton import param_witness as triton_witness
    except Exception:  # noqa: BLE001 — Triton is optional; the other two still apply
        pass
    else:
        impls.append(("kaon._fused_triton.param_witness", triton_witness))
    return impls


_WITNESS_IMPLS = _witness_impls()


@pytest.mark.parametrize("name,witness", _WITNESS_IMPLS,
                         ids=[n.split(".")[1] for n, _ in _WITNESS_IMPLS])
def test_param_witness_scans_in_c(name, witness):
    """The witness must build its tuples with ``map``, not generator expressions.

    Compared against a genexpr reference timed in the same loop, so the bound tracks the
    machine rather than an absolute number; the measured margin is ~1.6x, well clear of
    the 0.90 asserted here.

    The reference follows ``ft.SHAPE_WITNESS``: with that flag on, the FUSED witness carries a
    fourth per-param field (strides) and the other two implementations still do not — the native
    plan re-stacks by effective shape every step and needs no shape field. So this also pins
    which witness grows the field.
    """
    import time

    strides = False
    if "_fused_triton" in name:
        import kaon._fused_triton as ftm
        strides = ftm.SHAPE_WITNESS

    plist = [torch.empty(16 * 8 * 3 * 3) for _ in range(300)]
    plist += [torch.empty(1) for _ in range(128)]

    def genexpr_reference(pl):
        base = (tuple(id(p) for p in pl), tuple(p.data_ptr() for p in pl),
                tuple(p.is_contiguous() for p in pl))
        return (*base, tuple(p.stride() for p in pl)) if strides else base

    assert witness(plist) == genexpr_reference(plist), f"{name} changed the witness fields"

    def best_of(fn, reps=7, inner=20):
        out = []
        for _ in range(reps):
            t0 = time.perf_counter()
            for _ in range(inner):
                fn(plist)
            out.append((time.perf_counter() - t0) / inner)
        return min(out)

    ref = shipped = float("inf")
    for _ in range(3):  # interleaved so a scheduling hiccup cannot bias one arm
        ref = min(ref, best_of(genexpr_reference))
        shipped = min(shipped, best_of(witness))
    assert shipped <= 0.90 * ref, (
        f"{name} costs {shipped * 1e6:.1f} us on 428 params vs {ref * 1e6:.1f} us for plain "
        "generator expressions — the C-level map scan has been lost"
    )
