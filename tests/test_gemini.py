"""Tests for the Gemini (AdEMAMix two-EMA on koptim's factored backend) optimizer."""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
import torch

from koptim import Gemini
from koptim.gemini import schedule_alpha, schedule_beta3

from .conftest import train_steps


def _ref_ademamix_step(
    p: np.ndarray,
    g: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    v: np.ndarray,
    step: int,
    lr: float,
    beta1: float,
    beta2: float,
    beta3: float,
    alpha: float,
    eps1: float,
    weight_decay: float,
    cautious: bool,
    alpha_warmup: float | None = None,
    beta3_warmup: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Independent numpy AdEMAMix step mirroring Gemini's fp32 NON-factored path.

    This is the full per-coordinate second-moment branch (what Gemini uses for
    1-D params). It matches the kozistr reference EXCEPT the eps placement: Gemini
    folds Adafactor's ``eps1`` into ``g**2`` and uses the inv-sqrt form (no
    additive denominator eps), which we reproduce here. Bias correction, the
    slow-EMA mix and decoupled weight decay follow Gemini exactly.

    Returns ``(new_p, new_m1, new_m2, new_v)``.
    """
    alpha_t = schedule_alpha(alpha_warmup, step, alpha)
    beta3_t = schedule_beta3(beta3_warmup, step, beta1, beta3)
    new_m1 = beta1 * m1 + (1.0 - beta1) * g
    new_m2 = beta3_t * m2 + (1.0 - beta3_t) * g
    new_v = beta2 * v + (1.0 - beta2) * (g * g + eps1)
    bc1 = 1.0 - beta1**step
    bc2_sq = math.sqrt(1.0 - beta2**step)
    update = (new_m1 + alpha_t * new_m2) / np.sqrt(new_v) * bc2_sq
    delta = update * (lr / bc1)
    if weight_decay != 0.0:
        delta = delta + lr * weight_decay * p
    if cautious:
        mask = (delta * g > 0).astype(delta.dtype)
        denom = max(mask.mean(), 1e-8)
        delta = delta * mask / denom
    new_p = p - delta
    return new_p, new_m1, new_m2, new_v


def test_construct_and_step():
    """Construct and take a step on tiny CPU tensors (2-D + 1-D + conv)."""
    params = [
        torch.nn.Parameter(torch.randn(8, 4)),
        torch.nn.Parameter(torch.randn(5)),
        torch.nn.Parameter(torch.randn(3, 2, 3, 3)),
    ]
    opt = Gemini(params, lr=1e-3)
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()  # must not raise
    for p in params:
        assert torch.isfinite(p).all()


@pytest.mark.parametrize("cautious", [False, True])
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_matches_numpy_reference_1d(cautious, weight_decay):
    """The fp32 NON-factored (1-D) path matches an independent numpy AdEMAMix step."""
    lr, b1, b2, b3, alpha, eps1 = 1e-2, 0.9, 0.999, 0.9999, 5.0, 1e-30
    p = torch.nn.Parameter(torch.randn(40, dtype=torch.float64).float())
    opt = Gemini(
        [p], lr=lr, betas=(b1, b2, b3), alpha=alpha, eps=eps1,
        weight_decay=weight_decay, momentum_dtype="float32", cautious=cautious,
        foreach=False,
    )
    pr = p.detach().numpy().copy().astype(np.float64)
    m1r = np.zeros_like(pr)
    m2r = np.zeros_like(pr)
    vr = np.zeros_like(pr)
    gg = torch.Generator().manual_seed(3)
    for step in range(1, 16):
        g = torch.randn(40, generator=gg)
        p.grad = g.clone()
        opt.step()
        pr, m1r, m2r, vr = _ref_ademamix_step(
            pr, g.numpy().astype(np.float64), m1r, m2r, vr, step,
            lr, b1, b2, b3, alpha, eps1, weight_decay, cautious,
        )
    torch.testing.assert_close(p.detach().double(), torch.from_numpy(pr), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        opt.state[p]["m1_m"].double(), torch.from_numpy(m1r), rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        opt.state[p]["m2_m"].double(), torch.from_numpy(m2r), rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        opt.state[p]["v"].double(), torch.from_numpy(vr), rtol=1e-5, atol=1e-6
    )


def test_both_emas_update_at_their_own_betas():
    """m1 follows beta1 (fast), m2 follows beta3 (slow): after one step they differ.

    On step 1 from zero state both are ``(1-beta)*g`` with their own beta, so the
    fast EMA holds a much larger fraction of g than the slow EMA.
    """
    p = torch.nn.Parameter(torch.zeros(50))
    g = torch.randn(50)
    opt = Gemini([p], lr=1e-3, betas=(0.9, 0.999, 0.9999), alpha=5.0,
                 momentum_dtype="float32", cautious=False, foreach=False)
    p.grad = g.clone()
    opt.step()
    m1 = opt.state[p]["m1_m"]
    m2 = opt.state[p]["m2_m"]
    torch.testing.assert_close(m1, 0.1 * g, rtol=1e-5, atol=1e-6)   # 1-0.9
    torch.testing.assert_close(m2, 1e-4 * g, rtol=1e-4, atol=1e-7)  # 1-0.9999
    # The slow EMA moves ~1000x less than the fast one on the first step.
    assert m1.abs().sum() > 100 * m2.abs().sum()


def test_alpha_mix_scales_the_slow_ema():
    """The slow EMA enters the update scaled by alpha_t (numerator m1 + alpha*m2).

    Run two optimizers identical but for alpha; with the slow EMA non-zero the
    larger alpha must move the parameter more (same sign of contribution here).
    """
    torch.manual_seed(0)
    g_seq = [torch.randn(64) for _ in range(8)]

    def run(alpha):
        p = torch.nn.Parameter(torch.zeros(64))
        opt = Gemini([p], lr=1e-2, betas=(0.9, 0.999, 0.9999), alpha=alpha,
                     momentum_dtype="float32", cautious=False, foreach=False)
        for g in g_seq:
            p.grad = g.clone()
            opt.step()
        return p.detach().clone()

    p0 = run(0.0)   # plain-Adam-like (no slow EMA contribution)
    p5 = run(5.0)
    p10 = run(10.0)
    # With a non-trivial slow EMA, alpha changes the trajectory monotonically.
    assert not torch.allclose(p0, p5)
    assert (p10 - p0).abs().sum() > (p5 - p0).abs().sum()


def test_alpha_zero_drops_slow_ema_contribution():
    """alpha=0 makes the slow EMA irrelevant to the update (numerator is just m1)."""
    torch.manual_seed(0)
    g_seq = [torch.randn(8, 8) for _ in range(5)]
    pa = torch.nn.Parameter(torch.zeros(8, 8))
    pb = torch.nn.Parameter(torch.zeros(8, 8))
    oa = Gemini([pa], lr=1e-2, alpha=0.0, momentum_dtype="float32",
                cautious=False, foreach=False)
    ob = Gemini([pb], lr=1e-2, alpha=0.0, betas=(0.9, 0.999, 0.5),  # different beta3
                momentum_dtype="float32", cautious=False, foreach=False)
    for g in g_seq:
        pa.grad = g.clone()
        pb.grad = g.clone()
        oa.step()
        ob.step()
    # beta3 only feeds m2; with alpha=0 m2 never reaches the update, so the two
    # parameter trajectories are identical despite different beta3.
    torch.testing.assert_close(pa.detach(), pb.detach(), rtol=1e-6, atol=1e-7)


def test_warmup_schedules_match_formulas():
    """schedule_alpha / schedule_beta3 reproduce the paper / kozistr formulas."""
    alpha, b1, b3, t = 8.0, 0.9, 0.9999, 1000
    # alpha: linear ramp, capped.
    assert schedule_alpha(None, 5, alpha) == alpha
    assert schedule_alpha(t, 0, alpha) == 0.0
    assert schedule_alpha(t, 500, alpha) == pytest.approx(4.0)
    assert schedule_alpha(t, 5000, alpha) == alpha  # capped
    # beta3: constant when off; b1 at step 0; b3 (capped) at/after the horizon.
    assert schedule_beta3(None, 5, b1, b3) == b3
    assert schedule_beta3(t, 0, b1, b3) == pytest.approx(b1)
    assert schedule_beta3(t, t, b1, b3) == pytest.approx(b3)
    mid = schedule_beta3(t, 500, b1, b3)
    assert b1 < mid < b3  # monotone interior
    log_b1, log_b3 = math.log(b1), math.log(b3)
    s = 500 / t
    expect = math.exp(log_b1 * log_b3 / ((1.0 - s) * log_b3 + s * log_b1))
    assert mid == pytest.approx(expect)


def test_warmup_reference_parity():
    """End-to-end fp32 parity with the numpy reference WITH both warmups active."""
    lr, b1, b2, b3, alpha, eps1 = 1e-2, 0.9, 0.999, 0.9999, 8.0, 1e-30
    tw = 6
    p = torch.nn.Parameter(torch.randn(40, dtype=torch.float64).float())
    opt = Gemini(
        [p], lr=lr, betas=(b1, b2, b3), alpha=alpha, eps=eps1,
        alpha_warmup=tw, beta3_warmup=tw, momentum_dtype="float32",
        cautious=True, foreach=False,
    )
    pr = p.detach().numpy().copy().astype(np.float64)
    m1r = np.zeros_like(pr)
    m2r = np.zeros_like(pr)
    vr = np.zeros_like(pr)
    gg = torch.Generator().manual_seed(11)
    for step in range(1, 13):
        g = torch.randn(40, generator=gg)
        p.grad = g.clone()
        opt.step()
        pr, m1r, m2r, vr = _ref_ademamix_step(
            pr, g.numpy().astype(np.float64), m1r, m2r, vr, step,
            lr, b1, b2, b3, alpha, eps1, 0.0, True,
            alpha_warmup=tw, beta3_warmup=tw,
        )
    torch.testing.assert_close(p.detach().double(), torch.from_numpy(pr), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
def test_momentum_dtype_variants_construct_and_step(momentum_dtype):
    """Every momentum_dtype constructs, steps without NaN, and stores BOTH buffers."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(32, 16))
    opt = Gemini([p], lr=1e-3, momentum_dtype=momentum_dtype)
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()
    assert torch.isfinite(p).all()
    assert "m1_m" in opt.state[p]
    assert "m2_m" in opt.state[p]
    expected = {
        "bfloat16": torch.bfloat16, "float32": torch.float32,
        "int8": torch.int8, "4bit": torch.uint8,
    }[momentum_dtype]
    assert opt.state[p]["m1_m"].dtype == expected
    assert opt.state[p]["m2_m"].dtype == expected


def test_two_momenta_one_factored_v():
    """Gemini keeps TWO momentum buffers and a FACTORED second moment (the memory story).

    For a 2-D weight: two momenta (m1, m2) + factored row/col v (no full v). This
    is Adam's one momentum + one extra, with the second moment near-free.
    """
    p = torch.nn.Parameter(torch.randn(64, 64))
    opt = Gemini([p], lr=1e-3, momentum_dtype="float32")
    p.grad = torch.randn_like(p)
    opt.step()
    tensor_keys = {k for k, v in opt.state[p].items() if torch.is_tensor(v)}
    assert tensor_keys == {"m1_m", "m2_m", "row", "col"}, tensor_keys
    # Factored v: row + col floats, NOT 64*64.
    assert opt.state[p]["row"].numel() + opt.state[p]["col"].numel() == 128


def test_int8_is_one_byte_per_param_each_momentum():
    p = torch.nn.Parameter(torch.randn(128, 128))
    opt = Gemini([p], momentum_dtype="int8")
    p.grad = torch.randn_like(p)
    opt.step()
    assert opt.state[p]["m1_m"].numel() == p.numel()
    assert opt.state[p]["m2_m"].numel() == p.numel()
    assert opt.state[p]["m1_m"].dtype == torch.int8


def test_4bit_is_half_byte_per_param_each_momentum():
    p = torch.nn.Parameter(torch.randn(256, 256))
    opt = Gemini([p], momentum_dtype="4bit", momentum_4bit_block=128)
    p.grad = torch.randn_like(p)
    opt.step()
    for key in ("m1_m", "m2_m"):
        assert opt.state[p][key].dtype == torch.uint8
        assert opt.state[p][key].numel() == (p.numel() + 1) // 2


def _parity_params():
    g = torch.Generator().manual_seed(0)
    shapes = [
        (64, 128), (128, 64), (64, 128),   # 2-D, one shape repeated -> bucket N=2
        (32, 8, 3, 3),                     # conv
        (8, 96), (96, 8),                  # same numel, different shape (must not co-bucket)
        (40,), (40,), (128,), (320,),      # 1-D
    ]
    return [torch.nn.Parameter(torch.randn(*s, generator=g) * 0.05) for s in shapes]


@pytest.mark.parametrize(
    "cfg",
    [
        dict(momentum_dtype="float32"),
        dict(momentum_dtype="bfloat16"),
        dict(momentum_dtype="int8"),
        dict(momentum_dtype="int8", weight_decay=0.02),
        dict(momentum_dtype="4bit"),
        dict(momentum_dtype="4bit", weight_decay=0.02),
        dict(momentum_dtype="4bit", momentum_4bit_block=64),
        dict(weight_decay=0.02),
        dict(cautious=True),
        dict(cautious=False),
        dict(clip_threshold=1.0),
        dict(alpha=10.0, alpha_warmup=20, beta3_warmup=20),
    ],
)
def test_foreach_matches_per_param(cfg):
    """foreach=True is element-for-element equal to the per-parameter path (fp32).

    The hard part for Gemini: the bucketed path must update BOTH momenta exactly
    as the per-param path (m1 at beta1, m2 at beta3_t), including the warmup
    schedule and the per-bucket step counter.
    """
    pa = _parity_params()
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    oa = Gemini(pa, lr=1e-3, foreach=True, **cfg)
    ob = Gemini(pb, lr=1e-3, foreach=False, **cfg)
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
    """A tiny stack budget splits buckets and routes large weights to the loop —
    the result must still equal the per-parameter path exactly (int8 + WD)."""
    pa = _parity_params()
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    oa = Gemini(pa, lr=1e-3, momentum_dtype="int8", weight_decay=0.02,
                foreach=True, foreach_stack_budget=200)
    ob = Gemini(pb, lr=1e-3, momentum_dtype="int8", weight_decay=0.02, foreach=False)
    gg = torch.Generator().manual_seed(7)
    for _ in range(8):
        for a, b in zip(pa, pb, strict=False):
            grad = torch.randn(*a.shape, generator=gg) * 0.02
            a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb, strict=False):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


def test_cautious_masks_disagreeing_coords():
    """Cautious zeroes coords where the update sign disagrees with the gradient.

    Seed both momenta opposite the gradient on the first half so the numerator
    (m1 + alpha*m2) disagrees with g there; those coords must not move, survivors
    are rescaled up by 1/mean(mask).
    """
    n = 100
    p = torch.nn.Parameter(torch.zeros(n))
    opt = Gemini([p], lr=1.0, betas=(0.9, 0.999, 0.9999), alpha=5.0,
                 eps=1e-30, momentum_dtype="float32", cautious=True, foreach=False)
    g = torch.ones(n)
    seed = torch.where(torch.arange(n) < n // 2, torch.full((n,), -10.0), torch.zeros(n))
    opt.state[p]["step"] = 0
    opt.state[p]["v"] = torch.zeros(n)
    opt.state[p]["m1_m"] = seed.clone()
    opt.state[p]["m2_m"] = seed.clone()
    p.grad = g.clone()
    opt.step()
    moved = p.detach() != 0
    assert not moved[: n // 2].any(), "disagreeing coords must be masked (unchanged)"
    assert moved[n // 2 :].all(), "agreeing coords must move"


def test_overfits_regression():
    torch.manual_seed(0xC0DE)
    model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 8))
    opt = Gemini(model.parameters(), lr=3e-3, betas=(0.9, 0.999, 0.9999), alpha=5.0)
    x = torch.randn(64, 32)
    y = torch.randn(64, 8)
    initial = (model(x) - y).pow(2).mean().item()
    train_steps(model, opt, [(x, y)] * 150)
    final = (model(x) - y).pow(2).mean().item()
    assert final < 0.5 * initial, f"loss did not drop: {initial:.4f} -> {final:.4f}"


def test_bf16_weights_train_no_nan():
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 8)
    ).to(torch.bfloat16)
    opt = Gemini(model.parameters(), lr=3e-3, bf16_method="stochastic_rounding")
    x = torch.randn(64, 32, dtype=torch.bfloat16)
    y = torch.randn(64, 8, dtype=torch.bfloat16)
    for _ in range(30):
        opt.zero_grad()
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
def test_checkpoint_roundtrip_preserves_both_momenta_and_v(momentum_dtype):
    """torch.save/load resumes BIT-EXACTLY, preserving BOTH momenta + factored v dtype.

    torch's default load_state_dict upcasts state to the param dtype (fp32);
    Gemini overrides load_state_dict to restore the stored dtype of m1, m2 and the
    factored row/col of v.
    """
    torch.manual_seed(0)
    p_ref = torch.randn(16, 8)
    grads = [torch.randn(16, 8) for _ in range(10)]

    a = torch.nn.Parameter(p_ref.clone())
    opt_a = Gemini([a], lr=1e-3, momentum_dtype=momentum_dtype)
    for g in grads[:5]:
        a.grad = g.clone()
        opt_a.step()

    buf = io.BytesIO()
    torch.save(opt_a.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, weights_only=False)

    b = torch.nn.Parameter(a.detach().clone())
    opt_b = Gemini([b], lr=1e-3, momentum_dtype=momentum_dtype)
    opt_b.load_state_dict(sd)

    # Both momenta keep their stored dtype; the factored v stays fp32.
    assert opt_b.state[b]["m1_m"].dtype == opt_a.state[a]["m1_m"].dtype
    assert opt_b.state[b]["m2_m"].dtype == opt_a.state[a]["m2_m"].dtype
    assert opt_b.state[b]["row"].dtype == torch.float32
    assert opt_b.state[b]["col"].dtype == torch.float32
    assert opt_b.state[b]["step"] == opt_a.state[a]["step"]

    for g in grads[5:]:
        a.grad = g.clone()
        opt_a.step()
        b.grad = g.clone()
        opt_b.step()
    assert torch.equal(a, b), "resumed run must continue bit-exactly"


def test_kahan_runs():
    """bf16 + kahan path (per-param, +shift buffer) steps without NaN."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 8, dtype=torch.bfloat16))
    opt = Gemini([p], lr=1e-3, bf16_method="kahan")
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()
    assert torch.isfinite(p).all()
    assert "shift" in opt.state[p]


def test_conv_net_trains_no_nan():
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(4, 16, 3, padding=1), torch.nn.GELU(),
        torch.nn.Conv2d(16, 4, 3, padding=1),
    )
    opt = Gemini(net.parameters(), lr=1e-3)
    x = torch.randn(8, 4, 16, 16)
    y = torch.randn(8, 4, 16, 16)
    for _ in range(30):
        opt.zero_grad()
        loss = (net(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
    assert math.isfinite(loss.item())


def test_invalid_args_rejected():
    p = [torch.nn.Parameter(torch.randn(4, 4))]
    with pytest.raises(ValueError):
        Gemini(p, momentum_dtype="2bit")
    with pytest.raises(ValueError):
        Gemini(p, betas=(1.0, 0.999, 0.9999))
    with pytest.raises(ValueError):
        Gemini(p, betas=(0.9, 0.999, 1.0))
    with pytest.raises(ValueError):
        Gemini(p, lr=-1.0)
    with pytest.raises(ValueError):
        Gemini(p, alpha=-1.0)
    with pytest.raises(ValueError):
        Gemini(p, alpha_warmup=0)
    with pytest.raises(ValueError):
        Gemini(p, bf16_method="bogus")


def test_sparse_grad_rejected():
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = Gemini([p], lr=1e-3)
    p.grad = torch.sparse_coo_tensor(torch.tensor([[0], [0]]), torch.tensor([1.0]), (4, 4))
    with pytest.raises(RuntimeError):
        opt.step()
