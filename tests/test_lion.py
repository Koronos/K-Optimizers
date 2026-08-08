"""Tests for the Lion (Lion sign-momentum on Adakaon's backend) optimizer."""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
import torch

from kaon import Lion

from .conftest import train_steps


def _ref_lion_step(
    p: np.ndarray,
    g: np.ndarray,
    m: np.ndarray,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    cautious: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference numpy Lion step mirroring Lion's fp32 per-param path.

    Returns ``(new_p, new_m)``. Cautious masking and decoupled weight decay
    follow the same order Lion uses (WD folded into the delta, then the
    cautious mask on ``delta * g``), so this is bit-comparable up to fp rounding.
    """
    c = beta1 * m + (1.0 - beta1) * g
    update = np.sign(c)                       # +1 / 0 / -1
    new_m = beta2 * m + (1.0 - beta2) * g     # EMA updated AFTER the direction
    delta = update + weight_decay * p
    if cautious:
        mask = (delta * g > 0).astype(delta.dtype)
        denom = max(mask.mean(), 1e-8)
        delta = delta * mask / denom
    new_p = p - lr * delta
    return new_p, new_m


def test_construct_and_step():
    """Construct and take a step on tiny CPU tensors (2-D + 1-D + conv)."""
    params = [
        torch.nn.Parameter(torch.randn(8, 4)),
        torch.nn.Parameter(torch.randn(5)),
        torch.nn.Parameter(torch.randn(3, 2, 3, 3)),
    ]
    opt = Lion(params, lr=1e-4, betas=(0.9, 0.99))
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()  # must not raise
    for p in params:
        assert torch.isfinite(p).all()


@pytest.mark.parametrize("cautious", [False, True])
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_matches_numpy_reference(cautious, weight_decay):
    """The fp32 sign-momentum math matches an independent numpy reference."""
    torch.manual_seed(0)
    lr, b1, b2 = 0.01, 0.9, 0.99
    p = torch.nn.Parameter(torch.randn(16, 7, dtype=torch.float64).float())
    opt = Lion(
        [p], lr=lr, betas=(b1, b2), weight_decay=weight_decay,
        momentum_dtype="float32", cautious=cautious, foreach=False,
        gradient_centralization=False,  # the numpy reference has no GC
    )
    pr = p.detach().numpy().copy().astype(np.float64)
    mr = np.zeros_like(pr)
    gg = torch.Generator().manual_seed(3)
    for _ in range(12):
        g = torch.randn(16, 7, generator=gg)
        p.grad = g.clone()
        opt.step()
        pr, mr = _ref_lion_step(
            pr, g.numpy().astype(np.float64), mr, lr, b1, b2, weight_decay, cautious
        )
    torch.testing.assert_close(p.detach().double(), torch.from_numpy(pr), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        opt.state[p]["m"].double(), torch.from_numpy(mr), rtol=1e-5, atol=1e-6
    )


def test_update_is_sign_of_interpolated_momentum():
    """On the very first step (m == 0) the direction is exactly sign(g)."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(64))
    g = torch.randn(64)
    g[g.abs() < 1e-3] = 0.5  # avoid exact-zero coords so sign is unambiguous
    opt = Lion([p], lr=1.0, betas=(0.9, 0.99), weight_decay=0.0,
                    momentum_dtype="float32", cautious=False, foreach=False)
    p.grad = g.clone()
    opt.step()
    # p was 0; with lr=1 and no WD, the step is -sign(g), so p == -sign(g).
    torch.testing.assert_close(p.detach(), -torch.sign(g))


def test_cautious_masks_disagreeing_coords():
    """Cautious zeroes coords where the update sign disagrees with the gradient.

    Build a momentum that points opposite the gradient on chosen coords so the
    interpolated direction disagrees with g there; those coords must not move,
    and the surviving step magnitude is rescaled up by 1/mean(mask).
    """
    torch.manual_seed(0)
    n = 100
    p = torch.nn.Parameter(torch.zeros(n))
    opt = Lion([p], lr=1.0, betas=(0.5, 0.99), weight_decay=0.0,
                    momentum_dtype="float32", cautious=True, foreach=False)
    # Seed momentum opposite to the gradient on the first half of the coords.
    g = torch.ones(n)
    opt.state[p]["m"] = torch.where(
        torch.arange(n) < n // 2, torch.full((n,), -10.0), torch.zeros(n)
    )
    p.grad = g.clone()
    opt.step()
    # c = 0.5*m + 0.5*g: first half -> -4.5 (sign -1, disagrees with g>0 -> masked),
    # second half -> +0.5 (sign +1, agrees). Masked coords stay at 0.
    moved = p.detach() != 0
    assert not moved[: n // 2].any(), "disagreeing coords must be masked (unchanged)"
    assert moved[n // 2 :].all(), "agreeing coords must move"
    # Rescale preserves mean magnitude: surviving step = lr / mean(mask) = 1 / 0.5 = 2.
    torch.testing.assert_close(
        p.detach()[n // 2 :], torch.full((n // 2,), -2.0)
    )


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
def test_momentum_dtype_variants_construct_and_step(momentum_dtype):
    """Every momentum_dtype constructs, steps without NaN, and stores the buffer."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(32, 16))
    opt = Lion([p], lr=1e-3, betas=(0.9, 0.99), momentum_dtype=momentum_dtype)
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()
    assert torch.isfinite(p).all()
    assert "m" in opt.state[p]
    expected = {
        "bfloat16": torch.bfloat16, "float32": torch.float32,
        "int8": torch.int8, "4bit": torch.uint8,
    }[momentum_dtype]
    assert opt.state[p]["m"].dtype == expected


def test_4bit_is_half_byte_per_param():
    """The 4-bit store is a real ~0.5 B/param packed buffer."""
    p = torch.nn.Parameter(torch.randn(256, 256))
    opt = Lion([p], betas=(0.9, 0.99), momentum_dtype="4bit", momentum_4bit_block=128)
    p.grad = torch.randn_like(p)
    opt.step()
    st = opt.state[p]
    assert st["m"].dtype == torch.uint8
    assert st["m"].numel() == (p.numel() + 1) // 2  # exactly 0.5 B/param packed


def test_int8_is_one_byte_per_param():
    p = torch.nn.Parameter(torch.randn(128, 128))
    opt = Lion([p], betas=(0.9, 0.99), momentum_dtype="int8")
    p.grad = torch.randn_like(p)
    opt.step()
    assert opt.state[p]["m"].numel() == p.numel()  # one int8 byte per param


def test_single_momentum_buffer_no_second_moment():
    """Lion keeps ONE momentum buffer and NO second moment (the memory win)."""
    p = torch.nn.Parameter(torch.randn(64, 64))
    opt = Lion([p], lr=1e-3, betas=(0.9, 0.99), momentum_dtype="float32")
    p.grad = torch.randn_like(p)
    opt.step()
    tensor_keys = {k for k, v in opt.state[p].items() if torch.is_tensor(v)}
    assert tensor_keys == {"m"}, f"expected only the momentum buffer, got {tensor_keys}"


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
        dict(lr=1e-3, betas=(0.9, 0.99), momentum_dtype="float32"),
        dict(lr=1e-3, betas=(0.9, 0.99), momentum_dtype="bfloat16"),
        dict(lr=1e-3, betas=(0.9, 0.99), momentum_dtype="int8"),
        dict(lr=1e-3, betas=(0.9, 0.99), momentum_dtype="int8", weight_decay=0.02),
        dict(lr=1e-3, betas=(0.9, 0.99), momentum_dtype="4bit"),
        dict(lr=1e-3, betas=(0.9, 0.99), momentum_dtype="4bit", weight_decay=0.02),
        dict(lr=1e-3, betas=(0.9, 0.99), momentum_dtype="4bit", momentum_4bit_block=64),
        dict(lr=1e-3, betas=(0.9, 0.99), weight_decay=0.02),
        dict(lr=1e-3, betas=(0.9, 0.99), cautious=True),
        dict(lr=1e-3, betas=(0.9, 0.99), cautious=False),
    ],
)
def test_foreach_matches_per_param(cfg):
    """foreach=True is element-for-element equal to the per-parameter path (fp32)."""
    pa = _parity_params()
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    oa = Lion(pa, foreach=True, **cfg)
    ob = Lion(pb, foreach=False, **cfg)
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
    oa = Lion(pa, lr=1e-3, betas=(0.9, 0.99), momentum_dtype="int8",
                   weight_decay=0.02, foreach=True, foreach_stack_budget=200)
    ob = Lion(pb, lr=1e-3, betas=(0.9, 0.99), momentum_dtype="int8",
                   weight_decay=0.02, foreach=False)
    gg = torch.Generator().manual_seed(7)
    for _ in range(8):
        for a, b in zip(pa, pb, strict=False):
            grad = torch.randn(*a.shape, generator=gg) * 0.02
            a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb, strict=False):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


def test_overfits_regression():
    torch.manual_seed(0xC0DE)
    model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 8))
    opt = Lion(model.parameters(), lr=3e-3, betas=(0.9, 0.99))
    x = torch.randn(64, 32)
    y = torch.randn(64, 8)
    initial = (model(x) - y).pow(2).mean().item()
    train_steps(model, opt, [(x, y)] * 120)
    final = (model(x) - y).pow(2).mean().item()
    assert final < 0.5 * initial, f"loss did not drop: {initial:.4f} -> {final:.4f}"


def test_bf16_weights_train_no_nan():
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 8)
    ).to(torch.bfloat16)
    opt = Lion(model.parameters(), lr=3e-3, betas=(0.9, 0.99),
                    bf16_method="stochastic_rounding")
    x = torch.randn(64, 32, dtype=torch.bfloat16)
    y = torch.randn(64, 8, dtype=torch.bfloat16)
    for _ in range(30):
        opt.zero_grad()
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
def test_checkpoint_roundtrip_preserves_momentum_dtype(momentum_dtype):
    """torch.save/load resumes BIT-EXACTLY and keeps the configured momentum dtype.

    torch's default load_state_dict upcasts state to the param dtype (fp32);
    Lion overrides load_state_dict to restore the stored dtype.
    """
    torch.manual_seed(0)
    p_ref = torch.randn(16, 8)
    grads = [torch.randn(16, 8) for _ in range(10)]

    a = torch.nn.Parameter(p_ref.clone())
    opt_a = Lion([a], lr=1e-3, betas=(0.9, 0.99), momentum_dtype=momentum_dtype)
    for g in grads[:5]:
        a.grad = g.clone()
        opt_a.step()

    buf = io.BytesIO()
    torch.save(opt_a.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, weights_only=False)

    b = torch.nn.Parameter(a.detach().clone())
    opt_b = Lion([b], lr=1e-3, betas=(0.9, 0.99), momentum_dtype=momentum_dtype)
    opt_b.load_state_dict(sd)

    assert opt_b.state[b]["m"].dtype == opt_a.state[a]["m"].dtype

    for g in grads[5:]:
        a.grad = g.clone()
        opt_a.step()
        b.grad = g.clone()
        opt_b.step()
    assert torch.equal(a, b), "resumed run must continue bit-exactly"


def test_invalid_args_rejected():
    p = [torch.nn.Parameter(torch.randn(4, 4))]
    with pytest.raises(ValueError):
        Lion(p, momentum_dtype="2bit")
    with pytest.raises(ValueError):
        Lion(p, betas=(1.0, 0.99))
    with pytest.raises(ValueError):
        Lion(p, betas=(0.9, 1.0))
    with pytest.raises(ValueError):
        Lion(p, lr=-1.0)
    with pytest.raises(ValueError):
        Lion(p, bf16_method="bogus")


def test_kahan_runs():
    """bf16 + kahan path (per-param, +shift buffer) steps without NaN."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 8, dtype=torch.bfloat16))
    opt = Lion([p], lr=1e-3, betas=(0.9, 0.99), bf16_method="kahan")
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()
    assert torch.isfinite(p).all()
    assert "shift" in opt.state[p]


def test_sparse_grad_rejected():
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = Lion([p], lr=1e-3, betas=(0.9, 0.99))
    p.grad = torch.sparse_coo_tensor(torch.tensor([[0], [0]]), torch.tensor([1.0]), (4, 4))
    with pytest.raises(RuntimeError):
        opt.step()


def test_conv_net_trains_no_nan():
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(4, 16, 3, padding=1), torch.nn.GELU(),
        torch.nn.Conv2d(16, 4, 3, padding=1),
    )
    opt = Lion(net.parameters(), lr=1e-3, betas=(0.9, 0.99))
    x = torch.randn(8, 4, 16, 16)
    y = torch.randn(8, 4, 16, 16)
    for _ in range(30):
        opt.zero_grad()
        loss = (net(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
    assert math.isfinite(loss.item())


# ============================================================ 0-D scalar params
# LyCORIS ``use_scalar`` gates (and friends) are 0-D weights. They used to be gated
# out of the foreach path and stepped one-by-one, which is ~20 CUDA launches per
# scalar per step; they now ride the non-factored ``L == 1`` bucket as length-1
# views. These tests pin the three things that can silently break: the routing, the
# element-for-element parity with the per-param path, and the state layout (so a
# checkpoint crosses between the two paths).
def _scalar_shapes() -> list[tuple[int, ...]]:
    """0-D scalars plus shape-``(1,)`` bucket-mates.

    Every entry has ``numel() == 1``, so the whole bag lands in the same
    non-factored bucket — 0-D as length-1 views, ``(1,)`` as-is — which is exactly
    the mixed stacking the batched path has to get right.
    """
    return [(), (), (), (1,), (1,)]


def _scalar_bag(shapes, seed: int = 11) -> list[torch.nn.Parameter]:
    g = torch.Generator().manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(s, generator=g) * 0.05) for s in shapes]


def _scalar_parity(cfg, shapes=None, steps: int = 10, seed: int = 11, grad_seed: int = 7):
    """Drive one bag of params and one grad sequence through both paths.

    Returns ``(pa, pb, oa, ob)`` — ``a`` is ``foreach=True``, ``b`` is ``foreach=False``.
    """
    pa = _scalar_bag(_scalar_shapes() if shapes is None else shapes, seed)
    pb = [torch.nn.Parameter(p.detach().clone()) for p in pa]
    oa = Lion(pa, foreach=True, **cfg)
    ob = Lion(pb, foreach=False, **cfg)
    gg = torch.Generator().manual_seed(grad_seed)
    for _ in range(steps):
        for a, b in zip(pa, pb, strict=True):
            grad = torch.randn(a.shape, generator=gg) * 0.02
            a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    return pa, pb, oa, ob


_SCALAR_CFGS = [
    {'lr': 0.0001, 'betas': (0.9, 0.99), 'momentum_dtype': 'float32'},
    {'lr': 0.0001, 'betas': (0.9, 0.99), 'momentum_dtype': 'bfloat16'},
    {'lr': 0.0001, 'betas': (0.9, 0.99), 'momentum_dtype': 'int8'},
    {'lr': 0.0001, 'betas': (0.9, 0.99), 'momentum_dtype': '4bit'},
    {'lr': 0.0001, 'betas': (0.9, 0.99), 'momentum_dtype': 'float32', 'weight_decay': 0.02},
    {'lr': 0.0001, 'betas': (0.9, 0.99), 'momentum_dtype': 'int8', 'weight_decay': 0.02},
    {'lr': 0.0001, 'betas': (0.9, 0.99), 'momentum_dtype': '4bit', 'weight_decay': 0.02},
    {'lr': 0.0001, 'betas': (0.9, 0.99), 'momentum_dtype': 'float32', 'cautious': False},
]


@pytest.mark.parametrize("cfg", _SCALAR_CFGS)
def test_foreach_scalar_0d_matches_per_param(cfg):
    """0-D scalars through the batched bucket are element-for-element equal to the
    per-parameter path (fp32 weights, so stochastic rounding is not in play).

    For a length-1 slice every per-slice reduction the batched code does must
    degenerate to the per-param scalar one (the RMS clip's ``norm/sqrt(1)``, the
    cautious mask's mean over one element, the int8 absmax over one element) — this
    test is the proof of that, not an assumption.
    """
    pa, pb, oa, _ob = _scalar_parity(cfg)
    for a, b in zip(pa, pb, strict=True):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)
    # State keeps the per-param layout: a 0-D param keeps a 0-D momentum buffer
    # (checkpoint compat). 4bit is exempt — its storage is nibble-packed, so the
    # buffer is a flat (1,) byte array whatever the param's shape.
    if cfg["momentum_dtype"] != "4bit":
        for a in pa:
            if a.ndim == 0:
                assert oa.state[a]["m"].shape == a.shape


def test_foreach_scalar_0d_takes_batched_path(monkeypatch):
    """0-D scalars must actually ride the batched bucket, not silently fall back to
    the per-param loop (the LyCORIS use_scalar pathology this guards against)."""
    looped = []
    orig = Lion._step_one_param

    def spy(self, p, *args, **kwargs):
        looped.append(p)
        return orig(self, p, *args, **kwargs)

    monkeypatch.setattr(Lion, "_step_one_param", spy)
    params = _scalar_bag(_scalar_shapes())
    opt = Lion(params, foreach=True, lr=0.0001, betas=(0.9, 0.99))
    gg = torch.Generator().manual_seed(3)
    for _ in range(3):
        for p in params:
            p.grad = torch.randn(p.shape, generator=gg) * 0.02
        opt.step()
    assert not looped, f"{len(looped)} params fell back to the per-param loop"


def test_foreach_scalar_0d_mixed_with_other_shapes():
    """0-D scalars mixed with 1-D / 2-D / conv params: the scalars stay bit-exact vs
    the per-param path and the other buckets keep their existing contract (the file's
    own parity tests own the exact claim for those; here they get a 1e-7 bound so this
    test is not hostage to value-dependent last-ulp drift)."""
    shapes = [(), (), (1,), (40,), (40,), (8, 16), (8, 16), (4, 4, 3, 3)]
    pa, pb, _oa, _ob = _scalar_parity({'lr': 0.0001, 'betas': (0.9, 0.99), 'momentum_dtype': 'float32'}, shapes=shapes, steps=8, seed=3, grad_seed=5)
    for a, b in zip(pa, pb, strict=True):
        if a.numel() == 1:
            torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)
        else:
            torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=1e-7)


def test_scalar_0d_checkpoint_roundtrip_across_paths():
    """A checkpoint saved mid-training by one path resumes bit-exactly on the other:
    the 0-D state layout is identical on both, in both directions."""
    cfg = {'lr': 0.0001, 'betas': (0.9, 0.99), 'momentum_dtype': 'int8', 'weight_decay': 0.02}
    for save_foreach, load_foreach in [(True, False), (False, True)]:
        pa = _scalar_bag(_scalar_shapes(), seed=29)
        oa = Lion(pa, foreach=save_foreach, **cfg)
        gg = torch.Generator().manual_seed(31)
        grads = [[torch.randn(p.shape, generator=gg) * 0.02 for p in pa] for _ in range(8)]
        for step in range(4):
            for p, gr in zip(pa, grads[step], strict=True):
                p.grad = gr.clone()
            oa.step()
        buf = io.BytesIO()
        torch.save({"opt": oa.state_dict(), "params": [p.detach().clone() for p in pa]}, buf)
        buf.seek(0)
        ckpt = torch.load(buf, weights_only=False)
        pb = [torch.nn.Parameter(t.clone()) for t in ckpt["params"]]
        ob = Lion(pb, foreach=load_foreach, **cfg)
        ob.load_state_dict(ckpt["opt"])
        for step in range(4, 8):
            for plist, opt in ((pa, oa), (pb, ob)):
                for p, gr in zip(plist, grads[step], strict=True):
                    p.grad = gr.clone()
                opt.step()
        for a, b in zip(pa, pb, strict=True):
            torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


def test_int8_stacked_store_keeps_per_param_scale_layout():
    """The batched int8 requant must store ``m_scale`` in the SAME layout the
    per-param path allocates, or a param that took the batched path once can never be
    stepped per-param again.

    Regression: the stacked store hardcoded ``(row, 1)``, which reshaped a conv's
    ``(R,1,1,1)`` scale to ``(R,1)`` and a 0-D param's ``()`` scale to ``(1,)``; the
    next per-param step on that state raised on the mismatched broadcast. Both shapes
    are exercised here because they fail differently (conv silently mis-broadcasts
    against the trailing dims, 0-D errors outright).
    """
    for shape in [(), (6,), (5, 4), (4, 4, 3, 3)]:
        batched = [torch.nn.Parameter(torch.randn(shape) * 0.05) for _ in range(3)]
        per_param = [torch.nn.Parameter(p.detach().clone()) for p in batched]
        oa = Lion(batched, foreach=True, momentum_dtype="int8", lr=1e-4)
        ob = Lion(per_param, foreach=False, momentum_dtype="int8", lr=1e-4)
        for pa, pb in zip(batched, per_param, strict=True):
            g = torch.randn(shape) * 0.02
            pa.grad, pb.grad = g.clone(), g.clone()
        oa.step()
        ob.step()
        for pa, pb in zip(batched, per_param, strict=True):
            assert oa.state[pa]['m_scale'].shape == ob.state[pb]['m_scale'].shape, shape
        # ...and the batched-built state really does resume on the per-param path.
        resume = Lion(batched, foreach=False, momentum_dtype="int8", lr=1e-4)
        for p in batched:
            resume.state[p] = oa.state[p]
            p.grad = torch.randn(shape) * 0.02
        resume.step()
        assert all(torch.isfinite(p).all() for p in batched), shape
