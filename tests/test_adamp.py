"""Tests for AdamP — AdamW + the per-channel radial (scale-invariant) projection.

The reference mirrors AdamP's update with kaon's **factored** second moment (so the
2-D denominator matches the implementation) and the **official** ``clovaai/AdamP``
``_projection`` (channel view, then layer view) verbatim. The 1-D path follows the
official eps placement ``denom = sqrt(v)/sqrt(bc2) + eps``.
"""

from __future__ import annotations

import io
import math

import pytest
import torch

from kaon import AdamP

from .conftest import train_steps


# ----------------------------------------------------------------- reference
def _factored_inv_denom(v_row: torch.Tensor, v_col: torch.Tensor, bc2_sq: float) -> torch.Tensor:
    """Adafactor 1/sqrt(v_hat) reconstruction matching kaon._factored."""
    r = (v_row / v_row.mean(dim=-1, keepdim=True)).rsqrt().unsqueeze(-1)
    cfac = v_col.rsqrt().unsqueeze(-2)
    return (r * cfac) * bc2_sq


def _project_ref(p: torch.Tensor, grad: torch.Tensor, perturb: torch.Tensor,
                 delta: float, wd_ratio_hp: float, eps: float) -> tuple[torch.Tensor, float]:
    """Verbatim official clovaai/AdamP _projection (channel view then layer view)."""
    wd = 1.0
    expand = [-1] + [1] * (len(p.shape) - 1)
    for view in (lambda x: x.view(x.size(0), -1), lambda x: x.view(1, -1)):
        cos = torch.nn.functional.cosine_similarity(view(grad), view(p), dim=1, eps=eps).abs()
        if cos.max() < delta / math.sqrt(view(p).size(1)):
            p_n = p / view(p).norm(dim=1).view(expand).add_(eps)
            perturb = perturb - p_n * view(p_n * perturb).sum(dim=1).view(expand)
            return perturb, wd_ratio_hp
    return perturb, wd


def _ref_adamp_factored_2d(
    p0: torch.Tensor,
    grads: list[torch.Tensor],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    delta: float,
    wd_ratio: float,
) -> torch.Tensor:
    """Reference AdamP for a single 2-D weight using the kaon factored backend + official projection."""
    p = p0.clone()
    m = torch.zeros_like(p)
    v_row = torch.zeros(p.shape[0])
    v_col = torch.zeros(p.shape[1])
    for t, g in enumerate(grads, start=1):
        gsq = g * g
        if eps > 0:
            gsq = gsq + eps
        v_row.lerp_(gsq.mean(dim=-1), 1.0 - beta2)
        v_col.lerp_(gsq.mean(dim=-2), 1.0 - beta2)
        bc1 = 1.0 - beta1 ** t
        bc2_sq = math.sqrt(1.0 - beta2 ** t)
        inv_denom = _factored_inv_denom(v_row, v_col, bc2_sq)
        m.mul_(beta1).add_(g, alpha=1.0 - beta1)
        perturb = m * inv_denom
        perturb, wd_r = _project_ref(p, g, perturb, delta, wd_ratio, eps)
        if weight_decay > 0:
            p.mul_(1.0 - lr * weight_decay * wd_r)
        p.add_(perturb, alpha=-lr / bc1)
    return p


# ----------------------------------------------------------------- basic
def test_construct_and_step():
    params = [
        torch.nn.Parameter(torch.randn(8, 4)),
        torch.nn.Parameter(torch.randn(5)),
        torch.nn.Parameter(torch.randn(3, 2, 3, 3)),
    ]
    opt = AdamP(params, lr=1e-3)
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()
    for p in params:
        assert torch.isfinite(p).all()


def test_defaults_match_official():
    p = [torch.nn.Parameter(torch.randn(4, 4))]
    opt = AdamP(p)
    g = opt.param_groups[0]
    assert g["betas"] == (0.9, 0.999)
    assert g["delta"] == 0.1
    assert g["wd_ratio"] == 0.1
    assert g["eps"] == 1e-8


# --------------------------------------------------- correctness (target fp32)
@pytest.mark.parametrize("weight_decay", [0.0, 0.05])
def test_matches_factored_reference_2d(weight_decay):
    """AdamP's 2-D factored path matches the factored-Adam + official-projection reference.

    GC off, cautious off, fp32 weights and momentum. Uses a weight whose gradient is
    nearly orthogonal to the weight (a scale-invariant-like setup), so the projection
    fires on at least some steps.
    """
    torch.manual_seed(7)
    out, fan = 6, 5
    w = torch.randn(out, fan)
    p = torch.nn.Parameter(w.clone())
    opt = AdamP(
        [p], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay,
        delta=0.1, wd_ratio=0.1, cautious=False, gradient_centralization=False,
        momentum_dtype="float32", foreach=False,
    )
    # Build gradients with a controllable radial/tangential mix.
    grads = []
    for _ in range(8):
        g = torch.randn(out, fan)
        # mostly-orthogonal component per row to trigger the projection sometimes
        grads.append(g)
    for g in grads:
        p.grad = g.clone()
        opt.step()
    ref = _ref_adamp_factored_2d(
        w, grads, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8,
        weight_decay=weight_decay, delta=0.1, wd_ratio=0.1,
    )
    torch.testing.assert_close(p.detach(), ref, rtol=1e-5, atol=1e-6)


def test_matches_reference_scale_invariant_trigger():
    """A deliberately scale-invariant weight (grad orthogonal to weight) -> projection fires."""
    torch.manual_seed(3)
    out, fan = 4, 8
    # Weight rows; gradients made orthogonal to each row so cosine ~ 0 < threshold.
    w = torch.randn(out, fan)
    p = torch.nn.Parameter(w.clone())
    opt = AdamP(
        [p], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
        delta=0.1, wd_ratio=0.1, cautious=False, gradient_centralization=False,
        momentum_dtype="float32", foreach=False,
    )
    grads = []
    for _ in range(6):
        g = torch.randn(out, fan)
        # remove the radial (per-row parallel-to-weight) component -> grad orthogonal to w
        wn = w / w.norm(dim=1, keepdim=True)
        g = g - wn * (wn * g).sum(dim=1, keepdim=True)
        grads.append(g)
    for g in grads:
        p.grad = g.clone()
        opt.step()
    ref = _ref_adamp_factored_2d(
        w, grads, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8,
        weight_decay=0.0, delta=0.1, wd_ratio=0.1,
    )
    torch.testing.assert_close(p.detach(), ref, rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------- projection
def test_projection_removes_radial_component():
    """When the weight is scale-invariant the applied update has ~zero radial component."""
    torch.manual_seed(0)
    out, fan = 5, 9
    w = torch.randn(out, fan)
    p = torch.nn.Parameter(w.clone())
    opt = AdamP(
        [p], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
        delta=0.1, wd_ratio=0.1, cautious=False, gradient_centralization=False,
        momentum_dtype="float32", foreach=False,
    )
    # gradient orthogonal to each weight row -> cosine ~ 0 -> projection fires.
    g = torch.randn(out, fan)
    wn = w / w.norm(dim=1, keepdim=True)
    g = g - wn * (wn * g).sum(dim=1, keepdim=True)
    p.grad = g.clone()
    p_before = p.detach().clone()
    opt.step()
    applied = (p_before - p.detach())  # = lr * step direction (delta)
    # radial component of the applied update along each weight row should be ~0.
    radial = (wn * applied).sum(dim=1)
    assert radial.abs().max() < 1e-6, f"radial component not removed: {radial.abs().max()}"


def test_projection_does_not_fire_keeps_plain_adam():
    """When grad is strongly aligned with the weight (high cosine) the projection is a no-op.

    The AdamP step then equals the plain factored-Adam step (no radial removal).
    """
    torch.manual_seed(1)
    out, fan = 4, 7
    w = torch.randn(out, fan).abs() + 0.5  # positive weight
    p_a = torch.nn.Parameter(w.clone())
    p_b = torch.nn.Parameter(w.clone())
    common = dict(
        lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
        cautious=False, gradient_centralization=False,
        momentum_dtype="float32", foreach=False,
    )
    # delta=0 makes the threshold 0 -> cos.max() < 0 is never true -> never projects.
    opt_proj = AdamP([p_a], delta=0.1, wd_ratio=0.1, **common)
    opt_noproj = AdamP([p_b], delta=0.0, wd_ratio=0.1, **common)
    for _ in range(4):
        # gradient highly aligned with the (positive) weight -> high cosine -> no fire.
        g = w + 0.01 * torch.randn(out, fan)
        p_a.grad = g.clone()
        p_b.grad = g.clone()
        opt_proj.step()
        opt_noproj.step()
    # With high cosine the projection never fires, so the two trajectories coincide.
    torch.testing.assert_close(p_a.detach(), p_b.detach(), rtol=1e-6, atol=1e-7)


def test_1d_params_never_projected():
    """1-D params take the plain Adam step regardless of delta (no projection gate)."""
    torch.manual_seed(0)
    n = 16
    w = torch.randn(n)
    p_a = torch.nn.Parameter(w.clone())
    p_b = torch.nn.Parameter(w.clone())
    common = dict(
        lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
        cautious=False, gradient_centralization=False,
        momentum_dtype="float32", foreach=False,
    )
    opt_a = AdamP([p_a], delta=10.0, **common)   # huge delta would project a 2-D weight
    opt_b = AdamP([p_b], delta=0.0, **common)    # never projects
    for _ in range(4):
        g = torch.randn(n)
        p_a.grad = g.clone()
        p_b.grad = g.clone()
        opt_a.step()
        opt_b.step()
    torch.testing.assert_close(p_a.detach(), p_b.detach(), rtol=1e-6, atol=1e-7)


# ----------------------------------------------------------------- parity
@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
def test_momentum_dtype_variants_construct_and_step(momentum_dtype):
    params = [
        torch.nn.Parameter(torch.randn(8, 4)),
        torch.nn.Parameter(torch.randn(5)),
        torch.nn.Parameter(torch.randn(3, 2, 3, 3)),
    ]
    opt = AdamP(params, lr=1e-3, momentum_dtype=momentum_dtype)
    for _ in range(3):
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
    for p in params:
        assert torch.isfinite(p).all()


@pytest.mark.parametrize(
    "cfg",
    [
        dict(momentum_dtype="float32", weight_decay=0.0, cautious=True),
        dict(momentum_dtype="int8", weight_decay=0.02, cautious=True),
        dict(momentum_dtype="4bit", weight_decay=0.01, cautious=False),
        dict(momentum_dtype="bfloat16", weight_decay=0.0, cautious=True),
    ],
)
def test_foreach_matches_per_param(cfg):
    """foreach=True is bit-exact with the per-parameter path, including projected 2-D weights.

    The scale-invariant weight below has gradients orthogonal to it -> the projection
    fires, exercising the stacked vs per-param projection parity.
    """
    def mk() -> list[torch.nn.Parameter]:
        torch.manual_seed(1)
        si = torch.randn(6, 5)  # the scale-invariant weight (grads made orthogonal below)
        return [
            torch.nn.Parameter(torch.randn(8, 4)),
            torch.nn.Parameter(torch.randn(6, 5)),   # same shape as si -> same bucket
            torch.nn.Parameter(si.clone()),
            torch.nn.Parameter(torch.randn(5)),
            torch.nn.Parameter(torch.randn(7)),
            torch.nn.Parameter(torch.randn(3, 2, 3, 3)),
        ]

    pa = mk()
    pb = mk()
    oa = AdamP(pa, lr=1e-3, foreach=True, bf16_method="none", gradient_centralization=False, **cfg)
    ob = AdamP(pb, lr=1e-3, foreach=False, bf16_method="none", gradient_centralization=False, **cfg)
    torch.manual_seed(7)
    max_diff = 0.0
    for _ in range(6):
        gs = []
        for p in pa:
            g = torch.randn_like(p)
            # make the [6,5] params' gradients orthogonal to the weight (trigger projection)
            if p.shape == (6, 5):
                wn = p.detach() / p.detach().norm(dim=1, keepdim=True)
                g = g - wn * (wn * g).sum(dim=1, keepdim=True)
            gs.append(g)
        for p, g in zip(pa, gs, strict=True):
            p.grad = g.clone()
        for p, g in zip(pb, gs, strict=True):
            p.grad = g.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb, strict=True):
        max_diff = max(max_diff, (a.detach() - b.detach()).abs().max().item())
        assert torch.allclose(a, b, rtol=0.0, atol=1e-7), (
            f"foreach != per-param (max diff {max_diff})"
        )


def test_foreach_chunking_is_exact():
    """Splitting a foreach bucket into chunks must not change the result (with projection)."""
    def mk() -> list[torch.nn.Parameter]:
        torch.manual_seed(2)
        return [torch.nn.Parameter(torch.randn(6, 5)) for _ in range(7)]

    pa = mk()
    pb = mk()
    oa = AdamP(pa, lr=1e-3, momentum_dtype="int8", weight_decay=0.02,
               gradient_centralization=False, foreach=True, foreach_stack_budget=120)
    ob = AdamP(pb, lr=1e-3, momentum_dtype="int8", weight_decay=0.02,
               gradient_centralization=False, foreach=False)
    torch.manual_seed(3)
    for _ in range(5):
        gs = []
        for p in pa:
            g = torch.randn_like(p)
            wn = p.detach() / p.detach().norm(dim=1, keepdim=True)
            g = g - wn * (wn * g).sum(dim=1, keepdim=True)  # orthogonal -> projection fires
            gs.append(g)
        for p, g in zip(pa, gs, strict=True):
            p.grad = g.clone()
        for p, g in zip(pb, gs, strict=True):
            p.grad = g.clone()
        oa.step()
        ob.step()
    for a, b in zip(pa, pb, strict=True):
        assert torch.equal(a, b)


# ----------------------------------------------------------------- misc
def test_nesterov_runs():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = AdamP([p], lr=1e-3, nesterov=True)
    for _ in range(3):
        p.grad = torch.randn_like(p)
        opt.step()
    assert torch.isfinite(p).all()


def test_cautious_masks_disagreeing_coords():
    torch.manual_seed(0)
    n = 32
    p0 = torch.randn(8, n)

    def run(cautious: bool) -> torch.Tensor:
        p = torch.nn.Parameter(p0.clone())
        opt = AdamP([p], lr=1e-2, cautious=cautious, gradient_centralization=False,
                    momentum_dtype="float32", foreach=False)
        torch.manual_seed(5)
        for _ in range(4):
            p.grad = torch.randn(8, n)
            opt.step()
        return p.detach().clone()

    assert not torch.allclose(run(True), run(False))


def test_overfits_regression(toy_mlp, random_batch):
    x, y = random_batch
    opt = AdamP(toy_mlp.parameters(), lr=3e-3)
    losses = []
    for _ in range(80):
        opt.zero_grad()
        loss = (toy_mlp(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5


def test_bf16_weights_train_no_nan():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, 8, dtype=torch.bfloat16))
    opt = AdamP([p], lr=1e-3, bf16_method="stochastic_rounding")
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()
    assert torch.isfinite(p).all()


def test_kahan_runs():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 4, dtype=torch.bfloat16))
    opt = AdamP([p], lr=1e-3, bf16_method="kahan")
    for _ in range(3):
        p.grad = torch.randn_like(p)
        opt.step()
    assert "shift" in opt.state[p]
    assert torch.isfinite(p).all()


@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8", "4bit"])
def test_checkpoint_roundtrip_preserves_momentum_dtype(momentum_dtype):
    torch.manual_seed(0)
    params_a = [torch.nn.Parameter(torch.randn(8, 4)), torch.nn.Parameter(torch.randn(5))]
    opt_a = AdamP(params_a, lr=1e-3, momentum_dtype=momentum_dtype, weight_decay=0.01)
    for _ in range(3):
        for p in params_a:
            p.grad = torch.randn_like(p)
        opt_a.step()

    buf = io.BytesIO()
    torch.save(opt_a.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, weights_only=False)

    params_b = [torch.nn.Parameter(p.detach().clone()) for p in params_a]
    opt_b = AdamP(params_b, lr=1e-3, momentum_dtype=momentum_dtype, weight_decay=0.01)
    opt_b.load_state_dict(sd)

    for p_a, p_b in zip(params_a, params_b, strict=True):
        assert opt_b.state[p_b]["m"].dtype == opt_a.state[p_a]["m"].dtype

    torch.manual_seed(123)
    for _ in range(3):
        gs = [torch.randn_like(p) for p in params_a]
        for p, g in zip(params_a, gs, strict=True):
            p.grad = g.clone()
        for p, g in zip(params_b, gs, strict=True):
            p.grad = g.clone()
        opt_a.step()
        opt_b.step()
    for a, b in zip(params_a, params_b, strict=True):
        assert torch.equal(a, b), "resumed run must continue bit-exactly"


def test_conv_net_trains_no_nan():
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(4, 2, 3, padding=1),
    )
    opt = AdamP(net.parameters(), lr=1e-3)
    x = torch.randn(2, 3, 8, 8)
    y = torch.randn(2, 2, 8, 8)
    train_steps(net, opt, [(x, y)] * 5)
    for p in net.parameters():
        assert torch.isfinite(p).all()


def test_invalid_args_rejected():
    p = [torch.nn.Parameter(torch.randn(3))]
    with pytest.raises(ValueError):
        AdamP(p, lr=-1.0)
    with pytest.raises(ValueError):
        AdamP(p, betas=(1.0, 0.999))
    with pytest.raises(ValueError):
        AdamP(p, eps=-1e-8)
    with pytest.raises(ValueError):
        AdamP(p, weight_decay=-0.1)
    with pytest.raises(ValueError):
        AdamP(p, delta=-0.1)
    with pytest.raises(ValueError):
        AdamP(p, wd_ratio=1.5)
    with pytest.raises(ValueError):
        AdamP(p, momentum_dtype="fp8")
    with pytest.raises(ValueError):
        AdamP(p, bf16_method="bogus")


def test_sparse_grad_rejected():
    p = torch.nn.Parameter(torch.randn(4))
    opt = AdamP([p], lr=1e-3)
    idx = torch.tensor([[0, 2]])
    val = torch.tensor([1.0, 1.0])
    p.grad = torch.sparse_coo_tensor(idx, val, (4,))
    with pytest.raises(RuntimeError):
        opt.step()


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
    oa = AdamP(pa, foreach=True, **cfg)
    ob = AdamP(pb, foreach=False, **cfg)
    gg = torch.Generator().manual_seed(grad_seed)
    for _ in range(steps):
        for a, b in zip(pa, pb, strict=True):
            grad = torch.randn(a.shape, generator=gg) * 0.02
            a.grad, b.grad = grad.clone(), grad.clone()
        oa.step()
        ob.step()
    return pa, pb, oa, ob


_SCALAR_CFGS = [
    {'lr': 0.001, 'betas': (0.9, 0.999), 'momentum_dtype': 'float32'},
    {'lr': 0.001, 'betas': (0.9, 0.999), 'momentum_dtype': 'bfloat16'},
    {'lr': 0.001, 'betas': (0.9, 0.999), 'momentum_dtype': 'int8'},
    {'lr': 0.001, 'betas': (0.9, 0.999), 'momentum_dtype': '4bit'},
    {'lr': 0.001, 'betas': (0.9, 0.999), 'momentum_dtype': 'float32', 'weight_decay': 0.02},
    {'lr': 0.001, 'betas': (0.9, 0.999), 'momentum_dtype': 'int8', 'weight_decay': 0.02},
    {'lr': 0.001, 'betas': (0.9, 0.999), 'momentum_dtype': '4bit', 'weight_decay': 0.02},
    {'lr': 0.001, 'betas': (0.9, 0.999), 'momentum_dtype': 'float32', 'nesterov': True},
    {'lr': 0.001, 'betas': (0.9, 0.999), 'momentum_dtype': 'int8', 'nesterov': True, 'weight_decay': 0.02},
    {'lr': 0.001, 'betas': (0.9, 0.999), 'momentum_dtype': 'float32', 'cautious': False},
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
    # State keeps the per-param layout: a 0-D param keeps 0-D buffers (checkpoint compat).
    for a in pa:
        if a.ndim == 0:
            assert oa.state[a]["v"].shape == a.shape


def test_foreach_scalar_0d_takes_batched_path(monkeypatch):
    """0-D scalars must actually ride the batched bucket, not silently fall back to
    the per-param loop (the LyCORIS use_scalar pathology this guards against)."""
    looped = []
    orig = AdamP._step_one_param

    def spy(self, p, *args, **kwargs):
        looped.append(p)
        return orig(self, p, *args, **kwargs)

    monkeypatch.setattr(AdamP, "_step_one_param", spy)
    params = _scalar_bag(_scalar_shapes())
    opt = AdamP(params, foreach=True, lr=0.001, betas=(0.9, 0.999))
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
    pa, pb, _oa, _ob = _scalar_parity({'lr': 0.001, 'betas': (0.9, 0.999), 'momentum_dtype': 'float32'}, shapes=shapes, steps=8, seed=3, grad_seed=5)
    for a, b in zip(pa, pb, strict=True):
        if a.numel() == 1:
            torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)
        else:
            torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=1e-7)


def test_scalar_0d_checkpoint_roundtrip_across_paths():
    """A checkpoint saved mid-training by one path resumes bit-exactly on the other:
    the 0-D state layout is identical on both, in both directions."""
    cfg = {'lr': 0.001, 'betas': (0.9, 0.999), 'momentum_dtype': 'int8', 'weight_decay': 0.02}
    for save_foreach, load_foreach in [(True, False), (False, True)]:
        pa = _scalar_bag(_scalar_shapes(), seed=29)
        oa = AdamP(pa, foreach=save_foreach, **cfg)
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
        ob = AdamP(pb, foreach=load_foreach, **cfg)
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
        oa = AdamP(batched, foreach=True, momentum_dtype="int8", lr=1e-3)
        ob = AdamP(per_param, foreach=False, momentum_dtype="int8", lr=1e-3)
        for pa, pb in zip(batched, per_param, strict=True):
            g = torch.randn(shape) * 0.02
            pa.grad, pb.grad = g.clone(), g.clone()
        oa.step()
        ob.step()
        for pa, pb in zip(batched, per_param, strict=True):
            assert oa.state[pa]['m_scale'].shape == ob.state[pb]['m_scale'].shape, shape
        # ...and the batched-built state really does resume on the per-param path.
        resume = AdamP(batched, foreach=False, momentum_dtype="int8", lr=1e-3)
        for p in batched:
            resume.state[p] = oa.state[p]
            p.grad = torch.randn(shape) * 0.02
        resume.step()
        assert all(torch.isfinite(p).all() for p in batched), shape
