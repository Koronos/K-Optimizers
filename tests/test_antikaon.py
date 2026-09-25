"""Antikaon — momentum-free Adakaon + seeded Anti-PGD/RWP noise carried in the weights.

Pins the rule of docs/research/antikaon-design.md §1 against an fp32 reference written by
hand, the foreach == per-param bit-parity, the eval/train and checkpoint contracts (§4),
weight decay on the clean iterate, the antithetic pairing (§5), the rank-1 ARWP shaping
(§2.3), the LoRA ``B = 0`` case, and that every ``bf16_method`` of the writer works.
"""

from __future__ import annotations

import copy
import importlib.util
import math
import pathlib
import warnings

import pytest
import torch

from kaon import Adakaon, Antikaon
from kaon.antikaon import noise_seed_for

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------- helpers
def _bag(dtype=torch.float32, seed=0):
    """Two same-shape matrices (a foreach bucket), a conv, two 1-D, a 0-D scalar."""
    g = torch.Generator().manual_seed(seed)
    shapes = [(6, 5), (6, 5), (4, 3, 2, 2), (5,), (5,), ()]
    return [torch.nn.Parameter((torch.randn(s, generator=g) * 0.5).to(dtype)) for s in shapes]


def _grads(params, steps, seed=7):
    g = torch.Generator().manual_seed(seed)
    return [[torch.randn(p.shape, generator=g) * 0.02 for p in params] for _ in range(steps)]


def _run(opt, params, grads):
    for gs in grads:
        for p, gr in zip(params, gs, strict=True):
            p.grad = gr.to(p.dtype).clone()
        opt.step()


def _clone(params):
    return [torch.nn.Parameter(p.detach().clone()) for p in params]


# ----------------------------------------------------------------------------- §1 reference
def _reference(params0, grads, *, lr, k_sigma, wd, seed, clip=1.0, beta2=0.999, eps1=1e-30,
               s_cap=4.0, noise="rademacher", shape="v", antithetic=False):
    """The §1.2 rule, written from the design note (fp32, no kaon machinery but the seed)."""
    ws = [p.detach().clone().float() for p in params0]
    xis = [torch.zeros_like(w) for w in ws]
    stats = [None] * len(ws)
    for t, gs in enumerate(grads):
        for i, (w, g) in enumerate(zip(ws, gs, strict=True)):
            g = g.float()
            if g.ndim >= 2:
                gv = g.reshape(g.shape[0], -1)
                if stats[i] is None:
                    stats[i] = [torch.zeros(gv.shape[0]), torch.zeros(gv.shape[1])]
                row, col = stats[i]
                row.lerp_((gv * gv + eps1).mean(1), 1 - beta2)
                col.lerp_((gv * gv + eps1).mean(0), 1 - beta2)
                vhat = (row / row.mean())[:, None] * col[None, :]
                u = (gv / vhat.sqrt()).reshape(g.shape)
                vhat = vhat.reshape(g.shape)
            else:
                if stats[i] is None:
                    stats[i] = torch.zeros_like(g)
                stats[i].lerp_(g * g + eps1, 1 - beta2)
                vhat = stats[i]
                u = g / vhat.sqrt()
            u = u / max(1.0, float(u.pow(2).mean().sqrt()) / clip)
            delta = u + wd * (w - xis[i])                  # decay on z = w - xi_n
            if shape == "v":
                s = (vhat.mean() / vhat).pow(0.25).clamp(1 / s_cap, s_cap)
            else:
                s = torch.ones_like(w)
            draw, sign = (t // 2, -1.0 if t % 2 else 1.0) if antithetic else (t, 1.0)
            gen = torch.Generator().manual_seed(noise_seed_for(seed, i, draw))
            n = max(w.numel(), 1)
            if noise == "gaussian":
                eps = torch.empty(n).normal_(generator=gen)
            else:
                eps = torch.empty(n).bernoulli_(0.5, generator=gen) * 2 - 1
            xi_new = sign * k_sigma * lr * clip * s * eps.reshape(w.shape)
            ws[i] = w - lr * delta + xi_new - xis[i]
            xis[i] = xi_new
    return ws, xis


@pytest.mark.parametrize("foreach", [False, True])
@pytest.mark.parametrize("cfg", [
    {},
    {"noise": "gaussian"},
    {"shape": "none"},
    {"antithetic": True},
    {"s_cap": 1.1},
])
def test_matches_fp32_reference(foreach, cfg):
    params = _bag()
    p0 = [p.detach().clone() for p in params]
    grads = _grads(params, 7)
    lr, k, wd, seed = 1e-2, 5.0, 0.1, 1234
    opt = Antikaon(params, lr=lr, k_sigma=k, weight_decay=wd, noise_seed=seed, foreach=foreach,
                   gradient_centralization=False, **cfg)
    _run(opt, params, grads)
    ws, xis = _reference(p0, grads, lr=lr, k_sigma=k, wd=wd, seed=seed, **cfg)
    for p, w, xi in zip(params, ws, xis, strict=True):
        torch.testing.assert_close(p.detach(), w, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(opt.live_noise(p), xi, rtol=2e-5, atol=2e-7)


# ----------------------------------------------------------------------------- parity
@pytest.mark.parametrize("cfg", [
    {},
    {"weight_decay": 0.1},
    {"weight_decay": 0.1, "cautious": True, "cautious_wd": "full"},
    {"noise": "gaussian", "antithetic": True},
    {"shape": "none"},
    {"sigma_ref": "weight", "k_weight": 0.02, "weight_decay": 0.05},
    {"betas": (0.9, 0.999), "momentum_dtype": "float32", "weight_decay": 0.1},
])
def test_foreach_matches_per_param_bit_exact(cfg):
    """fp32 params: the two native paths must agree bit for bit, including a parameter
    that joins late (its chunk then mixes noise indices)."""
    pa = _bag()
    pb = _clone(pa)
    oa = Antikaon(pa, lr=1e-2, foreach=True, noise_seed=3, **cfg)
    ob = Antikaon(pb, lr=1e-2, foreach=False, noise_seed=3, **cfg)
    grads = _grads(pa, 8)
    for t, gs in enumerate(grads):
        for i, (a, b, g) in enumerate(zip(pa, pb, gs, strict=True)):
            if i == 1 and t < 3:        # late joiner in the (6, 5) bucket
                a.grad = b.grad = None
                continue
            a.grad, b.grad = g.clone(), g.clone()
        oa.step()
        ob.step()
    assert oa.state[pa[0]]["noise_step"] == 8 and oa.state[pa[1]]["noise_step"] == 5
    for a, b in zip(pa, pb, strict=True):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


@pytest.mark.parametrize("foreach", [False, True])
def test_k_sigma_zero_is_adakaon(foreach):
    pa = _bag()
    pb = _clone(pa)
    oa = Antikaon(pa, lr=1e-2, k_sigma=0.0, weight_decay=0.1, foreach=foreach)
    ob = Adakaon(pb, lr=1e-2, betas=(0.0, 0.999), cautious=False, weight_decay=0.1,
                 foreach=foreach)
    grads = _grads(pa, 5)
    _run(oa, pa, grads)
    _run(ob, pb, grads)
    for a, b in zip(pa, pb, strict=True):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


# ----------------------------------------------------------------------------- eval / train
def test_eval_train_round_trip_fp32():
    params = _bag()
    opt = Antikaon(params, lr=1e-2, noise_seed=5)
    _run(opt, params, _grads(params, 4))
    live = [p.detach().clone() for p in params]
    xis = [opt.live_noise(p) for p in params]
    opt.eval()
    for p, w, xi in zip(params, live, xis, strict=True):
        torch.testing.assert_close(p.detach(), w - xi, rtol=0, atol=0)
    opt.eval()                                    # idempotent
    for p, w, xi in zip(params, live, xis, strict=True):
        torch.testing.assert_close(p.detach(), w - xi, rtol=0, atol=0)
    opt.train()
    for p, w in zip(params, live, strict=True):
        torch.testing.assert_close(p.detach(), w, rtol=1e-6, atol=1e-7)
    opt.train()                                   # idempotent
    for p, w in zip(params, live, strict=True):
        torch.testing.assert_close(p.detach(), w, rtol=1e-6, atol=1e-7)


def test_eval_train_round_trip_bf16_is_rtn():
    torch.manual_seed(0)
    p = torch.nn.Parameter((torch.randn(64, 48) * 0.05).bfloat16())
    q = torch.nn.Parameter(torch.randn(48).bfloat16())
    opt = Antikaon([p, q], lr=1e-3, k_sigma=15.0, noise_seed=9)
    _run(opt, [p, q], _grads([p, q], 3))
    live = p.detach().clone()
    xi = opt.live_noise(p)
    opt.eval()
    assert torch.equal(p.detach(), (live.float() - xi).bfloat16())     # RTN of z
    opt.train()
    back = p.detach()
    changed = back != live
    # Each RTN errs by <= 1/2 ulp of its result, so the return is within ~1 ulp of
    # max(|w|, |z|); in practice only the binade-crossing coordinates move at all.
    assert changed.float().mean() < 0.2
    z = live.float() - xi
    bound = torch.finfo(torch.bfloat16).eps * torch.maximum(live.float().abs(), z.abs())
    assert ((back.float() - live.float()).abs() <= bound + 1e-12).all()


def test_step_in_eval_mode_raises():
    params = _bag()
    opt = Antikaon(params, lr=1e-2)
    _run(opt, params, _grads(params, 1))
    opt.eval()
    for p in params:
        p.grad = torch.zeros_like(p)
    with pytest.raises(RuntimeError, match="outside train mode"):
        opt.step()
    opt.train()
    opt.step()


def test_battery_evald_scores_the_clean_iterate():
    battery = _load("battery_for_antikaon_test", REPO / "benchmarks" / "control" / "battery.py")
    params = _bag()
    opt = Antikaon(params, lr=1e-2)
    _run(opt, params, _grads(params, 2))
    z = [p.detach() - opt.live_noise(p) for p in params]
    seen = battery.evald(opt, lambda: [p.detach().clone() for p in params])
    for a, b in zip(seen, z, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert opt._train_mode                          # evald put it back in train mode


# ----------------------------------------------------------------------------- checkpoints
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("foreach", [False, True])
def test_resume_is_bit_exact(dtype, foreach):
    """Save in eval mode at step n1, reload into FRESH params + optimizer (different
    constructor seed: the checkpoint's seed must win), continue: bit-identical to the run
    that did eval()/train() in-process and carried on."""
    kw = dict(lr=1e-2, k_sigma=5.0, weight_decay=0.1, foreach=foreach, antithetic=True)
    params = _bag(dtype)
    grads = _grads(params, 9)
    opt = Antikaon(params, noise_seed=11, **kw)
    _run(opt, params, grads[:4])
    opt.eval()
    sd = copy.deepcopy(opt.state_dict())
    model_ckpt = [p.detach().clone() for p in params]
    opt.train()
    _run(opt, params, grads[4:])

    params_b = [torch.nn.Parameter(w.clone()) for w in model_ckpt]
    opt_b = Antikaon(params_b, noise_seed=999, **kw)
    opt_b.load_state_dict(sd)                         # ends in train mode, xi re-installed
    assert opt_b.noise_seed == 11 and opt_b._train_mode
    _run(opt_b, params_b, grads[4:])
    for a, b in zip(params, params_b, strict=True):
        assert torch.equal(a.detach(), b.detach())


def test_resume_matches_uninterrupted_fp32():
    kw = dict(lr=1e-2, k_sigma=5.0, weight_decay=0.1, noise_seed=2)
    pa = _bag()
    pb = _clone(pa)
    grads = _grads(pa, 8)
    oa = Antikaon(pa, **kw)
    _run(oa, pa, grads)
    ob = Antikaon(pb, **kw)
    _run(ob, pb, grads[:3])
    ob.eval()
    sd = copy.deepcopy(ob.state_dict())
    pc = [torch.nn.Parameter(p.detach().clone()) for p in pb]
    oc = Antikaon(pc, **kw)
    oc.load_state_dict(sd)
    _run(oc, pc, grads[3:])
    for a, c in zip(pa, pc, strict=True):     # only the eval/train fp32 rounding separates them
        torch.testing.assert_close(a.detach(), c.detach(), rtol=1e-5, atol=1e-6)


def test_train_mode_checkpoint_is_rejected():
    params = _bag()
    opt = Antikaon(params, lr=1e-2)
    _run(opt, params, _grads(params, 2))
    sd = opt.state_dict()
    assert sd["_antikaon_meta"]["train_mode"] is True
    fresh = Antikaon(_clone(params), lr=1e-2)
    with pytest.raises(ValueError, match="saved in train mode"):
        fresh.load_state_dict(sd)


# ----------------------------------------------------------------------------- mechanism
@pytest.mark.parametrize("foreach", [False, True])
def test_weight_decay_acts_on_clean_iterate(foreach):
    """Zero gradients: the update is pure decay. With decay on z, z_n = z_0 (1 - lr wd)^n
    exactly; decay on the live w would add a -lr*wd*xi jitter (~2.5e-2 here)."""
    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(8, 6)), torch.nn.Parameter(torch.randn(8, 6)),
              torch.nn.Parameter(torch.randn(6))]
    z0 = [p.detach().clone() for p in params]
    lr, wd = 0.1, 0.5
    opt = Antikaon(params, lr=lr, k_sigma=5.0, weight_decay=wd, foreach=foreach,
                   gradient_centralization=False)
    steps = 4
    for _ in range(steps):
        for p in params:
            p.grad = torch.zeros_like(p)
        opt.step()
    for p, w0 in zip(params, z0, strict=True):
        z = p.detach() - opt.live_noise(p)
        torch.testing.assert_close(z, w0 * (1 - lr * wd) ** steps, rtol=1e-5, atol=1e-6)


def test_antithetic_pairs_are_opposite():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(10, 7))
    opt = Antikaon([p], lr=1e-2, antithetic=True, shape="none")
    seen = []
    for _ in range(4):
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
        seen.append(opt.live_noise(p))
    # noise index j = 0,1,2,3: (0,1) and (2,3) are antithetic pairs, pairs are independent
    assert torch.equal(seen[1], -seen[0])
    assert torch.equal(seen[3], -seen[2])
    assert (seen[2] != seen[0]).any() and (seen[2] != -seen[0]).any()
    # with the v-shaping the magnitude moves with v, the sign pattern still flips
    q = torch.nn.Parameter(torch.randn(10, 7))
    oq = Antikaon([q], lr=1e-2, antithetic=True)
    signs = []
    for _ in range(2):
        q.grad = torch.randn_like(q) * 0.01
        oq.step()
        signs.append(oq.live_noise(q).sign())
    assert torch.equal(signs[1], -signs[0])


def test_iid_noise_changes_every_step():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(10, 7))
    opt = Antikaon([p], lr=1e-2, shape="none")
    seen = []
    for _ in range(3):
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
        seen.append(opt.live_noise(p))
    assert (seen[0] != seen[1]).any() and (seen[1] != seen[2]).any()
    sigma = 5.0 * 1e-2
    for xi in seen:                               # Rademacher: |xi| == sigma exactly
        torch.testing.assert_close(xi.abs(), torch.full_like(xi, sigma))


def test_shape_factor_is_rank_one_arwp():
    torch.manual_seed(0)
    opt = Antikaon([torch.nn.Parameter(torch.zeros(3))], s_cap=1e6)
    row = torch.rand(1, 9) + 0.05
    col = torch.rand(1, 7) * 3 + 0.01
    s = opt._shape_factor((row, col))[0]
    vhat = (row[0] / row[0].mean())[:, None] * col[0][None, :]
    torch.testing.assert_close(s, (vhat.mean() / vhat).pow(0.25), rtol=1e-5, atol=0)
    sv = torch.linalg.svdvals(s.double())
    assert sv[1] / sv[0] < 1e-5                     # rank 1 up to fp32 rounding
    # the cap clamps per element
    capped = Antikaon([torch.nn.Parameter(torch.zeros(3))], s_cap=1.2)._shape_factor((row, col))
    assert capped.max() <= 1.2 + 1e-6 and capped.min() >= 1 / 1.2 - 1e-6
    # 1-D: (mean(v)/v)^(1/4); a 0-D parameter (length-1 row) gets exactly 1
    v = torch.rand(1, 11) + 0.1
    torch.testing.assert_close(opt._shape_factor((v,))[0], (v.mean() / v[0]).pow(0.25),
                               rtol=1e-5, atol=0)
    assert torch.equal(opt._shape_factor((torch.tensor([[0.37]]),)), torch.ones(1, 1))


@pytest.mark.parametrize("sigma_ref", ["step", "weight"])
@pytest.mark.parametrize("foreach", [False, True])
def test_lora_zero_init_b_receives_noise(sigma_ref, foreach):
    """A zero-init LoRA ``B`` with a zero gradient still gets the full step-unit radius:
    v is flat, so S == 1 and |xi| == k_sigma * lr * clip."""
    b1 = torch.nn.Parameter(torch.zeros(16, 4))
    b2 = torch.nn.Parameter(torch.zeros(16, 4))
    lr, k = 1e-3, 5.0
    opt = Antikaon([b1, b2], lr=lr, k_sigma=k, sigma_ref=sigma_ref, foreach=foreach,
                   gradient_centralization=False)
    for b in (b1, b2):
        b.grad = torch.zeros_like(b)
    opt.step()
    for b in (b1, b2):
        torch.testing.assert_close(b.detach().abs(), torch.full_like(b, k * lr),
                                   rtol=1e-6, atol=0)
        assert (b.detach() > 0).any() and (b.detach() < 0).any()


def test_live_noise_none_before_first_step_and_for_skipped_params():
    a, b = torch.nn.Parameter(torch.randn(4, 3)), torch.nn.Parameter(torch.randn(4, 3))
    opt = Antikaon([a, b], lr=1e-2)
    assert opt.live_noise(a) is None
    a.grad = torch.randn_like(a)
    opt.step()
    assert opt.live_noise(a) is not None and opt.live_noise(b) is None
    b_before = b.detach().clone()
    opt.eval()
    opt.train()
    assert torch.equal(b.detach(), b_before)


# ----------------------------------------------------------------------------- bf16 writers
def test_kahan_combined_write_tracks_fp32_clean_iterate():
    """Sub-ulp regime (lr 1e-5 on O(1) bf16 weights): the Kahan writer keeps the clean
    iterate (p + shift - xi) near the fp32 run; SR on the same stream wanders further."""
    torch.manual_seed(0)
    w0 = torch.randn(64, 32).bfloat16().float()     # representable: same start for every run
    b0 = torch.randn(32).bfloat16().float()
    grads = [[torch.randn(64, 32) * 0.02, torch.randn(32) * 0.02] for _ in range(60)]
    kw = dict(lr=1e-5, k_sigma=5.0, weight_decay=0.1, noise_seed=4)

    def run(dtype, method):
        ps = [torch.nn.Parameter(w0.clone().to(dtype)), torch.nn.Parameter(b0.clone().to(dtype))]
        opt = Antikaon(ps, bf16_method=method, **kw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _run(opt, ps, grads)
        zs = []
        for p in ps:
            w = p.detach().float()
            if method == "kahan" and dtype != torch.float32:
                w = w + opt.state[p]["shift"].float()
            zs.append(w - opt.live_noise(p))
        return opt, ps, zs

    _, _, z32 = run(torch.float32, "stochastic_rounding")
    ok, pk, zk = run(torch.bfloat16, "kahan")
    _, _, zsr = run(torch.bfloat16, "stochastic_rounding")
    assert "shift" in ok.state[pk[0]]
    err_k = sum(float((a - b).pow(2).sum()) for a, b in zip(zk, z32, strict=True)) ** 0.5
    err_sr = sum(float((a - b).pow(2).sum()) for a, b in zip(zsr, z32, strict=True)) ** 0.5
    assert err_k < 0.01 * err_sr, (err_k, err_sr)   # measured 6.3e-5 vs 0.17
    # eval/train work with the Kahan buffer present
    ok.eval()
    ok.train()


@pytest.mark.parametrize("foreach", [False, True])
def test_bf16_stochastic_rounding_runs_and_perturbs(foreach):
    params = _bag(torch.bfloat16)
    opt = Antikaon(params, lr=1e-2, k_sigma=15.0, foreach=foreach)
    _run(opt, params, _grads(params, 3))
    for p in params:
        assert p.dtype == torch.bfloat16 and torch.isfinite(p.float()).all()
        assert opt.live_noise(p).abs().max() > 0


def test_inert_warning_fires_in_sub_ulp_regime():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(32, 16).bfloat16())
    opt = Antikaon([p], lr=1e-6, k_sigma=1.5, inert_check_interval=1)
    with pytest.warns(UserWarning, match="below half"):
        for _ in range(Antikaon._INERT_PATIENCE + 2):
            p.grad = torch.randn(32, 16).bfloat16()
            opt.step()


def test_no_inert_warning_when_noise_is_representable():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(32, 16).bfloat16())
    opt = Antikaon([p], lr=1e-2, k_sigma=5.0, inert_check_interval=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(Antikaon._INERT_PATIENCE + 2):
            p.grad = torch.randn(32, 16).bfloat16()
            opt.step()


def test_fused_falls_back_with_warning():
    params = _bag()
    with pytest.warns(UserWarning, match="no Triton-fused path"):
        opt = Antikaon(params, lr=1e-2, fused=True)
    assert opt._fused is False
    _run(opt, params, _grads(params, 2))


def test_validation():
    p = [torch.nn.Parameter(torch.zeros(3))]
    for bad in ({"k_sigma": -1.0}, {"shape": "x"}, {"noise": "uniform"}, {"sigma_ref": "abs"},
                {"s_cap": 0.5}, {"inert_check_interval": 0}):
        with pytest.raises(ValueError):
            Antikaon(p, **bad)


def test_noise_seed_defaults_to_torch_seed():
    torch.manual_seed(1234)
    a = Antikaon([torch.nn.Parameter(torch.zeros(3))])
    torch.manual_seed(1234)
    b = Antikaon([torch.nn.Parameter(torch.zeros(3))])
    torch.manual_seed(99)
    c = Antikaon([torch.nn.Parameter(torch.zeros(3))])
    assert a.noise_seed == b.noise_seed != c.noise_seed


def test_registry_arms_construct_and_step():
    reg = _load("registry_for_antikaon_test", REPO / "benchmarks" / "control" / "registry.py")
    spec = reg.OPTIMIZERS["Antikaon"]
    makers = [spec["make"], *spec["variants"].values()]
    assert len(spec["variants"]) == 6
    for make in makers:
        params = _bag()
        opt = make(params, spec["lr"])
        assert isinstance(opt, Antikaon)
        _run(opt, params, _grads(params, 2))
        assert all(math.isfinite(float(p.detach().abs().sum())) for p in params)


# ----------------------------------------------------------------------------- review fixes
def _z(opt, p):
    xi = opt.live_noise(p)
    return p.detach().float() if xi is None else p.detach().float() - xi


@pytest.mark.parametrize("wd", [0.0, 0.1])
@pytest.mark.parametrize("foreach", [False, True])
def test_clean_iterate_is_adakaon_on_z_under_lr_changes(foreach, wd):
    """With external gradients, ``z = p - xi`` must follow momentum-free Adakaon applied to z
    exactly (up to fp32 rounding), through an lr that changes EVERY step, intermittent
    ``grad=None``, a param group added mid-run with its own lr, and interleaved eval/train.
    Guards the per-parameter frozen radius: recomputing ``noise_sigma`` from the current lr
    at removal time subtracts a different xi than the one installed and z drifts."""
    torch.manual_seed(0)
    shapes = [(6, 5), (6, 5), (5,), (4, 3, 2, 2)]
    pa = [torch.nn.Parameter(torch.randn(s) * 0.5) for s in shapes]
    pb = _clone(pa)
    kw = dict(weight_decay=wd, foreach=foreach, gradient_centralization=True)
    oa = Antikaon(pa, lr=1e-2, k_sigma=5.0, noise_seed=21, **kw)
    ob = Adakaon(pb, lr=1e-2, betas=(0.0, 0.999), cautious=False, **kw)
    g = torch.Generator().manual_seed(3)
    late_a: list[torch.nn.Parameter] = []
    late_b: list[torch.nn.Parameter] = []
    for t in range(14):
        if t == 5:                                   # (c) new group, different lr
            late_a = [torch.nn.Parameter(torch.randn(8, 4) * 0.3) for _ in range(2)]
            late_b = _clone(late_a)
            oa.add_param_group({"params": late_a, "lr": 3e-2})
            ob.add_param_group({"params": late_b, "lr": 3e-2})
        scale = 1.0 + 0.6 * math.sin(1.7 * t)        # (a) lr changes every step
        for ga, gb in zip(oa.param_groups, ob.param_groups, strict=True):
            base = 3e-2 if ga["params"] is late_a or ga["params"] == late_a else 1e-2
            ga["lr"] = gb["lr"] = base * scale
        for i, (a, b) in enumerate(zip(pa + late_a, pb + late_b, strict=True)):
            if i == 2 and t % 3 == 1:                # (b) intermittent grad=None
                a.grad = b.grad = None
                continue
            gr = torch.randn(a.shape, generator=g) * 0.02
            a.grad, b.grad = gr.clone(), gr.clone()
        oa.step()
        ob.step()
        if t in (4, 9):                              # (d) interleaved eval/train
            oa.eval()
            for a, b in zip(pa + late_a, pb + late_b, strict=True):
                torch.testing.assert_close(a.detach(), b.detach(), rtol=1e-5, atol=5e-6)
            oa.train()
        for a, b in zip(pa + late_a, pb + late_b, strict=True):
            torch.testing.assert_close(_z(oa, a), b.detach(), rtol=1e-5, atol=5e-6)


def test_kahan_eval_train_cycles_keep_clean_value():
    """bf16 + Kahan: eval/train go through the clean value ``p + shift`` and keep the RTN
    residual in ``shift``. Regime: xi (1.5e-3) >> ulp (~2.4e-4) >> step (lr 1e-4). Measured:
    eval-view error 0.8% of a step and 20-cycle drift RMS 0.08% of a step; a plain RTN round
    trip that ignores ``shift`` gives 490% and 11%."""
    torch.manual_seed(0)
    ps = [torch.nn.Parameter((torch.randn(64, 32) * 0.05).bfloat16()),
          torch.nn.Parameter((torch.randn(32) * 0.05).bfloat16())]
    lr = 1e-4
    opt = Antikaon(ps, lr=lr, k_sigma=15.0, bf16_method="kahan", noise_seed=8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _run(opt, ps, [[torch.randn(p.shape) * 0.02 for p in ps] for _ in range(5)])

    def clean(p):
        return p.detach().float() + opt.state[p]["shift"].float()

    before = [clean(p) for p in ps]
    xis = [opt.live_noise(p) for p in ps]
    for _ in range(20):
        opt.eval()
        for p, c, xi in zip(ps, before, xis, strict=True):   # eval view = clean - xi
            torch.testing.assert_close(clean(p), c - xi, rtol=0, atol=5e-2 * lr)
        opt.train()
    for p, c in zip(ps, before, strict=True):
        drift = (clean(p) - c).pow(2).mean().sqrt()
        assert drift < 1e-2 * lr, float(drift)


def test_noise_law_is_read_only():
    opt = Antikaon([torch.nn.Parameter(torch.zeros(3))])
    for name in ("k_sigma", "k_weight", "s_cap", "shape", "noise", "antithetic", "sigma_ref"):
        with pytest.raises(AttributeError):
            setattr(opt, name, getattr(opt, name))


def test_noise_backend_names_the_device_and_warns_on_mismatch():
    params = _bag()
    opt = Antikaon(params, lr=1e-2)
    _run(opt, params, _grads(params, 2))
    opt.eval()
    sd = copy.deepcopy(opt.state_dict())
    assert sd["_antikaon_meta"]["noise_backend"] == "torch-cpu"
    sd["_antikaon_meta"]["noise_backend"] = "torch-cuda"        # a checkpoint from a GPU run
    fresh = Antikaon([torch.nn.Parameter(p.detach().clone()) for p in params], lr=1e-2)
    with pytest.warns(UserWarning, match="noise backend 'torch-cuda'"):
        fresh.load_state_dict(sd)


@pytest.mark.parametrize("foreach", [False, True])
def test_weight_reference_reads_clean_iterate(foreach):
    torch.manual_seed(0)
    ps = [torch.nn.Parameter(torch.randn(6, 5) * 0.5) for _ in range(2)]
    kw_ = 0.3
    opt = Antikaon(ps, lr=1e-2, k_sigma=1.0, sigma_ref="weight", k_weight=kw_, foreach=foreach)
    _run(opt, ps, _grads(ps, 1))
    z = [_z(opt, p) for p in ps]                      # clean iterate entering step 2
    _run(opt, ps, _grads(ps, 1, seed=8))
    for p, zi in zip(ps, z, strict=True):
        want = (zi.square().mean(-1, keepdim=True).sqrt() * kw_).clamp(min=1e-2)
        torch.testing.assert_close(opt.state[p]["noise_sigma_rows"], want, rtol=1e-6, atol=0)


# ----------------------------------------------------------------------------- kahan8
# bf16_method="kahan8" (compact Kahan, docs/research/compact-kahan.md): the bf16 weight plus
# a uint8 residual in units of its own ulp. Antikaon's combined write goes through the same
# writers; the clean value (wd, sigma_ref="weight", eval/train) is decode(p, kahan_lo).
from kaon._compact_kahan import RESIDUAL_KEY  # noqa: E402


def _ck_bag(seed=0):
    """bf16 bag at the ~0.05 scale of a real network (ulp ~2.4e-4), incl. 0-D and conv."""
    g = torch.Generator().manual_seed(seed)
    shapes = [(8, 6), (8, 6), (4, 3, 2, 2), (6,), (6,), ()]
    return [torch.nn.Parameter((torch.randn(s, generator=g) * 0.05).bfloat16()) for s in shapes]


def _ck_unit(z: torch.Tensor) -> torch.Tensor:
    """Per-coordinate kahan8 grid unit ``ulp_bf16(z) / 256``."""
    e = torch.floor(torch.log2(z.abs().clamp_min(2.0 ** -126)))
    return torch.exp2(e - 7 - 8)


def _clean(opt, p):
    """Full-precision live value (decoded under kahan8, ``p + shift`` under kahan)."""
    return opt._read_clean(p, opt.state[p], opt.param_groups[0]["bf16_method"])


def _pin_residual_noise(monkeypatch):
    """Zero every torch.randint draw: the residual's rounding becomes deterministic (the
    Antikaon noise itself uses bernoulli_/normal_ and is untouched)."""
    real = torch.randint
    monkeypatch.setattr(torch, "randint", lambda *a, **k: torch.zeros_like(real(*a, **k)))


@pytest.mark.parametrize("cfg", [
    {},
    {"weight_decay": 0.1},
    {"weight_decay": 0.1, "cautious": True, "cautious_wd": "full"},
    {"sigma_ref": "weight", "k_weight": 0.02, "weight_decay": 0.05},
])
def test_kahan8_foreach_matches_per_param_bit_exact(cfg, monkeypatch):
    """With the residual's rounding noise pinned (the codec is then deterministic), the
    foreach bucket write (comp = the chunk's kahan_lo views) and the per-param write agree
    bit for bit — weights AND residual bytes — incl. a late joiner and eval/train."""
    _pin_residual_noise(monkeypatch)
    pa = _ck_bag()
    pb = _clone(pa)
    kw = dict(lr=1e-4, k_sigma=5.0, bf16_method="kahan8", noise_seed=3, **cfg)
    oa = Antikaon(pa, foreach=True, **kw)
    ob = Antikaon(pb, foreach=False, **kw)
    grads = _grads(pa, 8)
    for t, gs in enumerate(grads):
        for i, (a, b, g) in enumerate(zip(pa, pb, gs, strict=True)):
            if i == 1 and t < 3:
                a.grad = b.grad = None
                continue
            a.grad, b.grad = g.bfloat16(), g.bfloat16()
        oa.step()
        ob.step()
        if t == 4:
            oa.eval()
            ob.eval()
            oa.train()
            ob.train()
    for a, b in zip(pa, pb, strict=True):
        assert torch.equal(a.detach(), b.detach())
        assert oa.state[a][RESIDUAL_KEY].dtype == torch.uint8
        assert torch.equal(oa.state[a][RESIDUAL_KEY], ob.state[b][RESIDUAL_KEY])
    assert any(bool(oa.state[a][RESIDUAL_KEY].any()) for a in pa)   # the residual is in use


@pytest.mark.parametrize("foreach", [False, True])
def test_kahan8_resume_is_bit_exact(foreach):
    """Eval-mode checkpoint, fresh params + optimizer: bit-identical continuation, residual
    included (it is ordinary state; its noise comes from the checkpointed SR stream, and the
    eval/train re-encodes draw from that stream too)."""
    kw = dict(lr=1e-4, k_sigma=5.0, weight_decay=0.1, foreach=foreach, bf16_method="kahan8")
    params = _ck_bag()
    grads = _grads(params, 9)
    opt = Antikaon(params, noise_seed=11, **kw)
    _run(opt, params, grads[:4])
    opt.eval()
    sd = copy.deepcopy(opt.state_dict())
    model_ckpt = [p.detach().clone() for p in params]
    opt.train()
    _run(opt, params, grads[4:])

    params_b = [torch.nn.Parameter(w.clone()) for w in model_ckpt]
    opt_b = Antikaon(params_b, noise_seed=999, **kw)
    opt_b.load_state_dict(sd)
    for p in params_b:
        assert opt_b.state[p][RESIDUAL_KEY].dtype == torch.uint8
    _run(opt_b, params_b, grads[4:])
    for a, b in zip(params, params_b, strict=True):
        assert torch.equal(a.detach(), b.detach())
        assert torch.equal(opt.state[a][RESIDUAL_KEY], opt_b.state[b][RESIDUAL_KEY])


def test_kahan8_eval_train_cycles_keep_clean_value():
    """kahan8: eval/train go through decode(p, lo) and re-encode with the residual's SR.
    Regime: xi (1.5e-3) >> ulp (~2.4e-4) >> step (lr 1e-4). The eval view is the clean
    value minus xi to the residual grid, and 20 cycles (40 re-encodes) only add the
    unbiased grid walk (<= unit/2 * sqrt(40) RMS), far below a step."""
    torch.manual_seed(0)
    ps = [torch.nn.Parameter((torch.randn(64, 32) * 0.05).bfloat16()),
          torch.nn.Parameter((torch.randn(32) * 0.05).bfloat16())]
    lr = 1e-4
    opt = Antikaon(ps, lr=lr, k_sigma=15.0, bf16_method="kahan8", noise_seed=8)
    _run(opt, ps, [[torch.randn(p.shape) * 0.02 for p in ps] for _ in range(5)])
    before = [_clean(opt, p) for p in ps]
    xis = [opt.live_noise(p) for p in ps]
    cycles = 20
    for _ in range(cycles):
        opt.eval()
        for p, c, xi in zip(ps, before, xis, strict=True):
            want = c - xi
            # train writes land on the grid of |c|, eval writes on the grid of |z|: measure
            # in the coarser one
            err = (_clean(opt, p) - want).abs() / _ck_unit(torch.maximum(c.abs(), want.abs()))
            assert float(err.max()) <= 2.0 * math.sqrt(2 * cycles), float(err.max())
            # the forward pass in eval sees the nearest bf16 to the clean value (half-away
            # ties: within half an ulp, not necessarily the round-to-even cast)
            c_now = _clean(opt, p)
            half_ulp = 0.5 * torch.finfo(torch.bfloat16).eps * torch.exp2(
                torch.floor(torch.log2(c_now.abs().clamp_min(2.0 ** -126))))
            assert ((p.detach().float() - c_now).abs() <= half_ulp).all()
        opt.train()
    for p, c in zip(ps, before, strict=True):
        drift_units = ((_clean(opt, p) - c) / _ck_unit(c)).pow(2).mean().sqrt()
        assert float(drift_units) < 0.75 * math.sqrt(2 * cycles), float(drift_units)
        drift = (_clean(opt, p) - c).pow(2).mean().sqrt()
        assert drift < 5e-2 * lr, float(drift)


@pytest.mark.parametrize("method,foreach", [("kahan", False), ("kahan8", False),
                                            ("kahan8", True)])
def test_compensated_weight_decay_reads_the_clean_value(method, foreach, monkeypatch):
    """The decay term is ``wd * (w - xi_n)`` with ``w`` the COMPENSATED value
    (``p + shift`` / ``decode(p, lo)``), not the bare bf16 weight. Checked at the source:
    the ``w`` the decay receives, against a residual made non-zero by the previous steps."""
    seen: list[torch.Tensor] = []
    real = Antikaon._decay_term

    def spy(w, xi_old, wd):
        seen.append(w.detach().clone())
        return real(w, xi_old, wd)

    ps = _ck_bag()[:3] if method == "kahan8" else _ck_bag()[:1]
    opt = Antikaon(ps, lr=1e-4, k_sigma=5.0, weight_decay=0.1, foreach=foreach,
                   bf16_method=method)
    _run(opt, ps, _grads(ps, 3))
    want = [_clean(opt, p) for p in ps]
    assert any(bool((w != p.detach().float()).any()) for w, p in zip(want, ps, strict=True))
    monkeypatch.setattr(Antikaon, "_decay_term", staticmethod(spy))
    _run(opt, ps, _grads(ps, 1, seed=9))
    got = [w.reshape(-1) for w in seen]
    if foreach:                                  # one stacked call per bucket
        got = [row for w in seen for row in w.reshape(w.shape[0], -1)]
    assert len(got) == len(ps)
    for w in want:
        assert any(g.numel() == w.numel() and torch.equal(g, w.reshape(-1)) for g in got)


def test_kahan8_weight_decay_tracks_the_exact_decay():
    """Zero gradients: pure decay, z_n = z_0 (1 - lr wd)^n, with a big lr*wd (0.05/step) so a
    decay that ignores the residual misses ``lr*wd*residual`` visibly. RMS error of the
    clean iterate in residual grid units of the LIVE value (xi is 10x the weights here, so
    the live value sets the precision). Measured: 1.2-2.6 units with the clean read,
    12-26 units reading the bare bf16 weight (per-param and foreach)."""
    for foreach in (False, True):
        torch.manual_seed(0)
        z0 = [(torch.randn(8, 6) * 0.05).bfloat16(), (torch.randn(8, 6) * 0.05).bfloat16(),
              (torch.randn(6) * 0.05).bfloat16()]
        params = [torch.nn.Parameter(z.clone()) for z in z0]
        lr, wd, steps = 0.1, 0.5, 20
        opt = Antikaon(params, lr=lr, k_sigma=5.0, weight_decay=wd, foreach=foreach,
                       bf16_method="kahan8", gradient_centralization=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(steps):
                for p in params:
                    p.grad = torch.zeros_like(p)
                opt.step()
        for p, w0 in zip(params, z0, strict=True):
            live = _clean(opt, p)
            z = live - opt.live_noise(p)
            want = w0.float() * (1 - lr * wd) ** steps
            err = ((z - want) / _ck_unit(live)).pow(2).mean().sqrt()
            assert float(err) < 6.0, (foreach, float(err))


@pytest.mark.parametrize("foreach", [False, True])
def test_kahan8_weight_reference_reads_clean_value(foreach, monkeypatch):
    """sigma_ref="weight" reads RMS_row(decode(p, lo) - xi)."""
    _pin_residual_noise(monkeypatch)
    torch.manual_seed(0)
    ps = [torch.nn.Parameter((torch.randn(6, 5) * 0.05).bfloat16()) for _ in range(2)]
    kw_ = 0.3
    opt = Antikaon(ps, lr=1e-4, k_sigma=1.0, sigma_ref="weight", k_weight=kw_,
                   foreach=foreach, bf16_method="kahan8")
    _run(opt, ps, _grads(ps, 1))
    z = [_clean(opt, p) - opt.live_noise(p) for p in ps]
    _run(opt, ps, _grads(ps, 1, seed=8))
    for p, zi in zip(ps, z, strict=True):
        want = (zi.square().mean(-1, keepdim=True).sqrt() * kw_).clamp(min=1e-4)
        torch.testing.assert_close(opt.state[p]["noise_sigma_rows"], want, rtol=1e-6, atol=0)


def _lowlr_runs(foreach, steps=300, lr=1e-5, k_sigma=5.0):
    """bf16 at lr 1e-5 on ~0.05 weights (ulp ~2.4e-4): step ~0.04 ulp, xi ~0.2 ulp. The
    same bf16-representable start and the same (bf16-valued) gradients in every run, so the
    fp32 run is the same rule with the SAME xi (the shaping reads identical v).

    Returns the RMS distance of each bf16 run's clean iterate to the fp32 one, in bf16 ulps
    at RMS(w), and how far the fp32 run moved (max, same units)."""
    torch.manual_seed(0)
    w0 = [(torch.randn(64, 32) * 0.05).bfloat16(), (torch.randn(32) * 0.05).bfloat16()]
    gg = torch.Generator().manual_seed(3)
    g0 = [torch.randn(w.shape, generator=gg) for w in w0]
    grads = [[(g + 0.3 * torch.randn(g.shape, generator=gg)).bfloat16() for g in g0]
             for _ in range(steps)]
    kw = dict(lr=lr, k_sigma=k_sigma, weight_decay=0.1, noise_seed=4, foreach=foreach,
              gradient_centralization=False)      # GC on a bf16 grad rounds differently
    out = {}
    for name, dtype, method in (("fp32", torch.float32, "stochastic_rounding"),
                                ("kahan8", torch.bfloat16, "kahan8"),
                                ("sr", torch.bfloat16, "stochastic_rounding")):
        ps = [torch.nn.Parameter(w.clone().to(dtype)) for w in w0]
        opt = Antikaon(ps, bf16_method=method, **kw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _run(opt, ps, grads)
        out[name] = (opt, ps, [_clean(opt, p) - opt.live_noise(p) for p in ps])
    ref_xi = [out["fp32"][0].live_noise(p) for p in out["fp32"][1]]
    for name in ("kahan8", "sr"):                   # same xi in every run
        for p, xi in zip(out[name][1], ref_xi, strict=True):
            assert torch.equal(out[name][0].live_noise(p), xi)
    zf = torch.cat([z.flatten() for z in out["fp32"][2]])
    u = float(torch.finfo(torch.bfloat16).eps * zf.pow(2).mean().sqrt())   # ulp at RMS(w)
    w0f = torch.cat([w.float().flatten() for w in w0])
    err = {n: float((torch.cat([z.flatten() for z in out[n][2]]) - zf).pow(2).mean().sqrt()) / u
           for n in ("kahan8", "sr")}
    moved = float((zf - w0f).abs().max()) / u
    return err, moved


@pytest.mark.parametrize("foreach", [False, True])
def test_kahan8_low_lr_clean_iterate_tracks_fp32_far_better_than_sr(foreach):
    """THE regime kahan8 exists for: the clean iterate z = decode(p, lo) - xi of bf16
    Antikaon+kahan8 follows the fp32 run of the same rule (same xi, same grads) to the
    residual grid, while SR walks by whole ulps. Measured (300 steps, weight decay 0.1, RMS
    distance to the fp32 clean iterate in bf16 ulps at RMS(w); the fp32 run moved up to
    21 ulp): k_sigma 5 -> kahan8 0.020, SR 3.8-3.9 (~200x); k_sigma 1.5 -> kahan8 0.021,
    SR 2.6-2.8 (~130x); per-param and foreach alike."""
    err, moved = _lowlr_runs(foreach)
    assert moved > 5.0, moved
    assert err["kahan8"] < 0.05, err
    assert err["sr"] > 50 * err["kahan8"], err


def test_inert_warning_uses_the_compensated_resolution():
    """lr 1e-5, k_sigma 5 on ~0.05 bf16 weights (mean |w| 0.04): xi ~5e-5 is below half a
    bf16 ulp (~1.6e-4), so SR warns — but 80x above half kahan8's ulp/256 grid (~6e-7),
    where it is kept, so kahan8 must not warn. At lr 1e-7 (xi 5e-7) even the residual grid
    cannot hold it: warn."""
    def drive(method, lr):
        torch.manual_seed(0)
        p = torch.nn.Parameter((torch.randn(32, 16) * 0.05).bfloat16())
        opt = Antikaon([p], lr=lr, k_sigma=5.0, bf16_method=method, inert_check_interval=1)
        for _ in range(Antikaon._INERT_PATIENCE + 2):
            p.grad = torch.randn(32, 16).bfloat16()
            opt.step()

    with pytest.warns(UserWarning, match="below half"):
        drive("stochastic_rounding", 1e-5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        drive("kahan8", 1e-5)
    with pytest.warns(UserWarning, match="kahan8"):
        drive("kahan8", 1e-7)


def test_kahan8_fused_request_falls_back_to_foreach():
    params = _ck_bag()
    with pytest.warns(UserWarning, match="no Triton-fused path"):
        opt = Antikaon(params, lr=1e-4, fused=True, bf16_method="kahan8")
    assert opt._fused is False
    _run(opt, params, _grads(params, 3))
    for p in params:
        assert opt.state[p][RESIDUAL_KEY].dtype == torch.uint8
        assert torch.isfinite(_clean(opt, p)).all()
