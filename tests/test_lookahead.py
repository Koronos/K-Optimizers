"""Tests for :class:`kaon.lookahead.Lookahead`.

Covers the defining property — the k-step slow-weight interpolation
``phi += alpha*(theta - phi); theta <- phi`` — plus foreach/per-param parity, the
train()/eval() swap, and equivalence with the plain inner optimizer between syncs.
"""

from __future__ import annotations

import copy

import torch

from kaon.adakaon import Adakaon
from kaon.lookahead import Lookahead


def _make_params(shapes, *, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(*s, generator=g, dtype=dtype)) for s in shapes]


def _grad_seq(params, steps, *, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [
        [torch.randn(*p.shape, generator=g, dtype=torch.float32) for p in params]
        for _ in range(steps)
    ]


# ---------------------------------------------------------------- sync correctness
def test_sync_rule_k1():
    """k=1: every step is a sync. phi_new == phi + alpha*(theta_pre - phi); live==phi."""
    alpha = 0.5
    params = _make_params([(4, 5)])
    opt = Lookahead(params, lr=1e-2, k=1, alpha=alpha, slow_dtype="float32", foreach=False)
    grads = _grad_seq(params, 4)

    # Reference: run a plain Adakaon to get theta, apply the sync rule by hand.
    ref_params = [p.detach().clone().requires_grad_(True) for p in params]
    ref = Adakaon(ref_params, lr=1e-2, foreach=False)
    phi = [p.detach().clone() for p in params]  # phi_0 = theta_0

    for gs in grads:
        for p, g in zip(params, gs, strict=True):
            p.grad = g.clone()
        opt.step()
        for rp, g in zip(ref_params, gs, strict=True):
            rp.grad = g.clone()
        ref.step()
        # sync: phi += alpha*(theta - phi); theta <- phi
        for i, rp in enumerate(ref_params):
            phi[i] = phi[i] + alpha * (rp.detach() - phi[i])
            rp.data.copy_(phi[i])
        for p, rp in zip(params, ref_params, strict=True):
            torch.testing.assert_close(p.detach(), rp.detach(), rtol=1e-5, atol=1e-6)
        # stored phi matches and live == phi at a sync
        for p, ph in zip(params, phi, strict=True):
            stored = opt._dequant_phi(opt.state[p], "float32", p)
            torch.testing.assert_close(stored, ph, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(p.detach(), ph, rtol=1e-5, atol=1e-6)


def test_sync_rule_k3():
    """k=3: sync only every 3 steps; verify phi update + live reset at the sync."""
    alpha = 0.5
    k = 3
    params = _make_params([(6, 4)])
    opt = Lookahead(params, lr=1e-2, k=k, alpha=alpha, slow_dtype="float32", foreach=False)

    ref_params = [p.detach().clone().requires_grad_(True) for p in params]
    ref = Adakaon(ref_params, lr=1e-2, foreach=False)
    phi = [p.detach().clone() for p in params]
    grads = _grad_seq(params, 2 * k)

    for t, gs in enumerate(grads, start=1):
        for p, g in zip(params, gs, strict=True):
            p.grad = g.clone()
        opt.step()
        for rp, g in zip(ref_params, gs, strict=True):
            rp.grad = g.clone()
        ref.step()
        if t % k == 0:  # sync happens
            for i, rp in enumerate(ref_params):
                phi[i] = phi[i] + alpha * (rp.detach() - phi[i])
                rp.data.copy_(phi[i])
        for p, rp in zip(params, ref_params, strict=True):
            torch.testing.assert_close(p.detach(), rp.detach(), rtol=1e-5, atol=1e-6)


# --------------------------------------------------------- between-syncs == base opt
def test_between_syncs_equals_base():
    """For the first k-1 steps (no sync), Lookahead == plain Adakaon, exactly."""
    k = 4
    params = _make_params([(5, 7), (3,)])
    opt = Lookahead(params, lr=2e-3, k=k, alpha=0.5, slow_dtype="float32", foreach=False)
    ref_params = [p.detach().clone().requires_grad_(True) for p in params]
    ref = Adakaon(ref_params, lr=2e-3, foreach=False)
    grads = _grad_seq(params, k - 1)
    for gs in grads:
        for p, g in zip(params, gs, strict=True):
            p.grad = g.clone()
        for rp, g in zip(ref_params, gs, strict=True):
            rp.grad = g.clone()
        opt.step()
        ref.step()
        for p, rp in zip(params, ref_params, strict=True):
            torch.testing.assert_close(p.detach(), rp.detach(), rtol=0, atol=0)


# ------------------------------------------------------------- foreach == per-param
def _run(opt_factory, params, grads):
    opt = opt_factory(params)
    for gs in grads:
        for p, g in zip(params, gs, strict=True):
            p.grad = g.clone()
        opt.step()
    return opt


def _parity(momentum_dtype, slow_dtype):
    shapes = [(4, 6), (5, 3), (8,), (7,)]  # 2-D + 1-D, several per bucket
    k = 3
    grads = _grad_seq(_make_params(shapes), 2 * k + 1)  # >= 2 syncs

    p_loop = _make_params(shapes)
    p_fe = _make_params(shapes)

    def fac(foreach):
        def f(ps):
            return Lookahead(
                ps, lr=3e-3, k=k, alpha=0.5, slow_dtype=slow_dtype,
                momentum_dtype=momentum_dtype, bf16_method="none", foreach=foreach,
            )
        return f

    o_loop = _run(fac(False), p_loop, grads)
    o_fe = _run(fac(True), p_fe, grads)
    for a, b in zip(p_loop, p_fe, strict=True):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=1e-5, atol=1e-6)
    # stored phi matches too
    for a, b in zip(p_loop, p_fe, strict=True):
        pa = o_loop._dequant_phi(o_loop.state[a], slow_dtype, a)
        pb = o_fe._dequant_phi(o_fe.state[b], slow_dtype, b)
        torch.testing.assert_close(pa, pb, rtol=1e-5, atol=1e-6)


def test_parity_bf16_momentum():
    _parity("bfloat16", "bfloat16")


def test_parity_int8_momentum():
    _parity("int8", "int8")


def test_parity_float32():
    _parity("float32", "float32")


# ------------------------------------------------------------------- train / eval
def test_train_eval_roundtrip():
    """eval() exposes phi; train() returns the exact pre-eval fast weights."""
    params = _make_params([(4, 5), (6,)])
    opt = Lookahead(params, lr=1e-2, k=2, alpha=0.5, slow_dtype="float32", foreach=False)
    grads = _grad_seq(params, 5)
    for gs in grads:
        for p, g in zip(params, gs, strict=True):
            p.grad = g.clone()
        opt.step()

    theta = [p.detach().clone() for p in params]
    opt.eval()
    # live now == phi (the slow weights), which differ from theta after some steps
    for p in params:
        phi = opt._dequant_phi(opt.state[p], "float32", p)
        torch.testing.assert_close(p.detach(), phi, rtol=1e-5, atol=1e-6)
    opt.train()
    # back to the exact fast weights
    for p, t in zip(params, theta, strict=True):
        torch.testing.assert_close(p.detach(), t, rtol=0, atol=0)


def test_eval_idempotent_and_step_guard():
    params = _make_params([(3, 4)])
    opt = Lookahead(params, lr=1e-2, k=2, foreach=False)
    params[0].grad = torch.randn_like(params[0])
    opt.step()
    opt.eval()
    opt.eval()  # idempotent
    raised = False
    try:
        params[0].grad = torch.randn_like(params[0])
        opt.step()  # stepping in eval mode must error
    except RuntimeError:
        raised = True
    assert raised
    opt.train()


# ---------------------------------------------------------------- state_dict resume
def test_state_dict_roundtrip_int8():
    params = _make_params([(4, 6), (5,)])
    opt = Lookahead(params, lr=2e-3, k=2, slow_dtype="int8", momentum_dtype="int8", foreach=False)
    grads = _grad_seq(params, 5)
    for gs in grads:
        for p, g in zip(params, gs, strict=True):
            p.grad = g.clone()
        opt.step()
    sd = copy.deepcopy(opt.state_dict())

    params2 = _make_params([(4, 6), (5,)], seed=99)
    opt2 = Lookahead(params2, lr=2e-3, k=2, slow_dtype="int8", momentum_dtype="int8", foreach=False)
    opt2.load_state_dict(sd)
    # phi codes preserved as int8 (no fp32 upcast) and equal
    for p, p2 in zip(params, params2, strict=True):
        assert opt2.state[p2]["phi"].dtype == torch.int8
        a = opt._dequant_phi(opt.state[p], "int8", p)
        b = opt2._dequant_phi(opt2.state[p2], "int8", p2)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
