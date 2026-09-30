"""Tests for :class:`kaon.lookahead.Lookahead`.

Covers the defining property — the k-step slow-weight interpolation
``phi += alpha*(theta - phi); theta <- phi`` — plus foreach/per-param parity, the
train()/eval() swap, and equivalence with the plain inner optimizer between syncs.
"""

from __future__ import annotations

import copy

import pytest
import torch

from kaon._wrappers import CodecBuffer
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
            stored = CodecBuffer.read(opt.state[p], "phi", "float32", p)
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
        pa = CodecBuffer.read(o_loop.state[a], "phi", slow_dtype, a)
        pb = CodecBuffer.read(o_fe.state[b], "phi", slow_dtype, b)
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
        phi = CodecBuffer.read(opt.state[p], "phi", "float32", p)
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
        a = CodecBuffer.read(opt.state[p], "phi", "int8", p)
        b = CodecBuffer.read(opt2.state[p2], "phi", "int8", p2)
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_sync_foreach_matches_per_param():
    """foreach sync is element-for-element equal to the per-param path."""
    shape = (8, 8)
    params_loop = _make_params([shape] * 12, seed=4)
    params_fe = _make_params([shape] * 12, seed=4)
    k = 2
    grads = _grad_seq(params_loop, k, seed=5)

    opt_loop = Lookahead(
        params_loop, lr=1e-2, k=k, alpha=0.5, slow_dtype="float32",
        bf16_method="none", foreach=False,
    )
    opt_fe = Lookahead(
        params_fe, lr=1e-2, k=k, alpha=0.5, slow_dtype="float32",
        bf16_method="none", foreach=True,
    )
    for gs in grads:
        for p, g in zip(params_loop, gs, strict=True):
            p.grad = g.clone()
        for p, g in zip(params_fe, gs, strict=True):
            p.grad = g.clone()
        opt_loop.step()
        opt_fe.step()
    for a, b in zip(params_loop, params_fe, strict=True):
        torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0)


def test_sync_foreach_chunks_under_budget(monkeypatch):
    """Forced stack budget must split a large same-shape bucket into several chunks."""
    from kaon import lookahead as la_mod

    chunk_sizes: list[int] = []
    orig_sub = la_mod.subtract_batched_

    def spy_sub(weights, delta, bf16_method, **kw):
        chunk_sizes.append(len(weights))
        return orig_sub(weights, delta, bf16_method, **kw)

    monkeypatch.setattr(la_mod, "subtract_batched_", spy_sub)

    def fake_budget(_stack_budget, _cutoff, _bytes_per, _device):
        return 2 * 8 * 8  # two 8x8 tensors per chunk

    monkeypatch.setattr(la_mod, "foreach_budget", fake_budget)

    shape = (8, 8)
    params = _make_params([shape] * 12, seed=6)
    opt = Lookahead(
        params, lr=1e-2, k=1, alpha=0.5, slow_dtype="float32",
        bf16_method="none", foreach=True, foreach_stack_budget=1,
    )
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()
    assert sum(chunk_sizes) == 12
    assert max(chunk_sizes) == 2
    assert len(chunk_sizes) == 6


# ------------------------------------------------------------------ kahan on bf16 params
def _kahan_run(*, foreach, slow_dtype, bf16_method, steps=6, k=2, dtype=torch.bfloat16):
    """Run Lookahead(bf16 params) for ``steps`` steps with several syncs.

    Returns ``(opt, params, grads_fp32)``; the grads are drawn in the params' dtype and
    handed back upcast, so an fp32 reference run can consume the SAME rounded gradients
    and differ from this run only in the precision of the *weight* writes.
    """
    params = _make_params([(6, 5), (4,)], dtype=dtype, seed=11)
    g = torch.Generator().manual_seed(12)
    grads = [
        [torch.randn(*p.shape, generator=g, dtype=torch.float32).to(dtype) for p in params]
        for _ in range(steps)
    ]
    opt = Lookahead(
        params, lr=1e-2, k=k, alpha=0.5, slow_dtype=slow_dtype,
        bf16_method=bf16_method, foreach=foreach,
    )
    for gs in grads:
        for p, gr in zip(params, gs, strict=True):
            p.grad = gr.clone()
        opt.step()
    return opt, params, [[gr.float() for gr in gs] for gs in grads]


def test_kahan_sync_bf16_params():
    """kahan + bf16 params must survive the sync on every foreach/slow_dtype combination.

    The sync's ``theta <- phi`` write goes through ``subtract_one_``, whose kahan branch
    reads the compensation buffer ``shift``. The wrapper's own per-param state never holds
    one (it allocates ``phi``/``backup`` only), so handing it over raised ``KeyError:
    'shift'`` — on BOTH foreach settings, since ``_sync`` routes kahan to the per-param
    path regardless.
    """
    for foreach in (False, True):
        for slow_dtype in ("float32", "bfloat16"):
            opt, params, _ = _kahan_run(
                foreach=foreach, slow_dtype=slow_dtype, bf16_method="kahan"
            )
            for p in params:
                assert torch.isfinite(p.detach()).all(), (foreach, slow_dtype)
                phi = CodecBuffer.read(opt.state[p], "phi", slow_dtype, p)
                assert torch.isfinite(phi).all(), (foreach, slow_dtype)


def test_kahan_sync_uses_inner_shift_buffer():
    """The sync must accumulate its residue in the INNER's ``shift``, not a wrapper copy.

    Kahan compensation is a property of the WEIGHT, not of whoever writes it: the inner
    step and the sync both write the same ``p``, so a second buffer owned by the wrapper
    would split the residue and carry stale compensation across the ``theta <- phi`` reset.
    """
    opt, params, _ = _kahan_run(foreach=False, slow_dtype="float32", bf16_method="kahan", k=2)
    for p in params:
        assert "shift" not in opt.state[p], "wrapper must not allocate its own shift"
        assert "shift" in opt.inner.state[p]

    # A sync moves the inner's shift: the residue of ``theta <- phi`` lands there.
    p = params[0]
    before = opt.inner.state[p]["shift"].detach().clone()
    for q in params:
        q.grad = torch.randn_like(q)
    opt.step()  # la_step 1 -> no sync yet
    mid = opt.inner.state[p]["shift"].detach().clone()
    for q in params:
        q.grad = torch.randn_like(q)
    opt.step()  # la_step 2 == k -> sync
    after = opt.inner.state[p]["shift"].detach().clone()
    assert not torch.equal(before, mid) or not torch.equal(mid, after)
    assert torch.isfinite(after).all()

def test_kahan_sync_beats_uncompensated():
    """kahan must track an fp32 reference more closely than ``bf16_method="none"``.

    Same initial weights and the same (bf16-rounded) gradients in all three runs, so the
    only difference is how the bf16 weight writes — the inner step's AND the sync's —
    handle the bits that fall off the end. The kept sequence is the slow ``phi``, so that
    is the error that matters; the live ``theta`` is compared in L2 because its max-abs
    error sits on the bf16 grid and ties at short horizons.
    """
    steps, k, lr, shapes, seed = 24, 2, 1e-2, [(6, 5), (4,)], 11
    opt_k, p_k, grads = _kahan_run(
        foreach=False, slow_dtype="float32", bf16_method="kahan", steps=steps, k=k
    )
    opt_n, p_n, _ = _kahan_run(
        foreach=False, slow_dtype="float32", bf16_method="none", steps=steps, k=k
    )

    ref_params = [
        torch.nn.Parameter(p.detach().float())
        for p in _make_params(shapes, dtype=torch.bfloat16, seed=seed)
    ]
    ref = Lookahead(
        ref_params, lr=lr, k=k, alpha=0.5, slow_dtype="float32",
        bf16_method="none", foreach=False,
    )
    for gs in grads:
        for p, gr in zip(ref_params, gs, strict=True):
            p.grad = gr.clone()
        ref.step()

    def theta_err(params):
        return max(
            (p.detach().float() - r.detach()).norm().item()
            for p, r in zip(params, ref_params, strict=True)
        )

    def phi_err(opt, params, norm):
        return max(
            norm(
                CodecBuffer.read(opt.state[p], "phi", "float32", p)
                - CodecBuffer.read(ref.state[r], "phi", "float32", r)
            )
            for p, r in zip(params, ref_params, strict=True)
        )

    amax = lambda t: t.abs().max().item()   # noqa: E731
    l2 = lambda t: t.norm().item()          # noqa: E731

    for label, ek, en in (
        ("phi maxabs", phi_err(opt_k, p_k, amax), phi_err(opt_n, p_n, amax)),
        ("phi l2", phi_err(opt_k, p_k, l2), phi_err(opt_n, p_n, l2)),
        ("theta l2", theta_err(p_k), theta_err(p_n)),
    ):
        assert ek < en, f"{label}: kahan {ek:.3e} did not improve on none {en:.3e}"


def test_kahan_resume_is_bit_exact():
    """The shared ``shift`` travels in the INNER's state_dict, so a kahan resume is exact.

    The wrapper owns no compensation buffer, so there is nothing extra to checkpoint: the
    inner Adakaon's loader restores ``shift`` the way it restores every other state tensor.
    A run split by a save/load must land on the same bits as the uninterrupted one, on
    every ``slow_dtype``.
    """
    shapes, k, lr, steps = [(6, 5), (4,)], 3, 1e-2, 8

    def fresh():
        return _make_params(shapes, dtype=torch.bfloat16, seed=11)

    g = torch.Generator().manual_seed(12)
    grads = [
        [torch.randn(*s, generator=g, dtype=torch.float32).to(torch.bfloat16) for s in shapes]
        for _ in range(steps)
    ]

    def feed(opt, params, chunk):
        for gs in chunk:
            for p, gr in zip(params, gs, strict=True):
                p.grad = gr.clone()
            opt.step()

    def make_build(slow_dtype):
        def build(ps):
            return Lookahead(
                ps, lr=lr, k=k, alpha=0.5, slow_dtype=slow_dtype, bf16_method="kahan",
            )
        return build

    for slow_dtype in ("float32", "bfloat16", "int8", "4bit"):
        build = make_build(slow_dtype)

        cont = fresh()
        feed(build(cont), cont, grads)

        split = fresh()
        opt_a = build(split)
        feed(opt_a, split, grads[: steps // 2])
        sd = copy.deepcopy(opt_a.state_dict())
        resumed = [torch.nn.Parameter(p.detach().clone()) for p in split]
        opt_b = build(resumed)
        opt_b.load_state_dict(sd)
        feed(opt_b, resumed, grads[steps // 2 :])

        for a, b in zip(cont, resumed, strict=True):
            torch.testing.assert_close(a.detach(), b.detach(), rtol=0, atol=0,
                                       msg=f"slow_dtype={slow_dtype}")


# ------------------------------------------- 0.7.18 audit: eval-mode checkpoints & more
def _train(opt, params, grads):
    for gs in grads:
        for p, g in zip(params, gs, strict=True):
            p.grad = g.to(p.dtype)
        opt.step()


def test_state_dict_backup_only_in_eval_mode():
    """The eval-mode checkpoint carries the fast weights ``theta`` (the ``backup``) — the
    ONLY copy of them, since the live weights then hold ``phi`` — and a train-mode one
    does not (the live weights ARE theta)."""
    params = _make_params([(4, 5), (6,)])
    opt = Lookahead(params, lr=1e-2, k=3, foreach=False)
    _train(opt, params, _grad_seq(params, 4))
    assert all("backup" not in st for st in opt.state_dict()["lookahead"].values())
    opt.eval()
    sd = opt.state_dict()
    assert all("backup" in st for st in sd["lookahead"].values())
    opt.train()
    assert all("backup" not in st for st in opt.state.values())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_eval_mode_checkpoint_resumes_exactly(dtype):
    """Save model + optimizer in eval mode (the documented checkpoint point), resume into
    fresh objects, ``train()``: the fast weights come back byte-exact and the continued
    run matches the uninterrupted one bit for bit."""
    shapes = [(8, 6), (6,)]
    grads = _grad_seq(_make_params(shapes), 9, seed=3)
    params = _make_params(shapes, dtype=dtype)
    opt = Lookahead(params, lr=1e-2, k=3, foreach=False)
    _train(opt, params, grads[:4])            # mid-cycle: theta != phi
    opt.eval()
    model_sd = [p.detach().clone() for p in params]   # what model.state_dict() would hold
    opt_sd = copy.deepcopy(opt.state_dict())
    opt.train()
    _train(opt, params, grads[4:])            # uninterrupted reference

    params2 = [torch.nn.Parameter(w.clone()) for w in model_sd]
    opt2 = Lookahead(params2, lr=1e-2, k=3, foreach=False)
    opt2.load_state_dict(opt_sd)
    with pytest.raises(RuntimeError, match="train mode"):
        params2[0].grad = grads[4][0].to(dtype)
        opt2.step()
    opt2.train()
    _train(opt2, params2, grads[4:])
    for a, b in zip(params, params2, strict=True):
        assert torch.equal(a.detach(), b.detach())


def test_groups_without_grads_are_skipped():
    a, b = _make_params([(4, 4), (4, 4)])
    opt = Lookahead([{"params": [a]}, {"params": [b]}], lr=1e-2, k=2)
    b0 = b.detach().clone()
    for _ in range(4):
        a.grad = torch.randn(4, 4)
        opt.step()
    assert torch.equal(b.detach(), b0)
    assert "phi" not in opt.state.get(b, {})


def test_add_param_group_after_construction():
    a, c = _make_params([(4, 4), (5,)])
    opt = Lookahead([a], lr=1e-2, k=2, alpha=0.3)
    opt.add_param_group({"params": [c]})
    g = opt.param_groups[-1]
    assert (g["k"], g["alpha"], g["la_step"], g["train_mode"]) == (2, 0.3, 0, True)
    assert opt.inner.param_groups is opt.param_groups
    c0 = c.detach().clone()
    for _ in range(2):
        for p in (a, c):
            p.grad = torch.randn_like(p)
        opt.step()
    assert "phi" in opt.state[c]
    assert not torch.equal(c.detach(), c0)


def test_torch_lr_scheduler_drives_the_inner_lr():
    torch.manual_seed(0)
    (w,) = _make_params([(8, 8)])
    opt = Lookahead([w], lr=1e-2, k=100, betas=(0.0, 0.999), cautious=False,
                    gradient_centralization=False)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.1)
    g = torch.randn(8, 8)
    deltas = []
    for _ in range(2):
        before = w.detach().clone()
        w.grad = g.clone()
        opt.step()
        sched.step()
        deltas.append(float((w.detach() - before).abs().max()))
    assert opt.inner.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert deltas[1] < 0.5 * deltas[0]
