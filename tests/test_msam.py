"""Tests for MSAM — Momentum-SAM (zero-extra-pass sharpness perturbation).

Covers:
  1. Mechanics — after ``step()`` the live weights sit at (base-step result) +
     ``rho * m/||m||`` with the *global* momentum norm; the next ``step()`` removes the
     stale climb exactly, so the inner state/trajectory matches a twin base optimizer fed
     the same gradients.
  2. ``eval()``/``train()`` — eval shows the true (unperturbed) weights, train restores
     the exact perturbed point (zero drift on fp32).
  3. ``rho=0`` — transparent passthrough (bit-identical to the bare base optimizer).
  4. Codec momenta — the perturbation reads bf16 / int8 / 4bit momentum storage correctly
     (global climb norm == |rho|).
  5. state_dict — eval-mode checkpoint round-trips the inner state dtype-exactly.
  6. Composition — ``MSAM(base_optimizer=Lookahead)`` finds the innermost momentum and
     chains eval/train through the wrapped wrapper.
"""

from __future__ import annotations

import math

import torch

from kaon import Adakaon, Lookahead
from kaon._wrappers import CodecBuffer
from kaon.msam import MSAM


def _params(seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(4, 3, generator=g, dtype=torch.float32).requires_grad_(True)
    b = torch.randn(5, generator=g, dtype=torch.float32).requires_grad_(True)
    return [w, b]


def _attach_grads(params, seed=1):
    g = torch.Generator().manual_seed(seed)
    for p in params:
        p.grad = torch.randn(p.shape, generator=g, dtype=p.dtype)


def _read_momenta(opt):
    """fp32 momentum per param from the innermost optimizer's codec state."""
    owner = opt
    while hasattr(owner, "inner"):
        owner = owner.inner
    out = {}
    for group in opt.param_groups:
        md = group["momentum_dtype"]
        for p in group["params"]:
            st = owner.state.get(p)
            if st and "m" in st:
                out[p] = CodecBuffer.read(st, "m", md, p)
    return out


def _global_norm(momenta):
    return math.sqrt(sum(float((m * m).sum()) for m in momenta.values()))


KW = dict(lr=1e-3, betas=(0.2, 0.999), momentum_dtype="float32", foreach=False)


# --------------------------------------------------------------------------- 1
def test_step_perturbs_by_rho_normalized_momentum():
    """After step(): w_live == w_base_result + rho * m/||m|| (twin-verified)."""
    rho = -0.3
    pa, pb = _params(), _params()
    opt_a = MSAM(pa, rho=rho, **KW)
    opt_b = Adakaon(pb, **KW)

    for seed in (1, 2, 3):  # several steps: each must remove the stale climb first
        _attach_grads(pa, seed)
        _attach_grads(pb, seed)
        opt_a.step()
        opt_b.step()
        mom = _read_momenta(opt_a)
        gn = _global_norm(mom)
        for a, b in zip(pa, pb, strict=True):
            expected = b.data + (rho / (gn + opt_a.eps)) * mom[a]
            assert torch.allclose(a.data, expected, atol=1e-6, rtol=1e-5)
        # the twin's state must match exactly (same grads => same inner trajectory)
        mom_b = _read_momenta(opt_b)
        for a, b in zip(pa, pb, strict=True):
            assert torch.allclose(mom[a], mom_b[b], atol=1e-7, rtol=1e-6)


# --------------------------------------------------------------------------- 2
def test_eval_train_round_trip():
    """eval() shows the true weights (== twin); train() restores the exact climb."""
    rho = -0.3
    pa, pb = _params(), _params()
    opt_a = MSAM(pa, rho=rho, **KW)
    opt_b = Adakaon(pb, **KW)
    _attach_grads(pa, 1)
    _attach_grads(pb, 1)
    opt_a.step()
    opt_b.step()

    live = [p.data.clone() for p in pa]
    opt_a.eval()
    for a, b in zip(pa, pb, strict=True):  # eval == unperturbed == twin
        assert torch.allclose(a.data, b.data, atol=1e-6, rtol=1e-5)
    opt_a.train()
    for a, w in zip(pa, live, strict=True):  # restore is exact (fp32 add round trip)
        assert torch.equal(a.data, w)


def test_step_outside_train_mode_raises():
    pa = _params()
    opt = MSAM(pa, rho=-0.3, **KW)
    _attach_grads(pa)
    opt.step()
    opt.eval()
    _attach_grads(pa)
    try:
        opt.step()
        raise AssertionError("step() in eval mode must raise")
    except RuntimeError:
        pass


# --------------------------------------------------------------------------- 3
def test_rho_zero_is_passthrough():
    pa, pb = _params(), _params()
    opt_a = MSAM(pa, rho=0.0, **KW)
    opt_b = Adakaon(pb, **KW)
    for seed in (1, 2):
        _attach_grads(pa, seed)
        _attach_grads(pb, seed)
        opt_a.step()
        opt_b.step()
    for a, b in zip(pa, pb, strict=True):
        assert torch.equal(a.data, b.data)


# --------------------------------------------------------------------------- 4
def test_codec_momenta_climb_norm():
    """bf16/int8/4bit momentum storage: the global climb has L2 norm == |rho|."""
    for md in ("bfloat16", "int8", "4bit"):
        pa = _params()
        opt = MSAM(pa, rho=-0.25, lr=1e-3, betas=(0.2, 0.999), momentum_dtype=md, foreach=False)
        _attach_grads(pa, 1)
        opt.step()
        live = [p.data.clone() for p in pa]
        opt.eval()
        climb = math.sqrt(sum(float(((w - p.data) ** 2).sum()) for w, p in zip(live, pa, strict=True)))
        assert abs(climb - 0.25) < 1e-3, f"{md}: climb norm {climb} != rho"
        opt.train()


def test_none_norm_is_step_scaled():
    """norm='none': the climb is exactly rho * m (raw momentum, step-unit lookahead),
    capped per element at |rho| * clip_threshold * lr (the stability bound)."""
    rho = -1.5
    pa = _params()
    opt = MSAM(pa, rho=rho, norm="none", **KW)
    _attach_grads(pa, 1)
    opt.step()
    mom = _read_momenta(opt)
    live = [p.data.clone() for p in pa]
    opt.eval()
    bound = abs(rho) * 1.0 * KW["lr"]
    for w, p in zip(live, pa, strict=True):
        expected = (rho * mom[p]).clamp(-bound, bound)
        # atol 1e-6: (w - p) reconstructs e through an fp32 add at |p|~1, whose
        # cancellation error (~ulp(|p|)) sits just above 1e-7. Round-trip exactness is
        # asserted separately with torch.equal below.
        assert torch.allclose(w - p.data, expected, atol=1e-6, rtol=1e-6)
    opt.train()
    for a, w in zip(pa, live, strict=True):
        assert torch.equal(a.data, w)


def test_tensor_norm_per_param_radius():
    """norm='tensor': every param's climb has its own L2 radius == |rho| (and the
    eval/train round trip stays exact)."""
    rho = -0.25
    pa = _params()
    opt = MSAM(pa, rho=rho, norm="tensor", **KW)
    _attach_grads(pa, 1)
    opt.step()
    live = [p.data.clone() for p in pa]
    opt.eval()
    for w, p in zip(live, pa, strict=True):
        climb = math.sqrt(float(((w - p.data) ** 2).sum()))
        assert abs(climb - 0.25) < 1e-4, f"per-tensor climb {climb} != |rho|"
    opt.train()
    for a, w in zip(pa, live, strict=True):
        assert torch.equal(a.data, w)


def test_none_norm_spike_is_clamped():
    """The real-training NaN channel: a momentum spike (e.g. a dead factored channel
    concentrating ~sqrt(n)*lr on one coordinate) must NOT displace any weight beyond
    ``|rho| * clip_threshold * lr`` — and the climb/removal round trip stays exact."""
    rho, lr = -1.5, 1e-3
    pa = _params()
    opt = MSAM(pa, rho=rho, norm="none", lr=lr, betas=(0.2, 0.999), momentum_dtype="float32", foreach=False)
    _attach_grads(pa, 1)
    opt.step()
    # inject a runaway spike into the stored momentum (1000x any sane update)
    st = opt._momentum_owner().state[pa[0]]
    st["m"][0, 0] = 1.0
    _attach_grads(pa, 2)
    opt.step()  # removal of old climb + base step + NEW climb rides the spiked m
    live = [p.data.clone() for p in pa]
    opt.eval()
    bound = abs(rho) * 1.0 * lr  # |rho| * clip_threshold * lr
    worst = max((w - p.data).abs().max().item() for w, p in zip(live, pa, strict=True))
    assert worst <= bound * (1 + 1e-5), f"climb {worst} exceeds the per-element bound {bound}"
    opt.train()
    for a, w in zip(pa, live, strict=True):  # exact restore with the frozen bound
        assert torch.equal(a.data, w)


def test_none_norm_nonfinite_momentum_cannot_poison_weights():
    """A non-finite momentum coordinate (e.g. 0*inf from a blown quant block scale) must
    contribute ZERO climb — NaN passes through clamp(), so the sanitization is load-bearing
    (second real-training NaN, 2026-06-10)."""
    pa = _params()
    opt = MSAM(pa, rho=-1.5, norm="none", **KW)
    _attach_grads(pa, 1)
    opt.step()
    opt.eval()
    st = opt._momentum_owner().state[pa[0]]
    st["m"][0, 0] = float("nan")
    st["m"][0, 1] = float("inf")
    opt.train()  # re-climb through the poisoned momentum
    assert all(torch.isfinite(p).all() for p in pa), "climb wrote non-finite weights"


def test_climb_bound_frozen_across_lr_change():
    """An LR-scheduler change between steps must not corrupt the removal: the bound is
    frozen at climb time, so (climb at lr1) -> (lr change) -> (removal) is exact."""
    pa = _params()
    opt = MSAM(pa, rho=-1.5, norm="none", lr=1e-3, betas=(0.2, 0.999), momentum_dtype="float32", foreach=False)
    _attach_grads(pa, 1)
    opt.step()                                  # climb frozen at lr=1e-3
    for g in opt.param_groups:
        g["lr"] = 1e-4                          # scheduler moves lr between steps
    live = [p.data.clone() for p in pa]
    opt.eval()                                  # removal must subtract the SAME e
    opt.train()
    for a, w in zip(pa, live, strict=True):
        assert torch.equal(a.data, w)


# --------------------------------------------------------------------------- 5
def test_state_dict_eval_mode_round_trip():
    """Checkpoint in eval mode; a fresh MSAM resumes with the inner state dtype-exact."""
    pa = _params()
    opt = MSAM(pa, rho=-0.3, lr=1e-3, betas=(0.2, 0.999), momentum_dtype="bfloat16", foreach=False)
    for seed in (1, 2):
        _attach_grads(pa, seed)
        opt.step()
    opt.eval()
    sd = opt.state_dict()
    mom_before = {i: m.clone() for i, m in enumerate(_read_momenta(opt).values())}

    opt2 = MSAM(pa, rho=-0.3, lr=1e-3, betas=(0.2, 0.999), momentum_dtype="bfloat16", foreach=False)
    opt2.load_state_dict(sd)
    mom_after = list(_read_momenta(opt2).values())
    owner = opt2.inner
    for p in pa:
        assert owner.state[p]["m"].dtype == torch.bfloat16  # dtype preserved (not fp32-inflated)
    for i, m in enumerate(mom_after):
        assert torch.allclose(mom_before[i], m, atol=0, rtol=0)


# --------------------------------------------------------------------------- 6
def test_wraps_lookahead():
    """MSAM(base=Lookahead): perturbs from the innermost (Adakaon) momentum and chains
    eval/train through Lookahead's phi swap."""
    pa = _params()
    opt = MSAM(
        pa, base_optimizer=Lookahead, rho=-0.3,
        lr=1e-3, k=2, alpha=0.5, betas=(0.2, 0.999), momentum_dtype="float32", foreach=False,
    )
    for seed in (1, 2, 3):
        _attach_grads(pa, seed)
        opt.step()
    live = [p.data.clone() for p in pa]
    mom = _read_momenta(opt)
    assert _global_norm(mom) > 0  # found the innermost momentum (not Lookahead's phi)

    opt.eval()  # unperturb THEN swap to phi
    eval_w = [p.data.clone() for p in pa]
    # eval view must differ from live (perturbation + phi swap both applied)
    assert any(not torch.equal(a, b) for a, b in zip(live, eval_w, strict=True))
    opt.train()  # swap back to theta THEN re-perturb — exact restore
    for a, w in zip(pa, live, strict=True):
        assert torch.allclose(a.data, w, atol=1e-7, rtol=1e-6)
