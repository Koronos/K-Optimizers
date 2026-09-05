"""Unit tests for AdaPNM(fused=True) — the Triton-fused positive-negative-momentum step.

The fused path reuses the shared kaon Triton núcleo (gradient_centralize, factored_rc, the int8/4bit
dequant/requant primitives, sr_round) and adds only AdaPNM's two-momentum machinery: the pos/neg
buffers (roles alternate by step parity), the raw-grad EMA on the positive buffer, the pos-neg mix /
noise_norm renorm, decoupled WD applied BEFORE the step, and no RMS-clip.

Correctness criterion: AdaPNM(fused=True) must match AdaPNM(fused=False) — same math, same state.
Skips cleanly when CUDA or Triton is unavailable (the kernels are GPU-only).
"""
from __future__ import annotations

import pytest
import torch

import kaon._fused_triton as ft
from kaon import AdaPNM
from kaon._fused_triton import HAS_TRITON

pytestmark = pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()),
    reason="AdaPNM fused step requires CUDA + Triton",
)

DEV = "cuda"


def _bag(shapes, dtype=torch.float32, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return [torch.randn(*s, generator=g, device=DEV, dtype=dtype).requires_grad_(True) for s in shapes]


def _clone(ps):
    return [p.detach().clone().requires_grad_(True) for p in ps]


def _parts(opt):
    """(one_block, big, one_dim, native) from the cached fused partition (after a step)."""
    ob, big, od, nat = [], [], [], []
    for entry in opt._fused_part.values():
        o, b, d, n = entry[-4:]          # the leading witness field is not of interest here
        ob += o
        big += b
        od += d
        nat += n
    return ob, big, od, nat


def _assert_native_parity(fused_params, native_params, tol=1e-5):
    """Fused weights must track the native path within this file's fp32 parity bound.

    The two paths reduce in different orders (tiled Triton row/col reductions vs torch
    reductions), so they were never bit-identical; ``tol`` is the same relative bound
    the rest of this file's fp32 parity tests use. A wrong bias correction or a stale
    pointer cache moves the weights far above it.
    """
    d = max(
        (a.detach().float() - b.detach().float()).abs().max().item()
        for a, b in zip(fused_params, native_params, strict=True)
    )
    scale = max(b.detach().float().abs().max().item() for b in native_params)
    assert d / scale < tol, f"fused vs native rel={d / scale:.2e}"


def _run_parity(shapes, dtype, mdtype, *, cautious=True, gc=True, wd=0.0, steps=6, seed=1):
    """Step AdaPNM(fused=True) and native AdaPNM on identical params+grads; return max|Δp| and scale."""
    cfg = dict(lr=2e-3, betas=(0.8, 0.999), beta0=0.5, eps=1e-30, weight_decay=wd,
               cautious=cautious, gradient_centralization=gc, momentum_dtype=mdtype)
    pv = _bag(shapes, dtype, seed)
    pn = [p.detach().clone().requires_grad_(True) for p in pv]
    ov, on = AdaPNM(pv, fused=True, **cfg), AdaPNM(pn, **cfg)
    gen = torch.Generator(device=DEV).manual_seed(7)
    for _ in range(steps):
        gs = [torch.randn(*p.shape, generator=gen, device=DEV, dtype=dtype) for p in pv]
        for p, g in zip(pv, gs):
            p.grad = g.clone()
        for p, g in zip(pn, gs):
            p.grad = g.clone()
        ov.step()
        on.step()
    torch.cuda.synchronize()
    d = max((a.detach().float() - b.detach().float()).abs().max().item() for a, b in zip(pv, pn))
    scale = max(b.detach().float().abs().max().item() for b in pn)
    return d, scale, ov


def test_autolr_resets_rebuild_caches_and_preserve_native_parity():
    """AdaPNM resets parity counters together with state and fused pointer caches."""
    pv = _bag([(8, 16), (32,)], torch.float32, seed=41)
    pn = _clone(pv)
    cfg = dict(lr=2e-3, momentum_dtype="float32", cautious=False,
               gradient_centralization=False)
    fused = AdaPNM(pv, fused=True, **cfg)
    native = AdaPNM(pn, **cfg)
    gen = torch.Generator(device=DEV).manual_seed(42)
    retired_caches = []
    retired_state_tensors = []

    for contact in range(3):
        gs = [torch.randn(p.shape, generator=gen, device=DEV) for p in pv]
        for p, g in zip(pv, gs, strict=True):
            p.grad = g.clone()
        for p, g in zip(pn, gs, strict=True):
            p.grad = g.clone()
        fused.step()
        native.step()

        current_caches = tuple(fused._fused_ob_caches.values()) + tuple(fused._fused_od_caches.values())
        assert fused._fused_ob_caches and fused._fused_od_caches
        assert all(new is not old for new in current_caches for old in retired_caches)
        if contact == 2:
            break

        retired_caches.extend(current_caches)
        retired_state_tensors.extend(
            value
            for state in fused.state.values()
            for value in state.values()
            if torch.is_tensor(value)
        )
        fused._autolr_reset_base_state()
        native._autolr_reset_base_state()
        assert all(group["step"] == 0 for group in fused.param_groups)
        assert not fused.state
        assert not fused._fused_part
        assert not fused._fused_ob_caches
        assert not fused._fused_od_caches
    assert all(
        new is not old
        for state in fused.state.values()
        for new in state.values()
        if torch.is_tensor(new)
        for old in retired_state_tensors
    )
    for a, b in zip(pv, pn, strict=True):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_load_state_dict_invalidates_pointer_caches():
    ps = _bag([(8, 16), (32,)], seed=43)
    opt = AdaPNM(ps, fused=True, momentum_dtype="float32")
    for p in ps:
        p.grad = torch.ones_like(p)
    opt.step()
    old_caches = tuple(opt._fused_ob_caches.values()) + tuple(opt._fused_od_caches.values())
    assert old_caches

    opt.load_state_dict(opt.state_dict())
    assert not opt._fused_part
    assert not opt._fused_ob_caches
    assert not opt._fused_od_caches

    for p in ps:
        p.grad = torch.ones_like(p)
    opt.step()
    new_caches = tuple(opt._fused_ob_caches.values()) + tuple(opt._fused_od_caches.values())
    assert new_caches
    assert all(new is not old for new in new_caches for old in old_caches)


def test_late_grads_reuse_fused_caches_by_group_and_lag(monkeypatch):
    shapes = [(8, 16)] * 16 + [(64,)] * 8
    fused_params = _bag(shapes, torch.float32, seed=101)
    native_params = _clone(fused_params)
    cfg = dict(
        lr=2e-3,
        momentum_dtype="float32",
        cautious=False,
        gradient_centralization=False,
        clip_threshold=0.0,
        foreach=False,
    )
    fused = AdaPNM(fused_params, fused=True, **cfg)
    native = AdaPNM(native_params, fused=False, **cfg)
    constructions = {"one_block": {}, "one_dim": {}}

    def count_init(cls, category):
        original = cls.__init__

        def wrapped(cache, plist, state_for):
            group_step = fused.param_groups[0]["step"]
            lag = group_step - fused.state[plist[0]]["step"]
            counts = constructions[category]
            counts[lag] = counts.get(lag, 0) + 1
            original(cache, plist, state_for)

        return wrapped

    monkeypatch.setattr(
        ft.AdaPnmCache,
        "__init__",
        count_init(ft.AdaPnmCache, "one_block"),
    )
    monkeypatch.setattr(
        ft.OneDimPnmCache,
        "__init__",
        count_init(ft.OneDimPnmCache, "one_dim"),
    )

    late = set(range(8, 16)) | set(range(20, 24))
    gen = torch.Generator(device=DEV).manual_seed(103)
    for step in range(20):
        grads = [
            torch.randn(p.shape, generator=gen, device=DEV)
            for p in fused_params
        ]
        for index, (fused_p, native_p, grad) in enumerate(
            zip(fused_params, native_params, grads, strict=True)
        ):
            missing = step == 0 and index in late
            fused_p.grad = None if missing else grad.clone()
            native_p.grad = None if missing else grad.clone()
        fused.step()
        native.step()

    torch.cuda.synchronize()
    _assert_native_parity(fused_params, native_params)
    assert constructions == {
        "one_block": {0: 1, 1: 1},
        "one_dim": {0: 1, 1: 1},
    }
    gid = id(fused.param_groups[0])
    assert set(fused._fused_ob_caches) == {(gid, 0), (gid, 1)}
    assert set(fused._fused_od_caches) == {(gid, 0), (gid, 1)}

    ob_caches = dict(fused._fused_ob_caches)
    od_caches = dict(fused._fused_od_caches)
    for fused_p, native_p in zip(fused_params, native_params, strict=True):
        fused_p.grad = None
        native_p.grad = None
    fused.step()
    native.step()
    assert fused.param_groups[0]["step"] == 20
    assert all(fused._fused_ob_caches[key] is cache for key, cache in ob_caches.items())
    assert all(fused._fused_od_caches[key] is cache for key, cache in od_caches.items())

    for index, (fused_p, native_p) in enumerate(
        zip(fused_params, native_params, strict=True)
    ):
        grad = torch.ones_like(fused_p)
        fused_p.grad = None if index in late else grad
        native_p.grad = None if index in late else grad.clone()
    fused.step()
    native.step()
    assert set(fused._fused_ob_caches) == {(gid, 0)}
    assert set(fused._fused_od_caches) == {(gid, 0)}
    _assert_native_parity(fused_params, native_params)
    assert constructions == {
        "one_block": {0: 1, 1: 1},
        "one_dim": {0: 1, 1: 1},
    }


# ----------------------------------------------------------------- one-block parity
@pytest.mark.parametrize("cautious", [True, False])
@pytest.mark.parametrize("gc", [True, False])
def test_fp32_parity_exact(cautious, gc):
    # fp32 momentum -> exact vs native (the alternation + pos-neg mix reproduce native bit-closely)
    d, _, _ = _run_parity([(8, 16)] * 4, torch.float32, "float32", cautious=cautious, gc=gc)
    assert d < 1e-5, f"max|Δp|={d:.2e}"


def test_fp32_parity_weight_decay():
    d, _, _ = _run_parity([(8, 16), (16, 8)], torch.float32, "float32", wd=0.05)
    assert d < 1e-5, f"max|Δp|={d:.2e}"


def test_mixed_shapes_bucketing():
    d, _, ov = _run_parity([(8, 16), (16, 8), (12, 20), (16, 320)], torch.float32, "float32")
    assert d < 1e-5, f"max|Δp|={d:.2e}"
    assert len(_parts(ov)[0]) == 4 and len(_parts(ov)[1]) == 0


@pytest.mark.parametrize("mdtype", ["bfloat16", "int8", "4bit"])
def test_quant_momentum_parity(mdtype):
    # libdevice.rint requant -> int8/4bit track native closely; bf16 within ULP
    d, scale, _ = _run_parity([(8, 16)] * 4, torch.float32, mdtype)
    assert d / scale < 5e-3, f"{mdtype} rel={d/scale:.2e}"


def test_bf16_params_sr():
    d, scale, _ = _run_parity([(8, 16), (16, 8)], torch.bfloat16, "bfloat16")
    assert d / scale < 5e-2, f"rel={d/scale:.2e}"  # independent SR -> expectation match


# ----------------------------------------------------------------- chunked (big tensor) parity
def test_chunked_fp32_parity():
    d, _, ov = _run_parity([(1024, 512)], torch.float32, "float32")
    assert d < 1e-5, f"max|Δp|={d:.2e}"
    assert len(_parts(ov)[1]) == 1 and len(_parts(ov)[0]) == 0


@pytest.mark.parametrize("mdtype", ["int8", "4bit"])
def test_chunked_quant_parity(mdtype):
    d, scale, ov = _run_parity([(1024, 512)], torch.float32, mdtype)
    assert d / scale < 5e-4, f"{mdtype} rel={d/scale:.2e}"
    assert len(_parts(ov)[1]) == 1


def test_mixed_one_block_chunked_onedim():
    # small (one-block) + big (chunked) + 1-D (fused one-dim), all parity at once
    d, _, ov = _run_parity([(8, 16), (16, 320), (1024, 512), (64,)], torch.float32, "float32")
    assert d < 1e-5, f"max|Δp|={d:.2e}"
    ob, big, od, nat = _parts(ov)
    assert len(ob) == 2 and len(big) == 1 and len(od) == 1 and len(nat) == 0


# ----------------------------------------------------------------- memory + convergence
def test_two_momenta_memory():
    # AdaPNM carries TWO momenta (m_pos + m_neg); int8 -> ~2 B/param of momentum
    ps = _bag([(32, 48)] * 4, torch.float32)
    for p in ps:
        p.grad = torch.randn_like(p)
    opt = AdaPNM(ps, fused=True, momentum_dtype="int8")
    opt.step()
    torch.cuda.synchronize()
    st = opt.state[ps[0]]
    assert st["m_pos"].dtype == torch.int8 and st["m_neg"].dtype == torch.int8
    assert st["m_pos"].numel() == ps[0].numel() and st["m_neg"].numel() == ps[0].numel()


def test_fused_converges():
    torch.manual_seed(0)
    w = torch.randn(16, 24, device=DEV).requires_grad_(True)
    target = torch.randn(16, 24, device=DEV)
    opt = AdaPNM([w], fused=True, lr=5e-2, momentum_dtype="4bit")
    losses = []
    for _ in range(80):
        opt.zero_grad()
        loss = (w - target).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    torch.cuda.synchronize()
    assert torch.isfinite(w).all()
    assert losses[-1] < losses[0] * 0.3, f"{losses[0]:.3f} -> {losses[-1]:.3f}"


def test_alternation_runs_many_steps():
    # the pos/neg roles alternate by step parity -> run enough steps to exercise both phases repeatedly
    d, _, _ = _run_parity([(8, 16)] * 3, torch.float32, "float32", steps=11)
    assert d < 1e-5, f"max|Δp|={d:.2e}"


# ----------------------------------------------------------------- RMS-clip (divergence guard)
def _clip_trigger_step_rms(fused, clip, shape=(4, 4), warm=60):
    """rms(|Δp|) on a trigger step: a hot cell in a column kept cold during warmup. The cold col's
    EMA is tiny so c_factor=rsqrt(col) is huge -> the (v_hat-normalized) update RMS exceeds 1, which
    the clip must cap. Deterministic: fp32 momentum, cautious/GC off."""
    import math

    torch.manual_seed(0)
    rr, cc = shape
    p = (0.01 * torch.randn(rr, cc, device=DEV)).requires_grad_(True)
    opt = AdaPNM([p], fused=fused, clip_threshold=clip, lr=1e-2, betas=(0.8, 0.999), beta0=0.5,
                 eps=1e-8, cautious=False, gradient_centralization=False,
                 momentum_dtype="float32", bf16_method="none")
    g = torch.Generator(device=DEV).manual_seed(0)
    for _ in range(warm):
        gr = 0.05 * torch.randn(rr, cc, generator=g, device=DEV)
        gr[:, 0] = 0.0
        p.grad = gr
        opt.step()
    before = p.detach().clone()
    gr = 0.05 * torch.randn(rr, cc, generator=g, device=DEV)
    gr[:, 0] = 0.0
    gr[0, 0] = 30.0
    p.grad = gr
    opt.step()
    torch.cuda.synchronize()
    dp = p.detach() - before
    return float(dp.norm() / math.sqrt(dp.numel()))


@pytest.mark.parametrize("fused", [False, True])
def test_clip_bounds_factored_update(fused):
    # The guard that stopped the real Cosmos LoKr NaN: clip=1 caps rms(step) ~<= clip*step_size,
    # while clip=0 (unclamped PNM) leaves it several x larger. Same bound on native and fused.
    step_size = 1e-2 / (1.0 - 0.8 ** 61)
    r_unclipped = _clip_trigger_step_rms(fused, 0.0)
    r_clipped = _clip_trigger_step_rms(fused, 1.0)
    assert r_clipped <= 1.3 * step_size, f"clip should bound step rms ~<= step_size, got {r_clipped:.2e}"
    assert r_unclipped > 2.0 * r_clipped, f"clip=0 should be much larger: {r_unclipped:.2e} vs {r_clipped:.2e}"


def test_clip_default_on_and_disable():
    # default clip_threshold is 1.0 (the stability guard, matching Adakaon); 0 disables it
    p = _bag([(8, 8)], torch.float32)[0]
    assert AdaPNM([p]).param_groups[0]["clip_threshold"] == 1.0
    assert AdaPNM([p], clip_threshold=0.0).param_groups[0]["clip_threshold"] == 0.0


def test_clip_chunked_parity_with_native():
    # the big-tensor (chunked) path also clips: fused(clip=1) must still match native(clip=1).
    # A LONE big tensor keeps the per-tensor fused-chunked kernel (no batch to amortize).
    d, _, ov = _run_parity([(1024, 512)], torch.float32, "float32")
    assert d < 1e-5, f"max|Δp|={d:.2e}"
    assert len(_parts(ov)[1]) == 1  # confirms it took the (single-big) chunked path


def test_big_batched_routes_and_parity():
    # >=2 same-shape big (>cap) factors take the batched chunked kernel (~2 launches for the bucket);
    # result must still match a pure-native AdaPNM exactly (fp32).
    d, _, ov = _run_parity([(1024, 512)] * 3, torch.float32, "float32")
    assert d < 1e-5, f"max|Δp|={d:.2e}"
    ob, big, od, nat = _parts(ov)
    assert len(big) == 3 and len(ob) == 0  # all classified big; dispatched batched-chunked


# ----------------------------------------------------------------- one-block 1-D (biases / norms)
def test_one_dim_routes_and_parity_fp32():
    d, _, ov = _run_parity([(1024,)] * 4, torch.float32, "float32")
    ob, big, od, nat = _parts(ov)
    assert len(od) == 4 and len(ob) == 0 and len(big) == 0 and len(nat) == 0
    assert d < 1e-5, f"max|Δp|={d:.2e}"


@pytest.mark.parametrize("cautious", [True, False])
def test_one_dim_features(cautious):
    d, _, _ = _run_parity([(1024,), (512,)], torch.float32, "float32", cautious=cautious, wd=0.05)
    assert d < 1e-5, f"cautious={cautious} max|Δp|={d:.2e}"


def test_one_dim_bf16_momentum():
    d, scale, _ = _run_parity([(1024,)] * 3, torch.float32, "bfloat16")
    assert d / scale < 5e-3, f"rel={d/scale:.2e}"


def test_one_dim_bf16_params_sr():
    d, scale, _ = _run_parity([(1024,)] * 3, torch.bfloat16, "bfloat16")
    assert d / scale < 5e-2, f"rel={d/scale:.2e}"


def test_one_dim_alternation_many_steps():
    # cross the pos/neg parity boundary several times in the 1-D path
    d, _, _ = _run_parity([(1024,)] * 3, torch.float32, "float32", steps=11)
    assert d < 1e-5, f"max|Δp|={d:.2e}"


@pytest.mark.parametrize("mdtype", ["int8", "4bit"])
def test_one_dim_quant_routes_to_native(mdtype):
    d, scale, ov = _run_parity([(1024,)] * 3, torch.float32, mdtype)
    ob, big, od, nat = _parts(ov)
    assert len(od) == 0 and len(nat) == 3
    assert d / scale < 5e-4, f"{mdtype} rel={d/scale:.2e}"


# ----------------------------------------------------------------- conv (ndim>2) matrixized
def test_conv_one_block_parity():
    d, _, ov = _run_parity([(16, 8, 3, 3)] * 4, torch.float32, "float32")   # eff (16,72) -> one-block
    ob, big, od, nat = _parts(ov)
    assert len(ob) == 4 and len(nat) == 0
    assert d < 1e-5, f"max|Δp|={d:.2e}"


def test_conv_big_batched_parity():
    d, _, ov = _run_parity([(256, 128, 3, 3)] * 3, torch.float32, "float32")  # eff (256,1152) -> big
    assert len(_parts(ov)[1]) == 3
    assert d < 1e-5, f"max|Δp|={d:.2e}"


def test_conv_quant_routes_to_native():
    d, scale, ov = _run_parity([(16, 8, 3, 3)] * 4, torch.float32, "int8")
    ob, big, od, nat = _parts(ov)
    assert len(nat) == 4 and len(ob) == 0
    assert d / scale < 5e-4, f"rel={d/scale:.2e}"


# ----------------------------------------------------- candidate #4: fused reductions (no stack)
@pytest.mark.parametrize("gc", [True, False])
@pytest.mark.parametrize("shapes", [[(512, 512)] * 3, [(256, 128, 3, 3)] * 3])
def test_fused_reductions_matches_torch_reductions_fp32(shapes, gc):
    def run(fr):
        ps = _bag(shapes, torch.float32, 3)
        opt = AdaPNM(ps, fused=True, lr=2e-3, betas=(0.8, 0.999), beta0=0.5, eps=1e-30,
                     weight_decay=0.05, cautious=True, gradient_centralization=gc, momentum_dtype="float32")
        opt._fused_reductions = fr
        gen = torch.Generator(device=DEV).manual_seed(11)
        for _ in range(6):
            for p in ps:
                p.grad = torch.randn(*p.shape, generator=gen, device=DEV)
            opt.step()
        torch.cuda.synchronize()
        return ps
    a, b = run(False), run(True)
    d = max((x.detach() - y.detach()).abs().max().item() for x, y in zip(a, b))
    assert d < 1e-4, f"gc={gc} max|Δp|={d:.2e}"


@pytest.mark.parametrize("cautious", [True, False])
@pytest.mark.parametrize("gc", [True, False])
def test_big_batched_features(cautious, gc):
    d, _, _ = _run_parity([(512, 512)] * 3, torch.float32, "float32", cautious=cautious, gc=gc, wd=0.05)
    assert d < 1e-5, f"cautious={cautious} gc={gc} max|Δp|={d:.2e}"


@pytest.mark.parametrize("mdtype", ["bfloat16", "int8", "4bit"])
def test_big_batched_quant_parity(mdtype):
    d, scale, _ = _run_parity([(512, 512)] * 3, torch.float32, mdtype, wd=0.05)
    bound = 5e-3 if mdtype == "bfloat16" else 5e-4
    assert d / scale < bound, f"{mdtype} rel={d/scale:.2e}"


def test_big_batched_bf16_params_sr():
    d, scale, _ = _run_parity([(512, 512)] * 3, torch.bfloat16, "bfloat16")
    assert d / scale < 5e-2, f"rel={d/scale:.2e}"


def test_big_batched_alternation_many_steps():
    # cross the pos/neg parity boundary several times in the batched path (both swap orderings).
    d, _, _ = _run_parity([(512, 512)] * 3, torch.float32, "float32", steps=11)
    assert d < 1e-5, f"max|Δp|={d:.2e}"


def test_big_batched_matches_native_foreach_toggle():
    # batched chunked path (default) must equal the in-fused native-foreach fallback (toggle off).
    # fp32 momentum so the two paths match near-exactly (bf16 momentum rounds differently per path).
    cfg = dict(lr=2e-3, weight_decay=0.05, cautious=True, gradient_centralization=True,
               momentum_dtype="float32")
    pv = _bag([(512, 512)] * 3, torch.float32, seed=2)
    pn = _clone(pv)
    ov = AdaPNM(pv, fused=True, **cfg)
    on = AdaPNM(pn, fused=True, **cfg)
    on._fused_big_batched = False
    gen = torch.Generator(device=DEV).manual_seed(9)
    for _ in range(6):
        gs = [torch.randn(*p.shape, generator=gen, device=DEV) for p in pv]
        for p, g in zip(pv, gs):
            p.grad = g.clone()
        for p, g in zip(pn, gs):
            p.grad = g.clone()
        ov.step()
        on.step()
    torch.cuda.synchronize()
    d = max((a - b).abs().max().item() for a, b in zip(pv, pn))
    assert d < 1e-5, f"batched vs native-foreach max|Δp|={d:.2e}"


def test_fused_empty_step_does_not_advance_global_parity():
    p = _bag([(8, 16)], torch.float32, seed=71)[0]
    opt = AdaPNM(
        [p], fused=True, lr=2e-3, momentum_dtype="float32",
        cautious=False, gradient_centralization=False,
    )
    opt.step()
    assert opt.param_groups[0]["step"] == 0
    p.grad = torch.ones_like(p)
    opt.step()
    assert opt.param_groups[0]["step"] == 1
    assert opt.state[p]["step"] == 1
    assert torch.count_nonzero(opt.state[p]["m_pos"]) > 0
    assert torch.count_nonzero(opt.state[p]["m_neg"]) == 0


def test_fused_mixed_local_steps_use_local_bias_and_global_parity():
    active, late = _bag([(8, 16), (8, 16)], torch.float32, seed=73)
    expected = late.detach().clone().requires_grad_(True)
    cfg = dict(
        lr=2e-3, betas=(0.8, 0.9), beta0=0.0, clip_threshold=0.0,
        momentum_dtype="float32", cautious=False, gradient_centralization=False,
    )
    opt = AdaPNM([active, late], fused=True, **cfg)
    for _ in range(3):
        active.grad = torch.ones_like(active)
        late.grad = None
        opt.step()
    grad = torch.randn_like(late)
    active.grad = torch.ones_like(active)
    late.grad = grad.clone()
    opt.step()

    ref = AdaPNM([expected], fused=True, **cfg)
    expected.grad = grad.clone()
    ref.step()
    torch.cuda.synchronize()
    torch.testing.assert_close(late, expected, rtol=1e-5, atol=1e-6)
    assert opt.state[active]["step"] == 4
    assert opt.state[late]["step"] == 1
    assert torch.count_nonzero(opt.state[late]["m_pos"]) == 0
    assert torch.count_nonzero(opt.state[late]["m_neg"]) > 0


# ================================================================= audit 0.7.12 - AdaPNM fused safety
# Every test below has a confirmed repro against the pre-fix code; none is a performance or a
# numerics-QUALITY assertion - they guard a way AdaPNM's fused step could corrupt memory, refuse
# to compile, or silently disagree with the native step:
#
#   * ``p.data`` REBIND mid-training. ``_fused_partition`` and the AdaPnmCache/OneDimPnmCache
#     callers keyed on ``id(p)`` ONLY, so an external EMA / ``.to()`` / block-swap offloader that
#     rebinds the SAME parameter to fresh storage left the kernels writing the step into the
#     RETIRED buffer (measured: every param scribbled, ~1e-2 divergence, one-block and 1-D).
#   * ``momentum_4bit_block != 128`` on a one-block shape: ``_adapnm_tile_kernel`` hardcodes
#     128-element blocks AND passed ``NS = NB``, so it wrote 32 floats past a 32-entry
#     ``m_pos_scale``/``m_neg_scale`` (measured with a canary) and read the wrong layout back.
#   * Non-contiguous GRADS. ``conv_ok`` only checked the grad for ``ndim > 2``, and it was CACHED
#     in the partition, so a 2-D transposed / 1-D strided grad reached the kernels (which index
#     row-major off ``data_ptr``) - ~1e-2 divergence on all three routes, silently.
#   * NON-FINITE PROPAGATION. Policy is propagate: ``tl.where(keep, delta, 0)`` made the fused
#     cautious mask swallow an inf and FREEZE the tensor while native propagated it.
#   * A param group holding CPU and CUDA weights: the native foreach buckets were keyed on
#     ``(shape, dtype, lag)`` with no DEVICE, so ``torch.stack`` took the whole step down.

_SAFE_CFG = dict(lr=2e-3, betas=(0.9, 0.999), beta0=0.5, eps=1e-30, momentum_dtype="float32",
                 weight_decay=0.02, cautious=True, gradient_centralization=True)

# One representative bag per fused route (the routing is asserted, not assumed). AdaPNM has no
# 0-D route: its 1-D predicate requires ``ndim == 1``, so scalars go native.
_SAFE_ROUTES = {
    "one_block": [(8, 16)] * 4,
    "one_dim": [(32,)] * 3,
    "big": [(512, 512)] * 2,
}


def _plain_grads(gen, plist):
    return [torch.randn(tuple(p.shape), generator=gen, device=DEV, dtype=p.dtype) for p in plist]


def _transposed_grads(gen, plist):
    return [torch.randn((p.shape[1], p.shape[0]), generator=gen, device=DEV, dtype=p.dtype).t()
            for p in plist]


def _strided_grads(gen, plist):
    return [torch.randn((p.shape[0] * 2,), generator=gen, device=DEV, dtype=p.dtype)[::2]
            for p in plist]


def _drive(pairs, steps, gen, grads_for=_plain_grads, mutate=None):
    """Step every (params, optimizer) pair on IDENTICAL gradients for ``steps`` steps.

    The grads are re-DRAWN per optimizer from the same seed rather than cloned, so a fixture that
    returns a non-contiguous grad keeps that layout for every pair (``clone()`` would quietly
    compact it and defeat the strided-grad tests).
    """
    for step in range(steps):
        if mutate is not None:
            mutate(step, pairs)
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=gen, device=DEV).item())
        for plist, opt in pairs:
            draw = torch.Generator(device=DEV).manual_seed(seed)
            for p, g in zip(plist, grads_for(draw, plist), strict=True):
                p.grad = g
            opt.step()
    torch.cuda.synchronize()


def _maxdiff(pa, pb):
    return max((a.detach().float() - b.detach().float()).abs().max().item()
               for a, b in zip(pa, pb, strict=True))


def _safe_pair(shapes, cfg, dtype=torch.float32, seed=3):
    pv = _bag(shapes, dtype, seed)
    pn = _clone(pv)
    return pv, pn, AdaPNM(pv, fused=True, **cfg), AdaPNM(pn, **cfg)


def _assert_safe_route(opt, route):
    ob, big, od, nat = _parts(opt)
    assert not nat, f"{route}: {len(nat)} params fell to the native path"
    if route == "one_block":
        assert ob and not big and not od
    elif route == "big":
        assert big and not ob and not od
    else:
        assert od and not ob and not big


# ----------------------------------------------------------------- 1. p.data rebind
@pytest.mark.parametrize("route", list(_SAFE_ROUTES))
def test_pnm_data_rebind_stops_writing_the_retired_storage(route):
    """``p.data = p.data.clone()`` mid-training must move the kernels to the NEW buffer.

    The retired tensor is kept alive on purpose: on the pre-fix code the cached pointer arrays
    still addressed it, so the step landed in memory the optimizer no longer owns - in real
    training (an EMA, or Rengu-Flow's block-swap offloader) that buffer is freed and reused,
    which is a silent corruption or an illegal memory access.
    """
    pv, pn, ov, on = _safe_pair(_SAFE_ROUTES[route], _SAFE_CFG)
    retired: list[tuple[torch.Tensor, torch.Tensor]] = []

    def mutate(step, _pairs):
        if step != 2:
            return
        for plist in (pv, pn):
            for p in plist:
                old = p.data
                p.data = old.clone()
                if plist is pv:
                    retired.append((old, old.clone()))

    _drive([(pv, ov), (pn, on)], 5, torch.Generator(device=DEV).manual_seed(11), mutate=mutate)
    _assert_safe_route(ov, route)
    assert retired, "the mutation hook never ran"
    for old, snapshot in retired:
        assert torch.equal(old, snapshot), f"{route}: the fused step wrote the RETIRED storage"
    d = _maxdiff(pv, pn)
    assert d < 1e-5, f"{route}: max|dp| vs native = {d:.2e} after a rebind"


@pytest.mark.parametrize("route", list(_SAFE_ROUTES))
def test_pnm_data_rebind_to_bf16_stays_finite(route):
    """A rebind that also changes dtype re-buckets the caches (``lowp``/SR flip with it)."""
    pv = _bag(_SAFE_ROUTES[route], torch.float32, seed=5)
    ov = AdaPNM(pv, fused=True, **_SAFE_CFG)
    retired: list[tuple[torch.Tensor, torch.Tensor]] = []

    def mutate(step, _pairs):
        if step != 2:
            return
        for p in pv:
            old = p.data
            p.data = old.to(torch.bfloat16)
            retired.append((old, old.clone()))

    _drive([(pv, ov)], 5, torch.Generator(device=DEV).manual_seed(13), mutate=mutate)
    assert all(p.dtype == torch.bfloat16 for p in pv)
    for old, snapshot in retired:
        assert torch.equal(old, snapshot), f"{route}: the fused step wrote the RETIRED storage"
    for p in pv:
        assert torch.isfinite(p.detach().float()).all(), f"{route}: non-finite after a rebind"


# ----------------------------------------------------------------- 2. momentum_4bit_block != 128
@pytest.mark.parametrize("block", [0, 64, 256, 512])
def test_pnm_4bit_block_other_than_128_leaves_the_one_block_route(block):
    """``_adapnm_tile_kernel`` hardcodes ``BLK = min(R*C, 128)`` for BOTH momenta's dequant and
    the positive's requant, so any other ``momentum_4bit_block`` reads the wrong scale layout -
    and, when the real block is LARGER than 128, writes past the (shorter) scale buffer.
    """
    cfg = dict(_SAFE_CFG, momentum_dtype="4bit", momentum_4bit_block=block)
    pv, pn, ov, on = _safe_pair([(64, 128)] * 2, cfg, seed=23)   # (64,128) tile == 8192 == TILE_CAP
    _drive([(pv, ov), (pn, on)], 4, torch.Generator(device=DEV).manual_seed(29))
    ob, _big, _od, nat = _parts(ov)
    assert not ob and len(nat) == 2, "a non-128 4-bit block must not take the one-block kernel"
    per = 64 * 128
    expect_block = per if block == 0 else block
    for p in pv:
        st = ov.state[p]
        for pref in ("m_pos", "m_neg"):
            assert st[f"{pref}_block"] == expect_block
            assert st[f"{pref}_scale"].numel() == (per + expect_block - 1) // expect_block
    d = _maxdiff(pv, pn)
    assert d < 1e-5, f"max|dp|={d:.2e}"


def test_pnm_4bit_block_256_does_not_scribble_past_the_scale_buffers():
    """The direct memory-safety canary for the block above.

    ``m_pos_scale`` / ``m_neg_scale`` are re-pointed at the head of a longer sentinel buffer, so
    a requant that writes ``ceil(numel/128)`` scales into a ``ceil(numel/256)``-entry buffer
    lands in the tail. Pre-fix: 32 sentinel floats overwritten in EVERY one of the four buffers
    (both momenta take a turn as "positive" over two steps).
    """
    cfg = dict(_SAFE_CFG, momentum_dtype="4bit", momentum_4bit_block=256,
               cautious=False, gradient_centralization=False)
    pv = _bag([(64, 128)] * 2, torch.float32, seed=3)
    ov = AdaPNM(pv, fused=True, **cfg)
    _drive([(pv, ov)], 1, torch.Generator(device=DEV).manual_seed(4))
    canaries = []
    for p in pv:
        st = ov.state[p]
        for key in ("m_pos_scale", "m_neg_scale"):
            n = st[key].numel()
            buf = torch.full((n + 64,), 1e30, dtype=torch.float32, device=DEV)
            buf[:n].copy_(st[key])
            st[key] = buf[:n]                     # same numel, same values, sentinel tail
            canaries.append((key, buf, n))
    ov._invalidate_fused_caches()                 # the caches hold the OLD scale pointers
    _drive([(pv, ov)], 2, torch.Generator(device=DEV).manual_seed(6))
    scribbled = [(key, int((buf[n:] != 1e30).sum().item())) for key, buf, n in canaries]
    assert all(count == 0 for _key, count in scribbled), f"wrote past m_*_scale: {scribbled}"


def test_pnm_4bit_block_128_still_takes_the_one_block_route():
    """The guard must not cost the DEFAULT 4-bit configuration its fused route."""
    cfg = dict(_SAFE_CFG, momentum_dtype="4bit")
    pv, pn, ov, on = _safe_pair([(64, 128)] * 2, cfg, seed=23)
    _drive([(pv, ov), (pn, on)], 4, torch.Generator(device=DEV).manual_seed(29))
    ob, _big, _od, nat = _parts(ov)
    assert len(ob) == 2 and not nat
    scale = max(p.detach().abs().max().item() for p in pn)
    d = _maxdiff(pv, pn)
    assert d / scale < 8e-4, f"rel={d / scale:.2e}"


# ----------------------------------------------------------------- 3. non-contiguous grads
_NONCONTIG_CASES = [
    ([(8, 16)] * 3, _transposed_grads, "one_block"),
    ([(512, 512)] * 2, _transposed_grads, "big"),
    ([(64,)] * 3, _strided_grads, "one_dim"),
]


@pytest.mark.parametrize("shapes,grads,route", _NONCONTIG_CASES)
def test_pnm_non_contiguous_grads_match_native(shapes, grads, route):
    """The kernels read the grad from ``data_ptr()`` row-major - strides are invisible to them,
    so a non-contiguous grad must not reach them (it steps the transposed numbers instead)."""
    pv, pn, ov, on = _safe_pair(shapes, _SAFE_CFG, seed=31)
    gen = torch.Generator(device=DEV).manual_seed(37)
    assert not any(g.is_contiguous() for g in grads(gen, pv)), "the fixture grads are contiguous"
    _drive([(pv, ov), (pn, on)], 5, gen, grads_for=grads)
    d = _maxdiff(pv, pn)
    assert d < 1e-6, f"{route}: max|dp|={d:.2e} with non-contiguous grads"


@pytest.mark.parametrize("shapes,grads,route", _NONCONTIG_CASES)
def test_pnm_grad_layout_change_is_seen_after_a_contiguous_step(shapes, grads, route):
    """Contiguity belongs to THIS step's gradient, so it cannot be frozen into the cached routing.

    Step 1 runs contiguous (which is what populates the cached partition); step 2 onwards is
    strided. Pre-fix the cached routing kept dispatching those to the kernels.
    """
    pv, pn, ov, on = _safe_pair(shapes, _SAFE_CFG, seed=33)
    state = {"warm": True}

    def grads_for(gen, plist):
        return _plain_grads(gen, plist) if state["warm"] else grads(gen, plist)

    _drive([(pv, ov), (pn, on)], 1, torch.Generator(device=DEV).manual_seed(39),
           grads_for=grads_for)
    _assert_safe_route(ov, route)
    state["warm"] = False
    _drive([(pv, ov), (pn, on)], 4, torch.Generator(device=DEV).manual_seed(40),
           grads_for=grads_for)
    d = _maxdiff(pv, pn)
    assert d < 1e-6, f"{route}: max|dp|={d:.2e} after the grad layout changed"


def test_pnm_non_contiguous_grad_does_not_poison_the_contiguous_neighbours():
    """One strided grad demotes ONLY its own tensor for that step."""
    pv, pn, ov, on = _safe_pair([(8, 16)] * 3, _SAFE_CFG, seed=41)

    def grads(gen, plist):
        gs = _plain_grads(gen, plist)
        gs[1] = torch.randn((16, 8), generator=gen, device=DEV, dtype=torch.float32).t()
        return gs

    _drive([(pv, ov), (pn, on)], 5, torch.Generator(device=DEV).manual_seed(43), grads_for=grads)
    assert len(ov._fused_demoted) == 1, "the demotion memo never fired"
    (_demoted, _parts_in, out), = ov._fused_demoted.values()
    assert len(out[0]) == 2 and len(out[3]) == 1, "only the strided tensor should be demoted"
    d = _maxdiff(pv, pn)
    assert d < 1e-6, f"max|dp|={d:.2e}"


# ----------------------------------------------------------------- 4. non-finite propagation
@pytest.mark.parametrize("route", list(_SAFE_ROUTES))
def test_pnm_non_finite_grad_propagates_like_native(route):
    """Finiteness policy is PROPAGATE, and it has to be the same policy on both paths.

    ``tl.where(keep, delta, 0)`` made the fused cautious mask swallow the inf and FREEZE the
    tensor, while native's ``delta.mul_(mask)`` (0 * inf == NaN) propagates it. A frozen tensor
    is a silently dead weight; a NaN one is a training run that stops and gets fixed.
    """
    pv, pn, ov, on = _safe_pair(_SAFE_ROUTES[route], _SAFE_CFG, seed=47)

    def grads(gen, plist):
        gs = _plain_grads(gen, plist)
        gs[0].reshape(-1)[0] = float("inf")
        return gs

    _drive([(pv, ov), (pn, on)], 2, torch.Generator(device=DEV).manual_seed(53), grads_for=grads)
    fused_finite = torch.isfinite(pv[0].detach().float()).all().item()
    native_finite = torch.isfinite(pn[0].detach().float()).all().item()
    assert native_finite is False, "the native reference is expected to propagate"
    assert fused_finite == native_finite, "fused froze the tensor where native propagated"


def test_pnm_chunked_non_finite_grad_propagates_on_the_lone_big_tensor():
    """The per-tensor chunked pair (a LONE big tensor) has its own cautious site."""
    pv, pn, ov, on = _safe_pair([(512, 512)], _SAFE_CFG, seed=49)

    def grads(gen, plist):
        gs = _plain_grads(gen, plist)
        gs[0].reshape(-1)[0] = float("inf")
        return gs

    _drive([(pv, ov), (pn, on)], 2, torch.Generator(device=DEV).manual_seed(51), grads_for=grads)
    assert not torch.isfinite(pn[0].detach()).all(), "the native reference must propagate"
    assert not torch.isfinite(pv[0].detach()).all(), "fused froze the tensor"


def test_pnm_torch_reduction_batched_non_finite_grad_propagates():
    """The fifth cautious site: ``_adapnm_chunked_apply_batched``.

    The batched big route has two variants and only ``_adapnm_chunked_apply_batched_g`` (the
    default, grad via pointer array) is reached with ``_fused_reductions=True``. The
    ``_g``-less pair — the A/B baseline that stacks the grad in torch — carries its own
    cautious mask and would otherwise go untested.
    """
    pv, pn, ov, on = _safe_pair([(512, 512)] * 2, _SAFE_CFG, seed=57)
    ov._fused_reductions = False

    def grads(gen, plist):
        gs = _plain_grads(gen, plist)
        gs[0].reshape(-1)[0] = float("inf")
        return gs

    _drive([(pv, ov), (pn, on)], 2, torch.Generator(device=DEV).manual_seed(59), grads_for=grads)
    _assert_safe_route(ov, "big")
    assert not torch.isfinite(pn[0].detach()).all(), "the native reference must propagate"
    assert not torch.isfinite(pv[0].detach()).all(), "fused froze the tensor"
    assert torch.isfinite(pv[1].detach()).all(), "the finite neighbour was poisoned"


# ----------------------------------------------------------------- 5. device in the bucket keys
@pytest.mark.parametrize("gc", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_pnm_multi_device_group_step(fused, gc):
    """One param group holding CPU and CUDA weights of the same shape must step both.

    The native foreach path stacks a bucket with ``torch.stack`` and the bucket keys were
    ``(eff, dtype, matrixize, lag)`` / ``(numel, dtype, lag)`` with no device in them, so a CPU
    and a CUDA weight of the same shape landed in one bucket and the stack raised "Expected all
    tensors to be on the same device", taking the whole step down.

    ``gradient_centralization`` is parametrized rather than pinned off: the GC pre-pass
    (``kaon._backend.centralize_grads_``) already keys its stacking bucket on
    ``(shape, device, dtype)``, so a mixed-device group is safe through it too — this pins that
    the two bucketings agree instead of assuming one of them is the weak link.
    """
    params = [
        torch.randn(8, 8).requires_grad_(True),
        torch.randn(8, 8, device=DEV).requires_grad_(True),
        torch.randn(8).requires_grad_(True),
        torch.randn(8, device=DEV).requires_grad_(True),
    ]
    opt = AdaPNM(params, lr=1e-2, fused=fused, gradient_centralization=gc)
    before = [p.detach().clone() for p in params]
    gen = torch.Generator().manual_seed(61)
    for _ in range(2):
        for p in params:
            p.grad = torch.randn(tuple(p.shape), generator=gen).to(p.device)
        opt.step()
    torch.cuda.synchronize()
    for p, b in zip(params, before, strict=True):
        assert p.device == b.device, "a param changed device"
        assert torch.isfinite(p.detach()).all()
        assert not torch.equal(p.detach(), b), f"{tuple(p.shape)} on {p.device} never moved"


def test_pnm_big_shape_buckets_split_by_device():
    """``_fused_big`` groups by shape before handing a list to the big pointer cache, whose index
    arrays live on ``plist[0].device``. Without the device in that key, two GPUs of the same
    shape would be launched against pointer arrays built on the first one.

    Exercised on the grouping helper directly: reaching it end-to-end needs two CUDA devices
    (the big route requires ``p.is_cuda``), and this machine has one.
    """
    from kaon.adakaon import _same_shape_device_buckets

    a = torch.zeros(64, 64, device=DEV)
    b = torch.zeros(64, 64, device=DEV)
    c = torch.zeros(64, 64)                       # same shape + dtype, different device
    d = torch.zeros(64, 64, device=DEV, dtype=torch.bfloat16)
    buckets = _same_shape_device_buckets([a, b, c, d])
    assert len(buckets) == 3, f"expected shape x dtype x device buckets, got {len(buckets)}"
    assert sorted(len(v) for v in buckets.values()) == [1, 1, 2]
    for plist in buckets.values():
        assert len({p.device for p in plist}) == 1


def test_pnm_big_pointer_cache_is_reused_across_steps():
    """The big batched path rebuilt every pointer array (grad, p, both momenta) from a fresh
    ``torch.tensor([...])`` on EVERY step - H2D allocations per bucket per step that the
    one-block and 1-D routes have cached since 0.7.9. Cache it, and revalidate the witness."""
    pv = _bag([(512, 512)] * 3, torch.float32, seed=63)
    ov = AdaPNM(pv, fused=True, **_SAFE_CFG)
    gen = torch.Generator(device=DEV).manual_seed(67)
    _drive([(pv, ov)], 1, gen)
    assert ov._fused_big_caches, "the big route built no cache"
    first = tuple(ov._fused_big_caches.values())
    _drive([(pv, ov)], 3, gen)
    assert tuple(ov._fused_big_caches.values()) == first, "the big cache is rebuilt every step"
    # ... and a rebind still invalidates it.
    for p in pv:
        p.data = p.data.clone()
    _drive([(pv, ov)], 1, gen)
    assert all(new is not old for new in ov._fused_big_caches.values() for old in first), \
        "a p.data rebind must rebuild the big pointer cache"


# ----------------------------------------------------------------- 6. equal_to_1 specialization
@pytest.mark.parametrize("shapes", [[(20000, 1)] * 2, [(1, 20000)] * 2, [(20000, 1)], [(1, 20000)]])
def test_pnm_extreme_aspect_shapes_compile_and_match_native(shapes):
    """``R == 1`` / ``C == 1`` reach a kernel as an argument whose value is 1, which Triton
    SPECIALIZES into a Python int - no ``.to(tl.float32)`` on it. Regression guard for both the
    batched (>=2 same-shape) and the lone-tensor chunked routes."""
    pv, pn, ov, on = _safe_pair(shapes, _SAFE_CFG, seed=71)
    _drive([(pv, ov), (pn, on)], 3, torch.Generator(device=DEV).manual_seed(73))
    _assert_safe_route(ov, "big")
    scale = max(p.detach().abs().max().item() for p in pn)
    d = _maxdiff(pv, pn)
    assert d / scale < 1e-5, f"{shapes[0]} x{len(shapes)}: rel={d / scale:.2e}"
