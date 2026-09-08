"""Cross-process resume of the bf16 stochastic-rounding noise stream.

The bf16 SR weight write draws its noise from a counter, not from a persistent RNG
object: the Triton kernel (``kaon._fused_triton.sr_add_``, the default on CUDA) is seeded
per launch, and the torch reference path (``kaon.add_stochastic_``) seeds a private
generator per call. Either way the counter is optimizer state — a resume that restarts it
at 0 applies a *different* noise sequence than the run it continues, so the weights of a
resumed run diverge from the continuous run's even though every state tensor was restored
bit-exactly (measured ~4.7e-2 max abs on bf16 params after 4 steps).

Until 0.7.13 that counter was a **process-global, per-device** one that no checkpoint
saved, so:

* no optimizer with bf16 params on the native (non-Adakaon-fused) path resumed
  bit-identically in a fresh process, and
* Lookahead's ``phi`` sync — which goes through the same shared kernel — did not either,
  even over an Adakaon whose own fused counter (``_t``) was already checkpointed.

These tests pin the fix: every optimizer owns its own :class:`kaon._stochastic_rounding.
SRStream`, saves it in its ``state_dict``, and restores it on load. "A fresh process" is
simulated the way the public contract prescribes — ``torch.manual_seed`` plus
``kaon.reseed_stochastic_rounding()``, which is exactly what resets every SR stream a
process holds — and the resumed run must then land bit-identically on the continuous run's
weights.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
import torch

import kaon
import kaon._stochastic_rounding as srm
from kaon import (
    ADOPT,
    MSAM,
    SAM,
    AdaBelief,
    Adakaon,
    AdamP,
    AdaMuon,
    AdaPNM,
    KProdigy,
    Lion,
    Lookahead,
    Nekaon,
    ScheduleFree,
)

from .conftest import skip_if_no_cuda

SHAPES = [(64, 32), (48, 16), (24,), ()]
STEPS = 8

# Every optimizer that owns a bf16 SR weight write, plus the wrappers whose own writes
# (Lookahead's phi sync, SAM's climb) go through the same shared noise stream.
PLAIN = {
    "Adakaon": Adakaon,
    "AdaBelief": AdaBelief,
    "AdamP": AdamP,
    "AdaMuon": AdaMuon,
    "AdaPNM": AdaPNM,
    "ADOPT": ADOPT,
    "KProdigy": KProdigy,
    "Lion": Lion,
    "ScheduleFree": ScheduleFree,
}
WRAPPED = ("Lookahead", "SAM", "MSAM", "Nekaon")
# Optimizers with a train()/eval() weight view: default them to train mode after a build.
TRAIN_EVAL = ("ScheduleFree", "Lookahead", "MSAM", "Nekaon")
# MSAM refuses a train-mode save (the live weights carry the climb), so its checkpoint has
# to be taken in eval mode. The others are checkpointable in train mode, and are left there:
# a bf16 eval/train round trip is a lossy view swap for ScheduleFree (a closed-form lerp),
# which would mask what this file measures.
EVAL_TO_SAVE = ("MSAM", "Nekaon")


def _build(kind: str, params: list[torch.nn.Parameter], **kw: Any) -> Any:
    if kind == "Lookahead":
        opt: Any = Lookahead(params, k=2, alpha=0.5, **kw)
    elif kind == "SAM":
        opt = SAM(params, rho=0.05, **kw)
    elif kind == "MSAM":
        opt = MSAM(params, rho=0.3, **kw)
    elif kind == "Nekaon":
        opt = Nekaon(params, k=2, **kw)
    else:
        opt = PLAIN[kind](params, **kw)
    if kind in TRAIN_EVAL:
        opt.train()
    return opt


def _step(kind: str, opt: Any, params: list[torch.nn.Parameter], grads: list[torch.Tensor]) -> None:
    _apply_grads(params, grads)
    if kind == "SAM":
        opt.first_step()
        _apply_grads(params, grads)
        opt.second_step()
    else:
        opt.step()


def _apply_grads(params: list[torch.nn.Parameter], grads: list[torch.Tensor]) -> None:
    for p, g in zip(params, grads, strict=True):
        p.grad = g.detach().clone().to(device=p.device, dtype=p.dtype)


def _params(dtype: torch.dtype, device: str) -> list[torch.nn.Parameter]:
    gen = torch.Generator().manual_seed(11)
    return [
        torch.nn.Parameter(torch.randn(s, generator=gen).to(device=device, dtype=dtype))
        for s in SHAPES
    ]


def _grads(dtype: torch.dtype) -> list[list[torch.Tensor]]:
    gen = torch.Generator().manual_seed(23)
    return [[torch.randn(s, generator=gen).mul_(0.1) for s in SHAPES] for _ in range(STEPS)]


def _fresh_process() -> None:
    """What a resumed run does at start-up: seed the RNGs, reset every SR noise stream."""
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    kaon.reseed_stochastic_rounding()


def _run(
    kind: str,
    *,
    resume_at: int | None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    **kw: Any,
) -> list[torch.Tensor]:
    """Run ``STEPS`` steps and return the final weights.

    ``resume_at=None`` runs straight through. ``resume_at=k`` checkpoints after step ``k``,
    simulates a brand-new process, rebuilds the optimizer, loads, and continues — the
    weights must come out bit-identical either way.
    """
    _fresh_process()
    params = _params(dtype, device)
    opt = _build(kind, params, **kw)
    grads = _grads(dtype)
    for i, g in enumerate(grads):
        if resume_at is not None and i == resume_at:
            if kind in EVAL_TO_SAVE:
                opt.eval()
            sd = copy.deepcopy(opt.state_dict())
            _fresh_process()
            opt = _build(kind, params, **kw)
            if kind in EVAL_TO_SAVE:
                opt.eval()
            opt.load_state_dict(sd)
            if kind in TRAIN_EVAL:
                opt.train()
        _step(kind, opt, params, g)
    return [p.detach().float().cpu().clone() for p in params]


def _assert_same(ref: list[torch.Tensor], got: list[torch.Tensor], what: str) -> None:
    for i, (a, b) in enumerate(zip(ref, got, strict=True)):
        if not torch.equal(a, b):
            pytest.fail(
                f"{what}: param {i} differs after resume, max abs "
                f"{(a - b).abs().max().item():.3e}"
            )


# ======================================================= cross-process resume, per optimizer
@pytest.mark.parametrize("momentum_dtype", ["bfloat16", "float32", "int8"])
@pytest.mark.parametrize("kind", [*PLAIN, *WRAPPED])
def test_resume_in_a_fresh_process_is_bit_identical_native_bf16(kind, momentum_dtype):
    """bf16 params, native (foreach) path: a resumed run must not skew the noise stream."""
    skip_if_no_cuda()
    kw: dict[str, Any] = dict(lr=1e-2, momentum_dtype=momentum_dtype, foreach=True)
    ref = _run(kind, resume_at=None, **kw)
    got = _run(kind, resume_at=STEPS // 2, **kw)
    _assert_same(ref, got, f"{kind}/{momentum_dtype}/foreach")


@pytest.mark.parametrize("kind", [*PLAIN, *WRAPPED])
def test_resume_in_a_fresh_process_is_bit_identical_per_param(kind):
    """Same guarantee on the per-parameter path (one SR write per weight, not per bucket)."""
    skip_if_no_cuda()
    kw: dict[str, Any] = dict(lr=1e-2, momentum_dtype="bfloat16", foreach=False)
    ref = _run(kind, resume_at=None, **kw)
    got = _run(kind, resume_at=STEPS // 2, **kw)
    _assert_same(ref, got, f"{kind}/per-param")


@pytest.mark.parametrize("kind", ["Adakaon", "Lion", "ScheduleFree", "Lookahead"])
def test_resume_in_a_fresh_process_is_bit_identical_on_cpu(kind):
    """The torch reference SR path (no Triton on CPU) resumes exactly too."""
    kw: dict[str, Any] = dict(lr=1e-2, momentum_dtype="bfloat16", foreach=True)
    ref = _run(kind, resume_at=None, device="cpu", **kw)
    got = _run(kind, resume_at=STEPS // 2, device="cpu", **kw)
    _assert_same(ref, got, f"{kind}/cpu")


def test_resume_is_bit_identical_with_the_triton_kernel_disabled():
    """``SR_TRITON=False`` pins the torch path on CUDA; the counter is shared, so it resumes."""
    skip_if_no_cuda()
    from kaon import _backend as bk

    prev = bk.SR_TRITON
    bk.SR_TRITON = False
    try:
        kw: dict[str, Any] = dict(lr=1e-2, momentum_dtype="bfloat16", foreach=True)
        ref = _run("Adakaon", resume_at=None, **kw)
        got = _run("Adakaon", resume_at=STEPS // 2, **kw)
    finally:
        bk.SR_TRITON = prev
    _assert_same(ref, got, "Adakaon/SR_TRITON=False")


def test_adakaon_fused_resume_stays_bit_identical():
    """The fused path seeds off ``Adakaon._t`` (already checkpointed) — keep it that way."""
    skip_if_no_cuda()
    kw: dict[str, Any] = dict(lr=1e-2, momentum_dtype="bfloat16", fused=True)
    ref = _run("Adakaon", resume_at=None, **kw)
    got = _run("Adakaon", resume_at=STEPS // 2, **kw)
    _assert_same(ref, got, "Adakaon/fused")


# ================================================== two optimizers in one process
def test_two_optimizers_in_one_process_do_not_clobber_each_others_stream():
    """Restoring one optimizer's noise counter must not move the other's.

    The pre-0.7.13 counter was process-global: loading a checkpoint could only either
    leave it alone (breaking the loader's own resume) or overwrite it (breaking every
    other live optimizer's). Per-optimizer streams make both resumes exact at once.
    """
    skip_if_no_cuda()
    kw: dict[str, Any] = dict(lr=1e-2, momentum_dtype="bfloat16", foreach=True)

    def run(*, resume: bool) -> list[torch.Tensor]:
        _fresh_process()
        pa = _params(torch.bfloat16, "cuda")
        pb = _params(torch.bfloat16, "cuda")
        oa: Any = Adakaon(pa, **kw)
        ob: Any = Lion(pb, **kw)
        grads = _grads(torch.bfloat16)
        for i, g in enumerate(grads):
            if resume and i == STEPS // 2:
                sda = copy.deepcopy(oa.state_dict())
                sdb = copy.deepcopy(ob.state_dict())
                _fresh_process()
                oa = Adakaon(pa, **kw)
                ob = Lion(pb, **kw)
                # Load in the REVERSE of the construction order: a per-optimizer stream
                # must not care, a global counter would.
                ob.load_state_dict(sdb)
                oa.load_state_dict(sda)
            _step("Adakaon", oa, pa, g)
            _step("Lion", ob, pb, g)
        return [p.detach().float().cpu().clone() for p in (*pa, *pb)]

    _assert_same(run(resume=False), run(resume=True), "Adakaon+Lion")


def test_co_resident_optimizers_draw_independent_noise():
    """Two live optimizers must not share a noise sequence (unbiased AND independent).

    Stepping the same weights from the same grads through two same-config optimizers is
    the only way to observe it: identical arithmetic, so any difference in the result is
    the SR noise, and identical results would mean the two streams collided.
    """
    skip_if_no_cuda()
    _fresh_process()
    pa = _params(torch.bfloat16, "cuda")
    pb = _params(torch.bfloat16, "cuda")
    kw: dict[str, Any] = dict(lr=1e-1, momentum_dtype="float32", foreach=True)
    oa = Adakaon(pa, **kw)
    ob = Adakaon(pb, **kw)
    for g in _grads(torch.bfloat16):
        _step("Adakaon", oa, pa, g)
        _step("Adakaon", ob, pb, g)
    assert any(
        not torch.equal(a.detach().cpu(), b.detach().cpu()) for a, b in zip(pa, pb, strict=True)
    ), "two co-resident optimizers drew the SAME stochastic-rounding noise"


def test_lookahead_sync_has_its_own_stream_apart_from_the_inner_optimizer():
    """The wrapper's phi sync and the inner Adakaon's weight write are separate streams."""
    skip_if_no_cuda()
    _fresh_process()
    params = _params(torch.bfloat16, "cuda")
    opt = Lookahead(params, lr=1e-2, k=2, alpha=0.5, momentum_dtype="bfloat16")
    for g in _grads(torch.bfloat16):
        _step("Lookahead", opt, params, g)
    assert opt.sr_stream is not opt.inner.sr_stream
    assert opt.sr_stream.stream_id != opt.inner.sr_stream.stream_id
    assert opt.sr_stream.draws > 0 and opt.inner.sr_stream.draws > 0


# ============================================================ compatibility / contract
def test_a_checkpoint_without_the_sr_key_still_loads():
    """0.7.12 checkpoints have no ``_sr_meta``: back-fill a fresh stream, do not raise."""
    _fresh_process()
    params = _params(torch.bfloat16, "cpu")
    opt = Adakaon(params, lr=1e-2, momentum_dtype="bfloat16")
    for g in _grads(torch.bfloat16)[:3]:
        _step("Adakaon", opt, params, g)
    sd = copy.deepcopy(opt.state_dict())
    assert "_sr_meta" in sd
    del sd["_sr_meta"]

    opt2 = Adakaon(params, lr=1e-2, momentum_dtype="bfloat16")
    opt2.load_state_dict(sd)
    assert opt2.sr_stream.draws == 0
    _step("Adakaon", opt2, params, _grads(torch.bfloat16)[3])  # must still step


def test_a_new_checkpoint_only_adds_a_top_level_key():
    """0.7.12's loaders ignore unknown top-level keys, so a new checkpoint loads there."""
    _fresh_process()
    params = _params(torch.bfloat16, "cpu")
    opt = Adakaon(params, lr=1e-2, momentum_dtype="bfloat16")
    _step("Adakaon", opt, params, _grads(torch.bfloat16)[0])
    sd = opt.state_dict()
    assert set(sd) == {"state", "param_groups", "_adakaon_meta", "_sr_meta"}
    # CPU params take the torch reference SR path, so the generator state rides along.
    assert set(sd["_sr_meta"]) == {"stream", "draws", "gen"}
    assert isinstance(sd["_sr_meta"]["draws"], int)
    assert list(sd["_sr_meta"]["gen"]) == ["-1"], "one generator state, for the CPU device"


def test_the_triton_kernel_path_needs_no_generator_state():
    """On CUDA the whole position is the counter, so the checkpoint stays two integers."""
    skip_if_no_cuda()
    _fresh_process()
    params = _params(torch.bfloat16, "cuda")
    opt = Adakaon(params, lr=1e-2, momentum_dtype="bfloat16")
    _step("Adakaon", opt, params, _grads(torch.bfloat16)[0])
    assert set(opt.state_dict()["_sr_meta"]) == {"stream", "draws"}


def test_a_corrupt_sr_counter_is_rejected():
    _fresh_process()
    params = _params(torch.bfloat16, "cpu")
    opt = Adakaon(params, lr=1e-2, momentum_dtype="bfloat16")
    _step("Adakaon", opt, params, _grads(torch.bfloat16)[0])
    sd = copy.deepcopy(opt.state_dict())
    sd["_sr_meta"] = {"stream": 0, "draws": -1}
    with pytest.raises(ValueError, match="stochastic-rounding"):
        Adakaon(params, lr=1e-2, momentum_dtype="bfloat16").load_state_dict(sd)


def test_reseed_restarts_every_stream_and_the_stream_allocator():
    """``reseed_stochastic_rounding()`` is still the one call that resets ALL SR noise."""
    _fresh_process()
    a = srm.SRStream()
    b = srm.SRStream()
    assert a.stream_id != b.stream_id, "co-resident streams must get distinct identities"
    a.next_seed(torch.device("cpu"))
    a.next_seed(torch.device("cpu"))
    assert a.draws == 2
    kaon.reseed_stochastic_rounding()
    assert a.snapshot()["draws"] == 0, "reseed must restart a live stream"
    assert srm.SRStream().stream_id == a.stream_id, (
        "reseed must restart the allocator so a fresh optimizer reproduces its stream"
    )


def test_stream_zero_on_device_zero_keeps_the_0_7_12_seed_sequence():
    """Compatibility anchor: the first stream on cuda:0 reproduces the old global counter."""
    skip_if_no_cuda()
    _fresh_process()
    dev = torch.device("cuda", 0)
    base = torch.cuda.default_generators[0].initial_seed()
    stream = srm.SRStream()
    assert stream.stream_id == 0
    for k in (1, 2, 3):
        assert stream.next_seed(dev) == (base + k * 0x9E3779B1) & 0x7FFFFFFF


def test_a_checkpoint_taken_right_after_a_resume_keeps_the_torch_path_position():
    """Save -> load -> save (no step between) must not drop the generator state.

    The restored generator states are STAGED and only applied to the generator that is
    actually reached, so a re-save before the first step has to carry the staged copy
    through — otherwise the second checkpoint resumes from a fresh generator.
    """
    _fresh_process()
    params = _params(torch.bfloat16, "cpu")
    grads = _grads(torch.bfloat16)
    opt = Adakaon(params, lr=1e-2, momentum_dtype="bfloat16")
    for g in grads[:4]:
        _step("Adakaon", opt, params, g)
    first = copy.deepcopy(opt.state_dict())

    _fresh_process()
    opt2 = Adakaon(params, lr=1e-2, momentum_dtype="bfloat16")
    opt2.load_state_dict(first)
    second = copy.deepcopy(opt2.state_dict())        # re-saved without stepping
    assert second["_sr_meta"]["gen"].keys() == first["_sr_meta"]["gen"].keys()
    for key, state in first["_sr_meta"]["gen"].items():
        assert torch.equal(second["_sr_meta"]["gen"][key], state), key
