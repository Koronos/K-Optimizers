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
import io
import os
import subprocess
import sys
import textwrap
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


def _roundtrip(state_dict: dict[str, Any], map_location: Any) -> dict[str, Any]:
    """Serialize and read back the way a real resume does.

    ``map_location`` is the part that matters here: the torch reference path's generator
    state is a **CPU** byte tensor, and a resume that maps the whole checkpoint onto the
    training device (the common ``map_location="cuda"`` / ``map_location=device`` spelling)
    hands it back on CUDA — where ``Generator.set_state`` refuses it.
    """
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    buf.seek(0)
    return torch.load(buf, map_location=map_location, weights_only=False)


def _run(
    kind: str,
    *,
    resume_at: int | None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    map_location: Any = None,
    **kw: Any,
) -> list[torch.Tensor]:
    """Run ``STEPS`` steps and return the final weights.

    ``resume_at=None`` runs straight through. ``resume_at=k`` checkpoints after step ``k``,
    simulates a brand-new process, rebuilds the optimizer, loads, and continues — the
    weights must come out bit-identical either way. ``map_location`` routes the checkpoint
    through a real ``torch.save``/``torch.load`` instead of a deep copy.
    """
    _fresh_process()
    params = _params(dtype, device)
    opt = _build(kind, params, **kw)
    grads = _grads(dtype)
    for i, g in enumerate(grads):
        if resume_at is not None and i == resume_at:
            if kind in EVAL_TO_SAVE:
                opt.eval()
            if map_location is None:
                sd = copy.deepcopy(opt.state_dict())
            else:
                sd = _roundtrip(opt.state_dict(), map_location)
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
    cpu = torch.device("cpu")
    _fresh_process()
    a = srm.SRStream()
    b = srm.SRStream()
    assert a.stream_id is b.stream_id is None, "an id is claimed by DRAWING, not by existing"
    a.next_seed(cpu)
    a.next_seed(cpu)
    b.next_seed(cpu)
    assert a.stream_id != b.stream_id, "co-resident streams must get distinct identities"
    assert a.draws == 2
    kaon.reseed_stochastic_rounding()
    assert a.draws == 0 or a.snapshot() is None, "reseed must restart a live stream"
    fresh = srm.SRStream()
    fresh.next_seed(cpu)
    assert fresh.stream_id == 0, (
        "reseed must restart the allocator so a fresh optimizer reproduces stream 0"
    )


def test_stream_zero_on_device_zero_keeps_the_0_7_12_seed_sequence():
    """Compatibility anchor: the first stream on cuda:0 reproduces the old global counter."""
    skip_if_no_cuda()
    _fresh_process()
    dev = torch.device("cuda", 0)
    base = torch.cuda.default_generators[0].initial_seed()
    stream = srm.SRStream()
    for k in (1, 2, 3):
        assert stream.next_seed(dev) == (base + k * 0x9E3779B1) & 0x7FFFFFFF
    assert stream.stream_id == 0, "the first stream to draw is the compatibility anchor"


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


# ================================================ torch.load(map_location=...) round trips
# The torch reference path's position is a `torch.Generator` state — a **CPU** byte tensor,
# even for a CUDA generator. A resume almost always reads its checkpoint with
# `map_location` pointing at the training device, which moves that tensor to CUDA, and
# `Generator.set_state` only accepts a CPU ByteTensor. Because the restored state is
# *staged* and applied on the first draw, the failure lands on the first `step()` after a
# successful-looking load — the worst possible place. These pin the whole round trip.
MAP_LOCATION_CASES = {
    # SAM's climb calls `add_stochastic_` directly (sam.py), so the torch path — and a
    # `gen` payload — is what a plain bf16 CUDA SAM produces by DEFAULT.
    "SAM-default": ("SAM", dict(lr=1e-2, momentum_dtype="bfloat16", foreach=True)),
    # The documented switch that pins the reference implementation on CUDA.
    "Adakaon-SR_TRITON-off": ("Adakaon", dict(lr=1e-2, momentum_dtype="bfloat16",
                                              foreach=True)),
}


@pytest.mark.parametrize("case", list(MAP_LOCATION_CASES))
def test_a_checkpoint_read_with_map_location_cuda_resumes_bit_identically(case):
    """``torch.load(..., map_location=cuda)`` must not poison the staged generator state."""
    skip_if_no_cuda()
    from kaon import _backend as bk

    kind, kw = MAP_LOCATION_CASES[case]
    prev = bk.SR_TRITON
    if case == "Adakaon-SR_TRITON-off":
        bk.SR_TRITON = False
    try:
        ref = _run(kind, resume_at=None, **kw)
        got = _run(kind, resume_at=STEPS // 2, map_location=torch.device("cuda:0"), **kw)
    finally:
        bk.SR_TRITON = prev
    _assert_same(ref, got, f"{case}/map_location=cuda:0")


def test_a_channels_last_weight_resumes_bit_identically_through_map_location():
    """A strided weight the kernel cannot index falls to the torch path on CUDA.

    ``sr_add_supported`` rejects a non-contiguous target, so a ``channels_last`` conv
    weight emits a ``gen`` payload even on a Triton build with the kernel enabled — no
    opt-in knob required to reach the crash.
    """
    skip_if_no_cuda()

    def run(*, resume: bool) -> torch.Tensor:
        _fresh_process()
        p = torch.nn.Parameter(
            torch.randn(4, 3, 8, 8).to(device="cuda", dtype=torch.bfloat16)
            .to(memory_format=torch.channels_last)
        )
        assert not p.is_contiguous(), "the point of this test is a strided weight"
        opt: Any = Adakaon([p], lr=1e-2, momentum_dtype="bfloat16")
        gen = torch.Generator().manual_seed(5)
        for i in range(6):
            if resume and i == 3:
                sd = _roundtrip(opt.state_dict(), torch.device("cuda:0"))
                assert "gen" in sd["_sr_meta"], "the strided weight must take the torch path"
                _fresh_process()
                opt = Adakaon([p], lr=1e-2, momentum_dtype="bfloat16")
                opt.load_state_dict(sd)
            p.grad = (torch.randn(4, 3, 8, 8, generator=gen) * 0.1).to(
                device="cuda", dtype=torch.bfloat16
            )
            opt.step()
        return p.detach().float().cpu().clone()

    _assert_same([run(resume=False)], [run(resume=True)], "channels_last/map_location")


def test_a_cpu_checkpoint_resumes_onto_cuda():
    """Trained on CPU, resumed on GPU: the staged CPU generator state must still apply."""
    skip_if_no_cuda()
    _fresh_process()
    cpu_params = _params(torch.bfloat16, "cpu")
    opt = Adakaon(cpu_params, lr=1e-2, momentum_dtype="bfloat16")
    for g in _grads(torch.bfloat16)[:3]:
        _step("Adakaon", opt, cpu_params, g)
    sd = _roundtrip(opt.state_dict(), torch.device("cuda:0"))
    assert "gen" in sd["_sr_meta"]

    _fresh_process()
    gpu_params = [torch.nn.Parameter(p.detach().cuda()) for p in cpu_params]
    opt2 = Adakaon(gpu_params, lr=1e-2, momentum_dtype="bfloat16")
    opt2.load_state_dict(sd)
    _step("Adakaon", opt2, gpu_params, _grads(torch.bfloat16)[3])
    assert all(torch.isfinite(p).all() for p in gpu_params)


def test_an_invalid_generator_payload_is_rejected_at_load():
    """A corrupt ``gen`` blob must fail as a checkpoint error at LOAD, not as a TypeError
    from deep inside the first step."""
    _fresh_process()
    params = _params(torch.bfloat16, "cpu")
    opt = Adakaon(params, lr=1e-2, momentum_dtype="bfloat16")
    _step("Adakaon", opt, params, _grads(torch.bfloat16)[0])
    sd = copy.deepcopy(opt.state_dict())
    sd["_sr_meta"]["gen"] = {"-1": torch.zeros(3)}      # float32, and the wrong length
    with pytest.raises(ValueError, match="stochastic-rounding"):
        Adakaon(params, lr=1e-2, momentum_dtype="bfloat16").load_state_dict(sd)


# ============================================================== nested wrappers
# `SAM(base_optimizer=Lookahead, ...)` is reachable through the public API and makes THREE
# noise owners: SAM's climb, Lookahead's phi sync and the inner Adakaon's weight write. A
# single shared `_sr_wrap_meta` key means the outer wrapper's `state_dict` overwrites the
# intermediate one's, and the resume silently continues from the wrong position (measured
# 2.34e-2 on bf16 weights).
_NESTED_SCRIPT = """
import hashlib, sys, torch, kaon
from kaon import SAM, Lookahead

SHAPES = [(64, 32), (48, 16), (24,)]
STEPS, SPLIT = 8, 4
mode, ckpt = sys.argv[1], sys.argv[2]

def fresh():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    kaon.reseed_stochastic_rounding()

def params():
    g = torch.Generator().manual_seed(11)
    return [torch.nn.Parameter(torch.randn(s, generator=g).to("cuda", torch.bfloat16))
            for s in SHAPES]

def grads():
    g = torch.Generator().manual_seed(23)
    return [[torch.randn(s, generator=g).mul_(0.1) for s in SHAPES] for _ in range(STEPS)]

def step(o, ps, gs):
    for p, g in zip(ps, gs):
        p.grad = g.detach().clone().to(device=p.device, dtype=p.dtype)
    o.first_step()
    for p, g in zip(ps, gs):
        p.grad = g.detach().clone().to(device=p.device, dtype=p.dtype)
    o.second_step()

fresh()
ps, gs = params(), grads()
# SAM has no train()/eval() of its own; the inner Lookahead defaults to train mode.
o = SAM(ps, base_optimizer=Lookahead, rho=0.05, lr=1e-2, k=2, alpha=0.5,
        momentum_dtype="bfloat16")
lo, hi = (0, STEPS) if mode == "continuous" else (
    (0, SPLIT) if mode == "save" else (SPLIT, STEPS))
if mode == "resume":
    o.load_state_dict(torch.load(ckpt, map_location=torch.device("cuda:0"),
                                 weights_only=False))
    for p, w in zip(ps, torch.load(ckpt + ".w", weights_only=False)):
        p.data.copy_(w.to(p.device))
for i in range(lo, hi):
    step(o, ps, gs[i])
if mode == "save":
    torch.save(o.state_dict(), ckpt)
    torch.save([p.detach().cpu().clone() for p in ps], ckpt + ".w")
else:
    h = hashlib.blake2b(digest_size=12)
    for p in ps:
        h.update(p.detach().float().cpu().contiguous().reshape(-1).numpy().tobytes())
    print(h.hexdigest())
"""


def _run_script(script: str, tmp_path, *args: str) -> str:
    path = tmp_path / "nested_resume.py"
    path.write_text(textwrap.dedent(script), encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(kaon.__file__))
    out = subprocess.run(
        [sys.executable, str(path), *args], check=True, capture_output=True, text=True,
        env=env, cwd=str(tmp_path),
    )
    return out.stdout.strip()


def test_nested_wrappers_resume_bit_identically_across_two_real_processes(tmp_path):
    """SAM over Lookahead over Adakaon: three owners, three namespaced positions.

    Two genuinely separate interpreters, so nothing about the noise position can leak
    through process state — the only channel is the checkpoint.
    """
    skip_if_no_cuda()
    ckpt = str(tmp_path / "ck.pt")
    ref = _run_script(_NESTED_SCRIPT, tmp_path, "continuous", ckpt)
    _run_script(_NESTED_SCRIPT, tmp_path, "save", ckpt)
    got = _run_script(_NESTED_SCRIPT, tmp_path, "resume", ckpt)
    assert ref and got and ref == got, f"nested wrappers diverged: {ref} vs {got}"


def test_each_wrapper_gets_its_own_namespaced_sr_key():
    """The keys must not collide, or the outer wrapper silently eats the inner one's."""
    skip_if_no_cuda()
    _fresh_process()
    params = _params(torch.bfloat16, "cuda")
    opt = SAM(params, base_optimizer=Lookahead, rho=0.05, lr=1e-2, k=1, alpha=0.5,
              momentum_dtype="bfloat16")
    _step("SAM", opt, params, _grads(torch.bfloat16)[0])
    sd = opt.state_dict()
    wrap_keys = sorted(k for k in sd if k.startswith("_sr_wrap_meta"))
    assert wrap_keys == ["_sr_wrap_meta_lookahead", "_sr_wrap_meta_sam"], wrap_keys
    assert "_sr_meta" in sd, "the innermost Adakaon still uses the plain key"


# ============================================================== stream-id economy
def test_an_optimizer_that_never_rounds_does_not_claim_a_stream_id():
    """fp32 params / kahan never draw SR noise, so they must not shift anyone's stream.

    A diffusion run routinely holds fp32 and bf16 groups (or several optimizers); if a
    non-rounding optimizer burned a stream id just by stepping, it would move the bf16
    optimizer's noise off the 0.7.12 sequence for no reason at all.
    """
    skip_if_no_cuda()
    _fresh_process()
    fp32 = _params(torch.float32, "cuda")
    kahan_params = _params(torch.bfloat16, "cuda")
    bf16 = _params(torch.bfloat16, "cuda")
    plain = Adakaon(fp32, lr=1e-2, momentum_dtype="bfloat16")
    kahan = Adakaon(kahan_params, lr=1e-2, momentum_dtype="bfloat16", bf16_method="kahan")
    rounds = Adakaon(bf16, lr=1e-2, momentum_dtype="bfloat16")
    for g in _grads(torch.bfloat16)[:2]:
        _step("Adakaon", plain, fp32, g)
        _step("Adakaon", kahan, kahan_params, g)
        _step("Adakaon", rounds, bf16, g)
    assert plain.sr_stream.stream_id is None, "fp32 params must not claim a stream id"
    assert kahan.sr_stream.stream_id is None, "kahan must not claim a stream id"
    assert rounds.sr_stream.stream_id == 0, "the rounding optimizer keeps the anchor"
    assert "_sr_meta" not in plain.state_dict(), "no draws -> nothing to checkpoint"
    assert "_sr_meta" in rounds.state_dict()


# ==================================================== owners that draw for the FIRST time
# after a resume. Their id is not in the checkpoint (they had no position to save), so they
# take one from the allocator — which must already be past every id the load adopted, or
# they collide with a restored owner and the two share a seed sequence. Both scenarios below
# are ordinary settings, and both are checked across two real interpreters.
_LATE_SYNC_SCRIPT = """
import hashlib, sys, torch, kaon
from kaon import Lookahead

SHAPES = [(64, 32), (48, 16), (24,)]
STEPS, SPLIT, K = 10, 4, 6      # the phi sync draws for the first time at step 6 > SPLIT
mode, ckpt = sys.argv[1], sys.argv[2]

def fresh():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    kaon.reseed_stochastic_rounding()

def params():
    g = torch.Generator().manual_seed(11)
    return [torch.nn.Parameter(torch.randn(s, generator=g).to("cuda", torch.bfloat16))
            for s in SHAPES]

def grads():
    g = torch.Generator().manual_seed(23)
    return [[torch.randn(s, generator=g).mul_(0.1) for s in SHAPES] for _ in range(STEPS)]

fresh()
ps, gs = params(), grads()
o = Lookahead(ps, lr=1e-2, k=K, alpha=0.5, momentum_dtype="bfloat16")
lo, hi = (0, STEPS) if mode == "continuous" else (
    (0, SPLIT) if mode == "save" else (SPLIT, STEPS))
if mode == "resume":
    o.load_state_dict(torch.load(ckpt, map_location=torch.device("cuda:0"),
                                 weights_only=False))
    for p, w in zip(ps, torch.load(ckpt + ".w", weights_only=False)):
        p.data.copy_(w.to(p.device))
for i in range(lo, hi):
    for p, g in zip(ps, gs[i]):
        p.grad = g.detach().clone().to(device=p.device, dtype=p.dtype)
    o.step()
if mode == "save":
    torch.save(o.state_dict(), ckpt)
    torch.save([p.detach().cpu().clone() for p in ps], ckpt + ".w")
else:
    h = hashlib.blake2b(digest_size=12)
    for p in ps:
        h.update(p.detach().float().cpu().contiguous().reshape(-1).numpy().tobytes())
    print(h.hexdigest())
"""

_LATE_SECOND_SCRIPT = """
import hashlib, sys, torch, kaon
from kaon import Adakaon, Lion

SHAPES = [(64, 32), (48, 16), (24,)]
STEPS, SPLIT, THAW = 10, 4, 6    # the second optimizer starts stepping at 6 > SPLIT
mode, ckpt = sys.argv[1], sys.argv[2]

def fresh():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    kaon.reseed_stochastic_rounding()

def params(seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(s, generator=g).to("cuda", torch.bfloat16))
            for s in SHAPES]

def grads():
    g = torch.Generator().manual_seed(23)
    return [[torch.randn(s, generator=g).mul_(0.1) for s in SHAPES] for _ in range(STEPS)]

def step(o, ps, gs):
    for p, g in zip(ps, gs):
        p.grad = g.detach().clone().to(device=p.device, dtype=p.dtype)
    o.step()

fresh()
pa, pb, gs = params(11), params(12), grads()
oa = Adakaon(pa, lr=1e-2, momentum_dtype="bfloat16")
ob = Lion(pb, lr=1e-2, momentum_dtype="bfloat16")
lo, hi = (0, STEPS) if mode == "continuous" else (
    (0, SPLIT) if mode == "save" else (SPLIT, STEPS))
if mode == "resume":
    sd = torch.load(ckpt, map_location=torch.device("cuda:0"), weights_only=False)
    oa.load_state_dict(sd["a"])
    ob.load_state_dict(sd["b"])
    for p, w in zip(pa + pb, torch.load(ckpt + ".w", weights_only=False)):
        p.data.copy_(w.to(p.device))
for i in range(lo, hi):
    step(oa, pa, gs[i])
    if i >= THAW:
        step(ob, pb, gs[i])
if mode == "save":
    torch.save({"a": oa.state_dict(), "b": ob.state_dict()}, ckpt)
    torch.save([p.detach().cpu().clone() for p in pa + pb], ckpt + ".w")
else:
    h = hashlib.blake2b(digest_size=12)
    for p in pa + pb:
        h.update(p.detach().float().cpu().contiguous().reshape(-1).numpy().tobytes())
    print(h.hexdigest())
"""


@pytest.mark.parametrize(
    "script,name",
    [(_LATE_SYNC_SCRIPT, "lookahead-k6-sync-after-resume"),
     (_LATE_SECOND_SCRIPT, "second-optimizer-thawed-after-resume")],
    ids=["late-sync", "late-second-optimizer"],
)
def test_an_owner_that_first_draws_after_the_resume_gets_a_free_id(script, name, tmp_path):
    """The allocator must be past every restored id, across two real processes.

    ``Lookahead(k=6)`` checkpointed at step 4 has not synced yet, so its wrapper stream is
    absent from the checkpoint and claims an id only at step 6; a second optimizer unfrozen
    after the resume is the same shape of problem. Without the watermark both land on id 0
    — already taken by the restored inner optimizer — and the two owners then draw the same
    seeds (measured 1.56e-2 and 3.13e-2).
    """
    skip_if_no_cuda()
    ckpt = str(tmp_path / f"{name}.pt")
    ref = _run_script(script, tmp_path, "continuous", ckpt)
    _run_script(script, tmp_path, "save", ckpt)
    got = _run_script(script, tmp_path, "resume", ckpt)
    assert ref and got and ref == got, f"{name} diverged: {ref} vs {got}"


def test_restoring_an_id_pushes_the_allocator_past_it():
    """The unit-level invariant behind the two process tests above."""
    cpu = torch.device("cpu")
    _fresh_process()
    restored = srm.SRStream()
    restored.restore({"stream": 3, "draws": 12})
    late = srm.SRStream()
    late.next_seed(cpu)
    assert restored.stream_id == 3
    assert late.stream_id == 4, "a first-time drawer must not land on a restored id"


def test_reseed_hands_out_distinct_ids_to_live_and_new_streams():
    """A reseed restarts the allocator, so live streams must release their ids too.

    Otherwise a process that reseeds with optimizers already alive and then builds another
    one puts two owners on the same seed sequence — the pre-0.7.13 defect, reintroduced.
    """
    cpu = torch.device("cpu")
    _fresh_process()
    a, b = srm.SRStream(), srm.SRStream()
    a.next_seed(cpu)
    b.next_seed(cpu)
    assert {a.stream_id, b.stream_id} == {0, 1}

    kaon.reseed_stochastic_rounding()
    # The release is lazy, like every other part of a reseed: a stale id is still readable
    # but can never be USED, because any draw (or snapshot) applies the reset first.
    assert a.snapshot() is None, "a released stream has no position to checkpoint"
    c = srm.SRStream()
    a.next_seed(cpu)
    c.next_seed(cpu)
    b.next_seed(cpu)
    ids = [a.stream_id, c.stream_id, b.stream_id]
    assert ids == [0, 1, 2], f"re-claim must follow draw order and stay unique, got {ids}"
    assert len(set(ids)) == 3, "no two live streams may share an identity"


def test_reseed_keeps_two_runs_of_a_fresh_optimizer_identical():
    """The reproducibility contract the released allocator must not break: same seed +
    reseed + a freshly built optimizer reproduces the previous run bit for bit."""
    skip_if_no_cuda()

    def once() -> list[torch.Tensor]:
        _fresh_process()
        params = _params(torch.bfloat16, "cuda")
        opt = Adakaon(params, lr=1e-2, momentum_dtype="bfloat16")
        for g in _grads(torch.bfloat16)[:4]:
            _step("Adakaon", opt, params, g)
        return [p.detach().float().cpu().clone() for p in params]

    _assert_same(once(), once(), "manual_seed+reseed rerun")


def test_the_kernels_process_wide_fallback_keeps_stream_zero_across_a_reseed():
    """The fallback's id is PINNED: releasing it on a reseed would move the sequence that
    un-threaded callers of ``sr_add_`` rely on off the 0.7.12 anchor."""
    skip_if_no_cuda()
    import kaon._fused_triton as ft

    _fresh_process()
    dev = torch.device("cuda", 0)
    ft._PROCESS_SR_STREAM.next_seed(dev)
    kaon.reseed_stochastic_rounding()
    srm.SRStream().next_seed(dev)                    # an optimizer claims id 0 meanwhile
    base = torch.cuda.default_generators[0].initial_seed()
    assert ft._PROCESS_SR_STREAM.next_seed(dev) == (base + 0x9E3779B1) & 0x7FFFFFFF
    assert ft._PROCESS_SR_STREAM.stream_id == 0
