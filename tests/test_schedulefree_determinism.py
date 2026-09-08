"""ScheduleFree's reproducibility contract: what is bit-identical against itself, and why
its DEFAULT configuration is not.

`ScheduleFree` was reported as "not reproducible against itself": two identical runs in one
process, same `torch.manual_seed`, fp32 parameters, no gradient centralization, identical
gradients, diverge from the **second** step — while every other kaon optimizer has a noise
floor of exactly `0.0` in the same harness. That report is accurate, and this file pins the
cause, which is neither an unordered reduction nor an uninitialized buffer:

`momentum_dtype="bfloat16"` is ScheduleFree's **default**, and a bf16-stored `z` MUST be
written back with stochastic rounding or it stops moving altogether (the per-step
`lr_t*d` sits below the bf16 ULP of `z`; see `ScheduleFree._store_z`). So ScheduleFree is
the **only** kaon optimizer that draws SR noise for a model whose weights are fp32 — every
other one reaches the SR write only through a low-precision *weight*, which is why their
fp32 noise floor is 0 (`tests/test_sr_seed_checkpoint.py`'s
`test_an_optimizer_that_never_rounds_does_not_claim_a_stream_id` pins that invariant for
Adakaon). Drawing at all is what exposes ScheduleFree to the SR stream's documented
reproducibility protocol: a stream's identity is allocated in order of *first draw* from a
process-wide allocator that only `kaon.reseed_stochastic_rounding()` restarts, so the
second run of a two-run comparison lands on stream 1 instead of stream 0 and rounds `z`
off a different noise offset. That reaches the weights one step later, through the
`ckp1*(y - z)` averaging term — hence "from the second step", at ~one bf16 grid step of `z`.

Everything below is the evidence, in the order it settles the question:

* under the documented protocol (`torch.manual_seed` + `kaon.reseed_stochastic_rounding()`)
  ScheduleFree IS bit-identical against itself, on every device, both paths, every
  parameter and momentum dtype;
* a `z` that is not bf16 (`float32`/`int8`/`4bit`) draws no noise at all, claims no stream
  and needs no reseed;
* with a bf16 `z` the two runs claim *different* stream identities and diverge at the
  second step, and giving both runs the SAME (pinned) identity makes them bit-identical
  again with no reseed — the identity is the only variable;
* the desired end state — the default configuration reproducible under `torch.manual_seed`
  alone, like every other optimizer with fp32 weights — is a strict xfail at the bottom.
"""

from __future__ import annotations

import pytest
import torch

from kaon import ScheduleFree, reseed_stochastic_rounding
from kaon._stochastic_rounding import SRStream

from .conftest import skip_if_no_cuda

# Two same-shape 2-D weights (so the foreach path stacks with N > 1 and `_store_z_stacked`
# really writes back), a 1-D bias and a 0-D scalar.
SHAPES = [(8, 4), (8, 4), (5,), ()]
STEPS = 4
SEED = 1234

MOMENTUM_DTYPES = ["bfloat16", "float32", "int8", "4bit"]
# The three that store `z` through the codec's round-to-nearest write, i.e. no SR draw.
NO_SR_MOMENTUM = ["float32", "int8", "4bit"]


def _seed_torch() -> None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _bag(device: str, dtype: torch.dtype) -> list[torch.nn.Parameter]:
    """The parameter bag, drawn from a private generator.

    Deliberately NOT from the global RNG: the weights and the gradients have to be
    identical across the two runs whatever the global stream did, so the only thing left
    that can differ between them is kaon's own SR noise.
    """
    gen = torch.Generator().manual_seed(11)
    return [
        torch.nn.Parameter(torch.randn(s, generator=gen).to(device=device, dtype=dtype))
        for s in SHAPES
    ]


def _grads(device: str, dtype: torch.dtype, steps: int = STEPS) -> list[list[torch.Tensor]]:
    gen = torch.Generator().manual_seed(7)
    return [
        [torch.randn(s, generator=gen).to(device=device, dtype=dtype) for s in SHAPES]
        for _ in range(steps)
    ]


def _run(
    *,
    device: str = "cpu",
    param_dtype: torch.dtype = torch.float32,
    momentum_dtype: str = "bfloat16",
    foreach: bool = True,
    reseed: bool,
    steps: int = STEPS,
    stream: SRStream | None = None,
    bf16_method: str = "stochastic_rounding",
) -> tuple[list[list[torch.Tensor]], ScheduleFree]:
    """One run; returns the fp32 weight snapshot after every step, plus the optimizer.

    ``reseed`` is the whole difference between the documented protocol
    (``torch.manual_seed`` + ``kaon.reseed_stochastic_rounding()``) and the bare
    ``torch.manual_seed`` a user writes at the top of a script.
    """
    _seed_torch()
    if reseed:
        reseed_stochastic_rounding()
    params = _bag(device, param_dtype)
    opt = ScheduleFree(params, lr=1e-2, momentum_dtype=momentum_dtype, foreach=foreach,
                       bf16_method=bf16_method)
    if stream is not None:
        opt.__dict__["sr_stream"] = stream
    opt.train()
    snaps = []
    for grads in _grads(device, param_dtype, steps):
        for p, g in zip(params, grads, strict=True):
            p.grad = g.clone()
        opt.step()
        snaps.append([p.detach().float().cpu().clone() for p in params])
    return snaps, opt


def _first_difference(a: list[list[torch.Tensor]], b: list[list[torch.Tensor]]) -> int | None:
    """Index of the first step whose weights differ by a single bit, or ``None``."""
    for step, (xs, ys) in enumerate(zip(a, b, strict=True)):
        if any(not torch.equal(x, y) for x, y in zip(xs, ys, strict=True)):
            return step
    return None


def _max_abs(a: list[list[torch.Tensor]], b: list[list[torch.Tensor]]) -> float:
    return max(
        (x - y).abs().max().item()
        for xs, ys in zip(a, b, strict=True)
        for x, y in zip(xs, ys, strict=True)
    )


# ============================================================ the contract that DOES hold
@pytest.mark.parametrize("momentum_dtype", MOMENTUM_DTYPES)
@pytest.mark.parametrize("param_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("foreach", [True, False], ids=["foreach", "per_param"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_two_identical_runs_are_bit_identical(device, foreach, param_dtype, momentum_dtype):
    """Under the documented protocol ScheduleFree reproduces itself bit for bit.

    ``torch.manual_seed`` + ``kaon.reseed_stochastic_rounding()`` is the protocol the whole
    SR suite runs on (``tests/test_sr_seed_checkpoint.py``'s ``_fresh_process``), and it is
    what a rerun inside one process needs: re-seeding the global RNG to the *same* value is
    not observable through it, so nothing else can tell kaon that a new run started.

    This is the regression net for the reported "ScheduleFree is not reproducible against
    itself": there is no unordered reduction, no ``torch.empty`` state and no ``id()``-
    ordered iteration anywhere in the step — with the noise stream reset, every path and
    every dtype lands on the same bits.
    """
    if device == "cuda":
        skip_if_no_cuda()
    a, _ = _run(device=device, param_dtype=param_dtype,
                momentum_dtype=momentum_dtype, foreach=foreach, reseed=True)
    b, _ = _run(device=device, param_dtype=param_dtype,
                momentum_dtype=momentum_dtype, foreach=foreach, reseed=True)
    step = _first_difference(a, b)
    assert step is None, (
        f"two identical runs diverged at step {step} (max abs {_max_abs(a, b):.3e})"
    )


# ==================================================== a z that is not bf16 draws no noise
@pytest.mark.parametrize("momentum_dtype", NO_SR_MOMENTUM)
@pytest.mark.parametrize("foreach", [True, False], ids=["foreach", "per_param"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_a_non_bf16_z_needs_no_reseed_and_claims_no_stream(device, foreach, momentum_dtype):
    """fp32 weights + a ``z`` the codec writes round-to-nearest = no SR draw at all.

    This is the deterministic path ScheduleFree already has, and the answer to "how do I
    get a reproducible ScheduleFree without knowing anything about stochastic rounding":
    ``momentum_dtype="float32"`` (4 B/param for ``z``, the exact choice), or ``"int8"`` /
    ``"4bit"`` when the memory matters more than ``z``'s own stall risk at small
    ``lr*d``. Bit-identical here with NO ``reseed_stochastic_rounding()`` anywhere — the
    bare ``torch.manual_seed`` a user writes is enough.
    """
    if device == "cuda":
        skip_if_no_cuda()
    reseed_stochastic_rounding()      # hermetic: a clean run boundary for the allocator
    a, oa = _run(device=device, momentum_dtype=momentum_dtype, foreach=foreach, reseed=False)
    b, ob = _run(device=device, momentum_dtype=momentum_dtype, foreach=foreach, reseed=False)
    step = _first_difference(a, b)
    assert step is None, (
        f"an fp32 model with a {momentum_dtype!r} z is not supposed to draw any noise, "
        f"yet the runs diverged at step {step}"
    )
    for opt in (oa, ob):
        stream = opt.__dict__.get("sr_stream")
        assert stream is None or stream.stream_id is None, (
            "a ScheduleFree that never rounds must not claim a stream identity"
        )
        assert "_sr_meta" not in opt.state_dict(), "no draws -> nothing to checkpoint"


# ==================================================== the cause: the bf16 z's SR identity
@pytest.mark.parametrize("bf16_method", ["stochastic_rounding", "none"])
@pytest.mark.parametrize("foreach", [True, False], ids=["foreach", "per_param"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_a_bf16_z_makes_an_fp32_model_stochastic_from_the_second_step(
    device, foreach, bf16_method
):
    """THE reported symptom, and its mechanism, pinned.

    fp32 weights, fp32 gradients, no reseed between the runs — the harness the report used.
    The first step's weights still match (the ``z`` both runs read is the initialization
    ``z0 == x0 == p``, and the fp32 ``y`` write is exact); the differently-rounded ``z``
    reaches the weights one step later through the ``ckp1*(y - z)`` averaging term. Hence
    the divergence starting at step index 1.

    The mechanism is the stream *identity*: run 1 claims stream 0 from the process-wide
    allocator at its first draw and run 2 claims stream 1, so the two round ``z`` off
    different noise offsets. Documented as a known limitation of the 0.7.13 per-owner
    streams ("allocated in order of first draw"), and invisible for every other optimizer
    with fp32 weights because they never draw at all.

    ``bf16_method`` is parameterized to pin that it does NOT govern ``z``: the divergence
    is identical with the weight write's SR turned off (``"none"``), which is what rules
    out the ``y`` write-back as the source. With fp32 weights that branch is never taken
    anyway — every draw in this run belongs to ``_store_z``.
    """
    if device == "cuda":
        skip_if_no_cuda()
    reseed_stochastic_rounding()      # hermetic: run 1 must be the allocator's first
    a, oa = _run(device=device, momentum_dtype="bfloat16", foreach=foreach,
                 reseed=False, bf16_method=bf16_method)
    b, ob = _run(device=device, momentum_dtype="bfloat16", foreach=foreach,
                 reseed=False, bf16_method=bf16_method)
    assert oa.sr_stream.stream_id == 0
    assert ob.sr_stream.stream_id == 1, (
        "the second run must have taken the next free identity; if this changed, the "
        "allocator was fixed and the xfail at the bottom of this file should now XPASS"
    )
    assert _first_difference(a, b) == 1, (
        "a bf16 z is stochastically rounded, so two runs on different stream identities "
        "must agree on step 0 and part ways on step 1"
    )
    # A few bf16 grid steps of z (|z| ~ 3 for this bag -> ULP 2**-6), damped by ckp1 = 1/t.
    assert _max_abs(a, b) < 0.05, (
        f"the divergence must stay inside z's own bf16 grid, got {_max_abs(a, b):.3e}"
    )


@pytest.mark.parametrize("foreach", [True, False], ids=["foreach", "per_param"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pinning_the_stream_identity_makes_the_two_runs_identical(device, foreach):
    """The decisive test: the identity is the ONLY variable between the two runs.

    Hand both runs a stream pinned to id 0 (``SRStream(0)``, the same construction the
    Triton module's process-wide fallback uses) and the bf16 ``z`` rounds identically with
    no ``reseed_stochastic_rounding()`` anywhere — so nothing in ScheduleFree's arithmetic
    is non-deterministic, and nothing else in the step consumes entropy.
    """
    if device == "cuda":
        skip_if_no_cuda()
    a, _ = _run(device=device, momentum_dtype="bfloat16", foreach=foreach,
                reseed=False, stream=SRStream(0))
    b, _ = _run(device=device, momentum_dtype="bfloat16", foreach=foreach,
                reseed=False, stream=SRStream(0))
    step = _first_difference(a, b)
    assert step is None, (
        f"same identity, same noise: the runs must be bit-identical, diverged at step "
        f"{step} (max abs {_max_abs(a, b):.3e})"
    )


def test_fp32_weights_claim_a_stream_only_because_z_is_bf16():
    """Why ScheduleFree is the exception to a library-wide invariant.

    ``tests/test_sr_seed_checkpoint.py``'s
    ``test_an_optimizer_that_never_rounds_does_not_claim_a_stream_id`` pins "fp32 params ->
    no stream id" for Adakaon, and the 0.7.13 changelog leans on it: a mixed fp32 + bf16
    process is supposed to leave the bf16 optimizer on stream 0. A default ``ScheduleFree``
    over fp32 weights breaks that — its bf16 ``z`` rounds, so it takes an identity and
    pushes a co-resident bf16 optimizer off the compatibility anchor.

    Asserted as CURRENT behaviour, not desired: whoever changes the default ``z`` storage,
    or derives the identity from the owner, has to update this test and the changelog's
    "Known" entry together.
    """
    reseed_stochastic_rounding()
    _, bf16_z = _run(momentum_dtype="bfloat16", reseed=False, steps=1)
    _, fp32_z = _run(momentum_dtype="float32", reseed=False, steps=1)
    assert bf16_z.sr_stream.stream_id == 0, (
        "an fp32-weight ScheduleFree still rounds its bf16 z, so it claims stream 0"
    )
    assert "_sr_meta" in bf16_z.state_dict()
    assert fp32_z.sr_stream.stream_id is None
    assert "_sr_meta" not in fp32_z.state_dict()


# --------------------------------------------------------- known limitation (xfail pin)
_SF_REPRO_SITES = """The fix is one of two things, neither of them owner-local:
  kaon/schedulefree.py:236           the momentum_dtype default "bfloat16" -> "float32"
                                     (wire-visible: changes every run's memory footprint
                                     and trajectory)
  kaon/_stochastic_rounding.py:99    an SRStream identity derived from something stable
                                     about the owner instead of from the first-draw
                                     allocator `_next_stream` (already recorded as the
                                     0.7.13 follow-up; moves the noise of every process
                                     holding two rounding owners)
`_store_z` cannot simply stop rounding: a round-to-nearest bf16 write freezes z outright
(kaon/schedulefree.py:414, `_store_z`). An identity keyed on object lifetime (release-on-GC) is worse
still - it would make the noise depend on when the collector runs."""


@pytest.mark.xfail(strict=True, reason=f"bf16 z rounds by default. {_SF_REPRO_SITES}")
@pytest.mark.parametrize("foreach", [True, False], ids=["foreach", "per_param"])
def test_default_config_is_reproducible_under_manual_seed_alone(foreach):
    """**Desired** behaviour, not current: ``torch.manual_seed(s)`` alone must reproduce a
    default ScheduleFree over fp32 weights, exactly as it does every other kaon optimizer.

    Strict xfail: whoever removes the stream allocator's order sensitivity, or stops
    storing ``z`` in bf16 by default, flips this to XPASS.
    """
    reseed_stochastic_rounding()      # hermetic run boundary, then the bare protocol
    a, _ = _run(foreach=foreach, reseed=False)
    b, _ = _run(foreach=foreach, reseed=False)
    assert _first_difference(a, b) is None
