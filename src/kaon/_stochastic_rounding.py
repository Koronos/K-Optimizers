"""Stochastic rounding primitive for fp32 -> bf16 weight updates.

When training a model with bf16 parameters, the standard ``p += -lr * update``
truncates the fp32 result to bf16 using round-to-nearest-even. Small updates
that fall below the bf16 ULP of the weight round to zero, and the optimizer
effectively makes no progress on those parameters.

Two well-known mitigations exist:

1. **Kahan summation** — keep a per-parameter compensation buffer that
   accumulates the lost low-order bits across steps. Costs ~2 B/param of
   extra state, equal to the size of the model itself for bf16 weights.
2. **Stochastic rounding** — randomly round up or down with probability
   proportional to the fractional distance. The expected value is the
   exact fp32 result, so updates are preserved *in expectation* without
   any extra state.

This module implements (2): ``add_stochastic_(target, source, alpha)``.

The bf16 implementation uses the integer bit-manipulation trick from
``lodestone-rock/torchastic`` and ``AmericanPresidentJimmyCarter/adamw-bf16``:
add uniform noise in ``[0, 2**16)`` to the int32 view of the fp32 result,
then truncate to the upper 16 bits. The expected value of the rounding
step equals the unbiased rounding of the input.

For fp16 targets (different exponent layout) the bit trick does not apply
directly; ``NotImplementedError`` is raised — fall back to ``bf16_method='kahan'``.

**RNG isolation.** Noise is drawn from a per-device :class:`torch.Generator`
owned by this module (not the global CUDA/CPU RNG). The generator is seeded
from the global initial seed on first use and re-seeded whenever
:func:`torch.manual_seed` / :func:`torch.cuda.manual_seed_all` changes that
seed, so ``torch.manual_seed`` before each run still yields reproducible
training. Subsequent ``torch.rand`` calls are unaffected by stochastic-rounding
steps. Limitation: re-seeding to the *same* value inside one process (with no
different seed in between) is not observable through the global RNG, so the
module generator keeps its stream; call :func:`kaon.reseed_stochastic_rounding` (alias of :func:`reseed_generators`) after
``torch.manual_seed`` in that case (test suites, sweeps in one process).

**Per-owner noise streams: :class:`SRStream`.** The module generators above are the
*fallback*, for a caller that hands over no stream. What an optimizer's bf16 weight write
actually draws from is its OWN :class:`SRStream`, which holds one position per SR path:
a launch **counter** for the Triton kernel (``kaon._fused_triton.sr_add_``, whose
``tl.rand`` needs an integer seed, not a generator object) and a private running
:class:`torch.Generator` per device for the reference path here.

Both positions are *optimizer state*: an owner that restarts them at 0 on resume applies a
different noise sequence than the run it continues, so a resumed run's weights diverge from
the continuous run's even with every state tensor restored bit-exactly. Every kaon optimizer
therefore saves its stream in its ``state_dict`` (``_sr_meta``) and restores it on load —
see :class:`kaon._backend.SRSeedState`. Until 0.7.13 the kernel counter was a single
process-global, per-device one that no checkpoint saved, and no bf16 run on the native path
resumed bit-identically; the generators here were (and, for un-threaded callers, remain)
process-wide too.

:func:`reseed_generators` resets the generators here **and** every live
:class:`SRStream`, without needing a registry of live optimizers: it bumps a module
epoch that each stream compares on use (the same compare-on-use trick
:func:`_device_generator` uses for the global seed). It also restarts the stream-id
allocator, so a fresh optimizer built after the call gets stream 0 again and a rerun
inside one process reproduces its noise — which is what ``torch.manual_seed(s)`` +
this call promises.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

__all__ = ["SRStream", "add_stochastic_", "reseed_generators"]

# Per-device RNG — isolated from the global stream so SR does not perturb dataloader/dropout.
# Value is (generator, global_initial_seed at last sync). Used by any ``add_stochastic_``
# caller that does NOT hand over an :class:`SRStream` (SAM's climb, external callers).
_generators: dict[torch.device, tuple[torch.Generator, int]] = {}

# Odd Weyl increment (golden ratio * 2**32): consecutive draws land far apart in the
# Philox/MT stream, so two buckets rounded back to back never share lane noise.
_GOLDEN = 0x9E3779B1
# The Triton kernel takes the seed as a non-negative int32.
_SEED_MASK = 0x7FFFFFFF

# Process-wide stream bookkeeping, both reset by :func:`reseed_generators`:
#
# * ``_reseed_epoch`` — bumped on reseed. Every :class:`SRStream` compares it on use and
#   restarts its counter when it moved. That is how ONE call resets streams owned by
#   optimizers this module has no reference to (and streams created later), with no
#   registry, no weakrefs and no per-step cost.
# * ``_next_stream`` — the stream-id allocator, doubling as a **watermark**: an id is
#   handed out at a stream's first draw, and :meth:`SRStream.restore` pushes the allocator
#   past any id it adopts, so an owner that draws for the first time only AFTER a resume
#   cannot collide with a restored one. Restarting it is what makes ``torch.manual_seed(s)``
#   + reseed reproducible for a *freshly built* optimizer: it gets stream 0 again, so the
#   second run of a two-run comparison draws the first run's noise (the foreach-vs-per-param
#   and resume tests all rest on this).
_reseed_epoch = 0
_next_stream = 0


def _global_initial_seed(device: torch.device) -> int:
    """Initial seed of the global RNG for ``device`` (multi-GPU safe on CUDA)."""
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return torch.cuda.default_generators[idx].initial_seed()
    return torch.initial_seed()


def reseed_generators() -> None:
    """Reset EVERY kaon stochastic-rounding noise stream to follow the global RNG again.

    Needed only when ``torch.manual_seed`` is called again with the *same* seed in one
    process; a seed change is picked up automatically.

    "Every" means all three things kaon's SR noise can come from: this module's
    per-device generators, every live :class:`SRStream` (via the epoch — including the
    Triton weight-write kernel's process-wide fallback stream, which is what the bf16
    write uses on CUDA when the caller threads no stream of its own), and the stream-id
    allocator, so an optimizer built *after* this call reproduces stream 0's noise.
    Resetting only the generators here left ``torch.manual_seed(s)`` + this call
    NON-reproducible for any bf16 run on a Triton build.

    Call it BEFORE ``load_state_dict``, not after: a load restores the checkpoint's
    stream position, and this call would then throw it away.
    """
    global _reseed_epoch, _next_stream
    _generators.clear()
    _reseed_epoch += 1
    _next_stream = 0


def _mix32(x: int) -> int:
    """32-bit avalanche (Murmur3 finalizer). Pure Python int math: no tensor op, no sync,
    and it runs once per device per stream (the result is cached), never per draw."""
    x &= 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x ^= x >> 16
    return x


def _stream_offset(stream_id: int, device_key: int) -> int:
    """Additive separation between one stream/device pair and every other one.

    The seed of draw ``k`` is ``base + offset + k * _GOLDEN`` (mod 2**31), so an offset is
    just a starting point in the Weyl sequence. Two distinct pairs get avalanche-mixed,
    effectively unrelated offsets, which is what keeps their noise **independent**: before
    0.7.13 one global counter was shared by every optimizer and every device (and with
    ``manual_seed_all`` giving all CUDA devices the same base, two devices drew the *same*
    sequence over their different shards).

    ``stream_id == 0`` on ``cuda:0`` (or on the CPU) returns 0 **on purpose**: that is the
    compatibility anchor. A single-optimizer, single-device run — the overwhelmingly common
    case, and the one the whole test suite pins — keeps 0.7.12's ``base + k * _GOLDEN``
    kernel sequence bit-for-bit, and seeds its torch-path generator with ``base`` exactly as
    the shared :func:`_device_generator` did, so this change is observable only where the
    old design was actually broken (two stream owners in one process: two optimizers, or a
    wrapper plus its inner). CPU and ``cuda:0`` share the anchor: the kernel and the torch
    generator are different RNGs, so a stream that somehow wrote to both would not correlate
    anything by starting them at the same offset.

    Mixed offsets separate streams probabilistically rather than by construction: two
    streams reuse seeds only if their offsets happen to differ by ``j * _GOLDEN`` for a
    ``j`` below their draw counts (~1e-3 per pair at a million draws each). A partition of
    the 31-bit seed space into fixed strides would make that impossible, but only by
    capping the number of streams AND the draws each may take, and one bucket write per
    step burns draws fast. Unbiasedness — the only property SR is relied on for — does not
    depend on either choice.
    """
    if stream_id == 0 and device_key <= 0:
        return 0
    return _mix32(((stream_id + 1) * _GOLDEN) ^ ((device_key + 2) * 0x85EBCA6B))


class SRStream:
    """One owner's stochastic-rounding noise position — checkpointable optimizer state.

    A stream is an **identity** (``stream_id``) plus one position per SR path, because the
    two paths take their noise differently and each needs its own restorable position:

    * the **Triton kernel** is seeded once per launch, so its position is a *counter*
      (``draws``): launch ``k`` gets ``base + offset(stream_id, device) + k * _GOLDEN``
      (mod 2**31), where ``base`` is the global RNG's initial seed for that device;
    * the **torch reference** path (CPU, ``kaon._backend.SR_TRITON = False``, fp16, a
      strided view the kernel cannot index) draws from a private :class:`torch.Generator`
      per device, seeded ``base + offset(...)``, whose *state* is the position. Keeping one
      running generator rather than reseeding it per call is what preserves the reference
      path's ``foreach == per-param`` noise identity: a stacked draw of ``N*M`` numbers and
      ``N`` per-param draws of ``M`` consume the same sequence, so both paths round a bucket
      identically. (The kernel path never had that property — a stacked launch and ``N``
      per-param launches get different seeds — which is why the invariant is pinned on the
      reference path.)

    Both positions are checkpointed, so a resume reproduces either path and a run may
    cross between them.

    Why per owner rather than one per process: the counter has to be restorable from a
    checkpoint, and a process-global one cannot be. Restoring it from optimizer A's
    ``state_dict`` would rewind optimizer B's noise; not restoring it leaves every bf16
    run non-reproducible across a resume (measured ~4.7e-2 max abs on bf16 weights four
    steps after one). One stream per owner makes both resumes exact at once, and
    ``stream_id`` keeps their noise independent. Lookahead's ``phi`` sync counts as its own
    owner: it writes through the same shared kernel as the inner optimizer's weight write,
    so it needs its own position in the noise space.

    Cost per draw: one epoch compare, one dict hit, one bound ``initial_seed()`` call and
    three integer ops — measured level with 0.7.12's global counter (see CHANGELOG), with
    no extra kernel and no sync. Getting there is why the per-device cache is keyed by the
    ``torch.device`` **object** and holds a bound seed reader: ``device.type`` alone costs
    ~0.7 us (it builds a Python string), which at one draw per weight on the per-parameter
    path of a 428-adapter bag would have been ~0.3 ms/step of pure bookkeeping.

    ``stream_id`` comes from a process-wide allocator that :func:`reseed_generators`
    restarts, is claimed at the stream's first draw, and is carried in the checkpoint so a
    resume reproduces the identity even if the new process builds its optimizers in a
    different order. An owner that draws for the first time only *after* a resume has no
    id in the checkpoint and takes the next free one — the allocator doubles as a watermark
    that :meth:`restore` pushes past every id it adopts, so that "next free" is the same id
    the continuous run handed out.

    **Known limitation of that identity.** An id is handed out in order of *first draw*, so
    the noise a given optimizer receives depends on how many other streams drew before it.
    A checkpointed run is safe (the id travels with it), but a *new* run's trajectory moves
    if the draw order changes: reordering optimizer construction, adding a second optimizer,
    or running something that builds and steps optimizers first — :func:`kaon.tune` does
    exactly that — shifts the ids. It is unbiased either way and reproducible for a fixed
    program, which is the same guarantee 0.7.12's global counter gave (its noise depended on
    the *interleaving* of every optimizer's writes, so it was strictly more order-sensitive).
    Removing the order sensitivity needs an identity derived from something stable about the
    owner rather than from allocation order; recorded as a follow-up in the CHANGELOG.
    """

    __slots__ = ("_bound", "_epoch", "_generators", "_pending_gen", "_pinned_id",
                 "draws", "stream_id")

    def __init__(self, stream_id: int | None = None) -> None:
        # ``None`` = unclaimed. The id is taken from the allocator at the first REAL draw,
        # not here and not at first access, so an optimizer that never rounds (fp32 params,
        # ``bf16_method="kahan"``, Adakaon's fused path) does not consume one: a run holding
        # an fp32 group and a bf16 group — routine in diffusion — would otherwise push the
        # bf16 optimizer off stream 0 and off the 0.7.12 sequence for nothing. It also keeps
        # such an optimizer's ``state_dict`` free of an ``_sr_meta`` key.
        #
        # An id passed in explicitly is PINNED: it survives :meth:`reset`, where an
        # allocated one is released. Only the Triton module's process-wide fallback stream
        # uses that, and it has to stay on 0 for the lifetime of the process — that is the
        # id whose sequence reproduces 0.7.12's global counter.
        self._pinned_id = stream_id if stream_id is None else int(stream_id)
        self.stream_id = self._pinned_id
        self.draws = 0
        self._epoch = _reseed_epoch
        # torch.device -> (global initial seed bound to, kernel offset, seed reader, key)
        self._bound: dict[torch.device, tuple[int, int, Any, int]] = {}
        self._generators: dict[int, torch.Generator] = {}      # device key -> torch path RNG
        self._pending_gen: dict[int, Tensor] = {}              # restored generator states

    # ------------------------------------------------------------------- noise
    def next_seed(self, device: torch.device) -> int:
        """Consume one kernel launch's worth of noise and return its seed."""
        if self._epoch != _reseed_epoch:      # kaon.reseed_stochastic_rounding() happened
            self.reset()
        entry = self._bound.get(device)
        if entry is None:
            entry = self._bind(device)
        elif entry[2]() != entry[0]:          # torch.manual_seed moved under us
            entry = self._bind(device, restart=True)
        self.draws += 1
        return (entry[1] + self.draws * _GOLDEN) & _SEED_MASK

    def _bind(self, device: torch.device, *, restart: bool = False) -> tuple[int, int, Any, int]:
        """Cache this stream's noise offset on ``device``. Cold: once per device.

        ``restart`` means ``torch.manual_seed`` was called again with a DIFFERENT value,
        i.e. a new run, so the positions restart — exactly as 0.7.12's global counter and
        shared generators did (``torch.manual_seed(s)`` at the top of a run has to reproduce
        the sequence without any extra call).
        """
        global _next_stream
        if restart:
            self.draws = 0
            self._bound.clear()
            self._generators.clear()
        if self.stream_id is None:            # first real draw: claim an identity
            self.stream_id = _next_stream
            _next_stream += 1
        if device.type == "cuda":
            key = device.index if device.index is not None else torch.cuda.current_device()
            read_seed = torch.cuda.default_generators[key].initial_seed
        else:
            key, read_seed = -1, torch.initial_seed
        base = read_seed()
        entry = (base, (base + _stream_offset(self.stream_id, key)) & 0xFFFFFFFF, read_seed, key)
        self._bound[device] = entry
        return entry

    def generator(self, device: torch.device) -> torch.Generator:
        """This stream's private generator for the torch reference path.

        One running generator per device, seeded from ``base + offset`` on first use (so
        stream 0 seeds exactly as the shared :func:`_device_generator` did) and then left
        to run: its sequence, not a per-call reseed, is what keeps a stacked SR write and
        the equivalent per-param writes drawing the same numbers. Its state is what the
        checkpoint carries for this path.
        """
        if self._epoch != _reseed_epoch:
            self.reset()
        entry = self._bound.get(device)
        if entry is None:
            entry = self._bind(device)
        elif entry[2]() != entry[0]:
            entry = self._bind(device, restart=True)
        base, _offset, _read, key = entry
        gen = self._generators.get(key)
        if gen is None:
            gen = self._generators[key] = torch.Generator(device=device)
            gen.manual_seed((base + _stream_offset(self.stream_id, key)) & 0xFFFFFFFFFFFFFFFF)
            saved = self._pending_gen.pop(key, None)
            if saved is not None:      # a resume: continue the checkpointed sequence
                try:
                    gen.set_state(saved)
                except RuntimeError as exc:   # right dtype, wrong length for this RNG kind
                    raise ValueError(
                        "checkpoint's stochastic-rounding generator state does not fit "
                        f"this device's generator (device key {key}): {exc}"
                    ) from exc
        return gen

    # ------------------------------------------------------------------- state
    def reset(self) -> None:
        """Restart at draw 0, re-sync to the global seed, and **release the identity**.

        Releasing is what keeps the identities unique across a reseed:
        :func:`reseed_generators` restarts the allocator, so a stream that hung on to its
        old id would collide with the next stream to draw — two owners on one seed
        sequence, which is precisely the pre-0.7.13 defect. Every live stream re-claims on
        its next draw instead, in draw order, which is the order a clean run would have
        assigned anyway (the reseed is a run boundary; the allocator is back at 0).

        Like the rest of a reseed this lands lazily — a released ``stream_id`` is still
        readable on a stream that has not been touched since — but it can never be *used*:
        every draw and every :meth:`snapshot` applies the pending reset first. A stream
        constructed with an explicit id keeps it (``_pinned_id``): the Triton module's
        process-wide fallback must stay on stream 0 for the whole process.

        Only :func:`reseed_generators` gets here. A mid-process ``torch.manual_seed`` with
        a DIFFERENT value also restarts the positions (:meth:`_bind` with ``restart``) but
        deliberately keeps the identities: it does not touch the allocator, so re-claiming
        there would hand out fresh high ids and move the run off the compatibility anchor
        for no reason.
        """
        self.stream_id = self._pinned_id
        self.draws = 0
        self._epoch = _reseed_epoch
        self._bound.clear()
        self._generators.clear()
        self._pending_gen.clear()

    def _base_moved(self) -> bool:
        """Did the global seed change under this stream since its last draw?"""
        return any(read_seed() != base for base, _off, read_seed, _key in self._bound.values())

    def snapshot(self) -> dict[str, Any] | None:
        """What a checkpoint needs: the identity, the kernel counter, and — only if the
        torch reference path was actually used — that path's generator state per device
        (``"gen"``, ~5 KB per CPU device, 16 B per CUDA one; absent on the Triton path).

        ``None`` when the stream never drew: there is no position to restore, and a
        checkpoint key would only imply one.

        A pending restart (a reseed, or a changed global seed — both applied lazily on the
        next draw so they cost nothing per step) is forced to land FIRST. Otherwise the
        checkpoint would record a position the live run is about to abandon, and the resume
        would continue from a draw the saving run never reached.
        """
        if self._epoch != _reseed_epoch or self._base_moved():
            self.reset()
        if self.stream_id is None:
            return None
        meta: dict[str, Any] = {"stream": self.stream_id, "draws": self.draws}
        if self._generators or self._pending_gen:
            # str keys: an int/str round trip through JSON must not lose the device.
            # Staged-but-not-yet-applied states are carried through, so checkpointing
            # right after a resume (before the first step) does not drop the position a
            # live generator would still have been holding.
            gen = {str(k): state for k, state in self._pending_gen.items()}
            gen.update({str(k): g.get_state().clone() for k, g in self._generators.items()})
            meta["gen"] = gen
        return meta

    def restore(self, meta: dict[str, Any] | None) -> None:
        """Adopt a checkpointed position. ``None`` (a pre-0.7.13 checkpoint) resets.

        Back-filling a **reset** stream rather than leaving the live position where it is,
        is the conservative choice: an old checkpoint carries no position, and draw 0 is
        the position the run that wrote it started from — also exactly what an optimizer
        that never SR-wrote would have.

        The generator states are staged, not applied: the generators are rebuilt (and
        seeded from the *live* global seed) on first use, and the staged state is set on
        the one that is actually reached, so a checkpoint written on a device this process
        does not have costs nothing. They are also **normalised here**, which is load-bearing:
        ``Generator.set_state`` accepts only a CPU byte tensor, while a resume almost always
        reads its checkpoint with ``map_location`` pointing at the training device — that
        moves the state to CUDA and the staged-then-applied design made it blow up on the
        first ``step()`` after a load that looked fine. Validating the payload here also
        turns a corrupt blob into a checkpoint error at load time instead of a ``TypeError``
        from inside the optimizer.

        Restoring an id also pushes the allocator's **watermark** past it. Without that, an
        owner that had not drawn yet when the checkpoint was taken (so it carried no
        position at all) would claim a fresh ``0`` on the resume and collide with the owner
        restored *onto* 0 — two owners on one seed sequence. That is reachable with ordinary
        settings: ``Lookahead(k=6)`` checkpointed at step 4 syncs for the first time at step
        6 (measured 1.56e-2 divergence), and so does a second optimizer that starts stepping
        only after the resume (3.13e-2). With the watermark, a first-time drawer takes the
        next free id — the same one the continuous run gave it.
        """
        global _next_stream
        if not meta:
            self.reset()
            return
        stream_id = int(meta.get("stream", 0))
        draws = int(meta.get("draws", 0))
        if stream_id < 0 or draws < 0:
            raise ValueError(
                "checkpoint has an invalid stochastic-rounding noise stream "
                f"(stream={stream_id}, draws={draws})"
            )
        pending: dict[int, Tensor] = {}
        for key, state in (meta.get("gen") or {}).items():
            if not torch.is_tensor(state) or state.dtype != torch.uint8 or state.ndim != 1:
                raise ValueError(
                    "checkpoint has an invalid stochastic-rounding generator state for "
                    f"device {key!r}: expected a 1-D uint8 tensor, got "
                    f"{type(state).__name__}"
                    + (f" {tuple(state.shape)} {state.dtype}" if torch.is_tensor(state) else "")
                )
            # ``.cpu()`` is what makes ``map_location=<cuda device>`` survivable.
            pending[int(key)] = state.detach().cpu()
        self.stream_id = stream_id
        _next_stream = max(_next_stream, stream_id + 1)   # see the watermark note above
        self.draws = draws
        self._epoch = _reseed_epoch
        self._bound.clear()
        self._generators.clear()
        self._pending_gen = pending


def _device_generator(device: torch.device) -> torch.Generator:
    """Return the module-owned generator for ``device``, synced to the global seed."""
    seed = _global_initial_seed(device)
    entry = _generators.get(device)
    if entry is None or entry[1] != seed:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        _generators[device] = (gen, seed)
    return _generators[device][0]


@torch.no_grad()
def add_stochastic_(
    target: Tensor, source: Tensor, alpha: float = 1.0, sr: SRStream | None = None
) -> None:
    """In-place ``target += alpha * source`` with stochastic rounding.

    The rounding step happens on the final cast back to ``target.dtype``.
    For fp32 targets this is just a plain ``target.add_(source, alpha=alpha)``
    — there is no precision loss to compensate. For bf16 targets the
    integer bit trick (see module docstring) is used.

    Args:
        target: Destination tensor, modified in place.
        source: Tensor of the same shape as ``target``. Cast to fp32
            internally if not already.
        alpha: Scalar multiplier applied to ``source`` before adding.
        sr: the caller's :class:`SRStream`. Passing one makes the draw part of
            *checkpointed* state (so a resume reproduces it) and independent of every
            other owner's; ``None`` falls back to this module's shared per-device
            generator, whose position no checkpoint saves.

    Raises:
        NotImplementedError: For ``target.dtype`` other than bf16 or fp32.
    """
    if target.dtype == torch.float32:
        target.add_(source, alpha=alpha)
        return
    if target.dtype == torch.bfloat16:
        _add_stochastic_bf16_(target, source, alpha, sr)
        return
    raise NotImplementedError(
        f"add_stochastic_ for target dtype {target.dtype} is not implemented; "
        "currently only torch.bfloat16 and torch.float32 are supported "
        "(use bf16_method='kahan' for fp16 parameters)"
    )


@torch.no_grad()
def _add_stochastic_bf16_(
    target_bf16: Tensor,
    source: Tensor,
    alpha: float,
    sr: SRStream | None = None,
) -> None:
    source_fp32 = source if source.dtype == torch.float32 else source.float()

    # Compute the exact fp32 result of the addition.
    result_fp32 = target_bf16.float()
    result_fp32.add_(source_fp32, alpha=alpha)

    # Stochastic-round to bf16 via the int32 bit trick.
    # bf16 keeps the top 16 bits of an fp32 representation; the lower 16
    # are dropped on a normal cast. Adding uniform noise in [0, 2^16) to
    # those lower bits and then truncating makes the upper-bit "round up"
    # event happen with probability equal to the fractional distance,
    # which is exactly unbiased stochastic rounding.
    bits = result_fp32.view(torch.int32)
    noise = torch.randint(
        low=0,
        high=0x10000,
        size=bits.shape,
        dtype=torch.int32,
        device=bits.device,
        generator=_device_generator(bits.device) if sr is None else sr.generator(bits.device),
    )
    # Only NaN needs masking: CUDA canonicalizes fp32 NaN to 0x7FFFFFFF (and
    # -NaN to 0xFFFFFFFF); adding noise overflows to 0x8000xxxx and the AND
    # yields -0.0. ±inf and finites above bf16 max are already safe (inf noise
    # stays in-range; large finites may stochastically round to inf like RNE).
    # One extra kernel: fold the NaN test into the add operand. +-inf need no mask
    # (0x7F800000 + noise <= 0x7F80FFFF and the AND below restores it). Measured
    # cheaper than isfinite() + mul_ (two kernels) on CUDA and CPU.
    noise = torch.where(result_fp32 != result_fp32, 0, noise)
    # In two's complement, ``-0x10000`` is the int32 mask ``0xFFFF0000``.
    bits.add_(noise).bitwise_and_(-0x10000)

    # Lower 16 bits are now zero, so the bf16 cast is exact. copy_ fuses the
    # dtype conversion into the destination without a full-sized bf16 temp.
    # We benchmarked copying the top-16-bit lanes via an int16 view to skip
    # this convert; the strided copy is slower on CUDA than the fused contiguous
    # cast, so copy_ wins. A fused Triton kernel for the whole add+round remains
    # the real optimization — see CHANGELOG.
    target_bf16.copy_(result_fp32)
