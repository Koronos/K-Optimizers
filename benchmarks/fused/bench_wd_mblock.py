"""A/B harness for the 0.7.12 Adakaon `cautious_wd` + runtime-`m_block` batch.

Two questions, both measured rather than argued:

1. **``momentum_4bit_block`` as a runtime scalar.** The one-block tile kernel used to
   hardcode ``BLK = min(R*C, 128)``, so every other block size was DIVERTED to the native
   path by a routing guard (correct, but it gave up the fused route entirely). The block is
   now a runtime kernel argument and :class:`~kaon._fused_triton.PointerArrayCache` buckets
   by it. ``mblock`` measures what that route is worth: arm A is the OLD behaviour, produced
   in the same process by rewriting the optimizer's cached routing partition so the same
   tensors go native; arm B is the new fused route. Same build, same params, same state
   shapes — only the route differs.

2. **The JIT surface both changes add.** ``cautious_wd`` adds one ``WDFALL``-style constexpr
   (``WDFULL``) to ~13 kernels and the block adds one runtime int to the tile kernel.
   Triton keys its cache on constexprs and on an int argument's ``==1`` / ``%16==0``
   specialization, so the question is whether the number of COMPILED VARIANTS grows.
   ``jit`` counts the distinct compiled kernels a fixed workload produces
   (``triton.runtime.jit.JITFunction.cache``) for the default config and for the new knobs.

Statistics follow ``bench_perf_a2.py``: a PAIRED design (arms adjacent, order alternating,
geometric mean of the log ratio with a 95% CI), because the card is shared and a
``min``-of-N would report the contention rather than the effect. Launch counts are
contention-immune and are the primary evidence for the routing change.

Usage::

    python benchmarks/fused/bench_wd_mblock.py --case all
    python benchmarks/fused/bench_wd_mblock.py --case mblock --reps 300
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import torch

from kaon import Adakaon

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reuse the batch's statistics/profiling helpers rather than forking them.
A2 = _load("bench_perf_a2", f"{HERE}/bench_perf_a2.py")
bag, show, launches, peak_mb = A2.bag, A2.show, A2.launches, A2.peak_mb

DEV = "cuda"
CASES: dict = {}


def case(name: str):
    def deco(fn):
        CASES[name] = fn
        return fn
    return deco


def _force_native(opt: Adakaon) -> None:
    """Rewrite the cached routing partition so the one-block subset goes NATIVE instead.

    This reproduces the pre-0.7.12 behaviour for a non-128 ``momentum_4bit_block`` exactly:
    the guard in ``_fused_partition`` used to append those tensors to ``native``. The
    partition is keyed on the param witness and returned verbatim while nothing moves, so
    one rewrite after a warm step holds for the rest of the run.
    """
    for gid, part in list(opt._fused_part.items()):
        # The memo's LEADING fields are cache keys and must be written back untouched:
        # ``(witness, state_generation, *routes)`` since 0.7.14. Slice the four route lists
        # off the END and rebuild the whole entry — unpacking a fixed arity here raised
        # ``ValueError: too many values to unpack`` the moment the state-identity field was
        # added, and rewriting a short tuple would have made ``_fused_partition`` compare
        # its generation against a route LIST.
        head, (one_block, big, one_dim, native) = part[:-4], part[-4:]
        opt._fused_part[gid] = (*head, [], big, one_dim, native + one_block)


def _warm(opt: Adakaon) -> None:
    opt.step()


# ----------------------------------------------------------------- cases
@case("mblock")
def _mblock(args: argparse.Namespace) -> None:
    """Non-default 4-bit block: native degradation (old) vs the runtime-block tile kernel."""
    print("\n## momentum_4bit_block != 128 on one-block shapes   [native (old) / fused (new)]")
    shapes = [(64, 128)] * 200 + [(32, 64)] * 100
    for block in (64, 256, 0):
        for dt in (torch.float32, torch.bfloat16):
            ps = bag(shapes, dt, seed=block)
            common = dict(lr=1e-3, weight_decay=0.01, fused=True, momentum_dtype="4bit",
                          momentum_4bit_block=block, betas=(0.9, 0.999))
            old, new = Adakaon(ps, **common), Adakaon(ps, **common)
            _warm(old)
            _force_native(old)     # the pre-0.7.12 route for this block size
            _warm(old)
            _warm(new)
            print(f"  {len(shapes)} tensors, block={block}, {dt}")
            print(f"    {'native(old)':>12}: {launches(old.step):5d} launches "
                  f"{peak_mb(old.step):8.2f} MiB peak")
            print(f"    {'fused(new)':>12}: {launches(new.step):5d} launches "
                  f"{peak_mb(new.step):8.2f} MiB peak")
            show(f"block={block} {dt} (old/new)", old.step, new.step, args.reps)


@case("mblock_worst")
def _mblock_worst(args: argparse.Namespace) -> None:
    """The runtime block's worst case: a PADDED tile (no ``EXACT`` single-reduction) with a
    small block, i.e. ``requant_4bit``'s general ``for b in range(NB)`` loop running O(numel *
    NB). If the new route ever loses to the old native degradation it is here — a (60,120)
    weight at ``momentum_4bit_block=8`` is 900 blocks over a 64x128 tile.
    """
    print("\n## runtime block, worst case: padded tile x small block   [native (old) / fused (new)]")
    for shapes, label in (([(60, 120)] * 200, "(60,120) padded -> 64x128"),
                          ([(64, 128)] * 200, "(64,128) exact tile")):
        for block in (128, 64, 16, 8):
            ps = bag(shapes, torch.float32, seed=block)
            common = dict(lr=1e-3, weight_decay=0.01, fused=True, momentum_dtype="4bit",
                          momentum_4bit_block=block, betas=(0.9, 0.999))
            old, new = Adakaon(ps, **common), Adakaon(ps, **common)
            _warm(old)
            _force_native(old)
            _warm(old)
            _warm(new)
            bk = new._fused_ob_caches[id(new.param_groups[0])].buckets[0]
            nb = new.state[ps[0]]["m_scale"].numel()
            show(f"{label} block={block} NB={nb} exact4={bk['exact4']}",
                 old.step, new.step, args.reps)


@case("wd")
def _wd(args: argparse.Namespace) -> None:
    """``cautious_wd="full"`` vs ``"masked"``: the placement must not cost throughput."""
    print("\n## cautious_wd placement   [masked / full]")
    for dt in (torch.float32, torch.bfloat16):
        ps = A2.realistic_bag(dt)
        common = dict(lr=1e-3, weight_decay=0.01, fused=True, cautious=True,
                      betas=(0.9, 0.999))
        masked = Adakaon(ps, cautious_wd="masked", **common)
        full = Adakaon(ps, cautious_wd="full", **common)
        _warm(masked)
        _warm(full)
        print(f"  realistic mixed bag, {dt}")
        print(f"    {'masked':>8}: {launches(masked.step):4d} launches")
        print(f"    {'full':>8}: {launches(full.step):4d} launches")
        show(f"cautious_wd masked/full {dt}", masked.step, full.step, args.reps)


@case("jit")
def _jit(args: argparse.Namespace) -> None:
    """Compiled Triton variants — the JIT surface the 0.7.12 knobs add.

    A variant is one entry in a ``JITFunction``'s per-device cache: Triton keys it on the
    constexprs plus each int argument's ``== 1`` / ``% 16 == 0`` specialization.

    Three sections, in order of what they answer:

    1. ``BASELINE`` — the total for a workload that exercises every Adakaon fused route on
       every momentum storage and both param dtypes. **This is the number to compare across
       builds**: run it with a clean ``TRITON_CACHE_DIR`` on the old and the new code and the
       totals must match, which is what "the defaults cost no extra JIT" means.
    2. ``EXACT tiles`` — one distinct ``momentum_4bit_block`` per row on an unpadded
       power-of-two tile. Each costs +1 tile-kernel variant, because ``requant_4bit``'s
       single-reduction path needs ``FBLK`` as a constexpr for its ``(nb, FBLK)`` reshape.
    3. ``padded tiles`` — the same block sizes on a tile that needs padding (``EXACT`` off,
       ``FBLK == 0``). They all share ONE variant: that is the runtime ``m_blk`` argument
       doing its job — the kernel body does not specialize per block size.
    """
    import triton

    import kaon._fused_triton as ft

    def variants() -> dict[str, int]:
        """Compiled variants per Adakaon kernel, read off the per-device JIT caches.

        Triton 3.x keeps ``JITFunction.device_caches[dev] = (kernel_cache, ..., ...)``; the
        first slot maps one signature+specialization key to one compiled kernel.
        """
        out = {}
        for name in dir(ft):
            obj = getattr(ft, name)
            if not isinstance(obj, triton.runtime.jit.JITFunction):
                continue
            n = sum(len(dc[0]) for dc in getattr(obj, "device_caches", {}).values())
            if n:
                out[name] = n
        return out

    def step_once(shapes, dtype, **kw):
        ps = bag(shapes, dtype, seed=7)
        Adakaon(ps, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), fused=True, **kw).step()

    # 1. every route x every momentum storage x {fp32, bf16} params.
    routes = [(64, 128)] * 4 + [(512,)] * 4 + [()] * 2 + [(1024, 512)] * 2
    print("\n## JIT surface")
    for md in ("bfloat16", "float32", "int8", "4bit"):
        for dtype in (torch.float32, torch.bfloat16):
            step_once(routes, dtype, momentum_dtype=md)
    base = variants()
    print(f"  BASELINE (all routes x all momentum dtypes x fp32/bf16 params): "
          f"{sum(base.values())} variants across {len(base)} kernels")
    seen = base
    for label, shapes in (("EXACT tiles   (64,128)", [(64, 128)] * 4),
                          ("padded tiles  (60,120)", [(60, 120)] * 4)):
        for block in (128, 64, 256, 0, 32):
            step_once(shapes, torch.float32, momentum_dtype="4bit", momentum_4bit_block=block)
            cur = variants()
            added = {k: v - seen.get(k, 0) for k, v in cur.items() if v - seen.get(k, 0)}
            print(f"  {label}  block={block:<4} total={sum(cur.values()):3d}  new: {added or '-'}")
            seen = cur
    for arm in ("masked", "full"):
        step_once(routes, torch.float32, momentum_dtype="float32", cautious_wd=arm)
        cur = variants()
        added = {k: v - seen.get(k, 0) for k, v in cur.items() if v - seen.get(k, 0)}
        print(f"  cautious_wd={arm!r:<10}                     total={sum(cur.values()):3d}  "
              f"new: {added or '-'}")
        seen = cur


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all", help=f"one of {sorted(CASES)} or 'all'")
    ap.add_argument("--reps", type=int, default=150)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("CUDA required")
        sys.exit(1)
    names = sorted(CASES) if args.case == "all" else args.case.split(",")
    for n in names:
        CASES[n](args)


if __name__ == "__main__":
    main()
