"""Rakaon: experimental regularized factored preconditioning, without momentum.

Interpolates the factored variance toward its tensor mean before inversion.
This bounds anisotropy without a dense momentum or variance buffer. It is a
research candidate, not a claim of superior diffusion sample quality.
"""
from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

from kaon._backend import SRSeedState, subtract_batched_, subtract_one_
from kaon._momentum_codec import load_state_dict_preserving_dtypes

__all__ = ["Rakaon"]


class Rakaon(SRSeedState, Optimizer):
    """Momentum-free Adafactor with variance shrinkage and RMS update clipping.

    ``shrinkage=0`` uses factored variance; ``1`` uses a tensorwise RMS.
    For an R x C matrix, persistent state is 4*(R+C) bytes. At shrinkage=1
    only one fp32 variance scalar per tensor is needed. ``block_size=64``,
    ``256`` or ``1024`` (experimental, and only with ``shrinkage=1``) stores
    one fp32 variance per contiguous flat block; the final short block uses
    its actual length. Blocks are statistical partitions, not semantic model
    groups. Their updates use one global tensor RMS clip, then the usual
    weight write. Convolutions flatten to [out, rest]. Scalars/vectors use a
    dense second moment when ``shrinkage < 1``.
    Dense fp32 temporaries cover one parameter or bounded batches of matching
    shapes, never a flattened copy of the whole model.
    LR is absolute and constant by default; no hidden schedule or weight swap.
    Save model, optimizer, RNG and data position together for exact resumption.
    """

    def __init__(self, params, lr=1e-3, beta2=0.999, shrinkage=0.1,
                 clip_threshold=1.0, weight_decay=0.0, eps=1e-30,
                 stochastic_rounding=True, block_size=None):
        defaults = dict(lr=lr, beta2=beta2, shrinkage=shrinkage,
                        clip_threshold=clip_threshold, weight_decay=weight_decay,
                        eps=eps, stochastic_rounding=stochastic_rounding, block_size=block_size)
        super().__init__(params, defaults)

    def add_param_group(self, param_group):
        values = self.defaults | param_group
        for key in ("lr", "weight_decay"):
            if not math.isfinite(values[key]) or values[key] < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        for key in ("eps", "clip_threshold"):
            if not math.isfinite(values[key]) or values[key] <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not 0 <= values["beta2"] < 1:
            raise ValueError("beta2 must be in [0, 1)")
        if not 0 <= values["shrinkage"] <= 1:
            raise ValueError("shrinkage must be in [0, 1]")
        block_size = values["block_size"]
        if block_size is not None:
            if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
                raise ValueError("block_size must be a positive integer or None")
            if values["shrinkage"] != 1:
                raise ValueError("block_size requires shrinkage=1")
        super().add_param_group(param_group)

    def load_state_dict(self, state_dict):
        load_state_dict_preserving_dtypes(self, state_dict)
        for group in self.param_groups:
            group.setdefault("block_size", None)

    def _step_blocks(self, group):
        """One variance per contiguous block, global per-tensor update clipping.

        Blocks are statistical partitions, not inferred architectural components.
        Tail blocks use their actual length, so padding does not dilute variance.
        """
        block_size = group["block_size"]
        buckets = {}
        for p in group["params"]:
            if p.grad is None or not p.numel():
                continue
            state = self.state[p]
            if not state:
                state.update(step=0, isotropic=True, block_size=block_size,
                             variance=torch.zeros((p.numel() + block_size - 1) // block_size,
                                                  device=p.device, dtype=torch.float32))
            buckets.setdefault((p.device, p.dtype, p.shape), []).append(p)
        for params in buckets.values():
            elements = params[0].numel()
            # A tensor smaller than one block needs no padding. Requested block
            # size controls partitioning, not an unbounded scratch allocation.
            width = min(block_size, elements)
            blocks = (elements + width - 1) // width
            padded = blocks * width
            chunk_size = max(1, 262144 // padded)
            for offset in range(0, len(params), chunk_size):
                chunk = params[offset:offset + chunk_size]
                states = [self.state[p] for p in chunk]
                for state in states:
                    state["step"] += 1
                g = torch.stack([p.grad.float().reshape(-1) for p in chunk])
                if padded != elements:
                    g = torch.nn.functional.pad(g, (0, padded - elements))
                g = g.reshape(len(chunk), blocks, width)
                counts = g.new_full((blocks,), width)
                counts[-1] = elements - (blocks - 1) * width
                energy = g.square().sum(-1) / counts
                variance = torch.stack([state["variance"] for state in states])
                beta = group["beta2"]
                variance.lerp_(energy + group["eps"], 1 - beta)
                torch._foreach_copy_([state["variance"] for state in states], list(variance.unbind()))
                correction = g.new_tensor([1 - beta ** state["step"] for state in states])
                scale = (variance / correction[:, None]).clamp_min_(group["eps"]).rsqrt_()
                update_rms = (energy * scale.square() * counts).sum(1).div_(elements).sqrt_()
                scale.div_((update_rms / group["clip_threshold"]).clamp_min_(1)[:, None])
                update = (g * scale[:, :, None]).reshape(len(chunk), padded)[:, :elements]
                update = update.reshape(len(chunk), *chunk[0].shape)
                if group["weight_decay"]:
                    update.add_(torch.stack([p.float() for p in chunk]), alpha=group["weight_decay"])
                method = "stochastic_rounding" if group["stochastic_rounding"] else "none"
                subtract_batched_(chunk, update, method, alpha=group["lr"], sr=self.sr_stream)

    def _step_isotropic(self, group):
        # Batch matching tensors to amortize launch overhead on adapter bags.
        # The scratch budget is bounded; one oversized tensor runs alone.
        buckets = {}
        for p in group["params"]:
            if p.grad is None or not p.numel():
                continue
            state = self.state[p]
            if not state:
                state.update(step=0, isotropic=True,
                             variance=torch.zeros((), device=p.device, dtype=torch.float32))
            key = (p.device, p.dtype, p.shape)
            buckets.setdefault(key, []).append(p)
        for params in buckets.values():
            chunk_size = max(1, 262144 // params[0].numel())
            for offset in range(0, len(params), chunk_size):
                chunk = params[offset:offset + chunk_size]
                states = [self.state[p] for p in chunk]
                for state in states:
                    state["step"] += 1
                g = torch.stack([p.grad.float() for p in chunk])
                energy = g.reshape(len(chunk), -1).square().mean(1)
                variance = torch.stack([state["variance"] for state in states])
                beta = group["beta2"]
                variance.lerp_(energy + group["eps"], 1 - beta)
                torch._foreach_copy_([state["variance"] for state in states],
                                     list(variance.unbind()))
                correction = torch.tensor([1 - beta ** state["step"] for state in states],
                                          device=g.device, dtype=torch.float32)
                scale = (variance / correction).clamp_min_(group["eps"]).rsqrt_()
                scale.div_((energy.sqrt() * scale / group["clip_threshold"]).clamp_min_(1))
                update = g * scale.reshape(-1, *([1] * chunk[0].ndim))
                if group["weight_decay"]:
                    update.add_(torch.stack([p.float() for p in chunk]), alpha=group["weight_decay"])
                method = "stochastic_rounding" if group["stochastic_rounding"] else "none"
                subtract_batched_(chunk, update, method, alpha=group["lr"], sr=self.sr_stream)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Reject unsupported inputs before any parameter is updated.
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None and (p.grad.is_sparse or p.is_complex()):
                    raise RuntimeError("Rakaon requires dense real gradients")
                state = self.state.get(p)
                if state and state.get("isotropic", False) != (group["shrinkage"] == 1):
                    raise ValueError("Cannot switch isotropic state layout after initialization")
                if state and state.get("block_size") != group["block_size"]:
                    raise ValueError("Cannot switch block_size after initialization")
        for group in self.param_groups:
            if group["lr"] == 0:
                continue
            if group["block_size"] is not None:
                self._step_blocks(group)
                continue
            if group["shrinkage"] == 1:
                self._step_isotropic(group)
                continue
            for p in group["params"]:
                if p.grad is None or p.numel() == 0 or group["lr"] == 0:
                    continue
                g = p.grad.float()
                matrix = p.ndim >= 2
                g = g.reshape(p.shape[0], -1) if matrix else g
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["isotropic"] = group["shrinkage"] == 1
                    if matrix:
                        state["row"] = torch.zeros(g.shape[0], device=p.device, dtype=torch.float32)
                        state["col"] = torch.zeros(g.shape[1], device=p.device, dtype=torch.float32)
                    else:
                        state["variance"] = torch.zeros_like(g)
                state["step"] += 1
                beta = group["beta2"]
                correction = 1 - beta ** state["step"]
                sq = g.square().add_(group["eps"])
                if matrix:
                    row, col = state["row"], state["col"]
                    row.lerp_(sq.mean(dim=1), 1 - beta)
                    col.lerp_(sq.mean(dim=0), 1 - beta)
                    mean = row.mean().clamp_min(group["eps"])
                    variance = (row / mean).unsqueeze(1) * col.unsqueeze(0)
                else:
                    variance = state["variance"]
                    variance.lerp_(sq, 1 - beta)
                    mean = variance.mean()
                del sq
                s = group["shrinkage"]
                if s in (0, 1):
                    denom = variance / correction
                else:
                    denom = variance.mul(1 - s).add(mean * s).div_(correction)
                update = g / denom.clamp_min_(group["eps"]).sqrt_()
                update.div_((update.square().mean().sqrt() / group["clip_threshold"]).clamp_min_(1))
                update = update.reshape(p.shape)
                if group["weight_decay"]:
                    update.add_(p.float(), alpha=group["weight_decay"])
                update.mul_(group["lr"])
                method = "stochastic_rounding" if group["stochastic_rounding"] else "none"
                subtract_one_(p, update, state, method, sr=self.sr_stream)
        return loss
