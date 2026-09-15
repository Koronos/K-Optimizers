"""Experimental paired Gram SGD reference; not a recommended training optimizer.

Each parameter group must explicitly contain [A, B] for the update B @ A.
Absolute Tikhonov damping supports zero-B initialization but sacrifices exact
scale invariance. No momentum, clipping, decay or model-name inference is added.
"""
import math

import torch
from torch.optim import Optimizer

from kaon._backend import SRSeedState, subtract_one_
from kaon._momentum_codec import load_state_dict_preserving_dtypes


class PairedGram(SRSeedState, Optimizer):
    def __init__(self, params, lr=1e-3, damping=1e-3, stochastic_rounding=True):
        super().__init__(params, dict(lr=lr, damping=damping,
                                     stochastic_rounding=stochastic_rounding))

    def add_param_group(self, param_group):
        values = self.defaults | param_group
        if not math.isfinite(values["lr"]) or values["lr"] < 0:
            raise ValueError("lr must be finite and nonnegative")
        if not math.isfinite(values["damping"]) or values["damping"] <= 0:
            raise ValueError("damping must be finite and positive")
        param_group = dict(param_group)
        param_group["params"] = list(param_group["params"])
        self._pair(param_group)
        super().add_param_group(param_group)

    @staticmethod
    def _pair(group):
        if len(group["params"]) != 2:
            raise ValueError("each group requires the explicit ordered pair [A, B]")
        a, b = group["params"]
        if a is b or a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError("expected A[r,in] and B[out,r]")
        if not a.numel() or not b.numel() or a.device != b.device or a.dtype != b.dtype:
            raise ValueError("pair must be nonempty with matching device and dtype")
        if a.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("reference supports FP32 and BF16 parameters")
        return a, b

    def load_state_dict(self, state_dict):
        load_state_dict_preserving_dtypes(self, state_dict)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Check all structural failures before writing any parameter.
        for group in self.param_groups:
            a, b = self._pair(group)
            if (a.grad is None) != (b.grad is None):
                raise ValueError("both factors need gradients, or neither")
            if a.grad is not None and (a.grad.is_sparse or b.grad.is_sparse):
                raise ValueError("dense gradients required")
        for group in self.param_groups:
            a, b = self._pair(group)
            if a.grad is None or group["lr"] == 0:
                continue
            af, bf = a.float(), b.float()
            identity = torch.eye(a.shape[0], device=a.device, dtype=torch.float32)
            aa = af @ af.T + group["damping"] * identity
            bb = bf.T @ bf + group["damping"] * identity
            # Both solves use OLD factors, before either factor is written.
            da = torch.linalg.solve(bb, a.grad.float())
            db = torch.linalg.solve(aa, b.grad.float().T).T
            method = "stochastic_rounding" if group["stochastic_rounding"] else "none"
            for param, update in ((a, da), (b, db)):
                subtract_one_(param, update, self.state[param], method,
                              alpha=group["lr"], sr=self.sr_stream)
        return loss
