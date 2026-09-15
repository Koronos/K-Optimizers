"""Experimental stochastic lookahead with exact BF16 host restoration.

Keeps true weights in host RAM, not a persistent FP32 GPU mirror. Stochastic
perturbations are replayed across eval/train; checkpoints must contain true weights.
This is a reference implementation with per-tensor transfers, not a speed claim.
"""
import torch

from kaon import Nekaon
from kaon._stochastic_rounding import SRStream, add_stochastic_
from kaon._wrappers import CodecBuffer


class NekaonSROffload(Nekaon):
    def __init__(self, params, **kwargs):
        self._host_true = {}
        self._look_sr = SRStream()
        self._cycle_before = None
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            for param in group["params"]:
                if param.dtype != torch.bfloat16:
                    raise ValueError("this experimental path requires BF16 parameters")

    @torch.no_grad()
    def _apply(self, sign, scale=1.0):
        if sign < 0:
            for param, host in self._host_true.items():
                param.copy_(host, non_blocking=param.is_cuda)
            return
        params = self._momentum_params()
        if self._has_e:
            self._look_sr.restore(self._cycle_before)
        else:
            # Materialize stream identity before its first snapshot so replay
            # cannot accidentally acquire a new owner id on the first cycle.
            for param, _, _, _ in params:
                self._look_sr.generator(param.device)
            self._cycle_before = self._look_sr.snapshot()
        for param, state, dtype, group in params:
            host = self._host_true.get(param)
            if host is None:
                host = torch.empty_like(param, device="cpu", pin_memory=param.is_cuda)
                self._host_true[param] = host
            # Ordered on the same CUDA stream as subsequent weight writes and
            # restores; host storage must not be modified outside this class.
            host.copy_(param, non_blocking=param.is_cuda)
            delta = CodecBuffer.read(state, "m", dtype, param).float().clone()
            delta.mul_(self.rho * scale * self._climb_step_scale(group, sign))
            bound = self._climb_bound(group, sign)
            torch.nan_to_num_(delta, nan=0., posinf=bound, neginf=-bound)
            delta.clamp_(-bound, bound)
            add_stochastic_(param, delta, sr=self._look_sr)

    def state_dict(self):
        if self._train_mode:
            raise ValueError("Call eval() before saving true model and optimizer weights")
        state = super().state_dict()
        state["_sr_host_lookahead"] = {
            "version": 1,
            "before": self._cycle_before if self._has_e else self._look_sr.snapshot(),
        }
        return state

    def load_state_dict(self, state_dict):
        copied = dict(state_dict)
        meta = copied.pop("_sr_host_lookahead", None)
        if not meta or meta.get("version") != 1:
            raise ValueError("expected an NekaonSROffload checkpoint")
        self._host_true.clear()
        self._cycle_before = meta["before"]
        self._look_sr.restore(meta["before"])
        super().load_state_dict(copied)

    @property
    def host_snapshot_bytes(self):
        return sum(t.numel() * t.element_size() for t in self._host_true.values())

    def _warn_if_inert(self):
        # The inherited detector assumes deterministic round-to-nearest writes.
        # Its half-ULP condition does not diagnose stochastic perturbations.
        pass
