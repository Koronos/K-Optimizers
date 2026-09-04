"""Byte-identical optimizer resume under low-precision (bf16/fp16) params.

``torch.optim.Optimizer.load_state_dict`` casts every *floating* state tensor to the
param dtype. With bf16/fp16 weights that silently rounds ``m_scale`` / ``row`` /
``col`` / ``v`` / fp32 ``m`` through the param dtype before any dtype-restore
pass — measured ~0.3% relative drift per resume. These tests pin that
``load_state_dict_preserving_dtypes`` restores checkpoint tensors bit-identically
(and that the next step matches a no-save/load twin).
"""

from __future__ import annotations

import copy

import pytest
import torch

from kaon import Adakaon
from kaon._momentum_codec import load_state_dict_preserving_dtypes

MOMENTUM_DTYPES = ["int8", "4bit", "bfloat16", "float32"]
PARAM_DTYPES = [torch.bfloat16, torch.float16]


def _tensor_state(opt, p) -> dict[str, torch.Tensor]:
    return {k: v for k, v in opt.state[p].items() if torch.is_tensor(v)}


def _assert_state_bit_identical(ckpt_tensors: dict, live: dict) -> None:
    assert set(ckpt_tensors) == set(live)
    for k, ref in ckpt_tensors.items():
        got = live[k]
        assert got.dtype == ref.dtype, f"{k}: dtype {got.dtype} vs {ref.dtype}"
        assert got.shape == ref.shape, f"{k}: shape {tuple(got.shape)} vs {tuple(ref.shape)}"
        assert torch.equal(got.cpu(), ref.cpu()), f"{k}: values drifted vs checkpoint"


@pytest.mark.parametrize("momentum_dtype", MOMENTUM_DTYPES)
@pytest.mark.parametrize("param_dtype", PARAM_DTYPES, ids=["bf16", "fp16"])
def test_resume_state_tensors_bit_identical_lowp_params(momentum_dtype, param_dtype):
    """Every floating + quantized state tensor survives save/load bit-exactly."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, 8, dtype=param_dtype))
    opt = Adakaon(
        [p],
        lr=1e-3,
        betas=(0.9, 0.999),
        momentum_dtype=momentum_dtype,
        bf16_method="none",
        foreach=False,
        fused=False,
    )
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()

    sd = copy.deepcopy(opt.state_dict())
    ckpt = {k: v.detach().cpu().clone() for k, v in _tensor_state(opt, p).items()}

    p2 = torch.nn.Parameter(p.detach().clone())
    opt2 = Adakaon(
        [p2],
        lr=1e-3,
        betas=(0.9, 0.999),
        momentum_dtype=momentum_dtype,
        bf16_method="none",
        foreach=False,
        fused=False,
    )
    # Ensure state exists so load has somewhere to write (torch rebuilds it anyway).
    p2.grad = torch.zeros_like(p2)
    opt2.step()
    load_state_dict_preserving_dtypes(opt2, sd)

    _assert_state_bit_identical(ckpt, _tensor_state(opt2, p2))


@pytest.mark.parametrize("momentum_dtype", MOMENTUM_DTYPES)
@pytest.mark.parametrize("param_dtype", PARAM_DTYPES, ids=["bf16", "fp16"])
def test_resume_next_step_matches_no_saveload(momentum_dtype, param_dtype):
    """After an exact load, one more step equals a twin that never saved/loaded."""
    torch.manual_seed(1)
    w0 = torch.randn(12, 6, dtype=param_dtype)
    grads = [torch.randn(12, 6, dtype=param_dtype) for _ in range(6)]

    def _make(w):
        p = torch.nn.Parameter(w.clone())
        opt = Adakaon(
            [p],
            lr=1e-3,
            betas=(0.9, 0.999),
            momentum_dtype=momentum_dtype,
            bf16_method="none",
            foreach=False,
            fused=False,
        )
        return p, opt

    pa, oa = _make(w0)
    pb, ob = _make(w0)
    for g in grads[:4]:
        pa.grad = g.clone()
        pb.grad = g.clone()
        oa.step()
        ob.step()

    load_state_dict_preserving_dtypes(ob, copy.deepcopy(oa.state_dict()))
    pb.data.copy_(pa.detach())

    g = grads[4]
    pa.grad = g.clone()
    pb.grad = g.clone()
    oa.step()
    ob.step()

    assert torch.equal(pa.detach().cpu(), pb.detach().cpu())
    for k, va in _tensor_state(oa, pa).items():
        assert torch.equal(va.cpu(), _tensor_state(ob, pb)[k].cpu()), k


def test_resume_accepts_str_state_keys_json_drift():
    """JSON round-trips turn int state keys into str; resume must still be exact."""
    torch.manual_seed(2)
    p = torch.nn.Parameter(torch.randn(8, 4, dtype=torch.bfloat16))
    opt = Adakaon(
        [p],
        lr=1e-3,
        betas=(0.9, 0.999),
        momentum_dtype="int8",
        bf16_method="none",
        foreach=False,
        fused=False,
    )
    for _ in range(3):
        p.grad = torch.randn_like(p)
        opt.step()

    sd = copy.deepcopy(opt.state_dict())
    sd["state"] = {str(k): v for k, v in sd["state"].items()}
    ckpt = {k: v.detach().cpu().clone() for k, v in _tensor_state(opt, p).items()}

    p2 = torch.nn.Parameter(p.detach().clone())
    opt2 = Adakaon(
        [p2],
        lr=1e-3,
        betas=(0.9, 0.999),
        momentum_dtype="int8",
        bf16_method="none",
        foreach=False,
        fused=False,
    )
    p2.grad = torch.zeros_like(p2)
    opt2.step()
    load_state_dict_preserving_dtypes(opt2, sd)
    _assert_state_bit_identical(ckpt, _tensor_state(opt2, p2))
