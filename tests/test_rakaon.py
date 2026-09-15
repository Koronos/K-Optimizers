import copy

import pytest
import torch

from kaon import Rakaon


@pytest.mark.parametrize("shape", [(), (7,), (3, 5), (2, 3, 3, 3)])
@pytest.mark.parametrize("shrinkage", [0., 0.2, 1.])
def test_matches_dense_reference(shape, shrinkage):
    p = torch.nn.Parameter(torch.randn(shape))
    ref = p.detach().clone()
    opt = Rakaon([p], lr=.03, beta2=.8, shrinkage=shrinkage, weight_decay=.1)
    v = torch.zeros_like(p).reshape(p.shape[0], -1) if p.ndim >= 2 else torch.zeros_like(p)
    for step in range(1, 6):
        p.grad = torch.randn_like(p)
        g = p.grad.reshape(v.shape)
        v = .8 * v + .2 * (g.square() + 1e-30)
        estimate = v.mean(1)[:, None] * v.mean(0)[None, :] / v.mean() if p.ndim >= 2 else v
        estimate = ((1 - shrinkage) * estimate + shrinkage * v.mean()) / (1 - .8**step)
        u = g / estimate.sqrt()
        u = u / max(1., u.square().mean().sqrt().item())
        ref -= .03 * (u.reshape(shape) + .1 * ref)
        original_grad = p.grad.clone()
        opt.step()
        torch.testing.assert_close(p, ref)
        torch.testing.assert_close(p.grad, original_grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("shrinkage", [.1, 1.])
def test_resume_exact(dtype, device, shrinkage):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    p = torch.nn.Parameter(torch.randn(11, 13, device=device).to(dtype))
    opt = Rakaon([p], shrinkage=shrinkage)
    for _ in range(4):
        p.grad = torch.randn_like(p)
        opt.step()
    checkpoint = copy.deepcopy(opt.state_dict())
    q = torch.nn.Parameter(p.detach().clone())
    restored = Rakaon([q])
    restored.load_state_dict(checkpoint)
    for key in (("row", "col") if shrinkage < 1 else ("variance",)):
        torch.testing.assert_close(opt.state[p][key], restored.state[q][key], rtol=0, atol=0)
        assert restored.state[q][key].dtype == torch.float32
    for _ in range(4):
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
        rng = torch.get_rng_state()
        opt.step()
        torch.set_rng_state(rng)
        restored.step()
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_state_budget_and_zero_gradients():
    p = torch.nn.Parameter(torch.randn(256, 1024))
    opt = Rakaon([p])
    p.grad = torch.zeros_like(p)
    before = p.clone()
    opt.step()
    torch.testing.assert_close(p, before, rtol=0, atol=0)
    assert sum(v.numel() * v.element_size() for v in opt.state[p].values()
               if torch.is_tensor(v)) == 4 * (256 + 1024)
    p.grad = None
    opt.step()
    assert opt.state[p]["step"] == 1


def test_isotropic_state_budget():
    p = torch.nn.Parameter(torch.randn(256, 1024))
    opt = Rakaon([p], shrinkage=1)
    p.grad = torch.randn_like(p)
    opt.step()
    assert opt.state[p]["variance"].numel() == 1


@pytest.mark.parametrize("shrinkage, block_size", [(.1, None), (1., None), (1., 64)])
def test_state_buffers_stay_fp32_with_float64_default(shrinkage, block_size):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        p = torch.nn.Parameter(torch.randn(9, 17, dtype=torch.float32))
        opt = Rakaon([p], shrinkage=shrinkage, block_size=block_size)
        p.grad = torch.randn_like(p)
        opt.step()
        state = opt.state[p]
        assert all(value.dtype == torch.float32 for value in state.values()
                   if torch.is_tensor(value))
    finally:
        torch.set_default_dtype(previous)


def test_isotropic_batch_with_missing_gradients():
    params = [torch.nn.Parameter(torch.randn(3, 5)) for _ in range(6)]
    singles = [torch.nn.Parameter(p.detach().clone()) for p in params]
    opt = Rakaon(params, shrinkage=1)
    refs = [Rakaon([p], shrinkage=1) for p in singles]
    for step in range(6):
        for i, (p, q) in enumerate(zip(params, singles, strict=True)):
            p.grad = torch.randn_like(p) if (step + i) % 3 else None
            q.grad = p.grad.clone() if p.grad is not None else None
        opt.step()
        for ref in refs:
            ref.step()
        for p, q in zip(params, singles, strict=True):
            torch.testing.assert_close(p, q)


def test_layout_change_rejected_before_update():
    p = torch.nn.Parameter(torch.ones(3, 5))
    opt = Rakaon([p], shrinkage=1)
    p.grad = torch.ones_like(p)
    opt.step()
    before = p.clone()
    opt.param_groups[0]["shrinkage"] = .5
    with pytest.raises(ValueError, match="state layout"):
        opt.step()
    torch.testing.assert_close(p, before, rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [dict(lr=-1), dict(beta2=1), dict(shrinkage=1.1),
                                   dict(eps=0), dict(clip_threshold=0), dict(lr=float("nan"))])
def test_invalid(kwargs):
    with pytest.raises(ValueError):
        Rakaon([torch.nn.Parameter(torch.ones(2))], **kwargs)


def test_closure_groups_and_sparse_rejection():
    p = torch.nn.Parameter(torch.ones(3))
    q = torch.nn.Parameter(torch.ones(3))
    opt = Rakaon([{"params": [p], "lr": 0}, {"params": [q], "lr": .1}])
    def closure():
        opt.zero_grad()
        loss = p.square().sum() + q.square().sum()
        loss.backward()
        return loss
    assert opt.step(closure).item() == 6
    torch.testing.assert_close(p, torch.ones(3))
    assert q.max() < 1
    p.grad = torch.ones(3)
    q.grad = torch.ones(3).to_sparse()
    before = p.clone()
    with pytest.raises(RuntimeError, match="dense real"):
        opt.step()
    torch.testing.assert_close(p, before)
