import copy

import pytest
import torch

from kaon import Rakaon


@pytest.mark.parametrize("shape", [(), (7,), (3, 5), (2, 3, 3, 3)])
@pytest.mark.parametrize("block_size", [1, 4, 128])
def test_block_variance_matches_unpadded_reference(shape, block_size):
    p = torch.nn.Parameter(torch.randn(shape))
    q = p.detach().clone()
    opt = Rakaon([p], lr=.03, beta2=.8, shrinkage=1, block_size=block_size, weight_decay=.1)
    variance = [torch.tensor(0.) for _ in range((p.numel() + block_size - 1) // block_size)]
    for step in range(1, 6):
        p.grad = torch.randn_like(p)
        updates = []
        for i, block in enumerate(p.grad.reshape(-1).split(block_size)):
            variance[i] = .8 * variance[i] + .2 * (block.square().mean() + 1e-30)
            updates.append(block / (variance[i] / (1 - .8 ** step)).sqrt())
        update = torch.cat(updates).reshape(shape)
        update /= max(1., update.square().mean().sqrt().item())
        q -= .03 * (update + .1 * q)
        original = p.grad.clone()
        opt.step()
        torch.testing.assert_close(p, q)
        torch.testing.assert_close(p.grad, original, rtol=0, atol=0)
    assert opt.state[p]["variance"].numel() == len(variance)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_block_checkpoint_and_missing_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    params = [torch.nn.Parameter(torch.randn(7, 9, device=device).bfloat16()) for _ in range(3)]
    opt = Rakaon(params, shrinkage=1, block_size=16)
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()
    clones = [torch.nn.Parameter(p.detach().clone()) for p in params]
    restored = Rakaon(clones)
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    for i, (p, q) in enumerate(zip(params, clones, strict=True)):
        p.grad = torch.randn_like(p) if i != 1 else None
        q.grad = p.grad.clone() if p.grad is not None else None
    opt.step()
    restored.step()
    for p, q in zip(params, clones, strict=True):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        torch.testing.assert_close(opt.state[p]["variance"], restored.state[q]["variance"], rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [dict(block_size=0), dict(block_size=True),
                                   dict(block_size=1.5), dict(block_size=4, shrinkage=.1)])
def test_invalid_block_layout(kwargs):
    with pytest.raises(ValueError):
        Rakaon([torch.nn.Parameter(torch.ones(3))], **({"shrinkage": 1} | kwargs))
