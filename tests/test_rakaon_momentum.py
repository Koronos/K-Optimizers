import copy

import pytest
import torch

from kaon import Rakaon


@pytest.mark.parametrize("beta1", [.5, .9])
def test_momentum_matches_independent_reference(beta1):
    p = torch.nn.Parameter(torch.tensor([[1., -2.], [3., 4.]]))
    ref = p.detach().double().clone()
    momentum = torch.zeros_like(ref)
    variance = torch.tensor(0., dtype=torch.float64)
    opt = Rakaon([p], lr=.03, beta1=beta1, beta2=.8, shrinkage=1,
                 weight_decay=.1, stochastic_rounding=False)
    for step in range(1, 8):
        grad = torch.tensor([[step, -1.], [2., -step]])
        p.grad = grad.clone()
        g = grad.double()
        variance = .8 * variance + .2 * g.square().mean()
        u = g / (variance / (1 - .8**step)).sqrt()
        u /= max(1., u.square().mean().sqrt().item())
        momentum = beta1 * momentum + (1 - beta1) * u
        ref -= .03 * (momentum / (1 - beta1**step) + .1 * ref)
        opt.step()
        torch.testing.assert_close(p.double(), ref, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_momentum_bf16_resume_and_missing_gradient(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    p = torch.nn.Parameter(torch.ones(3, 5, device=device, dtype=torch.bfloat16))
    opt = Rakaon([p], beta1=.9, shrinkage=1)
    p.grad = torch.full_like(p, .3)
    opt.step()
    p.grad = None
    opt.step()
    assert opt.state[p]["step"] == 1
    q = torch.nn.Parameter(p.detach().clone())
    restored = Rakaon([q])
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    assert restored.state[q]["momentum"].dtype == torch.float32
    p.grad = torch.full_like(p, -.2)
    q.grad = p.grad.clone()
    opt.step()
    restored.step()
    torch.testing.assert_close(p, q, rtol=0, atol=0)
    torch.testing.assert_close(opt.state[p]["momentum"], restored.state[q]["momentum"], rtol=0, atol=0)


def test_legacy_checkpoint_and_beta_mutation():
    p = torch.nn.Parameter(torch.ones(3))
    opt = Rakaon([p], shrinkage=1)
    p.grad = torch.ones_like(p)
    opt.step()
    state = copy.deepcopy(opt.state_dict())
    state["param_groups"][0].pop("beta1")
    for value in state["state"].values():
        value.pop("momentum_beta1", None)
    opt.load_state_dict(state)
    opt.step()
    opt.param_groups[0]["beta1"] = .9
    with pytest.raises(ValueError, match="Cannot switch beta1"):
        opt.step()


@pytest.mark.parametrize("kwargs", [{"beta1": -1}, {"beta1": 1}, {"beta1": float("nan")},
                                    {"beta1": .9, "shrinkage": .1},
                                    {"beta1": .9, "block_size": 64}])
def test_invalid_momentum_configuration(kwargs):
    with pytest.raises(ValueError):
        Rakaon([torch.nn.Parameter(torch.ones(3))], **({"shrinkage": 1} | kwargs))
