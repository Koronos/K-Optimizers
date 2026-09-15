import copy

import pytest
import torch
from benchmarks.gram_lora import PairedGram


@pytest.mark.parametrize("zero_b", [False, True])
def test_simultaneous_update_matches_float64_reference(zero_b):
    torch.manual_seed(12)
    a = torch.nn.Parameter(torch.randn(3, 7))
    b = torch.nn.Parameter(torch.zeros(5, 3) if zero_b else torch.randn(5, 3))
    target = torch.randn(5, 7)
    (b @ a - target).square().mean().backward()
    old_a, old_b = a.detach().double().clone(), b.detach().double().clone()
    ga, gb = a.grad.clone(), b.grad.clone()
    eye = torch.eye(3, dtype=torch.float64)
    expected_a = old_a - .01 * torch.linalg.solve(old_b.T @ old_b + .001 * eye, ga.double())
    expected_b = old_b - .01 * torch.linalg.solve(old_a @ old_a.T + .001 * eye, gb.double().T).T
    opt = PairedGram([{"params": [a, b]}], lr=.01, stochastic_rounding=False)
    opt.step()
    torch.testing.assert_close(a.double(), expected_a, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(b.double(), expected_b, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(a.grad, ga, rtol=0, atol=0)
    torch.testing.assert_close(b.grad, gb, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_zero_initialization_decreases_loss_and_resumes_bf16(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(12)
    a = torch.nn.Parameter(torch.randn(3, 7, device=device).bfloat16())
    b = torch.nn.Parameter(torch.zeros(5, 3, device=device, dtype=torch.bfloat16))
    target = torch.randn(5, 7, device=device)
    opt = PairedGram([{"params": [a, b]}], lr=.03, damping=.1)
    initial = (b.float() @ a.float() - target).square().mean().item()
    for _ in range(20):
        opt.zero_grad()
        (b.float() @ a.float() - target).square().mean().backward()
        opt.step()
    assert (b.float() @ a.float() - target).square().mean().item() < initial
    ac, bc = [torch.nn.Parameter(p.detach().clone()) for p in (a, b)]
    restored = PairedGram([{"params": [ac, bc]}])
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    for p, q in ((a, ac), (b, bc)):
        q.grad = p.grad.clone()
    opt.step()
    restored.step()
    torch.testing.assert_close(a, ac, rtol=0, atol=0)
    torch.testing.assert_close(b, bc, rtol=0, atol=0)


def test_partial_pair_gradient_rejected_before_updates():
    a = torch.nn.Parameter(torch.ones(2, 3))
    b = torch.nn.Parameter(torch.ones(4, 2))
    opt = PairedGram([{"params": [a, b]}])
    a.grad = torch.ones_like(a)
    with pytest.raises(ValueError, match="both factors"):
        opt.step()
    assert torch.equal(a, torch.ones_like(a))


@pytest.mark.parametrize("damping", [0, -1, float("nan")])
def test_invalid_damping(damping):
    with pytest.raises(ValueError):
        PairedGram([{"params": [torch.nn.Parameter(torch.ones(2, 3)),
                                torch.nn.Parameter(torch.ones(4, 2))]}], damping=damping)
