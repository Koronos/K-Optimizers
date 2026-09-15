import copy

import pytest
import torch
from benchmarks.nekaon_sr_offload import NekaonSROffload


def make(device):
    p = torch.nn.Parameter(torch.full((64, 64), .01, dtype=torch.bfloat16, device=device))
    opt = NekaonSROffload([p], lr=1e-6, weight_decay=0, cautious=False,
                         gradient_centralization=False, momentum_dtype="4bit")
    return p, opt


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_subulp_effect_exact_restore_and_repeatable_eval(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(42)
    p, opt = make(device)
    p.grad = torch.ones_like(p)
    opt.step()
    live = p.detach().clone()
    opt.eval()
    true = p.detach().clone()
    assert (live != true).any(), "SR must expose some sub-ULP perturbations"
    assert opt.host_snapshot_bytes == p.numel() * 2
    assert all(t.device.type == "cpu" for t in opt._host_true.values())
    rng = torch.get_rng_state().clone()
    for _ in range(100):
        opt.train()
        torch.testing.assert_close(p, live, rtol=0, atol=0)
        opt.eval()
        torch.testing.assert_close(p, true, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_checkpoint_replays_same_live_weights_and_next_step(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(42)
    p, opt = make(device)
    p.grad = torch.ones_like(p)
    opt.step()
    opt.eval()
    checkpoint = copy.deepcopy(opt.state_dict())
    q, resumed = make(device)
    with torch.no_grad():
        q.copy_(p)
    resumed.load_state_dict(checkpoint)
    opt.train()
    torch.testing.assert_close(p, q, rtol=0, atol=0)
    p.grad = torch.full_like(p, -.4)
    q.grad = p.grad.clone()
    opt.step()
    resumed.step()
    torch.testing.assert_close(p, q, rtol=0, atol=0)
    opt.eval()
    resumed.eval()
    torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_train_checkpoint_rejected():
    _, opt = make("cpu")
    with pytest.raises(ValueError, match="eval"):
        opt.state_dict()


@pytest.mark.parametrize("fail", [False, True])
def test_checkpoint_boundary_saves_true_weights_and_restores_view(fail):
    from benchmarks.anima.checkpoint_view import checkpoint_true_view
    p, opt = make("cpu")
    p.grad = torch.ones_like(p)
    opt.step()
    live = p.detach().clone()
    opt.eval()
    true = p.detach().clone()
    opt.train()

    def write():
        torch.testing.assert_close(p, true, rtol=0, atol=0)
        opt.state_dict()
        if fail:
            raise OSError("simulated disk error")
        return 123

    if fail:
        with pytest.raises(OSError):
            checkpoint_true_view(opt, write)
    else:
        assert checkpoint_true_view(opt, write) == 123
    assert opt._train_mode
    torch.testing.assert_close(p, live, rtol=0, atol=0)
