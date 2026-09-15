"""Cross-process checkpoint coverage for Rakaon's bf16 state and SR stream."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_WORKER = r'''
import sys
import torch

from kaon import Rakaon, reseed_stochastic_rounding

mode, device, shrinkage, checkpoint, result = sys.argv[1:]
shrinkage = float(shrinkage)
device = torch.device(device)
torch.manual_seed(1729)
if device.type == "cuda":
    torch.cuda.manual_seed_all(1729)
reseed_stochastic_rounding()

p = torch.nn.Parameter(torch.randn(13, 17, device=device).to(torch.bfloat16))
opt = Rakaon([p], lr=0.03, beta2=0.8, shrinkage=shrinkage)

def advance(n):
    for _ in range(n):
        p.grad = torch.randn_like(p)
        opt.step()

if mode == "continuous":
    advance(6)
elif mode == "split":
    advance(3)
    torch.save({
        "model": p.detach().clone(),
        "optimizer": opt.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": ([state.cpu() for state in torch.cuda.get_rng_state_all()]
                      if device.type == "cuda" else None),
    }, checkpoint)
    raise SystemExit(0)
elif mode == "resume":
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    p.data.copy_(saved["model"])
    opt.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["torch_rng"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
    advance(3)
else:
    raise ValueError(mode)

torch.save({"model": p.detach().cpu(), "optimizer": opt.state_dict()}, result)
'''


def _run(tmp_path: Path, python: str, mode: str, device: str, shrinkage: float, name: str):
    checkpoint = tmp_path / "checkpoint.pt"
    result = tmp_path / f"{name}.pt"
    env = os.environ.copy()
    src = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
    args = [python, "-c", _WORKER, mode, device, str(shrinkage), str(checkpoint), str(result)]
    subprocess.run(args, check=True, env=env, capture_output=True, text=True)
    return (torch.load(result, map_location="cpu", weights_only=False)
            if mode != "split" else None)


@pytest.mark.parametrize("shrinkage", [0.1, 1.0])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bf16_checkpoint_resume_matches_continuous_process(tmp_path, shrinkage, device):
    """A fresh process must resume both FP32 statistics and bf16 SR noise exactly."""
    python = sys.executable
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    continuous = _run(tmp_path, python, "continuous", device, shrinkage, "continuous")
    _run(tmp_path, python, "split", device, shrinkage, "split")
    resumed = _run(tmp_path, python, "resume", device, shrinkage, "resumed")

    torch.testing.assert_close(continuous["model"], resumed["model"], rtol=0, atol=0)
    for key, state in continuous["optimizer"]["state"].items():
        other = resumed["optimizer"]["state"][key]
        for name, value in state.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value, other[name], rtol=0, atol=0)
                if name in {"row", "col", "variance"}:
                    assert value.dtype == torch.float32
    assert "_sr_meta" in continuous["optimizer"]
    assert "_sr_meta" in resumed["optimizer"]
