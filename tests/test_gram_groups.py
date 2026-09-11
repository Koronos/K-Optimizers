import pytest
import torch
from benchmarks.anima.gram_groups import pair_lora_groups


def layer():
    module = torch.nn.Module()
    module.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(7, 3, bias=False)})
    module.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(3, 5, bias=False)})
    module.scaling = {"default": 1.0}
    return module


def test_pairs_from_module_objects_and_preserves_lr():
    model = torch.nn.Sequential(layer(), layer())
    params = list(reversed(list(model.parameters())))
    groups = pair_lora_groups(model, [{"params": params, "lr": .02}])
    assert len(groups) == 2
    for module, group in zip(model, groups):
        assert group["params"][0] is module.lora_A["default"].weight
        assert group["params"][1] is module.lora_B["default"].weight
        assert group["lr"] == .02


@pytest.mark.parametrize("failure", ["scale", "split", "extra"])
def test_rejects_unsupported_pairing(failure):
    model = layer()
    params = list(model.parameters())
    groups = [{"params": params}]
    if failure == "scale":
        model.scaling["default"] = .5
    elif failure == "split":
        groups = [{"params": [p]} for p in params]
    else:
        params.append(torch.nn.Parameter(torch.ones(4)))
    with pytest.raises(ValueError):
        pair_lora_groups(model, groups)
