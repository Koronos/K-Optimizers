"""Seed adapter initialization before entering the unchanged Rengu trainer.

Launch with DeepSpeed --module benchmarks.anima.run_seeded. Rengu's train_seed
is applied later, before training; it does not seed adapter initialization.
ANIMA_INIT_SEED controls this earlier phase and defaults to 42.
"""
import hashlib
import os
import random
import runpy

import numpy as np
import torch


def main():
    seed = int(os.environ.get("ANIMA_INIT_SEED", "42"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[anima comparison] initialization seed={seed}", flush=True)
    # Cache construction can consume random draws before adapter creation.
    # Seed again at that exact boundary and fingerprint all trainable weights.
    # This hook exists only in this experiment process, not the Rengu checkout.
    from rengu_flow.model.cosmos_predict2.pipeline import CosmosPredict2Pipeline

    configure = CosmosPredict2Pipeline.configure_adapter

    def seeded_configure(self, adapter_config):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        result = configure(self, adapter_config)
        digest = hashlib.sha256()
        for name, param in self.transformer.named_parameters():
            if param.requires_grad:
                digest.update(name.encode())
                digest.update(str((tuple(param.shape), param.dtype)).encode())
                digest.update(param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        print(f"[anima comparison] adapter_initial_sha256={digest.hexdigest()}", flush=True)
        return result

    CosmosPredict2Pipeline.configure_adapter = seeded_configure
    get_groups = CosmosPredict2Pipeline.get_param_groups

    def counted_groups(self, parameters):
        groups = get_groups(self, parameters)
        if self.config["optimizer"]["type"] == "benchmarks.gram_lora.PairedGram":
            from benchmarks.anima.gram_groups import pair_lora_groups
            groups = pair_lora_groups(self.transformer, groups)
            print(f"[anima comparison] gram_pairs={len(groups)} lora_scale=1", flush=True)
        count = sum(param.numel() for group in groups for param in group["params"])
        print(f"[anima comparison] optimizer_parameter_count={count}", flush=True)
        return groups

    CosmosPredict2Pipeline.get_param_groups = counted_groups
    runpy.run_module("rengu_flow.main", run_name="__main__")


if __name__ == "__main__":
    main()
