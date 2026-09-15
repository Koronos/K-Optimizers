"""Seed adapter initialization before entering the unchanged Rengu trainer.

Launch with DeepSpeed --module benchmarks.anima.run_seeded. Rengu's train_seed
is applied later, before training; it does not seed adapter initialization.
ANIMA_INIT_SEED controls this earlier phase and defaults to 42.
"""
import hashlib
import os
import random
import runpy
import time

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
    # Rengu's DeepSpeed resume checkpoint path does not enter optimizer.eval().
    # Patch only this experimental class, in-process, around the ENTIRE save:
    # switching views inside optimizer.state_dict would be too late for model data.
    from benchmarks.anima.checkpoint_view import checkpoint_true_view
    from benchmarks.nekaon_sr_offload import NekaonSROffload
    from rengu_flow.utils.saver import Saver

    save_checkpoint = Saver.save_checkpoint

    def save_true_checkpoint(self, *args, **kwargs):
        optimizer = self.model_engine.optimizer
        if not isinstance(optimizer, NekaonSROffload):
            return save_checkpoint(self, *args, **kwargs)
        self._wait_async_export()
        return checkpoint_true_view(optimizer, lambda: save_checkpoint(self, *args, **kwargs))

    Saver.save_checkpoint = save_true_checkpoint
    if os.environ.get("ANIMA_OPT_TIMING") == "1":
        from kaon import Nekaon
        original_step = Nekaon._step_impl

        def timed_step(self, *args, **kwargs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = original_step(self, *args, **kwargs)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            host_bytes = getattr(self, "host_snapshot_bytes", 0)
            print(f"[optimizer diagnostic] seconds={seconds:.6f} host_snapshot_bytes={host_bytes}", flush=True)
            return result

        Nekaon._step_impl = timed_step
    runpy.run_module("rengu_flow.main", run_name="__main__")


if __name__ == "__main__":
    main()
