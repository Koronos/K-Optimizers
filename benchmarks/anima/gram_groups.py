"""Pair actual PEFT factor objects; never infer pairs from tensor dimensions."""


def pair_lora_groups(transformer, groups):
    owners = {}
    for index, group in enumerate(groups):
        for param in group["params"]:
            if id(param) in owners:
                raise ValueError("duplicate optimizer parameter")
            owners[id(param)] = index
    result, covered = [], set()
    for name, module in transformer.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        for adapter in module.lora_A:
            if adapter not in module.lora_B:
                raise ValueError(f"missing B factor: {name}/{adapter}")
            a, b = module.lora_A[adapter].weight, module.lora_B[adapter].weight
            if id(a) not in owners and id(b) not in owners:
                continue
            if id(a) not in owners or id(b) not in owners or owners[id(a)] != owners[id(b)]:
                raise ValueError(f"pair split across optimizer groups: {name}/{adapter}")
            if id(a) in covered or id(b) in covered:
                raise ValueError("factor reused by multiple adapters")
            scale = float(module.scaling[adapter])
            if scale != 1.0:
                raise ValueError(f"Gram pilot requires alpha/r=1, observed {scale}: {name}")
            source = groups[owners[id(a)]]
            result.append({**source, "params": [a, b]})
            covered.update((id(a), id(b)))
    if covered != set(owners):
        raise ValueError("some optimizer parameters are not explicit supported LoRA pairs")
    if not result:
        raise ValueError("no trainable LoRA pairs")
    return result
