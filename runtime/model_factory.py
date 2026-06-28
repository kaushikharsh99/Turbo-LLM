from models import (
MODEL_REGISTRY
)


def create_adapter(
model,
loader,
config,
):

    arch=config.architectures[0]
    
    if "Qwen3_5Moe" in arch:

        print(f"Detected: Qwen35MoeAdapter (arch={arch})")
        return MODEL_REGISTRY[
            "qwen3_5_moe"
        ](
            model,
            loader,
        )

    if "Qwen3Moe" in arch:

        print(f"Detected: Qwen3MoeAdapter (arch={arch})")
        return MODEL_REGISTRY[
            "qwen3_moe"
        ](
            model,
            loader,
        )

    if "Olmoe" in arch:

        print(f"Detected: OLMoEAdapter (arch={arch})")
        return MODEL_REGISTRY[
            "olmoe"
        ](
            model,
            loader,
        )

    if "Llama" in arch:

        print(f"Detected: LlamaAdapter (arch={arch})")
        return MODEL_REGISTRY[
            "llama"
        ](
            model,
            loader,
        )

    raise ValueError(
        f"Unsupported architecture: {arch}"
    )

