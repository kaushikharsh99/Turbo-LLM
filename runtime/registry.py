"""Explicit architecture capabilities; no substring-based dispatch."""
from dataclasses import dataclass


class CompatibilityError(ValueError):
    pass


@dataclass(frozen=True)
class Architecture:
    adapter: str
    streaming: bool
    checkpoint: str = "safetensors"


ARCHITECTURES = {
    "LlamaForCausalLM": Architecture("llama", False),
    "Qwen3MoeForCausalLM": Architecture("qwen3_moe", True),
    "Qwen3_5MoeForConditionalGeneration": Architecture("qwen3_5_moe", True),
    "Qwen3_5MoeForCausalLM": Architecture("qwen3_5_moe", True),
    "OlmoeForCausalLM": Architecture("olmoe", True),
}


def resolve_architecture(config):
    names = getattr(config, "architectures", None) or []
    for name in names:
        if name in ARCHITECTURES:
            return ARCHITECTURES[name]
    raise CompatibilityError(
        f"Unsupported architectures {names!r}. Registered: {', '.join(ARCHITECTURES)}. "
        "Add an adapter and capability registration, or use a supported GGUF with the llama_cpp backend."
    )
