from models.qwen3_moe import (
    Qwen3MoeAdapter
)
from models.qwen3_5_moe import (
    Qwen35MoeAdapter
)
from models.llama import (
    LlamaAdapter
)
from models.olmoe import (
    OLMoEAdapter
)

MODEL_REGISTRY = {

    "qwen3_moe":
    Qwen3MoeAdapter,

    "qwen3_5_moe":
    Qwen35MoeAdapter,

    "llama":
    LlamaAdapter,

    "olmoe":
    OLMoEAdapter,

}