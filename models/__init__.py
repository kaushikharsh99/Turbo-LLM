from models.qwen3_moe import (
    Qwen3MoeAdapter
)
from models.llama import (
    LlamaAdapter
)
from models.olmoe import (
    OLMoEAdapter
)

MODEL_REGISTRY={

"qwen3_moe":
Qwen3MoeAdapter,

"llama":
LlamaAdapter,

"olmoe":
OLMoEAdapter,

}
