from dataclasses import dataclass, field
from typing import Any
from model.layer import Layer
from model.tensor import Tensor

@dataclass(slots=True)
class ModelConfig:

    architecture: str

    num_layers: int

    hidden_size: int

    intermediate_size: int

    vocab_size: int

    max_position_embeddings: int

    num_attention_heads: int

    num_key_value_heads: int

    rope_theta: float

    rms_norm_eps: float

    num_experts: int

    experts_per_token: int

    head_dim: int = 128

    partial_rotary_factor: float = 1.0

    attn_output_gate: bool = False

@dataclass(slots=True)
class Model:

    config: ModelConfig

    embedding: Tensor

    layers: list[Layer] = field(default_factory=list)

    final_norm: Tensor | None = None

    lm_head: Tensor | None = None

    loaded: bool = False

    current_device: str = "disk"

    tensor_loader: Any = None

    layer_loader: Any = None

    def __len__(self):
        return len(self.layers)