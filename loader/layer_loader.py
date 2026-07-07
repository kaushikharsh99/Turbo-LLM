from dataclasses import dataclass, field
from typing import Dict

import torch

from .safetensor_loader import SafeTensorLoader

@dataclass
class AttentionWeights:
    q_proj: torch.Tensor | None = None
    k_proj: torch.Tensor | None = None
    v_proj: torch.Tensor | None = None
    o_proj: torch.Tensor | None = None


@dataclass
class MLPExpert:
    gate_proj: torch.Tensor | None = None
    up_proj: torch.Tensor | None = None
    down_proj: torch.Tensor | None = None

    gate_scale: torch.Tensor | None = None
    up_scale: torch.Tensor | None = None
    down_scale: torch.Tensor | None = None


@dataclass
class SharedExpert:
    gate_proj: torch.Tensor | None = None
    up_proj: torch.Tensor | None = None
    down_proj: torch.Tensor | None = None

    gate_scale: torch.Tensor | None = None
    up_scale: torch.Tensor | None = None
    down_scale: torch.Tensor | None = None


@dataclass
class Layer:

    attention: AttentionWeights = field(default_factory=AttentionWeights)

    input_norm: torch.Tensor | None = None
    post_attention_norm: torch.Tensor | None = None

    router: torch.Tensor | None = None

    experts: Dict[int, MLPExpert] = field(default_factory=dict)

    shared_expert: SharedExpert = field(default_factory=SharedExpert)


class LayerLoader:

    def __init__(self, tensor_loader: SafeTensorLoader):
        self.loader = tensor_loader

    def load_layer(self, layer_id: int) -> Layer:

        layer = Layer()

        if self.loader.has_tensor(f"model.language_model.layers.{layer_id}.input_layernorm.weight"):
            prefix = f"model.language_model.layers.{layer_id}"
        else:
            prefix = f"model.layers.{layer_id}"

        layer.attention.q_proj = self.loader.get_tensor(
            f"{prefix}.self_attn.q_proj.weight"
        )

        layer.attention.k_proj = self.loader.get_tensor(
            f"{prefix}.self_attn.k_proj.weight"
        )

        layer.attention.v_proj = self.loader.get_tensor(
            f"{prefix}.self_attn.v_proj.weight"
        )

        layer.attention.o_proj = self.loader.get_tensor(
            f"{prefix}.self_attn.o_proj.weight"
        )


        layer.input_norm = self.loader.get_tensor(
            f"{prefix}.input_layernorm.weight"
        )

        layer.post_attention_norm = self.loader.get_tensor(
            f"{prefix}.post_attention_layernorm.weight"
        )


        router_name = f"{prefix}.mlp.gate.weight"

        if self.loader.has_tensor(router_name):
            layer.router = self.loader.get_tensor(router_name)


        shared = layer.shared_expert

        shared_prefix = f"{prefix}.mlp.shared_expert"

        if self.loader.has_tensor(f"{shared_prefix}.gate_proj.weight"):

            shared.gate_proj = self.loader.get_tensor(
                f"{shared_prefix}.gate_proj.weight"
            )

            shared.up_proj = self.loader.get_tensor(
                f"{shared_prefix}.up_proj.weight"
            )

            shared.down_proj = self.loader.get_tensor(
                f"{shared_prefix}.down_proj.weight"
            )

            for name, attr in [
                ("gate_proj", "gate_scale"),
                ("up_proj", "up_scale"),
                ("down_proj", "down_scale"),
            ]:

                tensor_name = (
                    f"{shared_prefix}.{name}.weight_scale_inv"
                )

                if self.loader.has_tensor(tensor_name):
                    setattr(
                        shared,
                        attr,
                        self.loader.get_tensor(tensor_name),
                    )

        
        expert = 0

        while True:

            base = f"{prefix}.mlp.experts.{expert}"

            gate = f"{base}.gate_proj.weight"

            if not self.loader.has_tensor(gate):
                break

            e = MLPExpert()

            e.gate_proj = self.loader.get_tensor(gate)

            e.up_proj = self.loader.get_tensor(
                f"{base}.up_proj.weight"
            )

            e.down_proj = self.loader.get_tensor(
                f"{base}.down_proj.weight"
            )

            for name, attr in [
                ("gate_proj", "gate_scale"),
                ("up_proj", "up_scale"),
                ("down_proj", "down_scale"),
            ]:

                tensor_name = (
                    f"{base}.{name}.weight_scale_inv"
                )

                if self.loader.has_tensor(tensor_name):
                    setattr(
                        e,
                        attr,
                        self.loader.get_tensor(tensor_name),
                    )

            layer.experts[expert] = e

            expert += 1

        return layer