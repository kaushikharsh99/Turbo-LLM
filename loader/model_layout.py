import json
import os
import re


class ModelLayout:

    def __init__(self, snapshot_path):
        self.snapshot_path = snapshot_path

        with open(os.path.join(snapshot_path, "config.json")) as f:
            self.config = json.load(f)

        index_path = os.path.join(snapshot_path, "model.safetensors.index.json")

        with open(index_path) as f:
            index = json.load(f)

        self.weight_map = index["weight_map"]

        self.model_type = self.config.get("model_type", "unknown")

        self.layer_prefix = None
        self.expert_prefix = None

        self.gate_name = None
        self.up_name = None
        self.down_name = None

        self.router_name = None

        self._detect_layout()

    def _detect_layout(self):

        for tensor_name in self.weight_map.keys():

            m = re.match(
                r"(.*)\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.weight",
                tensor_name,
            )

            if m:
                self.layer_prefix = m.group(1)
                print("Detected layer prefix:", self.layer_prefix)
                print("Matched tensor:", tensor_name)
                self.expert_prefix = "mlp.experts"

                self.gate_name = "gate_proj"
                self.up_name = "up_proj"
                self.down_name = "down_proj"
                break

        if self.layer_prefix is None:
            raise RuntimeError(
                f"Unsupported model layout ({self.model_type})"
            )

        # Detect router tensor automatically
        for tensor_name in self.weight_map.keys():

            prefix = f"{self.layer_prefix}.0.mlp."

            if not tensor_name.startswith(prefix):
                continue

            suffix = tensor_name[len(prefix):]

            if not suffix.endswith(".weight"):
                continue

            name = suffix[:-7]  # remove ".weight"

            if "." in name:
                continue

            if name not in {
                self.gate_name,
                self.up_name,
                self.down_name,
            }:
                self.router_name = name
                break

        if self.router_name is None:
            raise RuntimeError(
                "Could not detect router tensor."
            )

    def expert_prefix_name(self, layer, expert):
        return (
            f"{self.layer_prefix}."
            f"{layer}."
            f"{self.expert_prefix}."
            f"{expert}"
        )

    def gate_tensor(self, layer, expert):
        return (
            f"{self.expert_prefix_name(layer, expert)}."
            f"{self.gate_name}.weight"
        )

    def up_tensor(self, layer, expert):
        return (
            f"{self.expert_prefix_name(layer, expert)}."
            f"{self.up_name}.weight"
        )

    def down_tensor(self, layer, expert):
        return (
            f"{self.expert_prefix_name(layer, expert)}."
            f"{self.down_name}.weight"
        )

    def router_tensor(self, layer):
        return (
            f"{self.layer_prefix}.{layer}.mlp.{self.router_name}.weight"
        )
    def embed_tensor(self):
        if self.layer_prefix.startswith("model.language_model"):
            return "model.language_model.embed_tokens.weight"
        return "model.embed_tokens.weight"


    def norm_tensor(self):
        if self.layer_prefix.startswith("model.language_model"):
            return "model.language_model.norm.weight"
        return "model.norm.weight"


    def lm_head_tensor(self):
        return "lm_head.weight"
    
    def layer_prefix_name(self, layer):
        return f"{self.layer_prefix}.{layer}"
    def module_name(self, tensor_name):
        """
        Convert a checkpoint tensor name into the corresponding
        PyTorch module parameter name.
        """

        if tensor_name.startswith("model.language_model."):
            return tensor_name.replace(
                "model.language_model.",
                "model.",
                1,
            )

        return tensor_name