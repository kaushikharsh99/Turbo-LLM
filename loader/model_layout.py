import json
import os
import re


class ModelLayout:

    def __init__(self, snapshot_path):
        if os.path.isfile(snapshot_path):
            self.snapshot_dir = os.path.dirname(snapshot_path)
            self.is_gguf_file = snapshot_path.endswith('.gguf')
        else:
            self.snapshot_dir = snapshot_path
            self.is_gguf_file = False
        self.snapshot_path = snapshot_path

        config_file = os.path.join(self.snapshot_dir, "config.json")
        if os.path.exists(config_file):
            with open(config_file) as f:
                self.config = json.load(f)
        else:
            self.config = {"model_type": "qwen2_moe"}

        index_path = os.path.join(self.snapshot_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            self.weight_map = index["weight_map"]
        else:
            self.weight_map = {}

        self.model_type = self.config.get("model_type", "unknown")

        self.layer_prefix = None
        self.expert_prefix = None

        self.gate_name = None
        self.up_name = None
        self.down_name = None

        self.router_name = None

        self._detect_layout()

    def _detect_layout(self):
        if self.is_gguf_file:
            from loader.gguf_loader import GGUFLoader
            gg_reader = GGUFLoader(self.snapshot_path)
            tensor_names = gg_reader.list_tensors()
            gg_reader.close()
        else:
            tensor_names = list(self.weight_map.keys())

        for tensor_name in tensor_names:
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
                
            # GGUF tensor naming convention
            if "blk." in tensor_name or "ffn_gate" in tensor_name:
                self.layer_prefix = "model.language_model.layers"
                self.expert_prefix = "mlp.experts"
                self.gate_name = "gate_proj"
                self.up_name = "up_proj"
                self.down_name = "down_proj"
                print(f"Detected GGUF tensor layout: {tensor_name}")
                break

        if self.layer_prefix is None:
            self.layer_prefix = "model.language_model.layers"
            self.expert_prefix = "mlp.experts"
            self.gate_name = "gate_proj"
            self.up_name = "up_proj"
            self.down_name = "down_proj"

        if self.is_gguf_file:
            self.router_name = "gate"
        else:
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
                self.router_name = "gate"

    def expert_prefix_name(self, layer, expert):
        return (
            f"{self.layer_prefix}."
            f"{layer}."
            f"{self.expert_prefix}."
            f"{expert}"
        )

    def gate_tensor(self, layer, expert):
        if self.is_gguf_file:
            return f"blk.{layer}.ffn_gate_exps.weight"
        return (
            f"{self.expert_prefix_name(layer, expert)}."
            f"{self.gate_name}.weight"
        )

    def up_tensor(self, layer, expert):
        if self.is_gguf_file:
            return f"blk.{layer}.ffn_up_exps.weight"
        return (
            f"{self.expert_prefix_name(layer, expert)}."
            f"{self.up_name}.weight"
        )

    def down_tensor(self, layer, expert):
        if self.is_gguf_file:
            return f"blk.{layer}.ffn_down_exps.weight"
        return (
            f"{self.expert_prefix_name(layer, expert)}."
            f"{self.down_name}.weight"
        )

    def router_tensor(self, layer):
        if self.is_gguf_file:
            return f"blk.{layer}.ffn_gate_inp.weight"
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