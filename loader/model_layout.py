import json
import os
import re


class ModelLayout:

    def __init__(self, snapshot_path):
        self.snapshot_path = snapshot_path

        with open(os.path.join(snapshot_path, "config.json")) as f:
            self.config = json.load(f)

        from loader.checkpoint import SafetensorsReader
        self.reader = SafetensorsReader(snapshot_path)
        self.weight_map = self.reader.weight_map

        self.model_type = self.config.get("model_type", "unknown")

        self.layer_prefix = None
        self.expert_prefix = None

        self.gate_name = None
        self.up_name = None
        self.down_name = None

        self.router_name = None

        self._expand_packed_experts()
        self._detect_layout()
        self._validate_experts()

    def _expand_packed_experts(self):
        # Qwen's HF packed storage is sliced per expert on CPU, never expanded
        # into a full resident copy. Execution still consumes the same interface.
        for source in list(self.weight_map):
            match = re.fullmatch(r"(.*\.mlp\.experts)\.gate_up_proj(?:\.weight)?", source)
            if not match:
                continue
            prefix = match.group(1)
            down = prefix + ".down_proj" + (".weight" if source.endswith(".weight") else "")
            shape = self.reader.shape(source)
            if down not in self.weight_map or len(shape) != 3 or shape[1] % 2:
                raise RuntimeError(f"Unsupported packed expert shape or missing down projection: {source}")
            down_shape = self.reader.shape(down)
            if down_shape != (shape[0], shape[2], shape[1] // 2):
                raise RuntimeError(f"Incompatible packed projections: {shape}, {down_shape}")
            if source + "_scale_inv" in self.weight_map or down + "_scale_inv" in self.weight_map:
                raise RuntimeError("Packed quantized expert scales are unsupported; use separate experts or GGUF")
            width = shape[1] // 2
            for expert in range(shape[0]):
                self.reader.add_slice(f"{prefix}.{expert}.gate_proj.weight", source,
                                      (expert, slice(0, width), slice(None)))
                self.reader.add_slice(f"{prefix}.{expert}.up_proj.weight", source,
                                      (expert, slice(width, None), slice(None)))
                self.reader.add_slice(f"{prefix}.{expert}.down_proj.weight", down,
                                      (expert, slice(None), slice(None)))

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
                f"Unsupported expert layout ({self.model_type}). Turbo streaming requires individually named mlp.experts.N.gate_proj/up_proj/down_proj.weight tensors; packed experts need conversion or the GGUF backend."
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

    def _validate_experts(self):
        from runtime.registry import CompatibilityError
        config = self.config.get("text_config", self.config)
        layers = config.get("num_hidden_layers")
        experts = config.get("num_experts")
        if layers is None or experts is None:
            # Layout-only probes may omit configuration dimensions.
            return
        missing = []
        for layer in range(layers):
            names = [self.router_tensor(layer)]
            for expert in range(experts):
                names += [self.gate_tensor(layer, expert), self.up_tensor(layer, expert),
                          self.down_tensor(layer, expert)]
            missing.extend(name for name in names if name not in self.weight_map)
        if missing:
            raise CompatibilityError(f"Incomplete expert checkpoint; missing {len(missing)} tensors, including {missing[:4]}")

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