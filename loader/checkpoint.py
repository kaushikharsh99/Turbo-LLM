"""Safetensors storage interface shared by layout detection and readers."""
import json
from pathlib import Path
from safetensors import safe_open
from runtime.registry import CompatibilityError


class SafetensorsReader:
    def __init__(self, path):
        self.path = Path(path)
        index = self.path / "model.safetensors.index.json"
        if index.exists():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:
            self.weight_map = {}
            for shard in sorted(self.path.glob("*.safetensors")):
                with safe_open(shard, framework="pt", device="cpu") as handle:
                    for name in handle.keys():
                        if name in self.weight_map:
                            raise CompatibilityError(f"Duplicate tensor {name} in {shard}")
                        self.weight_map[name] = shard.name
        if not self.weight_map:
            raise CompatibilityError(f"No safetensors checkpoint found in {path}")
        for shard in set(self.weight_map.values()):
            resolved = (self.path / shard).resolve()
            if not resolved.is_relative_to(self.path.resolve()) or not resolved.is_file():
                raise CompatibilityError(f"Missing or unsafe checkpoint shard: {shard}")

    def get_tensor(self, name):
        with safe_open(self.path / self.weight_map[name], framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)
