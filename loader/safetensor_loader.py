from pathlib import Path
from typing import Dict, List

import torch
from safetensors import safe_open


class SafeTensorLoader:

    def __init__(self, model_dir: str | Path, weight_map: Dict[str, str]):
        self.model_dir = Path(model_dir)
        self.weight_map = weight_map

        
        self._handles = {}

    def _get_handle(self, filename: str):
        
        if filename not in self._handles:

            path = self.model_dir / filename

            if not path.exists():
                raise FileNotFoundError(path)

            self._handles[filename] = safe_open(
                str(path),
                framework="pt",
                device="cpu",
            )

        return self._handles[filename]

    def has_tensor(self, tensor_name: str) -> bool:
        return tensor_name in self.weight_map

    def tensor_file(self, tensor_name: str) -> str:
        if tensor_name not in self.weight_map:
            raise KeyError(f"Tensor '{tensor_name}' not found.")

        return self.weight_map[tensor_name]

    def get_tensor(self, tensor_name: str) -> torch.Tensor:
        filename = self.tensor_file(tensor_name)

        handle = self._get_handle(filename)

        return handle.get_tensor(tensor_name)

    def get_shape(self, tensor_name: str):
        filename = self.tensor_file(tensor_name)

        handle = self._get_handle(filename)

        return handle.get_slice(tensor_name).get_shape()

    def get_dtype(self, tensor_name: str):
        filename = self.tensor_file(tensor_name)

        handle = self._get_handle(filename)

        return handle.get_slice(tensor_name).get_dtype()

    def tensors_in_file(self, filename: str) -> List[str]:
        return [
            tensor
            for tensor, file in self.weight_map.items()
            if file == filename
        ]

    def opened_files(self) -> List[str]:
        return list(self._handles.keys())

    def close(self):
        for handle in self._handles.values():
            handle.close()

        self._handles.clear()

    def __contains__(self, tensor_name: str):
        return tensor_name in self.weight_map

    def __len__(self):
        return len(self.weight_map)