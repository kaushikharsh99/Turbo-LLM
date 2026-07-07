from collections import OrderedDict
import torch
from typing import Optional, Dict, List, Any


class GPUMemory:
    """GPU cache for resident tensors (VRAM) with LRU eviction."""

    def __init__(self, max_bytes: int | float):
        self.max_bytes = max_bytes
        self.current_bytes = 0
        # Map tensor name -> GPU torch.Tensor
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        # Map tensor name -> Tensor metadata object
        self.registry: Dict[str, Any] = {}

    def exists(self, name: str) -> bool:
        """Checks if the tensor is resident in GPU memory."""
        return name in self.cache

    def get(self, name: str) -> Optional[torch.Tensor]:
        """Gets a tensor from the GPU cache and updates its LRU position."""
        if name in self.cache:
            self.cache.move_to_end(name)
            return self.cache[name]
        return None

    def add(self, name: str, data: torch.Tensor, metadata_obj: Any) -> List[str]:
        """
        Adds a tensor to the GPU cache.
        Evicts least recently used (LRU) tensors if memory limit is exceeded.
        Returns a list of evicted tensor names.
        """
        tensor_bytes = metadata_obj.size_bytes
        evicted_names: List[str] = []

        if name in self.cache:
            self.cache.move_to_end(name)
            self.current_bytes -= tensor_bytes

        # Evict LRU items if we are over the memory limit
        while self.cache and self.current_bytes + tensor_bytes > self.max_bytes:
            evict_name, evict_data = self.cache.popitem(last=False)
            evict_meta = self.registry.pop(evict_name)
            self.current_bytes -= evict_meta.size_bytes
            del evict_data  # Free CUDA tensor reference explicitly
            evicted_names.append(evict_name)

        self.cache[name] = data
        self.registry[name] = metadata_obj
        self.current_bytes += tensor_bytes

        return evicted_names

    def remove(self, name: str, empty_cache: bool = False) -> bool:
        """Removes a tensor from the GPU cache. Returns True if found and removed."""
        if name in self.cache:
            data = self.cache.pop(name)
            meta = self.registry.pop(name)
            self.current_bytes -= meta.size_bytes
            del data
            if empty_cache:
                torch.cuda.empty_cache()
            return True
        return False

    def clear(self):
        """Clears all resident GPU tensors and empties PyTorch's CUDA cache."""
        self.cache.clear()
        self.registry.clear()
        self.current_bytes = 0
        torch.cuda.empty_cache()
