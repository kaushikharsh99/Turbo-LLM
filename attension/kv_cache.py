import torch
from typing import Dict, Tuple


class KVCache:
    """Production-style Key-Value cache stored on GPU."""

    def __init__(self):
        # Maps layer_id -> k_cache (torch.Tensor)
        self.k_caches: Dict[int, torch.Tensor] = {}
        # Maps layer_id -> v_cache (torch.Tensor)
        self.v_caches: Dict[int, torch.Tensor] = {}

    def append(
        self, layer_id: int, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Appends new keys and values to the cache for a given layer.
        k, v shape: (batch_size, seq_len, num_kv_heads, head_dim)
        Returns the full concatenated (k_cache, v_cache) tensors.
        """
        # Ensure caches reside on GPU
        k = k.cuda()
        v = v.cuda()

        if layer_id not in self.k_caches:
            self.k_caches[layer_id] = k
            self.v_caches[layer_id] = v
        else:
            # Concatenate along the sequence dimension (dim 1)
            self.k_caches[layer_id] = torch.cat(
                (self.k_caches[layer_id], k), dim=1
            )
            self.v_caches[layer_id] = torch.cat(
                (self.v_caches[layer_id], v), dim=1
            )

        return self.k_caches[layer_id], self.v_caches[layer_id]

    def get(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the full cached k and v tensors for the layer."""
        if layer_id not in self.k_caches:
            raise KeyError(f"No KV cache initialized for layer {layer_id}")
        return self.k_caches[layer_id], self.v_caches[layer_id]

    def clear(self):
        """Clears all cached tensors."""
        self.k_caches.clear()
        self.v_caches.clear()

    def reset(self):
        """Resets the KV cache (alias for clear)."""
        self.clear()

    def sequence_length(self, layer_id: int) -> int:
        """Returns the current cached sequence length for a given layer."""
        if layer_id in self.k_caches:
            return self.k_caches[layer_id].shape[1]
        return 0
