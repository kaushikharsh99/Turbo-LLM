import torch
from typing import Dict, Tuple


class KVCache:
        """Production-style Key-Value cache stored on GPU."""

        def __init__(self, device_manager=None):
            # Maps layer_id -> k_cache (torch.Tensor)
            self.k_caches: Dict[int, torch.Tensor] = {}
            # Maps layer_id -> v_cache (torch.Tensor)
            self.v_caches: Dict[int, torch.Tensor] = {}
            # Maps layer_id -> conv_state (torch.Tensor)
            self.conv_states: Dict[int, torch.Tensor] = {}
            # Maps layer_id -> recurrent_state (torch.Tensor)
            self.recurrent_states: Dict[int, torch.Tensor] = {}
            self.device_manager = device_manager
            if device_manager is not None:
                self.device = device_manager.device
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def append(
            self, layer_id: int, k: torch.Tensor, v: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Appends new keys and values to the cache for a given layer.
            k, v shape: (batch_size, seq_len, num_kv_heads, head_dim)
            Returns the full concatenated (k_cache, v_cache) tensors.
            """
            # Ensure caches reside on active device
            k = k.to(self.device)
            v = v.to(self.device)

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
            """Clears all cached tensors and releases GPU memory."""

            # Explicitly delete cached tensors
            for cache in self.k_caches.values():
                del cache

            for cache in self.v_caches.values():
                del cache

            for state in self.conv_states.values():
                del state

            for state in self.recurrent_states.values():
                del state

            # Remove dictionary entries
            self.k_caches.clear()
            self.v_caches.clear()
            self.conv_states.clear()
            self.recurrent_states.clear()

            # Release unused cached memory
            if self.device_manager is not None:
                self.device_manager.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

        def reset(self):
            """Resets the KV cache (alias for clear)."""
            self.clear()

        def sequence_length(self, layer_id: int) -> int:
            """Returns the current cached sequence length for a given layer."""
            if layer_id in self.k_caches:
                return self.k_caches[layer_id].shape[1]
            return 0