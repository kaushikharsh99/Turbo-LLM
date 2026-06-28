import torch
from .base_cache import BaseCache


class KVCache(BaseCache):
    """
    Pre-allocated KV cache with O(1) per-token append.

    Instead of torch.cat (which copies the entire history every token),
    this allocates a fixed-size buffer up front and writes into it via
    slice assignment. Returns views of the valid portion — zero copies.

    Buffers are lazily allocated per-layer on first update(), so shapes
    (batch, num_kv_heads, head_dim, dtype) are inferred automatically.
    """
    def __init__(self, max_seq_len=4096, cache_type="standard"):
        self.max_seq_len = max_seq_len
        self.cache_type = cache_type

        self._keys = {}
        self._values = {}
        self._seq_lengths = {}

    # ------------------------------------------------------------------
    # HF-compatible interface (called by self_attn.forward)
    # ------------------------------------------------------------------

    def update(self, key_states, value_states, layer_idx):
        """
        Appends new KV states and returns the full accumulated states.
        Called by HuggingFace attention: past_key_values.update(k, v, layer_idx)

        key_states, value_states: [batch, num_kv_heads, new_seq_len, head_dim]
        Returns: (all_keys, all_values) as views — no copy.
        """
        if layer_idx not in self._seq_lengths:
            self._allocate_layer(layer_idx, key_states)

        start = self._seq_lengths[layer_idx]
        new_seq = key_states.shape[2]
        end = start + new_seq

        assert end <= self.max_seq_len, (
            f"KV cache overflow: tried to store {end} tokens but max_seq_len={self.max_seq_len}. "
            f"Increase max_seq_len or reduce prompt + max_new_tokens."
        )

        # O(1) write — just copy the new tokens into the pre-allocated slot
        self._keys[layer_idx][:, :, start:end, :] = key_states
        self._values[layer_idx][:, :, start:end, :] = value_states
        self._seq_lengths[layer_idx] = end

        # Return views of the valid portion (no allocation, no copy)
        return self._keys[layer_idx][:, :, :end, :], self._values[layer_idx][:, :, :end, :]

    # ------------------------------------------------------------------
    # Mask interface (called by create_causal_mask → _preprocess_mask_arguments)
    # ------------------------------------------------------------------

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the number of cached tokens for the given layer."""
        return self._seq_lengths.get(layer_idx, 0)

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        """
        Returns (kv_length, kv_offset) for causal mask construction.
        kv_length = cached tokens + new query tokens.
        """
        seq_len = self._seq_lengths.get(layer_idx, 0)
        if seq_len == 0:
            return query_length, 0
        return seq_len + query_length, 0

    # ------------------------------------------------------------------
    # Query / utility
    # ------------------------------------------------------------------

    def get(self, layer_idx):
        """Returns (key_view, value_view) for the valid portion, or (None, None)."""
        if layer_idx not in self._seq_lengths or self._seq_lengths[layer_idx] == 0:
            return None, None
        end = self._seq_lengths[layer_idx]
        return self._keys[layer_idx][:, :, :end, :], self._values[layer_idx][:, :, :end, :]

    def get_memory_bytes(self) -> int:
        """Returns bytes used by the valid (filled) portion of the KV cache."""
        total = 0
        for layer_idx, length in self._seq_lengths.items():
            if length > 0:
                k = self._keys[layer_idx]
                batch, heads, _, dim = k.shape
                # K + V, each: batch * heads * length * dim * element_size
                total += batch * heads * length * dim * k.element_size() * 2
        return total

    def get_allocated_bytes(self) -> int:
        """Returns total bytes allocated (including unused buffer space)."""
        total = 0
        for k in self._keys.values():
            total += k.element_size() * k.nelement()
        for v in self._values.values():
            total += v.element_size() * v.nelement()
        return total

    def clear(self):
        """Frees all buffers."""
        self._keys.clear()
        self._values.clear()
        self._seq_lengths.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _allocate_layer(self, layer_idx, reference_tensor):
        """Lazily allocate buffer for one layer, inferring shape from actual tensor."""
        batch, heads, _, dim = reference_tensor.shape
        dtype = reference_tensor.dtype
        device = reference_tensor.device

        self._keys[layer_idx] = torch.zeros(
            batch, heads, self.max_seq_len, dim,
            dtype=dtype, device=device
        )
        self._values[layer_idx] = torch.zeros(
            batch, heads, self.max_seq_len, dim,
            dtype=dtype, device=device
        )
        self._seq_lengths[layer_idx] = 0
    @property
    def is_empty(self):
        return len(self._seq_lengths) == 0
    
    def reset_layer(self, layer_idx):
        if layer_idx in self._seq_lengths:
            self._seq_lengths[layer_idx] = 0