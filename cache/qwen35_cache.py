from collections import defaultdict
import torch

from .base_cache import BaseCache


class LinearLayerState:

    def __init__(self):
        # Linear Attention
        self.conv_states = None
        self.recurrent_states = None

        # Transformer Attention
        self.key_states = None
        self.value_states = None


class Qwen35Cache(BaseCache):

    def __init__(self, num_layers, max_seq_len=None):
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.layers = defaultdict(LinearLayerState)

    def has_previous_state(self, layer_idx):

        layer = self.layers[layer_idx]

        return (
            layer.conv_states is not None
            and layer.recurrent_states is not None
        )

    def update_conv_state(self, state, layer_idx):

        self.layers[layer_idx].conv_states = state

    def update_recurrent_state(self, state, layer_idx):

        self.layers[layer_idx].recurrent_states = state

    def update(
        self,
        key_states,
        value_states,
        layer_idx,
    ):
        layer = self.layers[layer_idx]

        if layer.key_states is None:
            layer.key_states = key_states
            layer.value_states = value_states
        else:
            layer.key_states = torch.cat(
                [layer.key_states, key_states],
                dim=2,
            )
            layer.value_states = torch.cat(
                [layer.value_states, value_states],
                dim=2,
            )

        return layer.key_states, layer.value_states

    def get(self, layer_idx):

        layer = self.layers[layer_idx]

        return (
            layer.key_states,
            layer.value_states,
        )

    def clear(self):

        for layer in self.layers.values():
            layer.conv_states = None
            layer.recurrent_states = None
            layer.key_states = None
            layer.value_states = None

        self.layers.clear()

    def get_memory_bytes(self):

        total = 0

        for layer in self.layers.values():

            # Linear Attention cache
            if layer.conv_states is not None:
                total += (
                    layer.conv_states.numel()
                    * layer.conv_states.element_size()
                )

            if layer.recurrent_states is not None:
                total += (
                    layer.recurrent_states.numel()
                    * layer.recurrent_states.element_size()
                )

            # Transformer KV cache
            if layer.key_states is not None:
                total += (
                    layer.key_states.numel()
                    * layer.key_states.element_size()
                )

            if layer.value_states is not None:
                total += (
                    layer.value_states.numel()
                    * layer.value_states.element_size()
                )

        return total

    def get_allocated_bytes(self):
        return self.get_memory_bytes()
    
    def get_seq_length(self, layer_idx=0):
        return 0
    
    def get_mask_sizes(self, query_length, layer_idx):
        """
        Returns (kv_length, kv_offset) for HuggingFace mask creation.
        """
        if self.has_previous_state(layer_idx):
            return query_length + 1, 0

        return query_length, 0
    
