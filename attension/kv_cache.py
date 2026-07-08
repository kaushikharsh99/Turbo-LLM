import torch
from typing import Dict, Tuple

from kivi.quant import (
    quantize_k,
    quantize_v,
    unpack_and_dequant_kcache,
    unpack_and_dequant_vcache
)


class KVCache:
    """Production-style Key-Value cache stored on GPU/CPU/MPS with optional KIVI quantization."""

    def __init__(self, device_manager=None, k_bits=None, v_bits=None, group_size=32, residual_length=32):
        # Maps layer_id -> k_cache (torch.Tensor) for standard mode
        self.k_caches: Dict[int, torch.Tensor] = {}
        self.v_caches: Dict[int, torch.Tensor] = {}
        # Maps layer_id -> conv_state/recurrent_state
        self.conv_states: Dict[int, torch.Tensor] = {}
        self.recurrent_states: Dict[int, torch.Tensor] = {}

        self.device_manager = device_manager
        if device_manager is not None:
            self.device = device_manager.device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # KIVI settings
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.group_size = group_size
        self.residual_length = residual_length

        # KIVI storage mapping: layer_id -> quantized components
        self.k_codes: Dict[int, torch.Tensor] = {}
        self.k_scales: Dict[int, torch.Tensor] = {}
        self.k_mins: Dict[int, torch.Tensor] = {}

        self.v_codes: Dict[int, torch.Tensor] = {}
        self.v_scales: Dict[int, torch.Tensor] = {}
        self.v_mins: Dict[int, torch.Tensor] = {}

        # Residuals in full precision
        self.k_residuals: Dict[int, torch.Tensor] = {}
        self.v_residuals: Dict[int, torch.Tensor] = {}

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

        if k.numel() == 0 or v.numel() == 0:
            # Handle empty tensors safely
            return self.get(layer_id)

        # Check if KIVI is enabled
        if self.k_bits is not None or self.v_bits is not None:
            # -------------------------------------------------------------
            # KIVI mode
            # -------------------------------------------------------------
            # Transpose sequence and head dimensions to match KIVI's layout:
            # (B, T, nh, D) -> (B, nh, T, D)
            k_kivi = k.transpose(1, 2).contiguous()
            v_kivi = v.transpose(1, 2).contiguous()

            # --- Key Cache Processing ---
            if self.k_bits is not None:
                k_residual_existing = self.k_residuals.get(layer_id, None)
                if k_residual_existing is not None:
                    k_full = torch.cat((k_residual_existing, k_kivi), dim=2)
                else:
                    k_full = k_kivi

                L_total = k_full.shape[2]
                if L_total <= self.residual_length:
                    self.k_residuals[layer_id] = k_full
                else:
                    quant_len = (L_total - self.residual_length) - ((L_total - self.residual_length) % self.group_size)
                    if quant_len > 0:
                        k_to_quant = k_full[:, :, :quant_len, :].contiguous()
                        code_new, scale_new, mn_new = quantize_k(
                            k_to_quant, self.group_size, self.k_bits, use_triton=(self.device.type == "cuda")
                        )

                        if layer_id in self.k_codes:
                            self.k_codes[layer_id] = torch.cat((self.k_codes[layer_id], code_new), dim=2)
                            self.k_scales[layer_id] = torch.cat((self.k_scales[layer_id], scale_new), dim=2)
                            self.k_mins[layer_id] = torch.cat((self.k_mins[layer_id], mn_new), dim=2)
                        else:
                            self.k_codes[layer_id] = code_new
                            self.k_scales[layer_id] = scale_new
                            self.k_mins[layer_id] = mn_new

                        self.k_residuals[layer_id] = k_full[:, :, quant_len:, :].contiguous()
                    else:
                        self.k_residuals[layer_id] = k_full
            else:
                # Key is not quantized, append in full-precision to standard cache representation
                if layer_id not in self.k_caches:
                    self.k_caches[layer_id] = k
                else:
                    self.k_caches[layer_id] = torch.cat((self.k_caches[layer_id], k), dim=1)

            # --- Value Cache Processing ---
            if self.v_bits is not None:
                v_residual_existing = self.v_residuals.get(layer_id, None)
                if v_residual_existing is not None:
                    v_full = torch.cat((v_residual_existing, v_kivi), dim=2)
                else:
                    v_full = v_kivi

                L_total = v_full.shape[2]
                if L_total <= self.residual_length:
                    self.v_residuals[layer_id] = v_full
                else:
                    quant_len = L_total - self.residual_length
                    if quant_len > 0:
                        v_to_quant = v_full[:, :, :quant_len, :].contiguous()
                        code_new, scale_new, mn_new = quantize_v(
                            v_to_quant, self.group_size, self.v_bits, use_triton=(self.device.type == "cuda")
                        )

                        if layer_id in self.v_codes:
                            self.v_codes[layer_id] = torch.cat((self.v_codes[layer_id], code_new), dim=2)
                            self.v_scales[layer_id] = torch.cat((self.v_scales[layer_id], scale_new), dim=2)
                            self.v_mins[layer_id] = torch.cat((self.v_mins[layer_id], mn_new), dim=2)
                        else:
                            self.v_codes[layer_id] = code_new
                            self.v_scales[layer_id] = scale_new
                            self.v_mins[layer_id] = mn_new

                        self.v_residuals[layer_id] = v_full[:, :, quant_len:, :].contiguous()
                    else:
                        self.v_residuals[layer_id] = v_full
            else:
                # Value is not quantized, append in full-precision to standard cache representation
                if layer_id not in self.v_caches:
                    self.v_caches[layer_id] = v
                else:
                    self.v_caches[layer_id] = torch.cat((self.v_caches[layer_id], v), dim=1)

            # Reconstruct and return the full cache
            return self.get(layer_id)

        else:
            # -------------------------------------------------------------
            # Standard mode
            # -------------------------------------------------------------
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
        if self.k_bits is not None or self.v_bits is not None:
            # --- Key Cache Retrieval ---
            if self.k_bits is not None:
                if layer_id not in self.k_residuals:
                    raise KeyError(f"No KV cache initialized for layer {layer_id}")
                if layer_id in self.k_codes:
                    k_dequant = unpack_and_dequant_kcache(
                        self.k_codes[layer_id],
                        self.k_scales[layer_id],
                        self.k_mins[layer_id],
                        self.group_size,
                        self.k_bits,
                        dtype=self.k_residuals[layer_id].dtype
                    )
                    k_reconstruct = torch.cat((k_dequant, self.k_residuals[layer_id]), dim=2)
                else:
                    k_reconstruct = self.k_residuals[layer_id]
                # Transpose back: (B, nh, T, D) -> (B, T, nh, D)
                k_out = k_reconstruct.transpose(1, 2).contiguous()
            else:
                if layer_id not in self.k_caches:
                    raise KeyError(f"No KV cache initialized for layer {layer_id}")
                k_out = self.k_caches[layer_id]

            # --- Value Cache Retrieval ---
            if self.v_bits is not None:
                if layer_id not in self.v_residuals:
                    raise KeyError(f"No KV cache initialized for layer {layer_id}")
                if layer_id in self.v_codes:
                    v_dequant = unpack_and_dequant_vcache(
                        self.v_codes[layer_id],
                        self.v_scales[layer_id],
                        self.v_mins[layer_id],
                        self.group_size,
                        self.v_bits,
                        dtype=self.v_residuals[layer_id].dtype
                    )
                    v_reconstruct = torch.cat((v_dequant, self.v_residuals[layer_id]), dim=2)
                else:
                    v_reconstruct = self.v_residuals[layer_id]
                # Transpose back: (B, nh, T, D) -> (B, T, nh, D)
                v_out = v_reconstruct.transpose(1, 2).contiguous()
            else:
                if layer_id not in self.v_caches:
                    raise KeyError(f"No KV cache initialized for layer {layer_id}")
                v_out = self.v_caches[layer_id]

            return k_out, v_out

        else:
            if layer_id not in self.k_caches:
                raise KeyError(f"No KV cache initialized for layer {layer_id}")
            return self.k_caches[layer_id], self.v_caches[layer_id]

    def clear(self):
        """Clears all cached tensors and releases GPU/accelerator memory."""
        # Standard caches
        for cache in self.k_caches.values():
            del cache
        for cache in self.v_caches.values():
            del cache
        self.k_caches.clear()
        self.v_caches.clear()

        # KIVI caches
        for cache in self.k_codes.values():
            del cache
        for cache in self.k_scales.values():
            del cache
        for cache in self.k_mins.values():
            del cache
        self.k_codes.clear()
        self.k_scales.clear()
        self.k_mins.clear()

        for cache in self.v_codes.values():
            del cache
        for cache in self.v_scales.values():
            del cache
        for cache in self.v_mins.values():
            del cache
        self.v_codes.clear()
        self.v_scales.clear()
        self.v_mins.clear()

        # Residuals
        for cache in self.k_residuals.values():
            del cache
        for cache in self.v_residuals.values():
            del cache
        self.k_residuals.clear()
        self.v_residuals.clear()

        # State buffers
        for state in self.conv_states.values():
            del state
        for state in self.recurrent_states.values():
            del state
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
        if self.k_bits is not None:
            if layer_id in self.k_residuals:
                # In KIVI, sequence length is the sequence dimension of residuals + sequence dimension of codes
                res_len = self.k_residuals[layer_id].shape[2]
                code_len = 0
                if layer_id in self.k_codes:
                    # code shape is (B, nh, T_quant // (32 // bits), D), we need to unpack to get T_quant
                    pack_factor = 32 // self.k_bits
                    code_len = self.k_codes[layer_id].shape[2] * pack_factor
                return res_len + code_len
            return 0
        else:
            if layer_id in self.k_caches:
                return self.k_caches[layer_id].shape[1]
            return 0