import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from model.layer import Layer
from attension.attension import dequantize_weight, rms_norm
from attension.kv_cache import KVCache

# Import the PyTorch fallbacks from transformers
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)


class RMSNormGated:
    """RMSNorm with gating activation (gated SwiGLU-style normalization)."""

    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        self.weight = weight
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


class LinearAttention:
    """Computes hybrid linear causal self-attention (Gated DeltaNet) layers."""

    def __init__(self, config):
        self.config = config
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.num_v_heads = config.linear_num_value_heads
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

    def forward(
        self,
        hidden: torch.Tensor,
        layer: Layer,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        """
        Runs the forward pass for Gated DeltaNet linear attention.
        hidden shape: (batch_size, seq_len, hidden_size)
        Returns:
            output shape: (batch_size, seq_len, hidden_size)
        """
        if (
            layer.linear_attn is None
            or layer.linear_attn.conv1d is None
            or layer.linear_attn.conv1d.data is None
        ):
            return hidden

        batch_size, seq_len, hidden_size = hidden.shape
        dtype = hidden.dtype

        la = layer.linear_attn

        # 1. Dequantize FP8 projection weights
        w_in_qkv = dequantize_weight(
            la.in_proj_qkv.data,
            la.in_proj_qkv_scale.data if la.in_proj_qkv_scale else None,
            dtype,
        )
        w_in_z = dequantize_weight(
            la.in_proj_z.data,
            la.in_proj_z_scale.data if la.in_proj_z_scale else None,
            dtype,
        )
        w_out = dequantize_weight(
            la.out_proj.data,
            la.out_proj_scale.data if la.out_proj_scale else None,
            dtype,
        )

        # Other BF16 parameter weights (not quantized to FP8)
        w_in_b = la.in_proj_b.data.to(dtype)
        w_in_a = la.in_proj_a.data.to(dtype)
        w_conv1d = la.conv1d.data
        
        A_log = la.A_log.data
        dt_bias = la.dt_bias.data
        norm_weight = la.norm.data.to(dtype)

        # 1.5. Input RMSNorm
        if layer.attention_norm and layer.attention_norm.data is not None:
            normed_hidden = rms_norm(hidden, layer.attention_norm.data, self.config.rms_norm_eps)
        else:
            normed_hidden = hidden

        # 2. Project inputs
        mixed_qkv = torch.matmul(normed_hidden, w_in_qkv.t())  # (batch_size, seq_len, conv_dim)
        mixed_qkv = mixed_qkv.transpose(1, 2)  # (batch_size, conv_dim, seq_len)

        z = torch.matmul(normed_hidden, w_in_z.t())  # (batch_size, seq_len, value_dim)
        z = z.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        b = torch.matmul(normed_hidden, w_in_b.t())  # (batch_size, seq_len, num_v_heads)
        a = torch.matmul(normed_hidden, w_in_a.t())  # (batch_size, seq_len, num_v_heads)

        # 3. Causal Conv1D
        layer_id = layer.layer_id
        use_cache = kv_cache is not None

        if use_cache and seq_len == 1 and layer_id in kv_cache.conv_states:
            # Shift and append to cached state
            conv_state = kv_cache.conv_states[layer_id]  # (batch_size, conv_dim, kernel_size - 1)
            new_conv_state = torch.cat([conv_state, mixed_qkv], dim=-1)  # (batch_size, conv_dim, kernel_size)
            kv_cache.conv_states[layer_id] = new_conv_state[:, :, 1:]
            
            # depthwise grouped conv1d
            mixed_qkv = (new_conv_state * w_conv1d.squeeze(1)).sum(dim=-1, keepdim=True)  # (batch_size, conv_dim, 1)
            mixed_qkv = F.silu(mixed_qkv)

        else:
            # Full sequence (prefill) conv1d
            if use_cache:
                if seq_len >= self.conv_kernel_size - 1:
                    kv_cache.conv_states[layer_id] = mixed_qkv[:, :, -(self.conv_kernel_size - 1):]
                else:
                    kv_cache.conv_states[layer_id] = F.pad(mixed_qkv, (self.conv_kernel_size - 1 - seq_len, 0))

            conv_out = F.conv1d(
                mixed_qkv,
                w_conv1d,
                bias=None,
                stride=1,
                padding=self.conv_kernel_size - 1,
                groups=self.conv_dim,
            )
            conv_out = conv_out[:, :, :seq_len]
            mixed_qkv = F.silu(conv_out)

        mixed_qkv = mixed_qkv.transpose(1, 2)  # (batch_size, seq_len, conv_dim)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        beta = torch.sigmoid(b)
        
        # Discretized continuous state representation (using softplus & exp)
        g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
        g = g.to(dtype)

        # GQA matching (repeat heads)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        # 4. Gated Delta Rule Attention calculation
        recurrent_state = kv_cache.recurrent_states.get(layer_id, None) if use_cache else None

        if use_cache and seq_len == 1:
            core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state if (use_cache and recurrent_state is not None) else None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            
        if use_cache:
            kv_cache.recurrent_states[layer_id] = last_recurrent_state

        # 5. Normalization & Output projection
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)

        norm = RMSNormGated(norm_weight, self.config.rms_norm_eps)
        core_attn_out = norm.forward(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, self.value_dim)

        output = torch.matmul(core_attn_out, w_out.t())
        return hidden + output
