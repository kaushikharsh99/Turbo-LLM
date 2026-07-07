import math
import torch
from typing import Optional

from model.layer import Layer
from attension.rope import RoPE
from attension.kv_cache import KVCache


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standard Root Mean Square Normalization (RMSNorm)."""
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats keys/values heads for Grouped Query Attention (GQA).
    x shape: (batch_size, num_kv_heads, seq_len, head_dim)
    Returns shape: (batch_size, num_kv_heads * n_rep, seq_len, head_dim)
    """
    if n_rep == 1:
        return x
    batch_size, n_kv_heads, seq_len, head_dim = x.shape
    # Insert dimension, expand, and flatten back
    x = x.unsqueeze(2)  # (batch_size, n_kv_heads, 1, seq_len, head_dim)
    x = x.expand(batch_size, n_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch_size, n_kv_heads * n_rep, seq_len, head_dim)


def dequantize_weight(
    weight: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Casts weight to target_dtype and dequantizes using scale_inv if present."""
    if weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        w_float = weight.to(target_dtype)
        if scale is not None:
            scale_casted = scale.to(target_dtype)
            if len(scale_casted.shape) == 2:
                # 2D Block-wise dequantization (Qwen style)
                row_block_size = weight.shape[0] // scale_casted.shape[0]
                col_block_size = weight.shape[1] // scale_casted.shape[1]
                scale_expanded = scale_casted.repeat_interleave(
                    row_block_size, dim=0
                ).repeat_interleave(col_block_size, dim=1)
                w_float = w_float * scale_expanded
            else:
                # 1D Channel-wise dequantization
                w_float = w_float * scale_casted.view(-1, 1)
        return w_float
    elif weight.dtype != target_dtype:
        return weight.to(target_dtype)
    return weight


class Attention:
    """Attention computation subsystem for full causal self-attention layers."""

    def __init__(self, config):
        self.config = config
        self.rope = RoPE(config)
        self.eps = config.rms_norm_eps

    def forward(
        self,
        hidden: torch.Tensor,
        layer: Layer,
        kv_cache: KVCache,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Runs the attention forward pass on hidden state.
        hidden shape: (batch_size, seq_len, hidden_size)
        position_ids shape: (batch_size, seq_len)
        Returns shape: (batch_size, seq_len, hidden_size)
        """
        # If this is not a full attention layer (e.g. hybrid linear attention layer),
        # return the original hidden state for now.
        if (
            layer.attention is None
            or layer.attention.q_proj is None
            or layer.attention.q_proj.data is None
        ):
            return hidden

        batch_size, seq_len, hidden_size = hidden.shape

        # 1. Input RMSNorm
        normed_hidden = rms_norm(hidden, layer.attention_norm.data, self.eps)

        # 2. Query, Key, Value projections
        # Dequantize weight tensors first
        w_q = dequantize_weight(
            layer.attention.q_proj.data,
            layer.attention.q_scale.data if layer.attention.q_scale else None,
            normed_hidden.dtype,
        )
        w_k = dequantize_weight(
            layer.attention.k_proj.data,
            layer.attention.k_scale.data if layer.attention.k_scale else None,
            normed_hidden.dtype,
        )
        w_v = dequantize_weight(
            layer.attention.v_proj.data,
            layer.attention.v_scale.data if layer.attention.v_scale else None,
            normed_hidden.dtype,
        )

        q = torch.matmul(normed_hidden, w_q.t())
        k = torch.matmul(normed_hidden, w_k.t())
        v = torch.matmul(normed_hidden, w_v.t())

        # Reshape to (batch_size, seq_len, num_heads, head_dim)
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim

        q = q.view(batch_size, seq_len, num_heads, head_dim)
        k = k.view(batch_size, seq_len, num_kv_heads, head_dim)
        v = v.view(batch_size, seq_len, num_kv_heads, head_dim)

        # 3. Apply RoPE
        q, k = self.rope.apply(q, k, position_ids)

        # 4. Update KV Cache
        k, v = kv_cache.append(layer.layer_id, k, v)

        # 5. Grouped Query Attention (GQA) matching
        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Repeat KV heads for GQA/MQA
        n_rep = num_heads // num_kv_heads
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)

        # 6. Attention Scores computation
        # scores shape: (batch_size, num_heads, seq_len, kv_seq_len)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)

        # 7. Apply Causal Mask
        kv_seq_len = k.shape[2]
        if seq_len > 1:
            mask = torch.full(
                (seq_len, kv_seq_len), float("-inf"), device=scores.device
            )
            mask = torch.triu(mask, diagonal=kv_seq_len - seq_len + 1)
            scores = scores + mask.unsqueeze(0).unsqueeze(1)

        # 8. Softmax
        attn_probs = torch.softmax(scores, dim=-1)

        # 9. Compute context
        # context shape: (batch_size, num_heads, seq_len, head_dim)
        context = torch.matmul(attn_probs, v)

        # 10. Reshape back & project output
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, num_heads * head_dim)

        w_o = dequantize_weight(
            layer.attention.o_proj.data,
            layer.attention.o_scale.data if layer.attention.o_scale else None,
            context.dtype,
        )
        attn_output = torch.matmul(context, w_o.t())

        # 11. Add Residual Connection
        return hidden + attn_output
