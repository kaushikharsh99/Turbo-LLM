import torch
from typing import Tuple


class RoPE:
    """Qwen-style Rotary Position Embeddings (RoPE)."""

    def __init__(self, config):
        self.head_dim = config.head_dim
        self.partial_rotary_factor = config.partial_rotary_factor
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        self.max_seq_len = config.max_position_embeddings
        self.base = config.rope_theta

        # Precompute inverse frequencies
        # shape: (rotary_dim // 2,)
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim)
        )

        # Precompute sin/cos tables
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        # shape: (max_seq_len, rotary_dim // 2)
        freqs = torch.outer(t, inv_freq)

        # Concatenate frequencies to cover full rotary dimension
        # shape: (max_seq_len, rotary_dim)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Register cos and sin buffers
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half of the hidden dimensions."""
        half_dim = x.shape[-1] // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat((-x2, x1), dim=-1)

    def apply(
        self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies RoPE to query and key tensors.
        q, k shapes: (batch_size, seq_len, num_heads, head_dim)
        position_ids shape: (batch_size, seq_len)
        """
        device = position_ids.device
        dtype = q.dtype

        # Ensure precomputed tables are on the same device/dtype
        self.cos_cached = self.cos_cached.to(device=device, dtype=dtype)
        self.sin_cached = self.sin_cached.to(device=device, dtype=dtype)

        # Lookup and expand dims for broadcasting
        # shape: (batch_size, seq_len, 1, rotary_dim)
        cos = self.cos_cached[position_ids].unsqueeze(-2)
        sin = self.sin_cached[position_ids].unsqueeze(-2)

        if self.rotary_dim < self.head_dim:
            # Rotate only the first rotary_dim dimensions
            q_rot = q[..., :self.rotary_dim]
            q_pass = q[..., self.rotary_dim:]
            k_rot = k[..., :self.rotary_dim]
            k_pass = k[..., self.rotary_dim:]

            q_rot = (q_rot * cos) + (self._rotate_half(q_rot) * sin)
            k_rot = (k_rot * cos) + (self._rotate_half(k_rot) * sin)

            q = torch.cat((q_rot, q_pass), dim=-1)
            k = torch.cat((k_rot, k_pass), dim=-1)
        else:
            q = (q * cos) + (self._rotate_half(q) * sin)
            k = (k * cos) + (self._rotate_half(k) * sin)

        return q, k
