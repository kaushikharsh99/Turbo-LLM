import torch
from typing import Tuple


class RoPE:
    """Qwen3.5/3.6-MoE Text Rotary Position Embeddings (M-RoPE)."""

    def __init__(self, config):
        self.head_dim = config.head_dim
        self.partial_rotary_factor = config.partial_rotary_factor
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        self.base = config.rope_theta
        self.mrope_section = [11, 11, 10]  # default section sizes for 32 freq elements

        # Compute inv_freq in double precision to match HF
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float64) / self.rotary_dim)
        )
        self.inv_freq = inv_freq.float()

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half of the hidden dimensions."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_interleaved_mrope(self, freqs, mrope_section):
        freqs_t = freqs[0].clone()  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def apply(
        self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies M-RoPE to query and key tensors.
        q shape: (batch_size, seq_len, num_heads, head_dim)
        k shape: (batch_size, seq_len, num_kv_heads, head_dim)
        position_ids shape: (batch_size, seq_len)
        """
        device = q.device
        dtype = q.dtype

        # Expand position_ids to 3D: (3, batch_size, seq_len)
        if position_ids.ndim == 2:
            position_ids_expanded = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        else:
            position_ids_expanded = position_ids

        # Ensure inv_freq is on the correct device and dtype
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)

        # Expand for matrix multiplication
        inv_freq_expanded = inv_freq[None, None, :, None].expand(3, position_ids_expanded.shape[1], -1, 1)
        position_ids_expanded = position_ids_expanded[:, :, None, :].float()  # shape (3, bs, 1, seq_len)

        # Compute freqs: shape (3, bs, seq_len, rotary_dim // 2)
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)

        # Apply interleaved mrope
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)

        # Concatenate frequencies to cover full rotary dimension: shape (bs, seq_len, rotary_dim)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Compute cos and sin in float32 then cast back
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)

        # Broadcast shapes: (batch_size, seq_len, 1, rotary_dim)
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

        # Apply rotary embedding to active dims
        rotary_dim = cos.shape[-1]
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

        q_embed = (q_rot * cos) + (self._rotate_half(q_rot) * sin)
        k_embed = (k_rot * cos) + (self._rotate_half(k_rot) * sin)

        q_out = torch.cat([q_embed, q_pass], dim=-1)
        k_out = torch.cat([k_embed, k_pass], dim=-1)

        return q_out, k_out
