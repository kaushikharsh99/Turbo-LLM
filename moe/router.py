import torch
import torch.nn.functional as F
from typing import Tuple

class MoERouter:
    def __init__(self, config):
        self.config = config
        self.num_experts_per_tok = config.experts_per_token

    def forward(self, hidden_states: torch.Tensor, gate_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        hidden_states shape: (batch_size, seq_len, hidden_size)
        gate_weight shape: (num_experts, hidden_size)
        Returns:
            topk_weights: (batch_size, seq_len, k)
            topk_indices: (batch_size, seq_len, k)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_size)
        
        logits = torch.matmul(flat_hidden, gate_weight.t())
        
        # Softmax to get probabilities
        probs = F.softmax(
            logits.float(),
            dim=-1,
        ).to(logits.dtype)
        
        # Select Top-K
        topk_weights, topk_indices = torch.topk(probs, k=self.num_experts_per_tok, dim=-1)
        
        # Re-normalize weights
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # Reshape back
        topk_weights = topk_weights.view(batch_size, seq_len, -1)
        topk_indices = topk_indices.view(batch_size, seq_len, -1)
        
        return topk_weights, topk_indices
