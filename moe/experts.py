import torch
import torch.nn.functional as F
from typing import Dict, Any
from attension.attension import dequantize_weight

class MoEExperts:
    def __init__(self, config):
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        experts: list,  # list of Expert objects from layer.moe.experts
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        hidden_states shape: (batch_size, seq_len, hidden_size)
        experts: list of Expert metadata objects
        topk_weights shape: (batch_size, seq_len, k)
        topk_indices shape: (batch_size, seq_len, k)
        Returns:
            output shape: (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_tokens = batch_size * seq_len
        
        flat_hidden = hidden_states.view(num_tokens, hidden_size)
        flat_topk_weights = topk_weights.view(num_tokens, -1)
        flat_topk_indices = topk_indices.view(num_tokens, -1)
        
        final_output = torch.zeros_like(flat_hidden)
        dtype = flat_hidden.dtype
        
        # We iterate over the experts. Since experts is a list of all experts in this layer,
        # we can use the list index as the expert_id.
        for expert_id, expert in enumerate(experts):
            # Check if this expert has been loaded. If not, we skip it (Executor will load it).
            if expert.gate_proj is None or expert.gate_proj.data is None:
                continue
                
            # Find tokens that route to this expert
            mask = (flat_topk_indices == expert_id)
            if not mask.any():
                continue
            
            token_indices, k_indices = mask.nonzero(as_tuple=True)
            expert_input = flat_hidden[token_indices]
            
            # Dequantize weights
            w_gate = dequantize_weight(
                expert.gate_proj.data,
                expert.gate_scale.data if expert.gate_scale else None,
                dtype,
            )
            w_up = dequantize_weight(
                expert.up_proj.data,
                expert.up_scale.data if expert.up_scale else None,
                dtype,
            )
            w_down = dequantize_weight(
                expert.down_proj.data,
                expert.down_scale.data if expert.down_scale else None,
                dtype,
            )
            
            # SwiGLU FFN
            gate_out = torch.matmul(expert_input, w_gate.t())
            up_out = torch.matmul(expert_input, w_up.t())
            intermediate = F.silu(gate_out) * up_out
            expert_output = torch.matmul(intermediate, w_down.t())
            
            # Scale by routing weight
            routing_weight = flat_topk_weights[token_indices, k_indices].unsqueeze(-1)
            weighted_output = expert_output * routing_weight
            
            # Accumulate output
            final_output.index_add_(0, token_indices, weighted_output)
            
        return final_output.view(batch_size, seq_len, hidden_size)
