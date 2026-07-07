import torch
import torch.nn.functional as F
from attension.attension import dequantize_weight

class MoEExperts:
    def __init__(self, config):
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        experts: list, 
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_tokens = batch_size * seq_len
        
        flat_hidden = hidden_states.reshape(num_tokens, hidden_size)
        flat_topk_weights = topk_weights.reshape(num_tokens, -1)
        flat_topk_indices = topk_indices.reshape(num_tokens, -1)
        
        final_output = torch.zeros_like(flat_hidden)
        dtype = flat_hidden.dtype

        dispatch_table = {}

        _, top_k = flat_topk_indices.shape

        for token_idx in range(num_tokens):
            for routing_slot in range(top_k):

                expert_id = int(flat_topk_indices[token_idx, routing_slot])

                dispatch_table.setdefault(expert_id, []).append(
                    (token_idx, routing_slot)
                )

        for expert_id, routed_tokens in dispatch_table.items():

            expert = experts[expert_id]

            if expert.gate_proj is None or expert.gate_proj.data is None:
                continue

            token_indices = torch.tensor(
                [t for t, _ in routed_tokens],
                device=flat_hidden.device,
                dtype=torch.long,
            )

            routing_slots = torch.tensor(
                [k for _, k in routed_tokens],
                device=flat_hidden.device,
                dtype=torch.long,
            )

            expert_input = flat_hidden.index_select(0, token_indices)

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

            gate_out = torch.matmul(expert_input, w_gate.t())

            up_out = torch.matmul(expert_input, w_up.t())

            intermediate = F.silu(gate_out) * up_out

            expert_output = torch.matmul(intermediate, w_down.t())

            routing_weight = flat_topk_weights[
                token_indices,
                routing_slots,
            ].unsqueeze(-1)

            expert_output *= routing_weight

            final_output.index_add_(
                0,
                token_indices,
                expert_output,
            )
        return final_output.view(batch_size, seq_len, hidden_size)
