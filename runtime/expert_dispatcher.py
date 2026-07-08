import time
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any

from attension.attension import dequantize_weight

class ExpertDispatcher:
    """
    Handles expert-centric MoE dispatching.
    Groups routed tokens across all sequences in the batch by expert,
    gathers hidden states, executes GEMMs, and scatters outputs.
    Pure math kernel, does not manage memory or load/unload tensors.
    """

    def __init__(self, config):
        self.config = config
        self.num_experts = config.num_experts

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        experts: List[Any],
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
        layer_id: int,
        profiler: Any = None,
    ) -> torch.Tensor:
        """
        Groups tokens by expert, runs one batched GEMM per expert, and scatters back.
        hidden_states: (batch_size, seq_len, hidden_size)
        experts: list of Expert objects for this layer
        topk_weights: (batch_size, seq_len, k)
        topk_indices: (batch_size, seq_len, k)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_tokens = batch_size * seq_len
        dtype = hidden_states.dtype
        device = hidden_states.device

        flat_hidden = hidden_states.view(num_tokens, hidden_size)
        flat_weights = topk_weights.view(num_tokens, -1)
        flat_indices = topk_indices.view(num_tokens, -1)

        final_output = torch.zeros_like(flat_hidden)

        if profiler:
            profiler.start_layer_timer(
                layer_id,
                "Dispatch Table",
            )
        
        active_expert_ids = torch.unique(flat_indices).tolist()
    
        if profiler:
            profiler.stop_layer_timer(
                layer_id,
                "Dispatch Table",
            )

        if profiler and profiler.enabled:
            profiler.record_layer_active_experts(layer_id, len(active_expert_ids))

        # 2. Iterate through each active expert
        for expert_id in active_expert_ids:
            expert = experts[expert_id]

            if profiler:
                profiler.start_layer_timer(
                    layer_id,
                    "Gather",
                )
            # Find matching tokens
            mask = (flat_indices == expert_id)
            if not mask.any():
                continue

            token_indices, k_indices = mask.nonzero(as_tuple=True)
            num_routed_tokens = token_indices.numel()

            start_exec = 0.0

            if profiler and profiler.enabled:
                torch.cuda.synchronize()
                start_exec = time.perf_counter()
                    
            expert_input = flat_hidden[token_indices]

            if profiler:
                profiler.stop_layer_timer(
                    layer_id,
                    "Gather",
                )

            # Dequantize weights
            if profiler:
                profiler.start_layer_timer(
                    layer_id,
                    "FP8 Dequant",
                )
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

            if profiler:
                profiler.stop_layer_timer(
                    layer_id,
                    "FP8 Dequant",
                )
            if profiler:
                profiler.start_layer_timer(
                    layer_id,
                    "Expert Compute",
                )

            # SwiGLU FFN
            gate_out = torch.matmul(expert_input, w_gate.t())
            up_out = torch.matmul(expert_input, w_up.t())
            intermediate = F.silu(gate_out) * up_out
            expert_output = torch.matmul(intermediate, w_down.t())

            if profiler:
                profiler.stop_layer_timer(
                    layer_id,
                    "Expert Compute",
                )

            # Scale by routing weight
            routing_weight = flat_weights[token_indices, k_indices].unsqueeze(-1)
            weighted_output = expert_output * routing_weight

            if profiler:
                profiler.start_layer_timer(
                    layer_id,
                    "Scatter",
                )
            # Accumulate output (Scatter)
            final_output.index_add_(0, token_indices, weighted_output)

            if profiler and profiler.enabled:
                torch.cuda.synchronize()

                profiler.record_expert_execution(
                    expert_id=expert_id,
                    tokens_count=num_routed_tokens,
                    exec_time=time.perf_counter() - start_exec,
                )

            if profiler:
                profiler.stop_layer_timer(
                    layer_id,
                    "Scatter",
                )

        return final_output.view(batch_size, seq_len, hidden_size)
