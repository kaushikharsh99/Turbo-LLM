import torch
import torch.nn.functional as F
from attension.attension import dequantize_weight

class SharedExpert:
    def __init__(self, config):
        self.config = config

    def forward(self, hidden_states: torch.Tensor, shared_expert) -> torch.Tensor:
        """
        hidden_states shape: (batch_size, seq_len, hidden_size)
        shared_expert: SharedExpert metadata object from layer.moe.shared_expert
        Returns:
            output shape: (batch_size, seq_len, hidden_size)
        """
        if (
            shared_expert is None
            or shared_expert.gate_proj is None
            or shared_expert.up_proj is None
            or shared_expert.down_proj is None
            or shared_expert.shared_gate is None
            or shared_expert.gate_proj.data is None
            or shared_expert.up_proj.data is None
            or shared_expert.down_proj.data is None
            or shared_expert.shared_gate.data is None
        ):
            return torch.zeros_like(hidden_states)

        dtype = hidden_states.dtype
        
        # Dequantize weights
        w_gate = dequantize_weight(
            shared_expert.gate_proj.data,
            shared_expert.gate_scale.data if shared_expert.gate_scale else None,
            dtype,
        )
        w_up = dequantize_weight(
            shared_expert.up_proj.data,
            shared_expert.up_scale.data if shared_expert.up_scale else None,
            dtype,
        )
        w_down = dequantize_weight(
            shared_expert.down_proj.data,
            shared_expert.down_scale.data if shared_expert.down_scale else None,
            dtype,
        )
        
        # SwiGLU FFN
        gate_out = torch.matmul(hidden_states, w_gate.t())
        up_out = torch.matmul(hidden_states, w_up.t())

        intermediate = F.silu(gate_out) * up_out

        output = torch.matmul(intermediate, w_down.t())

        # Shared expert gate (HF implementation)
        w_shared_gate = shared_expert.shared_gate.data.to(dtype)

        gate = torch.matmul(
            hidden_states,
            w_shared_gate.t(),
        )

        gate = torch.sigmoid(gate)

        output = output * gate

        return output
