"""Sparse expert execution, loading one expert at a time during prefill."""
import torch
import torch.nn.functional as F


def moe_seq(hidden_states, gate_weights, up_weights, down_weights, top_k_weights):
    result = torch.zeros_like(hidden_states)
    for i, (gate, up, down) in enumerate(zip(gate_weights, up_weights, down_weights)):
        result += F.linear(F.silu(F.linear(hidden_states, gate)) * F.linear(hidden_states, up), down) * top_k_weights[:, i:i+1]
    return result


class MoEExecutor:
    def __init__(self, loader):
        self.loader = loader

    def _shared_expert(self, layer_id, hidden):
        prefix = self.loader.layout.layer_prefix_name(layer_id)
        base = f"{prefix}.mlp.shared_expert"
        if f"{base}.gate_proj.weight" not in self.loader.weight_map:
            return None
        gate = self.loader.load_weight(f"{base}.gate_proj.weight")
        up = self.loader.load_weight(f"{base}.up_proj.weight")
        down = self.loader.load_weight(f"{base}.down_proj.weight")
        score = self.loader.load_weight(f"{prefix}.mlp.shared_expert_gate.weight")
        return F.linear(F.silu(F.linear(hidden, gate)) * F.linear(hidden, up), down) * torch.sigmoid(F.linear(hidden, score))

    def execute_decode(self, layer_id, hidden_states, top_k_indices, top_k_weights):
        return self.execute_layer(layer_id, hidden_states, top_k_indices, top_k_weights)

    def execute_layer(self, layer_id, hidden_states, top_k_indices, top_k_weights):
        self.loader.clear_pinned_slots()
        result = torch.zeros_like(hidden_states)
        # Never hold every expert on the accelerator. Decode uses bounded cached
        # slots; prefill releases each expert before loading the next.
        for expert in torch.unique(top_k_indices).tolist():
            self.loader.clear_pinned_slots()
            token, choice = torch.where(top_k_indices == expert)
            if hidden_states.shape[0] == 1:
                gate, up, down = self.loader.load_expert(layer_id, expert)
            else:
                gate, up, down = self.loader.load_expert_dynamic(layer_id, expert)
            selected = hidden_states[token]
            output = F.linear(F.silu(F.linear(selected, gate)) * F.linear(selected, up), down)
            result.index_add_(0, token, output * top_k_weights[token, choice, None])
            del gate, up, down, selected, output
        shared = self._shared_expert(layer_id, hidden_states)
        if shared is not None:
            result += shared
        return result
