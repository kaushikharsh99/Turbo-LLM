import torch
import torch.nn.functional as F

class RouterExecutor:
    def __init__(self, loader, num_layers, top_k=8):
        self.loader = loader
        self.num_layers = num_layers
        self.top_k = top_k
        self.routers = {}
        # Preload router weights to VRAM (only 24 MB total)
        for i in range(self.num_layers):
            weight_name = self.loader.layout.router_tensor(i)
            self.routers[i] = self.loader.load_weight(weight_name)

    def compute_routing(self, layer_id, hidden_states, top_k=None):
        """
        hidden_states: [batch, seq_len, hidden_size] or [seq_len, hidden_size]
        returns: top_k_indices, top_k_weights
        """
        router_weight = self.routers[layer_id]
        
        # [batch * seq_len, hidden_size]
        orig_shape = hidden_states.shape
        hidden_flatten = hidden_states.view(-1, orig_shape[-1])
        
        # logits: [batch * seq_len, num_experts]
        logits = F.linear(hidden_flatten, router_weight)
        
        # probabilities — must use FP32 for numerical stability with 128 experts
        probs = F.softmax(logits, dtype=torch.float, dim=-1)
        
        # topk
        k = top_k if top_k is not None else self.top_k
        top_k_weights, top_k_indices = torch.topk(probs, k, dim=-1)
        
        # normalize weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # cast back to input dtype for expert MLP computations
        top_k_weights = top_k_weights.to(logits.dtype)
        
        return top_k_indices, top_k_weights
