import torch
import torch.nn.functional as F
import threading
import queue
import time

def moe_seq(hidden_states, gate_weights, up_weights, down_weights, top_k_weights):
    final_output = torch.zeros_like(hidden_states)
    top_k = top_k_weights.shape[1]
    for i in range(top_k):
        gate_proj = gate_weights[i]
        up_proj = up_weights[i]
        down_proj = down_weights[i]
        
        g = F.linear(hidden_states, gate_proj)
        u = F.linear(hidden_states, up_proj)
        inter = F.silu(g) * u
        o = F.linear(inter, down_proj)
        final_output += o * top_k_weights[0, i]
    return final_output

class MoEExecutor:
    def __init__(self, loader):
        self.loader = loader

    def execute_decode(self, layer_id, hidden_states, top_k_indices, top_k_weights):
        """
        Specialized decode path for a single token (seq_len == 1).
        hidden_states: [1, hidden_size]
        top_k_indices: [1, top_k]
        top_k_weights: [1, top_k]
        """
        # Reset instrumentation accumulators
        self.loader.load_ms_accum = 0.0
        self.loader.dequant_ms_accum = 0.0
        self.loader.evict_ms_accum = 0.0
        
        self.loader.clear_pinned_slots()
        expert_ids = top_k_indices[0].tolist()

        # 1. Fetch expert weights directly on main thread (handles cache hits & dequantization)
        gate_weights = []
        up_weights = []
        down_weights = []
        for exp_id in expert_ids:
            gate, up, down = self.loader.load_expert(layer_id, exp_id)
            gate_weights.append(gate)
            up_weights.append(up)
            down_weights.append(down)
            
        prefix = self.loader.layout.layer_prefix_name(layer_id)

        shared_gate = self.loader.load_weight(
            f"{prefix}.mlp.shared_expert.gate_proj.weight"
        )

        shared_up = self.loader.load_weight(
            f"{prefix}.mlp.shared_expert.up_proj.weight"
        )

        shared_down = self.loader.load_weight(
            f"{prefix}.mlp.shared_expert.down_proj.weight"
        )

        shared_gate_weight = self.loader.load_weight(
            f"{prefix}.mlp.shared_expert_gate.weight"
        )

        # 2. Sequential matrix multiplication (no stacked weight allocations or copy kernels)
        t_gemm_start = time.time()
        final_output = moe_seq(hidden_states, gate_weights, up_weights, down_weights, top_k_weights)
        
        # Shared expert
        shared_gate_out = F.linear(hidden_states, shared_gate)
        shared_up_out = F.linear(hidden_states, shared_up)

        shared_hidden = F.silu(shared_gate_out) * shared_up_out

        shared_output = F.linear(shared_hidden, shared_down)

        shared_gate_score = torch.sigmoid(
            F.linear(hidden_states, shared_gate_weight)
        )

        final_output += shared_gate_score * shared_output

        gemm_ms = (time.time() - t_gemm_start) * 1000.0
        
        if layer_id == 21:
            print(f"\nLayer 21\n")
            print(f"load:\n{self.loader.load_ms_accum:.1f}\n")
            print(f"dequant:\n{self.loader.dequant_ms_accum:.1f}\n")
            print(f"gemm:\n{gemm_ms:.1f}\n")
            print(f"evict:\n{self.loader.evict_ms_accum:.1f}")

        return final_output

    def execute_layer(self, layer_id, hidden_states, top_k_indices, top_k_weights):
        """
        hidden_states: [batch * seq_len, hidden_size]
        top_k_indices: [batch * seq_len, top_k]
        top_k_weights: [batch * seq_len, top_k]
        """
        num_tokens = hidden_states.shape[0]
        if num_tokens == 1:
            return self.execute_decode(layer_id, hidden_states, top_k_indices, top_k_weights)

        # ----------------------------------------------------
        # Sequential Path: Standard Token Grouping for Prefill
        # ----------------------------------------------------
        self.loader.clear_pinned_slots()
        final_output = torch.zeros_like(hidden_states)
        prefix = self.loader.layout.layer_prefix_name(layer_id)

        shared_gate = self.loader.load_weight(
            f"{prefix}.mlp.shared_expert.gate_proj.weight"
        )

        shared_up = self.loader.load_weight(
            f"{prefix}.mlp.shared_expert.up_proj.weight"
        )

        shared_down = self.loader.load_weight(
            f"{prefix}.mlp.shared_expert.down_proj.weight"
        )

        shared_gate_weight = self.loader.load_weight(
            f"{prefix}.mlp.shared_expert_gate.weight"
        )
        # Find unique experts needed for this batch of tokens
        unique_experts = torch.unique(top_k_indices).tolist()
        if not unique_experts:
            return final_output

        # Double-buffer queue (maxsize=1 to load at most 1 expert in advance)
        load_queue = queue.Queue(maxsize=1)
        stop_event = threading.Event()

        def loader_worker():
            try:
                for idx in range(len(unique_experts)):
                    if stop_event.is_set():
                        break
                    expert_id = unique_experts[idx]
                    gate_proj, up_proj, down_proj = self.loader.load_expert_dynamic(layer_id, expert_id)
                    load_queue.put((expert_id, gate_proj, up_proj, down_proj))
            except Exception as e:
                print(f"Error in MoE background loader thread: {e}")
                load_queue.put((None, None, None, None))

        # Start background loading
        loader_thread = threading.Thread(target=loader_worker)
        loader_thread.start()

        try:
            for idx in range(len(unique_experts)):
                target_expert_id = unique_experts[idx]
                expert_id, gate_proj, up_proj, down_proj = load_queue.get()
                if expert_id is None:
                    raise RuntimeError("Background loader thread failed.")
                assert expert_id == target_expert_id, f"Mismatched expert ID: expected {target_expert_id}, got {expert_id}"

                mask = (top_k_indices == expert_id)
                token_indices, top_k_pos = torch.where(mask)
                
                if len(token_indices) > 0:
                    expert_input = hidden_states[token_indices]
                    
                    gate_out = F.linear(expert_input, gate_proj)
                    up_out = F.linear(expert_input, up_proj)
                    intermediate = F.silu(gate_out) * up_out
                    expert_output = F.linear(intermediate, down_proj)
                    
                    weights = top_k_weights[token_indices, top_k_pos].unsqueeze(-1)
                    final_output[token_indices] += weights * expert_output
                    
                    del expert_input, gate_out, up_out, intermediate, expert_output, weights
                
                del gate_proj, up_proj, down_proj

        finally:
            stop_event.set()
            # Drain queue to unblock worker thread if it is blocked on put()
            while not load_queue.empty():
                try:
                    load_queue.get_nowait()
                except queue.Empty:
                    break
            loader_thread.join()
            shared_gate_out = F.linear(hidden_states, shared_gate)
            shared_up_out = F.linear(hidden_states, shared_up)

            shared_hidden = F.silu(shared_gate_out) * shared_up_out

            shared_output = F.linear(shared_hidden, shared_down)

            shared_gate_score = torch.sigmoid(
                F.linear(hidden_states, shared_gate_weight)
            )

            final_output += shared_gate_score * shared_output

        if self.loader.DEVICE == "cuda":
            torch.cuda.synchronize()
        elif self.loader.DEVICE == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        return final_output
