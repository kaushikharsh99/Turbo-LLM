import time
import torch
import torch.nn.functional as F
from typing import Optional, Any

from model.model import Model
from memory.memory_manager import MemoryManager
from attension.attension import Attention, rms_norm
from attension.linear_attention import LinearAttention
from attension.kv_cache import KVCache
from moe.router import MoERouter
from moe.experts import MoEExperts
from moe.shared_experts import SharedExpert
from moe.merge import MoEMerge
from runtime.expert_dispatcher import ExpertDispatcher

class Executor:
    """Orchestrates model execution layer-by-layer and interacts with MemoryManager."""

    def __init__(self, model: Model, memory_manager: MemoryManager):
        self.model = model
        self.memory_manager = memory_manager

        # Initialize math sub-modules
        self.attention = Attention(model.config)
        self.linear_attention = LinearAttention(model.config)
        self.router = MoERouter(model.config)
        self.experts = MoEExperts(model.config)
        self.shared_expert = SharedExpert(model.config)
        self.merge = MoEMerge(model.config)
        self.expert_dispatcher = ExpertDispatcher(model.config)

    def load_to_gpu(self, name: str, profiler: Optional[Any] = None) -> torch.Tensor:
        """Loads a tensor to GPU and records cache hit/miss and byte tracking for profiling."""
        if profiler and profiler.enabled:
            gpu_hit = self.memory_manager.gpu_memory.exists(name)
            cpu_hit = self.memory_manager.cpu_memory.exists(name)
            tensor_obj = self.memory_manager.tensor_registry[name]
            size_bytes = tensor_obj.size_bytes

            profiler.record_memory_movement(
                gpu_cache_hit=gpu_hit,
                cpu_cache_hit=cpu_hit,
                ram_to_gpu_bytes=0 if gpu_hit else size_bytes,
                ssd_to_ram_bytes=0 if (gpu_hit or cpu_hit) else size_bytes,
                tensor_load=not gpu_hit,
            )

            gpu_evictions_before = len(self.memory_manager.gpu_memory.cache)
            cpu_evictions_before = len(self.memory_manager.cpu_memory.cache)

            res = self.memory_manager.move_to_gpu(name)

            gpu_evictions_after = len(self.memory_manager.gpu_memory.cache)
            cpu_evictions_after = len(self.memory_manager.cpu_memory.cache)

            gpu_evicted = gpu_evictions_before + 1 - gpu_evictions_after
            cpu_evicted = cpu_evictions_before + (0 if cpu_hit else 1) - cpu_evictions_after

            for _ in range(max(0, gpu_evicted)):
                profiler.record_memory_movement(tensor_eviction=True)
            for _ in range(max(0, cpu_evicted)):
                profiler.record_memory_movement(tensor_eviction=True)

            return res
        else:
            return self.memory_manager.move_to_gpu(name)

    def forward_layer(
        self,
        hidden_states: torch.Tensor,
        layer_id: int,
        kv_cache: KVCache,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        profiler: Optional[Any] = None,
    ) -> torch.Tensor:
        """Runs the forward pass for a single layer block."""
        layer = self.model.layers[layer_id]

        # --- 1. Attention Block ---
        if layer.attention_norm:
            self.load_to_gpu(layer.attention_norm.name, profiler)

        if profiler:
            profiler.start_layer_timer(
                layer_id,
                "Linear Attention" if layer.linear_attn else "Attention",
            )

        if layer.linear_attn:
            la = layer.linear_attn
            for proj in [
                la.conv1d, la.dt_bias, la.A_log, la.norm, la.out_proj,
                la.in_proj_qkv, la.in_proj_z, la.in_proj_b, la.in_proj_a,
                la.out_proj_scale, la.in_proj_qkv_scale, la.in_proj_z_scale
            ]:
                if proj:
                    self.load_to_gpu(proj.name, profiler)

            attn_hidden = self.linear_attention.forward(
                hidden_states, layer, kv_cache,attention_mask,
            )
        else:
            if layer.attention and layer.attention.q_proj:
                self.load_to_gpu(layer.attention.q_proj.name, profiler)
                self.load_to_gpu(layer.attention.k_proj.name, profiler)
                self.load_to_gpu(layer.attention.v_proj.name, profiler)
                self.load_to_gpu(layer.attention.o_proj.name, profiler)

                if layer.attention.q_scale: self.load_to_gpu(layer.attention.q_scale.name, profiler)
                if layer.attention.k_scale: self.load_to_gpu(layer.attention.k_scale.name, profiler)
                if layer.attention.v_scale: self.load_to_gpu(layer.attention.v_scale.name, profiler)
                if layer.attention.o_scale: self.load_to_gpu(layer.attention.o_scale.name, profiler)
                if layer.attention.q_norm: self.load_to_gpu(layer.attention.q_norm.name, profiler)
                if layer.attention.k_norm: self.load_to_gpu(layer.attention.k_norm.name, profiler)

            attn_hidden = self.attention.forward(
                hidden_states, layer, kv_cache, position_ids, attention_mask
            )

        if profiler:
            profiler.stop_layer_timer(
                layer_id,
                "Linear Attention" if layer.linear_attn else "Attention",
            )

        # --- 2. Post-Attention Norm ---
        if layer.ffn_norm:
            self.load_to_gpu(layer.ffn_norm.name, profiler)

        eps = self.model.config.rms_norm_eps
        normed_ffn_hidden = rms_norm(
            attn_hidden, layer.ffn_norm.data, eps
        )

        # --- 3. MoE Block ---
        ffn_output = torch.zeros_like(attn_hidden)

        if layer.moe:
            if layer.moe.router and layer.moe.router.gate:
                self.load_to_gpu(layer.moe.router.gate.name, profiler)

                if profiler:
                    profiler.start_layer_timer(layer_id, "Router")

                topk_weights, topk_indices = self.router.forward(
                    normed_ffn_hidden,
                    layer.moe.router.gate.data,
                )

                if profiler:
                    profiler.stop_layer_timer(layer_id, "Router")

                # Find unique active experts
                active_expert_ids = torch.unique(topk_indices).tolist()
                
                if profiler:
                    profiler.start_layer_timer(
                        layer_id,
                        "Weight Loading",
                    )

                for exp_id in active_expert_ids:
                    expert = layer.moe.experts[exp_id]
                    self.load_to_gpu(expert.gate_proj.name, profiler)
                    self.load_to_gpu(expert.up_proj.name, profiler)
                    self.load_to_gpu(expert.down_proj.name, profiler)

                    if expert.gate_scale: self.load_to_gpu(expert.gate_scale.name, profiler)
                    if expert.up_scale: self.load_to_gpu(expert.up_scale.name, profiler)
                    if expert.down_scale: self.load_to_gpu(expert.down_scale.name, profiler)

                if profiler:
                    profiler.stop_layer_timer(
                        layer_id,
                        "Weight Loading",
                    )

                if profiler:
                    profiler.start_layer_timer(
                        layer_id,
                        "Experts",
                    )

                routed_output = self.expert_dispatcher.dispatch(
                    normed_ffn_hidden,
                    layer.moe.experts,
                    topk_weights,
                    topk_indices,
                    layer_id,
                    profiler,
                )

                if profiler:
                    profiler.stop_layer_timer(
                        layer_id,
                        "Experts",
                    )
            else:
                routed_output = torch.zeros_like(normed_ffn_hidden)

            if layer.moe.shared_expert and layer.moe.shared_expert.gate_proj:
                if profiler:
                    profiler.start_layer_timer(
                        layer_id,
                        "Shared Expert",
                    )

                shared = layer.moe.shared_expert
                self.load_to_gpu(shared.gate_proj.name, profiler)
                self.load_to_gpu(shared.up_proj.name, profiler)
                self.load_to_gpu(shared.down_proj.name, profiler)

                if shared.shared_gate: self.load_to_gpu(shared.shared_gate.name, profiler)
                if shared.gate_scale: self.load_to_gpu(shared.gate_scale.name, profiler)
                if shared.up_scale: self.load_to_gpu(shared.up_scale.name, profiler)
                if shared.down_scale: self.load_to_gpu(shared.down_scale.name, profiler)

                shared_output = self.shared_expert.forward(
                    normed_ffn_hidden, shared
                )

                if profiler:
                    profiler.stop_layer_timer(
                        layer_id,
                        "Shared Expert",
                    )
            else:
                shared_output = torch.zeros_like(normed_ffn_hidden)

            if profiler:
                profiler.start_layer_timer(
                    layer_id,
                    "Merge",
                )

            ffn_output = self.merge.forward(
                routed_output,
                shared_output,
            )

            if profiler:
                profiler.stop_layer_timer(
                    layer_id,
                    "Merge",
                )

        if profiler:
            profiler.start_layer_timer(
                layer_id,
                "GPU Free",
            )

        self.memory_manager.free_layer_gpu(layer_id)

        if profiler:
            profiler.stop_layer_timer(
                layer_id,
                "GPU Free",
            )

        return attn_hidden + ffn_output

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        profiler: Optional[Any] = None,
    ) -> torch.Tensor:

        if profiler:
            profiler.start_timer("Embedding")

        self.load_to_gpu(self.model.embedding.name, profiler)
        embedding_weight = self.model.embedding.data

        # shape: (batch_size, seq_len, hidden_size)
        hidden_states = F.embedding(input_ids, embedding_weight)

        if profiler:
            profiler.stop_timer("Embedding")

        # 2. Process all layers sequentially
        for layer_id in range(self.model.config.num_layers):

            if profiler:
                profiler.start_layer_timer(layer_id, "Total")

            hidden_states = self.forward_layer(
                hidden_states,
                layer_id,
                kv_cache,
                position_ids,
                attention_mask,
                profiler,
            )

            if profiler:
                profiler.stop_layer_timer(layer_id, "Total")

        if profiler:
            profiler.start_timer("Final RMSNorm")

        self.load_to_gpu(self.model.final_norm.name, profiler)
        eps = self.model.config.rms_norm_eps
        hidden_states = rms_norm(
            hidden_states, self.model.final_norm.data, eps
        )

        if profiler:
            profiler.stop_timer("Final RMSNorm")

        if profiler:
            profiler.start_timer("LM Head")

        self.load_to_gpu(self.model.lm_head.name, profiler)
        lm_head_weight = self.model.lm_head.data

        logits = torch.matmul(hidden_states, lm_head_weight.t())

        if profiler:
            profiler.stop_timer("LM Head")

        return logits
