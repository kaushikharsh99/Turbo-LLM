import time

import torch
from accelerate.utils import set_module_tensor_to_device


class LayerExecutor:

    def __init__(
        self,
        adapter,
        loader,
        router_exec,
        moe_exec,
        collector=None,
    ):
        self.adapter = adapter
        self.model = adapter.create_meta_model()
        self.loader = loader
        self.router_exec = router_exec
        self.moe_exec = moe_exec
        self.collector = collector

        self.layer_layout = adapter.create_layer_layout()

        self.preload_backbone_weights()

    def preload_backbone_weights(self):

        device_type = (
            "GPU VRAM"
            if self.loader.DEVICE in ("cuda", "mps")
            else "System RAM"
        )

        print(
            f"Preloading backbone weights for all {self.adapter.num_layers} layers to {device_type}..."
        )

        if getattr(self.loader, "gguf_loader", None) is not None:
            print("GGUF Mode: On-demand layer loading enabled (saving CPU RAM)...")
            return

        for layer_id in range(self.adapter.num_layers):

            prefix = self.loader.layout.layer_prefix_name(layer_id)

            for name in self.layer_layout.preload_weights(layer_id):

                full_name = f"{prefix}.{name}"


                weight = self.loader.load_weight(full_name)


                module_name = self.loader.layout.module_name(full_name)

                set_module_tensor_to_device(
                    self.model,
                    module_name,
                    device=self.loader.DEVICE,
                    value=weight,
                )

        print(
            f"Preloading embedding, norm, and lm_head weights to {device_type}..."
        )

        # Embedding
        embed_name = self.loader.layout.embed_tensor()

        embed_weight = self.loader.load_weight(embed_name)

        set_module_tensor_to_device(
            self.model,
            self.loader.layout.module_name(embed_name),
            device=self.loader.DEVICE,
            value=embed_weight,
        )

        # Final RMSNorm
        norm_name = self.loader.layout.norm_tensor()

        norm_weight = self.loader.load_weight(norm_name)

        set_module_tensor_to_device(
            self.model,
            self.loader.layout.module_name(norm_name),
            device=self.loader.DEVICE,
            value=norm_weight,
        )

        # LM Head
        lm_head_name = self.loader.layout.lm_head_tensor()
        
        # Check if lm_head shares weight with embed_tokens (tied embeddings)
        config = self.adapter.load_config()
        if getattr(config, "tie_word_embeddings", False):
            print(f"Detected tied word embeddings. Reusing embed_tokens weight for {lm_head_name}...")
            lm_head_weight = embed_weight
        else:
            try:
                lm_head_weight = self.loader.load_weight(lm_head_name)
            except torch.OutOfMemoryError:
                print(f"VRAM limited: Loading {lm_head_name} to CPU RAM fallback...")
                lm_head_weight = self.loader.load_weight(lm_head_name, device="cpu")

        set_module_tensor_to_device(
            self.model,
            self.loader.layout.module_name(lm_head_name),
            device=lm_head_weight.device,
            value=lm_head_weight,
        )
        # Move RoPE buffers (inv_freq, original_inv_freq) to GPU
        text_model = self.adapter.text_model

        if hasattr(text_model, "rotary_emb"):
            text_model.rotary_emb = text_model.rotary_emb.to(self.loader.DEVICE)

    @torch.no_grad()
    def execute_layer(
        self,
        layer_id,
        hidden_states,
        attention_mask,
        position_ids,
        position_embeddings,
        kv_cache=None,
    ):

        layer_module = self.adapter.layers()[layer_id]

        if not hasattr(self, "attn_times"):
            self.attn_times = []
            self.moe_times = []
            self.load_times = []
            self.dequant_times = []
            self.evict_times = []
            self.gemm_times = []
            self.shared_gate_times = []
            self.shared_up_times = []
            self.shared_down_times = []
            self.shared_score_times = []
            self.accum_times = []
            self.list_build_times = []
            self.concat_times = []
            self.boundary_times = []

        # ---------------- Attention ----------------

        if self.loader.DEVICE == "cuda":
            torch.cuda.synchronize()
        t_start_attn = time.perf_counter()

        residual = hidden_states

        normed_hidden = self.layer_layout.input_norm(layer_module)(
            hidden_states
        )

        if hasattr(layer_module, "linear_attn"):

            attn_output = layer_module.linear_attn(
                hidden_states=normed_hidden,
                cache_params=kv_cache,
                attention_mask=attention_mask,
            )

        else:

            attn_output, _ = layer_module.self_attn(
                hidden_states=normed_hidden,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=kv_cache,
            )

        hidden_states = residual + attn_output

        if self.loader.DEVICE == "cuda":
            torch.cuda.synchronize()
        self.attn_times.append(
            (time.perf_counter() - t_start_attn) * 1000.0
        )

        # Initialize router_times and shared_load_times arrays if not initialized
        if not hasattr(self, "router_times"):
            self.router_times = []
        if not hasattr(self, "shared_load_times"):
            self.shared_load_times = []

        # ---------------- MoE ----------------

        if self.loader.DEVICE == "cuda":
            torch.cuda.synchronize()
        t_start_moe = time.perf_counter()

        residual = hidden_states

        normed_attn = self.layer_layout.post_norm(layer_module)(
            hidden_states
        )

        t_route_start = time.perf_counter()
        top_k_indices, top_k_weights = (
            self.router_exec.compute_routing(
                layer_id,
                normed_attn,
            )
        )
        t_route_ms = (time.perf_counter() - t_route_start) * 1000.0

        if hasattr(self.loader, "prefetch_pipeline"):
            self.loader.prefetch_pipeline.working_set_est.record_layer_access(
                layer_id, top_k_indices[-1].tolist()
            )
            self.loader.prefetch_pipeline.submit_prefetch_candidates(
                current_layer=layer_id
            )

        if self.collector is not None:

            self.collector.record_layer(
                layer_id=layer_id,
                experts=top_k_indices[-1].tolist(),
                scores=top_k_weights[-1].tolist(),
            )
            
        original_shape = normed_attn.shape

        hidden_flat = normed_attn.view(
            -1,
            original_shape[-1],
        )

        moe_output = self.moe_exec.execute_layer(
            layer_id,
            hidden_flat,
            top_k_indices,
            top_k_weights,
        )

        if self.collector is not None and hasattr(self.collector, "record_layer_timing"):
            self.collector.record_layer_timing(
                layer_id=layer_id,
                load_ms=self.loader.load_ms_accum,
                dequant_ms=self.loader.dequant_ms_accum,
                evict_ms=self.loader.evict_ms_accum,
                gemm_ms=getattr(self.moe_exec, "last_gemm_ms", 0.0),
            )

        moe_output = moe_output.view(original_shape)

        hidden_states = residual + moe_output

        if self.loader.DEVICE == "cuda":
            torch.cuda.synchronize()
        self.moe_times.append(
            (time.perf_counter() - t_start_moe) * 1000.0
        )
        
        self.router_times.append(t_route_ms)
        self.shared_load_times.append(getattr(self.moe_exec, "last_shared_load_ms", 0.0))
        self.load_times.append(self.loader.load_ms_accum)
        self.dequant_times.append(self.loader.dequant_ms_accum)
        self.evict_times.append(self.loader.evict_ms_accum)
        self.gemm_times.append(getattr(self.moe_exec, "last_gemm_ms", 0.0))
        self.shared_gate_times.append(getattr(self.moe_exec, "last_shared_gate_ms", 0.0))
        self.shared_up_times.append(getattr(self.moe_exec, "last_shared_up_ms", 0.0))
        self.shared_down_times.append(getattr(self.moe_exec, "last_shared_down_ms", 0.0))
        self.shared_score_times.append(getattr(self.moe_exec, "last_shared_score_ms", 0.0))
        self.accum_times.append(getattr(self.moe_exec, "last_accum_ms", 0.0))
        self.list_build_times.append(getattr(self.moe_exec, "last_list_build_ms", 0.0))
        self.concat_times.append(getattr(self.moe_exec, "last_concat_ms", 0.0))
        self.boundary_times.append(getattr(self.moe_exec, "last_boundary_ms", 0.0))
        
        return hidden_states, None  