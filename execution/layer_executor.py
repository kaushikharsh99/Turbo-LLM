import torch
import time
import torch.nn.functional as F
from accelerate.utils import set_module_tensor_to_device

class LayerExecutor:
    def __init__(self, adapter, loader, router_exec, moe_exec):
        self.adapter = adapter
        self.model = adapter.create_meta_model()
        self.loader = loader
        self.router_exec = router_exec
        self.moe_exec = moe_exec
        
        # Non-MoE layer weight names inside a layer module
        self.non_moe_weight_names = [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "post_attention_layernorm.weight"
        ]
        self.preload_backbone_weights()
 
    def preload_backbone_weights(self):
        print(f"Preloading backbone weights for all {self.adapter.num_layers} layers to GPU VRAM...")
        for layer_id in range(self.adapter.num_layers):
            prefix = f"model.layers.{layer_id}"
            for name in self.non_moe_weight_names:
                full_name = f"{prefix}.{name}"
                w = self.loader.load_weight(full_name)
                set_module_tensor_to_device(
                self.model,
                full_name,
                device=self.loader.DEVICE,
                value=w,
                )
                
        print("Preloading embedding, norm, and lm_head weights to GPU VRAM...")
        embed_w = self.loader.load_weight("model.embed_tokens.weight")
        set_module_tensor_to_device(self.model, "model.embed_tokens.weight", device=self.loader.DEVICE, value=embed_w)
        
        norm_w = self.loader.load_weight("model.norm.weight")
        set_module_tensor_to_device(self.model, "model.norm.weight", device=self.loader.DEVICE, value=norm_w)
        
        lm_head_w = self.loader.load_weight("lm_head.weight")
        set_module_tensor_to_device(self.model, "lm_head.weight", device=self.loader.DEVICE, value=lm_head_w)
 
    @torch.no_grad()
    def execute_layer(self, layer_id, hidden_states, attention_mask, position_ids, position_embeddings, kv_cache=None):
        layer_module = self.adapter.layers()[layer_id]

        # Initialize profiling lists if not present
        if not hasattr(self, "attn_times"):
            self.attn_times = []
            self.moe_times = []

        # 2. Execute Layernorm & Self-Attention (weights are preloaded!)
        t_start_attn = time.time()
        residual = hidden_states
        normed_hidden = layer_module.input_layernorm(hidden_states)
        
        attn_output, _ = layer_module.self_attn(
            hidden_states=normed_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            past_key_values=kv_cache,
        )
        hidden_states = residual + attn_output
        self.attn_times.append(time.time() - t_start_attn)
        
        # 3. Post-attention Norm & MoE MLP
        t_start_moe = time.time()
        residual = hidden_states
        normed_attn = layer_module.post_attention_layernorm(hidden_states)
        
        # Router computation (preloaded!)
        top_k_indices, top_k_weights = self.router_exec.compute_routing(layer_id, normed_attn)
        
        # Execute experts sequentially
        orig_shape = normed_attn.shape
        hidden_flatten = normed_attn.view(-1, orig_shape[-1])
        moe_output = self.moe_exec.execute_layer(layer_id, hidden_flatten, top_k_indices, top_k_weights)
        moe_output = moe_output.view(orig_shape)
        
        hidden_states = residual + moe_output
        self.moe_times.append(time.time() - t_start_moe)
        
        kv = kv_cache.get(layer_id) if kv_cache is not None else None
        return hidden_states, kv
