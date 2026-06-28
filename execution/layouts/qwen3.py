from .base import BaseLayerLayout


class Qwen3LayerLayout(BaseLayerLayout):

    def preload_weights(self, layer):

        return [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "post_attention_layernorm.weight",
        ]

    def attention_module(self, layer_module):
        return layer_module.self_attn

    def input_norm(self, layer_module):
        return layer_module.input_layernorm

    def post_norm(self, layer_module):
        return layer_module.post_attention_layernorm