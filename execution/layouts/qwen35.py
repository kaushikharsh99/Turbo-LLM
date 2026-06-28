from .base import BaseLayerLayout


class Qwen35LayerLayout(BaseLayerLayout):

    def __init__(self, config):
        self.layer_types = config.layer_types

    def is_linear(self, layer):
        return self.layer_types[layer] == "linear_attention"

    def preload_weights(self, layer):

        weights = [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ]

        if self.is_linear(layer):

            weights += [
                "linear_attn.in_proj_qkv.weight",
                "linear_attn.in_proj_z.weight",
                "linear_attn.in_proj_a.weight",
                "linear_attn.in_proj_b.weight",
                "linear_attn.out_proj.weight",
                "linear_attn.norm.weight",

                # Missing parameters
                "linear_attn.A_log",
                "linear_attn.dt_bias",
                "linear_attn.conv1d.weight",
            ]

        else:

            weights += [
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
            ]

        return weights

    def attention_module(self, layer_module):
        return None

    def input_norm(self, layer_module):
        return layer_module.input_layernorm

    def post_norm(self, layer_module):
        return layer_module.post_attention_layernorm
