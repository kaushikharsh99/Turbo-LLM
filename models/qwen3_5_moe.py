from models.base import BaseModelAdapter
from execution.layouts import Qwen35LayerLayout
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import create_causal_mask
from cache.qwen35_cache import Qwen35Cache

class Qwen35MoeAdapter(BaseModelAdapter):

    def __init__(self, model, loader):
        self.model = model
        self.loader = loader

    @property
    def text_model(self):
        return self.model.model
    
    def create_layer_layout(self):
        return Qwen35LayerLayout(self.model.config)
    
    def load_config(self):
        return self.text_model.config

    def create_meta_model(self):
        return self.model

    def embed(self, input_ids):
        return self.text_model.embed_tokens(input_ids)

    def layers(self):
        return self.text_model.layers

    def forward_layer(
        self,
        layer_id,
        hidden,
        kv_cache,
        position_ids,
        attention_mask,
    ):
        layer = self.text_model.layers[layer_id]

        outputs = layer(
            hidden_states=hidden,
            position_embeddings=self.rotary_embeddings(hidden, position_ids),
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=kv_cache,
        )

        return outputs[0], kv_cache.get(layer_id) if kv_cache is not None else None

    def final_norm(self, hidden):
        return self.text_model.norm(hidden)

    def lm_head(self, hidden):
        return self.model.lm_head(hidden)

    def create_attention_mask(self, hidden, kv_cache, position_ids):
        return create_causal_mask(
            config=self.text_model.config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=kv_cache,
            position_ids=position_ids,
        )

    def rotary_embeddings(self, hidden, position_ids):
        return self.text_model.rotary_emb(hidden, position_ids)

    @property
    def num_layers(self):
        return self.text_model.config.num_hidden_layers

    @property
    def hidden_size(self):
        return self.text_model.config.hidden_size

    @property
    def is_moe(self):
        return True
    
    def create_cache(
        self,
        max_seq_len,
    ):
        return Qwen35Cache(
            num_layers=self.num_layers,
            max_seq_len=max_seq_len,
        )