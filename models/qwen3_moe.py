from models.base import BaseModelAdapter
from execution.layouts import Qwen3LayerLayout
from transformers.models.qwen3_moe.modeling_qwen3_moe import create_causal_mask
from cache.kv_cache import KVCache
class Qwen3MoeAdapter(BaseModelAdapter):

    def __init__(self, model, loader):
        self.model = model
        self.loader = loader

    @property
    def text_model(self):
        return self.model.model

    def create_layer_layout(self):
        return Qwen3LayerLayout()
    
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
            past_key_values=kv_cache,
            position_ids=position_ids,
            attention_mask=attention_mask,
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
            position_ids=position_ids
        )

    def rotary_embeddings(self, hidden, position_ids):
        return self.text_model.rotary_emb(
            hidden,
            position_ids=position_ids,
        )

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
        return KVCache(max_seq_len)