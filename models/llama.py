from models.base import BaseModelAdapter

class LlamaAdapter(BaseModelAdapter):

    def __init__(self, model, loader):
        self.model = model
        self.loader = loader

    def load_config(self):
        return self.model.config

    def create_meta_model(self):
        return self.model

    def embed(self, input_ids):
        return self.model.model.embed_tokens(input_ids)

    def layers(self):
        return self.model.model.layers

    def forward_layer(
        self,
        layer_id,
        hidden,
        kv_cache,
        position_ids,
        attention_mask,
    ):
        layer = self.model.model.layers[layer_id]
        # Dynamically compute rotary embeddings for Llama layers
        position_embeddings = self.model.model.rotary_emb(hidden, position_ids=position_ids)
        outputs = layer(
            hidden_states=hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=kv_cache,
            position_embeddings=position_embeddings,
        )
        return outputs[0], kv_cache.get(layer_id) if kv_cache is not None else None

    def final_norm(self, hidden):
        return self.model.model.norm(hidden)

    def lm_head(self, hidden):
        return self.model.lm_head(hidden)

    def create_attention_mask(self, hidden, kv_cache, position_ids):
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
        batch_size, seq_len = hidden.shape[0], hidden.shape[1]
        past_key_values_length = kv_cache.get_seq_length(0) if kv_cache is not None else 0
        return _prepare_4d_causal_attention_mask(
            attention_mask=None,
            input_shape=(batch_size, seq_len),
            inputs_embeds=hidden,
            past_key_values_length=past_key_values_length,
        )

    def rotary_embeddings(self, hidden, position_ids):
        return self.model.model.rotary_emb(hidden, position_ids=position_ids)

    @property
    def num_layers(self):
        return self.model.config.num_hidden_layers

    @property
    def hidden_size(self):
        return self.model.config.hidden_size

    @property
    def is_moe(self):
        return False
