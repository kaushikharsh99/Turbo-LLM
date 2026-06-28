from abc import ABC, abstractmethod

class BaseModelAdapter(ABC):

    @abstractmethod
    def load_config(self):
        pass

    @abstractmethod
    def create_meta_model(self):
        pass

    @abstractmethod
    def embed(self, input_ids):
        pass

    @abstractmethod
    def layers(self):
        pass

    @abstractmethod
    def forward_layer(
        self,
        layer_id,
        hidden,
        kv_cache,
        position_ids,
        attention_mask,
    ):
        pass

    @abstractmethod
    def final_norm(
        self,
        hidden,
    ):
        pass

    @abstractmethod
    def lm_head(
        self,
        hidden,
    ):
        pass

    @abstractmethod
    def create_attention_mask(
        self,
        hidden,
        kv_cache,
        position_ids,
    ):
        pass

    @abstractmethod
    def rotary_embeddings(
        self,
        hidden,
        position_ids,
    ):
        pass
    
    @abstractmethod
    def create_cache(
        self,
        max_seq_len,
    ):
        pass

    @property
    @abstractmethod
    def num_layers(self):
        pass

    @property
    @abstractmethod
    def hidden_size(self):
        pass

    @property
    @abstractmethod
    def is_moe(self):
        pass

    @property
    def capabilities(self):
        return {
            "is_moe": self.is_moe,
            "supports_flash": False,
            "supports_kv": True,
            "supports_residency": True,
        }
