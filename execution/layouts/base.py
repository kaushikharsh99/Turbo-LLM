from abc import ABC, abstractmethod


class BaseLayerLayout(ABC):

    @abstractmethod
    def preload_weights(self, layer):
        pass

    @abstractmethod
    def attention_module(self, layer_module):
        pass

    @abstractmethod
    def input_norm(self, layer_module):
        pass

    @abstractmethod
    def post_norm(self, layer_module):
        pass