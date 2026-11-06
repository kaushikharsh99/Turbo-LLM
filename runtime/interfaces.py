"""Structural contracts for storage and execution extensions."""
from typing import Protocol, Mapping, Any
import torch


class CheckpointReader(Protocol):
    weight_map: Mapping[str, str]
    def get_tensor(self, name: str) -> torch.Tensor: ...


class GenerationBackend(Protocol):
    def generate(self, prompt: str, **options: Any) -> str: ...


class ExpertStore(Protocol):
    DEVICE: str
    dtype: torch.dtype
    def load_weight(self, name: str, device=None, dtype=None) -> torch.Tensor: ...
    def load_expert(self, layer_id: int, expert_id: int): ...
    def load_expert_dynamic(self, layer_id: int, expert_id: int): ...
    def close(self) -> None: ...
