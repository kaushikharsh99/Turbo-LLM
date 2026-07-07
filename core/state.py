from dataclasses import dataclass, field

import torch

@dataclass(slots=True)
class RuntimeStats:

    generated_tokens: int = 0

    current_layer: int = 0

    current_step: int = 0

    sequence_length: int = 0

@dataclass(slots=True)
class RuntimeBuffers:

    hidden_states: torch.Tensor | None = None

    logits: torch.Tensor | None = None

    input_ids: torch.Tensor | None = None

    position_ids: torch.Tensor | None = None

@dataclass(slots=True)
class LoadedState:

    current_layer: int = -1

    next_layer: int = -1

    loaded_layers: set[int] = field(default_factory=set)

@dataclass(slots=True)
class EngineState:

    stats: RuntimeStats = field(default_factory=RuntimeStats)

    buffers: RuntimeBuffers = field(default_factory=RuntimeBuffers)

    loaded: LoadedState = field(default_factory=LoadedState)
    kv_cache = None

    profiler = None

    scheduler = None