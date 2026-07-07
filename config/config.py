
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(slots=True)
class EngineConfig:

    model_path: Path

    device: str = "cuda"

    dtype: torch.dtype = torch.float16

    max_vram_gb: float = 5.5

    max_ram_gb: float = 15.0

    max_new_tokens: int = 128

    temperature: float = 0.7

    top_p: float = 0.95

    batch_size: int = 1

    chat_mode: bool = True

    batch_mode: bool = False

    layer_streaming: bool = False

    double_buffering: bool = False

    async_loading: bool = False

    profiling: bool = True

    benchmark: bool = False

    verbose: bool = True

    validate_outputs: bool = False