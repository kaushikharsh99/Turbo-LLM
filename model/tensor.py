from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


@dataclass(slots=True)
class Tensor:

    name: str

    shape: tuple[int, ...]

    dtype: torch.dtype

    file_path: Optional[Path] = None

    file_offset: int = 0

    size_bytes: int = 0

    device: str = "disk"

    loaded: bool = False

    pinned: bool = False

    data: Optional[torch.Tensor] = field(default=None, repr=False)

    def numel(self) -> int:
        return int(torch.Size(self.shape).numel())

    @property
    def is_gpu(self) -> bool:
        return self.device == "cuda"

    @property
    def is_cpu(self) -> bool:
        return self.device == "cpu"

    @property
    def is_disk(self) -> bool:
        return self.device == "disk"