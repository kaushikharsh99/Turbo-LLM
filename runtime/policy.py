"""Execution policy independent of model architecture and checkpoint storage."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ExecutionPolicy:
    device: str
    dtype: torch.dtype

    @classmethod
    def from_config(cls, config=None):
        execution = (config or {}).get("execution", {})
        device = execution.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if device not in {"cpu", "cuda", "mps"}:
            raise ValueError("execution.device must be auto, cpu, cuda or mps")
        name = execution.get("dtype", "fp32" if device == "cpu" else "fp16")
        types = {"fp16": torch.float16, "float16": torch.float16,
                 "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                 "fp32": torch.float32, "float32": torch.float32}
        if name not in types:
            raise ValueError(f"Unsupported execution.dtype: {name}")
        return cls(device, types[name])
