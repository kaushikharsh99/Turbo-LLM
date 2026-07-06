from dataclasses import dataclass, field
from typing import Dict, Any, Optional

class EventType:
    GLOBAL = "global"
    TOKEN = "token"
    LAYER = "layer"
    ROUTER = "router"
    ATTENTION = "attention"
    LOOKUP = "lookup"
    LOAD = "load"
    GPU_COPY = "gpu_copy"
    DEQUANT = "dequant"
    GEMM = "gemm"
    SHARED_EXPERT = "shared_expert"
    NORM = "norm"
    SAMPLING = "sampling"
    LM_HEAD = "lm_head"
    RESIDUAL = "residual"
    OVERHEAD = "overhead"

@dataclass
class ProfileEvent:
    name: str
    event_type: str
    start_time: float  # Time in CPU perf_counter seconds
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    token_idx: Optional[int] = None
    layer_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
