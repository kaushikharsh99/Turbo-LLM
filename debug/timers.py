import time
import torch
from typing import Optional

class CPUTimer:
    def __init__(self, name: str, profiler, event_type: str, token_idx: Optional[int] = None, layer_id: Optional[int] = None, metadata: Optional[dict] = None):
        self.name = name
        self.profiler = profiler
        self.event_type = event_type
        self.token_idx = token_idx
        self.layer_id = layer_id
        self.metadata = metadata or {}
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000.0
        self.profiler.record_event(
            name=self.name,
            event_type=self.event_type,
            duration_ms=duration_ms,
            token_idx=self.token_idx,
            layer_id=self.layer_id,
            metadata=self.metadata
        )

class CUDATimer:
    def __init__(self, name: str, profiler, event_type: str, token_idx: Optional[int] = None, layer_id: Optional[int] = None, metadata: Optional[dict] = None):
        self.name = name
        self.profiler = profiler
        self.event_type = event_type
        self.token_idx = token_idx
        self.layer_id = layer_id
        self.metadata = metadata or {}
        self.start_event = None
        self.end_event = None
        self.start_cpu = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        self.start_cpu = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            self.end_event.record()
            self.profiler.register_pending_cuda_timer(self)
        else:
            duration_ms = (time.perf_counter() - self.start_cpu) * 1000.0
            self.profiler.record_event(
                name=self.name,
                event_type=self.event_type,
                duration_ms=duration_ms,
                token_idx=self.token_idx,
                layer_id=self.layer_id,
                metadata=self.metadata
            )
