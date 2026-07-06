import os
import psutil
import torch
import time
from typing import Dict, Any, List, Optional
from debug.events import ProfileEvent, EventType
from debug.timeline import Timeline

class Profiler:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.timeline = Timeline()
        self.pending_cuda_timers = []
        self.token_idx: Optional[int] = None
        self.layer_id: Optional[int] = None
        self.boundary_crossings = {
            "py_to_cpp": 0,
            "cpp_to_py": 0,
            "cuda_calls": 0,
            "syncs": 0
        }
        self.cache_stats = []  # list of dicts: {token_idx, layer_id, expert_id, source}
        self.memory_stats = []  # list of dicts: {token_idx, layer_id, event, vram_mb, ram_mb}
        self.model_info = {}
        self.process = psutil.Process(os.getpid())

    def reset(self):
        self.timeline.clear()
        self.pending_cuda_timers.clear()
        self.boundary_crossings = {
            "py_to_cpp": 0,
            "cpp_to_py": 0,
            "cuda_calls": 0,
            "syncs": 0
        }
        self.cache_stats.clear()
        self.memory_stats.clear()
        self.token_idx = None
        self.layer_id = None

    def set_token(self, idx: int):
        self.resolve_cuda_timers()
        self.token_idx = idx

    def set_layer(self, idx: int):
        self.layer_id = idx

    def record_event(self, name: str, event_type: str, duration_ms: float, token_idx: Optional[int] = None, layer_id: Optional[int] = None, metadata: Optional[dict] = None):
        t_idx = token_idx if token_idx is not None else self.token_idx
        l_idx = layer_id if layer_id is not None else self.layer_id
        event = ProfileEvent(
            name=name,
            event_type=event_type,
            start_time=time.perf_counter(),
            duration_ms=duration_ms,
            token_idx=t_idx,
            layer_id=l_idx,
            metadata=metadata or {}
        )
        self.timeline.record(event)

    def register_pending_cuda_timer(self, timer):
        t_idx = timer.token_idx if timer.token_idx is not None else self.token_idx
        l_idx = timer.layer_id if timer.layer_id is not None else self.layer_id
        self.pending_cuda_timers.append({
            "name": timer.name,
            "event_type": timer.event_type,
            "start_event": timer.start_event,
            "end_event": timer.end_event,
            "token_idx": t_idx,
            "layer_id": l_idx,
            "metadata": timer.metadata
        })

    def resolve_cuda_timers(self):
        if not self.pending_cuda_timers:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.boundary_crossings["syncs"] += 1
            
        for item in self.pending_cuda_timers:
            try:
                # elapsed_time returns milliseconds
                duration_ms = item["start_event"].elapsed_time(item["end_event"])
            except Exception:
                duration_ms = 0.0
            
            event = ProfileEvent(
                name=item["name"],
                event_type=item["event_type"],
                start_time=time.perf_counter(),  # approximate
                duration_ms=duration_ms,
                token_idx=item["token_idx"],
                layer_id=item["layer_id"],
                metadata=item["metadata"]
            )
            self.timeline.record(event)
        self.pending_cuda_timers.clear()

    def record_memory(self, label: str, token_idx: Optional[int] = None, layer_id: Optional[int] = None):
        t_idx = token_idx if token_idx is not None else self.token_idx
        l_idx = layer_id if layer_id is not None else self.layer_id
        
        vram = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        ram = self.process.memory_info().rss
        
        self.memory_stats.append({
            "token_idx": t_idx,
            "layer_id": l_idx,
            "event": label,
            "vram_mb": vram / 1024**2,
            "ram_mb": ram / 1024**2
        })

    def record_cache_lookup(self, layer_id: int, expert_id: int, source: str, token_idx: Optional[int] = None):
        t_idx = token_idx if token_idx is not None else self.token_idx
        self.cache_stats.append({
            "token_idx": t_idx,
            "layer_id": layer_id,
            "expert_id": expert_id,
            "source": source
        })

    def increment_crossing(self, crossing_type: str, count: int = 1):
        if crossing_type in self.boundary_crossings:
            self.boundary_crossings[crossing_type] += count
