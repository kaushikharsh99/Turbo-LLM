import json
import os
from debug.profiler import Profiler

def export_json(profiler: Profiler, filepath: str):
    data = []
    for event in profiler.timeline.events:
        data.append({
            "name": event.name,
            "event_type": event.event_type,
            "start_time_s": event.start_time,
            "duration_ms": event.duration_ms,
            "token_idx": event.token_idx,
            "layer_id": event.layer_id,
            "metadata": event.metadata
        })
        
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
