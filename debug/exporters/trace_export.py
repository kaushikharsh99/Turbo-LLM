import json
import os
from debug.profiler import Profiler

def export_chrome_trace(profiler: Profiler, filepath: str):
    events = []
    
    if not profiler.timeline.events:
        return
        
    # Find the earliest start time to align the trace to 0
    t_zero = min(e.start_time for e in profiler.timeline.events)
    
    for event in profiler.timeline.events:
        if event.duration_ms is None:
            continue
            
        start_us = int((event.start_time - t_zero) * 1_000_000)
        dur_us = int(event.duration_ms * 1000)
        
        cat = event.event_type
        
        args = event.metadata.copy()
        if event.token_idx is not None:
            args["token_idx"] = event.token_idx
        if event.layer_id is not None:
            args["layer_id"] = event.layer_id
            
        events.append({
            "name": event.name,
            "cat": cat,
            "ph": "X",
            "ts": start_us,
            "dur": dur_us,
            "pid": os.getpid(),
            "tid": event.layer_id if event.layer_id is not None else 0,
            "args": args
        })
        
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump({"traceEvents": events}, f, indent=4)
