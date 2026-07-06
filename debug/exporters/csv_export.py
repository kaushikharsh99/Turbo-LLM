import csv
import os
from debug.profiler import Profiler
from debug.events import EventType

def export_csvs(profiler: Profiler, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. token_stats.csv
    token_stats_path = os.path.join(output_dir, "token_stats.csv")
    with open(token_stats_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token_idx", "name", "duration_ms"])
        for event in profiler.timeline.events:
            if event.token_idx is not None and event.event_type not in (EventType.GLOBAL, EventType.LAYER):
                writer.writerow([event.token_idx, event.name, event.duration_ms])

    # 2. cache.csv
    cache_path = os.path.join(output_dir, "cache.csv")
    with open(cache_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token_idx", "layer_id", "expert_id", "source"])
        for stat in profiler.cache_stats:
            writer.writerow([stat["token_idx"], stat["layer_id"], stat["expert_id"], stat["source"]])

    # 3. memory.csv
    memory_path = os.path.join(output_dir, "memory.csv")
    with open(memory_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token_idx", "layer_id", "event", "vram_mb", "ram_mb"])
        for stat in profiler.memory_stats:
            writer.writerow([stat["token_idx"], stat["layer_id"], stat["event"], stat["vram_mb"], stat["ram_mb"]])
