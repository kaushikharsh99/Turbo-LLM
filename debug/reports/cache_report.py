from typing import Dict, Any, List
from debug.profiler import Profiler

def generate_cache_report(profiler: Profiler) -> Dict[str, Any]:
    gpu_hits = 0
    ram_hits = 0
    ssd_hits = 0
    
    # Layer specific cache sources
    layer_sources = {}  # layer_id -> {GPU: c, RAM: c, SSD: c}
    
    for stat in profiler.cache_stats:
        layer_id = stat["layer_id"]
        source = stat["source"]
        
        if source == "GPU":
            gpu_hits += 1
        elif source == "RAM":
            ram_hits += 1
        elif source == "SSD":
            ssd_hits += 1
            
        if layer_id not in layer_sources:
            layer_sources[layer_id] = {"GPU": 0, "RAM": 0, "SSD": 0}
        layer_sources[layer_id][source] += 1

    total = gpu_hits + ram_hits + ssd_hits
    hit_rate = (gpu_hits + ram_hits) / total * 100 if total > 0 else 0.0
    gpu_hit_rate = gpu_hits / total * 100 if total > 0 else 0.0

    return {
        "total_requests": total,
        "gpu_hits": gpu_hits,
        "ram_hits": ram_hits,
        "ssd_hits": ssd_hits,
        "hit_rate": hit_rate,
        "gpu_hit_rate": gpu_hit_rate,
        "layer_sources": layer_sources
    }
