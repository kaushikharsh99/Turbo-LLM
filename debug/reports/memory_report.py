from typing import Dict, Any, List
from debug.profiler import Profiler

def generate_memory_report(profiler: Profiler) -> Dict[str, Any]:
    if not profiler.memory_stats:
        return {}
        
    vrams = [s["vram_mb"] for s in profiler.memory_stats]
    rams = [s["ram_mb"] for s in profiler.memory_stats]
    
    peak_vram = max(vrams) if vrams else 0.0
    peak_ram = max(rams) if rams else 0.0
    start_vram = profiler.memory_stats[0]["vram_mb"]
    start_ram = profiler.memory_stats[0]["ram_mb"]
    
    return {
        "peak_vram_mb": peak_vram,
        "peak_ram_mb": peak_ram,
        "start_vram_mb": start_vram,
        "start_ram_mb": start_ram,
        "vram_delta_mb": peak_vram - start_vram,
        "ram_delta_mb": peak_ram - start_ram,
        "history": profiler.memory_stats
    }
