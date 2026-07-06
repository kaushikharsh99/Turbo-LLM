from typing import Dict, Any, List
from debug.profiler import Profiler
from debug.events import EventType

def generate_bottleneck_report(profiler: Profiler) -> Dict[str, Any]:
    # Aggregate durations by event type
    totals = {}
    total_time = 0.0
    
    for event in profiler.timeline.events:
        if event.duration_ms is not None:
            # We don't want to double count nested events (e.g. TOKEN contains layers)
            if event.event_type in (EventType.GLOBAL, EventType.TOKEN, EventType.LAYER):
                continue
            totals[event.name] = totals.get(event.name, 0.0) + event.duration_ms
            total_time += event.duration_ms

    # Calculate Python Overhead during decode phase (token_idx > 0)
    decode_events = [e for e in profiler.timeline.events if e.event_type == EventType.TOKEN and e.token_idx is not None and e.token_idx > 0]
    total_decode_ms = sum(e.duration_ms for e in decode_events if e.duration_ms is not None)
    
    # Sum of all non-global, non-token, non-layer components during decode
    component_sum = 0.0
    for event in profiler.timeline.events:
        if event.token_idx is not None and event.token_idx > 0:
            if event.event_type not in (EventType.GLOBAL, EventType.TOKEN, EventType.LAYER):
                if event.duration_ms is not None:
                    component_sum += event.duration_ms
                    
    python_overhead_ms = max(0.0, total_decode_ms - component_sum)
    if python_overhead_ms > 0:
        totals["Python Host Overhead"] = python_overhead_ms
        total_time += python_overhead_ms

    # Sort bottlenecks
    sorted_bottlenecks = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    
    # Formulate recommendation
    recommendation = "No critical bottlenecks found."
    if sorted_bottlenecks:
        top_name, top_val = sorted_bottlenecks[0]
        top_pct = (top_val / total_time) * 100 if total_time > 0 else 0
        
        if "load" in top_name.lower() or "ssd" in top_name.lower():
            recommendation = "Highest impact optimization: Reduce SSD loading overhead. Consider caching more experts, enabling asynchronous preloading, or using a faster storage drive. Estimated gain: 20-30%."
        elif "gemm" in top_name.lower():
            recommendation = "Highest impact optimization: Grouped GEMM is the primary bottleneck. Consider compiling custom Triton or CUDA kernels, or upgrading to a card supporting FP8 tensor cores directly."
        elif "dequant" in top_name.lower():
            recommendation = "Highest impact optimization: Dequantization overhead is high. Consider using pre-dequantized FP16 weights in cache or preloading dequantized layers."
        elif "attention" in top_name.lower():
            recommendation = "Highest impact optimization: Attention computation dominates. Install flash-linear-attention or Dao-AILab/causal-conv1d for fast-path hardware-accelerated kernels."
        elif "python" in top_name.lower() or "host" in top_name.lower():
            recommendation = "Highest impact optimization: Python host code overhead is significant. Move the remaining Python LayerExecutor loop and scheduler orchestration into C++."
        else:
            recommendation = f"Highest impact optimization: Focus on {top_name} (occupying {top_pct:.1f}% of total MoE/MLP runtime)."

    return {
        "totals": totals,
        "total_time_ms": total_time,
        "sorted_bottlenecks": sorted_bottlenecks,
        "recommendation": recommendation,
        "python_overhead_ms": python_overhead_ms,
        "total_decode_ms": total_decode_ms
    }
