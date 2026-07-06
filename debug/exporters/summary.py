import os
from debug.profiler import Profiler
from debug.events import EventType
from debug.reports import generate_bottleneck_report, generate_cache_report, generate_memory_report

def export_summary(profiler: Profiler, filepath: str) -> str:
    bottlenecks = generate_bottleneck_report(profiler)
    cache = generate_cache_report(profiler)
    memory = generate_memory_report(profiler)

    model_name = profiler.model_info.get("model", "N/A")
    prompt = profiler.model_info.get("prompt", "N/A")
    prompt_len = profiler.model_info.get("prompt_len", 0)
    generated_tokens = profiler.model_info.get("generated_tokens", 0)
    ttft_s = profiler.model_info.get("ttft", 0.0)
    decode_tokens = profiler.model_info.get("decode_tokens", 0)
    decode_duration = profiler.model_info.get("decode_duration", 0.0)
    total_duration = profiler.model_info.get("total_duration", 0.0)
    
    decode_tps = decode_tokens / decode_duration if decode_duration > 0 else 0.0
    peak_vram = memory.get("peak_vram_mb", 0.0)
    peak_ram = memory.get("peak_ram_mb", 0.0)

    lines = []
    lines.append("===================================================")
    lines.append("Turbo Profile")
    lines.append("===================================================")
    lines.append(f"Model:              {model_name}")
    lines.append(f"Prompt:             {prompt}")
    lines.append(f"Prompt Length:      {prompt_len} tokens")
    lines.append(f"Generated Tokens:   {generated_tokens}")
    lines.append(f"TTFT:               {ttft_s:.2f} s")
    lines.append(f"Decode TPS:         {decode_tps:.2f} tokens/s")
    lines.append(f"Peak VRAM:          {peak_vram:.2f} MB")
    lines.append(f"Peak RAM:           {peak_ram:.2f} MB")
    lines.append(f"Total Runtime:      {total_duration:.2f} s")
    lines.append("===================================================\n")

    # Phase 2: Token Timeline
    lines.append("Token Timeline:")
    lines.append("-" * 60)
    token_events = [e for e in profiler.timeline.events if e.event_type == EventType.TOKEN]
    token_events = sorted(token_events, key=lambda x: x.token_idx or 0)
    
    for te in token_events:
        lines.append(f"Token {te.token_idx:<2} | Total Duration: {te.duration_ms:.2f} ms")
        sub_ops = {}
        for event in profiler.timeline.events:
            if event.token_idx == te.token_idx and event.event_type not in (EventType.TOKEN, EventType.GLOBAL):
                sub_ops[event.name] = sub_ops.get(event.name, 0.0) + event.duration_ms
        for op_name, op_ms in sorted(sub_ops.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  └─ {op_name:<20} : {op_ms:.2f} ms")
    lines.append("")

    # Phase 3 & 4: Layer Cache and Latency Timeline
    lines.append("Layer Breakdown & Cache Summary:")
    lines.append("-" * 60)
    layer_durations = {}
    for event in profiler.timeline.events:
        if event.layer_id is not None and event.event_type != EventType.TOKEN:
            if event.layer_id not in layer_durations:
                layer_durations[event.layer_id] = {}
            layer_durations[event.layer_id][event.name] = layer_durations[event.layer_id].get(event.name, 0.0) + event.duration_ms

    for l_id in sorted(layer_durations.keys()):
        ops = layer_durations[l_id]
        total_layer_ms = sum(ops.values())
        lines.append(f"Layer {l_id:<2} (Total: {total_layer_ms:.2f} ms)")
        for op_name, op_ms in ops.items():
            lines.append(f"  ├─ {op_name:<20} : {op_ms:.2f} ms")
            
        if l_id in cache.get("layer_sources", {}):
            sources = cache["layer_sources"][l_id]
            lines.append(f"  └─ Cache: GPU={sources.get('GPU', 0)} RAM={sources.get('RAM', 0)} SSD={sources.get('SSD', 0)}")
    lines.append("")

    # Phase 6 & 8: Python & Boundary Crossings
    lines.append("Python vs C++ vs CUDA Overhead:")
    lines.append("-" * 60)
    python_overhead_pct = (bottlenecks.get('python_overhead_ms', 0.0) / (total_duration * 1000.0)) * 100 if total_duration > 0 else 0.0
    lines.append(f"Python Host Overhead: {bottlenecks.get('python_overhead_ms', 0.0):.2f} ms ({python_overhead_pct:.1f}%)")
    lines.append(f"Python -> C++ Calls:  {profiler.boundary_crossings.get('py_to_cpp', 0)}")
    lines.append(f"C++ -> Python Calls:  {profiler.boundary_crossings.get('cpp_to_py', 0)}")
    lines.append(f"CUDA synchronizations:{profiler.boundary_crossings.get('syncs', 0)}")
    lines.append("")

    # Phase 10: Bottleneck Analysis
    lines.append("================ Bottlenecks ================")
    for rank, (name, val) in enumerate(bottlenecks["sorted_bottlenecks"], 1):
        pct = (val / bottlenecks["total_time_ms"]) * 100 if bottlenecks["total_time_ms"] > 0 else 0.0
        lines.append(f"{rank}. {name:<25} {val:.2f} ms ({pct:.1f}%)")
    lines.append("")
    lines.append("Recommendation:")
    lines.append(bottlenecks["recommendation"])
    lines.append("=============================================")

    summary_text = "\n".join(lines)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(summary_text)
        
    return summary_text
