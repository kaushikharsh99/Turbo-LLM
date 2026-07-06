import os
from typing import Dict, Any

def export_expert_heat_map(profiler, output_path: str) -> str:
    """
    Exports a detailed Expert Heat Map and GPU Residency Report.
    """
    loader = getattr(profiler, "loader", None)
    if loader is None or not hasattr(loader, "residency_mgr") or loader.residency_mgr is None:
        report_text = "Expert Residency Data: Not available.\n"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        return report_text

    heat_map = loader.residency_mgr.get_heat_map_summary()

    lines = []
    lines.append("=" * 60)
    lines.append("EXPERT RESIDENCY & HEAT MAP REPORT")
    lines.append("=" * 60)
    lines.append(f"Total Expert Accesses : {heat_map['total_accesses']}")
    lines.append(f"Total SSD Reloads    : {heat_map['total_reloads']}")
    
    total_hits = loader.gpu_hits + loader.ram_hits + loader.ssd_hits
    gpu_hit_pct = (loader.gpu_hits / total_hits * 100.0) if total_hits > 0 else 0.0
    lines.append(f"Overall GPU Hit Rate : {gpu_hit_pct:.1f}%")
    lines.append("-" * 60)
    lines.append("Top Active Experts (Heat Map):")
    lines.append("-" * 60)

    for i, exp in enumerate(heat_map["top_experts"], 1):
        l_id = exp["layer_id"]
        e_id = exp["expert_id"]
        count = exp["access_count"]
        avg_dist = exp["avg_reuse_distance"]
        reloads = exp["ssd_reloads"]
        bar_len = min(20, count)
        bar = "█" * bar_len
        lines.append(
            f"{i:<2}. Layer {l_id:<2} Expert {e_id:<2} | Accesses: {count:<3} | {bar:<20} | Avg Reuse Dist: {avg_dist:.1f} layers | Reloads: {reloads}"
        )

    lines.append("=" * 60)

    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text
