import time
import torch
from typing import Dict, List, Set, Tuple, Optional, Any

class ExpertTracer:
    """
    Traces expert access patterns across layers and tokens to compute reuse distance,
    lifetimes, and access heat maps.
    """
    def __init__(self):
        self.access_history: List[Tuple[int, int, int, float]] = []  # (token_idx, layer_id, expert_id, timestamp)
        self.last_seen: Dict[Tuple[int, int], int] = {}              # (layer_id, expert_id) -> step_index
        self.reuse_distances: Dict[Tuple[int, int], List[int]] = {}  # (layer_id, expert_id) -> list of distances
        self.access_counts: Dict[Tuple[int, int], int] = {}          # (layer_id, expert_id) -> count
        self.ssd_reloads: Dict[Tuple[int, int], int] = {}           # (layer_id, expert_id) -> count
        self.total_accesses = 0
        self.step_counter = 0

    def record_access(self, token_idx: int, layer_id: int, expert_id: int, was_reloaded: bool = False) -> None:
        key = (layer_id, expert_id)
        self.total_accesses += 1
        self.step_counter += 1
        self.access_counts[key] = self.access_counts.get(key, 0) + 1

        if key in self.last_seen:
            distance = self.step_counter - self.last_seen[key]
            if key not in self.reuse_distances:
                self.reuse_distances[key] = []
            self.reuse_distances[key].append(distance)

        self.last_seen[key] = self.step_counter
        self.access_history.append((token_idx, layer_id, expert_id, time.perf_counter()))

        if was_reloaded:
            self.ssd_reloads[key] = self.ssd_reloads.get(key, 0) + 1

        from runtime.expert_knowledge_base import extern_ekb
        avg_dist = (sum(self.reuse_distances[key]) / len(self.reuse_distances[key])) if key in self.reuse_distances and self.reuse_distances[key] else None
        extern_ekb.update_expert_stat(
            layer_id,
            expert_id,
            load_time_ms=1.0,
            was_gpu_hit=not was_reloaded,
            was_ssd_load=was_reloaded,
            reuse_distance=avg_dist
        )

    def get_predicted_reuse_distance(self, layer_id: int, expert_id: int) -> float:
        key = (layer_id, expert_id)
        distances = self.reuse_distances.get(key)
        if not distances:
            return 999.0  # High default distance for new experts
        return sum(distances) / len(distances)


class ResidencyManager:
    """
    Layer-Aware GPU Residency Manager.
    Calculates retention scores based on reuse probability, reuse distance, and load cost.
    """
    def __init__(self, loader, max_vram_gb: float = 5.8):
        self.loader = loader
        self.max_vram_gb = max_vram_gb
        self.tracer = ExpertTracer()
        self.turbopredict_hooks: Set[Tuple[int, int]] = set()  # (layer_id, expert_id) predicted for near future

    def register_future_prediction(self, layer_id: int, expert_id: int) -> None:
        """Hook for TurboPredict to influence GPU residency."""
        self.turbopredict_hooks.add((layer_id, expert_id))

    def clear_future_predictions(self) -> None:
        self.turbopredict_hooks.clear()

    def compute_residency_score(self, layer_id: int, expert_id: int, hits: int, load_ms: float, vram_cost: float) -> float:
        key = (layer_id, expert_id)

        # 1. TurboPredict Hook override: infinite score if predicted for upcoming layer
        if key in self.turbopredict_hooks:
            return float('inf')

        # 2. Compute reuse distance score
        avg_distance = self.tracer.get_predicted_reuse_distance(layer_id, expert_id)
        distance_factor = 1.0 / (avg_distance + 1.0)

        # 3. Score = (Hits * Load_ms * DistanceFactor) / VRAM_cost
        cost = max(1.0, vram_cost)
        score = (hits * max(1.0, load_ms) * distance_factor * 1000.0) / cost
        return score

    def get_heat_map_summary(self) -> Dict[str, Any]:
        """Generate Expert Heat Map data and residency metrics."""
        sorted_experts = sorted(self.tracer.access_counts.items(), key=lambda x: x[1], reverse=True)
        top_experts = []
        for (l_id, exp_id), count in sorted_experts[:20]:
            distances = self.tracer.reuse_distances.get((l_id, exp_id), [])
            avg_dist = (sum(distances) / len(distances)) if distances else 0.0
            reloads = self.tracer.ssd_reloads.get((l_id, exp_id), 0)
            top_experts.append({
                "layer_id": l_id,
                "expert_id": exp_id,
                "access_count": count,
                "avg_reuse_distance": avg_dist,
                "ssd_reloads": reloads
            })

        total_reloads = sum(self.tracer.ssd_reloads.values())
        return {
            "total_accesses": self.tracer.total_accesses,
            "total_reloads": total_reloads,
            "top_experts": top_experts
        }
