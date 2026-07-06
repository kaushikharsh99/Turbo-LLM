import os
import json
import time
from typing import Dict, List, Tuple, Any, Optional

DEFAULT_EKB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cache",
    "expert_database.json"
)

class ExpertKnowledgeBase:
    """
    Persistent Self-Optimizing Expert Knowledge Base (EKB).
    Tracks expert reuse statistics and transition graphs using Exponential Moving Averages (EMA).
    """
    def __init__(self, db_path: str = DEFAULT_EKB_PATH, alpha: float = 0.05):
        self.db_path = db_path
        self.alpha = alpha  # EMA smoothing factor (0.05 = 5% weight to new observations)
        self.expert_stats: Dict[str, Dict[str, Any]] = {}
        self.transitions: Dict[str, Dict[str, Dict[str, float]]] = {}  # layer -> src_exp -> dst_exp -> weight
        self.load()

    def _key(self, layer_id: int, expert_id: int) -> str:
        return f"{layer_id}:{expert_id}"

    def update_expert_stat(
        self,
        layer_id: int,
        expert_id: int,
        load_time_ms: float = 1.0,
        was_gpu_hit: bool = False,
        was_ssd_load: bool = False,
        reuse_distance: Optional[float] = None
    ) -> None:
        key = self._key(layer_id, expert_id)
        if key not in self.expert_stats:
            self.expert_stats[key] = {
                "layer": layer_id,
                "expert": expert_id,
                "access_count": 0,
                "gpu_hits": 0,
                "ssd_loads": 0,
                "avg_reuse_distance": 20.0,
                "average_load_time_ms": load_time_ms,
                "keep_probability": 0.5,
                "last_updated": time.time()
            }

        stat = self.expert_stats[key]
        stat["access_count"] += 1
        if was_gpu_hit:
            stat["gpu_hits"] += 1
        if was_ssd_load:
            stat["ssd_loads"] += 1

        # EMA updates for load time and reuse distance
        stat["average_load_time_ms"] = (1.0 - self.alpha) * stat["average_load_time_ms"] + self.alpha * load_time_ms
        if reuse_distance is not None:
            stat["avg_reuse_distance"] = (1.0 - self.alpha) * stat["avg_reuse_distance"] + self.alpha * reuse_distance

        # Compute learned keep probability: higher hit rate & shorter reuse distance -> higher probability
        hit_ratio = stat["gpu_hits"] / max(1, stat["access_count"])
        dist_factor = 1.0 / (stat["avg_reuse_distance"] + 1.0)
        stat["keep_probability"] = min(1.0, max(0.0, 0.7 * hit_ratio + 0.3 * (dist_factor * 10.0)))
        stat["last_updated"] = time.time()

    def record_layer_transitions(self, layer_id: int, src_experts: List[int], dst_experts: List[int]) -> None:
        """
        Record transitions between layer L experts and layer L+1 experts to build the Expert Transition Graph.
        """
        l_str = str(layer_id)
        if l_str not in self.transitions:
            self.transitions[l_str] = {}

        layer_trans = self.transitions[l_str]
        for src in src_experts:
            src_str = str(src)
            if src_str not in layer_trans:
                layer_trans[src_str] = {}
            
            for dst in dst_experts:
                dst_str = str(dst)
                old_val = layer_trans[src_str].get(dst_str, 0.0)
                # EMA update for transition weight
                layer_trans[src_str][dst_str] = (1.0 - self.alpha) * old_val + self.alpha * 1.0

    def predict_next_layer_experts(self, layer_id: int, current_active_experts: List[int], top_k: int = 8, min_prob: float = 0.15) -> List[int]:
        """
        TurboPredict v0: Predicts upcoming Layer L+1 experts using the learned transition graph.
        """
        l_str = str(layer_id)
        if l_str not in self.transitions:
            return []

        scores: Dict[int, float] = {}
        layer_trans = self.transitions[l_str]

        for src in current_active_experts:
            src_str = str(src)
            if src_str in layer_trans:
                for dst_str, weight in layer_trans[src_str].items():
                    dst_id = int(dst_str)
                    scores[dst_id] = scores.get(dst_id, 0.0) + weight

        # Sort predicted experts by accumulated transition probability score
        sorted_preds = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        predicted = [dst_id for dst_id, score in sorted_preds if score >= min_prob]
        return predicted[:top_k]

    def save(self) -> None:
        """Persists updated knowledge base to JSON disk file."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        data = {
            "expert_stats": self.expert_stats,
            "transitions": self.transitions
        }
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Loads persistent knowledge base from JSON disk file if available."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.expert_stats = data.get("expert_stats", {})
                    self.transitions = data.get("transitions", {})
            except Exception:
                pass


extern_ekb = ExpertKnowledgeBase()
