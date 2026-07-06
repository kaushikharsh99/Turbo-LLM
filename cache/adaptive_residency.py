"""
TurboCache: Adaptive Residency Engine & Controller for LLM Expert Eviction
Implements a self-tuning, multi-term residency scoring function, online learning,
and eviction dataset sample collection.
"""
import math
from typing import Dict, Tuple, Optional, List

class EvictionSample:
    def __init__(self, layer_id: int, expert_id: int, evicted_step: int, residency_score: float):
        self.layer_id = layer_id
        self.expert_id = expert_id
        self.evicted_step = evicted_step
        self.residency_score = residency_score
        self.reloaded_step: Optional[int] = None
        self.is_bad_eviction: bool = False

class AdaptiveResidencyController:
    """
    TurboCache Controller:
    - Stage 1: Handcrafted Weighted Residency Score Baseline
    - Stage 2: Self-tuning Heuristic Weights via Runtime Eviction Feedback
    - Stage 3: Hybrid RL Nudge Safety Net Integration
    - Stage 4: Automatic Dataset & Sample Logging
    """
    def __init__(
        self,
        alpha: float = 1.0,   # Recency weight
        beta: float = 2.0,    # Frequency weight
        gamma: float = 1.5,   # Reuse Probability weight
        delta: float = 1.0,   # Load Latency weight
        epsilon: float = 0.5, # Memory Cost penalty
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        
        from cache.rl_controller import CacheRLAgent
        self.rl_agent = CacheRLAgent()
        
        self.stats: Dict[Tuple[int, int], dict] = {}
        self.current_step = 0
        self.eviction_logs: List[EvictionSample] = []
        
    def record_access(self, layer_id: int, expert_id: int, load_ms: float = 1.0, memory_bytes: float = 1.0):
        self.current_step += 1
        key = (layer_id, expert_id)
        
        if key not in self.stats:
            self.stats[key] = {
                "last_step": self.current_step,
                "frequency": 1,
                "reuse_count": 0,
                "load_ms": load_ms,
                "memory_bytes": memory_bytes,
            }
        else:
            stat = self.stats[key]
            step_delta = self.current_step - stat["last_step"]
            stat["last_step"] = self.current_step
            stat["frequency"] += 1
            if step_delta < 50:
                stat["reuse_count"] += 1
            stat["load_ms"] = load_ms
            stat["memory_bytes"] = memory_bytes

        # Check if this expert was prematurely evicted (feedback loop)
        for sample in reversed(self.eviction_logs[-50:]):
            if sample.layer_id == layer_id and sample.expert_id == expert_id and sample.reloaded_step is None:
                sample.reloaded_step = self.current_step
                steps_since_evict = self.current_step - sample.evicted_step
                if steps_since_evict < 40:
                    sample.is_bad_eviction = True
                    # Self-tune: wrong eviction -> increase frequency and recency weights
                    self.beta = min(5.0, self.beta * 1.02)
                    self.alpha = min(5.0, self.alpha * 1.01)

    def compute_residency_score(
        self,
        layer_id: int,
        expert_id: int,
        rl_nudge: float = 0.0
    ) -> float:
        key = (layer_id, expert_id)
        if key not in self.stats:
            return 0.0
            
        stat = self.stats[key]
        steps_ago = max(1, self.current_step - stat["last_step"])
        
        # 1. Recency: Exponential decay
        recency = math.exp(-0.05 * steps_ago)
        
        # 2. Frequency: Log-scaled count
        frequency = math.log1p(stat["frequency"])
        
        # 3. Reuse Probability: Ratio of quick reuses
        reuse_prob = stat["reuse_count"] / max(1, stat["frequency"])
        
        # 4. Load Latency: Normalized reload time
        load_latency = stat["load_ms"] / 10.0
        
        # 5. Memory Cost: Size cost penalty
        memory_cost = (stat["memory_bytes"] / (1024.0 * 1024.0)) / 10.0
        
        base_score = (
            self.alpha * recency +
            self.beta * frequency +
            self.gamma * reuse_prob +
            self.delta * load_latency
        )
        
        # Phase 2: Utility per Byte calculation (Favors smaller, high-value experts)
        expert_size_mb = max(0.1, stat["memory_bytes"] / (1024.0 * 1024.0))
        utility_per_byte = (base_score + rl_nudge) / expert_size_mb
        return utility_per_byte

    def log_eviction(self, layer_id: int, expert_id: int, score: float):
        sample = EvictionSample(
            layer_id=layer_id,
            expert_id=expert_id,
            evicted_step=self.current_step,
            residency_score=score
        )
        self.eviction_logs.append(sample)
        if len(self.eviction_logs) > 1000:
            self.eviction_logs.pop(0)

    def get_stats_summary(self) -> dict:
        bad_evictions = sum(1 for s in self.eviction_logs if s.is_bad_eviction)
        total_evictions = len(self.eviction_logs)
        bad_rate = (bad_evictions / total_evictions * 100.0) if total_evictions > 0 else 0.0
        res = {
            "alpha_recency": self.alpha,
            "beta_frequency": self.beta,
            "gamma_reuse": self.gamma,
            "total_evictions": total_evictions,
            "bad_eviction_rate": f"{bad_rate:.1f}%"
        }
        if hasattr(self, "rl_agent"):
            res["rl_agent"] = self.rl_agent.get_summary()
        return res
