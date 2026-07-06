"""
TurboPipeline: Async SSD -> RAM -> GPU Utility-Based Prefetch Engine
Hides expert load latency behind GPU compute using Working Set Estimation,
Utility Scoring, Multi-Level Priority Queues, and Bandwidth-Aware Prefetching.
"""
import time
import math
import queue
import threading
import torch
from typing import Dict, Set, List, Tuple, Optional

class WorkingSetEstimator:
    """
    Tracks expert access patterns across layers to estimate the active
    working set of experts likely needed over the next K layers.
    """
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        # key: layer_id -> Set of expert_ids seen historically
        self.layer_history: Dict[int, Set[int]] = {}
        # key: expert_id -> frequency across layers
        self.expert_frequency: Dict[int, int] = {}

    def record_layer_access(self, layer_id: int, expert_ids: List[int]):
        if layer_id not in self.layer_history:
            self.layer_history[layer_id] = set()
        self.layer_history[layer_id].update(expert_ids)
        
        for exp_id in expert_ids:
            self.expert_frequency[exp_id] = self.expert_frequency.get(exp_id, 0) + 1

    def estimate_working_set(self, current_layer: int, total_layers: int = 40) -> Set[Tuple[int, int]]:
        """
        Returns set of (layer_id, expert_id) expected in the lookahead window.
        """
        working_set = set()
        for offset in range(1, self.window_size + 1):
            next_layer = (current_layer + offset) % total_layers
            if next_layer in self.layer_history:
                for exp_id in self.layer_history[next_layer]:
                    working_set.add((next_layer, exp_id))
        return working_set


class UtilityScoringEngine:
    """
    Computes Utility Score for every expert candidate:
    Utility = (ReuseProb * LoadLatency * Confidence * PrefetchWindow) / MemoryCost
    """
    def __init__(
        self,
        w_reuse: float = 0.35,
        w_latency: float = 0.35,
        w_confidence: float = 0.20,
        w_cost: float = 0.10
    ):
        self.w_reuse = w_reuse
        self.w_latency = w_latency
        self.w_confidence = w_confidence
        self.w_cost = w_cost

    def compute_utility(
        self,
        reuse_prob: float,
        load_latency_ms: float,
        confidence: float,
        prefetch_window: int,
        memory_bytes: float
    ) -> float:
        norm_latency = max(0.1, load_latency_ms / 10.0)
        norm_cost = max(0.1, (memory_bytes / 1024.0**2) / 10.0)
        
        utility = (
            (reuse_prob ** self.w_reuse) *
            (norm_latency ** self.w_latency) *
            (confidence ** self.w_confidence) *
            prefetch_window
        ) / (norm_cost ** self.w_cost)
        
        return float(utility)


class MultiLevelPrefetchQueue:
    """
    3-Level Priority Queue:
    - HIGH PRIORITY: Imminent layer N+1 cache misses
    - MEDIUM PRIORITY: Working set candidates across next 2-4 layers
    - LOW PRIORITY: Historical hot experts (opportunistic prefetch when PCIe idle)
    """
    def __init__(self):
        self.high_prio = queue.Queue()
        self.med_prio = queue.Queue()
        self.low_prio = queue.Queue()
        self.pending_keys: Set[Tuple[int, int]] = set()
        self.lock = threading.Lock()

    def enqueue(self, layer_id: int, expert_id: int, priority: str = "high"):
        key = (layer_id, expert_id)
        with self.lock:
            if key in self.pending_keys:
                return
            self.pending_keys.add(key)
            
        item = (layer_id, expert_id)
        if priority == "high":
            self.high_prio.put(item)
        elif priority == "medium":
            self.med_prio.put(item)
        else:
            self.low_prio.put(item)

    def pop_next(self) -> Optional[Tuple[int, int]]:
        with self.lock:
            for q in (self.high_prio, self.med_prio, self.low_prio):
                if not q.empty():
                    item = q.get()
                    self.pending_keys.discard(item)
                    return item
        return None


class AsyncPrefetchPipeline:
    """
    Background worker thread preloading expert weights from SSD/RAM into GPU slots
    on a secondary CUDA transfer stream to hide load latency completely behind compute.
    """
    def __init__(self, expert_loader):
        self.loader = expert_loader
        self.working_set_est = WorkingSetEstimator(window_size=4)
        self.utility_engine = UtilityScoringEngine()
        
        # Priority prefetch queue
        self.queue = queue.PriorityQueue()  # (negative_utility, layer_id, expert_id)
        self.queued_keys: Set[Tuple[int, int]] = set()
        self.lock = threading.Lock()
        
        # Async worker thread
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        # Metrics (Phase 3)
        self.prefetched_count = 0
        self.hidden_latency_ms = 0.0
        self.bytes_transferred_pcie = 0
        self.bytes_saved_cache = 0

    def submit_prefetch_candidates(self, current_layer: int, total_layers: int = 40):
        working_set = self.working_set_est.estimate_working_set(current_layer, total_layers)
        
        with self.lock:
            for layer_id, exp_id in working_set:
                key = (layer_id, exp_id)
                if key in self.queued_keys:
                    continue
                if key in self.loader.expert_cache:
                    self.bytes_saved_cache += int(4.5 * 1024**2)
                    continue
                    
                confidence = 0.85 if layer_id == current_layer + 1 else 0.50
                utility = self.utility_engine.compute_utility(
                    reuse_prob=0.8,
                    load_latency_ms=15.0,
                    confidence=confidence,
                    prefetch_window=max(1, layer_id - current_layer),
                    memory_bytes=4.5 * 1024**2
                )
                
                self.queued_keys.add(key)
                self.queue.put((-utility, layer_id, exp_id))

    def _worker_loop(self):
        while self.running:
            try:
                neg_util, layer_id, exp_id = self.queue.get(timeout=0.05)
            except queue.Empty:
                continue
                
            key = (layer_id, exp_id)
            try:
                if key not in self.loader.expert_cache:
                    t0 = time.perf_counter()
                    # Perform background load directly into static GPU VRAM slots
                    self.loader.load_expert(layer_id, exp_id)
                    load_time_ms = (time.perf_counter() - t0) * 1000.0
                    expert_bytes = int(4.5 * 1024**2)
                    with self.lock:
                        self.prefetched_count += 1
                        self.hidden_latency_ms += load_time_ms
                        self.bytes_transferred_pcie += expert_bytes
            except Exception:
                pass
            finally:
                with self.lock:
                    self.queued_keys.discard(key)
                self.queue.task_done()

    def shutdown(self):
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=0.2)

    def get_summary(self) -> dict:
        transferred_mb = self.bytes_transferred_pcie / (1024.0 ** 2)
        saved_mb = self.bytes_saved_cache / (1024.0 ** 2)
        return {
            "prefetched_experts": self.prefetched_count,
            "hidden_load_latency": f"{self.hidden_latency_ms:.2f} ms",
            "bytes_transferred_pcie": f"{transferred_mb:.2f} MB",
            "bytes_saved_cache": f"{saved_mb:.2f} MB"
        }
