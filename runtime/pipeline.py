import torch
from typing import Dict, Optional, List
from runtime.futures import LayerFuture
from runtime.queues import PrefetchQueue
from runtime.loader_thread import LoaderThread

class PipelineManager:
    """
    Pipeline Manager / Scheduler orchestrating asynchronous layer prefetching
    and CUDA stream-overlapped GPU uploads.
    """
    def __init__(self, loader, enabled: bool = True):
        self.loader = loader
        self.enabled = enabled
        self.queue = PrefetchQueue()
        self.loader_thread = LoaderThread(self.loader, self.queue)
        self.futures: Dict[int, LayerFuture] = {}
        self.last_active_experts: Dict[int, List[int]] = {}  # layer_id -> list of recent expert_ids
        
        # Dual CUDA Streams: Copy Stream (Stream 1) for async uploads & Compute Stream (Stream 0) for GEMM
        self.copy_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        if self.enabled:
            self.loader_thread.start()

    def record_layer_experts(self, layer_id: int, expert_ids: List[int]) -> None:
        """Record expert IDs used by a layer to update EKB graph transitions."""
        from runtime.expert_knowledge_base import extern_ekb
        prev_layer = layer_id - 1
        if prev_layer in self.last_active_experts:
            extern_ekb.record_layer_transitions(prev_layer, self.last_active_experts[prev_layer], expert_ids)
        self.last_active_experts[layer_id] = list(set(expert_ids))

    def submit_prefetch(self, layer_id: int, expert_ids: Optional[List[int]] = None) -> Optional[LayerFuture]:
        """Submit a prefetch request for layer_id using TurboPredict v0 EKB graph predictions."""
        if not self.enabled:
            return None

        # Check if already prefetched or in-flight
        if layer_id in self.futures and not self.futures[layer_id].is_done():
            return self.futures[layer_id]

        future = LayerFuture(layer_id)
        self.futures[layer_id] = future

        from runtime.expert_knowledge_base import extern_ekb
        # Predict experts using TurboPredict v0 Graph
        if expert_ids is None:
            prev_layer_active = self.last_active_experts.get(layer_id - 1, [])
            predicted = extern_ekb.predict_next_layer_experts(layer_id - 1, prev_layer_active)
            expert_ids = predicted if predicted else self.last_active_experts.get(layer_id, None)

        # Register predictions in ResidencyManager to guarantee residency
        if expert_ids and hasattr(self.loader, "residency_mgr") and self.loader.residency_mgr:
            for exp_id in expert_ids:
                self.loader.residency_mgr.register_future_prediction(layer_id, exp_id)

        task = (layer_id, expert_ids, future, self.copy_stream)
        try:
            self.queue.put(task, block=False)
        except Exception:
            future.set_result(True)
        return future

    def await_layer(self, layer_id: int, timeout: Optional[float] = 1.0) -> None:
        """Block until layer_id's prefetch completes and synchronize CUDA streams."""
        if not self.enabled:
            return
        future = self.futures.get(layer_id)
        if future is not None and not future.is_done():
            try:
                future.wait(timeout=timeout)
            except Exception:
                pass
                
        # Synchronize default compute stream with copy_stream on GPU side
        if torch.cuda.is_available() and self.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self.copy_stream)

    def stop(self) -> None:
        if self.enabled:
            self.loader_thread.stop()
            self.queue.clear()
            self.futures.clear()
