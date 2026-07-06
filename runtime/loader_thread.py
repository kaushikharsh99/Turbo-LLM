import threading
import queue
import torch
import traceback
from typing import Optional, List, Tuple, Any
from runtime.futures import LayerFuture
from runtime.queues import PrefetchQueue

class LoaderThread:
    """
    Dedicated background worker thread for disk I/O (SSD -> Pinned RAM)
    and non-blocking GPU upload on copy_stream.
    """
    def __init__(self, loader, queue: PrefetchQueue):
        self.loader = loader
        self.queue = queue
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="TurboLLM-LoaderThread", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        try:
            self.queue.put(None, block=False)
        except Exception:
            pass
        self._thread.join(timeout=2.0)
        self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                task = self.queue.get(block=True, timeout=0.1)
                if task is None or self._stop_event.is_set():
                    break
                
                layer_id, expert_ids, future, copy_stream = task
                self._prefetch_layer(layer_id, expert_ids, copy_stream)
                future.set_result(True)
            except queue.Empty:
                continue
            except Exception as e:
                if 'future' in locals() and future is not None and not future.is_done():
                    future.set_exception(e)

    def _prefetch_layer(self, layer_id: int, expert_ids: Optional[List[int]], copy_stream: Optional[Any] = None) -> None:
        """
        Prefetches raw FP8 weights for layer_id from SSD into Pinned RAMCache (CPU memory),
        and issues non-blocking GPU transfer on copy_stream.
        """
        if not expert_ids:
            if hasattr(self.loader, "get_historical_experts"):
                expert_ids = self.loader.get_historical_experts(layer_id)
            else:
                expert_ids = list(range(8))

        # 1. Load from SSD to Pinned RAM
        for exp_id in expert_ids:
            key = (layer_id, exp_id)
            with self.loader.lock:
                if self.loader.ram_cache.get(key) is not None:
                    continue
                    
            try:
                gate_tensor_name = self.loader.layout.gate_tensor(layer_id, exp_id)
                gate_fp8_cpu = self.loader._get_tensor(gate_tensor_name)
                gate_scale_name = gate_tensor_name + "_scale_inv"
                gate_scale_cpu = self.loader._get_tensor(gate_scale_name) if gate_scale_name in self.loader.weight_map else None

                up_tensor_name = self.loader.layout.up_tensor(layer_id, exp_id)
                up_fp8_cpu = self.loader._get_tensor(up_tensor_name)
                up_scale_name = up_tensor_name + "_scale_inv"
                up_scale_cpu = self.loader._get_tensor(up_scale_name) if up_scale_name in self.loader.weight_map else None

                down_tensor_name = self.loader.layout.down_tensor(layer_id, exp_id)
                down_fp8_cpu = self.loader._get_tensor(down_tensor_name)
                down_scale_name = down_tensor_name + "_scale_inv"
                down_scale_cpu = self.loader._get_tensor(down_scale_name) if down_scale_name in self.loader.weight_map else None

                cached_expert = (
                    self.loader._pin_tensor(gate_fp8_cpu), self.loader._pin_tensor(gate_scale_cpu),
                    self.loader._pin_tensor(up_fp8_cpu), self.loader._pin_tensor(up_scale_cpu),
                    self.loader._pin_tensor(down_fp8_cpu), self.loader._pin_tensor(down_scale_cpu)
                )
                with self.loader.lock:
                    self.loader.ram_cache.put(key, cached_expert)
            except Exception:
                pass

        # 2. Issue non-blocking GPU transfer & pre-dequantize into static FP16 GPU slots
        if copy_stream is not None and torch.cuda.is_available():
            with torch.cuda.stream(copy_stream):
                for exp_id in expert_ids:
                    try:
                        self.loader.load_expert(layer_id, exp_id)
                    except Exception:
                        pass
