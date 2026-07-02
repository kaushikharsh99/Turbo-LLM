import os
import sys
import time
import torch
import gc
import queue
import threading
import psutil
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights

from loader.expert_loader import ExpertLoader
from runtime.model_factory import create_adapter
from runtime.engine import TurboEngine
from loader.model_manager import get_model, MODELS

class LogInterceptor:
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        self.queues = []
        self.history = []
        self.lock = threading.Lock()

    def write(self, message):
        self.original_stdout.write(message)
        if message.strip():
            with self.lock:
                self.history.append(message.strip())
                if len(self.history) > 500:
                    self.history.pop(0)
                for q in self.queues:
                    try:
                        q.put_nowait(message.strip())
                    except queue.Full:
                        pass

    def flush(self):
        self.original_stdout.flush()

    def register_queue(self, q):
        with self.lock:
            if q not in self.queues:
                self.queues.append(q)

    def unregister_queue(self, q):
        with self.lock:
            if q in self.queues:
                self.queues.remove(q)


class ServerMetricsCollector:
    def __init__(self, manager, token_queue):
        self.manager = manager
        self.token_queue = token_queue
        self.num_layers = None
        self.current_token = None
        self.start_time = None
        self.last_token_time = None
        self.generation_step = 0
        self.ttft = None
        self.thinking = "off"

    def begin_token(self, token_id, token_text, position, generation_step, thinking, temperature, top_p):
        self.generation_step = generation_step
        self.thinking = "on" if thinking else "off"
        if self.start_time is None:
            self.start_time = time.time()
        self.last_token_time = time.time()
        self.current_token = {
            "token_id": token_id,
            "token_text": token_text,
            "position": position,
            "generation_step": generation_step,
            "thinking": thinking,
            "temperature": temperature,
            "top_p": top_p,
            "layers": [None] * (self.num_layers or 0)
        }

    def record_layer(self, layer_id, experts, scores):
        if self.current_token and layer_id < len(self.current_token["layers"]):
            self.current_token["layers"][layer_id] = {
                "experts": experts,
                "scores": scores,
                "load_ms": 0.0,
                "dequant_ms": 0.0,
                "evict_ms": 0.0,
                "gemm_ms": 0.0,
            }

    def record_layer_timing(self, layer_id, load_ms, dequant_ms, evict_ms, gemm_ms):
        if self.current_token and layer_id < len(self.current_token["layers"]):
            layer_data = self.current_token["layers"][layer_id]
            if layer_data:
                layer_data.update({
                    "load_ms": load_ms,
                    "dequant_ms": dequant_ms,
                    "evict_ms": evict_ms,
                    "gemm_ms": gemm_ms,
                })
                # Broadcast live pipeline/layer info to websocket
                self.manager.broadcast_pipeline_update({
                    "type": "layer_progress",
                    "layer_id": layer_id,
                    "experts": layer_data["experts"],
                    "scores": layer_data["scores"],
                    "load_ms": load_ms,
                    "dequant_ms": dequant_ms,
                    "evict_ms": evict_ms,
                    "gemm_ms": gemm_ms,
                    "status": "Done" if load_ms + dequant_ms + gemm_ms > 0 else "Executing"
                })

    def finish_token(self):
        now = time.time()
        if self.generation_step == 1:
            self.ttft = now - self.start_time
            
        token_latency = now - (self.last_token_time or self.start_time)
        tps = 1.0 / token_latency if token_latency > 0 else 0.0
        
        # Calculate SSD/RAM/GPU hit rates from loader
        gpu_pct = 0.0
        ram_pct = 0.0
        ssd_pct = 0.0
        if self.manager.loader:
            total_hits = self.manager.loader.gpu_hits + self.manager.loader.ram_hits + self.manager.loader.ssd_hits
            if total_hits > 0:
                gpu_pct = (self.manager.loader.gpu_hits / total_hits) * 100
                ram_pct = (self.manager.loader.ram_hits / total_hits) * 100
                ssd_pct = (self.manager.loader.ssd_hits / total_hits) * 100

        # Construct final stats object for this token
        stats_payload = {
            "step": self.generation_step,
            "token": self.current_token["token_text"],
            "token_id": self.current_token["token_id"],
            "tps": tps,
            "latency_ms": token_latency * 1000.0,
            "ttft_ms": (self.ttft * 1000.0) if self.ttft else None,
            "gpu_hit_rate": gpu_pct,
            "ram_hit_rate": ram_pct,
            "ssd_hit_rate": ssd_pct,
            "layers": self.current_token["layers"]
        }

        # Save latest metrics to engine manager
        self.manager.last_metrics = stats_payload

        # Put the token and metrics in the queue for streaming
        self.token_queue.put({
            "type": "token",
            "token": self.current_token["token_text"],
            "token_id": self.current_token["token_id"],
            "step": self.generation_step,
            "metrics": stats_payload
        })
        
        # Broadcast stats via WebSocket
        self.manager.broadcast_stats_update(stats_payload)


class EngineManager:
    def __init__(self):
        self.active_model_id = None
        self.model_path = None
        self.status = "Unloaded"  # Unloaded, Loading, Loaded, Generating, Error
        self.error_message = None
        
        self.loader = None
        self.config = {
            "model": {"path": "./model"},
            "runtime": {"max_new_tokens": 128, "temperature": 0.7, "top_p": 0.95},
            "cache": {"gpu_limit": "auto", "ram_limit": "auto", "expert_limit": "auto"},
            "memory": {"max_vram_mb": 5800, "max_ram_percent": 35},
            "execution": {"dtype": "fp8", "profiling": True}
        }
        self.engine = None
        self.lock = threading.Lock()
        
        # Intercept output
        self.log_interceptor = LogInterceptor(sys.stdout)
        sys.stdout = self.log_interceptor

        # WebSocket subscriptions
        self.stats_websockets = []
        self.logs_websockets = []
        self.pipeline_websockets = []
        
        self.last_metrics = {}
        self.running_thread = None

        # Start stats broadcasting thread
        self.stats_thread = threading.Thread(target=self._broadcast_system_stats_loop, daemon=True)
        self.stats_thread.start()

    def get_installed_models(self):
        installed = []
        # Check standard cache directory ~/.turbollm/models
        if MODELS.exists():
            for folder in MODELS.iterdir():
                if folder.is_dir():
                    # Check if config.json or other weights exist
                    config_json = folder / "config.json"
                    model_name = folder.name.replace("_", "/")
                    installed.append({
                        "id": model_name,
                        "path": str(folder),
                        "status": "Installed" if self.active_model_id != model_name else "Running",
                        "size_gb": sum(f.stat().st_size for f in folder.glob('**/*') if f.is_file()) / (1024**3)
                    })
        # Check local model folder if it exists
        local_model_path = Path("./model")
        if local_model_path.exists():
            installed.append({
                "id": "local-model",
                "path": str(local_model_path.resolve()),
                "status": "Installed" if self.active_model_id != "local-model" else "Running",
                "size_gb": sum(f.stat().st_size for f in local_model_path.glob('**/*') if f.is_file()) / (1024**3)
            })
        return installed

    def load_model(self, model_id_or_path, custom_config=None):
        with self.lock:
            if self.status == "Loading":
                raise ValueError("A model is already loading.")
            
            self.unload_model_internal()
            self.status = "Loading"
            self.active_model_id = model_id_or_path
            self.error_message = None

        def _async_load():
            try:
                print(f"Server loading model: {model_id_or_path}")
                resolved_path = get_model(model_id_or_path)
                self.model_path = resolved_path
                
                # Merge custom config
                if custom_config:
                    for k, v in custom_config.items():
                        if isinstance(v, dict) and k in self.config:
                            self.config[k].update(v)
                        else:
                            self.config[k] = v
                
                # Create config and meta model
                hf_config = AutoConfig.from_pretrained(resolved_path, trust_remote_code=True)
                dtype = torch.float16
                if "execution" in self.config and "dtype" in self.config["execution"]:
                    dtype_str = self.config["execution"]["dtype"]
                    if dtype_str in ("bf16", "bfloat16"):
                        dtype = torch.bfloat16
                    elif dtype_str in ("fp32", "float32"):
                        dtype = torch.float32

                loader = ExpertLoader(resolved_path, config=self.config)
                arch = hf_config.architectures[0]
                is_moe = "Moe" in arch or "moe" in arch.lower()

                if is_moe:
                    with init_empty_weights():
                        causal_model = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True, torch_dtype=dtype)
                else:
                    causal_model = AutoModelForCausalLM.from_pretrained(
                        resolved_path,
                        config=hf_config,
                        trust_remote_code=True,
                        torch_dtype=dtype,
                    ).to(loader.DEVICE)

                adapter = create_adapter(causal_model, loader, hf_config)
                engine = TurboEngine(adapter)

                with self.lock:
                    self.loader = loader
                    self.adapter = adapter
                    self.engine = engine
                    self.status = "Loaded"
                print(f"Model successfully loaded: {model_id_or_path}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                with self.lock:
                    self.status = "Error"
                    self.error_message = str(e)
                print(f"Failed to load model {model_id_or_path}: {e}")

        # Start loading in background thread to not block fastapi
        threading.Thread(target=_async_load).start()

    def unload_model(self):
        with self.lock:
            self.unload_model_internal()
            self.status = "Unloaded"
            self.active_model_id = None
            self.model_path = None

    def unload_model_internal(self):
        if self.loader:
            try:
                self.loader.close()
            except Exception as e:
                print(f"Error closing loader: {e}")
        self.loader = None
        self.adapter = None
        self.engine = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_stream(self, prompt, max_new_tokens=None, temperature=None, top_p=None, chat=False, system_prompt=None, thinking="off"):
        if not self.engine:
            raise ValueError("No model loaded.")
        
        token_queue = queue.Queue()
        
        m_new_tokens = max_new_tokens if max_new_tokens is not None else self.config["runtime"]["max_new_tokens"]
        temp = temperature if temperature is not None else self.config["runtime"]["temperature"]
        tp = top_p if top_p is not None else self.config["runtime"]["top_p"]

        # Run config with server_mode=True to prevent loader from closing
        cfg = dict(self.config)
        cfg["server_mode"] = True

        collector = ServerMetricsCollector(self, token_queue)
        collector.num_layers = self.adapter.num_layers
        
        def _run_gen():
            try:
                with self.lock:
                    self.status = "Generating"
                
                self.engine.generate(
                    prompt=prompt,
                    max_new_tokens=m_new_tokens,
                    temperature=temp,
                    top_p=tp,
                    config=cfg,
                    chat=chat,
                    system_prompt=system_prompt,
                    collector=collector,
                    thinking=thinking
                )
                
                token_queue.put({"type": "done"})
            except Exception as e:
                import traceback
                traceback.print_exc()
                token_queue.put({"type": "error", "error": str(e)})
            finally:
                with self.lock:
                    if self.status == "Generating":
                        self.status = "Loaded"

        threading.Thread(target=_run_gen).start()
        
        while True:
            item = token_queue.get()
            yield item
            if item["type"] in ("done", "error"):
                break

    def broadcast_stats_update(self, stats_payload):
        # We also send this in the ws stats
        pass

    def broadcast_pipeline_update(self, payload):
        import json
        closed_sockets = []
        for ws in list(self.pipeline_websockets):
            try:
                # We need to run this synchronously or in the loop, but since we are in threads,
                # we'll send it asynchronously by calling write or scheduling it in the event loop.
                # In FastAPI we will handle this via queue or directly.
                pass
            except Exception:
                closed_sockets.append(ws)

    def _broadcast_system_stats_loop(self):
        import json
        while True:
            time.sleep(1.0)
            if not self.stats_websockets:
                continue

            # Fetch metrics
            ram = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            
            vram_allocated = 0.0
            vram_total = 0.0
            vram_free = 0.0
            if torch.cuda.is_available():
                vram_allocated = torch.cuda.memory_allocated() / (1024**2) # MB
                # Try getting total memory
                try:
                    free_mem, total_mem = torch.cuda.mem_get_info()
                    vram_total = total_mem / (1024**2)
                    vram_free = free_mem / (1024**2)
                except:
                    pass
            
            # Hit rates
            gpu_hits = 0
            ram_hits = 0
            ssd_hits = 0
            cache_limit = 0
            cache_count = 0
            if self.loader:
                gpu_hits = self.loader.gpu_hits
                ram_hits = self.loader.ram_hits
                ssd_hits = self.loader.ssd_hits
                cache_limit = self.loader.cache_limit
                cache_count = len(self.loader.expert_cache)

            stats = {
                "type": "stats",
                "status": self.status,
                "active_model": self.active_model_id,
                "ram_usage_percent": ram.percent,
                "ram_used_mb": ram.used / (1024**2),
                "ram_total_mb": ram.total / (1024**2),
                "cpu_percent": cpu,
                "vram_allocated_mb": vram_allocated,
                "vram_total_mb": vram_total,
                "vram_free_mb": vram_free,
                "gpu_hits": gpu_hits,
                "ram_hits": ram_hits,
                "ssd_hits": ssd_hits,
                "cache_limit": cache_limit,
                "cache_count": cache_count,
                "last_metrics": self.last_metrics
            }

            # We'll let the app.py broadcast this. We just update it in manager.
            self.last_system_stats = stats

# Singleton instance
engine_manager = EngineManager()
