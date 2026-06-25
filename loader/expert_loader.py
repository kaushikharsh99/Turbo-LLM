import json
import os
import torch
import time
import threading
from safetensors import safe_open
from cache.ram_cache import RAMCache

class ExpertLoader:
    DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    def __init__(self, snapshot_path, config=None):
        self.snapshot_path = snapshot_path
        self.config = config
        
        # Memory / VRAM config
        self.max_vram_gb = 5.8
        if config and "memory" in config and "max_vram_mb" in config:
            self.max_vram_gb = config["memory"]["max_vram_mb"] / 1024.0
            
        # Dtype config
        self.dtype = torch.float16
        if config and "execution" in config and "dtype" in config:
            dtype_str = config["execution"]["dtype"]
            if dtype_str in ("bf16", "bfloat16"):
                self.dtype = torch.bfloat16
            elif dtype_str in ("fp32", "float32"):
                self.dtype = torch.float32

        index_path = os.path.join(snapshot_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                self.index = json.load(f)
            self.weight_map = self.index["weight_map"]
        else:
            single_path = os.path.join(snapshot_path, "model.safetensors")
            if os.path.exists(single_path):
                from safetensors import safe_open
                with safe_open(single_path, framework="pt", device="cpu") as f:
                    keys = f.keys()
                self.weight_map = {k: "model.safetensors" for k in keys}
            else:
                import glob
                safetensors_files = glob.glob(os.path.join(snapshot_path, "*.safetensors"))
                if safetensors_files:
                    if len(safetensors_files) == 1:
                        single_path = safetensors_files[0]
                        filename = os.path.basename(single_path)
                        from safetensors import safe_open
                        with safe_open(single_path, framework="pt", device="cpu") as f:
                            keys = f.keys()
                        self.weight_map = {k: filename for k in keys}
                    else:
                        # Construct a weight map from all safetensors files
                        self.weight_map = {}
                        for sf in safetensors_files:
                            filename = os.path.basename(sf)
                            from safetensors import safe_open
                            with safe_open(sf, framework="pt", device="cpu") as f:
                                for k in f.keys():
                                    self.weight_map[k] = filename
                else:
                    self.weight_map = {}
        self.files = {} # Cache of safe_open handles
        self.expert_cache = {}
        self.expert_metadata = {}
        
        # Expert limit config
        self.cache_limit = 128  # Default limit
        if config and "cache" in config and "expert_limit" in config:
            limit = config["cache"]["expert_limit"]
            if limit != "auto":
                self.cache_limit = int(limit)
        
        # Thread lock for prefetching and cache management
        self.lock = threading.Lock()
        
        # Lazy allocation variables for static slots
        self.static_expert_gate = None
        self.static_expert_up = None
        self.static_expert_down = None
        self.num_slots = 0
        self.free_slots = []
        self.pinned_slots = set()
        
        # Instrumentation metrics
        self.load_ms_accum = 0.0
        self.dequant_ms_accum = 0.0
        self.evict_ms_accum = 0.0

        # RAM Cache and Hits Tracking
        self.ram_cache = RAMCache(config=config)
        self.gpu_hits = 0
        self.ram_hits = 0
        self.ssd_hits = 0
        self.load_counter = 0
        self.ssd_load_counter = 0

    def _get_tensor(self, weight_name):
        filename = self.weight_map[weight_name]
        if filename not in self.files:
            file_path = os.path.join(self.snapshot_path, filename)
            self.files[filename] = safe_open(file_path, framework="pt", device="cpu")
        
        # Qwen3 MoE might have stacked weights even in safetensors or might be individual.
        # Based on index.json, they are named model.layers.0.mlp.experts.0.gate_proj.weight
        return self.files[filename].get_tensor(weight_name)

    def load_weight(self, weight_name, device=DEVICE, dtype=None):
        if dtype is None:
            dtype = self.dtype
        tensor = self._get_tensor(weight_name)
        scale_name = f"{weight_name}_scale_inv"
        if scale_name in self.weight_map:
            # Direct transfer to GPU, then cast to dtype on CUDA (16x faster)
            if device == "mps" and tensor.dtype in (
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            ):
    # Convert unsupported FP8 weights before moving to MPS
                w = tensor.to(dtype=torch.float16).to(device=device)
            elif device == "cpu" and tensor.dtype in (
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            ):
                w = tensor.to(dtype=torch.float16)
            else:
                w = tensor.to(device=device).to(dtype=dtype)
            scale = self._get_tensor(scale_name).to(device=device).to(dtype=dtype)
            # Dequantize using reshape and broadcast (25x faster, 0 allocation copy)
            M, N = w.shape
            return (w.view(M // 128, 128, N // 128, 128) * scale.view(M // 128, 1, N // 128, 1)).view(M, N)
        else:
            return tensor.to(device=device).to(dtype=dtype)

    def load_weight_raw(self, weight_name, device=DEVICE):
        tensor = self._get_tensor(weight_name)
        scale_name = f"{weight_name}_scale_inv"
        if scale_name in self.weight_map:
            w_fp8 = self._prepare_fp8_tensor(tensor, device)
            scale = self._prepare_fp8_tensor(self._get_tensor(scale_name), device)
            return w_fp8, scale
        else:
            w_fp8 = self._prepare_fp8_tensor(tensor, device)
            return w_fp8, None

    def dequantize_weight(self, w_fp8, scale, dtype=None):
        if dtype is None:
            dtype = self.dtype

        # MPS cannot operate directly on FP8 tensors.
        # Convert on CPU first, then move to the target device.
        if self.DEVICE == "mps":
            w = w_fp8.cpu().to(dtype=dtype)

            if scale is None:
                return w.to("mps")

            scale_d = scale.cpu().to(dtype=dtype)

            M, N = w.shape
            w = (
                w.view(M // 128, 128, N // 128, 128)
                * scale_d.view(M // 128, 1, N // 128, 1)
            ).view(M, N)

            return w.to("mps")

        # Original path for CUDA / CPU
        if scale is None:
            return w_fp8.to(dtype=dtype)

        w = w_fp8.to(dtype=dtype)
        scale_d = scale.to(dtype=dtype)

        M, N = w.shape
        return (
            w.view(M // 128, 128, N // 128, 128)
            * scale_d.view(M // 128, 1, N // 128, 1)
        ).view(M, N)
    def load_weight_split(self, weight_name, device=DEVICE, dtype=None):
        """
        Loads the weight and returns the dequantized weight along with separated
        loading/copy time and dequantization time.
        """
        if dtype is None:
            dtype = self.dtype
        t_copy_start = time.time()
        tensor = self._get_tensor(weight_name)
        w_fp8 = self._prepare_fp8_tensor(tensor, device)
        
        scale_name = f"{weight_name}_scale_inv"
        if scale_name in self.weight_map:
            scale_fp8 = self._prepare_fp8_tensor(self._get_tensor(scale_name), device)
            copy_ms = (time.time() - t_copy_start) * 1000.0
            
            t_dequant_start = time.time()
            w = w_fp8.to(dtype=dtype)
            scale = scale_fp8.to(dtype=dtype)
            M, N = w.shape
            w_dequant = (w.view(M // 128, 128, N // 128, 128) * scale.view(M // 128, 1, N // 128, 1)).view(M, N)
            dequant_ms = (time.time() - t_dequant_start) * 1000.0
            
            return w_dequant.to(device=device), copy_ms, dequant_ms
        else:
            copy_ms = (time.time() - t_copy_start) * 1000.0
            t_dequant_start = time.time()
            w_dequant = w_fp8.to(dtype=dtype)
            dequant_ms = (time.time() - t_dequant_start) * 1000.0
            return w_dequant.to(device=device), copy_ms, dequant_ms

    def clear_pinned_slots(self):
        with self.lock:
            self.pinned_slots.clear()

    def adjust_cache_limit(self, max_vram_gb=None, is_decode=True):
        """
        Dynamically adjusts cache_limit based on current PyTorch allocated VRAM
        and actual physical GPU free VRAM to prevent OOM and respect the target budget.
        """
        if not torch.cuda.is_available():
            return
        
        if self.config and "cache" in self.config and "expert_limit" in self.config:
            if self.config["cache"]["expert_limit"] != "auto":
                self.cache_limit = int(self.config["cache"]["expert_limit"])
                return

        if max_vram_gb is None:
            max_vram_gb = self.max_vram_gb
        
        max_vram_bytes = max_vram_gb * 1024**3
        current_allocated = torch.cuda.memory_allocated()
        
        # Calculate size of all experts currently in cache (FP16 static format)
        cache_vram = 0
        if self.static_expert_gate is not None:
            cache_vram = (
                self.static_expert_gate.element_size() * self.static_expert_gate.nelement() +
                self.static_expert_up.element_size() * self.static_expert_up.nelement() +
                self.static_expert_down.element_size() * self.static_expert_down.nelement()
            )
                
        non_cache_allocated = current_allocated - cache_vram
        
        # Safety margin: 0.35 GB for prefill, 0.22 GB for decode
        safety_margin = (0.22 if is_decode else 0.35) * 1024**3
        available_by_budget = max_vram_bytes - non_cache_allocated - safety_margin
        
        # Budget based on physical GPU free memory
        free_mem, total_mem = torch.cuda.mem_get_info()
        available_by_physical = free_mem - safety_margin
        
        # Take the minimum of budget available and physical available
        available_bytes = min(available_by_budget, available_by_physical)
        
        # Estimate average size of 1 expert from actual weight shapes if available
        if hasattr(self, '_expert_gate_shape'):
            gate_els = 1
            for d in self._expert_gate_shape:
                gate_els *= d
            down_els = 1
            for d in self._expert_down_shape:
                down_els *= d
            # gate + up (same shape) + down, each in self.dtype
            elem_size = 2  # FP16/BF16 = 2 bytes
            avg_expert_size = (gate_els + gate_els + down_els) * elem_size
        else:
            avg_expert_size = 9.0 * 1024**2  # fallback ~9 MB for Qwen3
        dynamic_limit = int(available_bytes // avg_expert_size)
        
        # Keep limit bounds between 16 and 450
        self.cache_limit = max(16, min(450, dynamic_limit))
        
        print(f"[DEBUG Cache Limit] current_allocated: {current_allocated/1024**2:.2f} MB, "
              f"non_cache_allocated: {non_cache_allocated/1024**2:.2f} MB, "
              f"available_by_budget: {available_by_budget/1024**2:.2f} MB, "
              f"free_mem (physical): {free_mem/1024**2:.2f} MB, "
              f"available_by_physical: {available_by_physical/1024**2:.2f} MB, "
              f"dynamic_limit: {dynamic_limit}, chosen cap: {self.cache_limit}")

    def _prepare_fp8_tensor(self, tensor, device):
        if tensor is None:
            return None
        if device == "mps":
            return tensor
        return tensor.to(device=device)

    def load_expert_dynamic(self, layer_id, expert_id):
        """
        Loads and dequantizes expert weights dynamically on-the-fly.
        Used for prefill phase to avoid allocating the static cache pool.
        """
        key = (layer_id, expert_id)
        with self.lock:
            cached_expert = self.ram_cache.get(key)
            
        trigger_gc = False
        if cached_expert is not None:
            with self.lock:
                self.ram_hits += 1
            gate_fp8_cpu, gate_scale_cpu, up_fp8_cpu, up_scale_cpu, down_fp8_cpu, down_scale_cpu = cached_expert
            # Copy to GPU
            gate_fp8 = self._prepare_fp8_tensor(gate_fp8_cpu, self.DEVICE)
            gate_scale = self._prepare_fp8_tensor(gate_scale_cpu, self.DEVICE)
            up_fp8 = self._prepare_fp8_tensor(up_fp8_cpu, self.DEVICE)
            up_scale = self._prepare_fp8_tensor(up_scale_cpu, self.DEVICE)
            down_fp8 = self._prepare_fp8_tensor(down_fp8_cpu, self.DEVICE)
            down_scale = self._prepare_fp8_tensor(down_scale_cpu, self.DEVICE)
        else:
            with self.lock:
                self.ssd_hits += 1
                self.ssd_load_counter += 1
                trigger_gc = (self.ssd_load_counter % 20 == 0)
            prefix = f"model.layers.{layer_id}.mlp.experts.{expert_id}"
            
            gate_fp8_cpu = self._get_tensor(f"{prefix}.gate_proj.weight")
            gate_scale_name = f"{prefix}.gate_proj.weight_scale_inv"
            gate_scale_cpu = self._get_tensor(gate_scale_name) if gate_scale_name in self.weight_map else None

            up_fp8_cpu = self._get_tensor(f"{prefix}.up_proj.weight")
            up_scale_name = f"{prefix}.up_proj.weight_scale_inv"
            up_scale_cpu = self._get_tensor(up_scale_name) if up_scale_name in self.weight_map else None

            down_fp8_cpu = self._get_tensor(f"{prefix}.down_proj.weight")
            down_scale_name = f"{prefix}.down_proj.weight_scale_inv"
            down_scale_cpu = self._get_tensor(down_scale_name) if down_scale_name in self.weight_map else None

            cached_expert = (
                gate_fp8_cpu, gate_scale_cpu,
                up_fp8_cpu, up_scale_cpu,
                down_fp8_cpu, down_scale_cpu
            )
            with self.lock:
                self.ram_cache.put(key, cached_expert)
                
            # Copy to GPU
            gate_fp8 = self._prepare_fp8_tensor(gate_fp8_cpu, self.DEVICE)
            gate_scale = self._prepare_fp8_tensor(gate_scale_cpu, self.DEVICE)
            up_fp8 = self._prepare_fp8_tensor(up_fp8_cpu, self.DEVICE)
            up_scale = self._prepare_fp8_tensor(up_scale_cpu, self.DEVICE)
            down_fp8 = self._prepare_fp8_tensor(down_fp8_cpu, self.DEVICE)
            down_scale = self._prepare_fp8_tensor(down_scale_cpu, self.DEVICE)

        gate_proj = self.dequantize_weight(gate_fp8, gate_scale)
        up_proj = self.dequantize_weight(up_fp8, up_scale)
        down_proj = self.dequantize_weight(down_fp8, down_scale)
        
        if trigger_gc:
            import gc
            gc.collect(1)
            
        return gate_proj, up_proj, down_proj

    def load_expert(self, layer_id, expert_id, is_decode=True):
        """
        Loads, dequantizes, and caches the expert weights in static FP16 GPU slot buffers.
        On cache hits, returns slices of the FP16 slot buffers directly.
        """
        key = (layer_id, expert_id)
        
        with self.lock:
            # 1. Lazy allocation of static weight buffers
            if self.static_expert_gate is None:
                if self.DEVICE == "cuda":
                    torch.cuda.empty_cache() # Clear preloading temp memory first
                elif self.DEVICE == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
                
                # Probe actual expert weight shapes from the first available expert
                probe_gate = self._get_tensor("model.layers.0.mlp.experts.0.gate_proj.weight")
                probe_down = self._get_tensor("model.layers.0.mlp.experts.0.down_proj.weight")
                self._expert_gate_shape = probe_gate.shape  # e.g. (768, 2048) or (1024, 2048)
                self._expert_down_shape = probe_down.shape  # e.g. (2048, 768) or (2048, 1024)
                
                self.adjust_cache_limit(is_decode=is_decode)
                self.num_slots = self.cache_limit
                
                device_type = "GPU" if self.DEVICE in ("cuda", "mps") else "CPU"
                print(f"Allocating static {device_type} cache with {self.num_slots} expert {self.dtype} slots...")
                print(f"  Expert gate/up shape: {list(self._expert_gate_shape)}, down shape: {list(self._expert_down_shape)}")
                
                # Pre-allocate weights with actual shapes
                self.static_expert_gate = torch.zeros(self.num_slots, *self._expert_gate_shape, dtype=self.dtype, device=self.DEVICE)
                self.static_expert_up = torch.zeros(self.num_slots, *self._expert_gate_shape, dtype=self.dtype, device=self.DEVICE)
                self.static_expert_down = torch.zeros(self.num_slots, *self._expert_down_shape, dtype=self.dtype, device=self.DEVICE)
                
                self.free_slots = list(range(self.num_slots))
                self.pinned_slots = set()
                
            # 2. Check if already cached (Cache Hit)
            if key in self.expert_cache:
                slot_idx = self.expert_cache[key]
                self.expert_metadata[slot_idx]['hits'] += 1
                self.pinned_slots.add(slot_idx)
                self.gpu_hits += 1
                return (
                    self.static_expert_gate[slot_idx],
                    self.static_expert_up[slot_idx],
                    self.static_expert_down[slot_idx]
                )
                
            # 3. Cache miss: obtain a slot
            t_evict_start = time.time()
            if self.free_slots:
                slot_idx = self.free_slots.pop()
            else:
                # Evict the expert with the lowest score (excluding pinned slots)
                min_score = float('inf')
                slot_to_evict = None
                for s_idx, meta in self.expert_metadata.items():
                    if s_idx in self.pinned_slots:
                        continue
                    cost = max(1, meta['vram_cost'])
                    score = meta['hits'] * meta['load_ms'] / cost
                    if score < min_score:
                        min_score = score
                        slot_to_evict = s_idx
                        
                if slot_to_evict is not None:
                    evicted_key = self.expert_metadata[slot_to_evict]['key']
                    self.expert_cache.pop(evicted_key)
                    self.expert_metadata.pop(slot_to_evict)
                    slot_idx = slot_to_evict
                else:
                    slot_idx = 0
            
            self.pinned_slots.add(slot_idx)
            evict_ms = (time.time() - t_evict_start) * 1000.0
            self.evict_ms_accum += evict_ms
            
        # 4. Load raw weights from CPU (RAM Cache or SSD) to GPU (split timing)
        prefix = f"model.layers.{layer_id}.mlp.experts.{expert_id}"
        
        t_copy_start = time.time()
        with self.lock:
            cached_expert = self.ram_cache.get(key)
            
        trigger_gc = False
        if cached_expert is not None:
            with self.lock:
                self.ram_hits += 1
            gate_fp8_cpu, gate_scale_cpu, up_fp8_cpu, up_scale_cpu, down_fp8_cpu, down_scale_cpu = cached_expert
            # Copy to GPU
            gate_fp8 = self._prepare_fp8_tensor(gate_fp8_cpu, self.DEVICE)
            gate_scale = self._prepare_fp8_tensor(gate_scale_cpu, self.DEVICE)
            up_fp8 = self._prepare_fp8_tensor(up_fp8_cpu, self.DEVICE)
            up_scale = self._prepare_fp8_tensor(up_scale_cpu, self.DEVICE)
            down_fp8 = self._prepare_fp8_tensor(down_fp8_cpu, self.DEVICE)
            down_scale = self._prepare_fp8_tensor(down_scale_cpu, self.DEVICE)
        else:
            with self.lock:
                self.ssd_hits += 1
                self.ssd_load_counter += 1
                trigger_gc = (self.ssd_load_counter % 20 == 0)
            gate_fp8_cpu = self._get_tensor(f"{prefix}.gate_proj.weight")
            gate_scale_name = f"{prefix}.gate_proj.weight_scale_inv"
            gate_scale_cpu = self._get_tensor(gate_scale_name) if gate_scale_name in self.weight_map else None

            up_fp8_cpu = self._get_tensor(f"{prefix}.up_proj.weight")
            up_scale_name = f"{prefix}.up_proj.weight_scale_inv"
            up_scale_cpu = self._get_tensor(up_scale_name) if up_scale_name in self.weight_map else None

            down_fp8_cpu = self._get_tensor(f"{prefix}.down_proj.weight")
            down_scale_name = f"{prefix}.down_proj.weight_scale_inv"
            down_scale_cpu = self._get_tensor(down_scale_name) if down_scale_name in self.weight_map else None

            cached_expert = (
                gate_fp8_cpu, gate_scale_cpu,
                up_fp8_cpu, up_scale_cpu,
                down_fp8_cpu, down_scale_cpu
            )
            with self.lock:
                self.ram_cache.put(key, cached_expert)
                
            # Copy to GPU
            gate_fp8 = self._prepare_fp8_tensor(gate_fp8_cpu, self.DEVICE)
            gate_scale = self._prepare_fp8_tensor(gate_scale_cpu, self.DEVICE)
            up_fp8 = self._prepare_fp8_tensor(up_fp8_cpu, self.DEVICE)
            up_scale = self._prepare_fp8_tensor(up_scale_cpu, self.DEVICE)
            down_fp8 = self._prepare_fp8_tensor(down_fp8_cpu, self.DEVICE)
            down_scale = self._prepare_fp8_tensor(down_scale_cpu, self.DEVICE)

        copy_ms = (time.time() - t_copy_start) * 1000.0
        self.load_ms_accum += copy_ms
        
        # 5. Dequantize raw weights to FP16
        t_dequant_start = time.time()
        gate_proj = self.dequantize_weight(gate_fp8, gate_scale)
        up_proj = self.dequantize_weight(up_fp8, up_scale)
        down_proj = self.dequantize_weight(down_fp8, down_scale)
        dequant_ms = (time.time() - t_dequant_start) * 1000.0
        self.dequant_ms_accum += dequant_ms
        
        # 6. Copy dequantized weights to pre-allocated FP16 slots inside lock
        t_copy_static_start = time.time()
        with self.lock:
            # Check one more time in case another thread loaded concurrently
            if key in self.expert_cache:
                self.free_slots.append(slot_idx)
                slot_idx = self.expert_cache[key]
                self.expert_metadata[slot_idx]['hits'] += 1
                copy_to_static_ms = (time.time() - t_copy_static_start) * 1000.0
                self.load_ms_accum += copy_to_static_ms
                return (
                    self.static_expert_gate[slot_idx],
                    self.static_expert_up[slot_idx],
                    self.static_expert_down[slot_idx]
                )
                
            self.static_expert_gate[slot_idx].copy_(gate_proj)
            self.static_expert_up[slot_idx].copy_(up_proj)
            self.static_expert_down[slot_idx].copy_(down_proj)
            
            vram_cost = (
                gate_proj.nelement() * gate_proj.element_size()
            ) * 3  # Estimate same for gate, up, down
            
            self.expert_cache[key] = slot_idx
            self.expert_metadata[slot_idx] = {
                'hits': 1,
                'load_ms': copy_ms + dequant_ms,
                'vram_cost': vram_cost,
                'key': key
            }
            
        copy_to_static_ms = (time.time() - t_copy_static_start) * 1000.0
        self.load_ms_accum += copy_to_static_ms

        if trigger_gc:
            import gc
            gc.collect(1)
        
        return (
            self.static_expert_gate[slot_idx],
            self.static_expert_up[slot_idx],
            self.static_expert_down[slot_idx]
        )


    def load_expert_raw(self, layer_id, expert_id, is_decode=True):
        """
        Helper method to retrieve raw FP8 weight and scale tensors directly from disk/VRAM.
        """
        prefix = f"model.layers.{layer_id}.mlp.experts.{expert_id}"
        gate_fp8, gate_scale = self.load_weight_raw(f"{prefix}.gate_proj.weight")
        up_fp8, up_scale = self.load_weight_raw(f"{prefix}.up_proj.weight")
        down_fp8, down_scale = self.load_weight_raw(f"{prefix}.down_proj.weight")
        return gate_fp8, gate_scale, up_fp8, up_scale, down_fp8, down_scale

    def close(self):
        with self.lock:
            for f in self.files.values():
                del f
            self.files = {}
            self.expert_cache.clear()
            self.expert_metadata.clear()
            self.static_expert_gate = None
            self.static_expert_up = None
            self.static_expert_down = None
            self.free_slots.clear()