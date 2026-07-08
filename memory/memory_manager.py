import torch
from pathlib import Path
from typing import Dict, List, Optional, Any

from model.model import Model
from model.tensor import Tensor
from memory.cpu_memory import CPUMemory
from memory.gpu_memory import GPUMemory


class MemoryManager:
    """Coordinates weight loading and transitions between SSD -> CPU RAM -> GPU VRAM."""

    def __init__(self, model: Model, max_cpu_bytes: int | float, max_gpu_bytes: int | float, device_manager=None):
        self.model = model
        self.max_cpu_bytes = max_cpu_bytes
        self.max_gpu_bytes = max_gpu_bytes

        if device_manager is None:
            from runtime.device import DeviceManager
            device_manager = DeviceManager()
        self.device_manager = device_manager

        self.cpu_memory = CPUMemory(max_cpu_bytes)
        self.gpu_memory = GPUMemory(max_gpu_bytes, device_manager=device_manager)

        # Mapping of tensor name -> Tensor (metadata wrapper object)
        self.tensor_registry: Dict[str, Tensor] = {}
        self._register_model_tensors(model)

    def _register_model_tensors(self, model: Model):
        """Recursively registers all Tensor wrappers from the Model instance."""
        # 1. Embedding
        if model.embedding:
            self.tensor_registry[model.embedding.name] = model.embedding

        # 2. Layers
        for layer in model.layers:
            # Norms
            if layer.attention_norm:
                self.tensor_registry[layer.attention_norm.name] = layer.attention_norm
            if layer.ffn_norm:
                self.tensor_registry[layer.ffn_norm.name] = layer.ffn_norm

            # Attention
            if layer.attention:
                for proj in [
                    layer.attention.q_proj,
                    layer.attention.k_proj,
                    layer.attention.v_proj,
                    layer.attention.o_proj,
                    layer.attention.q_scale,
                    layer.attention.k_scale,
                    layer.attention.v_scale,
                    layer.attention.o_scale,
                    layer.attention.q_norm,
                    layer.attention.k_norm,
                ]:
                    if proj:
                        self.tensor_registry[proj.name] = proj

            # Linear Attention
            if layer.linear_attn:
                for proj in [
                    layer.linear_attn.conv1d,
                    layer.linear_attn.dt_bias,
                    layer.linear_attn.A_log,
                    layer.linear_attn.norm,
                    layer.linear_attn.out_proj,
                    layer.linear_attn.in_proj_qkv,
                    layer.linear_attn.in_proj_z,
                    layer.linear_attn.in_proj_b,
                    layer.linear_attn.in_proj_a,
                    layer.linear_attn.out_proj_scale,
                    layer.linear_attn.in_proj_qkv_scale,
                    layer.linear_attn.in_proj_z_scale,
                ]:
                    if proj:
                        self.tensor_registry[proj.name] = proj

            # MoE
            if layer.moe:
                if layer.moe.router and layer.moe.router.gate:
                    self.tensor_registry[layer.moe.router.gate.name] = layer.moe.router.gate

                if layer.moe.shared_expert:
                    for proj in [
                        layer.moe.shared_expert.gate_proj,
                        layer.moe.shared_expert.up_proj,
                        layer.moe.shared_expert.down_proj,
                        layer.moe.shared_expert.gate_scale,
                        layer.moe.shared_expert.up_scale,
                        layer.moe.shared_expert.down_scale,
                        layer.moe.shared_expert.shared_gate,
                    ]:
                        if proj:
                            self.tensor_registry[proj.name] = proj

                for expert in layer.moe.experts:
                    for proj in [
                        expert.gate_proj,
                        expert.up_proj,
                        expert.down_proj,
                        expert.gate_scale,
                        expert.up_scale,
                        expert.down_scale,
                    ]:
                        if proj:
                            self.tensor_registry[proj.name] = proj

        # 3. Final norm
        if model.final_norm:
            self.tensor_registry[model.final_norm.name] = model.final_norm

        # 4. LM Head
        if model.lm_head:
            self.tensor_registry[model.lm_head.name] = model.lm_head

    def load_tensor(self, name: str) -> torch.Tensor:
        """Loads a tensor from disk to CPU memory cache. Handles CPU cache evictions."""
        if name not in self.tensor_registry:
            raise KeyError(f"Tensor '{name}' is not registered in this model.")

        tensor_obj = self.tensor_registry[name]

        # Return CPU cached tensor if already loaded
        if self.cpu_memory.exists(name):
            return self.cpu_memory.get(name)

        if not self.model.tensor_loader:
            raise RuntimeError("Model does not have a tensor_loader initialized.")

        # Load from disk
        cpu_data = self.model.tensor_loader.get_tensor(name)

        # Cache in CPU memory and handle evicted tensors
        evicted_names = self.cpu_memory.add(name, cpu_data, tensor_obj)
        for evicted in evicted_names:
            evicted_obj = self.tensor_registry[evicted]
            if not self.gpu_memory.exists(evicted):
                evicted_obj.loaded = False
                evicted_obj.device = "disk"
                evicted_obj.data = None

        # Update metadata for newly loaded tensor (only if not already resident on GPU)
        if not self.gpu_memory.exists(name):
            tensor_obj.device = "cpu"
            tensor_obj.loaded = True
            tensor_obj.data = cpu_data

        return cpu_data

    def move_to_gpu(self, name: str) -> torch.Tensor:
        """Moves a tensor from CPU to GPU memory cache. Handles GPU cache evictions."""
        if name not in self.tensor_registry:
            raise KeyError(f"Tensor '{name}' is not registered in this model.")

        tensor_obj = self.tensor_registry[name]

        # Return GPU cached tensor if already resident
        if self.gpu_memory.exists(name):
            return self.gpu_memory.get(name)

        # Load to CPU RAM first if not present
        cpu_data = self.load_tensor(name)

        # Copy data to GPU
        gpu_data = cpu_data.to(self.device_manager.device)

        # Cache in GPU memory and handle evicted tensors
        evicted_names = self.gpu_memory.add(name, gpu_data, tensor_obj)
        for evicted in evicted_names:
            evicted_obj = self.tensor_registry[evicted]
            if self.cpu_memory.exists(evicted):
                evicted_obj.device = "cpu"
                evicted_obj.data = self.cpu_memory.get(evicted)
            else:
                evicted_obj.device = "disk"
                evicted_obj.loaded = False
                evicted_obj.data = None

        # Update metadata of resident tensor
        tensor_obj.device = self.device_manager.device.type
        tensor_obj.loaded = True
        tensor_obj.data = gpu_data

        return gpu_data

    def free_gpu(self, name: str):
        """Explicitly unloads a tensor from GPU resident cache."""
        if name not in self.tensor_registry:
            return

        tensor_obj = self.tensor_registry[name]

        if self.gpu_memory.exists(name):
            self.gpu_memory.remove(name)

            # Fallback to CPU copy if still present in RAM
            if self.cpu_memory.exists(name):
                tensor_obj.device = "cpu"
                tensor_obj.data = self.cpu_memory.get(name)
            else:
                tensor_obj.device = "disk"
                tensor_obj.loaded = False
                tensor_obj.data = None

    def free_cpu(self, name: str):
        """Explicitly unloads a tensor from CPU cache."""
        if name not in self.tensor_registry:
            return

        tensor_obj = self.tensor_registry[name]

        if self.cpu_memory.exists(name):
            self.cpu_memory.remove(name)

            # Mark as completely unloaded if it is also not in GPU
            if not self.gpu_memory.exists(name):
                tensor_obj.device = "disk"
                tensor_obj.loaded = False
                tensor_obj.data = None

    def get_tensor(self, name: str) -> torch.Tensor:
        """Gets PyTorch tensor data from wherever it currently resides (GPU preferred, CPU fallback)."""
        if self.gpu_memory.exists(name):
            return self.gpu_memory.get(name)
        elif self.cpu_memory.exists(name):
            return self.cpu_memory.get(name)
        else:
            # Load to CPU by default if not loaded
            return self.load_tensor(name)

    def free_layer_gpu(self, layer_id: int):
        """Unloads all tensors belonging to a specific layer from GPU resident memory."""
        layer_prefix = f"layers.{layer_id}."
        for name in list(self.tensor_registry.keys()):
            if layer_prefix in name:
                self.free_gpu(name)
