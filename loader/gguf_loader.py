"""
GGUFLoader: High-performance GGUF Model Reader & Quantized Expert Loader
Reads GGUF tensors (Q4_K_M, Q8_0, FP16) and dequantizes or loads raw quantized bytes into VRAM.
"""
import gguf
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

class GGUFLoader:
    def __init__(self, gguf_path: str):
        self.gguf_path = gguf_path
        self.reader = gguf.GGUFReader(gguf_path, mode='r')
        self.tensor_info: Dict[str, gguf.ReaderTensor] = {}
        for tensor in self.reader.tensors:
            self.tensor_info[tensor.name] = tensor

    def list_tensors(self) -> List[str]:
        return list(self.tensor_info.keys())

    def get_tensor_handle(self, name: str) -> Optional[gguf.ReaderTensor]:
        return self.tensor_info.get(name)

    def load_tensor(self, name: str, device: str = "cpu", dtype: torch.dtype = torch.float16, expert_idx: Optional[int] = None) -> torch.Tensor:
        if name not in self.tensor_info:
            raise KeyError(f"Tensor {name} not found in GGUF file {self.gguf_path}")
            
        t_info = self.tensor_info[name]
        data = t_info.data
        
        # If this is a 3D packed expert tensor (e.g. 256 experts), slice 1 expert's bytes to save 99% RAM!
        if expert_idx is not None and len(t_info.shape) == 3 and t_info.shape[2] > 1:
            num_experts = t_info.shape[2]
            bytes_per_exp = len(data) // num_experts
            start_b = expert_idx * bytes_per_exp
            end_b = start_b + bytes_per_exp
            data_slice = data[start_b:end_b]
            dq_np = gguf.quants.dequantize(data_slice, t_info.tensor_type)
            tensor_pt = torch.from_numpy(dq_np).squeeze(0).to(dtype=dtype)
        else:
            dq_np = gguf.quants.dequantize(data, t_info.tensor_type)
            tensor_pt = torch.from_numpy(dq_np).to(dtype=dtype)
            
        return tensor_pt.to(device=device)

    def close(self):
        del self.reader
        self.tensor_info.clear()
