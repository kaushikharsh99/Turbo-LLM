#!/usr/bin/env python3
import time
import torch
import numpy as np
import random
from loader.gguf_loader import GGUFLoader

def run_stress_test(gguf_path, num_loads=100):
    print("=" * 60)
    print("Turbo-LLM GGUF Stress Test & Loading Latency")
    print("=" * 60)
    
    loader = GGUFLoader(gguf_path)
    tensor_names = loader.list_tensors()
    print(f"Loaded model layout with {len(tensor_names)} tensors.")
    
    # Pick a random subset of tensors to load
    sample_names = random.choices(tensor_names, k=num_loads)
    
    lookup_times = []
    load_times = []
    
    print(f"Starting stress test: loading {num_loads} random tensors...")
    
    for i, name in enumerate(sample_names):
        # 1. Measure lookup latency
        t0 = time.perf_counter()
        handle = loader.get_tensor_handle(name)
        t1 = time.perf_counter()
        lookup_times.append(t1 - t0)
        
        # 2. Measure read and dequantize latency
        t2 = time.perf_counter()
        tensor = loader.load_tensor(name, device="cpu", dtype=torch.float16)
        t3 = time.perf_counter()
        load_times.append(t3 - t2)
        
        if (i + 1) % (num_loads // 5 or 1) == 0 or i == num_loads - 1:
            print(f"  Progress: {i + 1}/{num_loads} loaded...")
            
    avg_lookup_ms = np.mean(lookup_times) * 1000
    avg_load_ms = np.mean(load_times) * 1000
    total_time_s = np.sum(lookup_times) + np.sum(load_times)
    
    print("\nStress Test Results:")
    print(f"  Total time spent      : {total_time_s:.3f} s")
    print(f"  Avg Lookup Latency    : {avg_lookup_ms:.4f} ms")
    print(f"  Avg Read/Dequant Time : {avg_load_ms:.2f} ms")
    print(f"  Avg Speed             : {num_loads / total_time_s:.2f} tensors/s")
    print("=" * 60)
    
    loader.close()

if __name__ == "__main__":
    gguf_path = "/home/harsh/.turbollm/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    run_stress_test(gguf_path, num_loads=100)
