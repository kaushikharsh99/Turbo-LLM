import torch
import torch.nn.functional as F
import time
import sys
import os

# Make sure we can load the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import turbollm_cpp
except ImportError as e:
    print(f"Error: Could not import turbollm_cpp. Make sure to run 'pip install -e . --no-build-isolation' first. Details: {e}")
    sys.exit(1)

def moe_seq_py(hidden_states, gate_weights, up_weights, down_weights, top_k_weights):
    final_output = torch.zeros_like(hidden_states)
    top_k = top_k_weights.shape[1]
    for i in range(top_k):
        gate_proj = gate_weights[i]
        up_proj = up_weights[i]
        down_proj = down_weights[i]
        
        g = F.linear(hidden_states, gate_proj)
        u = F.linear(hidden_states, up_proj)
        inter = F.silu(g) * u
        o = F.linear(inter, down_proj)
        final_output += o * top_k_weights[0, i]
    return final_output

class MockLoader:
    def __init__(self, gate_weights, up_weights, down_weights):
        self.gate_weights = gate_weights
        self.up_weights = up_weights
        self.down_weights = down_weights

    def load_expert_raw(self, layer_id, expert_id):
        # Return dequantized weight as FP8 mockup, and None for scale.
        # This will execute C++ dequantize_fp8 correctly (falling back to simple cast).
        return (
            self.gate_weights[expert_id], None,
            self.up_weights[expert_id], None,
            self.down_weights[expert_id], None
        )

def run_verification_and_benchmark():
    # Setup test shapes and parameters matching MoE layers (e.g. Qwen / Mixtral layout)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Running verification on device: {device} (dtype: {dtype})")
    
    batch_size = 1
    hidden_size = 512
    intermediate_size = 256
    top_k = 8
    
    # Generate random inputs
    hidden_states = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    top_k_weights = torch.softmax(torch.randn(batch_size, top_k, dtype=dtype, device=device), dim=-1)
    
    # Experts' weights list
    gate_weights = [torch.randn(intermediate_size, hidden_size, dtype=dtype, device=device) for _ in range(top_k)]
    up_weights = [torch.randn(intermediate_size, hidden_size, dtype=dtype, device=device) for _ in range(top_k)]
    down_weights = [torch.randn(hidden_size, intermediate_size, dtype=dtype, device=device) for _ in range(top_k)]
    
    # 1. Run Python version
    output_py = moe_seq_py(hidden_states, gate_weights, up_weights, down_weights, top_k_weights)
    
    # 2. Run C++ version
    output_cpp = turbollm_cpp.execute_moe(hidden_states, gate_weights, up_weights, down_weights, top_k_weights)
    
    # 3. Run C++ cached version
    mock_loader = MockLoader(gate_weights, up_weights, down_weights)
    expert_ids = list(range(top_k))
    # Execute first time (cache miss -> loads and dequantizes)
    output_cpp_cached = turbollm_cpp.execute_moe_with_cache(0, expert_ids, hidden_states, top_k_weights, mock_loader)
    
    print("Python Output (first 5 elements):      ", output_py[0, :5])
    print("C++ Output (first 5 elements):          ", output_cpp[0, :5])
    print("C++ Cached Output (first 5 elements):   ", output_cpp_cached[0, :5])
    
    # Check numerical correctness
    tol = 1e-2 if dtype == torch.float16 else 1e-4
    all_close_standard = torch.allclose(output_py, output_cpp, atol=tol, rtol=tol)
    all_close_cached = torch.allclose(output_py, output_cpp_cached, atol=tol, rtol=tol)
    
    print(f"Standard C++ correctness check: {'PASSED' if all_close_standard else 'FAILED'}")
    print(f"Cached C++ correctness check:   {'PASSED' if all_close_cached else 'FAILED'}")
    
    if not (all_close_standard and all_close_cached):
        max_diff_std = (output_py - output_cpp).abs().max().item()
        max_diff_cached = (output_py - output_cpp_cached).abs().max().item()
        print(f"Max difference (Standard): {max_diff_std}, (Cached): {max_diff_cached}")
        
    # 4. Benchmark all three paths
    num_runs = 500
    
    # Warmup
    for _ in range(50):
        _ = moe_seq_py(hidden_states, gate_weights, up_weights, down_weights, top_k_weights)
        _ = turbollm_cpp.execute_moe(hidden_states, gate_weights, up_weights, down_weights, top_k_weights)
        _ = turbollm_cpp.execute_moe_with_cache(0, expert_ids, hidden_states, top_k_weights, mock_loader)
    if device == "cuda":
        torch.cuda.synchronize()
        
    # Python Benchmark
    t0 = time.time()
    for _ in range(num_runs):
        _ = moe_seq_py(hidden_states, gate_weights, up_weights, down_weights, top_k_weights)
    if device == "cuda":
        torch.cuda.synchronize()
    t_py = (time.time() - t0) * 1000.0 / num_runs
    
    # Standard C++ Benchmark
    t0 = time.time()
    for _ in range(num_runs):
        _ = turbollm_cpp.execute_moe(hidden_states, gate_weights, up_weights, down_weights, top_k_weights)
    if device == "cuda":
        torch.cuda.synchronize()
    t_cpp = (time.time() - t0) * 1000.0 / num_runs

    # Cached C++ Benchmark (representing a 100% cache hit rate path)
    t0 = time.time()
    for _ in range(num_runs):
        _ = turbollm_cpp.execute_moe_with_cache(0, expert_ids, hidden_states, top_k_weights, mock_loader)
    if device == "cuda":
        torch.cuda.synchronize()
    t_cpp_cached = (time.time() - t0) * 1000.0 / num_runs
    
    print(f"Benchmark results over {num_runs} runs:")
    print(f"  Python version:    {t_py:.4f} ms per step")
    print(f"  C++ version:       {t_cpp:.4f} ms per step")
    print(f"  C++ Cached version: {t_cpp_cached:.4f} ms per step")
    print(f"  Speedup (Standard C++): {(t_py / t_cpp - 1.0) * 100.0:.2f}%")
    print(f"  Speedup (C++ Cached):   {(t_py / t_cpp_cached - 1.0) * 100.0:.2f}%")

if __name__ == "__main__":
    run_verification_and_benchmark()
