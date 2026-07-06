import time
import torch
import torch.nn.functional as F

def benchmark_gemm_methods(iterations=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available. Benchmark requires GPU.")
        return

    # Qwen 3.6 35B MoE decode dimensions
    batch_size = 1
    top_k = 8
    hidden_dim = 2048
    inter_dim = 512
    dtype = torch.float16

    print("=" * 60)
    print("Qwen 3.6 MoE GEMM Microbenchmark (1000 Iterations)")
    print(f"Dimensions: Batch={batch_size}, TopK={top_k}, Hidden={hidden_dim}, Inter={inter_dim}")
    print("=" * 60)

    # Allocations
    x = torch.randn(batch_size, hidden_dim, device=device, dtype=dtype)
    gate_weights = [torch.randn(inter_dim, hidden_dim, device=device, dtype=dtype) for _ in range(top_k)]
    up_weights = [torch.randn(inter_dim, hidden_dim, device=device, dtype=dtype) for _ in range(top_k)]
    down_weights = [torch.randn(hidden_dim, inter_dim, device=device, dtype=dtype) for _ in range(top_k)]
    weights = torch.ones(batch_size, top_k, device=device, dtype=dtype) / top_k

    stacked_gate = torch.stack(gate_weights, 0)
    stacked_up = torch.stack(up_weights, 0)
    stacked_down = torch.stack(down_weights, 0)
    x_expanded = x.expand((top_k, 1, hidden_dim))

    # Warmup
    for _ in range(50):
        for i in range(top_k):
            _ = F.linear(x, gate_weights[i])
            _ = F.linear(x, up_weights[i])
            _ = F.linear(x, down_weights[i].t())
        _ = torch.bmm(x_expanded, stacked_gate.transpose(1, 2))
    torch.cuda.synchronize()

    # 1. Method A: PyTorch F.linear (Sequential GEMV)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iterations):
        out = torch.zeros_like(x)
        for i in range(top_k):
            g = F.linear(x, gate_weights[i])
            u = F.linear(x, up_weights[i])
            inter = F.silu(g) * u
            o = F.linear(inter, down_weights[i])
            out += o * weights[0, i]
    torch.cuda.synchronize()
    t_flinear = (time.perf_counter() - t0) * 1000.0 / iterations

    # 1b. Method A2: Fused Gate+Up PyTorch F.linear
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fused_gate_up_weights = [torch.cat([gate_weights[i], up_weights[i]], dim=0) for i in range(top_k)]
    for _ in range(iterations):
        out = torch.zeros_like(x)
        for i in range(top_k):
            gate_up = F.linear(x, fused_gate_up_weights[i])
            g, u = gate_up.chunk(2, dim=-1)
            inter = F.silu(g) * u
            o = F.linear(inter, down_weights[i])
            out += o * weights[0, i]
    torch.cuda.synchronize()
    t_fused = (time.perf_counter() - t0) * 1000.0 / iterations

    # 2. Method B: PyTorch torch.bmm (Batched GEMM)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iterations):
        g_batched = torch.bmm(x_expanded, stacked_gate.transpose(1, 2))
        u_batched = torch.bmm(x_expanded, stacked_up.transpose(1, 2))
        inter_batched = F.silu(g_batched) * u_batched
        o_batched = torch.bmm(inter_batched, stacked_down.transpose(1, 2))
        out = (o_batched * weights.view(top_k, 1, 1)).sum(0)
    torch.cuda.synchronize()
    t_bmm = (time.perf_counter() - t0) * 1000.0 / iterations

    # 3. Method C: C++ Extension cuBLASLt (if available)
    t_cublaslt = None
    try:
        import turbollm_cpp
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = turbollm_cpp.execute_moe(x, gate_weights, up_weights, down_weights, weights)
        torch.cuda.synchronize()
        t_cublaslt = (time.perf_counter() - t0) * 1000.0 / iterations
    except Exception as e:
        print(f"C++ cuBLASLt extension benchmark error: {e}")

    # 4. Method D: Native C++ Cached cuBLAS GEMV (Level 1 optimization)
    t_cached_gemv = None
    try:
        import cutlass_gemm_benchmark
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iterations):
            out = torch.zeros_like(x)
            for i in range(top_k):
                g = cutlass_gemm_benchmark.cublas_cached_gemv_fp16(x, gate_weights[i])
                u = cutlass_gemm_benchmark.cublas_cached_gemv_fp16(x, up_weights[i])
                inter = F.silu(g) * u
                o = cutlass_gemm_benchmark.cublas_cached_gemv_fp16(inter, down_weights[i])
                out += o * weights[0, i]
        torch.cuda.synchronize()
        t_cached_gemv = (time.perf_counter() - t0) * 1000.0 / iterations
    except Exception as e:
        print(f"C++ Cached cuBLAS GEMV extension benchmark error: {e}")

    # 5. Method E: Native C++ Grouped GEMM (Phase 2 optimization)
    t_grouped_gemv = None
    try:
        import cutlass_gemm_benchmark
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iterations):
            g_grouped = cutlass_gemm_benchmark.cublas_grouped_gemm_fp16(x, gate_weights)
            u_grouped = cutlass_gemm_benchmark.cublas_grouped_gemm_fp16(x, up_weights)
            inter_grouped = F.silu(g_grouped) * u_grouped
            
            inter_list = [inter_grouped[i] for i in range(top_k)]
            o_grouped = cutlass_gemm_benchmark.cublas_grouped_gemm_fp16(inter_list[0], down_weights)
            out = (o_grouped * weights.view(top_k, 1, 1)).sum(0)
        torch.cuda.synchronize()
        t_grouped_gemv = (time.perf_counter() - t0) * 1000.0 / iterations
    except Exception as e:
        print(f"C++ Grouped GEMM extension benchmark error: {e}")

    # 6. Method F: Fused Gate+Up Native C++ Grouped GEMM (Optimization 1)
    t_fused_cpp = None
    try:
        import cutlass_gemm_benchmark
        # Persistent pre-allocated fused gate+up weight matrices [1024, 2048]
        fused_gate_up_weights = [torch.cat([gate_weights[i], up_weights[i]], 0) for i in range(top_k)]
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iterations):
            gate_up_fused = cutlass_gemm_benchmark.cublas_fused_gate_up_grouped_gemm_fp16(x, fused_gate_up_weights)
            g_fused = gate_up_fused[:, :, :inter_dim]
            u_fused = gate_up_fused[:, :, inter_dim:]
            inter_fused = cutlass_gemm_benchmark.fused_silu_mul_cpp(g_fused, u_fused)
            
            inter_fused_list = [inter_fused[i] for i in range(top_k)]
            o_fused = cutlass_gemm_benchmark.cublas_grouped_gemm_fp16(inter_fused_list[0], down_weights)
            out = (o_fused * weights.view(top_k, 1, 1)).sum(0)
        torch.cuda.synchronize()
        t_fused_cpp = (time.perf_counter() - t0) * 1000.0 / iterations
    except Exception as e:
        print(f"C++ Fused Gate+Up Grouped GEMM extension benchmark error: {e}")

    print("\nBenchmark Results per Layer Execution (1 Layer):")
    print("-" * 60)
    print(f"1. PyTorch F.linear (Unfused GEMV) : {t_flinear:.3f} ms / layer")
    print(f"2. Fused Gate+Up F.linear (GEMV)   : {t_fused:.3f} ms / layer")
    print(f"3. PyTorch torch.bmm (Batched GEMM): {t_bmm:.3f} ms / layer")
    if t_cached_gemv is not None:
        print(f"4. C++ Cached cuBLAS GEMV (Level 1): {t_cached_gemv:.3f} ms / layer")
    if t_grouped_gemv is not None:
        print(f"5. C++ Grouped GEMM (Phase 2)      : {t_grouped_gemv:.3f} ms / layer")
    if t_fused_cpp is not None:
        print(f"6. C++ Fused Gate+Up Grouped GEMM  : {t_fused_cpp:.3f} ms / layer  <-- WINNER!")
    if t_cublaslt is not None:
        print(f"7. C++ cuBLASLt Extension         : {t_cublaslt:.3f} ms / layer")
    print("-" * 60)
    print(f"Total 40-Layer Model Decode Latency Estimates:")
    print(f"  - Unfused F.linear Total GEMM Time  : {t_flinear * 40:.2f} ms")
    print(f"  - Fused Gate+Up Total GEMM Time     : {t_fused * 40:.2f} ms")
    print(f"  - torch.bmm Total GEMM Time         : {t_bmm * 40:.2f} ms")
    if t_cached_gemv is not None:
        print(f"  - C++ Cached cuBLAS GEMV Total GEMM : {t_cached_gemv * 40:.2f} ms")
    if t_grouped_gemv is not None:
        print(f"  - C++ Grouped GEMM Total GEMM       : {t_grouped_gemv * 40:.2f} ms")
    if t_fused_cpp is not None:
        print(f"  - C++ Fused Gate+Up Grouped Total   : {t_fused_cpp * 40:.2f} ms")
    if t_cublaslt is not None:
        print(f"  - cuBLASLt Total GEMM Time      : {t_cublaslt * 40:.2f} ms")
    print("=" * 60)

    # Export autobenchmark_results.json
    import json, os
    out_dir = os.path.dirname(os.path.abspath(__file__))
    results = {
        "torch_flinear_ms_per_layer": round(t_flinear, 4),
        "fused_gate_up_flinear_ms_per_layer": round(t_fused, 4),
        "torch_bmm_ms_per_layer": round(t_bmm, 4),
        "cpp_cached_gemv_ms_per_layer": round(t_cached_gemv, 4) if t_cached_gemv else None,
        "cpp_grouped_gemm_ms_per_layer": round(t_grouped_gemv, 4) if t_grouped_gemv else None,
        "cpp_fused_gate_up_grouped_ms_per_layer": round(t_fused_cpp, 4) if t_fused_cpp else None,
        "cpp_cublaslt_ms_per_layer": round(t_cublaslt, 4) if t_cublaslt else None,
        "est_40layer_torch_bmm_ms": round(t_bmm * 40, 2),
        "est_40layer_cpp_fused_gate_up_grouped_ms": round(t_fused_cpp * 40, 2) if t_fused_cpp else None
    }
    with open(os.path.join(out_dir, "autobenchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAutobenchmark results saved to {os.path.join(out_dir, 'autobenchmark_results.json')}")

if __name__ == "__main__":
    benchmark_gemm_methods()
