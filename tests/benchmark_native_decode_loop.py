import torch
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks", "gemm"))

import cutlass_gemm_benchmark as cpp_gemm

def benchmark_decode_loop():
    device = "cuda"
    dtype = torch.float16
    num_layers = 40
    top_k = 8
    hidden_size = 2048
    inter_size = 512
    shared_size = 4096

    print("=" * 60)
    print(f"Native C++ Full Decode Loop Benchmark ({num_layers} Layers)")
    print("=" * 60)

    hidden_states = torch.randn(1, hidden_size, dtype=dtype, device=device)

    # Synthetic weight layers
    fused_gate_up_layers = []
    down_layers = []
    top_k_weights_layers = []
    shared_gate_layers = []
    shared_up_layers = []
    shared_down_layers = []
    shared_score_layers = []

    for l in range(num_layers):
        fused_gate_up = [torch.randn(inter_size * 2, hidden_size, dtype=dtype, device=device) for _ in range(top_k)]
        down = [torch.randn(hidden_size, inter_size, dtype=dtype, device=device) for _ in range(top_k)]
        top_k_w = torch.softmax(torch.randn(top_k, dtype=dtype, device=device), dim=-1)

        s_gate = torch.randn(shared_size, hidden_size, dtype=dtype, device=device)
        s_up = torch.randn(shared_size, hidden_size, dtype=dtype, device=device)
        s_down = torch.randn(hidden_size, shared_size, dtype=dtype, device=device)
        s_score = torch.randn(1, hidden_size, dtype=dtype, device=device)

        fused_gate_up_layers.append(fused_gate_up)
        down_layers.append(down)
        top_k_weights_layers.append(top_k_w)
        shared_gate_layers.append(s_gate)
        shared_up_layers.append(s_up)
        shared_down_layers.append(s_down)
        shared_score_layers.append(s_score)

    iterations = 200

    # 1. Python 40-Layer Loop (560 PyBind transitions / token)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iterations):
        curr = hidden_states
        for l in range(num_layers):
            g_u = cpp_gemm.cublas_fused_gate_up_grouped_gemm_fp16(curr, fused_gate_up_layers[l])
            g = g_u[:, :, :inter_size]
            u = g_u[:, :, inter_size:]
            inter = cpp_gemm.fused_silu_mul_cpp(g, u)
            o = cpp_gemm.cublas_grouped_gemm_fp16(inter[0], down_layers[l])
            moe_out = (o * top_k_weights_layers[l].view(top_k, 1, 1)).sum(0)

            sg = torch.matmul(curr, shared_gate_layers[l].t())
            su = torch.matmul(curr, shared_up_layers[l].t())
            sh = torch.nn.functional.silu(sg) * su
            sd = torch.matmul(sh, shared_down_layers[l].t())
            ssc = torch.sigmoid(torch.matmul(curr, shared_score_layers[l].t()))
            curr = curr + moe_out + ssc * sd
    torch.cuda.synchronize()
    t_py = (time.perf_counter() - t0) * 1000.0 / iterations

    # 2. Native C++ Single-Call Token Decode Loop (1 PyBind transition / token)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iterations):
        out_cxx = cpp_gemm.execute_full_token_decode_cxx(
            hidden_states,
            fused_gate_up_layers,
            down_layers,
            top_k_weights_layers,
            shared_gate_layers,
            shared_up_layers,
            shared_down_layers,
            shared_score_layers
        )
    torch.cuda.synchronize()
    t_cxx = (time.perf_counter() - t0) * 1000.0 / iterations

    print(f"1. Python 40-Layer Loop (560 transitions) : {t_py:.2f} ms / token")
    print(f"2. Native C++ Full Decode Loop (1 transition): {t_cxx:.2f} ms / token")
    print(f"Speedup from Native C++ Decode Loop         : {((t_py - t_cxx) / t_py) * 100:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    benchmark_decode_loop()
