#include <torch/extension.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <iostream>

// Shared cache for cuBLAS/cuBLASLt handles and descriptors
struct CublasBenchmarkCache {
    cublasHandle_t handle = nullptr;
    cublasLtHandle_t lt_handle = nullptr;
    void* workspace = nullptr;
    size_t workspace_size = 32 * 1024 * 1024; // 32 MB workspace

    CublasBenchmarkCache() {
        cublasCreate(&handle);
        cublasLtCreate(&lt_handle);
        cudaMalloc(&workspace, workspace_size);
    }

    ~CublasBenchmarkCache() {
        if (workspace) cudaFree(workspace);
        if (handle) cublasDestroy(handle);
        if (lt_handle) cublasLtDestroy(lt_handle);
    }
};

static CublasBenchmarkCache g_bench_cache;

// Level 1: Cached cuBLAS GEMV (cublasGemmEx) for M=1 decode shapes
torch::Tensor cublas_cached_gemv_fp16(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Input tensors must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat16 && B.dtype() == torch::kFloat16, "Input tensors must be Float16");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);

    cublasStatus_t status = cublasGemmEx(
        g_bench_cache.handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B.data_ptr(), CUDA_R_16F, K,
        A.data_ptr(), CUDA_R_16F, K,
        &beta,
        C.data_ptr(), CUDA_R_16F, N,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasGemmEx failed");
    return C;
}

// Grouped GEMM execution for top-k expert weights
torch::Tensor cublas_grouped_gemm_fp16(torch::Tensor x, std::vector<torch::Tensor> weights) {
    int problem_count = weights.size();
    int M = x.size(0);
    int K = x.size(1);
    int N = weights[0].size(0);

    auto outputs = torch::empty({problem_count, M, N}, x.options());

    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);

    for (int i = 0; i < problem_count; ++i) {
        cublasGemmEx(
            g_bench_cache.handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            weights[i].data_ptr(), CUDA_R_16F, K,
            x.data_ptr(), CUDA_R_16F, K,
            &beta,
            outputs[i].data_ptr(), CUDA_R_16F, N,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
    }

    return outputs;
}

// 🥇 1. Fused Gate+Up Grouped GEMM: Zero-allocation pre-concatenated weights [1024, 2048]
torch::Tensor cublas_fused_gate_up_grouped_gemm_fp16(torch::Tensor x, std::vector<torch::Tensor> fused_weights) {
    int problem_count = fused_weights.size();
    int M = x.size(0);
    int K = x.size(1);
    int N_fused = fused_weights[0].size(0); // 1024

    auto fused_outputs = torch::empty({problem_count, M, N_fused}, x.options());

    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);

    for (int i = 0; i < problem_count; ++i) {
        cublasGemmEx(
            g_bench_cache.handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N_fused, M, K,
            &alpha,
            fused_weights[i].data_ptr(), CUDA_R_16F, K,
            x.data_ptr(), CUDA_R_16F, K,
            &beta,
            fused_outputs[i].data_ptr(), CUDA_R_16F, N_fused,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
    }

    return fused_outputs;
}

// Vectorized zero-copy fused SiLU(gate) * up
torch::Tensor fused_silu_mul_cpp(torch::Tensor gate, torch::Tensor up) {
    auto out = torch::empty_like(gate);
    at::silu_out(out, gate);
    out.mul_(up);
    return out;
}

// 🚀 Native C++ Full Token Decode Loop across all MoE layers (1 boundary transition per token)
torch::Tensor execute_full_token_decode_cxx(
    torch::Tensor hidden_states,
    std::vector<std::vector<torch::Tensor>> fused_gate_up_layers,
    std::vector<std::vector<torch::Tensor>> down_layers,
    std::vector<torch::Tensor> top_k_weights_layers,
    std::vector<torch::Tensor> shared_gate_layers,
    std::vector<torch::Tensor> shared_up_layers,
    std::vector<torch::Tensor> shared_down_layers,
    std::vector<torch::Tensor> shared_score_layers
) {
    int num_layers = fused_gate_up_layers.size();
    auto current_hidden = hidden_states;
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);

    for (int l = 0; l < num_layers; ++l) {
        int problem_count = fused_gate_up_layers[l].size(); // top_k (8)
        int M = current_hidden.size(0);
        int K = current_hidden.size(1);
        int N_fused = fused_gate_up_layers[l][0].size(0); // 1024
        int N_sub = N_fused / 2; // 512

        // 1. Fused Gate+Up Grouped GEMM for top-k experts
        auto fused_out = torch::empty({problem_count, M, N_fused}, current_hidden.options());
        for (int i = 0; i < problem_count; ++i) {
            cublasGemmEx(
                g_bench_cache.handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                N_fused, M, K,
                &alpha,
                fused_gate_up_layers[l][i].data_ptr(), CUDA_R_16F, K,
                current_hidden.data_ptr(), CUDA_R_16F, K,
                &beta,
                fused_out[i].data_ptr(), CUDA_R_16F, N_fused,
                CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            );
        }

        // 2. Fused SiLU(Gate) * Up
        auto g_fused = fused_out.slice(2, 0, N_sub);
        auto u_fused = fused_out.slice(2, N_sub, N_fused);
        auto inter_fused = torch::empty_like(g_fused);
        at::silu_out(inter_fused, g_fused);
        inter_fused.mul_(u_fused);

        // 3. Down Grouped GEMM
        auto down_out = torch::empty({problem_count, M, K}, current_hidden.options());
        for (int i = 0; i < problem_count; ++i) {
            cublasGemmEx(
                g_bench_cache.handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                K, M, N_sub,
                &alpha,
                down_layers[l][i].data_ptr(), CUDA_R_16F, N_sub,
                inter_fused[i].data_ptr(), CUDA_R_16F, N_sub,
                &beta,
                down_out[i].data_ptr(), CUDA_R_16F, K,
                CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            );
        }

        // 4. Top-K weighted expert sum
        auto moe_out = (down_out * top_k_weights_layers[l].view({problem_count, 1, 1})).sum(0);

        // 5. Shared Expert projections
        int N_shared = shared_gate_layers[l].size(0);
        auto s_gate = torch::empty({M, N_shared}, current_hidden.options());
        auto s_up = torch::empty({M, N_shared}, current_hidden.options());
        auto s_down = torch::empty({M, K}, current_hidden.options());
        auto s_score = torch::empty({M, 1}, current_hidden.options());

        cublasGemmEx(g_bench_cache.handle, CUBLAS_OP_T, CUBLAS_OP_N, N_shared, M, K, &alpha, shared_gate_layers[l].data_ptr(), CUDA_R_16F, K, current_hidden.data_ptr(), CUDA_R_16F, K, &beta, s_gate.data_ptr(), CUDA_R_16F, N_shared, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cublasGemmEx(g_bench_cache.handle, CUBLAS_OP_T, CUBLAS_OP_N, N_shared, M, K, &alpha, shared_up_layers[l].data_ptr(), CUDA_R_16F, K, current_hidden.data_ptr(), CUDA_R_16F, K, &beta, s_up.data_ptr(), CUDA_R_16F, N_shared, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        auto s_hidden = torch::empty_like(s_gate);
        at::silu_out(s_hidden, s_gate);
        s_hidden.mul_(s_up);

        cublasGemmEx(g_bench_cache.handle, CUBLAS_OP_T, CUBLAS_OP_N, K, M, N_shared, &alpha, shared_down_layers[l].data_ptr(), CUDA_R_16F, N_shared, s_hidden.data_ptr(), CUDA_R_16F, N_shared, &beta, s_down.data_ptr(), CUDA_R_16F, K, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cublasGemmEx(g_bench_cache.handle, CUBLAS_OP_T, CUBLAS_OP_N, 1, M, K, &alpha, shared_score_layers[l].data_ptr(), CUDA_R_16F, K, current_hidden.data_ptr(), CUDA_R_16F, K, &beta, s_score.data_ptr(), CUDA_R_16F, 1, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        s_score = torch::sigmoid(s_score);

        // Accumulate output into residual
        current_hidden = current_hidden + moe_out + (s_score * s_down);
    }

    return current_hidden;
}
