#include <torch/extension.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <iostream>

struct GroupedGemmCache {
    cublasHandle_t handle = nullptr;
    GroupedGemmCache() {
        cublasCreate(&handle);
    }
    ~GroupedGemmCache() {
        if (handle) cublasDestroy(handle);
    }
};

static GroupedGemmCache g_grouped_cache;

// Grouped GEMM batched execution for top-k expert weights
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
            g_grouped_cache.handle,
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cublas_grouped_gemm_fp16", &cublas_grouped_gemm_fp16, "cuBLAS Grouped GEMM FP16");
}
