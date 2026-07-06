#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

struct PersistentGPUWorkspace {
    void* d_workspace = nullptr;
    size_t workspace_size = 4 * 1024 * 1024; // 4MB cuBLASLt scratch space

    // Pre-allocated batched activation buffers
    torch::Tensor batched_hidden_buf;  // [8, 1, hidden]
    torch::Tensor batched_gate_buf;    // [8, 1, inter]
    torch::Tensor batched_up_buf;      // [8, 1, inter]
    torch::Tensor batched_down_buf;    // [8, 1, hidden]
    torch::Tensor stacked_gate_w_buf;  // [8, inter, hidden]
    torch::Tensor stacked_up_w_buf;    // [8, inter, hidden]
    torch::Tensor stacked_down_w_buf;  // [8, hidden, inter]
    torch::Tensor combined_out_buf;    // [1, hidden]

    cublasLtHandle_t cublaslt_handle = nullptr;

    void init();
    void ensure_capacity(int batch_size, int top_k, int hidden_dim, int inter_dim, torch::Device device);
    ~PersistentGPUWorkspace();
};

extern PersistentGPUWorkspace g_gpu_workspace;

// Batched cuBLASLt GEMM helper for FP16 tensors: C[i] = A[i] * B[i]
void cublas_strided_batched_gemm_fp16(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C,
    int batch_count,
    int64_t strideA,
    int64_t strideB,
    int64_t strideC,
    bool transA = false,
    bool transB = false
);
