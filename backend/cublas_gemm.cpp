#include "cublas_gemm.h"
#include <c10/cuda/CUDAStream.h>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <tuple>

PersistentGPUWorkspace g_gpu_workspace;

void PersistentGPUWorkspace::init() {
    if (!cublaslt_handle) {
        cublasLtCreate(&cublaslt_handle);
    }
    if (!d_workspace) {
        cudaMalloc(&d_workspace, workspace_size);
    }
}

void PersistentGPUWorkspace::ensure_capacity(int batch_size, int top_k, int hidden_dim, int inter_dim, torch::Device device) {
    init();
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(device);
    
    if (!batched_gate_buf.defined() || batched_gate_buf.size(0) < top_k || batched_gate_buf.size(2) < inter_dim) {
        batched_hidden_buf = torch::empty({top_k, batch_size, hidden_dim}, options);
        batched_gate_buf = torch::empty({top_k, batch_size, inter_dim}, options);
        batched_up_buf = torch::empty({top_k, batch_size, inter_dim}, options);
        batched_down_buf = torch::empty({top_k, batch_size, hidden_dim}, options);
        
        stacked_gate_w_buf = torch::empty({top_k, inter_dim, hidden_dim}, options);
        stacked_up_w_buf = torch::empty({top_k, inter_dim, hidden_dim}, options);
        stacked_down_w_buf = torch::empty({top_k, hidden_dim, inter_dim}, options);

        combined_out_buf = torch::zeros({batch_size, hidden_dim}, options);
    }
}

PersistentGPUWorkspace::~PersistentGPUWorkspace() {
    if (d_workspace) {
        cudaFree(d_workspace);
        d_workspace = nullptr;
    }
    if (cublaslt_handle) {
        cublasLtDestroy(cublaslt_handle);
        cublaslt_handle = nullptr;
    }
}

// Simple Descriptor Cache key
struct GemmKey {
    int m, n, k;
    int batch_count;
    bool transA, transB;
    bool operator==(const GemmKey& o) const {
        return m == o.m && n == o.n && k == o.k && batch_count == o.batch_count && transA == o.transA && transB == o.transB;
    }
};

struct GemmKeyHash {
    size_t operator()(const GemmKey& key) const {
        return std::hash<int>()(key.m) ^ (std::hash<int>()(key.n) << 1) ^ (std::hash<int>()(key.k) << 2) ^ (std::hash<int>()(key.batch_count) << 3);
    }
};

struct CachedDescriptors {
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
};

static std::unordered_map<GemmKey, CachedDescriptors, GemmKeyHash> g_descriptor_cache;

void cublas_strided_batched_gemm_fp16(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C,
    int batch_count,
    int64_t strideA,
    int64_t strideB,
    int64_t strideC,
    bool transA,
    bool transB
) {
    g_gpu_workspace.init();

    int M = transA ? A.size(2) : A.size(1);
    int K = transA ? A.size(1) : A.size(2);
    int N = transB ? B.size(1) : B.size(2);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    float alpha = 1.0f;
    float beta = 0.0f;

    GemmKey key{M, N, K, batch_count, transA, transB};
    auto it = g_descriptor_cache.find(key);

    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;

    if (it == g_descriptor_cache.end()) {
        cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

        cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opB, sizeof(opB));
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opA, sizeof(opA));

        // Column-major representation: C^T = B^T * A^T
        int lda = transB ? K : N;
        int ldb = transA ? M : K;
        int ldc = N;

        cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, transB ? K : N, transB ? N : K, lda);
        cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, transA ? M : K, transA ? K : M, ldb);
        cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, N, M, ldc);

        // Configure strided batch attributes
        cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));
        cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB));

        cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));
        cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA));

        cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));
        cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC, sizeof(strideC));

        CachedDescriptors cd{operationDesc, Adesc, Bdesc, Cdesc};
        g_descriptor_cache[key] = cd;
    } else {
        operationDesc = it->second.operationDesc;
        Adesc = it->second.Adesc;
        Bdesc = it->second.Bdesc;
        Cdesc = it->second.Cdesc;
    }

    cublasLtMatmul(
        g_gpu_workspace.cublaslt_handle,
        operationDesc,
        &alpha,
        B.data_ptr(), Adesc,
        A.data_ptr(), Bdesc,
        &beta,
        C.data_ptr(), Cdesc,
        C.data_ptr(), Cdesc,
        NULL,
        g_gpu_workspace.d_workspace,
        g_gpu_workspace.workspace_size,
        stream
    );
}
