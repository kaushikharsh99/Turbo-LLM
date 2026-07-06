#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"

// CUTLASS Grouped GEMM definition for SM80 (Ampere RTX 3050)
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, 8,
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, 8,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<16, 32, 32>,
    cutlass::gemm::GemmShape<16, 32, 32>,
    cutlass::gemm::GemmShape<16, 16, 16>,
    cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2
>::GemmKernel;

using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

torch::Tensor cutlass_grouped_gemm_fp16(torch::Tensor x, std::vector<torch::Tensor> weights) {
    int problem_count = weights.size();
    int M = x.size(0);
    int K = x.size(1);
    int N = weights[0].size(0);

    auto outputs = torch::empty({problem_count, M, N}, x.options());

    std::vector<cutlass::gemm::GemmCoord> problem_sizes;
    std::vector<ElementA*> ptr_A;
    std::vector<ElementB*> ptr_B;
    std::vector<ElementC*> ptr_C;
    std::vector<int64_t> lda;
    std::vector<int64_t> ldb;
    std::vector<int64_t> ldc;

    for (int i = 0; i < problem_count; ++i) {
        problem_sizes.push_back({M, N, K});
        ptr_A.push_back(reinterpret_cast<ElementA*>(x.data_ptr<at::Half>()));
        ptr_B.push_back(reinterpret_cast<ElementB*>(weights[i].data_ptr<at::Half>()));
        ptr_C.push_back(reinterpret_cast<ElementC*>(outputs[i].data_ptr<at::Half>()));
        lda.push_back(K);
        ldb.push_back(K);
        ldc.push_back(N);
    }

    // Allocate GPU memory for grouped GEMM arguments
    cutlass::gemm::GemmCoord* d_problem_sizes;
    ElementA** d_ptr_A;
    ElementB** d_ptr_B;
    ElementC** d_ptr_C;
    int64_t* d_lda;
    int64_t* d_ldb;
    int64_t* d_ldc;

    cudaMalloc(&d_problem_sizes, sizeof(cutlass::gemm::GemmCoord) * problem_count);
    cudaMalloc(&d_ptr_A, sizeof(ElementA*) * problem_count);
    cudaMalloc(&d_ptr_B, sizeof(ElementB*) * problem_count);
    cudaMalloc(&d_ptr_C, sizeof(ElementC*) * problem_count);
    cudaMalloc(&d_lda, sizeof(int64_t) * problem_count);
    cudaMalloc(&d_ldb, sizeof(int64_t) * problem_count);
    cudaMalloc(&d_ldc, sizeof(int64_t) * problem_count);

    cudaMemcpy(d_problem_sizes, problem_sizes.data(), sizeof(cutlass::gemm::GemmCoord) * problem_count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ptr_A, ptr_A.data(), sizeof(ElementA*) * problem_count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ptr_B, ptr_B.data(), sizeof(ElementB*) * problem_count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ptr_C, ptr_C.data(), sizeof(ElementC*) * problem_count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lda, lda.data(), sizeof(int64_t) * problem_count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ldb, ldb.data(), sizeof(int64_t) * problem_count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ldc, ldc.data(), sizeof(int64_t) * problem_count, cudaMemcpyHostToDevice);

    typename GemmGrouped::Arguments args(
        d_problem_sizes,
        problem_count,
        1,
        {1.0f, 0.0f},
        d_ptr_A,
        d_ptr_B,
        d_ptr_C,
        d_ptr_C,
        d_lda.data(),
        d_ldb.data(),
        d_ldc.data(),
        d_ldc.data()
    );

    GemmGrouped gemm_op;
    cutlass::Status status = gemm_op(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS Grouped GEMM failed.");

    cudaFree(d_problem_sizes);
    cudaFree(d_ptr_A);
    cudaFree(d_ptr_B);
    cudaFree(d_ptr_C);
    cudaFree(d_lda);
    cudaFree(d_ldb);
    cudaFree(d_ldc);

    return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cutlass_grouped_gemm_fp16", &cutlass_grouped_gemm_fp16, "CUTLASS Grouped GEMM FP16");
}
