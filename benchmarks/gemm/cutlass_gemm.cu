#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

// CUTLASS single GEMM template for M x K * K x N (ColumnMajor B weight)
template<typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator>
torch::Tensor cutlass_gemm_dispatch(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor; // Transposed weight B (N x K)
    using LayoutC = cutlass::layout::RowMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80
    >;

    auto C = torch::empty({M, N}, A.options());

    typename Gemm::Arguments args(
        {M, N, K},
        {reinterpret_cast<ElementA*>(A.data_ptr<at::Half>()), K},
        {reinterpret_cast<ElementB*>(B.data_ptr<at::Half>()), K},
        {reinterpret_cast<ElementC*>(C.data_ptr<at::Half>()), N},
        {reinterpret_cast<ElementC*>(C.data_ptr<at::Half>()), N},
        {1.0f, 0.0f}
    );

    Gemm gemm_op;
    cutlass::Status status = gemm_op(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM execution failed.");

    return C;
}

torch::Tensor cutlass_gemm_fp16(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat16 && B.dtype() == torch::kFloat16, "Tensors must be Float16");
    return cutlass_gemm_dispatch<cutlass::half_t, cutlass::half_t, cutlass::half_t, float>(A, B);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cutlass_gemm_fp16", &cutlass_gemm_fp16, "CUTLASS FP16 GEMM kernel");
}
