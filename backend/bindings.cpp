#include <torch/extension.h>
#include <vector>
#include "moe_executor.h"

// Declarations for C++ GEMM functions in cutlass_gemm.cpp
torch::Tensor cublas_cached_gemv_fp16(torch::Tensor A, torch::Tensor B);
torch::Tensor cublas_grouped_gemm_fp16(torch::Tensor x, std::vector<torch::Tensor> weights);
torch::Tensor cublas_fused_gate_up_grouped_gemm_fp16(torch::Tensor x, std::vector<torch::Tensor> fused_weights);
torch::Tensor fused_silu_mul_cpp(torch::Tensor gate, torch::Tensor up);
torch::Tensor execute_full_token_decode_cxx(
    torch::Tensor hidden_states,
    std::vector<std::vector<torch::Tensor>> fused_gate_up_layers,
    std::vector<std::vector<torch::Tensor>> down_layers,
    std::vector<torch::Tensor> top_k_weights_layers,
    std::vector<torch::Tensor> shared_gate_layers,
    std::vector<torch::Tensor> shared_up_layers,
    std::vector<torch::Tensor> shared_down_layers,
    std::vector<torch::Tensor> shared_score_layers
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("execute_moe", &execute_moe, "Execute sequential MoE layer (LibTorch C++ implementation)");
    m.def("execute_moe_with_cache", &execute_moe_with_cache, "Execute sequential MoE layer with C++ dequant cache");
    m.def("get_cache_size", &get_cache_size, "Get current number of experts cached in C++ dequant cache");
    m.def("cublas_cached_gemv_fp16", &cublas_cached_gemv_fp16, "Cached cuBLAS GEMV FP16");
    m.def("cublas_grouped_gemm_fp16", &cublas_grouped_gemm_fp16, "cuBLAS Grouped GEMM FP16");
    m.def("cublas_fused_gate_up_grouped_gemm_fp16", &cublas_fused_gate_up_grouped_gemm_fp16, "Fused Gate+Up Grouped GEMM FP16");
    m.def("fused_silu_mul_cpp", &fused_silu_mul_cpp, "Fused SiLU(Gate) * Up kernel");
    m.def("execute_full_token_decode_cxx", &execute_full_token_decode_cxx, "Full 40-layer native C++ token decode loop");
}
