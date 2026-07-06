#pragma once

#include <torch/extension.h>
#include <vector>

torch::Tensor grouped_gemm_moe(
    torch::Tensor hidden_states,
    std::vector<torch::Tensor> gate_weights,
    std::vector<torch::Tensor> up_weights,
    std::vector<torch::Tensor> down_weights,
    torch::Tensor top_k_weights
);
