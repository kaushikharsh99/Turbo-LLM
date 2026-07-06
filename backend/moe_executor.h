#pragma once

#include <torch/extension.h>
#include <vector>

// Forward declare pybind11 namespace and object
namespace pybind11 {
    class object;
}

torch::Tensor execute_moe(
    torch::Tensor hidden_states,
    std::vector<torch::Tensor> gate_weights,
    std::vector<torch::Tensor> up_weights,
    std::vector<torch::Tensor> down_weights,
    torch::Tensor top_k_weights
);

torch::Tensor execute_moe_with_cache(
    int64_t layer_id,
    std::vector<int64_t> expert_ids,
    torch::Tensor hidden_states,
    torch::Tensor top_k_weights,
    pybind11::object loader
);
