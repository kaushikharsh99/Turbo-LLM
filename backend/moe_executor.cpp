#include "moe_executor.h"
#include "grouped_gemm.h"
#include "dequant_cache.h"
#include "fp8_dequant.h"
#include <pybind11/pybind11.h>
#include <future>
#include <vector>

namespace py = pybind11;

torch::Tensor execute_moe(
    torch::Tensor hidden_states,
    std::vector<torch::Tensor> gate_weights,
    std::vector<torch::Tensor> up_weights,
    std::vector<torch::Tensor> down_weights,
    torch::Tensor top_k_weights
) {
    return grouped_gemm_moe(hidden_states, gate_weights, up_weights, down_weights, top_k_weights);
}

torch::Tensor execute_moe_with_cache(
    int64_t layer_id,
    std::vector<int64_t> expert_ids,
    torch::Tensor hidden_states,
    torch::Tensor top_k_weights,
    py::object loader
) {
    // Dynamically synchronize cache limit from Python loader
    if (py::hasattr(loader, "cache_limit")) {
        int64_t py_limit = loader.attr("cache_limit").cast<int64_t>();
        g_dequant_cache.set_limit(py_limit);
    }

    std::vector<torch::Tensor> gate_weights(expert_ids.size());
    std::vector<torch::Tensor> up_weights(expert_ids.size());
    std::vector<torch::Tensor> down_weights(expert_ids.size());

    std::vector<size_t> miss_indices;
    
    for (size_t i = 0; i < expert_ids.size(); ++i) {
        int64_t exp_id = expert_ids[i];
        if (g_dequant_cache.has(layer_id, exp_id)) {
            auto cached = g_dequant_cache.get(layer_id, exp_id);
            gate_weights[i] = cached.gate;
            up_weights[i] = cached.up;
            down_weights[i] = cached.down;

            // Increment Python loader's gpu_hits count
            if (py::hasattr(loader, "gpu_hits")) {
                loader.attr("gpu_hits") = loader.attr("gpu_hits").cast<int64_t>() + 1;
            }
        } else {
            miss_indices.push_back(i);
        }
    }

    if (!miss_indices.empty()) {
        auto dtype = hidden_states.scalar_type();
        
        std::vector<std::vector<torch::Tensor>> raw_tensors_list;
        raw_tensors_list.reserve(miss_indices.size());
        
        for (size_t idx : miss_indices) {
            int64_t exp_id = expert_ids[idx];
            auto raw = loader.attr("load_expert_raw")(layer_id, exp_id);
            auto raw_tuple = raw.cast<py::tuple>();
            
            std::vector<torch::Tensor> exp_tensors;
            exp_tensors.reserve(6);
            for (size_t i = 0; i < 6; ++i) {
                if (raw_tuple[i].is_none()) {
                    exp_tensors.push_back(torch::Tensor());
                } else {
                    exp_tensors.push_back(raw_tuple[i].cast<torch::Tensor>());
                }
            }
            raw_tensors_list.push_back(exp_tensors);
        }

        // Asynchronously dequantize cache misses in parallel CPU threads (without GIL interference)
        std::vector<std::future<CachedExpert>> futures;
        for (size_t k = 0; k < miss_indices.size(); ++k) {
            auto exp_tensors = raw_tensors_list[k];
            futures.push_back(std::async(std::launch::async, [exp_tensors, dtype]() {
                auto gate_fp8 = exp_tensors[0];
                auto gate_scale = exp_tensors[1];
                auto up_fp8 = exp_tensors[2];
                auto up_scale = exp_tensors[3];
                auto down_fp8 = exp_tensors[4];
                auto down_scale = exp_tensors[5];

                auto gate_proj = dequantize_fp8(gate_fp8, gate_scale, dtype);
                auto up_proj = dequantize_fp8(up_fp8, up_scale, dtype);
                auto down_proj = dequantize_fp8(down_fp8, down_scale, dtype);

                return CachedExpert{gate_proj, up_proj, down_proj};
            }));
        }

        for (size_t k = 0; k < miss_indices.size(); ++k) {
            size_t idx = miss_indices[k];
            int64_t exp_id = expert_ids[idx];
            CachedExpert expert = futures[k].get();
            
            g_dequant_cache.put(layer_id, exp_id, expert);
            
            gate_weights[idx] = expert.gate;
            up_weights[idx] = expert.up;
            down_weights[idx] = expert.down;
        }
    }

    return grouped_gemm_moe(hidden_states, gate_weights, up_weights, down_weights, top_k_weights);
}

int64_t get_cache_size() {
    return g_dequant_cache.size();
}
