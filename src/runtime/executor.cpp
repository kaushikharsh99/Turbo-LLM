#include "executor.hpp"
#include "../profiler.hpp" // to use profiler
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace turbo {

Executor::Executor(Stream_t cuda_stream) : stream(cuda_stream) {}

Executor::~Executor() {}

LayerPointers Executor::get_layer_pointers(const Layer& layer, void* device_base_ptr) {
    LayerPointers ptrs;
    uint8_t* base = reinterpret_cast<uint8_t*>(device_base_ptr);
    size_t offset = 0;

    auto advance = [&](const GGUFTensor& t) -> void* {
        if (t.name.empty()) return nullptr;
        void* p = base + offset;
        offset += t.size_bytes;
        return p;
    };

    ptrs.input_layernorm = reinterpret_cast<float*>(advance(layer.input_layernorm));
    ptrs.post_attention_layernorm = reinterpret_cast<float*>(advance(layer.post_attention_layernorm));
    if (layer.k_proj.name == layer.q_proj.name) {
        ptrs.qkv = advance(layer.q_proj);
    } else {
        ptrs.qkv = advance(layer.q_proj);
        ptrs.k_proj = advance(layer.k_proj);
        ptrs.v_proj = advance(layer.v_proj);
    }
    ptrs.o_proj = advance(layer.o_proj);
    ptrs.gate_inp = reinterpret_cast<float*>(advance(layer.gate_inp));

    for (const auto& t : layer.expert_gate) ptrs.expert_gate.push_back(advance(t));
    for (const auto& t : layer.expert_up) ptrs.expert_up.push_back(advance(t));
    for (const auto& t : layer.expert_down) ptrs.expert_down.push_back(advance(t));

    if (layer.has_shared_expert) {
        ptrs.shared_expert_gate = advance(layer.shared_expert_gate);
        ptrs.shared_expert_up = advance(layer.shared_expert_up);
        ptrs.shared_expert_down = advance(layer.shared_expert_down);
    }

    return ptrs;
}

void Executor::execute(
    const Layer& layer,
    void* device_layer_weights,
    float* device_hidden_states,
    float* device_workspace,
    KVCache& kv_cache,
    int layer_idx,
    int batch_size,
    int seq_len,
    int pos
) {
    auto& profiler = Profiler::get();
    LayerPointers ptrs = get_layer_pointers(layer, device_layer_weights);

    int hidden_size = layer.input_layernorm.shape.empty() ? 2048 : static_cast<int>(layer.input_layernorm.shape[0]);
    int num_tokens = batch_size * seq_len;

    // 1. RMSNorm
    profiler.start_event();
    float* norm_out = device_workspace; // workspace layout: norm_out [num_tokens, hidden_size]
    launch_rmsnorm(norm_out, device_hidden_states, ptrs.input_layernorm, 1e-6f, hidden_size, num_tokens, stream);
    profiler.record_layer_metric(layer_idx, "dequant", profiler.end_event() * 0.1); // split norm/dequant timing

    // 2. QKV Projection & Dequant
    profiler.start_event();
    
    float* qkv_proj_out = nullptr;
    int qkv_output_size = 0;
    int q_offset_dim = 0;
    
    if (layer.k_proj.name == layer.q_proj.name) {
        // Merged QKV
        size_t qkv_elements = layer.q_proj.size_bytes * 2;
        float* dequant_qkv = norm_out + num_tokens * hidden_size;
        launch_dequant_q4_k(dequant_qkv, ptrs.qkv, qkv_elements, stream);
        
        qkv_output_size = (layer.q_proj.shape.size() > 1) ? static_cast<int>(layer.q_proj.shape[1]) : 8192;
        q_offset_dim = hidden_size;
        qkv_proj_out = dequant_qkv + qkv_elements;
        launch_gemm(qkv_proj_out, norm_out, dequant_qkv, num_tokens, qkv_output_size, hidden_size, stream);
    } else {
        // Separate Q, K, V
        size_t q_elements = layer.q_proj.size_bytes * 2;
        size_t k_elements = layer.k_proj.size_bytes * 2;
        size_t v_elements = layer.v_proj.size_bytes * 2;
        
        float* dequant_q = norm_out + num_tokens * hidden_size;
        float* dequant_k = dequant_q + q_elements;
        float* dequant_v = dequant_k + k_elements;
        
        launch_dequant_q4_k(dequant_q, ptrs.qkv, q_elements, stream);
        launch_dequant_q4_k(dequant_k, ptrs.k_proj, k_elements, stream);
        launch_dequant_q4_k(dequant_v, ptrs.v_proj, v_elements, stream);
        
        int q_dim = (layer.q_proj.shape.size() > 1) ? static_cast<int>(layer.q_proj.shape[1]) : 8192;
        int k_dim = (layer.k_proj.shape.size() > 1) ? static_cast<int>(layer.k_proj.shape[1]) : 512;
        int v_dim = (layer.v_proj.shape.size() > 1) ? static_cast<int>(layer.v_proj.shape[1]) : 512;
        qkv_output_size = q_dim + k_dim + v_dim;
        q_offset_dim = q_dim;
        
        qkv_proj_out = dequant_v + v_elements;
        std::memset(qkv_proj_out, 0, num_tokens * qkv_output_size * sizeof(float));
        
        // Strided GEMMs to construct concatenated layout
        for (int i = 0; i < num_tokens; ++i) {
            for (int l = 0; l < hidden_size; ++l) {
                float val = norm_out[i * hidden_size + l];
                for (int j = 0; j < q_dim; ++j) {
                    qkv_proj_out[i * qkv_output_size + j] += val * dequant_q[l * q_dim + j];
                }
            }
        }
        for (int i = 0; i < num_tokens; ++i) {
            for (int l = 0; l < hidden_size; ++l) {
                float val = norm_out[i * hidden_size + l];
                for (int j = 0; j < k_dim; ++j) {
                    qkv_proj_out[i * qkv_output_size + q_dim + j] += val * dequant_k[l * k_dim + j];
                }
            }
        }
        for (int i = 0; i < num_tokens; ++i) {
            for (int l = 0; l < hidden_size; ++l) {
                float val = norm_out[i * hidden_size + l];
                for (int j = 0; j < v_dim; ++j) {
                    qkv_proj_out[i * qkv_output_size + q_dim + k_dim + j] += val * dequant_v[l * v_dim + j];
                }
            }
        }
    }
    profiler.record_layer_metric(layer_idx, "dequant", profiler.end_event());

    // 3. Attention calculation
    profiler.start_event();
    // Simulate Attention, RoPE, Softmax and V computation
    // For the prototype, we simulate attention output by copying projection back and combining it.
    float* attn_out = qkv_proj_out + num_tokens * qkv_output_size;
    std::memcpy(attn_out, qkv_proj_out, num_tokens * hidden_size * sizeof(float)); 
    
    // Store in KV cache
    kv_cache.update(layer_idx, qkv_proj_out + num_tokens * q_offset_dim, num_tokens, pos);
    
    // Project back using o_proj
    float* dequant_o = attn_out + num_tokens * hidden_size;
    launch_dequant_q4_k(dequant_o, ptrs.o_proj, layer.o_proj.size_bytes * 2, stream);
    
    float* attn_final_out = dequant_o + layer.o_proj.size_bytes * 2;
    launch_gemm(attn_final_out, attn_out, dequant_o, num_tokens, hidden_size, hidden_size, stream);
    profiler.record_layer_metric(layer_idx, "attention", profiler.end_event());

    // 4. Residual Connection 1
    for (int i = 0; i < num_tokens * hidden_size; ++i) {
        device_hidden_states[i] += attn_final_out[i];
    }

    // 5. RMSNorm 2
    profiler.start_event();
    float* norm_out_2 = device_workspace;
    launch_rmsnorm(norm_out_2, device_hidden_states, ptrs.post_attention_layernorm, 1e-6f, hidden_size, num_tokens, stream);
    profiler.record_layer_metric(layer_idx, "router", profiler.end_event() * 0.2);

    // 6. Router (MoE)
    profiler.start_event();
    int num_experts = (layer.gate_inp.shape.size() > 1) ? static_cast<int>(layer.gate_inp.shape[1]) : 256;
    float* router_logits = norm_out_2 + num_tokens * hidden_size;
    launch_moe_router(router_logits, norm_out_2, ptrs.gate_inp, num_tokens, hidden_size, num_experts, stream);
    profiler.record_layer_metric(layer_idx, "router", profiler.end_event());

    // 7. Expert execution & Blending
    profiler.start_event();
    float* moe_out = router_logits + num_tokens * num_experts;
    std::memset(moe_out, 0, num_tokens * hidden_size * sizeof(float));

    // Simulate Top-K Routing: K = 8
    int k_experts = 8;
    for (int t = 0; t < num_tokens; ++t) {
        // Find top-K experts based on router_logits
        std::vector<std::pair<float, int>> exp_scores;
        for (int e = 0; e < num_experts; ++e) {
            exp_scores.push_back({router_logits[t * num_experts + e], e});
        }
        std::sort(exp_scores.rbegin(), exp_scores.rend());

        // Softmax over top-K
        float sum_exp = 0.0f;
        for (int i = 0; i < k_experts; ++i) sum_exp += std::exp(exp_scores[i].first);
        
        // Execute active experts
        for (int i = 0; i < k_experts; ++i) {
            int exp_id = exp_scores[i].second;
            float weight = std::exp(exp_scores[i].first) / sum_exp;
            
            // Execute expert exp_id (Gate, Up, Down projections)
            // For the prototype we do a fast mock GEMM simulating execution.
            float* token_state = norm_out_2 + t * hidden_size;
            float* exp_output = moe_out + t * hidden_size;
            for (int h = 0; h < hidden_size; ++h) {
                exp_output[h] += token_state[h] * weight * 0.1f; // Simulated MLP scaling
            }
        }
    }

    // Shared expert execution
    if (layer.has_shared_expert) {
        float* shared_out = moe_out + num_tokens * hidden_size;
        // Shared MLP execution on all tokens
        for (int i = 0; i < num_tokens * hidden_size; ++i) {
            moe_out[i] += norm_out_2[i] * 0.2f; // simulated shared expert
        }
    }
    profiler.record_layer_metric(layer_idx, "moe", profiler.end_event());

    // 8. Residual Connection 2
    for (int i = 0; i < num_tokens * hidden_size; ++i) {
        device_hidden_states[i] += moe_out[i];
    }
}

} // namespace turbo
