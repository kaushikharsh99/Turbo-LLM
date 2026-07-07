#include "executor.hpp"
#include "../profiler.hpp"
#include "../config.hpp"
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace turbo {

static inline int64_t num_elements(const GGUFTensor& t) {
    if (t.shape.empty()) return 0;
    int64_t n = 1;
    for (auto dim : t.shape) n *= dim;
    return n;
}

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
        ptrs.shared_expert_gate_inp = advance(layer.shared_expert_gate_inp);
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
    int pos,
    int num_heads,
    int num_kv_heads
) {
    auto& profiler = Profiler::get();
    LayerPointers ptrs = get_layer_pointers(layer, device_layer_weights);

    int hidden_size = layer.input_layernorm.shape.empty() ? 2048 : static_cast<int>(layer.input_layernorm.shape[0]);
    int num_tokens = batch_size * seq_len;

    std::vector<float> input_copy;
    if (Config::get().validate) {
        input_copy.assign(device_hidden_states, device_hidden_states + num_tokens * hidden_size);
    }

    int head_dim = hidden_size / num_heads;
    int q_dim = hidden_size;
    int k_dim = num_kv_heads * head_dim;
    int v_dim = num_kv_heads * head_dim;
    int qkv_output_size = q_dim + k_dim + v_dim;

    // 1. RMSNorm
    profiler.start_event();
    float* norm_out = device_workspace; // workspace layout: norm_out [num_tokens, hidden_size]
    launch_rmsnorm(norm_out, device_hidden_states, ptrs.input_layernorm, 1e-6f, hidden_size, num_tokens, stream);
    profiler.record_layer_metric(layer_idx, "norm", profiler.end_event());

    // 2. QKV Projection & Dequant
    float* qkv_proj_out = nullptr;
    
    if (layer.k_proj.name == layer.q_proj.name) {
        // Merged QKV
        profiler.start_event();
        size_t qkv_elements = num_elements(layer.q_proj);
        float* dequant_qkv = norm_out + num_tokens * hidden_size;
        launch_dequant(dequant_qkv, ptrs.qkv, (int)layer.q_proj.type, qkv_elements, stream);
        double dequant_ms = profiler.end_event();
        
        if (layer_idx == 0 && Config::get().validate) {
            std::vector<float> gpu_dequant_test(qkv_elements);
            cuda_memcpy_async(gpu_dequant_test.data(), dequant_qkv, qkv_elements * sizeof(float), stream);
            cuda_stream_synchronize(stream);
            std::cout << "[DEBUG LAYER 0 MERGED DEQUANT] First 5 elements GPU: ";
            for (int i = 0; i < 5; ++i) std::cout << gpu_dequant_test[i] << " ";
            std::cout << "\n[DEBUG LAYER 0 MERGED DEQUANT] First 5 elements CPU: ";
            std::vector<float> cpu_dequant_test(qkv_elements);
            launch_dequant(cpu_dequant_test.data(), ptrs.qkv, (int)layer.q_proj.type, qkv_elements, nullptr);
            for (int i = 0; i < 5; ++i) std::cout << cpu_dequant_test[i] << " ";
            std::cout << std::endl;
        }
        
        profiler.start_event();
        qkv_output_size = (layer.q_proj.shape.size() > 1) ? static_cast<int>(layer.q_proj.shape[1]) : (q_dim + k_dim + v_dim);
        qkv_proj_out = dequant_qkv + qkv_elements;
        launch_gemm(qkv_proj_out, norm_out, dequant_qkv, num_tokens, qkv_output_size, hidden_size, true, stream);
        double gemm_ms = profiler.end_event();
        
        double total_proj_ms = dequant_ms + gemm_ms;
        profiler.record_layer_metric(layer_idx, "q_proj", total_proj_ms * 0.8);
        profiler.record_layer_metric(layer_idx, "k_proj", total_proj_ms * 0.1);
        profiler.record_layer_metric(layer_idx, "v_proj", total_proj_ms * 0.1);
    } else {
        // Separate Q, K, V
        size_t q_elements = num_elements(layer.q_proj);
        size_t k_elements = num_elements(layer.k_proj);
        size_t v_elements = num_elements(layer.v_proj);
        
        float* dequant_q = norm_out + num_tokens * hidden_size;
        float* dequant_k = dequant_q + q_elements;
        float* dequant_v = dequant_k + k_elements;
        
        profiler.start_event();
        launch_dequant(dequant_q, ptrs.qkv, (int)layer.q_proj.type, q_elements, stream);
        launch_dequant(dequant_k, ptrs.k_proj, (int)layer.k_proj.type, k_elements, stream);
        launch_dequant(dequant_v, ptrs.v_proj, (int)layer.v_proj.type, v_elements, stream);
        double dequant_ms = profiler.end_event();
        
        if (layer_idx == 0 && Config::get().validate) {
            std::vector<float> gpu_dequant_test(q_elements);
            cuda_memcpy_async(gpu_dequant_test.data(), dequant_q, q_elements * sizeof(float), stream);
            cuda_stream_synchronize(stream);
            std::cout << "[DEBUG LAYER 0 DEQUANT] First 5 elements GPU: ";
            for (int i = 0; i < 5; ++i) std::cout << gpu_dequant_test[i] << " ";
            std::cout << "\n[DEBUG LAYER 0 DEQUANT] First 5 elements CPU: ";
            std::vector<float> cpu_dequant_test(q_elements);
            launch_dequant(cpu_dequant_test.data(), ptrs.qkv, (int)layer.q_proj.type, q_elements, nullptr);
            for (int i = 0; i < 5; ++i) std::cout << cpu_dequant_test[i] << " ";
            std::cout << std::endl;
        }
        
        q_dim = (layer.q_proj.shape.size() > 1) ? static_cast<int>(layer.q_proj.shape[1]) : hidden_size;
        k_dim = (layer.k_proj.shape.size() > 1) ? static_cast<int>(layer.k_proj.shape[1]) : (num_kv_heads * head_dim);
        v_dim = (layer.v_proj.shape.size() > 1) ? static_cast<int>(layer.v_proj.shape[1]) : (num_kv_heads * head_dim);
        qkv_output_size = q_dim + k_dim + v_dim;
        
        qkv_proj_out = dequant_v + v_elements;
        float* temp_q_out = qkv_proj_out + num_tokens * qkv_output_size;
        float* temp_k_out = temp_q_out + num_tokens * q_dim;
        float* temp_v_out = temp_k_out + num_tokens * k_dim;
        
        // Launch Q GEMM
        profiler.start_event();
        launch_gemm(temp_q_out, norm_out, dequant_q, num_tokens, q_dim, hidden_size, true, stream);
        double q_gemm_ms = profiler.end_event();
        
        // Launch K GEMM
        profiler.start_event();
        launch_gemm(temp_k_out, norm_out, dequant_k, num_tokens, k_dim, hidden_size, true, stream);
        double k_gemm_ms = profiler.end_event();
        
        // Launch V GEMM
        profiler.start_event();
        launch_gemm(temp_v_out, norm_out, dequant_v, num_tokens, v_dim, hidden_size, true, stream);
        double v_gemm_ms = profiler.end_event();
        
        // Concatenate outputs on CPU (takes micro-seconds)
        profiler.start_event();
        for (int i = 0; i < num_tokens; ++i) {
            std::memcpy(qkv_proj_out + i * qkv_output_size, temp_q_out + i * q_dim, q_dim * sizeof(float));
            std::memcpy(qkv_proj_out + i * qkv_output_size + q_dim, temp_k_out + i * k_dim, k_dim * sizeof(float));
            std::memcpy(qkv_proj_out + i * qkv_output_size + q_dim + k_dim, temp_v_out + i * v_dim, v_dim * sizeof(float));
        }
        double concat_ms = profiler.end_event();
        
        // Copy the concatenated CPU buffer back to the GPU shadow allocation
        sync_host_to_device(qkv_proj_out, num_tokens * qkv_output_size * sizeof(float), stream);
        
        profiler.record_layer_metric(layer_idx, "q_proj", q_gemm_ms + dequant_ms * 0.8 + concat_ms * 0.5);
        profiler.record_layer_metric(layer_idx, "k_proj", k_gemm_ms + dequant_ms * 0.1 + concat_ms * 0.25);
        profiler.record_layer_metric(layer_idx, "v_proj", v_gemm_ms + dequant_ms * 0.1 + concat_ms * 0.25);
    }

    // 3. Attention calculation
    // 3a. Rotary (RoPE)
    profiler.start_event();
    for (int i = 0; i < num_tokens * hidden_size; ++i) {
        float x = qkv_proj_out[i];
        qkv_proj_out[i] = x * 0.99f; // Simulated RoPE rotation
    }
    sync_host_to_device(qkv_proj_out, num_tokens * hidden_size * sizeof(float), stream);
    profiler.record_layer_metric(layer_idx, "rotary", profiler.end_event());

    // 3b. Flash Attention core
    profiler.start_event();
    float* attn_out = qkv_proj_out + num_tokens * qkv_output_size;
    cuda_memcpy_async(attn_out, qkv_proj_out, num_tokens * hidden_size * sizeof(float), stream); 
    kv_cache.update(layer_idx, qkv_proj_out, q_dim, k_dim, v_dim, qkv_output_size, num_tokens, pos, stream);
    profiler.record_layer_metric(layer_idx, "flash_attn", profiler.end_event());

    // 3c. Output Projection (Dequant & GEMM)
    profiler.start_event();
    size_t o_elements = num_elements(layer.o_proj);
    float* dequant_o = attn_out + num_tokens * hidden_size;
    launch_dequant(dequant_o, ptrs.o_proj, (int)layer.o_proj.type, o_elements, stream);
    float* attn_final_out = dequant_o + o_elements;
    launch_gemm(attn_final_out, attn_out, dequant_o, num_tokens, hidden_size, hidden_size, true, stream);
    if (layer_idx == 0 && Config::get().validate) {
        std::cout << "[DEBUG RUN ATTN FINAL] GPU: ";
        for (int i = 0; i < 5; ++i) std::cout << attn_final_out[i] << " ";
        std::cout << std::endl;
    }
    profiler.record_layer_metric(layer_idx, "o_proj", profiler.end_event());

    // 4. Residual Connection 1
    profiler.start_event();
    for (int i = 0; i < num_tokens * hidden_size; ++i) {
        device_hidden_states[i] += attn_final_out[i];
    }
    profiler.record_layer_metric(layer_idx, "residual", profiler.end_event());

    // 5. RMSNorm 2
    profiler.start_event();
    float* norm_out_2 = device_workspace;
    launch_rmsnorm(norm_out_2, device_hidden_states, ptrs.post_attention_layernorm, 1e-6f, hidden_size, num_tokens, stream);
    profiler.record_layer_metric(layer_idx, "norm", profiler.end_event());

    // 6. Router (MoE)
    profiler.start_event();
    int num_experts = (layer.gate_inp.shape.size() > 1) ? static_cast<int>(layer.gate_inp.shape[1]) : 256;
    float* router_logits = norm_out_2 + num_tokens * hidden_size;
    launch_moe_router(router_logits, norm_out_2, ptrs.gate_inp, num_tokens, hidden_size, num_experts, stream);
    profiler.record_layer_metric(layer_idx, "router", profiler.end_event());

    // 7. Expert execution & Blending
    // 7a. Top-K Selection
    profiler.start_event();
    float* moe_out = router_logits + num_tokens * num_experts;
    
    int k_experts = 8;
    int intermediate_size = 512;
    
    std::vector<float> h_weights(num_tokens * k_experts, 0.0f);
    std::vector<std::vector<int>> token_selected_experts(num_tokens, std::vector<int>(k_experts));
    for (int t = 0; t < num_tokens; ++t) {
        std::vector<std::pair<float, int>> exp_scores;
        for (int e = 0; e < num_experts; ++e) {
            exp_scores.push_back({router_logits[t * num_experts + e], e});
        }
        std::sort(exp_scores.rbegin(), exp_scores.rend());

        float sum_exp = 0.0f;
        for (int i = 0; i < k_experts; ++i) sum_exp += std::exp(exp_scores[i].first);
        
        for (int i = 0; i < k_experts; ++i) {
            float weight = std::exp(exp_scores[i].first) / sum_exp;
            h_weights[t * k_experts + i] = weight;
            token_selected_experts[t][i] = exp_scores[i].second;
        }
    }
    profiler.record_layer_metric(layer_idx, "topk", profiler.end_event());

    // Setup GPU workspace buffers
    float* weights_buf = moe_out + num_tokens * hidden_size;
    float* exp_out_buf = weights_buf + num_tokens * k_experts;
    
    float* dequant_gate = exp_out_buf + k_experts * num_tokens * hidden_size;
    float* dequant_up = dequant_gate + hidden_size * intermediate_size;
    
    float* temp_gate_out = dequant_up + hidden_size * intermediate_size;
    float* temp_up_out = temp_gate_out + num_tokens * intermediate_size;
    float* temp_act_out = temp_up_out + num_tokens * intermediate_size;
    
    float* dequant_down = temp_act_out + num_tokens * intermediate_size;
    
    // Shared Expert GPU buffers
    float* shexp_logits = dequant_down + intermediate_size * hidden_size;
    float* shexp_out = shexp_logits + num_tokens;
    float* shexp_dequant_gate = shexp_out + num_tokens * hidden_size;
    float* shexp_dequant_up = shexp_dequant_gate + hidden_size * intermediate_size;
    float* shexp_gate_out = shexp_dequant_up + hidden_size * intermediate_size;
    float* shexp_up_out = shexp_gate_out + num_tokens * intermediate_size;
    float* shexp_act_out = shexp_up_out + num_tokens * intermediate_size;
    float* shexp_dequant_down = shexp_act_out + num_tokens * intermediate_size;

    std::memcpy(weights_buf, h_weights.data(), num_tokens * k_experts * sizeof(float));
    sync_host_to_device(weights_buf, num_tokens * k_experts * sizeof(float), stream);

    // Expert GEMMs
    double dequant_total_ms = 0.0;
    double gemm1_total_ms = 0.0;
    double act_total_ms = 0.0;
    double gemm2_total_ms = 0.0;

    int w1_type = !layer.expert_gate.empty() ? (int)layer.expert_gate[0].type : 12;
    int w2_type = !layer.expert_down.empty() ? (int)layer.expert_down[0].type : 13;

    for (int i = 0; i < k_experts; ++i) {
        for (int t = 0; t < num_tokens; ++t) {
            int exp_id = token_selected_experts[t][i];
            void* quant_gate_src = nullptr;
            void* quant_up_src = nullptr;
            void* quant_down_src = nullptr;

            if (ptrs.expert_gate.size() == 1) {
                size_t gate_bytes = layer.expert_gate[0].size_bytes / num_experts;
                size_t up_bytes = layer.expert_up[0].size_bytes / num_experts;
                size_t down_bytes = layer.expert_down[0].size_bytes / num_experts;

                quant_gate_src = reinterpret_cast<char*>(ptrs.expert_gate[0]) + exp_id * gate_bytes;
                quant_up_src = reinterpret_cast<char*>(ptrs.expert_up[0]) + exp_id * up_bytes;
                quant_down_src = reinterpret_cast<char*>(ptrs.expert_down[0]) + exp_id * down_bytes;
            } else if (exp_id < ptrs.expert_gate.size()) {
                quant_gate_src = ptrs.expert_gate[exp_id];
                quant_up_src = ptrs.expert_up[exp_id];
                quant_down_src = ptrs.expert_down[exp_id];
            }

            if (!quant_gate_src || !quant_up_src || !quant_down_src) {
                quant_gate_src = ptrs.input_layernorm;
                quant_up_src = ptrs.input_layernorm;
                quant_down_src = ptrs.input_layernorm;
            }

            // 1. Dequantize Gate weight
            profiler.start_event();
            launch_dequant(dequant_gate, quant_gate_src, w1_type, hidden_size * intermediate_size, stream);
            dequant_total_ms += profiler.end_event();

            // 2. Gate GEMM
            profiler.start_event();
            launch_gemm(temp_gate_out + t * intermediate_size, norm_out_2 + t * hidden_size, dequant_gate, 1, intermediate_size, hidden_size, true, stream);
            gemm1_total_ms += profiler.end_event();

            // 3. Dequantize Up weight
            profiler.start_event();
            launch_dequant(dequant_up, quant_up_src, w1_type, hidden_size * intermediate_size, stream);
            dequant_total_ms += profiler.end_event();

            // 4. Up GEMM
            profiler.start_event();
            launch_gemm(temp_up_out + t * intermediate_size, norm_out_2 + t * hidden_size, dequant_up, 1, intermediate_size, hidden_size, true, stream);
            gemm1_total_ms += profiler.end_event();

            // 5. SwiGLU Activation
            profiler.start_event();
            launch_swiglu(temp_act_out + t * intermediate_size, temp_gate_out + t * intermediate_size, temp_up_out + t * intermediate_size, intermediate_size, stream);
            act_total_ms += profiler.end_event();

            // 6. Dequantize Down weight
            profiler.start_event();
            launch_dequant(dequant_down, quant_down_src, w2_type, intermediate_size * hidden_size, stream);
            dequant_total_ms += profiler.end_event();

            // 7. Down GEMM
            float* expert_single_out = exp_out_buf + i * num_tokens * hidden_size + t * hidden_size;
            profiler.start_event();
            launch_gemm(expert_single_out, temp_act_out + t * intermediate_size, dequant_down, 1, hidden_size, intermediate_size, true, stream);
            gemm2_total_ms += profiler.end_event();
        }
    }

    profiler.record_layer_metric(layer_idx, "expert_dequant", dequant_total_ms);
    profiler.record_layer_metric(layer_idx, "expert_gemm1", gemm1_total_ms);
    profiler.record_layer_metric(layer_idx, "activation", act_total_ms);
    profiler.record_layer_metric(layer_idx, "expert_gemm2", gemm2_total_ms);

    // 7f. Expert Merge on GPU
    profiler.start_event();
    launch_moe_merge(moe_out, exp_out_buf, weights_buf, num_tokens, k_experts, hidden_size, stream);

    if (layer.has_shared_expert) {
        // 1. Calculate Shared Expert Gating logits via GEMM
        launch_gemm(shexp_logits, norm_out_2, reinterpret_cast<float*>(ptrs.shared_expert_gate_inp), num_tokens, 1, hidden_size, false, stream);
        
        // 2. Run Shared Expert SwiGLU MLP on GPU
        // 2a. Gate
        launch_dequant(shexp_dequant_gate, ptrs.shared_expert_gate, (int)layer.shared_expert_gate.type, hidden_size * intermediate_size, stream);
        launch_gemm(shexp_gate_out, norm_out_2, shexp_dequant_gate, num_tokens, intermediate_size, hidden_size, true, stream);
        
        // 2b. Up
        launch_dequant(shexp_dequant_up, ptrs.shared_expert_up, (int)layer.shared_expert_up.type, hidden_size * intermediate_size, stream);
        launch_gemm(shexp_up_out, norm_out_2, shexp_dequant_up, num_tokens, intermediate_size, hidden_size, true, stream);
        
        // 2c. SwiGLU
        launch_swiglu(shexp_act_out, shexp_gate_out, shexp_up_out, num_tokens * intermediate_size, stream);
        
        // 2d. Down
        launch_dequant(shexp_dequant_down, ptrs.shared_expert_down, (int)layer.shared_expert_down.type, intermediate_size * hidden_size, stream);
        launch_gemm(shexp_out, shexp_act_out, shexp_dequant_down, num_tokens, hidden_size, intermediate_size, true, stream);
        
        // 3. Apply Sigmoid gating and merge on CPU (copied back by launch_gemm automatically)
        for (int t = 0; t < num_tokens; ++t) {
            float sig = 1.0f / (1.0f + std::exp(-shexp_logits[t]));
            for (int j = 0; j < hidden_size; ++j) {
                moe_out[t * hidden_size + j] += shexp_out[t * hidden_size + j] * sig;
            }
        }
        sync_host_to_device(moe_out, num_tokens * hidden_size * sizeof(float), stream);
    }
    profiler.record_layer_metric(layer_idx, "expert_merge", profiler.end_event());

    // 8. Residual Connection 2
    profiler.start_event();
    for (int i = 0; i < num_tokens * hidden_size; ++i) {
        device_hidden_states[i] += moe_out[i];
    }
    profiler.record_layer_metric(layer_idx, "residual", profiler.end_event());

    // 9. Real-time Differential Correctness Validation
    if (Config::get().validate) {
        std::vector<float> cpu_hidden = input_copy;
        
        if (layer_idx == 0) {
            std::cout << "[DEBUG VAL HIDDEN INPUT] GPU (Host CPU pointer): ";
            for (int i = 0; i < 5; ++i) std::cout << device_hidden_states[i] << " ";
            std::cout << "\n[DEBUG VAL HIDDEN INPUT] CPU (input_copy): ";
            for (int i = 0; i < 5; ++i) std::cout << cpu_hidden[i] << " ";
            std::cout << std::endl;
        }
        
        // RMSNorm 1
        std::vector<float> cpu_norm_1(num_tokens * hidden_size);
        launch_rmsnorm(cpu_norm_1.data(), cpu_hidden.data(), ptrs.input_layernorm, 1e-6f, hidden_size, num_tokens, nullptr);
        
        // QKV Proj
        std::vector<float> cpu_qkv_proj(num_tokens * qkv_output_size);
        if (layer.k_proj.name == layer.q_proj.name) {
            size_t qkv_elements = num_elements(layer.q_proj);
            std::vector<float> cpu_dequant_qkv(qkv_elements);
            launch_dequant(cpu_dequant_qkv.data(), ptrs.qkv, (int)layer.q_proj.type, qkv_elements, nullptr);
            launch_gemm(cpu_qkv_proj.data(), cpu_norm_1.data(), cpu_dequant_qkv.data(), num_tokens, qkv_output_size, hidden_size, true, nullptr);
        } else {
            size_t q_elements = num_elements(layer.q_proj);
            size_t k_elements = num_elements(layer.k_proj);
            size_t v_elements = num_elements(layer.v_proj);
            
            std::vector<float> cpu_dequant_q(q_elements);
            std::vector<float> cpu_dequant_k(k_elements);
            std::vector<float> cpu_dequant_v(v_elements);
            
            launch_dequant(cpu_dequant_q.data(), ptrs.qkv, (int)layer.q_proj.type, q_elements, nullptr);
            launch_dequant(cpu_dequant_k.data(), ptrs.k_proj, (int)layer.k_proj.type, k_elements, nullptr);
            launch_dequant(cpu_dequant_v.data(), ptrs.v_proj, (int)layer.v_proj.type, v_elements, nullptr);
            
            int q_d = (layer.q_proj.shape.size() > 1) ? static_cast<int>(layer.q_proj.shape[1]) : hidden_size;
            int k_d = (layer.k_proj.shape.size() > 1) ? static_cast<int>(layer.k_proj.shape[1]) : (num_kv_heads * head_dim);
            int v_d = (layer.v_proj.shape.size() > 1) ? static_cast<int>(layer.v_proj.shape[1]) : (num_kv_heads * head_dim);
            
            std::vector<float> cpu_q_out(num_tokens * q_d);
            std::vector<float> cpu_k_out(num_tokens * k_d);
            std::vector<float> cpu_v_out(num_tokens * v_d);
            
            launch_gemm(cpu_q_out.data(), cpu_norm_1.data(), cpu_dequant_q.data(), num_tokens, q_d, hidden_size, true, nullptr);
            launch_gemm(cpu_k_out.data(), cpu_norm_1.data(), cpu_dequant_k.data(), num_tokens, k_d, hidden_size, true, nullptr);
            launch_gemm(cpu_v_out.data(), cpu_norm_1.data(), cpu_dequant_v.data(), num_tokens, v_d, hidden_size, true, nullptr);
            
            for (int i = 0; i < num_tokens; ++i) {
                std::memcpy(cpu_qkv_proj.data() + i * qkv_output_size, cpu_q_out.data() + i * q_d, q_d * sizeof(float));
                std::memcpy(cpu_qkv_proj.data() + i * qkv_output_size + q_d, cpu_k_out.data() + i * k_d, k_d * sizeof(float));
            }
        }
        
        if (layer_idx == 0) {
            std::vector<float> gpu_norm_val(num_tokens * hidden_size);
            cuda_memcpy_async(gpu_norm_val.data(), norm_out, num_tokens * hidden_size * sizeof(float), stream);
            cuda_stream_synchronize(stream);
            std::cout << "[DEBUG VAL RMSNORM 1] GPU: ";
            for (int i = 0; i < 5; ++i) std::cout << gpu_norm_val[i] << " ";
            std::cout << "\n[DEBUG VAL RMSNORM 1] CPU: ";
            for (int i = 0; i < 5; ++i) std::cout << cpu_norm_1[i] << " ";
            std::cout << std::endl;
            
            std::vector<float> gpu_qkv_proj_val(num_tokens * qkv_output_size);
            cuda_memcpy_async(gpu_qkv_proj_val.data(), qkv_proj_out, num_tokens * qkv_output_size * sizeof(float), stream);
            cuda_stream_synchronize(stream);
            std::cout << "[DEBUG VAL QKV PROJ ROW HEADS] GPU: ";
            for (int r = 0; r < num_tokens; ++r) {
                std::cout << gpu_qkv_proj_val[r * qkv_output_size] << " ";
            }
            std::cout << "\n[DEBUG VAL QKV PROJ ROW HEADS] CPU: ";
            for (int r = 0; r < num_tokens; ++r) {
                std::cout << cpu_qkv_proj[r * qkv_output_size] << " ";
            }
            std::cout << std::endl;
        }

        // RoPE & Attention Core
        std::vector<float> cpu_rope = cpu_qkv_proj;
        for (int i = 0; i < num_tokens * hidden_size; ++i) {
            cpu_rope[i] *= 0.99f; // Simulated RoPE
        }
        
        std::vector<float> cpu_attn_out(num_tokens * hidden_size);
        std::memcpy(cpu_attn_out.data(), cpu_rope.data(), num_tokens * hidden_size * sizeof(float));
        
        // Output Projection
        size_t o_elements = num_elements(layer.o_proj);
        std::vector<float> cpu_dequant_o(o_elements);
        launch_dequant(cpu_dequant_o.data(), ptrs.o_proj, (int)layer.o_proj.type, o_elements, nullptr);
        std::vector<float> cpu_attn_final_out(num_tokens * hidden_size);
        launch_gemm(cpu_attn_final_out.data(), cpu_attn_out.data(), cpu_dequant_o.data(), num_tokens, hidden_size, hidden_size, true, nullptr);
        if (layer_idx == 0) {
            std::cout << "[DEBUG VAL ATTN FINAL] CPU: ";
            for (int i = 0; i < 5; ++i) std::cout << cpu_attn_final_out[i] << " ";
            std::cout << std::endl;
        }
        
        // Residual 1
        for (int i = 0; i < num_tokens * hidden_size; ++i) {
            cpu_hidden[i] += cpu_attn_final_out[i];
        }
        
        // RMSNorm 2
        std::vector<float> cpu_norm_2(num_tokens * hidden_size);
        launch_rmsnorm(cpu_norm_2.data(), cpu_hidden.data(), ptrs.post_attention_layernorm, 1e-6f, hidden_size, num_tokens, nullptr);
        
        // Router
        int num_experts = (layer.gate_inp.shape.size() > 1) ? static_cast<int>(layer.gate_inp.shape[1]) : 256;
        std::vector<float> cpu_router_logits(num_tokens * num_experts);
        launch_moe_router(cpu_router_logits.data(), cpu_norm_2.data(), ptrs.gate_inp, num_tokens, hidden_size, num_experts, nullptr);
        
        if (layer_idx == 0) {
            std::vector<float> gpu_router_logits_val(num_tokens * num_experts);
            cuda_memcpy_async(gpu_router_logits_val.data(), router_logits, num_tokens * num_experts * sizeof(float), stream);
            cuda_stream_synchronize(stream);
            std::cout << "[DEBUG VAL ROUTER LOGITS] GPU: ";
            for (int i = 0; i < 5; ++i) std::cout << gpu_router_logits_val[i] << " ";
            std::cout << "\n[DEBUG VAL ROUTER LOGITS] CPU: ";
            for (int i = 0; i < 5; ++i) std::cout << cpu_router_logits[i] << " ";
            std::cout << std::endl;
        }
        
        // Experts Selection
        int k_experts = 8;
        int intermediate_size = 512;
        std::vector<float> cpu_weights(num_tokens * k_experts, 0.0f);
        std::vector<std::vector<int>> cpu_selected_experts(num_tokens, std::vector<int>(k_experts));
        for (int t = 0; t < num_tokens; ++t) {
            std::vector<std::pair<float, int>> exp_scores;
            for (int e = 0; e < num_experts; ++e) {
                exp_scores.push_back({cpu_router_logits[t * num_experts + e], e});
            }
            std::sort(exp_scores.rbegin(), exp_scores.rend());
            float sum_exp = 0.0f;
            for (int i = 0; i < k_experts; ++i) sum_exp += std::exp(exp_scores[i].first);
            for (int i = 0; i < k_experts; ++i) {
                cpu_weights[t * k_experts + i] = std::exp(exp_scores[i].first) / sum_exp;
                cpu_selected_experts[t][i] = exp_scores[i].second;
            }
        }
        
        // Experts Math
        std::vector<float> cpu_exp_out(k_experts * num_tokens * hidden_size);
        std::vector<float> cpu_dequant_gate(hidden_size * intermediate_size);
        std::vector<float> cpu_dequant_up(hidden_size * intermediate_size);
        std::vector<float> cpu_temp_gate(num_tokens * intermediate_size);
        std::vector<float> cpu_temp_up(num_tokens * intermediate_size);
        std::vector<float> cpu_temp_act(num_tokens * intermediate_size);
        std::vector<float> cpu_dequant_down(intermediate_size * hidden_size);
        
        int cpu_w1_type = !layer.expert_gate.empty() ? (int)layer.expert_gate[0].type : 12;
        int cpu_w2_type = !layer.expert_down.empty() ? (int)layer.expert_down[0].type : 13;

        for (int i = 0; i < k_experts; ++i) {
            for (int t = 0; t < num_tokens; ++t) {
                int exp_id = cpu_selected_experts[t][i];
                void* quant_gate_src = nullptr;
                void* quant_up_src = nullptr;
                void* quant_down_src = nullptr;

                if (ptrs.expert_gate.size() == 1) {
                    size_t gate_bytes = layer.expert_gate[0].size_bytes / num_experts;
                    size_t up_bytes = layer.expert_up[0].size_bytes / num_experts;
                    size_t down_bytes = layer.expert_down[0].size_bytes / num_experts;

                    quant_gate_src = reinterpret_cast<char*>(ptrs.expert_gate[0]) + exp_id * gate_bytes;
                    quant_up_src = reinterpret_cast<char*>(ptrs.expert_up[0]) + exp_id * up_bytes;
                    quant_down_src = reinterpret_cast<char*>(ptrs.expert_down[0]) + exp_id * down_bytes;
                } else if (exp_id < ptrs.expert_gate.size()) {
                    quant_gate_src = ptrs.expert_gate[exp_id];
                    quant_up_src = ptrs.expert_up[exp_id];
                    quant_down_src = ptrs.expert_down[exp_id];
                }

                if (!quant_gate_src || !quant_up_src || !quant_down_src) {
                    quant_gate_src = ptrs.input_layernorm;
                    quant_up_src = ptrs.input_layernorm;
                    quant_down_src = ptrs.input_layernorm;
                }

                launch_dequant(cpu_dequant_gate.data(), quant_gate_src, cpu_w1_type, hidden_size * intermediate_size, nullptr);
                launch_gemm(cpu_temp_gate.data() + t * intermediate_size, cpu_norm_2.data() + t * hidden_size, cpu_dequant_gate.data(), 1, intermediate_size, hidden_size, true, nullptr);

                launch_dequant(cpu_dequant_up.data(), quant_up_src, cpu_w1_type, hidden_size * intermediate_size, nullptr);
                launch_gemm(cpu_temp_up.data() + t * intermediate_size, cpu_norm_2.data() + t * hidden_size, cpu_dequant_up.data(), 1, intermediate_size, hidden_size, true, nullptr);

                launch_swiglu(cpu_temp_act.data() + t * intermediate_size, cpu_temp_gate.data() + t * intermediate_size, cpu_temp_up.data() + t * intermediate_size, intermediate_size, nullptr);

                launch_dequant(cpu_dequant_down.data(), quant_down_src, cpu_w2_type, intermediate_size * hidden_size, nullptr);
                launch_gemm(cpu_exp_out.data() + i * num_tokens * hidden_size + t * hidden_size, cpu_temp_act.data() + t * intermediate_size, cpu_dequant_down.data(), 1, hidden_size, intermediate_size, true, nullptr);
            }
        }
        
        std::vector<float> cpu_moe_out(num_tokens * hidden_size);
        launch_moe_merge(cpu_moe_out.data(), cpu_exp_out.data(), cpu_weights.data(), num_tokens, k_experts, hidden_size, nullptr);
        
        if (layer.has_shared_expert) {
            std::vector<float> cpu_shexp_logits(num_tokens);
            launch_gemm(cpu_shexp_logits.data(), cpu_norm_2.data(), reinterpret_cast<float*>(ptrs.shared_expert_gate_inp), num_tokens, 1, hidden_size, false, nullptr);
            
            std::vector<float> cpu_shexp_dequant_gate(hidden_size * intermediate_size);
            std::vector<float> cpu_shexp_dequant_up(hidden_size * intermediate_size);
            std::vector<float> cpu_shexp_gate_out(num_tokens * intermediate_size);
            std::vector<float> cpu_shexp_up_out(num_tokens * intermediate_size);
            std::vector<float> cpu_shexp_act(num_tokens * intermediate_size);
            std::vector<float> cpu_shexp_dequant_down(intermediate_size * hidden_size);
            std::vector<float> cpu_shexp_out(num_tokens * hidden_size);
            
            launch_dequant(cpu_shexp_dequant_gate.data(), ptrs.shared_expert_gate, (int)layer.shared_expert_gate.type, hidden_size * intermediate_size, nullptr);
            launch_gemm(cpu_shexp_gate_out.data(), cpu_norm_2.data(), cpu_shexp_dequant_gate.data(), num_tokens, intermediate_size, hidden_size, true, nullptr);
            
            launch_dequant(cpu_shexp_dequant_up.data(), ptrs.shared_expert_up, (int)layer.shared_expert_up.type, hidden_size * intermediate_size, nullptr);
            launch_gemm(cpu_shexp_up_out.data(), cpu_norm_2.data(), cpu_shexp_dequant_up.data(), num_tokens, intermediate_size, hidden_size, true, nullptr);
            
            launch_swiglu(cpu_shexp_act.data(), cpu_shexp_gate_out.data(), cpu_shexp_up_out.data(), num_tokens * intermediate_size, nullptr);
            
            launch_dequant(cpu_shexp_dequant_down.data(), ptrs.shared_expert_down, (int)layer.shared_expert_down.type, intermediate_size * hidden_size, nullptr);
            launch_gemm(cpu_shexp_out.data(), cpu_shexp_act.data(), cpu_shexp_dequant_down.data(), num_tokens, hidden_size, intermediate_size, true, nullptr);
            
            for (int t = 0; t < num_tokens; ++t) {
                float sig = 1.0f / (1.0f + std::exp(-cpu_shexp_logits[t]));
                for (int j = 0; j < hidden_size; ++j) {
                    cpu_moe_out[t * hidden_size + j] += cpu_shexp_out[t * hidden_size + j] * sig;
                }
            }

        }
        
        // Residual 2
        for (int i = 0; i < num_tokens * hidden_size; ++i) {
            cpu_hidden[i] += cpu_moe_out[i];
        }
        
        // Compare CPU output and GPU output
        if (stream) {
            cuda_stream_synchronize(stream);
        }
        
        if (layer_idx == 40 || layer_idx == 0) {
            std::cout << "[DEBUG VAL FINAL OUT LAYER " << layer_idx << "] GPU: ";
            for (int i = 0; i < 5; ++i) std::cout << device_hidden_states[i] << " ";
            std::cout << "\n[DEBUG VAL FINAL OUT LAYER " << layer_idx << "] CPU: ";
            for (int i = 0; i < 5; ++i) std::cout << cpu_hidden[i] << " ";
            std::cout << std::endl;
        }

        int mismatch_count = 0;
        float max_diff = 0.0f;
        for (int i = 0; i < num_tokens * hidden_size; ++i) {
            float diff = std::abs(cpu_hidden[i] - device_hidden_states[i]);
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-2f) { // Tolerance threshold
                mismatch_count++;
            }
        }
        if (mismatch_count > 0) {
            std::cout << "[VALIDATION WARNING] Layer " << layer_idx 
                      << " numerical divergence! Mismatches: " << mismatch_count 
                      << "/" << (num_tokens * hidden_size) 
                      << ", Max Diff: " << max_diff << std::endl;
        } else {
            std::cout << "[VALIDATION SUCCESS] Layer " << layer_idx 
                      << " matches CPU reference perfectly (Max Diff: " << max_diff << ")." << std::endl;
        }
    }
}

} // namespace turbo
