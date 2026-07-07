#pragma once

#include "../model/model.hpp"
#include "cuda.hpp"
#include "kv_cache.hpp"

namespace turbo {

struct LayerPointers {
    float* input_layernorm = nullptr;
    float* post_attention_layernorm = nullptr;
    void* qkv = nullptr; // merged QKV, or Q projection if separate
    void* k_proj = nullptr; // K projection if separate
    void* v_proj = nullptr; // V projection if separate
    void* o_proj = nullptr;
    float* gate_inp = nullptr;
    std::vector<void*> expert_gate;
    std::vector<void*> expert_up;
    std::vector<void*> expert_down;
    void* shared_expert_gate_inp = nullptr;
    void* shared_expert_gate = nullptr;
    void* shared_expert_up = nullptr;
    void* shared_expert_down = nullptr;
};

class Executor {
private:
    Stream_t stream;

    LayerPointers get_layer_pointers(const Layer& layer, void* device_base_ptr);

public:
    Executor(Stream_t cuda_stream);
    ~Executor();

    // Executes a single layer forward pass
    void execute(
        const Layer& layer,
        void* device_layer_weights, // Contiguous VRAM layer weights
        float* device_hidden_states, // Input/Output [batch, seq_len, hidden_size]
        float* device_workspace,     // VRAM workspace for intermediate calculations
        KVCache& kv_cache,
        int layer_idx,
        int batch_size,
        int seq_len,
        int pos,
        int num_heads = 16,
        int num_kv_heads = 2
    );
};

} // namespace turbo
