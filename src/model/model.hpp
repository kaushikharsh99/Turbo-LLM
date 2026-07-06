#pragma once

#include "../gguf/loader.hpp"
#include <string>
#include <vector>

namespace turbo {

struct ModelConfig {
    std::string architecture;
    int num_layers = 0;
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int max_seq_len = 0;
    int num_experts = 0;
    int num_experts_per_tok = 0;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000000.0f;
    int vocab_size = 0;
};

struct Layer {
    int id = 0;

    // Norms
    GGUFTensor input_layernorm;
    GGUFTensor post_attention_layernorm;

    // Attention
    GGUFTensor q_proj;
    GGUFTensor k_proj;
    GGUFTensor v_proj;
    GGUFTensor o_proj;

    // MoE Router
    GGUFTensor gate_inp; // Router weight [num_experts, hidden_size]

    // Routed Experts (each expert has gate, up, down)
    std::vector<GGUFTensor> expert_gate; // [num_experts, intermediate_size, hidden_size]
    std::vector<GGUFTensor> expert_up;   // [num_experts, intermediate_size, hidden_size]
    std::vector<GGUFTensor> expert_down; // [num_experts, hidden_size, intermediate_size]

    // Shared Experts (if any)
    bool has_shared_expert = false;
    GGUFTensor shared_expert_gate;
    GGUFTensor shared_expert_up;
    GGUFTensor shared_expert_down;
};

struct Model {
    ModelConfig config;
    GGUFTensor token_embeddings;
    GGUFTensor output_norm;
    GGUFTensor output_weight; // LM Head

    std::vector<Layer> layers;

    // Tokenizer Vocabulary
    std::vector<std::string> vocab_tokens;
    std::vector<float> vocab_scores;
    std::vector<int32_t> vocab_types;

    void print() const;
};

} // namespace turbo
