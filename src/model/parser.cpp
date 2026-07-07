#include "parser.hpp"
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cstdint>

namespace turbo {

std::unique_ptr<Model> Parser::parse(const GGUFLoader& loader) {
    auto model = std::make_unique<Model>();
    const auto& tensors = loader.get_tensors();
    const auto& metadata = loader.get_metadata();

    // 1. Read configuration from metadata
    std::string arch = "qwen35moe";
    loader.get_metadata_string("general.architecture", arch);
    model->config.architecture = arch;

    // Use default values for Qwen3.6-35B-A3B if metadata lookup fails
    uint32_t num_layers = 41;
    loader.get_metadata_uint32(arch + ".block_count", num_layers);
    model->config.num_layers = num_layers;

    uint32_t hidden_size = 2048;
    loader.get_metadata_uint32(arch + ".embedding_length", hidden_size);
    model->config.hidden_size = hidden_size;

    // Intermediate size can be inferred or default
    model->config.intermediate_size = 512; // default for Qwen3.6-35B-A3B experts

    uint32_t num_heads = 16;
    loader.get_metadata_uint32(arch + ".attention.head_count", num_heads);
    model->config.num_heads = num_heads;

    uint32_t num_kv_heads = 2;
    loader.get_metadata_uint32(arch + ".attention.head_count_kv", num_kv_heads);
    model->config.num_kv_heads = num_kv_heads;

    uint32_t max_seq_len = 262144;
    loader.get_metadata_uint32(arch + ".context_length", max_seq_len);
    model->config.max_seq_len = max_seq_len;

    uint32_t num_experts = 256;
    loader.get_metadata_uint32(arch + ".expert_count", num_experts);
    model->config.num_experts = num_experts;

    uint32_t num_experts_per_tok = 8;
    loader.get_metadata_uint32(arch + ".expert_used_count", num_experts_per_tok);
    model->config.num_experts_per_tok = num_experts_per_tok;

    float rms_norm_eps = 1e-6f;
    loader.get_metadata_float(arch + ".attention.layer_norm_rms_epsilon", rms_norm_eps);
    model->config.rms_norm_eps = rms_norm_eps;

    float rope_theta = 10000000.0f;
    loader.get_metadata_float(arch + ".rope.freq_base", rope_theta);
    model->config.rope_theta = rope_theta;

    // 2. Read non-layer tensors
    auto find_tensor = [&](const std::string& name) -> GGUFTensor {
        auto it = tensors.find(name);
        if (it != tensors.end()) {
            return it->second;
        }
        // Try fallback
        for (const auto& pair : tensors) {
            if (pair.first.find(name) != std::string::npos) {
                return pair.second;
            }
        }
        return GGUFTensor{};
    };

    model->token_embeddings = find_tensor("token_embd.weight");
    model->output_norm = find_tensor("output_norm.weight");
    model->output_weight = find_tensor("output.weight");

    // Vocab size from token embeddings shape (usually shape[0] or shape[1])
    if (!model->token_embeddings.shape.empty()) {
        model->config.vocab_size = model->token_embeddings.shape.size() > 1 ? 
            model->token_embeddings.shape[1] : model->token_embeddings.shape[0];
    }

    // 3. Read layers
    model->layers.resize(model->config.num_layers);
    for (int l = 0; l < model->config.num_layers; ++l) {
        Layer& layer = model->layers[l];
        layer.id = l;
        std::string prefix = "blk." + std::to_string(l) + ".";

        layer.input_layernorm = find_tensor(prefix + "attn_norm.weight");
        layer.post_attention_layernorm = find_tensor(prefix + "post_attention_norm.weight");

        layer.q_proj = find_tensor(prefix + "attn_qkv.weight"); // Merged QKV
        if (!layer.q_proj.name.empty()) {
            layer.k_proj = layer.q_proj;
            layer.v_proj = layer.q_proj;
        } else {
            // Separate Q, K, V projections
            layer.q_proj = find_tensor(prefix + "attn_q.weight");
            layer.k_proj = find_tensor(prefix + "attn_k.weight");
            layer.v_proj = find_tensor(prefix + "attn_v.weight");
        }
        layer.o_proj = find_tensor(prefix + "ssm_out.weight");
        if (layer.o_proj.name.empty()) {
            layer.o_proj = find_tensor(prefix + "attn_output.weight");
        }

        layer.gate_inp = find_tensor(prefix + "ffn_gate_inp.weight");

        // Experts (try 3D grouped expert tensors first)
        GGUFTensor ffn_gate_exps = find_tensor(prefix + "ffn_gate_exps.weight");
        GGUFTensor ffn_up_exps = find_tensor(prefix + "ffn_up_exps.weight");
        GGUFTensor ffn_down_exps = find_tensor(prefix + "ffn_down_exps.weight");

        if (!ffn_gate_exps.name.empty()) {
            layer.expert_gate.push_back(ffn_gate_exps);
            layer.expert_up.push_back(ffn_up_exps);
            layer.expert_down.push_back(ffn_down_exps);
        } else {
            // Individual expert tensors
            for (int e = 0; e < model->config.num_experts; ++e) {
                GGUFTensor gate_e = find_tensor(prefix + "ffn_gate.weight." + std::to_string(e));
                GGUFTensor up_e = find_tensor(prefix + "ffn_up.weight." + std::to_string(e));
                GGUFTensor down_e = find_tensor(prefix + "ffn_down.weight." + std::to_string(e));
                if (gate_e.name.empty()) break;
                layer.expert_gate.push_back(gate_e);
                layer.expert_up.push_back(up_e);
                layer.expert_down.push_back(down_e);
            }
        }

        // Shared experts
        layer.shared_expert_gate_inp = find_tensor(prefix + "ffn_gate_inp_shexp.weight");
        layer.shared_expert_gate = find_tensor(prefix + "ffn_gate_shexp.weight");
        layer.shared_expert_up = find_tensor(prefix + "ffn_up_shexp.weight");
        layer.shared_expert_down = find_tensor(prefix + "ffn_down_shexp.weight");
        layer.has_shared_expert = !layer.shared_expert_gate.name.empty();
    }

    // 4. Parse vocabulary from metadata arrays
    auto tokens_it = metadata.find("tokenizer.ggml.tokens");
    if (tokens_it != metadata.end() && tokens_it->second.type == 9) {
        const auto& bytes = tokens_it->second.raw_data;
        if (bytes.size() >= 12) {
            size_t offset = 0;
            uint32_t item_type = *reinterpret_cast<const uint32_t*>(&bytes[offset]);
            offset += 4;
            uint64_t item_count = *reinterpret_cast<const uint64_t*>(&bytes[offset]);
            offset += 8;
            
            model->vocab_tokens.reserve(item_count);
            for (uint64_t i = 0; i < item_count; ++i) {
                if (offset + 8 > bytes.size()) break;
                uint64_t len = *reinterpret_cast<const uint64_t*>(&bytes[offset]);
                offset += 8;
                if (offset + len > bytes.size()) break;
                std::string s(reinterpret_cast<const char*>(&bytes[offset]), len);
                offset += len;
                model->vocab_tokens.push_back(s);
            }
        }
    }

    auto scores_it = metadata.find("tokenizer.ggml.scores");
    if (scores_it != metadata.end() && scores_it->second.type == 9) {
        const auto& bytes = scores_it->second.raw_data;
        if (bytes.size() >= 12) {
            size_t offset = 0;
            uint32_t item_type = *reinterpret_cast<const uint32_t*>(&bytes[offset]);
            offset += 4;
            uint64_t item_count = *reinterpret_cast<const uint64_t*>(&bytes[offset]);
            offset += 8;
            
            model->vocab_scores.reserve(item_count);
            for (uint64_t i = 0; i < item_count; ++i) {
                if (offset + 4 > bytes.size()) break;
                float s = *reinterpret_cast<const float*>(&bytes[offset]);
                offset += 4;
                model->vocab_scores.push_back(s);
            }
        }
    }

    auto types_it = metadata.find("tokenizer.ggml.token_type");
    if (types_it != metadata.end() && types_it->second.type == 9) {
        const auto& bytes = types_it->second.raw_data;
        if (bytes.size() >= 12) {
            size_t offset = 0;
            uint32_t item_type = *reinterpret_cast<const uint32_t*>(&bytes[offset]);
            offset += 4;
            uint64_t item_count = *reinterpret_cast<const uint64_t*>(&bytes[offset]);
            offset += 8;
            
            model->vocab_types.reserve(item_count);
            for (uint64_t i = 0; i < item_count; ++i) {
                if (offset + 4 > bytes.size()) break;
                int32_t t = *reinterpret_cast<const int32_t*>(&bytes[offset]);
                offset += 4;
                model->vocab_types.push_back(t);
            }
        }
    }

    return model;
}

} // namespace turbo
