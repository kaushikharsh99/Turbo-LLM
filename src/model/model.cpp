#include "model.hpp"
#include <iostream>

namespace turbo {

void Model::print() const {
    std::cout << "========================================\n"
              << "Model Information:\n"
              << "  Architecture:         " << config.architecture << "\n"
              << "  Num Layers:           " << config.num_layers << "\n"
              << "  Hidden Size:          " << config.hidden_size << "\n"
              << "  Intermediate Size:    " << config.intermediate_size << "\n"
              << "  Num Attention Heads:  " << config.num_heads << "\n"
              << "  Num KV Heads:         " << config.num_kv_heads << "\n"
              << "  Max Sequence Length:  " << config.max_seq_len << "\n"
              << "  Num Experts:          " << config.num_experts << "\n"
              << "  Experts Per Token:    " << config.num_experts_per_tok << "\n"
              << "  RMS Norm Epsilon:     " << config.rms_norm_eps << "\n"
              << "  RoPE Theta:           " << config.rope_theta << "\n"
              << "  Vocab Size:           " << config.vocab_size << "\n"
              << "========================================\n";
}

} // namespace turbo
