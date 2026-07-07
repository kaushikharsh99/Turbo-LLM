#pragma once

#include "../model/model.hpp"
#include "scheduler.hpp"
#include "executor.hpp"
#include "tokenizer.hpp"
#include "../config.hpp"
#include <memory>
#include <string>
#include <vector>

namespace turbo {

class ChatExecutor {
private:
    const Model& model;
    std::shared_ptr<MemoryManager> memory_manager;
    std::shared_ptr<Scheduler> scheduler;
    std::unique_ptr<Executor> executor;
    std::unique_ptr<Tokenizer> tokenizer;
    const Config& config;

    // Persistent buffers for embedding/LM head (loaded once)
    std::vector<uint8_t> embedding_data;    // Raw F16 embedding table
    std::vector<float> output_norm_weight;  // Output RMSNorm weight [hidden_size]
    std::vector<uint8_t> lm_head_data;      // Raw quantized LM head weight
    bool weights_loaded = false;

    void load_non_layer_weights();
    void embed_token(float* out, int token_id);

public:
    ChatExecutor(
        const Model& m, 
        std::shared_ptr<MemoryManager> mem_mgr, 
        std::shared_ptr<Scheduler> sched,
        const Config& cfg
    );
    ~ChatExecutor();

    void run(const std::string& prompt);
};

} // namespace turbo
