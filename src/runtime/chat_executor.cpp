#include "chat_executor.hpp"
#include "sampling.hpp"
#include "../profiler.hpp"
#include <iostream>
#include <vector>
#include <cstring>

namespace turbo {

ChatExecutor::ChatExecutor(
    const Model& m, 
    std::shared_ptr<MemoryManager> mem_mgr, 
    std::shared_ptr<Scheduler> sched,
    const Config& cfg
) : model(m), memory_manager(mem_mgr), scheduler(sched), config(cfg) {
    executor = std::make_unique<Executor>(scheduler->get_exec_stream());
    tokenizer = std::make_unique<Tokenizer>(model);
}

ChatExecutor::~ChatExecutor() {}

void ChatExecutor::run(const std::string& prompt) {
    auto& profiler = Profiler::get();
    profiler.start_generation();

    // 1. Tokenize prompt
    std::vector<int> tokens = tokenizer->encode(prompt);
    if (tokens.empty()) {
        std::cerr << "Empty prompt tokenization!" << std::endl;
        return;
    }

    std::cout << "Encoded Prompt: ";
    for (int t : tokens) std::cout << t << " ";
    std::cout << "\nStarting Generation Loop...\n\n" << prompt;

    // Pre-allocate CPU/GPU workspace buffers
    int hidden_size = model.config.hidden_size;
    int vocab_size = model.config.vocab_size;
    
    // Allocate space for device hidden states (for prefill/decode)
    float* dev_hidden = reinterpret_cast<float*>(memory_manager->get_device_io_buffer());
    float* dev_workspace = reinterpret_cast<float*>(memory_manager->get_device_workspace());

    // Initialize KV cache
    KVCache kv_cache(model.config.num_layers, model.config.num_kv_heads, 128, 4096, memory_manager->get_device_kv_cache());

    int num_generated = 0;
    int current_pos = 0;
    
    // We will do single-token generation loop for simplicity of layer prefetching overlap
    while (num_generated < config.max_new_tokens) {
        int seq_len = (num_generated == 0) ? static_cast<int>(tokens.size()) : 1;
        
        // 2. Embedding Lookup (simulated for dev_hidden)
        // Extract embedding vector(s) for current tokens
        // For decoding (seq_len = 1), copy one row of size hidden_size.
        // For prefill, copy seq_len rows.
        std::memset(dev_hidden, 0, seq_len * hidden_size * sizeof(float));
        
        // Mock embedding loading: set some values based on token ID
        for (int i = 0; i < seq_len; ++i) {
            int tok_id = (num_generated == 0) ? tokens[i] : tokens.back();
            dev_hidden[i * hidden_size] = static_cast<float>(tok_id % 100) * 0.01f;
        }

        // 3. Sequential Layer Execution with Double Buffering Prefetch
        scheduler->ensure_loaded(model, 0); // Load layer 0 immediately

        for (int l = 0; l < model.config.num_layers; ++l) {
            void* current_layer_weights = scheduler->get_active_device_buffer();
            
            // Execute current layer
            executor->execute(
                model.layers[l],
                current_layer_weights,
                dev_hidden,
                dev_workspace,
                kv_cache,
                l,
                1, // batch_size = 1
                seq_len,
                current_pos
            );

            // Asynchronously fetch next layer
            scheduler->prefetch_next(model, l);

            // Synchronize and swap double buffers for next layer
            scheduler->swap_buffers();
        }

        // 4. Output Layer Norm (RMSNorm)
        float* normed_out = dev_workspace;
        // Mock output norm weight: all 1s
        std::vector<float> mock_norm_weight(hidden_size, 1.0f);
        launch_rmsnorm(normed_out, dev_hidden + (seq_len - 1) * hidden_size, mock_norm_weight.data(), 1e-6f, hidden_size, 1, scheduler->get_exec_stream());

        // 5. LM Head GEMM
        // logits [vocab_size] = normed_out [1 x hidden_size] * output_weight [hidden_size x vocab_size]
        float* logits = normed_out + hidden_size;
        // Mock output head multiplication
        std::memset(logits, 0, vocab_size * sizeof(float));
        // Fill simulated logits
        for (int v = 0; v < vocab_size; ++v) {
            logits[v] = normed_out[v % hidden_size] * 0.05f;
        }

        // 6. Sample Token
        profiler.start_event();
        int sampled_token = Sampler::sample(logits, vocab_size, config.temperature, config.top_p);
        profiler.record_sampling(profiler.end_event());

        // Check for EOS
        if (sampled_token == tokenizer->get_eos_id() || sampled_token == tokenizer->get_pad_id()) {
            std::cout << " [EOS]" << std::endl;
            break;
        }

        // Print token
        std::string token_str = tokenizer->decode(sampled_token);
        std::cout << token_str << std::flush;

        tokens.push_back(sampled_token);
        current_pos += seq_len;
        num_generated++;
        
        profiler.update_vram(model.config.vocab_size * hidden_size * sizeof(float) + memory_manager->get_layer_size() * 2);
    }

    std::cout << "\n";
    profiler.end_generation(num_generated);
    profiler.print_summary();
}

} // namespace turbo
