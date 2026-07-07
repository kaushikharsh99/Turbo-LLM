#include "chat_executor.hpp"
#include "sampling.hpp"
#include "../profiler.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>

namespace turbo {

// F16 to F32 conversion helper
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            // Zero
            uint32_t result = sign << 31;
            float f;
            std::memcpy(&f, &result, 4);
            return f;
        }
        // Subnormal
        float f = (sign ? -1.0f : 1.0f) * std::ldexp(static_cast<float>(mant), -24);
        return f;
    } else if (exp == 31) {
        // Inf or NaN
        uint32_t result = (sign << 31) | (0xFF << 23) | (mant << 13);
        float f;
        std::memcpy(&f, &result, 4);
        return f;
    }

    // Normal number
    uint32_t result = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &result, 4);
    return f;
}

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

void ChatExecutor::load_non_layer_weights() {
    if (weights_loaded) return;

    auto loader = memory_manager->get_loader();

    // 1. Load token embeddings (F16, shape [hidden_size, vocab_size] in GGUF)
    std::cout << "Loading token embeddings (" 
              << (model.token_embeddings.size_bytes / (1024 * 1024)) << " MB)..." << std::flush;
    embedding_data.resize(model.token_embeddings.size_bytes);
    loader->read_tensor_data(model.token_embeddings, embedding_data.data());
    std::cout << " done.\n";

    // 2. Load output RMSNorm weight (F32, shape [hidden_size])
    int hidden_size = model.config.hidden_size;
    output_norm_weight.resize(hidden_size);
    loader->read_tensor_data(model.output_norm, output_norm_weight.data());
    std::cout << "Loaded output norm weight.\n";

    // 3. Load LM head weight (Q6_K, shape [hidden_size, vocab_size] in GGUF)
    std::cout << "Loading LM head weight (" 
              << (model.output_weight.size_bytes / (1024 * 1024)) << " MB)..." << std::flush;
    lm_head_data.resize(model.output_weight.size_bytes);
    loader->read_tensor_data(model.output_weight, lm_head_data.data());
    std::cout << " done.\n";

    weights_loaded = true;
}

void ChatExecutor::embed_token(float* out, int token_id) {
    int hidden_size = model.config.hidden_size;
    
    // Token embedding table is stored as F16 in GGUF
    // Shape is [hidden_size, vocab_size] — each column is one token's embedding
    // In memory (row-major): row i contains hidden_dim[i] values for all vocab tokens
    // But GGUF stores it as [vocab_size, hidden_size] contiguously per token
    // i.e., token t's embedding starts at offset t * hidden_size * sizeof(f16)
    
    const uint16_t* f16_data = reinterpret_cast<const uint16_t*>(embedding_data.data());
    size_t row_offset = static_cast<size_t>(token_id) * hidden_size;
    
    for (int i = 0; i < hidden_size; ++i) {
        out[i] = f16_to_f32(f16_data[row_offset + i]);
    }
}void ChatExecutor::run(const std::string& initial_prompt) {
    auto& profiler = Profiler::get();
    load_non_layer_weights();

    int hidden_size = model.config.hidden_size;
    int vocab_size = model.config.vocab_size;

    float* dev_hidden = reinterpret_cast<float*>(memory_manager->get_device_io_buffer());
    float* dev_workspace = reinterpret_cast<float*>(memory_manager->get_device_workspace());

    // Initialize KV cache
    KVCache kv_cache(model.config.num_layers, model.config.num_kv_heads, 128, 4096, memory_manager->get_device_kv_cache());

    std::vector<int> history_tokens;
    int current_pos = 0;

    bool first_turn = true;
    std::string user_input = initial_prompt;

    bool interactive = initial_prompt.empty();
    if (interactive) {
        std::cout << "\n=== Interactive Chat Mode (type 'exit' or 'quit' to end) ===\n";
    }

    while (true) {
        if (!initial_prompt.empty() && first_turn) {
            user_input = initial_prompt;
        } else if (interactive) {
            std::cout << "\nUser: ";
            std::fflush(stdout);
            if (!std::getline(std::cin, user_input)) {
                break;
            }
            if (user_input == "exit" || user_input == "quit") {
                break;
            }
            if (user_input.empty()) continue;
        } else {
            // One-shot mode, first turn is done, exit loop
            break;
        }
        first_turn = false;

        // Wrap prompt in ChatML template
        std::string formatted_prompt = "<|im_start|>user\n" + user_input + "<|im_end|>\n<|im_start|>assistant\n";
        
        std::vector<int> new_tokens = tokenizer->encode(formatted_prompt);
        if (new_tokens.empty()) continue;

        size_t prefill_len = new_tokens.size();
        history_tokens.insert(history_tokens.end(), new_tokens.begin(), new_tokens.end());

        if (interactive) {
            std::cout << "Assistant: " << std::flush;
        } else {
            std::cout << "\nStarting Generation Loop...\n\n" << user_input << " ";
        }

        int num_generated = 0;
        profiler.start_generation();

        while (num_generated < config.max_new_tokens) {
            profiler.start_step(num_generated, history_tokens.back());
            int seq_len = (num_generated == 0) ? static_cast<int>(prefill_len) : 1;

            // 2. Real Embedding Lookup
            std::memset(dev_hidden, 0, seq_len * hidden_size * sizeof(float));
            for (int i = 0; i < seq_len; ++i) {
                int idx_in_history = history_tokens.size() - seq_len + i;
                int tok_id = history_tokens[idx_in_history];
                embed_token(dev_hidden + i * hidden_size, tok_id);
            }

            // 3. Sequential Layer Execution with Double Buffering Prefetch
            scheduler->ensure_loaded(model, 0);

            for (int l = 0; l < model.config.num_layers; ++l) {
                void* current_layer_weights = scheduler->get_active_device_buffer();
                
                executor->execute(
                    model.layers[l],
                    current_layer_weights,
                    dev_hidden,
                    dev_workspace,
                    kv_cache,
                    l,
                    1, // batch_size = 1
                    seq_len,
                    current_pos,
                    model.config.num_heads,
                    model.config.num_kv_heads
                );

                scheduler->prefetch_next(model, l);
                scheduler->swap_buffers();
            }

            // 4. Real Output Layer Norm (RMSNorm)
            float* normed_out = dev_workspace;
            launch_rmsnorm(normed_out, dev_hidden + (seq_len - 1) * hidden_size, output_norm_weight.data(), 1e-6f, hidden_size, 1, scheduler->get_exec_stream());

            // 5. Real LM Head GEMM
            float* logits = normed_out + hidden_size;
            const int CHUNK_SIZE = 4096;
            std::vector<float> dequant_chunk(static_cast<size_t>(hidden_size) * CHUNK_SIZE);
            
            for (int v_start = 0; v_start < vocab_size; v_start += CHUNK_SIZE) {
                int v_end = std::min(v_start + CHUNK_SIZE, vocab_size);
                int chunk_v = v_end - v_start;
                size_t elements_per_row = hidden_size;
                size_t chunk_elements = static_cast<size_t>(chunk_v) * elements_per_row;
                size_t elements_before = static_cast<size_t>(v_start) * elements_per_row;
                size_t total_elements = static_cast<size_t>(vocab_size) * hidden_size;
                size_t bytes_before = (model.output_weight.size_bytes * elements_before) / total_elements;
                
                const void* chunk_quant = lm_head_data.data() + bytes_before;
                if (chunk_elements > dequant_chunk.size()) {
                    dequant_chunk.resize(chunk_elements);
                }
                
                launch_dequant(dequant_chunk.data(), chunk_quant, (int)model.output_weight.type, chunk_elements, nullptr);
                launch_gemm(logits + v_start, normed_out, dequant_chunk.data(), 1, chunk_v, hidden_size, true, nullptr);
            }

            // 6. Sample Token
            profiler.start_event();
            int sampled_token = Sampler::sample(logits, vocab_size, config.temperature, config.top_p);
            profiler.record_sampling(profiler.end_event());

            // Check for EOS
            if (sampled_token == tokenizer->get_eos_id() || sampled_token == tokenizer->get_pad_id() || sampled_token == 248046) {
                std::cout << std::endl;
                profiler.end_step();
                break;
            }

            // Print token
            std::string token_str = tokenizer->decode(sampled_token);
            std::cout << token_str << std::flush;

            history_tokens.push_back(sampled_token);
            current_pos += seq_len;
            num_generated++;
            prefill_len = 1;
            
            profiler.update_vram(model.config.vocab_size * hidden_size * sizeof(float) + memory_manager->get_layer_size() * 2);
            profiler.end_step();
        }

        // Add EOS token to history to maintain clean dialog sequence
        history_tokens.push_back(tokenizer->get_eos_id());
        current_pos += 1;

        profiler.end_generation(num_generated);
    }
    
    profiler.print_summary();
    profiler.export_json("profiler_run.json");
}

} // namespace turbo
