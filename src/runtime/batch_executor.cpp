#include "batch_executor.hpp"
#include "sampling.hpp"
#include "../profiler.hpp"
#include <iostream>
#include <vector>
#include <cstring>

namespace turbo {

BatchExecutor::BatchExecutor(
    const Model& m, 
    std::shared_ptr<MemoryManager> mem_mgr, 
    std::shared_ptr<Scheduler> sched,
    const Config& cfg
) : model(m), memory_manager(mem_mgr), scheduler(sched), config(cfg) {
    executor = std::make_unique<Executor>(scheduler->get_exec_stream());
    tokenizer = std::make_unique<Tokenizer>(model);
}

BatchExecutor::~BatchExecutor() {}

void BatchExecutor::run(const std::vector<std::string>& prompts) {
    auto& profiler = Profiler::get();
    profiler.start_generation();

    int batch_size = static_cast<int>(prompts.size());
    std::cout << "Starting Batch Execution Mode with batch size: " << batch_size << "\n";

    // 1. Tokenize all prompts
    std::vector<std::vector<int>> batch_tokens(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        batch_tokens[b] = tokenizer->encode(prompts[b]);
        std::cout << "  Prompt " << b << ": " << batch_tokens[b].size() << " tokens\n";
    }

    int hidden_size = model.config.hidden_size;
    int vocab_size = model.config.vocab_size;

    // Allocate workspace
    float* dev_hidden = reinterpret_cast<float*>(memory_manager->get_device_io_buffer());
    float* dev_workspace = reinterpret_cast<float*>(memory_manager->get_device_workspace());

    // Initialize KV cache
    KVCache kv_cache(model.config.num_layers, model.config.num_kv_heads, 128, 4096, memory_manager->get_device_kv_cache());

    std::vector<bool> active(batch_size, true);
    std::vector<std::vector<int>> generated_tokens(batch_size);
    int step = 0;
    int total_tokens_generated = 0;

    while (step < config.max_new_tokens) {
        bool any_active = false;
        for (int b = 0; b < batch_size; ++b) {
            if (active[b]) {
                any_active = true;
                break;
            }
        }
        if (!any_active) break;

        int active_token_id = step == 0 ? batch_tokens[0][0] : (generated_tokens[0].empty() ? 0 : generated_tokens[0].back());
        profiler.start_step(step, active_token_id);

        // 2. Setup concatenated hidden state for active batch elements
        // Shape: [batch_size, seq_len, hidden_size]
        // Here we mock the embedding loading for the active sequences
        std::memset(dev_hidden, 0, batch_size * hidden_size * sizeof(float));
        for (int b = 0; b < batch_size; ++b) {
            if (!active[b]) continue;
            int tok_id = step == 0 ? batch_tokens[b][0] : generated_tokens[b].back();
            // Mock embedding vector
            dev_hidden[b * hidden_size] = static_cast<float>(tok_id % 100) * 0.01f;
        }

        // 3. Batch Layer Execution (Double buffering prefetch)
        scheduler->ensure_loaded(model, 0); // Load layer 0 immediately

        for (int l = 0; l < model.config.num_layers; ++l) {
            void* current_layer_weights = scheduler->get_active_device_buffer();
            
            // Execute current layer over the ENTIRE batch
            executor->execute(
                model.layers[l],
                current_layer_weights,
                dev_hidden,
                dev_workspace,
                kv_cache,
                l,
                batch_size,
                1, // seq_len = 1 for decoding step
                step,
                model.config.num_heads,
                model.config.num_kv_heads
            );

            // Asynchronously fetch next layer
            scheduler->prefetch_next(model, l);

            // Synchronize and swap double buffers for next layer
            scheduler->swap_buffers();
        }

        // 4. Output RMSNorm + LM Head for all batch elements
        float* normed_out = dev_workspace;
        std::vector<float> mock_norm_weight(hidden_size, 1.0f);
        launch_rmsnorm(normed_out, dev_hidden, mock_norm_weight.data(), 1e-6f, hidden_size, batch_size, scheduler->get_exec_stream());

        // Compute logits and sample for each active prompt
        float* logits = normed_out + batch_size * hidden_size;
        for (int b = 0; b < batch_size; ++b) {
            if (!active[b]) continue;

            float* token_logits = logits + b * vocab_size;
            float* token_normed = normed_out + b * hidden_size;
            
            // Mock output head multiplication
            std::memset(token_logits, 0, vocab_size * sizeof(float));
            for (int v = 0; v < vocab_size; ++v) {
                token_logits[v] = token_normed[v % hidden_size] * 0.05f;
            }

            // 5. Sample token
            profiler.start_event();
            int sampled_token = Sampler::sample(token_logits, vocab_size, config.temperature, config.top_p);
            profiler.record_sampling(profiler.end_event());

            generated_tokens[b].push_back(sampled_token);
            total_tokens_generated++;

            // Check for EOS
            if (sampled_token == tokenizer->get_eos_id() || sampled_token == tokenizer->get_pad_id() || generated_tokens[b].size() >= config.max_new_tokens) {
                active[b] = false;
                std::cout << "  Prompt " << b << " finished generation (" << generated_tokens[b].size() << " tokens).\n";
            }
        }

        step++;
        profiler.update_vram(model.config.vocab_size * hidden_size * sizeof(float) + memory_manager->get_layer_size() * 2);
        profiler.end_step();
    }

    std::cout << "\n=== Generated Sequences ===\n";
    for (int b = 0; b < batch_size; ++b) {
        std::cout << "Prompt " << b << " output:\n" 
                  << tokenizer->decode(generated_tokens[b]) << "\n-----------------\n";
    }

    profiler.end_generation(total_tokens_generated);
    profiler.print_summary();
    profiler.export_json("batch_profiler_run.json");
    std::cout << "Exported detailed execution profile to: batch_profiler_run.json\n";
}

} // namespace turbo
