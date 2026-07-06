#include "config.hpp"
#include "profiler.hpp"
#include "gguf/loader.hpp"
#include "model/model.hpp"
#include "model/parser.hpp"
#include "runtime/memory.hpp"
#include "runtime/scheduler.hpp"
#include "runtime/chat_executor.hpp"
#include "runtime/batch_executor.hpp"
#include <iostream>
#include <memory>
#include <vector>

int main(int argc, char* argv[]) {
    // 1. Load configuration from command line arguments
    auto& config = turbo::Config::get();
    config.load_from_args(argc, argv);
    config.print();

    // 2. Initialize GGUF Loader
    std::cout << "Loading model file: " << config.model_path << " ...\n";
    auto loader = std::make_shared<turbo::GGUFLoader>(config.model_path);
    if (!loader->load_header_and_metadata()) {
        std::cerr << "Failed to parse model file headers!\n";
        return 1;
    }
    std::cout << "GGUF file headers parsed successfully.\n";

    // 3. Parse model structure from GGUF tensors
    std::cout << "Mapping GGUF tensors to Model structure...\n";
    auto model = turbo::Parser::parse(*loader);
    if (!model) {
        std::cerr << "Failed to construct Model object!\n";
        return 1;
    }
    model->print();

    // Debug layer 0 tensors
    std::cout << "DEBUG Layer 0 Tensors:\n"
              << "  input_layernorm: name=\"" << model->layers[0].input_layernorm.name << "\", size=" << model->layers[0].input_layernorm.size_bytes << "\n"
              << "  q_proj (QKV):    name=\"" << model->layers[0].q_proj.name << "\", size=" << model->layers[0].q_proj.size_bytes << "\n"
              << "  o_proj:          name=\"" << model->layers[0].o_proj.name << "\", size=" << model->layers[0].o_proj.size_bytes << "\n"
              << "  gate_inp:        name=\"" << model->layers[0].gate_inp.name << "\", size=" << model->layers[0].gate_inp.size_bytes << "\n";

    // 4. Initialize Memory Manager
    std::cout << "Initializing Memory Manager...\n";
    auto mem_mgr = std::make_shared<turbo::MemoryManager>(loader);
    try {
        mem_mgr->initialize(*model);
    } catch (const std::exception& e) {
        std::cerr << "Memory initialization error: " << e.what() << "\n";
        return 1;
    }

    // 5. Initialize Scheduler
    std::cout << "Initializing Scheduler...\n";
    auto scheduler = std::make_shared<turbo::Scheduler>(mem_mgr);

    // 6. Run Execution Engine
    if (config.chat_mode) {
        // Chat mode (batch size = 1)
        std::string prompt = "Explain Mixture of Experts in one simple sentence.";
        std::cout << "Running Chat Mode. Default prompt: \"" << prompt << "\"\n";
        
        turbo::ChatExecutor chat(*model, mem_mgr, scheduler, config);
        chat.run(prompt);
    } else {
        // Batch mode (maximizing throughput)
        std::vector<std::string> batch_prompts = {
            "Explain quantum computing.",
            "Write a poem about a computer.",
            "What is the theory of relativity?",
            "How does a neural network learn?"
        };
        
        turbo::BatchExecutor batch(*model, mem_mgr, scheduler, config);
        batch.run(batch_prompts);
    }

    std::cout << "\nTurbo LLM Execution Finished Successfully!\n";
    return 0;
}
