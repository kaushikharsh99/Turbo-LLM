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
#include <fstream>

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
        turbo::ChatExecutor chat(*model, mem_mgr, scheduler, config);
        chat.run(config.prompt_str);
    } else {
        // Batch mode (maximizing throughput)
        std::vector<std::string> batch_prompts;
        if (!config.prompts_jsonl_path.empty()) {
            std::cout << "Reading prompts from JSONL file: " << config.prompts_jsonl_path << "\n";
            std::ifstream infile(config.prompts_jsonl_path);
            if (!infile.is_open()) {
                std::cerr << "Failed to open prompts JSONL file: " << config.prompts_jsonl_path << "\n";
                return 1;
            }
            std::string line;
            while (std::getline(infile, line)) {
                if (line.empty()) continue;
                std::string prompt = "";
                size_t p_idx = line.find("\"prompt\"");
                if (p_idx == std::string::npos) p_idx = line.find("\"text\"");
                
                if (p_idx != std::string::npos) {
                    size_t colon_idx = line.find(":", p_idx);
                    if (colon_idx != std::string::npos) {
                        size_t start_quote = line.find("\"", colon_idx);
                        if (start_quote != std::string::npos) {
                            size_t end_quote = line.find("\"", start_quote + 1);
                            if (end_quote != std::string::npos) {
                                prompt = line.substr(start_quote + 1, end_quote - start_quote - 1);
                            }
                        }
                    }
                }
                if (!prompt.empty()) {
                    batch_prompts.push_back(prompt);
                }
            }
            std::cout << "Loaded " << batch_prompts.size() << " prompts from JSONL file.\n";
        }
        
        if (batch_prompts.empty()) {
            std::cout << "No prompts loaded from JSONL. Generating default batch prompts...\n";
            for (int i = 0; i < config.batch_size; ++i) {
                batch_prompts.push_back("Explain AI concept number " + std::to_string(i) + " in one simple sentence.");
            }
        }
        
        turbo::BatchExecutor batch(*model, mem_mgr, scheduler, config);
        batch.run(batch_prompts);
    }

    std::cout << "\nTurbo LLM Execution Finished Successfully!\n";
    return 0;
}
