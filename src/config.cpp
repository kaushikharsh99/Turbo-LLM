#include "config.hpp"
#include <iostream>
#include <cstring>
#include <cstdlib>

namespace turbo {

void Config::load_from_args(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "--model") == 0 || strcmp(argv[i], "-m") == 0) && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--max_new_tokens") == 0 && i + 1 < argc) {
            max_new_tokens = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            temperature = std::atof(argv[++i]);
        } else if (strcmp(argv[i], "--top_p") == 0 && i + 1 < argc) {
            top_p = std::atof(argv[++i]);
        } else if (strcmp(argv[i], "--chat") == 0) {
            chat_mode = true;
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            benchmarking = true;
        } else if (strcmp(argv[i], "--max_vram_mb") == 0 && i + 1 < argc) {
            max_vram_bytes = static_cast<size_t>(std::atoll(argv[++i])) * 1024ULL * 1024ULL;
        } else if (strcmp(argv[i], "--max_ram_mb") == 0 && i + 1 < argc) {
            max_ram_bytes = static_cast<size_t>(std::atoll(argv[++i])) * 1024ULL * 1024ULL;
        } else if (strcmp(argv[i], "--dtype") == 0 && i + 1 < argc) {
            dtype = argv[++i];
        }
    }
}

void Config::print() const {
    std::cout << "========================================\n"
              << "Turbo-LLM Configuration:\n"
              << "  Model Path:      " << model_path << "\n"
              << "  Max New Tokens:  " << max_new_tokens << "\n"
              << "  Temperature:     " << temperature << "\n"
              << "  Top P:           " << top_p << "\n"
              << "  Max VRAM:        " << (max_vram_bytes / (1024 * 1024)) << " MB\n"
              << "  Max RAM:         " << (max_ram_bytes / (1024 * 1024)) << " MB\n"
              << "  Chat Mode:       " << (chat_mode ? "Enabled" : "Disabled") << "\n"
              << "  Benchmarking:    " << (benchmarking ? "Enabled" : "Disabled") << "\n"
              << "  Data Type:       " << dtype << "\n"
              << "========================================\n";
}

} // namespace turbo
