#pragma once

#include <string>

namespace turbo {

struct Config {
    std::string model_path = "./model";
    int max_new_tokens = 50;
    float temperature = 0.7f;
    float top_p = 0.95f;
    size_t max_vram_bytes = 5800000000ULL; // ~5.8 GB
    size_t max_ram_bytes = 16000000000ULL;  // ~16 GB
    bool chat_mode = false;
    bool benchmarking = true;
    std::string dtype = "fp16";

    static Config& get() {
        static Config instance;
        return instance;
    }

    void load_from_args(int argc, char* argv[]);
    void print() const;
};

} // namespace turbo
