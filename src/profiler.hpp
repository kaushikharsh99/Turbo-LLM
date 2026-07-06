#pragma once

#include <string>
#include <chrono>
#include <map>
#include <vector>

namespace turbo {

struct LayerProfile {
    double ssd_read_ms = 0.0;
    double ram_copy_ms = 0.0;
    double vram_copy_ms = 0.0;
    double wait_ms = 0.0;
    double dequant_ms = 0.0;
    double attention_ms = 0.0;
    double router_ms = 0.0;
    double moe_ms = 0.0;
};

class Profiler {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point event_start;
    
    std::map<int, LayerProfile> layer_profiles;
    double total_sampling_ms = 0.0;
    double total_generation_ms = 0.0;
    size_t tokens_generated = 0;
    size_t peak_vram_bytes = 0;
    
public:
    static Profiler& get() {
        static Profiler instance;
        return instance;
    }

    void start_generation();
    void end_generation(size_t num_tokens);
    
    void start_event();
    double end_event(); // returns elapsed milliseconds

    void record_layer_metric(int layer_idx, const std::string& metric, double val_ms);
    void record_sampling(double val_ms);
    void update_vram(size_t bytes);
    
    void print_summary() const;
};

} // namespace turbo
