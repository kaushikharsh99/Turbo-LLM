#pragma once

#include <string>
#include <chrono>
#include <map>
#include <vector>

namespace turbo {

struct LayerProfile {
    // Pipeline / Memory
    double ssd_read_ms = 0.0;
    double wait_ms = 0.0;
    double vram_copy_ms = 0.0;

    // Attention
    double q_proj_ms = 0.0;
    double k_proj_ms = 0.0;
    double v_proj_ms = 0.0;
    double rotary_ms = 0.0;
    double flash_attn_ms = 0.0;
    double o_proj_ms = 0.0;

    // MoE
    double router_ms = 0.0;
    double topk_ms = 0.0;
    double expert_dequant_ms = 0.0;
    double expert_gemm1_ms = 0.0;
    double activation_ms = 0.0;
    double expert_gemm2_ms = 0.0;
    double expert_merge_ms = 0.0;

    // Others
    double residual_ms = 0.0;
    double norm_ms = 0.0;
    double sync_ms = 0.0;
};

struct StepProfile {
    int step = 0;
    int token_id = 0;
    std::map<int, LayerProfile> layers;
};

class Profiler {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point event_start;
    
    std::vector<StepProfile> step_profiles;
    int current_step_idx = -1;
    
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
    
    void start_step(int step_idx, int token_id);
    void end_step();
    
    void start_event();
    double end_event(); // returns elapsed milliseconds

    void record_layer_metric(int layer_idx, const std::string& metric, double val_ms);
    void record_sampling(double val_ms);
    void update_vram(size_t bytes);
    
    void print_summary() const;
    void export_json(const std::string& filepath) const;
};

} // namespace turbo
