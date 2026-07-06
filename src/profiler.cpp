#include "profiler.hpp"
#include <iostream>
#include <iomanip>
#include <numeric>

namespace turbo {

void Profiler::start_generation() {
    start_time = std::chrono::high_resolution_clock::now();
    layer_profiles.clear();
    total_sampling_ms = 0.0;
    total_generation_ms = 0.0;
    tokens_generated = 0;
    peak_vram_bytes = 0;
}

void Profiler::end_generation(size_t num_tokens) {
    auto end_time = std::chrono::high_resolution_clock::now();
    total_generation_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    tokens_generated = num_tokens;
}

void Profiler::start_event() {
    event_start = std::chrono::high_resolution_clock::now();
}

double Profiler::end_event() {
    auto event_end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(event_end - event_start).count();
}

void Profiler::record_layer_metric(int layer_idx, const std::string& metric, double val_ms) {
    auto& profile = layer_profiles[layer_idx];
    if (metric == "ssd_read") profile.ssd_read_ms = val_ms;
    else if (metric == "ram_copy") profile.ram_copy_ms = val_ms;
    else if (metric == "vram_copy") profile.vram_copy_ms = val_ms;
    else if (metric == "wait") profile.wait_ms = val_ms;
    else if (metric == "dequant") profile.dequant_ms = val_ms;
    else if (metric == "attention") profile.attention_ms = val_ms;
    else if (metric == "router") profile.router_ms = val_ms;
    else if (metric == "moe") profile.moe_ms = val_ms;
}

void Profiler::record_sampling(double val_ms) {
    total_sampling_ms += val_ms;
}

void Profiler::update_vram(size_t bytes) {
    if (bytes > peak_vram_bytes) {
        peak_vram_bytes = bytes;
    }
}

void Profiler::print_summary() const {
    if (tokens_generated == 0) return;

    double avg_throughput = (tokens_generated * 1000.0) / total_generation_ms;
    
    double total_ssd_read = 0, total_ram_copy = 0, total_vram_copy = 0;
    double total_wait = 0, total_dequant = 0, total_attn = 0, total_router = 0, total_moe = 0;
    
    for (const auto& pair : layer_profiles) {
        const auto& p = pair.second;
        total_ssd_read += p.ssd_read_ms;
        total_ram_copy += p.ram_copy_ms;
        total_vram_copy += p.vram_copy_ms;
        total_wait += p.wait_ms;
        total_dequant += p.dequant_ms;
        total_attn += p.attention_ms;
        total_router += p.router_ms;
        total_moe += p.moe_ms;
    }
    
    int n_layers = layer_profiles.size();
    double avg_layer_time = n_layers > 0 ? (total_dequant + total_attn + total_router + total_moe) / n_layers : 0.0;
    double avg_load_time = n_layers > 0 ? (total_ssd_read + total_ram_copy + total_vram_copy) / n_layers : 0.0;

    std::cout << "\n========================================\n"
              << "Turbo-LLM Profiler Summary:\n"
              << "========================================\n"
              << "Tokens Generated:   " << tokens_generated << "\n"
              << "Total Time:         " << std::fixed << std::setprecision(2) << total_generation_ms / 1000.0 << " s\n"
              << "Average Throughput: " << std::fixed << std::setprecision(2) << avg_throughput << " tokens/sec\n"
              << "Peak VRAM:          " << std::fixed << std::setprecision(2) << (double)peak_vram_bytes / (1024 * 1024) << " MB\n"
              << "----------------------------------------\n"
              << "Averages per layer execution:\n"
              << "  Execution Time:   " << avg_layer_time << " ms\n"
              << "  Load/Transfer:    " << avg_load_time << " ms\n"
              << "----------------------------------------\n"
              << "Execution Phase Breakdown:\n"
              << "  SSD Read:         " << total_ssd_read << " ms\n"
              << "  RAM Copy:         " << total_ram_copy << " ms\n"
              << "  VRAM Copy:        " << total_vram_copy << " ms\n"
              << "  Swap Wait:        " << total_wait << " ms\n"
              << "  Dequantization:   " << total_dequant << " ms\n"
              << "  Attention:        " << total_attn << " ms\n"
              << "  Router:           " << total_router << " ms\n"
              << "  MoE Experts:      " << total_moe << " ms\n"
              << "  Sampling:         " << total_sampling_ms << " ms\n"
              << "========================================\n";
}

} // namespace turbo
