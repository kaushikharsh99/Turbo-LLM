#include "profiler.hpp"
#include <iostream>
#include <iomanip>
#include <numeric>
#include <fstream>
#include <cmath>

namespace turbo {

void Profiler::start_generation() {
    start_time = std::chrono::high_resolution_clock::now();
    step_profiles.clear();
    current_step_idx = -1;
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

void Profiler::start_step(int step_idx, int token_id) {
    StepProfile step;
    step.step = step_idx;
    step.token_id = token_id;
    step_profiles.push_back(step);
    current_step_idx = step_profiles.size() - 1;
}

void Profiler::end_step() {
    // Currently no-op
}

void Profiler::start_event() {
    event_start = std::chrono::high_resolution_clock::now();
}

double Profiler::end_event() {
    auto event_end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(event_end - event_start).count();
}

void Profiler::record_layer_metric(int layer_idx, const std::string& metric, double val_ms) {
    if (current_step_idx < 0) {
        if (step_profiles.empty()) {
            step_profiles.push_back(StepProfile{});
        }
        current_step_idx = 0;
    }
    
    auto& profile = step_profiles[current_step_idx].layers[layer_idx];
    if (metric == "ssd_read") profile.ssd_read_ms = val_ms;
    else if (metric == "wait") profile.wait_ms = val_ms;
    else if (metric == "vram_copy") profile.vram_copy_ms = val_ms;
    
    else if (metric == "q_proj") profile.q_proj_ms = val_ms;
    else if (metric == "k_proj") profile.k_proj_ms = val_ms;
    else if (metric == "v_proj") profile.v_proj_ms = val_ms;
    else if (metric == "rotary") profile.rotary_ms = val_ms;
    else if (metric == "flash_attn") profile.flash_attn_ms = val_ms;
    else if (metric == "o_proj") profile.o_proj_ms = val_ms;
    
    else if (metric == "router") profile.router_ms = val_ms;
    else if (metric == "topk") profile.topk_ms = val_ms;
    else if (metric == "expert_dequant") profile.expert_dequant_ms = val_ms;
    else if (metric == "expert_gemm1") profile.expert_gemm1_ms = val_ms;
    else if (metric == "activation") profile.activation_ms = val_ms;
    else if (metric == "expert_gemm2") profile.expert_gemm2_ms = val_ms;
    else if (metric == "expert_merge") profile.expert_merge_ms = val_ms;
    
    else if (metric == "residual") profile.residual_ms = val_ms;
    else if (metric == "norm") profile.norm_ms = val_ms;
    else if (metric == "sync") profile.sync_ms = val_ms;
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
    
    double total_ssd_read = 0, total_vram_copy = 0, total_wait = 0;
    double total_q_proj = 0, total_k_proj = 0, total_v_proj = 0;
    double total_rotary = 0, total_flash_attn = 0, total_o_proj = 0;
    double total_router = 0, total_topk = 0, total_expert_dequant = 0;
    double total_expert_gemm1 = 0, total_activation = 0, total_expert_gemm2 = 0;
    double total_expert_merge = 0, total_residual = 0, total_norm = 0, total_sync = 0;
    
    int n_layers_total = 0;
    for (const auto& step : step_profiles) {
        for (const auto& pair : step.layers) {
            const auto& p = pair.second;
            total_ssd_read += p.ssd_read_ms;
            total_vram_copy += p.vram_copy_ms;
            total_wait += p.wait_ms;
            
            total_q_proj += p.q_proj_ms;
            total_k_proj += p.k_proj_ms;
            total_v_proj += p.v_proj_ms;
            total_rotary += p.rotary_ms;
            total_flash_attn += p.flash_attn_ms;
            total_o_proj += p.o_proj_ms;
            
            total_router += p.router_ms;
            total_topk += p.topk_ms;
            total_expert_dequant += p.expert_dequant_ms;
            total_expert_gemm1 += p.expert_gemm1_ms;
            total_activation += p.activation_ms;
            total_expert_gemm2 += p.expert_gemm2_ms;
            total_expert_merge += p.expert_merge_ms;
            
            total_residual += p.residual_ms;
            total_norm += p.norm_ms;
            total_sync += p.sync_ms;
            
            n_layers_total++;
        }
    }
    
    double avg_layer_exec = n_layers_total > 0 ? 
        (total_q_proj + total_k_proj + total_v_proj + total_rotary + total_flash_attn + total_o_proj +
         total_router + total_topk + total_expert_dequant + total_expert_gemm1 + total_activation +
         total_expert_gemm2 + total_expert_merge + total_residual + total_norm + total_sync) / n_layers_total : 0.0;
    
    double avg_load_time = n_layers_total > 0 ? (total_ssd_read + total_vram_copy) / n_layers_total : 0.0;

    std::cout << "\n========================================\n"
              << "Turbo-LLM Profiler Summary:\n"
              << "========================================\n"
              << "Tokens Generated:   " << tokens_generated << "\n"
              << "Total Time:         " << std::fixed << std::setprecision(2) << total_generation_ms / 1000.0 << " s\n"
              << "Average Throughput: " << std::fixed << std::setprecision(2) << avg_throughput << " tokens/sec\n"
              << "Peak VRAM:          " << std::fixed << std::setprecision(2) << (double)peak_vram_bytes / (1024 * 1024) << " MB\n"
              << "----------------------------------------\n"
              << "Averages per layer execution:\n"
              << "  Execution Time:   " << avg_layer_exec << " ms\n"
              << "  Load/Transfer:    " << avg_load_time << " ms\n"
              << "----------------------------------------\n"
              << "Execution Phase Breakdown:\n"
              << "  SSD Read:         " << total_ssd_read << " ms\n"
              << "  VRAM Copy:        " << total_vram_copy << " ms\n"
              << "  Swap Wait:        " << total_wait << " ms\n"
              << "  Attention:\n"
              << "    Q Proj GEMM:    " << total_q_proj << " ms\n"
              << "    K Proj GEMM:    " << total_k_proj << " ms\n"
              << "    V Proj GEMM:    " << total_v_proj << " ms\n"
              << "    Rotary RoPE:    " << total_rotary << " ms\n"
              << "    Attention Core: " << total_flash_attn << " ms\n"
              << "    O Proj GEMM:    " << total_o_proj << " ms\n"
              << "  MoE (Mixture of Experts):\n"
              << "    Router GEMM:    " << total_router << " ms\n"
              << "    TopK Selection: " << total_topk << " ms\n"
              << "    Exp Dequant:    " << total_expert_dequant << " ms\n"
              << "    Exp GEMM 1:     " << total_expert_gemm1 << " ms\n"
              << "    Activation:     " << total_activation << " ms\n"
              << "    Exp GEMM 2:     " << total_expert_gemm2 << " ms\n"
              << "    Exp Merge:      " << total_expert_merge << " ms\n"
              << "  Others:\n"
              << "    Residual:       " << total_residual << " ms\n"
              << "    RMSNorm:        " << total_norm << " ms\n"
              << "    Synchronization: " << total_sync << " ms\n"
              << "  Sampling:         " << total_sampling_ms << " ms\n"
              << "========================================\n";
}

void Profiler::export_json(const std::string& filepath) const {
    std::ofstream f(filepath);
    if (!f.is_open()) {
        std::cerr << "Failed to open export file: " << filepath << std::endl;
        return;
    }
    
    f << "{\n";
    f << "  \"tokens_generated\": " << tokens_generated << ",\n";
    f << "  \"total_time_ms\": " << total_generation_ms << ",\n";
    f << "  \"peak_vram_mb\": " << (double)peak_vram_bytes / (1024.0 * 1024.0) << ",\n";
    f << "  \"steps\": [\n";
    
    for (size_t s = 0; s < step_profiles.size(); ++s) {
        const auto& step = step_profiles[s];
        f << "    {\n";
        f << "      \"step\": " << step.step << ",\n";
        f << "      \"token_id\": " << step.token_id << ",\n";
        f << "      \"layers\": {\n";
        
        size_t l_idx = 0;
        for (const auto& pair : step.layers) {
            int l = pair.first;
            const auto& p = pair.second;
            f << "        \"" << l << "\": {\n";
            f << "          \"pipeline\": {\n";
            f << "            \"ssd_read_ms\": " << p.ssd_read_ms << ",\n";
            f << "            \"vram_copy_ms\": " << p.vram_copy_ms << ",\n";
            f << "            \"wait_ms\": " << p.wait_ms << "\n";
            f << "          },\n";
            f << "          \"attention\": {\n";
            f << "            \"q_proj_ms\": " << p.q_proj_ms << ",\n";
            f << "            \"k_proj_ms\": " << p.k_proj_ms << ",\n";
            f << "            \"v_proj_ms\": " << p.v_proj_ms << ",\n";
            f << "            \"rotary_ms\": " << p.rotary_ms << ",\n";
            f << "            \"flash_attn_ms\": " << p.flash_attn_ms << ",\n";
            f << "            \"o_proj_ms\": " << p.o_proj_ms << "\n";
            f << "          },\n";
            f << "          \"moe\": {\n";
            f << "            \"router_ms\": " << p.router_ms << ",\n";
            f << "            \"topk_ms\": " << p.topk_ms << ",\n";
            f << "            \"expert_dequant_ms\": " << p.expert_dequant_ms << ",\n";
            f << "            \"expert_gemm1_ms\": " << p.expert_gemm1_ms << ",\n";
            f << "            \"activation_ms\": " << p.activation_ms << ",\n";
            f << "            \"expert_gemm2_ms\": " << p.expert_gemm2_ms << ",\n";
            f << "            \"expert_merge_ms\": " << p.expert_merge_ms << "\n";
            f << "          },\n";
            f << "          \"others\": {\n";
            f << "            \"residual_ms\": " << p.residual_ms << ",\n";
            f << "            \"norm_ms\": " << p.norm_ms << ",\n";
            f << "            \"sync_ms\": " << p.sync_ms << "\n";
            f << "          }\n";
            
            f << "        }";
            if (++l_idx < step.layers.size()) f << ",";
            f << "\n";
        }
        f << "      }\n";
        f << "    }";
        if (s + 1 < step_profiles.size()) f << ",";
        f << "\n";
    }
    
    f << "  ]\n";
    f << "}\n";
}

} // namespace turbo
