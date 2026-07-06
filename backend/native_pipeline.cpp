#include "native_pipeline.h"
#include <iostream>

NativePipelineManager g_native_pipeline;

NativePipelineManager::NativePipelineManager() : ram_cache_(1024) {}

NativePipelineManager::~NativePipelineManager() {
    stop();
}

void NativePipelineManager::start() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (loader_thread_.joinable()) return;
    stop_flag_ = false;
    loader_thread_ = std::thread(&NativePipelineManager::worker_loop, this);
}

void NativePipelineManager::stop() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (stop_flag_) return;
        stop_flag_ = true;
    }
    cv_.notify_all();
    if (loader_thread_.joinable()) {
        loader_thread_.join();
    }
    ram_cache_.clear();
}

void NativePipelineManager::submit_prefetch(int layer_id, const std::vector<int>& expert_ids) {
    std::vector<int> exps = expert_ids;
    if (exps.empty()) {
        auto it = last_layer_experts_.find(layer_id);
        if (it != last_layer_experts_.end()) {
            exps = it->second;
        } else {
            for (int i = 0; i < 8; ++i) exps.push_back(i);
        }
    }

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        prefetch_queue_.push({layer_id, exps});
    }
    cv_.notify_one();
}

void NativePipelineManager::record_experts(int layer_id, const std::vector<int>& expert_ids) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    last_layer_experts_[layer_id] = expert_ids;
}

bool NativePipelineManager::get_expert(int layer_id, int expert_id, NativeExpertWeight& out_expert) {
    return ram_cache_.get(layer_id, expert_id, out_expert);
}

void NativePipelineManager::worker_loop() {
    while (true) {
        std::pair<int, std::vector<int>> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this] { return stop_flag_ || !prefetch_queue_.empty(); });

            if (stop_flag_ && prefetch_queue_.empty()) break;

            task = prefetch_queue_.front();
            prefetch_queue_.pop();
        }

        int layer_id = task.first;
        const auto& expert_ids = task.second;

        for (int exp_id : expert_ids) {
            if (ram_cache_.contains(layer_id, exp_id)) continue;

            // Load raw FP8 tensors directly from C++ native weight index
            try {
                std::string prefix = "model.language_model.layers." + std::to_string(layer_id) + ".mlp.experts." + std::to_string(exp_id) + ".";
                
                NativeExpertWeight exp;
                if (g_native_weight_index.has_tensor(prefix + "gate_proj.weight")) {
                    exp.gate_fp8 = g_native_weight_index.load_tensor(prefix + "gate_proj.weight", torch::kCPU);
                    exp.gate_scale = g_native_weight_index.load_tensor(prefix + "gate_proj.weight_scale_inv", torch::kCPU);
                    exp.up_fp8 = g_native_weight_index.load_tensor(prefix + "up_proj.weight", torch::kCPU);
                    exp.up_scale = g_native_weight_index.load_tensor(prefix + "up_proj.weight_scale_inv", torch::kCPU);
                    exp.down_fp8 = g_native_weight_index.load_tensor(prefix + "down_proj.weight", torch::kCPU);
                    exp.down_scale = g_native_weight_index.load_tensor(prefix + "down_proj.weight_scale_inv", torch::kCPU);

                    ram_cache_.put(layer_id, exp_id, exp);
                }
            } catch (...) {
                // Ignore missing tensor errors during speculative prefetch
            }
        }
    }
}
