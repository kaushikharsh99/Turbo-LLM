#pragma once

#include <torch/extension.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include "safetensors_reader.h"

struct NativeExpertWeight {
    torch::Tensor gate_fp8;
    torch::Tensor gate_scale;
    torch::Tensor up_fp8;
    torch::Tensor up_scale;
    torch::Tensor down_fp8;
    torch::Tensor down_scale;
};

class NativeRamCache {
private:
    size_t capacity_;
    std::unordered_map<int64_t, NativeExpertWeight> cache_;
    std::mutex mutex_;

    int64_t make_key(int layer_id, int expert_id) const {
        return (static_cast<int64_t>(layer_id) << 32) | static_cast<int64_t>(expert_id);
    }

public:
    explicit NativeRamCache(size_t capacity = 512) : capacity_(capacity) {}

    bool contains(int layer_id, int expert_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.find(make_key(layer_id, expert_id)) != cache_.end();
    }

    void put(int layer_id, int expert_id, const NativeExpertWeight& expert) {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_[make_key(layer_id, expert_id)] = expert;
    }

    bool get(int layer_id, int expert_id, NativeExpertWeight& out_expert) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(make_key(layer_id, expert_id));
        if (it != cache_.end()) {
            out_expert = it->second;
            return true;
        }
        return false;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
    }
};

class NativePipelineManager {
private:
    NativeRamCache ram_cache_;
    std::thread loader_thread_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::queue<std::pair<int, std::vector<int>>> prefetch_queue_;
    bool stop_flag_ = false;
    std::unordered_map<int, std::vector<int>> last_layer_experts_;

    void worker_loop();

public:
    NativePipelineManager();
    ~NativePipelineManager();

    void start();
    void stop();
    void submit_prefetch(int layer_id, const std::vector<int>& expert_ids = {});
    bool get_expert(int layer_id, int expert_id, NativeExpertWeight& out_expert);
    void record_experts(int layer_id, const std::vector<int>& expert_ids);
};

extern NativePipelineManager g_native_pipeline;
