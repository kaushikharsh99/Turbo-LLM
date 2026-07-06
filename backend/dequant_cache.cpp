#include "dequant_cache.h"
#include <algorithm>
#include <limits>

DequantCache g_dequant_cache;

void DequantCache::set_limit(size_t limit) {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_limit_ = limit;
}

bool DequantCache::has(int64_t layer_id, int64_t expert_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.find({layer_id, expert_id}) != cache_.end();
}

CachedExpert DequantCache::get(int64_t layer_id, int64_t expert_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto key = std::make_tuple(layer_id, expert_id);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        it->second.hits += 1; // Increment hits on access
        return it->second;
    }
    return CachedExpert{torch::Tensor(), torch::Tensor(), torch::Tensor(), 0};
}

void DequantCache::put(int64_t layer_id, int64_t expert_id, const CachedExpert& expert) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto key = std::make_tuple(layer_id, expert_id);

    // If already exists, update and return
    if (cache_.find(key) != cache_.end()) {
        cache_[key].gate = expert.gate;
        cache_[key].up = expert.up;
        cache_[key].down = expert.down;
        return;
    }

    // Evict lowest hit expert (using FIFO insertion order as a tie-breaker)
    if (cache_.size() >= cache_limit_) {
        int64_t min_hits = std::numeric_limits<int64_t>::max();
        std::tuple<int64_t, int64_t> key_to_evict;
        bool found = false;

        // Iterate in FIFO order to find minimum hits (gives oldest among same hit count)
        for (const auto& k : key_order_) {
            auto it = cache_.find(k);
            if (it != cache_.end() && it->second.hits < min_hits) {
                min_hits = it->second.hits;
                key_to_evict = k;
                found = true;
            }
        }

        if (found) {
            key_order_.erase(std::remove(key_order_.begin(), key_order_.end(), key_to_evict), key_order_.end());
            cache_.erase(key_to_evict);
        }
    }

    CachedExpert new_expert = expert;
    new_expert.hits = 1; // Initialize hits count
    cache_[key] = new_expert;
    key_order_.push_back(key);
}

size_t DequantCache::size() {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.size();
}

void DequantCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
    key_order_.clear();
}
