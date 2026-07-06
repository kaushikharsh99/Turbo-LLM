#include "dequant_cache.h"
#include <algorithm>

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
        return it->second;
    }
    return CachedExpert{torch::Tensor(), torch::Tensor(), torch::Tensor()};
}

void DequantCache::put(int64_t layer_id, int64_t expert_id, const CachedExpert& expert) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto key = std::make_tuple(layer_id, expert_id);

    // If already exists, update and return
    if (cache_.find(key) != cache_.end()) {
        cache_[key] = expert;
        return;
    }

    // Evict oldest elements FIFO style if we exceed the limit
    while (cache_.size() >= cache_limit_ && !key_order_.empty()) {
        auto oldest_key = key_order_.front();
        key_order_.erase(key_order_.begin());
        cache_.erase(oldest_key);
    }

    cache_[key] = expert;
    key_order_.push_back(key);
}

void DequantCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
    key_order_.clear();
}
