#pragma once

#include <torch/extension.h>
#include <unordered_map>
#include <tuple>
#include <mutex>
#include <vector>

struct CachedExpert {
    torch::Tensor gate;
    torch::Tensor up;
    torch::Tensor down;
};

struct TupleHash {
    template <class T1, class T2>
    std::size_t operator()(const std::tuple<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(std::get<0>(p));
        auto h2 = std::hash<T2>{}(std::get<1>(p));
        return h1 ^ (h2 << 1);
    }
};

class DequantCache {
private:
    std::unordered_map<std::tuple<int64_t, int64_t>, CachedExpert, TupleHash> cache_;
    std::vector<std::tuple<int64_t, int64_t>> key_order_; // Track FIFO insertion order
    std::mutex mutex_;
    size_t cache_limit_ = 16; // Safely scale down default limit

public:
    DequantCache() = default;
    
    void set_limit(size_t limit);
    bool has(int64_t layer_id, int64_t expert_id);
    CachedExpert get(int64_t layer_id, int64_t expert_id);
    void put(int64_t layer_id, int64_t expert_id, const CachedExpert& expert);
    void clear();
};

extern DequantCache g_dequant_cache;
