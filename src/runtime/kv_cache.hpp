#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>

namespace turbo {

class KVCache {
private:
    int num_layers;
    int num_kv_heads;
    int head_dim;
    int max_seq_len;
    void* device_storage = nullptr; // VRAM storage managed by MemoryManager
    
public:
    KVCache(int layers, int kv_heads, int dim, int seq_len, void* storage);
    ~KVCache();

    void update(int layer_idx, const float* qkv_proj_out, int q_dim, int k_dim, int v_dim, int qkv_output_size, int num_tokens, int pos, void* stream);
    
    void* get_layer_keys(int layer_idx);
    void* get_layer_values(int layer_idx);
    
    void clear();
};

} // namespace turbo
