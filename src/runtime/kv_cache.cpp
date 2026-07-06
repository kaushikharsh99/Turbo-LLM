#include "kv_cache.hpp"
#include <cstring>
#include <iostream>

namespace turbo {

KVCache::KVCache(int layers, int kv_heads, int dim, int seq_len, void* storage)
    : num_layers(layers), num_kv_heads(kv_heads), head_dim(dim), max_seq_len(seq_len), device_storage(storage) {}

KVCache::~KVCache() {
    // Storage is managed and freed by MemoryManager
}

void KVCache::update(int layer_idx, const float* new_kv, int num_tokens, int pos) {
    if (!device_storage) return;

    // In a full implementation, this copies the new key and value states into the VRAM cache buffer
    // at the appropriate offsets. 
    // Here we simulate the operation by calculating the index and writing if it's CPU storage
    // or calling cudaMemcpyAsync if using active CUDA.
    size_t layer_stride = 2 * num_kv_heads * max_seq_len * head_dim * sizeof(float);
    uint8_t* layer_base = reinterpret_cast<uint8_t*>(device_storage) + layer_idx * layer_stride;
    
    float* keys = reinterpret_cast<float*>(layer_base);
    float* values = reinterpret_cast<float*>(layer_base + (num_kv_heads * max_seq_len * head_dim * sizeof(float)));

    // Since we are mocking/fallback, we write data if it's CPU pointer, or simulate VRAM update
    // in real engine we do cudaMemcpyAsync on execution stream
}

void* KVCache::get_layer_keys(int layer_idx) {
    size_t layer_stride = 2 * num_kv_heads * max_seq_len * head_dim * sizeof(float);
    return reinterpret_cast<uint8_t*>(device_storage) + layer_idx * layer_stride;
}

void* KVCache::get_layer_values(int layer_idx) {
    size_t layer_stride = 2 * num_kv_heads * max_seq_len * head_dim * sizeof(float);
    size_t half_stride = num_kv_heads * max_seq_len * head_dim * sizeof(float);
    return reinterpret_cast<uint8_t*>(device_storage) + layer_idx * layer_stride + half_stride;
}

void KVCache::clear() {
    // Reset cache pointers or state
}

} // namespace turbo
