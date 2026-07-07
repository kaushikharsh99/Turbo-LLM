#include "kv_cache.hpp"
#include "cuda.hpp"
#include <cstring>
#include <iostream>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace turbo {

KVCache::KVCache(int layers, int kv_heads, int dim, int seq_len, void* storage)
    : num_layers(layers), num_kv_heads(kv_heads), head_dim(dim), max_seq_len(seq_len), device_storage(storage) {}

KVCache::~KVCache() {
    // Storage is managed and freed by MemoryManager
}

void KVCache::update(int layer_idx, const float* qkv_proj_out, int q_dim, int k_dim, int v_dim, int qkv_output_size, int num_tokens, int pos, void* stream) {
    if (!device_storage) return;

#ifdef USE_CUDA
    if (stream) {
        float* d_keys = reinterpret_cast<float*>(get_gpu_pointer(get_layer_keys(layer_idx)));
        float* d_values = reinterpret_cast<float*>(get_gpu_pointer(get_layer_values(layer_idx)));
        const float* d_qkv = reinterpret_cast<const float*>(get_gpu_pointer(qkv_proj_out));
        
        if (d_keys && d_values && d_qkv) {
            cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
            for (int i = 0; i < num_tokens; ++i) {
                // Copy key from GPU projection buffer to GPU key cache
                const float* src_key = d_qkv + i * qkv_output_size + q_dim;
                float* dst_key = d_keys + (pos + i) * k_dim;
                cudaMemcpyAsync(dst_key, src_key, k_dim * sizeof(float), cudaMemcpyDeviceToDevice, cuda_stream);
                
                // Copy value from GPU projection buffer to GPU value cache
                const float* src_val = d_qkv + i * qkv_output_size + q_dim + k_dim;
                float* dst_val = d_values + (pos + i) * v_dim;
                cudaMemcpyAsync(dst_val, src_val, v_dim * sizeof(float), cudaMemcpyDeviceToDevice, cuda_stream);
            }
        }
    }
#endif

    // Fallback Host CPU copy (and for validation/unit-tests)
    float* h_keys = reinterpret_cast<float*>(get_layer_keys(layer_idx));
    float* h_values = reinterpret_cast<float*>(get_layer_values(layer_idx));
    for (int i = 0; i < num_tokens; ++i) {
        const float* src_key = qkv_proj_out + i * qkv_output_size + q_dim;
        float* dst_key = h_keys + (pos + i) * k_dim;
        std::memcpy(dst_key, src_key, k_dim * sizeof(float));
        
        const float* src_val = qkv_proj_out + i * qkv_output_size + q_dim + k_dim;
        float* dst_val = h_values + (pos + i) * v_dim;
        std::memcpy(dst_val, src_val, v_dim * sizeof(float));
    }
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
