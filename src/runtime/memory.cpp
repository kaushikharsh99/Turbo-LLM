#include "memory.hpp"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cstdint>

namespace turbo {

MemoryManager::MemoryManager(std::shared_ptr<GGUFLoader> gguf_loader)
    : loader(gguf_loader) {}

MemoryManager::~MemoryManager() {
    wait_disk_load();

    if (host_buffer_a) cuda_free_pinned(host_buffer_a);
    if (host_buffer_b) cuda_free_pinned(host_buffer_b);
    if (device_buffer_a) cuda_free(device_buffer_a);
    if (device_buffer_b) cuda_free(device_buffer_b);
    if (device_kv_cache) cuda_free(device_kv_cache);
    if (device_workspace) cuda_free(device_workspace);
    if (device_io_buffer) cuda_free(device_io_buffer);
}

void MemoryManager::initialize(const Model& model) {
    // 1. Calculate maximum layer size
    size_t max_layer_bytes = 0;
    for (const auto& layer : model.layers) {
        size_t layer_bytes = 0;
        layer_bytes += layer.input_layernorm.size_bytes;
        layer_bytes += layer.post_attention_layernorm.size_bytes;
        if (layer.k_proj.name == layer.q_proj.name) {
            layer_bytes += layer.q_proj.size_bytes; // merged QKV
        } else {
            layer_bytes += layer.q_proj.size_bytes;
            layer_bytes += layer.k_proj.size_bytes;
            layer_bytes += layer.v_proj.size_bytes;
        }
        layer_bytes += layer.o_proj.size_bytes;
        layer_bytes += layer.gate_inp.size_bytes;
        
        for (const auto& t : layer.expert_gate) layer_bytes += t.size_bytes;
        for (const auto& t : layer.expert_up) layer_bytes += t.size_bytes;
        for (const auto& t : layer.expert_down) layer_bytes += t.size_bytes;
        
        if (layer.has_shared_expert) {
            layer_bytes += layer.shared_expert_gate.size_bytes;
            layer_bytes += layer.shared_expert_up.size_bytes;
            layer_bytes += layer.shared_expert_down.size_bytes;
        }
        max_layer_bytes = std::max(max_layer_bytes, layer_bytes);
    }

    host_buffer_size = max_layer_bytes;
    device_buffer_size = max_layer_bytes;

    std::cout << "Allocating Double Buffers. Host layer size: " 
              << (host_buffer_size / (1024 * 1024)) << " MB, Device layer size: "
              << (device_buffer_size / (1024 * 1024)) << " MB\n";

    // 2. Allocate pinned host buffers
    host_buffer_a = cuda_malloc_pinned(host_buffer_size);
    host_buffer_b = cuda_malloc_pinned(host_buffer_size);
    if (!host_buffer_a || !host_buffer_b) {
        throw std::runtime_error("Failed to allocate pinned host memory");
    }

    // 3. Allocate device layer buffers
    device_buffer_a = cuda_malloc(device_buffer_size);
    device_buffer_b = cuda_malloc(device_buffer_size);
    if (!device_buffer_a || !device_buffer_b) {
        throw std::runtime_error("Failed to allocate VRAM double buffers");
    }

    // 4. Allocate KV cache (e.g. 41 layers, 2 heads, seq_len 4096, dim 128)
    size_t kv_cache_bytes = model.config.num_layers * 2 * model.config.num_kv_heads * 4096 * 128 * sizeof(float);
    device_kv_cache = cuda_malloc(kv_cache_bytes);
    if (!device_kv_cache) {
        throw std::runtime_error("Failed to allocate VRAM KV Cache");
    }
    
    // 5. Allocate workspace & IO buffers
    size_t workspace_bytes = 512 * 1024 * 1024; // 512 MB
    device_workspace = cuda_malloc(workspace_bytes);
    device_io_buffer = cuda_malloc(workspace_bytes);
    if (!device_workspace || !device_io_buffer) {
        throw std::runtime_error("Failed to allocate VRAM workspaces");
    }

    std::cout << "Memory Initialization Successful. Total GPU allocations: " 
              << ((device_buffer_size * 2 + kv_cache_bytes + workspace_bytes * 2) / (1024 * 1024)) << " MB\n";
}

void MemoryManager::async_load_layer_to_pinned_ram(const Layer& layer, void* host_target_buffer) {
    wait_disk_load();
    
    std::lock_guard<std::mutex> lock(disk_mutex);
    disk_loading_in_progress = true;
    
    disk_thread = std::thread([this, layer, host_target_buffer]() {
        uint8_t* ptr = reinterpret_cast<uint8_t*>(host_target_buffer);
        size_t current_offset = 0;
        
        auto load_one = [&](const GGUFTensor& t) {
            if (t.name.empty()) return;
            loader->read_tensor_data(t, ptr + current_offset);
            current_offset += t.size_bytes;
        };

        load_one(layer.input_layernorm);
        load_one(layer.post_attention_layernorm);
        if (layer.k_proj.name == layer.q_proj.name) {
            load_one(layer.q_proj);
        } else {
            load_one(layer.q_proj);
            load_one(layer.k_proj);
            load_one(layer.v_proj);
        }
        load_one(layer.o_proj);
        load_one(layer.gate_inp);

        for (const auto& t : layer.expert_gate) load_one(t);
        for (const auto& t : layer.expert_up) load_one(t);
        for (const auto& t : layer.expert_down) load_one(t);

        if (layer.has_shared_expert) {
            load_one(layer.shared_expert_gate);
            load_one(layer.shared_expert_up);
            load_one(layer.shared_expert_down);
        }
        
        std::lock_guard<std::mutex> t_lock(disk_mutex);
        disk_loading_in_progress = false;
    });
}

void MemoryManager::wait_disk_load() {
    if (disk_thread.joinable()) {
        disk_thread.join();
    }
}

void MemoryManager::async_copy_to_device(const void* host_source_buffer, void* device_target_buffer, size_t size_bytes, Stream_t stream) {
    cuda_memcpy_async(device_target_buffer, host_source_buffer, size_bytes, stream);
}

void MemoryManager::wait_gpu_copy(Stream_t stream) {
    cuda_stream_synchronize(stream);
}

} // namespace turbo
