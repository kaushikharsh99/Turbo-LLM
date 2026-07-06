#pragma once

#include "../model/model.hpp"
#include "cuda.hpp"
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <future>
#include <mutex>

namespace turbo {

class MemoryManager {
private:
    size_t max_vram = 0;
    size_t max_ram = 0;
    
    // Host buffers (pinned host memory)
    void* host_buffer_a = nullptr;
    void* host_buffer_b = nullptr;
    size_t host_buffer_size = 0;

    // Device buffers (VRAM)
    void* device_buffer_a = nullptr;
    void* device_buffer_b = nullptr;
    size_t device_buffer_size = 0;

    void* device_kv_cache = nullptr;
    void* device_workspace = nullptr;
    void* device_io_buffer = nullptr;
    
    // Loader pointer
    std::shared_ptr<GGUFLoader> loader;

    // Async thread for disk loading
    std::thread disk_thread;
    std::mutex disk_mutex;
    bool disk_loading_in_progress = false;

public:
    MemoryManager(std::shared_ptr<GGUFLoader> gguf_loader);
    ~MemoryManager();

    void initialize(const Model& model);

    // Getters for device pointers
    void* get_device_buffer_a() { return device_buffer_a; }
    void* get_device_buffer_b() { return device_buffer_b; }
    void* get_device_kv_cache() { return device_kv_cache; }
    void* get_device_workspace() { return device_workspace; }
    void* get_device_io_buffer() { return device_io_buffer; }
    
    void* get_host_buffer_a() { return host_buffer_a; }
    void* get_host_buffer_b() { return host_buffer_b; }
    
    size_t get_layer_size() const { return host_buffer_size; }

    // Async loading from disk to pinned RAM
    void async_load_layer_to_pinned_ram(const Layer& layer, void* host_target_buffer);
    void wait_disk_load();

    // Async copying from pinned RAM to VRAM
    void async_copy_to_device(const void* host_source_buffer, void* device_target_buffer, size_t size_bytes, Stream_t stream);
    void wait_gpu_copy(Stream_t stream);
};

} // namespace turbo
