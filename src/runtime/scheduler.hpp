#pragma once

#include "../model/model.hpp"
#include "memory.hpp"
#include "cuda.hpp"
#include <map>
#include <chrono>

namespace turbo {

class Scheduler {
private:
    std::shared_ptr<MemoryManager> memory_manager;
    Stream_t exec_stream;
    Stream_t copy_stream;

    // Buffer pointers
    void* active_host_buf = nullptr;
    void* active_device_buf = nullptr;
    void* standby_host_buf = nullptr;
    void* standby_device_buf = nullptr;

    bool is_buffer_a_active = true;
    int prefetch_layer_idx = -1;
    int active_layer_idx = -1;
    std::map<int, std::chrono::high_resolution_clock::time_point> ssd_start_times;

public:
    Scheduler(std::shared_ptr<MemoryManager> mem_mgr);
    ~Scheduler();

    Stream_t get_exec_stream() { return exec_stream; }
    Stream_t get_copy_stream() { return copy_stream; }

    void* get_active_device_buffer() { return active_device_buf; }

    // Prefetches the next layer asynchronously
    void prefetch_next(const Model& model, int current_layer_idx);

    // Waits for the prefetch to finish and prepares it for execution
    void ensure_loaded(const Model& model, int layer_idx);

    // Swaps executing and prefetching buffers
    void swap_buffers();
};

} // namespace turbo
