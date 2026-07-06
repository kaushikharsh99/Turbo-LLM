#include "scheduler.hpp"
#include "../profiler.hpp"
#include <iostream>
#include <cstdint>
#include <algorithm>

namespace turbo {

Scheduler::Scheduler(std::shared_ptr<MemoryManager> mem_mgr)
    : memory_manager(mem_mgr) {
    exec_stream = cuda_stream_create();
    copy_stream = cuda_stream_create();

    // Assign initial buffer mapping
    active_host_buf = memory_manager->get_host_buffer_a();
    active_device_buf = memory_manager->get_device_buffer_a();
    standby_host_buf = memory_manager->get_host_buffer_b();
    standby_device_buf = memory_manager->get_device_buffer_b();
}

Scheduler::~Scheduler() {
    cuda_stream_destroy(exec_stream);
    cuda_stream_destroy(copy_stream);
}

void Scheduler::ensure_loaded(const Model& model, int layer_idx) {
    auto& profiler = Profiler::get();

    if (layer_idx == 0) {
        // 1. Load Layer 0 to active host buffer (SSD -> RAM A)
        profiler.start_event();
        ssd_start_times[0] = std::chrono::high_resolution_clock::now();
        memory_manager->async_load_layer_to_pinned_ram(model.layers[0], active_host_buf);
        memory_manager->wait_disk_load();
        double ssd_time = profiler.end_event();
        profiler.record_layer_metric(0, "ssd_read", ssd_time);
        profiler.record_layer_metric(0, "wait", ssd_time); // First layer wait is the read time

        // 2. Copy Layer 0 to active device buffer (RAM A -> VRAM A)
        profiler.start_event();
        memory_manager->async_copy_to_device(active_host_buf, active_device_buf, memory_manager->get_layer_size(), copy_stream);
        memory_manager->wait_gpu_copy(copy_stream);
        profiler.record_layer_metric(0, "vram_copy", profiler.end_event());

        // 3. Start prefetching Layer 1 to standby host buffer (SSD -> RAM B)
        if (model.config.num_layers > 1) {
            ssd_start_times[1] = std::chrono::high_resolution_clock::now();
            memory_manager->async_load_layer_to_pinned_ram(model.layers[1], standby_host_buf);
        }
    }
}

void Scheduler::prefetch_next(const Model& model, int current_layer_idx) {
    int next_layer = current_layer_idx + 1;
    if (next_layer >= model.config.num_layers) {
        prefetch_layer_idx = -1;
        return; // No more layers to prefetch
    }

    auto& profiler = Profiler::get();

    // 1. Wait for disk load of next_layer to finish (SSD -> RAM B)
    profiler.start_event();
    auto wait_start = std::chrono::high_resolution_clock::now();
    memory_manager->wait_disk_load();
    auto wait_end = std::chrono::high_resolution_clock::now();
    double wait_time = std::chrono::duration<double, std::milli>(wait_end - wait_start).count();
    double ssd_read_time = std::chrono::duration<double, std::milli>(wait_end - ssd_start_times[next_layer]).count();
    
    profiler.record_layer_metric(next_layer, "wait", wait_time);
    profiler.record_layer_metric(next_layer, "ssd_read", ssd_read_time);

    // 2. Launch async copy of next_layer from standby host to standby device buffer
    prefetch_layer_idx = next_layer;
    memory_manager->async_copy_to_device(standby_host_buf, standby_device_buf, memory_manager->get_layer_size(), copy_stream);

    // 3. Launch SSD read of layer next_layer + 1 into active host buffer (which is now idle)
    int prefetch_layer = next_layer + 1;
    if (prefetch_layer < model.config.num_layers) {
        ssd_start_times[prefetch_layer] = std::chrono::high_resolution_clock::now();
        memory_manager->async_load_layer_to_pinned_ram(model.layers[prefetch_layer], active_host_buf);
    }
}

void Scheduler::swap_buffers() {
    auto& profiler = Profiler::get();
    
    if (prefetch_layer_idx != -1) {
        // Wait for the copy stream (RAM -> VRAM transfer of next layer) to finish
        profiler.start_event();
        cuda_stream_synchronize(copy_stream);
        profiler.record_layer_metric(prefetch_layer_idx, "vram_copy", profiler.end_event());
        prefetch_layer_idx = -1;
    }

    // Wait for execution stream to finish current layer
    cuda_stream_synchronize(exec_stream);
    
    // Swap pointers
    std::swap(active_host_buf, standby_host_buf);
    std::swap(active_device_buf, standby_device_buf);
    is_buffer_a_active = !is_buffer_a_active;
}

} // namespace turbo
