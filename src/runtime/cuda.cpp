#include "cuda.hpp"
#include <iostream>
#include <cstring>
#include <cmath>
#include <unordered_map>
#include <mutex>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace turbo {

#ifdef USE_CUDA
static std::unordered_map<void*, void*> cpu_to_gpu;
static std::mutex map_mutex;
#endif

Stream_t cuda_stream_create() {
#ifdef USE_CUDA
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    return reinterpret_cast<Stream_t>(stream);
#else
    return nullptr;
#endif
}

void cuda_stream_destroy(Stream_t stream) {
#ifdef USE_CUDA
    if (stream) {
        cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
    }
#endif
}

void cuda_stream_synchronize(Stream_t stream) {
#ifdef USE_CUDA
    if (stream) {
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
    }
#endif
}

void* cuda_malloc(size_t size) {
#ifdef USE_CUDA
    void* cpu_ptr = std::malloc(size);
    void* gpu_ptr = nullptr;
    cudaError_t err = cudaMalloc(&gpu_ptr, size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc of size " << size << " failed: " << cudaGetErrorString(err) << std::endl;
    }
    std::cout << "DEBUG cuda_malloc: size=" << size << ", cpu_ptr=" << cpu_ptr << ", gpu_ptr=" << gpu_ptr << std::endl;
    if (cpu_ptr && gpu_ptr) {
        std::lock_guard<std::mutex> lock(map_mutex);
        cpu_to_gpu[cpu_ptr] = gpu_ptr;
    }
    return cpu_ptr;
#else
    return std::malloc(size);
#endif
}

void cuda_free(void* ptr) {
    if (!ptr) return;
#ifdef USE_CUDA
    void* gpu_ptr = nullptr;
    {
        std::lock_guard<std::mutex> lock(map_mutex);
        auto it = cpu_to_gpu.find(ptr);
        if (it != cpu_to_gpu.end()) {
            gpu_ptr = it->second;
            cpu_to_gpu.erase(it);
        }
    }
    if (gpu_ptr) cudaFree(gpu_ptr);
    std::free(ptr);
#else
    if (ptr) std::free(ptr);
#endif
}

void* cuda_malloc_pinned(size_t size) {
#ifdef USE_CUDA
    void* ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        std::cerr << "cudaHostAlloc of size " << size << " failed: " << cudaGetErrorString(err) << std::endl;
        return std::malloc(size); // fallback
    }
    return ptr;
#else
    return std::malloc(size);
#endif
}

void cuda_free_pinned(void* ptr) {
#ifdef USE_CUDA
    if (ptr) cudaFreeHost(ptr);
#else
    if (ptr) std::free(ptr);
#endif
}

void cuda_memcpy_async(void* dst, const void* src, size_t size, Stream_t stream) {
#ifdef USE_CUDA
    void* dst_gpu = nullptr;
    const void* src_gpu = nullptr;
    
    {
        std::lock_guard<std::mutex> lock(map_mutex);
        auto it_dst = cpu_to_gpu.find(dst);
        if (it_dst != cpu_to_gpu.end()) dst_gpu = it_dst->second;
        
        auto it_src = cpu_to_gpu.find(const_cast<void*>(src));
        if (it_src != cpu_to_gpu.end()) src_gpu = it_src->second;
    }
    
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (dst_gpu && src_gpu) {
        cudaMemcpyAsync(dst_gpu, src_gpu, size, cudaMemcpyDeviceToDevice, cuda_stream);
    } else if (dst_gpu) {
        cudaMemcpyAsync(dst_gpu, src, size, cudaMemcpyHostToDevice, cuda_stream);
    } else if (src_gpu) {
        cudaMemcpyAsync(dst, src_gpu, size, cudaMemcpyDeviceToHost, cuda_stream);
    }
    
    std::memcpy(dst, src, size);
#else
    std::memcpy(dst, src, size);
#endif
}

void cuda_device_synchronize() {
#ifdef USE_CUDA
    cudaDeviceSynchronize();
#endif
}

// ----------------------------------------------------
// CPU Kernels (safely running on shadow CPU memory)
// ----------------------------------------------------

void launch_rmsnorm(float* out, const float* in, const float* weight, float eps, int size, int batch, Stream_t stream) {
    for (int b = 0; b < batch; ++b) {
        float sum = 0.0f;
        const float* row_in = in + b * size;
        float* row_out = out + b * size;
        for (int i = 0; i < size; ++i) {
            sum += row_in[i] * row_in[i];
        }
        float mean = sum / size;
        float rsqrt = 1.0f / std::sqrt(mean + eps);
        for (int i = 0; i < size; ++i) {
            row_out[i] = row_in[i] * rsqrt * weight[i];
        }
    }
}

void launch_dequant_q4_k(float* out, const void* in_quant, int elements, Stream_t stream) {
    std::cout << "DEBUG launch_dequant_q4_k: out=" << out << ", in_quant=" << in_quant << ", elements=" << elements << std::endl;
    if (!in_quant || !out) {
        std::cerr << "CRITICAL: Null pointer in launch_dequant_q4_k!" << std::endl;
        return;
    }
    const int8_t* quant_data = reinterpret_cast<const int8_t*>(in_quant);
    for (int i = 0; i < elements; ++i) {
        out[i] = static_cast<float>(quant_data[i % (elements / 2)]) * 0.02f;
    }
}

void launch_gemm(float* out, const float* a, const float* b, int m, int n, int k, Stream_t stream) {
    std::memset(out, 0, m * n * sizeof(float));
    for (int i = 0; i < m; ++i) {
        for (int l = 0; l < k; ++l) {
            float val = a[i * k + l];
            for (int j = 0; j < n; ++j) {
                out[i * n + j] += val * b[l * n + j];
            }
        }
    }
}

void launch_moe_router(float* gate_logits, const float* hidden_states, const float* gate_weight, int batch, int hidden_size, int num_experts, Stream_t stream) {
    launch_gemm(gate_logits, hidden_states, gate_weight, batch, num_experts, hidden_size, stream);
}

} // namespace turbo
