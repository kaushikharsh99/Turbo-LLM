#pragma once

#include <cstddef>

namespace turbo {

using Stream_t = void*;

Stream_t cuda_stream_create();
void cuda_stream_destroy(Stream_t stream);
void cuda_stream_synchronize(Stream_t stream);

void* cuda_malloc(size_t size);
void cuda_free(void* ptr);

void* cuda_malloc_pinned(size_t size);
void cuda_free_pinned(void* ptr);

void cuda_memcpy_async(void* dst, const void* src, size_t size, Stream_t stream);
void cuda_device_synchronize();

// GPU execution kernels
void launch_rmsnorm(float* out, const float* in, const float* weight, float eps, int size, int batch, Stream_t stream);
void launch_dequant_q4_k(float* out, const void* in_quant, int elements, Stream_t stream);
void launch_gemm(float* out, const float* a, const float* b, int m, int n, int k, Stream_t stream);
void launch_moe_router(float* gate_logits, const float* hidden_states, const float* gate_weight, int batch, int hidden_size, int num_experts, Stream_t stream);

} // namespace turbo
