#include "runtime/cuda.hpp"
#include "runtime/kv_cache.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstring>
#include <cstdlib>

using namespace turbo;

// Helper to check numerical closeness
bool is_close(float a, float b, float tolerance = 1e-3f) {
    float diff = std::abs(a - b);
    if (diff < tolerance) return true;
    // Relative tolerance for larger values
    return diff / std::max(std::abs(a), std::abs(b)) < tolerance;
}

void test_rmsnorm() {
    std::cout << "Testing RMSNorm... ";
    int size = 1024;
    int batch = 4;
    std::vector<float> in(batch * size);
    std::vector<float> weight(size, 1.0f);
    for (int i = 0; i < batch * size; ++i) in[i] = static_cast<float>(i % 10) * 0.1f;
    
    std::vector<float> out_ref(batch * size, 0.0f);
    std::vector<float> out_test(batch * size, 0.0f);
    
    // CPU reference
    float eps = 1e-6f;
    for (int b = 0; b < batch; ++b) {
        float sum = 0.0f;
        for (int i = 0; i < size; ++i) sum += in[b * size + i] * in[b * size + i];
        float rsqrt = 1.0f / std::sqrt(sum / size + eps);
        for (int i = 0; i < size; ++i) out_ref[b * size + i] = in[b * size + i] * rsqrt * weight[i];
    }
    
    // Run test
    Stream_t stream = cuda_stream_create();
    float* d_in = reinterpret_cast<float*>(cuda_malloc(batch * size * sizeof(float)));
    float* d_out = reinterpret_cast<float*>(cuda_malloc(batch * size * sizeof(float)));
    float* d_weight = reinterpret_cast<float*>(cuda_malloc(size * sizeof(float)));
    
    std::memcpy(d_in, in.data(), batch * size * sizeof(float));
    std::memcpy(d_weight, weight.data(), size * sizeof(float));
    
    sync_host_to_device(d_in, batch * size * sizeof(float), stream);
    sync_host_to_device(d_weight, size * sizeof(float), stream);
    
    launch_rmsnorm(d_out, d_in, d_weight, eps, size, batch, stream);
    
    cuda_stream_synchronize(stream);
    cuda_memcpy_async(out_test.data(), d_out, batch * size * sizeof(float), stream);
    cuda_stream_synchronize(stream);
    
    for (int i = 0; i < batch * size; ++i) {
        assert(is_close(out_ref[i], out_test[i]) && "RMSNorm output mismatch");
    }
    
    cuda_free(d_in);
    cuda_free(d_out);
    cuda_free(d_weight);
    cuda_stream_destroy(stream);
    std::cout << "PASSED!\n";
}

void test_dequantize() {
    std::cout << "Testing GPU NVRTC Dequantization... ";
    int elements = 2048;
    std::vector<uint16_t> in_quant(elements);
    for (int i = 0; i < elements; ++i) {
        if (i % 2 == 0) in_quant[i] = 0x3C00; // 1.0f in FP16
        else in_quant[i] = 0x4000; // 2.0f in FP16
    }
    
    std::vector<float> out_ref(elements, 0.0f);
    std::vector<float> out_test(elements, 0.0f);
    
    // CPU reference
    for (int i = 0; i < elements; ++i) {
        out_ref[i] = (i % 2 == 0) ? 1.0f : 2.0f;
    }
    
    // GPU test
    Stream_t stream = cuda_stream_create();
    float* d_out = reinterpret_cast<float*>(cuda_malloc(elements * sizeof(float)));
    void* d_in = cuda_malloc(in_quant.size() * sizeof(uint16_t));
    
    std::memcpy(d_in, in_quant.data(), in_quant.size() * sizeof(uint16_t));
    sync_host_to_device(d_in, in_quant.size() * sizeof(uint16_t), stream);
    
    launch_dequant(d_out, d_in, 1, elements, stream);
    
    cuda_stream_synchronize(stream);
    cuda_memcpy_async(out_test.data(), d_out, elements * sizeof(float), stream);
    cuda_stream_synchronize(stream);
    
    for (int i = 0; i < elements; ++i) {
        assert(is_close(out_ref[i], out_test[i]) && "Dequantization output mismatch");
    }
    
    cuda_free(d_out);
    cuda_free(d_in);
    cuda_stream_destroy(stream);
    std::cout << "PASSED!\n";
}

void test_gemm() {
    std::cout << "Testing cuBLAS GEMM... ";
    int m = 8;
    int n = 16;
    int k = 32;
    std::vector<float> a(m * k);
    std::vector<float> b(k * n);
    for (int i = 0; i < m * k; ++i) a[i] = static_cast<float>(i % 5) * 0.1f;
    for (int i = 0; i < k * n; ++i) b[i] = static_cast<float>(i % 7) * 0.1f;
    
    std::vector<float> out_ref(m * n, 0.0f);
    std::vector<float> out_test(m * n, 0.0f);
    
    // CPU reference
    for (int i = 0; i < m; ++i) {
        for (int l = 0; l < k; ++l) {
            float val = a[i * k + l];
            for (int j = 0; j < n; ++j) {
                out_ref[i * n + j] += val * b[l * n + j];
            }
        }
    }
    
    // GPU test
    Stream_t stream = cuda_stream_create();
    float* d_a = reinterpret_cast<float*>(cuda_malloc(m * k * sizeof(float)));
    float* d_b = reinterpret_cast<float*>(cuda_malloc(k * n * sizeof(float)));
    float* d_out = reinterpret_cast<float*>(cuda_malloc(m * n * sizeof(float)));
    
    std::memcpy(d_a, a.data(), m * k * sizeof(float));
    std::memcpy(d_b, b.data(), k * n * sizeof(float));
    
    sync_host_to_device(d_a, m * k * sizeof(float), stream);
    sync_host_to_device(d_b, k * n * sizeof(float), stream);
    
    launch_gemm(d_out, d_a, d_b, m, n, k, false, stream);
    
    cuda_stream_synchronize(stream);
    cuda_memcpy_async(out_test.data(), d_out, m * n * sizeof(float), stream);
    cuda_stream_synchronize(stream);
    
    for (int i = 0; i < m * n; ++i) {
        assert(is_close(out_ref[i], out_test[i]) && "cuBLAS GEMM output mismatch");
    }
    
    cuda_free(d_a);
    cuda_free(d_b);
    cuda_free(d_out);
    cuda_stream_destroy(stream);
    std::cout << "PASSED!\n";
}

void test_gemm_transposed() {
    std::cout << "Testing cuBLAS GEMM Transposed... ";
    int m = 8;
    int n = 16;
    int k = 32;
    std::vector<float> a(m * k);
    std::vector<float> b(n * k); // shape n x k for transpose_b
    for (int i = 0; i < m * k; ++i) a[i] = static_cast<float>(i % 5) * 0.1f;
    for (int i = 0; i < n * k; ++i) b[i] = static_cast<float>(i % 7) * 0.1f;
    
    std::vector<float> out_ref(m * n, 0.0f);
    std::vector<float> out_test(m * n, 0.0f);
    
    // CPU reference for transpose_b
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += a[i * k + l] * b[j * k + l];
            }
            out_ref[i * n + j] = sum;
        }
    }
    
    // GPU test
    Stream_t stream = cuda_stream_create();
    float* d_a = reinterpret_cast<float*>(cuda_malloc(m * k * sizeof(float)));
    float* d_b = reinterpret_cast<float*>(cuda_malloc(n * k * sizeof(float)));
    float* d_out = reinterpret_cast<float*>(cuda_malloc(m * n * sizeof(float)));
    
    std::memcpy(d_a, a.data(), m * k * sizeof(float));
    std::memcpy(d_b, b.data(), n * k * sizeof(float));
    
    sync_host_to_device(d_a, m * k * sizeof(float), stream);
    sync_host_to_device(d_b, n * k * sizeof(float), stream);
    
    launch_gemm(d_out, d_a, d_b, m, n, k, true, stream);
    
    cuda_stream_synchronize(stream);
    cuda_memcpy_async(out_test.data(), d_out, m * n * sizeof(float), stream);
    cuda_stream_synchronize(stream);
    
    for (int i = 0; i < m * n; ++i) {
        assert(is_close(out_ref[i], out_test[i]) && "Transposed cuBLAS GEMM output mismatch");
    }
    
    cuda_free(d_a);
    cuda_free(d_b);
    cuda_free(d_out);
    cuda_stream_destroy(stream);
    std::cout << "PASSED!\n";
}

void test_activation() {
    std::cout << "Testing GPU MoE Activation (SiLU)... ";
    int elements = 1024;
    std::vector<float> in(elements);
    for (int i = 0; i < elements; ++i) in[i] = static_cast<float>(i - 512) * 0.01f;
    
    std::vector<float> out_ref(elements, 0.0f);
    std::vector<float> out_test(elements, 0.0f);
    
    // CPU reference (SiLU)
    for (int i = 0; i < elements; ++i) {
        float x = in[i];
        out_ref[i] = x / (1.0f + std::exp(-x));
    }
    
    // GPU test
    Stream_t stream = cuda_stream_create();
    float* d_in = reinterpret_cast<float*>(cuda_malloc(elements * sizeof(float)));
    float* d_out = reinterpret_cast<float*>(cuda_malloc(elements * sizeof(float)));
    
    std::memcpy(d_in, in.data(), elements * sizeof(float));
    sync_host_to_device(d_in, elements * sizeof(float), stream);
    
    launch_moe_activation(d_out, d_in, elements, stream);
    
    cuda_stream_synchronize(stream);
    cuda_memcpy_async(out_test.data(), d_out, elements * sizeof(float), stream);
    cuda_stream_synchronize(stream);
    
    for (int i = 0; i < elements; ++i) {
        assert(is_close(out_ref[i], out_test[i]) && "Activation output mismatch");
    }
    
    cuda_free(d_in);
    cuda_free(d_out);
    cuda_stream_destroy(stream);
    std::cout << "PASSED!\n";
}

void test_merge() {
    std::cout << "Testing GPU MoE Merge... ";
    int num_tokens = 4;
    int k_experts = 8;
    int hidden_size = 128;
    
    std::vector<float> exp_out(k_experts * num_tokens * hidden_size);
    std::vector<float> weights(num_tokens * k_experts);
    
    for (size_t i = 0; i < exp_out.size(); ++i) exp_out[i] = static_cast<float>(i % 100) * 0.01f;
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<float>(i % 5) * 0.1f;
    
    std::vector<float> out_ref(num_tokens * hidden_size, 0.0f);
    std::vector<float> out_test(num_tokens * hidden_size, 0.0f);
    
    // CPU reference
    for (int t = 0; t < num_tokens; ++t) {
        for (int i = 0; i < k_experts; ++i) {
            float weight = weights[t * k_experts + i];
            for (int h = 0; h < hidden_size; ++h) {
                out_ref[t * hidden_size + h] += exp_out[i * num_tokens * hidden_size + t * hidden_size + h] * weight;
            }
        }
    }
    
    // GPU test
    Stream_t stream = cuda_stream_create();
    float* d_exp_out = reinterpret_cast<float*>(cuda_malloc(exp_out.size() * sizeof(float)));
    float* d_weights = reinterpret_cast<float*>(cuda_malloc(weights.size() * sizeof(float)));
    float* d_moe_out = reinterpret_cast<float*>(cuda_malloc(out_ref.size() * sizeof(float)));
    
    std::memcpy(d_exp_out, exp_out.data(), exp_out.size() * sizeof(float));
    std::memcpy(d_weights, weights.data(), weights.size() * sizeof(float));
    
    sync_host_to_device(d_exp_out, exp_out.size() * sizeof(float), stream);
    sync_host_to_device(d_weights, weights.size() * sizeof(float), stream);
    
    launch_moe_merge(d_moe_out, d_exp_out, d_weights, num_tokens, k_experts, hidden_size, stream);
    
    cuda_stream_synchronize(stream);
    cuda_memcpy_async(out_test.data(), d_moe_out, out_ref.size() * sizeof(float), stream);
    cuda_stream_synchronize(stream);
    
    for (size_t i = 0; i < out_ref.size(); ++i) {
        assert(is_close(out_ref[i], out_test[i]) && "MoE Merge output mismatch");
    }
    
    cuda_free(d_exp_out);
    cuda_free(d_weights);
    cuda_free(d_moe_out);
    cuda_stream_destroy(stream);
    std::cout << "PASSED!\n";
}

void test_kv_cache() {
    std::cout << "Testing KVCache caching and retrieval... ";
    int layers = 2;
    int kv_heads = 2;
    int dim = 128;
    int seq_len = 10;
    int q_dim = 2048;
    int k_dim = kv_heads * dim;
    int v_dim = kv_heads * dim;
    int qkv_output_size = q_dim + k_dim + v_dim;
    
    size_t storage_size = layers * 2 * kv_heads * seq_len * dim * sizeof(float);
    
    // Allocate GPU storage shadow allocation
    Stream_t stream = cuda_stream_create();
    void* d_storage = cuda_malloc(storage_size);
    std::memset(d_storage, 0, storage_size);
    
    KVCache cache(layers, kv_heads, dim, seq_len, d_storage);
    
    // Let's create dummy projected outputs for 2 tokens at position pos = 3
    int num_tokens = 2;
    int pos = 3;
    std::vector<float> qkv_proj_out(num_tokens * qkv_output_size);
    for (size_t i = 0; i < qkv_proj_out.size(); ++i) {
        qkv_proj_out[i] = static_cast<float>(i) * 0.001f;
    }
    
    // Allocate GPU qkv buffer
    float* d_qkv = reinterpret_cast<float*>(cuda_malloc(qkv_proj_out.size() * sizeof(float)));
    std::memcpy(d_qkv, qkv_proj_out.data(), qkv_proj_out.size() * sizeof(float));
    sync_host_to_device(d_qkv, qkv_proj_out.size() * sizeof(float), stream);
    
    // Update cache for layer 1
    cache.update(1, d_qkv, q_dim, k_dim, v_dim, qkv_output_size, num_tokens, pos, stream);
    cuda_stream_synchronize(stream);
    
    // Copy the entire GPU KV cache storage back to CPU to verify
    std::vector<float> storage_test(storage_size / sizeof(float), 0.0f);
    cuda_memcpy_async(storage_test.data(), d_storage, storage_size, stream);
    cuda_stream_synchronize(stream);
    
    // Let's verify KVCache host fallback updated the memory too
    float* h_keys = reinterpret_cast<float*>(cache.get_layer_keys(1));
    float* h_values = reinterpret_cast<float*>(cache.get_layer_values(1));
    
    // Let's check correctness for token i = 0 and i = 1
    for (int i = 0; i < num_tokens; ++i) {
        for (int d = 0; d < k_dim; ++d) {
            float expected = qkv_proj_out[i * qkv_output_size + q_dim + d];
            // Test CPU value
            float val_cpu = h_keys[(pos + i) * k_dim + d];
            assert(is_close(expected, val_cpu) && "KVCache CPU key mismatch");
            
            // Test GPU value (copied back)
            size_t layer_stride = 2 * kv_heads * seq_len * dim;
            size_t key_offset = 1 * layer_stride + (pos + i) * k_dim + d;
            float val_gpu = storage_test[key_offset];
            assert(is_close(expected, val_gpu) && "KVCache GPU key mismatch");
        }
        
        for (int d = 0; d < v_dim; ++d) {
            float expected = qkv_proj_out[i * qkv_output_size + q_dim + k_dim + d];
            // Test CPU value
            float val_cpu = h_values[(pos + i) * v_dim + d];
            assert(is_close(expected, val_cpu) && "KVCache CPU value mismatch");
            
            // Test GPU value (copied back)
            size_t layer_stride = 2 * kv_heads * seq_len * dim;
            size_t half_stride = kv_heads * seq_len * dim;
            size_t val_offset = 1 * layer_stride + half_stride + (pos + i) * v_dim + d;
            float val_gpu = storage_test[val_offset];
            assert(is_close(expected, val_gpu) && "KVCache GPU value mismatch");
        }
    }
    
    cuda_free(d_qkv);
    cuda_free(d_storage);
    cuda_stream_destroy(stream);
    std::cout << "PASSED!\n";
}

int main() {
    std::cout << "==========================================================\n";
    std::cout << "Running Turbo LLM GPU Kernel Correctness Unit Tests\n";
    std::cout << "==========================================================\n";
    
    test_rmsnorm();
    test_dequantize();
    test_gemm();
    test_gemm_transposed();
    test_activation();
    test_merge();
    test_kv_cache();
    
    std::cout << "==========================================================\n";
    std::cout << "All Turbo LLM GPU Kernels verified mathematically correct!\n";
    std::cout << "==========================================================\n";
    return 0;
}
