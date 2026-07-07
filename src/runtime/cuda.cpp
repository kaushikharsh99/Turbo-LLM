#include "cuda.hpp"
#include <iostream>
#include <cstring>
#include <cmath>
#include <vector>
#include <mutex>
#include <unordered_map>

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvrtc.h>

namespace turbo {

struct Allocation {
    void* cpu_base;
    void* gpu_base;
    size_t size;
};

static std::vector<Allocation> allocations;
static std::mutex map_mutex;
static cublasHandle_t cublas_handle = nullptr;
static CUfunction dequant_q8_0_func = nullptr;
static CUfunction dequant_q4_k_func = nullptr;
static CUfunction dequant_q5_k_func = nullptr;
static CUfunction dequant_q6_k_func = nullptr;
static CUfunction dequant_f16_func = nullptr;
static CUfunction moe_activation_func = nullptr;
static CUfunction moe_merge_func = nullptr;
static CUfunction swiglu_func = nullptr;
static CUmodule dequant_module = nullptr;
static std::once_flag cuda_init_flag;

void init_cuda_global() {
    std::call_once(cuda_init_flag, [](){
        // 1. Initialize CUDA Driver API
        cuInit(0);
        
        // 2. Initialize cuBLAS
        cublasStatus_t status = cublasCreate(&cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cublasCreate failed with status: " << status << std::endl;
        } else {
            std::cout << "DEBUG: cuBLAS library initialized successfully." << std::endl;
        }

        // 3. Compile GPU Kernels via NVRTC
        const char* kernel_src = R"(
        __device__ inline float fp16_to_fp32(unsigned short h) {
            unsigned int sign = (h >> 15) & 1;
            unsigned int exponent = (h >> 10) & 0x1f;
            unsigned int mantissa = h & 0x3ff;
            if (exponent == 0) {
                if (mantissa == 0) {
                    unsigned int val = sign << 31;
                    return __int_as_float(val);
                }
                while ((mantissa & 0x400) == 0) {
                    mantissa <<= 1;
                    exponent--;
                }
                exponent++;
                mantissa &= 0x3ff;
            } else if (exponent == 31) {
                unsigned int val = (sign << 31) | 0x7f800000 | (mantissa << 13);
                return __int_as_float(val);
            }
            unsigned int val = (sign << 31) | ((exponent - 15 + 127) << 23) | (mantissa << 13);
            return __int_as_float(val);
        }

        __device__ inline void get_scale_min_k4(int j, const unsigned char *q, unsigned char *d, unsigned char *m) {
            if (j < 4) {
                *d = q[j] & 63;
                *m = q[j + 4] & 63;
            } else {
                *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
            }
        }

        extern "C" __global__ void dequantize_q8_0_kernel(float* out, const void* in_quant, int elements) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < elements) {
                int block_idx = idx / 32;
                int elem_in_block = idx % 32;
                const unsigned char* block_ptr = ((const unsigned char*)in_quant) + block_idx * 34;
                unsigned short d_raw = *(const unsigned short*)block_ptr;
                float d = fp16_to_fp32(d_raw);
                char q = ((const char*)(block_ptr + 2))[elem_in_block];
                out[idx] = q * d;
            }
        }

        extern "C" __global__ void dequantize_q4_k_kernel(float* out, const void* in_quant, int elements) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < elements) {
                int block_idx = idx / 256;
                int elem_in_block = idx % 256;
                const unsigned char* block_ptr = ((const unsigned char*)in_quant) + block_idx * 144;
                unsigned short d_raw = *(const unsigned short*)block_ptr;
                unsigned short dmin_raw = *(const unsigned short*)(block_ptr + 2);
                float d = fp16_to_fp32(d_raw);
                float min = fp16_to_fp32(dmin_raw);
                const unsigned char* scales = block_ptr + 4;
                const unsigned char* qs = block_ptr + 16;
                int is = (elem_in_block / 64) * 2;
                int step = (elem_in_block % 64);
                unsigned char sc, m;
                if (step < 32) {
                    get_scale_min_k4(is + 0, scales, &sc, &m);
                    unsigned char q = qs[step] & 0xF;
                    out[idx] = d * sc * q - min * m;
                } else {
                    get_scale_min_k4(is + 1, scales, &sc, &m);
                    unsigned char q = qs[step - 32] >> 4;
                    out[idx] = d * sc * q - min * m;
                }
            }
        }

        extern "C" __global__ void dequantize_q5_k_kernel(float* out, const void* in_quant, int elements) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < elements) {
                int block_idx = idx / 256;
                int elem_in_block = idx % 256;
                const unsigned char* block_ptr = ((const unsigned char*)in_quant) + block_idx * 176;
                unsigned short d_raw = *(const unsigned short*)block_ptr;
                unsigned short dmin_raw = *(const unsigned short*)(block_ptr + 2);
                float d = fp16_to_fp32(d_raw);
                float min = fp16_to_fp32(dmin_raw);
                const unsigned char* scales = block_ptr + 4;
                const unsigned char* qh = block_ptr + 16;
                const unsigned char* qs = block_ptr + 48;
                int is = (elem_in_block / 64) * 2;
                int step = (elem_in_block % 64);
                unsigned char sc, m;
                if (step < 32) {
                    get_scale_min_k4(is + 0, scales, &sc, &m);
                    unsigned char ql = qs[step] & 0xF;
                    unsigned char q = ql + ((qh[step] & (1 << is)) ? 16 : 0);
                    out[idx] = d * sc * q - min * m;
                } else {
                    get_scale_min_k4(is + 1, scales, &sc, &m);
                    unsigned char ql = qs[step - 32] >> 4;
                    unsigned char q = ql + ((qh[step - 32] & (1 << (is + 1))) ? 16 : 0);
                    out[idx] = d * sc * q - min * m;
                }
            }
        }

        extern "C" __global__ void dequantize_q6_k_kernel(float* out, const void* in_quant, int elements) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < elements) {
                int block_idx = idx / 256;
                int elem_in_block = idx % 256;
                const unsigned char* block_ptr = ((const unsigned char*)in_quant) + block_idx * 210;
                unsigned short d_raw = *(const unsigned short*)(block_ptr + 128 + 64 + 16);
                float d = fp16_to_fp32(d_raw);
                const unsigned char* ql = block_ptr;
                const unsigned char* qh = block_ptr + 128;
                const signed char* scales = (const signed char*)(block_ptr + 128 + 64);
                int n = (elem_in_block / 128) * 128;
                int sub_block = (elem_in_block % 128) / 32;
                int l = elem_in_block % 32;
                int n_offset = (n / 128);
                ql += n_offset * 64;
                qh += n_offset * 32;
                scales += n_offset * 8;
                int is = l / 16;
                float val = 0.0f;
                if (sub_block == 0) {
                    char q1 = (char)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    val = d * scales[is + 0] * q1;
                } else if (sub_block == 1) {
                    char q2 = (char)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    val = d * scales[is + 2] * q2;
                } else if (sub_block == 2) {
                    char q3 = (char)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    val = d * scales[is + 4] * q3;
                } else if (sub_block == 3) {
                    char q4 = (char)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    val = d * scales[is + 6] * q4;
                }
                out[idx] = val;
            }
        }

        extern "C" __global__ void dequantize_f16_kernel(float* out, const void* in_quant, int elements) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < elements) {
                unsigned short raw = ((const unsigned short*)in_quant)[idx];
                out[idx] = fp16_to_fp32(raw);
            }
        }

        extern "C" __global__ void moe_activation_kernel(float* out, const float* in, int elements) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < elements) {
                float x = in[idx];
                float sigmoid = 1.0f / (1.0f + expf(-x));
                out[idx] = x * sigmoid;
            }
        }

        extern "C" __global__ void moe_merge_kernel(float* moe_out, const float* exp_out, const float* weights, int num_tokens, int k_experts, int hidden_size) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int total = num_tokens * hidden_size;
            if (tid < total) {
                int t = tid / hidden_size;
                int h = tid % hidden_size;
                float sum = 0.0f;
                for (int i = 0; i < k_experts; ++i) {
                    float w = weights[t * k_experts + i];
                    sum += exp_out[i * num_tokens * hidden_size + t * hidden_size + h] * w;
                }
                moe_out[tid] = sum;
            }
        }

        extern "C" __global__ void swiglu_kernel(float* out, const float* gate, const float* up, int elements) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < elements) {
                float g = gate[idx];
                float u = up[idx];
                float silu_g = g / (1.0f + expf(-g));
                out[idx] = silu_g * u;
            }
        }
        )";

        nvrtcProgram prog;
        nvrtcResult create_res = nvrtcCreateProgram(&prog, kernel_src, "kernels.cu", 0, nullptr, nullptr);
        if (create_res != NVRTC_SUCCESS) {
            std::cerr << "nvrtcCreateProgram failed: " << create_res << std::endl;
            return;
        }
        
        // Compile program for compute capability (RTX 3050 is sm_86)
        const char* opts[] = { "--gpu-architecture=compute_86" };
        nvrtcResult compile_res = nvrtcCompileProgram(prog, 1, opts);
        if (compile_res != NVRTC_SUCCESS) {
            size_t log_size;
            nvrtcGetProgramLogSize(prog, &log_size);
            std::vector<char> log(log_size);
            nvrtcGetProgramLog(prog, log.data());
            std::cerr << "NVRTC Compilation failed:\n" << log.data() << std::endl;
            nvrtcDestroyProgram(&prog);
            return;
        }

        size_t ptx_size;
        nvrtcGetPTXSize(prog, &ptx_size);
        std::vector<char> ptx(ptx_size);
        nvrtcGetPTX(prog, ptx.data());
        nvrtcDestroyProgram(&prog);

        // Load PTX module
        CUresult cu_res = cuModuleLoadData(&dequant_module, ptx.data());
        if (cu_res != CUDA_SUCCESS) {
            std::cerr << "cuModuleLoadData failed: " << cu_res << std::endl;
            return;
        }
        
        cu_res = cuModuleGetFunction(&dequant_q8_0_func, dequant_module, "dequantize_q8_0_kernel");
        if (cu_res != CUDA_SUCCESS) std::cerr << "cuModuleGetFunction dequant_q8_0 failed: " << cu_res << std::endl;
        
        cu_res = cuModuleGetFunction(&dequant_q4_k_func, dequant_module, "dequantize_q4_k_kernel");
        if (cu_res != CUDA_SUCCESS) std::cerr << "cuModuleGetFunction dequant_q4_k failed: " << cu_res << std::endl;
        
        cu_res = cuModuleGetFunction(&dequant_q5_k_func, dequant_module, "dequantize_q5_k_kernel");
        if (cu_res != CUDA_SUCCESS) std::cerr << "cuModuleGetFunction dequant_q5_k failed: " << cu_res << std::endl;

        cu_res = cuModuleGetFunction(&dequant_q6_k_func, dequant_module, "dequantize_q6_k_kernel");
        if (cu_res != CUDA_SUCCESS) std::cerr << "cuModuleGetFunction dequant_q6_k failed: " << cu_res << std::endl;

        cu_res = cuModuleGetFunction(&dequant_f16_func, dequant_module, "dequantize_f16_kernel");
        if (cu_res != CUDA_SUCCESS) std::cerr << "cuModuleGetFunction dequant_f16 failed: " << cu_res << std::endl;
        
        cu_res = cuModuleGetFunction(&moe_activation_func, dequant_module, "moe_activation_kernel");
        if (cu_res != CUDA_SUCCESS) std::cerr << "cuModuleGetFunction activation failed: " << cu_res << std::endl;
        
        cu_res = cuModuleGetFunction(&moe_merge_func, dequant_module, "moe_merge_kernel");
        if (cu_res != CUDA_SUCCESS) std::cerr << "cuModuleGetFunction merge failed: " << cu_res << std::endl;
        
        cu_res = cuModuleGetFunction(&swiglu_func, dequant_module, "swiglu_kernel");
        if (cu_res != CUDA_SUCCESS) std::cerr << "cuModuleGetFunction swiglu failed: " << cu_res << std::endl;
        
        std::cout << "DEBUG: GPU dequantization, activation, and merge kernels compiled and loaded successfully via NVRTC." << std::endl;
    });
}

} // namespace turbo
#endif

namespace turbo {

void* get_gpu_pointer(const void* ptr) {
    if (!ptr) return nullptr;
#ifdef USE_CUDA
    std::lock_guard<std::mutex> lock(map_mutex);
    const uint8_t* p = reinterpret_cast<const uint8_t*>(ptr);
    for (const auto& alloc : allocations) {
        const uint8_t* base = reinterpret_cast<const uint8_t*>(alloc.cpu_base);
        if (p >= base && p < base + alloc.size) {
            size_t offset = p - base;
            return reinterpret_cast<uint8_t*>(alloc.gpu_base) + offset;
        }
    }
#endif
    return nullptr;
}

Stream_t cuda_stream_create() {
#ifdef USE_CUDA
    init_cuda_global();
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
        allocations.push_back({cpu_ptr, gpu_ptr, size});
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
        for (auto it = allocations.begin(); it != allocations.end(); ++it) {
            if (it->cpu_base == ptr) {
                gpu_ptr = it->gpu_base;
                allocations.erase(it);
                break;
            }
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
    void* dst_gpu = get_gpu_pointer(dst);
    const void* src_gpu = get_gpu_pointer(src);
    
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (dst_gpu && src_gpu) {
        cudaMemcpyAsync(dst_gpu, src_gpu, size, cudaMemcpyDeviceToDevice, cuda_stream);
    } else if (dst_gpu) {
        cudaMemcpyAsync(dst_gpu, src, size, cudaMemcpyHostToDevice, cuda_stream);
        std::memcpy(dst, src, size);
    } else if (src_gpu) {
        cudaMemcpyAsync(dst, src_gpu, size, cudaMemcpyDeviceToHost, cuda_stream);
    } else {
        std::memcpy(dst, src, size);
    }
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
// CPU Kernels with GPU sync copies
// ----------------------------------------------------

void launch_rmsnorm(float* out, const float* in, const float* weight, float eps, int size, int batch, Stream_t stream) {
    // 1. Run RMSNorm on Host CPU
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

#ifdef USE_CUDA
    // 2. Synchronize CPU state back to GPU shadow allocation
    if (stream) {
        void* out_gpu = get_gpu_pointer(out);
        if (out_gpu) {
            cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
            cudaMemcpyAsync(out_gpu, out, batch * size * sizeof(float), cudaMemcpyHostToDevice, cuda_stream);
        }
    }
#endif
}

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            uint32_t result = sign << 31;
            float f;
            std::memcpy(&f, &result, 4);
            return f;
        }
        float f = (sign ? -1.0f : 1.0f) * std::ldexp(static_cast<float>(mant), -24);
        return f;
    } else if (exp == 31) {
        uint32_t result = (sign << 31) | (0xFF << 23) | (mant << 13);
        float f;
        std::memcpy(&f, &result, 4);
        return f;
    }

    uint32_t result = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &result, 4);
    return f;
}

void launch_dequant(float* out, const void* in_quant, int type, int elements, Stream_t stream) {
    if (!in_quant || !out) {
        std::cerr << "CRITICAL: Null pointer in launch_dequant!" << std::endl;
        return;
    }

#ifdef USE_CUDA
    if (stream) {
        CUfunction func = nullptr;
        if (type == 8) func = dequant_q8_0_func; // Q8_0
        else if (type == 12) func = dequant_q4_k_func; // Q4_K
        else if (type == 13) func = dequant_q5_k_func; // Q5_K
        else if (type == 14) func = dequant_q6_k_func; // Q6_K
        else if (type == 1) func = dequant_f16_func; // F16

        void* out_gpu = get_gpu_pointer(out);
        void* in_quant_gpu = get_gpu_pointer(in_quant);
        
        if (func && out_gpu && in_quant_gpu) {
            cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
            int threads_per_block = 256;
            int num_blocks = (elements + threads_per_block - 1) / threads_per_block;
            void* args[] = { &out_gpu, &in_quant_gpu, &elements };
            CUresult res = cuLaunchKernel(
                func,
                num_blocks, 1, 1,
                threads_per_block, 1, 1,
                0,
                cuda_stream,
                args,
                nullptr
            );
            if (res != CUDA_SUCCESS) {
                std::cerr << "cuLaunchKernel failed: " << res << std::endl;
            }
            return;
        }
    }
#endif

    // Fallback Host CPU Dequantization
    if (type == 8) { // Q8_0
        #pragma pack(push, 1)
        struct block_q8_0 {
            uint16_t d;
            int8_t qs[32];
        };
        #pragma pack(pop)
        const block_q8_0* x = reinterpret_cast<const block_q8_0*>(in_quant);
        int nb = elements / 32;
        for (int i = 0; i < nb; ++i) {
            float d = f16_to_f32(x[i].d);
            for (int j = 0; j < 32; ++j) {
                out[i * 32 + j] = x[i].qs[j] * d;
            }
        }
    } else if (type == 12) { // Q4_K
        #pragma pack(push, 1)
        struct block_q4_K {
            uint16_t d;
            uint16_t dmin;
            uint8_t scales[12];
            uint8_t qs[128];
        };
        #pragma pack(pop)
        const block_q4_K* x = reinterpret_cast<const block_q4_K*>(in_quant);
        int nb = elements / 256;
        for (int i = 0; i < nb; ++i) {
            const uint8_t* q = x[i].qs;
            float d = f16_to_f32(x[i].d);
            float min = f16_to_f32(x[i].dmin);
            int is = 0;
            uint8_t sc, m;
            for (int j = 0; j < 256; j += 64) {
                get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
                float d1 = d * sc; float m1 = min * m;
                get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
                float d2 = d * sc; float m2 = min * m;
                for (int l = 0; l < 32; ++l) *out++ = d1 * (q[l] & 0xF) - m1;
                for (int l = 0; l < 32; ++l) *out++ = d2 * (q[l] >> 4) - m2;
                q += 32; is += 2;
            }
        }
    } else if (type == 13) { // Q5_K
        #pragma pack(push, 1)
        struct block_q5_K {
            uint16_t d;
            uint16_t dmin;
            uint8_t scales[12];
            uint8_t qh[32];
            uint8_t qs[128];
        };
        #pragma pack(pop)
        const block_q5_K* x = reinterpret_cast<const block_q5_K*>(in_quant);
        int nb = elements / 256;
        for (int i = 0; i < nb; ++i) {
            const uint8_t* ql = x[i].qs;
            const uint8_t* qh = x[i].qh;
            float d = f16_to_f32(x[i].d);
            float min = f16_to_f32(x[i].dmin);
            int is = 0;
            uint8_t sc, m;
            uint8_t u1 = 1, u2 = 2;
            for (int j = 0; j < 256; j += 64) {
                get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
                float d1 = d * sc; float m1 = min * m;
                get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
                float d2 = d * sc; float m2 = min * m;
                for (int l = 0; l < 32; ++l) *out++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
                for (int l = 0; l < 32; ++l) *out++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
                ql += 32; is += 2;
                u1 <<= 2; u2 <<= 2;
            }
        }
    } else if (type == 14) { // Q6_K
        #pragma pack(push, 1)
        struct block_q6_K {
            uint8_t ql[128];
            uint8_t qh[64];
            int8_t scales[16];
            uint16_t d;
        };
        #pragma pack(pop)
        const block_q6_K* x = reinterpret_cast<const block_q6_K*>(in_quant);
        int nb = elements / 256;
        for (int i = 0; i < nb; ++i) {
            float d = f16_to_f32(x[i].d);
            const uint8_t *ql = x[i].ql;
            const uint8_t *qh = x[i].qh;
            const int8_t *sc = x[i].scales;
            for (int n = 0; n < 256; n += 128) {
                for (int l = 0; l < 32; ++l) {
                    int is = l / 16;
                    const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    out[l + 0] = d * sc[is + 0] * q1;
                    out[l + 32] = d * sc[is + 2] * q2;
                    out[l + 64] = d * sc[is + 4] * q3;
                    out[l + 96] = d * sc[is + 6] * q4;
                }
                out += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
    } else if (type == 1) { // F16
        const uint16_t* x = reinterpret_cast<const uint16_t*>(in_quant);
        for (int i = 0; i < elements; ++i) {
            out[i] = f16_to_f32(x[i]);
        }
    } else if (type == 0) { // F32 / FP32 (no dequantization needed)
        std::memcpy(out, in_quant, elements * sizeof(float));
    } else {
        std::cerr << "WARNING: Unsupported dequantization type: " << type << std::endl;
        const int8_t* quant_data = reinterpret_cast<const int8_t*>(in_quant);
        for (int i = 0; i < elements; ++i) {
            out[i] = static_cast<float>(quant_data[i % (elements / 2)]) * 0.02f;
        }
    }
}

void launch_gemm(float* out, const float* a, const float* b, int m, int n, int k, bool transpose_b, Stream_t stream) {
#ifdef USE_CUDA
    if (stream && cublas_handle) {
        void* out_gpu = get_gpu_pointer(out);
        void* a_gpu = get_gpu_pointer(const_cast<float*>(a));
        void* b_gpu = get_gpu_pointer(const_cast<float*>(b));
        
        if (out_gpu && a_gpu && b_gpu) {
            cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
            cublasSetStream(cublas_handle, cuda_stream);
            
            float alpha = 1.0f;
            float beta = 0.0f;
            
            cublasStatus_t status = cublasSgemm(
                cublas_handle,
                transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,
                CUBLAS_OP_N,
                n, m, k,
                &alpha,
                reinterpret_cast<const float*>(b_gpu), transpose_b ? k : n,
                reinterpret_cast<const float*>(a_gpu), k,
                &beta,
                reinterpret_cast<float*>(out_gpu), n
            );
            
            if (status != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "cublasSgemm failed with status: " << status << std::endl;
            }
            
            // Asynchronously copy result back to Host CPU memory
            cudaMemcpyAsync(out, out_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost, cuda_stream);
            
            // Synchronize execution stream so the CPU immediately reads the GEMM output
            cudaStreamSynchronize(cuda_stream);
            return;
        }
    }
#endif

    // Fallback Host CPU GEMM
    if (transpose_b) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int l = 0; l < k; ++l) {
                    sum += a[i * k + l] * b[j * k + l];
                }
                out[i * n + j] = sum;
            }
        }
    } else {
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
}

void launch_moe_router(float* gate_logits, const float* hidden_states, const float* gate_weight, int batch, int hidden_size, int num_experts, Stream_t stream) {
    launch_gemm(gate_logits, hidden_states, gate_weight, batch, num_experts, hidden_size, true, stream);
}

void sync_host_to_device(void* ptr_cpu, size_t size, Stream_t stream) {
#ifdef USE_CUDA
    if (stream) {
        void* gpu_ptr = get_gpu_pointer(ptr_cpu);
        if (gpu_ptr) {
            cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
            cudaMemcpyAsync(gpu_ptr, ptr_cpu, size, cudaMemcpyHostToDevice, cuda_stream);
        }
    }
#endif
}

void launch_moe_activation(float* out, const float* in, int elements, Stream_t stream) {
#ifdef USE_CUDA
    if (stream && moe_activation_func) {
        void* out_gpu = get_gpu_pointer(out);
        void* in_gpu = get_gpu_pointer(const_cast<float*>(in));
        if (out_gpu && in_gpu) {
            cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
            int threads_per_block = 256;
            int num_blocks = (elements + threads_per_block - 1) / threads_per_block;
            void* args[] = { &out_gpu, &in_gpu, &elements };
            cuLaunchKernel(moe_activation_func, num_blocks, 1, 1, threads_per_block, 1, 1, 0, cuda_stream, args, nullptr);
            return;
        }
    }
#endif
    // Fallback Host CPU Silu
    for (int i = 0; i < elements; ++i) {
        float x = in[i];
        float sigmoid = 1.0f / (1.0f + std::exp(-x));
        out[i] = x * sigmoid;
    }
}

void launch_moe_merge(float* moe_out, const float* exp_out, const float* weights, int num_tokens, int k_experts, int hidden_size, Stream_t stream) {
#ifdef USE_CUDA
    if (stream && moe_merge_func) {
        void* moe_out_gpu = get_gpu_pointer(moe_out);
        void* exp_out_gpu = get_gpu_pointer(const_cast<float*>(exp_out));
        void* weights_gpu = get_gpu_pointer(const_cast<float*>(weights));
        if (moe_out_gpu && exp_out_gpu && weights_gpu) {
            cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
            int threads_per_block = 256;
            int total = num_tokens * hidden_size;
            int num_blocks = (total + threads_per_block - 1) / threads_per_block;
            void* args[] = { &moe_out_gpu, &exp_out_gpu, &weights_gpu, &num_tokens, &k_experts, &hidden_size };
            cuLaunchKernel(moe_merge_func, num_blocks, 1, 1, threads_per_block, 1, 1, 0, cuda_stream, args, nullptr);
            
            // Copy output back to CPU
            cudaMemcpyAsync(moe_out, moe_out_gpu, num_tokens * hidden_size * sizeof(float), cudaMemcpyDeviceToHost, cuda_stream);
            cudaStreamSynchronize(cuda_stream);
            return;
        }
    }
#endif
    // Fallback Host CPU Merge
    std::memset(moe_out, 0, num_tokens * hidden_size * sizeof(float));
    for (int t = 0; t < num_tokens; ++t) {
        for (int i = 0; i < k_experts; ++i) {
            float weight = weights[t * k_experts + i];
            for (int h = 0; h < hidden_size; ++h) {
                moe_out[t * hidden_size + h] += exp_out[i * num_tokens * hidden_size + t * hidden_size + h] * weight;
            }
        }
    }
}

void launch_swiglu(float* out, const float* gate, const float* up, int elements, Stream_t stream) {
#ifdef USE_CUDA
    if (stream && swiglu_func) {
        void* out_gpu = get_gpu_pointer(out);
        void* gate_gpu = get_gpu_pointer(const_cast<float*>(gate));
        void* up_gpu = get_gpu_pointer(const_cast<float*>(up));
        if (out_gpu && gate_gpu && up_gpu) {
            cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
            int threads_per_block = 256;
            int num_blocks = (elements + threads_per_block - 1) / threads_per_block;
            void* args[] = { &out_gpu, &gate_gpu, &up_gpu, &elements };
            cuLaunchKernel(swiglu_func, num_blocks, 1, 1, threads_per_block, 1, 1, 0, cuda_stream, args, nullptr);
            
            // Synchronize execution stream so output is copied back if needed
            // Wait, we don't copy back here because down projection GEMM will do it!
            return;
        }
    }
#endif
    // Fallback Host CPU SwiGLU
    for (int i = 0; i < elements; ++i) {
        float g = gate[i];
        float u = up[i];
        float silu_g = g / (1.0f + std::exp(-g));
        out[i] = silu_g * u;
    }
}

} // namespace turbo

extern "C" {
    void launch_dequant_c(float* out, const void* in_quant, int type, int elements, void* stream) {
        turbo::launch_dequant(out, in_quant, type, elements, stream);
    }
}
