#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Vectorized half2 kernel for SiLU(Gate) * Up
__global__ void fused_silu_mul_half2_kernel(
    const half2* __restrict__ gate,
    const half2* __restrict__ up,
    half2* __restrict__ out,
    int num_half2_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_half2_elements) {
        half2 g2 = gate[idx];
        half2 u2 = up[idx];

        float g0 = __half2float(g2.x);
        float g1 = __half2float(g2.y);

        float u0 = __half2float(u2.x);
        float u1 = __half2float(u2.y);

        float silu0 = g0 / (1.0f + expf(-g0));
        float silu1 = g1 / (1.0f + expf(-g1));

        float res0 = silu0 * u0;
        float res1 = silu1 * u1;

        out[idx] = __floats2half2_rn(res0, res1);
    }
}

torch::Tensor fused_silu_mul_cuda(torch::Tensor gate, torch::Tensor up) {
    TORCH_CHECK(gate.is_cuda() && up.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(gate.dtype() == torch::kFloat16 && up.dtype() == torch::kFloat16, "Inputs must be Float16");
    TORCH_CHECK(gate.sizes() == up.sizes(), "Gate and Up sizes must match");

    auto out = torch::empty_like(gate);
    int total_elements = gate.numel();
    int num_half2 = total_elements / 2;

    int threads = 256;
    int blocks = (num_half2 + threads - 1) / threads;

    fused_silu_mul_half2_kernel<<<blocks, threads>>>(
        reinterpret_cast<const half2*>(gate.data_ptr<at::Half>()),
        reinterpret_cast<const half2*>(up.data_ptr<at::Half>()),
        reinterpret_cast<half2*>(out.data_ptr<at::Half>()),
        num_half2
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_silu_mul_cuda", &fused_silu_mul_cuda, "Vectorized Fused SiLU(Gate) * Up CUDA kernel");
}
