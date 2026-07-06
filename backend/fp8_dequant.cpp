#include "fp8_dequant.h"

torch::Tensor dequantize_fp8(
    torch::Tensor w_fp8,
    torch::Tensor scale,
    torch::Dtype dtype
) {
    if (!w_fp8.defined()) {
        return torch::Tensor();
    }
    if (!scale.defined()) {
        return w_fp8.to(dtype);
    }
    
    auto w = w_fp8.to(dtype);
    auto scale_d = scale.to(dtype);
    
    int64_t M = w.size(0);
    int64_t N = w.size(1);
    
    auto w_reshaped = w.view({M / 128, 128, N / 128, 128});
    auto scale_reshaped = scale_d.view({M / 128, 1, N / 128, 1});
    
    return (w_reshaped * scale_reshaped).view({M, N});
}
