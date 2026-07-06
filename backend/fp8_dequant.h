#pragma once

#include <torch/extension.h>

torch::Tensor dequantize_fp8(
    torch::Tensor w_fp8,
    torch::Tensor scale,
    torch::Dtype dtype
);
