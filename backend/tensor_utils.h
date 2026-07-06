#pragma once

#include <torch/extension.h>

void print_tensor_info(const torch::Tensor& tensor, const std::string& name);
