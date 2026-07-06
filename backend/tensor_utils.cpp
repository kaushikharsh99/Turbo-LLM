#include "tensor_utils.h"
#include <iostream>

void print_tensor_info(const torch::Tensor& tensor, const std::string& name) {
    std::cout << "Tensor \"" << name << "\": "
              << "shape=" << tensor.sizes()
              << ", dtype=" << tensor.dtype()
              << ", device=" << tensor.device()
              << std::endl;
}
