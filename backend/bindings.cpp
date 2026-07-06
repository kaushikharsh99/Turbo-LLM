#include <torch/extension.h>
#include "moe_executor.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("execute_moe", &execute_moe, "Execute sequential MoE layer (LibTorch C++ implementation)");
    m.def("execute_moe_with_cache", &execute_moe_with_cache, "Execute sequential MoE layer with C++ dequant cache");
}
