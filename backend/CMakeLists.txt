cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(turbollm_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PyTorch (LibTorch)
find_package(Torch REQUIRED)
# Find pybind11
find_package(pybind11 REQUIRED)

pybind11_add_module(turbollm_cpp
    bindings.cpp
    moe_executor.cpp
    grouped_gemm.cpp
    dequant_cache.cpp
    fp8_dequant.cpp
    tensor_utils.cpp
)

target_link_libraries(turbollm_cpp PRIVATE ${TORCH_LIBRARIES})
