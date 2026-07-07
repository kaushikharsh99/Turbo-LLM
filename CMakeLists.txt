cmake_minimum_required(VERSION 3.18)
project(TurboPrototype LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to enable GPU acceleration via CUDA
option(USE_CUDA "Compile with CUDA support" OFF)

set(SOURCES
    src/main.cpp
    src/config.cpp
    src/profiler.cpp
    src/gguf/loader.cpp
    src/model/parser.cpp
    src/model/model.cpp
    src/runtime/executor.cpp
    src/runtime/chat_executor.cpp
    src/runtime/batch_executor.cpp
    src/runtime/memory.cpp
    src/runtime/cuda.cpp
    src/runtime/scheduler.cpp
    src/runtime/sampling.cpp
    src/runtime/kv_cache.cpp
    src/runtime/tokenizer.cpp
)

add_executable(turbo_prototype ${SOURCES})

target_include_directories(turbo_prototype PRIVATE src)

if(USE_CUDA)
    target_compile_definitions(turbo_prototype PRIVATE USE_CUDA)
    
    # Locate headers and libraries
    set(CUDA_INCLUDE_DIRS 
        "/usr/local/cuda-12.4/include"
    )
    set(CUDA_LIBRARIES 
        "/usr/local/cuda-12.4/lib64/libcudart.so"
        "/usr/local/cuda-12.4/lib64/libcublas.so"
        "/usr/local/cuda-12.4/lib64/libnvrtc.so"
        "/usr/lib64/libcuda.so"
    )
    
    target_include_directories(turbo_prototype PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(turbo_prototype PRIVATE ${CUDA_LIBRARIES})
    
    message(STATUS "CUDA NVRTC compilation enabled.")
endif()

# Correctness test executable
add_executable(turbo_tests
    tests/correctness_test.cpp
    src/runtime/cuda.cpp
    src/runtime/kv_cache.cpp
)
target_include_directories(turbo_tests PRIVATE src)

if(USE_CUDA)
    target_compile_definitions(turbo_tests PRIVATE USE_CUDA)
    target_include_directories(turbo_tests PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(turbo_tests PRIVATE ${CUDA_LIBRARIES})
endif()

# Shared library for Python reference runner ctypes loading
add_library(turbo_cuda SHARED
    src/runtime/cuda.cpp
)
target_include_directories(turbo_cuda PRIVATE src)
if(USE_CUDA)
    target_compile_definitions(turbo_cuda PRIVATE USE_CUDA)
    target_include_directories(turbo_cuda PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(turbo_cuda PRIVATE ${CUDA_LIBRARIES})
endif()
