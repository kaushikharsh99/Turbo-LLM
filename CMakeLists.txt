cmake_minimum_required(VERSION 3.18)
project(TurboPrototype LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to enable GPU acceleration via CUDA Runtime linking
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
    
    # Locate CUDA Toolkit paths manually
    set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
    set(CUDA_LIBRARIES "/usr/local/cuda/lib64/libcudart.so")
    
    target_include_directories(turbo_prototype PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(turbo_prototype PRIVATE ${CUDA_LIBRARIES})
    
    message(STATUS "CUDA Runtime linking enabled. Include: ${CUDA_INCLUDE_DIRS}")
endif()
