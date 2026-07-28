cmake_minimum_required(VERSION 4.1)

# This machine also has an older CUDA 12.0 toolkit installed via apt
# (/usr/lib/nvidia-cuda-toolkit), which CMake's default compiler search
# can pick over the real CUDA install and which doesn't support this
# GPU's architecture. Pin the compiler explicitly before project() so
# language detection uses the right nvcc.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS /usr/local/cuda/bin/nvcc)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()

project(simpleSigFreq LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(simpleSigFreq main.cpp main.cu)

target_link_libraries(simpleSigFreq PRIVATE CUDA::cufft)

set_target_properties(simpleSigFreq PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/exe)