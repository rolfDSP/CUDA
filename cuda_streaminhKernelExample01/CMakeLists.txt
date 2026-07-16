cmake_minimum_required(VERSION 4.1)
project(cuda_streaminhKernelExample01 CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(cuda_streaminhKernelExample01 main.cu)

set_target_properties(cuda_streaminhKernelExample01 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(cuda_streaminhKernelExample01 PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/exe)
