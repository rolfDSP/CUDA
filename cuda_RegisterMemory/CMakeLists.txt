cmake_minimum_required(VERSION 4.1)
project(cuda_RegisterMemory CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(cuda_RegisterMemory main.cu)

set_target_properties(cuda_RegisterMemory PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# Print per-kernel register usage at compile time.
target_compile_options(cuda_RegisterMemory PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
