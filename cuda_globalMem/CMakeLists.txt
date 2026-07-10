cmake_minimum_required(VERSION 4.1)
project(globalMem CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(globalMem main.cu)

set_target_properties(globalMem PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
