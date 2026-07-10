cmake_minimum_required(VERSION 4.1)
project(cuda_ConstMem CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(cuda_ConstMem main.cu)
