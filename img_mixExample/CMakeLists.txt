cmake_minimum_required(VERSION 4.1)
project(img_mixExample CUDA)

set(CMAKE_CUDA_STANDARD 20)

find_package(JPEG REQUIRED)

add_executable(img_mixExample main.cu)

target_link_libraries(img_mixExample PRIVATE JPEG::JPEG)

set_target_properties(img_mixExample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/exe)
