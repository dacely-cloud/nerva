cmake_minimum_required(VERSION 3.24)
project(nerva_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(nerva_cuda STATIC
    nerva_cuda_api.cpp
    smoke_kernel.cu
    graph_executor.cu
    token_ring_kernels.cu
    synthetic_decode.cu
)

target_include_directories(nerva_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
