cmake_minimum_required(VERSION 3.24)

if(NOT CMAKE_CUDA_COMPILER AND DEFINED ENV{CUDACXX} AND EXISTS "$ENV{CUDACXX}")
    set(CMAKE_CUDA_COMPILER "$ENV{CUDACXX}")
elseif(NOT CMAKE_CUDA_COMPILER AND DEFINED ENV{CUDA_HOME} AND EXISTS "$ENV{CUDA_HOME}/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "$ENV{CUDA_HOME}/bin/nvcc")
elseif(NOT CMAKE_CUDA_COMPILER AND DEFINED ENV{CUDA_PATH} AND EXISTS "$ENV{CUDA_PATH}/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "$ENV{CUDA_PATH}/bin/nvcc")
elseif(NOT CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

find_program(NERVA_NVCC_ON_PATH nvcc)
find_program(NERVA_NVIDIA_SMI nvidia-smi PATHS /usr/bin /usr/local/bin /bin)
if(CMAKE_CUDA_COMPILER)
    set(NERVA_NVCC_FOR_DETECTION "${CMAKE_CUDA_COMPILER}")
elseif(NERVA_NVCC_ON_PATH)
    set(NERVA_NVCC_FOR_DETECTION "${NERVA_NVCC_ON_PATH}")
endif()

function(nerva_normalize_cuda_arch out token)
    string(STRIP "${token}" arch)
    string(TOLOWER "${arch}" arch)
    string(REPLACE "\"" "" arch "${arch}")
    string(REPLACE "'" "" arch "${arch}")
    string(REGEX REPLACE "^sm_" "" arch "${arch}")
    string(REGEX REPLACE "^compute_" "" arch "${arch}")
    string(REGEX REPLACE "-(real|virtual)$" "" arch "${arch}")
    if(arch MATCHES "^[0-9]+\\.[0-9]+$")
        string(REPLACE "." "" arch "${arch}")
    endif()
    if(arch MATCHES "^[0-9]+[a-z]?$")
        set(${out} "${arch}" PARENT_SCOPE)
    else()
        set(${out} "" PARENT_SCOPE)
    endif()
endfunction()

function(nerva_cuda_arch_list_from_string out raw)
    string(REPLACE "," ";" tokens "${raw}")
    string(REPLACE " " ";" tokens "${tokens}")
    string(REPLACE "\n" ";" tokens "${tokens}")
    string(REPLACE "\t" ";" tokens "${tokens}")
    set(arches "")
    foreach(token IN LISTS tokens)
        nerva_normalize_cuda_arch(normalized "${token}")
        if(normalized)
            list(APPEND arches "${normalized}")
        endif()
    endforeach()
    if(arches)
        list(REMOVE_DUPLICATES arches)
    endif()
    set(${out} "${arches}" PARENT_SCOPE)
endfunction()

function(nerva_detect_local_cuda_arch out)
    if(NOT NERVA_NVIDIA_SMI)
        set(${out} "" PARENT_SCOPE)
        return()
    endif()
    execute_process(
        COMMAND "${NERVA_NVIDIA_SMI}" --query-gpu=compute_cap "--format=csv,noheader,nounits"
        OUTPUT_VARIABLE smi_output
        ERROR_QUIET
        RESULT_VARIABLE smi_result
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT smi_result EQUAL 0)
        execute_process(
            COMMAND "${NERVA_NVIDIA_SMI}" --query-gpu=compute_cap "--format=csv,noheader"
            OUTPUT_VARIABLE smi_output
            ERROR_QUIET
            RESULT_VARIABLE smi_result
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endif()
    if(NOT smi_result EQUAL 0)
        set(${out} "" PARENT_SCOPE)
        return()
    endif()
    string(REPLACE "\n" ";" smi_lines "${smi_output}")
    foreach(line IN LISTS smi_lines)
        nerva_normalize_cuda_arch(arch "${line}")
        if(arch)
            set(${out} "${arch}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${out} "" PARENT_SCOPE)
endfunction()

function(nerva_nvcc_supported_cuda_arches out)
    if(NOT NERVA_NVCC_FOR_DETECTION)
        set(${out} "" PARENT_SCOPE)
        return()
    endif()
    execute_process(
        COMMAND "${NERVA_NVCC_FOR_DETECTION}" --list-gpu-code
        OUTPUT_VARIABLE nvcc_codes
        ERROR_QUIET
        RESULT_VARIABLE nvcc_codes_result
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT nvcc_codes_result EQUAL 0)
        set(${out} "" PARENT_SCOPE)
        return()
    endif()
    nerva_cuda_arch_list_from_string(arches "${nvcc_codes}")
    set(${out} "${arches}" PARENT_SCOPE)
endfunction()

function(nerva_nvcc_version out_major out_minor)
    if(NOT NERVA_NVCC_FOR_DETECTION)
        set(${out_major} "" PARENT_SCOPE)
        set(${out_minor} "" PARENT_SCOPE)
        return()
    endif()
    execute_process(
        COMMAND "${NERVA_NVCC_FOR_DETECTION}" --version
        OUTPUT_VARIABLE nvcc_version_text
        ERROR_QUIET
        RESULT_VARIABLE nvcc_version_result
    )
    if(NOT nvcc_version_result EQUAL 0)
        set(${out_major} "" PARENT_SCOPE)
        set(${out_minor} "" PARENT_SCOPE)
        return()
    endif()
    string(REGEX MATCH "release ([0-9]+)\\.([0-9]+)" _ "${nvcc_version_text}")
    if(CMAKE_MATCH_1)
        set(${out_major} "${CMAKE_MATCH_1}" PARENT_SCOPE)
        set(${out_minor} "${CMAKE_MATCH_2}" PARENT_SCOPE)
    else()
        set(${out_major} "" PARENT_SCOPE)
        set(${out_minor} "" PARENT_SCOPE)
    endif()
endfunction()

function(nerva_default_cuda_arches out)
    nerva_nvcc_supported_cuda_arches(supported_arches)
    if(supported_arches)
        set(${out} "${supported_arches}" PARENT_SCOPE)
        return()
    endif()

    nerva_nvcc_version(cuda_major cuda_minor)
    if(cuda_major STREQUAL "")
        set(default_arches "75;80;86;89;90;120")
    elseif(cuda_major LESS 11)
        set(default_arches "50;52;60;61;70;75")
    elseif(cuda_major EQUAL 11 AND cuda_minor EQUAL 0)
        set(default_arches "52;60;61;70;75;80")
    elseif(cuda_major EQUAL 11 AND cuda_minor LESS 8)
        set(default_arches "52;60;61;70;75;80;86")
    elseif(cuda_major EQUAL 11)
        set(default_arches "52;60;61;70;75;80;86;89;90")
    elseif(cuda_major EQUAL 12 AND cuda_minor LESS 8)
        set(default_arches "60;61;70;75;80;86;89;90")
    elseif(cuda_major EQUAL 12)
        set(default_arches "60;61;70;75;80;86;89;90;120")
    else()
        set(default_arches "75;80;86;87;88;89;90;100;103;110;120;121")
    endif()
    set(${out} "${default_arches}" PARENT_SCOPE)
endfunction()

set(NERVA_CUDA_ARCHITECTURES "" CACHE STRING "CUDA architectures for NERVA native CUDA builds")
if(NOT CMAKE_CUDA_ARCHITECTURES)
    if(NERVA_CUDA_ARCHITECTURES)
        nerva_cuda_arch_list_from_string(selected_arches "${NERVA_CUDA_ARCHITECTURES}")
    elseif(DEFINED ENV{NERVA_CUDA_ARCHITECTURES})
        nerva_cuda_arch_list_from_string(selected_arches "$ENV{NERVA_CUDA_ARCHITECTURES}")
    elseif(DEFINED ENV{CUDAARCHS})
        nerva_cuda_arch_list_from_string(selected_arches "$ENV{CUDAARCHS}")
    elseif(DEFINED ENV{NERVA_CUDA_ARCH})
        nerva_cuda_arch_list_from_string(selected_arches "$ENV{NERVA_CUDA_ARCH}")
    elseif(DEFINED ENV{CUDA_ARCH})
        nerva_cuda_arch_list_from_string(selected_arches "$ENV{CUDA_ARCH}")
    else()
        nerva_detect_local_cuda_arch(selected_arches)
    endif()

    if(NOT selected_arches)
        nerva_default_cuda_arches(selected_arches)
    endif()

    set(CMAKE_CUDA_ARCHITECTURES "${selected_arches}" CACHE STRING "CUDA architectures" FORCE)
endif()
project(nerva_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)

message(STATUS "NERVA CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "NERVA CUDA toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "NERVA CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

add_library(nerva_cuda STATIC
    nerva_cuda_device_smoke.cu
    nerva_cuda_backend_contract.cu
    nerva_cuda_synthetic_graph.cu
    nerva_cuda_tiny_block.cu
    nerva_cuda_block_forward.cu
    nerva_cuda_greedy_sampler.cu
    nerva_cuda_hf_sampler.cu
    nerva_cuda_hf_decode_step.cu
    nerva_cuda_hf_decode_chain.cu
    hf_decode_sequence/attention_kernels.cu
    hf_decode_sequence/kernels.cu
    hf_decode_sequence/prefill_kernels.cu
    hf_decode_sequence/projection.cu
    hf_decode_sequence/sampler.cu
    hf_decode_sequence/weights.cu
    nerva_cuda_hf_decode_sequence.cu
    nerva_cuda_tiny_decode.cu
    nerva_cuda_tiered_attention.cu
    nerva_cuda_projection_bench.cu
    nerva_cuda_deepseek_quant.cu
    nerva_cuda_deepseek_router.cu
    nerva_cuda_deepseek_mla.cu
    nerva_cuda_experimental_rt.cu
    smoke_kernel.cu
    token_ring_kernels.cu
    synthetic_decode.cu
)

target_include_directories(nerva_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(DEFINED ENV{NERVA_OPTIX_SDK_ROOT} AND EXISTS "$ENV{NERVA_OPTIX_SDK_ROOT}/include/optix.h")
    target_include_directories(nerva_cuda PUBLIC "$ENV{NERVA_OPTIX_SDK_ROOT}/include")
    target_compile_definitions(nerva_cuda PUBLIC "NERVA_OPTIX_INCLUDE_DIR=\"$ENV{NERVA_OPTIX_SDK_ROOT}/include\"")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/optix-sdk/include/optix.h")
    target_include_directories(nerva_cuda PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/optix-sdk/include")
    target_compile_definitions(nerva_cuda PUBLIC "NERVA_OPTIX_INCLUDE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/optix-sdk/include\"")
elseif(EXISTS "/usr/local/optix-sdk/include/optix.h")
    target_include_directories(nerva_cuda PUBLIC "/usr/local/optix-sdk/include")
    target_compile_definitions(nerva_cuda PUBLIC "NERVA_OPTIX_INCLUDE_DIR=\"/usr/local/optix-sdk/include\"")
elseif(EXISTS "/opt/nvidia/optix-sdk/include/optix.h")
    target_include_directories(nerva_cuda PUBLIC "/opt/nvidia/optix-sdk/include")
    target_compile_definitions(nerva_cuda PUBLIC "NERVA_OPTIX_INCLUDE_DIR=\"/opt/nvidia/optix-sdk/include\"")
endif()
if(CUDAToolkit_INCLUDE_DIRS)
    list(GET CUDAToolkit_INCLUDE_DIRS 0 NERVA_CUDA_INCLUDE_DIR)
    target_compile_definitions(nerva_cuda PUBLIC "NERVA_CUDA_INCLUDE_DIR=\"${NERVA_CUDA_INCLUDE_DIR}\"")
endif()
target_link_libraries(nerva_cuda PUBLIC CUDA::cudart CUDA::cublas)
