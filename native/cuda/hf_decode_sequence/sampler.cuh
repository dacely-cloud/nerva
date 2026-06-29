#pragma once

#include "../nerva_cuda_api.h"

#include <cuda_runtime.h>
#include <stdint.h>

NervaCudaHfDecodeSamplerConfig default_hf_decode_sampler_config();
NervaCudaHfDecodeSamplerConfig normalize_hf_decode_sampler_config(
    NervaCudaHfDecodeSamplerConfig config);
bool hf_decode_sampler_config_matches(
    const NervaCudaHfDecodeSamplerConfig &lhs,
    const NervaCudaHfDecodeSamplerConfig &rhs);

cudaError_t launch_hf_decode_final_head_sampler(
    cudaStream_t stream, uint32_t *step_cursor, uint32_t max_steps,
    uint32_t has_eos_token, uint32_t eos_token, const float *scores,
    uint32_t vocab_size, NervaCudaSyntheticTokenSlot *slots,
    NervaCudaHfDecodeSamplerConfig sampler);
