#pragma once

#include "../nerva_cuda_api.h"

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

constexpr uint32_t kDTypeF16 = 0;
constexpr uint32_t kDTypeBF16 = 1;
constexpr uint32_t kRequestId = 1;
constexpr uint32_t kSequenceId = 1;
constexpr uint32_t kCompletionDeviceComplete = 1;
constexpr uint32_t kWeightStrategyGpuResident = 1;
constexpr uint32_t kWeightStrategyGpuStaged = 2;
constexpr uint32_t kDecodeThreads = 256;
constexpr uint32_t kDecodeNormThreads = 1024;
constexpr uint32_t kHeadThreadsMax = 256;
constexpr uint32_t kHeadThreadElements = 4;
constexpr uint32_t kPrefillChunkBaseTokens = 1024;
constexpr uint32_t kPrefillChunkMaxTokens = 8192;
constexpr uint32_t kProjectionBatchWorkspaceTokens = 32;
constexpr uint32_t kKvCacheBlockTokens = 16;
constexpr uint32_t kDecodeAttentionChunkTokens = 64;
constexpr uint32_t kGroupedGqaHeads = 4;
constexpr uint32_t kGroupedGqaThreadsPerHead = 64;
constexpr uint32_t kGroupedGqaThreads =
    kGroupedGqaHeads * kGroupedGqaThreadsPerHead;
constexpr uint32_t kGroupedGqaHeadDimMax =
    kGroupedGqaThreadsPerHead * kHeadThreadElements;
constexpr uint32_t kSharedWarpGqaThreadsPerHead = 32;
constexpr uint32_t kSharedWarpGqaThreads =
    kGroupedGqaHeads * kSharedWarpGqaThreadsPerHead;
constexpr uint32_t kSharedWarpGqaHeadDimMax =
    kSharedWarpGqaThreadsPerHead * kHeadThreadElements;
constexpr uint32_t kSharedWarpGqaTileTokens = kKvCacheBlockTokens;
constexpr uint32_t kSharedWarpGqaTileElements =
    kSharedWarpGqaTileTokens * kSharedWarpGqaHeadDimMax;
constexpr uint32_t kChunkedDecodeAttentionThreshold = 128;
constexpr uint64_t kMissingOffset = UINT64_MAX;
constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ull;
constexpr uint64_t kFnvPrime = 0x00000100000001b3ull;
constexpr size_t kCublasWorkspaceBytes = 64ull * 1024ull * 1024ull;
constexpr uint64_t kDescriptorStreamStagingBytes = 64ull * 1024ull * 1024ull;
constexpr uint64_t kPrefillAutotuneSafetyBytes = 256ull * 1024ull * 1024ull;

struct SequenceArenaLayout {
  uint64_t embeddings;
  uint64_t input;
  uint64_t scratch;
  uint64_t final_norm;
  uint64_t lm_head;
};

struct SequenceLayerLayout {
  uint64_t rms_attn;
  uint64_t rms_mlp;
  uint64_t w_q;
  uint64_t w_k;
  uint64_t q_norm;
  uint64_t k_norm;
  uint64_t w_v;
  uint64_t w_o;
  uint64_t q_bias;
  uint64_t k_bias;
  uint64_t v_bias;
  uint64_t o_bias;
  uint64_t w_gate;
  uint64_t w_up;
  uint64_t w_down;
};

struct PackedProjectionShape {
  uint64_t qkv_rows;
  uint64_t gate_up_rows;
  uint64_t qkv_elements_per_layer;
  uint64_t gate_up_elements_per_layer;
};

struct LayerScratch {
  float *input;
  float *attn_norm;
  float *q;
  float *k;
  float *v;
  float *attn;
  float *residual;
  float *mlp_norm;
  float *gate;
  float *up;
  float *ff;
  float *down;
};

__host__ __device__ inline LayerScratch layer_scratch_ptrs(
    float *scratch, uint32_t hidden, uint32_t attention_hidden,
    uint32_t kv_hidden, uint32_t intermediate) {
  LayerScratch out{};
  out.input = scratch;
  out.attn_norm = out.input + hidden;
  out.q = out.attn_norm + hidden;
  out.k = out.q + attention_hidden;
  out.v = out.k + kv_hidden;
  out.attn = out.v + kv_hidden;
  out.residual = out.attn + attention_hidden;
  out.mlp_norm = out.residual + hidden;
  out.gate = out.mlp_norm + hidden;
  out.up = out.gate + intermediate;
  out.ff = out.up + intermediate;
  out.down = out.ff + intermediate;
  return out;
}

__host__ __device__ __forceinline__ uint64_t kv_cache_page_offset(
    uint32_t layer_index, uint32_t kv_block_count, uint32_t physical_block,
    uint32_t block_offset, uint32_t kv_hidden, uint32_t kv_offset) {
  return (((static_cast<uint64_t>(layer_index) * kv_block_count +
            physical_block) *
               kKvCacheBlockTokens +
           block_offset) *
              kv_hidden +
          kv_offset);
}
