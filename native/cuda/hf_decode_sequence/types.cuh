#pragma once

#include "../nerva_cuda_api.h"

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

constexpr uint32_t kDTypeF16 = 0;
constexpr uint32_t kDTypeBF16 = 1;
constexpr uint32_t kDTypeF32 = 2;
constexpr uint32_t kMlpKindDense = 0;
constexpr uint32_t kMlpKindSparseMoe = 1;
constexpr uint32_t kAttentionKindFull = 0;
constexpr uint32_t kAttentionKindLinearGdn = 1;
constexpr uint32_t kAttentionKindDeepSeekMla = 2;
constexpr uint32_t kDeepSeekModeV3Mla = 1;
constexpr uint32_t kDeepSeekModeV32MlaIndexer = 2;
constexpr uint32_t kDeepSeekModeV4Swa = 3;
constexpr uint32_t kDeepSeekModeV4Compressed = 4;
constexpr uint32_t kDeepSeekModeV4CompressedIndexer = 5;
constexpr uint32_t kDeepSeekFlagSparseIndexer = 1u << 0;
constexpr uint32_t kDeepSeekFlagCompressor = 1u << 1;
constexpr uint32_t kDeepSeekFlagHashRouter = 1u << 2;
constexpr uint32_t kDeepSeekFlagMoe = 1u << 3;
constexpr uint32_t kDeepSeekFlagSlidingWindow = 1u << 4;
constexpr uint32_t kDeepSeekFlagRouterBias = 1u << 5;
constexpr uint32_t kSparseMoeExpertsMax = 256;
constexpr uint32_t kSparseMoeTopKMax = 16;
constexpr uint32_t kRequestId = 1;
constexpr uint32_t kSequenceId = 1;
constexpr uint32_t kCompletionDeviceComplete = 1;
constexpr uint32_t kWeightStrategyGpuResident = 1;
constexpr uint32_t kWeightStrategyGpuStaged = 2;
constexpr uint32_t kDecodeThreads = 256;
constexpr uint32_t kDeepSeekRuntimeCounterCompressorStateWrites = 0;
constexpr uint32_t kDeepSeekRuntimeCounterCompressedKvWrites = 1;
constexpr uint32_t kDeepSeekRuntimeCounterIndexerStateWrites = 2;
constexpr uint32_t kDeepSeekRuntimeCounterIndexerKvWrites = 3;
constexpr uint32_t kDeepSeekRuntimeCounterCompressedKvAttentionReads = 4;
constexpr uint32_t kDeepSeekRuntimeCounterCompressedKvAttentionSlotsScanned = 5;
constexpr uint32_t kDeepSeekRuntimeCounterSparseTopkSelections = 6;
constexpr uint32_t kDeepSeekRuntimeCounterSparseTopkSlotsSelected = 7;
constexpr uint32_t kDeepSeekRuntimeCounterSparseTopkCandidatesScored = 8;
constexpr uint32_t kDeepSeekRuntimeCounterV3GroupedRouterSelections = 9;
constexpr uint32_t kDeepSeekRuntimeCounterV4BiasRouterSelections = 10;
constexpr uint32_t kDeepSeekRuntimeCounterV4HashRouterSelections = 11;
constexpr uint32_t kDeepSeekRuntimeCounterRawAttentionTokensScanned = 12;
constexpr uint32_t kDeepSeekRuntimeCounterCount = 13;
constexpr uint32_t kDeepSeekV4AttentionAuxStreamCount = 3;
constexpr uint32_t kDeepSeekV4AttentionEventCount = 4;
constexpr uint32_t kDecodeNormThreads = 1024;
constexpr uint32_t kHeadThreadsMax = 256;
constexpr uint32_t kHeadThreadElements = 4;
constexpr uint32_t kPrefillChunkBaseTokens = 1024;
constexpr uint32_t kPrefillChunkMaxTokens = 8192;
constexpr uint32_t kProjectionBatchWorkspaceTokens = 32;
constexpr uint32_t kKvCacheBlockTokens = 16;
constexpr uint32_t kDeepSeekV32PackedKvBlockTokens = 64;
constexpr uint32_t kDeepSeekV32PackedKvNopeBytes = 512;
constexpr uint32_t kDeepSeekV32PackedKvScaleBytes = 16;
constexpr uint32_t kDeepSeekV32PackedKvRopeValues = 64;
constexpr uint32_t kDeepSeekV32PackedKvTokenBytes =
    kDeepSeekV32PackedKvNopeBytes + kDeepSeekV32PackedKvScaleBytes +
    kDeepSeekV32PackedKvRopeValues * 2u;
constexpr uint32_t kDeepSeekV32IndexerKvBlockTokens = 64;
constexpr uint32_t kDeepSeekV32IndexerKvTileTokens = 16;
constexpr uint32_t kDeepSeekV32IndexerKvTileHeadBytes = 16;
constexpr uint32_t kDeepSeekV4PackedKvDefaultBlockTokens = 64;
constexpr uint32_t kDeepSeekV4PackedKvC128BlockTokens = 2;
constexpr uint32_t kDeepSeekV4PackedKvAlignmentBytes = 576;
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

__host__ __device__ inline uint32_t deepseek_v4_packed_kv_block_tokens(
    uint32_t compress_ratio) {
  return compress_ratio >= 128u ? kDeepSeekV4PackedKvC128BlockTokens
                                : kDeepSeekV4PackedKvDefaultBlockTokens;
}

__host__ __device__ inline uint64_t deepseek_v4_round_up_u64(
    uint64_t value, uint64_t alignment) {
  if (alignment == 0) {
    return value;
  }
  const uint64_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  const uint64_t add = alignment - remainder;
  return value > UINT64_MAX - add ? UINT64_MAX : value + add;
}

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
  uint64_t w_q_gate;
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
  uint64_t w_router;
  uint64_t w_expert_gate_up;
  uint64_t w_expert_down;
  uint64_t w_shared_expert_gate;
  uint64_t w_shared_expert_up;
  uint64_t w_shared_expert_down;
  uint64_t w_shared_expert_router;
  uint64_t w_linear_conv;
  uint64_t w_linear_qkv;
  uint64_t w_linear_z;
  uint64_t w_linear_b;
  uint64_t w_linear_a;
  uint64_t w_linear_dt_bias;
  uint64_t w_linear_a_log;
  uint64_t w_linear_norm;
  uint64_t w_linear_out;
  uint64_t linear_conv_state;
  uint64_t linear_recurrent_state;
  uint32_t linear_key_heads;
  uint32_t linear_value_heads;
  uint32_t linear_key_head_dim;
  uint32_t linear_value_head_dim;
  uint32_t linear_conv_kernel;
  uint32_t mlp_kind;
  uint32_t moe_intermediate;
  uint32_t shared_expert_intermediate;
  uint32_t num_experts;
  uint32_t experts_per_token;
  uint32_t norm_topk_prob;
  uint32_t attention_kind;
  uint32_t deepseek_mode;
  uint32_t deepseek_flags;
  uint32_t deepseek_hc_mult;
  uint32_t deepseek_q_lora_rank;
  uint32_t deepseek_kv_lora_rank;
  uint32_t deepseek_o_lora_rank;
  uint32_t deepseek_o_groups;
  uint32_t deepseek_qk_nope_head_dim;
  uint32_t deepseek_qk_rope_head_dim;
  uint32_t deepseek_v_head_dim;
  uint32_t deepseek_compress_ratio;
  uint32_t deepseek_index_topk;
  uint32_t deepseek_index_n_heads;
  uint32_t deepseek_index_head_dim;
  uint32_t deepseek_router_num_groups;
  uint32_t deepseek_router_topk_groups;
  float deepseek_routed_scaling_factor;
  uint64_t deepseek_q_a_scale;
  uint64_t deepseek_q_b;
  uint64_t deepseek_q_b_scale;
  uint64_t deepseek_kv_a_scale;
  uint64_t deepseek_kv_b_scale;
  uint64_t deepseek_o_a_scale;
  uint64_t deepseek_o_b;
  uint64_t deepseek_o_b_scale;
  uint64_t deepseek_attention_sink;
  uint64_t deepseek_indexer_q;
  uint64_t deepseek_indexer_q_scale;
  uint64_t deepseek_indexer_k;
  uint64_t deepseek_indexer_k_scale;
  uint64_t deepseek_indexer_k_norm;
  uint64_t deepseek_indexer_k_norm_bias;
  uint64_t deepseek_indexer_weights;
  uint64_t deepseek_compressor_ape;
  uint64_t deepseek_compressor_wkv;
  uint64_t deepseek_compressor_wgate;
  uint64_t deepseek_compressor_norm;
  uint64_t deepseek_indexer_compressor_ape;
  uint64_t deepseek_indexer_compressor_wkv;
  uint64_t deepseek_indexer_compressor_wgate;
  uint64_t deepseek_indexer_compressor_norm;
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
  float *q_gate;
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
  out.q_gate = out.mlp_norm + hidden;
  out.gate = out.q_gate + attention_hidden;
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
