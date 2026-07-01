#include "../../deepseek_quant.cuh"
#include "../../deepseek_router.cuh"

constexpr uint32_t kDeepSeekSessionMaxCompressHeadSize = 512;
constexpr uint32_t kDeepSeekSessionMaxIndexerHeads = 128;
constexpr uint32_t kDeepSeekSessionMaxIndexerQueryValues = 8192;
constexpr uint32_t kDeepSeekSessionMaxSparseTopK = 2048;
constexpr uint32_t kDeepSeekSessionMaxMhcHcMult = 8;
constexpr uint32_t kDeepSeekSessionMaxMhcMixes =
    kDeepSeekSessionMaxMhcHcMult * (2u + kDeepSeekSessionMaxMhcHcMult);

__device__ float deepseek_fp8_scaled_weight(const uint16_t *arena,
                                            uint64_t weight_offset,
                                            uint64_t scale_offset,
                                            uint32_t rows, uint32_t cols,
                                            uint32_t row, uint32_t col);
__device__ float deepseek_rope_value_serial(float left, float right,
                                            uint32_t offset, uint32_t dim,
                                            uint32_t position, float theta,
                                            bool second,
                                            const SequenceLayerLayout &layout);
__device__ __forceinline__ uint16_t deepseek_session_f32_to_bf16_bits(
    float value);
__device__ __forceinline__ float deepseek_session_bf16_bits_to_f32(
    uint16_t bits);
__device__ uint8_t deepseek_session_f32_to_f8_e4m3fn_bits_nearest(
    float value);
__device__ float deepseek_swiglu(float gate, float up, float swiglu_limit);
__device__ bool deepseek_session_sparse_score_is_better(
    float candidate, int32_t slot, float current, int32_t current_slot);

__device__ __forceinline__ bool deepseek_v32_packed_physical_block(
    const uint32_t *kv_block_table, uint32_t kv_block_count,
    uint32_t packed_block_count, uint32_t logical_packed_block,
    uint32_t *physical_packed_block) {
  if (physical_packed_block == nullptr ||
      logical_packed_block >= packed_block_count) {
    return false;
  }
  if (kv_block_table == nullptr) {
    *physical_packed_block = logical_packed_block;
    return true;
  }
  constexpr uint32_t normal_blocks_per_packed =
      kDeepSeekV32PackedKvBlockTokens / kKvCacheBlockTokens;
  const uint32_t logical_normal_base =
      logical_packed_block * normal_blocks_per_packed;
  if (normal_blocks_per_packed == 0 || logical_normal_base >= kv_block_count) {
    return false;
  }
  const uint32_t available_normal_blocks =
      min(normal_blocks_per_packed, kv_block_count - logical_normal_base);
  const uint32_t physical_normal_base = kv_block_table[logical_normal_base];
  if (physical_normal_base % normal_blocks_per_packed != 0) {
    return false;
  }
  for (uint32_t offset = 1; offset < available_normal_blocks; ++offset) {
    if (kv_block_table[logical_normal_base + offset] !=
        physical_normal_base + offset) {
      return false;
    }
  }
  const uint32_t physical = physical_normal_base / normal_blocks_per_packed;
  if (physical >= packed_block_count) {
    return false;
  }
  *physical_packed_block = physical;
  return true;
}

__device__ void deepseek_session_apply_v4_mhc_pre_state(
    const uint16_t *arena, SequenceLayerLayout layout, uint32_t dtype,
    uint32_t hidden, uint32_t position, float rms_eps, const float *layer_input,
    uint32_t initialize_residual, uint64_t hc_base_offset,
    uint64_t hc_fn_offset, uint64_t hc_scale_offset, uint64_t norm_weight_offset,
    float *deepseek_mhc_residual, float *deepseek_mhc_post_mix,
    float *deepseek_mhc_comb_mix, float *temp_layer_input,
    uint16_t *projection_input);
__device__ void deepseek_session_finish_v4_mhc_head_norm(
    const uint16_t *arena, SequenceArenaLayout arena_layout,
    SequenceLayerLayout layout, uint32_t dtype, uint32_t final_norm_weight_dtype,
    uint32_t hidden, uint32_t position, float rms_eps, const float *layer_output,
    float *deepseek_mhc_residual, float *deepseek_mhc_post_mix,
    float *deepseek_mhc_comb_mix, float *temp_layer_input,
    uint16_t *projection_input);
