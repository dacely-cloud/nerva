#include "weights.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

uint64_t push(uint64_t &cursor, uint64_t len) {
  const uint64_t offset = cursor;
  cursor += len;
  return offset;
}

uint64_t push_optional(uint64_t &cursor, uint64_t len, const uint16_t *ptr) {
  if (ptr == nullptr) {
    return kMissingOffset;
  }
  return push(cursor, len);
}

uint64_t linear_gdn_key_dim(const NervaCudaHfDecodeChainLayer &layer) {
  return static_cast<uint64_t>(layer.linear_key_heads) * layer.linear_key_head_dim;
}

uint64_t linear_gdn_value_dim(const NervaCudaHfDecodeChainLayer &layer) {
  return static_cast<uint64_t>(layer.linear_value_heads) * layer.linear_value_head_dim;
}

uint64_t linear_gdn_conv_dim(const NervaCudaHfDecodeChainLayer &layer) {
  return linear_gdn_key_dim(layer) * 2u + linear_gdn_value_dim(layer);
}

bool linear_gdn_dims_valid(const NervaCudaHfDecodeChainLayer &layer) {
  return layer.linear_key_heads != 0 && layer.linear_value_heads != 0 &&
         layer.linear_key_head_dim != 0 && layer.linear_value_head_dim != 0 &&
         layer.linear_conv_kernel != 0;
}

uint64_t hash_tokens(const uint32_t *tokens, uint32_t count) {
  uint64_t hash = kFnvOffset;
  for (uint32_t index = 0; index < count; ++index) {
    uint32_t token = tokens[index];
    for (uint32_t byte = 0; byte < 4; ++byte) {
      hash ^= static_cast<uint64_t>((token >> (8u * byte)) & 0xffu);
      hash *= kFnvPrime;
    }
  }
  return hash;
}

void hash_u32(uint64_t &hash, uint32_t value) {
  for (uint32_t byte = 0; byte < 4; ++byte) {
    hash ^= static_cast<uint64_t>((value >> (8u * byte)) & 0xffu);
    hash *= kFnvPrime;
  }
}

void hash_u64(uint64_t &hash, uint64_t value) {
  for (uint32_t byte = 0; byte < 8; ++byte) {
    hash ^= static_cast<uint64_t>((value >> (8u * byte)) & 0xffu);
    hash *= kFnvPrime;
  }
}

void hash_descriptor(uint64_t &hash,
                     const NervaCudaHfDecodeSequenceWeightBlock &descriptor) {
  hash_u64(hash, descriptor.block_id);
  hash_u64(hash, descriptor.block_version);
  hash_u64(hash, descriptor.offset_bytes);
  hash_u64(hash, descriptor.bytes);
  hash_u32(hash, descriptor.strategy);
}

bool descriptor_has_memory_source(
    const NervaCudaHfDecodeSequenceWeightBlock &descriptor) {
  return descriptor.host_source != nullptr;
}

bool descriptor_has_file_source(
    const NervaCudaHfDecodeSequenceWeightBlock &descriptor) {
  return descriptor.source_file != nullptr && descriptor.source_file_len != 0;
}

bool descriptor_has_source(
    const NervaCudaHfDecodeSequenceWeightBlock &descriptor) {
  return descriptor_has_memory_source(descriptor) ||
         descriptor_has_file_source(descriptor);
}

template <typename Request>
bool descriptors_require_file_staging_impl(const Request *request) {
  if (request == nullptr || request->planned_weight_descriptor_count == 0 ||
      request->planned_weight_descriptors == nullptr) {
    return false;
  }
  for (uint32_t index = 0; index < request->planned_weight_descriptor_count; ++index) {
    if (descriptor_has_file_source(request->planned_weight_descriptors[index])) {
      return true;
    }
  }
  return false;
}

template <typename Request>
uint64_t pinned_weight_staging_bytes_impl(const Request *request,
                                     uint64_t full_weight_bytes) {
  if (request->planned_weight_blocks == 0 && request->planned_weight_bytes == 0) {
    return full_weight_bytes;
  }
  if (!descriptors_require_file_staging_impl(request)) {
    return sizeof(uint16_t);
  }
  uint64_t bytes = std::min(full_weight_bytes, kDescriptorStreamStagingBytes);
  bytes -= bytes % sizeof(uint16_t);
  return bytes == 0 ? sizeof(uint16_t) : bytes;
}

template <typename Request>
bool has_declared_weight_plan_impl(const Request *request) {
  return request->planned_weight_blocks != 0 || request->planned_weight_bytes != 0;
}

void debug_valid_request_failure(const char *reason) {
  if (getenv("NERVA_DEBUG_VALID_REQUEST") != nullptr) {
    fprintf(stderr, "[nerva-debug] valid_request failed: %s\n", reason);
  }
}

bool valid_layer(const NervaCudaHfDecodeChainLayer &layer, bool require_sources) {
  const bool dense = layer.mlp_kind == kMlpKindDense;
  const bool sparse_moe = layer.mlp_kind == kMlpKindSparseMoe;
  if (!dense && !sparse_moe) {
    return false;
  }
  if (layer.attention_kind == kAttentionKindLinearGdn) {
    if (!linear_gdn_dims_valid(layer)) {
      return false;
    }
  } else if (layer.attention_kind != kAttentionKindFull) {
    return false;
  }
  if (sparse_moe &&
      (layer.moe_intermediate == 0 || layer.num_experts == 0 ||
       layer.experts_per_token == 0 ||
       layer.experts_per_token > layer.num_experts ||
       layer.num_experts > kSparseMoeExpertsMax ||
       layer.experts_per_token > kSparseMoeTopKMax)) {
    return false;
  }
  if (!require_sources) {
    return true;
  }
  if (layer.rms_attn_weight == nullptr || layer.rms_mlp_weight == nullptr) {
    return false;
  }
  if (layer.attention_kind == kAttentionKindLinearGdn) {
    if (layer.w_linear_conv == nullptr || layer.w_linear_qkv == nullptr ||
        layer.w_linear_z == nullptr || layer.w_linear_b == nullptr ||
        layer.w_linear_a == nullptr || layer.w_linear_dt_bias == nullptr ||
        layer.w_linear_a_log == nullptr || layer.w_linear_norm == nullptr ||
        layer.w_linear_out == nullptr) {
      return false;
    }
  } else if (layer.w_q == nullptr || layer.w_k == nullptr ||
             layer.w_v == nullptr || layer.w_o == nullptr) {
    return false;
  }
  if (dense) {
    return layer.w_gate != nullptr && layer.w_up != nullptr &&
           layer.w_down != nullptr;
  }
  if (layer.w_router == nullptr || layer.w_expert_gate_up == nullptr ||
      layer.w_expert_down == nullptr) {
    return false;
  }
  if (layer.shared_expert_intermediate != 0 &&
      (layer.w_shared_expert_gate == nullptr ||
       layer.w_shared_expert_up == nullptr ||
       layer.w_shared_expert_down == nullptr ||
       layer.w_shared_expert_router == nullptr)) {
    return false;
  }
  return true;
}

bool valid_request(const NervaCudaHfDecodeSequenceRequest *request) {
  if (request == nullptr) {
    debug_valid_request_failure("null request");
    return false;
  }
  const bool declared_weight_plan = has_declared_weight_plan_impl(request);
  if (request->layers == nullptr || request->output_tokens == nullptr ||
      request->prompt_tokens == nullptr ||
      (!declared_weight_plan &&
       (request->embeddings == nullptr || request->final_norm_weight == nullptr ||
        request->lm_head == nullptr)) ||
      request->output_token_capacity < request->steps || request->layer_count == 0 ||
      request->steps == 0 || request->prompt_token_count == 0 ||
      request->hidden == 0 || request->heads == 0 || request->kv_heads == 0 ||
      request->head_dim == 0 || request->intermediate == 0 ||
      request->vocab_size == 0 || request->seed_token >= request->vocab_size ||
      request->kv_heads > request->heads || request->heads % request->kv_heads != 0 ||
      request->dtype > kDTypeBF16 ||
      !std::isfinite(request->sampler.temperature) ||
      request->sampler.temperature < 0.0f ||
      !std::isfinite(request->sampler.top_p) ||
      request->sampler.top_p <= 0.0f || request->sampler.top_p > 1.0f ||
      (request->rope_theta > 0.0f && request->head_dim % 2 != 0) ||
      request->prompt_token_count > UINT32_MAX - request->steps + 1u) {
    debug_valid_request_failure("shape/source/sampler invariant");
    return false;
  }
  for (uint32_t index = 0; index < request->prompt_token_count; ++index) {
    if (request->prompt_tokens[index] >= request->vocab_size) {
      debug_valid_request_failure("prompt token outside vocab");
      return false;
    }
  }
  if (request->prompt_tokens[request->prompt_token_count - 1u] != request->seed_token) {
    debug_valid_request_failure("seed token does not match prompt tail");
    return false;
  }
  for (uint32_t index = 0; index < request->layer_count; ++index) {
    if (!valid_layer(request->layers[index], !declared_weight_plan)) {
      if (getenv("NERVA_DEBUG_VALID_REQUEST") != nullptr) {
        const auto &layer = request->layers[index];
        fprintf(stderr,
                "[nerva-debug] invalid layer %u mlp_kind=%u moe_intermediate=%u "
                "shared_intermediate=%u num_experts=%u experts_per_token=%u norm_topk=%u attention_kind=%u "
                "rms=%p mlp=%p q=%p q_gate=%p k=%p v=%p o=%p gate=%p up=%p down=%p "
                "router=%p expert_gate_up=%p expert_down=%p shared_gate=%p "
                "shared_up=%p shared_down=%p shared_router=%p\n",
                index, layer.mlp_kind, layer.moe_intermediate,
                layer.shared_expert_intermediate,
                layer.num_experts, layer.experts_per_token,
                layer.norm_topk_prob, layer.attention_kind,
                static_cast<const void *>(layer.rms_attn_weight),
                static_cast<const void *>(layer.rms_mlp_weight),
                static_cast<const void *>(layer.w_q),
                static_cast<const void *>(layer.w_q_gate),
                static_cast<const void *>(layer.w_k),
                static_cast<const void *>(layer.w_v),
                static_cast<const void *>(layer.w_o),
                static_cast<const void *>(layer.w_gate),
                static_cast<const void *>(layer.w_up),
                static_cast<const void *>(layer.w_down),
                static_cast<const void *>(layer.w_router),
                static_cast<const void *>(layer.w_expert_gate_up),
                static_cast<const void *>(layer.w_expert_down),
                static_cast<const void *>(layer.w_shared_expert_gate),
                static_cast<const void *>(layer.w_shared_expert_up),
                static_cast<const void *>(layer.w_shared_expert_down),
                static_cast<const void *>(layer.w_shared_expert_router));
      }
      return false;
    }
    const auto &layer = request->layers[index];
    if (layer.mlp_kind == kMlpKindSparseMoe &&
        layer.moe_intermediate > request->intermediate) {
      debug_valid_request_failure("MoE intermediate exceeds scratch capacity");
      return false;
    }
    if (layer.mlp_kind == kMlpKindSparseMoe &&
        layer.shared_expert_intermediate > request->intermediate) {
      debug_valid_request_failure("MoE shared expert intermediate exceeds scratch capacity");
      return false;
    }
  }
  if (declared_weight_plan) {
    if (request->planned_weight_blocks == 0 || request->planned_weight_bytes == 0) {
      debug_valid_request_failure("incomplete declared weight plan");
      return false;
    }
    if (request->planned_weight_descriptors == nullptr ||
        request->planned_weight_descriptor_count != request->planned_weight_blocks ||
        request->planned_weight_descriptor_hash == 0) {
      debug_valid_request_failure("invalid declared descriptor table");
      return false;
    }
    if (request->planned_gpu_resident_blocks > request->planned_weight_blocks ||
        request->planned_gpu_staged_blocks >
            request->planned_weight_blocks - request->planned_gpu_resident_blocks) {
      debug_valid_request_failure("invalid declared block counts");
      return false;
    }
    if (request->planned_gpu_resident_weight_bytes > request->planned_weight_bytes ||
        request->planned_gpu_staged_weight_bytes >
            request->planned_weight_bytes - request->planned_gpu_resident_weight_bytes) {
      debug_valid_request_failure("invalid declared byte counts");
      return false;
    }
  }
  return true;
}

void clear_result(const NervaCudaHfDecodeSequenceRequest *request,
                  NervaCudaHfDecodeSequenceResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
  if (request != nullptr) {
    out->dtype = request->dtype;
    out->hidden = request->hidden;
    out->heads = request->heads;
    out->kv_heads = request->kv_heads;
    out->head_dim = request->head_dim;
    out->intermediate = request->intermediate;
    out->vocab_size = request->vocab_size;
    out->layer_count = request->layer_count;
    out->steps = request->steps;
    out->seed_token = request->seed_token;
    out->planned_weight_blocks = request->planned_weight_blocks;
    out->planned_gpu_resident_blocks = request->planned_gpu_resident_blocks;
    out->planned_gpu_staged_blocks = request->planned_gpu_staged_blocks;
    out->planned_weight_bytes = request->planned_weight_bytes;
    out->planned_gpu_resident_weight_bytes =
        request->planned_gpu_resident_weight_bytes;
    out->planned_gpu_staged_weight_bytes =
        request->planned_gpu_staged_weight_bytes;
    out->planned_weight_descriptor_count =
        request->planned_weight_descriptor_count;
    out->planned_weight_descriptor_hash =
        request->planned_weight_descriptor_hash;
  }
}

void clear_session_create_result(
    const NervaCudaHfDecodeSequenceSessionCreateRequest *request,
    NervaCudaHfDecodeSequenceSessionCreateResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
  if (request != nullptr) {
    out->dtype = request->dtype;
    out->hidden = request->hidden;
    out->heads = request->heads;
    out->kv_heads = request->kv_heads;
    out->head_dim = request->head_dim;
    out->intermediate = request->intermediate;
    out->vocab_size = request->vocab_size;
    out->layer_count = request->layer_count;
    out->max_context_tokens = request->max_context_tokens;
    out->planned_weight_blocks = request->planned_weight_blocks;
    out->planned_gpu_resident_blocks = request->planned_gpu_resident_blocks;
    out->planned_gpu_staged_blocks = request->planned_gpu_staged_blocks;
    out->planned_weight_bytes = request->planned_weight_bytes;
    out->planned_gpu_resident_weight_bytes =
        request->planned_gpu_resident_weight_bytes;
    out->planned_gpu_staged_weight_bytes =
        request->planned_gpu_staged_weight_bytes;
    out->planned_weight_descriptor_count =
        request->planned_weight_descriptor_count;
    out->planned_weight_descriptor_hash =
        request->planned_weight_descriptor_hash;
    out->experimental_rt_decode_requested =
        request->experimental_rt_decode == 0 ? 0u : 1u;
    out->experimental_rt_decode_enabled = 0;
    out->experimental_rt_mode = request->experimental_rt_mode;
    out->experimental_rt_page_tokens = request->experimental_rt_page_tokens;
    out->experimental_rt_pages = request->experimental_rt_pages;
    out->experimental_rt_local_window_tokens =
        request->experimental_rt_local_window_tokens;
    out->experimental_rt_sink_tokens = request->experimental_rt_sink_tokens;
  }
}

template <typename Request, typename Result>
bool validate_weight_descriptors_impl(const Request *request,
                                 uint64_t resident_weight_bytes,
                                 Result *out) {
  if (request->planned_weight_blocks == 0) {
    return true;
  }
  uint64_t cursor = 0;
  uint64_t descriptor_hash = kFnvOffset;
  uint64_t resident_bytes = 0;
  uint64_t staged_bytes = 0;
  uint32_t resident_blocks = 0;
  uint32_t staged_blocks = 0;
  for (uint32_t index = 0; index < request->planned_weight_descriptor_count; ++index) {
    const auto &descriptor = request->planned_weight_descriptors[index];
    if (descriptor.bytes == 0 || descriptor.reserved != 0 ||
        !descriptor_has_source(descriptor) || descriptor.offset_bytes != cursor ||
        descriptor.offset_bytes % sizeof(uint16_t) != 0 ||
        descriptor.bytes % sizeof(uint16_t) != 0) {
      return false;
    }
    const uint64_t next_cursor = cursor + descriptor.bytes;
    if (next_cursor < cursor) {
      return false;
    }
    cursor = next_cursor;
    hash_descriptor(descriptor_hash, descriptor);
    if (descriptor.strategy == kWeightStrategyGpuResident) {
      resident_blocks += 1;
      const uint64_t next_resident_bytes = resident_bytes + descriptor.bytes;
      if (next_resident_bytes < resident_bytes) {
        return false;
      }
      resident_bytes = next_resident_bytes;
    } else if (descriptor.strategy == kWeightStrategyGpuStaged) {
      staged_blocks += 1;
      const uint64_t next_staged_bytes = staged_bytes + descriptor.bytes;
      if (next_staged_bytes < staged_bytes) {
        return false;
      }
      staged_bytes = next_staged_bytes;
    } else {
      return false;
    }
  }
  if (cursor != resident_weight_bytes || cursor != request->planned_weight_bytes ||
      descriptor_hash != request->planned_weight_descriptor_hash ||
      resident_blocks != request->planned_gpu_resident_blocks ||
      staged_blocks != request->planned_gpu_staged_blocks ||
      resident_bytes != request->planned_gpu_resident_weight_bytes ||
      staged_bytes != request->planned_gpu_staged_weight_bytes) {
    return false;
  }
  out->planned_weight_descriptor_hash = descriptor_hash;
  return true;
}


bool should_pack_cublas_weights(uint32_t hidden, uint32_t attention_hidden) {
  return hidden >= 128 && attention_hidden == hidden;
}

PackedProjectionShape packed_projection_shape(uint64_t hidden,
                                              uint64_t attention_hidden,
                                              uint64_t kv_hidden,
                                              uint64_t intermediate) {
  PackedProjectionShape shape{};
  shape.qkv_rows = attention_hidden + kv_hidden * 2;
  shape.gate_up_rows = intermediate * 2;
  shape.qkv_elements_per_layer = shape.qkv_rows * hidden;
  shape.gate_up_elements_per_layer = shape.gate_up_rows * hidden;
  return shape;
}

void pack_layer(SequenceLayerLayout &layout, uint64_t &cursor,
                const NervaCudaHfDecodeChainLayer &layer, uint64_t hidden,
                uint64_t attention_hidden, uint64_t kv_hidden, uint64_t head_dim,
                uint64_t intermediate) {
  layout.mlp_kind = layer.mlp_kind;
  layout.moe_intermediate = layer.moe_intermediate;
  layout.shared_expert_intermediate = layer.shared_expert_intermediate;
  layout.num_experts = layer.num_experts;
  layout.experts_per_token = layer.experts_per_token;
  layout.norm_topk_prob = layer.norm_topk_prob == 0 ? 0u : 1u;
  layout.attention_kind = layer.attention_kind;
  layout.rms_attn = kMissingOffset;
  layout.rms_mlp = kMissingOffset;
  layout.w_q = kMissingOffset;
  layout.w_q_gate = kMissingOffset;
  layout.q_norm = kMissingOffset;
  layout.w_k = kMissingOffset;
  layout.k_norm = kMissingOffset;
  layout.w_v = kMissingOffset;
  layout.w_o = kMissingOffset;
  layout.w_gate = kMissingOffset;
  layout.w_up = kMissingOffset;
  layout.w_down = kMissingOffset;
  layout.w_router = kMissingOffset;
  layout.w_expert_gate_up = kMissingOffset;
  layout.w_expert_down = kMissingOffset;
  layout.w_shared_expert_gate = kMissingOffset;
  layout.w_shared_expert_up = kMissingOffset;
  layout.w_shared_expert_down = kMissingOffset;
  layout.w_shared_expert_router = kMissingOffset;
  layout.w_linear_conv = kMissingOffset;
  layout.w_linear_qkv = kMissingOffset;
  layout.w_linear_z = kMissingOffset;
  layout.w_linear_b = kMissingOffset;
  layout.w_linear_a = kMissingOffset;
  layout.w_linear_dt_bias = kMissingOffset;
  layout.w_linear_a_log = kMissingOffset;
  layout.w_linear_norm = kMissingOffset;
  layout.w_linear_out = kMissingOffset;
  layout.linear_conv_state = kMissingOffset;
  layout.linear_recurrent_state = kMissingOffset;
  layout.linear_key_heads = layer.linear_key_heads;
  layout.linear_value_heads = layer.linear_value_heads;
  layout.linear_key_head_dim = layer.linear_key_head_dim;
  layout.linear_value_head_dim = layer.linear_value_head_dim;
  layout.linear_conv_kernel = layer.linear_conv_kernel;
  if (layer.attention_kind == kAttentionKindLinearGdn) {
    const uint64_t value_dim = linear_gdn_value_dim(layer);
    const uint64_t conv_dim = linear_gdn_conv_dim(layer);
    layout.rms_attn = push(cursor, hidden);
    layout.w_linear_conv =
        push(cursor, conv_dim * layer.linear_conv_kernel);
    layout.w_linear_qkv = push(cursor, conv_dim * hidden);
    layout.w_linear_z = push(cursor, value_dim * hidden);
    layout.w_linear_b = push(cursor, layer.linear_value_heads * hidden);
    layout.w_linear_a = push(cursor, layer.linear_value_heads * hidden);
    layout.w_linear_dt_bias = push(cursor, layer.linear_value_heads);
    layout.w_linear_a_log = push(cursor, layer.linear_value_heads * 2u);
    layout.w_linear_norm = push(cursor, layer.linear_value_head_dim * 2u);
    layout.w_linear_out = push(cursor, hidden * value_dim);
    layout.rms_mlp = push(cursor, hidden);
  } else {
    layout.rms_attn = push(cursor, hidden);
    layout.w_q = push(cursor, attention_hidden * hidden);
    layout.w_q_gate =
        push_optional(cursor, attention_hidden * hidden, layer.w_q_gate);
    layout.q_norm = push_optional(cursor, head_dim, layer.q_norm_weight);
    layout.w_k = push(cursor, kv_hidden * hidden);
    layout.k_norm = push_optional(cursor, head_dim, layer.k_norm_weight);
    layout.w_v = push(cursor, kv_hidden * hidden);
    layout.w_o = push(cursor, hidden * attention_hidden);
    layout.rms_mlp = push(cursor, hidden);
  }
  if (layer.mlp_kind == kMlpKindSparseMoe) {
    const uint64_t moe_intermediate = layer.moe_intermediate;
    const uint64_t num_experts = layer.num_experts;
    const uint64_t shared_intermediate = layer.shared_expert_intermediate;
    layout.w_router = push(cursor, num_experts * hidden);
    layout.w_expert_gate_up =
        push(cursor, num_experts * moe_intermediate * 2u * hidden);
    layout.w_expert_down =
        push(cursor, num_experts * hidden * moe_intermediate);
    if (shared_intermediate != 0) {
      layout.w_shared_expert_gate = push(cursor, shared_intermediate * hidden);
      layout.w_shared_expert_up = push(cursor, shared_intermediate * hidden);
      layout.w_shared_expert_down = push(cursor, hidden * shared_intermediate);
      layout.w_shared_expert_router = push(cursor, hidden);
    }
  } else {
    layout.w_gate = push(cursor, intermediate * hidden);
    layout.w_up = push(cursor, intermediate * hidden);
    layout.w_down = push(cursor, hidden * intermediate);
  }
  layout.q_bias = push_optional(cursor, attention_hidden, layer.q_bias);
  layout.k_bias = push_optional(cursor, kv_hidden, layer.k_bias);
  layout.v_bias = push_optional(cursor, kv_hidden, layer.v_bias);
  layout.o_bias = push_optional(cursor, hidden, layer.o_bias);
}

uint64_t linear_gdn_conv_state_elements(const SequenceLayerLayout &layout) {
  if (layout.attention_kind != kAttentionKindLinearGdn ||
      layout.linear_conv_kernel <= 1) {
    return 0;
  }
  const uint64_t key_dim =
      static_cast<uint64_t>(layout.linear_key_heads) *
      layout.linear_key_head_dim;
  const uint64_t value_dim =
      static_cast<uint64_t>(layout.linear_value_heads) *
      layout.linear_value_head_dim;
  const uint64_t conv_dim = key_dim * 2u + value_dim;
  return conv_dim * (layout.linear_conv_kernel - 1u);
}

uint64_t linear_gdn_recurrent_state_elements(const SequenceLayerLayout &layout) {
  if (layout.attention_kind != kAttentionKindLinearGdn) {
    return 0;
  }
  return static_cast<uint64_t>(layout.linear_value_heads) *
         layout.linear_value_head_dim * layout.linear_key_head_dim;
}

void assign_linear_gdn_state_offsets(std::vector<SequenceLayerLayout> &layouts,
                                     uint64_t *conv_state_elements,
                                     uint64_t *recurrent_state_elements) {
  uint64_t conv_cursor = 0;
  uint64_t recurrent_cursor = 0;
  for (SequenceLayerLayout &layout : layouts) {
    layout.linear_conv_state = kMissingOffset;
    layout.linear_recurrent_state = kMissingOffset;
    const uint64_t conv_elements = linear_gdn_conv_state_elements(layout);
    if (conv_elements != 0) {
      layout.linear_conv_state = conv_cursor;
      conv_cursor += conv_elements;
    }
    const uint64_t recurrent_elements =
        linear_gdn_recurrent_state_elements(layout);
    if (recurrent_elements != 0) {
      layout.linear_recurrent_state = recurrent_cursor;
      recurrent_cursor += recurrent_elements;
    }
  }
  if (conv_state_elements != nullptr) {
    *conv_state_elements = conv_cursor;
  }
  if (recurrent_state_elements != nullptr) {
    *recurrent_state_elements = recurrent_cursor;
  }
}

void copy_optional(uint16_t *arena, uint64_t offset, const uint16_t *src, uint64_t elements) {
  if (src != nullptr) {
    memcpy(arena + offset, src, elements * sizeof(uint16_t));
  }
}

void copy_layer(uint16_t *arena, const SequenceLayerLayout &layout,
                const NervaCudaHfDecodeChainLayer &layer, uint64_t hidden,
                uint64_t attention_hidden, uint64_t kv_hidden, uint64_t head_dim,
                uint64_t intermediate) {
  memcpy(arena + layout.rms_attn, layer.rms_attn_weight, hidden * sizeof(uint16_t));
  memcpy(arena + layout.rms_mlp, layer.rms_mlp_weight, hidden * sizeof(uint16_t));
  if (layout.attention_kind == kAttentionKindLinearGdn) {
    const uint64_t value_dim =
        static_cast<uint64_t>(layout.linear_value_heads) *
        layout.linear_value_head_dim;
    const uint64_t conv_dim =
        (static_cast<uint64_t>(layout.linear_key_heads) *
         layout.linear_key_head_dim * 2u) +
        value_dim;
    memcpy(arena + layout.w_linear_conv, layer.w_linear_conv,
           conv_dim * layout.linear_conv_kernel * sizeof(uint16_t));
    memcpy(arena + layout.w_linear_qkv, layer.w_linear_qkv,
           conv_dim * hidden * sizeof(uint16_t));
    memcpy(arena + layout.w_linear_z, layer.w_linear_z,
           value_dim * hidden * sizeof(uint16_t));
    memcpy(arena + layout.w_linear_b, layer.w_linear_b,
           static_cast<uint64_t>(layout.linear_value_heads) * hidden *
               sizeof(uint16_t));
    memcpy(arena + layout.w_linear_a, layer.w_linear_a,
           static_cast<uint64_t>(layout.linear_value_heads) * hidden *
               sizeof(uint16_t));
    memcpy(arena + layout.w_linear_dt_bias, layer.w_linear_dt_bias,
           static_cast<uint64_t>(layout.linear_value_heads) * sizeof(uint16_t));
    memcpy(arena + layout.w_linear_a_log, layer.w_linear_a_log,
           static_cast<uint64_t>(layout.linear_value_heads) * sizeof(float));
    memcpy(arena + layout.w_linear_norm, layer.w_linear_norm,
           static_cast<uint64_t>(layout.linear_value_head_dim) * sizeof(float));
    memcpy(arena + layout.w_linear_out, layer.w_linear_out,
           hidden * value_dim * sizeof(uint16_t));
  } else {
  memcpy(arena + layout.w_q, layer.w_q, attention_hidden * hidden * sizeof(uint16_t));
  copy_optional(arena, layout.w_q_gate, layer.w_q_gate,
                attention_hidden * hidden);
  copy_optional(arena, layout.q_norm, layer.q_norm_weight, head_dim);
  memcpy(arena + layout.w_k, layer.w_k, kv_hidden * hidden * sizeof(uint16_t));
  copy_optional(arena, layout.k_norm, layer.k_norm_weight, head_dim);
  memcpy(arena + layout.w_v, layer.w_v, kv_hidden * hidden * sizeof(uint16_t));
  memcpy(arena + layout.w_o, layer.w_o, hidden * attention_hidden * sizeof(uint16_t));
  copy_optional(arena, layout.q_bias, layer.q_bias, attention_hidden);
  copy_optional(arena, layout.k_bias, layer.k_bias, kv_hidden);
  copy_optional(arena, layout.v_bias, layer.v_bias, kv_hidden);
  copy_optional(arena, layout.o_bias, layer.o_bias, hidden);
  }
  if (layout.mlp_kind == kMlpKindSparseMoe) {
    const uint64_t moe_intermediate = layout.moe_intermediate;
    const uint64_t num_experts = layout.num_experts;
    const uint64_t shared_intermediate = layout.shared_expert_intermediate;
    memcpy(arena + layout.w_router, layer.w_router,
           num_experts * hidden * sizeof(uint16_t));
    memcpy(arena + layout.w_expert_gate_up, layer.w_expert_gate_up,
           num_experts * moe_intermediate * 2u * hidden * sizeof(uint16_t));
    memcpy(arena + layout.w_expert_down, layer.w_expert_down,
           num_experts * hidden * moe_intermediate * sizeof(uint16_t));
    if (shared_intermediate != 0) {
      memcpy(arena + layout.w_shared_expert_gate, layer.w_shared_expert_gate,
             shared_intermediate * hidden * sizeof(uint16_t));
      memcpy(arena + layout.w_shared_expert_up, layer.w_shared_expert_up,
             shared_intermediate * hidden * sizeof(uint16_t));
      memcpy(arena + layout.w_shared_expert_down, layer.w_shared_expert_down,
             hidden * shared_intermediate * sizeof(uint16_t));
      memcpy(arena + layout.w_shared_expert_router, layer.w_shared_expert_router,
             hidden * sizeof(uint16_t));
    }
  } else {
    memcpy(arena + layout.w_gate, layer.w_gate,
           intermediate * hidden * sizeof(uint16_t));
    memcpy(arena + layout.w_up, layer.w_up,
           intermediate * hidden * sizeof(uint16_t));
    memcpy(arena + layout.w_down, layer.w_down,
           hidden * intermediate * sizeof(uint16_t));
  }
}

bool descriptor_destination_bytes(
    const NervaCudaHfDecodeSequenceWeightBlock &descriptor,
    uint64_t arena_bytes, uint64_t embedding_bytes, uint64_t scratch_gap_bytes,
    uint64_t *destination_bytes) {
  if (descriptor.offset_bytes % sizeof(uint16_t) != 0 ||
      descriptor.bytes % sizeof(uint16_t) != 0) {
    return false;
  }
  uint64_t translated = descriptor.offset_bytes;
  if (translated >= embedding_bytes) {
    translated += scratch_gap_bytes;
  }
  if (translated > arena_bytes || descriptor.bytes > arena_bytes - translated) {
    return false;
  }
  *destination_bytes = translated;
  return true;
}

struct NativeLoadProgress {
  std::chrono::steady_clock::time_point start;
  uint32_t last_percent;
};

void report_native_load_progress(uint64_t done, uint64_t total,
                                 NativeLoadProgress *progress) {
  const char *mode = getenv("NERVA_NATIVE_LOAD_PROGRESS");
  if (mode != nullptr && strcmp(mode, "quiet") == 0) {
    return;
  }
  if (total == 0 || progress == nullptr) {
    return;
  }
  const uint32_t percent =
      done >= total ? 100u : static_cast<uint32_t>((done * 100u) / total);
  const uint32_t displayed_percent =
      percent >= 100u ? 100u : (percent / 5u) * 5u;
  if (displayed_percent == progress->last_percent) {
    return;
  }
  const auto now = std::chrono::steady_clock::now();
  const double elapsed_s =
      std::chrono::duration<double>(now - progress->start).count();
  const double done_gb = static_cast<double>(done) / 1000000000.0;
  const double total_gb = static_cast<double>(total) / 1000000000.0;
  const double gb_s = elapsed_s > 0.0 ? done_gb / elapsed_s : 0.0;
  if (mode != nullptr && strcmp(mode, "color") == 0) {
    fprintf(stderr,
            "\x1b[2m[nerva-load]\x1b[0m "
            "\x1b[38;2;255;106;42mweights H2D\x1b[0m "
            "\x1b[38;2;112;223;158m%3u%%\x1b[0m  "
            "%.2f/%.2f GB  \x1b[38;2;87;190;255m%.2f GB/s\x1b[0m\n",
            displayed_percent, done_gb, total_gb, gb_s);
  } else if (mode != nullptr && strcmp(mode, "ansi") == 0) {
    fprintf(stderr,
            "\x1b[2m[nerva-load]\x1b[0m "
            "\x1b[93mweights H2D\x1b[0m "
            "\x1b[92m%3u%%\x1b[0m  "
            "%.2f/%.2f GB  \x1b[96m%.2f GB/s\x1b[0m\n",
            displayed_percent, done_gb, total_gb, gb_s);
  } else {
    fprintf(stderr,
            "[nerva-load] weights H2D %3u%%  %.2f/%.2f GB  %.2f GB/s\n",
            displayed_percent, done_gb, total_gb, gb_s);
  }
  fflush(stderr);
  progress->last_percent = displayed_percent;
}

cudaError_t copy_file_descriptor_to_device(
    uint16_t *device_destination, uint16_t *staging, uint64_t staging_bytes,
    const NervaCudaHfDecodeSequenceWeightBlock &descriptor, cudaStream_t stream,
    uint64_t *setup_sync_calls, uint64_t *progress_done,
    uint64_t progress_total, NativeLoadProgress *progress) {
  if (device_destination == nullptr || staging == nullptr || staging_bytes == 0 ||
      staging_bytes % sizeof(uint16_t) != 0 ||
      !descriptor_has_file_source(descriptor)) {
    return cudaErrorInvalidValue;
  }
  std::string path(descriptor.source_file, descriptor.source_file_len);
  FILE *file = fopen(path.c_str(), "rb");
  if (file == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (fseek(file, static_cast<long>(descriptor.file_offset_begin), SEEK_SET) != 0) {
    fclose(file);
    return cudaErrorInvalidValue;
  }
  uint64_t remaining = descriptor.bytes;
  uint64_t destination_offset_elements = 0;
  while (remaining != 0) {
    const uint64_t chunk_bytes = std::min(remaining, staging_bytes);
    const size_t read = fread(staging, 1, static_cast<size_t>(chunk_bytes), file);
    if (read != static_cast<size_t>(chunk_bytes)) {
      fclose(file);
      return cudaErrorInvalidValue;
    }
    cudaError_t err = cudaMemcpyAsync(
        device_destination + destination_offset_elements, staging, chunk_bytes,
        cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
      fclose(file);
      return err;
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
      fclose(file);
      return err;
    }
    if (setup_sync_calls != nullptr) {
      *setup_sync_calls += 1;
    }
    remaining -= chunk_bytes;
    destination_offset_elements += chunk_bytes / sizeof(uint16_t);
    if (progress_done != nullptr) {
      *progress_done += chunk_bytes;
      report_native_load_progress(*progress_done, progress_total, progress);
    }
  }
  fclose(file);
  return cudaSuccess;
}

template <typename Request, typename Result>
cudaError_t copy_weight_descriptors_to_device_impl(
    uint16_t *device_arena, uint16_t *staging, uint64_t staging_bytes,
    const Request *request, uint64_t arena_bytes,
    uint64_t embedding_bytes, uint64_t scratch_gap_bytes, cudaStream_t stream,
    Result *out, uint64_t *setup_sync_calls) {
  uint64_t progress_done = 0;
  NativeLoadProgress progress = {std::chrono::steady_clock::now(), UINT32_MAX};
  const bool report_progress = descriptors_require_file_staging_impl(request);
  if (report_progress) {
    report_native_load_progress(0, request->planned_weight_bytes, &progress);
  }
  for (uint32_t index = 0; index < request->planned_weight_descriptor_count; ++index) {
    const auto &descriptor = request->planned_weight_descriptors[index];
    uint64_t destination_bytes = 0;
    if (!descriptor_destination_bytes(descriptor, arena_bytes, embedding_bytes,
                                      scratch_gap_bytes, &destination_bytes)) {
      return cudaErrorInvalidValue;
    }
    uint16_t *destination = device_arena + destination_bytes / sizeof(uint16_t);
    if (descriptor_has_file_source(descriptor)) {
      cudaError_t err = copy_file_descriptor_to_device(
          destination, staging, staging_bytes, descriptor, stream, setup_sync_calls,
          &progress_done, request->planned_weight_bytes,
          &progress);
      if (err != cudaSuccess) {
        return err;
      }
    } else if (descriptor_has_memory_source(descriptor)) {
      cudaError_t err = cudaMemcpyAsync(destination, descriptor.host_source,
                                        descriptor.bytes, cudaMemcpyHostToDevice,
                                        stream);
      if (err != cudaSuccess) {
        return err;
      }
      if (report_progress) {
        progress_done += descriptor.bytes;
        report_native_load_progress(progress_done, request->planned_weight_bytes,
                                    &progress);
      }
    } else {
      return cudaErrorInvalidValue;
    }
    out->h2d_bytes += descriptor.bytes;
    if (descriptor.strategy == kWeightStrategyGpuResident) {
      out->descriptor_gpu_resident_h2d_bytes += descriptor.bytes;
    } else if (descriptor.strategy == kWeightStrategyGpuStaged) {
      out->descriptor_gpu_staged_h2d_bytes += descriptor.bytes;
    }
  }
  if (report_progress) {
    report_native_load_progress(request->planned_weight_bytes,
                                request->planned_weight_bytes,
                                &progress);
  }
  return cudaSuccess;
}


bool has_declared_weight_plan(const NervaCudaHfDecodeSequenceRequest *request) {
  return has_declared_weight_plan_impl(request);
}

bool has_declared_weight_plan(
    const NervaCudaHfDecodeSequenceSessionCreateRequest *request) {
  return has_declared_weight_plan_impl(request);
}

uint64_t pinned_weight_staging_bytes(
    const NervaCudaHfDecodeSequenceRequest *request,
    uint64_t full_weight_bytes) {
  return pinned_weight_staging_bytes_impl(request, full_weight_bytes);
}

uint64_t pinned_weight_staging_bytes(
    const NervaCudaHfDecodeSequenceSessionCreateRequest *request,
    uint64_t full_weight_bytes) {
  return pinned_weight_staging_bytes_impl(request, full_weight_bytes);
}

bool validate_weight_descriptors(const NervaCudaHfDecodeSequenceRequest *request,
                                 uint64_t resident_weight_bytes,
                                 NervaCudaHfDecodeSequenceResult *out) {
  return validate_weight_descriptors_impl(request, resident_weight_bytes, out);
}

bool validate_weight_descriptors(
    const NervaCudaHfDecodeSequenceSessionCreateRequest *request,
    uint64_t resident_weight_bytes,
    NervaCudaHfDecodeSequenceSessionCreateResult *out) {
  return validate_weight_descriptors_impl(request, resident_weight_bytes, out);
}

cudaError_t copy_weight_descriptors_to_device(
    uint16_t *device_arena, uint16_t *staging, uint64_t staging_bytes,
    const NervaCudaHfDecodeSequenceRequest *request, uint64_t arena_bytes,
    uint64_t embedding_bytes, uint64_t scratch_gap_bytes, cudaStream_t stream,
    NervaCudaHfDecodeSequenceResult *out, uint64_t *setup_sync_calls) {
  return copy_weight_descriptors_to_device_impl(
      device_arena, staging, staging_bytes, request, arena_bytes,
      embedding_bytes, scratch_gap_bytes, stream, out, setup_sync_calls);
}

cudaError_t copy_weight_descriptors_to_device(
    uint16_t *device_arena, uint16_t *staging, uint64_t staging_bytes,
    const NervaCudaHfDecodeSequenceSessionCreateRequest *request,
    uint64_t arena_bytes, uint64_t embedding_bytes, uint64_t scratch_gap_bytes,
    cudaStream_t stream, NervaCudaHfDecodeSequenceSessionCreateResult *out,
    uint64_t *setup_sync_calls) {
  return copy_weight_descriptors_to_device_impl(
      device_arena, staging, staging_bytes, request, arena_bytes,
      embedding_bytes, scratch_gap_bytes, stream, out, setup_sync_calls);
}
