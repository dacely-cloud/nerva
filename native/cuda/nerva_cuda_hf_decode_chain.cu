#include "nerva_cuda_api.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include <vector>

namespace {

constexpr uint32_t kDTypeF16 = 0;
constexpr uint32_t kDTypeBF16 = 1;
constexpr uint32_t kRequestId = 1;
constexpr uint32_t kSequenceId = 1;
constexpr uint32_t kCompletionDeviceComplete = 1;
constexpr uint64_t kMissingOffset = UINT64_MAX;
constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ull;
constexpr uint64_t kFnvPrime = 0x00000100000001b3ull;

struct ChainArenaLayout {
  uint64_t input;
  uint64_t scratch;
  uint64_t final_norm;
  uint64_t lm_head;
};

struct ChainLayerLayout {
  uint64_t rms_attn;
  uint64_t rms_mlp;
  uint64_t w_q;
  uint64_t w_k;
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

__device__ float encoded_to_f32(uint16_t value, uint32_t dtype) {
  if (dtype == kDTypeBF16) {
    return __uint_as_float(static_cast<uint32_t>(value) << 16);
  }
  return __half2float(__ushort_as_half(value));
}

__device__ uint16_t f32_to_encoded(float value, uint32_t dtype) {
  if (dtype == kDTypeBF16) {
    uint32_t bits = __float_as_uint(value);
    uint32_t lsb = (bits >> 16) & 1u;
    return static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16);
  }
  return __half_as_ushort(__float2half_rn(value));
}

__device__ float silu(float value) {
  return value / (1.0f + expf(-value));
}

__device__ void mat_vec(const uint16_t *matrix, const float *input, uint32_t rows,
                        uint32_t cols, uint32_t dtype, float *output) {
  for (uint32_t row = 0; row < rows; ++row) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < cols; ++col) {
      sum += encoded_to_f32(matrix[row * cols + col], dtype) * input[col];
    }
    output[row] = sum;
  }
}

__device__ void rms_norm(const float *input, const uint16_t *weight, uint32_t hidden,
                         uint32_t dtype, float eps, float *output) {
  float mean_square = 0.0f;
  for (uint32_t index = 0; index < hidden; ++index) {
    mean_square += input[index] * input[index];
  }
  const float scale = rsqrtf(mean_square / static_cast<float>(hidden) + eps);
  for (uint32_t index = 0; index < hidden; ++index) {
    output[index] = input[index] * scale * encoded_to_f32(weight[index], dtype);
  }
}

__device__ void add_bias(const uint16_t *arena, uint64_t offset, uint32_t len,
                         uint32_t dtype, float *output) {
  if (offset == kMissingOffset) {
    return;
  }
  const uint16_t *bias = arena + offset;
  for (uint32_t index = 0; index < len; ++index) {
    output[index] += encoded_to_f32(bias[index], dtype);
  }
}

__device__ void apply_rope(float *values, uint32_t heads, uint32_t head_dim,
                           uint32_t position, float theta) {
  if (theta <= 0.0f) {
    return;
  }
  const uint32_t half = head_dim / 2;
  for (uint32_t head = 0; head < heads; ++head) {
    const uint32_t start = head * head_dim;
    for (uint32_t offset = 0; offset < half; ++offset) {
      const uint32_t first = start + offset;
      const uint32_t second = first + half;
      const float exponent = static_cast<float>(2 * offset) / static_cast<float>(head_dim);
      float angle = static_cast<float>(position) / powf(theta, exponent);
      float sin_value = 0.0f;
      float cos_value = 0.0f;
      sincosf(angle, &sin_value, &cos_value);
      const float left = values[first];
      const float right = values[second];
      values[first] = left * cos_value - right * sin_value;
      values[second] = right * cos_value + left * sin_value;
    }
  }
}

__device__ void run_layer(uint16_t *arena, ChainLayerLayout layout, uint64_t input_offset,
                          uint64_t output_offset, uint32_t dtype, uint32_t hidden,
                          uint32_t heads, uint32_t kv_heads, uint32_t head_dim,
                          uint32_t intermediate, uint32_t position, float rms_eps,
                          float rope_theta, float *scratch) {
  const uint32_t attention_hidden = heads * head_dim;
  const uint32_t kv_hidden = kv_heads * head_dim;
  float *input = scratch;
  float *attn_norm = input + hidden;
  float *q = attn_norm + hidden;
  float *k = q + attention_hidden;
  float *v = k + kv_hidden;
  float *attn = v + kv_hidden;
  float *residual = attn + attention_hidden;
  float *mlp_norm = residual + hidden;
  float *gate = mlp_norm + hidden;
  float *up = gate + intermediate;
  float *ff = up + intermediate;
  float *down = ff + intermediate;

  for (uint32_t index = 0; index < hidden; ++index) {
    input[index] = encoded_to_f32(arena[input_offset + index], dtype);
  }
  rms_norm(input, arena + layout.rms_attn, hidden, dtype, rms_eps, attn_norm);
  mat_vec(arena + layout.w_q, attn_norm, attention_hidden, hidden, dtype, q);
  mat_vec(arena + layout.w_k, attn_norm, kv_hidden, hidden, dtype, k);
  mat_vec(arena + layout.w_v, attn_norm, kv_hidden, hidden, dtype, v);
  add_bias(arena, layout.q_bias, attention_hidden, dtype, q);
  add_bias(arena, layout.k_bias, kv_hidden, dtype, k);
  add_bias(arena, layout.v_bias, kv_hidden, dtype, v);
  apply_rope(q, heads, head_dim, position, rope_theta);
  apply_rope(k, kv_heads, head_dim, position, rope_theta);

  for (uint32_t head = 0; head < heads; ++head) {
    const uint32_t kv_head = head / (heads / kv_heads);
    for (uint32_t offset = 0; offset < head_dim; ++offset) {
      attn[head * head_dim + offset] = v[kv_head * head_dim + offset];
    }
  }
  mat_vec(arena + layout.w_o, attn, hidden, attention_hidden, dtype, residual);
  add_bias(arena, layout.o_bias, hidden, dtype, residual);
  for (uint32_t index = 0; index < hidden; ++index) {
    residual[index] += input[index];
  }

  rms_norm(residual, arena + layout.rms_mlp, hidden, dtype, rms_eps, mlp_norm);
  mat_vec(arena + layout.w_gate, mlp_norm, intermediate, hidden, dtype, gate);
  mat_vec(arena + layout.w_up, mlp_norm, intermediate, hidden, dtype, up);
  for (uint32_t index = 0; index < intermediate; ++index) {
    ff[index] = silu(gate[index]) * up[index];
  }
  mat_vec(arena + layout.w_down, ff, hidden, intermediate, dtype, down);
  for (uint32_t index = 0; index < hidden; ++index) {
    arena[output_offset + index] = f32_to_encoded(residual[index] + down[index], dtype);
  }
}

__global__ void hf_decode_chain_kernel(
    uint16_t *arena, ChainArenaLayout arena_layout, ChainLayerLayout *layers,
    uint32_t layer_count, uint32_t dtype, uint32_t hidden, uint32_t heads,
    uint32_t kv_heads, uint32_t head_dim, uint32_t intermediate, uint32_t vocab_size,
    uint32_t position, uint64_t token_index, float rms_eps, float rope_theta,
    float *scratch, NervaCudaSyntheticTokenSlot *slot) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  uint64_t input_offset = arena_layout.input;
  uint64_t output_offset = arena_layout.scratch;
  for (uint32_t layer_index = 0; layer_index < layer_count; ++layer_index) {
    run_layer(arena, layers[layer_index], input_offset, output_offset, dtype, hidden,
              heads, kv_heads, head_dim, intermediate, position, rms_eps, rope_theta,
              scratch);
    const uint64_t next_input = output_offset;
    output_offset = input_offset;
    input_offset = next_input;
  }

  float *decoded = scratch;
  float *final_norm = decoded + hidden;
  float *logits = final_norm + hidden;
  for (uint32_t index = 0; index < hidden; ++index) {
    decoded[index] = encoded_to_f32(arena[input_offset + index], dtype);
  }
  rms_norm(decoded, arena + arena_layout.final_norm, hidden, dtype, rms_eps, final_norm);
  mat_vec(arena + arena_layout.lm_head, final_norm, vocab_size, hidden, dtype, logits);
  uint32_t best_index = 0;
  float best_value = logits[0];
  for (uint32_t index = 1; index < vocab_size; ++index) {
    const float value = logits[index];
    if (isfinite(value) && value > best_value) {
      best_value = value;
      best_index = index;
    }
  }

  slot->request_id = kRequestId;
  slot->sequence_id = kSequenceId;
  slot->token_index = token_index;
  slot->token = best_index;
  slot->version = slot->version + 1ull;
  slot->completion = kCompletionDeviceComplete;
  slot->host_copied = 0;
}

uint64_t hash_token(uint32_t token) {
  uint64_t hash = kFnvOffset;
  for (uint32_t byte = 0; byte < 4; ++byte) {
    hash ^= static_cast<uint64_t>((token >> (8u * byte)) & 0xffu);
    hash *= kFnvPrime;
  }
  return hash;
}

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

bool valid_layer(const NervaCudaHfDecodeChainLayer &layer) {
  return layer.rms_attn_weight != nullptr && layer.rms_mlp_weight != nullptr &&
         layer.w_q != nullptr && layer.w_k != nullptr && layer.w_v != nullptr &&
         layer.w_o != nullptr && layer.w_gate != nullptr && layer.w_up != nullptr &&
         layer.w_down != nullptr;
}

bool valid_request(const NervaCudaHfDecodeChainRequest *request) {
  if (request == nullptr || request->input == nullptr || request->layers == nullptr ||
      request->final_norm_weight == nullptr || request->lm_head == nullptr ||
      request->layer_count == 0 || request->hidden == 0 || request->heads == 0 ||
      request->kv_heads == 0 || request->head_dim == 0 ||
      request->intermediate == 0 || request->vocab_size == 0 ||
      request->kv_heads > request->heads || request->heads % request->kv_heads != 0 ||
      request->dtype > kDTypeBF16 ||
      (request->rope_theta > 0.0f && request->head_dim % 2 != 0)) {
    return false;
  }
  for (uint32_t index = 0; index < request->layer_count; ++index) {
    if (!valid_layer(request->layers[index])) {
      return false;
    }
  }
  return true;
}

void clear_result(const NervaCudaHfDecodeChainRequest *request,
                  NervaCudaHfDecodeChainResult *out) {
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
    out->token_index = request->token_index;
  }
}

int fail(NervaCudaHfDecodeChainResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  return -1;
}

void pack_layer(ChainLayerLayout &layout, uint64_t &cursor,
                const NervaCudaHfDecodeChainLayer &layer, uint64_t hidden,
                uint64_t attention_hidden, uint64_t kv_hidden, uint64_t intermediate) {
  layout.rms_attn = push(cursor, hidden);
  layout.rms_mlp = push(cursor, hidden);
  layout.w_q = push(cursor, attention_hidden * hidden);
  layout.w_k = push(cursor, kv_hidden * hidden);
  layout.w_v = push(cursor, kv_hidden * hidden);
  layout.w_o = push(cursor, hidden * attention_hidden);
  layout.q_bias = push_optional(cursor, attention_hidden, layer.q_bias);
  layout.k_bias = push_optional(cursor, kv_hidden, layer.k_bias);
  layout.v_bias = push_optional(cursor, kv_hidden, layer.v_bias);
  layout.o_bias = push_optional(cursor, hidden, layer.o_bias);
  layout.w_gate = push(cursor, intermediate * hidden);
  layout.w_up = push(cursor, intermediate * hidden);
  layout.w_down = push(cursor, hidden * intermediate);
}

void copy_optional(uint16_t *arena, uint64_t offset, const uint16_t *src, uint64_t elements) {
  if (src != nullptr) {
    memcpy(arena + offset, src, elements * sizeof(uint16_t));
  }
}

void copy_layer(uint16_t *arena, const ChainLayerLayout &layout,
                const NervaCudaHfDecodeChainLayer &layer, uint64_t hidden,
                uint64_t attention_hidden, uint64_t kv_hidden, uint64_t intermediate) {
  memcpy(arena + layout.rms_attn, layer.rms_attn_weight, hidden * sizeof(uint16_t));
  memcpy(arena + layout.rms_mlp, layer.rms_mlp_weight, hidden * sizeof(uint16_t));
  memcpy(arena + layout.w_q, layer.w_q, attention_hidden * hidden * sizeof(uint16_t));
  memcpy(arena + layout.w_k, layer.w_k, kv_hidden * hidden * sizeof(uint16_t));
  memcpy(arena + layout.w_v, layer.w_v, kv_hidden * hidden * sizeof(uint16_t));
  memcpy(arena + layout.w_o, layer.w_o, hidden * attention_hidden * sizeof(uint16_t));
  copy_optional(arena, layout.q_bias, layer.q_bias, attention_hidden);
  copy_optional(arena, layout.k_bias, layer.k_bias, kv_hidden);
  copy_optional(arena, layout.v_bias, layer.v_bias, kv_hidden);
  copy_optional(arena, layout.o_bias, layer.o_bias, hidden);
  memcpy(arena + layout.w_gate, layer.w_gate, intermediate * hidden * sizeof(uint16_t));
  memcpy(arena + layout.w_up, layer.w_up, intermediate * hidden * sizeof(uint16_t));
  memcpy(arena + layout.w_down, layer.w_down, hidden * intermediate * sizeof(uint16_t));
}

}  // namespace

extern "C" int nerva_cuda_hf_decode_chain_u16(
    const NervaCudaHfDecodeChainRequest *request,
    NervaCudaHfDecodeChainResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_result(request, out);
  if (!valid_request(request)) {
    return -1;
  }
  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  if (out->device_count <= 0) {
    out->cuda_error = static_cast<int32_t>(cudaErrorNoDevice);
    return -1;
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail(out, err);
  }

  const uint64_t hidden = request->hidden;
  const uint64_t attention_hidden = request->heads * request->head_dim;
  const uint64_t kv_hidden = request->kv_heads * request->head_dim;
  const uint64_t intermediate = request->intermediate;
  const uint64_t vocab_size = request->vocab_size;
  ChainArenaLayout arena_layout{};
  std::vector<ChainLayerLayout> layouts(request->layer_count);
  uint64_t elements = 0;
  arena_layout.input = push(elements, hidden);
  arena_layout.scratch = push(elements, hidden);
  for (uint32_t index = 0; index < request->layer_count; ++index) {
    pack_layer(layouts[index], elements, request->layers[index], hidden,
               attention_hidden, kv_hidden, intermediate);
  }
  arena_layout.final_norm = push(elements, hidden);
  arena_layout.lm_head = push(elements, vocab_size * hidden);
  const uint64_t arena_bytes = elements * sizeof(uint16_t);
  const uint64_t layout_bytes = layouts.size() * sizeof(ChainLayerLayout);
  const uint64_t block_scratch =
      hidden * 4 + attention_hidden * 2 + kv_hidden * 2 + intermediate * 4;
  const uint64_t scratch_bytes = (block_scratch + hidden + vocab_size) * sizeof(float);

  uint16_t *host_arena = nullptr;
  uint16_t *device_arena = nullptr;
  ChainLayerLayout *device_layouts = nullptr;
  float *device_scratch = nullptr;
  NervaCudaSyntheticTokenSlot *host_slot = nullptr;
  NervaCudaSyntheticTokenSlot *device_slot = nullptr;
  cudaStream_t stream = nullptr;

  err = cudaHostAlloc(reinterpret_cast<void **>(&host_arena), arena_bytes, cudaHostAllocDefault);
  if (err == cudaSuccess)
    err = cudaHostAlloc(reinterpret_cast<void **>(&host_slot), sizeof(*host_slot),
                        cudaHostAllocDefault);
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void **>(&device_arena), arena_bytes);
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void **>(&device_layouts), layout_bytes);
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void **>(&device_scratch), scratch_bytes);
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void **>(&device_slot), sizeof(*device_slot));
  if (err == cudaSuccess) err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) {
    fail(out, err);
    cudaFree(device_slot);
    cudaFree(device_scratch);
    cudaFree(device_layouts);
    cudaFree(device_arena);
    cudaFreeHost(host_slot);
    cudaFreeHost(host_arena);
    return -1;
  }

  memset(host_arena, 0, arena_bytes);
  memset(host_slot, 0, sizeof(*host_slot));
  memcpy(host_arena + arena_layout.input, request->input, hidden * sizeof(uint16_t));
  for (uint32_t index = 0; index < request->layer_count; ++index) {
    copy_layer(host_arena, layouts[index], request->layers[index], hidden,
               attention_hidden, kv_hidden, intermediate);
  }
  memcpy(host_arena + arena_layout.final_norm, request->final_norm_weight,
         hidden * sizeof(uint16_t));
  memcpy(host_arena + arena_layout.lm_head, request->lm_head,
         vocab_size * hidden * sizeof(uint16_t));

  err = cudaMemcpyAsync(device_arena, host_arena, arena_bytes, cudaMemcpyHostToDevice, stream);
  if (err == cudaSuccess) {
    out->h2d_bytes = arena_bytes;
    err = cudaMemcpyAsync(device_layouts, layouts.data(), layout_bytes,
                          cudaMemcpyHostToDevice, stream);
    out->h2d_bytes += layout_bytes;
  }
  if (err == cudaSuccess) {
    err = cudaMemsetAsync(device_slot, 0, sizeof(*device_slot), stream);
  }
  if (err == cudaSuccess) {
    hf_decode_chain_kernel<<<1, 1, 0, stream>>>(
        device_arena, arena_layout, device_layouts, request->layer_count, request->dtype,
        request->hidden, request->heads, request->kv_heads, request->head_dim,
        request->intermediate, request->vocab_size, request->position,
        request->token_index, request->rms_eps, request->rope_theta, device_scratch,
        device_slot);
    out->kernel_launches = 1;
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cudaMemcpyAsync(host_slot, device_slot, sizeof(*host_slot),
                          cudaMemcpyDeviceToHost, stream);
    out->d2h_bytes = sizeof(*host_slot);
  }
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
    out->sync_calls = 1;
  }

  if (err == cudaSuccess) {
    out->token_index = host_slot->token_index;
    out->token = host_slot->token;
    out->slot_version = host_slot->version;
    out->completion = host_slot->completion;
    out->output_hash = hash_token(host_slot->token);
    out->resident_weight_bytes = arena_bytes - (hidden * 2 * sizeof(uint16_t));
    out->device_arena_bytes = arena_bytes + layout_bytes + scratch_bytes + sizeof(*device_slot);
    out->pinned_host_bytes = arena_bytes + sizeof(*host_slot);
    out->status = host_slot->request_id == kRequestId &&
                          host_slot->sequence_id == kSequenceId &&
                          host_slot->completion == kCompletionDeviceComplete
                      ? 0
                      : -1;
  } else {
    fail(out, err);
  }

  cudaStreamDestroy(stream);
  cudaFree(device_slot);
  cudaFree(device_scratch);
  cudaFree(device_layouts);
  cudaFree(device_arena);
  cudaFreeHost(host_slot);
  cudaFreeHost(host_arena);
  return out->status == 0 ? 0 : -1;
}
