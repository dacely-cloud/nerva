#include "nerva_cuda_api.h"
#include "deepseek_router.cuh"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

namespace {

constexpr uint32_t kV3NumExperts = 8;
constexpr uint32_t kV3NumGroups = 2;
constexpr uint32_t kV3TopKGroups = 1;
constexpr uint32_t kV3TopK = 2;
constexpr uint32_t kV3ExpertsPerGroup = kV3NumExperts / kV3NumGroups;

constexpr uint32_t kV4NumExperts = 4;
constexpr uint32_t kV4TopK = 2;
constexpr uint32_t kV4HashTopK = 3;

constexpr float kV3Logits[kV3NumExperts] = {
    -2.0f, 0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -3.0f,
};
constexpr float kV3Bias[kV3NumExperts] = {
    0.0f, 0.0f, 0.0f, 4.0f, 0.0f, 0.0f, -4.0f, 0.0f,
};
constexpr uint32_t kV3ExpectedIds[kV3TopK] = {3, 2};

constexpr float kV4Logits[kV4NumExperts] = {-2.0f, 0.0f, 1.0f, 3.0f};
constexpr float kV4Bias[kV4NumExperts] = {0.0f, 3.0f, 0.0f, -3.0f};
constexpr uint32_t kV4ExpectedIds[kV4TopK] = {1, 2};

constexpr float kV4HashLogits[kV4NumExperts] = {4.0f, -1.0f, 0.0f, 2.0f};
constexpr uint32_t kV4HashExpectedIds[kV4HashTopK] = {2, 1, 3};

struct DeviceRouterOutput {
  uint32_t v3_ids[kV3TopK];
  uint32_t v4_ids[kV4TopK];
  uint32_t v4_hash_ids[kV4HashTopK];
  float v3_weights[kV3TopK];
  float v4_weights[kV4TopK];
  float v4_hash_weights[kV4HashTopK];
};

__device__ __host__ float sigmoid_score(float value) {
  return 1.0f / (1.0f + expf(-value));
}

__device__ __host__ float softplus_score(float value) {
  if (value > 20.0f) {
    return value;
  }
  if (value < -20.0f) {
    return expf(value);
  }
  return log1pf(expf(value));
}

__device__ __host__ float sqrtsoftplus_score(float value) {
  return sqrtf(softplus_score(value));
}

__device__ bool better_score(float lhs_score,
                             uint32_t lhs_id,
                             float rhs_score,
                             uint32_t rhs_id) {
  return lhs_score > rhs_score ||
         (lhs_score == rhs_score && lhs_id < rhs_id);
}

__device__ void insert_topk(float score,
                            uint32_t id,
                            float *top_scores,
                            uint32_t *top_ids,
                            uint32_t k) {
  for (uint32_t slot = 0; slot < k; ++slot) {
    if (better_score(score, id, top_scores[slot], top_ids[slot])) {
      for (uint32_t shift = k - 1; shift > slot; --shift) {
        top_scores[shift] = top_scores[shift - 1];
        top_ids[shift] = top_ids[shift - 1];
      }
      top_scores[slot] = score;
      top_ids[slot] = id;
      return;
    }
  }
}

__device__ float top2_sum_group(const float *scores, uint32_t start) {
  float first = -INFINITY;
  float second = -INFINITY;
  for (uint32_t i = 0; i < kV3ExpertsPerGroup; ++i) {
    const float value = scores[start + i];
    if (value > first) {
      second = first;
      first = value;
    } else if (value > second) {
      second = value;
    }
  }
  return first + second;
}

__device__ void run_v3_grouped_route(DeviceRouterOutput *out) {
  const float logits[kV3NumExperts] = {
      -2.0f, 0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -3.0f,
  };
  const float bias[kV3NumExperts] = {
      0.0f, 0.0f, 0.0f, 4.0f, 0.0f, 0.0f, -4.0f, 0.0f,
  };
  float raw_scores[kV3NumExperts];
  float choice_scores[kV3NumExperts];
  for (uint32_t i = 0; i < kV3NumExperts; ++i) {
    raw_scores[i] = sigmoid_score(logits[i]);
    choice_scores[i] = raw_scores[i] + bias[i];
  }

  float group_scores[kV3NumGroups];
  for (uint32_t group = 0; group < kV3NumGroups; ++group) {
    group_scores[group] = top2_sum_group(choice_scores, group * kV3ExpertsPerGroup);
  }

  uint32_t selected_group = 0;
  float selected_group_score = group_scores[0];
  for (uint32_t group = 1; group < kV3NumGroups; ++group) {
    if (better_score(group_scores[group], group, selected_group_score, selected_group)) {
      selected_group = group;
      selected_group_score = group_scores[group];
    }
  }

  float top_scores[kV3TopK] = {-INFINITY, -INFINITY};
  uint32_t top_ids[kV3TopK] = {0, 1};
  const uint32_t start = selected_group * kV3ExpertsPerGroup;
  for (uint32_t i = 0; i < kV3ExpertsPerGroup; ++i) {
    const uint32_t expert = start + i;
    insert_topk(choice_scores[expert], expert, top_scores, top_ids, kV3TopK);
  }

  float weight_sum = 0.0f;
  for (uint32_t i = 0; i < kV3TopK; ++i) {
    weight_sum += raw_scores[top_ids[i]];
  }
  const float scale = 2.5f / weight_sum;
  for (uint32_t i = 0; i < kV3TopK; ++i) {
    out->v3_ids[i] = top_ids[i];
    out->v3_weights[i] = raw_scores[top_ids[i]] * scale;
  }
}

__device__ void run_v4_route(DeviceRouterOutput *out) {
  const float logits[kV4NumExperts] = {-2.0f, 0.0f, 1.0f, 3.0f};
  const float bias[kV4NumExperts] = {0.0f, 3.0f, 0.0f, -3.0f};
  float raw_scores[kV4NumExperts];
  float top_scores[kV4TopK] = {-INFINITY, -INFINITY};
  uint32_t top_ids[kV4TopK] = {0, 1};
  for (uint32_t i = 0; i < kV4NumExperts; ++i) {
    raw_scores[i] = sqrtsoftplus_score(logits[i]);
    insert_topk(raw_scores[i] + bias[i], i, top_scores, top_ids, kV4TopK);
  }

  float weight_sum = 0.0f;
  for (uint32_t i = 0; i < kV4TopK; ++i) {
    weight_sum += raw_scores[top_ids[i]];
  }
  const float scale = 1.5f / weight_sum;
  for (uint32_t i = 0; i < kV4TopK; ++i) {
    out->v4_ids[i] = top_ids[i];
    out->v4_weights[i] = raw_scores[top_ids[i]] * scale;
  }
}

__device__ void run_v4_hash_route(DeviceRouterOutput *out) {
  const float logits[kV4NumExperts] = {4.0f, -1.0f, 0.0f, 2.0f};
  const uint32_t hash_ids[kV4HashTopK] = {2, 1, 3};
  float weight_sum = 0.0f;
  for (uint32_t i = 0; i < kV4HashTopK; ++i) {
    const uint32_t expert = hash_ids[i];
    out->v4_hash_ids[i] = expert;
    out->v4_hash_weights[i] = sqrtsoftplus_score(logits[expert]);
    weight_sum += out->v4_hash_weights[i];
  }
  for (uint32_t i = 0; i < kV4HashTopK; ++i) {
    out->v4_hash_weights[i] /= weight_sum;
  }
}

__global__ void deepseek_router_smoke_kernel(DeviceRouterOutput *out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  run_v3_grouped_route(out);
  run_v4_route(out);
  run_v4_hash_route(out);
}

__global__ void deepseek_router_route_kernel(
    uint32_t router_kind,
    const float *logits,
    const float *correction_bias,
    const uint32_t *hash_route_table,
    uint32_t hash_route_table_len,
    uint32_t num_experts,
    uint32_t num_groups,
    uint32_t top_k_groups,
    uint32_t top_k,
    uint32_t norm_topk_prob,
    uint32_t route_token,
    float routed_scaling_factor,
    uint32_t *expert_ids,
    float *weights,
    int32_t *route_error) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  namespace dsr = nerva::deepseek::router;
  int status = -1;
  if (router_kind == dsr::kV3GroupedSigmoid) {
    status = dsr::route_v3_grouped_sigmoid(logits,
                                           correction_bias,
                                           num_experts,
                                           num_groups,
                                           top_k_groups,
                                           top_k,
                                           norm_topk_prob,
                                           routed_scaling_factor,
                                           expert_ids,
                                           weights);
  } else if (router_kind == dsr::kV4SqrtSoftplus) {
    status = dsr::route_v4_sqrtsoftplus(logits,
                                        correction_bias,
                                        num_experts,
                                        top_k,
                                        norm_topk_prob,
                                        routed_scaling_factor,
                                        expert_ids,
                                        weights);
  } else if (router_kind == dsr::kV4Hash) {
    status = dsr::route_v4_hash(logits,
                                hash_route_table,
                                hash_route_table_len,
                                route_token,
                                num_experts,
                                top_k,
                                norm_topk_prob,
                                routed_scaling_factor,
                                expert_ids,
                                weights);
  }
  *route_error = status;
}

uint32_t f32_bits(float value) {
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

uint64_t mix_hash_u32(uint64_t hash, uint32_t value) {
  for (uint32_t byte = 0; byte < 4; ++byte) {
    hash ^= static_cast<uint8_t>((value >> (byte * 8)) & 0xffu);
    hash *= 1099511628211ull;
  }
  return hash;
}

uint64_t hash_route(const uint32_t *ids, const float *weights, uint32_t len) {
  uint64_t hash = 1469598103934665603ull;
  for (uint32_t i = 0; i < len; ++i) {
    hash = mix_hash_u32(hash, ids[i]);
    hash = mix_hash_u32(hash, f32_bits(weights[i]));
  }
  return hash;
}

void expected_v3(float *weights) {
  const float raw3 = sigmoid_score(kV3Logits[3]);
  const float raw2 = sigmoid_score(kV3Logits[2]);
  const float scale = 2.5f / (raw3 + raw2);
  weights[0] = raw3 * scale;
  weights[1] = raw2 * scale;
}

void expected_v4(float *weights) {
  const float raw1 = sqrtsoftplus_score(kV4Logits[1]);
  const float raw2 = sqrtsoftplus_score(kV4Logits[2]);
  const float scale = 1.5f / (raw1 + raw2);
  weights[0] = raw1 * scale;
  weights[1] = raw2 * scale;
}

void expected_v4_hash(float *weights) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < kV4HashTopK; ++i) {
    weights[i] = sqrtsoftplus_score(kV4HashLogits[kV4HashExpectedIds[i]]);
    sum += weights[i];
  }
  for (uint32_t i = 0; i < kV4HashTopK; ++i) {
    weights[i] /= sum;
  }
}

void compare_route(const uint32_t *actual_ids,
                   const float *actual_weights,
                   const uint32_t *expected_ids,
                   const float *expected_weights,
                   uint32_t len,
                   uint64_t *mismatches,
                   float *max_abs_diff) {
  *mismatches = 0;
  *max_abs_diff = 0.0f;
  for (uint32_t i = 0; i < len; ++i) {
    if (actual_ids[i] != expected_ids[i]) {
      *mismatches += 1;
    }
    const float diff = fabsf(actual_weights[i] - expected_weights[i]);
    if (diff > *max_abs_diff) {
      *max_abs_diff = diff;
    }
    if (diff > 1e-6f) {
      *mismatches += 1;
    }
  }
}

void clear_result(NervaCudaDeepSeekRouterSmokeResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
  out->v3_num_experts = kV3NumExperts;
  out->v3_num_groups = kV3NumGroups;
  out->v3_top_k_groups = kV3TopKGroups;
  out->v3_top_k = kV3TopK;
  out->v4_num_experts = kV4NumExperts;
  out->v4_top_k = kV4TopK;
  out->v4_hash_top_k = kV4HashTopK;
}

int fail(NervaCudaDeepSeekRouterSmokeResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

void clear_route_result(const NervaCudaDeepSeekRouterRouteRequest *request,
                        NervaCudaDeepSeekRouterRouteResult *out) {
  memset(out, 0, sizeof(*out));
  out->status = -1;
  out->route_error = -1;
  if (request != nullptr) {
    out->router_kind = request->router_kind;
    out->num_experts = request->num_experts;
    out->num_groups = request->num_groups;
    out->top_k_groups = request->top_k_groups;
    out->top_k = request->top_k;
  }
}

int fail_route(NervaCudaDeepSeekRouterRouteResult *out, cudaError_t err) {
  out->cuda_error = static_cast<int32_t>(err);
  out->status = -1;
  return -1;
}

bool validate_route_request(const NervaCudaDeepSeekRouterRouteRequest *request) {
  namespace dsr = nerva::deepseek::router;
  if (request == nullptr || request->logits == nullptr ||
      request->expert_ids == nullptr || request->weights == nullptr ||
      request->num_experts == 0 || request->top_k == 0 ||
      request->top_k > dsr::kMaxTopK || request->top_k > request->num_experts) {
    return false;
  }
  if (request->router_kind == dsr::kV3GroupedSigmoid) {
    return request->num_groups > 0 && request->num_groups <= dsr::kMaxGroups &&
           request->top_k_groups > 0 &&
           request->top_k_groups <= request->num_groups &&
           request->num_experts % request->num_groups == 0;
  }
  if (request->router_kind == dsr::kV4SqrtSoftplus) {
    return true;
  }
  if (request->router_kind == dsr::kV4Hash) {
    const uint64_t end =
        static_cast<uint64_t>(request->route_token) * request->top_k +
        request->top_k;
    return request->hash_route_table != nullptr &&
           end <= request->hash_route_table_len;
  }
  return false;
}

}  // namespace

extern "C" int nerva_cuda_deepseek_router_smoke(
    NervaCudaDeepSeekRouterSmokeResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_result(out);

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail(out, err);
  }
  if (out->device_count <= 0) {
    return fail(out, cudaErrorNoDevice);
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail(out, err);
  }

  DeviceRouterOutput *device_output = nullptr;
  DeviceRouterOutput *host_output = nullptr;
  cudaStream_t stream = nullptr;
  const uint64_t output_bytes = sizeof(DeviceRouterOutput);

  err = cudaMalloc(reinterpret_cast<void **>(&device_output), output_bytes);
  if (err != cudaSuccess) goto cleanup;
  out->device_arena_bytes = output_bytes;

  err = cudaHostAlloc(reinterpret_cast<void **>(&host_output),
                      output_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  out->pinned_host_bytes = output_bytes;

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) goto cleanup;

  deepseek_router_smoke_kernel<<<1, 1, 0, stream>>>(device_output);
  out->kernel_launches += 1;
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(host_output,
                        device_output,
                        output_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->d2h_bytes = output_bytes;

  err = cudaStreamSynchronize(stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) goto cleanup;

  memcpy(out->v3_expert_ids, host_output->v3_ids, sizeof(out->v3_expert_ids));
  memcpy(out->v4_expert_ids, host_output->v4_ids, sizeof(out->v4_expert_ids));
  memcpy(out->v4_hash_expert_ids,
         host_output->v4_hash_ids,
         sizeof(out->v4_hash_expert_ids));
  memcpy(out->v3_weights, host_output->v3_weights, sizeof(out->v3_weights));
  memcpy(out->v4_weights, host_output->v4_weights, sizeof(out->v4_weights));
  memcpy(out->v4_hash_weights,
         host_output->v4_hash_weights,
         sizeof(out->v4_hash_weights));

  float v3_expected_weights[kV3TopK];
  float v4_expected_weights[kV4TopK];
  float v4_hash_expected_weights[kV4HashTopK];
  expected_v3(v3_expected_weights);
  expected_v4(v4_expected_weights);
  expected_v4_hash(v4_hash_expected_weights);

  compare_route(out->v3_expert_ids,
                out->v3_weights,
                kV3ExpectedIds,
                v3_expected_weights,
                kV3TopK,
                &out->v3_mismatches,
                &out->v3_max_abs_diff);
  compare_route(out->v4_expert_ids,
                out->v4_weights,
                kV4ExpectedIds,
                v4_expected_weights,
                kV4TopK,
                &out->v4_mismatches,
                &out->v4_max_abs_diff);
  compare_route(out->v4_hash_expert_ids,
                out->v4_hash_weights,
                kV4HashExpectedIds,
                v4_hash_expected_weights,
                kV4HashTopK,
                &out->v4_hash_mismatches,
                &out->v4_hash_max_abs_diff);

  out->v3_output_hash = hash_route(out->v3_expert_ids, out->v3_weights, kV3TopK);
  out->v4_output_hash = hash_route(out->v4_expert_ids, out->v4_weights, kV4TopK);
  out->v4_hash_output_hash =
      hash_route(out->v4_hash_expert_ids, out->v4_hash_weights, kV4HashTopK);
  out->status = (out->v3_mismatches == 0 && out->v4_mismatches == 0 &&
                 out->v4_hash_mismatches == 0)
                    ? 0
                    : -1;

cleanup:
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (host_output != nullptr) cudaFreeHost(host_output);
  if (device_output != nullptr) cudaFree(device_output);

  if (err != cudaSuccess) {
    return fail(out, err);
  }
  return out->status == 0 ? 0 : -1;
}

extern "C" int nerva_cuda_deepseek_router_route(
    const NervaCudaDeepSeekRouterRouteRequest *request,
    NervaCudaDeepSeekRouterRouteResult *out) {
  if (out == nullptr) {
    return -1;
  }
  clear_route_result(request, out);
  if (!validate_route_request(request)) {
    return -1;
  }

  cudaError_t err = cudaGetDeviceCount(&out->device_count);
  if (err != cudaSuccess) {
    return fail_route(out, err);
  }
  if (out->device_count <= 0) {
    return fail_route(out, cudaErrorNoDevice);
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return fail_route(out, err);
  }

  float *d_logits = nullptr;
  float *d_bias = nullptr;
  uint32_t *d_hash_route_table = nullptr;
  uint32_t *d_ids = nullptr;
  float *d_weights = nullptr;
  int32_t *d_route_error = nullptr;
  uint32_t *h_ids = nullptr;
  float *h_weights = nullptr;
  int32_t h_route_error = -1;
  cudaStream_t stream = nullptr;

  const uint64_t logits_bytes =
      sizeof(float) * static_cast<uint64_t>(request->num_experts);
  const uint64_t bias_bytes =
      request->correction_bias == nullptr
          ? 0
          : sizeof(float) * static_cast<uint64_t>(request->num_experts);
  const uint64_t hash_bytes =
      request->hash_route_table == nullptr
          ? 0
          : sizeof(uint32_t) *
                static_cast<uint64_t>(request->hash_route_table_len);
  const uint64_t ids_bytes =
      sizeof(uint32_t) * static_cast<uint64_t>(request->top_k);
  const uint64_t weights_bytes =
      sizeof(float) * static_cast<uint64_t>(request->top_k);
  const uint64_t route_error_bytes = sizeof(int32_t);

  err = cudaMalloc(reinterpret_cast<void **>(&d_logits), logits_bytes);
  if (err != cudaSuccess) goto cleanup;
  if (bias_bytes != 0) {
    err = cudaMalloc(reinterpret_cast<void **>(&d_bias), bias_bytes);
    if (err != cudaSuccess) goto cleanup;
  }
  if (hash_bytes != 0) {
    err = cudaMalloc(reinterpret_cast<void **>(&d_hash_route_table), hash_bytes);
    if (err != cudaSuccess) goto cleanup;
  }
  err = cudaMalloc(reinterpret_cast<void **>(&d_ids), ids_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_weights), weights_bytes);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMalloc(reinterpret_cast<void **>(&d_route_error), route_error_bytes);
  if (err != cudaSuccess) goto cleanup;
  out->device_arena_bytes =
      logits_bytes + bias_bytes + hash_bytes + ids_bytes + weights_bytes +
      route_error_bytes;

  err = cudaHostAlloc(reinterpret_cast<void **>(&h_ids),
                      ids_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  err = cudaHostAlloc(reinterpret_cast<void **>(&h_weights),
                      weights_bytes,
                      cudaHostAllocDefault);
  if (err != cudaSuccess) goto cleanup;
  out->pinned_host_bytes = ids_bytes + weights_bytes;

  err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(d_logits,
                        request->logits,
                        logits_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->h2d_bytes += logits_bytes;
  if (bias_bytes != 0) {
    err = cudaMemcpyAsync(d_bias,
                          request->correction_bias,
                          bias_bytes,
                          cudaMemcpyHostToDevice,
                          stream);
    if (err != cudaSuccess) goto cleanup;
    out->h2d_bytes += bias_bytes;
  }
  if (hash_bytes != 0) {
    err = cudaMemcpyAsync(d_hash_route_table,
                          request->hash_route_table,
                          hash_bytes,
                          cudaMemcpyHostToDevice,
                          stream);
    if (err != cudaSuccess) goto cleanup;
    out->h2d_bytes += hash_bytes;
  }
  err = cudaMemcpyAsync(d_route_error,
                        &h_route_error,
                        route_error_bytes,
                        cudaMemcpyHostToDevice,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->h2d_bytes += route_error_bytes;

  deepseek_router_route_kernel<<<1, 1, 0, stream>>>(
      request->router_kind,
      d_logits,
      d_bias,
      d_hash_route_table,
      request->hash_route_table_len,
      request->num_experts,
      request->num_groups,
      request->top_k_groups,
      request->top_k,
      request->norm_topk_prob,
      request->route_token,
      request->routed_scaling_factor,
      d_ids,
      d_weights,
      d_route_error);
  out->kernel_launches += 1;
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;

  err = cudaMemcpyAsync(h_ids,
                        d_ids,
                        ids_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(h_weights,
                        d_weights,
                        weights_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  err = cudaMemcpyAsync(&h_route_error,
                        d_route_error,
                        route_error_bytes,
                        cudaMemcpyDeviceToHost,
                        stream);
  if (err != cudaSuccess) goto cleanup;
  out->d2h_bytes = ids_bytes + weights_bytes + route_error_bytes;

  err = cudaStreamSynchronize(stream);
  out->sync_calls += 1;
  if (err != cudaSuccess) goto cleanup;

  out->route_error = h_route_error;
  if (h_route_error == 0) {
    memcpy(request->expert_ids, h_ids, ids_bytes);
    memcpy(request->weights, h_weights, weights_bytes);
    out->output_hash = hash_route(request->expert_ids,
                                  request->weights,
                                  request->top_k);
    out->status = 0;
  }

cleanup:
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (h_weights != nullptr) cudaFreeHost(h_weights);
  if (h_ids != nullptr) cudaFreeHost(h_ids);
  if (d_route_error != nullptr) cudaFree(d_route_error);
  if (d_weights != nullptr) cudaFree(d_weights);
  if (d_ids != nullptr) cudaFree(d_ids);
  if (d_hash_route_table != nullptr) cudaFree(d_hash_route_table);
  if (d_bias != nullptr) cudaFree(d_bias);
  if (d_logits != nullptr) cudaFree(d_logits);

  if (err != cudaSuccess) {
    return fail_route(out, err);
  }
  return out->status == 0 ? 0 : -1;
}
